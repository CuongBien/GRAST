"""Checkpoint reload, question/gold separation, marked-answer EM and fixed pilot reports."""

import json
from pathlib import Path

import pytest
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation import capability
from llm_scratch.evaluation.capability import (
    generate_questions,
    left_pad,
    load_protocol,
    run_capability,
)
from llm_scratch.evaluation.checkpoints import load_for_evaluation
from llm_scratch.evaluation.pilot import (
    audit_question_splits,
    load_manifest,
    load_questions,
    load_references,
)
from llm_scratch.evaluation.pilot_metrics import (
    extract_generated_final,
    score_predictions,
    summarize,
)
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.research.capability_acceptance import make_fixture
from llm_scratch.research.training_acceptance import _same

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/wp07_capability.toml"


@pytest.fixture
def inputs(tmp_path):
    config, data, codec_root, checkpoints = make_fixture(tmp_path / "inputs", CONFIG)
    return config, data, codec_root, checkpoints


def codec():
    return ByteBPETokenizer(vocab_size=264)


def response(answer, c=None):
    c = codec() if c is None else c
    return [
        *c.encode("R1 = 1 / 2 = 1/2."),
        c.special_id("</think>"),
        c.special_id("<final>"),
        *c.encode(answer),
        c.special_id("</final>"),
        c.eos_token_id,
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.50", "1/2"),
        ("2/4", "1/2"),
        ("−2", "-2"),
        ("1,000", "1000"),
        ("NaN", None),
        ("1/0", None),
        ("2 and 3", None),
        ("", None),
    ],
)
def test_marked_final_uses_numeric_v1_exact_rationals(text, expected):
    result = extract_generated_final(response(text), codec(), "eos")
    assert result["predicted_answer"] == expected
    assert result["answer_valid"] == (expected is not None)


@pytest.mark.parametrize(
    "kind", ["bare", "rationale_number", "literal_markers", "duplicate", "reversed"]
)
def test_never_guess_last_number_or_accept_malformed_final_markers(kind):
    c = codec()
    if kind == "bare":
        ids = c.encode("5")
    elif kind == "rationale_number":
        ids = c.encode("The last number is 5.")
    elif kind == "literal_markers":
        ids = c.encode("<final>5</final>")
    elif kind == "duplicate":
        ids = response("5") + response("5")
    else:
        ids = [c.special_id("</final>"), *c.encode("5"), c.special_id("<final>")]
    result = extract_generated_final(ids, c, "max_new_tokens")
    assert not result["answer_valid"] and not result["format_valid"]


def test_complete_final_answer_and_eos_format_are_separate_diagnostics():
    c = codec()
    full = response("1/2")
    assert extract_generated_final(full, c, "eos")["format_valid"]
    short = extract_generated_final(full[:-1], c, "max_new_tokens")
    assert short["answer_valid"] and not short["format_valid"]


def test_accuracy_denominator_includes_invalid_wrong_and_exhausted_questions():
    c = codec()
    rows = [
        {
            "question_id": "a",
            "difficulty": "easy",
            "generated_ids": response("0.5"),
            "finish_reason": "eos",
        },
        {
            "question_id": "b",
            "difficulty": "easy",
            "generated_ids": c.encode("3"),
            "finish_reason": "max_new_tokens",
        },
        {
            "question_id": "c",
            "difficulty": "medium",
            "generated_ids": response("2"),
            "finish_reason": "eos",
        },
        {
            "question_id": "d",
            "difficulty": "hard",
            "generated_ids": [],
            "finish_reason": "max_new_tokens",
        },
    ]
    scored = score_predictions(rows, {"a": "1/2", "b": "3", "c": "3", "d": "3"}, c)
    summary = summarize(scored)
    assert summary["overall"]["questions"] == 4 and summary["overall"]["exact_match"] == 0.25
    assert summary["overall"]["answer_valid_rate"] == 0.5
    assert summary["overall"]["format_valid_rate"] == 0.5
    assert summary["by_difficulty"]["easy"]["exact_match"] == 0.5
    assert sum(m["questions"] for m in summary["by_difficulty"].values()) == 4
    with pytest.raises(ValueError, match="sets must match"):
        score_predictions(rows[:-1], {"a": "1/2", "b": "3", "c": "3", "d": "3"}, c)
    with pytest.raises(ValueError):
        score_predictions(rows, {"a": None, "b": "3", "c": "3", "d": "3"}, c)


def test_pilot_question_loader_never_opens_references(inputs, monkeypatch):
    manifest = load_manifest(inputs[1])
    original = Path.open

    def guard(path, *args, **kwargs):
        if "references" in path.name:
            raise AssertionError("Gold read by question loader")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    val = load_questions(inputs[1], manifest, "validation", expected_records=3)
    test = load_questions(inputs[1], manifest, "test", expected_records=3)
    audit_question_splits([val, test])
    assert all("canonical_answer" not in row and "answer" not in row for row in val + test)


def test_pilot_corruption_duplicate_or_gold_inputs_are_rejected(inputs):
    manifest = load_manifest(inputs[1])
    path = inputs[1] / "pilot/validation.questions.jsonl"
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["canonical_answer"] = "5"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_questions(inputs[1], manifest, "validation", expected_records=3)
    spec = manifest["files"]["pilot/validation.questions.jsonl"]
    spec.update(sha256=file_sha256(path), size_bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="question-only"):
        load_questions(inputs[1], manifest, "validation", expected_records=3)
    rows[0].pop("canonical_answer")
    with pytest.raises(ValueError, match="duplication"):
        audit_question_splits([rows, rows])


def test_references_must_match_every_question_id(inputs):
    manifest = load_manifest(inputs[1])
    questions = load_questions(inputs[1], manifest, "validation", expected_records=3)
    ids = {q["question_id"] for q in questions}
    assert len(load_references(inputs[1], manifest, "validation", ids)) == 3
    with pytest.raises(ValueError, match="IDs"):
        load_references(inputs[1], manifest, "validation", ids - {next(iter(ids))})


def test_checkpoint_reload_preserves_weights_tying_rng_and_history(inputs):
    c = ByteBPETokenizer.load(inputs[2])
    before = torch.get_rng_state().clone()
    model, meta, history = load_for_evaluation(
        inputs[3][1], c, expected_sha256=file_sha256(inputs[3][1])
    )
    assert torch.equal(before, torch.get_rng_state())
    payload = torch.load(inputs[3][1], map_location="cpu", weights_only=True)
    assert _same(model.state_dict(), payload["state"]["model"])
    assert model.token_embedding.weight is model.lm_head.weight and not model.training
    assert meta["stage"] == "sft" and meta["step"] == 1 and history["train"]
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match="SHA256"):
        load_for_evaluation(inputs[3][1], c, expected_sha256="0" * 64)


@pytest.mark.parametrize("kind", ["nan", "tokenizer", "history"])
def test_checkpoint_rejects_bad_weights_metadata_or_history(inputs, tmp_path, kind):
    p = torch.load(inputs[3][1], map_location="cpu", weights_only=True)
    if kind == "nan":
        p["state"]["model"]["final_norm.weight"][0] = float("nan")
    elif kind == "tokenizer":
        p["state"]["model"]["_extra_state"]["tokenizer_fingerprint"] = "0" * 64
    else:
        p["state"]["data_state"]["extra_state"]["history"]["train"].clear()
    bad = tmp_path / "bad.pt"
    torch.save(p, bad)
    with pytest.raises(ValueError):
        load_for_evaluation(bad, ByteBPETokenizer.load(inputs[2]))


def test_left_padding_keeps_prefix_contiguous_and_cached_generation_reproducible(inputs):
    c = ByteBPETokenizer.load(inputs[2])
    model, _, _ = load_for_evaluation(inputs[3][1], c)
    manifest = load_manifest(inputs[1])
    questions = load_questions(inputs[1], manifest, "validation", expected_records=3)
    prefixes = [encode_reasoning_prompt(c, q["question"]) for q in questions]
    ids, mask = left_pad(prefixes, c.pad_token_id, "cpu")
    assert mask[:, -1].all()
    for i, prefix in enumerate(prefixes):
        assert ids[i][mask[i]].tolist() == prefix
        assert mask[i].tolist() == sorted(mask[i].tolist())
    policy = GenerationConfig(max_new_tokens=8, eos_token_id=c.eos_token_id)
    before = torch.get_rng_state().clone()
    a = generate(model, ids, mask, config=policy, use_cache=True)
    b = generate(model, ids, mask, config=policy, use_cache=False)
    assert torch.equal(a.generated_ids, b.generated_ids) and torch.equal(
        before, torch.get_rng_state()
    )

    def log(*args, **kwargs):
        pass

    first = generate_questions(
        model, c, questions, policy, batch_size=2, seed=42, precision="float32", emit=log
    )
    second = generate_questions(
        model, c, questions, policy, batch_size=2, seed=42, precision="float32", emit=log
    )
    assert first == second
    with pytest.raises(ValueError, match="without gold"):
        generate_questions(
            model,
            c,
            [questions[0] | {"canonical_answer": "5"}],
            policy,
            batch_size=2,
            seed=42,
            precision="float32",
            emit=log,
        )


@pytest.mark.parametrize(
    "old,new",
    [
        ("batch_size = 2", "batch_size = 0"),
        ("do_sample = false", "do_sample = true"),
        ('precision = "float32"', 'precision = "int8"'),
        ("cache_check_tokens = 8", "cache_check_tokens = 8\nthreshold = 0.9"),
    ],
)
def test_protocol_rejects_invalid_or_adaptive_pilot_settings(inputs, old, new):
    source = inputs[0].read_text(encoding="utf-8")
    inputs[0].write_text(source.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protocol(inputs[0])


def test_test_diagnostic_subset_is_forbidden_before_any_evaluation(inputs, tmp_path):
    with pytest.raises(ValueError, match="validation-only"):
        run_capability(
            inputs[0],
            inputs[1],
            inputs[2],
            split="test",
            diagnostic_limit=1,
            output_root=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def test_full_runner_saves_model_predictions_before_gold_and_exports_all_questions(
    inputs, tmp_path, monkeypatch
):
    original = capability.load_references
    output_root = tmp_path / "runs"

    def check_gold_read(root, manifest, split, ids):
        run = next(output_root.iterdir())
        predictions = list((run / "artifacts").glob(f"*/{split}.predictions.jsonl"))
        assert predictions, "Gold accessed before model-only predictions were saved"
        for path in predictions:
            rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
            assert all("reference" not in r and "canonical_answer" not in r for r in rows)
        return original(root, manifest, split, ids)

    monkeypatch.setattr(capability, "load_references", check_gold_read)
    path = run_capability(inputs[0], inputs[1], inputs[2], split="both", output_root=output_root)
    report = json.loads((path / "artifacts/capability_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    for record in report["checkpoints"].values():
        for split in ("validation", "test"):
            assert record["pilot"][split]["overall"]["questions"] == 3
        assert record["text_samples"] and record["cache_equivalence"]["status"] == "passed"
    assert (path / "artifacts/stage_loss_curves.png").read_bytes().startswith(b"\x89PNG")
    assert (path / "artifacts/stage_loss_curves.svg").exists()
    assert (path / "artifacts/pilot_metrics.csv").exists()
    assert (path / "artifacts/pilot_exact_match.png").read_bytes().startswith(b"\x89PNG")
