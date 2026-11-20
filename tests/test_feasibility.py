"""Separate copy/arithmetic scoring, complete sampled coverage, and bounded training."""

import json
from pathlib import Path

import pytest
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.coverage import balanced_validation, sample_candidates, score_coverage
from llm_scratch.evaluation.pilot import load_manifest, load_questions
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.feasibility import diagnostic_batches, load_protocol, run_feasibility
from llm_scratch.research.feasibility_acceptance import make_inputs
from llm_scratch.research.feasibility_data import make_tasks, question_only, score_tasks
from llm_scratch.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/wp07_feasibility.toml"


def answer(codec, text, reasoning="R1 = 40 + 41 = 81."):
    return [
        *codec.encode(reasoning),
        codec.special_id("</think>"),
        codec.special_id("<final>"),
        *codec.encode(text),
        codec.special_id("</final>"),
        codec.eos_token_id,
    ]


def test_authored_pairs_balanced_disjoint_reproducible_and_not_pilot_easy_range():
    tasks = make_tasks()
    assert tasks == make_tasks() and tasks != make_tasks(seed=43)
    assert len(tasks["train"]) == 64 and len(tasks["validation"]) == 16
    pairs = {s: {tuple(r["operands"]) for r in rows} for s, rows in tasks.items()}
    assert not pairs["train"] & pairs["validation"]
    for rows in tasks.values():
        assert len({r["question"] for r in rows}) == len(rows)
        assert sum(r["task"] == "copy" for r in rows) == len(rows) // 2
        for r in rows:
            assert all(40 <= n <= 99 for n in r["operands"])
            if r["task"] == "arithmetic":
                a, b = r["operands"]
                assert r["final_answer"] == str(a + b if r["operator"] == "+" else a - b)


@pytest.mark.parametrize("counts", [(0, 8), (3, 8), (32, 3), (3600, 2)])
def test_invalid_authored_dimensions(counts):
    with pytest.raises(ValueError):
        make_tasks(*counts)


def test_task_scorer_separates_copy_equation_and_numeric_final():
    codec = ByteBPETokenizer(vocab_size=264)
    rows = make_tasks(2, 2)["train"]
    preds = []
    for r in rows:
        preds.append(
            {
                "question_id": r["question_id"],
                "generated_ids": answer(codec, r["final_answer"], r["reasoning"]),
                "finish_reason": "eos",
            }
        )
    scored, metrics = score_tasks(preds, rows, codec)
    assert metrics["copy"]["correct"] == metrics["arithmetic"]["correct"] == 2
    assert metrics["arithmetic"]["operands_and_operator_correct"] == 2
    assert metrics["arithmetic"]["first_equation_arithmetic_correct"] == 2
    preds[1]["generated_ids"] = answer(codec, rows[1]["final_answer"], "R1 = 20 + 20 = 30.")
    _, metrics = score_tasks(preds, rows, codec)
    assert metrics["arithmetic"]["correct"] == 2
    assert metrics["arithmetic"]["operands_and_operator_correct"] == 1
    assert metrics["arithmetic"]["first_equation_arithmetic_correct"] == 1
    preds[0]["generated_ids"] = codec.encode(rows[0]["final_answer"])
    _, metrics = score_tasks(preds, rows, codec)
    assert metrics["copy"]["correct"] == 1
    with pytest.raises(ValueError):
        score_tasks(preds[:-1], rows, codec)


def test_training_labels_mask_exactly_the_generation_prefix():
    codec = ByteBPETokenizer(vocab_size=264)
    rows = make_tasks(2, 2)["train"]
    batches = diagnostic_batches(rows, codec, 3)
    assert [b["input_ids"].shape[0] for b in batches] == [3, 1]
    for batch in batches:
        for ids, labels, mask in zip(
            batch["input_ids"], batch["labels"], batch["attention_mask"], strict=True
        ):
            first = int((labels != -100).nonzero()[0])
            assert ids[first - 1] == codec.special_id("<think>")
            assert torch.equal(
                ids[first:][mask[first:].bool()], labels[first:][mask[first:].bool()]
            )
            assert (labels[~mask.bool()] == -100).all()
    with pytest.raises(ValueError):
        diagnostic_batches(rows, codec, 2, [0, 0, 2, 3])


def test_selection_is_balanced_order_independent_and_rejects_test_or_gold(tmp_path):
    config, dataset, _, _ = make_inputs(tmp_path / "inputs", CONFIG)
    manifest = load_manifest(dataset)
    rows = load_questions(dataset, manifest, "validation", expected_records=3)
    assert balanced_validation(rows, 1, 42) == balanced_validation(rows[::-1], 1, 42)
    assert {q["difficulty"] for q in balanced_validation(rows, 1, 42)} == {"easy", "medium", "hard"}
    for bad in (
        [r | {"split": "test"} for r in rows],
        [r | {"answer": "3"} for r in rows],
        rows + [rows[0]],
    ):
        with pytest.raises(ValueError):
            balanced_validation(bad, 1, 42)
    with pytest.raises(ValueError):
        balanced_validation(rows, 2, 42)
    assert load_protocol(config).data["coverage"]["candidates"] == 2


def test_coverage_includes_invalid_and_checks_all_candidate_indices():
    codec = ByteBPETokenizer(vocab_size=264)
    predictions = []
    for qid, difficulty, texts in (("a", "easy", ["5", "3"]), ("b", "hard", [None, "4"])):
        for i, text in enumerate(texts):
            predictions.append(
                {
                    "question_id": qid,
                    "difficulty": difficulty,
                    "split": "validation",
                    "candidate_index": i,
                    "generated_ids": [] if text is None else answer(codec, text),
                    "finish_reason": "eos",
                }
            )
    scored, questions, summary = score_coverage(
        predictions, {"a": "5", "b": "5"}, codec, candidates=2
    )
    overall = summary["overall"]
    assert overall["observed_coverage_at_k"] == 0.5
    assert overall["candidate_accuracy"] == 0.25 and overall["answer_valid_rate"] == 0.75
    assert overall["questions"] == 2 and overall["candidates"] == 4
    assert questions[1]["correct_candidates"] == 0 and len(scored) == 4
    for bad in (
        predictions[:-1],
        predictions + [predictions[0]],
        [r | {"candidate_index": 0} for r in predictions],
    ):
        with pytest.raises(ValueError):
            score_coverage(bad, {"a": "5", "b": "5"}, codec, candidates=2)
    with pytest.raises(ValueError):
        score_coverage(predictions, {"a": None, "b": "5"}, codec, candidates=2)


def test_sampling_replays_private_rng_seeds_and_never_accepts_gold():
    codec = ByteBPETokenizer(vocab_size=264)
    seed_everything(42)
    model = CausalLM(
        ModelConfig(
            vocab_size=264,
            context_length=512,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            ffn_hidden_size=32,
        ),
        tokenizer_fingerprint=codec.fingerprint,
    ).eval()
    questions = [question_only(make_tasks(2, 2)["validation"])[0]]
    state = torch.get_rng_state().clone()
    kwargs = dict(
        candidates=3, batch_size=2, seed=42, precision="float32", emit=lambda *a, **k: None
    )
    policy = GenerationConfig(
        max_new_tokens=5, do_sample=True, temperature=0.8, top_k=40, top_p=0.95
    )
    first = sample_candidates(model, codec, questions, policy, **kwargs)
    second = sample_candidates(model, codec, questions, policy, **kwargs)
    assert first == second and torch.equal(state, torch.get_rng_state())
    assert [r["candidate_index"] for r in first] == [0, 1, 2]
    assert first[0]["batch_seed"] == first[1]["batch_seed"] != first[2]["batch_seed"]
    assert first[0]["generated_ids"] != first[1]["generated_ids"]
    with pytest.raises(ValueError):
        sample_candidates(model, codec, [questions[0] | {"gold": "5"}], policy, **kwargs)


@pytest.mark.parametrize(
    "old,new",
    [
        ("candidates = 16", "candidates = 0"),
        ("per_difficulty = 10", "per_difficulty = true"),
        ("eval_steps = [0, 25, 50, 100, 200]", "eval_steps = [0, 50]"),
        ('precision = "bfloat16"', 'precision = "unknown"'),
    ],
)
def test_protocol_rejects_invalid_settings(tmp_path, old, new):
    config = tmp_path / "bad.toml"
    config.write_text(CONFIG.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protocol(config)


def test_full_bounded_pipeline_never_reads_test_and_persists_before_gold(tmp_path, monkeypatch):
    config, dataset, codec, checkpoint = make_inputs(tmp_path / "inputs", CONFIG)
    digest = file_sha256(checkpoint)
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        if path.name.startswith("test."):
            raise AssertionError("WP07.6.1 must not read held-out test")
        if path.name == "validation.references.jsonl":
            assert list((tmp_path / "runs").glob("*/artifacts/coverage.predictions.jsonl"))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    run = run_feasibility(config, dataset, codec, output_root=tmp_path / "runs")
    report = json.loads((run / "artifacts/feasibility_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed" and report["source_checkpoint_unchanged"]
    assert report["coverage"]["overall"]["candidates"] == 6
    assert report["overfit"]["steps"] == report["overfit"]["attempts"] == 1
    assert report["overfit"]["optimizer_initial_state_entries"] == 0
    assert [r["step"] for r in report["overfit"]["evaluations"]] == [0, 1]
    assert file_sha256(checkpoint) == digest
    assert (run / "logs/events.jsonl").stat().st_size > 0
    restored = torch.load(run / "artifacts/overfit/diagnostic-final.pt", weights_only=True)
    assert restored["state"]["step"] == 1
    assert (
        restored["state"]["data_state"]["extra_state"]["contract"]["purpose"]
        == "authored-overfit-diagnostic-only"
    )
