"""Recovery data exclusions, gold-free generation, selection and exact restart."""

import copy
import json
from pathlib import Path

import pytest
import torch

from llm_scratch.curriculum.acceptance import fixture
from llm_scratch.curriculum.evaluation import (
    export_selected,
    load_selected,
    snapshot,
    weights_digest,
)
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import Expression
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.easy_recovery import data, runner
from llm_scratch.evaluation.checkpoints import load_for_evaluation
from llm_scratch.research.training_acceptance import _same
from llm_scratch.utils.config import canonical_json
from llm_scratch.utils.io import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recovery_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("easy-recovery")
    _, old_data, tokenizer, old_checkpoint = fixture(root / "source", ROOT)
    codec = ByteBPETokenizer.load(tokenizer)
    model, _, _ = load_for_evaluation(old_checkpoint, codec)
    weights = snapshot(model)
    source = root / "selected-source.pt"
    export_selected(
        source,
        weights,
        {"step": 1, "weights_sha256": weights_digest(weights)},
        {"selection_ids": []},
    )
    old_meta = json.loads((old_data / "manifest.json").read_text(encoding="utf-8"))
    prior = root / "prior-questions.jsonl"
    prior.write_text("", encoding="utf-8")
    protocol = json.loads((ROOT / "configs/data/easy_recovery.json").read_text(encoding="utf-8"))
    protocol.update(
        source_dataset_fingerprint=old_meta["fingerprint"],
        source_checkpoint_sha256=file_sha256(source),
        prior_questions_sha256=file_sha256(prior),
        max_length=256,
        selection={"wp03": 1, "authored": 1},
        gate={"wp03": 1, "authored": 2},
    )
    write_json(root / "data.json", protocol)
    prepared = data.prepare(root / "data.json", old_data, tokenizer, source, prior, root / "data")
    recipe = (ROOT / "configs/training/easy_recovery.toml").read_text(encoding="utf-8")
    for old, new in {
        "vocab_size = 8192": "vocab_size = 264",
        "context_length = 1024": "context_length = 512",
        "hidden_size = 512": "hidden_size = 16",
        "num_layers = 10": "num_layers = 1",
        "num_heads = 8": "num_heads = 4",
        "ffn_hidden_size = 2048": "ffn_hidden_size = 32",
        "dropout = 0.0": "dropout = 0.1",
        "max_steps = 1000": "max_steps = 4",
        "warmup_steps = 50": "warmup_steps = 0",
        'precision = "float16"': 'precision = "float32"',
        "block_size = 128": "block_size = 256",
        "batch_size = 8": "batch_size = 2",
        "gradient_accumulation_steps = 4": "gradient_accumulation_steps = 2",
        "validate_every = 100": "validate_every = 2",
        "checkpoint_every = 100": "checkpoint_every = 3",
        "milestones = [500, 1000]": "milestones = [2, 4]",
        "max_new_tokens = 64": "max_new_tokens = 8",
        "d2ea45abc8f81b516552c184d75c595b95fa2bd1abdc260538e497da4b627dc8": file_sha256(source),
    }.items():
        recipe = recipe.replace(old, new)
    config = root / "recovery.toml"
    config.write_text(recipe, encoding="utf-8")
    return {
        "config": config,
        "prepared_root": prepared,
        "tokenizer": tokenizer,
        "source_checkpoint": source,
        "output_root": root / "runs",
    }


def test_exclusion_includes_intermediate_results_and_operand_order():
    tree = Expression(
        operator="+",
        left=Expression(value=1),
        right=Expression(operator="-", left=Expression(value=8), right=Expression(value=3)),
    )
    excluded = data.facts(
        [{"task": "multi", "expression": tree.as_dict()}, {"task": "copy", "operands": [7, 6]}]
    )
    assert excluded == {data.pair(3, 8), data.pair(5, 1), data.pair(6, 7)}


def test_data_is_disjoint_deterministic_and_balanced_without_test_files(recovery_inputs):
    codec = ByteBPETokenizer.load(recovery_inputs["tokenizer"])
    parts = {
        k: data.Corpus(recovery_inputs["prepared_root"], k, codec)
        for k in ("train", "selection", "gate")
    }
    data.audit({k: v.unique_examples for k, v in parts.items()})
    train = parts["train"]
    assert train.meta["test_files_read"] is False
    assert train.repeat > 1 and len(train) > len(train.unique_examples)
    cfg = train.meta["config"]
    rows = parts["selection"].examples + parts["gate"].examples
    a, _ = data.partition(rows, set(), cfg)
    b, _ = data.partition(list(reversed(rows)), set(), cfg)
    assert a == b
    collided = copy.deepcopy(parts["selection"].examples[0])
    with pytest.raises(ValueError, match="leakage"):
        data.audit({"selection": [collided], "gate": [collided]})
    changed = copy.deepcopy(cfg)
    changed["gate"]["wp03"] = 100
    with pytest.raises(ValueError, match="Insufficient"):
        data.partition(rows, set(), changed)


def test_correct_final_does_not_verify_wrong_reasoning(recovery_inputs):
    codec = ByteBPETokenizer.load(recovery_inputs["tokenizer"])
    rows = data.Corpus(recovery_inputs["prepared_root"], "selection", codec).examples
    predictions = [
        {
            "question_id": r["question_id"],
            "generated_ids": codec.encode(
                r["reasoning"] + "</think><final>" + r["final_answer"] + "</final>",
                add_eos=True,
                allowed_special={"</think>", "<final>", "</final>"},
            ),
            "finish_reason": "eos",
        }
        for r in rows
    ]
    _, good = runner.score(predictions, rows, codec)
    assert good["accuracy"] == good["verified_accuracy"] == 1
    predictions[0]["generated_ids"] = codec.encode(
        "Wrong trace.</think><final>" + rows[0]["final_answer"] + "</final>",
        add_eos=True,
        allowed_special={"</think>", "<final>", "</final>"},
    )
    _, bad = runner.score(predictions, rows, codec)
    assert bad["accuracy"] == 1 and bad["verified_accuracy"] == 0.5
    with pytest.raises(ValueError, match="exactly once"):
        runner.score(predictions[:1], rows, codec)


def test_full_resume_exact_and_partial_run_never_reads_gate(recovery_inputs):
    # A gate payload is deliberately unavailable during controlled stops.
    gate = recovery_inputs["prepared_root"] / "gate.jsonl"
    payload = gate.read_bytes()
    gate.unlink()
    try:
        partial = runner.run(**recovery_inputs, stop_after=1)
        report = json.loads(
            (partial / "artifacts/recovery_report.json").read_text(encoding="utf-8")
        )
        assert not report["gate_evaluated"] and report["best"]["step"] == 0
    finally:
        gate.write_bytes(payload)
    full = runner.run(**recovery_inputs)
    resumed = runner.run(**recovery_inputs, resume=partial / "checkpoints/latest.pt")
    a = torch.load(full / "checkpoints/latest.pt", weights_only=True)
    b = torch.load(resumed / "checkpoints/latest.pt", weights_only=True)
    assert _same(a, b), "Exact resume changed weights/optimizer/RNG/cursor/selection"
    assert a["state"]["step"] == 4 and a["state"]["attempts"] == 4
    codec = ByteBPETokenizer.load(recovery_inputs["tokenizer"])
    model, meta = load_selected(resumed / "checkpoints/selected_model.pt", codec)
    extra = b["state"]["data_state"]["extra_state"]
    assert _same(model.state_dict(), extra["best_weights"])
    assert meta["selection"] == extra["best"]
    summary = json.loads((resumed / "artifacts/recovery_report.json").read_text(encoding="utf-8"))
    assert summary["gate_evaluated"] and summary["status"] == "completed"
    assert (resumed / "artifacts/loss_curves.png").exists()
    assert (full / "checkpoints/step-00000002.pt").exists()
    assert (resumed / "checkpoints/step-00000002.pt").exists()
    assert [r["step"] for r in extra["generated"]] == [0, 2, 4]
    # This path never consults the final gate when deciding the selected weights.
    assert "gate" not in extra and "gate_report" not in extra

    # Tampering must fail before another optimizer update can consume data.
    bad = copy.deepcopy(b)
    bad["state"]["data_state"]["extra_state"]["best"]["weights_sha256"] = "0" * 64
    bad_path = partial.parent / "altered-selection.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="Selected weights checksum"):
        runner.run(**recovery_inputs, resume=bad_path)


def test_portable_kit_round_trip_and_payload_guards(recovery_inputs, tmp_path):
    import zipfile

    from llm_scratch.easy_recovery import bundle

    wheel = tmp_path / "fixture.whl"
    # Extraction verifies transport; the installed-wheel acceptance uses a real build.
    wheel.write_bytes(b"offline wheel transport fixture")
    target = tmp_path / "kit.zip"
    info = bundle.create(
        recovery_inputs["prepared_root"],
        recovery_inputs["tokenizer"],
        recovery_inputs["config"],
        wheel,
        target,
        "a" * 40,
    )
    output = bundle.extract(target, tmp_path / "kit", info["sha256"])
    assert not list(output.rglob("*.pt"))
    with pytest.raises(ValueError, match="SHA256"):
        bundle.extract(target, tmp_path / "bad-sha", "0" * 64)
    (output / "data/gate.jsonl").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="payload mismatch"):
        bundle.verify(output)
    malicious = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(target) as source, zipfile.ZipFile(malicious, "w") as dest:
        for name in source.namelist():
            dest.writestr(name, source.read(name))
        dest.writestr("../outside.pt", "forbidden")
    with pytest.raises(ValueError, match="Unsafe"):
        bundle.extract(malicious, tmp_path / "unsafe", file_sha256(malicious))


def test_reject_wrong_source_and_invalid_recipe_before_training(recovery_inputs, tmp_path):
    cfg = recovery_inputs["config"].read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(cfg.replace("max_new_tokens = 8", "max_new_tokens = 0"), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        runner.load_recipe(bad)
    other = dict(recovery_inputs)
    source = tmp_path / "source.pt"
    source.write_bytes(b"corrupted")
    other["source_checkpoint"] = source
    with pytest.raises(ValueError, match="checksum"):
        runner.run(**other, stop_after=1)


def test_question_projection_cannot_include_gold(recovery_inputs):
    from llm_scratch.curriculum.evaluation import questions_only
    from llm_scratch.evaluation.pilot import QUESTION_KEYS

    codec = ByteBPETokenizer.load(recovery_inputs["tokenizer"])
    rows = data.Corpus(recovery_inputs["prepared_root"], "selection", codec).examples
    assert all(set(q) == QUESTION_KEYS for q in questions_only(rows))
    changed = [r | {"reasoning": "poison", "final_answer": "999999"} for r in rows]
    assert canonical_json(questions_only(rows)) == canonical_json(questions_only(changed))
