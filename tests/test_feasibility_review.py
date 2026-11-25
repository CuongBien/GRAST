"""Frozen evidence chain, question-only selection, context constraints and probe scoring."""

import copy
import json
from pathlib import Path

import pytest

from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.research.feasibility_review import (
    audit_receipt,
    audit_sources,
    build_decision,
    choose_gsm,
    gsm_rows,
    load_config,
    load_probe,
    prepare,
    question_only,
    run,
)
from llm_scratch.research.review_acceptance import check
from llm_scratch.research.sample_bank import load_config as load_bank_config
from llm_scratch.utils.config import config_digest

ROOT = Path(__file__).resolve().parents[1]


def raw_rows():
    return [
        {"question_id": f"g{i}", "question": f"Compute {i} + 3.", "canonical_answer": str(i + 3)}
        for i in range(10)
    ]


def test_selector_uses_questions_only_is_stable_and_context_eligible():
    codec = ByteBPETokenizer(vocab_size=264)
    rows = raw_rows()
    questions = question_only(rows)
    selected, n = choose_gsm(
        questions, codec, count=3, seed=42, context_length=100, max_new_tokens=8
    )
    reordered, _ = choose_gsm(
        list(reversed(questions)), codec, count=3, seed=42, context_length=100, max_new_tokens=8
    )
    assert selected == reordered and n == 10
    for r in rows:
        r["canonical_answer"] = "999"
    assert question_only(rows) == questions
    bad = copy.deepcopy(questions)
    bad[0]["canonical_answer"] = "7"
    with pytest.raises(ValueError, match="gold"):
        choose_gsm(bad, codec, count=3, seed=42, context_length=100, max_new_tokens=8)
    with pytest.raises(ValueError, match="context-eligible"):
        choose_gsm(questions, codec, count=3, seed=42, context_length=10, max_new_tokens=8)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return check(tmp_path_factory.mktemp("wp08-review-parent") / "fixture", ROOT)


def test_actual_review_pipeline_and_no_automatic_gate_for_new_evidence(fixture):
    config, _, path = fixture
    report = json.loads((path / "artifacts/feasibility_report.json").read_text(encoding="utf-8"))
    assert len(report["gsm8k"]) == 2
    assert all(r["summary"]["overall"]["questions"] == 2 for r in report["gsm8k"])
    assert report["decision"]["confidence_efficacy_gate"] == "requires-new-review"
    assert not report["decision"]["automatic_training_or_threshold_fit"]
    assert load_config(config).data["gsm_probe"]["questions"] == 2


def test_preparation_does_not_open_heldout_test(fixture, tmp_path, monkeypatch):
    config, _, _ = fixture
    original = Path.open

    def guarded(p, *args, **kwargs):
        assert p.name != "test.jsonl"
        return original(p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    prepare(config, tmp_path / "probe")


@pytest.mark.parametrize("kind", ["checksum", "receipt", "escape"])
def test_source_receipt_tampering_is_rejected(fixture, tmp_path, kind):
    config, _, _ = fixture
    d = load_config(config).data
    i = d["inputs"]
    r = json.loads(Path(i["pilot_receipt"]).read_text(encoding="utf-8"))
    expected = i["pilot_receipt_fingerprint"]
    if kind == "checksum":
        next(iter(r["files"].values()))["sha256"] = "0" * 64
        expected = config_digest(r)
    elif kind == "receipt":
        r["run_directory"] = "bad"
    else:
        key = next(iter(r["files"]))
        r["files"]["../escaped.json"] = r["files"].pop(key)
        expected = config_digest(r)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_receipt(path, expected, i["project_root"], "WP08.2")


@pytest.mark.parametrize("kind", ["budget", "question", "manifest"])
def test_frozen_probe_changes_fail(fixture, tmp_path, kind):
    config, probe, _ = fixture
    d = load_config(config).data
    bank, _, _ = audit_sources(d)
    cp = load_bank_config(d["inputs"]["checkpoint_config"]).data["checkpoints"]
    root = tmp_path / "probe"
    root.mkdir()
    m = json.loads((probe / "manifest.json").read_text(encoding="utf-8"))
    q = (probe / "questions.jsonl").read_text(encoding="utf-8")
    if kind == "budget":
        d["gsm_probe"]["max_new_tokens"] += 1
    elif kind == "question":
        q += "\n"
    else:
        m["eligible_records"] += 1
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    (root / "questions.jsonl").write_text(q, encoding="utf-8")
    with pytest.raises(ValueError):
        load_probe(root, d, bank, cp)


def test_gsm_validation_identity_count_and_test_split_guards(fixture):
    config, _, _ = fixture
    d = load_config(config).data
    changed = copy.deepcopy(d)
    changed["inputs"]["wp03_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        gsm_rows(changed)
    changed = copy.deepcopy(d)
    changed["gsm_probe"]["expected_validation_records"] += 1
    with pytest.raises(ValueError, match="count"):
        gsm_rows(changed)


def test_posthoc_review_is_bound_to_actual_snapshot():
    gsm = [{"summary": {"overall": {"questions": 16}}}]
    p = {
        "candidates": 2880,
        "bank_fingerprint": "6941830812e5fe91ba583553556cbc68a10ecf9b84dcad2bb5850c97cab1c809",
    }
    r = {
        "decisions": 1440,
        "replay_fingerprint": "7fa3de8093ae73094a74c82b95e74cc7545858c2df05f65c965f1fee242e6a20",
    }
    decision = build_decision(p, r, gsm)
    assert decision["engineering_gate"] == "passed"
    assert decision["confidence_efficacy_gate"] == "no-go-with-current-model-and-benchmark"
    assert decision["primary_diagnostic_benchmark"] == "wp03-easy-validation"
    other = build_decision(p, dict(r, replay_fingerprint="0" * 64), gsm)
    assert other["confidence_efficacy_gate"] == "requires-new-review"
    assert other["medium_hard_role"] == "requires-review"
    assert "new human feasibility review" in other["next_gate"]
    with pytest.raises(ValueError):
        build_decision(p, r, [])


@pytest.mark.parametrize("field", ["upstream_split", "split"])
def test_test_rows_are_rejected_even_with_matching_checksum(fixture, tmp_path, field):
    config, _, _ = fixture
    d = load_config(config).data
    root = tmp_path / "dataset"
    (root / "gsm8k").mkdir(parents=True)
    original = Path(d["inputs"]["dataset"])
    m = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(s)
        for s in (original / "gsm8k/validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[0][field] = "test"
    path = root / "gsm8k/validation.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    from llm_scratch.data.datasets import file_sha256

    m["files"]["gsm8k/validation.jsonl"].update(
        sha256=file_sha256(path), size_bytes=path.stat().st_size
    )
    m["dataset_fingerprint"] = config_digest(
        {k: v for k, v in m.items() if k != "dataset_fingerprint"}
    )
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    d["inputs"].update(dataset=str(root), wp03_fingerprint=m["dataset_fingerprint"])
    with pytest.raises(ValueError, match="official train"):
        gsm_rows(d)


def test_scoring_projection_follows_both_prediction_files(fixture, tmp_path, monkeypatch):
    import llm_scratch.research.feasibility_review as module

    config, probe, _ = fixture
    original = module.normalize_example
    reads = []

    def guarded(row, source, split):
        assert len(list((tmp_path / "runs").glob("*/artifacts/*.predictions.jsonl"))) == 2
        reads.append(row["question_id"])
        return original(row, source, split)

    monkeypatch.setattr(module, "normalize_example", guarded)
    run(config, probe, output_root=tmp_path / "runs")
    assert len(reads) == 2
