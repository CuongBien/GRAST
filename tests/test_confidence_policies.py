"""Threshold freeze/holdout boundaries, committed replay and counterfactual labels."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.calibrators import CalibrationContext
from llm_scratch.reasoning.confidence_stopping import ConfidencePolicy
from llm_scratch.reasoning.stopping import Policy
from llm_scratch.research import confidence_policies as policies
from llm_scratch.research.confidence_pilot import read_rows, write_rows
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_selection_constraint_tie_break_infeasibility_and_partition():
    a = ConfidencePolicy("a", "raw_threshold", 2, 16, 0.75)
    b = ConfidencePolicy("b", "raw_threshold", 2, 16, 0.9)
    bad = ConfidencePolicy("bad", "cges_lns", 2, 16, 0.9)

    def metric(id, accuracy, tokens):
        return {"policy_id": id, "accuracy": accuracy, "mean_costs": {"prefix_tokens": tokens}}

    metrics = [
        metric("fixed-k16", 0.8, 160),
        metric("a", 0.76, 30),
        metric("b", 0.76, 30),
        metric("bad", 0.74, 10),
    ]
    result = policies.select_thresholds(
        metrics, [a, b, bad], partition="calibration_threshold", max_regret=0.05
    )
    assert result["choices"]["raw_threshold"]["selected_policy"]["id"] == "b"
    assert result["choices"]["cges_lns"]["selected_policy"] is None
    with pytest.raises(ValueError, match="threshold partition only"):
        policies.select_thresholds(metrics, [a, b], partition="evaluation", max_regret=0.05)


def test_gold_free_commit_repeat_counterfactual_labels_and_forgery(tmp_path, monkeypatch):
    groups, refs = policies.synthetic_groups("synthetic_threshold", 4, 42)
    specs = (
        Policy("fixed-k16", "fixed", 16, 16),
        ConfidencePolicy("raw", "raw_threshold", 2, 16, 0.9),
    )
    fp = config_digest([asdict(g) for g in groups])
    original = Path.open

    def guard(path, *args, **kwargs):
        assert "references" not in path.name and "scored" not in path.name
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard)
        a = policies.commit_replay(
            groups, specs, fp, tmp_path / "a", context=CalibrationContext("synthetic")
        )
        b = policies.commit_replay(
            groups, specs, fp, tmp_path / "b", context=CalibrationContext("synthetic")
        )
    assert a == b and a["records"] == 24
    before = (tmp_path / "a/decisions.jsonl").read_bytes()
    m = policies.score_committed(
        groups,
        specs,
        fp,
        tmp_path / "a",
        refs,
        tmp_path / "score",
        {q: 12 for q in refs},
        context=CalibrationContext("synthetic"),
    )
    changed = policies.score_committed(
        groups,
        specs,
        fp,
        tmp_path / "a",
        {q: "999999" for q in refs},
        tmp_path / "counterfactual",
        {q: 12 for q in refs},
        context=CalibrationContext("synthetic"),
    )
    assert m[0]["accuracy"] == 0.5 and changed[0]["accuracy"] == 0
    assert (tmp_path / "a/decisions.jsonl").read_bytes() == before
    assert m[0]["wrong_early_stops"] == 0
    rows = read_rows(tmp_path / "a/decisions.jsonl")
    rows[0]["answer"] = "999999"
    write_rows(tmp_path / "a/decisions.jsonl", rows)
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    manifest["decisions_sha256"] = file_sha256(tmp_path / "a/decisions.jsonl")
    manifest["fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    write_json(tmp_path / "a/manifest.json", manifest)
    with pytest.raises(ValueError, match="disagree"):
        policies.score_committed(
            groups,
            specs,
            fp,
            tmp_path / "a",
            refs,
            tmp_path / "forged",
            {q: 12 for q in refs},
            context=CalibrationContext("synthetic"),
        )


def test_full_synthetic_selection_frozen_before_evaluation_and_no_real_io(tmp_path, monkeypatch):
    cfg = json.loads((ROOT / "configs/evaluation/wp09_policies.json").read_text())
    cfg["baseline_config"] = str(ROOT / "configs/evaluation/wp09_baselines.json")
    cfg["calibration_config"] = str(ROOT / "configs/evaluation/wp09_calibration.json")
    cfg["artifacts"]["root"] = str(tmp_path / "runs")
    path = tmp_path / "policies.json"
    write_json(path, cfg)
    original = policies.synthetic_groups
    seen = []

    def spy(partition, *args):
        seen.append(partition)
        if partition == "synthetic_evaluation":
            assert (
                len(list((tmp_path / "runs").glob("*/artifacts/synthetic/frozen_selection.json")))
                == 1
            )
        return original(partition, *args)

    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic policy verification must not load real banks/checkpoints")

    monkeypatch.setattr(policies, "synthetic_groups", spy)
    monkeypatch.setattr(policies, "SampleBank", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    run = policies.run(path, synthetic_only=True)
    r = json.loads((run / "artifacts/report.json").read_text())
    s = r["synthetic"]
    assert seen == ["synthetic_threshold", "synthetic_evaluation"]
    assert s["threshold_decisions"] == 7296 and s["evaluation_decisions"] == 2112
    assert len(s["evaluation_metrics"]) == 11
    assert not s["evaluation_used_for_selection"] and not s["model_efficacy_claim"]
    frozen = run / "artifacts/synthetic/frozen_selection.json"
    choices = policies.load_selection(
        frozen, expected_fingerprint=s["selection"]["fingerprint"], expected_scope="synthetic_only"
    )
    assert len(choices) == 3
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        policies.load_selection(
            frozen, expected_fingerprint="f" * 64, expected_scope="synthetic_only"
        )
    with pytest.raises(ValueError, match="scope/schema"):
        policies.load_selection(
            frozen,
            expected_fingerprint=s["selection"]["fingerprint"],
            expected_scope="model_validation",
        )
    assert r["real"] is None and not r["real_evaluation_scored"]
