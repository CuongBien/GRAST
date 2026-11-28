"""Offline baselines, question weighting, atomic batch accounting and gold boundary."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import test_confidence_pilot
import test_policy_replay
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.stopping import Policy
from llm_scratch.research import confidence_baselines as baselines
from llm_scratch.research import confidence_pilot
from llm_scratch.research.policy_replay import parse_policies, replay_one
from llm_scratch.research.sample_bank import SampleBank
from llm_scratch.utils.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def batch_group():
    group = test_policy_replay.observation(["4"] * 6, [2, 4, 6, 8, 1, 3])
    return replace(
        group,
        candidates=tuple(
            replace(c, batch_start=i // 4 * 4, batch_row=i % 4, batch_size=min(4, 6 - i // 4 * 4))
            for i, c in enumerate(group.candidates)
        ),
    )


def test_partial_batch_charges_whole_batch_and_padding_slots():
    group = batch_group()
    result = baselines.costs(group, 2, 10)
    assert result == {
        "prefix_tokens": 6,
        "batch_tokens": 20,
        "padded_generation_slots": 32,
        "prefix_prompt_tokens": 20,
        "batch_prompt_tokens": 40,
        "charged_candidates": 4,
        "launched_batches": 1,
    }
    tail = baselines.costs(group, 5, 10)
    assert tail["prefix_tokens"] == 21 and tail["batch_tokens"] == 24
    assert tail["padded_generation_slots"] == 38 and tail["charged_candidates"] == 6
    full = baselines.costs(group, 6, 10)
    assert full["prefix_tokens"] == full["batch_tokens"] == 24
    with pytest.raises(ValueError, match="Invalid cost"):
        baselines.costs(group, 0, 10)


def test_future_batch_cost_is_not_a_stopping_input():
    group = batch_group()
    first = replay_one(group, Policy("u2", "agreement", 2, 6))
    candidates = list(group.candidates)
    candidates[2] = replace(candidates[2], token_count=100)
    changed = replace(group, candidates=tuple(candidates))
    assert replay_one(changed, Policy("u2", "agreement", 2, 6)) == first
    a, b = baselines.costs(group, 2, 10), baselines.costs(changed, 2, 10)
    assert a["prefix_tokens"] == b["prefix_tokens"] and b["batch_tokens"] > a["batch_tokens"]
    candidates = list(group.candidates)
    candidates[4] = replace(candidates[4], token_count=100)
    assert baselines.costs(replace(group, candidates=tuple(candidates)), 2, 10) == a


def test_question_normalization_not_unequal_greedy_sample_totals():
    def row(q, tokens):
        return {
            "question_id": q,
            "difficulty": "test",
            "correct": False,
            "draws": 1,
            **{k: tokens for k in baselines.COST_KEYS},
        }

    reference = {q: {k: v for k in baselines.COST_KEYS} for q, v in [("a", 100), ("b", 200)]}
    _, sample = baselines.grouped_summary(
        [row("a", 10), row("a", 14), row("b", 28), row("b", 32)], reference
    )
    _, greedy = baselines.grouped_summary([row("a", 12), row("b", 30)], reference)
    assert sample["observations"] == 4 and greedy["observations"] == 2
    assert sample["unique_questions"] == greedy["unique_questions"] == 2
    assert sample["mean_costs"] == greedy["mean_costs"]
    assert sample["prefix_savings"] == greedy["prefix_savings"] == 0.86


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root, benchmark, checkpoint, plan = test_confidence_pilot.inputs.__wrapped__(tmp_path_factory)
    run = confidence_pilot.generate_pilot(
        root / "inputs", plan["fingerprint"], checkpoint, root / "runs", "cpu"
    )
    bank = SampleBank(run / "artifacts/bank")
    refs = {r["question_id"]: r for r in confidence_pilot.read_rows(benchmark / "references.jsonl")}
    confidence_pilot.write_rows(
        root / "references.jsonl",
        [
            {"question_id": q, "split": "validation", "canonical_answer": refs[q]["final_answer"]}
            for q in plan["question_ids"]
        ],
    )
    cfg = json.loads((ROOT / "configs/evaluation/wp09_baselines.json").read_text())
    cfg["artifacts"]["root"] = str(root / "baseline-runs")
    g = json.loads((run / "artifacts/generation_commit.json").read_text())
    cfg["inputs"].update(
        pilot_inputs=str(root / "inputs"),
        plan_fingerprint=plan["fingerprint"],
        bank=str(bank.root),
        bank_fingerprint=bank.fingerprint,
        generation_commit=str(run / "artifacts/generation_commit.json"),
        generation_fingerprint=g["fingerprint"],
        greedy=str(run / "artifacts/greedy.predictions.jsonl"),
        greedy_sha256=g["greedy_sha256"],
        references=str(root / "references.jsonl"),
        references_sha256=file_sha256(root / "references.jsonl"),
    )
    path = root / "baselines.json"
    write_json(path, cfg)
    return root, path, bank, run, plan


def test_greedy_is_separate_and_rejects_labels_or_bad_prompt(source):
    root, _, bank, run, plan = source
    _, _, codec = confidence_pilot.load_inputs(root / "inputs", plan["fingerprint"])
    raw = confidence_pilot.read_rows(run / "artifacts/greedy.predictions.jsonl")
    gs = baselines.greedy_decisions(raw, bank, codec)
    assert len(gs) == 2 and all(r["trial_seed"] is None and r["policy_id"] == "greedy" for r in gs)
    modified = copy.deepcopy(raw)
    modified[0]["correct"] = True
    with pytest.raises(ValueError, match="model-only"):
        baselines.greedy_decisions(modified, bank, codec)
    modified = copy.deepcopy(raw)
    modified[0]["question"] = "different"
    with pytest.raises(ValueError, match="Invalid dedicated greedy"):
        baselines.greedy_decisions(modified, bank, codec)


def test_decision_boundary_repeat_and_actual_run_without_model_load(source, monkeypatch):
    root, config, bank, pilot_run, plan = source
    cfg = baselines.load_config(config).data
    policies = parse_policies(cfg["policies"])
    _, _, codec = confidence_pilot.load_inputs(root / "inputs", plan["fingerprint"])
    raw = confidence_pilot.read_rows(pilot_run / "artifacts/greedy.predictions.jsonl")
    original = Path.open

    def guard(path, *args, **kwargs):
        assert "references" not in path.name and "scored" not in path.name
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard)
        first = baselines.commit_decisions(bank, policies, raw, codec, root / "decisions-a")
        second = baselines.commit_decisions(bank, policies, raw, codec, root / "decisions-b")
    assert first == second and first["sampled_decisions"] == 48 and first["greedy_decisions"] == 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline replay must not load any model checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", forbidden)
        run = baselines.run(config)
    report = json.loads((run / "artifacts/analysis/report.json").read_text())
    assert report["sampled_decisions"] == 48 and report["greedy_decisions"] == 2
    assert len(report["metrics"]) == 9 and report["unique_questions"] == 2
    assert not any(
        report[k]
        for k in (
            "new_inference_performed",
            "calibrator_fitted",
            "threshold_partition_scored",
            "evaluation_scored",
        )
    )
    fixed = next(r for r in report["metrics"] if r["policy_id"] == "fixed-k16")
    assert fixed["prefix_savings"] == fixed["batch_savings"] == fixed["padded_slot_savings"] == 0
    assert (run / "artifacts/analysis/cost_proxies.png").is_file()
    assert len(confidence_pilot.read_rows(run / "artifacts/analysis/per_question.jsonl")) == 18
    with pytest.raises(ValueError, match="outside bank"):
        baselines.score_baselines(
            bank,
            codec,
            root / "decisions-a",
            root / "references.jsonl",
            cfg["inputs"]["references_sha256"],
            bank.root / "labels",
            policies,
            "fixed-k16",
            raw,
        )
    forged = confidence_pilot.read_rows(root / "decisions-a/greedy.decisions.jsonl")
    forged[0]["answer"] = "123"
    confidence_pilot.write_rows(root / "decisions-a/greedy.decisions.jsonl", forged)
    commit = json.loads((root / "decisions-a/commit.json").read_text())
    commit["greedy_decisions_sha256"] = file_sha256(root / "decisions-a/greedy.decisions.jsonl")
    from llm_scratch.utils.config import config_digest

    commit["fingerprint"] = config_digest({k: v for k, v in commit.items() if k != "fingerprint"})
    write_json(root / "decisions-a/commit.json", commit)
    with pytest.raises(ValueError, match="frozen raw completions"):
        baselines.score_baselines(
            bank,
            codec,
            root / "decisions-a",
            root / "references.jsonl",
            cfg["inputs"]["references_sha256"],
            root / "forged-score",
            policies,
            "fixed-k16",
            raw,
        )
