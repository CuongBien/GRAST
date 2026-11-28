"""Independent signal/reliability oracles and gold-free frozen-prefix noninterference."""

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import test_confidence_baselines
import test_policy_replay
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.calibration import agreement_reliability
from llm_scratch.reasoning.confidence import prefix_signals
from llm_scratch.research import confidence_diagnostics as diagnostic
from llm_scratch.research.confidence_pilot import read_rows, write_rows
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_entropy_canonical_classes_invalid_bucket_and_denominator():
    signals = prefix_signals(test_policy_replay.observation(["4", "4", "8", None]))
    assert signals["answer"] == "4" and signals["agreement"] == 0.5
    assert signals["valid_fraction"] == 0.75
    assert signals["answer_entropy_nats"] == pytest.approx(1.0397207708399179)
    assert signals["answer_entropy_normalized"] == pytest.approx(0.75)
    assert signals["invalid_answers"] == 1 and signals["representative_index"] == 0


@pytest.mark.parametrize(
    "answers,agreement", [(["4"], 1), ([None], 0), ([None, None], 0), (["4", "4"], 1)]
)
def test_degenerate_zero_entropy_does_not_imply_correctness(answers, agreement):
    result = prefix_signals(test_policy_replay.observation(answers))
    assert result["answer_entropy_nats"] == result["answer_entropy_normalized"] == 0
    assert result["agreement"] == agreement
    assert "correct" not in result and "reference" not in result
    if all(a is None for a in answers):
        assert result["winner_member_logprob_mean"] is None
        assert result["representative_logprob_mean"] is None


def test_raw_logprob_length_normalization_pooling_and_winner_tie():
    group = test_policy_replay.observation(["4", "8"], [1, 3])
    cs = [
        replace(
            c,
            generated_logprobs=tuple([p] * c.token_count),
            logprob_sum=math.fsum([p] * c.token_count),
            logprob_mean=p,
        )
        for c, p in zip(group.candidates, [-0.2, -0.6], strict=True)
    ]
    result = prefix_signals(replace(group, candidates=tuple(cs)))
    assert result["mean_candidate_logprob_mean"] == pytest.approx(-0.4)
    assert result["pooled_token_logprob_mean"] == pytest.approx(-0.5)
    assert result["mean_sequence_logprob_sum"] == pytest.approx(-1)
    assert result["winner_member_logprob_mean"] == result["representative_logprob_mean"] == -0.2
    assert result["answer_entropy_normalized"] == pytest.approx(1)
    assert result["observed_generated_tokens"] == 4
    cs[1] = replace(cs[1], extracted_answer="4")
    result = prefix_signals(replace(group, candidates=tuple(cs)))
    assert result["winner_member_logprob_mean"] == pytest.approx(-0.4)
    assert result["representative_logprob_mean"] == -0.2


def test_future_answers_and_logprobs_never_change_observed_prefix():
    group = test_policy_replay.observation(["4", "4", "8", None])
    original = prefix_signals(group.prefix(2))
    cs = list(group.candidates)
    cs[2] = replace(cs[2], extracted_answer="999", logprob_sum=-math.inf)
    changed = replace(group, candidates=tuple(cs))
    assert prefix_signals(changed.prefix(2)) == original
    with pytest.raises(ValueError, match="Inconsistent"):
        prefix_signals(changed)
    with pytest.raises(ValueError, match="nonempty policy prefix"):
        prefix_signals(group.prefix(0))
    with pytest.raises(ValueError, match="nonempty policy prefix"):
        prefix_signals({"candidates": group.candidates, "reference": "4"})


def test_reliability_question_weights_empty_bins_and_exact_boundaries():
    def row(q, seed, score, correct):
        return {
            "question_id": q,
            "trial_seed": seed,
            "draws": 2,
            "agreement": score,
            "correct": correct,
        }

    rows = [row("a", 1, 0.2, False), row("a", 2, 0.8, True), row("b", 1, 1, False)]
    result = agreement_reliability(rows, [0, 0.1, 0.5, 1])
    assert result["unique_questions"] == 2 and result["observations"] == 3
    assert result["accuracy"] == 0.25 and result["mean_agreement"] == 0.75
    assert result["raw_agreement_ece"] == pytest.approx(0.5)
    assert result["raw_agreement_brier"] == pytest.approx(0.52)
    assert result["bins"][0]["accuracy"] is None
    assert result["bins"][0]["mean_agreement"] is None
    assert result["bins"][2]["observations"] == 2
    assert result["bins"][2]["question_weight_mass"] == 0.75
    rows[0]["agreement"] = 0.5
    result = agreement_reliability(rows, [0, 0.1, 0.5, 1])
    assert result["bins"][1]["observations"] == 0
    assert result["bins"][2]["observations"] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        agreement_reliability(rows + rows[:1], [0, 1])
    with pytest.raises(ValueError, match="one nonempty prefix"):
        agreement_reliability(rows + [row("c", 1, 1, False) | {"draws": 4}], [0, 1])


@pytest.mark.parametrize("correct,ece,brier", [(False, 0.75, 0.625), (True, 0.25, 0.125)])
def test_single_outcome_ece_brier_descriptive_auc_undefined(correct, ece, brier):
    rows = [
        {"question_id": "a", "trial_seed": i, "draws": 2, "agreement": a, "correct": correct}
        for i, a in enumerate([1, 0.5])
    ]
    result = agreement_reliability(rows, [0, 0.5, 1])
    assert result["raw_agreement_ece"] == ece
    assert result["raw_agreement_brier"] == brier
    assert result["correctness_classes"] == 1 and result["roc_auc"] is None
    assert result["discrimination_status"] == "undefined_single_outcome_class"


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root, _, bank, _, plan = test_confidence_baselines.source.__wrapped__(tmp_path_factory)
    cfg = json.loads((ROOT / "configs/evaluation/wp09_confidence_diagnostics.json").read_text())
    cfg["artifacts"]["root"] = str(root / "diagnostic-runs")
    cfg["inputs"].update(
        pilot_inputs=str(root / "inputs"),
        plan_fingerprint=plan["fingerprint"],
        bank=str(bank.root),
        bank_fingerprint=bank.fingerprint,
        references=str(root / "references.jsonl"),
        references_sha256=file_sha256(root / "references.jsonl"),
    )
    path = root / "diagnostic.json"
    write_json(path, cfg)
    return root, path, bank


def test_trace_commit_gold_boundary_prefix_only_and_repeat(source, monkeypatch):
    root, _, bank = source
    opened, sizes = [], []
    original_open, original_signal = Path.open, diagnostic.prefix_signals

    def guard(path, *args, **kwargs):
        opened.append(path.name)
        assert "references" not in path.name and "scored" not in path.name
        return original_open(path, *args, **kwargs)

    def spy(observation):
        sizes.append(len(observation.candidates))
        assert not hasattr(observation, "reference")
        assert [c.candidate_index for c in observation.candidates] == list(
            range(len(observation.candidates))
        )
        return original_signal(observation)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard)
        patch.setattr(diagnostic, "prefix_signals", spy)
        a = diagnostic.create_trace(bank, root / "trace-a")
        b = diagnostic.create_trace(bank, root / "trace-b")
    assert a == b and a["records"] == 96
    assert sizes == list(range(1, 17)) * 12
    assert "signals.jsonl" in opened
    assert diagnostic.load_trace(bank, root / "trace-a")[0] == a


def test_full_offline_run_labels_only_after_commit_and_forgery_rejected(source, monkeypatch):
    root, config, bank = source

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic replay must not load a model checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", forbidden)
        run = diagnostic.run(config)
    art = run / "artifacts"
    report = json.loads((art / "analysis/report.json").read_text())
    assert report["prefix_observations"] == 96 and report["unique_questions"] == 2
    assert len(report["metrics"]) == 16 and len(report["metrics"][0]["bins"]) == 10
    assert not any(
        report[k]
        for k in [
            "new_inference_performed",
            "calibrator_fitted",
            "threshold_search_performed",
            "threshold_partition_scored",
            "evaluation_scored",
        ]
    )
    assert (art / "analysis/reliability_diagnostic.png").is_file()
    original_trace = (art / "trace/signals.jsonl").read_bytes()
    refs = read_rows(root / "references.jsonl")
    changed = [r | {"canonical_answer": "123456789"} for r in refs]
    write_rows(root / "changed-references.jsonl", changed)
    result = diagnostic.score_trace(
        bank,
        art / "trace",
        root / "changed-references.jsonl",
        file_sha256(root / "changed-references.jsonl"),
        root / "changed-analysis",
        [i / 10 for i in range(11)],
        [2, 4, 8, 16],
    )
    assert result["correct_prefix_observations"] == 0
    assert (art / "trace/signals.jsonl").read_bytes() == original_trace
    with pytest.raises(ValueError, match="outside bank/trace"):
        diagnostic.score_trace(
            bank,
            art / "trace",
            root / "references.jsonl",
            file_sha256(root / "references.jsonl"),
            art / "trace/labels",
            [0, 1],
            [2],
        )
    # A forger cannot hide edited signals by recomputing just the file/manifest hashes.
    forged = read_rows(art / "trace/signals.jsonl")
    forged[0]["mean_candidate_logprob_mean"] = -123
    write_rows(art / "trace/signals.jsonl", forged)
    manifest = json.loads((art / "trace/manifest.json").read_text())
    manifest["signals_sha256"] = file_sha256(art / "trace/signals.jsonl")
    manifest["fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    write_json(art / "trace/manifest.json", manifest)
    with pytest.raises(ValueError, match="disagree"):
        diagnostic.load_trace(bank, art / "trace")


def test_frozen_config_rejects_threshold_and_bin_search(source):
    root, config, _ = source
    cfg = json.loads(config.read_text())
    bad = copy.deepcopy(cfg)
    bad["threshold_search"] = True
    path = root / "bad-diagnostic.json"
    write_json(path, bad)
    with pytest.raises(ValueError, match="Invalid frozen"):
        diagnostic.load_config(path)
    bad = copy.deepcopy(cfg)
    bad["reliability_bin_edges"] = [0, 0.99, 1]
    write_json(path, bad)
    with pytest.raises(ValueError, match="Invalid frozen"):
        diagnostic.load_config(path)
