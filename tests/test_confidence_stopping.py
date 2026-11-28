"""Analytical CGES likelihoods, bounded stopping and no future/gold/context leakage."""

import math
from dataclasses import replace

import pytest
import test_policy_replay

from llm_scratch.reasoning.calibrators import CalibrationContext, CalibrationSpec, fit_calibrator
from llm_scratch.reasoning.confidence_stopping import (
    ConfidencePolicy,
    cges_summary,
    evaluate_confidence_prefix,
    replay_confidence,
)
from llm_scratch.research.confidence_calibration import synthetic_examples


def group(answers, confidences):
    g = test_policy_replay.observation(answers)
    cs = []
    for c, p in zip(g.candidates, confidences, strict=True):
        logs = tuple([math.log(p)] * c.token_count)
        cs.append(
            replace(
                c,
                generated_logprobs=logs,
                logprob_sum=math.fsum(logs),
                logprob_mean=math.fsum(logs) / c.token_count,
            )
        )
    return replace(g, candidates=tuple(cs))


def test_cges_likelihood_matches_direct_product_and_keeps_unseen_mass():
    g = group(["4", "8", "4"], [0.6, 0.7, 0.8])
    s = cges_summary(g, 4, 1e-6)
    likelihoods = {
        a: math.prod(
            p if answer == a else (1 - p) / 3
            for answer, p in zip(["4", "8", "4"], [0.6, 0.7, 0.8], strict=True)
        )
        for a in ["4", "8", "unseen1", "unseen2"]
    }
    total = sum(likelihoods.values())
    assert s["confidence"] == pytest.approx(likelihoods["4"] / total)
    assert s["unseen_mass"] == pytest.approx(
        (likelihoods["unseen1"] + likelihoods["unseen2"]) / total
    )
    one = cges_summary(group(["4"], [0.8]), 3, 1e-6)
    assert one["confidence"] == pytest.approx(0.8) and one["unseen_mass"] == pytest.approx(0.2)
    two = cges_summary(group(["4", "4"], [0.8, 0.8]), 3, 1e-6)
    assert two["confidence"] == pytest.approx(32 / 33)


def test_cges_tie_earliest_response_low_confidence_and_finite_extremes():
    assert cges_summary(group(["8", "4"], [0.7, 0.7]), 3, 1e-6)["answer"] == "8"
    low = cges_summary(group(["4", "4"], [0.01, 0.01]), 3, 1e-6)
    assert low["confidence"] < 0.001 and low["unseen_mass"] > 0.999
    s = cges_summary(group(["4"] * 16, [1 - 1e-14] * 16), 17, 1e-6)
    assert math.isfinite(s["confidence"]) and 0 <= s["confidence"] <= 1
    with pytest.raises(ValueError, match="unseen slots"):
        cges_summary(group(["4", "8"], [0.5, 0.5]), 2, 1e-6)


def test_threshold_inclusive_minimum_invalid_never_triggers_and_budget_abstention():
    p = ConfidencePolicy("raw", "raw_threshold", 2, 3, 0.5, 4)
    g = group(["4", "8", "4"], [0.8] * 3)
    r = replay_confidence(g, p)
    assert r["draws"] == 2 and r["answer"] == "4" and r["stop_reason"] == "confidence_threshold"
    assert not evaluate_confidence_prefix(g.prefix(1), p)["stop"]
    invalid = group(["4", None, None], [0.99] * 3)
    cp = ConfidencePolicy("cges", "cges_lns", 2, 3, 0.9, 4)
    assert not evaluate_confidence_prefix(invalid.prefix(2), cp)["stop"]
    assert replay_confidence(invalid, cp)["stop_reason"] == "budget_exhausted"
    all_invalid = replay_confidence(group([None] * 3, [0.9] * 3), p)
    assert all_invalid["answer"] is None and all_invalid["draws"] == 3


def test_only_prefixes_reach_decisions_and_future_mutation_does_not_change_stop(monkeypatch):
    import llm_scratch.reasoning.confidence_stopping as module

    g = group(["4", "4", "8", "8"], [0.9] * 4)
    p = ConfidencePolicy("cg", "cges_lns", 2, 4, 0.9, 5)
    original = module.evaluate_confidence_prefix
    seen = []

    def spy(observation, policy, **kwargs):
        seen.append(len(observation.candidates))
        assert not hasattr(observation, "reference")
        return original(observation, policy, **kwargs)

    monkeypatch.setattr(module, "evaluate_confidence_prefix", spy)
    a = replay_confidence(g, p)
    cs = list(g.candidates)
    cs[2] = replace(cs[2], extracted_answer="9999", logprob_sum=-math.inf)
    assert replay_confidence(replace(g, candidates=tuple(cs)), p) == a
    assert seen == [1, 2, 1, 2]


def test_calibrated_policy_pins_artifact_and_rejects_model_context_for_synthetic():
    model = fit_calibrator(
        synthetic_examples("calibration_fit", 32, 42),
        CalibrationSpec(),
        CalibrationContext("synthetic"),
    )
    p = ConfidencePolicy("cal", "calibrated_threshold", 2, 3, 0.99, 4, 1e-6, model.fingerprint)
    g = group(["4"] * 3, [0.98] * 3)
    r = replay_confidence(g, p, calibrator=model, context=CalibrationContext("synthetic"))
    assert r["draws"] == 3 and r["stop_reason"] == "budget_exhausted"
    with pytest.raises(ValueError, match="context mismatch"):
        replay_confidence(
            g,
            p,
            calibrator=model,
            context=CalibrationContext("model_validation", "a" * 64, "b" * 64),
        )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        replay_confidence(
            g,
            replace(p, calibrator_fingerprint="f" * 64),
            calibrator=model,
            context=CalibrationContext("synthetic"),
        )
    with pytest.raises(ValueError, match="do not accept"):
        replay_confidence(
            g, ConfidencePolicy("raw", "raw_threshold", 2, 3, 0.9, 4), calibrator=model
        )
