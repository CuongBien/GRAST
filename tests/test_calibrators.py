"""Analytical logistic oracles, grouped weighting and fit/prediction boundaries."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import test_policy_replay

from llm_scratch.reasoning.calibrators import (
    CalibrationContext,
    CalibrationExample,
    CalibrationSpec,
    FeatureVector,
    LogisticCalibrator,
    example_weights,
    features_from_prefix,
    fit_calibrator,
    fit_support,
    load_calibrator,
    logistic_objective,
    require_disjoint_assessment,
    sigmoid,
)
from llm_scratch.research.confidence_calibration import predict_inputs
from llm_scratch.utils.config import config_digest


def example(i, correct, *, partition="calibration_fit", group=None, values=None, seed=42, n=4):
    return CalibrationExample(
        partition,
        group or f"group-{i}",
        f"q-{i}",
        seed,
        n,
        FeatureVector(values or (0.75, 0.2, -0.1, math.log(n) / math.log(16))),
        correct,
    )


def spec(**kwargs):
    return CalibrationSpec(min_positive_groups=1, min_negative_groups=1, **kwargs)


def test_constant_feature_solution_is_empirical_prior_not_saturated():
    rows = [example(i, i < 3) for i in range(4)]
    context = CalibrationContext("synthetic")
    model = fit_calibrator(rows, spec(), context)
    p = model.predict_features([rows[0].features], context=context)[0]
    assert p == pytest.approx(0.75, abs=1e-10)
    assert model.payload["parameters"]["intercept"] == pytest.approx(math.log(3))
    assert model.payload["parameters"]["coefficients"] == pytest.approx([0] * 4, abs=1e-10)
    assert model.payload["parameters"]["scale"] == [1] * 4
    assert model.payload["optimizer"]["gradient_inf_norm"] <= 1e-8


def test_group_question_seed_prefix_weights_and_fit_only_standardization():
    a = example(1, True, group="a")
    rows = [
        a,
        replace(a, draws=8, features=FeatureVector((0.75, 0.2, -0.1, 0.75))),
        replace(a, trial_seed=43),
        example(2, False, group="a"),
        example(3, False, group="b"),
    ]
    assert example_weights(rows) == pytest.approx([0.0625, 0.0625, 0.125, 0.25, 0.5])
    model = fit_calibrator(rows, spec(), CalibrationContext("synthetic"))
    expected = np.average([e.features.values for e in rows], axis=0, weights=[1, 1, 2, 4, 8])
    assert model.payload["parameters"]["center"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="Duplicate"):
        example_weights(rows + [a])
    with pytest.raises(ValueError, match="multiple independent"):
        example_weights(rows + [replace(a, trial_seed=44, group_id="other")])


def test_analytical_gradient_hessian_finite_differences_and_sigmoid_extremes():
    x = np.array([[1, -1, 0.2], [1, 2, 0.1], [1, 0.3, 2], [1, 1, 1]], dtype=np.float64)
    y, w = np.array([0, 1, 1, 0]), np.array([0.1, 0.2, 0.3, 0.4])
    theta, penalty = np.array([0.2, -0.4, 0.6]), np.array([0, 0.01, 0.01])
    _, grad, hessian = logistic_objective(theta, x, y, w, penalty)
    epsilon = 1e-5
    for i in range(3):
        delta = np.eye(3)[i] * epsilon
        plus = logistic_objective(theta + delta, x, y, w, penalty)
        minus = logistic_objective(theta - delta, x, y, w, penalty)
        assert grad[i] == pytest.approx((plus[0] - minus[0]) / (2 * epsilon), abs=1e-9)
        assert hessian[:, i] == pytest.approx((plus[1] - minus[1]) / (2 * epsilon), abs=1e-9)
    assert sigmoid([-1e300, 0, 1e300]).tolist() == [0, 0.5, 1]


@pytest.mark.parametrize(
    "partition", ["calibration_threshold", "evaluation", "synthetic_assessment"]
)
def test_reserved_or_assessment_labels_cannot_fit(partition):
    rows = [example(1, True, partition=partition), example(2, False, partition=partition)]
    with pytest.raises(ValueError, match="calibration_fit only"):
        fit_calibrator(rows, spec(), CalibrationContext("synthetic"))


@pytest.mark.parametrize("correct", [True, False])
def test_single_outcome_refused_and_repeated_prefixes_do_not_supply_groups(correct):
    rows = [example(i, correct) for i in range(10)]
    with pytest.raises(ValueError, match="no degenerate fit"):
        fit_calibrator(rows, spec(), CalibrationContext("synthetic"))
    rows = [example(i, i % 2 == 0, group="same") for i in range(20)]
    support = fit_support(rows, CalibrationSpec())
    assert support["positive_groups"] == support["negative_groups"] == 1
    assert not support["eligible"]


def test_prefix_features_future_invariance_abstention_and_context_binding():
    g = test_policy_replay.observation(["4", "4", "8"])
    f = features_from_prefix(g.prefix(2))
    assert f.values == pytest.approx((1, 0, -1, 0.25))
    cs = list(g.candidates)
    cs[2] = replace(cs[2], extracted_answer="999", logprob_sum=-math.inf)
    assert features_from_prefix(replace(g, candidates=tuple(cs)).prefix(2)) == f
    assert features_from_prefix(g.prefix(1)) is None
    assert features_from_prefix(test_policy_replay.observation([None, None])) is None
    rows = [example(1, True), example(2, False)]
    model = fit_calibrator(rows, spec(), CalibrationContext("synthetic"))
    assert model.predict_prefix(
        g.prefix(2), context=CalibrationContext("synthetic")
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="context mismatch"):
        model.predict_prefix(
            g.prefix(2), context=CalibrationContext("model_validation", "a" * 64, "b" * 64)
        )
    with pytest.raises(ValueError, match="observable"):
        predict_inputs(model, [{"correct": True}], CalibrationContext("synthetic"))


def test_serialization_pinning_immutability_schema_and_disjoint_groups(tmp_path):
    context = CalibrationContext("synthetic")
    rows = [example(1, True), example(2, False)]
    model = fit_calibrator(rows, spec(), context)
    path = tmp_path / "calibrator.json"
    model.save(path)
    loaded = load_calibrator(path, expected_fingerprint=model.fingerprint)
    assert loaded.predict_features([rows[0].features], context=context) == model.predict_features(
        [rows[0].features], context=context
    )
    assert model == fit_calibrator(rows, spec(), context)
    altered = model.payload
    altered["parameters"]["intercept"] = 100
    assert model.payload["parameters"]["intercept"] == 0
    with pytest.raises(ValueError, match="overwrite"):
        model.save(path)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_calibrator(path, expected_fingerprint="f" * 64)
    altered["parameters"]["scale"][0] = 0
    altered["fingerprint"] = config_digest({k: v for k, v in altered.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="parameters/training"):
        LogisticCalibrator(json.dumps(altered))
    with pytest.raises(ValueError, match="disjoint"):
        require_disjoint_assessment(model, [replace(rows[0], partition="synthetic_assessment")])
    require_disjoint_assessment(model, [example(3, True, partition="synthetic_assessment")])


def test_nonconverged_fit_and_invalid_feature_values_refused():
    rows = [example(i, i > 2, values=(0.5 if i < 3 else 1, 0.2, -0.1, 0.5)) for i in range(6)]
    with pytest.raises(ValueError, match="did not converge"):
        fit_calibrator(rows, spec(max_iterations=1), CalibrationContext("synthetic"))
    with pytest.raises(ValueError, match="observable"):
        FeatureVector((1, float("nan"), -1, 0.5))
    with pytest.raises(ValueError, match="observable"):
        FeatureVector((1, 0, 1, 0.5))
