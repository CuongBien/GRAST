"""Group-weighted logistic calibration; inference consumes only observable prefixes."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from llm_scratch.reasoning.confidence import PROTOCOL as SIGNAL_PROTOCOL
from llm_scratch.reasoning.confidence import prefix_signals
from llm_scratch.research.bank_schema import digest, identifier, integer
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import canonical_json, config_digest

PROTOCOL = "group-weighted-l2-logistic-prefix-v1"
FEATURES = ("agreement", "answer_entropy_normalized", "winner_member_logprob_mean", "log_draws")


@dataclass(frozen=True)
class FeatureVector:
    values: tuple[float, ...]

    def __post_init__(self):
        v = self.values
        if (
            type(v) is not tuple
            or len(v) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
            or not 0 < v[0] <= 1
            or not 0 <= v[1] <= 1 + 1e-12
            or v[2] > 0
            or not 0.25 <= v[3] <= 1
        ):
            raise ValueError("Invalid ordered observable calibration features")


def features_from_prefix(observation):
    s = prefix_signals(observation)
    if s["draws"] < 2 or s["answer"] is None:
        return None
    if s["draws"] > 16:
        raise ValueError("Calibration supports prefixes 2..16")
    return FeatureVector(
        (
            s["agreement"],
            s["answer_entropy_normalized"],
            s["winner_member_logprob_mean"],
            math.log(s["draws"]) / math.log(16),
        )
    )


@dataclass(frozen=True)
class CalibrationContext:
    domain: str
    checkpoint_sha256: str | None = None
    tokenizer_fingerprint: str | None = None

    def __post_init__(self):
        if self.domain == "synthetic":
            valid = self.checkpoint_sha256 is None and self.tokenizer_fingerprint is None
        else:
            valid = (
                self.domain == "model_validation"
                and digest(self.checkpoint_sha256)
                and digest(self.tokenizer_fingerprint)
            )
        if not valid:
            raise ValueError("Invalid synthetic/model calibration context")


@dataclass(frozen=True)
class CalibrationSpec:
    l2: float = 0.01
    gradient_tolerance: float = 1e-8
    max_iterations: int = 100
    min_positive_groups: int = 5
    min_negative_groups: int = 5

    def __post_init__(self):
        if (
            type(self.l2) not in (int, float)
            or not math.isfinite(self.l2)
            or self.l2 <= 0
            or type(self.gradient_tolerance) not in (int, float)
            or not math.isfinite(self.gradient_tolerance)
            or not 0 < self.gradient_tolerance < 1
            or not integer(self.max_iterations, 1, 1000)
            or not integer(self.min_positive_groups, 1)
            or not integer(self.min_negative_groups, 1)
        ):
            raise ValueError("Invalid fixed calibration specification")


@dataclass(frozen=True)
class CalibrationExample:
    partition: str
    group_id: str
    question_id: str
    trial_seed: int
    draws: int
    features: FeatureVector
    correct: bool

    def __post_init__(self):
        if (
            self.partition
            not in {
                "calibration_fit",
                "calibration_threshold",
                "evaluation",
                "synthetic_assessment",
            }
            or not identifier(self.group_id)
            or not identifier(self.question_id)
            or not integer(self.trial_seed, 0, 2**32 - 1)
            or not integer(self.draws, 2, 16)
            or type(self.features) is not FeatureVector
            or type(self.correct) is not bool
            or not math.isclose(
                self.features.values[3], math.log(self.draws) / math.log(16), abs_tol=1e-12
            )
        ):
            raise ValueError("Invalid labeled calibration example")


def example_weights(examples):
    """Equal group, question within group, seed within question, prefix within seed."""
    if not examples or any(type(e) is not CalibrationExample for e in examples):
        raise ValueError("Require nonempty typed calibration examples")
    if len({e.partition for e in examples}) != 1:
        raise ValueError("Do not mix calibration partitions")
    if len({(e.question_id, e.trial_seed, e.draws) for e in examples}) != len(examples):
        raise ValueError("Duplicate question-seed-prefix calibration example")
    q_groups, group_qs, q_seeds = {}, {}, {}
    counts = Counter((e.question_id, e.trial_seed) for e in examples)
    for e in examples:
        if e.question_id in q_groups and q_groups[e.question_id] != e.group_id:
            raise ValueError("Question spans multiple independent groups")
        q_groups[e.question_id] = e.group_id
        group_qs.setdefault(e.group_id, set()).add(e.question_id)
        q_seeds.setdefault(e.question_id, set()).add(e.trial_seed)
    return np.array(
        [
            1
            / (
                len(group_qs)
                * len(group_qs[e.group_id])
                * len(q_seeds[e.question_id])
                * counts[e.question_id, e.trial_seed]
            )
            for e in examples
        ],
        dtype=np.float64,
    )


def fit_support(examples, spec):
    example_weights(examples)
    pos = {e.group_id for e in examples if e.correct}
    neg = {e.group_id for e in examples if not e.correct}
    return {
        "rows": len(examples),
        "unique_questions": len({e.question_id for e in examples}),
        "groups": len({e.group_id for e in examples}),
        "positive_groups": len(pos),
        "negative_groups": len(neg),
        "positive_rows": sum(e.correct for e in examples),
        "negative_rows": sum(not e.correct for e in examples),
        "eligible": len(pos) >= spec.min_positive_groups and len(neg) >= spec.min_negative_groups,
    }


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    result[~positive] = ez / (1 + ez)
    return result


def logistic_objective(theta, x, y, weights, penalty):
    z = x @ theta
    p = sigmoid(z)
    loss = float(weights @ (np.logaddexp(0, z) - y * z) + 0.5 * np.sum(penalty * theta**2))
    gradient = x.T @ (weights * (p - y)) + penalty * theta
    hessian = x.T @ ((weights * p * (1 - p))[:, None] * x) + np.diag(penalty)
    return loss, gradient, hessian


def fit_calibrator(examples, spec, context):
    if type(spec) is not CalibrationSpec or type(context) is not CalibrationContext:
        raise ValueError("Require an explicit calibration specification and context")
    if not examples or any(e.partition != "calibration_fit" for e in examples):
        raise ValueError("Fit accepts calibration_fit only")
    support = fit_support(examples, spec)
    if not support["eligible"]:
        raise ValueError("Insufficient independent positive/negative groups; no degenerate fit")
    w = example_weights(examples)
    raw = np.array([e.features.values for e in examples], dtype=np.float64)
    y = np.array([e.correct for e in examples], dtype=np.float64)
    center = w @ raw
    scale = np.sqrt(w @ ((raw - center) ** 2))
    scale[scale < 1e-12] = 1.0
    x = np.column_stack([np.ones(len(raw)), (raw - center) / scale])
    penalty = np.array([0, *([spec.l2] * len(FEATURES))], dtype=np.float64)
    theta = np.zeros(len(FEATURES) + 1, dtype=np.float64)
    prior = float(w @ y)
    theta[0] = math.log(prior / (1 - prior))
    for iteration in range(spec.max_iterations + 1):
        loss, gradient, hessian = logistic_objective(theta, x, y, w, penalty)
        norm = float(np.max(np.abs(gradient)))
        if norm <= spec.gradient_tolerance:
            break
        if iteration == spec.max_iterations:
            raise ValueError("Calibration optimizer did not converge; artifact not exported")
        direction = np.linalg.solve(hessian, gradient)
        step = 1.0
        for _ in range(40):
            trial = theta - step * direction
            candidate_loss = logistic_objective(trial, x, y, w, penalty)[0]
            if math.isfinite(candidate_loss) and candidate_loss <= loss - 1e-4 * step * float(
                gradient @ direction
            ):
                theta = trial
                break
            step *= 0.5
        else:
            raise ValueError("Calibration line search failed; artifact not exported")
    payload = {
        "schema_version": 1,
        "work_package": "WP09.4B",
        "protocol": PROTOCOL,
        "signal_protocol": SIGNAL_PROTOCOL,
        "features": list(FEATURES),
        "context": asdict(context),
        "spec": asdict(spec),
        "parameters": {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "coefficients": theta[1:].tolist(),
            "intercept": float(theta[0]),
        },
        "training": {
            "partition": "calibration_fit",
            "examples_sha256": config_digest([asdict(e) for e in examples]),
            "group_ids": sorted({e.group_id for e in examples}),
            "question_ids": sorted({e.question_id for e in examples}),
            **support,
        },
        "optimizer": {"iterations": iteration, "objective": loss, "gradient_inf_norm": norm},
    }
    payload["fingerprint"] = config_digest(payload)
    return LogisticCalibrator(canonical_json(payload))


@dataclass(frozen=True)
class LogisticCalibrator:
    """Immutable JSON artifact; no per-question labels or references at prediction time."""

    serialized: str

    def __post_init__(self):
        p = self.payload
        if (
            set(p)
            != {
                "schema_version",
                "work_package",
                "protocol",
                "signal_protocol",
                "features",
                "context",
                "spec",
                "parameters",
                "training",
                "optimizer",
                "fingerprint",
            }
            or p["schema_version"] != 1
            or p["work_package"] != "WP09.4B"
            or p["protocol"] != PROTOCOL
            or p["signal_protocol"] != SIGNAL_PROTOCOL
            or p["features"] != list(FEATURES)
            or config_digest({k: v for k, v in p.items() if k != "fingerprint"}) != p["fingerprint"]
        ):
            raise ValueError("Invalid calibration artifact schema/protocol/fingerprint")
        CalibrationContext(**p["context"])
        spec = CalibrationSpec(**p["spec"])
        params, t, opt = p["parameters"], p["training"], p["optimizer"]
        if (
            set(params) != {"center", "scale", "coefficients", "intercept"}
            or any(
                type(params[k]) is not list or len(params[k]) != 4
                for k in ("center", "scale", "coefficients")
            )
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in [
                    params["intercept"],
                    *params["center"],
                    *params["scale"],
                    *params["coefficients"],
                ]
            )
            or any(s <= 0 for s in params["scale"])
            or set(t)
            != {
                "partition",
                "examples_sha256",
                "group_ids",
                "question_ids",
                "rows",
                "unique_questions",
                "groups",
                "positive_groups",
                "negative_groups",
                "positive_rows",
                "negative_rows",
                "eligible",
            }
            or t["partition"] != "calibration_fit"
            or not digest(t["examples_sha256"])
            or any(
                type(t[k]) is not list
                or not t[k]
                or t[k] != sorted(set(t[k]))
                or any(not identifier(v) for v in t[k])
                for k in ("group_ids", "question_ids")
            )
            or any(
                not integer(t[k], 1)
                for k in (
                    "rows",
                    "unique_questions",
                    "groups",
                    "positive_groups",
                    "negative_groups",
                    "positive_rows",
                    "negative_rows",
                )
            )
            or t["groups"] != len(t["group_ids"])
            or t["unique_questions"] != len(t["question_ids"])
            or t["positive_rows"] + t["negative_rows"] != t["rows"]
            or t["eligible"] is not True
            or not spec.min_positive_groups <= t["positive_groups"] <= t["groups"]
            or not spec.min_negative_groups <= t["negative_groups"] <= t["groups"]
            or set(opt) != {"iterations", "objective", "gradient_inf_norm"}
            or not integer(opt["iterations"], 0, spec.max_iterations)
            or any(
                type(opt[k]) not in (int, float) or not math.isfinite(opt[k]) or opt[k] < 0
                for k in ("objective", "gradient_inf_norm")
            )
            or opt["gradient_inf_norm"] > spec.gradient_tolerance
        ):
            raise ValueError("Invalid calibration parameters/training/convergence metadata")

    @property
    def payload(self):
        return json.loads(self.serialized)

    @property
    def fingerprint(self):
        return self._compiled[-1]

    @cached_property
    def _compiled(self):
        # Immutable prediction parameters; payload still returns a fresh audit copy.
        p = self.payload
        params = p["parameters"]
        return (
            CalibrationContext(**p["context"]),
            tuple(params["center"]),
            tuple(params["scale"]),
            tuple(params["coefficients"]),
            params["intercept"],
            p["fingerprint"],
        )

    def predict_features(self, features, *, context):
        bound, center, scale, coefficients, intercept, _ = self._compiled
        if type(context) is not CalibrationContext or context != bound:
            raise ValueError("Calibration context mismatch; synthetic is not model validation")
        if not features or any(type(f) is not FeatureVector for f in features):
            raise ValueError("Prediction accepts observable feature vectors only")
        raw = np.array([f.values for f in features], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            z = ((raw - center) / scale) @ coefficients + intercept
        if not np.isfinite(z).all():
            raise ValueError("Features outside finite calibration numeric range")
        return sigmoid(z).tolist()

    def predict_prefix(self, observation, *, context):
        if type(context) is not CalibrationContext or context != self._compiled[0]:
            raise ValueError("Calibration context mismatch")
        vector = features_from_prefix(observation)
        return self.predict_features([vector], context=context)[0] if vector is not None else None

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise ValueError("Do not overwrite a frozen calibration artifact")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(path, canonical_json(self.payload) + "\n")


def load_calibrator(path, *, expected_fingerprint):
    model = LogisticCalibrator(Path(path).read_text(encoding="utf-8"))
    if not digest(expected_fingerprint) or model.fingerprint != expected_fingerprint:
        raise ValueError("Frozen calibration fingerprint mismatch")
    return model


def require_disjoint_assessment(model, examples):
    example_weights(examples)
    t = model.payload["training"]
    if (
        any(e.partition == "calibration_fit" for e in examples)
        or set(t["group_ids"]) & {e.group_id for e in examples}
        or set(t["question_ids"]) & {e.question_id for e in examples}
    ):
        raise ValueError("Assessment must be group/question-disjoint from calibration fit")
