"""Prefix-only threshold policies and an explicit online CGES-style adaptation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from llm_scratch.reasoning.calibrators import LogisticCalibrator
from llm_scratch.reasoning.confidence import prefix_signals
from llm_scratch.reasoning.voting import majority
from llm_scratch.research.bank_schema import digest, identifier, integer

PROTOCOL = "prefix-threshold-cges-fixed-capacity-unseen-slots-v1"


@dataclass(frozen=True)
class ConfidencePolicy:
    id: str
    kind: str
    min_draws: int
    max_draws: int
    threshold: float
    capacity: int = 17
    epsilon: float = 1e-6
    calibrator_fingerprint: str | None = None

    def __post_init__(self):
        if (
            not identifier(self.id)
            or self.kind not in {"raw_threshold", "calibrated_threshold", "cges_lns"}
            or not integer(self.min_draws, 2, 16)
            or not integer(self.max_draws, self.min_draws, 16)
            or type(self.threshold) not in (int, float)
            or not math.isfinite(self.threshold)
            or not 0 < self.threshold < 1
            or not integer(self.capacity, self.max_draws + 1, 1024)
            or type(self.epsilon) not in (int, float)
            or not math.isfinite(self.epsilon)
            or not 0 < self.epsilon < 0.5
            or (
                not digest(self.calibrator_fingerprint)
                if self.kind == "calibrated_threshold"
                else self.calibrator_fingerprint is not None
            )
        ):
            raise ValueError("Invalid bounded confidence policy")


def cges_summary(observation, capacity, epsilon):
    """Fixed anonymous hypotheses remain in normalization; no future answer set.

    Factor out the likelihood shared by all hypotheses. Each observed answer gains
    log(c*(M-1)/(1-c)) for its own responses. Every unseen slot retains score zero.
    Response confidence c is exp(mean raw token logprob), clipped for finite logs.
    This is an uncalibrated pseudo-posterior, not a correctness guarantee.
    """
    prefix_signals(observation)  # Validate observed tokens/logprobs and canonical answer order.
    if not integer(capacity, len(observation.candidates) + 1, 1024) or not 0 < epsilon < 0.5:
        raise ValueError("CGES capacity must retain unseen slots throughout this prefix")
    scores, first = {}, {}
    for c in observation.candidates:
        if c.answer_valid:
            a = c.extracted_answer
            p = min(1 - epsilon, max(epsilon, math.exp(c.logprob_mean)))
            scores[a] = scores.get(a, 0.0) + math.log(p) + math.log(capacity - 1) - math.log1p(-p)
            first.setdefault(a, c.candidate_index)
    unseen = capacity - len(scores)
    if not scores:
        return {
            "answer": None,
            "representative_index": None,
            "confidence": None,
            "unseen_mass": 1.0,
            "observed_classes": 0,
        }
    peak = max(0.0, max(scores.values()))
    denominator = math.fsum(math.exp(s - peak) for s in scores.values()) + unseen * math.exp(-peak)
    answer = min(scores, key=lambda a: (-scores[a], first[a]))
    return {
        "answer": answer,
        "representative_index": first[answer],
        "confidence": math.exp(scores[answer] - peak) / denominator,
        "unseen_mass": unseen * math.exp(-peak) / denominator,
        "observed_classes": len(scores),
    }


def evaluate_confidence_prefix(observation, policy, *, calibrator=None, context=None):
    if type(policy) is not ConfidencePolicy:
        raise ValueError("Expected a frozen confidence policy")
    vote = majority(observation)
    n = vote.observed_draws
    if not 1 <= n <= policy.max_draws:
        raise ValueError("Policy requires a nonempty bounded prefix")
    extra = {}
    if policy.kind == "calibrated_threshold":
        if (
            type(calibrator) is not LogisticCalibrator
            or calibrator.fingerprint != policy.calibrator_fingerprint
        ):
            raise ValueError("Frozen calibrator missing or fingerprint mismatch")
        score = calibrator.predict_prefix(observation, context=context)
        answer, index = vote.answer, vote.representative_index
        score_kind = "calibrated_prefix_winner"
    else:
        if calibrator is not None:
            raise ValueError("Raw/CGES policies do not accept a prefix calibrator")
        if policy.kind == "raw_threshold":
            score, answer, index = (
                vote.share if vote.answer is not None else None,
                vote.answer,
                vote.representative_index,
            )
            score_kind = "raw_agreement"
        else:
            state = cges_summary(observation, policy.capacity, policy.epsilon)
            score, answer, index = (
                state["confidence"],
                state["answer"],
                state["representative_index"],
            )
            extra = {
                "unseen_mass": state["unseen_mass"],
                "observed_classes": state["observed_classes"],
            }
            score_kind = "cges_lns_pseudo_posterior"
    hit = (
        n >= policy.min_draws
        and observation.candidates[-1].answer_valid
        and answer is not None
        and score is not None
        and score >= policy.threshold
    )
    reason = (
        "confidence_threshold" if hit else ("budget_exhausted" if n == policy.max_draws else None)
    )
    return {
        "draws": n,
        "stop": reason is not None,
        "reason": reason,
        "answer": answer,
        "representative_index": index,
        "confidence": score,
        "score_kind": score_kind,
        **extra,
    }


def replay_confidence(group, policy, *, calibrator=None, context=None):
    if len(group.candidates) < policy.max_draws:
        raise ValueError("Bank is too short for confidence policy")
    history = []
    for n in range(1, policy.max_draws + 1):
        prefix = group.prefix(n)
        step = evaluate_confidence_prefix(prefix, policy, calibrator=calibrator, context=context)
        history.append(step)
        if step["stop"]:
            return {
                "checkpoint_id": group.checkpoint_id,
                "question_id": group.question_id,
                "difficulty": group.difficulty,
                "trial_seed": group.trial_seed,
                "policy_id": policy.id,
                "draws": n,
                "answer": step["answer"],
                "representative_index": step["representative_index"],
                "stop_reason": step["reason"],
                "observed_generated_tokens": sum(c.token_count for c in prefix.candidates),
                "history": history,
            }
    raise RuntimeError("Bounded confidence policy did not terminate")
