"""Declarative fixed-budget and agreement rules, with no references or future input."""

from dataclasses import dataclass

from llm_scratch.reasoning.voting import majority
from llm_scratch.research.bank_schema import identifier, integer


@dataclass(frozen=True)
class Policy:
    id: str
    kind: str
    min_draws: int
    max_draws: int
    threshold_numerator: int = 1
    threshold_denominator: int = 1

    def __post_init__(self):
        if (
            not identifier(self.id)
            or self.kind not in ("fixed", "agreement")
            or not integer(self.min_draws, 1, 16)
            or not integer(self.max_draws, self.min_draws, 16)
            or not integer(self.threshold_numerator, 1)
            or not integer(self.threshold_denominator, self.threshold_numerator)
        ):
            raise ValueError("Invalid declarative replay policy")
        if self.kind == "fixed" and (
            self.min_draws != self.max_draws
            or self.threshold_numerator != 1
            or self.threshold_denominator != 1
        ):
            raise ValueError("Fixed policies require equal budgets and neutral threshold1/1")


def evaluate_prefix(observation, policy):
    """Only this prefix and a fixed policy spec enter the stopping decision."""
    if not isinstance(policy, Policy):
        raise ValueError("Expected a declarative policy")
    vote = majority(observation)
    n = vote.observed_draws
    if not 1 <= n <= policy.max_draws:
        raise ValueError("Policy must receive a nonempty prefix within its declared budget")
    reason = None
    if policy.kind == "fixed":
        if n == policy.max_draws:
            reason = "fixed_budget"
    elif (
        n >= policy.min_draws
        and vote.answer is not None
        and vote.top_votes * policy.threshold_denominator >= n * policy.threshold_numerator
    ):
        reason = "agreement"
    elif n == policy.max_draws:
        reason = "budget_exhausted"
    return {
        "draws": n,
        "stop": reason is not None,
        "reason": reason,
        "answer": vote.answer,
        "representative_index": vote.representative_index,
        "top_votes": vote.top_votes,
        "valid_votes": vote.valid_votes,
        "vote_share": vote.share,
    }
