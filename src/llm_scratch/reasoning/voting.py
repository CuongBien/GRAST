"""Canonical-answer plurality; ties go to the first observed answer class."""

from dataclasses import dataclass

from llm_scratch.reasoning.answer_extraction import normalize_answer
from llm_scratch.research.bank_schema import PolicyObservation


@dataclass(frozen=True)
class Vote:
    answer: str | None
    representative_index: int | None
    top_votes: int
    valid_votes: int
    observed_draws: int

    @property
    def share(self):
        return self.top_votes / self.observed_draws if self.observed_draws else 0.0


def majority(observation):
    """Invalid answers abstain but remain in the denominator; format is diagnostic."""
    if not isinstance(observation, PolicyObservation):
        raise ValueError("Voting accepts a policy observation only")
    counts, first = {}, {}
    for index, candidate in enumerate(observation.candidates):
        if candidate.candidate_index != index:
            raise ValueError("Voting requires a contiguous ordered prefix")
        if candidate.answer_valid:
            answer = candidate.extracted_answer
            if not isinstance(answer, str) or normalize_answer(answer) != answer:
                raise ValueError("Eligible votes must be canonical numeric-v1 answers")
            counts[answer] = counts.get(answer, 0) + 1
            first.setdefault(answer, index)
    winner = min(counts, key=lambda a: (-counts[a], first[a])) if counts else None
    return Vote(
        winner,
        first.get(winner),
        counts.get(winner, 0),
        sum(counts.values()),
        len(observation.candidates),
    )
