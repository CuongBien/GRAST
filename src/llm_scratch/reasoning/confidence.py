"""Gold-free current-prefix signals; none are calibrated correctness probabilities."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean

from llm_scratch.reasoning.voting import majority
from llm_scratch.research.bank_schema import PolicyObservation

PROTOCOL = "prefix-answer-entropy-invalid-bucket-raw-model-logprobs-v1"


def prefix_signals(observation):
    """Consume only the supplied prefix, without references or future completions.

    Entropy uses canonical answer classes plus one invalid bucket, divided by all
    draws. Normalize by log(draws), not log(observed classes). Invalid abstains from
    voting even though it contributes to entropy. Logprobs include generated EOS
    and reasoning/format tokens; they are not probabilities of a correct answer.
    """
    if type(observation) is not PolicyObservation or not observation.candidates:
        raise ValueError("Signals require a nonempty policy prefix")
    vote = majority(observation)
    candidates = observation.candidates
    for c in candidates:
        if (
            type(c.token_count) is not int
            or c.token_count <= 0
            or len(c.generated_ids) != c.token_count
            or len(c.generated_logprobs) != c.token_count
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or p > 0
                for p in c.generated_logprobs
            )
        ):
            raise ValueError("Invalid current-prefix token logprobs")
        total = math.fsum(c.generated_logprobs)
        if not math.isclose(c.logprob_sum, total, rel_tol=1e-12, abs_tol=1e-8) or not math.isclose(
            c.logprob_mean, total / c.token_count, rel_tol=1e-12, abs_tol=1e-8
        ):
            raise ValueError("Inconsistent current-prefix logprob summaries")
    n = vote.observed_draws
    counts = Counter(c.extracted_answer if c.answer_valid else None for c in candidates)
    entropy = -math.fsum((v / n) * math.log(v / n) for v in counts.values())
    winners = [c for c in candidates if c.answer_valid and c.extracted_answer == vote.answer]
    return {
        "draws": n,
        "answer": vote.answer,
        "representative_index": vote.representative_index,
        "valid_answers": vote.valid_votes,
        "invalid_answers": n - vote.valid_votes,
        "valid_fraction": vote.valid_votes / n,
        "agreement": vote.share,
        "answer_entropy_nats": max(0.0, entropy),
        "answer_entropy_normalized": max(0.0, entropy) / math.log(n) if n > 1 else 0.0,
        "observed_generated_tokens": sum(c.token_count for c in candidates),
        "mean_sequence_logprob_sum": mean(c.logprob_sum for c in candidates),
        "mean_candidate_logprob_mean": mean(c.logprob_mean for c in candidates),
        "pooled_token_logprob_mean": math.fsum(c.logprob_sum for c in candidates)
        / sum(c.token_count for c in candidates),
        "winner_member_logprob_mean": mean(c.logprob_mean for c in winners) if winners else None,
        "representative_logprob_mean": (
            candidates[vote.representative_index].logprob_mean if winners else None
        ),
    }
