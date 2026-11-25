"""Scorer-only observed prefix coverage; no policy or confidence fitting."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from llm_scratch.data.sft import math_reasoning


def trace_matches(candidate, expression, codec):
    """Conservative exact canonical-trace check; other valid derivations stay unverified."""
    end = codec.special_id("</think>")
    ids = candidate.generated_ids
    if ids.count(end) != 1:
        return False
    rationale = codec.decode(ids[: ids.index(end)], skip_special_tokens=False)
    return " ".join(rationale.split()) == " ".join(math_reasoning(expression).split())


def summarize_bank(bank, labels, expressions, codec):
    groups = bank.observations()
    expected = {
        (g.checkpoint_id, g.question_id, g.trial_seed, c.candidate_index)
        for g in groups
        for c in g.candidates
    }
    keys = [
        (r["checkpoint_id"], r["question_id"], r["trial_seed"], r["candidate_index"])
        for r in labels
    ]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Scored labels must cover the exact ordered bank grid")
    correct = {k: r["correct"] for k, r in zip(keys, labels, strict=True)}
    if any(type(v) is not bool for v in correct.values()):
        raise ValueError("Correctness labels must be boolean")
    if set(expressions) != {g.question_id for g in groups}:
        raise ValueError("Trace expressions must cover the exact pilot")
    audits, per_question = [], []
    prefixes = [k for k in (1, 2, 4, 8, 16) if k <= bank.manifest["candidates_per_question"]]
    if bank.manifest["candidates_per_question"] not in prefixes:
        prefixes.append(bank.manifest["candidates_per_question"])
    for group in groups:
        rows = []
        for c in group.candidates:
            identity = (group.checkpoint_id, group.question_id, group.trial_seed, c.candidate_index)
            trace = trace_matches(c, expressions[group.question_id], codec)
            row = {
                "checkpoint_id": group.checkpoint_id,
                "question_id": group.question_id,
                "trial_seed": group.trial_seed,
                "candidate_index": c.candidate_index,
                "correct": correct[identity],
                "canonical_trace_match": trace,
                "verified_correct": correct[identity] and trace and c.format_valid,
                "format_valid": c.format_valid,
                "answer_valid": c.answer_valid,
                "finish_reason": c.finish_reason,
                "token_count": c.token_count,
            }
            rows.append(row)
            audits.append(row)
        for k in prefixes:
            prefix = rows[:k]
            per_question.append(
                {
                    "checkpoint_id": group.checkpoint_id,
                    "question_id": group.question_id,
                    "trial_seed": group.trial_seed,
                    "difficulty": group.difficulty,
                    "k": k,
                    "covered": any(r["correct"] for r in prefix),
                    "verified_covered": any(r["verified_correct"] for r in prefix),
                    "correct_candidates": sum(r["correct"] for r in prefix),
                    "verified_correct_candidates": sum(r["verified_correct"] for r in prefix),
                    "answer_valid_candidates": sum(r["answer_valid"] for r in prefix),
                    "format_valid_candidates": sum(r["format_valid"] for r in prefix),
                    "budget_exhausted_candidates": sum(
                        r["finish_reason"] == "max_new_tokens" for r in prefix
                    ),
                    "generated_tokens": sum(r["token_count"] for r in prefix),
                }
            )

    def aggregate(rows):
        n = len(rows)
        total = sum(r["k"] for r in rows)
        covered = sum(r["covered"] for r in rows)
        verified = sum(r["verified_covered"] for r in rows)
        return {
            "question_trials": n,
            "covered_question_trials": covered,
            "observed_prefix_coverage": covered / n,
            "verified_covered_question_trials": verified,
            "verified_prefix_coverage": verified / n,
            "candidates": total,
            "correct_candidates": sum(r["correct_candidates"] for r in rows),
            "verified_correct_candidates": sum(r["verified_correct_candidates"] for r in rows),
            "candidate_accuracy": sum(r["correct_candidates"] for r in rows) / total,
            "format_valid_rate": sum(r["format_valid_candidates"] for r in rows) / total,
            "answer_valid_rate": sum(r["answer_valid_candidates"] for r in rows) / total,
            "budget_exhausted_candidates": sum(r["budget_exhausted_candidates"] for r in rows),
            "generated_tokens": sum(r["generated_tokens"] for r in rows),
        }

    scopes = ["all", *sorted({g.difficulty for g in groups})]
    checkpoints = []
    for cp in bank.manifest["checkpoints"]:
        seeds = []
        for seed in bank.manifest["trial_seeds"]:
            selected = [
                r
                for r in per_question
                if r["checkpoint_id"] == cp["id"] and r["trial_seed"] == seed
            ]
            seeds.append(
                {
                    "seed": seed,
                    "scopes": {
                        scope: {
                            str(k): aggregate(
                                [
                                    r
                                    for r in selected
                                    if r["k"] == k and (scope == "all" or r["difficulty"] == scope)
                                ]
                            )
                            for k in prefixes
                        }
                        for scope in scopes
                    },
                }
            )
        checkpoints.append(
            {
                "checkpoint_id": cp["id"],
                "seeds": seeds,
                "seed_summary": {
                    scope: {
                        str(k): {
                            "coverage_mean": mean(
                                s["scopes"][scope][str(k)]["observed_prefix_coverage"]
                                for s in seeds
                            ),
                            "coverage_min": min(
                                s["scopes"][scope][str(k)]["observed_prefix_coverage"]
                                for s in seeds
                            ),
                            "coverage_max": max(
                                s["scopes"][scope][str(k)]["observed_prefix_coverage"]
                                for s in seeds
                            ),
                            "verified_coverage_mean": mean(
                                s["scopes"][scope][str(k)]["verified_prefix_coverage"]
                                for s in seeds
                            ),
                        }
                        for k in prefixes
                    }
                    for scope in scopes
                },
            }
        )
    return (
        audits,
        per_question,
        {
            "checkpoints": checkpoints,
            "prefixes": prefixes,
            "unique_questions": len({g.question_id for g in groups}),
            "question_difficulty_counts": dict(
                Counter(
                    g.difficulty
                    for g in groups
                    if g.checkpoint_id == groups[0].checkpoint_id
                    and g.trial_seed == groups[0].trial_seed
                )
            ),
            "interpretation": (
                "Observed prefix coverage, not an unbiased pass@k estimator; seeds reuse "
                "the same questions. No confidence fitting or benchmark decision."
            ),
            "trace_protocol": (
                "Whitespace-normalized exact WP03 canonical rationale plus valid format "
                "and correct final; other derivations are unverified, not proven wrong."
            ),
        },
    )
