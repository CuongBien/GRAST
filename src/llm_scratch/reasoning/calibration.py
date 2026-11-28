"""Descriptive raw-agreement reliability; no calibrator fitting or threshold search."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter


def agreement_reliability(rows, edges):
    """Equal question weights, seeds averaged within question, at one prefix size.

    Empty bins are null, not zero-accuracy observations. The rightmost bin includes
    1.0. Repeated seeds and prefixes do not increase the independent question count.
    ECE/Brier remain descriptive even when correctness has only one outcome class.
    """
    if (
        not rows
        or len({r["draws"] for r in rows}) != 1
        or len(edges) < 2
        or edges[0] != 0
        or edges[-1] != 1
        or any(type(e) not in (int, float) or not math.isfinite(e) for e in edges)
        or any(a >= b for a, b in zip(edges[:-1], edges[1:], strict=True))
    ):
        raise ValueError("Reliability requires one nonempty prefix size and ordered bins [0,1]")
    if len({(r["question_id"], r["trial_seed"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate question-seed reliability observation")
    for r in rows:
        if (
            type(r["correct"]) is not bool
            or type(r["agreement"]) not in (float, int)
            or not math.isfinite(r["agreement"])
            or not 0 <= r["agreement"] <= 1
        ):
            raise ValueError("Invalid raw agreement/correctness")
    counts = Counter(r["question_id"] for r in rows)
    weighted = [(r, 1 / (len(counts) * counts[r["question_id"]])) for r in rows]
    bins = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = [
            (r, w)
            for r, w in weighted
            if min(bisect_right(edges, r["agreement"]) - 1, len(edges) - 2) == index
        ]
        mass = math.fsum(w for _, w in selected)
        confidence = math.fsum(w * r["agreement"] for r, w in selected) / mass if mass else None
        accuracy = math.fsum(w * r["correct"] for r, w in selected) / mass if mass else None
        bins.append(
            {
                "lower": lo,
                "upper": hi,
                "upper_inclusive": index == len(edges) - 2,
                "observations": len(selected),
                "unique_questions": len({r["question_id"] for r, _ in selected}),
                "question_weight_mass": mass,
                "mean_agreement": confidence,
                "accuracy": accuracy,
                "absolute_gap": abs(confidence - accuracy) if mass else None,
            }
        )
    return {
        "unique_questions": len(counts),
        "observations": len(rows),
        "accuracy": math.fsum(w * r["correct"] for r, w in weighted),
        "mean_agreement": math.fsum(w * r["agreement"] for r, w in weighted),
        "raw_agreement_ece": math.fsum(
            b["question_weight_mass"] * b["absolute_gap"] for b in bins if b["observations"]
        ),
        "raw_agreement_brier": math.fsum(
            w * (r["agreement"] - r["correct"]) ** 2 for r, w in weighted
        ),
        "correctness_classes": len({r["correct"] for r in rows}),
        "discrimination_status": (
            "undefined_single_outcome_class"
            if len({r["correct"] for r in rows}) == 1
            else "not_estimated_in_diagnostic_phase"
        ),
        "roc_auc": None,
        "bins": bins,
    }
