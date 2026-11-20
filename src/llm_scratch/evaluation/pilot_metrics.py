"""Exact canonical final-answer scoring on all pilot questions, with format diagnostics."""

from __future__ import annotations

from collections import Counter
from statistics import mean

from llm_scratch.reasoning.answer_extraction import extract_answer, normalize_answer


def extract_generated_final(ids, codec, finish_reason):
    opening, closing = codec.special_id("<final>"), codec.special_id("</final>")
    end_think = codec.special_id("</think>")
    raw = codec.decode(ids, skip_special_tokens=False)
    answer = None
    marked = ids.count(opening) == ids.count(closing) == 1 and ids.index(opening) < ids.index(
        closing
    )
    if marked:
        span = ids[ids.index(opening) + 1 : ids.index(closing)]
        candidate = normalize_answer(codec.decode(span, skip_special_tokens=False))
        # Decoded numeric-v1 additionally rejects duplicated literal final markers in rationale.
        if candidate is not None and extract_answer(raw) == candidate:
            answer = candidate
    format_valid = (
        marked
        and ids.count(end_think) == 1
        and 0 < ids.index(end_think) < ids.index(opening)
        and ids.index(end_think) + 1 == ids.index(opening)
        and ids.count(codec.eos_token_id) == 1
        and ids[-1] == codec.eos_token_id
        and ids.index(closing) + 1 == len(ids) - 1
        and finish_reason == "eos"
        and not any(
            t in ids for t in (codec.bos_token_id, codec.pad_token_id, codec.special_id("<think>"))
        )
    )
    return {
        "predicted_answer": answer,
        "format_valid": bool(format_valid),
        "answer_valid": answer is not None,
        "generated_text": raw,
    }


def score_predictions(predictions, references, codec):
    if len({r["question_id"] for r in predictions}) != len(predictions) or (
        {r["question_id"] for r in predictions} != set(references)
    ):
        raise ValueError("Prediction/reference sets must match exactly; no dropped questions")
    scored = []
    for row in predictions:
        ref = references[row["question_id"]]
        if (
            not isinstance(ref, str)
            or normalize_answer(ref) is None
            or normalize_answer(ref) != ref
        ):
            raise ValueError("Reference is not canonical numeric-v1")
        result = extract_generated_final(row["generated_ids"], codec, row["finish_reason"])
        scored.append(
            row | result | {"reference": ref, "correct": result["predicted_answer"] == ref}
        )
    return scored


def _summary(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty pilot")
    count = len(rows)
    return {
        "questions": count,
        "correct": sum(r["correct"] for r in rows),
        "exact_match": sum(r["correct"] for r in rows) / count,
        "answer_valid_rate": sum(r["answer_valid"] for r in rows) / count,
        "format_valid_rate": sum(r["format_valid"] for r in rows) / count,
        "eos_rate": sum(r["finish_reason"] == "eos" for r in rows) / count,
        "budget_exhausted_rate": sum(r["finish_reason"] == "max_new_tokens" for r in rows) / count,
        "mean_generated_tokens": mean(len(r["generated_ids"]) for r in rows),
        "generated_tokens": sum(len(r["generated_ids"]) for r in rows),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
    }


def summarize(scored):
    return {
        "overall": _summary(scored),
        "by_difficulty": {
            difficulty: _summary([r for r in scored if r["difficulty"] == difficulty])
            for difficulty in sorted({r["difficulty"] for r in scored})
        },
    }
