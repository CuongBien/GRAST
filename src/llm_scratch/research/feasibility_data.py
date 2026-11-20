"""Authored paired operand-copy/arithmetic data, independent of the WP03 pilot."""

from __future__ import annotations

import random
import re

from llm_scratch.evaluation.pilot import QUESTION_KEYS
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.utils.config import config_digest


def make_tasks(train_pairs=32, validation_pairs=8, seed=42):
    """Each pair supplies one copy task and one arithmetic task; no pilot data read.

    Operands 40..99 cannot coincide with WP03's one-operation easy tasks (1..30).
    Authored validation is a disjoint pair holdout, not the WP03 pilot validation.
    """
    if any(type(n) is not int or n < 2 or n % 2 for n in (train_pairs, validation_pairs)):
        raise ValueError("Need positive even pair counts >=2")
    if train_pairs + validation_pairs > 3600:
        raise ValueError("Authored pair space exhausted")
    pool = [(a, b) for a in range(40, 100) for b in range(40, 100)]
    random.Random(seed).shuffle(pool)
    result = {}
    offset = 0
    for split, count in (("train", train_pairs), ("validation", validation_pairs)):
        rows = []
        for i, (a, b) in enumerate(pool[offset : offset + count]):
            op = "+" if i % 2 == 0 else "-"
            answer = a + b if op == "+" else a - b
            operation = "sum" if op == "+" else "difference"
            for task in ("copy", "arithmetic"):
                if task == "copy":
                    question = (
                        f"Copy both operands in order: left = {a}, right = {b}. "
                        "Return the two numbers separated by a comma; do not calculate."
                    )
                    reasoning, final = f"The operands are {a} and {b}.", f"{a},{b}"
                else:
                    question = (
                        f"R1 is the {operation} of {a} and {b}. "
                        "What is R1? Give only the numeric answer."
                    )
                    reasoning, final = f"R1 = {a} {op} {b} = {answer}.", str(answer)
                qid = f"authored-{split}-{task}-{config_digest([a, b, op])[:16]}"
                rows.append(
                    {
                        "question_id": qid,
                        "question": question,
                        "source": "grast-feasibility-authored-v1",
                        "split": split,
                        "difficulty": task if task == "copy" else f"arithmetic-{op}",
                        "operation_count": int(task == "arithmetic"),
                        "reasoning_steps": 1,
                        "prompt_length": "short",
                        "prompt_word_count": len(question.split()),
                        "task": task,
                        "operands": [a, b],
                        "operator": op,
                        "reasoning": reasoning,
                        "final_answer": final,
                    }
                )
        result[split] = rows
        offset += count
    return result


def question_only(rows):
    return [{k: row[k] for k in QUESTION_KEYS} for row in rows]


def _copy_answer(ids, codec):
    opening, closing = codec.special_id("<final>"), codec.special_id("</final>")
    text = codec.decode(ids, skip_special_tokens=False)
    if (
        ids.count(opening) != 1
        or ids.count(closing) != 1
        or ids.index(opening) >= ids.index(closing)
        or text.count("<final>") != 1
        or text.count("</final>") != 1
    ):
        return None
    span = codec.decode(ids[ids.index(opening) + 1 : ids.index(closing)])
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", span)
    return None if match is None else [int(match[1]), int(match[2])]


def score_tasks(predictions, references, codec):
    refs = {r["question_id"]: r for r in references}
    if (
        len(refs) != len(references)
        or len(predictions) != len(refs)
        or ({p["question_id"] for p in predictions} != set(refs))
    ):
        raise ValueError("Diagnostic prediction/reference sets must match without drops")
    scored = []
    for pred in predictions:
        ref = refs[pred["question_id"]]
        result = extract_generated_final(pred["generated_ids"], codec, pred["finish_reason"])
        copy = _copy_answer(pred["generated_ids"], codec) if ref["task"] == "copy" else None
        # This is a strict first-equation diagnostic, not a general CoT verifier.
        trace = result["generated_text"].split("</think>", 1)[0]
        equation = re.match(r"\s*R1 = (-?\d+) ([+-]) (-?\d+) = (-?\d+)\.", trace)
        operands_correct = equation_correct = False
        if equation is not None:
            a, op, b, value = equation.groups()
            a, b, value = int(a), int(b), int(value)
            operands_correct = [a, b] == ref["operands"] and op == ref["operator"]
            equation_correct = (a + b if op == "+" else a - b) == value
        correct = (
            copy == ref["operands"]
            if ref["task"] == "copy"
            else result["predicted_answer"] == ref["final_answer"]
        )
        scored.append(
            pred
            | result
            | {
                "task": ref["task"],
                "reference": ref["final_answer"],
                "copy_pair": copy,
                "correct": correct,
                "answer_valid": copy is not None
                if ref["task"] == "copy"
                else result["answer_valid"],
                "first_equation_parsed": equation is not None,
                "operands_and_operator_correct": operands_correct,
                "first_equation_arithmetic_correct": equation_correct,
            }
        )
    summary = {}
    for task in ("copy", "arithmetic"):
        rows = [r for r in scored if r["task"] == task]
        if not rows:
            raise ValueError("Both copy and arithmetic tasks are required")
        n = len(rows)
        summary[task] = {
            "questions": n,
            "correct": sum(r["correct"] for r in rows),
            "exact_match": sum(r["correct"] for r in rows) / n,
            "answer_valid_rate": sum(r["answer_valid"] for r in rows) / n,
            "format_valid_rate": sum(r["format_valid"] for r in rows) / n,
            "operands_and_operator_correct": sum(r["operands_and_operator_correct"] for r in rows)
            if task == "arithmetic"
            else None,
            "first_equation_arithmetic_correct": sum(
                r["first_equation_arithmetic_correct"] for r in rows
            )
            if task == "arithmetic"
            else None,
        }
    return scored, summary
