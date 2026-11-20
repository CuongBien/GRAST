"""Fixed validation subset and observed coverage@K; gold stays outside generation."""

from __future__ import annotations

from collections import Counter

import torch

from llm_scratch.evaluation.capability import left_pad
from llm_scratch.evaluation.pilot import QUESTION_KEYS
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.inference.generate import generate
from llm_scratch.reasoning.answer_extraction import normalize_answer
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.utils.config import config_digest


def balanced_validation(questions, per_difficulty, seed):
    if type(per_difficulty) is not int or per_difficulty < 1:
        raise ValueError("Subset size must be positive")
    if not questions or any(
        set(q) != QUESTION_KEYS or q["split"] != "validation" for q in questions
    ):
        raise ValueError("Coverage selection accepts validation questions only, without gold")
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate validation question IDs")
    selected = []
    for difficulty in ("easy", "medium", "hard"):
        group = [q for q in questions if q["difficulty"] == difficulty]
        if len(group) < per_difficulty:
            raise ValueError("Not enough questions for balanced validation")
        selected.extend(
            sorted(group, key=lambda q: config_digest([seed, q["question_id"]]))[:per_difficulty]
        )
    return selected


def batch_seed(seed, question_id, candidate_start):
    return int(config_digest(["coverage-v1", seed, question_id, candidate_start])[:8], 16)


def sample_candidates(
    model, codec, questions, policy, *, candidates, batch_size, seed, precision, emit
):
    if (
        type(candidates) is not int
        or candidates < 1
        or type(batch_size) is not int
        or batch_size < 1
        or not policy.do_sample
    ):
        raise ValueError("Coverage needs positive candidate/batch counts and sampled generation")
    if not questions or any(
        set(q) != QUESTION_KEYS or q["split"] != "validation" for q in questions
    ):
        raise ValueError("Sampling accepts validation question-only rows")
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate candidate questions")
    predictions = []
    device = model.token_embedding.weight.device
    for index, q in enumerate(questions):
        prefix = encode_reasoning_prompt(codec, q["question"])
        for start in range(0, candidates, batch_size):
            count = min(batch_size, candidates - start)
            ids, mask = left_pad([prefix] * count, codec.pad_token_id, device)
            draw_seed = batch_seed(seed, q["question_id"], start)
            emit("coverage_batch_started", question=index + 1, total=len(questions), start=start)

            def progress(step, token_ids, active):
                if (step + 1) % 32 == 0:
                    emit("coverage_decode_progress", decode_step=step + 1, active=int(active.sum()))

            with torch.autocast(
                device.type, dtype=getattr(torch, precision), enabled=precision != "float32"
            ):
                output = generate(
                    model,
                    ids,
                    mask,
                    config=policy,
                    seed=draw_seed,
                    use_cache=True,
                    on_token=progress,
                )
            for row in range(count):
                tokens = output.generated_ids[row][output.generated_mask[row]].tolist()
                predictions.append(
                    {
                        "question_id": q["question_id"],
                        "question": q["question"],
                        "split": "validation",
                        "difficulty": q["difficulty"],
                        "candidate_index": start + row,
                        "batch_seed": draw_seed,
                        "batch_start": start,
                        "batch_row": row,
                        "generated_ids": tokens,
                        "finish_reason": output.finish_reasons[row],
                        "generated_text": codec.decode(tokens, skip_special_tokens=False),
                    }
                )
            emit("coverage_batch_finished", question=index + 1, candidates_completed=start + count)
    return predictions


def score_coverage(predictions, references, codec, *, candidates):
    if type(candidates) is not int or candidates < 1 or not references:
        raise ValueError("Coverage requires positive candidate count and nonempty references")
    if any(not isinstance(v, str) or normalize_answer(v) != v for v in references.values()):
        raise ValueError("Coverage references must be canonical numeric-v1 strings")
    grouped = {qid: [] for qid in references}
    scored = []
    for p in predictions:
        if p.get("split") != "validation" or p["question_id"] not in grouped:
            raise ValueError("Only complete validation predictions can be scored")
        result = extract_generated_final(p["generated_ids"], codec, p["finish_reason"])
        row = (
            p
            | result
            | {
                "reference": references[p["question_id"]],
                "correct": result["predicted_answer"] == references[p["question_id"]],
            }
        )
        grouped[p["question_id"]].append(row)
        scored.append(row)
    per_question = []
    for qid, rows in grouped.items():
        if (
            len(rows) != candidates
            or any(type(r["candidate_index"]) is not int for r in rows)
            or {r["candidate_index"] for r in rows} != set(range(candidates))
            or len({r["difficulty"] for r in rows}) != 1
        ):
            raise ValueError("Each question needs exactly K distinct candidate indices")
        answers = Counter(r["predicted_answer"] for r in rows if r["answer_valid"])
        per_question.append(
            {
                "question_id": qid,
                "difficulty": rows[0]["difficulty"],
                "correct_candidates": sum(r["correct"] for r in rows),
                "covered": any(r["correct"] for r in rows),
                "valid_candidates": sum(r["answer_valid"] for r in rows),
                "unique_valid_answers": len(answers),
                "answer_counts": dict(answers),
            }
        )

    def summarize(rows):
        qids = {r["question_id"] for r in rows}
        samples = [r for r in scored if r["question_id"] in qids]
        return {
            "questions": len(rows),
            "candidates_per_question": candidates,
            "covered_questions": sum(r["covered"] for r in rows),
            "observed_coverage_at_k": sum(r["covered"] for r in rows) / len(rows),
            "candidates": len(samples),
            "correct_candidates": sum(r["correct"] for r in samples),
            "candidate_accuracy": sum(r["correct"] for r in samples) / len(samples),
            "answer_valid_rate": sum(r["answer_valid"] for r in samples) / len(samples),
            "format_valid_rate": sum(r["format_valid"] for r in samples) / len(samples),
            "generated_tokens": sum(len(r["generated_ids"]) for r in samples),
            "budget_exhausted": sum(r["finish_reason"] == "max_new_tokens" for r in samples),
        }

    return (
        scored,
        per_question,
        {
            "overall": summarize(per_question),
            "by_difficulty": {
                d: summarize([r for r in per_question if r["difficulty"] == d])
                for d in sorted({r["difficulty"] for r in per_question})
            },
        },
    )
