"""Strict gold-free schema and immutable observations for WP08 sample banks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from llm_scratch.evaluation.pilot import QUESTION_KEYS
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.utils.config import config_digest

LOGPROB_PROTOCOL = "fp32-raw-model-log-softmax-natural-log-eos-included-padding-excluded-v1"
BATCHING_PROTOCOL = "one-question-per-batch-private-batch-rng-fixed-layout-v1"
TOKENIZER_FILES = {
    "tokenizer/vocab.json",
    "tokenizer/merges.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer_manifest.json",
}
PAYLOAD_FILES = {"questions.jsonl", "candidates.jsonl"} | TOKENIZER_FILES
CANDIDATE_KEYS = {
    "checkpoint_id",
    "question_id",
    "trial_seed",
    "candidate_index",
    "generation_seed",
    "batch_start",
    "batch_size",
    "batch_row",
    "prompt_ids_sha256",
    "generated_ids",
    "generated_logprobs",
    "generated_text",
    "finish_reason",
    "extracted_answer",
    "answer_valid",
    "format_valid",
    "token_count",
    "logprob_sum",
    "logprob_mean",
}


def integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def identifier(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", value) is not None
    )


def validate_questions(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Bank requires nonempty question-only rows")
    seen, texts = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != QUESTION_KEYS:
            raise ValueError("Bank accepts the question-only schema; gold/unknown fields forbidden")
        if not identifier(row["question_id"]) or row["question_id"] in seen:
            raise ValueError("Invalid/duplicate question identity or text")
        if row["split"] != "validation":
            raise ValueError("WP08 foundation accepts validation only")
        if any(
            not isinstance(row[k], str) or not row[k].strip()
            for k in ("question", "source", "difficulty", "prompt_length")
        ):
            raise ValueError("Question text/source/strata must be nonempty strings")
        if row["question"] in texts:
            raise ValueError("Duplicate question text")
        if any(
            not integer(row[k]) for k in ("operation_count", "reasoning_steps", "prompt_word_count")
        ):
            raise ValueError("Question metadata counts must be nonnegative integers")
        if row["prompt_word_count"] != len(row["question"].split()):
            raise ValueError("Question word-count metadata mismatch")
        seen.add(row["question_id"])
        texts.add(row["question"])


def generation_seed(checkpoint_sha256, question_id, trial_seed, batch_start):
    return int(
        config_digest(["wp08-bank-v1", checkpoint_sha256, question_id, trial_seed, batch_start])[
            :8
        ],
        16,
    )


def validate_grid(candidates, batch_size, trial_seeds):
    if not integer(candidates, 1, 16) or not integer(batch_size, 1, 16):
        raise ValueError("Bank supports1..16 candidates and batch size1..16")
    if (
        not isinstance(trial_seeds, list)
        or not trial_seeds
        or any(not integer(v, 0, 2**32 - 1) for v in trial_seeds)
        or len(set(trial_seeds)) != len(trial_seeds)
    ):
        raise ValueError("Trial seeds must be distinct uint32 integers in declared order")


def describe_output(ids, logprobs, finish_reason, codec):
    if not isinstance(ids, list) or not ids or any(not integer(t, 0, len(codec) - 1) for t in ids):
        raise ValueError("Candidate tokens must be nonempty valid token IDs")
    if (
        not isinstance(logprobs, list)
        or len(ids) != len(logprobs)
        or any(type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in logprobs)
    ):
        raise ValueError("Require one finite nonpositive raw-model logprob per generated token")
    if finish_reason not in ("eos", "max_new_tokens"):
        raise ValueError("Unknown finish reason")
    if (finish_reason == "eos") != (ids[-1] == codec.eos_token_id) or codec.eos_token_id in ids[
        :-1
    ]:
        raise ValueError("EOS must occur exactly at the end of an EOS completion")
    result = extract_generated_final(ids, codec, finish_reason)
    total = math.fsum(logprobs)
    return {
        "generated_text": result["generated_text"],
        "extracted_answer": result["predicted_answer"],
        "answer_valid": result["answer_valid"],
        "format_valid": result["format_valid"],
        "token_count": len(ids),
        "logprob_sum": total,
        "logprob_mean": total / len(ids),
    }


@dataclass(frozen=True)
class CandidateObservation:
    candidate_index: int
    generation_seed: int
    batch_start: int
    batch_row: int
    batch_size: int
    generated_ids: tuple[int, ...]
    generated_logprobs: tuple[float, ...]
    generated_text: str
    finish_reason: str
    extracted_answer: str | None
    answer_valid: bool
    format_valid: bool
    token_count: int
    logprob_sum: float
    logprob_mean: float


@dataclass(frozen=True)
class PolicyObservation:
    checkpoint_id: str
    question_id: str
    question: str
    difficulty: str
    trial_seed: int
    candidates: tuple[CandidateObservation, ...]

    def prefix(self, k):
        """A fresh observation carries only the first k candidates, no future rows."""
        if not integer(k, 0, len(self.candidates)):
            raise ValueError("Prefix must be within the available ordered bank")
        return replace(self, candidates=self.candidates[:k])
