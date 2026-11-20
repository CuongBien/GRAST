"""Manifest-verified fixed pilot inputs; reference labels are a separate scorer-only read."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llm_scratch.data.datasets import file_sha256, normalize_text
from llm_scratch.reasoning.answer_extraction import normalize_answer
from llm_scratch.utils.config import config_digest

QUESTION_KEYS = {
    "question_id",
    "question",
    "source",
    "split",
    "difficulty",
    "operation_count",
    "reasoning_steps",
    "prompt_length",
    "prompt_word_count",
}


def load_manifest(root: Path):
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        data.get("schema_version") != 1
        or data.get("work_package") != "WP03"
        or data.get("scope") not in ("full", "math-only")
        or data.get("answer_protocol") != "numeric-v1"
        or config_digest({k: v for k, v in data.items() if k != "dataset_fingerprint"})
        != data.get("dataset_fingerprint")
    ):
        raise ValueError("Need a valid frozen WP03 numeric-v1 manifest")
    return data


def _read(root, manifest, split, role):
    if split not in ("validation", "test") or role not in ("questions", "references"):
        raise ValueError("Only fixed validation/test pilot roles are supported")
    relative = f"pilot/{split}.{role}.jsonl"
    spec = manifest["files"].get(relative, {})
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or spec.get("path") != relative
        or spec.get("role") != role
    ):
        raise ValueError("Invalid pilot role/path")
    if path.stat().st_size != spec["size_bytes"] or file_sha256(path) != spec["sha256"]:
        raise ValueError("Pilot file checksum mismatch")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != spec["records"] or not rows:
        raise ValueError("Pilot record count mismatch")
    if len({r["question_id"] for r in rows}) != len(rows) or any(
        r.get("split") != split for r in rows
    ):
        raise ValueError("Pilot IDs/split mismatch")
    return rows


def load_questions(root: Path, manifest: dict, split: str, *, expected_records: int):
    rows = _read(root, manifest, split, "questions")
    if any(set(r) != QUESTION_KEYS or not normalize_text(r["question"]) for r in rows):
        raise ValueError("Pilot generation inputs must contain question-only fields, never gold")
    spec = manifest["pilot"][split]
    if (
        type(expected_records) is not int
        or len(rows) != expected_records
        or len(rows) != spec["records"]
        or dict(Counter(r["difficulty"] for r in rows)) != spec["difficulty_counts"]
    ):
        raise ValueError("Fixed pilot counts/difficulty strata differ from locked manifest")
    return rows


def load_references(root: Path, manifest: dict, split: str, question_ids: set[str]):
    rows = _read(root, manifest, split, "references")
    if {r["question_id"] for r in rows} != question_ids or any(
        set(r) != {"question_id", "canonical_answer", "split"}
        or not isinstance(r["canonical_answer"], str)
        or normalize_answer(r["canonical_answer"]) is None
        or normalize_answer(r["canonical_answer"]) != r["canonical_answer"]
        for r in rows
    ):
        raise ValueError("Pilot reference IDs/canonical answers mismatch")
    return {r["question_id"]: r["canonical_answer"] for r in rows}


def audit_question_splits(groups):
    seen_ids, seen_text = set(), set()
    for rows in groups:
        for row in rows:
            text = config_digest(normalize_text(row["question"]))
            if row["question_id"] in seen_ids or text in seen_text:
                raise ValueError("Pilot question duplication within/across splits")
            seen_ids.add(row["question_id"])
            seen_text.add(text)
