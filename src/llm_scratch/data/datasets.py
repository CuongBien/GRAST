"""Pinned-file downloads, bounded Parquet selection and label-independent split/dedup."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

from llm_scratch.reasoning.answer_extraction import extract_answer
from llm_scratch.utils.config import config_digest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_fingerprint(row: dict) -> str:
    return config_digest(normalize_text(row.get("question", row.get("text", ""))))


def download_source(spec: dict, cache_root: Path, *, offline: bool = False) -> Path:
    """Fetch an entire pinned source and verify SHA-256, including on cache hits."""
    digest = spec["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Source requires a full SHA-256")
    if (
        not spec["url"].startswith("https://")
        or type(spec["size_bytes"]) is not int
        or spec["size_bytes"] <= 0
    ):
        raise ValueError("Source requires HTTPS and a positive byte size")
    path = cache_root / "raw" / digest / "source"
    if path.is_file():
        if path.stat().st_size != spec["size_bytes"] or file_sha256(path) != digest:
            raise ValueError(f"Corrupt source cache: {path}")
        return path
    if offline:
        raise FileNotFoundError(f"Offline source cache miss: {digest}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
    try:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "GRAST-WP03/1"})
        total = 0
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=60) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                total += len(block)
                if total > spec["size_bytes"]:
                    raise ValueError("Source exceeded its pinned byte size")
                output.write(block)
        if total != spec["size_bytes"] or file_sha256(temporary) != digest:
            raise ValueError(f"Source checksum/size mismatch: {spec['url']}")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def read_stories(path: Path, limit: int, upstream_split: str) -> list[dict]:
    import pyarrow.parquet as pq

    if type(limit) is not int or limit < 1:
        raise ValueError("Story limit must be positive")
    rows = []
    source_index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
        for text in batch.column(0).to_pylist():
            index = source_index
            source_index += 1
            if not isinstance(text, str) or not normalize_text(text):
                continue
            rows.append(
                {
                    "question_id": "tinystories-" + config_digest(normalize_text(text)),
                    "source": "tinystories",
                    "upstream_split": upstream_split,
                    "source_index": index,
                    "text": text.strip(),
                }
            )
            if len(rows) == limit:
                return rows
    if len(rows) < limit:
        raise ValueError(f"Only {len(rows)} stories available; requested {limit}")
    return rows


def read_gsm8k(path: Path, upstream_split: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            if not isinstance(item.get("question"), str) or not normalize_text(item["question"]):
                raise ValueError(f"Invalid GSM8K question at {index}")
            answer = extract_answer(item.get("answer"))
            if answer is None:
                raise ValueError(f"Invalid GSM8K final answer at {upstream_split}:{index}")
            rows.append(
                {
                    "question_id": "gsm8k-" + config_digest(normalize_text(item["question"])),
                    "question": item["question"],
                    "answer": item["answer"],
                    "canonical_answer": answer,
                    "source": "gsm8k",
                    "upstream_split": upstream_split,
                    "source_index": index,
                    "difficulty": "ungraded",
                }
            )
    return rows


def deduplicate(rows: list[dict], *, excluded: set[str] | None = None) -> tuple[list[dict], dict]:
    """Dedup NFKC/case/whitespace text; a held-out exclusion set has priority."""
    seen, kept = {}, []
    report = {"input": len(rows), "within_source_duplicates": 0, "heldout_overlap_removed": 0}
    for row in rows:
        key = text_fingerprint(row)
        if excluded and key in excluded:
            report["heldout_overlap_removed"] += 1
            continue
        if key in seen:
            if seen[key].get("canonical_answer") != row.get("canonical_answer"):
                raise ValueError("Duplicate question has conflicting canonical answers")
            report["within_source_duplicates"] += 1
            continue
        seen[key] = row
        kept.append(row)
    report["output"] = len(kept)
    return sorted(kept, key=lambda row: row["question_id"]), report


def split_rows(
    rows: list[dict], counts: dict[str, int], seed: int, namespace: str
) -> dict[str, list[dict]]:
    """Rank stable IDs by hash; neither answers nor input ordering affect assignment."""
    if sum(counts.values()) != len(rows) or any(
        type(n) is not int or n < 0 for n in counts.values()
    ):
        raise ValueError("Split counts must be nonnegative integers summing to input size")
    if len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate stable IDs before splitting")
    ranked = sorted(rows, key=lambda row: config_digest([seed, namespace, row["question_id"]]))
    result, start = {}, 0
    for split in ("train", "validation", "test"):
        count = counts.get(split, 0)
        result[split] = sorted(
            [{**row, "split": split} for row in ranked[start : start + count]],
            key=lambda row: row["question_id"],
        )
        start += count
    if start != len(rows):
        raise ValueError("Unknown split name")
    return result


def audit_splits(datasets: dict[str, dict[str, list[dict]]]) -> dict:
    """Check exact-normalized text and procedural semantic IDs across all sources/splits."""
    text_sets = {split: set() for split in ("train", "validation", "test")}
    semantic_sets = {split: set() for split in text_sets}
    ids = set()
    for splits in datasets.values():
        for split, rows in splits.items():
            for row in rows:
                if row.get("split") != split:
                    raise ValueError("Record split label does not match its output partition")
                if row["question_id"] in ids:
                    raise ValueError("Repeated stable ID across datasets/splits")
                ids.add(row["question_id"])
                text_sets[split].add(text_fingerprint(row))
                if "semantic_id" in row:
                    semantic_sets[split].add(row["semantic_id"])
    overlaps = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{first}/{second}"] = {
            "normalized_text": len(text_sets[first] & text_sets[second]),
            "procedural_semantic_id": len(semantic_sets[first] & semantic_sets[second]),
        }
    if any(any(values.values()) for values in overlaps.values()):
        raise ValueError(f"Cross-split leakage: {overlaps}")
    return {
        "normalization": "NFKC + casefold + whitespace-v1",
        "overlaps": overlaps,
        "passed": True,
        "limitations": (
            "Exact-normalized text and +/* associative/commutative tree grouping; "
            "not semantic paraphrase detection"
        ),
    }
