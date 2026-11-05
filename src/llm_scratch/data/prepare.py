"""Create deterministic dataset artifacts, held-out pilots and a portable manifest."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

from llm_scratch.data.datasets import (
    audit_splits,
    deduplicate,
    download_source,
    file_sha256,
    read_gsm8k,
    read_stories,
    split_rows,
    text_fingerprint,
)
from llm_scratch.data.difficulty import generate_math
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run


def _write_rows(root: Path, relative: str, rows: list[dict], role: str) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(rows, key=lambda item: item["question_id"]):
            line = canonical_json(row)
            # Preserve decoded Unicode, but keep splitlines()-based JSONL readers safe.
            for separator in ("\u0085", "\u2028", "\u2029"):
                line = line.replace(separator, f"\\u{ord(separator):04x}")
            stream.write(line + "\n")
    return {
        "path": relative,
        "records": len(rows),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "role": role,
    }


def _prepare_external(data: dict, run: Run, offline: bool) -> tuple[dict, dict]:
    cache, seed = Path(data["artifacts"]["cache_root"]), data["reproducibility"]["seed"]
    paths = {}
    for name in ("tinystories", "gsm8k"):
        paths[name] = {}
        for split, spec in data["sources"][name]["files"].items():
            run.log(
                "source_verification_started",
                dataset=name,
                upstream_split=split,
                sha256=spec["sha256"],
            )
            paths[name][split] = download_source(spec, cache, offline=offline)
            run.log("source_verified", dataset=name, upstream_split=split)
    tiny_train, train_report = deduplicate(
        read_stories(paths["tinystories"]["train"], data["tinystories"]["train_records"], "train")
    )
    tiny_heldout, held_report = deduplicate(
        read_stories(
            paths["tinystories"]["validation"], data["tinystories"]["heldout_records"], "validation"
        )
    )
    # Official held-out content wins; training text is removed on overlap.
    tiny_train, tiny_overlap = deduplicate(
        tiny_train, excluded={text_fingerprint(row) for row in tiny_heldout}
    )
    if len(tiny_heldout) < 2 or not tiny_train:
        raise ValueError("TinyStories has insufficient distinct records")
    tiny = split_rows(
        tiny_heldout,
        {"validation": len(tiny_heldout) // 2, "test": len(tiny_heldout) - len(tiny_heldout) // 2},
        seed,
        "tinystories-heldout-v1",
    )
    tiny["train"] = [{**row, "split": "train"} for row in tiny_train]
    gsm_test, gsm_test_report = deduplicate(read_gsm8k(paths["gsm8k"]["test"], "test"))
    gsm_train, gsm_train_report = deduplicate(
        read_gsm8k(paths["gsm8k"]["train"], "train"),
        excluded={text_fingerprint(row) for row in gsm_test},
    )
    val_count = data["gsm8k"]["validation_records"]
    if val_count >= len(gsm_train):
        raise ValueError("GSM8K validation must leave training records")
    gsm = split_rows(
        gsm_train,
        {"train": len(gsm_train) - val_count, "validation": val_count},
        seed,
        "gsm8k-train-v1",
    )
    gsm["test"] = [{**row, "split": "test"} for row in gsm_test]
    reports = {
        "tinystories": {
            "train": train_report,
            "heldout": held_report,
            "train_vs_heldout": tiny_overlap,
        },
        "gsm8k": {"train": gsm_train_report, "test": gsm_test_report},
    }
    return {"tinystories": tiny, "gsm8k": gsm}, reports


def prepare_data(config: LoadedConfig, *, offline: bool = False, math_only: bool = False) -> Path:
    data = config.data
    seed, math = data["reproducibility"]["seed"], data["math"]
    with Run(config) as run:
        run.metadata["data_options"] = {"offline": offline, "math_only": math_only}
        run.metadata["execution_device"] = "cpu"
        # Inventory records installed PyTorch/CUDA/GPU; data preparation does not require them.
        run.metadata["runtime_inventory"] = inspect_environment("cpu")
        write_json(run.path / "metadata.json", run.metadata)
        write_json(run.path / "config_sources" / "sources.lock.json", data["sources"])
        datasets, dedup = ({}, {}) if math_only else _prepare_external(data, run, offline)
        counts = {split: math[f"{split}_per_level"] for split in ("train", "validation", "test")}
        questions = generate_math(math["levels"], sum(counts.values()), seed)
        math_splits = {split: [] for split in counts}
        for level in math["levels"]:
            rows = [row for row in questions if row["difficulty"] == level["name"]]
            splits = split_rows(rows, counts, seed, "arithmetic-split-v1")
            for split, items in splits.items():
                math_splits[split].extend(items)
        datasets["math"] = math_splits
        leakage = audit_splits(datasets)
        root = run.path / "artifacts" / "datasets"
        files = {}
        for name, splits in datasets.items():
            for split, rows in splits.items():
                relative = f"{name}/{split}.jsonl"
                files[relative] = _write_rows(
                    root,
                    relative,
                    rows,
                    "pretraining" if name == "tinystories" else "training_or_evaluation",
                )
        pilot = {}
        for split in ("validation", "test"):
            selected = []
            for level in math["levels"]:
                candidates = [
                    row for row in math_splits[split] if row["difficulty"] == level["name"]
                ]
                candidates.sort(
                    key=lambda row: config_digest([seed, "pilot-v1", split, row["question_id"]])
                )
                selected.extend(candidates[: math["pilot_per_level_per_split"]])
            prompts = [
                {
                    key: row[key]
                    for key in (
                        "question_id",
                        "question",
                        "source",
                        "split",
                        "difficulty",
                        "operation_count",
                        "reasoning_steps",
                        "prompt_length",
                        "prompt_word_count",
                    )
                }
                for row in selected
            ]
            references = [
                {
                    "question_id": row["question_id"],
                    "canonical_answer": row["canonical_answer"],
                    "split": split,
                }
                for row in selected
            ]
            for role, rows in (("questions", prompts), ("references", references)):
                relative = f"pilot/{split}.{role}.jsonl"
                files[relative] = _write_rows(root, relative, rows, role)
            pilot[split] = {
                "records": len(selected),
                "difficulty_counts": dict(
                    sorted(Counter(row["difficulty"] for row in selected).items())
                ),
            }
        portable = copy.deepcopy(data)
        portable["artifacts"] = {"root": "<run-root>", "cache_root": "<cache-root>"}
        manifest = {
            "schema_version": 1,
            "work_package": "WP03",
            "scope": "math-only" if math_only else "full",
            "data_config_sha256": config_digest(portable),
            "seed": seed,
            "generator_version": "arithmetic-v1",
            "answer_protocol": "numeric-v1",
            "sources": {"math": "grast-generated"} if math_only else data["sources"],
            "levels": math["levels"],
            "files": files,
            "pilot": pilot,
            "split_policy": {
                "tinystories": (
                    "Official validation prefix split into internal validation/test; "
                    "train overlap removed"
                ),
                "gsm8k": (
                    "Official test preserved except exact duplicate collapse; "
                    "validation carved only from official train"
                ),
                "math": "Stable-ID hash ranking, stratified by configured difficulty",
            },
            "selection_policy": (
                "No model results or gold answers enter split ranking/pilot selection. "
                "Thresholds/calibrators may use validation only; test is held out."
            ),
            "deduplication": dedup,
            "leakage_audit": leakage,
            "difficulty_interpretation": (
                "reasoning_steps is expression dependency depth, not measured model/human "
                "reasoning effort; prompt length is controlled wording, word count is measured"
            ),
        }
        manifest["dataset_fingerprint"] = config_digest(manifest)
        write_json(root / "manifest.json", manifest)
        write_json(root / "leakage_report.json", {"deduplication": dedup, **leakage})
        run.metadata["dataset_fingerprint"] = manifest["dataset_fingerprint"]
        run.metadata["data_scope"] = manifest["scope"]
        write_json(run.path / "metadata.json", run.metadata)
        run.log(
            "datasets_prepared",
            dataset_fingerprint=manifest["dataset_fingerprint"],
            pilot=pilot,
            scope=manifest["scope"],
        )
        write_json(
            run.path / "result.json",
            {
                "status": "completed",
                "scope": manifest["scope"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "manifest": "artifacts/datasets/manifest.json",
                "leakage_audit_passed": True,
            },
        )
    return run.path
