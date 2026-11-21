"""Grouped authored copy/arithmetic plus unchanged WP03 train/validation bridges."""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import Counter
from pathlib import Path

import torch

from llm_scratch.data.collator import collate_tokenized
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import (
    Expression,
    build_expression,
    expression_from_dict,
    render_question,
)
from llm_scratch.data.sft import math_reasoning, normalize_example
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.pilot import load_manifest
from llm_scratch.reasoning.formatting import encode_sft_example
from llm_scratch.utils.config import canonical_json, config_digest
from llm_scratch.utils.io import write_json

TASKS = ("copy", "single", "multi")


def emit(event, **fields):
    print(canonical_json({"event": event, **fields}), flush=True)


def load_data_config(path):
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    keys = {
        "schema_version",
        "seed",
        "operand_min",
        "operand_max",
        "validation_percent",
        "multi_train_per_level",
        "multi_validation_per_level",
        "max_length",
        "wp03_fingerprint",
        "tokenizer_fingerprint",
    }
    if set(cfg) != keys or cfg["schema_version"] != 1:
        raise ValueError("Invalid curriculum data configuration")
    if any(type(cfg[k]) is not int for k in keys - {"wp03_fingerprint", "tokenizer_fingerprint"}):
        raise ValueError("Curriculum counts must be integers")
    if not (
        31 <= cfg["operand_min"] < cfg["operand_max"] <= 999
        and 1 <= cfg["validation_percent"] <= 49
        and 0 <= cfg["seed"] < 2**32
    ):
        raise ValueError("Invalid range/seed/split; new operands must be outside WP03 1..30")
    if (
        min(cfg["multi_train_per_level"], cfg["multi_validation_per_level"]) < 1
        or cfg["max_length"] < 8
    ):
        raise ValueError("Invalid curriculum sizes/context cap")
    return cfg


def group_split(group, cfg):
    return (
        "validation"
        if int(config_digest([cfg["seed"], group])[:16], 16) % 100 < cfg["validation_percent"]
        else "train"
    )


def leaves(tree):
    return [tree.value] if tree.operator is None else leaves(tree.left) + leaves(tree.right)


def copy_text(a, b):
    return (
        f"Copy both operands in order: left = {a}, right = {b}. "
        "Return the two numbers separated by a comma; do not calculate."
    )


def authored(cfg):
    rows = {"train": [], "validation": []}
    for a in range(cfg["operand_min"], cfg["operand_max"] + 1):
        for b in range(cfg["operand_min"], cfg["operand_max"] + 1):
            group = "pair-" + config_digest(sorted([a, b]))
            split = group_split(group, cfg)
            common = {"origin": "authored", "split": split, "group": group, "operands": [a, b]}
            rows[split].append(
                common
                | {
                    "task": "copy",
                    "difficulty": "copy",
                    "prompt_length": "short",
                    "prompt": copy_text(a, b),
                    "reasoning": f"The operands are {a} and {b}.",
                    "final_answer": f"{a},{b}",
                    "expression": None,
                    "semantic_id": "copy-" + config_digest([a, b]),
                }
            )
            for op in ("+", "-"):
                if op == "+" and a > b:
                    continue  # Commutative arithmetic duplicates are not separate examples.
                tree = Expression(operator=op, left=Expression(value=a), right=Expression(value=b))
                rows[split].append(math_row(tree, "single", split, group))
    rng = random.Random(cfg["seed"])
    seen = set()
    for operations in (2, 3):
        counts = Counter()
        limits = {
            "train": cfg["multi_train_per_level"],
            "validation": cfg["multi_validation_per_level"],
        }
        attempts = 0
        while any(counts[s] < n for s, n in limits.items()):
            attempts += 1
            if attempts > 1000 * sum(limits.values()):
                raise ValueError("Insufficient distinct multi-step operand groups")
            tree = build_expression(
                rng, operations, 2, cfg["operand_min"], cfg["operand_max"], ["+", "-"]
            )
            group = "leaves-" + config_digest(sorted(leaves(tree)))
            split = group_split(group, cfg)
            semantic = config_digest(tree.canonical())
            if counts[split] >= limits[split] or semantic in seen:
                continue
            seen.add(semantic)
            rows[split].append(math_row(tree, "multi", split, group))
            counts[split] += 1
    return rows


def math_row(tree, task, split, group):
    length = "short" if tree.operations == 1 else "medium"
    return {
        "origin": "authored",
        "split": split,
        "group": group,
        "task": task,
        "difficulty": f"ops-{tree.operations}",
        "prompt_length": length,
        "operands": leaves(tree),
        "prompt": render_question(tree, length),
        "reasoning": math_reasoning(tree.as_dict()),
        "final_answer": str(tree.evaluate()),
        "expression": tree.as_dict(),
        "semantic_id": "math-" + config_digest(tree.canonical()),
    }


def bridge(root, cfg):
    """Read only frozen WP03 math train/validation; never open pilot or test files."""
    manifest = load_manifest(root)
    if manifest["dataset_fingerprint"] != cfg["wp03_fingerprint"]:
        raise ValueError("WP03 source fingerprint mismatch")
    result, sources = {}, {}
    for split in ("train", "validation"):
        relative = f"math/{split}.jsonl"
        path, spec = root / relative, manifest["files"][relative]
        if (
            spec["path"] != relative
            or file_sha256(path) != spec["sha256"]
            or path.stat().st_size != spec["size_bytes"]
        ):
            raise ValueError("WP03 source checksum mismatch")
        records = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        if len(records) != spec["records"]:
            raise ValueError("WP03 source count mismatch")
        result[split] = []
        for raw in records:
            normalized = normalize_example(raw, "math", split)
            tree = expression_from_dict(raw["expression"])
            if not all(1 <= n <= 30 for n in leaves(tree)):
                raise ValueError("WP03 bridge no longer disjoint from new operand range")
            result[split].append(
                {
                    "origin": "wp03",
                    "split": split,
                    "group": "wp03-" + raw["semantic_id"],
                    "task": "single" if tree.operations == 1 else "multi",
                    "difficulty": raw["difficulty"],
                    "prompt_length": raw["prompt_length"],
                    "operands": leaves(tree),
                    "prompt": normalized["prompt"],
                    "reasoning": normalized["reasoning"],
                    "final_answer": normalized["final_answer"],
                    "expression": raw["expression"],
                    "semantic_id": "wp03-" + raw["semantic_id"],
                }
            )
        sources[relative] = spec
    return result, sources


def validate_row(row, codec, max_length):
    task = row["task"]
    if (
        task not in TASKS
        or row["split"] not in ("train", "validation")
        or row["origin"] not in ("wp03", "authored")
    ):
        raise ValueError("Invalid curriculum row task/source/split")
    if task == "copy":
        a, b = row["operands"]
        if (
            row["origin"] != "authored"
            or row["expression"] is not None
            or (row["prompt"], row["reasoning"], row["final_answer"])
            != (copy_text(a, b), f"The operands are {a} and {b}.", f"{a},{b}")
        ):
            raise ValueError("Copy operands/prompt/target mismatch")
        semantic, group = "copy-" + config_digest([a, b]), "pair-" + config_digest(sorted([a, b]))
    else:
        tree = expression_from_dict(row["expression"])
        if (
            leaves(tree) != row["operands"]
            or tree.operations < 1
            or (task == "single") != (tree.operations == 1)
        ):
            raise ValueError("Arithmetic task/operands mismatch")
        if (row["prompt"], row["reasoning"], row["final_answer"]) != (
            render_question(tree, row["prompt_length"]),
            math_reasoning(tree.as_dict()),
            str(tree.evaluate()),
        ):
            raise ValueError("Arithmetic prompt/trace/answer mismatch")
        prefix = "wp03-" if row["origin"] == "wp03" else "math-"
        semantic = prefix + config_digest(tree.canonical())
        group = (
            semantic
            if row["origin"] == "wp03"
            else ("pair-" if task == "single" else "leaves-") + config_digest(sorted(leaves(tree)))
        )
    if (
        row["semantic_id"] != semantic
        or row["group"] != group
        or row["question_id"] != "curriculum-" + config_digest([row["origin"], semantic])
    ):
        raise ValueError("Curriculum semantic/group identity mismatch")
    encoded = encode_sft_example(codec, row["prompt"], row["reasoning"], row["final_answer"])
    if any(row.get(k) != v for k, v in encoded.items()) or len(encoded["input_ids"]) > max_length:
        raise ValueError("Curriculum encoding/mask/length mismatch; no truncation allowed")


def audit(train, validation):
    seen_ids, seen_prompts, seen_groups = set(), set(), set()
    for rows in (train, validation):
        ids, prompts = {r["question_id"] for r in rows}, {r["prompt"] for r in rows}
        groups = {r["group"] for r in rows}
        if (
            len(ids) != len(rows)
            or len(prompts) != len(rows)
            or ids & seen_ids
            or prompts & seen_prompts
            or groups & seen_groups
        ):
            raise ValueError("Curriculum duplicate or train/validation group leakage")
        seen_ids |= ids
        seen_prompts |= prompts
        seen_groups |= groups


def prepare(config_path, dataset_root, tokenizer, output):
    cfg = load_data_config(config_path)
    codec = ByteBPETokenizer.load(tokenizer, expected_fingerprint=cfg["tokenizer_fingerprint"])
    emit("curriculum_preparation_started", config=cfg)
    rows = authored(cfg)
    bridges, sources = bridge(Path(dataset_root), cfg)
    for split in rows:
        rows[split].extend(bridges[split])
        for index, row in enumerate(rows[split]):
            row["question_id"] = "curriculum-" + config_digest([row["origin"], row["semantic_id"]])
            row.update(
                encode_sft_example(codec, row["prompt"], row["reasoning"], row["final_answer"])
            )
            validate_row(row, codec, cfg["max_length"])
            if (index + 1) % 1000 == 0:
                emit(
                    "curriculum_encoding_progress",
                    split=split,
                    completed=index + 1,
                    total=len(rows[split]),
                )
    audit(rows["train"], rows["validation"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specs = {}
    for split, records in rows.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in records:
                f.write(canonical_json(row) + "\n")
        specs[split] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "records": len(records),
            "tasks": dict(Counter(r["task"] for r in records)),
            "origins": dict(Counter(r["origin"] for r in records)),
            "target_tokens": sum(sum(t != -100 for t in r["labels"]) for r in records),
            "max_tokens": max(len(r["input_ids"]) for r in records),
        }
        emit("curriculum_split_prepared", split=split, **specs[split])
    meta = {
        "schema_version": 1,
        "work_package": "WP07.6.2",
        "scope": "curriculum_train_validation_only",
        "config": cfg,
        "source_files": sources,
        "vocab_size": len(codec),
        "tokenizer_fingerprint": codec.fingerprint,
        "block_size": cfg["max_length"],
        "protocol": "independent-bos-question-think-masked-v1",
        "splits": specs,
        "split_policy": (
            "Authored unordered pairs shared across copy/single; sorted leaf tuples for multi. "
            "WP03 original semantic splits unchanged."
        ),
        "test_files_read": False,
        "truncated_examples": 0,
    }
    meta["fingerprint"] = config_digest(meta)
    write_json(output / "manifest.json", meta)
    emit("curriculum_data_ready", output=str(output), fingerprint=meta["fingerprint"])
    return output


class Corpus:
    def __init__(self, root, split, codec):
        self.root, self.split, self.codec = Path(root), split, codec
        if split not in ("train", "validation"):
            raise ValueError("Only curriculum train/validation are supported")
        self.meta = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        m = self.meta
        if (
            m.get("work_package") != "WP07.6.2"
            or m.get("scope") != "curriculum_train_validation_only"
            or m.get("schema_version") != 1
            or config_digest({k: v for k, v in m.items() if k != "fingerprint"})
            != m.get("fingerprint")
            or m.get("tokenizer_fingerprint") != codec.fingerprint
            or m.get("vocab_size") != len(codec)
        ):
            raise ValueError("Invalid curriculum manifest/tokenizer")
        spec = m["splits"][split]
        if spec["file"] != f"{split}.jsonl":
            raise ValueError("Invalid curriculum split path")
        path = self.root / spec["file"]
        if file_sha256(path) != spec["sha256"] or path.stat().st_size != spec["size_bytes"]:
            raise ValueError("Curriculum payload checksum mismatch")
        self.examples = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        if len(self.examples) != spec["records"] or not self.examples:
            raise ValueError("Curriculum record count mismatch")
        for row in self.examples:
            if row["split"] != split:
                raise ValueError("Curriculum row split mismatch")
            validate_row(row, codec, m["block_size"])
            if row["origin"] == "authored" and (
                group_split(row["group"], m["config"]) != split
                or not all(
                    m["config"]["operand_min"] <= v <= m["config"]["operand_max"]
                    for v in row["operands"]
                )
            ):
                raise ValueError("Authored curriculum range/group split mismatch")
        if (
            dict(Counter(r["task"] for r in self.examples)) != spec["tasks"]
            or sum(sum(t != -100 for t in r["labels"]) for r in self.examples)
            != spec["target_tokens"]
        ):
            raise ValueError("Curriculum task/target counts mismatch")
        self.stamps = {
            p: (p.stat().st_size, p.stat().st_mtime_ns) for p in (path, self.root / "manifest.json")
        }

    def assert_unchanged(self):
        if any((p.stat().st_size, p.stat().st_mtime_ns) != s for p, s in self.stamps.items()):
            raise RuntimeError("Curriculum files changed after verification")

    def view(self, tasks=TASKS):
        return View(self, tasks)


class View:
    def __init__(self, base, tasks):
        self.base, self.split, self.meta = base, base.split, base.meta
        self.tasks = tuple(tasks)
        if (
            not self.tasks
            or len(set(self.tasks)) != len(self.tasks)
            or set(self.tasks) - set(TASKS)
        ):
            raise ValueError("Invalid curriculum task pool")
        self.examples = [r for r in base.examples if r["task"] in tasks]
        if not self.examples:
            raise ValueError("Empty curriculum task pool")

    @property
    def identity(self):
        self.base.assert_unchanged()
        return {
            "fingerprint": self.meta["fingerprint"],
            "split": self.split,
            "tasks": list(self.tasks),
        }

    def __len__(self):
        return len(self.examples)

    def batch(self, indices):
        self.base.assert_unchanged()
        if not indices or any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise ValueError("Invalid curriculum batch indices")
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in collate_tokenized(
                [self.examples[i] for i in indices], self.base.codec
            ).items()
        }

    def validation_batches(self, size):
        if self.split != "validation" or type(size) is not int or size < 1:
            raise ValueError("Need validation split and positive batch size")
        for start in range(0, len(self), size):
            yield self.batch(list(range(start, min(start + size, len(self)))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/data/wp07_curriculum.toml"))
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.config, a.dataset_root, a.tokenizer, a.output)


if __name__ == "__main__":
    main()
