"""Freeze disjoint operand-pair partitions using existing train/validation only."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from llm_scratch.curriculum.data import Corpus as SourceCorpus
from llm_scratch.curriculum.data import validate_row
from llm_scratch.curriculum.evaluation import load_selected
from llm_scratch.data.collator import collate_tokenized
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import expression_from_dict
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.utils.config import canonical_json, config_digest
from llm_scratch.utils.io import write_json


def pair(a, b):
    return "pair-" + config_digest(sorted([int(a), int(b)]))


def facts(rows):
    """Exclude integer operand pairs at EVERY old arithmetic node, including subtasks."""
    result = set()

    def visit(tree):
        if tree.operator:
            a, b = tree.left.evaluate(), tree.right.evaluate()
            if a.denominator == b.denominator == 1:
                result.add(pair(a, b))
            visit(tree.left)
            visit(tree.right)

    for row in rows:
        if row["task"] == "copy":
            result.add(pair(*row["operands"]))
        else:
            visit(expression_from_dict(row["expression"]))
    return result


def partition(rows, excluded, cfg):
    """Choose by ID hash alone; never rank by answers, scores, or model output."""
    pools = {o: {} for o in ("wp03", "authored")}
    for row in sorted(rows, key=lambda r: config_digest([cfg["seed"], r["question_id"]])):
        if row["task"] != "single":
            continue
        group = pair(*row["operands"])
        if group not in excluded:
            pools[row["origin"]].setdefault(group, row)
    selected = {k: [] for k in ("selection", "gate")}
    for origin, groups in pools.items():
        ranked = sorted(groups, key=lambda g: config_digest([cfg["seed"], origin, g]))
        required = sum(cfg[k][origin] for k in selected)
        if len(ranked) < required:
            raise ValueError(
                f"Insufficient clean {origin} operand pairs: {len(ranked)} < {required}"
            )
        offset = 0
        for key in selected:
            n = cfg[key][origin]
            selected[key].extend(groups[g] for g in ranked[offset : offset + n])
            offset += n
    return selected, {o: len(g) for o, g in pools.items()}


def audit(parts):
    seen_ids, seen_prompts, seen_groups = set(), set(), set()
    for name, rows in parts.items():
        ids = {r["question_id"] for r in rows}
        prompts = {r["prompt"] for r in rows}
        groups = {pair(*r["operands"]) for r in rows}
        if (
            not rows
            or len(ids) != len(rows)
            or len(prompts) != len(rows)
            or ids & seen_ids
            or prompts & seen_prompts
            or groups & seen_groups
            or (name != "train" and len(groups) != len(rows))
        ):
            raise ValueError("Recovery duplicate or operand-pair partition leakage")
        seen_ids |= ids
        seen_prompts |= prompts
        seen_groups |= groups


def prepare(config, source, tokenizer, checkpoint, prior_questions, output):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    if (
        set(cfg)
        != {
            "schema_version",
            "seed",
            "max_length",
            "source_dataset_fingerprint",
            "source_checkpoint_sha256",
            "prior_questions_sha256",
            "selection",
            "gate",
        }
        or cfg["schema_version"] != 1
        or type(cfg["seed"]) is not int
        or not 0 <= cfg["seed"] < 2**32
        or type(cfg["max_length"]) is not int
        or cfg["max_length"] < 16
        or any(
            set(cfg[k]) != {"wp03", "authored"}
            or any(type(v) is not int or v < 1 for v in cfg[k].values())
            for k in ("selection", "gate")
        )
        or file_sha256(prior_questions) != cfg["prior_questions_sha256"]
    ):
        raise ValueError("Invalid recovery data protocol/prior question checksum")
    codec = ByteBPETokenizer.load(tokenizer)
    train = SourceCorpus(source, "train", codec)
    val = SourceCorpus(source, "validation", codec)
    if train.meta["fingerprint"] != cfg["source_dataset_fingerprint"]:
        raise ValueError("Source curriculum fingerprint mismatch")
    model, source_meta = load_selected(checkpoint, codec, cfg["source_checkpoint_sha256"])
    del model
    old_ids = set(source_meta["contract"]["selection_ids"])
    prior = [json.loads(s) for s in Path(prior_questions).read_text(encoding="utf-8").splitlines()]
    old_prompts = {r["question"] for r in prior}
    if not old_ids <= {r["question_id"] for r in val.examples} or not old_prompts <= {
        r["prompt"] for r in val.examples
    }:
        raise ValueError("Prior selection/pilot must map to source validation")
    exposed = [r for r in val.examples if r["question_id"] in old_ids or r["prompt"] in old_prompts]
    excluded = facts(train.examples) | facts(exposed)
    parts, pools = partition(val.examples, excluded, cfg)
    parts = {"train": [r for r in train.examples if r["task"] == "single"], **parts}
    audit(parts)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specs = {}
    for name, records in parts.items():
        for r in records:
            validate_row(r, codec, cfg["max_length"])
        path = output / f"{name}.jsonl"
        path.write_text(
            "".join(canonical_json(r) + "\n" for r in records), encoding="utf-8", newline="\n"
        )
        specs[name] = {
            "sha256": file_sha256(path),
            "records": len(records),
            "origins": dict(Counter(r["origin"] for r in records)),
            "operators": dict(Counter(r["expression"]["operator"] for r in records)),
            "ids": [r["question_id"] for r in records] if name != "train" else None,
        }
    meta = {
        "schema_version": 1,
        "scope": "easy-recovery-v1",
        "config": cfg,
        "vocab_size": len(codec),
        "tokenizer_fingerprint": codec.fingerprint,
        "block_size": cfg["max_length"],
        "splits": specs,
        "clean_pair_pool": pools,
        "excluded_pairs": len(excluded),
        "excluded_pairs_sha256": config_digest(sorted(excluded)),
        "evaluation_pair_groups": {
            k: [pair(*r["operands"]) for r in parts[k]] for k in ("selection", "gate")
        },
        "test_files_read": False,
        "scope_limit": (
            "Development validation; previously used for loss. Pretraining exposure unknown."
        ),
        "train_sampling": (
            "Repeat WP03 references ceil(authored/WP03) times; shuffle virtual epoch."
        ),
    }
    meta["fingerprint"] = config_digest(meta)
    write_json(output / "manifest.json", meta)
    print(
        canonical_json(
            {
                "event": "easy_recovery_data_ready",
                "output": str(output),
                "fingerprint": meta["fingerprint"],
                "clean_pair_pool": pools,
            }
        ),
        flush=True,
    )
    return output


class Corpus:
    def __init__(self, root, partition_name, codec):
        self.root, self.partition, self.codec = Path(root), partition_name, codec
        if partition_name not in ("train", "selection", "gate"):
            raise ValueError("Unknown recovery partition")
        self.split = "train" if partition_name == "train" else "validation"
        self.meta = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        m = self.meta
        if (
            m.get("scope") != "easy-recovery-v1"
            or m.get("schema_version") != 1
            or config_digest({k: v for k, v in m.items() if k != "fingerprint"})
            != m.get("fingerprint")
            or m.get("tokenizer_fingerprint") != codec.fingerprint
            or m.get("vocab_size") != len(codec)
        ):
            raise ValueError("Recovery manifest/tokenizer mismatch")
        path = self.root / f"{partition_name}.jsonl"
        spec = m["splits"][partition_name]
        if file_sha256(path) != spec["sha256"]:
            raise ValueError("Recovery data checksum mismatch")
        self.examples = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        if not self.examples or len(self.examples) != spec["records"]:
            raise ValueError("Recovery data count mismatch")
        for row in self.examples:
            validate_row(row, codec, m["block_size"])
            if row["task"] != "single" or row["split"] != self.split:
                raise ValueError("Recovery must retain original easy train/validation membership")
        audit({partition_name: self.examples})
        self.stamps = {
            p: (p.stat().st_size, p.stat().st_mtime_ns) for p in (path, self.root / "manifest.json")
        }
        self.unique_examples = self.examples
        self.repeat = 1
        if self.split == "train":
            groups = {
                o: [r for r in self.examples if r["origin"] == o] for o in ("wp03", "authored")
            }
            if not all(groups.values()):
                raise ValueError("Both training origins are required")
            self.repeat = math.ceil(len(groups["authored"]) / len(groups["wp03"]))
            self.examples = groups["wp03"] * self.repeat + groups["authored"]

    def assert_unchanged(self):
        if any(
            (p.stat().st_size, p.stat().st_mtime_ns) != stamp for p, stamp in self.stamps.items()
        ):
            raise RuntimeError("Recovery data changed during execution")

    @property
    def identity(self):
        self.assert_unchanged()
        return {
            "fingerprint": self.meta["fingerprint"],
            "partition": self.partition,
            "wp03_repeat": self.repeat,
            "virtual_rows": len(self),
        }

    def __len__(self):
        return len(self.examples)

    def batch(self, indices):
        self.assert_unchanged()
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in collate_tokenized([self.examples[i] for i in indices], self.codec).items()
        }

    def validation_batches(self, size):
        if self.split != "validation" or type(size) is not int or size < 1:
            raise ValueError("Validation requires a positive batch size")
        for start in range(0, len(self), size):
            yield self.batch(list(range(start, min(start + size, len(self)))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "tokenizer", "checkpoint", "prior-questions", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.config, a.source, a.tokenizer, a.checkpoint, a.prior_questions, a.output)


if __name__ == "__main__":
    main()
