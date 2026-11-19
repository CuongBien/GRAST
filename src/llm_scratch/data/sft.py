"""Verified WP03 train/validation -> independent prompt-masked CoT examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from llm_scratch.data.collator import collate_tokenized
from llm_scratch.data.datasets import file_sha256, normalize_text
from llm_scratch.data.difficulty import expression_from_dict, render_question
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.reasoning.answer_extraction import extract_answer, normalize_answer
from llm_scratch.reasoning.formatting import encode_reasoning_prompt, encode_sft_example
from llm_scratch.utils.config import canonical_json, config_digest
from llm_scratch.utils.io import write_json

PROTOCOL = "cot-independent-bos-question-lf-think-reasoning-final-eos-prompt-masked-v1"


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def math_reasoning(expression: dict) -> str:
    """Mirror WP03 postorder R1/R2 names, compute exact Fractions at every node."""
    tree = expression_from_dict(expression)
    steps = []

    def visit(node):
        if node.operator is None:
            return str(node.value)
        left, right = visit(node.left), visit(node.right)
        value = node.evaluate()
        name = f"R{len(steps) + 1}"
        steps.append(f"{name} = {left} {node.operator} {right} = {value}.")
        return name

    visit(tree)
    if not steps:
        raise ValueError("Arithmetic rationale requires at least one operation")
    return "\n".join(steps)


def normalize_example(row: dict, source: str, split: str) -> dict:
    expected_source = "grast-arithmetic-v1" if source == "math" else "gsm8k"
    if (
        row.get("source") != expected_source
        or row.get("split") != split
        or not isinstance(row.get("question_id"), str)
        or not row["question_id"]
        or not isinstance(row.get("question"), str)
        or not normalize_text(row["question"])
    ):
        raise ValueError("Invalid source/split/question identity")
    final = normalize_answer(row.get("canonical_answer"))
    if final is None or extract_answer(row.get("answer")) != final:
        raise ValueError("Source final answer and canonical answer disagree")
    if source == "math":
        tree = expression_from_dict(row["expression"])
        if (
            normalize_answer(str(tree.evaluate())) != final
            or render_question(tree, row["prompt_length"]) != row["question"]
        ):
            raise ValueError("Arithmetic prompt/expression/final answer disagree")
        reasoning = math_reasoning(row["expression"])
    else:
        if row.get("upstream_split") != "train" or row["answer"].count("####") != 1:
            raise ValueError("GSM8K SFT only accepts internal splits of official train")
        reasoning = row["answer"].split("####", 1)[0].strip()
        if not reasoning:
            raise ValueError("GSM8K rationale is empty")
    return {
        "question_id": row["question_id"],
        "source": row["source"],
        "split": split,
        "difficulty": row.get("difficulty", "ungraded"),
        "prompt": row["question"],
        "reasoning": reasoning,
        "final_answer": final,
    }


def prepare_sft(dataset_root, tokenizer_root, output, *, max_length=512, sources=("math",)) -> Path:
    if type(max_length) is not int or max_length < 8:
        raise ValueError("max_length must be an integer >=8")
    sources = tuple(sources)
    if not sources or len(set(sources)) != len(sources) or set(sources) - {"math", "gsm8k"}:
        raise ValueError(
            "Choose unique sources from math/gsm8k; default math preserves GSM8K benchmark"
        )
    root, output = Path(dataset_root).resolve(), Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("work_package") != "WP03"
        or manifest.get("scope") not in ("full", "math-only")
        or config_digest({k: v for k, v in manifest.items() if k != "dataset_fingerprint"})
        != manifest.get("dataset_fingerprint")
    ):
        raise ValueError("Need a verified WP03 dataset manifest")
    codec = ByteBPETokenizer.load(tokenizer_root)
    output.mkdir(parents=True, exist_ok=False)
    seen, ids, splits, source_files = set(), set(), {}, {}
    for split in ("train", "validation"):
        records, prompt_tokens, target_tokens, lengths = 0, 0, 0, []
        counts = Counter()
        destination = output / f"{split}.jsonl"
        with destination.open("x", encoding="utf-8", newline="\n") as target:
            for source in sorted(sources):
                relative = f"{source}/{split}.jsonl"
                spec = manifest["files"].get(relative, {})
                path = (root / relative).resolve()
                if (
                    not path.is_relative_to(root)
                    or spec.get("path") != relative
                    or spec.get("role") != "training_or_evaluation"
                ):
                    raise ValueError(f"Invalid SFT source: {relative}")
                digest, size, count = hashlib.sha256(), 0, 0
                emit("sft_normalization_started", source=relative)
                with path.open("rb") as stream:
                    for line in stream:
                        digest.update(line)
                        size += len(line)
                        example = normalize_example(json.loads(line), source, split)
                        fingerprint = config_digest(normalize_text(example["prompt"]))
                        if fingerprint in seen or example["question_id"] in ids:
                            raise ValueError(
                                "Duplicate question within/across SFT train/validation"
                            )
                        seen.add(fingerprint)
                        ids.add(example["question_id"])
                        encoded = encode_sft_example(
                            codec, example["prompt"], example["reasoning"], example["final_answer"]
                        )
                        length = len(encoded["input_ids"])
                        if length > max_length:
                            raise ValueError(
                                f"Example {example['question_id']} needs {length} tokens > "
                                f"max_length={max_length}; no silent truncation/filtering"
                            )
                        prefix = len(encode_reasoning_prompt(codec, example["prompt"]))
                        target.write(canonical_json(example | encoded) + "\n")
                        prompt_tokens += prefix
                        target_tokens += length - prefix
                        lengths.append(length)
                        counts[f"{source}/{example['difficulty']}"] += 1
                        count += 1
                        records += 1
                        if records % 1000 == 0:
                            emit("sft_normalization_progress", split=split, examples=records)
                if (
                    digest.hexdigest() != spec["sha256"]
                    or size != spec["size_bytes"]
                    or (count != spec["records"] or count < 1)
                ):
                    raise ValueError(f"SFT source checksum/count mismatch: {relative}")
                source_files[relative] = {k: spec[k] for k in ("sha256", "records", "size_bytes")}
        if not records:
            raise ValueError("SFT split is empty")
        splits[split] = {
            "file": destination.name,
            "sha256": file_sha256(destination),
            "size_bytes": destination.stat().st_size,
            "records": records,
            "prompt_tokens": prompt_tokens,
            "target_tokens": target_tokens,
            "min_length": min(lengths),
            "max_length": max(lengths),
            "counts": dict(counts),
        }
        emit("sft_normalization_completed", split=split, **splits[split])
    meta = {
        "schema_version": 1,
        "work_package": "WP07.4",
        "scope": "cot_sft_train_validation_only",
        "protocol": PROTOCOL,
        "sources": sorted(sources),
        "source_files": source_files,
        "source_dataset_fingerprint": manifest["dataset_fingerprint"],
        "source_origin": manifest.get("origin", "wp03_prepared_sources"),
        "block_size": max_length,
        "vocab_size": len(codec),
        "tokenizer_fingerprint": codec.fingerprint,
        "splits": splits,
        "truncated_examples": 0,
        "filtered_examples": 0,
    }
    meta["fingerprint"] = config_digest(meta)
    write_json(output / "manifest.json", meta)
    emit("sft_data_ready", output=str(output), fingerprint=meta["fingerprint"])
    return output


class SFTCorpus:
    """Independent examples with dynamic right padding and no cross-example attention."""

    def __init__(self, root, split, codec: ByteBPETokenizer):
        self.root, self.split, self.codec = Path(root).resolve(), split, codec
        if split not in ("train", "validation"):
            raise ValueError("Only SFT train/validation splits can be loaded")
        manifest_path = self.root / "manifest.json"
        self.meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        m = self.meta
        if (
            m.get("schema_version") != 1
            or m.get("work_package") != "WP07.4"
            or m.get("scope") != "cot_sft_train_validation_only"
            or m.get("protocol") != PROTOCOL
            or m.get("vocab_size") != len(codec)
            or m.get("tokenizer_fingerprint") != codec.fingerprint
            or type(m.get("block_size")) is not int
            or m["block_size"] < 8
            or config_digest({k: v for k, v in m.items() if k != "fingerprint"})
            != m.get("fingerprint")
        ):
            raise ValueError("Invalid SFT manifest/tokenizer/protocol")
        spec = m["splits"][split]
        if spec.get("file") != f"{split}.jsonl":
            raise ValueError("Invalid SFT split filename")
        path = self.root / spec["file"]
        if file_sha256(path) != spec["sha256"] or path.stat().st_size != spec["size_bytes"]:
            raise ValueError("SFT payload checksum mismatch")
        self.examples = []
        seen = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                final = row.get("final_answer")
                if (
                    row.get("split") != split
                    or not isinstance(row.get("prompt"), str)
                    or not normalize_text(row["prompt"])
                    or not isinstance(row.get("reasoning"), str)
                    or not row["reasoning"].strip()
                    or normalize_answer(final) != final
                    or final is None
                    or row.get("question_id") in seen
                ):
                    raise ValueError("Invalid normalized SFT example")
                encoded = encode_sft_example(codec, row["prompt"], row["reasoning"], final)
                if any(row.get(k) != v for k, v in encoded.items()) or (
                    len(encoded["input_ids"]) > m["block_size"]
                ):
                    raise ValueError("SFT prompt mask/target/framing/length mismatch")
                seen.add(row["question_id"])
                self.examples.append(row)
        if len(self.examples) != spec["records"] or not self.examples:
            raise ValueError("SFT record count mismatch")
        target_count = sum(sum(v != -100 for v in r["labels"]) for r in self.examples)
        if target_count != spec["target_tokens"]:
            raise ValueError("SFT target count mismatch")
        self._stamps = {p: self._stamp(p) for p in (manifest_path, path)}

    @staticmethod
    def _stamp(path):
        s = path.stat()
        return s.st_ino, s.st_size, s.st_mtime_ns

    def assert_unchanged(self):
        if any(self._stamp(p) != stamp for p, stamp in self._stamps.items()):
            raise RuntimeError("SFT files changed after verification")

    @property
    def identity(self):
        self.assert_unchanged()
        return {
            k: self.meta[k]
            for k in (
                "fingerprint",
                "tokenizer_fingerprint",
                "block_size",
                "vocab_size",
                "protocol",
            )
        } | {"split": self.split}

    def __len__(self):
        return len(self.examples)

    def batch(self, indices):
        self.assert_unchanged()
        if not indices or any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise ValueError("Invalid SFT batch indices")
        examples = [self.examples[i] for i in indices]
        padded = collate_tokenized(examples, self.codec)
        return {
            k: torch.tensor(
                v, dtype=torch.bool if k == "attention_mask" else torch.long, device="cpu"
            )
            for k, v in padded.items()
        }

    def validation_batches(self, batch_size):
        if self.split != "validation" or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Need validation split and a positive batch size")
        for start in range(0, len(self), batch_size):
            yield self.batch(list(range(start, min(start + batch_size, len(self)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sources", nargs="+", choices=("math", "gsm8k"), default=["math"])
    args = parser.parse_args()
    prepare_sft(
        args.dataset_root,
        args.tokenizer,
        args.output,
        max_length=args.max_length,
        sources=args.sources,
    )


if __name__ == "__main__":
    main()
