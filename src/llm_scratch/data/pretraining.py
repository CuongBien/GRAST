"""WP07.2 immutable WP03/WP04 token streams and loss-preserving causal blocks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from llm_scratch.data.datasets import file_sha256, normalize_text
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json

PACKING = "continuous-bos-eos-overlap1-ignore-bos-target-v1"


def _emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def prepare_pretraining(
    dataset_root: str | Path, tokenizer_root: str | Path, output: str | Path, *, block_size: int
) -> Path:
    """Read train/validation only; publish manifest last, never silently reuse output."""
    if type(block_size) is not int or block_size < 3:
        raise ValueError("block_size must be an integer >=3")
    root, output = Path(dataset_root).resolve(), Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("work_package") != "WP03"
        or manifest.get("scope") != "full"
        or config_digest({k: v for k, v in manifest.items() if k != "dataset_fingerprint"})
        != manifest.get("dataset_fingerprint")
    ):
        raise ValueError("Need a valid full WP03 dataset manifest")
    codec = ByteBPETokenizer.load(tokenizer_root)
    if len(codec) > 2**32:
        raise ValueError("Vocabulary cannot fit uint32 token stream")
    output.mkdir(parents=True, exist_ok=False)
    seen, splits = set(), {}
    for split in ("train", "validation"):
        relative = f"tinystories/{split}.jsonl"
        spec = manifest["files"].get(relative)
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not spec
            or spec.get("path") != relative
            or spec.get("role") != "pretraining"
        ):
            raise ValueError(f"Invalid WP03 pretraining source: {relative}")
        digest, size, records, tokens = hashlib.sha256(), 0, 0, 0
        target = output / f"{split}.u32"
        _emit("tokenization_started", split=split, expected_documents=spec["records"])
        with path.open("rb") as source, target.open("xb") as destination:
            for line in source:
                digest.update(line)
                size += len(line)
                row = json.loads(line)
                if (
                    row.get("split") != split
                    or row.get("source") != "tinystories"
                    or not isinstance(row.get("text"), str)
                    or not row["text"].strip()
                ):
                    raise ValueError(f"Invalid source row at {relative}:{records + 1}")
                fingerprint = config_digest(normalize_text(row["text"]))
                if fingerprint in seen:
                    raise ValueError("Duplicate story within/across train and validation")
                seen.add(fingerprint)
                ids = codec.encode(row["text"], add_bos=True, add_eos=True)
                destination.write(np.asarray(ids, dtype="<u4").tobytes())
                tokens += len(ids)
                records += 1
                if records % 1000 == 0:
                    _emit("tokenization_progress", split=split, documents=records, tokens=tokens)
        if (
            digest.hexdigest() != spec["sha256"]
            or size != spec["size_bytes"]
            or records != spec["records"]
            or records < 1
            or tokens < 3
        ):
            raise ValueError(f"Source checksum/count mismatch: {relative}")
        splits[split] = {
            "file": target.name,
            "source_file": relative,
            "source_sha256": digest.hexdigest(),
            "documents": records,
            "tokens": tokens,
            "blocks": (tokens - 2) // (block_size - 1) + 1,
            "supervised_pairs": tokens - records,
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        _emit("tokenization_completed", split=split, **splits[split])
    result = {
        "schema_version": 1,
        "work_package": "WP07.2",
        "scope": "pretraining_train_validation_only",
        "source_dataset_fingerprint": manifest["dataset_fingerprint"],
        "source_origin": manifest.get("origin", "wp03_prepared_sources"),
        "packing": PACKING,
        "block_size": block_size,
        "dtype": "<u4",
        "tokenizer_fingerprint": codec.fingerprint,
        "vocab_size": len(codec),
        "bos_token_id": codec.bos_token_id,
        "eos_token_id": codec.eos_token_id,
        "pad_token_id": codec.pad_token_id,
        "splits": splits,
    }
    result["fingerprint"] = config_digest(result)
    write_json(output / "manifest.json", result)
    _emit("pretraining_data_ready", fingerprint=result["fingerprint"], output=str(output))
    return output


class PackedCorpus:
    """Read-only mmap; relocation preserves identity, corruption is checked before use."""

    def __init__(self, root: str | Path, split: str):
        if split not in ("train", "validation"):
            raise ValueError("Only train/validation corpora may enter this pipeline")
        root = Path(root).resolve()
        meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            type(meta.get("schema_version")) is not int
            or meta.get("schema_version") != 1
            or meta.get("work_package") != "WP07.2"
            or meta.get("scope") != "pretraining_train_validation_only"
            or meta.get("packing") != PACKING
            or meta.get("dtype") != "<u4"
            or config_digest({k: v for k, v in meta.items() if k != "fingerprint"})
            != meta.get("fingerprint")
        ):
            raise ValueError("Invalid packed-corpus manifest/fingerprint")
        if type(meta["block_size"]) is not int or meta["block_size"] < 3:
            raise ValueError("Invalid block size")
        spec = meta["splits"][split]
        if spec["file"] != f"{split}.u32":
            raise ValueError("Invalid token filename")
        path = (root / spec["file"]).resolve()
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != spec["size_bytes"]
            or spec["size_bytes"] != spec["tokens"] * 4
            or file_sha256(path) != spec["sha256"]
        ):
            raise ValueError("Packed tokens checksum/size mismatch")
        self.meta, self.split = meta, split
        self._paths = (root / "manifest.json", path)
        self._stamps = tuple(self._stamp(p) for p in self._paths)
        self.tokens = np.memmap(path, dtype="<u4", mode="r")
        if (
            len(self.tokens) < 3
            or self.tokens.max() >= meta["vocab_size"]
            or int(self.tokens[0]) != meta["bos_token_id"]
            or int(self.tokens[-1]) != meta["eos_token_id"]
            or int(np.count_nonzero(self.tokens == meta["bos_token_id"])) != spec["documents"]
            or int(np.count_nonzero(self.tokens == meta["eos_token_id"])) != spec["documents"]
            or spec["blocks"] != (len(self.tokens) - 2) // (meta["block_size"] - 1) + 1
            or spec["supervised_pairs"] != len(self.tokens) - spec["documents"]
        ):
            raise ValueError("Invalid packed token vocabulary/boundaries/counts")

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def assert_unchanged(self):
        if tuple(self._stamp(p) for p in self._paths) != self._stamps:
            raise RuntimeError(
                "Immutable corpus changed after verification; reopen and verify hashes"
            )

    def __len__(self):
        return self.meta["splits"][self.split]["blocks"]

    @property
    def identity(self):
        self.assert_unchanged()
        return {
            "fingerprint": self.meta["fingerprint"],
            "split": self.split,
            "tokenizer_fingerprint": self.meta["tokenizer_fingerprint"],
            "block_size": self.meta["block_size"],
            "vocab_size": self.meta["vocab_size"],
        }

    def batch(self, indices: list[int]) -> dict[str, torch.Tensor]:
        self.assert_unchanged()
        if not indices or any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise ValueError("Invalid packed block indices")
        width = self.meta["block_size"]
        ids = torch.full(
            (len(indices), width), self.meta["pad_token_id"], dtype=torch.long, device="cpu"
        )
        labels = torch.full_like(ids, -100, device="cpu")
        mask = torch.zeros_like(ids, dtype=torch.bool, device="cpu")
        for row, index in enumerate(indices):
            start = index * (width - 1)
            values = torch.tensor(
                np.asarray(self.tokens[start : start + width], dtype=np.int64), device="cpu"
            )
            ids[row, : len(values)] = values
            labels[row, : len(values)] = values.masked_fill(
                values == self.meta["bos_token_id"], -100
            )
            mask[row, : len(values)] = True
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    def validation_batches(self, batch_size: int):
        if self.split != "validation" or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Validation requires a validation corpus and positive batch size")
        for start in range(0, len(self), batch_size):
            yield self.batch(list(range(start, min(start + batch_size, len(self)))))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare immutable WP07 pretraining data; train/validation only"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    args = parser.parse_args()
    output = prepare_pretraining(
        args.dataset_root, args.tokenizer, args.output, block_size=args.block_size
    )
    print(json.dumps({"status": "passed", "prepared_root": str(output)}), flush=True)


if __name__ == "__main__":
    main()
