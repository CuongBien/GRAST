"""WP04 train-only BPE artifact pipeline with frozen held-out round-trip/statistics."""

from __future__ import annotations

import json
import re
import time
import tomllib
from dataclasses import replace
from pathlib import Path, PurePosixPath

from llm_scratch.data.collator import encode_pretraining
from llm_scratch.data.config import _fields
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.reasoning.formatting import encode_reasoning_prompt, encode_sft_example
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run


def load_tokenizer_training_config(
    path: str | Path,
    *,
    output_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> LoadedConfig:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    _fields(
        data,
        {
            "schema_version": int,
            "tokenizer_config": str,
            "dataset_manifest": str,
            "dataset_runs_root": str,
            "experiment": dict,
            "reproducibility": dict,
            "artifacts": dict,
            "training": dict,
            "evaluation": dict,
        },
        "tokenizer training",
    )
    if data["schema_version"] != 1:
        raise ValueError("Unsupported tokenizer training schema")
    _fields(data["experiment"], {"name": str, "phase": int}, "experiment")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", data["experiment"]["name"])
        or data["experiment"]["phase"] != 1
    ):
        raise ValueError("Invalid phase/name")
    _fields(data["reproducibility"], {"seed": int}, "reproducibility")
    if not 0 <= data["reproducibility"]["seed"] < 2**32:
        raise ValueError("Invalid seed")
    _fields(data["artifacts"], {"root": str}, "artifacts")
    data["artifacts"]["root"] = (
        str(Path(output_root).resolve())
        if output_root is not None
        else str((path.parent / data["artifacts"]["root"]).resolve())
    )
    _fields(
        data["training"],
        {
            "file": str,
            "field": str,
            "max_records": int,
            "min_frequency": int,
            "max_piece_bytes": int,
        },
        "training",
    )
    if data["training"]["max_records"] < 1 or not data["training"]["field"]:
        raise ValueError("Training record limit/field must be nonempty")
    if PurePosixPath(data["training"]["file"]).name != "train.jsonl":
        raise ValueError("Tokenizer training is restricted to a train split")
    _fields(data["evaluation"], {"files": list}, "evaluation")
    if not data["evaluation"]["files"]:
        raise ValueError("Need evaluation files")
    for spec in data["evaluation"]["files"]:
        _fields(spec, {"file": str, "field": str}, "evaluation source")
        if not spec["field"]:
            raise ValueError("Empty evaluation field")
    tokenizer_path = (path.parent / data.pop("tokenizer_config")).resolve()
    tokenizer_text = tokenizer_path.read_text(encoding="utf-8")
    component = tomllib.loads(tokenizer_text)
    _fields(component, {"tokenizer": dict}, "tokenizer component")
    _fields(
        component["tokenizer"],
        {"algorithm": str, "vocab_size": int, "special_tokens": list},
        "tokenizer",
    )
    if component["tokenizer"]["algorithm"] != "bpe":
        raise ValueError("Only from-scratch BPE is supported")
    data["tokenizer"] = component["tokenizer"]
    ByteBPETokenizer(
        vocab_size=data["tokenizer"]["vocab_size"],
        min_frequency=data["training"]["min_frequency"],
        special_tokens=data["tokenizer"]["special_tokens"],
        max_piece_bytes=data["training"]["max_piece_bytes"],
    )
    data["dataset_manifest"] = str((path.parent / data["dataset_manifest"]).resolve())
    data["dataset_runs_root"] = str((path.parent / data["dataset_runs_root"]).resolve())
    data["dataset_root"] = str(Path(dataset_root).resolve()) if dataset_root is not None else None
    return LoadedConfig(data, {"training": text, "tokenizer": tokenizer_text}, path)


def _dataset(config: LoadedConfig) -> tuple[LoadedConfig, dict, Path]:
    data = config.data
    manifest = json.loads(Path(data["dataset_manifest"]).read_text(encoding="utf-8"))
    fingerprint = manifest.get("dataset_fingerprint")
    if manifest.get("work_package") != "WP03" or manifest.get("scope") != "full":
        raise ValueError("Need a full WP03 dataset manifest")
    if (
        config_digest(
            {key: value for key, value in manifest.items() if key != "dataset_fingerprint"}
        )
        != fingerprint
    ):
        raise ValueError("Dataset manifest fingerprint mismatch")
    root = Path(data["dataset_root"]) if data["dataset_root"] is not None else None
    if root is None:
        for candidate in sorted(
            Path(data["dataset_runs_root"]).glob("*/artifacts/datasets/manifest.json"), reverse=True
        ):
            lifecycle = candidate.parents[2] / "run.json"
            if (
                lifecycle.is_file()
                and json.loads(lifecycle.read_text())["status"] == "completed"
                and json.loads(candidate.read_text(encoding="utf-8")) == manifest
            ):
                root = candidate.parent
                break
    if root is None:
        raise FileNotFoundError("No matching WP03 data: run prepare-data or pass --dataset-root")
    root = root.resolve()
    return (
        replace(
            config, data={**data, "dataset_root": str(root), "dataset_fingerprint": fingerprint}
        ),
        manifest,
        root,
    )


def _data_file(root: Path, relative: str) -> Path:
    if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise ValueError("Dataset file must be relative to its root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path escapes its root")
    return path


def _texts(path: Path, field: str, expected_count: int, limit: int | None = None):
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line).get(field)
            if not isinstance(value, str):
                raise ValueError(f"Expected text field {field!r} in {path}")
            count += 1
            yield value
            if limit is not None and count >= limit:
                break
    expected = min(expected_count, limit) if limit is not None else expected_count
    if count != expected:
        raise ValueError(f"Record count mismatch in {path}: {count} != {expected}")


def train_tokenizer(config: LoadedConfig) -> Path:
    config, manifest, root = _dataset(config)
    data, training = config.data, config.data["training"]
    required = {training["file"]} | {spec["file"] for spec in data["evaluation"]["files"]}
    with Run(config) as run:
        run.metadata["execution_device"] = "cpu"
        run.metadata["runtime_inventory"] = inspect_environment("cpu")
        run.metadata["dataset_fingerprint"] = manifest["dataset_fingerprint"]
        write_json(run.path / "metadata.json", run.metadata)
        write_json(run.path / "config_sources" / "dataset_manifest.json", manifest)
        for relative in sorted(required):
            info = manifest["files"].get(relative)
            if info is None or info["path"] != relative:
                raise ValueError(f"Dataset manifest does not contain {relative}")
            path = _data_file(root, relative)
            if file_sha256(path) != info["sha256"] or path.stat().st_size != info["size_bytes"]:
                raise ValueError(f"Dataset payload checksum mismatch: {relative}")
        run.log("dataset_verified", dataset_fingerprint=manifest["dataset_fingerprint"])
        start = time.perf_counter()

        def progress(merges, frequency):
            run.log("bpe_merge_progress", merges=merges, pair_frequency=frequency)

        tokenizer = ByteBPETokenizer.train(
            _texts(
                _data_file(root, training["file"]),
                training["field"],
                manifest["files"][training["file"]]["records"],
                training["max_records"],
            ),
            vocab_size=data["tokenizer"]["vocab_size"],
            special_tokens=data["tokenizer"]["special_tokens"],
            min_frequency=training["min_frequency"],
            max_piece_bytes=training["max_piece_bytes"],
            progress=progress,
        )
        training_seconds = time.perf_counter() - start
        output = run.path / "artifacts" / "tokenizer"
        tokenizer.save(output)
        loaded = ByteBPETokenizer.load(output, expected_fingerprint=tokenizer.fingerprint)
        reports, save_load_checks = {}, 0
        for spec in data["evaluation"]["files"]:
            relative, field = spec["file"], spec["field"]
            characters = utf8_bytes = tokens = documents = unknown = 0
            for text in _texts(
                _data_file(root, relative), field, manifest["files"][relative]["records"]
            ):
                ids = loaded.encode(text)
                if loaded.decode(ids, errors="strict") != text:
                    raise AssertionError(f"Round-trip failed: {relative}:{documents}")
                if documents < 16:
                    if tokenizer.encode(text) != ids:
                        raise AssertionError("Save/load changed token IDs")
                    save_load_checks += 1
                documents += 1
                characters += len(text)
                utf8_bytes += len(text.encode("utf-8"))
                tokens += len(ids)
                unknown += ids.count(loaded.unk_token_id)
            reports[f"{relative}:{field}"] = {
                "documents": documents,
                "characters": characters,
                "utf8_bytes": utf8_bytes,
                "tokens": tokens,
                "tokens_per_character": tokens / characters if characters else None,
                "bytes_per_token": utf8_bytes / tokens if tokens else None,
                "unknown_tokens": unknown,
                "round_trip_documents": documents,
            }
            run.log("tokenization_evaluated", source=relative, documents=documents)
        if loaded.fingerprint != tokenizer.fingerprint:
            raise AssertionError("Evaluation mutated the codec")
        sample_path = _data_file(root, training["file"])
        sample = next(
            _texts(
                sample_path, training["field"], manifest["files"][training["file"]]["records"], 1
            )
        )
        pretraining = encode_pretraining(loaded, sample)
        if (
            loaded.decode(pretraining["input_ids"], skip_special_tokens=True, errors="strict")
            != sample
        ):
            raise AssertionError("Pretraining framing changed text")
        question = "What is 2 + 3?"
        sft = encode_sft_example(loaded, question, "Add the quantities.", "5")
        prompt = encode_reasoning_prompt(loaded, question)
        if sft["input_ids"][: len(prompt)] != prompt:
            raise AssertionError("SFT and inference prefix differ")
        report = {
            "schema_version": 1,
            "work_package": "WP04",
            "tokenizer_fingerprint": loaded.fingerprint,
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "training_source": {**training, "split": "train"},
            "training": tokenizer.training_statistics,
            "training_seconds": training_seconds,
            "vocab_size": len(loaded),
            "target_vocab_size": loaded.target_vocab_size,
            "special_token_ids": loaded.special_tokens,
            "evaluation": reports,
            "save_load_id_checks": save_load_checks,
            "integration_checks": {
                "pretraining_round_trip": True,
                "sft_inference_prefix_equal": True,
            },
            "protocol": (
                "All metrics use frozen ordinary-text encoding without BOS/EOS; characters are "
                "Unicode code points. Held-out data never enters training or vocabulary selection."
            ),
        }
        write_json(run.path / "artifacts" / "tokenizer_report.json", report)
        run.metadata["tokenizer_fingerprint"] = loaded.fingerprint
        write_json(run.path / "metadata.json", run.metadata)
        write_json(
            run.path / "result.json",
            {
                "status": "completed",
                "tokenizer_fingerprint": loaded.fingerprint,
                "vocab_size": len(loaded),
                "report": "artifacts/tokenizer_report.json",
                "tokenizer": "artifacts/tokenizer",
            },
        )
        run.log(
            "tokenizer_completed", tokenizer_fingerprint=loaded.fingerprint, vocab_size=len(loaded)
        )
    return run.path
