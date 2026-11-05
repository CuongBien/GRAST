"""Strict WP03 data config, separate from the WP02 model/run config schema."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from llm_scratch.utils.config import LoadedConfig


def _fields(table: dict, fields: dict, label: str) -> None:
    if not isinstance(table, dict) or set(table) != set(fields):
        raise ValueError(f"{label}: expected fields {sorted(fields)}")
    for key, kind in fields.items():
        if type(table[key]) is not kind:
            raise ValueError(f"{label}.{key} must be {kind.__name__}")


def load_data_config(path: str | Path, *, output_root: str | Path | None = None) -> LoadedConfig:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    _fields(
        data,
        {
            "schema_version": int,
            "sources_lock": str,
            "experiment": dict,
            "reproducibility": dict,
            "artifacts": dict,
            "tinystories": dict,
            "gsm8k": dict,
            "math": dict,
        },
        "data config",
    )
    if data["schema_version"] != 1:
        raise ValueError("Unsupported data schema_version")
    _fields(data["experiment"], {"name": str, "phase": int}, "experiment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", data["experiment"]["name"]):
        raise ValueError("Invalid experiment.name")
    if data["experiment"]["phase"] != 1:
        raise ValueError("WP03 data belongs to phase 1")
    _fields(data["reproducibility"], {"seed": int}, "reproducibility")
    if not 0 <= data["reproducibility"]["seed"] < 2**32:
        raise ValueError("Invalid seed")
    _fields(data["artifacts"], {"root": str, "cache_root": str}, "artifacts")
    for key, value in data["artifacts"].items():
        if not value.strip():
            raise ValueError(f"Empty artifacts.{key}")
        data["artifacts"][key] = str((path.parent / value).resolve())
    if output_root is not None:
        data["artifacts"]["root"] = str(Path(output_root).resolve())
    _fields(data["tinystories"], {"train_records": int, "heldout_records": int}, "tinystories")
    _fields(data["gsm8k"], {"validation_records": int}, "gsm8k")
    _fields(
        data["math"],
        {
            "train_per_level": int,
            "validation_per_level": int,
            "test_per_level": int,
            "pilot_per_level_per_split": int,
            "levels": list,
        },
        "math",
    )
    for section in ("tinystories", "gsm8k", "math"):
        if any(value <= 0 for value in data[section].values() if type(value) is int):
            raise ValueError(f"{section} counts must be positive")
    if data["tinystories"]["heldout_records"] < 2:
        raise ValueError("Need at least two held-out TinyStories records")
    math = data["math"]
    if math["pilot_per_level_per_split"] > min(
        math["validation_per_level"], math["test_per_level"]
    ):
        raise ValueError("Pilot cannot exceed validation/test pool per level")
    if not math["levels"]:
        raise ValueError("Need at least one difficulty level")
    level_names = set()
    for level in math["levels"]:
        _fields(
            level,
            {
                "name": str,
                "operations": int,
                "reasoning_steps": int,
                "prompt_length": str,
                "operand_min": int,
                "operand_max": int,
                "operators": list,
            },
            "math.level",
        )
        if (
            not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", level["name"])
            or level["name"] in level_names
        ):
            raise ValueError("Difficulty names must be safe and unique")
        level_names.add(level["name"])
        count, depth = level["operations"], level["reasoning_steps"]
        if not 1 <= depth <= 8 or not depth <= count <= min(2**depth - 1, 31):
            raise ValueError("Invalid operation count/dependency depth")
        if not -1000 <= level["operand_min"] <= level["operand_max"] <= 1000:
            raise ValueError("Invalid operand range")
        if level["prompt_length"] not in ("short", "medium", "long"):
            raise ValueError("Invalid prompt_length")
        if not level["operators"] or any(
            op not in ("+", "-", "*", "/") for op in level["operators"]
        ):
            raise ValueError("Unsupported arithmetic operator")
    lock_path = path.parent / data.pop("sources_lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _fields(lock, {"schema_version": int, "tinystories": dict, "gsm8k": dict}, "sources lock")
    if lock["schema_version"] != 1:
        raise ValueError("Unsupported sources lock version")
    for name, splits in (("tinystories", {"train", "validation"}), ("gsm8k", {"train", "test"})):
        source = lock[name]
        _fields(
            source,
            {
                "repository": str,
                "revision": str,
                "license": str,
                "card_url": str,
                "selection": str,
                "files": dict,
            },
            name,
        )
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]) or not source["license"]:
            raise ValueError("Sources must pin a full revision and declare license metadata")
        if set(source["files"]) != splits:
            raise ValueError(f"Invalid {name} upstream splits")
        for spec in source["files"].values():
            _fields(spec, {"url": str, "sha256": str, "size_bytes": int}, "source file")
            if not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) or spec["size_bytes"] <= 0:
                raise ValueError("Source requires valid SHA-256 and size")
            if not spec["url"].startswith("https://") or source["revision"] not in spec["url"]:
                raise ValueError("Source URL must use HTTPS and the pinned revision")
    data["sources"] = lock
    return LoadedConfig(data, {"data": text}, path)
