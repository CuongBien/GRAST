"""Compose TOML research configurations and reject unknown or invalid settings."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = {
    "experiment": {"name": str, "phase": int},
    "model": {
        "vocab_size": int,
        "context_length": int,
        "hidden_size": int,
        "num_layers": int,
        "num_heads": int,
        "ffn_hidden_size": int,
        "dropout": float,
        "tie_embeddings": bool,
    },
    "tokenizer": {"algorithm": str, "vocab_size": int, "special_tokens": list},
    "training": {
        "stage": str,
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "max_steps": int,
        "learning_rate": float,
        "weight_decay": float,
        "warmup_steps": int,
        "max_grad_norm": float,
        "precision": str,
    },
    "generation": {
        "max_new_tokens": int,
        "temperature": float,
        "top_p": float,
        "k_values": list,
    },
    "reproducibility": {"seed": int, "deterministic": bool, "num_threads": int},
    "environment": {"device": str, "require_gpu_name": str},
    "artifacts": {"root": str},
}
COMPONENTS = ("model", "tokenizer", "training", "generation")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def config_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _keys(actual: dict, expected: set[str], label: str) -> None:
    if set(actual) != expected:
        raise ValueError(
            f"{label}: missing={sorted(expected - set(actual))}, "
            f"unknown={sorted(set(actual) - expected)}"
        )


def validate_config(data: dict) -> None:
    """Validate a fully resolved snapshot; booleans do not count as integers."""
    _keys(data, set(SCHEMA) | {"schema_version"}, "config")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version; expected 1")
    for section, fields in SCHEMA.items():
        table = data[section]
        if not isinstance(table, dict):
            raise ValueError(f"{section} must be a table")
        _keys(table, set(fields), section)
        for name, expected in fields.items():
            value = table[name]
            valid = type(value) is expected
            if expected is float:
                valid = type(value) in (float, int) and math.isfinite(value)
            if not valid:
                raise ValueError(f"{section}.{name} must be {expected.__name__}")
            if expected is int and name not in ("seed", "warmup_steps") and value <= 0:
                raise ValueError(f"{section}.{name} must be positive")
    exp, model, tok = data["experiment"], data["model"], data["tokenizer"]
    train, gen, repro = data["training"], data["generation"], data["reproducibility"]
    env = data["environment"]
    checks = [
        (bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", exp["name"])), "experiment.name"),
        (exp["phase"] in (1, 2), "experiment.phase"),
        (model["hidden_size"] % model["num_heads"] == 0, "model.hidden_size / num_heads"),
        (0 <= model["dropout"] < 1, "model.dropout"),
        (tok["algorithm"] == "bpe", "tokenizer.algorithm"),
        (tok["vocab_size"] == model["vocab_size"], "model/tokenizer vocab_size mismatch"),
        (train["stage"] in ("pretrain", "sft"), "training.stage"),
        (train["precision"] in ("float32", "bfloat16", "float16"), "training.precision"),
        (0 <= train["warmup_steps"] <= train["max_steps"], "training.warmup_steps"),
        (train["learning_rate"] > 0, "training.learning_rate"),
        (train["weight_decay"] >= 0, "training.weight_decay"),
        (train["max_grad_norm"] > 0, "training.max_grad_norm"),
        (0 < gen["max_new_tokens"] <= model["context_length"], "generation.max_new_tokens"),
        (gen["temperature"] > 0 and 0 < gen["top_p"] <= 1, "generation.temperature/top_p"),
        (0 <= repro["seed"] < 2**32, "reproducibility.seed"),
        (env["device"] in ("cpu", "cuda", "auto"), "environment.device"),
        (not env["require_gpu_name"] or env["device"] != "cpu", "GPU requirement on CPU"),
        (bool(data["artifacts"]["root"].strip()), "artifacts.root"),
    ]
    for valid, label in checks:
        if not valid:
            raise ValueError(f"Invalid {label}")
    tokens = tok["special_tokens"]
    if not tokens or any(type(t) is not str or not t for t in tokens):
        raise ValueError("tokenizer.special_tokens must be nonempty strings")
    if len(set(tokens)) != len(tokens) or len(tokens) >= tok["vocab_size"]:
        raise ValueError("tokenizer.special_tokens must be unique and fit vocabulary")
    ks = gen["k_values"]
    if not ks or any(type(k) is not int or k not in (1, 2, 4, 8, 16) for k in ks):
        raise ValueError("generation.k_values must contain values from 1,2,4,8,16")
    if ks != sorted(set(ks)):
        raise ValueError("generation.k_values must be sorted and unique")


@dataclass(frozen=True)
class LoadedConfig:
    data: dict[str, Any]
    sources: dict[str, str]
    path: Path

    @property
    def sha256(self) -> str:
        return config_digest(self.data)


def load_config(
    path: str | Path,
    *,
    seed: int | None = None,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> LoadedConfig:
    path = Path(path).resolve()
    sources = {"experiment": path.read_text(encoding="utf-8")}
    data = tomllib.loads(sources["experiment"])
    expected = {
        "schema_version",
        "components",
        "experiment",
        "reproducibility",
        "environment",
        "artifacts",
    }
    _keys(data, expected, "experiment file")
    components = data.pop("components")
    if not isinstance(components, dict):
        raise ValueError("components must be a table")
    _keys(components, set(COMPONENTS), "components")
    for section, relative in components.items():
        if not isinstance(relative, str):
            raise ValueError(f"components.{section} must be a path string")
        sources[section] = (path.parent / relative).read_text(encoding="utf-8")
        component = tomllib.loads(sources[section])
        _keys(component, {section}, f"component {section}")
        data[section] = component[section]
    validate_config(data)
    if seed is not None:
        data["reproducibility"]["seed"] = seed
    if device is not None:
        data["environment"]["device"] = device
    if output_root is not None:
        data["artifacts"]["root"] = str(Path(output_root).resolve())
    validate_config(data)
    data["artifacts"]["root"] = str((path.parent / data["artifacts"]["root"]).resolve())
    return LoadedConfig(data, sources, path)
