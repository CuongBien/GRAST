"""Strict, separate WP06.5 online profiling protocol schema."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_scratch.model.config import ModelConfig
from llm_scratch.utils.config import LoadedConfig


@dataclass(frozen=True)
class ProfileConfig:
    prompt_lengths: list[int]
    new_tokens: int
    batch_size: int
    warmup: int
    repeats: int
    context_length: int
    precision: str
    device: str

    def __post_init__(self):
        if (
            type(self.prompt_lengths) is not list
            or not self.prompt_lengths
            or any(type(n) is not int or n < 1 for n in self.prompt_lengths)
            or self.prompt_lengths != sorted(set(self.prompt_lengths))
        ):
            raise ValueError("prompt_lengths must be sorted unique positive integers")
        for name in ("new_tokens", "batch_size", "warmup", "repeats", "context_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.new_tokens < 2:
            raise ValueError("new_tokens >= 2 required for next-token latency")
        if max(self.prompt_lengths) + self.new_tokens > self.context_length:
            raise ValueError("Profiling context must fit prompt plus all new tokens")
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported profiling precision")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Profiling device must be cpu or cuda (no fallback)")


def load_profile_config(path: str | Path, *, device=None, output_root=None) -> LoadedConfig:
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    if set(data) != {
        "schema_version",
        "model_config",
        "experiment",
        "profiling",
        "reproducibility",
        "artifacts",
    }:
        raise ValueError("Invalid profiling config tables/keys")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Expected profiling schema_version=1")
    if data["experiment"] != {"name": "wp06-5-inference-profile"}:
        raise ValueError("Invalid profiling experiment name")
    repro = data["reproducibility"]
    if set(repro) != {"seed", "deterministic", "num_threads"}:
        raise ValueError("Invalid reproducibility keys")
    if (
        type(repro["seed"]) is not int
        or not 0 <= repro["seed"] < 2**32
        or type(repro["deterministic"]) is not bool
        or type(repro["num_threads"]) is not int
        or repro["num_threads"] < 1
    ):
        raise ValueError("Invalid reproducibility values")
    try:
        policy = ProfileConfig(**data["profiling"])
    except TypeError as error:
        raise ValueError(f"Invalid profiling settings: {error}") from error
    if device is not None:
        policy = ProfileConfig(**(asdict(policy) | {"device": device}))
    model_source = (path.parent / data.pop("model_config")).read_text(encoding="utf-8")
    model_data = tomllib.loads(model_source)
    if set(model_data) != {"model"}:
        raise ValueError("Expected [model] configuration")
    architecture = ModelConfig.from_dict(model_data["model"])
    if policy.context_length < architecture.context_length:
        raise ValueError("Profiling context override only permits extension")
    if architecture.vocab_size < 9:
        raise ValueError("Profiling synthetic IDs require vocab_size >= 9")
    if set(data["artifacts"]) != {"root"}:
        raise ValueError("Expected artifacts.root")
    root = (
        Path(output_root).resolve()
        if output_root
        else (path.parent / data["artifacts"]["root"]).resolve()
    )
    data.update(profiling=asdict(policy), source_model=architecture.to_dict())
    data["artifacts"]["root"] = str(root)
    return LoadedConfig(data, {"profiling": source, "model": model_source}, path)
