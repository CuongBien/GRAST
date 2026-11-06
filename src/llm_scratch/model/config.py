"""Validated, serializable decoder configuration, independent of the torch runtime."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """MHA baseline; architecture sizes remain subject to the feasibility gate.

    The eight WP02 model fields are accepted unchanged. New architectural defaults
    are explicit in to_dict(), which the future model checkpoint must persist.
    """

    vocab_size: int
    context_length: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_hidden_size: int
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "context_length",
            "hidden_size",
            "num_layers",
            "num_heads",
            "ffn_hidden_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        for name in ("dropout", "rope_theta", "rms_norm_eps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"model.{name} must be finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 1 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta must be > 1 and rms_norm_eps must be positive")
        if type(self.tie_embeddings) is not bool:
            raise ValueError("tie_embeddings must be a boolean")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        if not isinstance(data, dict):
            raise ValueError("Model configuration must be a dictionary")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"Incomplete model configuration: {error}") from error
