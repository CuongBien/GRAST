"""Runtime generation settings, separate from WP02's candidate-count experiment grid."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_id: int | None = None
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        for name in ("max_new_tokens", "top_k", "pad_token_id"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.do_sample) is not bool:
            raise ValueError("do_sample must be boolean")
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.top_p > 1:
            raise ValueError("top_p must be <= 1")
        if self.eos_token_id is not None and (
            type(self.eos_token_id) is not int or self.eos_token_id < 0
        ):
            raise ValueError("eos_token_id must be a nonnegative integer or None")

    def validate_vocab(self, vocab_size: int) -> None:
        if self.top_k > vocab_size:
            raise ValueError("top_k exceeds vocabulary size; 0 disables top-k")
        for value in (self.pad_token_id, self.eos_token_id):
            if value is not None and value >= vocab_size:
                raise ValueError("Special token ID outside vocabulary")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GenerationConfig:
        if not isinstance(data, dict):
            raise ValueError("generation settings must be a dictionary")
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"Invalid generation settings: {error}") from error
