"""WP07 training-engine settings, independent from legacy WP02 debug snapshots."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    stage: str = "pretrain"
    max_steps: int = 10
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 2
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    precision: str = "float32"

    def __post_init__(self):
        if self.stage not in ("pretrain", "sft"):
            raise ValueError("stage must be pretrain or sft")
        for name in ("max_steps", "gradient_accumulation_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0,max_steps)")
        for name in (
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "eps",
            "min_lr_ratio",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.learning_rate, self.eps, self.max_grad_norm) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer rate/epsilon/weight decay/gradient clipping")
        if not 0 <= self.min_lr_ratio <= 1 or not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Invalid LR ratio or Adam betas")
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported precision")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TrainingConfig:
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"Invalid training settings: {error}") from error
