"""AdamW over unique trainable Parameters and update-indexed warmup/cosine LR."""

import math

import torch

from llm_scratch.training.config import TrainingConfig


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    # parameters() deduplicates tied embedding/LM head. Norm/bias vectors have no decay.
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        [g for g in groups if g["params"]],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        foreach=False,
        fused=False,
    )


class WarmupCosine:
    """LR for update 0 is set at init; step() follows a successful optimizer update.

    Warmup W uses (i+1)/W, then cosine from base LR to min_lr_ratio on the final
    planned update. A one-update decay tail uses base LR. AMP skips do not advance.
    """

    def __init__(self, optimizer, config: TrainingConfig):
        self.optimizer, self.config = optimizer, config
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.completed_updates = 0
        self._apply()

    def multiplier(self, index: int) -> float:
        cfg = self.config
        index = min(index, cfg.max_steps - 1)
        if index < cfg.warmup_steps:
            return (index + 1) / cfg.warmup_steps
        tail = cfg.max_steps - cfg.warmup_steps
        progress = (index - cfg.warmup_steps) / (tail - 1) if tail > 1 else 0.0
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    def _apply(self):
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * self.multiplier(self.completed_updates)

    def step(self):
        if self.completed_updates >= self.config.max_steps:
            raise RuntimeError("Scheduler budget exhausted")
        self.completed_updates += 1
        self._apply()

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "config": self.config.to_dict(),
            "base_lrs": self.base_lrs.copy(),
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: dict):
        if (
            state.get("schema_version") != 1
            or state.get("config") != self.config.to_dict()
            or state.get("base_lrs") != self.base_lrs
            or type(state.get("completed_updates")) is not int
            or not 0 <= state["completed_updates"] <= self.config.max_steps
        ):
            raise ValueError("Incompatible scheduler state")
        self.completed_updates = state["completed_updates"]
        self._apply()
