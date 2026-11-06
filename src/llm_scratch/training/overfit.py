"""Diagnostic optimization of exactly one fixed batch; not a pretraining/SFT trainer."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from llm_scratch.evaluation.architecture import measure_batch
from llm_scratch.model.transformer import CausalLM


def overfit_batch(
    model: CausalLM,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    max_grad_norm: float = 1.0,
    progress: Callable[[dict], None] | None = None,
) -> list[dict]:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (
        type(max_grad_norm) not in (int, float)
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be finite and positive")
    if model.config.dropout != 0:
        raise ValueError("Fixed diagnostic protocol requires dropout=0")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimized) != len(parameters) or {id(p) for p in optimized} != {
        id(p) for p in parameters
    }:
        raise ValueError("Optimizer must cover every unique trainable parameter exactly once")
    history = [{"step": 0, **measure_batch(model, batch)}]
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.num_loss_tokens != history[0]["num_loss_tokens"] or not torch.isfinite(
            output.loss
        ):
            raise AssertionError("Missing/changed supervision or non-finite training loss")
        output.loss.backward()
        if any(parameter.grad is None for parameter in parameters):
            raise AssertionError("A trainable parameter is disconnected from diagnostic loss")
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=True, foreach=False
        )
        optimizer.step()
        record = {
            "step": step,
            **measure_batch(model, batch),
            "gradient_norm_before_clip": norm.item(),
        }
        history.append(record)
        if progress is not None:
            progress(record)
    return history
