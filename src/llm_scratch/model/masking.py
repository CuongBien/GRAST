"""Shared hidden-state/padding contract for causal attention and decoder blocks."""

from __future__ import annotations

import torch
from torch import Tensor

from llm_scratch.model.config import ModelConfig


def valid_token_mask(x: Tensor, attention_mask: Tensor | None, config: ModelConfig) -> Tensor:
    """Validate before normalization/projection and return bool [batch, sequence]."""
    if not x.is_floating_point() or x.ndim != 3 or x.shape[-1] != config.hidden_size:
        raise ValueError("Expected floating [batch, sequence, hidden_size]")
    batch, length, _ = x.shape
    if batch == 0 or not 1 <= length <= config.context_length:
        raise ValueError("Need a nonempty batch/sequence within configured context length")
    if attention_mask is None:
        return torch.ones((batch, length), dtype=torch.bool, device=x.device)
    if (
        attention_mask.shape != (batch, length)
        or attention_mask.device != x.device
        or attention_mask.dtype
        not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("attention_mask must be bool/integer [batch, sequence] on x.device")
    if torch.any((attention_mask != 0) & (attention_mask != 1)):
        raise ValueError("attention_mask values must be 0 or 1")
    return attention_mask.bool()
