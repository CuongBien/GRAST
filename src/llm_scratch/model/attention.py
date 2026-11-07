"""WP05.2 explicit causal MHA with RoPE, padding masks and safe masked softmax."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm_scratch.model.cache import KVCacheStep
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import RotaryEmbedding
from llm_scratch.model.masking import valid_token_mask


def _scaled_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    allowed: Tensor,
    *,
    dropout_p: float,
    training: bool,
) -> Tensor:
    """Compute scores/softmax/value reduction in FP32 (FP64 for FP64 inputs).

    Never run softmax on an all-negative-infinity row: substitute finite scores
    first, then force every forbidden probability to zero, including empty rows.
    Explicitly disable autocast so the score matmul cannot fall back to FP16/BF16.
    """
    compute_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = (query.to(compute_dtype) / math.sqrt(query.shape[-1])) @ key.to(
            compute_dtype
        ).transpose(-2, -1)
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = scores.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0)
        probabilities = scores.softmax(dim=-1).masked_fill(~allowed, 0.0)
        probabilities = F.dropout(probabilities, p=dropout_p, training=training)
        attended = probabilities @ value.to(compute_dtype)
    return attended.to(value.dtype)


class CausalSelfAttention(nn.Module):
    """Bias-free MHA; input/output [batch, sequence, hidden_size].

    attention_mask is bool or integer [batch, sequence], 1/True = valid token.
    Padded inputs are cleared before projection; padded query outputs are exactly
    zero. A fully padded example has zero output/gradients, without NaNs.
    Causality follows sequence order, independently of explicit RoPE position IDs.
    Optional cache_step appends already-rotated K and raw V for the supplied layer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length, config.rope_theta)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        *,
        cache_step: KVCacheStep | None = None,
        layer_index: int | None = None,
    ) -> Tensor:
        if cache_step is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Cached attention requires eval and no_grad/inference_mode")
            if layer_index is None:
                raise ValueError("Cached attention requires layer_index")
            if position_ids is not None:
                expected = cache_step.position_ids
                if (
                    not isinstance(position_ids, Tensor)
                    or position_ids.shape not in ((x.shape[1],), expected.shape)
                    or not torch.equal(position_ids.expand_as(expected), expected)
                ):
                    raise ValueError("position_ids do not match cache transaction")
            position_ids = cache_step.position_ids
        elif layer_index is not None:
            raise ValueError("layer_index requires cache_step")
        valid = valid_token_mask(x, attention_mask, self.config)
        batch, length, hidden_size = x.shape
        # Clear even nonfinite padding values so zero probabilities never multiply NaN/Inf V.
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)

        def heads(projection: nn.Linear) -> Tensor:
            return (
                projection(x)
                .view(batch, length, self.config.num_heads, self.config.head_dim)
                .transpose(1, 2)
            )

        query = self.rope(heads(self.q_proj), position_ids)
        key = self.rope(heads(self.k_proj), position_ids)
        value = heads(self.v_proj)
        if cache_step is None:
            causal = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()
            key_mask = valid
        else:
            if cache_step.end - cache_step.start != length or not torch.equal(
                valid, cache_step.query_mask
            ):
                raise ValueError("Cached attention chunk/mask does not match transaction")
            key, value = cache_step.update(layer_index, key, value)
            query_offsets = torch.arange(cache_step.start, cache_step.end, device=x.device)
            key_offsets = torch.arange(cache_step.end, device=x.device)
            causal = key_offsets[None, :] <= query_offsets[:, None]
            key_mask = cache_step.attention_mask
        allowed = causal[None, None, :, :] & key_mask[:, None, None, :] & valid[:, None, :, None]
        attended = _scaled_attention(
            query, key, value, allowed, dropout_p=self.config.dropout, training=self.training
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, length, hidden_size)
        return self.out_proj(merged).masked_fill(~valid.unsqueeze(-1), 0.0)
