"""Embedding, normalization, RoPE and SwiGLU built from torch/autograd primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_size(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class TokenEmbedding(nn.Module):
    """[batch, sequence] IDs -> [batch, sequence, hidden_size], without scaling.

    PAD has an ordinary learned row: padding is handled by attention/loss masks.
    This also permits the future LM head to share exactly the same Parameter.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        _positive_size(vocab_size, "vocab_size")
        _positive_size(hidden_size, "hidden_size")
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an int32/int64 tensor [batch, sequence]")
        return F.embedding(input_ids, self.weight)


class RMSNorm(nn.Module):
    """x / sqrt(mean(x**2) + eps) * learned scale, without mean subtraction/bias.

    Reductions and scaling use FP32 for FP16/BF16 inputs, then return input dtype.
    FP64 is retained for high-precision reference and gradient checks.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        _positive_size(hidden_size, "hidden_size")
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point() or x.ndim < 1 or x.shape[-1] != self.weight.numel():
            raise ValueError("RMSNorm expects floating input with final dimension hidden_size")
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        value = x.to(compute_dtype)
        normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.to(compute_dtype)).to(x.dtype)


class SwiGLU(nn.Module):
    """Token-wise, bias-free FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        _positive_size(hidden_size, "hidden_size")
        _positive_size(ffn_hidden_size, "ffn_hidden_size")
        self.hidden_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point() or x.ndim < 1 or x.shape[-1] != self.hidden_size:
            raise ValueError("SwiGLU expects floating input with final dimension hidden_size")
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Adjacent-pair RoPE on [batch, heads, sequence, head_dim].

    Call separately for Q and K, never V. Positions are [sequence] or [batch, sequence].
    Frequencies are computed from integer indices on each call so casting the module
    to BF16/FP16 cannot quantize a persistent frequency buffer. No KV cache yet.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        _positive_size(head_dim, "head_dim")
        _positive_size(max_seq_len, "max_seq_len")
        if head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 1:
            raise ValueError("theta must be finite and > 1")
        self.head_dim, self.max_seq_len, self.theta = head_dim, max_seq_len, theta

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        if not x.is_floating_point() or x.ndim != 4 or x.shape[-1] != self.head_dim:
            raise ValueError("RoPE expects floating [batch, heads, sequence, head_dim]")
        batch, _, length, _ = x.shape
        if length > self.max_seq_len:
            raise ValueError("Sequence exceeds configured context length")
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device)
        if (
            position_ids.dtype not in (torch.int32, torch.int64)
            or position_ids.shape not in ((length,), (batch, length))
            or position_ids.device != x.device
        ):
            raise ValueError(
                "position_ids must be integer [sequence] or [batch, sequence] on x.device"
            )
        if torch.any(position_ids < 0) or torch.any(position_ids >= self.max_seq_len):
            raise ValueError("position_ids outside configured context length")
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        exponents = torch.arange(0, self.head_dim, 2, device=x.device, dtype=compute_dtype)
        inverse_frequency = self.theta ** (-exponents / self.head_dim)
        angles = position_ids.to(compute_dtype).unsqueeze(-1) * inverse_frequency
        if position_ids.ndim == 1:
            angles = angles.unsqueeze(0)
        cosine, sine = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        value = x.to(compute_dtype)
        even, odd = value[..., 0::2], value[..., 1::2]
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        return rotated.flatten(-2).to(x.dtype)
