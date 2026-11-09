"""Summarize measured wall-clock samples; CUDA allocator bytes remain distinct."""

from __future__ import annotations

import math

import numpy as np


def distribution(values: list[float]) -> dict:
    if not values or any(not math.isfinite(n) or n <= 0 for n in values):
        raise ValueError("Metrics must contain finite positive measured values")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize_samples(samples: list[dict], batch_size: int, new_tokens: int) -> dict:
    if (
        not samples
        or type(batch_size) is not int
        or batch_size < 1
        or type(new_tokens) is not int
        or new_tokens < 2
    ):
        raise ValueError("Need samples, positive batch size and >=2 new tokens")
    for sample in samples:
        if len(sample["next_token_latencies_ms"]) != new_tokens - 1 or sample["total_ms"] < sample[
            "ttft_ms"
        ] + sum(sample["next_token_latencies_ms"]):
            raise ValueError("Inconsistent token count/timing boundaries")
    decode_times = [sum(s["next_token_latencies_ms"]) for s in samples]
    result = {
        "ttft_ms": distribution([s["ttft_ms"] for s in samples]),
        "next_token_latency_ms": distribution(
            [n for s in samples for n in s["next_token_latencies_ms"]]
        ),
        "total_ms": distribution([s["total_ms"] for s in samples]),
        "overall_tokens_per_second": distribution(
            [batch_size * new_tokens * 1000 / s["total_ms"] for s in samples]
        ),
        "decode_tokens_per_second": distribution(
            [batch_size * (new_tokens - 1) * 1000 / n for n in decode_times]
        ),
        "vram_bytes": None,
    }
    memory = [s["vram_bytes"] for s in samples]
    if any(m is None for m in memory) and not all(m is None for m in memory):
        raise ValueError("Cannot mix CPU and CUDA memory samples")
    if memory[0] is not None:
        result["vram_bytes"] = {
            key: {"min": min(m[key] for m in memory), "max": max(m[key] for m in memory)}
            for key in memory[0]
        }
    return result
