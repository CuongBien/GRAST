"""WP06.1 bounded static MHA cache with transactional, layer-consistent append."""

from __future__ import annotations

import torch
from torch import Tensor

from llm_scratch.model.config import ModelConfig


class KVCache:
    """One request/batch, fixed device/dtype/geometry, no autograd or silent eviction.

    K is stored after RoPE, V before rotation. Buffers are [B,H,capacity,D]. A step
    writes only the uncommitted tail; all layers must complete before length advances.
    Reset invalidates logical contents but retains storage. Returned views are borrowed:
    callers must not modify them or retain them across reset/reuse.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        *,
        max_length: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(config, ModelConfig):
            raise ValueError("config must be ModelConfig")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        capacity = config.context_length if max_length is None else max_length
        if type(capacity) is not int or not 1 <= capacity <= config.context_length:
            raise ValueError("max_length must be positive and within model context_length")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("Unsupported KV floating dtype")
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Cache supports CPU or CUDA")
        self._config, self._batch_size, self._capacity, self._dtype = (
            config,
            batch_size,
            capacity,
            dtype,
        )
        shape = (batch_size, config.num_heads, capacity, config.head_dim)
        # Ordinary no-grad buffers remain reusable even when created inside inference_mode.
        with torch.inference_mode(False), torch.no_grad():
            self._layers = tuple(
                (
                    torch.empty(shape, device=target, dtype=dtype),
                    torch.empty(shape, device=target, dtype=dtype),
                )
                for _ in range(config.num_layers)
            )
            self._valid = torch.empty((batch_size, capacity), device=target, dtype=torch.bool)
            self._positions = torch.empty((batch_size, capacity), device=target, dtype=torch.long)
        self._device = self._valid.device  # Canonicalize e.g. cuda -> cuda:0, cpu:0 -> cpu.
        self._length = 0
        self._active: KVCacheStep | None = None

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._capacity

    @property
    def sequence_length(self) -> int:
        return self._length

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def attention_mask(self) -> Tensor:
        return self._valid[:, : self._length]

    @property
    def position_ids(self) -> Tensor:
        return self._positions[:, : self._length]

    def _layer_index(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < self.config.num_layers:
            raise ValueError("Invalid cache layer index")

    def get_layer(self, index: int) -> tuple[Tensor, Tensor]:
        """Return only committed K/V views, including during an unfinished append."""
        self._layer_index(index)
        key, value = self._layers[index]
        return key[:, :, : self._length], value[:, :, : self._length]

    def begin_step(
        self,
        num_tokens: int,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> KVCacheStep:
        if self._active is not None:
            raise RuntimeError("A cache append is already active")
        if type(num_tokens) is not int or num_tokens < 1:
            raise ValueError("num_tokens must be a positive integer")
        start, end = self._length, self._length + num_tokens
        if end > self.max_length:
            raise ValueError("Cache capacity/context limit exceeded; no tokens were evicted")
        shape = (self.batch_size, num_tokens)
        if attention_mask is None:
            valid = torch.ones(shape, device=self.device, dtype=torch.bool)
        else:
            if (
                not isinstance(attention_mask, Tensor)
                or attention_mask.shape != shape
                or attention_mask.device != self.device
                or attention_mask.layout != torch.strided
                or attention_mask.dtype
                not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            ):
                raise ValueError(
                    "attention_mask must be bool/integer [B,new_tokens] on cache device"
                )
            if torch.any((attention_mask != 0) & (attention_mask != 1)):
                raise ValueError("attention_mask values must be 0 or 1")
            valid = attention_mask.bool()
        if position_ids is None:
            positions = torch.arange(start, end, device=self.device).expand(self.batch_size, -1)
        else:
            if (
                not isinstance(position_ids, Tensor)
                or position_ids.shape not in ((num_tokens,), shape)
                or position_ids.device != self.device
                or position_ids.layout != torch.strided
                or position_ids.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "position_ids must be integer [new_tokens] or [B,new_tokens] on cache device"
                )
            if torch.any(position_ids < 0) or torch.any(position_ids >= self.config.context_length):
                raise ValueError("RoPE position_ids outside model context_length")
            positions = position_ids.expand(shape)
        with torch.no_grad():
            self._valid[:, start:end].copy_(valid)
            self._positions[:, start:end].copy_(positions)
        step = KVCacheStep(self, start, end)
        self._active = step
        return step

    def reset(self) -> None:
        if self._active is not None:
            raise RuntimeError("Abort/commit the active append before reset")
        self._length = 0

    def memory_bytes(self) -> dict[str, int]:
        """Tensor payload, not CUDA allocator reserved/peak memory or measured VRAM."""
        kv_capacity = sum(t.numel() * t.element_size() for pair in self._layers for t in pair)
        metadata_capacity = sum(
            t.numel() * t.element_size() for t in (self._valid, self._positions)
        )
        return {
            "kv_used_bytes": kv_capacity * self.sequence_length // self.max_length,
            "kv_capacity_bytes": kv_capacity,
            "metadata_used_bytes": metadata_capacity * self.sequence_length // self.max_length,
            "metadata_capacity_bytes": metadata_capacity,
            "tensor_capacity_bytes": kv_capacity + metadata_capacity,
        }


class KVCacheStep:
    """Append transaction; update() exposes current-layer prefix for attention.

    Use a with block around the entire model forward. Exceptions or incomplete layer
    sets abort; existing committed data stays valid. Uncommitted bytes are not scrubbed.
    """

    def __init__(self, cache: KVCache, start: int, end: int):
        self._cache, self._start, self._end = cache, start, end
        self._written: set[int] = set()
        self._closed = False

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end

    def _check_open(self) -> None:
        if self._closed or self._cache._active is not self:
            raise RuntimeError("Cache append is no longer active")

    @property
    def attention_mask(self) -> Tensor:
        self._check_open()
        return self._cache._valid[:, : self.end]

    @property
    def query_mask(self) -> Tensor:
        self._check_open()
        return self._cache._valid[:, self.start : self.end]

    @property
    def position_ids(self) -> Tensor:
        self._check_open()
        return self._cache._positions[:, self.start : self.end]

    def update(self, layer_index: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        self._check_open()
        cache = self._cache
        cache._layer_index(layer_index)
        if layer_index in self._written:
            raise RuntimeError("Layer already written in this append")
        shape = (
            cache.batch_size,
            cache.config.num_heads,
            self.end - self.start,
            cache.config.head_dim,
        )
        for tensor in (key, value):
            if (
                not isinstance(tensor, Tensor)
                or tensor.shape != shape
                or tensor.dtype != cache.dtype
                or tensor.device != cache.device
                or tensor.layout != torch.strided
            ):
                raise ValueError("K/V must match [B,heads,new_tokens,head_dim], cache dtype/device")
            if tensor.requires_grad:
                raise ValueError("K/V must not require gradients; use no_grad/inference_mode")
        stored_key, stored_value = cache._layers[layer_index]
        with torch.no_grad():
            stored_key[:, :, self.start : self.end].copy_(key)
            stored_value[:, :, self.start : self.end].copy_(value)
        self._written.add(layer_index)
        return stored_key[:, :, : self.end], stored_value[:, :, : self.end]

    def commit(self) -> None:
        self._check_open()
        if len(self._written) != self._cache.config.num_layers:
            raise RuntimeError("Cannot commit an append with incomplete layers")
        self._cache._length = self.end
        self._cache._active = None
        self._closed = True

    def abort(self) -> None:
        self._check_open()
        self._cache._active = None
        self._closed = True

    def __enter__(self) -> KVCacheStep:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._closed:
            return
        if exc_type is not None:
            self.abort()
            return
        try:
            self.commit()
        except BaseException:
            self.abort()
            raise
