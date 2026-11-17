"""WP07.1 single-device training engine; corpus order/packing lives in WP07.2."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from llm_scratch.model.transformer import CausalLM
from llm_scratch.training.checkpoint import save_checkpoint
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.optim import WarmupCosine, build_optimizer
from llm_scratch.utils.seed import restore_rng_state


class TrainingEngine:
    def __init__(
        self,
        model: CausalLM,
        config: TrainingConfig,
        *,
        generator: np.random.Generator,
        log: Callable[[dict], None] | None = None,
    ):
        self.model, self.config, self.generator, self.log = model, config, generator, log
        self.device = model.token_embedding.weight.device
        if self.device.type not in ("cpu", "cuda") or any(
            p.dtype != torch.float32 for p in model.parameters()
        ):
            raise ValueError("Engine needs FP32 master weights on CPU/CUDA; use autocast for AMP")
        if config.precision == "float16" and self.device.type != "cuda":
            raise ValueError("FP16 training requires CUDA")
        if (
            config.precision == "bfloat16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("Selected CUDA device does not support BF16")
        self.optimizer = build_optimizer(model, config)
        self.scheduler = WarmupCosine(self.optimizer, config)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=config.precision == "float16")
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.step = self.attempts = self.tokens_seen = self.tokens_optimized = 0
        self._busy = self._failed = False
        self._emit("trainer_initialized", config=config.to_dict(), device=str(self.device))

    def _emit(self, event, **fields):
        record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        if self.log is not None:
            self.log(record)
        return record

    def _autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=getattr(torch, self.config.precision),
            enabled=self.config.precision != "float32",
        )

    def _batch(self, batch):
        if (
            not isinstance(batch, dict)
            or not {"input_ids", "labels"} <= set(batch)
            or set(batch) - {"input_ids", "labels", "attention_mask"}
        ):
            raise ValueError("Expected input_ids, labels and optional attention_mask")
        ids, labels = batch["input_ids"], batch["labels"]
        if (
            not isinstance(ids, torch.Tensor)
            or ids.ndim != 2
            or 0 in ids.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or not isinstance(labels, torch.Tensor)
            or labels.shape != ids.shape
            or labels.dtype not in (torch.int32, torch.int64)
            or ids.shape[1] > self.model.config.context_length
            or any(
                not isinstance(v, torch.Tensor) or v.layout != torch.strided for v in batch.values()
            )
        ):
            raise ValueError("Training IDs/labels must be matching nonempty integer [B,T]")
        result = {k: v.to(self.device) for k, v in batch.items()}
        ids, labels = result["input_ids"], result["labels"]
        if torch.any((ids < 0) | (ids >= self.model.config.vocab_size)) or torch.any(
            (labels != -100) & ((labels < 0) | (labels >= self.model.config.vocab_size))
        ):
            raise ValueError("Training IDs/labels outside vocabulary")
        mask = result.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
        if (
            mask.shape != ids.shape
            or mask.dtype
            not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or torch.any((mask != 0) & (mask != 1))
        ):
            raise ValueError("Invalid attention_mask")
        result["attention_mask"] = mask.bool()
        count = int(
            ((labels[:, 1:] != -100) & mask[:, 1:].bool() & mask[:, :-1].bool()).sum().item()
        )
        return result, count

    def train_update(self, microbatches: Sequence[dict]) -> dict:
        if self._busy or self._failed or self.step >= self.config.max_steps:
            raise RuntimeError("Trainer busy, failed (reload checkpoint), or step budget exhausted")
        if not 1 <= len(microbatches) <= self.config.gradient_accumulation_steps:
            raise ValueError("Need 1..gradient_accumulation_steps microbatches per update")
        if {id(p) for p in self.model.parameters() if p.requires_grad} != {
            id(p) for p in self.parameters
        }:
            raise RuntimeError("Trainable parameters changed after optimizer creation")
        started = time.perf_counter()
        batches = [self._batch(batch) for batch in microbatches]
        total = sum(n for _, n in batches)
        if total == 0:
            raise ValueError("Accumulation window has no supervised next-token pairs")
        self._emit(
            "update_started",
            step=self.step + 1,
            attempt=self.attempts + 1,
            microbatches=len(batches),
            loss_tokens=total,
        )
        self._busy = True
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        used_lrs = [g["lr"] for g in self.optimizer.param_groups]
        numerator = 0.0
        try:
            for batch, count in batches:
                if count == 0:
                    continue  # Empty supervision contributes neither gradients nor dropout RNG.
                with self._autocast():
                    output = self.model(**batch)
                if output.num_loss_tokens != count or not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite loss or inconsistent supervision count")
                numerator += output.loss.detach().item() * count
                self.scaler.scale(output.loss * (count / total)).backward()
            if any(p.grad is None for p in self.parameters):
                raise RuntimeError("A trainable parameter is disconnected from loss")
            self.scaler.unscale_(self.optimizer)
            finite = bool(
                torch.stack([torch.isfinite(p.grad).all() for p in self.parameters]).all()
            )
            if not finite and not self.scaler.is_enabled():
                raise FloatingPointError("Non-finite gradients")
            norm = None
            if finite:
                norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters,
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                    foreach=False,
                ).item()
            self.scaler.step(self.optimizer)  # FP16 overflow skips this effective batch atomically.
            self.scaler.update()
            self.attempts += 1
            self.tokens_seen += total
            if finite:
                self.step += 1
                self.tokens_optimized += total
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except BaseException:
            self._failed = True
            self.optimizer.zero_grad(set_to_none=True)
            self._emit("training_failed", step=self.step, attempt=self.attempts + 1)
            raise
        finally:
            self._busy = False
        elapsed = time.perf_counter() - started
        return self._emit(
            "train_update",
            step=self.step,
            attempt=self.attempts,
            loss=numerator / total,
            loss_tokens=total,
            tokens_seen=self.tokens_seen,
            tokens_optimized=self.tokens_optimized,
            learning_rates=used_lrs,
            gradient_norm_before_clip=norm,
            skipped=not finite,
            amp_scale=self.scaler.get_scale(),
            elapsed_seconds=elapsed,
            supervised_tokens_per_second=total / elapsed,
        )

    @torch.no_grad()
    def evaluate(self, batches: Sequence[dict]) -> dict:
        if self._busy or self._failed:
            raise RuntimeError("Trainer busy/failed")
        was_training = self.model.training
        numerator, total = 0.0, 0
        self.model.eval()
        try:
            for original in batches:
                batch, count = self._batch(original)
                if count == 0:
                    continue
                with self._autocast():
                    output = self.model(**batch)
                if not torch.isfinite(output.loss) or output.num_loss_tokens != count:
                    raise FloatingPointError("Invalid validation loss")
                numerator += output.loss.item() * count
                total += count
            if total == 0:
                raise ValueError("Validation has no supervised tokens")
        finally:
            self.model.train(was_training)
        return self._emit("validation", step=self.step, loss=numerator / total, loss_tokens=total)

    def _runtime(self):
        return {
            "device": str(self.device),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "num_threads": torch.get_num_threads(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else None,
            "capability": list(torch.cuda.get_device_capability(self.device))
            if self.device.type == "cuda"
            else None,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }

    def save(self, path: str | Path, *, data_state: dict):
        if not isinstance(data_state, dict):
            raise ValueError("data_state must be an explicit dictionary")
        if self._busy or self._failed or any(p.grad is not None for p in self.parameters):
            raise RuntimeError("Checkpoint only at clean optimizer-update boundaries")
        state = {
            "training_schema_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "training_config": self.config.to_dict(),
            "runtime": self._runtime(),
            "step": self.step,
            "attempts": self.attempts,
            "tokens_seen": self.tokens_seen,
            "tokens_optimized": self.tokens_optimized,
            "data_state": data_state,
        }
        save_checkpoint(path, state, generator=self.generator)
        self._emit("checkpoint_saved", step=self.step, path=str(path))

    def load(
        self, path: str | Path, *, data_validator: Callable[[dict, int], None] | None = None
    ) -> dict:
        if self._busy:
            raise RuntimeError("Cannot load during update")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("state", {})
        if (
            payload.get("format_version") != 1
            or state.get("training_schema_version") != 1
            or state.get("training_config") != self.config.to_dict()
            or state.get("runtime") != self._runtime()
        ):
            raise ValueError(
                "Incompatible training config/runtime; exact resume requires same recipe/runtime"
            )
        if state["step"] != state["scheduler"]["completed_updates"]:
            raise ValueError("Checkpoint scheduler/update counters disagree")
        if (
            any(
                type(state.get(n)) is not int or state[n] < 0
                for n in ("step", "attempts", "tokens_seen", "tokens_optimized")
            )
            or not state["step"] <= min(state["attempts"], self.config.max_steps)
            or state["tokens_optimized"] > state["tokens_seen"]
            or not isinstance(state.get("data_state"), dict)
        ):
            raise ValueError("Invalid checkpoint progress/data state")
        if data_validator is not None:
            data_validator(state["data_state"], state["attempts"])
        try:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler = torch.amp.GradScaler(
                self.device.type, enabled=self.config.precision == "float16"
            )
            self.scaler.load_state_dict(state["scaler"])
            for name in ("step", "attempts", "tokens_seen", "tokens_optimized"):
                setattr(self, name, state[name])
            self.optimizer.zero_grad(set_to_none=True)
            restore_rng_state(payload["rng"], self.generator)
        except BaseException:
            self._failed = True
            raise
        self._failed = False
        self._emit("checkpoint_loaded", step=self.step, path=str(path))
        return state["data_state"]
