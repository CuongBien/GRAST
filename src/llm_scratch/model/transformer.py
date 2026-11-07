"""From-scratch decoder-only LM: pre-norm blocks, tied/untied head and causal CE."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm_scratch.model.attention import CausalSelfAttention
from llm_scratch.model.cache import KVCache, KVCacheStep
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import RMSNorm, RotaryEmbedding, SwiGLU, TokenEmbedding
from llm_scratch.model.masking import valid_token_mask


class DecoderBlock(nn.Module):
    """Two pre-norm residual sublayers, input/output [batch, sequence, hidden].

    Clear padding before the first norm/residual so nonfinite padding cannot enter
    RMSNorm gradients or bypass attention through the identity path. Mask again
    after each residual. Valid-token residuals are exact identity when branches vanish.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.ffn_hidden_size)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        *,
        cache_step: KVCacheStep | None = None,
        layer_index: int | None = None,
    ) -> Tensor:
        if cache_step is None and layer_index is not None:
            raise ValueError("layer_index requires cache_step")
        valid = valid_token_mask(x, attention_mask, self.config)
        padding = ~valid.unsqueeze(-1)
        x = x.masked_fill(padding, 0.0)
        if cache_step is None:
            attended = self.attention(self.attention_norm(x), valid, position_ids)
        else:
            attended = self.attention(
                self.attention_norm(x),
                valid,
                position_ids,
                cache_step=cache_step,
                layer_index=layer_index,
            )
        x = (x + self.residual_dropout(attended)).masked_fill(padding, 0.0)
        transformed = self.ffn(self.ffn_norm(x))
        return (x + self.residual_dropout(transformed)).masked_fill(padding, 0.0)


@dataclass(frozen=True)
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    num_loss_tokens: int = 0
    cache: KVCache | None = None


def _check_fingerprint(value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
    ):
        raise ValueError("tokenizer_fingerprint must be a lowercase SHA-256 or None")


class CausalLM(nn.Module):
    """Decoder-only LM; labels are unshifted, -100 excludes prompt targets.

    Loss predicts labels[:, 1:] from logits[:, :-1]. Both endpoints must be valid
    according to attention_mask: left padding/interior gaps never become examples.
    No supervised next-token pairs returns a differentiable zero, including T=1.
    """

    def __init__(self, config: ModelConfig, *, tokenizer_fingerprint: str | None = None):
        super().__init__()
        _check_fingerprint(tokenizer_fingerprint)
        self.config = config
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self._cache_revision = 0
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        position_ids: Tensor | None = None,
        *,
        use_cache: bool = False,
        cache: KVCache | None = None,
    ) -> CausalLMOutput:
        if type(use_cache) is not bool:
            raise ValueError("use_cache must be a boolean")
        if cache is not None and (not use_cache or not isinstance(cache, KVCache)):
            raise ValueError("A KVCache argument requires use_cache=True")
        if use_cache:
            if any(module.training for module in self.modules()) or torch.is_grad_enabled():
                raise ValueError("Cached forward requires model.eval() and no_grad/inference_mode")
            if labels is not None:
                raise ValueError("Cached forward does not accept training labels")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be int32/int64 [batch, sequence]")
        batch, length = input_ids.shape
        if batch == 0 or not 1 <= length <= self.config.context_length:
            raise ValueError("Need a nonempty batch/sequence within configured context length")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("input_ids and model must be on the same device")
        if torch.any((input_ids < 0) | (input_ids >= self.config.vocab_size)):
            raise ValueError("input_ids outside vocabulary")
        if labels is not None:
            if (
                labels.shape != input_ids.shape
                or labels.device != input_ids.device
                or labels.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("labels must be int32/int64 [batch, sequence] on input_ids.device")
            if torch.any((labels != -100) & ((labels < 0) | (labels >= self.config.vocab_size))):
                raise ValueError("labels must be -100 or a valid vocabulary ID")
        hidden = self.token_embedding(input_ids)
        valid = valid_token_mask(hidden, attention_mask, self.config)
        if use_cache:
            if cache is None:
                cache = self.new_cache(batch)
            signature = self._cache_signature()
            cache.validate_for_model(self, signature, batch, self._kv_dtype())
            if cache.requires_explicit_positions and position_ids is None:
                raise ValueError("Explicit position_ids required after custom-position cache use")
            with cache.begin_step(length, attention_mask=valid, position_ids=position_ids) as step:
                logits = self._hidden_logits(hidden, valid, step.position_ids, cache_step=step)
                if self._cache_signature() != signature:
                    raise RuntimeError("Model changed during cached forward; append aborted")
            cache.record_model_commit(self, signature, explicit_positions=position_ids is not None)
            return CausalLMOutput(logits=logits, cache=cache)
        logits = self._hidden_logits(hidden, valid, position_ids)
        if labels is None:
            return CausalLMOutput(logits=logits)
        targets = labels[:, 1:].long().masked_fill(~(valid[:, :-1] & valid[:, 1:]), -100)
        count = int((targets != -100).sum().item())
        compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=logits.device.type, enabled=False):
            loss = F.cross_entropy(
                logits[:, :-1].to(compute_dtype).reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / max(count, 1)
        return CausalLMOutput(logits=logits, loss=loss, num_loss_tokens=count)

    def _hidden_logits(
        self,
        hidden: Tensor,
        valid: Tensor,
        position_ids: Tensor | None,
        *,
        cache_step: KVCacheStep | None = None,
    ) -> Tensor:
        padding = ~valid.unsqueeze(-1)
        hidden = hidden.masked_fill(padding, 0.0)
        for index, block in enumerate(self.blocks):
            if cache_step is None:
                hidden = block(hidden, valid, position_ids)
            else:
                hidden = block(
                    hidden, valid, position_ids, cache_step=cache_step, layer_index=index
                )
        hidden = self.final_norm(hidden).masked_fill(padding, 0.0)
        return self.lm_head(hidden).masked_fill(padding, 0.0)

    def _kv_dtype(self) -> torch.dtype:
        weight = self.token_embedding.weight
        if weight.dtype != torch.float64 and torch.is_autocast_enabled(weight.device.type):
            return torch.get_autocast_dtype(weight.device.type)
        return weight.dtype

    def new_cache(self, batch_size: int, *, max_length: int | None = None) -> KVCache:
        return KVCache(
            self.config,
            batch_size,
            max_length=max_length,
            device=self.token_embedding.weight.device,
            dtype=self._kv_dtype(),
        )

    def _cache_signature(self) -> tuple:
        tensors = []
        for name, value in (*self.named_parameters(), *self.named_buffers()):
            if torch.is_inference(value):
                raise ValueError(
                    "Construct model outside inference_mode for versioned cache binding"
                )
            tensors.append(
                (
                    name,
                    id(value),
                    value.data_ptr(),
                    value._version,
                    value.dtype,
                    value.device,
                    tuple(value.shape),
                )
            )
        modules = []
        for name, module in self.named_modules():
            attributes = ()
            if isinstance(module, RotaryEmbedding):
                attributes = (module.theta, module.max_seq_len, module.head_dim)
            elif isinstance(module, RMSNorm):
                attributes = (module.eps,)
            modules.append((name, id(module), type(module), module.training, attributes))
        return (
            self.config,
            self.tokenizer_fingerprint,
            self._cache_revision,
            tuple(tensors),
            tuple(modules),
        )

    def _apply(self, fn, recurse=True):
        # A root dtype/device round trip can reuse an old pointer/version after
        # quantizing values. Invalidate even when the eventual dtype matches again.
        self._cache_revision += 1
        return super()._apply(fn, recurse=recurse)

    def parameter_counts(self) -> dict[str, int]:
        """Count actual unique Parameters; tied head is charged only to embedding."""
        seen, counts = set(), {}
        for name, module in (
            ("token_embedding", self.token_embedding),
            ("blocks", self.blocks),
            ("final_norm", self.final_norm),
            ("lm_head_unique", self.lm_head),
        ):
            count = 0
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    count += parameter.numel()
                    seen.add(id(parameter))
            counts[name] = count
        counts["total"] = sum(counts.values())
        return counts

    def get_extra_state(self) -> dict:
        return {
            "schema_version": 1,
            "model_config": self.config.to_dict(),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }

    def set_extra_state(self, state: dict) -> None:
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "model_config", "tokenizer_fingerprint"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
        ):
            raise ValueError("Invalid model checkpoint metadata")
        if ModelConfig.from_dict(state["model_config"]) != self.config:
            raise ValueError("Checkpoint model configuration mismatch")
        fingerprint = state["tokenizer_fingerprint"]
        _check_fingerprint(fingerprint)
        if self.tokenizer_fingerprint is not None and self.tokenizer_fingerprint != fingerprint:
            raise ValueError("Checkpoint tokenizer fingerprint mismatch")
        self.tokenizer_fingerprint = fingerprint

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._cache_revision += 1
        if self.config.tie_embeddings:
            embedding = state_dict.get("token_embedding.weight")
            head = state_dict.get("lm_head.weight")
            if (
                isinstance(embedding, Tensor)
                and isinstance(head, Tensor)
                and not torch.equal(embedding, head)
            ):
                raise ValueError("Tied embedding/head checkpoint weights disagree")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        return result
