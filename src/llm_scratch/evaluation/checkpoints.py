"""Read model/history/provenance without restoring optimizer, RNG or training cursors."""

from __future__ import annotations

from pathlib import Path

import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.utils.config import config_digest


def load_for_evaluation(path: Path, codec: ByteBPETokenizer, *, expected_sha256=None):
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Evaluation checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state", {})
    if (
        payload.get("format_version") != 1
        or state.get("training_schema_version") != 1
        or state.get("training_config", {}).get("stage") not in ("pretrain", "sft")
        or type(state.get("step")) is not int
        or state["step"] < 1
        or type(state.get("attempts")) is not int
        or state["attempts"] < state["step"]
    ):
        raise ValueError("Expected a trained pretrain/SFT engine checkpoint")
    weights = state["model"]
    extra = weights.get("_extra_state", {})
    if extra.get("schema_version") != 1 or extra.get("tokenizer_fingerprint") != codec.fingerprint:
        raise ValueError("Checkpoint tokenizer metadata mismatch")
    architecture = ModelConfig.from_dict(extra["model_config"])
    if architecture.vocab_size != len(codec):
        raise ValueError("Checkpoint vocabulary mismatch")
    # Model construction must not advance a caller's RNG or inherit a default CUDA/meta device.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        model = CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint)
    for name, value in weights.items():
        if name != "_extra_state" and (
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Nonfinite/invalid checkpoint tensor: {name}")
    model.load_state_dict(weights)
    model.eval()
    runner = state.get("data_state", {}).get("extra_state", {})
    history = runner.get("history")
    if (
        not isinstance(history, dict)
        or set(history) != {"train", "validation"}
        or not history["train"]
        or not history["validation"]
        or history["train"][-1]["step"] != state["step"]
        or history["train"][-1]["attempt"] != state["attempts"]
        or len(history["train"]) != state["attempts"]
        or runner.get("history_sha256") != config_digest(history)
    ):
        raise ValueError("Evaluation requires checkpoint-owned training/validation history")
    meta = {
        "checkpoint_sha256": digest,
        "stage": state["training_config"]["stage"],
        "step": state["step"],
        "attempts": state["attempts"],
        "tokens_optimized": state["tokens_optimized"],
        "source_runtime": state["runtime"],
        "model_config": architecture.to_dict(),
        "tokenizer_fingerprint": codec.fingerprint,
        "parameter_counts": model.parameter_counts(),
        "initialization": runner.get("contract", {}).get("initialization"),
    }
    # Mmap-backed training state is deliberately discarded; only model/history travel onward.
    return model, meta, history
