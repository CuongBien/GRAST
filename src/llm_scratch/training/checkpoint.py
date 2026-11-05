"""Atomic checkpoints for tensors/primitive state and all managed RNGs; no trainer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_scratch.utils.seed import capture_rng_state, restore_rng_state

if TYPE_CHECKING:
    import numpy as np


def save_checkpoint(
    path: str | Path, state: dict[str, Any], *, generator: np.random.Generator
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(
                {"format_version": 1, "state": state, "rng": capture_rng_state(generator)}, stream
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: str | Path, *, generator: np.random.Generator) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version")
    restore_rng_state(payload["rng"], generator)
    return payload["state"]
