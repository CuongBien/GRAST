"""Consumed-window cursor; no DataLoader workers/prefetch or global RNG draws."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from llm_scratch.data.pretraining import PackedCorpus
from llm_scratch.utils.config import config_digest


@dataclass(frozen=True)
class BatchWindow:
    batches: tuple[dict, ...]
    indices: tuple[tuple[int, ...], ...]


class TrainingCursor:
    def __init__(
        self, corpus: PackedCorpus, *, batch_size: int, accumulation_steps: int, seed: int
    ):
        if corpus.split != "train":
            raise ValueError("Training cursor requires train split")
        for name, value in (("batch_size", batch_size), ("accumulation_steps", accumulation_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Invalid shuffle seed")
        self.corpus, self.batch_size, self.accumulation_steps, self.seed = (
            corpus,
            batch_size,
            accumulation_steps,
            seed,
        )
        self.epoch = self.offset = self.windows_consumed = 0
        self._order = self._permutation(0)
        self._pending = None

    @property
    def contract(self):
        return {
            "schema_version": 1,
            "corpus": self.corpus.identity,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "seed": self.seed,
            "shuffle": "private-cpu-randperm-per-epoch-v1",
            "torch": str(torch.__version__),
        }

    def _permutation(self, epoch):
        digest = config_digest({"dataset": self.corpus.identity, "seed": self.seed, "epoch": epoch})
        rng = torch.Generator(device="cpu").manual_seed(int(digest[:16], 16) % (2**63))
        return torch.randperm(len(self.corpus), generator=rng, device="cpu").tolist()

    def _state(self):
        return {
            "contract": self.contract,
            "epoch": self.epoch,
            "offset": self.offset,
            "windows_consumed": self.windows_consumed,
            "order_sha256": config_digest(self._order),
        }

    def state_dict(self):
        if self._pending is not None:
            raise RuntimeError("Cursor checkpoint requires a consumed-window boundary")
        return self._state()

    def validate_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"contract", "epoch", "offset", "windows_consumed", "order_sha256"}
            or state["contract"] != self.contract
        ):
            raise ValueError("Incompatible dataset/tokenizer/packing/shuffle/batch contract")
        if any(
            type(state[k]) is not int or state[k] < 0
            for k in ("epoch", "offset", "windows_consumed")
        ):
            raise ValueError("Invalid cursor counters")
        width = self.batch_size * self.accumulation_steps
        per_epoch = (len(self.corpus) + width - 1) // width
        if (
            state["offset"] >= len(self.corpus)
            or state["offset"] % width
            or state["windows_consumed"] != state["epoch"] * per_epoch + state["offset"] // width
            or state["order_sha256"] != config_digest(self._permutation(state["epoch"]))
        ):
            raise ValueError("Invalid cursor boundary/order/counters")

    def load_state_dict(self, state):
        if self._pending is not None:
            raise RuntimeError("Cannot load during pending data window")
        self.validate_state(state)
        self.epoch, self.offset, self.windows_consumed = (
            state["epoch"],
            state["offset"],
            state["windows_consumed"],
        )
        self._order = self._permutation(self.epoch)

    def peek(self) -> BatchWindow:
        if self._pending is not None:
            raise RuntimeError("Commit/abort the pending window first")
        end = min(self.offset + self.batch_size * self.accumulation_steps, len(self.corpus))
        indices = tuple(
            tuple(self._order[i : min(i + self.batch_size, end)])
            for i in range(self.offset, end, self.batch_size)
        )
        window = BatchWindow(tuple(self.corpus.batch(list(i)) for i in indices), indices)
        self._pending = window
        return window

    def commit(self, window: BatchWindow):
        if self._pending is not window:
            raise RuntimeError("Stale/foreign data window")
        self.offset += sum(map(len, window.indices))
        self.windows_consumed += 1
        if self.offset == len(self.corpus):
            self.epoch += 1
            self.offset = 0
            self._order = self._permutation(self.epoch)
        self._pending = None

    def abort(self, window: BatchWindow):
        if self._pending is not window:
            raise RuntimeError("Stale/foreign data window")
        self._pending = None


def window_fingerprint(window: BatchWindow) -> str:
    digest = hashlib.sha256()
    for batch in window.batches:
        for key in ("input_ids", "labels", "attention_mask"):
            value = batch[key].contiguous().numpy()
            digest.update(key.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()
