"""Bind engine and consumed train cursor into one full-resume checkpoint contract."""

from __future__ import annotations

from itertools import islice

from llm_scratch.data.pretraining import PackedCorpus
from llm_scratch.training.data import TrainingCursor, window_fingerprint
from llm_scratch.training.trainer import TrainingEngine


class PretrainingSession:
    def __init__(self, engine: TrainingEngine, train: TrainingCursor, validation: PackedCorpus):
        if (
            engine.config.stage != "pretrain"
            or validation.split != "validation"
            or train.corpus.meta["fingerprint"] != validation.meta["fingerprint"]
            or engine.model.config.vocab_size != train.corpus.meta["vocab_size"]
            or engine.model.tokenizer_fingerprint != train.corpus.meta["tokenizer_fingerprint"]
            or engine.model.config.context_length < train.corpus.meta["block_size"]
            or engine.config.gradient_accumulation_steps != train.accumulation_steps
        ):
            raise ValueError("Model/tokenizer/train/validation/accumulation contract mismatch")
        if engine.attempts != train.windows_consumed:
            raise ValueError("Engine and consumed data cursor disagree")
        self.engine, self.train, self.validation = engine, train, validation
        self._failed = False

    @property
    def contract(self):
        return {
            "schema_version": 1,
            "stage": "pretrain",
            "train": self.train.contract,
            "validation": self.validation.identity,
            "validation_order": "fixed-sequential-v1",
        }

    def update(self):
        if self._failed or self.engine.attempts != self.train.windows_consumed:
            raise RuntimeError("Session failed/desynchronized; load a full session checkpoint")
        window = self.train.peek()
        digest = window_fingerprint(window)
        epoch = self.train.epoch
        try:
            result = self.engine.train_update(window.batches)
            self.train.commit(window)  # Also commit on AMP skip: data/RNG were consumed.
        except BaseException:
            if self.train._pending is window:
                self.train.abort(window)
            self._failed = True
            raise
        self.engine._emit(
            "data_window_committed",
            step=self.engine.step,
            attempt=self.engine.attempts,
            epoch=epoch,
            block_indices=window.indices,
            batch_sha256=digest,
            next_epoch=self.train.epoch,
            next_offset=self.train.offset,
        )
        return result | {
            "data_epoch": epoch,
            "block_indices": window.indices,
            "batch_sha256": digest,
        }

    def validate(self, *, max_batches: int | None = None):
        if self._failed:
            raise RuntimeError("Session failed")
        if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
            raise ValueError("max_batches must be positive or None for full validation")
        before = self.train.state_dict()
        batches = self.validation.validation_batches(self.train.batch_size)
        result = self.engine.evaluate(
            batches if max_batches is None else islice(batches, max_batches)
        )
        if self.train.state_dict() != before:
            raise AssertionError("Validation changed the consumed train cursor")
        self.engine._emit(
            "validation_data_scope",
            step=self.engine.step,
            dataset=self.validation.identity,
            max_batches=max_batches,
            batch_size=self.train.batch_size,
            order="fixed-sequential-v1",
        )
        return result

    def save(self, path):
        if self._failed or self.engine.attempts != self.train.windows_consumed:
            raise RuntimeError("Cannot save failed/desynchronized session")
        self.engine.save(
            path,
            data_state={"session_contract": self.contract, "train_cursor": self.train.state_dict()},
        )

    def _validate_data(self, state, attempts):
        if (
            not isinstance(state, dict)
            or set(state) != {"session_contract", "train_cursor"}
            or state["session_contract"] != self.contract
        ):
            raise ValueError("Checkpoint dataset/tokenizer/packing/validation identity mismatch")
        self.train.validate_state(state["train_cursor"])
        if state["train_cursor"]["windows_consumed"] != attempts:
            raise ValueError("Checkpoint engine/data cursor counters disagree")

    def load(self, path):
        if self.train._pending is not None:
            raise RuntimeError("Cannot load during pending data window")
        state = self.engine.load(path, data_validator=self._validate_data)
        self.train.load_state_dict(state["train_cursor"])
        self._failed = False
        self.engine._emit(
            "full_data_resume_loaded",
            step=self.engine.step,
            epoch=self.train.epoch,
            next_offset=self.train.offset,
            dataset_fingerprint=self.train.corpus.meta["fingerprint"],
        )
