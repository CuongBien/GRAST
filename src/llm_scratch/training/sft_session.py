"""SFT consumes independent examples using the same transactional cursor/engine contract."""

from llm_scratch.data.datasets import normalize_text
from llm_scratch.training.session import PretrainingSession


class SFTSession(PretrainingSession):
    def __init__(self, engine, train, validation):
        if (
            engine.config.stage != "sft"
            or validation.split != "validation"
            or train.corpus.meta["fingerprint"] != validation.meta["fingerprint"]
            or engine.model.config.vocab_size != train.corpus.meta["vocab_size"]
            or engine.model.tokenizer_fingerprint != train.corpus.meta["tokenizer_fingerprint"]
            or engine.model.config.context_length < train.corpus.meta["block_size"]
            or engine.config.gradient_accumulation_steps != train.accumulation_steps
            or engine.attempts != train.windows_consumed
        ):
            raise ValueError("SFT model/tokenizer/context/data/cursor contract mismatch")
        train_ids = {r["question_id"] for r in train.corpus.examples}
        train_questions = {normalize_text(r["prompt"]) for r in train.corpus.examples}
        if any(
            r["question_id"] in train_ids or normalize_text(r["prompt"]) in train_questions
            for r in validation.examples
        ):
            raise ValueError("SFT train/validation overlap")
        self.engine, self.train, self.validation = engine, train, validation
        self._failed = False

    @property
    def contract(self):
        return super().contract | {"stage": "sft"}
