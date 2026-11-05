"""One token-ID framing for SFT and inference, with user text always encoded literally."""

from llm_scratch.data.tokenizer import ByteBPETokenizer


def encode_reasoning_prompt(tokenizer: ByteBPETokenizer, question: str) -> list[int]:
    """Inference prefix is exactly the prefix used by encode_sft_example."""
    return [
        tokenizer.bos_token_id,
        *tokenizer.encode(question + "\n"),
        tokenizer.special_id("<think>"),
    ]


def encode_sft_example(
    tokenizer: ByteBPETokenizer,
    question: str,
    reasoning: str,
    final_answer: str,
) -> dict[str, list[int]]:
    prefix = encode_reasoning_prompt(tokenizer, question)
    target = [
        *tokenizer.encode(reasoning),
        tokenizer.special_id("</think>"),
        tokenizer.special_id("<final>"),
        *tokenizer.encode(final_answer),
        tokenizer.special_id("</final>"),
        tokenizer.eos_token_id,
    ]
    ids = prefix + target
    # Labels are unshifted; the future causal LM loss performs its usual one-token shift.
    return {
        "input_ids": ids,
        "labels": [-100] * len(prefix) + target,
        "attention_mask": [1] * len(ids),
    }
