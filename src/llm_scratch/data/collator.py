"""Tokenizer-facing pretraining examples and right padding; no tensors/training loop."""

from llm_scratch.data.tokenizer import ByteBPETokenizer


def encode_pretraining(tokenizer: ByteBPETokenizer, text: str) -> dict[str, list[int]]:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return {"input_ids": ids, "labels": ids.copy(), "attention_mask": [1] * len(ids)}


def collate_tokenized(
    examples: list[dict],
    tokenizer: ByteBPETokenizer,
    *,
    max_length: int | None = None,
) -> dict[str, list[list[int]]]:
    """Right-pad IDs, zero attention and -100 labels. Never silently truncate."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    lengths = [len(example["input_ids"]) for example in examples]
    if not all(lengths):
        raise ValueError("Tokenized examples must be nonempty")
    width = max(lengths) if max_length is None else max_length
    if type(width) is not int or width < max(lengths):
        raise ValueError("max_length would truncate an example")
    result = {"input_ids": [], "labels": [], "attention_mask": []}
    for example in examples:
        ids = list(example["input_ids"])
        labels = list(example.get("labels", ids))
        if list(example.get("attention_mask", [1] * len(ids))) != [1] * len(ids):
            raise ValueError("Collator expects unpadded examples")
        if len(ids) != len(labels):
            raise ValueError("labels and input_ids must have equal lengths")
        for token in ids:
            tokenizer.token_bytes(token)
        if any(
            type(label) is not int or (label != -100 and not 0 <= label < len(tokenizer))
            for label in labels
        ):
            raise ValueError("Invalid language-model label")
        padding = width - len(ids)
        result["input_ids"].append(ids + [tokenizer.pad_token_id] * padding)
        result["labels"].append(labels + [-100] * padding)
        result["attention_mask"].append([1] * len(ids) + [0] * padding)
    return result
