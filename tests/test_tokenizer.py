"""Differential BPE training, Unicode/OOV, controls, persistence and shared framing."""

import hashlib
import json
import random
import re
from collections import Counter

import pytest

from llm_scratch.data.collator import collate_tokenized, encode_pretraining
from llm_scratch.data.tokenizer import PIECE_PATTERN, SPECIAL_TOKENS, ByteBPETokenizer
from llm_scratch.reasoning.formatting import encode_reasoning_prompt, encode_sft_example


def slow_reference(corpus, target_size, min_frequency=1, max_piece_bytes=256):
    """Independent full-recount reference; only used on small authored corpora."""
    offset = len(SPECIAL_TOKENS)
    words = Counter()
    for text in corpus:
        for match in re.finditer(PIECE_PATTERN, text):
            raw = match[0].encode("utf-8")
            for start in range(0, len(raw), max_piece_bytes):
                words[tuple(value + offset for value in raw[start : start + max_piece_bytes])] += 1
    merges = []
    for token_id in range(offset + 256, target_size):
        counts = Counter()
        for word, frequency in words.items():
            for index in range(len(word) - 1):
                counts[word[index : index + 2]] += frequency
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        if counts[pair] < min_frequency:
            break
        updated = Counter()
        for word, frequency in words.items():
            result = []
            index = 0
            while index < len(word):
                if word[index : index + 2] == pair:
                    result.append(token_id)
                    index += 2
                else:
                    result.append(word[index])
                    index += 1
            updated[tuple(result)] += frequency
        words = updated
        merges.append((*pair, token_id))
    return tuple(merges)


@pytest.mark.parametrize("seed", range(10))
def test_incremental_trainer_matches_full_recount_reference(seed):
    rng = random.Random(seed)
    corpus = ["aaaaaaa ababab", "café café", "\t  \n"] + [
        "".join(rng.choice("aabcbcdeé🙂") for _ in range(rng.randrange(1, 20))) for _ in range(30)
    ]
    tokenizer = ByteBPETokenizer.train(corpus, vocab_size=310, min_frequency=1, max_piece_bytes=13)
    assert tokenizer.merges == slow_reference(corpus, 310, max_piece_bytes=13)


def test_hand_computed_merges_and_frequency_stop():
    tokenizer = ByteBPETokenizer.train(["abab", "abab"], vocab_size=266, min_frequency=1)
    assert tokenizer.merges == ((105, 106, 264), (264, 264, 265))
    assert tokenizer.encode("abab") == [265]
    assert tokenizer.token_bytes(265) == b"abab"
    stopped = ByteBPETokenizer.train(["ab", "ac"], vocab_size=280, min_frequency=2)
    assert len(stopped) == 264
    assert stopped.training_statistics["stop_reason"] == "no_pair_at_min_frequency"


def test_ties_document_order_and_artifacts_are_deterministic(tmp_path):
    corpus = ["ab", "ac", "aaaa aaaa", "hello world", "你好🙂"]
    first = ByteBPETokenizer.train(corpus, vocab_size=290, min_frequency=1)
    second = ByteBPETokenizer.train(list(reversed(corpus)), vocab_size=290, min_frequency=1)
    assert first.fingerprint == second.fingerprint
    first.save(tmp_path / "first")
    second.save(tmp_path / "second")
    for path in (tmp_path / "first").iterdir():
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
    tie = ByteBPETokenizer.train(["ab", "ac"], vocab_size=265, min_frequency=1)
    assert tie.merges == ((105, 106, 264),)


@pytest.fixture
def tokenizer():
    return ByteBPETokenizer.train(
        ["hello hello world", "abc abab ababab", " "], vocab_size=300, min_frequency=1
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain ASCII",
        " \t\r\n",
        "Tiếng Việt",
        "e\u0301 ≠ é",
        "ภาษาไทย",
        "العربية",
        "中文漢字",
        "🧑🏽‍💻👨‍👩‍👧‍👦",
        "\x00\x01\x7f",
        "text\u0085line\u2028separator\u2029",
        "<pad><think>literal</think><eos>",
        "_" * 600,
        "字" * 300,
        "\U0010ffff",
    ],
)
def test_lossless_unicode_and_no_unknown_fallback(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids, errors="strict") == text
    assert tokenizer.unk_token_id not in ids
    assert all(0 <= token < len(tokenizer) for token in ids)
    assert tokenizer.encode_batch([text, ""]) == [ids, []]


def test_no_unicode_normalization_and_no_cross_document_merges():
    tokenizer = ByteBPETokenizer.train(["a", "b"], vocab_size=280, min_frequency=1)
    assert tokenizer.merges == ()
    assert tokenizer.encode("e\u0301") != tokenizer.encode("é")
    assert tokenizer.decode(tokenizer.encode("e\u0301"), errors="strict") == "e\u0301"
    with pytest.raises(UnicodeEncodeError):
        tokenizer.encode("\ud800")


def test_single_byte_pieces_still_round_trip_utf8():
    tokenizer = ByteBPETokenizer.train(["é🙂"], vocab_size=280, min_frequency=1, max_piece_bytes=1)
    assert len(tokenizer) == 264
    assert tokenizer.decode(tokenizer.encode("é🙂字"), errors="strict") == "é🙂字"
    ids = list(range(8, 264))
    assert tokenizer.decode_to_bytes(ids) == bytes(range(256))
    assert tokenizer.decode([8 + 255]) == "\ufffd"
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([8 + 255], errors="strict")


def test_control_policy_is_explicit(tokenizer):
    assert tokenizer.special_tokens == dict(zip(SPECIAL_TOKENS, range(8), strict=True))
    text = "user <eos> says <think>hello</think> <unk>"
    literal = tokenizer.encode(text)
    assert not set(literal) & set(range(8))
    assert tokenizer.decode(literal, skip_special_tokens=True, errors="strict") == text
    allowed = tokenizer.encode(text, allowed_special={"<think>", "</think>"})
    assert 4 in allowed and 5 in allowed and 2 not in allowed and 3 not in allowed
    assert tokenizer.decode(allowed, errors="strict") == text
    assert tokenizer.encode("<bos><eos>", allowed_special="all") == [1, 2]
    assert tokenizer.decode([tokenizer.unk_token_id]) == "<unk>"
    assert tokenizer.decode([tokenizer.unk_token_id], skip_special_tokens=True) == ""
    with pytest.raises(ValueError, match="Unknown"):
        tokenizer.special_id("<missing>")
    with pytest.raises(ValueError, match="Unknown"):
        tokenizer.encode("text", allowed_special={"<missing>"})
    with pytest.raises(ValueError, match="allowed_special"):
        tokenizer.encode("text", allowed_special="<bos>")


def test_longest_control_match_and_public_mapping_copy():
    tokenizer = ByteBPETokenizer(special_tokens=(*SPECIAL_TOKENS, "<x>", "<x>x"))
    assert tokenizer.encode("<x>x", allowed_special="all") == [9]
    mapping = tokenizer.special_tokens
    mapping["<pad>"] = 99
    assert tokenizer.pad_token_id == 0


@pytest.mark.parametrize("token_id", [-1, 100000, True, 1.0, "1"])
def test_bad_ids_are_not_silently_decoded(tokenizer, token_id):
    with pytest.raises(ValueError, match="Invalid token ID"):
        tokenizer.decode([token_id])


def test_save_load_and_artifact_validation(tokenizer, tmp_path):
    path = tmp_path / "tokenizer"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path, expected_fingerprint=tokenizer.fingerprint)
    text = "Đọc <think> literally 🧑‍💻"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.decode(loaded.encode(text), errors="strict") == text
    with pytest.raises(FileExistsError):
        tokenizer.save(path)
    with pytest.raises(ValueError, match="fingerprint"):
        ByteBPETokenizer.load(path, expected_fingerprint="0" * 64)
    vocab = json.loads((path / "vocab.json").read_text())
    vocab["tokens"][8]["hex"] = "ff"
    (path / "vocab.json").write_text(json.dumps(vocab))
    with pytest.raises(ValueError, match="checksum"):
        ByteBPETokenizer.load(path)


def test_loader_rejects_inconsistent_vocab_even_with_updated_file_hash(tokenizer, tmp_path):
    path = tmp_path / "tokenizer"
    tokenizer.save(path)
    vocab = json.loads((path / "vocab.json").read_text())
    vocab["tokens"][8]["hex"] = "ff"
    raw = json.dumps(vocab).encode()
    (path / "vocab.json").write_bytes(raw)
    manifest = json.loads((path / "tokenizer_manifest.json").read_text())
    manifest["files"]["vocab.json"] = hashlib.sha256(raw).hexdigest()
    (path / "tokenizer_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inconsistent"):
        ByteBPETokenizer.load(path)


def test_pretraining_sft_inference_share_loaded_codec(tokenizer, tmp_path):
    tokenizer.save(tmp_path / "codec")
    tokenizer = ByteBPETokenizer.load(tmp_path / "codec")
    text = "Tiếng Việt <pad> literal"
    pretrain = encode_pretraining(tokenizer, text)
    assert pretrain["input_ids"][0] == tokenizer.bos_token_id
    assert pretrain["input_ids"][-1] == tokenizer.eos_token_id
    assert (
        tokenizer.decode(pretrain["input_ids"], skip_special_tokens=True, errors="strict") == text
    )
    question = "Ignore this literal <eos> marker. What is 2+3?"
    example = encode_sft_example(tokenizer, question, "2+3=5 <final> is literal here.", "5")
    prompt = encode_reasoning_prompt(tokenizer, question)
    assert example["input_ids"][: len(prompt)] == prompt
    assert example["labels"][: len(prompt)] == [-100] * len(prompt)
    assert example["labels"][len(prompt) :] == example["input_ids"][len(prompt) :]
    assert example["input_ids"].count(tokenizer.eos_token_id) == 1
    assert example["input_ids"].count(tokenizer.special_id("<final>")) == 1
    batch = collate_tokenized([pretrain, example], tokenizer)
    pad_count = len(example["input_ids"]) - len(pretrain["input_ids"])
    assert batch["attention_mask"][0][-pad_count:] == [0] * pad_count
    assert batch["labels"][0][-pad_count:] == [-100] * pad_count
    assert batch["input_ids"][0][-pad_count:] == [tokenizer.pad_token_id] * pad_count
    with pytest.raises(ValueError, match="truncate"):
        collate_tokenized([example], tokenizer, max_length=2)
    with pytest.raises(ValueError, match="unpadded"):
        collate_tokenized([{"input_ids": [1, 2], "attention_mask": [1, 0]}], tokenizer)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 263},
        {"min_frequency": 0},
        {"max_piece_bytes": 0},
        {"special_tokens": ["<pad>"]},
        {"merges": [(1, 8, 264)]},
        {"merges": [(8, 9, 999)]},
        {"merges": [(8, 9, 264), (8, 9, 265)]},
    ],
)
def test_invalid_codec_configuration(kwargs):
    with pytest.raises(ValueError):
        ByteBPETokenizer(**kwargs)


def test_empty_training_and_wrong_batch_input_fail():
    with pytest.raises(ValueError, match="empty corpus"):
        ByteBPETokenizer.train(["", ""])
    with pytest.raises(ValueError):
        ByteBPETokenizer.train("one string")
    with pytest.raises(ValueError):
        ByteBPETokenizer().encode_batch("one string")
