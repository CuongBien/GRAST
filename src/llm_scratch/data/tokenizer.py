"""Pure-Python byte BPE: deterministic training, lossless Unicode and explicit controls."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path

from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<think>", "</think>", "<final>", "</final>")
PIECE_PATTERN = r"\s+|[^\W\d_]+|\d+|_+|[^\w\s]+"
_PIECES = re.compile(PIECE_PATTERN)


def _pieces(text: str, max_piece_bytes: int) -> Iterable[bytes]:
    """Partition without normalization; byte slicing can bisect UTF-8, decoding rejoins it."""
    for match in _PIECES.finditer(text):
        value = match[0].encode("utf-8", errors="strict")
        for start in range(0, len(value), max_piece_bytes):
            yield value[start : start + max_piece_bytes]


def _merge(
    tokens: tuple[int, ...] | list[int], pair: tuple[int, int], new_id: int
) -> tuple[int, ...]:
    result, index = [], 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            result.append(new_id)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return tuple(result)


class ByteBPETokenizer:
    """All 256 bytes are present. Ordinary unseen text never becomes <unk>.

    encode() treats control spellings as literal text unless allowed_special explicitly
    names them. BOS/EOS insertion is opt-in. Decode invalid IDs fails; incomplete UTF-8
    defaults to replacement, with errors='strict' available for validation.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: Iterable[str] = SPECIAL_TOKENS,
        max_piece_bytes: int = 256,
        merges: Iterable[tuple[int, int, int]] = (),
    ):
        if isinstance(special_tokens, (str, bytes)):
            raise ValueError("special_tokens must be a sequence of token strings")
        names = tuple(special_tokens)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Special tokens must be nonempty strings")
        if len(set(names)) != len(names) or not set(SPECIAL_TOKENS) <= set(names):
            raise ValueError("Special tokens must be unique and contain all project controls")
        for name in names:
            name.encode("utf-8", errors="strict")
        base_size = len(names) + 256
        if type(vocab_size) is not int or vocab_size < base_size:
            raise ValueError(f"vocab_size must be at least {base_size}")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if type(max_piece_bytes) is not int or not 1 <= max_piece_bytes <= 4096:
            raise ValueError("max_piece_bytes must be between 1 and 4096")
        self._special_names, self._special_ids = (
            names,
            dict(zip(names, range(len(names)), strict=True)),
        )
        self._target_size, self._min_frequency, self._max_piece_bytes = (
            vocab_size,
            min_frequency,
            max_piece_bytes,
        )
        self._vocab = [name.encode("utf-8") for name in names] + [
            bytes([byte]) for byte in range(256)
        ]
        checked, seen = [], set()
        for raw in merges:
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != 3
                or any(type(i) is not int for i in raw)
            ):
                raise ValueError("Each merge must contain three integer token IDs")
            left, right, new_id = raw
            if new_id != len(self._vocab) or not all(
                len(names) <= i < new_id for i in (left, right)
            ):
                raise ValueError(
                    "Merge IDs must be consecutive and reference previous ordinary tokens"
                )
            if (left, right) in seen or new_id >= vocab_size:
                raise ValueError("Duplicate pair or merges exceeding target vocabulary")
            seen.add((left, right))
            self._vocab.append(self._vocab[left] + self._vocab[right])
            checked.append((left, right, new_id))
        self._merges = tuple(checked)
        self._vocab = tuple(self._vocab)
        self._ranks = {
            (left, right): (rank, token_id)
            for rank, (left, right, token_id) in enumerate(self._merges)
        }
        self._encode_piece = lru_cache(maxsize=32768)(self._encode_piece_uncached)
        self.training_statistics: dict = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def target_vocab_size(self) -> int:
        return self._target_size

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self._special_ids)

    @property
    def merges(self) -> tuple[tuple[int, int, int], ...]:
        return self._merges

    def special_id(self, name: str) -> int:
        try:
            return self._special_ids[name]
        except KeyError as error:
            raise ValueError(f"Unknown special token: {name}") from error

    @property
    def pad_token_id(self) -> int:
        return self.special_id("<pad>")

    @property
    def bos_token_id(self) -> int:
        return self.special_id("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self.special_id("<eos>")

    @property
    def unk_token_id(self) -> int:
        return self.special_id("<unk>")

    def _configuration(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm": "byte-bpe-v1",
            "normalization": "none",
            "pretokenizer": "unicode-groups-v1",
            "pattern": PIECE_PATTERN,
            "max_piece_bytes": self._max_piece_bytes,
            "target_vocab_size": self._target_size,
            "vocab_size": self.vocab_size,
            "min_frequency": self._min_frequency,
            "special_tokens": list(self._special_names),
        }

    @property
    def fingerprint(self) -> str:
        return config_digest({"config": self._configuration(), "merges": self._merges})

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: Iterable[str] = SPECIAL_TOKENS,
        max_piece_bytes: int = 256,
        progress: Callable[[int, int], None] | None = None,
    ) -> ByteBPETokenizer:
        """Weighted piece counts, incremental pair updates and a lazily validated heap.

        Ties choose the lexicographically smallest pair of IDs. No RNG or external BPE
        library is used. Merges never cross a piece or document boundary.
        """
        base = cls(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            max_piece_bytes=max_piece_bytes,
        )
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be an iterable of documents, not one string")
        offset, frequencies = len(base._special_names), Counter()
        documents, characters, byte_count = 0, 0, 0
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Training documents must be strings")
            documents += 1
            characters += len(text)
            byte_count += len(text.encode("utf-8", errors="strict"))
            for piece in _pieces(text, max_piece_bytes):
                frequencies[tuple(byte + offset for byte in piece)] += 1
        if byte_count == 0:
            raise ValueError("Cannot train on an empty corpus")
        words = sorted(frequencies)
        weights = [frequencies[word] for word in words]
        pair_counts, pair_words = Counter(), {}
        for index, word in enumerate(words):
            for pair, count in Counter(zip(word, word[1:], strict=False)).items():
                pair_counts[pair] += count * weights[index]
                pair_words.setdefault(pair, set()).add(index)
        heap = [(-count, *pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        merges = []
        while len(base) + len(merges) < vocab_size:
            while heap:
                negative_count, left, right = heapq.heappop(heap)
                pair = (left, right)
                if pair_counts.get(pair, 0) == -negative_count and -negative_count > 0:
                    break
            else:
                break
            frequency = -negative_count
            if frequency < min_frequency:
                break
            new_id = len(base) + len(merges)
            changed = set()
            for index in sorted(pair_words.get(pair, ())):
                old_word = words[index]
                new_word = _merge(old_word, pair, new_id)
                old_pairs = Counter(zip(old_word, old_word[1:], strict=False))
                new_pairs = Counter(zip(new_word, new_word[1:], strict=False))
                for affected in old_pairs.keys() | new_pairs.keys():
                    delta = (new_pairs[affected] - old_pairs[affected]) * weights[index]
                    if delta:
                        total = pair_counts.get(affected, 0) + delta
                        if total < 0:
                            raise RuntimeError("Negative pair count in BPE trainer")
                        if total:
                            pair_counts[affected] = total
                        else:
                            pair_counts.pop(affected, None)
                        changed.add(affected)
                    if old_pairs[affected] and not new_pairs[affected]:
                        pair_words[affected].discard(index)
                        if not pair_words[affected]:
                            del pair_words[affected]
                    elif new_pairs[affected] and not old_pairs[affected]:
                        pair_words.setdefault(affected, set()).add(index)
                words[index] = new_word
            for affected in changed:
                count = pair_counts.get(affected, 0)
                if count:
                    heapq.heappush(heap, (-count, *affected))
            merges.append((*pair, new_id))
            if progress and (len(merges) % 256 == 0 or new_id + 1 == vocab_size):
                progress(len(merges), frequency)
        tokenizer = cls(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=base._special_names,
            max_piece_bytes=max_piece_bytes,
            merges=merges,
        )
        tokenizer.training_statistics = {
            "documents": documents,
            "characters": characters,
            "utf8_bytes": byte_count,
            "unique_pieces": len(words),
            "merges_learned": len(merges),
            "target_vocab_size": vocab_size,
            "actual_vocab_size": len(tokenizer),
            "stop_reason": "target_reached"
            if len(tokenizer) == vocab_size
            else "no_pair_at_min_frequency",
        }
        return tokenizer

    def _encode_piece_uncached(self, piece: bytes) -> tuple[int, ...]:
        tokens = tuple(byte + len(self._special_names) for byte in piece)
        while len(tokens) > 1:
            candidates = (
                (self._ranks[pair][0], pair)
                for pair in zip(tokens, tokens[1:], strict=False)
                if pair in self._ranks
            )
            best = min(candidates, default=None)
            if best is None:
                break
            pair = best[1]
            tokens = _merge(tokens, pair, self._ranks[pair][1])
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Iterable[str] | str = (),
    ) -> list[int]:
        if not isinstance(text, str):
            raise ValueError("encode requires a string")
        if isinstance(allowed_special, str) and allowed_special != "all":
            raise ValueError("allowed_special must be an iterable of names or 'all'")
        allowed = set(self._special_ids) if allowed_special == "all" else set(allowed_special)
        if not allowed <= set(self._special_ids):
            raise ValueError("Unknown token in allowed_special")
        ids = [self.bos_token_id] if add_bos else []

        def ordinary(value: str) -> None:
            for piece in _pieces(value, self._max_piece_bytes):
                ids.extend(self._encode_piece(piece))

        if allowed:
            pattern = re.compile(
                "|".join(
                    re.escape(name) for name in sorted(allowed, key=lambda name: (-len(name), name))
                )
            )
            start = 0
            for match in pattern.finditer(text):
                ordinary(text[start : match.start()])
                ids.append(self._special_ids[match[0]])
                start = match.end()
            ordinary(text[start:])
        else:
            ordinary(text)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_batch(self, texts: Iterable[str], **kwargs) -> list[list[int]]:
        if isinstance(texts, (str, bytes)):
            raise ValueError("encode_batch requires a sequence of documents")
        return [self.encode(text, **kwargs) for text in texts]

    def token_bytes(self, token_id: int) -> bytes:
        if type(token_id) is not int or not 0 <= token_id < len(self):
            raise ValueError(f"Invalid token ID: {token_id}")
        return self._vocab[token_id]

    def decode_to_bytes(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> bytes:
        pieces = []
        for token_id in ids:
            value = self.token_bytes(token_id)
            if not skip_special_tokens or token_id >= len(self._special_names):
                pieces.append(value)
        return b"".join(pieces)

    def decode(
        self, ids: Iterable[int], *, skip_special_tokens: bool = False, errors: str = "replace"
    ) -> str:
        if errors not in ("strict", "replace"):
            raise ValueError("errors must be strict or replace")
        return self.decode_to_bytes(ids, skip_special_tokens=skip_special_tokens).decode(
            "utf-8", errors=errors
        )

    def _vocab_payload(self) -> dict:
        return {
            "schema_version": 1,
            "tokens": [
                {"id": index, "special": self._special_names[index]}
                if index < len(self._special_names)
                else {"id": index, "hex": value.hex()}
                for index, value in enumerate(self._vocab)
            ],
        }

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "vocab.json", self._vocab_payload())
        write_json(path / "merges.json", {"schema_version": 1, "merges": self._merges})
        write_json(path / "tokenizer_config.json", self._configuration())
        files = {
            name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ("vocab.json", "merges.json", "tokenizer_config.json")
        }
        # Written last: a partial save is never a loadable artifact.
        write_json(
            path / "tokenizer_manifest.json",
            {
                "schema_version": 1,
                "tokenizer_fingerprint": self.fingerprint,
                "files": files,
            },
        )

    @classmethod
    def load(
        cls, directory: str | Path, *, expected_fingerprint: str | None = None
    ) -> ByteBPETokenizer:
        path = Path(directory)
        manifest = json.loads((path / "tokenizer_manifest.json").read_text(encoding="utf-8"))
        names = {"vocab.json", "merges.json", "tokenizer_config.json"}
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "tokenizer_fingerprint",
            "files",
        }:
            raise ValueError("Invalid tokenizer artifact manifest")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or not isinstance(manifest["files"], dict)
            or set(manifest["files"]) != names
        ):
            raise ValueError("Invalid tokenizer artifact manifest")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Tokenizer checksum mismatch: {name}")
        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        merges = json.loads((path / "merges.json").read_text(encoding="utf-8"))
        fields = {
            "schema_version",
            "algorithm",
            "normalization",
            "pretokenizer",
            "pattern",
            "max_piece_bytes",
            "target_vocab_size",
            "vocab_size",
            "min_frequency",
            "special_tokens",
        }
        if not isinstance(config, dict) or set(config) != fields:
            raise ValueError("Invalid tokenizer configuration fields")
        if (
            not isinstance(merges, dict)
            or set(merges) != {"schema_version", "merges"}
            or type(merges["schema_version"]) is not int
            or merges["schema_version"] != 1
            or not isinstance(merges["merges"], list)
        ):
            raise ValueError("Unsupported merges format")
        tokenizer = cls(
            vocab_size=config["target_vocab_size"],
            min_frequency=config["min_frequency"],
            special_tokens=config["special_tokens"],
            max_piece_bytes=config["max_piece_bytes"],
            merges=merges["merges"],
        )
        if config != tokenizer._configuration():
            raise ValueError("Unsupported or inconsistent tokenizer configuration")
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        if vocab != tokenizer._vocab_payload():
            raise ValueError("Vocabulary is inconsistent with byte alphabet and merges")
        if tokenizer.fingerprint != manifest["tokenizer_fingerprint"] or (
            expected_fingerprint is not None and tokenizer.fingerprint != expected_fingerprint
        ):
            raise ValueError("Tokenizer fingerprint mismatch")
        return tokenizer
