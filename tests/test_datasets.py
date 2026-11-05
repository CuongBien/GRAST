"""Split/leakage contracts and a full offline pipeline using authored fixture data."""

import copy
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_scratch.cli import main
from llm_scratch.data.config import load_data_config
from llm_scratch.data.datasets import (
    audit_splits,
    deduplicate,
    download_source,
    file_sha256,
    normalize_text,
    split_rows,
)
from llm_scratch.data.prepare import prepare_data

ROOT = Path(__file__).resolve().parents[1]


def test_split_is_order_independent_label_independent_and_seeded():
    rows = [
        {"question_id": str(i), "question": f"question {i}", "canonical_answer": str(i)}
        for i in range(30)
    ]
    counts = {"train": 20, "validation": 5, "test": 5}
    first = split_rows(rows, counts, 42, "test")
    random.Random(7).shuffle(rows)
    for row in rows:
        row["canonical_answer"] = "999"
    second = split_rows(rows, counts, 42, "test")

    def ids(partitions):
        return {key: [row["question_id"] for row in value] for key, value in partitions.items()}

    assert ids(first) == ids(second)
    assert ids(first) != ids(split_rows(rows, counts, 43, "test"))
    assert [len(first[key]) for key in counts] == [20, 5, 5]


def test_unicode_whitespace_dedup_and_conflicting_labels():
    rows = [
        {"question_id": "a", "question": "WHAT is ３ + 2?", "canonical_answer": "5"},
        {"question_id": "b", "question": " what  is 3 + 2? ", "canonical_answer": "5"},
    ]
    kept, report = deduplicate(rows)
    assert len(kept) == 1
    assert report["within_source_duplicates"] == 1
    rows[1]["canonical_answer"] = "6"
    with pytest.raises(ValueError, match="conflicting"):
        deduplicate(rows)
    assert normalize_text(" Hello\n WORLD ") == "hello world"


def test_audit_detects_cross_source_text_leakage_and_semantic_variants():
    first = {"question_id": "a", "question": "How many?", "split": "train", "semantic_id": "same"}
    second = {
        "question_id": "b",
        "question": " HOW  MANY? ",
        "split": "test",
        "semantic_id": "other",
    }
    with pytest.raises(ValueError, match="leakage"):
        audit_splits({"a": {"train": [first]}, "b": {"test": [second]}})
    second["question"] = "Different rendering"
    second["semantic_id"] = "same"
    with pytest.raises(ValueError, match="leakage"):
        audit_splits({"a": {"train": [first]}, "b": {"test": [second]}})


def test_download_verifies_checksum_and_offline_never_fetches(tmp_path, monkeypatch):
    content = b"authored fixture\n"
    spec = {
        "url": "https://fixtures.invalid/data",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    calls = []

    def fake_open(*args, **kwargs):
        calls.append(True)
        return io.BytesIO(content)

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    path = download_source(spec, tmp_path)
    assert calls == [True]
    assert download_source(spec, tmp_path, offline=True) == path
    assert calls == [True]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        download_source(spec, tmp_path, offline=True)


def test_bad_download_does_not_leave_a_trusted_cache_file(tmp_path, monkeypatch):
    spec = {"url": "https://fixtures.invalid/data", "sha256": "a" * 64, "size_bytes": 3}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="checksum"):
        download_source(spec, tmp_path)
    assert not list(tmp_path.rglob("source"))
    assert not list(tmp_path.rglob(".download-*"))
    with pytest.raises(FileNotFoundError, match="Offline"):
        download_source(spec, tmp_path, offline=True)


@pytest.fixture
def fixture_config(tmp_path):
    """Source fixtures are authored here; no remote dataset/network is used in CI."""
    cache = tmp_path / "cache"
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    stories = [f"A fox shares a drum in tale {index}." for index in range(8)]
    tiny_train = stories + ["  A FOX shares a drum in tale 0. ", "A bear rests under a tree."]
    tiny_val = [
        "A bear rests under a tree.",
        "A kite floats\u2028beside a lake.",
        "A rabbit carries a basket.",
        "A bird sings near a pond.",
    ]
    for split, records in (("train", tiny_train), ("validation", tiny_val)):
        pq.write_table(pa.table({"text": records}), source_dir / f"tiny_{split}.parquet")
    gsm_train = [
        {
            "question": f"Authored training problem {i}: what is the next number?",
            "answer": f"#### {i + 1}",
        }
        for i in range(10)
    ]
    gsm_train += [
        copy.deepcopy(gsm_train[0]),
        {"question": "What is twenty plus one?", "answer": "#### 21"},
    ]
    gsm_test = [{"question": "What is twenty plus one?", "answer": "#### 21"}] + [
        {
            "question": f"Authored held-out problem {i}: what is the next number?",
            "answer": f"#### {i + 1}",
        }
        for i in range(3)
    ]
    for split, rows in (("train", gsm_train), ("test", gsm_test)):
        (source_dir / f"gsm_{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    lock = json.loads((ROOT / "configs/data/sources.lock.json").read_text())
    for name, prefix, suffix in (("tinystories", "tiny", "parquet"), ("gsm8k", "gsm", "jsonl")):
        lock[name]["revision"] = "a" * 40
        lock[name]["license"] = "authored-test-fixture"
        for split in lock[name]["files"]:
            path = source_dir / f"{prefix}_{split}.{suffix}"
            digest = file_sha256(path)
            spec = {
                "url": f"https://fixtures.invalid/{'a' * 40}/{name}/{split}",
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
            lock[name]["files"][split] = spec
            destination = cache / "raw" / digest / "source"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(path.read_bytes())
    (tmp_path / "sources.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    text = (ROOT / "configs/data/wp03.toml").read_text()
    for old, new in [
        ('root = "../../artifacts/runs"', f'root = "{(tmp_path / "runs").as_posix()}"'),
        ('cache_root = "../../artifacts/datasets/cache"', f'cache_root = "{cache.as_posix()}"'),
        ("train_records = 10000", "train_records = 10"),
        ("heldout_records = 2000", "heldout_records = 4"),
        ("validation_records = 500", "validation_records = 2"),
        ("train_per_level = 500", "train_per_level = 3"),
        ("validation_per_level = 100", "validation_per_level = 2"),
        ("test_per_level = 100", "test_per_level = 2"),
        ("pilot_per_level_per_split = 50", "pilot_per_level_per_split = 1"),
    ]:
        text = text.replace(old, new)
    path = tmp_path / "data.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_full_offline_pipeline_is_portable_and_pilot_has_no_gold(fixture_config, tmp_path):
    first = prepare_data(load_data_config(fixture_config), offline=True)
    second = prepare_data(
        load_data_config(fixture_config, output_root=tmp_path / "other-runs"), offline=True
    )
    relative = "artifacts/datasets/manifest.json"
    assert (first / relative).read_bytes() == (second / relative).read_bytes()
    manifest = json.loads((first / relative).read_text())
    assert manifest["scope"] == "full" and manifest["leakage_audit"]["passed"]
    assert manifest["deduplication"]["tinystories"]["train"]["within_source_duplicates"] == 1
    assert (
        manifest["deduplication"]["tinystories"]["train_vs_heldout"]["heldout_overlap_removed"] == 1
    )
    assert manifest["deduplication"]["gsm8k"]["train"]["heldout_overlap_removed"] == 1
    base = first / "artifacts/datasets"
    for file in manifest["files"].values():
        assert file_sha256(base / file["path"]) == file["sha256"]
    memberships = {}
    for split in ("validation", "test"):
        prompts = [
            json.loads(line)
            for line in (base / f"pilot/{split}.questions.jsonl").read_text().splitlines()
        ]
        refs = [
            json.loads(line)
            for line in (base / f"pilot/{split}.references.jsonl").read_text().splitlines()
        ]
        original = [
            json.loads(line) for line in (base / f"math/{split}.jsonl").read_text().splitlines()
        ]
        assert len(prompts) == len(refs) == 3
        assert all(not ({"answer", "canonical_answer", "expression"} & set(row)) for row in prompts)
        memberships[split] = {row["question_id"] for row in prompts}
        assert memberships[split] == {row["question_id"] for row in refs}
        assert memberships[split] <= {row["question_id"] for row in original}
    assert not memberships["validation"] & memberships["test"]
    official_test = [
        json.loads(line) for line in (base / "gsm8k/test.jsonl").read_text().splitlines()
    ]
    assert len(official_test) == 4 and all(row["upstream_split"] == "test" for row in official_test)
    tiny_heldout = [
        json.loads(line)
        for split in ("validation", "test")
        for line in (base / f"tinystories/{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("\u2028" in row["text"] for row in tiny_heldout)
    assert json.loads((first / "run.json").read_text())["status"] == "completed"
    metadata = json.loads((first / "metadata.json").read_text())
    assert metadata["execution_device"] == "cpu"
    assert "cuda_version" in metadata["runtime_inventory"]


def test_data_config_errors_are_rejected(fixture_config):
    text = fixture_config.read_text()
    fixture_config.write_text(text.replace("operations = 1", "operations = 2"), encoding="utf-8")
    with pytest.raises(ValueError, match="operation count"):
        load_data_config(fixture_config)


def test_prepare_cli_failure_is_logged_on_missing_offline_cache(fixture_config, tmp_path):
    data = load_data_config(fixture_config)
    raw_root = Path(data.data["artifacts"]["cache_root"]) / "raw"
    first_source = next(raw_root.iterdir())
    # Keep good inputs intact, but ask the CLI to use a missing cache offline.
    fixture_config.write_text(
        fixture_config.read_text().replace(
            raw_root.parent.as_posix(), (tmp_path / "missing-cache").as_posix()
        ),
        encoding="utf-8",
    )
    assert main(["prepare-data", "--config", str(fixture_config), "--offline"]) == 1
    assert first_source.is_dir()
    (run_path,) = Path(data.data["artifacts"]["root"]).iterdir()
    status = json.loads((run_path / "run.json").read_text())
    assert status["status"] == "failed" and "Offline source cache miss" in status["error"]
    assert not (run_path / "result.json").exists()
