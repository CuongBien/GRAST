"""Offline tokenizer acceptance and held-out isolation with authored JSONL fixtures."""

import json
from pathlib import Path

import pytest

from llm_scratch.cli import main
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import SPECIAL_TOKENS, ByteBPETokenizer
from llm_scratch.data.train_tokenizer import load_tokenizer_training_config, train_tokenizer
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "datasets"
    files = {}
    sources = {
        "tinystories/train.jsonl": ["A cat sat.", "A cat ran.", "A fox sat.", "Hello Việt Nam!"],
        "tinystories/validation.jsonl": ["An unseen elephant 🐘."],
        "math/test.jsonl": ["What is 12 + 34?", "Compute 7 × 8."],
    }
    for relative, texts in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
            encoding="utf-8",
        )
        files[relative] = {
            "path": relative,
            "records": len(texts),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path = tmp_path / "manifest.json"
    manifest = {"work_package": "WP03", "scope": "full", "files": files}
    manifest["dataset_fingerprint"] = config_digest(manifest)
    write_json(manifest_path, manifest)
    component = tmp_path / "bpe.toml"
    component.write_text(
        '[tokenizer]\nalgorithm="bpe"\nvocab_size=288\nspecial_tokens='
        + json.dumps(SPECIAL_TOKENS)
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "train.toml"
    config.write_text(
        """schema_version=1
tokenizer_config="bpe.toml"
dataset_manifest="manifest.json"
dataset_runs_root="runs"
[experiment]
name="tokenizer-fixture"
phase=1
[reproducibility]
seed=42
[artifacts]
root="runs"
[training]
file="tinystories/train.jsonl"
field="text"
max_records=4
min_frequency=1
max_piece_bytes=256
[evaluation]
files=[
{file="tinystories/train.jsonl",field="text"},
{file="tinystories/validation.jsonl",field="text"},
{file="math/test.jsonl",field="text"}]
""",
        encoding="utf-8",
    )
    return config, root, manifest_path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_acceptance_and_held_out_changes_cannot_change_codec(dataset, capsys):
    config, root, manifest_path = dataset
    assert main(["train-tokenizer", "--config", str(config), "--dataset-root", str(root)]) == 0
    response = json.loads(capsys.readouterr().out)
    first = Path(response["run_directory"])
    report = read(first / "artifacts/tokenizer_report.json")
    snapshot = read(first / "config.json")
    manifest = read(manifest_path)
    assert snapshot["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert snapshot["dataset_root"] == str(root.resolve())
    assert report["training"]["documents"] == 4
    assert report["save_load_id_checks"] == 7
    assert all(report["integration_checks"].values())
    for metric in report["evaluation"].values():
        assert metric["round_trip_documents"] == metric["documents"]
        assert metric["unknown_tokens"] == 0
        assert metric["tokens_per_character"] == metric["tokens"] / metric["characters"]
    assert read(first / "run.json")["status"] == "completed"
    assert (first / "logs/run.log").stat().st_size > 0
    assert read(first / "metadata.json")["tokenizer_fingerprint"] == report["tokenizer_fingerprint"]
    artifact = first / "artifacts/tokenizer"
    codec = ByteBPETokenizer.load(artifact, expected_fingerprint=report["tokenizer_fingerprint"])
    assert (
        codec.decode(codec.encode("Entirely new 文字 🐉"), errors="strict")
        == "Entirely new 文字 🐉"
    )

    heldout = root / "tinystories/validation.jsonl"
    heldout.write_text(json.dumps({"text": "zzzzzzzz " * 100}) + "\n", encoding="utf-8")
    info = manifest["files"]["tinystories/validation.jsonl"]
    info.update(sha256=file_sha256(heldout), size_bytes=heldout.stat().st_size)
    manifest.pop("dataset_fingerprint")
    manifest["dataset_fingerprint"] = config_digest(manifest)
    write_json(manifest_path, manifest)
    second = train_tokenizer(load_tokenizer_training_config(config, dataset_root=root))
    assert (
        read(second / "artifacts/tokenizer_report.json")["tokenizer_fingerprint"]
        == codec.fingerprint
    )
    for path in artifact.iterdir():
        assert path.read_bytes() == (second / "artifacts/tokenizer" / path.name).read_bytes()


@pytest.mark.parametrize("split", ["validation", "test"])
def test_cannot_fit_on_held_out_split(dataset, split):
    config, root, _ = dataset
    config.write_text(
        config.read_text().replace(
            'file="tinystories/train.jsonl"', f'file="tinystories/{split}.jsonl"'
        )
    )
    with pytest.raises(ValueError, match="train split"):
        load_tokenizer_training_config(config, dataset_root=root)


def test_corrupt_payload_fails_with_durable_log(dataset):
    config, root, _ = dataset
    (root / "tinystories/train.jsonl").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        train_tokenizer(load_tokenizer_training_config(config, dataset_root=root))
    failed = next((config.parent / "runs").iterdir())
    assert read(failed / "run.json")["status"] == "failed"
    assert "run_failed" in (failed / "logs/events.jsonl").read_text()
    assert not (failed / "artifacts/tokenizer").exists()


def test_corrupt_manifest_fails_before_creating_run(dataset):
    config, root, manifest_path = dataset
    manifest = read(manifest_path)
    manifest["files"]["tinystories/train.jsonl"]["records"] = 100
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="fingerprint"):
        train_tokenizer(load_tokenizer_training_config(config, dataset_root=root))
    assert not (config.parent / "runs").exists()
