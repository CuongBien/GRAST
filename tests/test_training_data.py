"""Packing coverage, train/held-out identity and transactional consumed-data resume."""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.pretraining import PackedCorpus, prepare_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.data_acceptance import offline_sources
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.data import BatchWindow, TrainingCursor, window_fingerprint
from llm_scratch.training.session import PretrainingSession
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.seed import seed_everything

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def prepared(root, block_size=32):
    root.mkdir(parents=True, exist_ok=True)
    source = offline_sources(root / "source")
    codec = ByteBPETokenizer(vocab_size=264)
    codec.save(root / "codec")
    output = prepare_pretraining(source, root / "codec", root / "packed", block_size=block_size)
    return output, source, codec


def new_session(root, device="cpu", precision="float32", batch_size=2, seed=42):
    rng = seed_everything(42)
    train, val = PackedCorpus(root, "train"), PackedCorpus(root, "validation")
    cfg = ModelConfig(
        vocab_size=train.meta["vocab_size"],
        context_length=train.meta["block_size"],
        hidden_size=16,
        num_heads=4,
        num_layers=2,
        ffn_hidden_size=32,
        dropout=0.2,
    )
    model = CausalLM(cfg, tokenizer_fingerprint=train.meta["tokenizer_fingerprint"]).to(device)
    engine = TrainingEngine(
        model,
        TrainingConfig(
            max_steps=5, warmup_steps=1, gradient_accumulation_steps=2, precision=precision
        ),
        generator=rng,
    )
    return PretrainingSession(
        engine, TrainingCursor(train, batch_size=batch_size, accumulation_steps=2, seed=seed), val
    )


def rehash_source(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, spec in manifest["files"].items():
        spec.update(
            sha256=file_sha256(root / relative), size_bytes=(root / relative).stat().st_size
        )
    manifest["dataset_fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "dataset_fingerprint"}
    )
    write_json(root / "manifest.json", manifest)


@pytest.mark.parametrize("width", [3, 7, 32, 128])
def test_packing_preserves_all_targets_once_unicode_and_tail_padding(tmp_path, width):
    root, source, codec = prepared(tmp_path, width)
    for split in ("train", "validation"):
        corpus = PackedCorpus(root, split)
        rows = [
            json.loads(line)
            for line in (source / f"tinystories/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        tokens = [
            token for row in rows for token in codec.encode(row["text"], add_bos=True, add_eos=True)
        ]
        assert corpus.tokens.tolist() == tokens
        observed = []
        for index in range(len(corpus)):
            batch = corpus.batch([index])
            valid = batch["attention_mask"][0]
            ids, labels = batch["input_ids"][0], batch["labels"][0]
            start = index * (width - 1)
            assert ids[valid].tolist() == tokens[start : start + width]
            assert (labels[~valid] == -100).all()
            for j in range(1, width):
                if valid[j] and valid[j - 1] and labels[j] != -100:
                    observed.append((start + j, int(labels[j])))
        expected = [(i, t) for i, t in enumerate(tokens) if i > 0 and t != codec.bos_token_id]
        assert observed == expected
        assert len(observed) == corpus.meta["splits"][split]["supervised_pairs"]
    with pytest.raises(ValueError):
        PackedCorpus(root, "test")


def test_prepare_reads_no_test_file_and_publishes_deterministic_portable_identity(
    tmp_path, monkeypatch
):
    source = offline_sources(tmp_path / "source")
    codec = ByteBPETokenizer(vocab_size=264)
    codec.save(tmp_path / "codec")
    forbidden = source / "tinystories/test.jsonl"
    forbidden.write_text("not JSON - never open")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"]["tinystories/test.jsonl"] = {
        "path": "tinystories/test.jsonl",
        "sha256": file_sha256(forbidden),
        "size_bytes": forbidden.stat().st_size,
        "records": 1,
        "role": "pretraining",
    }
    manifest["dataset_fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "dataset_fingerprint"}
    )
    write_json(source / "manifest.json", manifest)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.resolve() == forbidden.resolve():
            raise AssertionError("Read test set")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    a = prepare_pretraining(source, tmp_path / "codec", tmp_path / "a", block_size=32)
    b = prepare_pretraining(source, tmp_path / "codec", tmp_path / "b", block_size=32)
    assert PackedCorpus(a, "train").identity == PackedCorpus(b, "train").identity
    with pytest.raises(FileExistsError):
        prepare_pretraining(source, tmp_path / "codec", a, block_size=32)
    shutil.copytree(a, tmp_path / "moved")
    assert PackedCorpus(tmp_path / "moved", "train").identity == PackedCorpus(a, "train").identity


@pytest.mark.parametrize(
    "bad", ["source_checksum", "duplicate_train_validation", "token_payload", "manifest"]
)
def test_corruption_and_leakage_are_rejected(tmp_path, bad):
    root, source, _ = prepared(tmp_path / "good")
    if bad in ("source_checksum", "duplicate_train_validation"):
        target = source / "tinystories/validation.jsonl"
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        if bad == "source_checksum":
            rows[0]["text"] += " changed"
        else:
            train = json.loads((source / "tinystories/train.jsonl").read_text().splitlines()[0])
            rows[0]["text"] = "  " + train["text"].upper() + "  "
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if bad == "duplicate_train_validation":
            rehash_source(source)
        with pytest.raises(ValueError):
            prepare_pretraining(source, tmp_path / "good/codec", tmp_path / "bad", block_size=32)
        assert not (tmp_path / "bad/manifest.json").exists()
    else:
        clone = tmp_path / "bad"
        shutil.copytree(root, clone)
        if bad == "token_payload":
            with (clone / "train.u32").open("r+b") as f:
                f.write(b"\xff\xff\xff\xff")
        else:
            meta = json.loads((clone / "manifest.json").read_text())
            meta["block_size"] = 64
            write_json(clone / "manifest.json", meta)
        with pytest.raises(ValueError):
            PackedCorpus(clone, "train")


def test_cursor_abort_commit_partial_epoch_and_shuffle_leave_global_rng_unchanged(tmp_path):
    root, _, _ = prepared(tmp_path)
    cursor = TrainingCursor(
        PackedCorpus(root, "train"), batch_size=2, accumulation_steps=2, seed=42
    )
    before = cursor.state_dict()
    rng = torch.get_rng_state().clone()
    a = cursor.peek()
    digest = window_fingerprint(a)
    with pytest.raises(RuntimeError):
        cursor.state_dict()
    with pytest.raises(RuntimeError):
        cursor.peek()
    with pytest.raises(RuntimeError):
        cursor.commit(BatchWindow(a.batches, a.indices))
    cursor.abort(a)
    assert cursor.state_dict() == before
    b = cursor.peek()
    assert window_fingerprint(b) == digest and b.indices == a.indices
    cursor.commit(b)
    with pytest.raises(RuntimeError):
        cursor.commit(b)
    indices = [i for batch in b.indices for i in batch]
    while cursor.epoch == 0:
        window = cursor.peek()
        indices.extend(i for batch in window.indices for i in batch)
        cursor.commit(window)
    assert sorted(indices) == list(range(len(cursor.corpus)))
    assert cursor.epoch == 1 and cursor.offset == 0 and cursor.windows_consumed == 3
    assert torch.equal(rng, torch.get_rng_state())
    cursor.validate_state(cursor.state_dict())
    for key, value in (
        ("offset", 1),
        ("epoch", True),
        ("windows_consumed", 0),
        ("order_sha256", "bad"),
    ):
        with pytest.raises(ValueError):
            cursor.validate_state(cursor.state_dict() | {key: value})


@pytest.mark.parametrize("device", DEVICES)
def test_actual_data_next_batches_losses_and_full_state_resume_exact(tmp_path, device):
    root, _, _ = prepared(tmp_path)
    session = new_session(root, device)
    for _ in range(2):
        session.update()
    assert session.train.offset == 8 and session.train.epoch == 0
    path = tmp_path / "resume.pt"
    session.save(path)
    rng = torch.get_rng_state().clone()
    cursor = session.train.state_dict()
    value = session.validate()["loss"]
    assert session.train.state_dict() == cursor and torch.equal(rng, torch.get_rng_state())
    reference = [session.update() for _ in range(3)]
    final = copy.deepcopy(session.engine.model.state_dict())
    opt = copy.deepcopy(session.engine.optimizer.state_dict())
    restored = new_session(root, device)
    restored.load(path)
    assert restored.validate()["loss"] == value
    for old in reference:
        new = restored.update()
        for key in ("loss", "step", "attempt", "batch_sha256", "block_indices", "learning_rates"):
            assert old[key] == new[key]
    for k, v in final.items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, restored.engine.model.state_dict()[k], rtol=0, atol=0)
        else:
            assert v == restored.engine.model.state_dict()[k]
    from llm_scratch.research.training_acceptance import _same

    assert _same(opt, restored.engine.optimizer.state_dict())
    assert session.train.state_dict() == restored.train.state_dict()
    assert session.engine.scheduler.state_dict() == restored.engine.scheduler.state_dict()


@pytest.mark.parametrize("change", ["seed", "batch", "dataset", "cursor"])
def test_full_resume_rejects_data_mismatch_before_weights_or_rng_change(tmp_path, change):
    root, _, _ = prepared(tmp_path / "a")
    original = new_session(root)
    original.update()
    original.save(tmp_path / "checkpoint.pt")
    if change == "dataset":
        clone = tmp_path / "other"
        shutil.copytree(root, clone)
        meta = json.loads((clone / "manifest.json").read_text())
        meta["source_dataset_fingerprint"] = "f" * 64
        meta["fingerprint"] = config_digest({k: v for k, v in meta.items() if k != "fingerprint"})
        write_json(clone / "manifest.json", meta)
        root = clone
    session = new_session(
        root, batch_size=1 if change == "batch" else 2, seed=43 if change == "seed" else 42
    )
    path = tmp_path / "checkpoint.pt"
    if change == "cursor":
        payload = torch.load(path, weights_only=True)
        payload["state"]["data_state"]["train_cursor"]["offset"] = 1
        path = tmp_path / "bad.pt"
        torch.save(payload, path)
    rng = torch.get_rng_state().clone()
    weights = copy.deepcopy(session.engine.model.state_dict())
    with pytest.raises(ValueError):
        session.load(path)
    assert torch.equal(rng, torch.get_rng_state()) and session.engine.attempts == 0
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, session.engine.model.state_dict()[k], rtol=0, atol=0)
    assert session.train.windows_consumed == 0


def test_failure_after_optimizer_aborts_cursor_and_requires_full_checkpoint_reload(
    tmp_path, monkeypatch
):
    root, _, _ = prepared(tmp_path)
    session = new_session(root)
    session.save(tmp_path / "initial.pt")
    before = session.train.state_dict()
    update = session.engine.train_update

    def fail_after(batches):
        update(batches)
        raise RuntimeError("after optimizer")

    monkeypatch.setattr(session.engine, "train_update", fail_after)
    with pytest.raises(RuntimeError, match="after optimizer"):
        session.update()
    assert session.train.state_dict() == before and session.engine.attempts == 1
    with pytest.raises(RuntimeError):
        session.update()
    with pytest.raises(RuntimeError):
        session.save(tmp_path / "bad.pt")
    session.load(tmp_path / "initial.pt")
    monkeypatch.setattr(session.engine, "train_update", update)
    session.update()
    assert session.train.windows_consumed == session.engine.attempts == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 data-consumption on overflow")
def test_amp_skip_consumes_data_and_resume_keeps_next_window(tmp_path):
    root, _, _ = prepared(tmp_path)
    session = new_session(root, "cuda", "float16")
    handle = session.engine.parameters[0].register_hook(
        lambda grad: torch.full_like(grad, float("inf"))
    )
    try:
        record = session.update()
    finally:
        handle.remove()
    assert record["skipped"] and session.engine.step == 0 and session.train.windows_consumed == 1
    path = tmp_path / "skip.pt"
    session.save(path)
    expected = session.train.peek()
    digest = window_fingerprint(expected)
    session.train.abort(expected)
    resumed = new_session(root, "cuda", "float16")
    resumed.load(path)
    actual = resumed.train.peek()
    assert window_fingerprint(actual) == digest
    resumed.train.abort(actual)


def test_full_data_resume_in_fresh_process_with_relocated_corpus(tmp_path):
    root, _, _ = prepared(tmp_path / "source")
    session = new_session(root)
    for _ in range(2):
        session.update()
    checkpoint = tmp_path / "resume.pt"
    session.save(checkpoint)
    expected = session.update()
    shutil.copytree(root, tmp_path / "relocated")
    code = """
import json,sys
from test_training_data import new_session
session=new_session(sys.argv[1]); session.load(sys.argv[2])
record=session.update()
print(json.dumps({'marker':'fresh_data_resume','batch_sha256':record['batch_sha256'],'loss':record['loss']}),flush=True)
"""
    process = subprocess.run(
        [sys.executable, "-u", "-c", code, str(tmp_path / "relocated"), str(checkpoint)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    last = json.loads(process.stdout.splitlines()[-1])
    assert (
        last["marker"] == "fresh_data_resume" and last["batch_sha256"] == expected["batch_sha256"]
    )
    assert last["loss"] == expected["loss"]


@pytest.mark.parametrize("which", ["manifest.json", "train.u32"])
def test_loaded_corpus_refuses_mutation_during_training(tmp_path, which):
    root, _, _ = prepared(tmp_path)
    corpus = PackedCorpus(root, "train")
    path = root / which
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(RuntimeError, match="changed after verification"):
        corpus.batch([0])


@pytest.mark.parametrize("changed", ["tokenizer", "stage", "vocab", "context"])
def test_session_refuses_incompatible_model_or_stage(tmp_path, changed):
    root, _, _ = prepared(tmp_path)
    session = new_session(root)
    if changed == "tokenizer":
        session.engine.model.tokenizer_fingerprint = "f" * 64
    elif changed == "stage":
        from dataclasses import replace

        session.engine.config = replace(session.engine.config, stage="sft")
    else:
        from dataclasses import replace

        session.engine.model.config = replace(
            session.engine.model.config,
            **({"vocab_size": 300} if changed == "vocab" else {"context_length": 16}),
        )
    with pytest.raises(ValueError, match="contract mismatch"):
        PretrainingSession(session.engine, session.train, session.validation)


def test_cursor_and_batches_stay_on_cpu_under_another_default_device(tmp_path):
    root, _, _ = prepared(tmp_path)
    corpus = PackedCorpus(root, "train")
    with torch.device("meta"):
        cursor = TrainingCursor(corpus, batch_size=2, accumulation_steps=2, seed=42)
        window = cursor.peek()
        assert all(t.device.type == "cpu" for batch in window.batches for t in batch.values())
        assert len(window_fingerprint(window)) == 64
        cursor.abort(window)
