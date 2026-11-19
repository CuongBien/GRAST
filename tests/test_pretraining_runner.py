"""Runner continuation, metric recovery, durable failures and portable input bundles."""

import copy
import csv
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.pretraining import prepare_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.data.training_bundle import create_bundle, extract_bundle
from llm_scratch.research.data_acceptance import make_session, offline_sources
from llm_scratch.research.training_acceptance import _same
from llm_scratch.training import pretrain
from llm_scratch.training.pretrain import PretrainingJob, load_recipe, run_pretraining
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.seed import capture_rng_state, seed_everything

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/wp07_pretrain_smoke.toml"


@pytest.fixture
def inputs(tmp_path):
    source = offline_sources(tmp_path / "sources")
    codec = tmp_path / "tokenizer"
    ByteBPETokenizer(vocab_size=264).save(codec)
    packed = prepare_pretraining(source, codec, tmp_path / "packed", block_size=32)
    return packed, codec


@pytest.fixture
def fast_curves(monkeypatch):
    # State/failure unit tests do not need to redraw the same chart repeatedly.
    monkeypatch.setattr(pretrain, "export_curves", lambda *args: None)


def run(inputs, tmp_path, **kwargs):
    return run_pretraining(CONFIG, *inputs, output_root=tmp_path / "runs", **kwargs)


def payload(run_path):
    return torch.load(run_path / "checkpoints/latest.pt", map_location="cpu", weights_only=True)


def new_job(root):
    from llm_scratch.model.config import ModelConfig
    from llm_scratch.training.config import TrainingConfig

    recipe = load_recipe(CONFIG).data
    rng = seed_everything(**recipe["reproducibility"])
    session = make_session(
        root,
        ModelConfig.from_dict(recipe["model"]),
        TrainingConfig.from_dict(recipe["training"]),
        recipe["data"],
        "cpu",
        rng,
    )
    return PretrainingJob(session, recipe)


def test_runner_relocated_subprocess_resume_matches_uninterrupted_and_recovers_curves(
    inputs,
    tmp_path,
    capsys,
):
    full = run(inputs, tmp_path)
    first = run(inputs, tmp_path, stop_after=2)
    # Neither the previous run directory nor a JSONL tail is required for continuation.
    relocated = tmp_path / "portable"
    relocated.mkdir()
    checkpoint = relocated / "resume.pt"
    shutil.copyfile(first / "checkpoints/latest.pt", checkpoint)
    shutil.rmtree(first)
    destination = tmp_path / "subprocess_runs"
    process = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "llm_scratch.training.pretrain",
            "--config",
            str(CONFIG),
            "--prepared-root",
            str(inputs[0]),
            "--tokenizer",
            str(inputs[1]),
            "--resume",
            str(checkpoint),
            "--output-root",
            str(destination),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    resumed = next(destination.iterdir())
    a, b = payload(full), payload(resumed)
    assert _same(a, b), "Model/optimizer/scaler/scheduler/cursor/RNG/history must all match"
    assert '"event": "train_update"' in process.stdout
    assert '"event": "checkpoint_saved"' in process.stdout
    assert '"event": "validation"' in process.stdout
    for path in (full, resumed):
        assert (path / "checkpoints/final.pt").exists()
        assert (path / "checkpoints/step-00000004.pt").exists()
        assert (path / "artifacts/loss_curves.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (path / "artifacts/loss_curves.svg").read_text()
        history = json.loads((path / "artifacts/history.json").read_text())
        rows = list(csv.DictReader(io.StringIO((path / "artifacts/losses.csv").read_text())))
        assert len(rows) == len(history["train"]) + len(history["validation"]) == 10
        assert json.loads((path / "run.json").read_text())["status"] == "completed"
        assert (path / "artifacts/tokenizer/tokenizer_manifest.json").exists()
    assert '"event": "pretraining_started"' in capsys.readouterr().out


@pytest.mark.parametrize("change", ["recipe", "history", "missing"])
def test_runner_rejects_mismatch_before_weights_or_rng_mutate(
    inputs, tmp_path, fast_curves, change
):
    first = run(inputs, tmp_path, stop_after=2)
    saved = payload(first)
    extra = saved["state"]["data_state"]["extra_state"]
    if change == "recipe":
        extra["contract"]["recipe"]["runner"]["validate_every"] = 3
    elif change == "history":
        extra["history"]["train"].pop()
        extra["history_sha256"] = config_digest(extra["history"])
    else:
        del saved["state"]["data_state"]["extra_state"]
    bad = tmp_path / "bad.pt"
    torch.save(saved, bad)
    job = new_job(inputs[0])
    before = copy.deepcopy(job.session.engine.model.state_dict())
    rng = capture_rng_state(job.session.engine.generator)
    with pytest.raises(ValueError):
        job.load(bad)
    assert _same(before, job.session.engine.model.state_dict())
    assert _same(rng, capture_rng_state(job.session.engine.generator))


def test_checkpoint_write_failure_preserves_previous_boundary(
    inputs, tmp_path, fast_curves, monkeypatch
):
    original = PretrainingJob.save

    def fail_at_four(self, path):
        if self.session.engine.step == 4:
            raise OSError("simulated Drive quota error")
        return original(self, path)

    monkeypatch.setattr(PretrainingJob, "save", fail_at_four)
    with pytest.raises(OSError, match="Drive quota"):
        run(inputs, tmp_path)
    directory = next((tmp_path / "runs").iterdir())
    assert payload(directory)["state"]["step"] == 2
    assert json.loads((directory / "run.json").read_text())["status"] == "failed"
    assert '"event":"pretraining_error"' in (directory / "logs/events.jsonl").read_text()


@pytest.mark.parametrize("during_update", [False, True])
def test_keyboard_interrupt_does_not_save_partial_update(
    inputs,
    tmp_path,
    fast_curves,
    monkeypatch,
    during_update,
):
    original = PretrainingJob.update

    def interrupt(self):
        if self.session.engine.step == 2:
            if during_update:
                self.session._failed = True
                self.session.engine._failed = True
            raise KeyboardInterrupt
        return original(self)

    monkeypatch.setattr(PretrainingJob, "update", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(inputs, tmp_path)
    directory = next((tmp_path / "runs").iterdir())
    assert payload(directory)["state"]["step"] == 2
    assert len(payload(directory)["state"]["data_state"]["extra_state"]["history"]["train"]) == 2


def test_resume_completed_checkpoint_is_noop_with_portable_final(inputs, tmp_path, fast_curves):
    full = run(inputs, tmp_path)
    resumed = run(inputs, tmp_path, resume=full / "checkpoints/latest.pt")
    assert _same(payload(full), payload(resumed))
    assert (resumed / "checkpoints/final.pt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 scaler required")
def test_runner_amp_skips_persist_scaler_cursor_and_history(
    inputs, tmp_path, fast_curves, monkeypatch
):
    original_update = pretrain.TrainingEngine.train_update
    original_save = PretrainingJob.save
    calls = 0
    skipped_checkpoint = tmp_path / "after-skips.pt"

    def overflow_twice(self, batches):
        nonlocal calls
        calls += 1
        handle = None
        if calls <= 2:
            handle = next(self.model.parameters()).register_hook(lambda grad: grad * float("inf"))
        try:
            return original_update(self, batches)
        finally:
            if handle is not None:
                handle.remove()

    def save_and_copy(self, path):
        original_save(self, path)
        if self.session.engine.attempts == 2:
            shutil.copyfile(path, skipped_checkpoint)

    monkeypatch.setattr(pretrain.TrainingEngine, "train_update", overflow_twice)
    monkeypatch.setattr(PretrainingJob, "save", save_and_copy)
    full = run(inputs, tmp_path, device="cuda", precision="float16", stop_after=2)
    skipped = torch.load(skipped_checkpoint, weights_only=True, map_location="cpu")["state"]
    assert skipped["step"] == 0 and skipped["attempts"] == 2
    assert skipped["data_state"]["train_cursor"]["windows_consumed"] == 2
    assert all(r["skipped"] for r in skipped["data_state"]["extra_state"]["history"]["train"])
    resumed = run(
        inputs,
        tmp_path,
        device="cuda",
        precision="float16",
        stop_after=2,
        resume=skipped_checkpoint,
    )
    assert _same(payload(full), payload(resumed))


@pytest.mark.parametrize(
    "old,new",
    [
        ("block_size = 32", "block_size = true"),
        ("block_size = 32", "block_size = 129"),
        ("validate_every = 2", "validate_every = 0"),
        ("milestones = [2, 4, 6]", "milestones = [4, 2]"),
        ("seed = 42", "seed = -1"),
        ('stage = "pretrain"', 'stage = "sft"'),
        ("validation_batches = 2", "validation_batches = -1"),
        ("deterministic = true", "deterministic = 1"),
    ],
)
def test_recipe_invalid_settings_are_rejected(tmp_path, old, new):
    config = tmp_path / "recipe.toml"
    config.write_text(CONFIG.read_text().replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_recipe(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 scaler required")
def test_overflow_budget_bounds_smoke_even_with_long_recipe(
    inputs, tmp_path, fast_curves, monkeypatch
):
    config = tmp_path / "bounded.toml"
    config.write_text(
        CONFIG.read_text().replace("max_attempts_per_step = 16", "max_attempts_per_step = 2"),
        encoding="utf-8",
    )
    original = pretrain.TrainingEngine.train_update

    def always_overflow(self, batches):
        handle = next(self.model.parameters()).register_hook(lambda grad: grad * float("inf"))
        try:
            return original(self, batches)
        finally:
            handle.remove()

    monkeypatch.setattr(pretrain.TrainingEngine, "train_update", always_overflow)
    with pytest.raises(RuntimeError, match="attempt budget"):
        run_pretraining(
            config,
            *inputs,
            device="cuda",
            precision="float16",
            output_root=tmp_path / "runs",
            stop_after=1,
        )
    saved = payload(next((tmp_path / "runs").iterdir()))["state"]
    assert saved["step"] == 0 and saved["attempts"] == 2


def test_bundle_roundtrip_and_external_sha(inputs, tmp_path):
    archive = tmp_path / "inputs.zip"
    result = create_bundle(*inputs, archive)
    output = extract_bundle(archive, tmp_path / "extracted", expected_sha256=result["sha256"])
    assert file_sha256(output / "pretraining_data/train.u32") == file_sha256(
        inputs[0] / "train.u32"
    )
    assert set(result["manifest"]["files"]) == {
        i.filename for i in zipfile.ZipFile(archive).infolist()
    } - {"bundle_manifest.json"}
    with pytest.raises(ValueError, match="SHA256"):
        extract_bundle(archive, tmp_path / "wrong", expected_sha256="0" * 64)
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("corruption", ["traversal", "duplicate", "payload", "manifest"])
def test_bundle_refuses_unsafe_or_corrupted_entries(inputs, tmp_path, corruption):
    archive = tmp_path / "inputs.zip"
    create_bundle(*inputs, archive)
    with zipfile.ZipFile(archive) as source:
        entries = {n: source.read(n) for n in source.namelist()}
    if corruption == "traversal":
        entries["../escape"] = b"forbidden"
    elif corruption == "payload":
        entries["pretraining_data/train.u32"] += b"BAD"
    elif corruption == "manifest":
        data = json.loads(entries["bundle_manifest.json"])
        data["block_size"] = 128
        entries["bundle_manifest.json"] = json.dumps(data).encode()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as target:
        for name, value in entries.items():
            target.writestr(name, value)
        if corruption == "duplicate":
            with pytest.warns(UserWarning):
                target.writestr("bundle_manifest.json", entries["bundle_manifest.json"])
    with pytest.raises(ValueError):
        extract_bundle(bad, tmp_path / "bad_output")
    assert not (tmp_path / "escape").exists()


def test_colab_training_cells_compile_and_stream_output():
    notebook = json.loads(
        (ROOT / "notebooks/wp07_pretraining_colab.ipynb").read_text(encoding="utf-8")
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    source = json.dumps(notebook)
    assert "Popen" in source and "stderr=subprocess.STDOUT" in source and "flush=True" in source
    assert "--resume" in source and "drive.mount" in source
