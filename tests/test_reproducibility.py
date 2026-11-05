"""Behavioral acceptance checks for WP02, with real RNGs and isolated processes."""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_scratch.cli import main
from llm_scratch.smoke import run_smoke
from llm_scratch.training.checkpoint import load_checkpoint, save_checkpoint
from llm_scratch.utils.config import load_config, validate_config
from llm_scratch.utils.environment import git_metadata, inspect_environment
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything

CONFIG = Path(__file__).resolve().parents[1] / "configs/experiments/local_smoke.toml"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "num_heads", 7),
        ("model", "hidden_size", True),
        ("tokenizer", "vocab_size", 2048),
        ("generation", "k_values", [1, 2, 2]),
        ("training", "learning_rate", float("nan")),
        ("training", "warmup_steps", -1),
        ("reproducibility", "seed", -1),
        ("environment", "device", "cdua"),
    ],
)
def test_invalid_config_is_rejected(section, key, value):
    data = copy.deepcopy(load_config(CONFIG).data)
    data[section][key] = value
    with pytest.raises(ValueError):
        validate_config(data)


def test_composition_paths_and_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(CONFIG, seed=7)
    assert config.data["reproducibility"]["seed"] == 7
    assert Path(config.data["artifacts"]["root"]) == CONFIG.parents[2] / "artifacts/runs"
    assert set(config.sources) == {"experiment", "model", "tokenizer", "training", "generation"}
    config.data["model"]["typo"] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_config(config.data)


def _draw(generator):
    return (
        random.random(),
        np.random.random(),
        generator.random(),
        torch.rand(4).tolist(),
        (torch.randn(4, 4) @ torch.randn(4, 4)).tolist(),
    )


def test_seed_and_checkpoint_restore_all_rngs(tmp_path):
    generator = seed_everything(42)
    first = _draw(generator)
    assert _draw(seed_everything(42)) == first
    assert _draw(seed_everything(43)) != first
    generator = seed_everything(42)
    path = tmp_path / "state.pt"
    tensor = torch.arange(4)
    save_checkpoint(path, {"step": 3, "tensor": tensor}, generator=generator)
    expected = _draw(generator)
    state = load_checkpoint(path, generator=generator)
    assert _draw(generator) == expected
    assert state["step"] == 3
    assert torch.equal(state["tensor"], tensor)
    assert not list(tmp_path.glob(".state.pt.*"))


def test_run_snapshot_is_frozen_and_failure_is_logged(tmp_path):
    config = load_config(CONFIG, output_root=tmp_path)
    with pytest.raises(RuntimeError, match="intentional failure"):
        with Run(config) as run:
            snapshot = (run.path / "config.json").read_bytes()
            config.data["reproducibility"]["seed"] = 999
            assert (run.path / "config.json").read_bytes() == snapshot
            raise RuntimeError("intentional failure")
    status = json.loads((run.path / "run.json").read_text())
    assert status["status"] == "failed"
    assert "intentional failure" in (run.path / "logs/run.log").read_text()
    assert json.loads((run.path / "metadata.json").read_text())["seed"] == 42


def test_interrupted_checkpoint_save_preserves_previous_file(tmp_path, monkeypatch):
    generator = seed_everything(42)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"step": 1}, generator=generator)
    previous = path.read_bytes()

    def interrupted_save(payload, stream):
        stream.write(b"incomplete")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        save_checkpoint(path, {"step": 2}, generator=generator)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".checkpoint.pt.*"))


def test_smoke_repeats_across_independent_runs_and_seeds(tmp_path):
    config = load_config(CONFIG, output_root=tmp_path)
    first = run_smoke(config)
    second = run_smoke(config)
    third = run_smoke(load_config(CONFIG, output_root=tmp_path, seed=43))
    results = [json.loads((path / "result.json").read_text()) for path in (first, second, third)]
    assert first != second
    assert results[0]["probe_sha256"] == results[1]["probe_sha256"]
    assert results[0]["probe_sha256"] != results[2]["probe_sha256"]
    for path in (first, second, third):
        assert json.loads((path / "run.json").read_text())["status"] == "completed"
        for repeat in (1, 2):
            assert (path / f"checkpoints/repeat_{repeat}.pt").is_file()
            artifact = json.loads((path / f"artifacts/repeat_{repeat}.json").read_text())
            assert artifact["checkpoint_resume_equal"]
            assert artifact["settings"]["deterministic_algorithms"]
            assert artifact["settings"]["python_hash_seed"] == str(
                results[(first, second, third).index(path)]["seed"]
            )


def test_cuda_gate_does_not_fall_back_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = inspect_environment("cuda", "A100")
    assert not report["validation"]["ok"]
    assert report["selected_device"] is None
    with pytest.raises(RuntimeError, match="CUDA requested"):
        run_smoke(load_config(CONFIG, output_root=tmp_path, device="cuda"))
    (run_path,) = tmp_path.iterdir()
    assert json.loads((run_path / "run.json").read_text())["status"] == "failed"
    assert not (run_path / "result.json").exists()


def test_environment_cli_saves_report(tmp_path):
    output = tmp_path / "environment.json"
    assert main(["check-env", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["pytorch_version"]
    assert report["numpy_version"]
    assert "cuda_version" in report


def test_a100_gate_rejects_a_different_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name="NVIDIA RTX 4060", total_memory=8 * 1024**3, major=8, minor=9
        ),
    )
    report = inspect_environment("cuda", "A100")
    assert not report["validation"]["ok"]
    assert any("A100" in message for message in report["validation"]["errors"])


def test_unborn_git_is_recorded_without_inventing_a_commit(tmp_path):
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "untracked.txt").write_text("new")
    report = git_metadata(tmp_path)
    assert report["state"] == "unborn"
    assert report["commit"] is None
    assert report["dirty"] is True
