"""WP05 final gate: overfit, negative controls, replay and fresh-process repeatability."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from llm_scratch.evaluation.architecture import audit_model
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.architecture_acceptance import (
    fixture_texts,
    load_acceptance_config,
    run_acceptance,
)
from llm_scratch.training.checkpoint import load_checkpoint, save_checkpoint
from llm_scratch.training.overfit import overfit_batch
from llm_scratch.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/wp05_acceptance.toml"
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def setup(device="cpu", tied=True):
    seed_everything(42)
    model = CausalLM(
        ModelConfig(
            vocab_size=16,
            context_length=32,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            ffn_hidden_size=64,
            tie_embeddings=tied,
        )
    ).to(device)
    ids = torch.tensor([[1, 3, 4, 5, 2, 0], [1, 6, 7, 8, 9, 2]], device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], device=device)
    labels = ids.clone()
    labels[1, :3] = -100
    return model, {"input_ids": ids, "attention_mask": mask, "labels": labels}


def optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0, betas=(0.9, 0.95), foreach=False
    )


@pytest.mark.parametrize("device", DEVICES)
def test_full_model_overfits_fixed_batch_with_audits_before_and_after(device):
    model, batch = setup(device)
    original = {name: tensor.clone() for name, tensor in batch.items()}
    before = audit_model(model, batch)
    history = overfit_batch(model, batch, optimizer(model), steps=50)
    after = audit_model(model, batch)
    assert len(history) == 51 and history[-1]["step"] == 50
    assert history[-1]["loss"] <= 0.2 * history[0]["loss"]
    assert history[-1]["token_accuracy"] >= 0.95
    assert all(record["num_loss_tokens"] == 7 for record in history)
    assert all(record["gradient_norm_before_clip"] >= 0 for record in history[1:])
    assert all(before["checks"].values()) and all(after["checks"].values())
    for name in batch:
        torch.testing.assert_close(batch[name], original[name], rtol=0, atol=0)


def test_untied_model_passes_causal_padding_audit_without_accumulating_grads():
    model, batch = setup(tied=False)
    model.train()
    assert all(audit_model(model, batch)["checks"].values())
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_negative_control_future_leaking_attention_fails_audit(monkeypatch):
    model, batch = setup()

    def leaky(x, attention_mask=None, position_ids=None):
        return x.mean(dim=1, keepdim=True).expand_as(x)

    monkeypatch.setattr(model.blocks[0].attention, "forward", leaky)
    with pytest.raises(AssertionError):
        audit_model(model, batch)
    assert model.training


def test_negative_control_padding_logits_fail_audit(monkeypatch):
    model, batch = setup()
    forward = model.forward

    def bad_forward(input_ids, attention_mask=None, labels=None, position_ids=None):
        output = forward(input_ids, attention_mask, labels, position_ids)
        return replace(output, logits=output.logits + (~attention_mask.bool()).unsqueeze(-1))

    monkeypatch.setattr(model, "forward", bad_forward)
    with pytest.raises(AssertionError, match="padding logits"):
        audit_model(model, batch)


def test_optimizer_checkpoint_continuation_matches_uninterrupted_updates(tmp_path):
    model, batch = setup()
    opt = optimizer(model)
    generator = seed_everything(42)
    overfit_batch(model, batch, opt, steps=8)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        {"model": model.state_dict(), "optimizer": opt.state_dict()},
        generator=generator,
    )
    expected_history = overfit_batch(model, batch, opt, steps=8)
    restored = CausalLM(model.config)
    restored_opt = optimizer(restored)
    state = load_checkpoint(checkpoint, generator=generator)
    restored.load_state_dict(state["model"])
    restored_opt.load_state_dict(state["optimizer"])
    actual_history = overfit_batch(restored, batch, restored_opt, steps=8)
    assert actual_history == expected_history
    for name, value in model.state_dict().items():
        actual = restored.state_dict()[name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        else:
            assert actual == value


def test_no_supervision_rejects_before_weight_decay_or_optimizer_update():
    model, batch = setup()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.5)
    batch["labels"].fill_(-100)
    weights = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(ValueError, match="supervised"):
        overfit_batch(model, batch, opt, steps=1)
    assert not opt.state
    for actual, before in zip(model.parameters(), weights, strict=True):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


def test_nonfinite_gradient_rejected_before_optimizer_step():
    model, batch = setup()
    opt = optimizer(model)
    before = model.token_embedding.weight.detach().clone()
    handle = model.token_embedding.weight.register_hook(lambda gradient: gradient * float("nan"))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            overfit_batch(model, batch, opt, steps=1)
    finally:
        handle.remove()
    assert not opt.state
    torch.testing.assert_close(model.token_embedding.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [0, -1, True])
def test_invalid_step_count(steps):
    model, batch = setup()
    with pytest.raises(ValueError, match="steps"):
        overfit_batch(model, batch, optimizer(model), steps=steps)


def test_partial_optimizer_and_dropout_rejected():
    model, batch = setup()
    partial = torch.optim.AdamW(list(model.parameters())[:-1])
    with pytest.raises(ValueError, match="every unique"):
        overfit_batch(model, batch, partial, steps=1)
    dropout_model = CausalLM(ModelConfig.from_dict(model.config.to_dict() | {"dropout": 0.1}))
    with pytest.raises(ValueError, match="dropout"):
        overfit_batch(dropout_model, batch, optimizer(dropout_model), steps=1)


def test_invalid_audit_length_preserves_mode():
    model, batch = setup()
    short = {name: value[:, :1] for name, value in batch.items()}
    with pytest.raises(ValueError, match=">=3"):
        audit_model(model, short)
    assert model.training


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fresh_process_default_protocol_repeats_exactly_and_records_evidence(tmp_path):
    reports = []
    for _ in range(2):
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_architecture.py"),
                "--output-root",
                str(tmp_path / "runs"),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        run = Path(json.loads(process.stdout.splitlines()[-1])["run_directory"])
        report = read(run / "artifacts/acceptance_report.json")
        assert read(run / "run.json")["status"] == "completed"
        assert report["status"] == "passed" and report["checkpoint_round_trip_exact"]
        assert report["final"]["step"] == 200 and len(report["history"]) == 201
        assert report["loss_ratio"] <= 0.2 and report["final"]["token_accuracy"] >= 0.95
        assert report["parameter_counts"]["total"] == 100160
        assert (run / "logs/events.jsonl").stat().st_size > 0
        snapshot = read(run / "config.json")
        assert snapshot["batch_sha256"] == report["batch_sha256"]
        assert snapshot["acceptance"] == report["acceptance"]
        assert (run / "checkpoints/overfit.pt").is_file()
        reports.append(report)
    assert reports[0] == reports[1]


def test_failed_gate_has_failed_lifecycle_log_and_no_accepted_checkpoint(tmp_path):
    config = load_acceptance_config(CONFIG, output_root=tmp_path)
    config.data["optimization"]["steps"] = 1
    with pytest.raises(AssertionError, match="thresholds"):
        run_acceptance(config)
    run = next(tmp_path.iterdir())
    assert read(run / "run.json")["status"] == "failed"
    assert read(run / "artifacts/acceptance_report.json")["status"] == "failed"
    assert not (run / "checkpoints/overfit.pt").exists()
    assert "run_failed" in (run / "logs/events.jsonl").read_text()


@pytest.mark.parametrize(
    "profile,device", [("unknown", "cpu"), ("tiny", "cdua"), ("candidate", "cuda")]
)
def test_invalid_profile_device_or_missing_candidate_tokenizer(profile, device):
    with pytest.raises(ValueError):
        load_acceptance_config(CONFIG, profile=profile, device=device)


def test_fixture_unknown_stage_and_wrong_types_rejected():
    fixture = read(ROOT / "experiments/wp05_overfit_fixture.json")
    malformed = copy.deepcopy(fixture)
    malformed["records"][0]["stage"] = "test"
    with pytest.raises(ValueError, match="Unknown"):
        fixture_texts(malformed)
    malformed = copy.deepcopy(fixture)
    malformed["records"] = [42]
    with pytest.raises(ValueError, match="objects"):
        fixture_texts(malformed)
    fixture["records"] = []
    with pytest.raises(ValueError, match="nonempty"):
        fixture_texts(fixture)
