"""WP07.1 numerical accumulation, optimizer/scheduler, AMP, logs and exact continuation."""

import copy
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.optim import WarmupCosine, build_optimizer
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.seed import seed_everything

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def config(**changes):
    return TrainingConfig(**(dict(max_steps=6, warmup_steps=1, learning_rate=0.003) | changes))


def model(device="cpu", dropout=0.0):
    return CausalLM(
        ModelConfig(
            vocab_size=32,
            context_length=16,
            hidden_size=16,
            num_heads=4,
            num_layers=2,
            ffn_hidden_size=32,
            dropout=dropout,
        )
    ).to(device)


def batches():
    # 4 supervised pairs versus 1: a mean of microbatch means would be wrong.
    a = torch.tensor([[1, 3, 4, 5, 2]])
    b = torch.tensor([[1, 6, 7, 2, 0]])
    return [
        dict(input_ids=a, labels=a.clone(), attention_mask=torch.ones_like(a)),
        dict(
            input_ids=b,
            labels=torch.tensor([[-100, -100, -100, 2, -100]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
        ),
    ]


def new_engine(device="cpu", cfg=None, dropout=0.0):
    rng = seed_everything(42)
    return TrainingEngine(model(device, dropout), cfg or config(), generator=rng)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("device", DEVICES)
def test_token_weighted_accumulation_matches_independent_combined_batch(device):
    engine = new_engine(device)
    reference = copy.deepcopy(engine.model)
    opt = build_optimizer(reference, engine.config)
    scheduler = WarmupCosine(opt, engine.config)
    batch = {k: torch.cat([b[k] for b in batches()]).to(device) for k in batches()[0]}
    output = reference(**batch)
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        reference.parameters(), engine.config.max_grad_norm, error_if_nonfinite=True, foreach=False
    )
    opt.step()
    scheduler.step()
    report = engine.train_update(batches())
    assert report["loss_tokens"] == 5 and report["loss"] == pytest.approx(
        output.loss.item(), abs=1e-6
    )
    for a, b in zip(engine.model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert engine.scheduler.state_dict() == scheduler.state_dict()
    assert all(p.grad is None for p in engine.model.parameters())


def test_optimizer_unique_parameters_decay_groups_and_scheduler_endpoints():
    engine = new_engine(
        cfg=config(max_steps=6, warmup_steps=2, learning_rate=1.0, min_lr_ratio=0.1)
    )
    groups = engine.optimizer.param_groups
    params = [p for g in groups for p in g["params"]]
    assert len(params) == len({id(p) for p in engine.model.parameters()})
    assert sum(p is engine.model.token_embedding.weight for p in params) == 1
    assert all(p.ndim >= 2 for p in groups[0]["params"])
    assert all(p.ndim < 2 for p in groups[1]["params"]) and groups[1]["weight_decay"] == 0
    rates = []
    for _ in range(6):
        rates.append(groups[0]["lr"])
        engine.scheduler.step()
    assert rates == pytest.approx([0.5, 1, 1, 0.775, 0.325, 0.1])
    with pytest.raises(RuntimeError):
        engine.scheduler.step()
    single = new_engine(cfg=config(max_steps=1, warmup_steps=0))
    assert single.optimizer.param_groups[0]["lr"] == single.config.learning_rate


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_training_validation_and_logs_are_finite_and_loss_decreases(device, precision, capsys):
    engine = new_engine(device, cfg=config(precision=precision))
    initial = engine.evaluate(batches())["loss"]
    for _ in range(6):
        engine.train_update(batches())
    assert engine.evaluate(batches())["loss"] < initial
    assert engine.step == engine.scheduler.completed_updates == 6
    assert engine.tokens_seen == engine.tokens_optimized == 30
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len([r for r in logs if r["event"] == "train_update"]) == 6
    assert len([r for r in logs if r["event"] == "update_started"]) == 6
    assert all(p.dtype == torch.float32 for p in engine.model.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_exact_checkpoint_continuation_with_dropout_all_rngs_and_optimizer(tmp_path, device):
    engine = new_engine(device, dropout=0.2)
    for _ in range(2):
        engine.train_update(batches())
    path = tmp_path / "resume.pt"
    engine.save(path, data_state={"next_window": 2, "fixture": "authored-v1"})
    draws = (
        random.random(),
        np.random.random(),
        engine.generator.random(),
        torch.rand(3, device=device),
    )
    expected = []
    for _ in range(4):
        expected.append(engine.train_update(batches()))
    final = copy.deepcopy(engine.model.state_dict())
    opt = copy.deepcopy(engine.optimizer.state_dict())
    resumed = new_engine(device, dropout=0.2)
    data = resumed.load(path)
    assert data == {"next_window": 2, "fixture": "authored-v1"}
    actual = (
        random.random(),
        np.random.random(),
        resumed.generator.random(),
        torch.rand(3, device=device),
    )
    assert draws[:3] == actual[:3]
    torch.testing.assert_close(draws[3], actual[3], rtol=0, atol=0)
    for reference in expected:
        report = resumed.train_update(batches())
        assert report["loss"] == reference["loss"]
        assert report["learning_rates"] == reference["learning_rates"]
    assert_nested_equal(final, resumed.model.state_dict())
    assert_nested_equal(opt, resumed.optimizer.state_dict())
    assert engine.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert engine.scaler.state_dict() == resumed.scaler.state_dict()


def test_zero_supervision_short_window_validation_and_input_errors_do_not_advance():
    engine = new_engine()
    empty = {**batches()[0], "labels": torch.full((1, 5), -100)}
    with pytest.raises(ValueError, match="no supervised"):
        engine.train_update([empty])
    assert engine.step == 0 and engine.attempts == 0
    record = engine.train_update([empty, batches()[0]])
    assert record["loss_tokens"] == 4
    engine.train_update([batches()[0]])  # Short final accumulation window is allowed.
    assert engine.step == 2
    for invalid in ([], batches() * 2):
        with pytest.raises(ValueError):
            engine.train_update(invalid)
    with pytest.raises(ValueError):
        engine.evaluate([empty])
    assert engine.model.training
    with pytest.raises(ValueError):
        engine.train_update([{**batches()[0], "attention_mask": [[1] * 5]}])


def test_validation_is_weighted_and_preserves_gradients_mode_rng_and_progress():
    engine = new_engine(dropout=0.2)
    engine.model.train()
    before = torch.get_rng_state().clone()
    out = engine.evaluate(batches())
    assert out["loss_tokens"] == 5 and engine.step == 0 and engine.model.training
    assert torch.equal(before, torch.get_rng_state())
    engine.model.eval()
    batch = {k: torch.cat([b[k] for b in batches()]) for k in batches()[0]}
    with torch.no_grad():
        reference = engine.model(**batch)
    assert out["loss"] == pytest.approx(reference.loss.item(), abs=1e-6)
    assert all(p.grad is None for p in engine.parameters)


def test_checkpoint_rejects_incompatible_config_and_corrupt_counter_before_rng_change(tmp_path):
    engine = new_engine()
    engine.train_update(batches())
    path = tmp_path / "state.pt"
    engine.save(path, data_state={})
    other = new_engine(cfg=replace(engine.config, learning_rate=0.0001))
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Incompatible"):
        other.load(path)
    assert torch.equal(rng, torch.get_rng_state())
    payload = torch.load(path, weights_only=True)
    payload["state"]["step"] = 4
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="counters"):
        engine.load(tmp_path / "bad.pt")
    assert torch.equal(rng, torch.get_rng_state())


def test_nonfinite_failure_requires_reload_and_save_boundary_is_enforced(tmp_path):
    engine = new_engine()
    path = tmp_path / "initial.pt"
    engine.save(path, data_state={})
    handle = engine.parameters[0].register_hook(lambda grad: grad * float("nan"))
    try:
        with pytest.raises(FloatingPointError, match="gradients"):
            engine.train_update(batches())
    finally:
        handle.remove()
    assert engine.step == 0
    with pytest.raises(RuntimeError):
        engine.train_update(batches())
    with pytest.raises(RuntimeError):
        engine.save(tmp_path / "failed.pt", data_state={})
    engine.load(path)
    engine.train_update(batches())
    engine.parameters[0].grad = torch.zeros_like(engine.parameters[0])
    with pytest.raises(RuntimeError, match="boundaries"):
        engine.save(tmp_path / "mid.pt", data_state={})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 GradScaler behavior")
def test_fp16_overflow_skip_does_not_advance_scheduler_and_scaler_resumes(tmp_path):
    engine = new_engine("cuda", cfg=config(precision="float16"))
    weights = copy.deepcopy(engine.model.state_dict())
    initial_scale = engine.scaler.get_scale()
    handle = engine.parameters[0].register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        report = engine.train_update(batches())
    finally:
        handle.remove()
    assert report["skipped"] and engine.step == engine.scheduler.completed_updates == 0
    assert engine.attempts == 1 and engine.tokens_seen == 5 and engine.tokens_optimized == 0
    assert engine.scaler.get_scale() < initial_scale
    assert_nested_equal(weights, engine.model.state_dict())
    path = tmp_path / "overflow.pt"
    engine.save(path, data_state={"next_window": 1})
    expected = engine.train_update(batches())
    final = copy.deepcopy(engine.model.state_dict())
    restored = new_engine("cuda", cfg=config(precision="float16"))
    restored.load(path)
    actual = restored.train_update(batches())
    assert actual["skipped"] == expected["skipped"] and actual["amp_scale"] == expected["amp_scale"]
    assert_nested_equal(final, restored.model.state_dict())


@pytest.mark.parametrize(
    "changes",
    [
        {"learning_rate": 0},
        {"weight_decay": -1},
        {"beta1": 1},
        {"beta2": True},
        {"eps": 0},
        {"max_steps": True},
        {"gradient_accumulation_steps": 0},
        {"warmup_steps": 6},
        {"precision": "int8"},
        {"min_lr_ratio": 1.1},
        {"max_grad_norm": float("nan")},
    ],
)
def test_invalid_training_settings(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_fp16_cpu_and_half_master_weights_are_rejected():
    rng = seed_everything(42)
    with pytest.raises(ValueError, match="CUDA"):
        TrainingEngine(model(), config(precision="float16"), generator=rng)
    with pytest.raises(ValueError, match="master"):
        TrainingEngine(model().half(), config(), generator=rng)


def test_checkpoint_continuation_in_fresh_python_process(tmp_path):
    engine = new_engine(dropout=0.2)
    for _ in range(2):
        engine.train_update(batches())
    checkpoint = tmp_path / "restart.pt"
    final = tmp_path / "fresh_final.pt"
    engine.save(checkpoint, data_state={"next_window": 2})
    for _ in range(4):
        engine.train_update(batches())
    program = """
import sys, torch
from test_training_engine import new_engine, batches
engine = new_engine(dropout=.2)
assert engine.load(sys.argv[1]) == {'next_window': 2}
for _ in range(4):
    engine.train_update(batches())
torch.save({'model': engine.model.state_dict(), 'optimizer': engine.optimizer.state_dict(),
            'scheduler': engine.scheduler.state_dict(),
            'scaler': engine.scaler.state_dict()}, sys.argv[2])
print('fresh_process_resume_passed', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-u", "-c", program, str(checkpoint), str(final)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh_process_resume_passed" in result.stdout
    state = torch.load(final, weights_only=True)
    assert_nested_equal(engine.model.state_dict(), state["model"])
    assert_nested_equal(engine.optimizer.state_dict(), state["optimizer"])
    assert engine.scheduler.state_dict() == state["scheduler"]
    assert engine.scaler.state_dict() == state["scaler"]
