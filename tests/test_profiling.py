"""Protocol semantics, real tiny timing, CUDA accounting, no performance thresholds."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from llm_scratch.evaluation.latency_vram import distribution, summarize_samples
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.inference.profile_config import ProfileConfig, load_profile_config
from llm_scratch.inference.profiling import (
    copy_for_inference_context,
    measure_request,
    profile_matrix,
)
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.utils.seed import seed_everything

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
ROOT = Path(__file__).resolve().parents[1]


def model(device="cpu"):
    seed_everything(42)
    return (
        CausalLM(
            ModelConfig(
                vocab_size=32,
                context_length=24,
                hidden_size=16,
                num_layers=2,
                num_heads=4,
                ffn_hidden_size=32,
            )
        )
        .to(device)
        .eval()
    )


def policy(**changes):
    values = dict(
        prompt_lengths=[4, 8],
        new_tokens=3,
        batch_size=1,
        warmup=1,
        repeats=2,
        context_length=24,
        precision="float32",
        device="cpu",
    )
    return ProfileConfig(**(values | changes))


def test_quantiles_and_token_throughput_denominators_are_distinct():
    samples = [
        dict(ttft_ms=10.0, next_token_latencies_ms=[10.0, 10.0], total_ms=40.0, vram_bytes=None),
        dict(ttft_ms=20.0, next_token_latencies_ms=[20.0, 20.0], total_ms=70.0, vram_bytes=None),
    ]
    result = summarize_samples(samples, batch_size=2, new_tokens=3)
    assert result["ttft_ms"]["p50"] == 15
    assert result["next_token_latency_ms"]["count"] == 4
    assert result["decode_tokens_per_second"]["p50"] == 150
    assert result["overall_tokens_per_second"]["max"] == 150
    assert result["overall_tokens_per_second"]["min"] == pytest.approx(6000 / 70)
    assert result["vram_bytes"] is None
    for invalid in ([], [0], [-1], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            distribution(invalid)
    with pytest.raises(ValueError):
        summarize_samples(samples, 2, 4)


def test_cpu_and_cuda_samples_cannot_be_mixed():
    sample = dict(ttft_ms=1.0, next_token_latencies_ms=[1.0], total_ms=3.0, vram_bytes=None)
    with pytest.raises(ValueError, match="mix"):
        summarize_samples([sample, sample | {"vram_bytes": {"peak_allocated": 42}}], 1, 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_lengths": []},
        {"prompt_lengths": [8, 4]},
        {"prompt_lengths": [4, 4]},
        {"prompt_lengths": [True]},
        {"new_tokens": 1},
        {"new_tokens": True},
        {"warmup": 0},
        {"repeats": 0},
        {"batch_size": 0},
        {"context_length": 9},
        {"precision": "int8"},
        {"device": "auto"},
    ],
)
def test_invalid_protocol_rejected(changes):
    with pytest.raises(ValueError):
        policy(**changes)


def test_context_copy_preserves_weights_alias_metadata_and_source():
    source = model()
    source.tokenizer_fingerprint = "a" * 64
    original = source.get_extra_state()
    extended = copy_for_inference_context(source, 32)
    assert source.config.context_length == 24
    assert extended.config.context_length == 32
    assert source.get_extra_state() == original
    assert extended.tokenizer_fingerprint == source.tokenizer_fingerprint
    assert extended.lm_head.weight is extended.token_embedding.weight
    assert extended.config == replace(source.config, context_length=32)
    for left, right in zip(source.parameters(), extended.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    ids = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(source(ids).logits, extended(ids).logits, rtol=0, atol=0)
        assert extended(torch.ones(1, 30, dtype=torch.long)).logits.shape == (1, 30, 32)
    with pytest.raises(ValueError, match="configuration mismatch"):
        source.load_state_dict(extended.state_dict())
    assert source.config.context_length == 24
    for value in (True, 23):
        with pytest.raises(ValueError):
            copy_for_inference_context(source, value)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cached", [False, True])
def test_observer_gets_completed_tokens_without_changing_generation(device, cached):
    net = model(device)
    ids = torch.tensor([[1, 3], [1, 4]], device=device)
    seen = []

    def observe(step, tokens, valid):
        assert not torch.is_grad_enabled()
        seen.append((step, tokens.clone(), valid.clone()))

    cfg = GenerationConfig(max_new_tokens=3)
    a = generate(net, ids, config=cfg, use_cache=cached, on_token=observe)
    b = generate(net, ids, config=cfg, use_cache=cached)
    torch.testing.assert_close(a.sequences, b.sequences)
    assert [s[0] for s in seen] == [0, 1, 2]
    for step, tokens, valid in seen:
        torch.testing.assert_close(tokens, a.generated_ids[:, step])
        torch.testing.assert_close(valid, a.generated_mask[:, step])
    with pytest.raises(ValueError, match="on_token"):
        generate(net, ids, config=cfg, on_token=1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cached", [False, True])
def test_real_measurement_has_exact_budget_monotonic_samples_and_memory(device, cached):
    net = model(device)
    ids = torch.tensor([[1, 3, 4]], device=device)
    sample = measure_request(
        net, ids, torch.ones_like(ids), new_tokens=3, use_cache=cached, seed=42
    )
    ticks = sample["token_ready_offsets_ns"]
    assert 0 < ticks[0] < ticks[1] < ticks[2] <= sample["request_elapsed_ns"]
    assert sample["ttft_ms"] == ticks[0] / 1e6
    assert len(sample["generated_ids"][0]) == 3
    assert len(sample["next_token_latencies_ms"]) == 2
    if device == "cpu":
        assert sample["vram_bytes"] is None
    else:
        mem = sample["vram_bytes"]
        assert mem["peak_allocated"] >= max(mem["baseline_allocated"], mem["end_allocated"])
        assert mem["peak_reserved"] >= mem["peak_allocated"]
        assert mem["peak_allocated_above_baseline"] >= 0
    with pytest.raises(ValueError):
        measure_request(net, ids, torch.ones_like(ids), new_tokens=1, use_cache=cached, seed=42)


def test_matrix_repeats_orders_conditions_and_uses_matching_prompt_ids():
    net = model()
    records = profile_matrix(net, policy(), seed=42)
    assert [(r["prompt_length"], r["mode"]) for r in records] == [
        (4, "uncached"),
        (4, "cached"),
        (8, "cached"),
        (8, "uncached"),
    ]
    for a, b in (records[:2], records[2:]):
        assert a["prompt_ids_sha256"] == b["prompt_ids_sha256"]
        assert a["samples"][0]["generated_ids"] == b["samples"][1]["generated_ids"]
    for record in records:
        assert len(record["samples"]) == 2
        assert record["summary"]["next_token_latency_ms"]["count"] == 4
        assert record["summary"]["vram_bytes"] is None
        if record["mode"] == "cached":
            payload = record["cache_payload_formula_bytes"]
            assert payload["capacity"] == record["prompt_length"] + 3
            assert payload["used_columns"] == record["prompt_length"] + 2
    with pytest.raises(ValueError, match="context"):
        profile_matrix(net, policy(context_length=32), seed=42)


def test_default_and_smoke_protocols_keep_baseline_context_explicit():
    full = load_profile_config(ROOT / "configs/experiments/wp06_profiling.toml")
    assert full.data["source_model"]["context_length"] == 1024
    assert full.data["profiling"]["context_length"] == 2064
    assert full.data["profiling"]["prompt_lengths"] == [128, 512, 1024, 2048]
    smoke = load_profile_config(
        ROOT / "configs/experiments/wp06_profiling_smoke.toml", device="cpu"
    )
    assert smoke.data["profiling"]["new_tokens"] == 4


def test_scripts_profile_preserves_stdlib_cprofile_and_torch_optimizer():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cProfile, torch; cProfile.run('x=1'); "
            "torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.1); "
            "print('stdlib_profile_ok')",
        ],
        cwd=ROOT / "scripts",
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "stdlib_profile_ok" in result.stdout
