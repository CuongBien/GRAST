"""Actual cached generation versus full-prefix reference, not teacher-forced replay alone."""

import weakref

import pytest
import torch

from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM, CausalLMOutput
from llm_scratch.research.generation_equivalence import (
    check_generation_equivalence,
    trace_generation,
)
from llm_scratch.utils.seed import seed_everything

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def model(device="cpu", dtype=torch.float32, **changes):
    seed_everything(42)
    cfg = dict(
        vocab_size=32,
        context_length=24,
        hidden_size=16,
        num_heads=4,
        num_layers=2,
        ffn_hidden_size=32,
    )
    return CausalLM(ModelConfig(**(cfg | changes))).to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["full", "left_right", "gaps"])
def test_actual_greedy_logits_and_tokens_match_across_precision_padding(device, dtype, layout):
    net = model(device, dtype)
    ids = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8]], device=device)
    mask = torch.ones_like(ids)
    if layout == "left_right":
        mask[0, -2:] = 0
        mask[1, :2] = 0
    elif layout == "gaps":
        mask[0, 1] = 0
        mask[1, [0, 2]] = 0
    report = check_generation_equivalence(net, ids, mask, config=GenerationConfig(max_new_tokens=6))
    assert report["generated_lengths"] == [6, 6]
    assert report["cached_input_lengths"] == [4, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "settings",
    [
        dict(temperature=0.7),
        dict(temperature=1.3, top_k=7),
        dict(top_p=0.65),
        dict(temperature=0.8, top_k=12, top_p=0.9),
    ],
)
def test_controlled_sampling_uses_same_rng_schedule_and_matches_ids(device, settings):
    net = model(device)
    ids = torch.tensor([[1, 5, 0], [0, 1, 7]], device=device)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]], device=device)
    config = GenerationConfig(max_new_tokens=8, do_sample=True, **settings)
    report = check_generation_equivalence(net, ids, mask, config=config, seed=123)
    assert report["checks"]["token_ids_exact"]
    a = generate(net, ids, mask, config=config, seed=123, use_cache=True)
    b = generate(net, ids, mask, config=config, seed=123, use_cache=True)
    torch.testing.assert_close(a.sequences, b.sequences)


class CachedTransitions(CausalLM):
    """Real model/cache operations, independent known output distribution for EOS cases."""

    def __init__(self, device="cpu"):
        super().__init__(
            ModelConfig(
                vocab_size=12,
                context_length=12,
                hidden_size=8,
                num_heads=2,
                num_layers=1,
                ffn_hidden_size=16,
            )
        )
        self.to(device).eval()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        output = super().forward(input_ids, attention_mask, **kwargs)
        transition = torch.tensor([9, 2, 6, 4, 2, 2, 7, 8, 2, 2, 2, 2], device=input_ids.device)
        chosen = transition[input_ids].masked_fill(~attention_mask.bool(), 9)
        logits = torch.full_like(output.logits, -torch.inf).scatter(-1, chosen[..., None], 0.0)
        return CausalLMOutput(logits=logits, cache=output.cache)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("pad", [0, 2])
def test_mixed_eos_finished_rows_and_pad_equal_eos(device, sampled, pad):
    ids = torch.tensor([[1, 0, 0], [0, 3, 0]], device=device)
    mask = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device)
    policy = GenerationConfig(max_new_tokens=6, do_sample=sampled, eos_token_id=2, pad_token_id=pad)
    report = check_generation_equivalence(CachedTransitions(device), ids, mask, config=policy)
    assert report["generated_ids"] == [[2], [4, 2]]
    assert report["generated_mask"] == [[True, False], [True, True]]
    assert report["finish_reasons"] == ["eos", "eos"]
    assert [step["active_rows"] for step in report["steps"]] == [2, 1]


def test_all_rows_eos_on_prefill_has_no_unnecessary_decode():
    net = CachedTransitions()
    report = check_generation_equivalence(
        net,
        torch.tensor([[1], [4]]),
        torch.ones(2, 1, dtype=torch.long),
        config=GenerationConfig(max_new_tokens=4, eos_token_id=2),
    )
    assert report["cached_input_lengths"] == [1]
    assert report["generated_ids"] == [[2], [2]]


@pytest.mark.parametrize("device", DEVICES)
def test_real_model_mixed_eos_uses_only_active_readout_logits(device):
    net = model(device)
    ids = torch.tensor([[1, 3, 4], [1, 6, 8]], device=device)
    with torch.no_grad():
        first = net(ids).logits[:, -1].argmax(-1)
    assert first[0] != first[1]  # Fixed fixture has distinct first-token predictions.
    policy = GenerationConfig(max_new_tokens=4, eos_token_id=int(first[0]))
    report = check_generation_equivalence(net, ids, torch.ones_like(ids), config=policy)
    assert report["generated_lengths"][0] == 1
    assert report["generated_lengths"][1] > 1
    assert report["steps"][1]["active_rows"] == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [0, 1, 5])
def test_context_and_rope_offset_exact_boundaries(device, budget):
    net = model(device, context_length=12)
    ids = torch.tensor([[1, 3, 4]], device=device)
    report = check_generation_equivalence(
        net,
        ids,
        torch.ones_like(ids),
        config=GenerationConfig(max_new_tokens=budget),
        position_offset=12 - 3 - budget,
    )
    assert report["generated_lengths"] == [budget]
    assert report["cached_input_lengths"] == ([3] + [1] * (budget - 1) if budget else [])
    for use_cache in (False, True):
        with pytest.raises(ValueError, match="context"):
            generate(
                net,
                ids,
                config=GenerationConfig(max_new_tokens=budget),
                position_offset=12 - 3 - budget + 1,
                use_cache=use_cache,
            )


def test_projection_lengths_cache_masks_positions_and_storage_stay_correct():
    net = model(context_length=16)
    ids = torch.tensor([[1, 3, 0], [0, 1, 4]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    lengths, snapshots = [], []

    def projected(module, args):
        lengths.append(args[0].shape[1])

    def cache_snapshot(module, args, kwargs, out):
        cache = out.cache
        snapshots.append(
            (
                cache.sequence_length,
                cache.attention_mask.clone(),
                cache.position_ids.clone(),
                cache.get_layer(0)[0].untyped_storage().data_ptr(),
                cache.get_layer(0)[0][:, :, :3].clone(),
            )
        )

    a = net.blocks[0].attention.k_proj.register_forward_pre_hook(projected)
    b = net.register_forward_hook(cache_snapshot, with_kwargs=True)
    try:
        out = generate(
            net,
            ids,
            mask,
            config=GenerationConfig(max_new_tokens=4),
            use_cache=True,
            position_offset=2,
        )
    finally:
        a.remove()
        b.remove()
    assert lengths == [3, 1, 1, 1]
    assert [s[0] for s in snapshots] == [3, 4, 5, 6]
    assert len({s[3] for s in snapshots}) == 1
    for length, valid, positions, _, prefix in snapshots:
        torch.testing.assert_close(valid, out.attention_mask[:, :length])
        torch.testing.assert_close(positions, torch.arange(2, 2 + length).expand(2, -1))
        torch.testing.assert_close(prefix, snapshots[0][4], rtol=0, atol=0)
    assert out.sequences.shape[1] == 7  # Last sampled token is not forwarded into private cache.


def test_finished_row_decode_mask_is_retained_without_reactivating_padding():
    net = CachedTransitions()
    snapshots = []

    def capture(module, args, kwargs, output):
        snapshots.append(output.cache.attention_mask.clone())

    handle = net.register_forward_hook(capture, with_kwargs=True)
    try:
        out = generate(
            net,
            torch.tensor([[1], [2]]),
            config=GenerationConfig(max_new_tokens=4, eos_token_id=2),
            use_cache=True,
        )
    finally:
        handle.remove()
    assert out.generated_mask.tolist() == [[True, False, False, False], [True, True, True, True]]
    assert snapshots[-1].tolist() == [[True, True, False, False], [True, True, True, True]]


def test_zero_budget_allocates_no_cache_and_does_no_forward(monkeypatch):
    net = model()

    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected model/cache work")

    monkeypatch.setattr(net, "forward", unexpected)
    monkeypatch.setattr(net, "new_cache", unexpected)
    out = generate(
        net,
        torch.ones(2, 24, dtype=torch.long),
        config=GenerationConfig(max_new_tokens=0),
        use_cache=True,
    )
    assert out.generated_ids.shape == (2, 0)


def test_private_cache_is_released_and_consecutive_requests_are_isolated(monkeypatch):
    net = model()
    original = net.new_cache
    references = []

    def create(*args, **kwargs):
        cache = original(*args, **kwargs)
        references.append(weakref.ref(cache))
        return cache

    monkeypatch.setattr(net, "new_cache", create)
    ids = torch.tensor([[1, 3]])
    config = GenerationConfig(max_new_tokens=4, do_sample=True)
    a = generate(net, ids, config=config, use_cache=True)
    generate(net, torch.tensor([[1, 8, 9]]), config=config, use_cache=True)
    b = generate(net, ids, config=config, use_cache=True)
    torch.testing.assert_close(a.sequences, b.sequences)
    assert len(references) == 3 and all(ref() is None for ref in references)


@pytest.mark.parametrize("device", DEVICES)
def test_cached_autocast_equivalence_and_global_rng_is_unchanged(device):
    net = model(device)
    ids = torch.tensor([[1, 3, 4]], device=device)
    policy = GenerationConfig(max_new_tokens=5, do_sample=True, top_k=12, top_p=0.9)
    cpu = torch.get_rng_state().clone()
    gpu = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    with torch.autocast(
        device_type=device, dtype=torch.bfloat16 if device == "cpu" else torch.float16
    ):
        report = check_generation_equivalence(net, ids, torch.ones_like(ids), config=policy)
    assert report["status"] == "passed"
    assert torch.equal(cpu, torch.get_rng_state())
    if gpu is not None:
        assert torch.equal(gpu, torch.cuda.get_rng_state(device))


def test_trace_hook_removed_on_failure_and_next_request_works(monkeypatch):
    net = model()
    head = net.lm_head.forward

    def fail(*args, **kwargs):
        raise RuntimeError("head failure")

    monkeypatch.setattr(net.lm_head, "forward", fail)
    before = len(net._forward_hooks)
    with pytest.raises(RuntimeError, match="head failure"):
        trace_generation(
            net,
            torch.tensor([[1, 3]]),
            torch.ones(1, 2, dtype=torch.long),
            config=GenerationConfig(max_new_tokens=3),
            seed=42,
            use_cache=True,
        )
    assert len(net._forward_hooks) == before
    monkeypatch.setattr(net.lm_head, "forward", head)
    out = generate(
        net, torch.tensor([[1, 3]]), config=GenerationConfig(max_new_tokens=3), use_cache=True
    )
    assert out.generated_lengths.tolist() == [3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_cache": 1},
        {"position_offset": True},
        {"position_offset": -1},
        {"position_offset": 1.5},
    ],
)
def test_invalid_cache_offset_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        generate(model(), torch.tensor([[1]]), config=GenerationConfig(max_new_tokens=1), **kwargs)
