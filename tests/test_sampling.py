"""Distribution-level sampling tests with analytical supports and explicit RNGs."""

import math

import pytest
import torch

from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.sampling import filter_logits, sample_next_token

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_temperature_distribution_preserves_inputs_and_precision(device, dtype):
    logits = torch.tensor([[0.0, 2.0, -2.0, -torch.inf]], device=device, dtype=dtype)
    saved = logits.clone()
    result = filter_logits(logits, GenerationConfig(temperature=2))
    expected = torch.tensor([[0.0, 1.0, -1.0, -torch.inf]], device=device).softmax(-1)
    torch.testing.assert_close(result.softmax(-1).float(), expected)
    torch.testing.assert_close(logits, saved)
    assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert not result.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_exact_top_k_tie_order_and_combined_nucleus_after_renormalization(device):
    tied = torch.tensor([[1.0, 2.0, 2.0, 2.0, 0.0]], device=device)
    kept = torch.isfinite(filter_logits(tied, GenerationConfig(top_k=2)))
    assert kept.tolist() == [[False, True, True, False, False]]
    # Token IDs deliberately differ from descending probability order.
    logits = torch.tensor([[0.1, 0.4, 0.3, 0.2]], device=device, dtype=torch.float64).log()
    # After top-k=2: .4/.7 > .55, so nucleus keeps token 1 alone.
    result = filter_logits(logits, GenerationConfig(top_k=2, top_p=0.55))
    assert torch.isfinite(result).tolist() == [[False, True, False, False]]
    assert torch.isfinite(filter_logits(logits, GenerationConfig(top_k=0))).all()
    torch.testing.assert_close(
        filter_logits(logits, GenerationConfig(top_k=4)),
        filter_logits(logits, GenerationConfig()),
    )


@pytest.mark.parametrize(
    "p,expected", [(0.01, [0]), (0.5, [0]), (0.6, [0, 1]), (1.0, [0, 1, 2, 3])]
)
def test_nucleus_minimal_prefix_includes_crossing_and_handles_exact_boundary(p, expected):
    logits = torch.tensor([[0.5, 0.25, 0.125, 0.125]], dtype=torch.float64).log()
    support = torch.where(torch.isfinite(filter_logits(logits, GenerationConfig(top_p=p)))[0])[0]
    assert support.tolist() == expected


@pytest.mark.parametrize("temperature", [5e-324, 1e-300, 1e-10, 1e300])
def test_extreme_finite_temperature_never_removes_all_candidates(temperature):
    logits = torch.tensor([[3.0, 1.0, -torch.inf]])
    result = filter_logits(logits, GenerationConfig(temperature=temperature))
    probs = result.softmax(-1)
    assert torch.isfinite(probs).all() and probs[0, 2] == 0
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))
    if temperature < 1:
        assert probs.tolist() == [[1.0, 0.0, 0.0]]


def test_large_float64_scores_with_large_temperature_do_not_overflow_subtraction():
    result = filter_logits(
        torch.tensor([[1e308, -1e308]], dtype=torch.float64), GenerationConfig(temperature=1e308)
    )
    torch.testing.assert_close(result, torch.tensor([[0.0, -2.0]], dtype=torch.float64))


def test_tiny_nucleus_probability_keeps_a_finite_candidate():
    result = filter_logits(torch.tensor([[0.0, 2.0, 1.0]]), GenerationConfig(top_p=5e-324))
    assert torch.isfinite(result).tolist() == [[False, True, False]]


@pytest.mark.parametrize("temperature", [5e-324, 1e300])
def test_extreme_temperature_does_not_change_top_k_rank(temperature):
    result = filter_logits(
        torch.tensor([[0.0, 2.0, 1.0]]), GenerationConfig(temperature=temperature, top_k=1)
    )
    assert torch.isfinite(result).tolist() == [[False, True, False]]


@pytest.mark.parametrize("device", DEVICES)
def test_sampling_replays_advances_private_rng_and_leaves_global_rng_unchanged(device):
    logits = torch.zeros(128, 10, device=device)
    config = GenerationConfig(do_sample=True, top_k=4, top_p=0.8)
    a = torch.Generator(device=device).manual_seed(42)
    b = torch.Generator(device=device).manual_seed(42)
    cpu_state = torch.get_rng_state().clone()
    gpu_state = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    first = sample_next_token(logits, config, generator=a)
    again = sample_next_token(logits, config, generator=b)
    second = sample_next_token(logits, config, generator=a)
    assert torch.equal(first, again) and not torch.equal(first, second)
    assert first.min() >= 0 and first.max() < 4
    assert torch.equal(cpu_state, torch.get_rng_state())
    if gpu_state is not None:
        assert torch.equal(gpu_state, torch.cuda.get_rng_state(device))


@pytest.mark.parametrize("device", DEVICES)
def test_sampled_frequencies_follow_temperature_distribution(device):
    logits = (
        torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64, device=device).log().expand(12000, -1)
    )
    generator = torch.Generator(device=device).manual_seed(51)
    samples = sample_next_token(logits, GenerationConfig(do_sample=True), generator=generator)
    frequencies = torch.bincount(samples, minlength=3).double() / samples.numel()
    torch.testing.assert_close(
        frequencies.cpu(), torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64), rtol=0, atol=0.02
    )


def test_greedy_ties_raw_argmax_and_no_rng_consumption():
    generator = torch.Generator().manual_seed(7)
    state = generator.get_state().clone()
    logits = torch.tensor([[2.0, 2.0, -torch.inf], [-2.0, 0.0, 3.0]])
    result = sample_next_token(
        logits, GenerationConfig(top_k=1, top_p=0.01, temperature=1e10), generator=generator
    )
    assert result.tolist() == [0, 2]
    assert torch.equal(state, generator.get_state())


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 0},
        {"temperature": -1},
        {"temperature": math.nan},
        {"temperature": math.inf},
        {"temperature": True},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": False},
        {"top_k": -1},
        {"top_k": True},
        {"top_k": 2.0},
        {"do_sample": 1},
        {"max_new_tokens": -1},
        {"max_new_tokens": True},
        {"eos_token_id": True},
        {"pad_token_id": -1},
        {"eos_token_id": -1},
    ],
)
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        GenerationConfig(**changes)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(2, 3, dtype=torch.long),
        torch.zeros(0, 3),
        torch.zeros(2, 0),
        torch.zeros(3),
        torch.tensor([[float("nan"), 0]]),
        torch.tensor([[float("inf"), 0]]),
        torch.tensor([[-torch.inf, -torch.inf]]),
    ],
)
def test_invalid_logits_rejected_in_greedy_and_sampling(logits):
    for do_sample in (False, True):
        with pytest.raises(ValueError):
            sample_next_token(
                logits, GenerationConfig(do_sample=do_sample), generator=torch.Generator()
            )


def test_vocab_unknown_keys_and_missing_generator_are_rejected():
    for config in (
        GenerationConfig(top_k=4),
        GenerationConfig(eos_token_id=3),
        GenerationConfig(pad_token_id=3),
    ):
        with pytest.raises(ValueError):
            sample_next_token(torch.zeros(1, 3), config)
    with pytest.raises(ValueError, match="generator"):
        sample_next_token(torch.zeros(1, 3), GenerationConfig(do_sample=True))
    with pytest.raises(ValueError):
        GenerationConfig.from_dict({"k_values": [1, 2]})
    with pytest.raises(ValueError):
        GenerationConfig.from_dict([])
    config = GenerationConfig(top_k=2)
    assert GenerationConfig.from_dict(config.to_dict()) == config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for wrong-device RNG")
def test_sampling_rejects_wrong_generator_device():
    with pytest.raises(ValueError, match="device"):
        sample_next_token(
            torch.zeros(1, 5, device="cuda"),
            GenerationConfig(do_sample=True),
            generator=torch.Generator(),
        )
