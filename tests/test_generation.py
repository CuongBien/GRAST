"""Generation contracts: hand-authored transitions, padding/EOS and real model baseline."""

import pytest
import torch

from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM, CausalLMOutput

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def tiny_config():
    return ModelConfig(
        vocab_size=12,
        context_length=16,
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        ffn_hidden_size=16,
    )


class TransitionLM(CausalLM):
    """Independent known transitions; padded readouts deliberately predict bad ID 9."""

    def __init__(self, device="cpu"):
        super().__init__(tiny_config())
        self.to(device).eval()
        self.seen = []

    def forward(self, input_ids, attention_mask=None):
        assert not torch.is_grad_enabled()
        self.seen.append((input_ids.clone(), attention_mask.clone()))
        transitions = torch.tensor([9, 2, 6, 4, 2, 2, 7, 8, 2, 2, 2, 2], device=input_ids.device)
        next_ids = transitions[input_ids].masked_fill(~attention_mask, 9)
        logits = torch.full((*input_ids.shape, 12), -torch.inf, device=input_ids.device)
        return CausalLMOutput(logits.scatter(-1, next_ids[..., None], 0.0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("do_sample", [False, True])
def test_batched_eos_stops_rows_keeps_eos_and_masks_tail(device, do_sample):
    net = TransitionLM(device)
    ids = torch.tensor([[1, 0, 0], [0, 3, 0]], device=device)
    mask = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device)
    saved_ids, saved_mask = ids.clone(), mask.clone()
    out = generate(
        net,
        ids,
        mask,
        config=GenerationConfig(max_new_tokens=5, do_sample=do_sample, eos_token_id=2),
    )
    assert out.generated_ids.tolist() == [[2, 0], [4, 2]]
    assert out.generated_mask.tolist() == [[True, False], [True, True]]
    assert out.generated_lengths.tolist() == [1, 2]
    assert out.finish_reasons == ("eos", "eos")
    assert len(net.seen) == 2
    torch.testing.assert_close(out.sequences[:, :3], ids)
    torch.testing.assert_close(out.attention_mask[:, :3], mask.bool())
    torch.testing.assert_close(ids, saved_ids)
    torch.testing.assert_close(mask, saved_mask)
    assert out.sequences.dtype == torch.long and out.attention_mask.dtype == torch.bool
    assert net.training is False and not out.sequences.requires_grad


def test_prompt_eos_is_context_and_eos_none_generates_to_exact_budget():
    net = TransitionLM()
    out = generate(net, torch.tensor([[2]]), config=GenerationConfig(max_new_tokens=3))
    assert out.generated_ids.tolist() == [[6, 7, 8]]
    assert out.finish_reasons == ("max_new_tokens",)
    assert [x.shape[1] for x, _ in net.seen] == [1, 2, 3]


def test_pad_equal_eos_and_finish_at_last_budget_step():
    net = TransitionLM()
    out = generate(
        net,
        torch.tensor([[1], [3]]),
        config=GenerationConfig(max_new_tokens=2, eos_token_id=2, pad_token_id=2),
    )
    assert out.generated_ids.tolist() == [[2, 2], [4, 2]]
    assert out.generated_lengths.tolist() == [1, 2]
    assert out.finish_reasons == ("eos", "eos")


def test_distinguishes_eos_from_budget_and_does_not_suppress_generated_pad():
    net = TransitionLM()
    out = generate(
        net,
        torch.tensor([[1], [2]]),
        config=GenerationConfig(max_new_tokens=1, eos_token_id=2, pad_token_id=6),
    )
    assert out.generated_ids.tolist() == [[2], [6]]
    assert out.generated_mask.all()
    assert out.finish_reasons == ("eos", "max_new_tokens")


def test_finished_rows_are_removed_before_sampling(monkeypatch):
    import importlib

    module = importlib.import_module("llm_scratch.inference.generate")
    original = module.sample_next_token
    batches = []

    def record(logits, config, **kwargs):
        batches.append(logits.shape[0])
        return original(logits, config, **kwargs)

    monkeypatch.setattr(module, "sample_next_token", record)
    generate(
        TransitionLM(),
        torch.tensor([[1], [3]]),
        config=GenerationConfig(max_new_tokens=4, do_sample=True, eos_token_id=2),
    )
    assert batches == [2, 1]


def test_zero_budget_full_context_has_no_forward_or_rng_effect():
    net = TransitionLM()
    state = torch.get_rng_state().clone()
    ids = torch.ones(2, 16, dtype=torch.int32)
    out = generate(net, ids, config=GenerationConfig(max_new_tokens=0))
    assert out.generated_ids.shape == (2, 0) and out.generated_lengths.tolist() == [0, 0]
    assert out.finish_reasons == ("max_new_tokens", "max_new_tokens")
    assert not net.seen
    assert torch.equal(state, torch.get_rng_state())


def test_context_boundary_rejects_upfront_even_if_eos_could_stop_early():
    net = TransitionLM()
    out = generate(
        net, torch.ones(1, 15, dtype=torch.long), config=GenerationConfig(max_new_tokens=1)
    )
    assert out.sequences.shape == (1, 16)
    net.seen.clear()
    with pytest.raises(ValueError, match="context"):
        generate(
            net,
            torch.ones(1, 15, dtype=torch.long),
            config=GenerationConfig(max_new_tokens=2, eos_token_id=2),
        )
    assert not net.seen


@pytest.mark.parametrize("device", DEVICES)
def test_real_model_greedy_matches_independent_full_prefix_loop_with_gaps(device):
    torch.manual_seed(2)
    net = CausalLM(tiny_config()).to(device).eval()
    ids = torch.tensor([[1, 3, 4, 0], [0, 1, 0, 5]], device=device)
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 0, 1]], device=device).bool()
    out = generate(net, ids, mask, config=GenerationConfig(max_new_tokens=4))
    expected, valid = ids.clone(), mask.clone()
    with torch.no_grad():
        for step in range(4):
            readout = (
                torch.tensor([2, 3], device=device)
                if step == 0
                else torch.full((2,), expected.shape[1] - 1, device=device)
            )
            logits = net(expected, valid).logits[torch.arange(2, device=device), readout]
            expected = torch.cat((expected, logits.argmax(-1)[:, None]), -1)
            valid = torch.cat((valid, torch.ones(2, 1, dtype=torch.bool, device=device)), -1)
    torch.testing.assert_close(out.sequences, expected)
    torch.testing.assert_close(out.attention_mask, valid)
    assert out.generated_lengths.tolist() == [4, 4]


@pytest.mark.parametrize("device", DEVICES)
def test_seeded_real_generation_replays_and_does_not_advance_global_rng(device):
    net = CausalLM(tiny_config()).to(device).eval()
    ids = torch.tensor([[1, 4], [1, 5]], device=device)
    config = GenerationConfig(max_new_tokens=6, do_sample=True, top_k=8, top_p=0.9)
    cpu_state = torch.get_rng_state().clone()
    gpu_state = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    a = generate(net, ids, config=config, seed=123)
    b = generate(net, ids, config=config, seed=123)
    c = generate(net, ids, config=config, seed=456)
    assert torch.equal(a.generated_ids, b.generated_ids)
    assert not torch.equal(a.generated_ids, c.generated_ids)
    assert torch.equal(cpu_state, torch.get_rng_state())
    if gpu_state is not None:
        assert torch.equal(gpu_state, torch.cuda.get_rng_state(device))


@pytest.mark.parametrize(
    "ids,mask",
    [
        (torch.ones(0, 3, dtype=torch.long), None),
        (torch.ones(1, 0, dtype=torch.long), None),
        (torch.ones(3, dtype=torch.long), None),
        (torch.ones(1, 3), None),
        (torch.tensor([[12]]), None),
        (torch.tensor([[-1]]), None),
        (torch.ones(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)),
        (torch.ones(1, 3, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)),
        (torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3)),
        (torch.ones(1, 3, dtype=torch.long), torch.tensor([[1, 2, 1]])),
    ],
)
def test_invalid_prompts_masks_rejected_before_forward(ids, mask):
    net = TransitionLM()
    with pytest.raises(ValueError):
        generate(net, ids, mask, config=GenerationConfig(max_new_tokens=0))
    assert not net.seen


def test_mode_seed_and_vocab_validation_leave_model_unchanged():
    net = TransitionLM()
    ids = torch.tensor([[1]])
    net.train()
    with pytest.raises(ValueError, match="eval"):
        generate(net, ids)
    assert net.training
    net.eval()
    net.blocks[0].ffn.train()
    with pytest.raises(ValueError, match="eval"):
        generate(net, ids)
    net.eval()
    for seed in (True, -1, 2**32, 1.5):
        with pytest.raises(ValueError, match="seed"):
            generate(net, ids, seed=seed)
    for config in (GenerationConfig(top_k=13), GenerationConfig(eos_token_id=12)):
        with pytest.raises(ValueError):
            generate(net, ids, config=config)
    assert not net.seen
