"""Raw-model likelihood matches independent teacher forcing and preserves generation."""

import pytest
import torch

from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM, CausalLMOutput

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def model(device="cpu"):
    torch.manual_seed(17)
    return (
        CausalLM(
            ModelConfig(
                vocab_size=12,
                context_length=16,
                hidden_size=8,
                num_heads=2,
                num_layers=1,
                ffn_hidden_size=16,
            )
        )
        .to(device)
        .eval()
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("offset", [0, 2])
def test_raw_scores_match_teacher_forcing_cached_uncached_and_preserve_rng(device, offset):
    net = model(device)
    ids = torch.tensor([[0, 1, 3], [1, 4, 5]], device=device)
    mask = ids != 0
    policy = GenerationConfig(max_new_tokens=4, do_sample=True, temperature=0.7, top_k=4, top_p=0.9)
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    plain = generate(net, ids, mask, config=policy, seed=21, position_offset=offset)
    assert plain.generated_logprobs is None
    outputs = []
    for cache in (False, True):
        out = generate(
            net,
            ids,
            mask,
            config=policy,
            seed=21,
            position_offset=offset,
            use_cache=cache,
            return_logprobs=True,
        )
        assert torch.equal(out.generated_ids, plain.generated_ids)
        assert torch.equal(out.generated_mask, plain.generated_mask)
        with torch.no_grad():
            positions = torch.arange(offset, offset + out.sequences.shape[1], device=device)
            logits = net(out.sequences, out.attention_mask, position_ids=positions).logits
            expected = (
                logits[:, 2:-1]
                .float()
                .log_softmax(-1)
                .gather(-1, out.generated_ids[..., None])
                .squeeze(-1)
            )
        torch.testing.assert_close(out.generated_logprobs, expected, atol=2e-6, rtol=2e-6)
        assert out.generated_logprobs.dtype == torch.float32
        assert not out.generated_logprobs.requires_grad
        outputs.append(out)
    torch.testing.assert_close(
        outputs[0].generated_logprobs, outputs[1].generated_logprobs, atol=2e-6, rtol=2e-6
    )
    assert torch.equal(torch.get_rng_state(), cpu_state)
    if cuda_state is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_state)


class KnownLM(CausalLM):
    def __init__(self):
        super().__init__(model().config)
        self.eval()

    def forward(self, input_ids, attention_mask=None):
        # Row 1 emits PAD as a real sampled token, then EOS. Row 0 immediately EOS.
        target = torch.where(input_ids == 3, 0, 2)
        logits = torch.zeros((*input_ids.shape, 12))
        logits.scatter_(-1, target[..., None], 2.0)
        return CausalLMOutput(logits)


def test_raw_not_filtered_scores_real_pad_eos_and_masked_tails():
    out = generate(
        KnownLM(),
        torch.tensor([[1], [3]]),
        return_logprobs=True,
        config=GenerationConfig(
            max_new_tokens=3, do_sample=True, top_k=1, temperature=0.1, eos_token_id=2
        ),
    )
    assert out.generated_ids.tolist() == [[2, 0], [0, 2]]
    assert out.generated_mask.tolist() == [[True, False], [True, True]]
    # Filtered top-1 sampling logprob would be zero; raw-model logprob is negative.
    expected = torch.tensor([2.0] + [0.0] * 11).log_softmax(0)[0]
    torch.testing.assert_close(out.generated_logprobs[out.generated_mask], expected.expand(3))
    assert out.generated_logprobs[0, 1] == 0
    assert out.finish_reasons == ("eos", "eos")


def test_zero_budget_and_flag_validation():
    net = model()
    out = generate(
        net, torch.tensor([[1]]), return_logprobs=True, config=GenerationConfig(max_new_tokens=0)
    )
    assert out.generated_logprobs.shape == (1, 0)
    with pytest.raises(ValueError, match="boolean"):
        generate(net, torch.tensor([[1]]), return_logprobs=1)
