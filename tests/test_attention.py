"""WP05.2 numerical reference, causal/padding isolation and low-precision autograd."""

import io

import pytest
import torch
from torch.nn import functional as F

from llm_scratch.data.collator import collate_tokenized, encode_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.attention import CausalSelfAttention
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import TokenEmbedding

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def config(**changes):
    fields = dict(
        vocab_size=280,
        context_length=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_hidden_size=16,
    )
    return ModelConfig(**(fields | changes))


def complex_rope(x, position_ids, theta):
    """Independent adjacent-pair reference using complex-number multiplication."""
    pairs = torch.view_as_complex(x.contiguous().reshape(*x.shape[:-1], -1, 2))
    frequencies = theta ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device).double() / x.shape[-1]
    )
    angles = position_ids.double().unsqueeze(-1) * frequencies
    rotations = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)
    return torch.view_as_real(pairs * rotations).flatten(-2)


def torch_reference(x, parameters, cfg, mask, positions=None):
    """Native math SDPA + complex RoPE; no helper from the implementation is used."""
    batch, length, hidden = x.shape
    clean = x.masked_fill(~mask.unsqueeze(-1), 0)
    projections = [
        (clean @ parameters[f"{name}_proj.weight"].T)
        .reshape(batch, length, cfg.num_heads, cfg.head_dim)
        .transpose(1, 2)
        for name in ("q", "k", "v")
    ]
    positions = (
        torch.arange(length, device=x.device).expand(batch, -1) if positions is None else positions
    )
    q, k = [complex_rope(value, positions, cfg.rope_theta) for value in projections[:2]]
    allowed = (
        torch.ones(length, length, device=x.device, dtype=torch.bool).tril()[None, None]
        & mask[:, None, None, :]
        & mask[:, None, :, None]
    )
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        attended = F.scaled_dot_product_attention(q, k, projections[2], attn_mask=allowed)
    joined = attended.transpose(1, 2).reshape(batch, length, hidden)
    return (joined @ parameters["out_proj.weight"].T).masked_fill(~mask.unsqueeze(-1), 0)


@pytest.mark.parametrize("device", DEVICES)
def test_reference_output_input_and_all_projection_gradients(device):
    torch.manual_seed(42)
    cfg = config()
    module = CausalSelfAttention(cfg).to(device).double()
    x = torch.randn(2, 5, 8, device=device, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], device=device, dtype=torch.bool)
    positions = torch.tensor([[0, 2, 4, 6, 8], [0, 1, 2, 3, 4]], device=device)
    parameters = {
        name: value.detach().clone().requires_grad_() for name, value in module.named_parameters()
    }
    actual = module(x, mask, positions)
    expected = torch_reference(reference_x, parameters, cfg, mask, positions)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-10, atol=1e-12)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=1e-10, atol=1e-12)


def test_hand_computed_uniform_causal_attention_averages_only_visible_values():
    module = CausalSelfAttention(config(hidden_size=4, num_heads=1)).double()
    with torch.no_grad():
        module.q_proj.weight.zero_()
        module.k_proj.weight.zero_()
        module.v_proj.weight.copy_(torch.eye(4))
        module.out_proj.weight.copy_(torch.eye(4))
    x = torch.arange(16, dtype=torch.float64).view(1, 4, 4)
    actual = module(x, torch.tensor([[1, 0, 1, 1]]))
    expected = torch.stack((x[0, 0], torch.zeros(4), x[0, [0, 2]].mean(0), x[0, [0, 2, 3]].mean(0)))
    torch.testing.assert_close(actual[0], expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_future_perturbation_prefix_equivalence_and_zero_future_gradients(
    device, explicit_positions
):
    module = CausalSelfAttention(config()).to(device).double().eval()
    x = torch.randn(2, 6, 8, device=device, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(5, -1, -1, device=device) if explicit_positions else None
    original = module(x, position_ids=positions)
    changed = x.detach().clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 1000
    perturbed = module(changed, position_ids=positions)
    torch.testing.assert_close(original[:, :3], perturbed[:, :3], rtol=0, atol=0)
    prefix_positions = positions[:3] if positions is not None else None
    prefix = module(x[:, :3], position_ids=prefix_positions)
    torch.testing.assert_close(original[:, :3], prefix, rtol=1e-10, atol=1e-12)
    original[:, 2].sum().backward()
    assert torch.count_nonzero(x.grad[:, 3:]) == 0
    assert x.grad[:, :3].abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mask_rows", [[[1, 1, 1, 0, 0]], [[0, 0, 1, 1, 1]], [[1, 0, 1, 0, 1]]])
def test_padding_inputs_are_isolated_even_when_nonfinite(device, mask_rows):
    module = CausalSelfAttention(config()).to(device).double()
    mask = torch.tensor(mask_rows, device=device, dtype=torch.bool)
    x = torch.randn(1, 5, 8, device=device, dtype=torch.float64, requires_grad=True)
    expected = module(x, mask)
    changed = x.detach().clone()
    changed[~mask] = float("nan")
    changed[~mask, 0] = float("inf")
    changed.requires_grad_()
    actual = module(changed, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[~mask]) == 0
    actual.sum().backward()
    assert torch.isfinite(changed.grad).all()
    assert torch.count_nonzero(changed.grad[~mask]) == 0
    assert changed.grad[mask].abs().sum() > 0


@pytest.mark.parametrize("left_padding", [False, True])
def test_padded_valid_outputs_match_unpadded_sequence(left_padding):
    module = CausalSelfAttention(config()).double().eval()
    x = torch.randn(1, 3, 8, dtype=torch.float64)
    padding = torch.randn(1, 2, 8, dtype=torch.float64)
    combined = torch.cat([padding, x] if left_padding else [x, padding], dim=1)
    mask = torch.tensor([[0, 0, 1, 1, 1] if left_padding else [1, 1, 1, 0, 0]], dtype=torch.bool)
    torch.testing.assert_close(
        module(combined, mask)[mask], module(x).flatten(0, 1), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("device", DEVICES)
def test_fully_padded_example_has_zero_outputs_and_all_gradients(device):
    module = CausalSelfAttention(config()).to(device)
    x = torch.randn(2, 4, 8, device=device, requires_grad=True)
    output = module(x, torch.zeros(2, 4, device=device, dtype=torch.long))
    assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0
    output.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) == 0
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == 0


def test_finite_difference_input_gradcheck_with_mixed_padding():
    module = CausalSelfAttention(config(hidden_size=4, num_heads=1)).double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda value: module(value, mask), (x,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_device_shapes_and_finite_gradients(device, dtype):
    module = CausalSelfAttention(config()).to(device=device, dtype=dtype)
    x = torch.randn(2, 5, 8, device=device, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], device=device)
    output = module(x, mask)
    assert output.dtype == dtype and output.device == x.device and output.shape == x.shape
    assert torch.isfinite(output).all() and torch.count_nonzero(output[1]) == 0
    output.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_fp16_large_scores_and_autocast_do_not_overflow(device):
    module = CausalSelfAttention(config()).to(device).eval()
    with torch.no_grad():
        for projection in (module.q_proj, module.k_proj):
            projection.weight.copy_(torch.eye(8, device=device) * 10)
        module.v_proj.weight.copy_(torch.eye(8, device=device))
        module.out_proj.weight.copy_(torch.eye(8, device=device))
    x = torch.full((1, 4, 8), 100.0, device=device, requires_grad=True)
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device, dtype=dtype):
        output = module(x)
    torch.testing.assert_close(output.float(), x, rtol=0.01, atol=0.01)
    output.float().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    half_module = CausalSelfAttention(config()).to(device=device, dtype=torch.float16)
    half_module.load_state_dict(module.state_dict())
    assert torch.isfinite(half_module(x.detach().half())).all()


def test_dropout_respects_train_eval_seed_and_padding():
    module = CausalSelfAttention(config(dropout=0.5))
    x = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]])
    torch.manual_seed(42)
    first = module(x, mask)
    torch.manual_seed(42)
    repeated = module(x, mask)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(first, module(x, mask))
    assert torch.count_nonzero(first[1]) == 0
    module.eval()
    baseline = CausalSelfAttention(config()).eval()
    baseline.load_state_dict(module.state_dict())
    torch.testing.assert_close(module(x, mask), baseline(x, mask), rtol=0, atol=0)
    torch.testing.assert_close(module(x, mask), module(x, mask), rtol=0, atol=0)


def test_parameter_count_state_dict_and_noncontiguous_input():
    module = CausalSelfAttention(config())
    assert sum(parameter.numel() for parameter in module.parameters()) == 4 * 8**2
    assert all(
        projection.bias is None
        for projection in (module.q_proj, module.k_proj, module.v_proj, module.out_proj)
    )
    x = torch.randn(2, 8, 5).transpose(1, 2)
    assert not x.is_contiguous()
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    restored = CausalSelfAttention(config())
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored(x), module(x), rtol=0, atol=0)


def test_wp04_embedding_attention_integration_and_mask_conventions():
    tokenizer = ByteBPETokenizer.train(["A story. Một câu chuyện."], vocab_size=280)
    batch = collate_tokenized(
        [encode_pretraining(tokenizer, text) for text in ("A story.", "🙂")], tokenizer
    )
    ids, mask = torch.tensor(batch["input_ids"]), torch.tensor(batch["attention_mask"])
    embedding = TokenEmbedding(len(tokenizer), 8)
    module = CausalSelfAttention(config(context_length=64, vocab_size=len(tokenizer)))
    output = module(embedding(ids), mask)
    assert output.shape == (*ids.shape, 8)
    assert torch.count_nonzero(output[~mask.bool()]) == 0
    output.square().sum().backward()
    assert torch.isfinite(embedding.weight.grad).all()
    torch.testing.assert_close(module(embedding(ids), mask), module(embedding(ids), mask.bool()))
    full = torch.ones_like(mask)
    torch.testing.assert_close(module(embedding(ids)), module(embedding(ids), full), rtol=0, atol=0)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, 3),
        torch.ones(2, 2, dtype=torch.long),
        torch.full((2, 3), 2),
        torch.full((2, 3), -1),
    ],
)
def test_bad_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="attention_mask"):
        CausalSelfAttention(config())(torch.randn(2, 3, 8), mask)


@pytest.mark.parametrize("shape", [(2, 3, 7), (2, 8), (0, 3, 8), (1, 0, 8), (1, 17, 8)])
def test_bad_input_shapes_are_rejected(shape):
    with pytest.raises(ValueError):
        CausalSelfAttention(config())(torch.randn(shape))


def test_nonfloating_input_is_rejected():
    with pytest.raises(ValueError):
        CausalSelfAttention(config())(torch.ones(1, 3, 8, dtype=torch.long))
