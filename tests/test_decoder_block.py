"""WP05.3 independent SwiGLU/block numerics, residual Jacobians and causal isolation."""

import io

import pytest
import torch
from torch.func import functional_call
from torch.nn import functional as F

from llm_scratch.data.collator import collate_tokenized, encode_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import SwiGLU, TokenEmbedding
from llm_scratch.model.transformer import DecoderBlock

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def config(**changes):
    return ModelConfig(
        **(
            dict(
                vocab_size=280,
                context_length=32,
                hidden_size=8,
                num_heads=2,
                num_layers=2,
                ffn_hidden_size=12,
            )
            | changes
        )
    )


def reference_ffn(x, parameters, prefix=""):
    gate = x @ parameters[prefix + "gate_proj.weight"].T
    up = x @ parameters[prefix + "up_proj.weight"].T
    return ((gate * gate.sigmoid()) * up) @ parameters[prefix + "down_proj.weight"].T


def reference_block(x, parameters, cfg, mask, positions):
    """Native RMSNorm/SDPA + complex RoPE and matrix-form SwiGLU; no model helpers."""
    x = x.masked_fill(~mask.unsqueeze(-1), 0)
    normalized = F.rms_norm(
        x, (cfg.hidden_size,), parameters["attention_norm.weight"], cfg.rms_norm_eps
    )
    heads = [
        (normalized @ parameters[f"attention.{name}_proj.weight"].T)
        .reshape(x.shape[0], x.shape[1], cfg.num_heads, cfg.head_dim)
        .transpose(1, 2)
        for name in ("q", "k", "v")
    ]
    frequencies = cfg.rope_theta ** (
        -torch.arange(0, cfg.head_dim, 2, device=x.device).double() / cfg.head_dim
    )
    angles = positions.double().unsqueeze(-1) * frequencies
    rotations = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)
    rotated = [
        torch.view_as_real(
            torch.view_as_complex(value.contiguous().reshape(*value.shape[:-1], -1, 2)) * rotations
        ).flatten(-2)
        for value in heads[:2]
    ]
    allowed = (
        torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).tril()[None, None]
        & mask[:, None, None, :]
        & mask[:, None, :, None]
    )
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        attended = F.scaled_dot_product_attention(*rotated, heads[2], attn_mask=allowed)
    attention_branch = (
        attended.transpose(1, 2).reshape_as(x) @ parameters["attention.out_proj.weight"].T
    )
    residual = (x + attention_branch).masked_fill(~mask.unsqueeze(-1), 0)
    normalized = F.rms_norm(
        residual, (cfg.hidden_size,), parameters["ffn_norm.weight"], cfg.rms_norm_eps
    )
    return (residual + reference_ffn(normalized, parameters, "ffn.")).masked_fill(
        ~mask.unsqueeze(-1), 0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_swiglu_reference_outputs_and_input_parameter_gradients(device):
    module = SwiGLU(8, 12).to(device).double()
    x = torch.randn(2, 3, 8, device=device, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    parameters = {
        name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()
    }
    actual, expected = module(x), reference_ffn(reference_x, parameters)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-10, atol=1e-12)
    for name, parameter in module.named_parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=1e-10, atol=1e-12)


def test_swiglu_finite_difference_input_and_all_parameters():
    module = SwiGLU(2, 3).double()
    x = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    parameters = tuple(p.detach().clone().requires_grad_() for p in module.parameters())
    names = tuple(name for name, _ in module.named_parameters())
    assert torch.autograd.gradcheck(
        lambda value, *weights: functional_call(
            module, dict(zip(names, weights, strict=True)), (value,)
        ),
        (x, *parameters),
    )


def test_swiglu_gate_zeroing_and_token_independence():
    module = SwiGLU(8, 12)
    x = torch.randn(2, 3, 8)
    before = module(x)
    x[1, 2] += 50
    torch.testing.assert_close(module(x)[0], before[0], rtol=0, atol=0)
    with torch.no_grad():
        module.gate_proj.weight.zero_()
    assert torch.count_nonzero(module(x)) == 0
    assert sum(p.numel() for p in module.parameters()) == 3 * 8 * 12


@pytest.mark.parametrize("device", DEVICES)
def test_full_pre_norm_block_independent_reference_and_all_gradients(device):
    cfg = config()
    module = DecoderBlock(cfg).to(device).double()
    with torch.no_grad():
        module.attention_norm.weight.copy_(torch.linspace(0.5, 1.5, 8, device=device))
        module.ffn_norm.weight.copy_(torch.linspace(0.7, 1.7, 8, device=device))
    x = torch.randn(2, 4, 8, device=device, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], device=device, dtype=torch.bool)
    positions = torch.tensor([[0, 2, 4, 6], [0, 1, 3, 5]], device=device)
    parameters = {
        name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()
    }
    actual = module(x, mask, positions)
    expected = reference_block(reference_x, parameters, cfg, mask, positions)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-10, atol=1e-12)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0, name
        torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_branches_exact_residual_identity_and_jacobian(device):
    module = DecoderBlock(config(dropout=0.5)).to(device).double().train()
    with torch.no_grad():
        module.attention.out_proj.weight.zero_()
        module.ffn.down_proj.weight.zero_()
    x = torch.randn(2, 4, 8, device=device, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], device=device, dtype=torch.bool)
    output = module(x, mask)
    torch.testing.assert_close(output, x.masked_fill(~mask.unsqueeze(-1), 0), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(x.grad, mask.unsqueeze(-1).expand_as(x).double(), rtol=0, atol=0)


def test_decoder_block_finite_difference_gradcheck():
    module = DecoderBlock(config(hidden_size=4, num_heads=1, ffn_hidden_size=6)).double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[0, 1, 1]])
    assert torch.autograd.gradcheck(lambda value: module(value, mask), (x,))


def test_zero_branch_full_input_jacobian_is_masked_identity():
    module = DecoderBlock(config(hidden_size=4, num_heads=1, ffn_hidden_size=6)).double()
    with torch.no_grad():
        module.attention.out_proj.weight.zero_()
        module.ffn.down_proj.weight.zero_()
    x = torch.randn(1, 3, 4, dtype=torch.float64)
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: module(value, mask), x, vectorize=True
    )
    expected = torch.diag(mask.unsqueeze(-1).expand_as(x).flatten().double())
    torch.testing.assert_close(jacobian.reshape(x.numel(), x.numel()), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth", [1, 2])
def test_block_stack_causal_prefix_and_zero_future_gradients(device, depth):
    blocks = torch.nn.ModuleList(
        [DecoderBlock(config()).to(device).double().eval() for _ in range(depth)]
    )

    def run(value, positions):
        for block in blocks:
            value = block(value, position_ids=positions)
        return value

    x = torch.randn(2, 6, 8, device=device, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(5, -1, -1, device=device)
    original = run(x, positions)
    changed = x.detach().clone()
    changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 100
    torch.testing.assert_close(run(changed, positions)[:, :3], original[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(
        run(x[:, :3], positions[:3]), original[:, :3], rtol=1e-10, atol=1e-12
    )
    original[:, 2].sum().backward()
    assert torch.count_nonzero(x.grad[:, 3:]) == 0
    assert x.grad[:, :3].abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mask_rows", [[[1, 1, 1, 0, 0]], [[0, 0, 1, 1, 1]], [[1, 0, 1, 0, 1]]])
def test_nonfinite_padding_is_cleared_before_norm_and_residual(device, mask_rows):
    module = DecoderBlock(config()).to(device).double()
    mask = torch.tensor(mask_rows, device=device, dtype=torch.bool)
    x = torch.randn(1, 5, 8, device=device, dtype=torch.float64)
    expected = module(x, mask)
    changed = x.clone().masked_fill(~mask.unsqueeze(-1), float("nan"))
    changed[~mask, 0] = float("inf")
    changed.requires_grad_()
    actual = module(changed, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[~mask]) == 0
    assert torch.isfinite(changed.grad).all() and torch.count_nonzero(changed.grad[~mask]) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_all_padding_has_zero_outputs_and_input_parameter_gradients(device):
    module = DecoderBlock(config()).to(device)
    x = torch.full((2, 4, 8), float("nan"), device=device, requires_grad=True)
    output = module(x, torch.zeros(2, 4, device=device, dtype=torch.long))
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) == 0
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_low_precision_ffn_block_shapes_and_finite_gradients(device, dtype):
    block = DecoderBlock(config()).to(device=device, dtype=dtype)
    ffn = SwiGLU(8, 12).to(device=device, dtype=dtype)
    x = torch.randn(2, 4, 8, device=device, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]], device=device)
    output = block(x, mask)
    direct = ffn(x)
    assert output.shape == direct.shape == x.shape
    assert output.dtype == direct.dtype == dtype
    assert output.device == direct.device == x.device
    assert torch.isfinite(output).all() and torch.count_nonzero(output[1]) == 0
    (output.float().square().sum() + direct.float().square().sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        torch.isfinite(p.grad).all() for p in list(block.parameters()) + list(ffn.parameters())
    )


@pytest.mark.parametrize("device", DEVICES)
def test_autocast_keeps_float32_residual_stream_and_finite_gradients(device):
    block = DecoderBlock(config()).to(device)
    x = torch.randn(1, 4, 8, device=device, requires_grad=True)
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device, dtype=dtype):
        output = block(x)
    assert output.dtype == torch.float32
    output.square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters())


def test_dropout_train_eval_and_state_dict_with_noncontiguous_input():
    module = DecoderBlock(config(dropout=0.5))
    x = torch.randn(2, 8, 5).transpose(1, 2)
    assert not x.is_contiguous()
    torch.manual_seed(42)
    first = module(x)
    torch.manual_seed(42)
    torch.testing.assert_close(first, module(x), rtol=0, atol=0)
    assert not torch.equal(first, module(x))
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    restored = DecoderBlock(config()).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    module.eval()
    torch.testing.assert_close(module(x), restored(x), rtol=0, atol=0)
    assert sum(p.numel() for p in module.parameters()) == 4 * 8**2 + 3 * 8 * 12 + 2 * 8


def test_wp04_ids_mask_embedding_and_block_integration():
    tokenizer = ByteBPETokenizer.train(["A story. Một câu chuyện."], vocab_size=280)
    batch = collate_tokenized(
        [encode_pretraining(tokenizer, s) for s in ("A story.", "🙂")], tokenizer
    )
    ids, mask = torch.tensor(batch["input_ids"]), torch.tensor(batch["attention_mask"])
    embedding = TokenEmbedding(len(tokenizer), 8)
    block = DecoderBlock(config(vocab_size=len(tokenizer), context_length=64))
    output = block(embedding(ids), mask)
    assert output.shape == (*ids.shape, 8)
    assert torch.count_nonzero(output[~mask.bool()]) == 0
    output.square().sum().backward()
    assert torch.isfinite(embedding.weight.grad).all()
    torch.testing.assert_close(
        block(embedding(ids), mask), block(embedding(ids), mask.bool()), rtol=0, atol=0
    )


@pytest.mark.parametrize("args", [(0, 8), (8, 0), (True, 8), (8, 1.5)])
def test_invalid_swiglu_construction(args):
    with pytest.raises(ValueError):
        SwiGLU(*args)


@pytest.mark.parametrize(
    "x", [torch.randn(2, 7), torch.tensor(1.0), torch.ones(2, 8, dtype=torch.long)]
)
def test_invalid_swiglu_input(x):
    with pytest.raises(ValueError):
        SwiGLU(8, 12)(x)


@pytest.mark.parametrize(
    "mask", [torch.ones(2, 3), torch.ones(2, 2, dtype=torch.long), torch.full((2, 3), -1)]
)
def test_block_uses_strict_shared_mask_validation(mask):
    with pytest.raises(ValueError, match="attention_mask"):
        DecoderBlock(config())(torch.randn(2, 3, 8), mask)


@pytest.mark.parametrize("shape", [(2, 3, 7), (0, 3, 8), (1, 0, 8), (1, 33, 8)])
def test_invalid_block_input_shapes(shape):
    with pytest.raises(ValueError):
        DecoderBlock(config())(torch.randn(shape))
