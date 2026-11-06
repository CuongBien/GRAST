"""WP05.1 numerical references, autograd, dtype/device and tokenizer integration."""

import io
import math
from pathlib import Path

import pytest
import torch
from torch.func import functional_call

from llm_scratch.data.collator import collate_tokenized, encode_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import RMSNorm, RotaryEmbedding, TokenEmbedding
from llm_scratch.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def candidate_config():
    return load_config(ROOT / "configs/experiments/local_smoke.toml").data["model"]


def test_model_config_preserves_wp02_and_serializes_new_defaults():
    data = candidate_config()
    config = ModelConfig.from_dict(data)
    assert config.head_dim == 64
    assert all(config.to_dict()[name] == value for name, value in data.items())
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 10000.0
    assert ModelConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vocab_size", 0),
        ("context_length", -1),
        ("hidden_size", True),
        ("num_layers", 1.5),
        ("num_heads", 7),
        ("ffn_hidden_size", 0),
        ("dropout", 1),
        ("dropout", -0.1),
        ("rope_theta", 1),
        ("rope_theta", float("inf")),
        ("rms_norm_eps", 0),
        ("rms_norm_eps", float("nan")),
        ("tie_embeddings", 1),
    ],
)
def test_invalid_model_config(key, value):
    with pytest.raises(ValueError):
        ModelConfig.from_dict({**candidate_config(), key: value})


def test_model_config_rejects_odd_rotary_dimension_unknown_and_missing_fields():
    data = candidate_config()
    with pytest.raises(ValueError, match="even"):
        ModelConfig.from_dict({**data, "hidden_size": 24, "num_heads": 8})
    with pytest.raises(ValueError, match="Unknown"):
        ModelConfig.from_dict({**data, "typo": 1})
    with pytest.raises(ValueError, match="Incomplete"):
        ModelConfig.from_dict({})


@pytest.mark.parametrize("device", DEVICES)
def test_embedding_lookup_parameter_count_and_repeated_token_gradients(device):
    embedding = TokenEmbedding(12, 6).to(device)
    ids = torch.tensor([[0, 4, 4], [2, 0, 11]], device=device)
    output = embedding(ids)
    assert output.shape == (2, 3, 6)
    torch.testing.assert_close(output, embedding.weight[ids])
    assert sum(p.numel() for p in embedding.parameters()) == 72
    output.sum().backward()
    counts = torch.bincount(ids.flatten(), minlength=12).to(output.dtype)
    torch.testing.assert_close(embedding.weight.grad, counts[:, None].expand(-1, 6))
    assert embedding.weight.grad[0].sum() > 0  # PAD row is deliberately not frozen.


@pytest.mark.parametrize("ids", [torch.ones(2, 3), torch.ones(3, dtype=torch.long)])
def test_embedding_requires_batched_integer_ids(ids):
    with pytest.raises(ValueError, match="input_ids"):
        TokenEmbedding(10, 4)(ids)


@pytest.mark.parametrize("token_id", [-1, 10])
def test_embedding_rejects_out_of_vocabulary_ids(token_id):
    with pytest.raises((IndexError, RuntimeError)):
        TokenEmbedding(10, 4)(torch.tensor([[token_id]]))


def test_embedding_uses_wp04_adapter_ids_and_saves_exact_weight():
    tokenizer = ByteBPETokenizer.train(["A story. Một câu chuyện."], vocab_size=280)
    examples = [encode_pretraining(tokenizer, text) for text in ("A story.", "🙂")]
    batch = collate_tokenized(examples, tokenizer)
    embedding = TokenEmbedding(len(tokenizer), 8)
    ids = torch.tensor(batch["input_ids"])
    output = embedding(ids)
    assert output.shape == (*ids.shape, 8)
    stream = io.BytesIO()
    torch.save(embedding.state_dict(), stream)
    stream.seek(0)
    restored = TokenEmbedding(len(tokenizer), 8)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored(ids), output, rtol=0, atol=0)


def test_rmsnorm_matches_torch_reference_outputs_and_gradients():
    torch.manual_seed(42)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    norm = RMSNorm(8, eps=1e-6).double()
    reference = torch.nn.RMSNorm(8, eps=1e-6).double()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
        reference.weight.copy_(norm.weight)
    output, expected = norm(x), reference(reference_x)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(norm.weight.grad, reference.weight.grad, rtol=1e-11, atol=1e-11)


def test_rmsnorm_finite_difference_gradcheck_input_and_scale():
    norm = RMSNorm(4).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    weight = norm.weight.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value, scale: functional_call(norm, {"weight": scale}, (value,)), (x, weight)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rmsnorm_low_precision_avoids_square_overflow_and_preserves_dtype(device, dtype):
    norm = RMSNorm(4).to(device=device, dtype=dtype)
    x = torch.tensor(
        [[[1000.0, -2000.0, 3000.0, -4000.0], [0, 0, 0, 0]]], device=device, dtype=dtype
    )
    x.requires_grad_()
    output = norm(x)
    reference = x.float() / (x.float().square().mean(-1, keepdim=True) + norm.eps).sqrt()
    torch.testing.assert_close(output, reference.to(dtype))
    assert output.dtype == dtype and output.device == x.device
    assert torch.isfinite(output).all()
    output.float().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(norm.weight.grad).all()


def test_rmsnorm_operates_per_token_without_centering():
    norm = RMSNorm(4)
    x = torch.ones(2, 3, 4)
    before = norm(x)
    x[1, 2] *= 3
    after = norm(x)
    assert before.mean() > 0.99
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert sum(p.numel() for p in norm.parameters()) == 4


def matrix_rope_reference(x, positions, theta):
    """Independent 2x2 rotation-matrix reference in float64 on CPU."""
    output = torch.empty_like(x)
    for batch in range(x.shape[0]):
        for head in range(x.shape[1]):
            for time in range(x.shape[2]):
                for pair in range(x.shape[3] // 2):
                    angle = int(positions[batch, time]) * theta ** (-2 * pair / x.shape[3])
                    rotation = torch.tensor(
                        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
                        dtype=x.dtype,
                    )
                    output[batch, head, time, 2 * pair : 2 * pair + 2] = (
                        rotation @ x[batch, head, time, 2 * pair : 2 * pair + 2]
                    )
    return output


def test_rope_matches_independent_rotation_matrices_and_preserves_norm():
    x = torch.randn(2, 3, 4, 8, dtype=torch.float64)
    positions = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    rope = RotaryEmbedding(8, 16)
    output = rope(x, positions)
    torch.testing.assert_close(output, matrix_rope_reference(x, positions, 10000.0))
    torch.testing.assert_close(output.square().sum(-1), x.square().sum(-1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(output[0, :, 0], x[0, :, 0], rtol=0, atol=0)
    assert list(rope.parameters()) == []


def test_rope_relative_dot_product_and_chunk_positions():
    rope = RotaryEmbedding(8, 32)
    q = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    k = torch.randn_like(q)
    first = (rope(q, torch.tensor([2])) * rope(k, torch.tensor([7]))).sum(-1)
    shifted = (rope(q, torch.tensor([8])) * rope(k, torch.tensor([13]))).sum(-1)
    torch.testing.assert_close(first, shifted, rtol=1e-12, atol=1e-12)
    x = torch.randn(2, 2, 9, 8)
    full = rope(x)
    pieces = [rope(x[:, :, :4]), rope(x[:, :, 4:], torch.arange(4, 9))]
    torch.testing.assert_close(torch.cat(pieces, dim=2), full, rtol=0, atol=0)


def test_rope_finite_difference_gradcheck():
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(RotaryEmbedding(4, 8), (x,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rope_dtype_device_gradients_and_high_positions(device, dtype):
    rope = RotaryEmbedding(8, 4096).to(device=device, dtype=dtype)
    x = torch.randn(2, 3, 4, 8, device=device, dtype=dtype, requires_grad=True)
    positions = torch.arange(2049, 2053, device=device)
    output = rope(x, positions)
    reference = rope(x.detach().double(), positions).to(dtype)
    torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
    assert output.shape == x.shape and output.dtype == dtype and output.device == x.device
    output.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "positions",
    [torch.tensor([-1, 0]), torch.tensor([0, 8]), torch.tensor([0.0, 1.0]), torch.tensor([0])],
)
def test_rope_rejects_invalid_positions(positions):
    with pytest.raises(ValueError, match="position_ids"):
        RotaryEmbedding(4, 8)(torch.zeros(1, 2, 2, 4), positions)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TokenEmbedding(0, 8),
        lambda: TokenEmbedding(10, True),
        lambda: RMSNorm(0),
        lambda: RMSNorm(4, eps=0),
        lambda: RMSNorm(4, eps=float("nan")),
        lambda: RotaryEmbedding(3, 8),
        lambda: RotaryEmbedding(4, 0),
        lambda: RotaryEmbedding(4, 8, theta=1),
    ],
)
def test_invalid_layer_construction(factory):
    with pytest.raises(ValueError):
        factory()


def test_invalid_layer_input_shapes_and_context():
    with pytest.raises(ValueError):
        RMSNorm(4)(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        RMSNorm(4)(torch.zeros(2, 4, dtype=torch.long))
    rope = RotaryEmbedding(4, 8)
    with pytest.raises(ValueError):
        rope(torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="context"):
        rope(torch.zeros(1, 1, 9, 4))
