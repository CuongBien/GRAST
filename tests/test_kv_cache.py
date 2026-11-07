"""WP06.1 storage/transaction contract; model logits/generation are later phases."""

import pytest
import torch

from llm_scratch.model.cache import KVCache, KVCacheStep
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import RotaryEmbedding

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def config(**changes):
    return ModelConfig(
        **(
            dict(
                vocab_size=16,
                context_length=16,
                hidden_size=8,
                num_layers=3,
                num_heads=2,
                ffn_hidden_size=12,
            )
            | changes
        )
    )


def tensors(cache, length, layer, start=0):
    shape = (cache.batch_size, cache.config.num_heads, length, cache.config.head_dim)
    key = (
        torch.arange(
            torch.tensor(shape).prod().item(), device=cache.device, dtype=cache.dtype
        ).view(shape)
        + layer * 100
        + start
    )
    return key, key + 0.5


def append_all(cache, length):
    with cache.begin_step(length) as step:
        for layer in range(cache.config.num_layers):
            step.update(layer, *tensors(cache, length, layer, step.start))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_prefill_and_multiple_appends_match_independent_concatenation(device, dtype):
    cache = KVCache(config(), 2, device=device, dtype=dtype)
    references = [[] for _ in range(3)]
    for length in (4, 1, 2, 1):
        old_length = cache.sequence_length
        with cache.begin_step(length) as step:
            assert step.start == old_length and step.end == old_length + length
            assert cache.sequence_length == old_length
            for layer in (2, 0, 1):
                key, value = tensors(cache, length, layer, old_length)
                references[layer].append((key.clone(), value.clone()))
                full_key, full_value = step.update(layer, key, value)
                torch.testing.assert_close(
                    full_key,
                    torch.cat([pair[0] for pair in references[layer]], dim=2),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    full_value,
                    torch.cat([pair[1] for pair in references[layer]], dim=2),
                    rtol=0,
                    atol=0,
                )
                assert cache.get_layer(layer)[0].shape[2] == old_length
                assert not full_key.requires_grad and full_key.grad_fn is None
        assert cache.sequence_length == old_length + length
    for layer in range(3):
        key, value = cache.get_layer(layer)
        assert key.dtype == dtype and key.device == torch.device(
            device if device == "cpu" else "cuda:0"
        )
        torch.testing.assert_close(
            key, torch.cat([pair[0] for pair in references[layer]], dim=2), rtol=0, atol=0
        )
        torch.testing.assert_close(
            value, torch.cat([pair[1] for pair in references[layer]], dim=2), rtol=0, atol=0
        )
    assert cache.attention_mask.all()
    torch.testing.assert_close(
        cache.position_ids, torch.arange(8, device=cache.device).expand(2, -1)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_rotated_keys_unrotated_values_and_metadata_round_trip(device):
    cache = KVCache(config(), 2, device=device)
    rope = RotaryEmbedding(4, 16)
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]], device=device)
    positions = torch.tensor([[0, 3, 4], [8, 9, 0]], device=device, dtype=torch.int32)
    expected = []
    with cache.begin_step(3, attention_mask=mask, position_ids=positions) as step:
        for layer in range(3):
            key, value = tensors(cache, 3, layer)
            rotated = rope(key, step.position_ids)
            expected.append((rotated.clone(), value.clone()))
            step.update(layer, rotated, value)
        torch.testing.assert_close(step.query_mask, mask.bool())
        torch.testing.assert_close(step.position_ids, positions.long())
    # Source mutations must not mutate cache metadata/storage.
    mask.zero_()
    positions.zero_()
    assert cache.attention_mask.tolist() == [[False, True, True], [True, True, False]]
    assert cache.position_ids.tolist() == [[0, 3, 4], [8, 9, 0]]
    for layer in range(3):
        key, value = cache.get_layer(layer)
        torch.testing.assert_close(key, expected[layer][0], rtol=0, atol=0)
        torch.testing.assert_close(value, expected[layer][1], rtol=0, atol=0)
    with cache.begin_step(1, position_ids=torch.tensor([12], device=device)) as step:
        assert step.position_ids.tolist() == [[12], [12]]
        assert step.attention_mask.shape == (2, 4)
        for layer in range(3):
            step.update(layer, *tensors(cache, 1, layer))


@pytest.mark.parametrize("device", DEVICES)
def test_incomplete_and_exception_appends_preserve_committed_prefix_and_retry(device):
    cache = KVCache(config(), 1, device=device)
    append_all(cache, 3)
    previous = [tuple(tensor.clone() for tensor in cache.get_layer(layer)) for layer in range(3)]
    previous_positions = cache.position_ids.clone()
    with pytest.raises(RuntimeError, match="incomplete"):
        with cache.begin_step(2) as step:
            step.update(0, *tensors(cache, 2, 0, 300))
    assert cache.sequence_length == 3
    with pytest.raises(ValueError, match="forward failed"):
        with cache.begin_step(2) as step:
            for layer in range(3):
                step.update(layer, *tensors(cache, 2, layer, 700))
            raise ValueError("forward failed after all layers before commit")
    for layer in range(3):
        for actual, expected in zip(cache.get_layer(layer), previous[layer], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache.position_ids, previous_positions, rtol=0, atol=0)
    append_all(cache, 1)
    assert cache.sequence_length == 4


def test_failed_key_value_validation_does_not_mark_layer_or_expose_tail():
    cache = KVCache(config(), 1)
    with cache.begin_step(2) as step:
        key, value = tensors(cache, 2, 0)
        with pytest.raises(ValueError):
            step.update(0, key, value.double())
        assert cache.sequence_length == 0 and cache.get_layer(0)[0].numel() == 0
        step.update(0, key, value)
        key.fill_(999)
        value.fill_(999)
        for layer in (1, 2):
            step.update(layer, *tensors(cache, 2, layer))
    assert not torch.any(cache.get_layer(0)[0] == 999)


def test_noncontiguous_projection_views_copy_exactly():
    cache = KVCache(config(), 2)
    source = torch.randn(2, 3, 2, 4).transpose(1, 2)
    assert not source.is_contiguous()
    with cache.begin_step(3) as step:
        for layer in range(3):
            step.update(layer, source, -source)
    torch.testing.assert_close(cache.get_layer(0)[0], source, rtol=0, atol=0)
    torch.testing.assert_close(cache.get_layer(0)[1], -source, rtol=0, atol=0)


def test_explicit_rope_positions_are_bounded_by_model_context_not_storage_capacity():
    cache = KVCache(config(), 1, max_length=4)
    with cache.begin_step(2, position_ids=torch.tensor([14, 15])) as step:
        for layer in range(3):
            step.update(layer, *tensors(cache, 2, layer))
    assert cache.position_ids.tolist() == [[14, 15]]
    # Default remains physical sequence offset, not a guess based on custom positions.
    append_all(cache, 1)
    assert cache.position_ids.tolist() == [[14, 15, 2]]


def test_duplicate_layer_closed_transaction_and_manual_lifecycle():
    cache = KVCache(config(), 1)
    step = cache.begin_step(1)
    step.update(0, *tensors(cache, 1, 0))
    with pytest.raises(RuntimeError, match="already written"):
        step.update(0, *tensors(cache, 1, 0))
    with pytest.raises(RuntimeError, match="incomplete"):
        step.commit()
    assert cache.sequence_length == 0
    for layer in (1, 2):
        step.update(layer, *tensors(cache, 1, layer))
    step.commit()
    assert cache.sequence_length == 1
    for action in (
        step.commit,
        step.abort,
        lambda: step.update(0, *tensors(cache, 1, 0)),
        lambda: step.position_ids,
    ):
        with pytest.raises(RuntimeError, match="no longer active"):
            action()
    with cache.begin_step(1) as cancelled:
        cancelled.abort()
    assert cache.sequence_length == 1
    forged = KVCacheStep(cache, 1, 2)
    with pytest.raises(RuntimeError, match="no longer active"):
        forged.update(0, *tensors(cache, 1, 0))


def test_nested_append_and_reset_during_append_rejected_without_discarding_data():
    cache = KVCache(config(), 1)
    with cache.begin_step(1) as step:
        with pytest.raises(RuntimeError, match="already active"):
            cache.begin_step(1)
        with pytest.raises(RuntimeError, match="active append"):
            cache.reset()
        for layer in range(3):
            step.update(layer, *tensors(cache, 1, layer))
    assert cache.sequence_length == 1


@pytest.mark.parametrize("device", DEVICES)
def test_capacity_limit_and_reset_reuse_same_storage(device):
    cache = KVCache(config(), 2, max_length=4, device=device)
    pointers = [cache.get_layer(layer)[0].untyped_storage().data_ptr() for layer in range(3)]
    append_all(cache, 4)
    memory = cache.memory_bytes()
    assert memory["kv_used_bytes"] == memory["kv_capacity_bytes"]
    with pytest.raises(ValueError, match="capacity/context"):
        cache.begin_step(1)
    assert cache.sequence_length == 4
    cache.reset()
    assert cache.sequence_length == 0 and cache.memory_bytes()["kv_used_bytes"] == 0
    assert cache.memory_bytes()["tensor_capacity_bytes"] == memory["tensor_capacity_bytes"]
    assert cache.attention_mask.shape == cache.position_ids.shape == (2, 0)
    append_all(cache, 1)
    assert cache.sequence_length == 1 and cache.position_ids.tolist() == [[0], [0]]
    assert pointers == [
        cache.get_layer(layer)[0].untyped_storage().data_ptr() for layer in range(3)
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_payload_accounting_matches_real_storage_sizes(dtype):
    cache = KVCache(config(), 2, max_length=8, dtype=dtype)
    expected_kv = 2 * 3 * 2 * 2 * 8 * 4 * torch.empty((), dtype=dtype).element_size()
    assert cache.memory_bytes() == {
        "kv_used_bytes": 0,
        "kv_capacity_bytes": expected_kv,
        "metadata_used_bytes": 0,
        "metadata_capacity_bytes": 2 * 8 * (1 + 8),
        "tensor_capacity_bytes": expected_kv + 144,
    }
    actual_bytes = sum(
        t.untyped_storage().nbytes() for layer in range(3) for t in cache.get_layer(layer)
    )
    assert actual_bytes == expected_kv
    append_all(cache, 3)
    assert cache.memory_bytes()["kv_used_bytes"] == expected_kv * 3 // 8
    assert cache.memory_bytes()["metadata_used_bytes"] == 2 * 3 * 9


@pytest.mark.parametrize("device", DEVICES)
def test_cache_created_and_updated_inside_inference_mode_can_be_reused_outside(device):
    with torch.inference_mode():
        cache = KVCache(config(), 1, device=device)
        append_all(cache, 2)
        assert not cache.get_layer(0)[0].requires_grad
    append_all(cache, 1)
    assert cache.sequence_length == 3
    cache.reset()
    with torch.inference_mode():
        append_all(cache, 1)
    assert cache.sequence_length == 1


@pytest.mark.parametrize("kind", ["key", "value"])
def test_autograd_inputs_rejected(kind):
    cache = KVCache(config(), 1)
    with cache.begin_step(1) as step:
        key, value = tensors(cache, 1, 0)
        (key if kind == "key" else value).requires_grad_()
        with pytest.raises(ValueError, match="gradients"):
            step.update(0, key, value)
        step.abort()


@pytest.mark.parametrize(
    "key",
    [
        torch.ones(1, 2, 1, 4, dtype=torch.float64),
        torch.ones(1, 2, 2, 4),
        torch.ones(1, 2, 1, 1),
        [1],
    ],
)
def test_no_implicit_dtype_conversion_or_broadcasting(key):
    cache = KVCache(config(), 1)
    with cache.begin_step(1) as step:
        with pytest.raises(ValueError, match="K/V must match"):
            step.update(0, key, tensors(cache, 1, 0)[1])
        step.abort()


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(1, 2),
        torch.ones(1, 1, dtype=torch.long),
        torch.tensor([[1, 2]]),
        torch.tensor([[1, -1]]),
        [[1, 1]],
    ],
)
def test_invalid_masks_do_not_open_transaction(mask):
    cache = KVCache(config(), 1)
    with pytest.raises(ValueError, match="attention_mask"):
        cache.begin_step(2, attention_mask=mask)
    append_all(cache, 1)


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([0.0, 1.0]),
        torch.tensor([0]),
        torch.tensor([-1, 1]),
        torch.tensor([0, 16]),
        [0, 1],
    ],
)
def test_invalid_positions_do_not_open_transaction(positions):
    cache = KVCache(config(), 1)
    with pytest.raises(ValueError, match="position_ids"):
        cache.begin_step(2, position_ids=positions)
    append_all(cache, 1)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_invalid_append_lengths(length):
    with pytest.raises(ValueError, match="num_tokens"):
        KVCache(config(), 1).begin_step(length)


@pytest.mark.parametrize("index", [-1, 3, True, 1.5])
def test_invalid_layer_indices(index):
    cache = KVCache(config(), 1)
    with pytest.raises(ValueError, match="layer index"):
        cache.get_layer(index)
    with cache.begin_step(1) as step:
        with pytest.raises(ValueError, match="layer index"):
            step.update(index, *tensors(cache, 1, 0))
        step.abort()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 1, "max_length": 0},
        {"batch_size": 1, "max_length": 17},
        {"batch_size": 1, "max_length": True},
        {"batch_size": 1, "dtype": torch.int64},
        {"batch_size": 1, "device": "meta"},
    ],
)
def test_invalid_cache_geometry_dtype_or_device(kwargs):
    with pytest.raises(ValueError):
        KVCache(config(), **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires two different devices")
def test_device_mismatch_rejected_without_silent_transfer():
    cache = KVCache(config(), 1, device="cuda")
    with pytest.raises(ValueError, match="cache device"):
        cache.begin_step(1, attention_mask=torch.ones(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="cache device"):
        cache.begin_step(1, position_ids=torch.zeros(1, dtype=torch.long))
    with cache.begin_step(1) as step:
        cpu = torch.zeros(1, 2, 1, 4)
        with pytest.raises(ValueError, match="dtype/device"):
            step.update(0, cpu, cpu)
        step.abort()
