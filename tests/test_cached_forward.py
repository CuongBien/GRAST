"""WP06.2 cached model equivalence, transactional failure and request ownership."""

import copy

import pytest
import torch

from llm_scratch.model.cache import KVCache
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.utils.seed import seed_everything

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def config(**changes):
    return ModelConfig(
        **(
            dict(
                vocab_size=32,
                context_length=32,
                hidden_size=16,
                num_heads=4,
                num_layers=3,
                ffn_hidden_size=32,
            )
            | changes
        )
    )


def model(device="cpu", dtype=torch.float32, **changes):
    seed_everything(42)
    return CausalLM(config(**changes)).to(device=device, dtype=dtype).eval()


def assert_equivalent(actual, expected, dtype):
    tolerances = {
        torch.float64: (1e-10, 1e-12),
        torch.float32: (1e-4, 1e-5),
        torch.float16: (0.005, 0.002),
        torch.bfloat16: (0.02, 0.01),
    }
    rtol, atol = tolerances[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["full", "left_right", "interior_all_pad"])
def test_prefill_token_and_chunk_logits_match_uncached_prefixes(device, dtype, layout):
    net = model(device, dtype)
    ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 2], [1, 9, 10, 11, 12, 13, 14, 2]], device=device)
    mask = torch.ones_like(ids)
    if layout == "left_right":
        mask[0, 4:] = 0
        mask[1, :2] = 0
    elif layout == "interior_all_pad":
        mask[0, [1, 4]] = 0
        mask[1] = 0
    cache = None
    start, chunks = 0, []
    with torch.inference_mode():
        for length in (3, 1, 2, 1, 1):
            end = start + length
            result = net(ids[:, start:end], mask[:, start:end], use_cache=True, cache=cache)
            cache = result.cache
            assert cache is not None and cache.sequence_length == end
            expected = net(ids[:, :end], mask[:, :end]).logits[:, start:end]
            assert_equivalent(result.logits, expected, dtype)
            assert result.loss is None and result.num_loss_tokens == 0
            chunks.append(result.logits)
            start = end
        full = net(ids, mask).logits
        assert_equivalent(torch.cat(chunks, dim=1), full, dtype)
    assert cache.sequence_length == ids.shape[1]
    torch.testing.assert_close(cache.attention_mask, mask.bool(), rtol=0, atol=0)
    torch.testing.assert_close(cache.position_ids, torch.arange(8, device=device).expand(2, -1))
    assert torch.count_nonzero(torch.cat(chunks, dim=1)[~mask.bool()]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_custom_positions_match_uncached_and_require_explicit_continuation(device):
    net = model(device, torch.float64)
    ids = torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2]], device=device)
    positions = torch.tensor([[2, 5, 9, 12], [7, 6, 5, 4]], device=device)
    with torch.no_grad():
        prefix = net(ids[:, :2], position_ids=positions[:, :2], use_cache=True)
        cache = prefix.cache
        assert cache.requires_explicit_positions
        with pytest.raises(ValueError, match="Explicit position_ids"):
            net(ids[:, 2:3], use_cache=True, cache=cache)
        assert cache.sequence_length == 2
        result = net(ids[:, 2:], position_ids=positions[:, 2:], use_cache=True, cache=cache)
        assert_equivalent(
            result.logits, net(ids, position_ids=positions).logits[:, 2:], torch.float64
        )
    torch.testing.assert_close(cache.position_ids, positions)
    cache.reset()
    assert not cache.requires_explicit_positions


def test_decode_projects_only_new_tokens_and_never_rotates_past_keys_again():
    net = model(dtype=torch.float64)
    ids = torch.tensor([[1, 3, 4, 5, 6, 7, 2]])
    lengths = []
    handle = net.blocks[0].attention.k_proj.register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1])
    )
    try:
        with torch.no_grad():
            cache = net(ids[:, :4], use_cache=True).cache
            keys = [cache.get_layer(i)[0].clone() for i in range(3)]
            net(ids[:, 4:5], use_cache=True, cache=cache)
            net(ids[:, 5:], use_cache=True, cache=cache)
        assert lengths == [4, 1, 2]
        for index, old in enumerate(keys):
            torch.testing.assert_close(cache.get_layer(index)[0][:, :, :4], old, rtol=0, atol=0)
    finally:
        handle.remove()


def test_context_capacity_boundary_and_invalid_inputs_do_not_advance_cache():
    net = model(context_length=6)
    ids = torch.tensor([[1, 3, 4, 5, 6, 2]])
    with torch.no_grad():
        cache = net(ids[:, :4], use_cache=True).cache
        with pytest.raises(ValueError, match="attention_mask"):
            net(
                ids[:, 4:5],
                attention_mask=torch.ones(1, 5, dtype=torch.long),
                use_cache=True,
                cache=cache,
            )
        with pytest.raises(ValueError, match="context_length"):
            net(ids[:, 4:5], position_ids=torch.tensor([6]), use_cache=True, cache=cache)
        assert cache.sequence_length == 4
        final = net(ids[:, 4:], use_cache=True, cache=cache)
        assert_equivalent(final.logits, net(ids).logits[:, 4:], torch.float32)
        with pytest.raises(ValueError, match="capacity/context"):
            net(ids[:, :1], use_cache=True, cache=cache)
    assert cache.sequence_length == 6


def test_small_external_cache_and_root_model_binding_validation():
    net = model()
    ids = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        small = net.new_cache(1, max_length=3)
        assert net(ids, use_cache=True, cache=small).cache is small
        with pytest.raises(ValueError, match="capacity/context"):
            net(ids[:, :1], use_cache=True, cache=small)
        for cache in (
            KVCache(net.config, 2),
            KVCache(net.config, 1, dtype=torch.float64),
            KVCache(config(context_length=64), 1),
        ):
            with pytest.raises(ValueError, match="mismatch|precision"):
                net(ids, use_cache=True, cache=cache)
            assert cache.sequence_length == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "inplace",
        "reload",
        "dtype_roundtrip",
        "tokenizer_metadata",
        "replace_parameter",
        "rope_metadata",
    ],
)
def test_changed_model_state_rejects_old_prefix_until_reset(mutation):
    net = model()
    ids = torch.tensor([[1, 3, 4, 2]])
    with torch.no_grad():
        cache = net(ids[:, :3], use_cache=True).cache
        if mutation == "inplace":
            net.blocks[0].attention.q_proj.weight.add_(0.01)
        elif mutation == "reload":
            net.load_state_dict(copy.deepcopy(net.state_dict()))
        elif mutation == "dtype_roundtrip":
            net.half().float()
        elif mutation == "tokenizer_metadata":
            net.tokenizer_fingerprint = "a" * 64
        elif mutation == "rope_metadata":
            net.blocks[0].attention.rope.theta = 20000
        else:
            net.final_norm.weight = torch.nn.Parameter(net.final_norm.weight.clone())
        with pytest.raises(RuntimeError, match="weights/metadata changed"):
            net(ids[:, 3:], use_cache=True, cache=cache)
        assert cache.sequence_length == 3
        cache.reset()
        result = net(ids, use_cache=True, cache=cache)
        assert_equivalent(result.logits, net(ids).logits, torch.float32)


def test_different_model_and_raw_storage_append_cannot_reuse_owned_prefix():
    first, second = model(), model()
    ids = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        cache = first(ids[:, :2], use_cache=True).cache
        with pytest.raises(RuntimeError, match="another model"):
            second(ids[:, 2:], use_cache=True, cache=cache)
        with cache.begin_step(1) as step:
            for layer in range(first.config.num_layers):
                value = torch.zeros(1, 4, 1, 4)
                step.update(layer, value, value)
        with pytest.raises(RuntimeError, match="unbound"):
            first(ids[:, 2:], use_cache=True, cache=cache)
        cache.reset()
        second(ids, use_cache=True, cache=cache)


def test_failure_after_all_layers_rolls_back_and_retry_matches_uncached():
    net = model(dtype=torch.float64)
    ids = torch.tensor([[1, 3, 4, 5]])
    with torch.no_grad():
        cache = net(ids[:, :3], use_cache=True).cache
        old = [tuple(t.clone() for t in cache.get_layer(i)) for i in range(3)]

        def fail(module, args, result):
            raise RuntimeError("head failed")

        handle = net.lm_head.register_forward_hook(fail)
        try:
            with pytest.raises(RuntimeError, match="head failed"):
                net(ids[:, 3:], use_cache=True, cache=cache)
        finally:
            handle.remove()
        assert cache.sequence_length == 3
        for i in range(3):
            for actual, expected in zip(cache.get_layer(i), old[i], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        retry = net(ids[:, 3:], use_cache=True, cache=cache)
        assert_equivalent(retry.logits, net(ids).logits[:, 3:], torch.float64)


def test_mutation_during_forward_aborts_append_and_marks_existing_prefix_stale():
    net = model()
    ids = torch.tensor([[1, 3, 4, 5]])
    with torch.no_grad():
        cache = net(ids[:, :3], use_cache=True).cache

        def mutate(module, args, result):
            net.final_norm.weight.add_(0.01)

        handle = net.lm_head.register_forward_hook(mutate)
        try:
            with pytest.raises(RuntimeError, match="changed during cached forward"):
                net(ids[:, 3:], use_cache=True, cache=cache)
        finally:
            handle.remove()
        assert cache.sequence_length == 3
        with pytest.raises(RuntimeError, match="weights/metadata changed"):
            net(ids[:, 3:], use_cache=True, cache=cache)


@pytest.mark.parametrize("device", DEVICES)
def test_autocast_cache_dtype_and_precision_switch_guard(device):
    net = model(device)
    ids = torch.tensor([[1, 3, 4, 2]], device=device)
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
        result = net(ids[:, :3], use_cache=True)
        assert result.cache.dtype == dtype
        decoded = net(ids[:, 3:], use_cache=True, cache=result.cache)
        assert_equivalent(decoded.logits, net(ids).logits[:, 3:], dtype)
    with torch.no_grad(), pytest.raises(ValueError, match="precision"):
        net(ids[:, :1], use_cache=True, cache=result.cache)
    assert result.cache.sequence_length == 4


def test_cached_forward_rejects_training_grad_labels_and_ambiguous_arguments():
    net = model()
    ids = torch.tensor([[1, 3, 2]])
    with pytest.raises(ValueError, match="eval.*no_grad"):
        net(ids, use_cache=True)
    with torch.no_grad():
        net.train()
        with pytest.raises(ValueError, match="eval"):
            net(ids, use_cache=True)
        net.eval()
        net.blocks[0].train()
        with pytest.raises(ValueError, match="eval"):
            net(ids, use_cache=True)
        net.eval()
        with pytest.raises(ValueError, match="training labels"):
            net(ids, use_cache=True, labels=ids)
        with pytest.raises(ValueError, match="use_cache=True"):
            net(ids, cache=net.new_cache(1))
        with pytest.raises(ValueError, match="boolean"):
            net(ids, use_cache=1)
        with pytest.raises(ValueError, match="use_cache=True"):
            net(ids, use_cache=True, cache="wrong")
    with torch.inference_mode():
        unusual = CausalLM(config()).eval()
        with pytest.raises(ValueError, match="Construct model outside inference_mode"):
            unusual(ids, use_cache=True)


def test_all_padding_prefix_can_be_followed_by_valid_tokens():
    net = model(dtype=torch.float64)
    ids = torch.tensor([[1, 3, 4, 2]])
    mask = torch.tensor([[0, 0, 1, 1]])
    with torch.no_grad():
        result = net(ids[:, :2], mask[:, :2], use_cache=True)
        assert torch.count_nonzero(result.logits) == 0
        output = net(ids[:, 2:], mask[:, 2:], use_cache=True, cache=result.cache)
        assert_equivalent(output.logits, net(ids, mask).logits[:, 2:], torch.float64)


def test_independent_requests_can_interleave_without_model_global_cache_state():
    net = model(dtype=torch.float64)
    first = torch.tensor([[1, 3, 4, 2]])
    second = torch.tensor([[1, 9, 8, 7, 2]])
    with torch.no_grad():
        a = net(first[:, :2], use_cache=True).cache
        b = net(second[:, :3], use_cache=True).cache
        a_result = net(first[:, 2:], use_cache=True, cache=a)
        b_result = net(second[:, 3:], use_cache=True, cache=b)
        assert a is not b and a.sequence_length == 4 and b.sequence_length == 5
        assert_equivalent(a_result.logits, net(first).logits[:, 2:], torch.float64)
        assert_equivalent(b_result.logits, net(second).logits[:, 3:], torch.float64)
        a.reset()
        assert b.sequence_length == 5


def test_optimizer_update_invalidates_prior_inference_cache():
    net = model()
    ids = torch.tensor([[1, 3, 4, 2]])
    with torch.no_grad():
        cache = net(ids[:, :3], use_cache=True).cache
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    net.train()
    net(ids, labels=ids).loss.backward()
    optimizer.step()
    net.eval()
    with torch.no_grad(), pytest.raises(RuntimeError, match="weights/metadata changed"):
        net(ids[:, 3:], use_cache=True, cache=cache)


def test_cache_owner_is_weak_reference_not_a_model_lifetime_extension():
    import gc
    import weakref

    net = model()
    ids = torch.tensor([[1, 3, 2]])
    with torch.no_grad():
        cache = net(ids, use_cache=True).cache
    reference = weakref.ref(net)
    del net
    gc.collect()
    assert reference() is None
    another = model()
    with torch.no_grad(), pytest.raises(RuntimeError, match="another model"):
        another(ids[:, :1], use_cache=True, cache=cache)
