"""WP05.4 full LM assembly, shifted/masked CE, tying, checkpoint and dtype contracts."""

import copy
import io
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from llm_scratch.data.collator import collate_tokenized, encode_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.reasoning.formatting import encode_reasoning_prompt, encode_sft_example
from llm_scratch.training.checkpoint import load_checkpoint, save_checkpoint
from llm_scratch.utils.config import load_config
from llm_scratch.utils.seed import seed_everything

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def config(**changes):
    return ModelConfig(
        **(
            dict(
                vocab_size=16,
                context_length=32,
                hidden_size=8,
                num_heads=2,
                num_layers=2,
                ffn_hidden_size=12,
            )
            | changes
        )
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tied", [True, False])
def test_full_model_shapes_unique_parameters_and_all_gradients(device, tied):
    cfg = config(tie_embeddings=tied)
    model = CausalLM(cfg).to(device)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 2, 0]], device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], device=device)
    output = model(ids, mask, labels=ids)
    assert output.logits.shape == (2, 5, 16)
    assert output.num_loss_tokens == 7
    assert torch.isfinite(output.loss) and output.loss > 0
    output.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert (model.lm_head.weight is model.token_embedding.weight) == tied
    assert len(model.blocks) == 2 and model.blocks[0] is not model.blocks[1]
    assert model.blocks[0].attention.q_proj.weight is not model.blocks[1].attention.q_proj.weight
    counts = model.parameter_counts()
    assert counts == {
        "token_embedding": 128,
        "blocks": 1120,
        "final_norm": 8,
        "lm_head_unique": 0 if tied else 128,
        "total": 1256 if tied else 1384,
    }
    assert (
        counts["total"]
        == cfg.expected_parameter_count
        == sum(p.numel() for p in model.parameters())
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.token_embedding.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.token_embedding.weight)
    assert (model.lm_head.weight is model.token_embedding.weight) == tied


def test_candidate_count_is_46148096_with_tied_head_and_50342400_untied():
    path = Path(__file__).resolve().parents[1] / "configs/experiments/local_smoke.toml"
    cfg = ModelConfig.from_dict(load_config(path).data["model"])
    assert cfg.expected_parameter_count == 46_148_096
    assert (
        ModelConfig.from_dict(cfg.to_dict() | {"tie_embeddings": False}).expected_parameter_count
        == 50_342_400
    )


def test_full_assembly_matches_embedding_blocks_final_norm_and_head_in_order():
    model = CausalLM(config()).double().eval()
    ids = torch.tensor([[1, 3, 4, 2], [1, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    positions = torch.tensor([[0, 2, 4, 6], [0, 1, 2, 3]])
    observed = []
    handles = [
        block.register_forward_hook(lambda module, args, output, i=i: observed.append(i))
        for i, block in enumerate(model.blocks)
    ]
    actual = model(ids, mask, position_ids=positions)
    for handle in handles:
        handle.remove()
    assert observed == [0, 1]
    hidden = F.embedding(ids, model.token_embedding.weight).masked_fill(~mask.unsqueeze(-1), 0)
    for block in model.blocks:
        hidden = block(hidden, mask, positions)
    hidden = F.rms_norm(hidden, (8,), model.final_norm.weight, model.config.rms_norm_eps)
    expected = (hidden @ model.lm_head.weight.T).masked_fill(~mask.unsqueeze(-1), 0)
    torch.testing.assert_close(actual.logits, expected, rtol=1e-12, atol=1e-12)
    assert actual.loss is None and actual.num_loss_tokens == 0


@pytest.mark.parametrize("device", DEVICES)
def test_loss_exact_next_token_shift_prompt_mask_and_logit_gradients(device):
    model = CausalLM(config()).to(device).double()
    ids = torch.tensor([[1, 4, 5, 6, 2]], device=device)
    labels = torch.tensor([[-100, -100, 9, 10, 2]], device=device)
    output = model(ids, labels=labels)
    output.logits.retain_grad()
    expected = -torch.stack(
        [
            output.logits[0, query].log_softmax(-1)[target]
            for query, target in [(1, 9), (2, 10), (3, 2)]
        ]
    ).mean()
    assert output.num_loss_tokens == 3
    torch.testing.assert_close(output.loss, expected, rtol=1e-12, atol=1e-12)
    output.loss.backward()
    assert torch.count_nonzero(output.logits.grad[:, [0, 4]]) == 0
    assert output.logits.grad[:, 1:4].abs().sum() > 0
    changed_labels = labels.clone()
    changed_labels[0, 0] = 12  # First label is never a next-token target.
    torch.testing.assert_close(model(ids, labels=changed_labels).loss, output.loss, rtol=0, atol=0)
    wrong_unshifted = F.cross_entropy(
        output.logits[:, 2:].reshape(-1, 16), labels[:, 2:].reshape(-1)
    )
    assert not torch.isclose(output.loss, wrong_unshifted, rtol=1e-8, atol=1e-8)


def test_uniform_logits_loss_is_log_vocab_with_correct_target_denominator():
    model = CausalLM(config()).double()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    ids = torch.tensor([[1, 3, 4, 2], [1, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    labels = ids.clone()
    labels[0, 1] = -100
    output = model(ids, mask, labels)
    assert output.num_loss_tokens == 3
    torch.testing.assert_close(output.loss, torch.tensor(math.log(16), dtype=torch.float64))


@pytest.mark.parametrize("mask_row", [[0, 0, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 1, 1, 0]])
def test_loss_excludes_pairs_touching_padding_and_does_not_bridge_gaps(mask_row):
    model = CausalLM(config()).double().eval()
    ids = torch.tensor([[1, 3, 4, 5, 2]])
    mask = torch.tensor([mask_row], dtype=torch.bool)
    output = model(ids, mask, labels=ids)
    pairs = [i for i in range(4) if mask_row[i] and mask_row[i + 1]]
    expected = -torch.stack(
        [output.logits[0, i].log_softmax(-1)[ids[0, i + 1]] for i in pairs]
    ).mean()
    assert output.num_loss_tokens == len(pairs)
    torch.testing.assert_close(output.loss, expected, rtol=1e-12, atol=1e-12)
    changed_ids = ids.clone()
    changed_ids[~mask] = 15
    changed = model(changed_ids, mask, labels=changed_ids)
    torch.testing.assert_close(output.logits, changed.logits, rtol=0, atol=0)
    torch.testing.assert_close(output.loss, changed.loss, rtol=0, atol=0)
    assert torch.count_nonzero(output.logits[~mask]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["single_token", "all_ignored", "all_padding", "isolated_tokens"])
def test_no_supervised_pairs_has_differentiable_zero_and_all_gradients_zero(device, case):
    model = CausalLM(config()).to(device)
    ids = torch.tensor([[1]] if case == "single_token" else [[1, 4, 5, 2]], device=device)
    mask = torch.ones_like(ids)
    labels = ids.clone()
    if case == "all_ignored":
        labels.fill_(-100)
    elif case == "all_padding":
        mask.zero_()
    elif case == "isolated_tokens":
        mask[0] = torch.tensor([1, 0, 1, 0], device=device)
    output = model(ids, mask, labels)
    assert output.num_loss_tokens == 0 and output.loss.requires_grad
    assert output.loss.item() == 0
    output.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) == 0, name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_logits_fp32_ce_and_weight_tying_after_conversion(device, dtype):
    model = CausalLM(config()).to(device=device, dtype=dtype)
    ids = torch.tensor([[1, 3, 4, 2]], device=device, dtype=torch.int32)
    output = model(ids, labels=ids)
    assert output.logits.dtype == dtype and output.logits.device == ids.device
    assert output.loss.dtype == torch.float32 and torch.isfinite(output.loss)
    assert model.lm_head.weight is model.token_embedding.weight
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("device", DEVICES)
def test_autocast_returns_low_precision_logits_and_fp32_loss(device):
    model = CausalLM(config()).to(device)
    ids = torch.tensor([[1, 3, 4, 2]], device=device)
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device, dtype=dtype):
        output = model(ids, labels=ids)
    assert output.logits.dtype == dtype and output.loss.dtype == torch.float32
    output.loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("assign", [False, True])
def test_state_dict_weights_only_metadata_and_alias_round_trip(tied, assign):
    model = CausalLM(config(tie_embeddings=tied), tokenizer_fingerprint="a" * 64).double().eval()
    ids = torch.tensor([[1, 3, 4, 2]])
    expected = model(ids, labels=ids)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    assert state["_extra_state"]["model_config"] == model.config.to_dict()
    restored = CausalLM(model.config).double().eval()
    restored.load_state_dict(state, assign=assign)
    actual = restored(ids, labels=ids)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    assert restored.tokenizer_fingerprint == "a" * 64
    assert (restored.lm_head.weight is restored.token_embedding.weight) == tied
    assert restored.parameter_counts() == model.parameter_counts()


def test_checkpoint_config_fingerprint_and_disagreeing_tied_weights_are_rejected():
    model = CausalLM(config(), tokenizer_fingerprint="a" * 64)
    with pytest.raises(ValueError, match="configuration mismatch"):
        CausalLM(config(rope_theta=20000)).load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CausalLM(config(), tokenizer_fingerprint="b" * 64).load_state_dict(model.state_dict())
    state = copy.deepcopy(model.state_dict())
    state["lm_head.weight"] = state["lm_head.weight"].clone() + 1
    with pytest.raises(ValueError, match="disagree"):
        CausalLM(config()).load_state_dict(state)
    for bad in (
        {},
        {"schema_version": True, "model_config": config().to_dict(), "tokenizer_fingerprint": None},
    ):
        with pytest.raises(ValueError, match="metadata"):
            model.set_extra_state(bad)


def test_model_state_works_with_existing_atomic_rng_checkpoint(tmp_path):
    generator = seed_everything(42)
    model = CausalLM(config(), tokenizer_fingerprint="a" * 64).eval()
    ids = torch.tensor([[1, 3, 4, 2]])
    expected = model(ids).logits
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"step": 0, "model": model.state_dict()}, generator=generator)
    restored = CausalLM(config()).eval()
    restored.load_state_dict(load_checkpoint(path, generator=generator)["model"])
    torch.testing.assert_close(restored(ids).logits, expected, rtol=0, atol=0)
    assert restored.tokenizer_fingerprint == "a" * 64


def test_real_tokenizer_pretrain_sft_prefix_and_first_answer_target():
    tokenizer = ByteBPETokenizer.train(
        ["A story. Một câu chuyện. What is 2 + 3? Add 2 and 3."], vocab_size=300
    )
    pretrain = encode_pretraining(tokenizer, "A story.")
    sft = encode_sft_example(tokenizer, "What is 2 + 3?", "Add 2 and 3.", "5")
    batch = {
        name: torch.tensor(value)
        for name, value in collate_tokenized([pretrain, sft], tokenizer).items()
    }
    model = CausalLM(
        config(vocab_size=len(tokenizer), context_length=128),
        tokenizer_fingerprint=tokenizer.fingerprint,
    ).eval()
    output = model(**batch)
    assert torch.isfinite(output.loss)
    prefix = encode_reasoning_prompt(tokenizer, "What is 2 + 3?")
    assert sft["input_ids"][: len(prefix)] == prefix
    prefix_logits = model(torch.tensor([prefix])).logits
    torch.testing.assert_close(
        prefix_logits[0], output.logits[1, : len(prefix)], rtol=1e-5, atol=1e-6
    )
    expected_pairs = sum(label != -100 for label in pretrain["labels"][1:]) + sum(
        label != -100 for label in sft["labels"][1:]
    )
    assert output.num_loss_tokens == expected_pairs
    assert sft["labels"][len(prefix)] != -100 and sft["labels"][len(prefix) - 1] == -100
    output.loss.backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0


def test_training_dropout_eval_and_basic_future_token_isolation():
    model = CausalLM(config(dropout=0.5))
    ids = torch.tensor([[1, 3, 4, 5, 2]])
    torch.manual_seed(42)
    first = model(ids).logits
    torch.manual_seed(42)
    torch.testing.assert_close(first, model(ids).logits, rtol=0, atol=0)
    assert not torch.equal(first, model(ids).logits)
    model.eval()
    changed = ids.clone()
    changed[:, 3:] = 15
    torch.testing.assert_close(
        model(ids).logits[:, :3], model(changed).logits[:, :3], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "ids",
    [
        torch.ones(1, 3),
        torch.ones(3, dtype=torch.long),
        torch.tensor([[-1]]),
        torch.tensor([[16]]),
        torch.empty(0, 3, dtype=torch.long),
        torch.empty(1, 0, dtype=torch.long),
        torch.ones(1, 33, dtype=torch.long),
    ],
)
def test_invalid_input_ids_rejected(ids):
    with pytest.raises(ValueError):
        CausalLM(config())(ids)


@pytest.mark.parametrize(
    "labels",
    [
        torch.ones(1, 3),
        torch.ones(1, 2, dtype=torch.long),
        torch.tensor([[1, -1, 2]]),
        torch.tensor([[1, 16, 2]]),
    ],
)
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        CausalLM(config())(torch.tensor([[1, 3, 2]]), labels=labels)


@pytest.mark.parametrize("fingerprint", ["", "A" * 64, "g" * 64, 42])
def test_invalid_tokenizer_fingerprint_rejected(fingerprint):
    with pytest.raises(ValueError, match="fingerprint"):
        CausalLM(config(), tokenizer_fingerprint=fingerprint)
