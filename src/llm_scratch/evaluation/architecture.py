"""Fixed-batch metrics and adversarial causal/padding audits of the full CausalLM."""

from __future__ import annotations

import torch

from llm_scratch.model.transformer import CausalLM


def supervised_pairs(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = batch["attention_mask"].bool()
    return valid[:, :-1] & valid[:, 1:] & (batch["labels"][:, 1:] != -100)


def measure_batch(model: CausalLM, batch: dict[str, torch.Tensor]) -> dict:
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(**batch)
            pairs = supervised_pairs(batch)
            count = int(pairs.sum())
            if not count or count != output.num_loss_tokens:
                raise ValueError("Diagnostic batch must contain supervised next-token pairs")
            correct = (output.logits[:, :-1].argmax(-1) == batch["labels"][:, 1:]) & pairs
            if not torch.isfinite(output.loss):
                raise AssertionError("Non-finite diagnostic loss")
            return {
                "loss": output.loss.item(),
                "token_accuracy": int(correct.sum()) / count,
                "num_loss_tokens": count,
            }
    finally:
        model.train(training)


def audit_model(model: CausalLM, batch: dict[str, torch.Tensor]) -> dict:
    """Run in eval with fixed weights; restore mode and never accumulate parameter grads.

    Include negative controls in tests: a deliberately leaky attention or nonzero PAD
    logits must make this audit fail rather than produce an unconditional success report.
    """
    training = model.training
    ids, mask = batch["input_ids"], batch["attention_mask"].bool()
    length = ids.shape[1]
    if length < 3 or length + 2 > model.config.context_length:
        raise ValueError("Audit needs >=3 tokens and space for two extra padding positions")
    rtol, atol = (
        (1e-10, 1e-12) if model.token_embedding.weight.dtype == torch.float64 else (1e-4, 1e-5)
    )
    model.eval()
    try:
        with torch.no_grad():
            base = model(ids, mask).logits
            if not torch.isfinite(base).all() or torch.count_nonzero(base[~mask]):
                raise AssertionError("Non-finite logits or nonzero padding logits")
            cutoffs = sorted({1, length // 2, length - 1})
            max_future_difference = 0.0
            for cutoff in cutoffs:
                changed = ids.clone()
                changed[:, cutoff:] = (changed[:, cutoff:] + 1) % model.config.vocab_size
                perturbed = model(changed, mask).logits
                torch.testing.assert_close(base[:, :cutoff], perturbed[:, :cutoff], rtol=0, atol=0)
                prefix = model(ids[:, :cutoff], mask[:, :cutoff]).logits
                torch.testing.assert_close(base[:, :cutoff], prefix, rtol=rtol, atol=atol)
                max_future_difference = max(
                    max_future_difference,
                    (base[:, :cutoff] - perturbed[:, :cutoff]).abs().max().item(),
                )
            changed = ids.masked_fill(~mask, model.config.vocab_size - 1)
            torch.testing.assert_close(base, model(changed, mask).logits, rtol=0, atol=0)
            labels = batch["labels"].clone()
            selected = labels != -100
            labels[selected] = (labels[selected] + 1) % model.config.vocab_size
            torch.testing.assert_close(base, model(ids, mask, labels=labels).logits, rtol=0, atol=0)
            pad_ids = torch.zeros(ids.shape[0], 2, dtype=ids.dtype, device=ids.device)
            pad_mask = torch.zeros_like(pad_ids, dtype=torch.bool)
            for left in (False, True):
                padded_ids = torch.cat([pad_ids, ids] if left else [ids, pad_ids], dim=1)
                padded_mask = torch.cat([pad_mask, mask] if left else [mask, pad_mask], dim=1)
                result = model(padded_ids, padded_mask).logits
                compared = result[:, 2:] if left else result[:, :-2]
                torch.testing.assert_close(base, compared, rtol=rtol, atol=atol)
                if torch.count_nonzero(result[~padded_mask]):
                    raise AssertionError("Added left/right padding has nonzero logits")
            middle = length // 2
            hole_ids = torch.cat([ids[:, :middle], pad_ids, ids[:, middle:]], dim=1)
            hole_mask = torch.cat([mask[:, :middle], pad_mask, mask[:, middle:]], dim=1)
            original_positions = torch.arange(length, device=ids.device)
            positions = torch.cat(
                [
                    original_positions[:middle],
                    torch.zeros(2, device=ids.device, dtype=torch.long),
                    original_positions[middle:],
                ]
            )
            result = model(hole_ids, hole_mask, position_ids=positions).logits
            torch.testing.assert_close(
                base,
                torch.cat([result[:, :middle], result[:, middle + 2 :]], dim=1),
                rtol=rtol,
                atol=atol,
            )
            if torch.count_nonzero(result[~hole_mask]):
                raise AssertionError("Interior padding has nonzero logits")
        captured = []

        def capture(module, args, value):
            captured.append(value)

        handle = model.token_embedding.register_forward_hook(capture)
        try:
            output = model(ids, mask).logits
            cutoff = length // 2
            gradient = torch.autograd.grad(output[:, cutoff - 1].square().sum(), captured[0])[0]
        finally:
            handle.remove()
        if not torch.isfinite(gradient).all() or torch.count_nonzero(gradient[:, cutoff:]):
            raise AssertionError("Prefix logits depend on future input embeddings")
        if torch.count_nonzero(gradient[~mask]) or gradient[:, :cutoff].abs().sum() == 0:
            raise AssertionError(
                "Padding gradients are nonzero or causal gradient check is vacuous"
            )
        padded = model(ids, torch.zeros_like(mask), labels=batch["labels"])
        gradients = torch.autograd.grad(padded.loss, tuple(model.parameters()))
        if padded.loss.item() != 0 or padded.num_loss_tokens or torch.count_nonzero(padded.logits):
            raise AssertionError("Fully padded loss/logits must be zero")
        if any(
            not torch.isfinite(value).all() or torch.count_nonzero(value) for value in gradients
        ):
            raise AssertionError("Fully padded parameter gradients must be finite zero")
        return {
            "cutoffs": cutoffs,
            "cross_shape_tolerance": {"rtol": rtol, "atol": atol},
            "max_future_prefix_difference": max_future_difference,
            "checks": {
                "future_perturbations": True,
                "prefix_only_equivalence": True,
                "zero_future_embedding_gradients": True,
                "zero_padding_embedding_gradients": True,
                "padding_value_isolation": True,
                "left_right_padding": True,
                "interior_padding_explicit_positions": True,
                "no_gold_label_input": True,
                "fully_padded_logits_loss_gradients_zero": True,
            },
        }
    finally:
        model.train(training)
