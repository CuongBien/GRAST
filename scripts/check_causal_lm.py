"""WP05.4 full candidate LM forward/backward, causal CE and unique parameter counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm_scratch.data.collator import collate_tokenized, encode_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.reasoning.formatting import encode_sft_example
from llm_scratch.utils.config import load_config
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/local_smoke.toml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, device=args.device, output_root=args.output_root)
    config.data["experiment"]["name"] = "wp05-4-causal-lm"
    architecture = ModelConfig.from_dict(config.data["model"])
    if architecture.context_length < 4:
        raise ValueError("LM probe needs context_length >= 4")
    with Run(config) as run:
        seed_everything(
            config.data["reproducibility"]["seed"],
            deterministic=config.data["reproducibility"]["deterministic"],
            num_threads=config.data["reproducibility"]["num_threads"],
        )
        environment = inspect_environment(
            args.device, config.data["environment"]["require_gpu_name"]
        )
        write_json(run.path / "artifacts/environment.json", environment)
        if not environment["validation"]["ok"]:
            raise RuntimeError(environment["validation"]["errors"])
        tokenizer = ByteBPETokenizer.load(args.tokenizer) if args.tokenizer else None
        if tokenizer is not None:
            if len(tokenizer) != architecture.vocab_size:
                raise ValueError("Model vocabulary size does not match tokenizer artifact")
            examples = [
                encode_pretraining(tokenizer, "A small fox shares a story."),
                encode_sft_example(tokenizer, "What is 2 + 3?", "Add 2 and 3.", "5"),
            ]
            batch = {
                name: torch.tensor(value, device=args.device)
                for name, value in collate_tokenized(examples, tokenizer).items()
            }
            source = "wp04_tokenizer_pretrain_sft_examples"
        else:
            length = min(16, architecture.context_length)
            ids = torch.randint(architecture.vocab_size, (2, length), device=args.device)
            mask = torch.ones_like(ids)
            mask[0, -(length // 4) :] = 0
            mask[1, : length // 4] = 0
            labels = ids.clone()
            labels[0, : length // 4] = -100
            batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
            source = "seeded_synthetic_token_ids"
        if batch["input_ids"].shape[1] > architecture.context_length:
            raise ValueError("Probe examples exceed context; use a larger model context")
        fingerprint = tokenizer.fingerprint if tokenizer is not None else None
        model = CausalLM(architecture, tokenizer_fingerprint=fingerprint).to(args.device).eval()
        counts = model.parameter_counts()
        if counts["total"] != architecture.expected_parameter_count:
            raise AssertionError("Actual unique parameter count differs from architecture formula")
        if (model.lm_head.weight is model.token_embedding.weight) != architecture.tie_embeddings:
            raise AssertionError("LM head sharing does not match configuration")
        output = model(**batch)
        mask = batch["attention_mask"].bool()
        pairs = mask[:, :-1] & mask[:, 1:] & (batch["labels"][:, 1:] != -100)
        with torch.no_grad():
            log_probabilities = output.logits[:, :-1].float().log_softmax(-1)
            targets = batch["labels"][:, 1:].clamp_min(0).unsqueeze(-1)
            manual_loss = -log_probabilities.gather(-1, targets).squeeze(-1)[pairs].mean()
            torch.testing.assert_close(output.loss, manual_loss)
            if output.num_loss_tokens != int(pairs.sum()):
                raise AssertionError("Wrong supervised next-token count")
            changed = batch["input_ids"].clone()
            cutoff = changed.shape[1] // 2
            changed[:, cutoff:] = (changed[:, cutoff:] + 1) % architecture.vocab_size
            changed_output = model(changed, batch["attention_mask"]).logits
            torch.testing.assert_close(
                output.logits[:, :cutoff], changed_output[:, :cutoff], rtol=0, atol=0
            )
        if not torch.isfinite(output.logits).all() or not torch.isfinite(output.loss):
            raise AssertionError("Non-finite model logits/loss")
        if torch.count_nonzero(output.logits[~mask]):
            raise AssertionError("Padding logits must be zero")
        output.loss.backward()
        gradient_norms = {
            "token_embedding": 0.0,
            "blocks": 0.0,
            "final_norm": 0.0,
            "lm_head_unique": 0.0,
        }
        checked = 0
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Missing/non-finite gradient: {name}")
            norm = parameter.grad.norm().item()
            if norm <= 0:
                raise AssertionError(f"Zero gradient in normal LM probe: {name}")
            group = "lm_head_unique" if name.startswith("lm_head.") else name.split(".")[0]
            gradient_norms[group] += norm**2
            checked += 1
        report = {
            "work_package": "WP05.4",
            "status": "passed",
            "device": args.device,
            "batch_source": source,
            "tokenizer_fingerprint": fingerprint,
            "model_config": architecture.to_dict(),
            "input_shape": list(batch["input_ids"].shape),
            "logits_shape": list(output.logits.shape),
            "loss": output.loss.item(),
            "manual_next_token_loss": manual_loss.item(),
            "num_loss_tokens": output.num_loss_tokens,
            "parameter_counts": counts,
            "expected_parameter_count": architecture.expected_parameter_count,
            "unique_parameters_with_finite_nonzero_gradients": checked,
            "gradient_l2_by_component": {
                name: value**0.5 for name, value in gradient_norms.items()
            },
            "checks": {
                "finite_logits_loss_gradients": True,
                "weight_tying_correct": True,
                "parameter_formula_matches_actual": True,
                "manual_shifted_ce_matches": True,
                "supervised_pair_count_correct": True,
                "padding_logits_zero": True,
                "basic_future_perturbation_prefix_unchanged": True,
            },
            "limitation": "Untrained model forward/backward probe; no overfit or quality claim.",
        }
        write_json(run.path / "artifacts/model_config.json", architecture.to_dict())
        write_json(run.path / "artifacts/causal_lm_report.json", report)
        run.metadata.update(
            execution_device=args.device,
            tokenizer_fingerprint=fingerprint,
            probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log(
            "causal_lm_passed", device=args.device, checks=report["checks"], parameter_counts=counts
        )
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
