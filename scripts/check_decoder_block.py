"""WP05.3 candidate-size decoder block: residual identity, masking and autograd."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import DecoderBlock
from llm_scratch.utils.config import load_config
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/local_smoke.toml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, device=args.device, output_root=args.output_root)
    config.data["experiment"]["name"] = "wp05-3-decoder-block"
    architecture = ModelConfig.from_dict(config.data["model"])
    if architecture.context_length < 4:
        raise ValueError("Block isolation probe needs context_length >= 4")
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
        run.metadata["execution_device"] = args.device
        run.metadata["probe_script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        write_json(run.path / "metadata.json", run.metadata)
        write_json(run.path / "artifacts/model_config.json", architecture.to_dict())
        block = DecoderBlock(architecture).to(args.device).eval()
        length = min(16, architecture.context_length)
        x = torch.randn(2, length, architecture.hidden_size, device=args.device, requires_grad=True)
        mask = torch.ones(2, length, device=args.device, dtype=torch.bool)
        mask[0, -(length // 4) :] = False
        mask[1, : length // 4] = False
        output = block(x, mask)
        cutoff = length // 2
        with torch.no_grad():
            future = x.detach().clone()
            future[:, cutoff:] += 100 * torch.randn_like(future[:, cutoff:])
            future_output = block(future, mask)
            changed_padding = x.detach().masked_fill(~mask.unsqueeze(-1), float("nan"))
            padding_output = block(changed_padding, mask)
        torch.testing.assert_close(output[:, :cutoff], future_output[:, :cutoff], rtol=0, atol=0)
        torch.testing.assert_close(output, padding_output, rtol=0, atol=0)
        prefix_grad = torch.autograd.grad(output[:, cutoff - 1].sum(), x, retain_graph=True)[0]
        if torch.count_nonzero(prefix_grad[:, cutoff:]):
            raise AssertionError("Future inputs received gradients from prefix outputs")
        output.square().mean().backward()
        if not torch.isfinite(output).all() or not torch.isfinite(x.grad).all():
            raise AssertionError("Non-finite block output/input gradients")
        gradient_norms = {}
        for name, parameter in block.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Missing or non-finite gradient: {name}")
            gradient_norms[name] = parameter.grad.norm().item()
            if gradient_norms[name] <= 0:
                raise AssertionError(f"Zero gradient in normal block probe: {name}")
        if torch.count_nonzero(output[~mask]) or torch.count_nonzero(x.grad[~mask]):
            raise AssertionError("Padding outputs/input gradients must be zero")
        block.zero_grad(set_to_none=True)
        padded_x = torch.full_like(x, float("nan"), requires_grad=True)
        padded_output = block(padded_x, torch.zeros_like(mask))
        padded_output.sum().backward()
        for value in (padded_output, padded_x.grad, *(p.grad for p in block.parameters())):
            if value is None or not torch.isfinite(value).all() or torch.count_nonzero(value):
                raise AssertionError(
                    "Fully padded examples must have finite zero outputs/gradients"
                )
        block.zero_grad(set_to_none=True)
        with torch.no_grad():
            block.attention.out_proj.weight.zero_()
            block.ffn.down_proj.weight.zero_()
        identity_x = x.detach().clone().requires_grad_()
        identity_output = block(identity_x, mask)
        torch.testing.assert_close(
            identity_output, identity_x.masked_fill(~mask.unsqueeze(-1), 0), rtol=0, atol=0
        )
        identity_output.sum().backward()
        torch.testing.assert_close(
            identity_x.grad, mask.unsqueeze(-1).expand_as(identity_x).float(), rtol=0, atol=0
        )
        report = {
            "work_package": "WP05.3",
            "status": "passed",
            "device": args.device,
            "block_count": 1,
            "model_config": architecture.to_dict(),
            "input_shape": list(x.shape),
            "output_shape": list(output.shape),
            "parameter_count": sum(p.numel() for p in block.parameters()),
            "expected_parameter_count": 4 * architecture.hidden_size**2
            + 3 * architecture.hidden_size * architecture.ffn_hidden_size
            + 2 * architecture.hidden_size,
            "gradient_l2_before_zeroing_branches": gradient_norms,
            "checks": {
                "finite_outputs_and_gradients": True,
                "all_trainable_parameters_nonzero_gradients": True,
                "future_perturbation_prefix_unchanged": True,
                "prefix_to_future_input_gradients_zero": True,
                "nonfinite_padding_perturbation_unchanged": True,
                "padding_outputs_and_input_gradients_zero": True,
                "fully_padded_outputs_all_gradients_finite_zero": True,
                "zero_branches_exact_identity_and_input_jacobian": True,
            },
            "future_prefix_max_abs_difference": (output[:, :cutoff] - future_output[:, :cutoff])
            .abs()
            .max()
            .item(),
            "padding_max_abs_difference": (output - padding_output).abs().max().item(),
            "limitation": "Block probe only; full decoder, LM head, CE loss and overfit follow.",
        }
        if report["parameter_count"] != report["expected_parameter_count"]:
            raise AssertionError(
                "Block parameter count differs from attention + SwiGLU + two norms"
            )
        write_json(run.path / "artifacts/decoder_block_report.json", report)
        run.log("decoder_block_passed", device=args.device, checks=report["checks"])
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
