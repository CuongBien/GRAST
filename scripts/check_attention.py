"""WP05.2 candidate-size causal/padding-isolation and forward/backward probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm_scratch.model.attention import CausalSelfAttention
from llm_scratch.model.config import ModelConfig
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
    config.data["experiment"]["name"] = "wp05-2-attention"
    architecture = ModelConfig.from_dict(config.data["model"])
    if architecture.context_length < 4:
        raise ValueError("Attention isolation probe needs context_length >= 4")
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
        attention = CausalSelfAttention(architecture).to(args.device).eval()
        length = min(16, architecture.context_length)
        x = torch.randn(2, length, architecture.hidden_size, device=args.device, requires_grad=True)
        mask = torch.ones(2, length, device=args.device, dtype=torch.bool)
        mask[0, -(length // 4) :] = False
        mask[1, : length // 4] = False
        output = attention(x, mask)
        cutoff = length // 2
        with torch.no_grad():
            changed_future = x.detach().clone()
            changed_future[:, cutoff:] += 100 * torch.randn_like(changed_future[:, cutoff:])
            future_output = attention(changed_future, mask)
            changed_padding = x.detach().clone().masked_fill(~mask.unsqueeze(-1), float("nan"))
            padding_output = attention(changed_padding, mask)
        torch.testing.assert_close(output[:, :cutoff], future_output[:, :cutoff], rtol=0, atol=0)
        torch.testing.assert_close(output, padding_output, rtol=0, atol=0)
        prefix_gradient = torch.autograd.grad(output[:, cutoff - 1].sum(), x, retain_graph=True)[0]
        if torch.count_nonzero(prefix_gradient[:, cutoff:]):
            raise AssertionError("Future inputs received gradients from prefix outputs")
        output.square().mean().backward()
        if not torch.isfinite(x.grad).all() or not all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in attention.parameters()
        ):
            raise AssertionError("Non-finite attention gradients")
        if torch.count_nonzero(output[~mask]) or torch.count_nonzero(x.grad[~mask]):
            raise AssertionError("Padding output/input gradients must be zero")
        padded_x = torch.randn_like(x, requires_grad=True)
        attention.zero_grad(set_to_none=True)
        padded_output = attention(padded_x, torch.zeros_like(mask))
        padded_output.sum().backward()
        if (
            not torch.isfinite(padded_output).all()
            or torch.count_nonzero(padded_output)
            or not torch.isfinite(padded_x.grad).all()
            or torch.count_nonzero(padded_x.grad)
            or not all(
                p.grad is not None and torch.count_nonzero(p.grad) == 0
                for p in attention.parameters()
            )
        ):
            raise AssertionError("Fully padded examples must have finite zero outputs/gradients")
        report = {
            "work_package": "WP05.2",
            "status": "passed",
            "device": args.device,
            "model_config": architecture.to_dict(),
            "input_shape": list(x.shape),
            "output_shape": list(output.shape),
            "parameter_count": sum(p.numel() for p in attention.parameters()),
            "expected_parameter_count": 4 * architecture.hidden_size**2,
            "checks": {
                "finite_gradients": True,
                "future_perturbation_prefix_unchanged": True,
                "prefix_to_future_input_gradients_zero": True,
                "padding_perturbation_output_unchanged": True,
                "padded_query_outputs_zero": True,
                "padding_input_gradients_zero": True,
                "fully_padded_outputs_and_gradients_finite_zero": True,
            },
            "future_prefix_max_abs_difference": (output[:, :cutoff] - future_output[:, :cutoff])
            .abs()
            .max()
            .item(),
            "padding_max_abs_difference": (output - padding_output).abs().max().item(),
            "limitation": "Attention module only; no decoder, language-model loss or overfit yet.",
        }
        if report["parameter_count"] != report["expected_parameter_count"]:
            raise AssertionError(
                "Attention parameter count differs from four bias-free projections"
            )
        write_json(run.path / "artifacts/attention_report.json", report)
        run.log("attention_passed", device=args.device, checks=report["checks"])
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
