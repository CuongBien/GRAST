"""WP05.1 candidate-size forward/backward probe; no decoder or language-model loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import RMSNorm, RotaryEmbedding, TokenEmbedding
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
    config.data["experiment"]["name"] = "wp05-1-primitives"
    architecture = ModelConfig.from_dict(config.data["model"])
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
        embedding = TokenEmbedding(architecture.vocab_size, architecture.hidden_size).to(
            args.device
        )
        norm = RMSNorm(architecture.hidden_size, architecture.rms_norm_eps).to(args.device)
        rope = RotaryEmbedding(
            architecture.head_dim, architecture.context_length, architecture.rope_theta
        ).to(args.device)
        length = min(16, architecture.context_length)
        ids = torch.randint(architecture.vocab_size, (2, length), device=args.device)
        hidden = norm(embedding(ids))
        heads = hidden.view(2, length, architecture.num_heads, architecture.head_dim).transpose(
            1, 2
        )
        rotated = rope(heads)
        # A scalar solely for exercising autograd; this is not cross-entropy loss.
        probe_objective = rotated.square().mean()
        probe_objective.backward()
        parameters = list(embedding.parameters()) + list(norm.parameters())
        finite_gradients = all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in parameters
        )
        norm_error = (rotated.square().sum(-1) - heads.square().sum(-1)).abs().max().item()
        if not finite_gradients or not torch.isfinite(rotated).all():
            raise AssertionError("Non-finite primitive outputs/gradients")
        torch.testing.assert_close(rotated.square().sum(-1), heads.square().sum(-1))
        report = {
            "work_package": "WP05.1",
            "status": "passed",
            "device": args.device,
            "model_config": architecture.to_dict(),
            "input_shape": list(ids.shape),
            "hidden_shape": list(hidden.shape),
            "rotated_shape": list(rotated.shape),
            "parameter_counts": {
                "embedding": embedding.weight.numel(),
                "rmsnorm": norm.weight.numel(),
                "rope": sum(p.numel() for p in rope.parameters()),
            },
            "finite_gradients": bool(finite_gradients),
            "gradient_l2": {
                "embedding": embedding.weight.grad.norm().item(),
                "rmsnorm": norm.weight.grad.norm().item(),
            },
            "rope_max_squared_norm_error": norm_error,
            "probe_objective": probe_objective.item(),
            "limitation": "Primitive autograd probe; no attention, decoder, CE loss or overfit.",
        }
        write_json(run.path / "artifacts/primitives_report.json", report)
        run.log(
            "primitives_passed", device=args.device, parameter_counts=report["parameter_counts"]
        )
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
