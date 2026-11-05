"""WP02 acceptance probe: two fresh processes, deterministic draws and checkpoint resume."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from llm_scratch.training.checkpoint import load_checkpoint, save_checkpoint
from llm_scratch.utils.config import LoadedConfig, config_digest, validate_config
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything, worker_environment


def _probe(generator, device: str) -> dict:
    import numpy as np
    import torch

    matrix = torch.randn(8, 8, device=device, dtype=torch.float32)
    product = matrix @ matrix.T
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {
        "python": [random.random() for _ in range(8)],
        "numpy_global": np.random.random(8).tolist(),
        "numpy_generator": generator.random(8).tolist(),
        "torch_cpu": torch.rand(8).tolist(),
        "torch_device": torch.rand(8, device=device).cpu().tolist(),
        "matrix_product": product.cpu().tolist(),
    }


def _worker(run_path: Path, repeat: int) -> None:
    import torch

    data = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    validate_config(data)
    repro = data["reproducibility"]
    generator = seed_everything(**repro)
    report = inspect_environment(**data["environment"])
    if not report["validation"]["ok"]:
        raise RuntimeError("; ".join(report["validation"]["errors"]))
    device = report["selected_device"]
    first = _probe(generator, device)
    checkpoint = run_path / "checkpoints" / f"repeat_{repeat}.pt"
    save_checkpoint(checkpoint, {"step": 0, "purpose": "rng-smoke-only"}, generator=generator)
    expected = _probe(generator, device)
    restored = load_checkpoint(checkpoint, generator=generator)
    actual = _probe(generator, device)
    if expected != actual or restored["step"] != 0:
        raise AssertionError("Checkpoint did not restore the exact RNG continuation")
    payload = {
        "probe": first,
        "after_resume": actual,
        "checkpoint_resume_equal": True,
        "settings": {
            "device": device,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    write_json(run_path / "artifacts" / f"repeat_{repeat}.json", payload)


def run_smoke(config: LoadedConfig) -> Path:
    """Create an auditable run even if environment validation or a worker fails."""
    with Run(config) as run:
        report = inspect_environment(**config.data["environment"])
        run.metadata["environment"] = report
        write_json(run.path / "metadata.json", run.metadata)
        if not report["validation"]["ok"]:
            raise RuntimeError("; ".join(report["validation"]["errors"]))
        run.log("environment_checked", device=report["selected_device"])
        env = worker_environment(**config.data["reproducibility"])
        outputs = []
        for repeat in (1, 2):
            command = [
                sys.executable,
                "-m",
                "llm_scratch.smoke",
                "--worker",
                str(run.path),
                "--repeat",
                str(repeat),
            ]
            try:
                process = subprocess.run(
                    command, env=env, capture_output=True, text=True, timeout=120
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"Repeat {repeat} exceeded the 120-second smoke timeout"
                ) from error
            (run.path / "logs" / f"repeat_{repeat}.stdout.log").write_text(
                process.stdout, encoding="utf-8"
            )
            (run.path / "logs" / f"repeat_{repeat}.stderr.log").write_text(
                process.stderr, encoding="utf-8"
            )
            if process.returncode:
                error_log = run.path / "logs" / f"repeat_{repeat}.stderr.log"
                raise RuntimeError(f"Repeat {repeat} failed; see {error_log}")
            outputs.append(
                json.loads(
                    (run.path / "artifacts" / f"repeat_{repeat}.json").read_text(encoding="utf-8")
                )
            )
            run.log("repeat_completed", repeat=repeat)
        if outputs[0] != outputs[1]:
            raise AssertionError("Fresh-process outputs differ for the same seed")
        summary = {
            "repeat_equal": True,
            "checkpoint_resume_equal": True,
            "probe_sha256": config_digest(outputs[0]),
            "seed": config.data["reproducibility"]["seed"],
            "device": report["selected_device"],
            "config_sha256": config.sha256,
            "scope": "RNG and tiny float32 matrix probe only; no model or training",
        }
        write_json(run.path / "result.json", summary)
        run.log("reproducibility_verified", **summary)
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    _worker(args.worker, args.repeat)
