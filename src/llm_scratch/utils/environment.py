"""Record runtime versions and enforce requested CPU/CUDA/A100 environment gates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_metadata(directory: Path) -> dict:
    root = _command(["git", "rev-parse", "--show-toplevel"], directory)
    if root is None:
        return {"state": "unavailable", "commit": None, "dirty": None}
    repo = Path(root)
    commit = _command(["git", "rev-parse", "--verify", "HEAD"], repo)
    status = _command(["git", "status", "--porcelain", "--untracked-files=all"], repo)
    return {
        "state": "committed" if commit else "unborn",
        "commit": commit,
        "branch": _command(["git", "symbolic-ref", "--short", "HEAD"], repo),
        "dirty": bool(status) if status is not None else None,
        "untracked_files": sum(line.startswith("??") for line in (status or "").splitlines()),
    }


def package_fingerprint() -> dict:
    """Fingerprint actual imported code, including an uncommitted editable install."""
    import llm_scratch

    root = Path(llm_scratch.__file__).parent
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    from llm_scratch.utils.config import config_digest

    return {"sha256": config_digest(files), "files": files}


def inspect_environment(device: str = "cpu", require_gpu_name: str = "") -> dict:
    errors: list[str] = []
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "requested_device": device,
        "selected_device": None,
        "numpy_version": None,
        "pytorch_version": None,
        "cuda_version": None,
        "cudnn_version": None,
        "cuda_available": False,
        "gpus": [],
        "nvidia_smi": None,
        "process_environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    if device not in ("cpu", "cuda", "auto"):
        errors.append(f"Unsupported device: {device}")
    executable = shutil.which("nvidia-smi")
    if executable:
        report["nvidia_smi"] = _command(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
    try:
        import numpy as np

        report["numpy_version"] = np.__version__
    except ImportError:
        errors.append("NumPy is missing; install the project dependencies")
    try:
        import torch

        available = torch.cuda.is_available()
        report.update(
            pytorch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            cuda_available=available,
        )
        if available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_bytes": props.total_memory,
                        "compute_capability": [props.major, props.minor],
                    }
                )
        selected = "cuda:0" if available and device in ("auto", "cuda") else "cpu"
        report["selected_device"] = selected
        if device == "cuda" and not available:
            report["selected_device"] = None
            errors.append("CUDA requested but unavailable; no CPU fallback is allowed")
        if require_gpu_name and (
            selected == "cpu" or require_gpu_name.lower() not in report["gpus"][0]["name"].lower()
        ):
            errors.append(
                f"Required GPU name contains {require_gpu_name!r}; selected device does not match"
            )
    except (ImportError, OSError, RuntimeError) as error:
        errors.append(f"PyTorch runtime is unavailable: {error}. Install the cpu or cuda extra.")
    report["validation"] = {"ok": not errors, "errors": errors}
    return report


def collect_metadata(config_path: Path, config_sha256: str, seed: int) -> dict:
    packages = {
        item.metadata["Name"]: item.version
        for item in importlib.metadata.distributions()
        if item.metadata["Name"]
    }
    return {
        "schema_version": 1,
        "git": git_metadata(config_path.parent),
        "package_source": package_fingerprint(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "seed": seed,
        "packages": dict(sorted(packages.items())),
        "invocation": sys.argv,
    }
