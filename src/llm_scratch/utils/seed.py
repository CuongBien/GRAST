"""Seed global RNGs and explicit NumPy generators; configure deterministic PyTorch."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def worker_environment(seed: int, deterministic: bool, num_threads: int) -> dict[str, str]:
    """Set hash seed before interpreter startup, not by mutating a running interpreter."""
    env = os.environ.copy()
    env.update(
        PYTHONHASHSEED=str(seed), OMP_NUM_THREADS=str(num_threads), MKL_NUM_THREADS=str(num_threads)
    )
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def seed_everything(
    seed: int, *, deterministic: bool = True, num_threads: int = 1
) -> np.random.Generator:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(num_threads) is not int or num_threads < 1:
        raise ValueError("num_threads must be positive")
    if deterministic:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("Invalid CUBLAS_WORKSPACE_CONFIG for deterministic execution")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return np.random.default_rng(seed)


def seed_worker(worker_id: int) -> None:
    """Use with a seeded torch.Generator and DataLoader.worker_init_fn."""
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def capture_rng_state(generator: np.random.Generator) -> dict:
    import numpy as np
    import torch

    legacy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [legacy[0], legacy[1].tolist(), legacy[2], legacy[3], legacy[4]],
        "numpy_generator": generator.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state: dict, generator: np.random.Generator) -> None:
    import numpy as np
    import torch

    cuda = state["torch_cuda"]
    if cuda and (not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()):
        raise ValueError("Checkpoint CUDA RNG topology differs from the current runtime")
    legacy = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((legacy[0], np.array(legacy[1], dtype=np.uint32), *legacy[2:]))
    generator.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)
