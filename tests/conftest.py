"""Configure cuBLAS before any test imports torch or initializes a CUDA context."""

import os

# WP02 tests enable deterministic algorithms later in the same pytest process.
# Setting this after earlier model tests have initialized cuBLAS is too late.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
