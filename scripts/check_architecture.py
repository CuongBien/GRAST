"""Launch a fresh module process; scripts/profile.py must not shadow stdlib profile."""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "llm_scratch.research.architecture_acceptance", *sys.argv[1:]],
        check=True,
    )
