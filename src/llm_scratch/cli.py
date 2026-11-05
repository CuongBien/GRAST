"""Command-line entry points for reproducibility checks; no model execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_scratch.utils.config import load_config
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grast")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke", help="Create a run and verify two seeded processes plus RNG resume"
    )
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--seed", type=int)
    smoke.add_argument("--device", choices=("cpu", "cuda", "auto"))
    smoke.add_argument("--output-root", type=Path)
    check = commands.add_parser(
        "check-env", help="Inspect versions and enforce CPU/CUDA/GPU requirements"
    )
    check.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    check.add_argument("--require-gpu-name", default="")
    check.add_argument("--output", type=Path)
    prepare = commands.add_parser(
        "prepare-data", help="Prepare pinned WP03 datasets and fixed pilot"
    )
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path)
    prepare.add_argument("--offline", action="store_true")
    prepare.add_argument(
        "--math-only",
        action="store_true",
        help="Generate synthetic data only; not full WP03 acceptance",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-data":
            from llm_scratch.data.config import load_data_config
            from llm_scratch.data.prepare import prepare_data

            config = load_data_config(args.config, output_root=args.output_root)
            path = prepare_data(config, offline=args.offline, math_only=args.math_only)
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "scope": "math-only" if args.math_only else "full",
                        "run_directory": str(path),
                        "manifest": str(path / "artifacts/datasets/manifest.json"),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "check-env":
            report = inspect_environment(args.device, args.require_gpu_name)
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["validation"]["ok"] else 1
        from llm_scratch.smoke import run_smoke

        config = load_config(
            args.config, seed=args.seed, device=args.device, output_root=args.output_root
        )
        path = run_smoke(config)
        print(json.dumps({"status": "passed", "run_directory": str(path)}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, AssertionError, ImportError) as error:
        print(f"grast: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
