"""WP06.5 CLI; also preserve stdlib profile API when cProfile imports this filename."""

if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(
        subprocess.call([sys.executable, "-m", "llm_scratch.inference.profiling", *sys.argv[1:]])
    )
else:
    import importlib.util
    import sysconfig
    from pathlib import Path

    _spec = importlib.util.spec_from_file_location(
        "_grast_stdlib_profile", Path(sysconfig.get_path("stdlib")) / "profile.py"
    )
    _stdlib = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_stdlib)
    globals().update(
        {name: value for name, value in vars(_stdlib).items() if not name.startswith("__")}
    )
