"""Notebook source pin, input identities, cell syntax and mandatory subprocess streaming."""

import ast
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp07_curriculum_colab.ipynb"


def cells():
    return [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    ]


def test_curriculum_notebook_compiles_pins_source_and_keeps_weights_separate():
    sources = cells()
    for i, source in enumerate(sources):
        compile(source, f"curriculum-notebook-cell-{i}", "exec")
    combined = "\n".join(sources)
    assert re.search(r'SOURCE_COMMIT = "[a-f0-9]{40}"', combined)
    assert '"--no-deps"' in combined
    assert "5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42" in combined
    assert "02c9e6a9d969365921d00cc119e614295a25f1d1bde53fb7808141f2ffcee217" in combined
    assert "RESUME_CHECKPOINT = None" in combined and "STOP_AFTER = None" in combined
    assert 'saved["state"]["training_config"]["precision"]' in combined
    assert "files.download" in combined and '".pt", ".pth", ".ckpt"' in combined
    assert "capture_output" not in combined and "subprocess.run" not in combined


@pytest.mark.parametrize("exit_code", [0, 3])
def test_notebook_stream_prints_lines_tracks_exact_run_and_reports_failure(exit_code, capsys):
    source = next(s for s in cells() if "def stream(" in s)
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "stream"
    )
    ns = {
        "os": os,
        "json": json,
        "subprocess": subprocess,
        "signal": signal,
        "LAST_RUN_DIRECTORY": None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-stream", "exec"), ns)
    child = (
        "import json; print('live output',flush=True); "
        "print(json.dumps({'event':'curriculum_started','run_directory':'exact-run'}),flush=True); "
        "raise SystemExit(" + str(exit_code) + ")"
    )
    if exit_code:
        with pytest.raises(RuntimeError, match="exit code 3"):
            ns["stream"]([sys.executable, "-u", "-c", child])
    else:
        ns["stream"]([sys.executable, "-u", "-c", child])
    assert ns["LAST_RUN_DIRECTORY"] == "exact-run"
    assert "live output" in capsys.readouterr().out


@pytest.mark.parametrize("unresponsive", [False, True])
def test_notebook_interrupt_stops_child_before_return(unresponsive):
    source = next(s for s in cells() if "def stream(" in s)
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "stream"
    )

    class Process:
        signals = []
        killed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def stdout(self):
            raise KeyboardInterrupt

        def send_signal(self, value):
            self.signals.append(value)

        def wait(self, timeout=None):
            if unresponsive and timeout is not None:
                raise subprocess.TimeoutExpired("child", timeout)
            return 0

        def kill(self):
            self.killed = True

    process = Process()

    class Subprocess:
        PIPE = subprocess.PIPE
        STDOUT = subprocess.STDOUT
        TimeoutExpired = subprocess.TimeoutExpired

        @staticmethod
        def Popen(*args, **kwargs):
            return process

    ns = {
        "os": os,
        "json": json,
        "subprocess": Subprocess,
        "signal": signal,
        "LAST_RUN_DIRECTORY": None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-stream", "exec"), ns)
    with pytest.raises(KeyboardInterrupt):
        ns["stream"](["child"])
    assert process.signals == [signal.SIGINT]
    assert process.killed == unresponsive
