"""Colab input pinning, compilation, live streaming and interruption cleanup."""

import ast
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/easy_recovery_colab.ipynb"


def cells():
    return [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    ]


def stream_namespace(subprocess_module=subprocess):
    source = next(s for s in cells() if "def stream(" in s)
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "stream"
    )
    ns = {
        "os": os,
        "json": json,
        "subprocess": subprocess_module,
        "signal": signal,
        "LAST_RUN_DIRECTORY": None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "easy-notebook-stream", "exec"), ns)
    return ns


def test_notebook_compiles_defaults_to_smoke_and_pins_wheel_kit():
    for index, source in enumerate(cells()):
        compile(source, f"easy-notebook-{index}", "exec")
    source = "\n".join(cells())
    normalized = source.replace('"', "'")
    assert 'KIT_SHA256 = "' in source
    assert re.search(r'KIT_SHA256 = "[a-f0-9]{64}"', source)
    assert "SMOKE_ONLY = True" in source and "RESUME_CHECKPOINT = None" in source
    assert "'--no-deps'" in normalized and "'--stop-after', '2'" in normalized
    assert "saved['state']['training_config']['precision']" in normalized
    assert "'grast-0.1.0-py3-none-any.whl'" in normalized
    assert "archive.extractall(KIT_ROOT)" in source and "set(names) != EXPECTED_FILES" in source
    assert "subprocess.run" not in source and "capture_output" not in source
    assert "files.download" in source and "'.pt', '.pth', '.ckpt', '.safetensors'" in normalized


@pytest.mark.parametrize(
    "major,bf16,expected", [(7, True, "float16"), (8, True, "bfloat16"), (8, False, "float16")]
)
def test_t4_uses_fp16_even_when_bf16_emulation_is_available(major, bf16, expected):
    from types import SimpleNamespace

    source = next(s for s in cells() if "import torch" in s)
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PRECISION" for t in n.targets)
    )
    cuda = SimpleNamespace(
        get_device_capability=lambda _: (major, 0), is_bf16_supported=lambda: bf16
    )
    ns = {"torch": SimpleNamespace(cuda=cuda)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "easy-notebook-dtype", "exec"), ns)
    assert ns["PRECISION"] == expected


@pytest.mark.parametrize("exit_code", [0, 3])
def test_stream_tracks_exact_run_and_prints_child_output(exit_code, capsys):
    ns = stream_namespace()
    child = (
        "import json; print('live output',flush=True); "
        "print(json.dumps({'event':'easy_recovery_started',"
        "'run_directory':'exact-run'}),flush=True); "
        "raise SystemExit(" + str(exit_code) + ")"
    )
    command = [sys.executable, "-u", "-c", child]
    if exit_code:
        with pytest.raises(RuntimeError, match="exit code 3"):
            ns["stream"](command)
    else:
        ns["stream"](command)
    assert ns["LAST_RUN_DIRECTORY"] == "exact-run"
    assert "live output" in capsys.readouterr().out


@pytest.mark.parametrize("unresponsive", [False, True])
def test_stream_interrupt_waits_for_child_exit(unresponsive):
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
        PIPE, STDOUT, TimeoutExpired = subprocess.PIPE, subprocess.STDOUT, subprocess.TimeoutExpired

        @staticmethod
        def Popen(*args, **kwargs):
            return process

    ns = stream_namespace(Subprocess)
    with pytest.raises(KeyboardInterrupt):
        ns["stream"](["child"])
    assert process.signals == [signal.SIGINT]
    assert process.killed == unresponsive
