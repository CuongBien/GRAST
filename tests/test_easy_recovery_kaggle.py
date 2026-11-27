"""Kaggle's shipped cells: mounted inputs, streaming, real smoke/resume and outputs."""

import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
import test_easy_recovery

from llm_scratch.data.datasets import file_sha256
from llm_scratch.easy_recovery.bundle import create

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/easy_recovery_kaggle.ipynb"


def cells():
    return [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    ]


def helpers(tmp_path):
    ns = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "signal": signal,
        "subprocess": subprocess,
        "sys": sys,
        "tempfile": tempfile,
        "zipfile": zipfile,
        "LAST_RUN_DIRECTORY": None,
        "CONSOLE_LOG": tmp_path / "console.log",
        "CHILD_ENV": {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    constants = {
        "SOURCE_COMMIT",
        "TOTAL_UPDATES",
        "KIT_NAME",
        "KIT_SHA256",
        "MANIFEST_SHA256",
        "SOURCE_SHA256",
        "SOURCE_BYTES",
        "EXPECTED_FILES",
    }
    for source in cells():
        nodes = [
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef)
            or (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
            )
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    recovery_inputs = test_easy_recovery.recovery_inputs.__wrapped__(tmp_path_factory)
    root = tmp_path_factory.mktemp("easy-kaggle-kit")
    wheel = root / "fixture.whl"
    wheel.write_bytes(b"transport-only fixture; installation is separately smoke-verified")
    archive = root / "kit.zip"
    info = create(
        recovery_inputs["prepared_root"],
        recovery_inputs["tokenizer"],
        recovery_inputs["config"],
        wheel,
        archive,
        "1ff2079f5833345b10ad5f852e3a297f17d072ab",
    )
    with zipfile.ZipFile(archive) as z:
        manifest_sha = hashlib.sha256(z.read("bundle_manifest.json")).hexdigest()
    return archive, info["sha256"], manifest_sha, recovery_inputs


def mounted(kit, tmp_path):
    archive, digest, manifest_sha, kwargs = kit
    ns = helpers(tmp_path)
    ns.update(
        KIT_SHA256=digest,
        MANIFEST_SHA256=manifest_sha,
        TOTAL_UPDATES=4,
        SOURCE_SHA256=file_sha256(kwargs["source_checkpoint"]),
        SOURCE_BYTES=kwargs["source_checkpoint"].stat().st_size,
    )
    root = tmp_path / "input/datasets/owner/easy/versions/1"
    root.mkdir(parents=True)
    path = root / ns["KIT_NAME"]
    path.write_bytes(archive.read_bytes())
    return ns, path, root


def test_cells_compile_pin_same_kit_and_never_require_colab_or_network(tmp_path):
    for i, source in enumerate(cells()):
        compile(source, f"easy-kaggle-cell-{i}", "exec")
    source = "\n".join(cells())
    ns = helpers(tmp_path)
    assert ns["KIT_SHA256"] == "c8b15e13ab936af43a7efc649fedc1191f210fa4d9a097c95bc63bb68ab606cf"
    assert (
        ns["MANIFEST_SHA256"] == "d58c11c093e80035a627dad338956b70514a8ef9dfa8121a54ec3c0e3231af98"
    )
    assert "SMOKE_ONLY = True" in source and 'RESUME = ""' in source
    assert 'Path("/kaggle/input")' in source
    assert 'Path("/kaggle/working/grast-easy-recovery")' in source
    assert not any(
        s in source
        for s in (
            "google.colab",
            "drive.mount",
            "files.download",
            "files.upload",
            "capture_output",
            '"git", "clone"',
        )
    )
    assert not any(
        isinstance(n, ast.ImportFrom) and (n.module or "").startswith("llm_scratch")
        for s in cells()
        for n in ast.walk(ast.parse(s))
    )
    install = ns["wheel_install_command"](tmp_path, tmp_path / "site")
    assert "--no-index" in install and "--no-deps" in install and "--target" in install
    assert "OUTPUT_INDEX = None" in source


def test_zip_expanded_nested_discovery_and_payloads(kit, tmp_path):
    ns, path, root = mounted(kit, tmp_path)
    assert ns["find_kit"](tmp_path / "input") == ("zip", path)
    expanded = root / "expanded"
    with zipfile.ZipFile(path) as z:
        z.extractall(expanded)
    with pytest.raises(ValueError, match="Multiple"):
        ns["find_kit"](tmp_path / "input")
    assert ns["prepare_kit"](tmp_path / "input", tmp_path / "scratch", expanded) == expanded
    path.unlink()
    assert ns["find_kit"](tmp_path / "input") == ("expanded", expanded)
    (expanded / "data/gate.jsonl").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="payload SHA256"):
        ns["prepare_kit"](tmp_path / "input", tmp_path / "scratch")


def test_explicit_paths_fail_without_falling_back_and_archive_rejects_traversal(kit, tmp_path):
    ns, path, root = mounted(kit, tmp_path)
    with pytest.raises(FileNotFoundError, match="Explicit"):
        ns["find_kit"](tmp_path / "input", root / "missing.zip")
    renamed = path.with_name("renamed.zip")
    path.rename(renamed)
    with pytest.raises(FileNotFoundError, match="Add Input"):
        ns["find_kit"](tmp_path / "input")
    assert ns["find_kit"](tmp_path / "input", renamed) == ("zip", renamed)
    malicious = root / "unsafe.zip"
    with zipfile.ZipFile(renamed) as source, zipfile.ZipFile(malicious, "w") as dest:
        for name in source.namelist():
            dest.writestr(name, source.read(name))
        dest.writestr("../outside", "forbidden")
    ns["KIT_SHA256"] = file_sha256(malicious)
    with pytest.raises(ValueError, match="Unsafe"):
        ns["prepare_kit"](tmp_path / "input", tmp_path / "scratch", malicious)
    malicious.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        ns["find_kit"](tmp_path / "input", malicious)


def test_selected_source_found_by_hash_and_explicit_mismatch_never_falls_back(tmp_path):
    ns = helpers(tmp_path)
    source = tmp_path / "renamed.pt"
    source.write_bytes(b"selected2200")
    wrong = tmp_path / "selected_model.pt"
    wrong.write_bytes(b"wrong")
    ns["SOURCE_BYTES"] = source.stat().st_size
    digest = file_sha256(source)
    assert ns["resolve_source"](tmp_path, digest) == source
    with pytest.raises(ValueError, match="expected SHA256"):
        ns["resolve_source"](tmp_path, digest, wrong)


@pytest.mark.parametrize(
    "major,supported,saved,expected",
    [
        (7, True, "", "float16"),
        (8, True, "", "bfloat16"),
        (8, False, "", "float16"),
        (8, True, "float16", "float16"),
    ],
)
def test_precision_requires_native_bf16_and_preserves_resume(
    tmp_path, major, supported, saved, expected
):
    ns = helpers(tmp_path)
    assert ns["select_precision"](major, supported, saved) == expected
    with pytest.raises(RuntimeError, match="native BF16"):
        ns["select_precision"](7, True, "bfloat16")


@pytest.mark.parametrize("code", [0, 3])
def test_stream_logs_exact_run_and_child_failures(tmp_path, capsys, code):
    ns = helpers(tmp_path)
    child = (
        "import json; print('live output',flush=True); "
        "print(json.dumps({'event':'easy_recovery_started',"
        "'run_directory':'exact-run'}),flush=True); "
        "raise SystemExit(" + str(code) + ")"
    )
    if code:
        with pytest.raises(subprocess.CalledProcessError):
            ns["run_logged"]([sys.executable, "-u", "-c", child])
    else:
        ns["run_logged"]([sys.executable, "-u", "-c", child])
    assert ns["LAST_RUN_DIRECTORY"] == "exact-run"
    assert "live output" in capsys.readouterr().out
    assert "live output" in ns["CONSOLE_LOG"].read_text(encoding="utf-8")


@pytest.mark.parametrize("unresponsive", [False, True])
def test_interrupt_waits_for_child_or_kills_before_return(tmp_path, unresponsive):
    ns = helpers(tmp_path)

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

        def poll(self):
            return None

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

    ns["subprocess"] = Subprocess
    with pytest.raises(KeyboardInterrupt):
        ns["run_logged"](["child"])
    assert process.signals == [signal.SIGINT] and process.killed == unresponsive


def test_failed_start_cannot_reuse_previous_output(tmp_path):
    source = next(s for s in cells() if "command = training_command(INPUTS" in s)
    nodes = [n for n in ast.parse(source).body if not isinstance(n, ast.FunctionDef)]
    ns = {
        "LAST_RUN_DIRECTORY": "old-run",
        "OUTPUT_INDEX": {"old": True},
        "INPUTS": tmp_path,
        "OUTPUT_ROOT": tmp_path,
        "PRECISION": "float32",
        "SMOKE_ONLY": True,
        "INIT_WEIGHTS": tmp_path,
        "RESUME": "",
        "CONSOLE_LOG": tmp_path / "console.log",
        "training_command": lambda *args, **kwargs: ["fake-child"],
    }

    def failed_start(*args):
        raise RuntimeError("failed before run started")

    ns["run_logged"] = failed_start
    with pytest.raises(RuntimeError, match="failed before run started"):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "kaggle-training-cell", "exec"), ns)
    assert ns["OUTPUT_INDEX"] is None and ns["LAST_RUN_DIRECTORY"] is None


def test_real_cpu_smoke_resume_and_weight_free_summary(kit, tmp_path):
    ns, archive, root = mounted(kit, tmp_path)
    inputs = ns["prepare_kit"](tmp_path / "input", tmp_path / "scratch")
    original = {
        p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()
    }
    source = root / "renamed-source.pt"
    source.write_bytes(kit[3]["source_checkpoint"].read_bytes())
    work = tmp_path / "working"
    work.mkdir()
    command = ns["training_command"](inputs, work / "runs", "float32", True, source, device="cpu")
    assert command[command.index("--stop-after") + 1] == "2"
    ns["run_logged"](command)
    run = Path(ns["LAST_RUN_DIRECTORY"])
    index = ns["export_output"](run, work)
    assert index["report"]["status"] == "controlled_stop" and not index["report"]["gate_evaluated"]
    assert ns["resume_settings"](index["resume_checkpoint"], inputs) == ("float32", 2)
    with pytest.raises(ValueError, match="full easy-recovery"):
        ns["resume_settings"](index["selected_model"], inputs)
    with pytest.raises(ValueError, match="Smoke starts fresh"):
        ns["training_command"](
            inputs,
            work / "runs",
            "float32",
            True,
            source,
            resume=index["resume_checkpoint"],
            device="cpu",
        )
    command = ns["training_command"](
        inputs,
        work / "runs",
        "float32",
        False,
        source,
        resume=index["resume_checkpoint"],
        device="cpu",
    )
    assert "--stop-after" not in command
    ns["LAST_RUN_DIRECTORY"] = None
    ns["run_logged"](command)
    final = ns["export_output"](ns["LAST_RUN_DIRECTORY"], work)
    assert final["report"]["status"] == "completed" and final["report"]["step"] == 4
    with zipfile.ZipFile(final["summary"]) as z:
        assert "artifacts/recovery_report.json" in z.namelist()
        assert not any(
            Path(n).suffix in {".pt", ".pth", ".ckpt", ".safetensors"} for n in z.namelist()
        )
    assert all(file_sha256(root / name) == digest for name, digest in original.items())
