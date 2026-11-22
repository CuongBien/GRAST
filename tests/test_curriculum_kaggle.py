"""Exercise the shipped Kaggle adapter with real subprocesses and offline curriculum inputs."""

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
import torch

from llm_scratch.curriculum.acceptance import fixture
from llm_scratch.curriculum.bundle import create
from llm_scratch.data.datasets import file_sha256

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp07_curriculum_kaggle.ipynb"


def cells():
    return [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    ]


def helpers(tmp_path):
    namespace = {
        "Path": Path,
        "sys": sys,
        "os": os,
        "hashlib": hashlib,
        "json": json,
        "subprocess": subprocess,
        "signal": signal,
        "tempfile": tempfile,
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
        "BUNDLE_NAME",
        "BUNDLE_SHA256",
        "MANIFEST_SHA256",
        "SFT_SHA256",
    }
    for source in cells():
        selected = [
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef)
            or (
                isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
            )
        ]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("curriculum-kaggle-fixture")
    config, data, tokenizer, weights = fixture(root / "inputs", ROOT)
    path = root / "fixture.zip"
    info = create(data, tokenizer, path)
    with zipfile.ZipFile(path) as z:
        manifest_sha = hashlib.sha256(z.read("bundle_manifest.json")).hexdigest()
    return config, path, weights, info["sha256"], manifest_sha


def mounted(bundle, tmp_path):
    config, archive, weights, digest, manifest_sha = bundle
    ns = helpers(tmp_path)
    ns.update(BUNDLE_SHA256=digest, MANIFEST_SHA256=manifest_sha)
    root = tmp_path / "input/datasets/owner/data/versions/1"
    root.mkdir(parents=True)
    path = root / ns["BUNDLE_NAME"]
    path.write_bytes(archive.read_bytes())
    return ns, path, root


def test_cells_compile_pin_same_training_source_and_require_no_interactive_dialogs(tmp_path):
    for i, source in enumerate(cells()):
        compile(source, f"kaggle-curriculum-cell-{i}", "exec")
    combined = "\n".join(cells())
    ns = helpers(tmp_path)
    assert ns["SOURCE_COMMIT"] == "2383ab9a2900dc276eec7305a7e7d8cdfc5f2903"
    assert ns["BUNDLE_SHA256"] == "02c9e6a9d969365921d00cc119e614295a25f1d1bde53fb7808141f2ffcee217"
    assert ns["SFT_SHA256"] == "5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42"
    assert "SMOKE_ONLY = True" in combined and 'RESUME = ""' in combined
    assert (
        'Path("/kaggle/input")' in combined
        and 'Path("/kaggle/working/grast-wp07-curriculum")' in combined
    )
    assert not any(
        token in combined
        for token in (
            "google.colab",
            "files.upload",
            "files.download",
            "drive.mount",
            "capture_output",
        )
    )
    assert not any(
        isinstance(n, ast.ImportFrom) and (n.module or "").startswith("llm_scratch")
        for s in cells()
        for n in ast.walk(ast.parse(s))
    )
    assert 'CHILD_ENV["PYTHONPATH"]' in combined and '"--no-deps"' in combined


def test_zip_discovery_and_renamed_explicit_archive(bundle, tmp_path):
    ns, path, root = mounted(bundle, tmp_path)
    assert ns["find_bundle"](tmp_path / "input") == ("zip", path)
    renamed = root / "renamed.zip"
    path.rename(renamed)
    with pytest.raises(FileNotFoundError, match="Add Input"):
        ns["find_bundle"](tmp_path / "input")
    assert ns["find_bundle"](tmp_path / "input", renamed) == ("zip", renamed)


def test_expanded_layout_checked_by_manifest_identity_and_payloads(bundle, tmp_path):
    ns, path, root = mounted(bundle, tmp_path)
    with zipfile.ZipFile(path) as z:
        z.extractall(root / "expanded")  # Authored fixture, created by the tested bundle creator.
    path.unlink()
    assert ns["prepare_inputs"](tmp_path / "input", tmp_path / "scratch") == root / "expanded"
    (root / "expanded/tokenizer/vocab.json").write_bytes(b"{}")
    with pytest.raises(subprocess.CalledProcessError):
        ns["prepare_inputs"](tmp_path / "input", tmp_path / "scratch")


def test_bundle_ambiguity_corruption_and_explicit_path_never_falls_back(bundle, tmp_path):
    ns, path, root = mounted(bundle, tmp_path)
    with zipfile.ZipFile(path) as z:
        z.extractall(root / "expanded")
    with pytest.raises(ValueError, match="Multiple"):
        ns["find_bundle"](tmp_path / "input")
    assert ns["find_bundle"](tmp_path / "input", path) == ("zip", path)
    with pytest.raises(FileNotFoundError, match="Explicit"):
        ns["find_bundle"](tmp_path / "input", root / "missing.zip")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        ns["find_bundle"](tmp_path / "input", path)
    manifest = root / "expanded/bundle_manifest.json"
    meta = json.loads(manifest.read_text())
    meta["dataset_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="SHA256"):
        ns["find_bundle"](tmp_path / "input", manifest.parent)


@pytest.mark.parametrize("explicit", [False, True])
def test_sft_resolver_accepts_renamed_weights_by_hash(tmp_path, explicit):
    ns = helpers(tmp_path)
    (tmp_path / "final.pt").write_bytes(b"wrong-pretraining")
    correct = tmp_path / "renamed-sft.pt"
    correct.write_bytes(b"sft500")
    assert (
        ns["resolve_sft_checkpoint"](tmp_path, file_sha256(correct), correct if explicit else "")
        == correct
    )


def test_missing_wrong_and_explicit_sft_fail_without_silent_fallback(tmp_path):
    ns = helpers(tmp_path)
    (tmp_path / "summary.zip").write_bytes(b"no-weights")
    with pytest.raises(FileNotFoundError, match="do not contain model weights"):
        ns["resolve_sft_checkpoint"](tmp_path, "0" * 64)
    valid = tmp_path / "valid.pt"
    valid.write_bytes(b"valid")
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Expected SHA256"):
        ns["resolve_sft_checkpoint"](tmp_path, file_sha256(valid), wrong)
    with pytest.raises(FileNotFoundError, match="Explicit"):
        ns["resolve_sft_checkpoint"](tmp_path, file_sha256(valid), tmp_path / "missing.pt")


@pytest.mark.parametrize(
    "major,supported,saved,expected",
    [
        (6, False, "", "float16"),
        (7, True, "", "float16"),
        (8, True, "", "bfloat16"),
        (8, False, "", "float16"),
        (8, True, "float16", "float16"),
        (7, True, "float32", "float32"),
    ],
)
def test_precision_preserves_resume_and_requires_native_bf16(
    tmp_path, major, supported, saved, expected
):
    assert helpers(tmp_path)["select_precision"](major, supported, saved) == expected


def test_unsupported_bf16_resume_is_not_silently_changed(tmp_path):
    with pytest.raises(RuntimeError, match="native BF16"):
        helpers(tmp_path)["select_precision"](7, True, "bfloat16")


@pytest.mark.parametrize(
    "smoke,resume_step,expected_goal",
    [(True, None, 2), (True, 2, 4), (True, 2799, 2800), (False, None, None), (False, 4, None)],
)
def test_smoke_full_resume_commands_keep_original_recipe(
    tmp_path, smoke, resume_step, expected_goal
):
    ns = helpers(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"path-only")
    kwargs = (
        {"resume": checkpoint, "resume_step": resume_step, "init_checkpoint": None}
        if resume_step is not None
        else {"init_checkpoint": checkpoint}
    )
    cmd = ns["training_command"](ROOT, tmp_path, tmp_path / "runs", "float16", smoke, **kwargs)
    assert cmd[cmd.index("--config") + 1] == ROOT / "configs/training/wp07_curriculum.toml"
    assert ("--resume" in cmd) == (resume_step is not None)
    assert ("--init-checkpoint" in cmd) == (resume_step is None)
    if expected_goal is None:
        assert "--stop-after" not in cmd
    else:
        assert int(cmd[cmd.index("--stop-after") + 1]) == expected_goal


def test_resume_header_distinguishes_selected_and_old_sft(bundle, tmp_path):
    ns = helpers(tmp_path)
    with pytest.raises(ValueError, match="full curriculum"):
        ns["resume_settings"](bundle[2])  # Valid old SFT checkpoint, not curriculum.
    selected = tmp_path / "selected_model.pt"
    torch.save({"format": "grast-curriculum-selected-v1"}, selected)
    with pytest.raises(ValueError, match="full curriculum"):
        ns["resume_settings"](selected)
    with pytest.raises(FileNotFoundError, match="Attach previous"):
        ns["resume_settings"](tmp_path / "missing.pt")


@pytest.mark.parametrize("exit_code", [0, 3])
def test_stream_records_output_and_exact_run_even_with_non_json_objects(
    tmp_path, capsys, exit_code
):
    ns = helpers(tmp_path)
    code = (
        "import sys,json;print('42',flush=True);"
        "print('live stderr',file=sys.stderr,flush=True);"
        "print(json.dumps({'event':'curriculum_started','run_directory':'chosen-run'}),flush=True);"
        "raise SystemExit(" + str(exit_code) + ")"
    )
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError):
            ns["run_logged"]([sys.executable, "-u", "-c", code])
    else:
        ns["run_logged"]([sys.executable, "-u", "-c", code])
    assert ns["LAST_RUN_DIRECTORY"] == "chosen-run"
    assert (
        "live stderr" in capsys.readouterr().out and "live stderr" in ns["CONSOLE_LOG"].read_text()
    )


@pytest.mark.parametrize("stuck", [False, True])
def test_stop_waits_for_child_and_kills_when_needed(tmp_path, stuck):
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
            if stuck and timeout is not None:
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

    ns["subprocess"] = Subprocess
    with pytest.raises(KeyboardInterrupt):
        ns["run_logged"](["child"])
    assert process.signals == [signal.SIGINT] and process.killed == stuck


def test_offline_adapter_extracts_launches_smoke_and_resumes_same_runner(bundle, tmp_path):
    config, _, weights, _, _ = bundle
    ns, path, root = mounted(bundle, tmp_path)
    inputs = ns["prepare_inputs"](tmp_path / "input", tmp_path / "scratch")
    selected = ns["resolve_sft_checkpoint"](weights.parent, file_sha256(weights), weights)
    command = ns["training_command"](
        ROOT,
        inputs,
        tmp_path / "runs",
        "float32",
        True,
        init_checkpoint=selected,
        device="cpu",
        recipe_path=config,
    )
    ns["run_logged"](command, cwd=ROOT)
    first = Path(ns["LAST_RUN_DIRECTORY"])
    assert json.loads((first / "artifacts/curriculum_report.json").read_text())["step"] == 2
    precision, step = ns["resume_settings"](first / "checkpoints/latest.pt")
    assert (precision, step) == ("float32", 2)
    ns["LAST_RUN_DIRECTORY"] = None
    command = ns["training_command"](
        ROOT,
        inputs,
        tmp_path / "runs",
        precision,
        False,
        resume=first / "checkpoints/latest.pt",
        resume_step=step,
        device="cpu",
        recipe_path=config,
    )
    ns["run_logged"](command, cwd=ROOT)
    resumed = Path(ns["LAST_RUN_DIRECTORY"])
    report = json.loads((resumed / "artifacts/curriculum_report.json").read_text())
    assert report["step"] == 6 and report["phase_windows"] == {"copy": 2, "single": 2, "multi": 2}
    assert resumed != first and (resumed / "checkpoints/selected_model.pt").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GPU preflight requires CUDA")
def test_gpu_preflight_executes_real_forward_backward(tmp_path):
    ns = helpers(tmp_path)
    precision = ns["gpu_preflight"](tmp_path)
    report = json.loads((tmp_path / "kaggle_environment.json").read_text())
    assert report["precision"] == precision and report["forward_backward_probe"] == "passed"
    assert report["device"] == "cuda:0" and not report["uses_distributed_training"]
