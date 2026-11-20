"""Exercise notebook source activation without processing a new editable-install .pth."""

import ast
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("wp07_sft_kaggle.ipynb", "wp07_sft_colab.ipynb")


def setup_source(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "def run_logged" in "".join(c["source"])
    )


def activation_source(name):
    tree = ast.parse(setup_source(name))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "activate_project_source"
    )
    return ast.unparse(function)


def run_kernel(name, tmp_path, body):
    env = os.environ.copy()
    # Like a kernel started before pip install -e: dependencies are available,
    # but site startup never evaluates the editable project's new .pth file.
    env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    script = activation_source(name) + "\n" + body
    return subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_running_kernel_can_import_project_and_launch_child_without_restarting(name, tmp_path):
    body = f"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

try:
    importlib.import_module("llm_scratch")
except ModuleNotFoundError as error:
    assert error.name == "llm_scratch"
else:
    raise AssertionError("Test must start without project import visibility")
repo = Path({str(ROOT)!r})
previous_pythonpath = os.environ["PYTHONPATH"]
source = activate_project_source(repo)
assert sys.path[0] == str(source)
assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(source), previous_pythonpath]
activate_project_source(repo)
assert sys.path.count(str(source)) == 1
assert os.environ["PYTHONPATH"].split(os.pathsep).count(str(source)) == 1
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.sft_bundle import extract_bundle, verify_directory
from llm_scratch.training.sft import main
import llm_scratch
assert Path(llm_scratch.__file__).resolve().parent == source / "llm_scratch"
child = subprocess.run(
    [sys.executable, "-S", "-c",
     "from llm_scratch.data.sft_bundle import extract_bundle; "
     "from llm_scratch.training.sft import main; "
     "import llm_scratch; print(llm_scratch.__file__)"],
    text=True, capture_output=True, check=True,
)
assert Path(child.stdout.strip()).resolve().parent == source / "llm_scratch"
print("running-kernel-and-child-imports-passed")
"""
    result = run_kernel(name, tmp_path, body)
    assert "Kernel imports verified:" in result.stdout
    assert "running-kernel-and-child-imports-passed" in result.stdout


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_bootstrap_rejects_missing_checkout_and_foreign_cached_module(name, tmp_path):
    body = f"""
import sys
from pathlib import Path
from types import ModuleType

try:
    activate_project_source(Path("missing-checkout"))
except FileNotFoundError as error:
    assert "setup cell first" in str(error)
else:
    raise AssertionError("Missing checkout was accepted")
foreign = ModuleType("llm_scratch")
foreign.__file__ = str(Path("other-project/llm_scratch/__init__.py").resolve())
sys.modules["llm_scratch"] = foreign
before = list(sys.path)
try:
    activate_project_source(Path({str(ROOT)!r}))
except RuntimeError as error:
    assert "Restart the kernel" in str(error)
else:
    raise AssertionError("Foreign cached project module was accepted")
assert sys.path == before
print("invalid-kernel-state-rejected")
"""
    assert "invalid-kernel-state-rejected" in run_kernel(name, tmp_path, body).stdout


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_shipped_setup_activates_source_after_install_and_checks_child_imports(name):
    source = setup_source(name)
    assert source.index('"pip", "install", "--no-deps"') < source.index(
        "PROJECT_SOURCE = activate_project_source(REPO)"
    )
    assert "Subprocess imports verified:" in source
    assert "from llm_scratch.data.sft_bundle import extract_bundle" in source
    assert "from llm_scratch.training.sft import main" in source
    compile(source, name, "exec")
