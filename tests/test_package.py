"""Smoke checks for installed package metadata and importability, not model behavior."""

import importlib
import pkgutil
from importlib.metadata import version

import llm_scratch


def test_installed_version_matches_package() -> None:
    assert version("grast") == llm_scratch.__version__


def test_package_modules_import_without_runtime_dependencies() -> None:
    for module in pkgutil.walk_packages(llm_scratch.__path__, prefix="llm_scratch."):
        importlib.import_module(module.name)
