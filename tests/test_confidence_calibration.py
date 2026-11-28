"""Synthetic end-to-end acceptance, real-pilot no-fit boundary and source locks."""

import json
from pathlib import Path

import pytest
import test_confidence_diagnostics
import torch

from llm_scratch.reasoning.calibrators import CalibrationSpec
from llm_scratch.research import confidence_calibration as calibration
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_runner_repeat_without_bank_or_weights(tmp_path, monkeypatch):
    cfg = json.loads((ROOT / "configs/evaluation/wp09_calibration.json").read_text())
    cfg["diagnostic_config"] = str(ROOT / "configs/evaluation/wp09_confidence_diagnostics.json")
    cfg["artifacts"]["root"] = str(tmp_path / "runs")
    path = tmp_path / "calibration.json"
    write_json(path, cfg)

    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic acceptance must not read banks or model checkpoints")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(calibration, "SampleBank", forbidden)
    a = calibration.run(path, synthetic_only=True)
    b = calibration.run(path, synthetic_only=True)
    for name in [
        "report.json",
        "synthetic/calibrator.json",
        "synthetic/assessment.predictions.jsonl",
        "synthetic/prediction_commit.json",
        "synthetic/assessment.scored.jsonl",
    ]:
        assert (a / "artifacts" / name).read_bytes() == (b / "artifacts" / name).read_bytes()
    r = json.loads((a / "artifacts/report.json").read_text())
    assert r["real_pilot"] is None and not r["real_calibrator_fitted"]
    s = r["synthetic"]
    assert s["fit_support"]["groups"] == s["assessment_support"]["groups"] == 128
    assert s["calibrated"]["brier"] < s["raw"]["brier"]
    assert s["calibrated"]["log_loss"] < s["raw"]["log_loss"]
    assert (a / "artifacts/synthetic/synthetic_calibration.png").is_file()
    assert not r["threshold_search_performed"] and not r["evaluation_scored"]


def test_real_pilot_scan_never_fits_and_outputs_cannot_enter_bank(tmp_path_factory, monkeypatch):
    root, config, bank = test_confidence_diagnostics.source.__wrapped__(tmp_path_factory)

    def forbidden(*args, **kwargs):
        raise AssertionError("Eligibility scan must not fit or load a model")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(calibration, "fit_calibrator", forbidden)
    report = calibration.real_eligibility(config, CalibrationSpec(), root / "eligibility")
    assert not report["real_calibrator_fitted"]
    assert not report["support"]["eligible"]
    assert not list((root / "eligibility").rglob("calibrator.json"))
    assert (root / "eligibility/trace/manifest.json").is_file()
    with pytest.raises(ValueError, match="outside bank"):
        calibration.real_eligibility(config, CalibrationSpec(), bank.root / "bad-output")
    bank.assert_unchanged()


def test_config_lock_rejects_changed_source_and_reserved_training_options(tmp_path):
    cfg = json.loads((ROOT / "configs/evaluation/wp09_calibration.json").read_text())
    cfg["diagnostic_config"] = str(ROOT / "configs/evaluation/wp09_confidence_diagnostics.json")
    cfg["artifacts"]["root"] = str(tmp_path / "runs")
    path = tmp_path / "bad.json"
    cfg["fit_partition"] = "evaluation"
    write_json(path, cfg)
    with pytest.raises(ValueError, match="Invalid frozen"):
        calibration.load_config(path)
    del cfg["fit_partition"]
    cfg["diagnostic_config_fingerprint"] = "f" * 64
    write_json(path, cfg)
    with pytest.raises(ValueError, match="checksum mismatch"):
        calibration.load_config(path)
    cfg["diagnostic_config_fingerprint"] = config_digest(
        json.loads(Path(cfg["diagnostic_config"]).read_text())
    )
    write_json(path, cfg)
    assert calibration.load_config(path).data["calibration_spec"]["l2"] == 0.01
    source_copy = tmp_path / "source-with-crlf.json"
    source_copy.write_bytes(
        Path(cfg["diagnostic_config"])
        .read_text()
        .replace("\r\n", "\n")
        .replace("\n", "\r\n")
        .encode()
    )
    cfg["diagnostic_config"] = str(source_copy)
    write_json(path, cfg)
    assert calibration.load_config(path).data["calibration_spec"]["l2"] == 0.01
