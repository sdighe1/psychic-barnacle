"""Canonical filesystem locations for the project.

Everything is resolved relative to the repository root so the code works no
matter what the current working directory is (tests, scripts, the Streamlit
app, …).
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # src/wcpredictor
SRC_DIR = PACKAGE_DIR.parent                            # src
REPO_ROOT = SRC_DIR.parent                              # repo root

DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "config"

for _d in (DATA_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Saved artifacts
MODEL_PATH = OUTPUTS_DIR / "model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
PREDICTIONS_2026_PATH = OUTPUTS_DIR / "predictions_2026.json"
CALIBRATION_PLOT_PATH = OUTPUTS_DIR / "calibration.png"

WC2026_CONFIG_PATH = CONFIG_DIR / "wc2026.yaml"
