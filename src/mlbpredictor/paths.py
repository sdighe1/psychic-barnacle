"""Canonical filesystem locations for the MLB predictor.

Everything resolves relative to the repository root so the code works regardless
of the current working directory (tests, scripts, the Streamlit app, ...). Mirrors
``wcpredictor/paths.py``.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # src/mlbpredictor
SRC_DIR = PACKAGE_DIR.parent                            # src
REPO_ROOT = SRC_DIR.parent                              # repo root

DATA_DIR = REPO_ROOT / "data" / "mlb"                   # raw + parsed cache (gitignored)
RETRO_DIR = DATA_DIR / "retrosheet"                     # downloaded event / gamelog files
CACHE_DIR = DATA_DIR / "cache"                          # parsed aggregates (parquet/csv)
OUTPUTS_DIR = REPO_ROOT / "outputs"                     # committed artifacts
CONFIG_DIR = REPO_ROOT / "config"

for _d in (DATA_DIR, RETRO_DIR, CACHE_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Saved artifacts (committed, small).
MODEL_PATH = OUTPUTS_DIR / "mlb_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "mlb_metrics.json"
CALIBRATION_PLOT_PATH = OUTPUTS_DIR / "mlb_calibration.png"
PREDICTIONS_LATEST_PATH = OUTPUTS_DIR / "predictions_latest.json"


def predictions_path(date_str: str) -> Path:
    """Path for a given day's slate predictions, e.g. ``predictions_20260712.json``."""
    return OUTPUTS_DIR / f"predictions_{date_str}.json"


# Config files.
MLB_CONFIG_PATH = CONFIG_DIR / "mlb.yaml"
SLATE_EXAMPLE_PATH = CONFIG_DIR / "slate.example.yaml"
