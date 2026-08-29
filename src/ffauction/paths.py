"""Canonical filesystem locations for the project.

Everything is resolved relative to the repository root so the code works no
matter what the current working directory is (tests, scripts, the Streamlit
app, ...).
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # src/ffauction
SRC_DIR = PACKAGE_DIR.parent                            # src
REPO_ROOT = SRC_DIR.parent                              # repo root

DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "config"

for _d in (DATA_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Committed artifacts (regenerate with scripts/build_projections.py)
PROJECTIONS_PATH = OUTPUTS_DIR / "projections.csv"   # baseline player projections
META_PATH = OUTPUTS_DIR / "meta.json"                # data vintage + backtest accuracy

# Config
LEAGUE_CONFIG_PATH = CONFIG_DIR / "league.yaml"

# Curated non-skill baselines bundled with the repo (K / D-ST lack per-player
# component stats in the offensive dataset, so we ship a small ranked seed).
KDST_BASELINE_PATH = DATA_DIR / "kdst_baseline.csv"

# Where the app persists an in-progress draft so a refresh does not lose it.
DRAFT_STATE_PATH = OUTPUTS_DIR / "draft_state.json"
