"""Matplotlib figures for the training pipeline and the dashboard.

Kept dependency-light (matplotlib only). ``save_calibration_plot`` is used by the
trainer; the ``*_figure`` builders are used by the Streamlit app.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_calibration_plot(reliability: dict, path) -> None:
    """Reliability curve of the home-win probability."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    mp = reliability.get("mean_pred", [])
    mo = reliability.get("mean_obs", [])
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="perfect")
    ax.plot(mp, mo, "o-", color="#2563eb", label="model")
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title("Moneyline calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def win_prob_figure(home_team: str, away_team: str, p_home: float, p_away: float):
    fig, ax = plt.subplots(figsize=(5, 1.6))
    ax.barh([0], [p_away], color="#f59e0b", label=away_team)
    ax.barh([0], [p_home], left=[p_away], color="#2563eb", label=home_team)
    ax.set_xlim(0, 1); ax.set_yticks([])
    ax.text(p_away / 2, 0, f"{away_team} {p_away:.0%}", va="center", ha="center",
            color="white", fontsize=9, fontweight="bold")
    ax.text(p_away + p_home / 2, 0, f"{home_team} {p_home:.0%}", va="center", ha="center",
            color="white", fontsize=9, fontweight="bold")
    ax.set_title("Win probability")
    fig.tight_layout()
    return fig


def total_distribution_figure(total_runs: np.ndarray, weights: np.ndarray, line: float,
                              interval: tuple | None = None):
    """Histogram of the projected total, with the line and (optionally) a shaded
    credible interval ``(lo, hi)``."""
    fig, ax = plt.subplots(figsize=(5, 3))
    maxr = int(np.percentile(total_runs, 99)) + 1
    bins = np.arange(0, maxr + 1)
    ax.hist(total_runs, bins=bins, weights=weights, density=True,
            color="#2563eb", alpha=0.75)
    if interval is not None:
        ax.axvspan(interval[0], interval[1], color="#93c5fd", alpha=0.35,
                   label=f"{interval[0]}–{interval[1]} interval")
    ax.axvline(line, color="#ef4444", linestyle="--", label=f"line {line}")
    ax.set_xlabel("Total runs"); ax.set_ylabel("Probability")
    ax.set_title("Projected total runs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# Confidence-level display colors (light/neutral, theme-agnostic).
CONF_COLOR = {"High": "#16a34a", "Medium": "#d97706", "Low": "#dc2626"}


def confidence_badge_md(level: str) -> str:
    """A small colored HTML badge for a High/Medium/Low confidence level."""
    color = CONF_COLOR.get(level, "#6b7280")
    return (f"<span style='background:{color};color:white;padding:2px 8px;"
            f"border-radius:10px;font-size:0.8em;font-weight:600'>{level}</span>")
