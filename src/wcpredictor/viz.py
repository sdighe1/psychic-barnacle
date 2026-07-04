"""Shared plotting helpers (matplotlib, Agg-safe for headless use)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette: blue/amber is the canonical colour-blind-safe categorical pair;
# draw is a neutral (not a hue). Sequential heatmap uses a single blue ramp.
HOME = "#2563eb"
AWAY = "#f59e0b"
DRAW = "#9ca3af"
ACCENT = "#2563eb"
GRID = "#d1d5db"
INK = "#111827"
MUTED = "#6b7280"


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def wdl_bar_figure(p_home: float, p_draw: float, p_away: float, home: str, away: str):
    """A single 100%-stacked horizontal bar: home win / draw / away win."""
    fig, ax = plt.subplots(figsize=(7, 1.35), dpi=120)
    segs = [(p_home, HOME, f"{home}  {p_home:.0%}"),
            (p_draw, DRAW, f"Draw  {p_draw:.0%}"),
            (p_away, AWAY, f"{away}  {p_away:.0%}")]
    left = 0.0
    for val, color, label in segs:
        ax.barh(0, val, left=left, color=color, edgecolor="white", linewidth=2, height=0.6)
        if val > 0.11:
            ax.text(left + val / 2, 0, label, ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        left += val
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def champion_bar_figure(teams: list[str], probs: list[float], max_rows: int = 16):
    """Sorted horizontal bar of championship probabilities, direct-labelled."""
    pairs = sorted(zip(teams, probs), key=lambda x: x[1])[-max_rows:]
    labels = [t for t, _ in pairs]
    vals = [p * 100 for _, p in pairs]
    fig, ax = plt.subplots(figsize=(6.4, max(3.2, 0.34 * len(labels))), dpi=120)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=ACCENT, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlabel("Championship probability (%)", fontsize=10, color=MUTED)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.012, yi, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(vals) * 1.14)
    _clean(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def save_calibration_plot(calib: dict, path, title: str = "Calibration (ensemble)") -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.plot([0, 1], [0, 1], "--", color=GRID, label="Perfect")
    ax.plot(calib["mean_pred"], calib["mean_obs"], "o-", color=ACCENT, label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def scoreline_heatmap_figure(matrix: np.ndarray, home: str, away: str, max_display: int = 6):
    """Heatmap of the scoreline probability matrix (home goals × away goals)."""
    M = matrix[: max_display + 1, : max_display + 1]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=120)
    im = ax.imshow(M * 100, cmap="Blues", origin="upper")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xlabel(f"{away} goals")
    ax.set_ylabel(f"{home} goals")
    ax.set_title("Scoreline probability (%)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j] * 100
            if v >= 1.0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > M.max() * 100 * 0.55 else "#111827", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
