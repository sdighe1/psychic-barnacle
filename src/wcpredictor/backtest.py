"""Backtesting metrics and evaluation helpers.

Primary metric is the **Ranked Probability Score** (RPS), the standard proper
score for ordered outcomes (home win → draw → away win) in football forecasting
(Constantinou & Fenton, 2012). We also report log-loss, multiclass Brier,
accuracy and exact-score hit-rate, plus reliability data for a calibration plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_LABEL_IDX = {"H": 0, "D": 1, "A": 2}
_PROB_COLS = ["p_home", "p_draw", "p_away"]


def _onehot(y_idx: np.ndarray) -> np.ndarray:
    O = np.zeros((len(y_idx), 3))
    O[np.arange(len(y_idx)), y_idx] = 1.0
    return O


def rps(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Mean Ranked Probability Score for ordered [home, draw, away] (lower better).

    RPS = 1/(r-1) · Σ_{i=1}^{r-1} (CumP_i − CumO_i)²  with r = 3 outcomes.
    """
    O = _onehot(y_idx)
    cp = np.cumsum(probs, axis=1)
    co = np.cumsum(O, axis=1)
    return float(np.mean(((cp[:, :-1] - co[:, :-1]) ** 2).sum(axis=1)) / (3 - 1))


def log_loss(probs: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx])))


def brier(probs: np.ndarray, y_idx: np.ndarray) -> float:
    O = _onehot(y_idx)
    return float(np.mean(((probs - O) ** 2).sum(axis=1)))


def accuracy(probs: np.ndarray, y_idx: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == y_idx))


def evaluate(frame: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """Compute all metrics for a prediction ``frame`` against actuals in ``feat``."""
    y_idx = feat["outcome"].map(_LABEL_IDX).to_numpy()
    probs = frame[_PROB_COLS].to_numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)
    # Exact-score hit-rate from rounded expected goals.
    ph = np.rint(frame["exp_home"].to_numpy()).astype(int)
    pa = np.rint(frame["exp_away"].to_numpy()).astype(int)
    hit = np.mean((ph == feat["home_score"].to_numpy()) & (pa == feat["away_score"].to_numpy()))
    return {
        "rps": round(rps(probs, y_idx), 4),
        "log_loss": round(log_loss(probs, y_idx), 4),
        "brier": round(brier(probs, y_idx), 4),
        "accuracy": round(accuracy(probs, y_idx), 4),
        "exact_score": round(float(hit), 4),
        "n": int(len(feat)),
    }


def reliability_data(frame: pd.DataFrame, feat: pd.DataFrame, n_bins: int = 10) -> dict:
    """Pooled reliability curve across all three classes (for a calibration plot)."""
    y_idx = feat["outcome"].map(_LABEL_IDX).to_numpy()
    O = _onehot(y_idx)
    probs = frame[_PROB_COLS].to_numpy()
    p = probs.reshape(-1)
    o = O.reshape(-1)
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mean_pred, mean_obs, counts = [], [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() > 0:
            mean_pred.append(float(p[m].mean()))
            mean_obs.append(float(o[m].mean()))
            counts.append(int(m.sum()))
    return {"mean_pred": mean_pred, "mean_obs": mean_obs, "counts": counts}
