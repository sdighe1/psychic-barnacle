"""Backtest metrics for the MLB predictor.

Moneyline is a two-class problem (home/away), so we report **log-loss**, **Brier**
and **accuracy** plus a reliability curve for calibration. For the score we report
**MAE/RMSE on total runs** and on each side's runs, and the run-line / totals
hit behaviour. Lower log-loss / Brier / MAE is better; a home-field base rate is
the no-skill floor. (There is no draw class — MLB games always resolve.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def log_loss(p_home: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p_home, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p_home: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p_home - y) ** 2))


def accuracy(p_home: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p_home >= 0.5).astype(int) == y))


def evaluate_moneyline(p_home: np.ndarray, y: np.ndarray) -> dict:
    p_home = np.asarray(p_home, float)
    y = np.asarray(y, float)
    return {
        "log_loss": round(log_loss(p_home, y), 4),
        "brier": round(brier(p_home, y), 4),
        "accuracy": round(accuracy(p_home, y), 4),
        "n": int(len(y)),
    }


def evaluate_totals(exp_total: np.ndarray, actual_total: np.ndarray) -> dict:
    e = np.asarray(exp_total, float)
    a = np.asarray(actual_total, float)
    return {
        "runs_mae": round(float(np.mean(np.abs(e - a))), 3),
        "runs_rmse": round(float(np.sqrt(np.mean((e - a) ** 2))), 3),
        "mean_pred": round(float(e.mean()), 3),
        "mean_actual": round(float(a.mean()), 3),
    }


def evaluate_side_runs(exp_home, home_actual, exp_away, away_actual) -> dict:
    eh, ah = np.asarray(exp_home, float), np.asarray(home_actual, float)
    ea, aa = np.asarray(exp_away, float), np.asarray(away_actual, float)
    err = np.concatenate([eh - ah, ea - aa])
    return {
        "team_runs_mae": round(float(np.mean(np.abs(err))), 3),
        "team_runs_rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
    }


def reliability(p_home: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability curve of the home-win probability (for a calibration plot)."""
    p = np.asarray(p_home, float)
    y = np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mean_pred, mean_obs, counts = [], [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() > 0:
            mean_pred.append(float(p[m].mean()))
            mean_obs.append(float(y[m].mean()))
            counts.append(int(m.sum()))
    return {"mean_pred": mean_pred, "mean_obs": mean_obs, "counts": counts}
