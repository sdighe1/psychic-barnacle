"""Measure projection accuracy and blend sources by it.

A projection source is only as good as its track record, so we **backtest**:
rebuild a season's projections from data available *before* it, then compare to
what actually happened. Those metrics both populate the app's "model card" and
set the **weights** for an accuracy-weighted consensus when more than one source
is present.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import scoring
from .projections import build_projections
from .scoring import OVERRIDE_COLUMN, STAT_COLUMNS


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def backtest_model(
    history: pd.DataFrame, test_season: int, pool: int = 175, fmt: str = "half_ppr"
) -> Dict:
    """Project ``test_season`` from prior seasons and score it against actuals.

    Returns MAE / RMSE / rank-correlation over the top ``pool`` projected
    players, plus per-position MAE and the sample size.
    """
    train = history[history["season"] < test_season]
    actual_rows = history[history["season"] == test_season]
    if train.empty or actual_rows.empty:
        return {}

    proj = build_projections(train, target_season=test_season)
    proj["proj_pts"] = scoring.project_points(proj, fmt)

    actual = actual_rows.copy()
    actual["actual_pts"] = scoring.project_points(actual, fmt)
    merged = proj.merge(actual[["player_id", "actual_pts"]], on="player_id", how="left")
    merged["actual_pts"] = merged["actual_pts"].fillna(0.0)

    top = merged.sort_values("proj_pts", ascending=False).head(pool)
    err = (top["proj_pts"] - top["actual_pts"]).to_numpy()
    metrics = {
        "test_season": int(test_season),
        "n": int(len(top)),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "rank_corr": round(_spearman(top["proj_pts"].to_numpy(), top["actual_pts"].to_numpy()), 3),
        "pos_mae": {
            pos: round(float(np.mean(np.abs(g["proj_pts"] - g["actual_pts"]))), 2)
            for pos, g in top.groupby("position")
        },
    }
    return metrics


def accuracy_weights(mae_by_source: Dict[str, float]) -> Dict[str, float]:
    """Turn per-source MAE into normalised weights (lower error -> more weight)."""
    inv = {s: 1.0 / max(m, 1e-6) for s, m in mae_by_source.items() if m and m == m}
    total = sum(inv.values())
    if total <= 0:
        n = len(mae_by_source) or 1
        return {s: 1.0 / n for s in mae_by_source}
    return {s: v / total for s, v in inv.items()}


def weighted_consensus(
    sources: Dict[str, pd.DataFrame], weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Blend several projection frames into one accuracy-weighted stat line.

    Frames are aligned on ``player_id``; component stats are averaged with the
    given weights (renormalised per player over the sources that actually cover
    them). Player meta comes from the highest-weighted covering source.
    """
    names = list(sources)
    if not names:
        raise ValueError("no sources to blend")
    weights = weights or {s: 1.0 / len(names) for s in names}

    stat_cols = STAT_COLUMNS + [OVERRIDE_COLUMN]
    # accumulate weighted sums and weight totals per player
    acc: Dict[str, Dict] = {}
    for name in names:
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        for _, row in sources[name].iterrows():
            pid = str(row["player_id"]) if "player_id" in row and pd.notna(row["player_id"]) else None
            if not pid:
                continue
            rec = acc.setdefault(pid, {"w": 0.0, "meta_w": -1.0, "meta": None,
                                       **{c: 0.0 for c in stat_cols}, "cov": {c: 0.0 for c in stat_cols}})
            for c in stat_cols:
                val = row.get(c, np.nan)
                if pd.notna(val):
                    rec[c] += w * float(val)
                    rec["cov"][c] += w
            rec["w"] += w
            if w > rec["meta_w"]:
                rec["meta_w"] = w
                rec["meta"] = {k: row.get(k) for k in ("player", "position", "team", "age", "proj_games")}

    rows = []
    for pid, rec in acc.items():
        out = {"player_id": pid}
        out.update(rec["meta"] or {})
        for c in stat_cols:
            cov = rec["cov"][c]
            out[c] = (rec[c] / cov) if cov > 0 else np.nan
        out["source"] = "consensus"
        rows.append(out)
    return pd.DataFrame(rows)
