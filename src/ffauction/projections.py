"""Build a data-driven baseline projection for the upcoming season.

Method (per returning skill player)
-----------------------------------
1. **Recency-weighted per-game rates** over the last few seasons (most recent
   weighted highest, seasons with more games count more).
2. **Regression to the positional mean** -- rates are shrunk toward the
   position's average, so small-sample lines are pulled back to earth.
3. **Aging curve** -- counting stats are scaled by a position/age multiplier.
4. **Projected games** -- recent availability regressed toward a positional
   default, then totals = rate x games x aging.

The result is a projected **stat line** per player; fantasy points for any
scoring format are derived later by :mod:`ffauction.scoring`. This is an honest,
open-data baseline -- not a market-beating projection -- and is designed to be
blended with (or replaced by) provider numbers via :mod:`ffauction.providers`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .scoring import STAT_COLUMNS

# Recency weights, newest first (extended/truncated to the seasons available).
DEFAULT_RECENCY = [0.6, 0.3, 0.1]
# Shrinkage strength, in "effective games" of league-average play.
SHRINK_GAMES = 8.0
# Counting stats that get regressed & aged (everything scored except overrides).
_RATE_STATS = [c for c in STAT_COLUMNS if c != "st_td"] + ["carries", "targets"]

# Projected-games regression target by position.
_DEFAULT_GAMES = {"QB": 16.0, "RB": 15.0, "WR": 15.5, "TE": 15.5}


def _aging_multiplier(pos: str, age: float) -> float:
    """Rough age curve (1.0 = prime). Approximate, position-specific."""
    if age is None or age != age:      # NaN
        return 1.0
    a = float(age)
    if pos == "RB":
        pts = {21: 0.97, 22: 1.0, 26: 1.0, 27: 0.96, 28: 0.90, 29: 0.83, 30: 0.76, 31: 0.68}
    elif pos == "WR":
        pts = {21: 0.94, 23: 1.0, 28: 1.0, 30: 0.95, 31: 0.90, 32: 0.84, 33: 0.77, 34: 0.70}
    elif pos == "TE":
        pts = {22: 0.90, 24: 0.98, 26: 1.0, 30: 1.0, 31: 0.94, 32: 0.88, 33: 0.80, 34: 0.72}
    else:  # QB
        pts = {22: 0.95, 25: 1.0, 36: 1.0, 37: 0.95, 38: 0.90, 39: 0.83, 40: 0.75}
    ages = sorted(pts)
    if a <= ages[0]:
        return pts[ages[0]]
    if a >= ages[-1]:
        return pts[ages[-1]]
    for lo, hi in zip(ages, ages[1:]):
        if lo <= a <= hi:
            frac = (a - lo) / (hi - lo)
            return pts[lo] + frac * (pts[hi] - pts[lo])
    return 1.0


def build_projections(
    history: pd.DataFrame,
    target_season: int,
    recency: Optional[List[float]] = None,
    keep_per_pos: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Project ``target_season`` stat lines from a player-season ``history``
    table (see :func:`data.load_player_seasons`)."""
    recency = recency or DEFAULT_RECENCY
    keep_per_pos = keep_per_pos or {"QB": 40, "RB": 85, "WR": 95, "TE": 40}

    hist = history.copy()
    seasons = sorted(hist["season"].unique(), reverse=True)
    window = seasons[: len(recency)]
    weight_by_season = {s: w for s, w in zip(window, recency)}
    hist = hist[hist["season"].isin(window)].copy()
    hist["w"] = hist["season"].map(weight_by_season).astype(float)

    latest = max(window)
    # Only project players active in the most recent season (drops retirees).
    active_ids = set(hist.loc[hist["season"] == latest, "player_id"])
    hist = hist[hist["player_id"].isin(active_ids)]

    for c in _RATE_STATS:
        if c not in hist.columns:
            hist[c] = 0.0
        hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0.0)
    hist["games"] = pd.to_numeric(hist["games"], errors="coerce").fillna(0.0).clip(lower=0)

    # Weighted sums per player: sum(w*stat) and sum(w*games).
    for c in _RATE_STATS:
        hist[f"ws_{c}"] = hist["w"] * hist[c]
    hist["w_games"] = hist["w"] * hist["games"]
    hist["ws_st_td"] = hist["w"] * hist.get("st_td", 0.0)

    agg = {f"ws_{c}": "sum" for c in _RATE_STATS}
    agg["w_games"] = "sum"
    agg["ws_st_td"] = "sum"
    grp = hist.groupby("player_id").agg(agg).reset_index()

    # Player meta from the most recent season on record.
    meta = (
        hist.sort_values("season")
        .groupby("player_id")
        .agg(player=("player", "last"), position=("position", "last"),
             team=("team", "last"), age_latest=("age", "last"),
             season_latest=("season", "last"))
        .reset_index()
    )
    # Last-season games (for the availability projection).
    last_games = (
        hist[hist["season"] == latest][["player_id", "games"]]
        .rename(columns={"games": "last_games"})
    )
    df = grp.merge(meta, on="player_id").merge(last_games, on="player_id", how="left")
    df["last_games"] = df["last_games"].fillna(df["w_games"])

    # Positional mean per-game rate (games-weighted).
    pos_rate: Dict[str, Dict[str, float]] = {}
    for pos, g in df.groupby("position"):
        gg = max(g["w_games"].sum(), 1e-9)
        pos_rate[pos] = {c: g[f"ws_{c}"].sum() / gg for c in _RATE_STATS}

    # Shrink each rate toward the positional mean, then project totals.
    proj_age = df["age_latest"] + (target_season - df["season_latest"])
    games_proj = np.clip(
        0.65 * df["last_games"] + 0.35 * df["position"].map(_DEFAULT_GAMES).fillna(15.0),
        1.0, 17.0,
    )
    df["proj_games"] = games_proj.round(1)
    aging = pd.Series(
        [_aging_multiplier(p, a) for p, a in zip(df["position"], proj_age)],
        index=df.index,
    )

    out = pd.DataFrame({
        "player_id": df["player_id"], "player": df["player"], "position": df["position"],
        "team": df["team"], "age": proj_age.round(0), "proj_games": df["proj_games"],
    })
    for c in _RATE_STATS:
        pm = df["position"].map(lambda p: pos_rate.get(p, {}).get(c, 0.0))
        rate = (df[f"ws_{c}"] + SHRINK_GAMES * pm) / (df["w_games"] + SHRINK_GAMES)
        out[c] = (rate * df["proj_games"] * aging).round(1)

    # Special-teams TDs: light per-game projection, no aging.
    st_rate = df["ws_st_td"] / df["w_games"].replace(0, np.nan)
    out["st_td"] = (st_rate.fillna(0.0) * df["proj_games"]).round(2)
    out["proj_points_override"] = np.nan
    out["source"] = "history"

    # Trim to a useful depth per position.
    from . import scoring
    out["_pts"] = scoring.project_points(out, "half_ppr")
    parts = []
    for pos, n in keep_per_pos.items():
        parts.append(out[out["position"] == pos].sort_values("_pts", ascending=False).head(n))
    out = pd.concat(parts, ignore_index=True).drop(columns="_pts")
    return out
