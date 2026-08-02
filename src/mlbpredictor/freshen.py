"""Freshen the deployed model with current-season form (live, via statsapi).

The committed model is trained through the last complete Retrosheet season (Retrosheet
publishes with a lag, so mid-season data is never in the offline mirror). For predicting
*this* season's games that leaves the model stale — it ignores months of current form.
This module, run at prediction time when statsapi is reachable, returns a **freshened copy**
of the predictor:

* **Elo** — reverted a touch toward the mean for the new season, then updated through every
  completed game of the current season (so ratings reflect current team strength);
* **Projections** — each player's multi-year Marcel projection is treated as a prior and
  updated with their current-season-to-date rate line (weighted by current PAs/BF).

It is idempotent: it always starts from the committed base model, so re-running each morning
simply re-applies the latest current-season data. Leak-safe for "today": only *completed*
games move Elo, and today's games aren't final yet.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import requests

from .config import load_config
from .ids import retro_for_mlbam
from .livedata import STATSAPI_TEAM_ID_TO_RETRO

_IDX = {  # PA_OUTCOMES order: 1B,2B,3B,HR,BB,HBP,SO,OUT
    "1B": 0, "2B": 1, "3B": 2, "HR": 3, "BB": 4, "HBP": 5, "SO": 6, "OUT": 7,
}


def _statsapi(path: str, params: dict) -> dict:
    cfg = load_config()["live"]
    r = requests.get(f"{cfg['statsapi_base']}{path}", params=params, timeout=cfg["timeout_seconds"] + 20)
    r.raise_for_status()
    return r.json()


def _num(st: dict, key: str) -> float:
    v = st.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _rate_vec(st: dict, denom_key: str, hbp_key: str) -> tuple[np.ndarray, float] | None:
    """Build a PA-outcome rate vector + sample size from a statsapi stat line."""
    denom = _num(st, denom_key)
    if denom < 1:
        return None
    h, d, t, hr = _num(st, "hits"), _num(st, "doubles"), _num(st, "triples"), _num(st, "homeRuns")
    single = max(h - d - t - hr, 0.0)
    bb, hbp, so = _num(st, "baseOnBalls"), _num(st, hbp_key), _num(st, "strikeOuts")
    out = max(denom - (single + d + t + hr + bb + hbp + so), 0.0)
    vec = np.array([single, d, t, hr, bb, hbp, so, out], dtype=float)
    s = vec.sum()
    return (vec / s, denom) if s > 0 else None


def fetch_season_results(season: int, start: str, end: str) -> pd.DataFrame:
    """Completed regular-season games in ``[start, end]`` (teams as Retrosheet codes)."""
    js = _statsapi("/schedule", {"sportId": 1, "startDate": start, "endDate": end, "gameType": "R"})
    rows = []
    for day in js.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            t = g.get("teams", {})
            h, a = t.get("home", {}), t.get("away", {})
            ht = STATSAPI_TEAM_ID_TO_RETRO.get(h.get("team", {}).get("id"))
            at = STATSAPI_TEAM_ID_TO_RETRO.get(a.get("team", {}).get("id"))
            hs, as_ = h.get("score"), a.get("score")
            if not ht or not at or hs is None or as_ is None:
                continue
            rows.append({"date": day["date"], "home_team": ht, "away_team": at,
                         "home_score": int(hs), "away_score": int(as_), "season": season})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


def fetch_season_rates(season: int, group: str) -> dict[str, tuple[np.ndarray, float]]:
    """Current-season rate vectors keyed by Retrosheet id (``group`` = hitting|pitching)."""
    js = _statsapi("/stats", {"stats": "season", "group": group, "season": season,
                              "sportId": 1, "gameType": "R", "limit": 4000, "playerPool": "all"})
    denom_key, hbp_key = ("plateAppearances", "hitByPitch") if group == "hitting" \
        else ("battersFaced", "hitBatsmen")
    out: dict[str, tuple[np.ndarray, float]] = {}
    for split in (js.get("stats", [{}])[0].get("splits", []) or []):
        rid = retro_for_mlbam(split.get("player", {}).get("id"))
        if not rid:
            continue
        rv = _rate_vec(split.get("stat", {}), denom_key, hbp_key)
        if rv:
            out[rid] = rv
    return out


def freshen_elo(base_elo, results: pd.DataFrame, revert: float = 0.20):
    """Copy of ``base_elo`` reverted toward the mean, then updated with ``results``."""
    elo = copy.deepcopy(base_elo)
    if revert > 0:
        for t in list(elo.ratings_):
            elo.ratings_[t] = 1500.0 + (1 - revert) * (elo.ratings_[t] - 1500.0)
    if not results.empty:
        elo.fit_transform(results)
    return elo


def _blend(base_map, league_vec, curr: dict, w_base: float) -> None:
    for rid, (vec, n) in curr.items():
        base = base_map.get(rid, league_vec)
        base_map[rid] = (base * w_base + vec * n) / (w_base + n)


def freshen_projection(base_ps, curr_bat: dict, curr_pit: dict,
                       w_bat: float = 400.0, w_pit: float = 500.0):
    """Copy of the projection with each player's rate updated by current-season data."""
    ps = copy.deepcopy(base_ps)
    _blend(ps.bat_, ps.league_bat, curr_bat, w_bat)
    _blend(ps.pit_, ps.league_pit, curr_pit, w_pit)
    return ps


def freshen_predictor(predictor, season: int, through_date: str):
    """Return ``(freshened_predictor, info)`` updated with ``season`` form through a date.

    On any statsapi failure the original predictor is returned unchanged (info notes it).
    """
    cfg = load_config().get("freshen", {})
    try:
        results = fetch_season_results(season, f"{season}-03-01", through_date)
        curr_bat = fetch_season_rates(season, "hitting")
        curr_pit = fetch_season_rates(season, "pitching")
    except requests.RequestException as exc:
        return predictor, {"freshened": False, "error": str(exc)}

    p = copy.deepcopy(predictor)
    p.fb.elo = freshen_elo(p.fb.elo, results, revert=float(cfg.get("elo_revert", 0.20)))
    p.fb.deploy_proj = freshen_projection(
        p.fb.deploy_proj, curr_bat, curr_pit,
        w_bat=float(cfg.get("w_base_pa", 400)), w_pit=float(cfg.get("w_base_bf", 500)))
    return p, {"freshened": True, "season": season, "games": int(len(results)),
               "batters": len(curr_bat), "pitchers": len(curr_pit)}
