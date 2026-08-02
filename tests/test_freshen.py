"""Current-season freshening: rate-vector building, Elo update, projection blend."""
import types

import numpy as np
import pandas as pd

import mlbpredictor.freshen as fr
from mlbpredictor.freshen import (fetch_season_rates, freshen_elo, freshen_projection,
                                  _rate_vec)
from mlbpredictor.ratings import EloRatingSystem

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


def test_rate_vec_builds_and_normalises():
    st = {"plateAppearances": 700, "hits": 180, "doubles": 30, "triples": 5,
          "homeRuns": 40, "baseOnBalls": 90, "hitByPitch": 5, "strikeOuts": 120}
    vec, pa = _rate_vec(st, "plateAppearances", "hitByPitch")
    assert pa == 700
    assert abs(vec.sum() - 1.0) < 1e-9
    # singles = 180 - 30 - 5 - 40 = 105
    assert abs(vec[0] * 700 - 105) < 1e-6
    assert abs(vec[3] * 700 - 40) < 1e-6         # HR


def test_rate_vec_empty_returns_none():
    assert _rate_vec({"plateAppearances": 0}, "plateAppearances", "hitByPitch") is None


def test_fetch_season_rates_maps_ids(monkeypatch):
    payload = {"stats": [{"splits": [
        {"player": {"id": 660271}, "stat": {"plateAppearances": 600, "hits": 200,
         "doubles": 40, "triples": 5, "homeRuns": 50, "baseOnBalls": 80,
         "hitByPitch": 3, "strikeOuts": 100}},
        {"player": {"id": 999}, "stat": {"plateAppearances": 500, "hits": 100}},  # no retro id
    ]}]}
    monkeypatch.setattr(fr, "_statsapi", lambda path, params: payload)
    monkeypatch.setattr(fr, "retro_for_mlbam",
                        lambda i: "ohtas001" if int(i) == 660271 else None)
    rates = fetch_season_rates(2026, "hitting")
    assert set(rates) == {"ohtas001"}
    vec, pa = rates["ohtas001"]
    assert pa == 600 and abs(vec.sum() - 1.0) < 1e-9


def test_freshen_elo_reverts_and_updates():
    games = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-01", "2025-04-02"]),
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "home_score": [8, 7], "away_score": [1, 2], "season": [2025, 2025]})
    elo = EloRatingSystem()
    elo.fit_transform(games)
    a0 = elo.rating("A")
    # revert only (no new games) pulls A back toward 1500
    reverted = freshen_elo(elo, pd.DataFrame(), revert=0.25)
    assert 1500 < reverted.rating("A") < a0
    # applying new current-season wins for A pushes it back up, base unchanged
    new = pd.DataFrame({"date": pd.to_datetime(["2026-04-01"]), "home_team": ["A"],
                        "away_team": ["B"], "home_score": [6], "away_score": [0],
                        "season": [2026]})
    updated = freshen_elo(elo, new, revert=0.0)
    assert updated.rating("A") > a0
    assert elo.rating("A") == a0                 # original not mutated


def test_freshen_projection_blends_toward_current():
    base = types.SimpleNamespace(
        bat_={"x": LEAGUE.copy()}, pit_={}, league_bat=LEAGUE.copy(), league_pit=LEAGUE.copy())
    hot = LEAGUE.copy(); hot[3] *= 3; hot /= hot.sum()      # a big HR surge
    fresh = freshen_projection(base, {"x": (hot, 600.0)}, {}, w_bat=400.0)
    assert fresh.bat_["x"][3] > base.bat_["x"][3]           # HR rate moved up
    assert base.bat_["x"][3] == LEAGUE[3]                   # original untouched
    # a tiny sample barely moves the projection
    fresh_small = freshen_projection(base, {"x": (hot, 5.0)}, {}, w_bat=400.0)
    assert abs(fresh_small.bat_["x"][3] - LEAGUE[3]) < abs(fresh.bat_["x"][3] - LEAGUE[3])
