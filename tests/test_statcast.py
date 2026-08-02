"""Statcast expected-stats de-luck: parsing, luck-multiplier math, projection nudge."""
import numpy as np
import pandas as pd

import mlbpredictor.statcast as sc
from mlbpredictor.offense import woba
from mlbpredictor.statcast import (apply_luck, load_expected, luck_multipliers,
                                   multipliers_for_season)

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


# ------------------------------- apply_luck --------------------------- #
def test_apply_luck_scales_woba_up_and_down():
    up = apply_luck(LEAGUE, 1.10)
    down = apply_luck(LEAGUE, 0.90)
    assert abs(up.sum() - 1.0) < 1e-9 and abs(down.sum() - 1.0) < 1e-9
    assert woba(up) > woba(LEAGUE) > woba(down)
    # proportional target is approximately hit (hits carry the correction, OUT absorbs)
    assert abs(woba(up) - 1.10 * woba(LEAGUE)) < 0.01


def test_apply_luck_identity():
    assert np.allclose(apply_luck(LEAGUE, 1.0), LEAGUE)
    assert np.allclose(apply_luck(LEAGUE, None), LEAGUE)


# ----------------------------- multipliers ---------------------------- #
def test_luck_multipliers_regress_and_clip():
    df = pd.DataFrame({
        "retro_id": ["big", "tiny", "capped"],
        "pa": [600, 5, 600],
        "woba": [0.300, 0.300, 0.300],
        "est_woba": [0.360, 0.360, 0.900],      # unlucky; tiny sample; absurd gap
    })
    m = luck_multipliers(df, regress_pa=200.0, clip=0.15)
    # 600 PA: delta 0.06 * 600/800 = 0.045 -> mult 1.15 (right at the clip)
    assert abs(m["big"] - 1.15) < 1e-6
    # 5 PA regresses almost fully to 1.0
    assert 1.0 < m["tiny"] < 1.02
    # an extreme gap is clipped to +15%
    assert m["capped"] == 1.15


def test_multipliers_disabled_returns_empty(monkeypatch):
    # When disabled, no fetch is attempted and the maps are empty (host-independent).
    monkeypatch.setattr(sc, "_sc_cfg", lambda: {"enabled": False})
    called = {"n": 0}
    monkeypatch.setattr(sc, "load_expected", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    assert multipliers_for_season(2024) == ({}, {})
    assert called["n"] == 0                       # never attempted the network


def test_multipliers_enabled_but_unreachable_is_soft(monkeypatch):
    monkeypatch.setattr(sc, "_sc_cfg", lambda: {"enabled": True, "regress_pa": 200.0, "clip": 0.15})
    monkeypatch.setattr(sc, "load_expected", lambda kind, year: None)   # host unreachable
    assert multipliers_for_season(2024) == ({}, {})


# ------------------------------- parsing ------------------------------ #
_CSV = ("last_name, first_name,player_id,year,pa,woba,est_woba,est_ba\n"
        "Judge,Aaron,592450,2024,700,0.400,0.430,0.310\n"
        "Nobody,Zed,999999,2024,50,0.250,0.240,0.200\n")


def test_load_expected_parses_and_maps(monkeypatch):
    monkeypatch.setattr(sc, "cached_text", lambda url, name, **k: _CSV)
    monkeypatch.setattr(sc, "retro_for_mlbam",
                        lambda i: {592450: "judga001"}.get(int(i)) if i else None)
    df = load_expected("batter", 2024)
    assert list(df["retro_id"]) == ["judga001"]   # unmapped id dropped
    r = df.iloc[0]
    assert r["pa"] == 700 and abs(r["est_woba"] - 0.430) < 1e-9


def test_load_expected_missing_columns_returns_none(monkeypatch):
    monkeypatch.setattr(sc, "cached_text", lambda url, name, **k: "a,b,c\n1,2,3\n")
    assert load_expected("batter", 2024) is None
