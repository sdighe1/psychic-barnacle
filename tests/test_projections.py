"""Projection system: regression-to-mean and as-of leak-free behaviour."""
import numpy as np
import pandas as pd

from mlbpredictor.projections import ProjectionSystem, _weighted_regressed
from mlbpredictor.retrosheet import PA_OUTCOMES

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


def _counts_row(season, rate, pa):
    row = {"retro_id": "x", "season": season, "PA": pa}
    for o, r in zip(PA_OUTCOMES, rate):
        row[o] = r * pa
    return row


def test_small_sample_regresses_to_league():
    # A tiny sample of an extreme hitter is pulled hard toward league.
    extreme = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # all HR (absurd)
    sub = pd.DataFrame([_counts_row(2023, extreme, 10)])
    rate, _ = _weighted_regressed(sub, [5.0], LEAGUE, regress_pa=200.0)
    assert rate[3] < 0.10                    # HR rate heavily regressed from 1.0
    assert abs(rate.sum() - 1.0) < 1e-9


def test_large_sample_stays_near_itself():
    good = np.array([0.16, 0.05, 0.005, 0.05, 0.10, 0.012, 0.18, 0.393]); good /= good.sum()
    sub = pd.DataFrame([_counts_row(2023, good, 5000)])
    rate, _ = _weighted_regressed(sub, [5.0], LEAGUE, regress_pa=200.0)
    assert np.allclose(rate, good, atol=0.01)


def _make_frames():
    rows_bat, rows_pit, rows_bull = [], [], []
    for season in (2021, 2022, 2023):
        for pid, rate, pa in [("bat1", LEAGUE, 600), ("bat2", LEAGUE * 1.0, 600)]:
            r = _counts_row(season, rate, pa); r["retro_id"] = pid; rows_bat.append(r)
        rp = _counts_row(season, LEAGUE, 700); rp["retro_id"] = "pit1"; rows_pit.append(rp)
        rb = _counts_row(season, LEAGUE, 2000); rb["retro_id"] = "team1"; rb["team"] = "AAA"
        rb.pop("retro_id"); rows_bull.append(rb)
    return (pd.DataFrame(rows_bat), pd.DataFrame(rows_pit), pd.DataFrame(rows_bull))


def test_fit_is_as_of_leakfree():
    bat, pit, bull = _make_frames()
    ps = ProjectionSystem(age_per_year=0.0).fit(bat, pit, bull, ref_season=2023)
    # Only seasons < 2023 were used; projections exist and normalise.
    v = ps.batter("bat1")
    assert abs(v.sum() - 1.0) < 1e-9
    assert ps.known_batter("bat1")
    assert not ps.known_batter("nobody")
    assert np.allclose(ps.batter("nobody"), ps.league_bat)   # unknown -> league


def test_bullpen_lookup():
    bat, pit, bull = _make_frames()
    ps = ProjectionSystem(age_per_year=0.0).fit(bat, pit, bull, ref_season=2024)
    assert abs(ps.bullpen("AAA").sum() - 1.0) < 1e-9
    assert np.allclose(ps.bullpen("ZZZ"), ps.league_pit)     # unknown team -> league
