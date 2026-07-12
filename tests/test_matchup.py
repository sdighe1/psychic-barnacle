"""Odds-ratio matchup engine: check the limiting behaviour and normalisation."""
import numpy as np

from mlbpredictor.matchup import matchup_probs, scale_offense
from mlbpredictor.retrosheet import PA_OUTCOMES

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


def test_average_batter_yields_pitcher_rates():
    pitcher = np.array([0.12, 0.04, 0.003, 0.025, 0.07, 0.01, 0.30, 0.432])
    pitcher = pitcher / pitcher.sum()
    p = matchup_probs(LEAGUE, pitcher, LEAGUE)
    assert np.allclose(p, pitcher, atol=1e-9)


def test_average_pitcher_yields_batter_rates():
    batter = np.array([0.16, 0.05, 0.005, 0.05, 0.10, 0.012, 0.18, 0.393])
    batter = batter / batter.sum()
    p = matchup_probs(batter, LEAGUE, LEAGUE)
    assert np.allclose(p, batter, atol=1e-9)


def test_two_average_players_yield_league():
    p = matchup_probs(LEAGUE, LEAGUE, LEAGUE)
    assert np.allclose(p, LEAGUE, atol=1e-9)


def test_probs_sum_to_one():
    b = np.array([0.20, 0.06, 0.006, 0.06, 0.11, 0.013, 0.15, 0.401]); b /= b.sum()
    pit = np.array([0.10, 0.03, 0.002, 0.02, 0.06, 0.009, 0.33, 0.449]); pit /= pit.sum()
    for pf in (0.9, 1.0, 1.15):
        p = matchup_probs(b, pit, LEAGUE, park_factor=pf)
        assert abs(p.sum() - 1.0) < 1e-9
        assert (p >= 0).all()


def test_park_factor_increases_offense():
    i_out = PA_OUTCOMES.index("OUT")
    hitters_park = scale_offense(LEAGUE, 1.2)
    pitchers_park = scale_offense(LEAGUE, 0.85)
    assert hitters_park[i_out] < LEAGUE[i_out]       # fewer outs in a hitters' park
    assert pitchers_park[i_out] > LEAGUE[i_out]
    assert abs(hitters_park.sum() - 1.0) < 1e-9
