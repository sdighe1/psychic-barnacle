"""Analytic expected-runs estimate (the fast path used by the run model & features).

The Monte-Carlo simulator produces runs by playing games out; for the backtest and
the ensemble we need a closed-form estimate over thousands of games. We compute a
lineup's aggregate per-PA outcome rates against the opposing pitching (via the same
odds-ratio matchup), convert to **wOBA**, and scale to runs relative to league:

    expected_runs = league_runs_per_game · (team_wOBA / league_wOBA)

Park is applied once, inside the matchup. This tracks the simulator closely while
being vectorisable and instant.
"""
from __future__ import annotations

import numpy as np

from .matchup import matchup_probs
from .retrosheet import PA_OUTCOMES

# Standard wOBA linear weights (modern era), aligned to PA_OUTCOMES order.
#                    1B     2B     3B     HR    BB     HBP    SO   OUT
WOBA_WEIGHTS = np.array([0.88, 1.25, 1.58, 2.02, 0.69, 0.72, 0.0, 0.0])

SP_SHARE = 0.62          # fraction of a team's PAs that face the starter


def woba(rates: np.ndarray) -> float:
    """wOBA implied by a per-PA outcome distribution."""
    return float(np.dot(WOBA_WEIGHTS, rates))


def pitch_blend(starter: np.ndarray, bullpen: np.ndarray, sp_share: float = SP_SHARE) -> np.ndarray:
    v = sp_share * starter + (1.0 - sp_share) * bullpen
    s = v.sum()
    return v / s if s > 0 else v


def team_pa_rates(lineup_vecs, starter_vec, bullpen_vec, league_vec,
                  park_factor: float = 1.0, sp_share: float = SP_SHARE) -> np.ndarray:
    """Aggregate per-PA outcome rates for a lineup vs the opposing pitching."""
    pitch = pitch_blend(starter_vec, bullpen_vec, sp_share)
    acc = np.zeros(len(PA_OUTCOMES))
    for b in lineup_vecs:
        acc += matchup_probs(b, pitch, league_vec, park_factor)
    return acc / len(lineup_vecs)


class OffenseModel:
    """Maps projection inputs to an expected run total, calibrated to league."""

    def __init__(self, league_vec: np.ndarray, league_runs_per_game: float,
                 sp_share: float = SP_SHARE):
        self.league_vec = league_vec
        self.league_woba = woba(league_vec)
        self.lg_rpg = float(league_runs_per_game)
        self.sp_share = sp_share

    def expected_runs(self, lineup_vecs, starter_vec, bullpen_vec,
                      park_factor: float = 1.0) -> float:
        q = team_pa_rates(lineup_vecs, starter_vec, bullpen_vec, self.league_vec,
                          park_factor, self.sp_share)
        ratio = woba(q) / self.league_woba if self.league_woba > 0 else 1.0
        return float(np.clip(self.lg_rpg * ratio, 0.5, 20.0))
