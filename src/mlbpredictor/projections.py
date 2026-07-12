"""Marcel-style player projections → per-PA outcome rate vectors.

For each batter, starting pitcher and team bullpen we build a probability vector
over :data:`~mlbpredictor.retrosheet.PA_OUTCOMES`
``[1B, 2B, 3B, HR, BB, HBP, SO, OUT]`` by:

1. **Weighting** the most recent (up to) three completed seasons, newest first;
2. **Regressing to the league mean** — mixing in a fixed number of league-average
   PAs so small samples are pulled toward average;
3. a light **age adjustment** for batters (young = a touch better, old = worse).

Everything is computed *as of* a reference season using only earlier data, so a
backtest that projects season *Y* from seasons ``< Y`` is leak-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .ids import birth_year_for
from .retrosheet import PA_OUTCOMES

_IDX = {o: i for i, o in enumerate(PA_OUTCOMES)}
_OUT = _IDX["OUT"]
_N = len(PA_OUTCOMES)


def counts_row_to_vec(row: pd.Series) -> np.ndarray:
    return np.array([float(row[o]) for o in PA_OUTCOMES], dtype=float)


def _weighted_regressed(sub: pd.DataFrame, weights: list[float],
                        league_vec: np.ndarray, regress_pa: float) -> tuple[np.ndarray, float]:
    """Weighted multi-season rate, regressed to league by the *actual* sample size.

    The season weights set how much each year informs the blended *rate*; the
    regression toward league then uses the player's real accumulated PA (not the
    weight-inflated total), so a 10-PA cameo regresses almost fully to league while
    a multi-season regular barely moves. Returns ``(rate_vector, actual_pa)``.
    """
    sub = sub.sort_values("season", ascending=False).head(len(weights))
    wc = np.zeros(_N)
    wpa = 0.0
    pa_actual = 0.0
    for w, (_, r) in zip(weights, sub.iterrows()):
        wc += w * counts_row_to_vec(r)
        wpa += w * float(r["PA"])
        pa_actual += float(r["PA"])
    weighted_rate = wc / wpa if wpa > 0 else league_vec.copy()
    rate = (weighted_rate * pa_actual + league_vec * regress_pa) / (pa_actual + regress_pa)
    return rate, pa_actual


def _age_adjust(rate: np.ndarray, retro_id: str, ref_season: int,
                peak: float, per_year: float) -> np.ndarray:
    """Nudge a batter's non-out outcomes by a small age factor (normalisation-safe)."""
    if per_year <= 0:
        return rate
    by = birth_year_for(retro_id)
    if by is None:
        return rate
    age = ref_season - by
    factor = 1.0 + per_year * (peak - age)
    factor = float(np.clip(factor, 0.90, 1.10))
    adj = rate.copy()
    adj[:_OUT] *= factor                       # scale hits/walks/etc
    adj[_OUT] = max(1e-6, 1.0 - adj[:_OUT].sum())
    return adj / adj.sum()


class ProjectionSystem:
    """Builds and serves projected PA-outcome rate vectors."""

    def __init__(self, weights: list[float] | None = None,
                 bat_regress_pa: float | None = None, pit_regress_bf: float | None = None,
                 age_peak: float | None = None, age_per_year: float | None = None):
        cfg = load_config()["projections"]
        self.weights = weights or cfg["season_weights"]
        self.bat_regress_pa = bat_regress_pa if bat_regress_pa is not None else cfg["bat_regress_pa"]
        self.pit_regress_bf = pit_regress_bf if pit_regress_bf is not None else cfg["pit_regress_bf"]
        self.age_peak = age_peak if age_peak is not None else cfg["age_peak"]
        self.age_per_year = age_per_year if age_per_year is not None else cfg["age_adj_per_year"]

        self.ref_season = 0
        self.league_bat = np.full(_N, 1.0 / _N)
        self.league_pit = np.full(_N, 1.0 / _N)
        self.bat_: dict[str, np.ndarray] = {}
        self.pit_: dict[str, np.ndarray] = {}
        self.bull_: dict[str, np.ndarray] = {}
        self._bat_pa: dict[str, float] = {}
        self._pit_pa: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _league_vec(agg: pd.DataFrame) -> np.ndarray:
        tot = np.array([float(agg[o].sum()) for o in PA_OUTCOMES])
        s = tot.sum()
        return tot / s if s > 0 else np.full(_N, 1.0 / _N)

    def fit(self, batting: pd.DataFrame, pitching: pd.DataFrame, bullpen: pd.DataFrame,
            ref_season: int) -> "ProjectionSystem":
        """Fit projections *as of* ``ref_season`` using only seasons ``< ref_season``."""
        self.ref_season = int(ref_season)
        bat = batting[batting["season"] < ref_season]
        pit = pitching[pitching["season"] < ref_season]
        bull = bullpen[bullpen["season"] < ref_season]
        if bat.empty or pit.empty:
            raise ValueError(f"No projection data before season {ref_season}.")

        self.league_bat = self._league_vec(bat)
        self.league_pit = self._league_vec(pit)

        for rid, sub in bat.groupby("retro_id"):
            rate, _ = _weighted_regressed(sub, self.weights, self.league_bat, self.bat_regress_pa)
            rate = _age_adjust(rate, rid, self.ref_season, self.age_peak, self.age_per_year)
            self.bat_[rid] = rate
            self._bat_pa[rid] = float(sub["PA"].sum())
        for rid, sub in pit.groupby("retro_id"):
            rate, _ = _weighted_regressed(sub, self.weights, self.league_pit, self.pit_regress_bf)
            self.pit_[rid] = rate
            self._pit_pa[rid] = float(sub["PA"].sum())
        for team, sub in bull.groupby("team"):
            rate, _ = _weighted_regressed(sub, self.weights, self.league_pit, self.pit_regress_bf)
            self.bull_[team] = rate
        return self

    # ------------------------------------------------------------------ #
    def batter(self, retro_id: str) -> np.ndarray:
        """Projected rate vector for a batter (league average if unknown)."""
        return self.bat_.get(retro_id, self.league_bat)

    def pitcher(self, retro_id: str) -> np.ndarray:
        """Projected rate vector allowed by a (starting) pitcher; league avg if unknown."""
        return self.pit_.get(retro_id, self.league_pit)

    def bullpen(self, team: str) -> np.ndarray:
        """Projected rate vector allowed by a team's bullpen; league avg if unknown."""
        return self.bull_.get(team, self.league_pit)

    def known_batter(self, retro_id: str) -> bool:
        return retro_id in self.bat_

    def known_pitcher(self, retro_id: str) -> bool:
        return retro_id in self.pit_
