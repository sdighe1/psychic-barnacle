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


_BAT_SPLIT_REG = 900.0      # PA of regression for batter platoon splits (they're noisy)
_PIT_SPLIT_REG = 1200.0     # BF of regression for pitcher platoon splits


def _weighted_split(sub: pd.DataFrame, weights: list[float], suffix: str):
    """Raw weighted split rate + PA for the ``_vL``/``_vR`` columns (no regression)."""
    cols = [o + suffix for o in PA_OUTCOMES]
    if not all(c in sub.columns for c in cols) or ("PA" + suffix) not in sub.columns:
        return None, 0.0                       # split columns absent (e.g. synthetic frames)
    sub = sub.sort_values("season", ascending=False).head(len(weights))
    wc = np.zeros(_N)
    wpa = 0.0
    pa = 0.0
    for w, (_, r) in zip(weights, sub.iterrows()):
        wc += w * np.array([float(r[o + suffix]) for o in PA_OUTCOMES])
        wpa += w * float(r["PA" + suffix])
        pa += float(r["PA" + suffix])
    return (wc / wpa if wpa > 0 else None), pa


def _safe_ratio(side: np.ndarray, overall: np.ndarray) -> np.ndarray:
    return np.where(overall > 1e-9, side / np.maximum(overall, 1e-9), 1.0)


def _platoon_project(overall, factor, obs_rate, obs_pa, reg) -> np.ndarray:
    """League-platoon-adjusted overall (prior), blended with the observed split."""
    prior = overall * factor
    s = prior.sum()
    prior = prior / s if s > 0 else overall
    if obs_rate is None or obs_pa <= 0:
        return prior
    blended = (obs_rate * obs_pa + prior * reg) / (obs_pa + reg)
    bs = blended.sum()
    return blended / bs if bs > 0 else prior


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
        # Platoon splits (vs L / vs R). Batter keys = opposing pitcher hand;
        # pitcher/bullpen keys = batter's effective hand.
        self.bat_vL_: dict[str, np.ndarray] = {}
        self.bat_vR_: dict[str, np.ndarray] = {}
        self.pit_vL_: dict[str, np.ndarray] = {}
        self.pit_vR_: dict[str, np.ndarray] = {}
        self.bull_vL_: dict[str, np.ndarray] = {}
        self.bull_vR_: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _league_vec(agg: pd.DataFrame, suffix: str = "") -> np.ndarray:
        cols = [o + suffix for o in PA_OUTCOMES]
        if not all(c in agg.columns for c in cols):        # split columns absent
            return ProjectionSystem._league_vec(agg) if suffix else np.full(_N, 1.0 / _N)
        tot = np.array([float(agg[c].sum()) for c in cols])
        s = tot.sum()
        return tot / s if s > 0 else np.full(_N, 1.0 / _N)

    def _fit_splits(self, sub, overall, factor_L, factor_R, reg):
        """Return ``(vs_L, vs_R)`` platoon projections for one player.

        Prior = the player's (league-platoon-adjusted) overall; blended with their
        observed split, heavily regressed since splits are noisy.
        """
        oL, paL = _weighted_split(sub, self.weights, "_vL")
        oR, paR = _weighted_split(sub, self.weights, "_vR")
        return (_platoon_project(overall, factor_L, oL, paL, reg),
                _platoon_project(overall, factor_R, oR, paR, reg))

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
        # League platoon factors: how the average player's rates shift by opposing hand.
        pf_bat_L = _safe_ratio(self._league_vec(bat, "_vL"), self.league_bat)
        pf_bat_R = _safe_ratio(self._league_vec(bat, "_vR"), self.league_bat)
        pf_pit_L = _safe_ratio(self._league_vec(pit, "_vL"), self.league_pit)
        pf_pit_R = _safe_ratio(self._league_vec(pit, "_vR"), self.league_pit)
        pf_bull_L = _safe_ratio(self._league_vec(bull, "_vL"), self._league_vec(bull))
        pf_bull_R = _safe_ratio(self._league_vec(bull, "_vR"), self._league_vec(bull))

        for rid, sub in bat.groupby("retro_id"):
            rate, _ = _weighted_regressed(sub, self.weights, self.league_bat, self.bat_regress_pa)
            rate = _age_adjust(rate, rid, self.ref_season, self.age_peak, self.age_per_year)
            self.bat_[rid] = rate
            self._bat_pa[rid] = float(sub["PA"].sum())
            self.bat_vL_[rid], self.bat_vR_[rid] = self._fit_splits(
                sub, rate, pf_bat_L, pf_bat_R, _BAT_SPLIT_REG)
        for rid, sub in pit.groupby("retro_id"):
            rate, _ = _weighted_regressed(sub, self.weights, self.league_pit, self.pit_regress_bf)
            self.pit_[rid] = rate
            self._pit_pa[rid] = float(sub["PA"].sum())
            self.pit_vL_[rid], self.pit_vR_[rid] = self._fit_splits(
                sub, rate, pf_pit_L, pf_pit_R, _PIT_SPLIT_REG)
        for team, sub in bull.groupby("team"):
            rate, _ = _weighted_regressed(sub, self.weights, self.league_pit, self.pit_regress_bf)
            self.bull_[team] = rate
            self.bull_vL_[team], self.bull_vR_[team] = self._fit_splits(
                sub, rate, pf_bull_L, pf_bull_R, _PIT_SPLIT_REG)
        return self

    # ------------------------------------------------------------------ #
    def batter(self, retro_id: str, vs: str | None = None) -> np.ndarray:
        """Batter rate vector; ``vs`` = opposing pitcher hand ('L'/'R') for the split."""
        overall = self.bat_.get(retro_id, self.league_bat)
        if vs == "L":
            return self.bat_vL_.get(retro_id, overall)
        if vs == "R":
            return self.bat_vR_.get(retro_id, overall)
        return overall

    def pitcher(self, retro_id: str, vs: str | None = None) -> np.ndarray:
        """Pitcher allowed-rate vector; ``vs`` = batter's effective hand ('L'/'R')."""
        overall = self.pit_.get(retro_id, self.league_pit)
        if vs == "L":
            return self.pit_vL_.get(retro_id, overall)
        if vs == "R":
            return self.pit_vR_.get(retro_id, overall)
        return overall

    def bullpen(self, team: str, vs: str | None = None) -> np.ndarray:
        """Bullpen allowed-rate vector; ``vs`` = batter's effective hand ('L'/'R')."""
        overall = self.bull_.get(team, self.league_pit)
        if vs == "L":
            return self.bull_vL_.get(team, overall)
        if vs == "R":
            return self.bull_vR_.get(team, overall)
        return overall

    def known_batter(self, retro_id: str) -> bool:
        return retro_id in self.bat_

    def known_pitcher(self, retro_id: str) -> bool:
        return retro_id in self.pit_
