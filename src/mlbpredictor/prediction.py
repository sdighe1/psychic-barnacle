"""The prediction container returned for a game.

:class:`GamePrediction` bundles the calibrated moneyline (from the ensemble) with
the Monte-Carlo simulation, **reweighted** so the simulation's win rate matches the
calibrated probability. Every downstream number — projected score, totals, run
line, first-five-innings and the per-player prop lines — is then a weighted
statistic of the same reconciled simulation, so the moneyline, the score and the
props never disagree.

Uncertainty is reported two ways:

* **Prediction intervals** — central credible intervals (default 50% and 90%) on
  every count output, taken as weighted quantiles of the reconciled simulation.
* **A game confidence level** (High/Medium/Low) for the moneyline pick, a composite
  of how decisive the edge is, how much the ensemble components agree, the quality of
  the inputs (confirmed vs. guessed lineups, roster coverage) and the simulation's
  effective sample size. The interval width expresses uncertainty for the totals and
  props (there is no "pick" there — the line is set at our own projection).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulate import GameSimResult, _IS_HIT, _TOTAL_BASES

_CONF_HI, _CONF_MED = 0.60, 0.40
_EDGE_SCALE = 0.20          # a 20-point moneyline edge (p≈0.70) is "fully decisive" for MLB


def american_odds(p: float) -> int:
    """Fair American moneyline from a win probability."""
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _nearest_half(x: float) -> float:
    return round(x * 2) / 2.0


def _band(score: float) -> str:
    return "High" if score >= _CONF_HI else "Medium" if score >= _CONF_MED else "Low"


@dataclass
class GamePrediction:
    home_team: str
    away_team: str
    p_home: float
    p_away: float
    sim: GameSimResult
    weights: np.ndarray
    home_batters: list = field(default_factory=list)   # [(id, name), ...] order 1..9
    away_batters: list = field(default_factory=list)
    home_sp: tuple = ("", "")
    away_sp: tuple = ("", "")
    park: str | None = None
    date: str | None = None
    # uncertainty inputs
    member_probs: dict = field(default_factory=dict)   # ensemble member -> p_home (this game)
    lineup_confirmed: bool = False                     # real posted lineups vs. defaults
    roster_coverage: float = 1.0                       # fraction of 20 slots with real projections
    interval_levels: tuple = (0.5, 0.9)

    # ---- weighted helpers ------------------------------------------------ #
    def _wavg(self, x: np.ndarray) -> float:
        return float(np.average(x, weights=self.weights))

    def _wprob(self, mask: np.ndarray) -> float:
        return float(np.average(mask.astype(float), weights=self.weights))

    def _wquantile(self, x: np.ndarray, qs) -> np.ndarray:
        """Weighted quantile(s) of ``x`` under the importance weights."""
        x = np.asarray(x, dtype=float)
        order = np.argsort(x, kind="mergesort")
        xs, ws = x[order], self.weights[order]
        cw = (np.cumsum(ws) - 0.5 * ws) / ws.sum()     # midpoint cumulative weights
        return np.interp(qs, cw, xs)

    def interval(self, x: np.ndarray, level: float, as_int: bool = True):
        """Central ``level`` credible interval ``[lo, hi]`` of ``x``."""
        lo, hi = self._wquantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
        if as_int:
            return [int(round(lo)), int(round(hi))]
        return [round(float(lo), 1), round(float(hi), 1)]

    def _ranges(self, x: np.ndarray, as_int: bool = True) -> dict:
        return {f"range_{int(round(l * 100))}": self.interval(x, l, as_int)
                for l in self.interval_levels}

    @property
    def n_eff(self) -> float:
        """Effective sample size of the reweighted simulation (Kish)."""
        w = self.weights
        return float(w.sum() ** 2 / np.sum(w * w))

    # ---- outcome / score ------------------------------------------------- #
    @property
    def favorite(self) -> str:
        return self.home_team if self.p_home >= self.p_away else self.away_team

    @property
    def exp_home_runs(self) -> float:
        return self._wavg(self.sim.home_runs)

    @property
    def exp_away_runs(self) -> float:
        return self._wavg(self.sim.away_runs)

    def projected_score(self) -> tuple[int, int]:
        return int(round(self.exp_away_runs)), int(round(self.exp_home_runs))

    @property
    def fair_ml_home(self) -> int:
        return american_odds(self.p_home)

    @property
    def fair_ml_away(self) -> int:
        return american_odds(self.p_away)

    @property
    def p_home_ci(self) -> list:
        """Model-disagreement band on the win prob: [min, max] across ensemble members.

        This reflects how much the components differ — not a frequentist CI.
        """
        v = list(self.member_probs.values()) + [self.p_home]  # always brackets the estimate
        return [round(float(min(v)), 3), round(float(max(v)), 3)]

    # ---- confidence ------------------------------------------------------ #
    def _components(self) -> dict:
        edge = max(self.p_home, self.p_away) - 0.5
        decisiveness = min(edge / _EDGE_SCALE, 1.0)
        spread = float(np.std(list(self.member_probs.values()))) if self.member_probs else 0.0
        agreement = 1.0 - min(spread / 0.15, 1.0)
        input_quality = 0.5 * float(self.lineup_confirmed) + 0.5 * float(self.roster_coverage)
        stability = min(self.n_eff / (0.5 * len(self.weights)), 1.0)
        return {"decisiveness": decisiveness, "agreement": agreement,
                "input_quality": input_quality, "stability": stability}

    def confidence(self) -> dict:
        """Composite confidence in the moneyline pick → level + score + components."""
        c = self._components()
        # Quality (0-1) modulates, but decisiveness dominates so a coin flip stays Low
        # even with perfect inputs — there is no *pick* to be confident about.
        quality = 0.35 * c["agreement"] + 0.40 * c["input_quality"] + 0.25 * c["stability"]
        score = 0.65 * c["decisiveness"] + 0.35 * quality
        return {
            "level": _band(score), "score": round(score, 3),
            "components": {k: round(v, 3) for k, v in c.items()},
            "n_eff": int(round(self.n_eff)),
        }

    @property
    def confidence_label(self) -> str:
        return self.confidence()["level"]

    # ---- totals / run line ---------------------------------------------- #
    @property
    def exp_total(self) -> float:
        return self.exp_home_runs + self.exp_away_runs

    def total_line(self) -> float:
        return _nearest_half(self.exp_total)

    def p_over(self, line: float) -> float:
        return self._wprob(self.sim.total() > line)

    def p_home_runline(self, line: float = -1.5) -> float:
        return self._wprob((self.sim.home_runs - self.sim.away_runs) + line > 0)

    def p_away_runline(self, line: float = 1.5) -> float:
        return self._wprob((self.sim.away_runs - self.sim.home_runs) + line > 0)

    # ---- first five innings --------------------------------------------- #
    def f5(self) -> dict:
        a, h = self.sim.away_runs_f5, self.sim.home_runs_f5
        diff = h - a
        p_home = self._wprob(diff > 0) + 0.5 * self._wprob(diff == 0)
        return {
            "p_home": round(p_home, 4),
            "p_away": round(1 - p_home, 4),
            "exp_total": round(self._wavg(a + h), 2),
            "line": _nearest_half(self._wavg(a + h)),
            **self._ranges(a + h),
        }

    # ---- props ----------------------------------------------------------- #
    def pitcher_props(self) -> list[dict]:
        # A starter's line is recorded against the team he faces: the HOME starter
        # pitches to the away batters, so his stats live in ``away_pitch`` (and
        # vice-versa).
        out = []
        for (pid, name), pitch in (
            (self.home_sp, self.sim.away_pitch),
            (self.away_sp, self.sim.home_pitch),
        ):
            k = pitch["sp_k"]
            ip = pitch["sp_out"] / 3.0
            kline = _nearest_half(self._wavg(k))
            out.append({
                "player": name, "id": pid,
                "proj_k": round(self._wavg(k), 2), "k_range": self._ranges(k),
                "k_line": kline, "p_over_k": round(self._wprob(k > kline), 3),
                "proj_ip": round(self._wavg(ip), 2), "ip_range": self._ranges(ip, as_int=False),
                "proj_hits_allowed": round(self._wavg(pitch["sp_h"]), 2),
                "hits_allowed_range": self._ranges(pitch["sp_h"]),
                "proj_earned_runs": round(self._wavg(pitch["sp_r"]), 2),
                "earned_runs_range": self._ranges(pitch["sp_r"]),
            })
        return out

    def batter_props(self, team: str = "home") -> list[dict]:
        box = self.sim.home_box if team == "home" else self.sim.away_box
        names = self.home_batters if team == "home" else self.away_batters
        out = []
        for pos in range(9):
            hits = (box[:, pos, :4] * _IS_HIT[:4]).sum(axis=1)
            tb = (box[:, pos, :4] * _TOTAL_BASES[:4]).sum(axis=1)
            hr = box[:, pos, 3]
            pid, name = names[pos] if pos < len(names) else ("", f"Batter {pos+1}")
            out.append({
                "order": pos + 1, "player": name, "id": pid,
                "proj_hits": round(self._wavg(hits), 2), "hits_range": self._ranges(hits),
                "p_1plus_hit": round(self._wprob(hits >= 1), 3),
                "proj_tb": round(self._wavg(tb), 2), "tb_range": self._ranges(tb),
                "p_over_1p5_tb": round(self._wprob(tb > 1.5), 3),
                "proj_hr": round(self._wavg(hr), 3), "p_hr": round(self._wprob(hr >= 1), 3),
                "proj_runs": round(self._wavg(box[:, pos, 8]), 2),
                "proj_rbi": round(self._wavg(box[:, pos, 9]), 2),
            })
        return out

    # ---- serialisation --------------------------------------------------- #
    def to_dict(self) -> dict:
        a, h = self.projected_score()
        line = self.total_line()
        return {
            "home_team": self.home_team, "away_team": self.away_team,
            "date": self.date, "park": self.park,
            "confidence": self.confidence(),
            "data_quality": {
                "lineup_confirmed": bool(self.lineup_confirmed),
                "roster_coverage": round(float(self.roster_coverage), 3),
                "n_eff": int(round(self.n_eff)), "n_sims": int(len(self.weights)),
            },
            "moneyline": {
                "p_home": round(self.p_home, 4), "p_away": round(self.p_away, 4),
                "fair_home": self.fair_ml_home, "fair_away": self.fair_ml_away,
                "favorite": self.favorite, "p_home_ci": self.p_home_ci,
            },
            "score": {
                "projected_away": a, "projected_home": h,
                "exp_away_runs": round(self.exp_away_runs, 2),
                "exp_home_runs": round(self.exp_home_runs, 2),
                "away_range": self._ranges(self.sim.away_runs),
                "home_range": self._ranges(self.sim.home_runs),
            },
            "total": {
                "exp_total": round(self.exp_total, 2), "line": line,
                "p_over": round(self.p_over(line), 4), "p_under": round(1 - self.p_over(line), 4),
                **self._ranges(self.sim.total()),
            },
            "run_line": {
                "home_-1.5": round(self.p_home_runline(-1.5), 4),
                "away_+1.5": round(self.p_away_runline(1.5), 4),
                "away_-1.5": round(self._wprob((self.sim.away_runs - self.sim.home_runs) - 1.5 > 0), 4),
                "home_+1.5": round(self._wprob((self.sim.home_runs - self.sim.away_runs) + 1.5 > 0), 4),
            },
            "first_5_innings": self.f5(),
            "pitcher_props": self.pitcher_props(),
            "home_batter_props": self.batter_props("home"),
            "away_batter_props": self.batter_props("away"),
            "starters": {"home": self.home_sp[1], "away": self.away_sp[1]},
        }

    def summary(self) -> str:
        a, h = self.projected_score()
        fav = self.favorite
        fav_p = self.p_home if fav == self.home_team else self.p_away
        conf = self.confidence()["level"]
        return (f"{self.away_team} {a} @ {self.home_team} {h}  |  "
                f"{fav} {fav_p:.0%} ({american_odds(fav_p):+d}) [{conf}]  |  "
                f"total {self.total_line()} (O {self.p_over(self.total_line()):.0%})")
