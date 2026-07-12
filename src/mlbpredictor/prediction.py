"""The prediction container returned for a game.

:class:`GamePrediction` bundles the calibrated moneyline (from the ensemble) with
the Monte-Carlo simulation, **reweighted** so the simulation's win rate matches the
calibrated probability. Every downstream number — projected score, totals, run
line, first-five-innings and the per-player prop lines — is then a weighted
statistic of the same reconciled simulation, so the moneyline, the score and the
props never disagree. Fair prices are reported as probabilities and American odds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulate import GameSimResult, _IS_HIT, _TOTAL_BASES


def american_odds(p: float) -> int:
    """Fair American moneyline from a win probability."""
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _nearest_half(x: float) -> float:
    return round(x * 2) / 2.0


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

    # ---- weighted helpers ------------------------------------------------ #
    def _w(self) -> np.ndarray:
        return self.weights

    def _wavg(self, x: np.ndarray) -> float:
        return float(np.average(x, weights=self.weights))

    def _wprob(self, mask: np.ndarray) -> float:
        return float(np.average(mask.astype(float), weights=self.weights))

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
            outs = pitch["sp_out"]
            kline = _nearest_half(self._wavg(k))
            out.append({
                "player": name, "id": pid,
                "proj_k": round(self._wavg(k), 2),
                "k_line": kline, "p_over_k": round(self._wprob(k > kline), 3),
                "proj_ip": round(self._wavg(outs) / 3.0, 2),
                "proj_hits_allowed": round(self._wavg(pitch["sp_h"]), 2),
                "proj_earned_runs": round(self._wavg(pitch["sp_r"]), 2),
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
                "proj_hits": round(self._wavg(hits), 2),
                "p_1plus_hit": round(self._wprob(hits >= 1), 3),
                "proj_tb": round(self._wavg(tb), 2),
                "p_over_1p5_tb": round(self._wprob(tb > 1.5), 3),
                "proj_hr": round(self._wavg(hr), 3),
                "p_hr": round(self._wprob(hr >= 1), 3),
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
            "moneyline": {
                "p_home": round(self.p_home, 4), "p_away": round(self.p_away, 4),
                "fair_home": self.fair_ml_home, "fair_away": self.fair_ml_away,
                "favorite": self.favorite,
            },
            "score": {
                "projected_away": a, "projected_home": h,
                "exp_away_runs": round(self.exp_away_runs, 2),
                "exp_home_runs": round(self.exp_home_runs, 2),
            },
            "total": {
                "exp_total": round(self.exp_total, 2), "line": line,
                "p_over": round(self.p_over(line), 4), "p_under": round(1 - self.p_over(line), 4),
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
        return (f"{self.away_team} {a} @ {self.home_team} {h}  |  "
                f"{fav} {fav_p:.0%} ({american_odds(fav_p):+d})  |  "
                f"total {self.total_line()} (O {self.p_over(self.total_line()):.0%})")
