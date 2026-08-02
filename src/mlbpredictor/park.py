"""Park run factors from game logs.

Classic quality-controlled estimate: for each park's home team, compare total
runs per game **at home** to that same team's total runs per game **on the road**
(same team both ways, so team quality roughly cancels). The ratio is regressed
toward 1.0 by sample size and averaged across seasons. A factor > 1 means a hitter's
park (Coors), < 1 a pitcher's park (Petco).

The simulator uses the factor as a mild offensive-environment multiplier for both
teams at the venue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_CLIP = (0.85, 1.30)


class ParkFactors:
    def __init__(self, regress_games: float = 150.0):
        self.regress_games = float(regress_games)
        self.factors_: dict[str, float] = {}

    def fit(self, game_logs: pd.DataFrame) -> "ParkFactors":
        gl = game_logs.copy()
        gl["total"] = gl["home_score"] + gl["away_score"]
        home = gl.groupby("home_team")["total"].agg(hr="sum", hn="size")
        road = gl.groupby("away_team")["total"].agg(rr="sum", rn="size")
        # Each team's primary home park (mode).
        team_park = gl.groupby("home_team")["park"].agg(
            lambda s: s.value_counts().index[0])

        acc: dict[str, list[float]] = {}
        for team in home.index:
            if team not in road.index:
                continue
            hn = float(home.loc[team, "hn"])
            rn = float(road.loc[team, "rn"])
            if hn < 20 or rn < 20:
                continue
            hrpg = home.loc[team, "hr"] / hn
            rrpg = road.loc[team, "rr"] / rn
            if rrpg <= 0:
                continue
            raw = hrpg / rrpg
            pf = 1.0 + (raw - 1.0) * (hn / (hn + self.regress_games))
            acc.setdefault(team_park[team], []).append(pf)

        self.factors_ = {p: float(np.clip(np.mean(v), *_CLIP)) for p, v in acc.items()}
        return self

    def factor(self, park_id: str | None) -> float:
        return self.factors_.get(park_id or "", 1.0)
