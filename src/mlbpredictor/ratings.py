"""Sequential, leak-free team Elo for MLB.

A classic Elo with a home-field edge and a margin-of-victory multiplier (538-style).
Ratings are updated game-by-game in date order; the *pre-game* ratings are recorded
as features, so there is no look-ahead leakage. Elo captures durable team strength
(roster, depth, bullpen) and is a strong, low-variance moneyline baseline. The
specific-start pitching effect is layered on separately via projection features.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

BASE = 1500.0


class EloRatingSystem:
    def __init__(self, k: float = 6.0, home_adv: float = 24.0, mov: bool = True,
                 revert: float = 0.0):
        self.k = float(k)
        self.home_adv = float(home_adv)
        self.mov = mov
        self.revert = float(revert)          # optional season-to-season regression
        self.ratings_: dict[str, float] = {}
        self.last_played_: dict[str, pd.Timestamp] = {}
        self.games_: dict[str, int] = {}

    def rating(self, team: str) -> float:
        return self.ratings_.get(team, BASE)

    @staticmethod
    def expected(elo_home: float, elo_away: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-(elo_home - elo_away) / 400.0))

    def _mov_mult(self, margin: int, elo_diff_winner: float) -> float:
        if not self.mov:
            return 1.0
        # 538's autocorrelation-corrected MOV multiplier.
        return math.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))

    def fit_transform(self, games: pd.DataFrame) -> pd.DataFrame:
        """Add pre-game ``home_elo``/``away_elo`` columns and update ratings in order."""
        df = games.sort_values(["date"], kind="mergesort").reset_index(drop=True)
        home_elo = np.empty(len(df))
        away_elo = np.empty(len(df))
        prev_season: dict[str, int] = {}

        for i, r in enumerate(df.itertuples(index=False)):
            h, a = r.home_team, r.away_team
            season = getattr(r, "season", None)
            # Optional soft season reversion toward the mean.
            if self.revert > 0 and season is not None:
                for t in (h, a):
                    if prev_season.get(t) not in (None, season):
                        self.ratings_[t] = BASE + (1 - self.revert) * (self.rating(t) - BASE)
                    prev_season[t] = season

            rh, ra = self.rating(h), self.rating(a)
            home_elo[i] = rh
            away_elo[i] = ra

            exp_home = self.expected(rh + self.home_adv, ra)
            home_win = 1.0 if r.home_score > r.away_score else 0.0
            margin = r.home_score - r.away_score
            winner_diff = (rh - ra) if home_win else (ra - rh)
            mult = self._mov_mult(margin, winner_diff)
            delta = self.k * mult * (home_win - exp_home)
            self.ratings_[h] = rh + delta
            self.ratings_[a] = ra - delta
            self.last_played_[h] = r.date
            self.last_played_[a] = r.date
            self.games_[h] = self.games_.get(h, 0) + 1
            self.games_[a] = self.games_.get(a, 0) + 1

        df = df.copy()
        df["home_elo"] = home_elo
        df["away_elo"] = away_elo
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        return df

    def rankings(self, top: int | None = None) -> pd.DataFrame:
        s = pd.Series(self.ratings_).sort_values(ascending=False)
        if top:
            s = s.head(top)
        return s.rename("elo").reset_index().rename(columns={"index": "team"})
