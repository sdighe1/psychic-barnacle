"""A World-Football-style Elo rating system.

Ratings are built by walking the match history in chronological order, so the
rating attached to any match is strictly *pre-match* (it never sees the result
it is about to predict). Key ingredients, following the eloratings.net scheme:

- **Importance weight** ``K`` per match (World Cup > continental > qualifier >
  friendly), supplied by :func:`wcpredictor.data.tournament_weights`.
- **Margin-of-victory** multiplier so blowouts move ratings more.
- **Home advantage** added to the home side's expectation unless the match is at
  a neutral venue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INITIAL_RATING = 1500.0


def mov_multiplier(goal_diff: int) -> float:
    """Goal-difference multiplier (eloratings.net)."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


class EloRatingSystem:
    """Sequential Elo ratings with margin-of-victory and home advantage.

    Parameters
    ----------
    home_advantage:
        Elo points added to the home team's expectation at non-neutral venues.
    initial_rating:
        Rating assigned to a team the first time it is seen.
    """

    def __init__(self, home_advantage: float = 65.0, initial_rating: float = INITIAL_RATING):
        self.home_advantage = float(home_advantage)
        self.initial_rating = float(initial_rating)
        self.ratings_: dict[str, float] = {}
        self.last_played_: dict[str, pd.Timestamp] = {}

    # ------------------------------------------------------------------ #
    def expected_home(self, r_home: float, r_away: float, neutral: bool) -> float:
        """Win-expectancy of the home side (draws split this continuously)."""
        adj = 0.0 if neutral else self.home_advantage
        return 1.0 / (1.0 + 10.0 ** (-((r_home + adj) - r_away) / 400.0))

    def rating(self, team: str) -> float:
        return self.ratings_.get(team, self.initial_rating)

    # ------------------------------------------------------------------ #
    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Process ``matches`` in order; return a copy with pre-match ratings.

        Adds columns ``home_elo``/``away_elo`` (the ratings *before* each match)
        and leaves ``self.ratings_`` holding the latest rating for every team.
        """
        df = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
        n = len(df)
        home_pre = np.empty(n)
        away_pre = np.empty(n)

        ratings = dict(self.ratings_)
        last_played = dict(self.last_played_)
        init = self.initial_rating
        hfa = self.home_advantage

        for i, row in enumerate(df.itertuples(index=False)):
            rh = ratings.get(row.home_team, init)
            ra = ratings.get(row.away_team, init)
            home_pre[i] = rh
            away_pre[i] = ra

            adj = 0.0 if row.neutral else hfa
            we = 1.0 / (1.0 + 10.0 ** (-((rh + adj) - ra) / 400.0))
            if row.home_score > row.away_score:
                w = 1.0
            elif row.home_score < row.away_score:
                w = 0.0
            else:
                w = 0.5
            g = mov_multiplier(row.home_score - row.away_score)
            delta = row.k_weight * g * (w - we)
            ratings[row.home_team] = rh + delta
            ratings[row.away_team] = ra - delta
            last_played[row.home_team] = row.date
            last_played[row.away_team] = row.date

        self.ratings_ = ratings
        self.last_played_ = last_played
        df = df.copy()
        df["home_elo"] = home_pre
        df["away_elo"] = away_pre
        return df

    # ------------------------------------------------------------------ #
    def rankings(self, top: int | None = None) -> pd.DataFrame:
        """Current ratings as a sorted table."""
        s = pd.Series(self.ratings_, name="elo").sort_values(ascending=False)
        out = s.reset_index().rename(columns={"index": "team"})
        out.index = np.arange(1, len(out) + 1)
        out.index.name = "rank"
        return out.head(top) if top else out
