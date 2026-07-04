"""Feature engineering for the ML component of the predictor.

All features are computed strictly *as-of* the match (only information available
before kick-off), so the training frame is free of look-ahead leakage:

- Elo of each side (pre-match) and their difference
- Rolling recent **form**: goals for / against and points-per-game over the last
  ``form_window`` matches
- **Rest days** since each team's previous match
- Tournament **importance** and the **neutral**-venue flag

:class:`FeatureBuilder` fits on the full history and can then produce an
identically-shaped feature vector for any hypothetical future fixture, so the
training and inference paths never diverge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import EloRatingSystem

FEATURES = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "importance",
    "neutral",
    "home_form_gf",
    "home_form_ga",
    "home_form_pts",
    "away_form_gf",
    "away_form_ga",
    "away_form_pts",
    "home_rest",
    "away_rest",
]

DEFAULT_FORM = {"gf": 1.2, "ga": 1.2, "pts": 1.3}
DEFAULT_REST = 90.0
MAX_REST = 365.0


def _team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) with that team's goals for/against and points."""
    base = df.reset_index(drop=True)
    mid = np.arange(len(base))
    home = pd.DataFrame({
        "match_id": mid, "team": base["home_team"].values, "date": base["date"].values,
        "gf": base["home_score"].values, "ga": base["away_score"].values, "is_home": True,
    })
    away = pd.DataFrame({
        "match_id": mid, "team": base["away_team"].values, "date": base["date"].values,
        "gf": base["away_score"].values, "ga": base["home_score"].values, "is_home": False,
    })
    tp = pd.concat([home, away], ignore_index=True)
    tp["points"] = np.where(tp.gf > tp.ga, 3, np.where(tp.gf == tp.ga, 1, 0))
    return tp


class FeatureBuilder:
    def __init__(self, form_window: int = 5, home_advantage: float = 65.0):
        self.form_window = int(form_window)
        self.elo = EloRatingSystem(home_advantage=home_advantage)
        self.current_form_: dict[str, dict[str, float]] = {}
        self.last_played_: dict[str, pd.Timestamp] = {}

    # ------------------------------------------------------------------ #
    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Fit Elo + form state and return the training feature frame."""
        df = self.elo.fit_transform(matches)  # adds home_elo/away_elo, sorts by date
        tp = _team_perspective(df)
        tp = tp.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)

        w = self.form_window
        grp = tp.groupby("team", sort=False)
        for col in ("gf", "ga", "points"):
            tp[f"form_{col}"] = (
                grp[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
        tp["rest"] = (
            grp["date"].apply(lambda s: (s - s.shift(1)).dt.days)
            .reset_index(level=0, drop=True)
        )

        # Fill cold-start values and cap rest.
        tp["form_gf"] = tp["form_gf"].fillna(DEFAULT_FORM["gf"])
        tp["form_ga"] = tp["form_ga"].fillna(DEFAULT_FORM["ga"])
        tp["form_points"] = tp["form_points"].fillna(DEFAULT_FORM["pts"])
        tp["rest"] = tp["rest"].fillna(DEFAULT_REST).clip(upper=MAX_REST)

        # Snapshot the latest state per team for inference.
        latest = tp.sort_values("date").groupby("team").tail(self.form_window)
        for team, g in latest.groupby("team"):
            self.current_form_[team] = {
                "gf": float(g["gf"].mean()),
                "ga": float(g["ga"].mean()),
                "pts": float(np.where(g["gf"] > g["ga"], 3, np.where(g["gf"] == g["ga"], 1, 0)).mean()),
            }
        self.last_played_ = dict(self.elo.last_played_)

        # Merge team-perspective features back onto matches.
        home_tp = tp[tp.is_home].set_index("match_id")
        away_tp = tp[~tp.is_home].set_index("match_id")
        df = df.copy()
        df["home_form_gf"] = home_tp["form_gf"].reindex(range(len(df))).values
        df["home_form_ga"] = home_tp["form_ga"].reindex(range(len(df))).values
        df["home_form_pts"] = home_tp["form_points"].reindex(range(len(df))).values
        df["home_rest"] = home_tp["rest"].reindex(range(len(df))).values
        df["away_form_gf"] = away_tp["form_gf"].reindex(range(len(df))).values
        df["away_form_ga"] = away_tp["form_ga"].reindex(range(len(df))).values
        df["away_form_pts"] = away_tp["form_points"].reindex(range(len(df))).values
        df["away_rest"] = away_tp["rest"].reindex(range(len(df))).values
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["neutral"] = df["neutral"].astype(int)
        return df

    # ------------------------------------------------------------------ #
    def _form(self, team: str) -> dict[str, float]:
        return self.current_form_.get(team, {"gf": DEFAULT_FORM["gf"], "ga": DEFAULT_FORM["ga"], "pts": DEFAULT_FORM["pts"]})

    def match_features(self, home: str, away: str, neutral: bool = False,
                       date: pd.Timestamp | None = None) -> pd.DataFrame:
        """Feature row for one hypothetical fixture, using the fitted state."""
        rh = self.elo.rating(home)
        ra = self.elo.rating(away)
        fh = self._form(home)
        fa = self._form(away)
        if date is not None:
            rest_h = _rest(self.last_played_.get(home), date)
            rest_a = _rest(self.last_played_.get(away), date)
        else:
            rest_h = rest_a = DEFAULT_REST
        row = {
            "elo_diff": rh - ra,
            "home_elo": rh,
            "away_elo": ra,
            "importance": 5,          # World Cup finals by default
            "neutral": int(neutral),
            "home_form_gf": fh["gf"], "home_form_ga": fh["ga"], "home_form_pts": fh["pts"],
            "away_form_gf": fa["gf"], "away_form_ga": fa["ga"], "away_form_pts": fa["pts"],
            "home_rest": rest_h, "away_rest": rest_a,
        }
        return pd.DataFrame([row], columns=FEATURES)


def _rest(last: pd.Timestamp | None, date: pd.Timestamp) -> float:
    if last is None or pd.isna(last):
        return DEFAULT_REST
    d = (pd.Timestamp(date) - pd.Timestamp(last)).days
    return float(min(max(d, 0), MAX_REST))
