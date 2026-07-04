"""Reference models used to demonstrate the main model's lift in the backtest.

- :class:`BaseRateModel` — always predicts the training-set outcome frequencies.
  This is the "no-skill" floor for log-loss / RPS.
- :class:`EloLogisticModel` — a multinomial logistic regression on the Elo
  difference (+ neutral flag). A strong, simple, well-known baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..prediction import Prediction

_LABELS = ["H", "D", "A"]


class BaseRateModel:
    def __init__(self):
        self.rates_ = {"H": 0.49, "D": 0.27, "A": 0.24}
        self.mean_home_goals_ = 1.5
        self.mean_away_goals_ = 1.1

    def fit(self, feat: pd.DataFrame) -> "BaseRateModel":
        counts = feat["outcome"].value_counts(normalize=True)
        self.rates_ = {k: float(counts.get(k, 0.0)) for k in _LABELS}
        self.mean_home_goals_ = float(feat["home_score"].mean())
        self.mean_away_goals_ = float(feat["away_score"].mean())
        return self

    def predict_frame(self, feat: pd.DataFrame) -> pd.DataFrame:
        n = len(feat)
        return pd.DataFrame({
            "p_home": np.full(n, self.rates_["H"]),
            "p_draw": np.full(n, self.rates_["D"]),
            "p_away": np.full(n, self.rates_["A"]),
            "exp_home": np.full(n, self.mean_home_goals_),
            "exp_away": np.full(n, self.mean_away_goals_),
        }, index=feat.index)

    def predict_match(self, home: str, away: str, neutral: bool = False, **_) -> Prediction:
        return Prediction(home, away, self.rates_["H"], self.rates_["D"], self.rates_["A"],
                          self.mean_home_goals_, self.mean_away_goals_)


class EloLogisticModel:
    """Multinomial logistic regression on ``[elo_diff, neutral]``."""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.cols = ["elo_diff", "neutral"]

    def fit(self, feat: pd.DataFrame) -> "EloLogisticModel":
        X = feat[self.cols].to_numpy(dtype=float)
        y = feat["outcome"].to_numpy()
        self.clf.fit(X, y)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_frame(self, feat: pd.DataFrame) -> pd.DataFrame:
        X = feat[self.cols].to_numpy(dtype=float)
        P = self.clf.predict_proba(X)
        col = {c: i for i, c in enumerate(self.classes_)}
        sup = feat["elo_diff"].to_numpy(dtype=float) / 400.0
        return pd.DataFrame({
            "p_home": P[:, col["H"]],
            "p_draw": P[:, col["D"]],
            "p_away": P[:, col["A"]],
            "exp_home": np.clip(1.35 + 0.55 * sup, 0.2, 5.0),
            "exp_away": np.clip(1.35 - 0.55 * sup, 0.2, 5.0),
        }, index=feat.index)

    def _probs(self, elo_diff: float, neutral: bool) -> dict[str, float]:
        X = np.array([[elo_diff, int(neutral)]], dtype=float)
        p = self.clf.predict_proba(X)[0]
        return {c: float(p[i]) for i, c in enumerate(self.classes_)}

    def predict_match(self, home: str, away: str, neutral: bool = False,
                      elo_diff: float = 0.0, **_) -> Prediction:
        p = self._probs(elo_diff, neutral)
        # Rough expected goals from a logistic supremacy proxy (baseline only).
        sup = elo_diff / 400.0
        exp_h = float(np.clip(1.35 + 0.55 * sup, 0.2, 5.0))
        exp_a = float(np.clip(1.35 - 0.55 * sup, 0.2, 5.0))
        return Prediction(home, away, p.get("H", 0.0), p.get("D", 0.0), p.get("A", 0.0), exp_h, exp_a)
