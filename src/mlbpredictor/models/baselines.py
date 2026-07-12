"""Reference moneyline models (2-class: home win / away win — no draws in MLB).

- :class:`HomeBaseRate` — always predicts the training home-win frequency (~0.54).
  The no-skill floor for log-loss / Brier.
- :class:`EloLogistic` — logistic regression of home-win on the pre-game Elo
  difference. A strong, simple, well-understood baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class HomeBaseRate:
    def __init__(self):
        self.rate_ = 0.54

    def fit(self, feat: pd.DataFrame) -> "HomeBaseRate":
        self.rate_ = float(feat["home_win"].mean())
        return self

    def predict_p_home(self, feat: pd.DataFrame) -> np.ndarray:
        return np.full(len(feat), self.rate_)


class EloLogistic:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, C=1.0)

    def fit(self, feat: pd.DataFrame) -> "EloLogistic":
        X = feat[["elo_diff"]].to_numpy(dtype=float)
        self.clf.fit(X, feat["home_win"].to_numpy())
        self._pos = list(self.clf.classes_).index(1)
        return self

    def predict_p_home(self, feat: pd.DataFrame) -> np.ndarray:
        X = feat[["elo_diff"]].to_numpy(dtype=float)
        return self.clf.predict_proba(X)[:, self._pos]
