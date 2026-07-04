"""Gradient-boosting model over the engineered features.

Uses scikit-learn's histogram gradient boosting (no native build deps): a
classifier for win/draw/loss and two regressors for the goal counts. Captures
non-linear interactions (form × rest × importance × Elo) that the parametric
models miss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..features import FEATURES
from ..prediction import Prediction

_LABELS = ["H", "D", "A"]


class GBoostModel:
    def __init__(self, learning_rate: float = 0.05, max_iter: int = 400,
                 min_samples_leaf: int = 40, l2_regularization: float = 1.0,
                 random_state: int = 0):
        common = dict(learning_rate=learning_rate, max_iter=max_iter,
                      min_samples_leaf=min_samples_leaf,
                      l2_regularization=l2_regularization, random_state=random_state)
        self.clf = HistGradientBoostingClassifier(**common)
        self.reg_home = HistGradientBoostingRegressor(loss="poisson", **common)
        self.reg_away = HistGradientBoostingRegressor(loss="poisson", **common)

    def fit(self, feat: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "GBoostModel":
        X = feat[FEATURES].to_numpy(dtype=float)
        self.clf.fit(X, feat["outcome"].to_numpy(), sample_weight=sample_weight)
        self.classes_ = list(self.clf.classes_)
        self.reg_home.fit(X, feat["home_score"].to_numpy(dtype=float), sample_weight=sample_weight)
        self.reg_away.fit(X, feat["away_score"].to_numpy(dtype=float), sample_weight=sample_weight)
        return self

    def predict_frame(self, feat: pd.DataFrame) -> pd.DataFrame:
        X = feat[FEATURES].to_numpy(dtype=float)
        P = self.clf.predict_proba(X)
        col = {c: i for i, c in enumerate(self.classes_)}
        out = pd.DataFrame({
            "p_home": P[:, col["H"]],
            "p_draw": P[:, col["D"]],
            "p_away": P[:, col["A"]],
            "exp_home": np.clip(self.reg_home.predict(X), 0.05, 8.0),
            "exp_away": np.clip(self.reg_away.predict(X), 0.05, 8.0),
        }, index=feat.index)
        return out

    def predict_match(self, home: str, away: str, neutral: bool = False,
                      feat_row: pd.DataFrame | None = None, **_) -> Prediction:
        if feat_row is None:
            raise ValueError("GBoostModel.predict_match needs a feature row.")
        r = self.predict_frame(feat_row).iloc[0]
        return Prediction(home, away, float(r.p_home), float(r.p_draw), float(r.p_away),
                          float(r.exp_home), float(r.exp_away))
