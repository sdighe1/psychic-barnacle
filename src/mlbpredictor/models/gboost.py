"""Gradient-boosting moneyline model over the engineered game features.

scikit-learn histogram gradient boosting (no native build deps) on the as-of
features (Elo, expected runs, starter quality, park, rest). Captures non-linear
interactions the parametric run model misses. Heavily regularised because the
per-game signal in baseball is genuinely weak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..features import FEATURES


class GBoostModel:
    def __init__(self, learning_rate: float = 0.03, max_iter: int = 300,
                 max_depth: int = 3, min_samples_leaf: int = 80,
                 l2_regularization: float = 1.0, random_state: int = 0):
        self.clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
            random_state=random_state)

    def fit(self, feat: pd.DataFrame) -> "GBoostModel":
        X = feat[FEATURES].to_numpy(dtype=float)
        self.clf.fit(X, feat["home_win"].to_numpy())
        self._pos = list(self.clf.classes_).index(1)
        return self

    def predict_p_home(self, feat: pd.DataFrame) -> np.ndarray:
        X = feat[FEATURES].to_numpy(dtype=float)
        return self.clf.predict_proba(X)[:, self._pos]
