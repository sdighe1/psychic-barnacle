"""High-level prediction API.

:class:`Predictor` bundles a fitted :class:`~wcpredictor.features.FeatureBuilder`
with the fitted :class:`~wcpredictor.models.ensemble.EnsembleModel`, so callers
just do::

    pred = Predictor.load()
    print(pred.predict("Spain", "France").summary())

It pickles cleanly via joblib, which is how the trained model is persisted to
``outputs/model.joblib``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import pandas as pd

from .features import FeatureBuilder
from .models.baselines import BaseRateModel, EloLogisticModel
from .models.dixon_coles import DixonColesModel
from .models.ensemble import EnsembleModel
from .models.gboost import GBoostModel
from .paths import MODEL_PATH
from .prediction import Prediction


def build_members(home_advantage: float = 65.0) -> dict:
    """Fresh, unfitted component models keyed by name."""
    return {
        "dixon_coles": DixonColesModel(),
        "gboost": GBoostModel(),
        "elo_logistic": EloLogisticModel(),
        "base_rate": BaseRateModel(),
    }


def fit_members(members: dict, feat: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> dict:
    """Fit every component model on the feature frame ``feat``."""
    members["dixon_coles"].fit(feat, ref_date=ref_date)
    members["gboost"].fit(feat)
    members["elo_logistic"].fit(feat)
    members["base_rate"].fit(feat)
    return members


@dataclass
class Predictor:
    fb: FeatureBuilder
    ensemble: EnsembleModel
    metrics: dict = field(default_factory=dict)
    trained_through: str = ""

    # ------------------------------------------------------------------ #
    def predict(self, home: str, away: str, neutral: bool = True,
                date: pd.Timestamp | None = None) -> Prediction:
        """Predict a single fixture. World Cup matches default to ``neutral=True``."""
        feat_row = self.fb.match_features(home, away, neutral, date=date)
        return self.ensemble.predict_match(home, away, neutral, feat_row=feat_row)

    def teams(self) -> list[str]:
        return sorted(self.fb.elo.ratings_)

    def rankings(self, top: int | None = None) -> pd.DataFrame:
        return self.fb.elo.rankings(top)

    @property
    def weight_map(self) -> dict:
        return self.ensemble.weight_map

    # ------------------------------------------------------------------ #
    def save(self, path=MODEL_PATH) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH) -> "Predictor":
        return joblib.load(path)


def load_predictor(path=MODEL_PATH) -> Predictor:
    return Predictor.load(path)
