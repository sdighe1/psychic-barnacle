"""Calibrated moneyline ensemble (2-class).

Blends the members' home-win probabilities with non-negative weights that sum to
one (chosen to minimise validation log-loss), then applies a single temperature
for calibration. The negative-binomial run model is a member *and* the source of
the expected-runs / totals numbers the prediction carries, so the moneyline and the
projected score stay consistent. The daily simulator's run distribution is later
rescaled to this calibrated win probability (see :mod:`mlbpredictor.predict`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

_EPS = 1e-12


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _temper(p: np.ndarray, T: float) -> np.ndarray:
    logit = np.log(np.clip(p, _EPS, 1 - _EPS) / np.clip(1 - p, _EPS, 1 - _EPS))
    return _sigmoid(logit / T)


class EnsembleModel:
    RUNDIST = "rundist"

    def __init__(self, rundist, members: dict, ridge: float = 0.004):
        """``members`` maps name -> model; one entry (``rundist``) is the run model.

        ``ridge`` is a small L2 pull toward equal weights in the blend objective, so
        the optimiser can't collapse onto a single (possibly overfit) component.
        """
        self.rundist = rundist
        self.members = members
        self.names = list(members.keys())
        self.ridge = float(ridge)
        self.weights = np.ones(len(self.names)) / len(self.names)
        self.temperature = 1.0

    # ------------------------------------------------------------------ #
    def _member_probs(self, feat: pd.DataFrame) -> np.ndarray:
        cols = []
        for n in self.names:
            if n == self.RUNDIST:
                fr = self.rundist.predict_frame(feat["exp_home_runs"].to_numpy(),
                                                feat["exp_away_runs"].to_numpy())
                cols.append(fr["p_home_win"].to_numpy())
            else:
                cols.append(self.members[n].predict_p_home(feat))
        return np.vstack(cols)                      # (m, N)

    def fit_blend(self, val: pd.DataFrame) -> "EnsembleModel":
        P = self._member_probs(val)                 # (m, N)
        y = val["home_win"].to_numpy(dtype=float)
        m = len(self.names)

        def obj(w):
            w = np.clip(w, 0, None)
            s = w.sum()
            if s == 0:
                return 1e9
            wn = w / s
            b = wn @ P
            return _logloss(y, b) + self.ridge * float(np.sum((wn - 1.0 / m) ** 2))

        cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        res = minimize(obj, np.ones(m) / m, method="SLSQP", bounds=[(0.0, 1.0)] * m,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        self.weights = np.clip(res.x, 0, None)
        self.weights /= self.weights.sum()

        b = self.weights @ P
        tr = minimize_scalar(lambda T: _logloss(y, _temper(b, T)),
                             bounds=(0.5, 3.0), method="bounded")
        self.temperature = float(tr.x)
        return self

    @property
    def weight_map(self) -> dict:
        return {n: float(w) for n, w in zip(self.names, self.weights)}

    # ------------------------------------------------------------------ #
    def predict_p_home(self, feat: pd.DataFrame) -> np.ndarray:
        b = self.weights @ self._member_probs(feat)
        return _temper(b, self.temperature)

    def member_probs_for(self, feat: pd.DataFrame) -> dict:
        """Per-component home-win probabilities, keyed by member name.

        Used to gauge ensemble *agreement* for a game (a confidence signal): tight
        agreement across Elo / run model / gradient boosting → higher confidence.
        Returns ``{name: np.ndarray}`` (one value per row in ``feat``).
        """
        P = self._member_probs(feat)                # (m, N)
        return {n: P[i] for i, n in enumerate(self.names)}

    def predict_frame(self, feat: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_p_home(feat)
        rd = self.rundist.predict_frame(feat["exp_home_runs"].to_numpy(),
                                        feat["exp_away_runs"].to_numpy())
        return pd.DataFrame({
            "p_home": p, "p_away": 1 - p,
            "exp_home_runs": rd["exp_home_runs"].to_numpy(),
            "exp_away_runs": rd["exp_away_runs"].to_numpy(),
            "exp_total": rd["exp_total"].to_numpy(),
        }, index=feat.index)
