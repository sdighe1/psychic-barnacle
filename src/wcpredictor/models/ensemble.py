"""Calibrated blend of the component models.

The ensemble:

1. **Blends** the members' win/draw/loss probabilities with non-negative weights
   that sum to 1, chosen to minimise validation log-loss.
2. **Temperature-scales** the blend (a single parameter) for calibration.
3. For a scoreline, takes the **Dixon-Coles probability matrix and rescales its
   home-win / draw / away-win regions** to match the blended (more accurate)
   outcome probabilities — so the projected exact score is always consistent
   with the headline W/D/L numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from ..prediction import Prediction

_LABEL_IDX = {"H": 0, "D": 1, "A": 2}


def _temper(B: np.ndarray, T: float) -> np.ndarray:
    Bt = np.clip(B, 1e-12, 1.0) ** (1.0 / T)
    return Bt / Bt.sum(axis=1, keepdims=True)


def _logloss(y_idx: np.ndarray, B: np.ndarray) -> float:
    B = np.clip(B, 1e-12, 1.0)
    return float(-np.mean(np.log(B[np.arange(len(y_idx)), y_idx])))


def rescale_matrix(M: np.ndarray, p_home: float, p_draw: float, p_away: float) -> np.ndarray:
    """Scale the three outcome regions of ``M`` to the target probabilities."""
    M = M.copy()
    home_mask = np.tril(np.ones_like(M, dtype=bool), -1)   # home > away
    away_mask = np.triu(np.ones_like(M, dtype=bool), 1)    # away > home
    draw_mask = np.eye(M.shape[0], dtype=bool)
    for mask, target in ((home_mask, p_home), (draw_mask, p_draw), (away_mask, p_away)):
        s = M[mask].sum()
        if s > 0:
            M[mask] *= target / s
    return M / M.sum()


class EnsembleModel:
    def __init__(self, members: dict, dc_name: str = "dixon_coles"):
        self.members = members
        self.names = list(members.keys())
        self.dc_name = dc_name
        self.weights = np.ones(len(self.names)) / len(self.names)
        self.temperature = 1.0

    # ------------------------------------------------------------------ #
    def _member_frames(self, feat: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {n: m.predict_frame(feat) for n, m in self.members.items()}

    def _stack(self, frames: dict[str, pd.DataFrame]) -> np.ndarray:
        return np.stack([frames[n][["p_home", "p_draw", "p_away"]].to_numpy() for n in self.names])

    def _blend(self, frames: dict[str, pd.DataFrame]) -> np.ndarray:
        P = self._stack(frames)                       # (m, N, 3)
        B = np.tensordot(self.weights, P, axes=(0, 0))  # (N, 3)
        B = np.clip(B, 1e-12, 1.0)
        B = B / B.sum(axis=1, keepdims=True)
        return _temper(B, self.temperature)

    # ------------------------------------------------------------------ #
    def fit_blend(self, feat_val: pd.DataFrame) -> "EnsembleModel":
        frames = self._member_frames(feat_val)
        P = self._stack(frames)                        # (m, N, 3)
        y = feat_val["outcome"].map(_LABEL_IDX).to_numpy()
        m = len(self.names)

        def obj(w):
            w = np.clip(w, 0, None)
            if w.sum() == 0:
                return 1e9
            w = w / w.sum()
            B = np.tensordot(w, P, axes=(0, 0))
            B = np.clip(B, 1e-12, 1.0)
            B = B / B.sum(axis=1, keepdims=True)
            return _logloss(y, B)

        cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        res = minimize(obj, np.ones(m) / m, method="SLSQP", bounds=[(0.0, 1.0)] * m,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        self.weights = np.clip(res.x, 0, None)
        self.weights /= self.weights.sum()

        # Temperature on the blended (untempered) probs.
        B = np.tensordot(self.weights, P, axes=(0, 0))
        B = np.clip(B, 1e-12, 1.0)
        B = B / B.sum(axis=1, keepdims=True)
        tr = minimize_scalar(lambda T: _logloss(y, _temper(B, T)),
                             bounds=(0.5, 3.0), method="bounded")
        self.temperature = float(tr.x)
        return self

    @property
    def weight_map(self) -> dict[str, float]:
        return {n: float(w) for n, w in zip(self.names, self.weights)}

    # ------------------------------------------------------------------ #
    def predict_frame(self, feat: pd.DataFrame) -> pd.DataFrame:
        frames = self._member_frames(feat)
        B = self._blend(frames)
        dc = frames[self.dc_name]
        return pd.DataFrame({
            "p_home": B[:, 0], "p_draw": B[:, 1], "p_away": B[:, 2],
            "exp_home": dc["exp_home"].to_numpy(), "exp_away": dc["exp_away"].to_numpy(),
        }, index=feat.index)

    def predict_match(self, home: str, away: str, neutral: bool = False,
                      feat_row: pd.DataFrame | None = None, **_) -> Prediction:
        if feat_row is None:
            raise ValueError("EnsembleModel.predict_match needs a feature row.")
        # DC is asked by team name (it owns the scoreline matrix); the other
        # members score the engineered feature row.
        prob_rows = {}
        dc_pred = self.members[self.dc_name].predict_match(home, away, neutral)
        prob_rows[self.dc_name] = np.array([dc_pred.p_home, dc_pred.p_draw, dc_pred.p_away])
        for name, model in self.members.items():
            if name == self.dc_name:
                continue
            r = model.predict_frame(feat_row).iloc[0]
            prob_rows[name] = np.array([r["p_home"], r["p_draw"], r["p_away"]])
        P = np.stack([prob_rows[n] for n in self.names])       # (m, 3)
        B = np.tensordot(self.weights, P, axes=(0, 0))
        B = np.clip(B, 1e-12, 1.0)
        B = B / B.sum()
        B = _temper(B[None, :], self.temperature)[0]
        M = rescale_matrix(dc_pred.score_matrix, B[0], B[1], B[2])
        goals = np.arange(M.shape[0])
        exp_h = float((M.sum(axis=1) * goals).sum())
        exp_a = float((M.sum(axis=0) * goals).sum())
        return Prediction(home, away, float(B[0]), float(B[1]), float(B[2]), exp_h, exp_a, score_matrix=M)
