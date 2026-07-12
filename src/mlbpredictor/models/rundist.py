"""Negative-binomial run distribution — the fast run/total/run-line component.

Given each team's expected runs, model runs as independent Negative Binomials
(baseball run totals are over-dispersed relative to Poisson). From the two
distributions we derive the moneyline, the totals distribution and run-line
probabilities in closed form — instantly, over thousands of games — so this is the
run-based component in the ensemble and the backtest. A small home-field run bump
gives the home side its ~54% edge before ensemble calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom


class RunDistModel:
    def __init__(self, nb_size: float = 4.0, hfa_runs: float = 0.18, max_runs: int = 26):
        self.nb_size = float(nb_size)
        self.hfa_runs = float(hfa_runs)
        self.max_runs = int(max_runs)
        self._k = np.arange(max_runs + 1)

    # ------------------------------------------------------------------ #
    def fit_dispersion(self, team_runs: np.ndarray) -> "RunDistModel":
        """Method-of-moments Negative-Binomial size from observed team run totals."""
        m = float(np.mean(team_runs))
        v = float(np.var(team_runs))
        if v > m > 0:
            self.nb_size = float(np.clip(m * m / (v - m), 1.5, 20.0))
        return self

    def _pmf(self, mu: float) -> np.ndarray:
        mu = max(mu, 1e-3)
        p = self.nb_size / (self.nb_size + mu)
        pmf = nbinom.pmf(self._k, self.nb_size, p)
        pmf[-1] += max(0.0, 1.0 - pmf.sum())      # dump the tail into the last bin
        return pmf

    def _pmf_matrix(self, mu: np.ndarray) -> np.ndarray:
        mu = np.clip(mu, 1e-3, None)
        p = self.nb_size / (self.nb_size + mu)          # (N,)
        pmf = nbinom.pmf(self._k[None, :], self.nb_size, p[:, None])  # (N, K+1)
        pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
        return pmf

    # ------------------------------------------------------------------ #
    def predict(self, exp_home: float, exp_away: float) -> dict:
        """Full single-game prediction: probabilities + distributions."""
        ph = self._pmf(exp_home + self.hfa_runs)
        pa = self._pmf(exp_away)
        joint = np.outer(ph, pa)                          # joint[i,j]=P(home=i,away=j)
        p_home = float(np.tril(joint, -1).sum())          # home > away
        p_tie = float(np.trace(joint))
        p_home_win = p_home + 0.5 * p_tie
        total_pmf = np.convolve(ph, pa)
        margin = self._k[:, None] - self._k[None, :]      # home - away
        return {
            "p_home_win": p_home_win,
            "exp_home_runs": float(exp_home + self.hfa_runs),
            "exp_away_runs": float(exp_away),
            "total_pmf": total_pmf,
            "p_home_minus_1p5": float(joint[margin >= 2].sum()),
            "p_away_minus_1p5": float(joint[margin <= -2].sum()),
            "joint": joint,
        }

    @staticmethod
    def over_prob(total_pmf: np.ndarray, line: float) -> float:
        k = np.arange(len(total_pmf))
        over = total_pmf[k > line].sum()
        push = total_pmf[k == line].sum() if float(line).is_integer() else 0.0
        return float(over + 0.5 * push)

    # ------------------------------------------------------------------ #
    def predict_frame(self, exp_home: np.ndarray, exp_away: np.ndarray) -> pd.DataFrame:
        """Vectorised moneyline + expected totals for a whole schedule (fast)."""
        ph = self._pmf_matrix(np.asarray(exp_home, float) + self.hfa_runs)
        pa = self._pmf_matrix(np.asarray(exp_away, float))
        cdf_away = np.cumsum(pa, axis=1)
        p_away_lt = np.concatenate([np.zeros((len(pa), 1)), cdf_away[:, :-1]], axis=1)
        p_home_gt = (ph * p_away_lt).sum(axis=1)          # P(home > away)
        p_tie = (ph * pa).sum(axis=1)
        p_home_win = p_home_gt + 0.5 * p_tie
        exp_h = np.asarray(exp_home, float) + self.hfa_runs
        exp_a = np.asarray(exp_away, float)
        return pd.DataFrame({
            "p_home_win": np.clip(p_home_win, 1e-6, 1 - 1e-6),
            "exp_home_runs": exp_h,
            "exp_away_runs": exp_a,
            "exp_total": exp_h + exp_a,
        })
