"""Time-weighted Dixon-Coles bivariate-Poisson goals model.

Each team ``t`` gets an attack rating ``a_t`` and a defence rating ``d_t``; with a
home-advantage term ``h`` the expected goals are::

    log λ_home = a_home + d_away + h · (venue is not neutral)
    log μ_away = a_away + d_home

Goals are Poisson with the Dixon-Coles low-score correlation correction ``ρ``
(Dixon & Coles, 1997), which fixes the well-known under-prediction of 0-0/1-0/
0-1/1-1 results. Recent matches are up-weighted with an exponential time decay.

Fitting is a single vectorised MLE (``scipy.optimize.minimize``, L-BFGS-B) with an
L2 ridge on the team ratings that both regularises sparse teams and resolves the
attack/defence level degeneracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from ..prediction import Prediction

MAX_GOALS = 10


def _dc_tau(hg, ag, lam, mu, rho):
    """Dixon-Coles correction for the four low-score cells (vectorised)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def dc_score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Full probability matrix ``M[i, j] = P(home i, away j)``."""
    i = np.arange(max_goals + 1)
    # Poisson pmf via logs for stability.
    log_ph = i * np.log(lam) - lam - gammaln(i + 1)
    log_pa = i * np.log(mu) - mu - gammaln(i + 1)
    M = np.exp(log_ph[:, None] + log_pa[None, :])
    M[0, 0] *= 1.0 - lam * mu * rho
    M[0, 1] *= 1.0 + lam * rho
    M[1, 0] *= 1.0 + mu * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 1e-15, None)
    return M / M.sum()


def dc_region_probs(lam: np.ndarray, mu: np.ndarray, rho: float,
                    max_goals: int = MAX_GOALS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised P(home win), P(draw), P(away win) for arrays of ``lam``/``mu``.

    Uses independent-Poisson region sums, then applies the Dixon-Coles correction
    to the four low-score cells analytically (no per-row matrix construction).
    """
    k = np.arange(max_goals + 1)
    gl = gammaln(k + 1.0)
    log_lam = np.log(lam)[:, None]
    log_mu = np.log(mu)[:, None]
    ph = np.exp(k[None, :] * log_lam - lam[:, None] - gl[None, :])   # (n, G+1)
    pa = np.exp(k[None, :] * log_mu - mu[:, None] - gl[None, :])
    ca = np.cumsum(pa, axis=1)                                       # P(away <= j)
    p_away_lt = np.concatenate([np.zeros((len(lam), 1)), ca[:, :-1]], axis=1)  # P(away < i)
    ch = np.cumsum(ph, axis=1)
    p_home_lt = np.concatenate([np.zeros((len(lam), 1)), ch[:, :-1]], axis=1)
    p_home = (ph * p_away_lt).sum(axis=1)
    p_away = (pa * p_home_lt).sum(axis=1)
    p_draw = (ph * pa).sum(axis=1)
    # Dixon-Coles corrections on the 2x2 low-score block.
    d00 = -ph[:, 0] * pa[:, 0] * lam * mu * rho     # draw cell (0,0)
    d11 = -ph[:, 1] * pa[:, 1] * rho                # draw cell (1,1)
    d01 = ph[:, 0] * pa[:, 1] * lam * rho           # away cell (0,1)
    d10 = ph[:, 1] * pa[:, 0] * mu * rho            # home cell (1,0)
    p_home = p_home + d10
    p_away = p_away + d01
    p_draw = p_draw + d00 + d11
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


class DixonColesModel:
    def __init__(self, half_life_days: float = 730.0, max_age_years: float = 12.0,
                 l2: float = 2.0, max_goals: int = MAX_GOALS):
        self.half_life_days = float(half_life_days)
        self.max_age_years = float(max_age_years)
        self.l2 = float(l2)
        self.max_goals = int(max_goals)
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_ = 0.25
        self.rho_ = -0.05
        self.teams_: list[str] = []

    # ------------------------------------------------------------------ #
    def fit(self, matches: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> "DixonColesModel":
        df = matches.dropna(subset=["home_score", "away_score"]).copy()
        ref = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max()
        min_date = ref - pd.Timedelta(days=365.25 * self.max_age_years)
        df = df[df["date"] >= min_date]
        if df.empty:
            raise ValueError("No matches in the Dixon-Coles fitting window.")

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        hi = df["home_team"].map(idx).to_numpy()
        ai = df["away_team"].map(idx).to_numpy()
        hg = df["home_score"].to_numpy().astype(int)
        ag = df["away_score"].to_numpy().astype(int)
        not_neutral = (~df["neutral"].astype(bool)).to_numpy().astype(float)

        age_days = (ref - df["date"]).dt.days.to_numpy().astype(float)
        weights = 0.5 ** (age_days / self.half_life_days)
        lg_hg = gammaln(hg + 1.0)
        lg_ag = gammaln(ag + 1.0)

        def nll(params):
            atk = params[:n]
            dfn = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]
            log_lam = atk[hi] + dfn[ai] + home_adv * not_neutral
            log_mu = atk[ai] + dfn[hi]
            lam = np.exp(log_lam)
            mu = np.exp(log_mu)
            ll_home = hg * log_lam - lam - lg_hg
            ll_away = ag * log_mu - mu - lg_ag
            tau = _dc_tau(hg, ag, lam, mu, rho)
            ll_tau = np.log(np.clip(tau, 1e-12, None))
            ll = weights * (ll_home + ll_away + ll_tau)
            ridge = self.l2 * (np.dot(atk, atk) + np.dot(dfn, dfn))
            return -ll.sum() + ridge

        x0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 2.0), (-0.25, 0.25)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "maxfun": 100000})

        p = res.x
        self.teams_ = teams
        self.attack_ = {t: float(p[idx[t]]) for t in teams}
        self.defense_ = {t: float(p[n + idx[t]]) for t in teams}
        self.home_adv_ = float(p[2 * n])
        self.rho_ = float(p[2 * n + 1])
        return self

    # ------------------------------------------------------------------ #
    def _lambda_mu(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        ah = self.attack_.get(home, 0.0)
        dh = self.defense_.get(home, 0.0)
        aa = self.attack_.get(away, 0.0)
        da = self.defense_.get(away, 0.0)
        h = 0.0 if neutral else self.home_adv_
        lam = float(np.exp(ah + da + h))
        mu = float(np.exp(aa + dh))
        # Guard against pathological values.
        return min(lam, 8.0), min(mu, 8.0)

    def _rates_frame(self, feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ah = feat["home_team"].map(self.attack_).fillna(0.0).to_numpy()
        dh = feat["home_team"].map(self.defense_).fillna(0.0).to_numpy()
        aa = feat["away_team"].map(self.attack_).fillna(0.0).to_numpy()
        da = feat["away_team"].map(self.defense_).fillna(0.0).to_numpy()
        h = np.where(feat["neutral"].astype(bool).to_numpy(), 0.0, self.home_adv_)
        lam = np.clip(np.exp(ah + da + h), 0.01, 8.0)
        mu = np.clip(np.exp(aa + dh), 0.01, 8.0)
        return lam, mu

    def predict_frame(self, feat: pd.DataFrame) -> pd.DataFrame:
        lam, mu = self._rates_frame(feat)
        p_home, p_draw, p_away = dc_region_probs(lam, mu, self.rho_, self.max_goals)
        return pd.DataFrame({
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "exp_home": lam, "exp_away": mu,
        }, index=feat.index)

    def predict_match(self, home: str, away: str, neutral: bool = False, **_) -> Prediction:
        lam, mu = self._lambda_mu(home, away, neutral)
        M = dc_score_matrix(lam, mu, self.rho_, self.max_goals)
        p_home = float(np.tril(M, -1).sum())   # home goals > away goals
        p_away = float(np.triu(M, 1).sum())
        p_draw = float(np.trace(M))
        goals = np.arange(self.max_goals + 1)
        exp_h = float((M.sum(axis=1) * goals).sum())
        exp_a = float((M.sum(axis=0) * goals).sum())
        return Prediction(home, away, p_home, p_draw, p_away, exp_h, exp_a, score_matrix=M)
