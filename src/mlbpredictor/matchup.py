"""Batter-vs-pitcher matchup via the odds-ratio (log5) method.

Given a batter's and a pitcher's projected PA-outcome rate vectors and the league
baseline, the expected outcome distribution for that matchup is the multiplicative
odds-ratio blend (Tango's generalisation of Bill James' log5 to multiple outcomes):

    p_o  ∝  batter_o · pitcher_o / league_o     (then renormalised)

This has the right limits: an average batter facing pitcher P yields P's rates; an
average pitcher facing batter B yields B's rates; two average players yield league.
An optional park factor mildly scales the offensive (non-out) events at the venue.
"""
from __future__ import annotations

import numpy as np

from .retrosheet import PA_OUTCOMES

_OUT = PA_OUTCOMES.index("OUT")
_EPS = 1e-9


def scale_offense(vec: np.ndarray, factor: float) -> np.ndarray:
    """Scale non-out outcomes by ``factor`` and rebalance OUT (normalisation-safe)."""
    if factor == 1.0:
        return vec
    adj = vec.astype(float).copy()
    adj[:_OUT] *= factor
    adj[_OUT + 1:] *= factor
    non_out = adj.sum() - adj[_OUT]
    adj[_OUT] = max(1e-6, 1.0 - non_out)
    return adj / adj.sum()


def matchup_probs(batter: np.ndarray, pitcher: np.ndarray, league: np.ndarray,
                  park_factor: float = 1.0) -> np.ndarray:
    """Return the PA-outcome distribution for a batter vs a pitcher.

    All inputs are length-``len(PA_OUTCOMES)`` probability vectors.
    """
    num = batter * pitcher / np.maximum(league, _EPS)
    s = num.sum()
    p = num / s if s > 0 else league.copy()
    return scale_offense(p, park_factor)
