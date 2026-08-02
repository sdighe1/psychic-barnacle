"""Backtest calibration helpers: interval coverage and confidence-band accuracy."""
import numpy as np
from scipy.stats import nbinom

from mlbpredictor.backtest import (accuracy_by_confidence, evaluate_total_intervals,
                                   interval_coverage)
from mlbpredictor.models.rundist import RunDistModel


def test_interval_coverage_basic():
    lo = np.array([2, 3, 4])
    hi = np.array([6, 7, 8])
    actual = np.array([5, 10, 4])                 # in, out, in (boundary)
    assert abs(interval_coverage(lo, hi, actual) - 2 / 3) < 1e-9


def test_total_interval_coverage_near_nominal():
    rng = np.random.default_rng(0)
    n = 4000
    rd = RunDistModel(nb_size=4.0, hfa_runs=0.0)
    eh = rng.uniform(3.5, 5.5, n)
    ea = rng.uniform(3.5, 5.5, n)
    # draw actual totals from the very same model
    p_h = rd.nb_size / (rd.nb_size + eh)
    p_a = rd.nb_size / (rd.nb_size + ea)
    actual = nbinom.rvs(rd.nb_size, p_h, random_state=rng) + \
        nbinom.rvs(rd.nb_size, p_a, random_state=rng)
    cov = evaluate_total_intervals(rd, eh, ea, actual, levels=(0.5, 0.9))
    assert 0.42 <= cov["coverage_50"] <= 0.60      # ~50%
    assert 0.85 <= cov["coverage_90"] <= 0.96      # ~90%


def test_accuracy_rises_with_confidence():
    rng = np.random.default_rng(1)
    n = 6000
    p_home = rng.uniform(0.30, 0.70, n)
    y = (rng.random(n) < p_home).astype(int)       # outcomes follow the true prob
    spread = np.full(n, 0.02)                       # components agree
    bands = accuracy_by_confidence(p_home, spread, y)
    # decisive (High) games are more predictable than coin-flips (Low)
    if "High" in bands and "Low" in bands:
        assert bands["High"]["accuracy"] > bands["Low"]["accuracy"]
    assert all(0.0 <= m["accuracy"] <= 1.0 for m in bands.values())
