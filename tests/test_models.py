"""Tests for Elo, the offense/expected-runs model, the run model and the ensemble."""
import numpy as np
import pandas as pd

from mlbpredictor.backtest import log_loss
from mlbpredictor.models.baselines import EloLogistic, HomeBaseRate
from mlbpredictor.models.ensemble import EnsembleModel
from mlbpredictor.models.rundist import RunDistModel
from mlbpredictor.offense import OffenseModel
from mlbpredictor.ratings import EloRatingSystem

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


# --------------------------- offense ---------------------------------- #
def test_expected_runs_league_average():
    off = OffenseModel(LEAGUE, league_runs_per_game=4.5)
    r = off.expected_runs([LEAGUE] * 9, LEAGUE, LEAGUE, park_factor=1.0)
    assert abs(r - 4.5) < 0.05                       # avg lineup vs avg arms -> league


def test_expected_runs_better_lineup_scores_more():
    off = OffenseModel(LEAGUE, 4.5)
    strong = LEAGUE.copy(); strong[:4] *= 1.4; strong[6] *= 0.8; strong /= strong.sum()
    r_strong = off.expected_runs([strong] * 9, LEAGUE, LEAGUE)
    r_avg = off.expected_runs([LEAGUE] * 9, LEAGUE, LEAGUE)
    assert r_strong > r_avg + 0.5
    # a hitters' park raises run expectancy
    assert off.expected_runs([LEAGUE] * 9, LEAGUE, LEAGUE, 1.15) > r_avg


# ----------------------------- Elo ------------------------------------ #
def test_elo_updates_and_conserves():
    games = pd.DataFrame({
        "date": pd.to_datetime(["2022-04-01", "2022-04-02", "2022-04-03"]),
        "home_team": ["A", "A", "B"], "away_team": ["B", "B", "A"],
        "home_score": [5, 6, 1], "away_score": [1, 2, 0], "season": [2022, 2022, 2022],
    })
    elo = EloRatingSystem()
    out = elo.fit_transform(games)
    assert {"home_elo", "away_elo", "elo_diff"}.issubset(out.columns)
    # A won all three -> A above 1500, B below, total conserved.
    assert elo.rating("A") > 1500 > elo.rating("B")
    assert abs((elo.rating("A") + elo.rating("B")) - 3000) < 1e-6


def test_elo_expected_monotonic():
    assert EloRatingSystem.expected(1600, 1500) > EloRatingSystem.expected(1500, 1500) > \
        EloRatingSystem.expected(1400, 1500)


# --------------------------- run model -------------------------------- #
def test_rundist_home_edge_and_bounds():
    rd = RunDistModel()
    out = rd.predict(4.5, 4.5)
    assert 0.5 < out["p_home_win"] < 0.6            # small home edge from hfa_runs
    assert abs(sum(out["total_pmf"]) - 1.0) < 1e-6
    hi = rd.predict(6.0, 3.5)
    assert hi["p_home_win"] > out["p_home_win"]     # more runs -> more likely to win


def test_rundist_over_prob_monotonic():
    rd = RunDistModel()
    pmf = rd.predict(4.5, 4.5)["total_pmf"]
    assert rd.over_prob(pmf, 6.5) > rd.over_prob(pmf, 8.5) > rd.over_prob(pmf, 10.5)


def test_rundist_frame_matches_scalar():
    rd = RunDistModel()
    fr = rd.predict_frame(np.array([5.0]), np.array([4.0]))
    sc = rd.predict(5.0, 4.0)
    assert abs(fr["p_home_win"].iloc[0] - sc["p_home_win"]) < 1e-9


# ---------------------------- ensemble -------------------------------- #
def _synthetic_feat(n=400, seed=0):
    rng = np.random.default_rng(seed)
    eh = rng.normal(4.6, 1.0, n).clip(1)
    ea = rng.normal(4.6, 1.0, n).clip(1)
    elo_diff = (eh - ea) * 60 + rng.normal(0, 40, n)
    # true home win prob driven by run diff + home edge
    p = 1 / (1 + np.exp(-((eh - ea) * 0.35 + 0.1)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "exp_home_runs": eh, "exp_away_runs": ea, "exp_run_diff": eh - ea,
        "elo_diff": elo_diff, "home_sp_quality": rng.normal(0.31, 0.02, n),
        "away_sp_quality": rng.normal(0.31, 0.02, n), "park_factor": 1.0,
        "home_rest": 3.0, "away_rest": 3.0, "home_win": y,
    })


def test_ensemble_blends_and_calibrates():
    tr, val, te = _synthetic_feat(600, 1), _synthetic_feat(400, 2), _synthetic_feat(400, 3)
    rd = RunDistModel()
    members = {"rundist": rd, "elo_logistic": EloLogistic().fit(tr),
               "base_rate": HomeBaseRate().fit(tr)}
    ens = EnsembleModel(rd, members).fit_blend(val)
    w = ens.weight_map
    assert abs(sum(w.values()) - 1.0) < 1e-6
    p = ens.predict_p_home(te)
    assert ((p >= 0) & (p <= 1)).all()
    # ensemble should be no worse than the base rate on the test set
    base = HomeBaseRate().fit(tr).predict_p_home(te)
    assert log_loss(p, te["home_win"].to_numpy()) <= log_loss(base, te["home_win"].to_numpy()) + 1e-6


def test_ensemble_member_probs_for():
    tr = _synthetic_feat(300, 1)
    rd = RunDistModel()
    members = {"rundist": rd, "elo_logistic": EloLogistic().fit(tr),
               "base_rate": HomeBaseRate().fit(tr)}
    ens = EnsembleModel(rd, members).fit_blend(tr)
    mp = ens.member_probs_for(tr)
    assert set(mp.keys()) == set(members.keys())
    for arr in mp.values():
        assert len(arr) == len(tr) and ((arr >= 0) & (arr <= 1)).all()


def test_rundist_total_quantiles_monotonic():
    rd = RunDistModel()
    eh, ea = np.array([4.0, 6.5]), np.array([4.0, 5.5])   # totals ~8 vs ~12
    q = rd.total_quantiles(eh, ea, [0.1, 0.5, 0.9])
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()   # increasing in q
    assert q[1, 1] > q[0, 1]                       # higher-scoring game has a higher median
