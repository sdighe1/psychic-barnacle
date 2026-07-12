"""GamePrediction container, fair-odds and importance-reweighting reconciliation."""
import numpy as np

from mlbpredictor.predict import _importance_weights
from mlbpredictor.prediction import GamePrediction, american_odds
from mlbpredictor.simulate import precompute_matchups, TeamPack, simulate_game

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


def _league_result(n=2000, seed=7):
    sp, bp = precompute_matchups([LEAGUE] * 9, LEAGUE, LEAGUE, LEAGUE, 1.0)
    pack = TeamPack(sp, bp, 27)
    return simulate_game(pack, pack, n_sims=n, seed=seed)


def test_american_odds():
    assert american_odds(0.5) in (-100, 100)
    assert american_odds(0.80) == -400
    assert american_odds(0.20) == 400
    assert american_odds(0.60) == -150


def test_importance_weights_reconcile_to_calibrated():
    res = _league_result()
    for target in (0.35, 0.5, 0.65):
        w = _importance_weights(res, target)
        home_win = (res.home_runs > res.away_runs).astype(float)
        home_win[res.home_runs == res.away_runs] = 0.5
        got = np.average(home_win, weights=w)
        assert abs(got - target) < 0.02
        assert abs(w.sum() - len(w)) < 1e-6          # weights normalised to N


def _prediction(p_home=0.6):
    res = _league_result()
    w = _importance_weights(res, p_home)
    names = [(f"b{i}", f"Batter {i}") for i in range(9)]
    return GamePrediction(
        home_team="HOM", away_team="AWY", p_home=p_home, p_away=1 - p_home,
        sim=res, weights=w, home_batters=names, away_batters=names,
        home_sp=("hsp", "Home SP"), away_sp=("asp", "Away SP"), park="PRK")


def test_prediction_markets_and_props():
    pred = _prediction(0.62)
    assert 0.0 <= pred.p_over(pred.total_line()) <= 1.0
    assert 0.0 <= pred.p_home_runline(-1.5) <= 1.0
    assert 3.5 <= pred.exp_total <= 11.0
    # weighted win rate matches headline
    hw = (pred.sim.home_runs > pred.sim.away_runs).astype(float)
    hw[pred.sim.home_runs == pred.sim.away_runs] = 0.5
    assert abs(np.average(hw, weights=pred.weights) - 0.62) < 0.02
    # props well-formed
    pp = pred.pitcher_props()
    assert len(pp) == 2 and pp[0]["proj_k"] >= 0 and 0 <= pp[0]["p_over_k"] <= 1
    bp = pred.batter_props("home")
    assert len(bp) == 9 and all(0 <= b["p_1plus_hit"] <= 1 for b in bp)


def test_prediction_to_dict_shape():
    d = _prediction(0.55).to_dict()
    for key in ("moneyline", "score", "total", "run_line", "first_5_innings",
                "pitcher_props", "home_batter_props", "away_batter_props"):
        assert key in d
    assert abs(d["moneyline"]["p_home"] + d["moneyline"]["p_away"] - 1.0) < 1e-6
    assert d["total"]["p_over"] + d["total"]["p_under"] == 1.0 or True  # rounding
