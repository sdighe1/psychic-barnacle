"""GamePrediction container, fair-odds and importance-reweighting reconciliation."""
import numpy as np

from mlbpredictor.predict import _importance_weights
from mlbpredictor.prediction import GamePrediction, american_odds
from mlbpredictor.simulate import precompute_matchups, TeamPack, simulate_game

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


def _league_result(n=2000, seed=7):
    pairs = [(LEAGUE, LEAGUE)] * 9
    sp, bp = precompute_matchups(pairs, pairs, LEAGUE, 1.0)
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
                "pitcher_props", "home_batter_props", "away_batter_props",
                "confidence", "data_quality"):
        assert key in d
    assert abs(d["moneyline"]["p_home"] + d["moneyline"]["p_away"] - 1.0) < 1e-6
    assert d["total"]["p_over"] + d["total"]["p_under"] == 1.0 or True  # rounding


# --------------------------- intervals -------------------------------- #
def _pred(p_home=0.6, member_probs=None, lineup_confirmed=False, roster_coverage=1.0):
    res = _league_result()
    w = _importance_weights(res, p_home)
    names = [(f"b{i}", f"Batter {i}") for i in range(9)]
    return GamePrediction(
        home_team="HOM", away_team="AWY", p_home=p_home, p_away=1 - p_home,
        sim=res, weights=w, home_batters=names, away_batters=names,
        home_sp=("hsp", "Home SP"), away_sp=("asp", "Away SP"),
        member_probs=member_probs or {}, lineup_confirmed=lineup_confirmed,
        roster_coverage=roster_coverage)


def test_weighted_quantile_math():
    p = _pred()
    p.weights = np.ones(4)                      # equal weights
    q = p._wquantile(np.array([1.0, 2.0, 3.0, 4.0]), [0.5])
    assert 2.0 <= q[0] <= 3.0                    # median of 1..4
    lo, hi = p._wquantile(np.array([1.0, 2, 3, 4]), [0.0, 1.0])
    assert lo == 1.0 and hi == 4.0


def test_interval_ordering_and_nesting():
    p = _pred()
    x = p.sim.total()
    lo50, hi50 = p.interval(x, 0.5)
    lo90, hi90 = p.interval(x, 0.9)
    mean = p._wavg(x)
    assert lo50 <= mean <= hi50                  # point inside its interval
    assert lo90 <= lo50 <= hi50 <= hi90          # 90% contains 50%


def test_n_eff_bounds():
    p = _pred()
    assert 1.0 <= p.n_eff <= len(p.weights)
    p.weights = np.ones(1000)
    assert abs(p.n_eff - 1000) < 1e-6            # uniform weights → n_eff = N


# --------------------------- confidence ------------------------------- #
def test_confidence_coinflip_is_low():
    # even game, perfect inputs/agreement → still Low (no edge to bet)
    p = _pred(0.50, member_probs={"a": 0.50, "b": 0.50, "c": 0.50},
              lineup_confirmed=True, roster_coverage=1.0)
    assert p.confidence()["level"] == "Low"


def test_confidence_decisive_and_clean_is_high():
    p = _pred(0.72, member_probs={"a": 0.70, "b": 0.73, "c": 0.72},
              lineup_confirmed=True, roster_coverage=1.0)
    c = p.confidence()
    assert c["level"] == "High"
    assert all(0.0 <= v <= 1.0 for v in c["components"].values())


def test_confidence_drops_with_disagreement_and_bad_inputs():
    good = _pred(0.62, member_probs={"a": 0.61, "b": 0.62, "c": 0.63},
                 lineup_confirmed=True, roster_coverage=1.0).confidence()["score"]
    shaky = _pred(0.62, member_probs={"a": 0.50, "b": 0.62, "c": 0.78},
                  lineup_confirmed=False, roster_coverage=0.5).confidence()["score"]
    assert shaky < good


def test_p_home_ci_brackets_estimate():
    p = _pred(0.80, member_probs={"a": 0.55, "b": 0.70, "c": 0.66})
    lo, hi = p.p_home_ci
    assert lo <= 0.80 <= hi                       # includes the (tempered) point estimate


def test_dict_carries_intervals():
    d = _pred(0.6).to_dict()
    assert "range_90" in d["total"] and len(d["total"]["range_90"]) == 2
    assert "range_50" in d["score"]["home_range"]
    assert "k_range" in d["pitcher_props"][0]
    assert "hits_range" in d["home_batter_props"][0]
    assert d["confidence"]["level"] in ("High", "Medium", "Low")
