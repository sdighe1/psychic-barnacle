"""CLV / market-comparison metrics (pure, no network)."""
import numpy as np
import pandas as pd

from mlbpredictor.clv import (american_to_decimal, american_to_prob, clv_report,
                              no_vig_two_way, normalize_odds_frame)


def test_american_to_prob():
    assert abs(american_to_prob(-110) - 110 / 210) < 1e-9
    assert abs(american_to_prob(+150) - 100 / 250) < 1e-9
    assert abs(american_to_prob("+100") - 0.5) < 1e-9
    assert abs(american_to_prob("EVEN") - 0.5) < 1e-9


def test_american_to_decimal():
    assert abs(american_to_decimal(-110) - (1 + 100 / 110)) < 1e-9
    assert abs(american_to_decimal(+150) - 2.5) < 1e-9


def test_no_vig_removes_margin():
    ph, pa = american_to_prob(-150), american_to_prob(+130)
    assert ph + pa > 1.0                              # has vig
    nh, na = no_vig_two_way(ph, pa)
    assert abs(nh + na - 1.0) < 1e-9                   # vig removed
    assert nh > na                                     # favorite still favored


def _frame():
    # g1: model loves the home favorite and it wins; g3: model bets away and loses; g2: no edge.
    return pd.DataFrame({
        "model_p_home": [0.90, 0.50, 0.20],
        "actual_home_win": [1, 0, 1],
        "close_home_ml": [-300, -110, -110],
        "close_away_ml": [+250, -110, -110],
    })


def test_clv_report_bet_selection_and_roi():
    rep = clv_report(_frame(), edge_threshold=0.02)
    assert rep["n_games"] == 3
    b = rep["betting"]
    assert b["n_bets"] == 2 and b["n_home_bets"] == 1 and b["n_away_bets"] == 1
    # g1 home bet wins (+0.333u), g3 away bet loses (-1u) -> ROI negative, hit rate 0.5
    assert abs(b["hit_rate"] - 0.5) < 1e-9
    assert b["roi"] < 0
    assert "log_loss" in rep["market"] and "log_loss" in rep["model"]


def test_clv_report_sharp_model_beats_market():
    # A model that matches outcomes exactly should out-sharpen a noisy market.
    df = pd.DataFrame({
        "model_p_home": [0.95, 0.05, 0.95, 0.05],
        "actual_home_win": [1, 0, 1, 0],
        "close_home_ml": [-120, +120, -120, +120],
        "close_away_ml": [+110, -110, +110, -110],
    })
    rep = clv_report(df)
    assert rep["model"]["log_loss"] < rep["market"]["log_loss"]
    assert rep["model"]["beats_market_logloss"] is True


def test_clv_beat_the_close_with_opening_lines():
    df = _frame().assign(open_home_ml=[-250, -110, -105], open_away_ml=[+210, -110, -115])
    rep = clv_report(df, edge_threshold=0.02)
    assert "clv" in rep
    assert 0.0 <= rep["clv"]["beat_close_rate"] <= 1.0


def test_normalize_odds_frame_maps_teams_and_dates():
    df = pd.DataFrame({"date": ["2025-04-04"], "home_team": ["LAD"], "away_team": ["SD"],
                       "close_home_ml": ["-145"], "close_away_ml": ["+122"]})
    out = normalize_odds_frame(df)
    assert out.iloc[0]["home_team"] == "LAN" and out.iloc[0]["away_team"] == "SDN"
    assert out.iloc[0]["date"] == "2025-04-04"
