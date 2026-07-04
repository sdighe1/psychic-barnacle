import numpy as np
import pandas as pd

from wcpredictor.elo import EloRatingSystem, mov_multiplier


def test_mov_multiplier():
    assert mov_multiplier(0) == 1.0
    assert mov_multiplier(1) == 1.0
    assert mov_multiplier(2) == 1.5
    assert mov_multiplier(3) == (11 + 3) / 8


def test_expected_symmetry():
    elo = EloRatingSystem(home_advantage=0.0)
    assert abs(elo.expected_home(1500, 1500, neutral=True) - 0.5) < 1e-9
    # Home advantage tilts the expectation upward.
    elo2 = EloRatingSystem(home_advantage=100.0)
    assert elo2.expected_home(1500, 1500, neutral=False) > 0.5


def test_stronger_team_gains_rating():
    # A beats B 3-0 repeatedly at neutral venues.
    n = 30
    df = pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=n, freq="30D"),
        "home_team": ["A"] * n,
        "away_team": ["B"] * n,
        "home_score": [3] * n,
        "away_score": [0] * n,
        "neutral": [True] * n,
        "k_weight": [40.0] * n,
    })
    elo = EloRatingSystem()
    out = elo.fit_transform(df)
    assert elo.rating("A") > 1500 > elo.rating("B")
    # Pre-match ratings are as-of: first row sees the initial 1500.
    assert out["home_elo"].iloc[0] == 1500.0
    assert out["home_elo"].iloc[-1] > out["home_elo"].iloc[0]
