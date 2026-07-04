import pandas as pd

from wcpredictor.data import load_matches, normalize_team, tournament_weights


def test_normalize_team():
    assert normalize_team("West Germany") == "Germany"
    assert normalize_team("Czech Republic") == "Czechia"
    assert normalize_team("  Brazil ") == "Brazil"
    assert normalize_team("Nowhereland") == "Nowhereland"


def test_tournament_weights():
    assert tournament_weights("FIFA World Cup") == (60.0, 5)
    # Qualifiers must never score as finals.
    assert tournament_weights("FIFA World Cup qualification") == (40.0, 3)
    assert tournament_weights("Friendly") == (20.0, 1)
    assert tournament_weights("UEFA Euro") == (50.0, 4)


def test_load_matches_schema():
    df = load_matches()
    for col in ["date", "home_team", "away_team", "home_score", "away_score",
                "neutral", "outcome", "k_weight", "importance"]:
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["neutral"].dtype == bool
    assert set(df["outcome"].unique()) <= {"H", "D", "A"}
    # Chronologically sorted, no team playing itself.
    assert df["date"].is_monotonic_increasing
    assert (df["home_team"] != df["away_team"]).all()
