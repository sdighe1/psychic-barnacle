"""Unit tests for the Retrosheet parser (pure, no network)."""
import pandas as pd

from mlbpredictor.retrosheet import (PA_OUTCOMES, classify_batter_event,
                                     parse_event_text, parse_gamelog_text)


def test_classifier_covers_all_outcomes():
    cases = {
        "8/F78XD+": "OUT", "E3/G9L.B-1": "OUT", "64(1)3/GDP/G6": "OUT",
        "W": "BB", "IW": "BB", "W+WP.1-2": "BB",
        "S7/G6": "1B", "S9/G34.1-3": "1B",
        "D7/L7L": "2B", "T9/F9LD": "3B",
        "HR/F9LS.1-H": "HR", "H/L7": "HR",
        "HP.1-2": "HBP", "K": "SO", "K+SB2": "SO",
        "FC5/G5.1X2(654)": "OUT",
        # non-PA baserunning / admin events:
        "WP.2-3": None, "SB2": None, "NP": None, "DI": None, "PB.2-3": None, "CS2(24)": None,
    }
    for ev, exp in cases.items():
        assert classify_batter_event(ev) == exp, f"{ev} -> {classify_batter_event(ev)} != {exp}"


def test_classifier_empty_and_none():
    assert classify_batter_event("") is None
    assert classify_batter_event(None) is None


EVENT_SNIPPET = """id,TST202304070
info,visteam,AAA
info,hometeam,BBB
start,bat1,"Batter 1",0,1,7
start,bat2,"Batter 2",0,2,4
start,pitA,"Pitcher A",0,0,1
start,bat3,"Batter 3",1,1,3
start,bat4,"Batter 4",1,2,9
start,pitB,"Pitcher B",1,0,1
play,1,0,bat1,00,X,S7
play,1,0,bat2,00,X,K
play,1,1,bat3,00,X,HR/F9
play,1,1,bat4,00,X,W
sub,pitC,"Pitcher C",1,0,1
play,2,0,bat1,00,X,D7
"""


def test_parse_event_attribution():
    bat, pit, bull = parse_event_text(EVENT_SNIPPET)
    # Batters
    assert dict(bat["bat1"]) == {"1B": 1, "PA": 2, "2B": 1}
    assert bat["bat2"]["SO"] == 1
    assert bat["bat3"]["HR"] == 1
    assert bat["bat4"]["BB"] == 1
    # Pitchers: side-0 batters face the HOME pitcher (pitB, then reliever pitC).
    assert pit["pitB"]["1B"] == 1 and pit["pitB"]["SO"] == 1 and pit["pitB"]["PA"] == 2
    assert pit["pitA"]["HR"] == 1 and pit["pitA"]["BB"] == 1     # faced home batters
    assert pit["pitC"]["2B"] == 1
    # Bullpen: pitC is a reliever for the home team BBB.
    assert bull["BBB"]["2B"] == 1 and bull["BBB"]["PA"] == 1
    assert "AAA" not in bull                                     # no relievers used by AAA


def _make_gamelog_row():
    rec = [""] * 161
    rec[0] = "20230330"      # date
    rec[1] = "0"             # game number
    rec[3] = "MIL"           # visiting team
    rec[6] = "CHN"           # home team
    rec[9] = "3"             # visitor score
    rec[10] = "4"            # home score
    rec[12] = "D"            # day/night
    rec[16] = "CHI11"        # park
    rec[101] = "burnc002"    # visitor SP
    rec[103] = "strom001"    # home SP
    for i in range(9):
        rec[105 + i * 3] = f"v{i+1}"     # visitor batting order ids
        rec[132 + i * 3] = f"h{i+1}"     # home batting order ids
    return ",".join(rec)


def test_parse_gamelog_fields():
    df = parse_gamelog_text(_make_gamelog_row() + "\n")
    assert len(df) == 1
    r = df.iloc[0]
    assert r["home_team"] == "CHN" and r["away_team"] == "MIL"
    assert r["home_score"] == 4 and r["away_score"] == 3
    assert r["home_sp"] == "strom001" and r["away_sp"] == "burnc002"
    assert r["park"] == "CHI11"
    assert r["home_lineup"] == [f"h{i+1}" for i in range(9)]
    assert r["away_lineup"] == [f"v{i+1}" for i in range(9)]
    assert r["date"] == pd.Timestamp("2023-03-30")


def test_pa_outcomes_partition():
    # Every classified event is one of the declared outcomes.
    for ev in ["S7", "D8", "T9", "HR", "W", "HP", "K", "43"]:
        assert classify_batter_event(ev) in PA_OUTCOMES
