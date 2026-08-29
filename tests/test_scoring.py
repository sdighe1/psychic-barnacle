"""Scoring: known stat lines -> known ESPN points for each format."""
from __future__ import annotations

import pandas as pd

from ffauction import scoring


def test_wr_line_across_formats():
    line = {"rec": 100, "rec_yds": 1000, "rec_td": 8}
    # base (no reception pts): 1000*0.1 + 8*6 = 148
    assert scoring.points_from_stats(line, scoring.rules_for_format("standard")) == 148.0
    assert scoring.points_from_stats(line, scoring.rules_for_format("half_ppr")) == 148.0 + 50.0
    assert scoring.points_from_stats(line, scoring.rules_for_format("ppr")) == 148.0 + 100.0


def test_qb_line():
    line = {"pass_yds": 4500, "pass_td": 35, "pass_int": 10, "rush_yds": 300, "rush_td": 3}
    # 4500*.04 + 35*4 + 10*-2 + 300*.1 + 3*6 = 180 + 140 - 20 + 30 + 18
    assert scoring.points_from_stats(line, scoring.rules_for_format("standard")) == 348.0


def test_fumbles_and_return_tds():
    line = {"rush_yds": 100, "fumbles_lost": 2, "st_td": 1}
    # 10 - 4 + 6 = 12
    assert scoring.points_from_stats(line, scoring.rules_for_format("half_ppr")) == 12.0


def test_override_is_format_agnostic():
    line = {"proj_points_override": 130.0}
    for fmt in scoring.FORMATS:
        assert scoring.points_from_stats(line, scoring.rules_for_format(fmt)) == 130.0


def test_vectorised_matches_scalar():
    df = pd.DataFrame([
        {"rec": 80, "rec_yds": 1100, "rec_td": 9},
        {"pass_yds": 4000, "pass_td": 30, "pass_int": 12},
    ])
    series = scoring.project_points(df, "half_ppr")
    scalar = [scoring.points_from_stats(r, scoring.rules_for_format("half_ppr"))
              for _, r in df.iterrows()]
    assert list(series) == [round(s, 2) for s in scalar]


def test_missing_columns_default_to_zero():
    assert scoring.points_from_stats({}, scoring.rules_for_format("ppr")) == 0.0
