"""Projection build, provider import and accuracy blending (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ffauction import accuracy, projections, providers, scoring
from ffauction.scoring import OVERRIDE_COLUMN, STAT_COLUMNS


def _history() -> pd.DataFrame:
    """Three seasons of synthetic player-seasons in the data.py schema."""
    rows = []
    specs = [
        ("p_qb", "Star QB", "QB", dict(pass_yds=4600, pass_td=34, pass_int=10, rush_yds=250)),
        ("p_rb", "Star RB", "RB", dict(rush_yds=1300, rush_td=12, rec=45, rec_yds=380, carries=280, targets=60)),
        ("p_wr", "Star WR", "WR", dict(rec=105, rec_yds=1400, rec_td=11, targets=150)),
        ("p_te", "Star TE", "TE", dict(rec=80, rec_yds=900, rec_td=7, targets=110)),
        ("p_wr2", "Depth WR", "WR", dict(rec=40, rec_yds=520, rec_td=3, targets=70)),
    ]
    for season in (2022, 2023, 2024):
        for pid, name, pos, stats in specs:
            row = {c: 0.0 for c in STAT_COLUMNS + ["carries", "targets"]}
            row.update(stats)
            row.update({"player_id": pid, "season": season, "player": name,
                        "position": pos, "team": "FA", "age": 25.0 + (season - 2022),
                        "games": 16})
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_projections_schema_and_ranges():
    proj = projections.build_projections(_history(), target_season=2025)
    for col in ["player_id", "player", "position", "proj_games", "source"] + STAT_COLUMNS:
        assert col in proj.columns
    assert proj["proj_games"].between(1, 17).all()
    assert (proj["source"] == "history").all()
    pts = scoring.project_points(proj, "half_ppr")
    assert pts.max() > 0
    # the star WR should out-project the depth WR
    wr = proj[proj["position"] == "WR"].assign(pts=scoring.project_points(proj[proj["position"] == "WR"], "half_ppr"))
    assert wr.sort_values("pts", ascending=False).iloc[0]["player"] == "Star WR"


def test_only_recent_active_players_projected():
    hist = _history()
    # a retiree who last played in 2022 should be dropped
    extra = hist[hist["player_id"] == "p_wr"].copy()
    extra["player_id"] = "p_retired"
    extra["player"] = "Retired WR"
    extra = extra[extra["season"] == 2022]
    proj = projections.build_projections(pd.concat([hist, extra]), target_season=2025)
    assert "p_retired" not in set(proj["player_id"])


def test_provider_component_import_and_match():
    raw = pd.DataFrame({"Player": ["Star WR", "New Guy"], "POS": ["WR", "RB"],
                        "TEAM": ["FA", "FA"], "Rec": [110, 50], "RecYds": [1500, 400], "RecTD": [12, 2]})
    prov = providers.normalize_provider_frame(raw)
    assert "name_key" in prov.columns and prov["rec"].iloc[0] == 110
    baseline = pd.DataFrame({"player_id": ["W1"], "player": ["Star WR"], "position": ["WR"]})
    matched = providers.match_to_baseline(prov, baseline)
    assert matched.loc[matched["player"] == "Star WR", "player_id"].iloc[0] == "W1"
    assert pd.isna(matched.loc[matched["player"] == "New Guy", "player_id"].iloc[0])


def test_provider_points_only_sets_override():
    raw = pd.DataFrame({"Player": ["Kicker Guy"], "POS": ["K"], "FPTS": [150]})
    prov = providers.normalize_provider_frame(raw)
    assert prov[OVERRIDE_COLUMN].iloc[0] == 150


def test_accuracy_weights_favor_lower_error():
    w = accuracy.accuracy_weights({"good": 10.0, "bad": 40.0})
    assert w["good"] > w["bad"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_weighted_consensus_blends_sources():
    a = pd.DataFrame([{"player_id": "x", "player": "X", "position": "WR", "team": "FA",
                       "age": 26, "proj_games": 16, **{c: 0.0 for c in STAT_COLUMNS},
                       "rec_yds": 1000, OVERRIDE_COLUMN: np.nan}])
    b = pd.DataFrame([{"player_id": "x", "player": "X", "position": "WR", "team": "FA",
                       "age": 26, "proj_games": 16, **{c: 0.0 for c in STAT_COLUMNS},
                       "rec_yds": 1200, OVERRIDE_COLUMN: np.nan}])
    blend = accuracy.weighted_consensus({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert abs(blend.loc[blend["player_id"] == "x", "rec_yds"].iloc[0] - 1100) < 1e-6
