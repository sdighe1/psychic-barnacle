"""FantasyPros ECR parsing + rank-anchoring (offline, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ffauction import fantasypros, projections, scoring
from ffauction.scoring import OVERRIDE_COLUMN, STAT_COLUMNS


def _raw_ecr() -> pd.DataFrame:
    """Mimic db_fpecr.parquet with two scrape dates and several ecr types."""
    rows = [
        # latest scrape, redraft positional (rp) -- what we want
        ("Star WR", "WR", "CIN", 1.1, "rp", "2026-08-28"),
        ("Rook WR", "WR", "SEA", 2.4, "rp", "2026-08-28"),
        ("Star RB", "RB", "DET", 1.2, "rp", "2026-08-28"),
        ("Star QB", "QB", "BUF", 1.0, "rp", "2026-08-28"),
        ("Top K", "PK", "DAL", 1.0, "rp", "2026-08-28"),        # PK -> K
        ("Top DEF", "DST", "HOU", 1.0, "rp", "2026-08-28"),
        ("Idp Guy", "LB", "MIA", 1.0, "rp", "2026-08-28"),      # dropped (not fantasy pos)
        # noise that must be filtered out
        ("Star WR", "WR", "CIN", 5.0, "do", "2026-08-28"),      # wrong ecr_type
        ("Old WR", "WR", "CIN", 1.0, "rp", "2020-01-01"),       # old scrape
    ]
    return pd.DataFrame(rows, columns=["player", "pos", "team", "ecr", "ecr_type", "scrape_date"])


def test_parse_ecr_keeps_latest_redraft_positional():
    ranks, date = fantasypros.parse_ecr(_raw_ecr())
    assert date == "2026-08-28"
    assert set(ranks["position"]) == {"WR", "RB", "QB", "K", "DST"}   # PK->K, LB dropped
    assert "Old WR" not in set(ranks["player"])                       # old scrape gone
    # dedupe kept the redraft (rp) row, not the dynasty (do) one
    assert (ranks[ranks["player"] == "Star WR"]["ecr"] == 1.1).all()


def _model() -> pd.DataFrame:
    rows = []
    specs = {"WR": [1400, 1100, 900], "RB": [1300, 1000], "QB": [4600, 4000], "TE": [900, 700]}
    for pos, vals in specs.items():
        for i, y in enumerate(vals):
            line = {c: 0.0 for c in STAT_COLUMNS}
            if pos == "QB":
                line["pass_yds"] = y; line["pass_td"] = 30 - i * 4
            elif pos == "RB":
                line["rush_yds"] = y; line["rush_td"] = 10 - i * 2
            else:
                line["rec_yds"] = y; line["rec"] = 90 - i * 20; line["rec_td"] = 9 - i * 2
            rows.append({"player_id": f"m_{pos}{i}", "player": f"Model {pos}{i}",
                         "position": pos, "team": "FA", "age": 26.0, "proj_games": 16.0,
                         "source": "history", OVERRIDE_COLUMN: np.nan, **line})
    return pd.DataFrame(rows)


def _fp_ranks() -> pd.DataFrame:
    return pd.DataFrame([
        ("Real WR1", "WR", "CIN", 1.1), ("Rookie WR", "WR", "SEA", 2.2), ("Real WR3", "WR", "LAR", 3.5),
        ("Real RB1", "RB", "DET", 1.0), ("Real RB2", "RB", "ATL", 2.0),
        ("Real QB1", "QB", "BUF", 1.0),
        ("Real TE1", "TE", "LV", 1.0),
        ("Kicker A", "K", "DAL", 1.0), ("Kicker B", "K", "HOU", 2.0),
        ("Defense A", "DST", "HOU", 1.0),
    ], columns=["player", "position", "team", "ecr"])


def _kdst() -> pd.DataFrame:
    return pd.DataFrame({"player": ["KA", "KB", "DA", "DB"], "position": ["K", "K", "DST", "DST"],
                        "team": ["DAL", "HOU", "HOU", "DEN"], "proj_points": [150, 140, 145, 138]})


def test_anchor_orders_by_ecr_with_monotonic_points():
    out = projections.anchor_to_rankings(_model(), _fp_ranks(), _kdst())
    wr = out[out["position"] == "WR"].copy()
    wr["pts"] = scoring.project_points(wr, "half_ppr")
    wr = wr.sort_values("ecr")
    # better ECR (lower) -> more projected points
    assert list(wr["pts"]) == sorted(wr["pts"], reverse=True)
    assert "Rookie WR" in set(out["player"])                     # rookie not in model, still projected
    assert out.loc[out["player"] == "Real WR1", "ecr"].iloc[0] == 1.1


def test_anchor_kdst_uses_baseline_magnitudes_in_fp_order():
    out = projections.anchor_to_rankings(_model(), _fp_ranks(), _kdst())
    k = out[out["position"] == "K"].sort_values("ecr")
    # top FantasyPros kicker gets the top curated magnitude
    assert k.iloc[0][OVERRIDE_COLUMN] == 150
    assert (k[OVERRIDE_COLUMN].to_numpy()[:2] == [150, 140]).all()
    assert (out[out["position"] == "DST"][OVERRIDE_COLUMN] > 0).all()


def test_anchor_preserves_format_flexibility():
    out = projections.anchor_to_rankings(_model(), _fp_ranks(), _kdst())
    wr1 = out[out["player"] == "Real WR1"].iloc[0]
    std = scoring.points_from_stats(wr1, scoring.rules_for_format("standard"))
    ppr = scoring.points_from_stats(wr1, scoring.rules_for_format("ppr"))
    assert ppr > std                                            # receptions add points in PPR


def test_anchor_player_ids_are_stable_slugs():
    out = projections.anchor_to_rankings(_model(), _fp_ranks(), _kdst())
    assert out[out["player"] == "Real WR1"]["player_id"].iloc[0] == "WR:real wr1"
