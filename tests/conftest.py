"""Shared pytest fixtures: a small synthetic, network-free player pool."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ffauction.league import LeagueSettings  # noqa: E402
from ffauction.scoring import OVERRIDE_COLUMN, STAT_COLUMNS  # noqa: E402

# Players per position and the top projected points at each (descending).
_POOL_SPEC = {
    "QB": (8, 380), "RB": (16, 320), "WR": (18, 300), "TE": (8, 210),
    "K": (5, 150), "DST": (5, 140),
}


def _stat_line(pos: str, pts: float) -> dict:
    """Realise a target point total via a single stat (no receptions), so the
    projection is identical across scoring formats -- convenient for exact math
    in valuation/draft tests."""
    line = {c: 0.0 for c in STAT_COLUMNS}
    line[OVERRIDE_COLUMN] = float("nan")
    if pos == "QB":
        line["pass_yds"] = pts / 0.04
    elif pos in ("RB",):
        line["rush_yds"] = pts / 0.1
    elif pos in ("WR", "TE"):
        line["rec_yds"] = pts / 0.1
    else:  # K / DST
        line[OVERRIDE_COLUMN] = pts
    return line


@pytest.fixture
def pool() -> pd.DataFrame:
    rows = []
    for pos, (n, top) in _POOL_SPEC.items():
        for i in range(n):
            pts = top - i * (top * 0.05)          # descending, still positive
            rows.append({
                "player_id": f"{pos}{i:02d}", "player": f"{pos} Player {i}",
                "position": pos, "team": "FA", "age": 26.0, "proj_games": 16.0,
                "source": "synthetic", **_stat_line(pos, pts),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def league() -> LeagueSettings:
    # 4 teams keeps the money pool small enough to fit the synthetic pool while
    # still exercising FLEX, caps and replacement levels.
    return LeagueSettings(teams=4, budget=100)
