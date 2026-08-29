"""Historical NFL data from nflverse (via ``nfl_data_py``).

We pull season-level offensive stats and the roster table (for name, position,
team and age), then normalise them into the project's stat-line schema. This is
open data and always available, so the projection baseline can be rebuilt from
scratch with ``scripts/build_projections.py``.

Kickers and team defenses are not in the offensive stats feed; they come from a
small curated baseline instead (see ``data/kdst_baseline.csv``).
"""
from __future__ import annotations

import contextlib
import io
from typing import List

import pandas as pd

from .league import normalize_position

# nflverse seasonal-stat column -> our schema (component stats).
_SEASONAL_MAP = {
    "passing_yards": "pass_yds",
    "passing_tds": "pass_td",
    "interceptions": "pass_int",
    "passing_2pt_conversions": "pass_2pt",
    "rushing_yards": "rush_yds",
    "rushing_tds": "rush_td",
    "rushing_2pt_conversions": "rush_2pt",
    "receptions": "rec",
    "receiving_yards": "rec_yds",
    "receiving_tds": "rec_td",
    "receiving_2pt_conversions": "rec_2pt",
    "special_teams_tds": "st_td",
}
_OPPORTUNITY = ["carries", "targets"]      # kept for context, not scored
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def _quiet_import():
    """Import nfl_data_py while muting its noisy stdout/progress bars."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import nfl_data_py as nfl  # noqa: WPS433 (deliberate lazy import)
    return nfl


def latest_available_season(newest: int = 2025, floor: int = 2015) -> int:
    """Probe nflverse for the most recent season with seasonal data."""
    nfl = _quiet_import()
    for yr in range(newest, floor - 1, -1):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                nfl.import_seasonal_data([yr], s_type="REG")
            return yr
        except Exception:
            continue
    raise RuntimeError("no seasonal data available from nflverse")


def load_player_seasons(years: List[int]) -> pd.DataFrame:
    """Return one normalised row per (player, season) for the given years.

    Columns: ``player_id, season, player, position, team, age, games`` plus the
    stat-line component columns and opportunity (``carries``, ``targets``).
    Years that are missing upstream are skipped.
    """
    nfl = _quiet_import()
    frames = []
    for yr in years:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                s = nfl.import_seasonal_data([yr], s_type="REG")
        except Exception:
            continue
        s = s.rename(columns=_SEASONAL_MAP)
        # fumbles lost across sack/rush/receiving
        for c in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"):
            if c not in s.columns:
                s[c] = 0.0
        s["fumbles_lost"] = (
            s["sack_fumbles_lost"] + s["rushing_fumbles_lost"] + s["receiving_fumbles_lost"]
        )
        s["season"] = yr
        frames.append(s)

    if not frames:
        raise RuntimeError(f"no seasonal data for years {years}")

    stats = pd.concat(frames, ignore_index=True)
    keep = ["player_id", "season", "games"] + list(_SEASONAL_MAP.values()) + ["fumbles_lost"] + _OPPORTUNITY
    keep = [c for c in keep if c in stats.columns]
    stats = stats[keep].copy()

    meta = _load_roster_meta(nfl, years)
    df = stats.merge(meta, on=["player_id", "season"], how="left")
    df["position"] = df["position"].map(normalize_position)
    df = df[df["position"].isin(SKILL_POSITIONS)].reset_index(drop=True)
    return df


def _load_roster_meta(nfl, years: List[int]) -> pd.DataFrame:
    """player_id + season -> player name, position, team, age (one row each)."""
    frames = []
    for yr in years:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                r = nfl.import_seasonal_rosters([yr])
        except Exception:
            continue
        cols = {c: c for c in ("player_id", "player_name", "position", "team", "age") if c in r.columns}
        r = r[list(cols)].copy()
        r["season"] = yr
        # rosters can be weekly -> collapse to one row per player-season
        r = (
            r.sort_values("age")
            .groupby(["player_id", "season"], as_index=False)
            .agg({"player_name": "last", "position": "last", "team": "last", "age": "max"})
        )
        frames.append(r)
    meta = pd.concat(frames, ignore_index=True)
    return meta.rename(columns={"player_name": "player"})
