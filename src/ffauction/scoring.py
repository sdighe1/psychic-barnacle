"""ESPN fantasy scoring.

Fantasy points are computed from a projected **stat line** (not stored as a
single number), so the same projections re-price instantly when you switch
between ESPN **Standard** (0 PPR), **Half-PPR** (0.5) and **Full-PPR** (1.0).

The component-stat column names below are the schema used everywhere in the
project (``outputs/projections.csv`` and the synthetic test fixtures).

Kickers and team defenses do not have per-player offensive component stats, so
their projection carries a format-independent ``proj_points_override`` that is
simply added on top (their component stats are all zero).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

# --- projection schema -------------------------------------------------------
# Component stat columns (everything defaults to 0 when absent).
STAT_COLUMNS = [
    "pass_yds", "pass_td", "pass_int", "pass_2pt",
    "rush_yds", "rush_td", "rush_2pt",
    "rec", "rec_yds", "rec_td", "rec_2pt",
    "fumbles_lost", "st_td",
]
OVERRIDE_COLUMN = "proj_points_override"  # K / D-ST direct points (format-agnostic)

# --- scoring formats ---------------------------------------------------------
FORMATS = {"standard": 0.0, "half_ppr": 0.5, "ppr": 1.0}
FORMAT_LABELS = {"standard": "Standard (non-PPR)", "half_ppr": "Half-PPR", "ppr": "Full PPR"}


@dataclass(frozen=True)
class ScoringRules:
    """ESPN default scoring values (per unit). Only ``pts_per_reception`` varies
    between the three supported formats."""
    pts_per_pass_yd: float = 0.04         # 1 pt / 25 yds
    pts_per_pass_td: float = 4.0
    pts_per_interception: float = -2.0
    pts_per_pass_2pt: float = 2.0
    pts_per_rush_yd: float = 0.1          # 1 pt / 10 yds
    pts_per_rush_td: float = 6.0
    pts_per_rush_2pt: float = 2.0
    pts_per_rec_yd: float = 0.1           # 1 pt / 10 yds
    pts_per_rec_td: float = 6.0
    pts_per_rec_2pt: float = 2.0
    pts_per_reception: float = 0.5        # 0 / 0.5 / 1 depending on format
    pts_per_fumble_lost: float = -2.0
    pts_per_st_td: float = 6.0            # kick / punt return TDs


def rules_for_format(fmt: str) -> ScoringRules:
    """Return the ESPN scoring rules for ``standard`` / ``half_ppr`` / ``ppr``."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown scoring format {fmt!r}; expected one of {list(FORMATS)}")
    return ScoringRules(pts_per_reception=FORMATS[fmt])


def _g(row: Mapping, key: str) -> float:
    v = row.get(key, 0.0) if hasattr(row, "get") else getattr(row, key, 0.0)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else v  # treat NaN as 0


def points_from_stats(row: Mapping, rules: ScoringRules) -> float:
    """Fantasy points for one projected stat line (a dict / pandas row)."""
    r = rules
    pts = (
        _g(row, "pass_yds") * r.pts_per_pass_yd
        + _g(row, "pass_td") * r.pts_per_pass_td
        + _g(row, "pass_int") * r.pts_per_interception
        + _g(row, "pass_2pt") * r.pts_per_pass_2pt
        + _g(row, "rush_yds") * r.pts_per_rush_yd
        + _g(row, "rush_td") * r.pts_per_rush_td
        + _g(row, "rush_2pt") * r.pts_per_rush_2pt
        + _g(row, "rec_yds") * r.pts_per_rec_yd
        + _g(row, "rec_td") * r.pts_per_rec_td
        + _g(row, "rec_2pt") * r.pts_per_rec_2pt
        + _g(row, "rec") * r.pts_per_reception
        + _g(row, "fumbles_lost") * r.pts_per_fumble_lost
        + _g(row, "st_td") * r.pts_per_st_td
    )
    return pts + _g(row, OVERRIDE_COLUMN)


def project_points(df: pd.DataFrame, fmt: str) -> pd.Series:
    """Vectorised fantasy points for a projections DataFrame under ``fmt``."""
    r = rules_for_format(fmt)

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    pts = (
        col("pass_yds") * r.pts_per_pass_yd
        + col("pass_td") * r.pts_per_pass_td
        + col("pass_int") * r.pts_per_interception
        + col("pass_2pt") * r.pts_per_pass_2pt
        + col("rush_yds") * r.pts_per_rush_yd
        + col("rush_td") * r.pts_per_rush_td
        + col("rush_2pt") * r.pts_per_rush_2pt
        + col("rec_yds") * r.pts_per_rec_yd
        + col("rec_td") * r.pts_per_rec_td
        + col("rec_2pt") * r.pts_per_rec_2pt
        + col("rec") * r.pts_per_reception
        + col("fumbles_lost") * r.pts_per_fumble_lost
        + col("st_td") * r.pts_per_st_td
        + col(OVERRIDE_COLUMN)
    )
    return pts.round(2)
