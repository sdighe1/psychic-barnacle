"""Ingest external projection sources (FantasyPros / PFF / ESPN / your own).

The baseline model is one source. Any provider export can be normalised to the
same stat-line schema and either **blended** into an accuracy-weighted consensus
(see :mod:`ffauction.accuracy`) or used to **override** the baseline for the
players it covers. Two import shapes are supported:

* **Component stats** -- columns mapped to the projection schema; fully
  format-flexible (re-scores for Standard/Half/PPR).
* **Points only** -- a single projected-points column, stored as a
  format-agnostic override for the matched players.

Robust CSV import is the reliable path; ``fetch_url_csv`` is a thin best-effort
helper for public CSV endpoints and never something the app depends on.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd

from .league import normalize_position
from .scoring import OVERRIDE_COLUMN, STAT_COLUMNS

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Common header spellings -> our component-stat schema.
DEFAULT_COLUMN_MAP = {
    "pass_yds": ["pass_yds", "passing_yards", "pass yds", "py"],
    "pass_td": ["pass_td", "pass_tds", "passing_tds", "ptd"],
    "pass_int": ["pass_int", "interceptions", "int", "ints"],
    "rush_yds": ["rush_yds", "rushing_yards", "ry"],
    "rush_td": ["rush_td", "rush_tds", "rushing_tds", "rtd"],
    "rec": ["rec", "receptions", "receptions_rec", "catches"],
    "rec_yds": ["rec_yds", "receiving_yards", "recy"],
    "rec_td": ["rec_td", "rec_tds", "receiving_tds", "rectd"],
    "fumbles_lost": ["fumbles_lost", "fl", "fum_lost", "fumbles"],
}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and common suffixes for fuzzy matching."""
    if name is None:
        return ""
    s = re.sub(r"[.'`]", "", str(name).lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(toks).strip()


def _find_col(columns, candidates) -> Optional[str]:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def normalize_provider_frame(
    raw: pd.DataFrame,
    name_col: Optional[str] = None,
    pos_col: Optional[str] = None,
    team_col: Optional[str] = None,
    points_col: Optional[str] = None,
    column_map: Optional[Dict[str, list]] = None,
    source: str = "provider",
) -> pd.DataFrame:
    """Normalise a provider export into the project's projection schema.

    Auto-detects common column names; pass explicit ``*_col`` overrides when a
    file uses unusual headers.
    """
    column_map = column_map or DEFAULT_COLUMN_MAP
    cols = list(raw.columns)
    name_col = name_col or _find_col(cols, ["player", "name", "player_name", "playername"])
    pos_col = pos_col or _find_col(cols, ["pos", "position"])
    team_col = team_col or _find_col(cols, ["team", "tm", "nfl_team"])
    points_col = points_col or _find_col(cols, ["fpts", "points", "proj_points", "fantasy_points"])
    if name_col is None:
        raise ValueError("could not find a player-name column in the provider file")

    out = pd.DataFrame()
    out["player"] = raw[name_col].astype(str)
    out["name_key"] = out["player"].map(normalize_name)
    out["position"] = (raw[pos_col].map(normalize_position) if pos_col else "")
    out["team"] = (raw[team_col].astype(str) if team_col else "")

    have_components = False
    for schema_col, cands in column_map.items():
        src = _find_col(cols, cands)
        if src is not None:
            out[schema_col] = pd.to_numeric(raw[src], errors="coerce").fillna(0.0)
            have_components = True
    for c in STAT_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0

    out[OVERRIDE_COLUMN] = float("nan")
    if points_col is not None and not have_components:
        # points-only import: store as a format-agnostic override
        out[OVERRIDE_COLUMN] = pd.to_numeric(raw[points_col], errors="coerce")
    out["source"] = source
    return out


def match_to_baseline(provider: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Attach baseline ``player_id`` to provider rows by (name_key, position).

    Unmatched provider rows keep an empty ``player_id`` (they are new players you
    can still add to the board).
    """
    base = baseline.copy()
    base["name_key"] = base["player"].map(normalize_name)
    key = base.drop_duplicates(["name_key", "position"])[["name_key", "position", "player_id"]]
    merged = provider.merge(key, on=["name_key", "position"], how="left")
    return merged


def fetch_url_csv(url: str, timeout: int = 20) -> Optional[pd.DataFrame]:
    """Best-effort fetch of a public CSV endpoint. Returns None on any failure
    (network, auth, format) -- callers must tolerate None."""
    try:
        import io

        import requests

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "ffauction/0.1"})
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception:
        return None
