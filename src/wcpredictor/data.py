"""Load, cache and clean the historical international-match dataset.

Source: Mart Jürisoo's *International football results from 1872 to present*
(https://github.com/martj42/international_results), an open, community-maintained
CSV of essentially every men's international. We use three files:

- ``results.csv``    — one row per match (the core training data)
- ``shootouts.csv``  — penalty-shootout winners (used to resolve knockout draws)
- ``goalscorers.csv``— goal events (not required by the model, downloaded lazily)

Downloads are cached under ``data/`` so the pipeline runs offline after the
first fetch.
"""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd
import requests

from .paths import DATA_DIR

RAW_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results": "results.csv",
    "shootouts": "shootouts.csv",
    "goalscorers": "goalscorers.csv",
}

# --------------------------------------------------------------------------- #
# Team-name normalisation
# --------------------------------------------------------------------------- #
# Merge historical predecessor states into their modern FIFA lineage so a team's
# rating/history is continuous. This mirrors the conventions used by
# eloratings.net. Kept deliberately small — over-merging distorts strength.
TEAM_ALIASES = {
    "West Germany": "Germany",
    "East Germany": "East Germany",  # kept distinct (dissolved, not a lineage of Germany)
    "Soviet Union": "Russia",
    "CIS": "Russia",
    "Czechoslovakia": "Czechia",
    "Czech Republic": "Czechia",
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Zaïre": "DR Congo",
    "Zaire": "DR Congo",
    "Congo-Kinshasa": "DR Congo",
    "Congo DR": "DR Congo",
    "Republic of Ireland": "Ireland",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
}


def normalize_team(name: str) -> str:
    """Map a raw team name to its canonical form."""
    if not isinstance(name, str):
        return name
    return TEAM_ALIASES.get(name.strip(), name.strip())


# --------------------------------------------------------------------------- #
# Tournament importance  (drives Elo K-factor and is a model feature)
# --------------------------------------------------------------------------- #
# Base K weights follow the World Football Elo Ratings scheme.
_IMPORTANCE_RULES = [
    # (substring matcher, K weight, ordinal importance 1..5)
    ("FIFA World Cup", 60, 5),          # World Cup finals
    ("Copa América", 50, 4),
    ("UEFA Euro", 50, 4),
    ("African Cup of Nations", 50, 4),
    ("AFC Asian Cup", 50, 4),
    ("Gold Cup", 50, 4),
    ("Confederations Cup", 50, 4),
    ("UEFA Nations League", 45, 4),
    ("CONCACAF Nations League", 40, 3),
    ("qualification", 40, 3),           # any qualifier
    ("Friendly", 20, 1),
]


def tournament_weights(tournament: str) -> tuple[float, int]:
    """Return ``(k_weight, importance)`` for a tournament string.

    ``qualification`` matches (e.g. "FIFA World Cup qualification") are caught by
    the generic rule and never by the finals rule, because that rule is checked
    first only for exact finals names — see the ordering guard below.
    """
    t = tournament or ""
    # Qualifiers first so "FIFA World Cup qualification" doesn't score as finals.
    if "qualification" in t:
        return 40.0, 3
    for key, k, imp in _IMPORTANCE_RULES:
        if key in t:
            return float(k), imp
    return 30.0, 2  # other competitive tournaments


# --------------------------------------------------------------------------- #
# Download / cache
# --------------------------------------------------------------------------- #
def _fetch(name: str, refresh: bool = False) -> pd.DataFrame:
    fname = FILES[name]
    cache = DATA_DIR / fname
    if refresh or not cache.exists():
        url = f"{RAW_BASE}/{fname}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    return pd.read_csv(cache)


@lru_cache(maxsize=4)
def load_matches(refresh: bool = False) -> pd.DataFrame:
    """Return the cleaned, chronologically-sorted match table.

    Columns: ``date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral, outcome, k_weight, importance``.
    """
    df = _fetch("results", refresh=refresh)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # neutral may be bool or "TRUE"/"FALSE"
    df["neutral"] = df["neutral"].map(_to_bool)

    df["outcome"] = df.apply(
        lambda r: "H" if r.home_score > r.away_score
        else ("A" if r.home_score < r.away_score else "D"),
        axis=1,
    )
    weights = df["tournament"].map(tournament_weights)
    df["k_weight"] = [w[0] for w in weights]
    df["importance"] = [w[1] for w in weights]

    # Drop self-matches / obviously bad rows, sort chronologically, stable index.
    df = df[df["home_team"] != df["away_team"]]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().upper() in {"TRUE", "1", "YES", "T"}


@lru_cache(maxsize=2)
def load_shootouts(refresh: bool = False) -> pd.DataFrame:
    """Penalty-shootout results, team names normalised."""
    df = _fetch("shootouts", refresh=refresh)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("home_team", "away_team", "winner"):
        if col in df.columns:
            df[col] = df[col].map(normalize_team)
    return df


def team_list(df: pd.DataFrame | None = None, min_matches: int = 1) -> list[str]:
    """Sorted list of teams appearing at least ``min_matches`` times."""
    if df is None:
        df = load_matches()
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    return sorted(counts[counts >= min_matches].index.tolist())


def wc2026_matches(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows for the 2026 FIFA World Cup finals already present in the data."""
    if df is None:
        df = load_matches()
    mask = (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)
    return df[mask].copy()
