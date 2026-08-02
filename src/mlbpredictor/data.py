"""High-level data loading: game logs and per-player rate aggregates.

Downloads from the Chadwick Bureau Retrosheet mirror (GitHub raw, reachable
offline) and caches parsed results under ``data/mlb/``:

* :func:`load_game_logs` — one row per game (scores, starters, lineups, park).
  Small; the raw ``GL{year}.TXT`` is cached.
* :func:`load_rate_aggregates` — season batting / pitching / bullpen PA-outcome
  counts, summed across all 30 team event files for the season. Heavier to build,
  so only the compact parquet aggregate is cached (raw event files are streamed,
  not kept).
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict

import pandas as pd

from .config import load_config
from .ids import bats_of, throws_of
from .net import cached_text, fetch_text
from .paths import CACHE_DIR
from .retrosheet import (PA_OUTCOMES, counts_to_frame, parse_event_text,
                         parse_gamelog_text)

# Bump when the parsed-aggregate schema changes (e.g. platoon splits added).
_AGG_VERSION = "pl1"


# --------------------------------------------------------------------------- #
# Game logs
# --------------------------------------------------------------------------- #
def _season_gamelog(season: int, refresh: bool = False) -> pd.DataFrame:
    cfg = load_config()["data"]
    url = f"{cfg['retro_base']}/{season}/GL{season}.TXT"
    text = cached_text(url, f"GL{season}.TXT", refresh=refresh)
    if not text:
        return pd.DataFrame()
    df = parse_gamelog_text(text)
    df["season"] = season
    return df


def load_game_logs(seasons: list[int] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Concatenated game logs for ``seasons`` (defaults to config ``elo_seasons``)."""
    seasons = seasons or load_config()["data"]["elo_seasons"]
    frames = [_season_gamelog(s, refresh=refresh) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No game logs could be loaded.")
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "home_team"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Team → league (drives event-file extension: NL park = .EVN, AL park = .EVA)
# --------------------------------------------------------------------------- #
def _team_leagues(season: int) -> dict[str, str]:
    cfg = load_config()["data"]
    text = cached_text(f"{cfg['retro_base']}/{season}/TEAM{season}", f"TEAM{season}.txt")
    out: dict[str, str] = {}
    if text:
        for rec in csv.reader(io.StringIO(text)):
            if len(rec) >= 2 and rec[0]:
                out[rec[0]] = rec[1]
    return out


def _merge_counts(dst: dict, src: dict) -> None:
    for key, c in src.items():
        d = dst[key]
        for k, v in c.items():
            d[k] += v


# --------------------------------------------------------------------------- #
# Rate aggregates
# --------------------------------------------------------------------------- #
def _build_season_aggregates(season: int) -> dict[str, pd.DataFrame]:
    """Parse every team event file for ``season`` → summed batting/pitching/bullpen."""
    cfg = load_config()["data"]
    leagues = _team_leagues(season)
    if not leagues:
        raise RuntimeError(f"Could not read TEAM{season} (team list) for event files.")

    bat: dict = defaultdict(lambda: defaultdict(int))
    pit: dict = defaultdict(lambda: defaultdict(int))
    bull: dict = defaultdict(lambda: defaultdict(int))

    for team, league in sorted(leagues.items()):
        ext = "EVN" if league.upper() == "N" else "EVA"
        url = f"{cfg['retro_base']}/{season}/{season}{team}.{ext}"
        text = fetch_text(url)                      # streamed, not cached raw
        if text is None and ext == "EVN":
            text = fetch_text(f"{cfg['retro_base']}/{season}/{season}{team}.EVA")
        if not text:
            continue
        b, p, bl = parse_event_text(text, bats_fn=bats_of, throws_fn=throws_of)
        _merge_counts(bat, b)
        _merge_counts(pit, p)
        _merge_counts(bull, bl)

    return {
        "batting": counts_to_frame(bat, "retro_id", season),
        "pitching": counts_to_frame(pit, "retro_id", season),
        "bullpen": counts_to_frame(bull, "team", season),
    }


def _season_aggregate(kind: str, season: int, refresh: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / f"{kind}_{_AGG_VERSION}_{season}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    aggs = _build_season_aggregates(season)
    for k, frame in aggs.items():
        frame.to_parquet(CACHE_DIR / f"{k}_{_AGG_VERSION}_{season}.parquet", index=False)
    return aggs[kind]


def load_rate_aggregates(seasons: list[int] | None = None,
                         refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Return concatenated ``{'batting','pitching','bullpen'}`` frames over ``seasons``.

    Each is one row per (player-or-team, season) with PA/BF and the PA-outcome
    counts in :data:`PA_OUTCOMES`.
    """
    seasons = seasons or load_config()["data"]["rate_seasons"]
    out: dict[str, list[pd.DataFrame]] = {"batting": [], "pitching": [], "bullpen": []}
    for s in seasons:
        for kind in out:
            out[kind].append(_season_aggregate(kind, s, refresh=refresh))
    return {kind: pd.concat(frames, ignore_index=True) for kind, frames in out.items()}


def league_rates(agg: pd.DataFrame) -> pd.Series:
    """League-wide PA-outcome *rates* (per PA) from a counts frame."""
    tot = agg[PA_OUTCOMES].sum()
    pa = float(agg["PA"].sum())
    return (tot / pa) if pa > 0 else tot * 0.0
