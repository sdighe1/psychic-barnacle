"""Pure Retrosheet parsing (no network) — event files and game logs.

Two inputs from the Chadwick Bureau mirror:

* **Event files** ``{year}{TEAM}.EV{N,A}`` — one record per line: ``id`` (new game),
  ``info`` (metadata), ``start``/``sub`` (lineup, pos 1 = pitcher), ``play`` (one
  plate appearance or a baserunning event), ``data`` (earned runs). We classify each
  ``play`` into a completed-PA outcome and attribute it to the batter and the
  pitcher then on the mound, splitting relief work into a team "bullpen" bucket.

* **Game logs** ``GL{year}.TXT`` — the canonical 161-field, one-row-per-game file
  with final scores, starting pitchers and the two starting batting orders. Used
  for the game-level backtest, Elo, park factors and default lineups.

Keeping this module free of I/O makes the fiddly event grammar easy to unit-test.
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict

import pandas as pd

# Completed-PA outcome categories (the simulator's alphabet).
PA_OUTCOMES = ["1B", "2B", "3B", "HR", "BB", "HBP", "SO", "OUT"]

# Leading tokens that are baserunning / administrative events, NOT a PA result.
# (The same batter's plate appearance continues on a later ``play`` record.)
_NON_PA_PREFIXES = ("SB", "CS", "POCS", "PO", "DI", "PB", "WP", "BK", "OA", "FLE", "NP")


def classify_batter_event(raw: str) -> str | None:
    """Map a Retrosheet play-event string to a PA outcome, or ``None``.

    Returns one of :data:`PA_OUTCOMES`, or ``None`` when the record is not a
    completed plate appearance (stolen base, wild pitch, no-play, ...).
    """
    if not raw:
        return None
    ev = raw.split(".", 1)[0].strip().upper()   # drop baserunner-advance suffix
    if not ev:
        return None
    for p in _NON_PA_PREFIXES:                   # e.g. WP before W, SB before S, DI before D
        if ev.startswith(p):
            return None
    c0 = ev[0]
    if c0 == "K":
        return "SO"
    if c0 == "W":                                # walk (WP already excluded)
        return "BB"
    if c0 == "I":                                # I / IW intentional walk
        return "BB"
    if ev.startswith("HP"):
        return "HBP"
    if c0 == "H":                                # HR / H / H9 (HP already handled)
        return "HR"
    if c0 == "S":                                # single (SB excluded above)
        return "1B"
    if c0 == "D":                                # double (DI excluded above)
        return "2B"
    if c0 == "T":                                # triple
        return "3B"
    if c0 == "E" or ev.startswith("FC") or ev.startswith("FLE"):
        return "OUT"                             # reached on error / fielder's choice
    if c0.isdigit():                             # fielded out (8, 43, 6(1)3, ...)
        return "OUT"
    return None                                  # catcher interference etc. — negligible


def _to_int(v, default: int | None = None) -> int | None:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# Event-file parsing → per-player rate aggregates
# --------------------------------------------------------------------------- #
def parse_event_text(text: str) -> tuple[dict, dict, dict]:
    """Aggregate one event file into batting / pitching / bullpen outcome counts.

    Returns three dicts keyed by player id (bullpen keyed by team id), each mapping
    outcome → count and carrying a ``"PA"`` total::

        bat[batter_id][outcome], pit[pitcher_id][outcome], bullpen[team_id][outcome]
    """
    bat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pit: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bull: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    home = vis = None
    cur_p: list[str | None] = [None, None]      # current pitcher by side (0=vis, 1=home)
    start_p: list[str | None] = [None, None]    # each side's starting pitcher

    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        tag = rec[0]
        if tag == "id":
            home = vis = None
            cur_p = [None, None]
            start_p = [None, None]
        elif tag == "info" and len(rec) >= 3:
            if rec[1] == "hometeam":
                home = rec[2]
            elif rec[1] == "visteam":
                vis = rec[2]
        elif tag in ("start", "sub") and len(rec) >= 6:
            side = _to_int(rec[3])
            if side in (0, 1) and rec[5].strip() == "1":   # a pitcher
                cur_p[side] = rec[1]
                if start_p[side] is None:
                    start_p[side] = rec[1]
        elif tag == "play" and len(rec) >= 7:
            bside = _to_int(rec[2])
            if bside not in (0, 1):
                continue
            outcome = classify_batter_event(rec[6])
            if outcome is None:
                continue
            batter = rec[3]
            bat[batter][outcome] += 1
            bat[batter]["PA"] += 1
            pside = 1 - bside
            pid = cur_p[pside]
            if pid is not None:
                pit[pid][outcome] += 1
                pit[pid]["PA"] += 1
                if start_p[pside] is not None and pid != start_p[pside]:
                    pteam = home if pside == 1 else vis
                    if pteam:
                        bull[pteam][outcome] += 1
                        bull[pteam]["PA"] += 1
    return bat, pit, bull


def counts_to_frame(counts: dict, key_name: str, season: int) -> pd.DataFrame:
    """Turn a ``{id: {outcome: n}}`` dict into a tidy DataFrame (one row per id)."""
    rows = []
    for key, c in counts.items():
        row = {key_name: key, "season": season, "PA": int(c.get("PA", 0))}
        for o in PA_OUTCOMES:
            row[o] = int(c.get(o, 0))
        rows.append(row)
    cols = [key_name, "season", "PA", *PA_OUTCOMES]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Game-log parsing → one row per game
# --------------------------------------------------------------------------- #
# 0-indexed field positions in the 161-field Retrosheet game log.
_GL = {
    "date": 0, "number": 1, "visteam": 3, "hometeam": 6,
    "vis_score": 9, "home_score": 10, "outs": 11, "daynight": 12, "park": 16,
    "vis_sp": 101, "home_sp": 103,
}
_GL_VIS_BAT0 = 105    # visitor batting-order slot 1 id; slot i id at 105 + i*3
_GL_HOME_BAT0 = 132   # home batting-order slot 1 id; slot i id at 132 + i*3


def parse_gamelog_text(text: str) -> pd.DataFrame:
    """Parse a ``GL{year}.TXT`` file into a game-level DataFrame.

    Columns: ``date`` (Timestamp), ``game_no``, ``home_team``, ``away_team``,
    ``home_score``, ``away_score``, ``park``, ``daynight``, ``home_sp``,
    ``away_sp``, ``home_lineup`` (list of 9 ids), ``away_lineup`` (list of 9 ids).
    """
    rows = []
    for rec in csv.reader(io.StringIO(text)):
        if len(rec) < 160:
            continue
        hs = _to_int(rec[_GL["home_score"]])
        vs = _to_int(rec[_GL["vis_score"]])
        if hs is None or vs is None:
            continue
        away_lineup = [rec[_GL_VIS_BAT0 + i * 3] for i in range(9)]
        home_lineup = [rec[_GL_HOME_BAT0 + i * 3] for i in range(9)]
        rows.append({
            "date": rec[_GL["date"]],
            "game_no": rec[_GL["number"]],
            "home_team": rec[_GL["hometeam"]],
            "away_team": rec[_GL["visteam"]],
            "home_score": hs,
            "away_score": vs,
            "park": rec[_GL["park"]],
            "daynight": rec[_GL["daynight"]],
            "home_sp": rec[_GL["home_sp"]],
            "away_sp": rec[_GL["vis_sp"]],
            "home_lineup": home_lineup,
            "away_lineup": away_lineup,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values(
            ["date", "home_team"], kind="mergesort").reset_index(drop=True)
    return df


def parse_roster_text(text: str) -> pd.DataFrame:
    """Parse a ``TEAMyyyy.ROS`` roster file (id,last,first,bats,throws,team,pos)."""
    rows = []
    for rec in csv.reader(io.StringIO(text)):
        if len(rec) >= 6:
            rows.append({"retro_id": rec[0], "bats": rec[3], "throws": rec[4]})
    return pd.DataFrame(rows, columns=["retro_id", "bats", "throws"])
