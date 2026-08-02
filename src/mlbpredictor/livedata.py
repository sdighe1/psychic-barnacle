"""Today's slate: live from the MLB Stats API, or a manual file fallback.

**Auto-fetch** hits ``statsapi.mlb.com`` for the day's schedule, probable pitchers
and (when posted) lineups. That host is blocked by some environment network
policies — if it is unreachable we transparently fall back to a **manual slate**
file (``config/slate.yaml``) that you fill in with the morning's matchups and
probable starters. To enable auto-fetch, allow ``statsapi.mlb.com`` in the
environment's network policy (see the README / config).

Everything is normalised to Retrosheet ids/team codes (the model's native keys) via
the Chadwick register bridge in :mod:`mlbpredictor.ids`.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date, timedelta

import requests
import yaml

from .config import load_config
from .ids import retro_for_mlbam, retro_for_name

# MLBAM team id -> Retrosheet team code (stable).
STATSAPI_TEAM_ID_TO_RETRO = {
    108: "ANA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHN", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KCA", 119: "LAN",
    120: "WAS", 121: "NYN", 133: "OAK", 134: "PIT", 135: "SDN", 136: "SEA",
    137: "SFN", 138: "SLN", 139: "TBA", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CHA", 146: "MIA", 147: "NYA", 158: "MIL",
}
RETRO_TO_STATSAPI_TEAM_ID = {v: k for k, v in STATSAPI_TEAM_ID_TO_RETRO.items()}

# Common abbreviations / names -> Retrosheet code (for manual slates).
ABBREV_TO_RETRO = {
    "LAA": "ANA", "ANA": "ANA", "ARI": "ARI", "AZ": "ARI", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CHN": "CHN", "CWS": "CHA", "CHA": "CHA", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCA", "KCA": "KCA", "KCR": "KCA",
    "LAD": "LAN", "LAN": "LAN", "WSH": "WAS", "WSN": "WAS", "WAS": "WAS", "NYM": "NYN",
    "NYN": "NYN", "OAK": "OAK", "ATH": "OAK", "PIT": "PIT", "SD": "SDN", "SDP": "SDN",
    "SDN": "SDN", "SEA": "SEA", "SF": "SFN", "SFG": "SFN", "SFN": "SFN", "STL": "SLN",
    "SLN": "SLN", "TB": "TBA", "TBR": "TBA", "TBA": "TBA", "TEX": "TEX", "TOR": "TOR",
    "MIN": "MIN", "PHI": "PHI", "ATL": "ATL", "CWS_": "CHA", "MIA": "MIA", "FLA": "MIA",
    "NYY": "NYA", "NYA": "NYA", "MIL": "MIL",
}

_RETRO_ID = re.compile(r"^[a-z]{4,5}[a-z]?\d{3}$")   # e.g. acunr001


@dataclass
class GameInput:
    home_team: str
    away_team: str
    home_sp: str
    away_sp: str
    home_lineup: list[str] = field(default_factory=list)
    away_lineup: list[str] = field(default_factory=list)
    park: str | None = None
    source: str = "manual"
    # True when a full 9-man lineup was actually provided (posted/entered), not defaulted.
    home_lineup_confirmed: bool = False
    away_lineup_confirmed: bool = False

    @property
    def lineups_confirmed(self) -> bool:
        return self.home_lineup_confirmed and self.away_lineup_confirmed


# --------------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------------- #
def resolve_team(token: str) -> str:
    """Map an abbreviation / retro code to a Retrosheet team code."""
    t = str(token).strip()
    if t in ABBREV_TO_RETRO:
        return ABBREV_TO_RETRO[t]
    return t.upper()


def resolve_player(token: str | None) -> str | None:
    """Map a token to a Retrosheet id: pass through retro ids, else look up by name."""
    if not token:
        return None
    t = str(token).strip()
    if _RETRO_ID.match(t):
        return t
    return retro_for_name(t)


# --------------------------------------------------------------------------- #
# Live: MLB Stats API
# --------------------------------------------------------------------------- #
def statsapi_reachable() -> bool:
    cfg = load_config()["live"]
    try:
        r = requests.get(f"{cfg['statsapi_base']}/sports", timeout=6)
        return r.status_code == 200
    except requests.RequestException:
        return False


def fetch_statsapi_slate(date: str) -> list[GameInput]:
    """Fetch the day's games (probable pitchers + posted lineups) from statsapi.

    ``date`` is ``YYYY-MM-DD``. Raises ``requests.RequestException`` if unreachable.
    """
    cfg = load_config()["live"]
    url = (f"{cfg['statsapi_base']}/schedule?sportId=1&date={date}"
           f"&hydrate={cfg['schedule_hydrate']}")
    resp = requests.get(url, timeout=cfg["timeout_seconds"])
    resp.raise_for_status()
    return parse_statsapi_schedule(resp.json())


def _retro_team(team_obj: dict) -> str | None:
    tid = team_obj.get("id")
    if tid in STATSAPI_TEAM_ID_TO_RETRO:
        return STATSAPI_TEAM_ID_TO_RETRO[tid]
    return ABBREV_TO_RETRO.get(str(team_obj.get("abbreviation", "")).upper())


def _lineup_retro(players: list) -> list[str]:
    out = []
    for p in players or []:
        rid = retro_for_mlbam(p.get("id")) or retro_for_name(p.get("fullName", ""))
        if rid:
            out.append(rid)
    return out


def parse_statsapi_schedule(payload: dict) -> list[GameInput]:
    """Parse a statsapi ``/schedule`` JSON payload into GameInputs (pure)."""
    games: list[GameInput] = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            ht, at = _retro_team(home.get("team", {})), _retro_team(away.get("team", {}))
            if not ht or not at:
                continue
            hsp = retro_for_mlbam((home.get("probablePitcher") or {}).get("id")) \
                or retro_for_name((home.get("probablePitcher") or {}).get("fullName", ""))
            asp = retro_for_mlbam((away.get("probablePitcher") or {}).get("id")) \
                or retro_for_name((away.get("probablePitcher") or {}).get("fullName", ""))
            lineups = g.get("lineups", {}) or {}
            hl = _lineup_retro(lineups.get("homePlayers"))
            al = _lineup_retro(lineups.get("awayPlayers"))
            games.append(GameInput(
                home_team=ht, away_team=at, home_sp=hsp, away_sp=asp,
                home_lineup=hl, away_lineup=al, park=None, source="statsapi",
                home_lineup_confirmed=len(hl) >= 9, away_lineup_confirmed=len(al) >= 9))
    return games


# --------------------------------------------------------------------------- #
# Live: bullpen availability (active roster + recent usage) for specific-reliever pens
# --------------------------------------------------------------------------- #
def _statsapi_get(path: str, params: dict) -> dict:
    cfg = load_config()["live"]
    r = requests.get(f"{cfg['statsapi_base']}{path}", params=params, timeout=cfg["timeout_seconds"])
    r.raise_for_status()
    return r.json()


def parse_roster_pitchers(payload: dict) -> list[str]:
    """Retrosheet ids of the pitchers on an active-roster payload (pure)."""
    out = []
    for e in payload.get("roster", []) or []:
        pos = e.get("position", {}) or {}
        if pos.get("abbreviation") == "P" or pos.get("type") == "Pitcher":
            rid = retro_for_mlbam((e.get("person") or {}).get("id"))
            if rid:
                out.append(rid)
    return out


def parse_boxscore_pitchers(payload: dict) -> list[tuple[str | None, str, bool]]:
    """``(retro_team, retro_id, started)`` for every pitcher in a boxscore (pure)."""
    out = []
    for side in ("home", "away"):
        t = payload.get("teams", {}).get(side, {}) or {}
        team = STATSAPI_TEAM_ID_TO_RETRO.get((t.get("team") or {}).get("id"))
        players = t.get("players", {}) or {}
        for pid in t.get("pitchers", []) or []:
            pdata = players.get(f"ID{pid}", {}) or {}
            gs = ((pdata.get("stats") or {}).get("pitching") or {}).get("gamesStarted")
            rid = retro_for_mlbam((pdata.get("person") or {}).get("id") or pid)
            if rid:
                out.append((team, rid, bool(gs)))
    return out


def fetch_pitcher_roles(season: int) -> dict[str, float]:
    """``{retro_id: gamesStarted/games}`` for the season (≈1 = starter, ≈0 = reliever)."""
    js = _statsapi_get("/stats", {"stats": "season", "group": "pitching", "season": season,
                                  "sportId": 1, "gameType": "R", "limit": 4000, "playerPool": "all"})
    roles: dict[str, float] = {}
    for s in (js.get("stats", [{}])[0].get("splits", []) or []):
        rid = retro_for_mlbam(s.get("player", {}).get("id"))
        st = s.get("stat", {}) or {}
        g = st.get("gamesPitched") or st.get("gamesPlayed") or 0
        gs = st.get("gamesStarted") or 0
        if rid and g:
            roles[rid] = float(gs) / float(g)
    return roles


def fetch_recent_appearances(date: str, lookback_days: int) -> dict[str, set[int]]:
    """``{retro_id: {days_ago,...}}`` for pitchers who appeared in the prior games."""
    d0 = _date.fromisoformat(date)
    start = (d0 - timedelta(days=int(lookback_days))).isoformat()
    end = (d0 - timedelta(days=1)).isoformat()
    js = _statsapi_get("/schedule", {"sportId": 1, "startDate": start, "endDate": end, "gameType": "R"})
    appearances: dict[str, set[int]] = defaultdict(set)
    for day in js.get("dates", []):
        try:
            days_ago = (d0 - _date.fromisoformat(day.get("date", ""))).days
        except ValueError:
            continue
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            try:
                box = _statsapi_get(f"/game/{g.get('gamePk')}/boxscore", {})
            except requests.RequestException:
                continue
            for _team, rid, _started in parse_boxscore_pitchers(box):
                appearances[rid].add(days_ago)
    return dict(appearances)


def fetch_team_relievers(team_code: str, date: str, roles: dict[str, float],
                         gs_frac: float) -> list[str]:
    """Active-roster relievers for a team (rostered pitchers minus season starters)."""
    tid = RETRO_TO_STATSAPI_TEAM_ID.get(team_code)
    if not tid:
        return []
    js = _statsapi_get(f"/teams/{tid}/roster", {"rosterType": "active", "date": date})
    return [r for r in parse_roster_pitchers(js) if roles.get(r, 0.0) < gs_frac]


def fetch_bullpen_usage(date: str, team_codes, cfg: dict | None = None) -> dict[str, dict]:
    """Per-team available relievers + recent-usage days, for ``team_bullpen_vector``.

    Returns ``{team_code: {"relievers": [ids], "appearances": {id: {days_ago}}}}``.
    Each statsapi call is guarded — a partial failure yields an empty entry for that
    team, so the caller cleanly falls back to the season-aggregate bullpen.
    """
    cfg = cfg or load_config().get("bullpen", {})
    season = int(str(date)[:4])
    try:
        roles = fetch_pitcher_roles(season)
    except requests.RequestException:
        roles = {}
    try:
        appearances = fetch_recent_appearances(date, int(cfg.get("lookback_days", 2)))
    except requests.RequestException:
        appearances = {}
    gs_frac = float(cfg.get("starter_gs_frac", 0.5))
    usage: dict[str, dict] = {}
    for tc in sorted(set(team_codes)):
        try:
            relievers = fetch_team_relievers(tc, date, roles, gs_frac)
        except requests.RequestException:
            relievers = []
        usage[tc] = {"relievers": relievers,
                     "appearances": {r: appearances.get(r, set()) for r in relievers}}
    return usage


# --------------------------------------------------------------------------- #
# Manual slate file
# --------------------------------------------------------------------------- #
def _resolve_starter(token, team_label: str) -> str:
    """Resolve a starter token, warning if a non-empty token fails to resolve."""
    rid = resolve_player(token)
    if token and not rid:
        print(f"  [slate] warning: could not resolve {team_label} starter "
              f"'{token}' — will fall back to the team's recent starter.")
    return rid or ""


def _resolve_lineup(tokens, team_label: str) -> list[str]:
    ids = [resolve_player(x) for x in (tokens or [])]
    if tokens and any(r is None for r in ids):
        bad = [t for t, r in zip(tokens, ids) if r is None]
        print(f"  [slate] warning: {team_label} lineup has unresolved names {bad}.")
    return [r for r in ids if r]


def load_slate_file(path) -> list[GameInput]:
    """Load a manual slate YAML: a list of games under key ``games``."""
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    out = []
    for g in doc.get("games", []):
        home, away = resolve_team(g["home"]), resolve_team(g["away"])
        hl = _resolve_lineup(g.get("home_lineup"), home)
        al = _resolve_lineup(g.get("away_lineup"), away)
        out.append(GameInput(
            home_team=home, away_team=away,
            home_sp=_resolve_starter(g.get("home_sp"), home),
            away_sp=_resolve_starter(g.get("away_sp"), away),
            home_lineup=hl, away_lineup=al, park=g.get("park"), source="manual",
            home_lineup_confirmed=len(hl) >= 9, away_lineup_confirmed=len(al) >= 9))
    return out


# --------------------------------------------------------------------------- #
# Fill gaps from the model's stored defaults
# --------------------------------------------------------------------------- #
def fill_defaults(game: GameInput, predictor) -> GameInput:
    """Fill missing lineups / starter / park from the model's most-recent defaults."""
    fb = predictor.fb
    if len(game.home_lineup) < 9:
        game.home_lineup = fb.default_lineups.get(game.home_team, game.home_lineup)
    if len(game.away_lineup) < 9:
        game.away_lineup = fb.default_lineups.get(game.away_team, game.away_lineup)
    if not game.home_sp:
        game.home_sp = fb.default_starter.get(game.home_team, "")
    if not game.away_sp:
        game.away_sp = fb.default_starter.get(game.away_team, "")
    if not game.park:
        game.park = fb.default_park.get(game.home_team)
    return game


def get_slate(date: str, slate_path=None, predictor=None, prefer_live: bool = True) -> list[GameInput]:
    """Return the day's slate: live statsapi if reachable, else the manual file.

    Missing lineups/starters/park are filled from the model's stored defaults.
    """
    games: list[GameInput] = []
    live_used = False
    if prefer_live and statsapi_reachable():
        live_used = True
        try:
            games = fetch_statsapi_slate(date)
        except requests.RequestException:
            live_used = False            # unreachable mid-request → allow file fallback
    # Only fall back to the manual file when the live feed was NOT used successfully.
    # (If statsapi was reached and simply had no games — an off day — return empty
    # rather than predicting stale games from the example slate.)
    if not games and slate_path and not live_used:
        games = load_slate_file(slate_path)
    if predictor is not None:
        games = [fill_defaults(g, predictor) for g in games]
    return games
