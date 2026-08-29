"""League settings: roster, budget, scoring format and per-position caps.

Defaults come from the user's ESPN league (see ``config/league.yaml``): 10
teams, $200 auction budget, a 9-man starting lineup with one FLEX, four bench
spots (13 draftable), plus per-position maximums. Everything is editable in the
YAML file and, at runtime, in the app sidebar.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict

import yaml

from .paths import LEAGUE_CONFIG_PATH

# Positions that can occupy the FLEX slot.
FLEX_ELIGIBLE = ("RB", "WR", "TE")
# All draftable positions, in display order.
POSITIONS = ("QB", "RB", "WR", "TE", "DST", "K")

# ESPN uses a few labels; normalise everything to these codes.
POSITION_ALIASES = {
    "D/ST": "DST", "D-ST": "DST", "DEF": "DST", "DST": "DST",
    "PK": "K", "K": "K", "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "FB": "RB",
}


def normalize_position(pos: str) -> str:
    if pos is None:
        return ""
    return POSITION_ALIASES.get(str(pos).strip().upper(), str(pos).strip().upper())


@dataclass(frozen=True)
class LeagueSettings:
    teams: int = 10
    budget: int = 200
    scoring_format: str = "half_ppr"           # standard | half_ppr | ppr
    # Starting-lineup slots.
    starters: Dict[str, int] = field(default_factory=lambda: {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1,
    })
    bench: int = 4
    # Maximum rosterable players per position (ESPN caps).
    position_max: Dict[str, int] = field(default_factory=lambda: {
        "QB": 4, "RB": 8, "WR": 8, "TE": 3, "DST": 3, "K": 3,
    })

    # --- derived ------------------------------------------------------------
    @property
    def roster_size(self) -> int:
        """Draftable spots per team = starters + bench (IR is not drafted)."""
        return sum(self.starters.values()) + self.bench

    @property
    def total_money(self) -> int:
        return self.teams * self.budget

    @property
    def total_roster_spots(self) -> int:
        return self.teams * self.roster_size

    def base_starters(self, pos: str) -> int:
        """Dedicated starter slots for a position across the whole league
        (excludes FLEX, which is shared by RB/WR/TE)."""
        return self.teams * self.starters.get(pos, 0)

    @property
    def flex_slots(self) -> int:
        return self.teams * self.starters.get("FLEX", 0)

    def cap(self, pos: str) -> int:
        return int(self.position_max.get(pos, self.roster_size))

    def with_updates(self, **kwargs) -> "LeagueSettings":
        return replace(self, **kwargs)


def load_league(path=LEAGUE_CONFIG_PATH) -> LeagueSettings:
    """Load league settings from YAML, falling back to defaults for anything
    the file omits."""
    defaults = LeagueSettings()
    try:
        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return defaults

    starters = {**defaults.starters, **(cfg.get("starters") or {})}
    position_max = {**defaults.position_max, **(cfg.get("position_max") or {})}
    return LeagueSettings(
        teams=int(cfg.get("teams", defaults.teams)),
        budget=int(cfg.get("budget", defaults.budget)),
        scoring_format=str(cfg.get("scoring_format", defaults.scoring_format)),
        starters={k: int(v) for k, v in starters.items()},
        bench=int(cfg.get("bench", defaults.bench)),
        position_max={k: int(v) for k, v in position_max.items()},
    )
