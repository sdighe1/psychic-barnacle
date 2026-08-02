"""Load ``config/mlb.yaml`` with sensible code defaults.

Every key has a default here, so the package works even if the YAML is missing or
partial. ``load_config()`` deep-merges the file over the defaults.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import yaml

from .paths import MLB_CONFIG_PATH

DEFAULTS: dict = {
    "data": {
        "retro_base": "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons",
        "register_base": "https://raw.githubusercontent.com/chadwickbureau/register/master/data",
        "register_shards": 10,
        "elo_seasons": [2017, 2018, 2019, 2021, 2022, 2023, 2024],
        "rate_seasons": [2022, 2023, 2024],
    },
    "projections": {
        "season_weights": [5.0, 4.0, 3.0],
        "bat_regress_pa": 200.0,
        "pit_regress_bf": 300.0,
        "age_peak": 27.0,
        "age_adj_per_year": 0.003,
    },
    "intervals": {
        "levels": [0.5, 0.9],
    },
    "freshen": {
        "enabled": True,
        "elo_revert": 0.20,
        "w_base_pa": 400,
        "w_base_bf": 500,
    },
    "simulation": {
        "n_sims": 5000,
        "starter_max_tto": 3.0,
        "starter_max_batters": 27,
        "tto_factors": [0.97, 1.00, 1.05],
        "extra_innings_ghost_runner": True,
        "random_seed": 0,
    },
    "bullpen": {
        # Live specific-reliever bullpen (statsapi active roster + recent usage). When
        # off / offline the season-aggregate team bullpen is used unchanged.
        "use_live_relievers": True,
        "quality_temp": 0.06,     # softmax temperature over −wOBA-against (lower ⇒ favor best arms)
        "min_relievers": 3,       # need this many known relievers, else season aggregate
        "lookback_days": 2,       # recent days scanned for fatigue
        "b2b_penalty": 0.15,      # pitched both of the last 2 days ⇒ likely down today
        "day1_penalty": 0.60,     # pitched yesterday only
        "day2_penalty": 0.90,     # pitched 2 days ago only
        "starter_gs_frac": 0.5,   # season gamesStarted/games ≥ this ⇒ treat as a starter, exclude
    },
    "live": {
        "statsapi_base": "https://statsapi.mlb.com/api/v1",
        "schedule_hydrate": "probablePitcher(note),lineups,team,linescore",
        "timeout_seconds": 20,
    },
    "statcast": {
        # xwOBA "de-luck" of projections. Needs baseballsavant.mlb.com whitelisted, so
        # it is OFF by default and fails soft (projections untouched) when unavailable.
        "enabled": False,
        "base_url": "https://baseballsavant.mlb.com/leaderboard/expected_statistics",
        "regress_pa": 200.0,   # PA of regression of the luck delta toward zero
        "clip": 0.15,          # cap the per-player wOBA luck correction at ±15%
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Return the merged config dict (file over defaults). Cached."""
    file_cfg: dict = {}
    if MLB_CONFIG_PATH.exists():
        with open(MLB_CONFIG_PATH) as f:
            file_cfg = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULTS, file_cfg)
