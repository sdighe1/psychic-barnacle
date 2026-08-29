"""Value-Based Drafting: turn projected points into auction dollars.

Pipeline
--------
1. ``compute_values`` scores every player for the league's format, finds each
   position's **replacement level** (with proper FLEX handling), and converts
   Value-Over-Replacement (VORP) into an **optimal** dollar value that sums to
   the league's money pool.
2. ``live_expected_prices`` re-derives dollars from the money and value that are
   *still on the board* as players get drafted -- i.e. auction **inflation** --
   giving a live "what will this actually cost now" estimate.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import scoring
from .league import FLEX_ELIGIBLE, LeagueSettings, normalize_position

# Kicker and team defense have a real projected-points spread but the auction
# market pays ~$1 for them regardless (the spread is unpredictable). Excluding
# them from the dollar pool keeps their value at $1 and lets skill players
# absorb that money -- matching how auctions actually price these positions.
STREAM_POSITIONS = ("K", "DST")


# --------------------------------------------------------------------------- #
# Replacement levels
# --------------------------------------------------------------------------- #
def replacement_levels(df: pd.DataFrame, league: LeagueSettings) -> Dict[str, float]:
    """Projected points of the last *startable* player at each position.

    RB/WR/TE share the league's FLEX slots: dedicated starters are counted
    first, then the best remaining RB/WR/TE fill the FLEX pool, and each
    position's replacement level is the worst starter (or flex) it lands.
    """
    levels: Dict[str, float] = {}

    # Non-flex positions: replacement = the (base starters + 1)-th best.
    for pos in ("QB", "DST", "K"):
        pts = np.sort(df.loc[df["position"] == pos, "proj_points"].to_numpy())[::-1]
        n = league.base_starters(pos)
        levels[pos] = _nth_or_floor(pts, n)

    # Flex pool: rank all RB/WR/TE together, seat dedicated starters, then FLEX.
    startable_counts = {pos: league.base_starters(pos) for pos in FLEX_ELIGIBLE}
    pool = (
        df[df["position"].isin(FLEX_ELIGIBLE)][["position", "proj_points"]]
        .sort_values("proj_points", ascending=False)
        .reset_index(drop=True)
    )
    # Which rows are already dedicated starters?
    seated = {pos: 0 for pos in FLEX_ELIGIBLE}
    is_flex_candidate = np.zeros(len(pool), dtype=bool)
    for i, pos in enumerate(pool["position"].to_numpy()):
        if seated[pos] < startable_counts[pos]:
            seated[pos] += 1
        else:
            is_flex_candidate[i] = True
    # Best `flex_slots` of the leftovers win the flex jobs.
    flex_idx = np.where(is_flex_candidate)[0][: league.flex_slots]
    for i in flex_idx:
        startable_counts[pool.at[i, "position"]] += 1

    for pos in FLEX_ELIGIBLE:
        pts = np.sort(df.loc[df["position"] == pos, "proj_points"].to_numpy())[::-1]
        levels[pos] = _nth_or_floor(pts, startable_counts[pos])

    return levels


def _nth_or_floor(sorted_desc: np.ndarray, n: int) -> float:
    """Points of the n-th ranked player (1-indexed); if there are fewer than n
    players, fall back to the worst available (or 0)."""
    if len(sorted_desc) == 0 or n <= 0:
        return 0.0
    idx = min(n, len(sorted_desc)) - 1
    return float(sorted_desc[idx])


# --------------------------------------------------------------------------- #
# Dollar values
# --------------------------------------------------------------------------- #
def _dollars(vorp: np.ndarray, dollar_per_point: float) -> np.ndarray:
    return np.maximum(1.0, 1.0 + np.maximum(vorp, 0.0) * dollar_per_point)


def _dollar_per_point(vorp_pos: np.ndarray, discretionary: float, n_pool: int) -> float:
    """Dollars per VORP point given a discretionary budget and the positive
    VORP of the players expected to be rostered."""
    top = np.sort(vorp_pos)[::-1][: max(n_pool, 0)]
    total = float(top[top > 0].sum())
    if total <= 0 or discretionary <= 0:
        return 0.0
    return discretionary / total


def compute_values(df: pd.DataFrame, league: LeagueSettings) -> pd.DataFrame:
    """Add ``proj_points``, ``vorp``, ``optimal_dollar``, ``tier`` and ranks."""
    out = df.copy()
    out["position"] = out["position"].map(normalize_position)
    out["proj_points"] = scoring.project_points(out, league.scoring_format)

    repl = replacement_levels(out, league)
    out["replacement"] = out["position"].map(repl).fillna(0.0)
    out["vorp"] = (out["proj_points"] - out["replacement"]).round(2)

    discretionary = league.total_money - league.total_roster_spots  # reserve $1/spot
    pool = ~out["position"].isin(STREAM_POSITIONS)
    dpp = _dollar_per_point(out.loc[pool, "vorp"].to_numpy(), discretionary, league.total_roster_spots)
    dollars = np.rint(_dollars(out["vorp"].to_numpy(), dpp))
    out["optimal_dollar"] = np.where(pool.to_numpy(), dollars, 1).astype(int)

    out["tier"] = assign_tiers(out)
    # Rank by auction VALUE (VORP), not raw points, so positional scarcity is
    # baked into the board order; break ties on raw projected points.
    out = out.sort_values(["vorp", "proj_points"], ascending=False).reset_index(drop=True)
    out["overall_rank"] = np.arange(1, len(out) + 1)
    out["pos_rank"] = out.groupby("position")["proj_points"].rank(ascending=False, method="first").astype(int)
    return out


def live_expected_prices(
    values: pd.DataFrame,
    league: LeagueSettings,
    drafted_ids: Iterable[str],
    total_spent: float,
) -> pd.Series:
    """Inflation-adjusted market price for every player, given the picks so far.

    Returns a Series aligned to ``values.index``. Money and positive VORP still
    on the board set a fresh dollars-per-point; early bargains push remaining
    prices up, early overspends push them down.
    """
    drafted = set(map(str, drafted_ids))
    is_drafted = values["player_id"].astype(str).isin(drafted).to_numpy()
    n_drafted = int(is_drafted.sum())

    remaining_money = league.total_money - float(total_spent)
    remaining_spots = league.total_roster_spots - n_drafted
    if remaining_spots <= 0:
        return pd.Series(np.maximum(1, values["optimal_dollar"]), index=values.index)

    discretionary = remaining_money - remaining_spots  # keep $1 per open spot
    pool = ~values["position"].isin(STREAM_POSITIONS)
    avail_pool = (~is_drafted) & pool.to_numpy()
    dpp = _dollar_per_point(values.loc[avail_pool, "vorp"].to_numpy(), discretionary, remaining_spots)

    prices = np.rint(_dollars(values["vorp"].to_numpy(), dpp))
    prices = np.where(pool.to_numpy(), prices, 1).astype(float)
    return pd.Series(prices, index=values.index)


# --------------------------------------------------------------------------- #
# Tiers (gap-based, per position)
# --------------------------------------------------------------------------- #
def assign_tiers(df: pd.DataFrame, min_gap: float = 3.0) -> pd.Series:
    """Group each position's players into tiers, breaking where the drop in
    projected points is unusually large. Tier 1 is the best."""
    tiers = pd.Series(1, index=df.index, dtype=int)
    for pos, grp in df.groupby("position"):
        ordered = grp.sort_values("proj_points", ascending=False)
        pts = ordered["proj_points"].to_numpy()
        if len(pts) <= 1:
            continue
        gaps = -np.diff(pts)
        gaps = gaps[gaps >= 0]
        thr = max(min_gap, float(np.mean(gaps) + np.std(gaps))) if len(gaps) else min_gap
        t, assigned = 1, []
        prev: Optional[float] = None
        for p in pts:
            if prev is not None and (prev - p) > thr:
                t += 1
            assigned.append(t)
            prev = p
        tiers.loc[ordered.index] = assigned
    return tiers
