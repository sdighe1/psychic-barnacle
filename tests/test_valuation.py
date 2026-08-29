"""Valuation: replacement levels, dollar-pool identity, VORP and tiers."""
from __future__ import annotations

from ffauction import valuation


def test_replacement_levels_exist_for_all_positions(pool, league):
    vals = valuation.compute_values(pool, league)
    repl = valuation.replacement_levels(vals, league)
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        assert pos in repl and repl[pos] > 0


def test_dollar_values_sum_to_money_pool(pool, league):
    vals = valuation.compute_values(pool, league)
    top = vals.sort_values("vorp", ascending=False).head(league.total_roster_spots)
    total = int(top["optimal_dollar"].sum())
    # Identity holds up to integer rounding of each value.
    assert abs(total - league.total_money) <= 0.03 * league.total_money


def test_vorp_positive_for_studs_and_min_dollar(pool, league):
    vals = valuation.compute_values(pool, league)
    assert vals.iloc[0]["vorp"] > 0                     # best player above replacement
    assert vals["optimal_dollar"].min() >= 1           # every player at least $1
    # A clearly below-replacement player is valued at the $1 floor.
    assert vals.sort_values("vorp").iloc[0]["optimal_dollar"] == 1


def test_flex_pulls_rbwr_replacement_below_base(pool, league):
    """With FLEX, more RB/WR are startable than the dedicated slots alone, so
    their replacement level sits at or below the pure base-starter cutoff."""
    vals = valuation.compute_values(pool, league)
    repl = valuation.replacement_levels(vals, league)
    rb = vals[vals["position"] == "RB"].sort_values("proj_points", ascending=False)
    base_cutoff = rb.iloc[league.base_starters("RB") - 1]["proj_points"]
    assert repl["RB"] <= base_cutoff + 1e-6


def test_tiers_start_at_one_and_best_is_tier_one(pool, league):
    vals = valuation.compute_values(pool, league)
    assert vals["tier"].min() == 1
    for pos in ("RB", "WR"):
        grp = vals[vals["position"] == pos].sort_values("proj_points", ascending=False)
        assert grp.iloc[0]["tier"] == 1
        assert grp["tier"].is_monotonic_increasing
