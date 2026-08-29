"""Draft engine: max bid, position caps, inflation, recommendations, slots."""
from __future__ import annotations

import pytest

from ffauction import valuation
from ffauction.draft import DraftState


@pytest.fixture
def ds(pool, league):
    return DraftState(league, valuation.compute_values(pool, league))


def test_max_bid_leaves_a_dollar_per_open_slot(ds):
    top = str(ds.values.iloc[0]["player_id"])
    # fresh: budget - (roster_size - 1)
    assert ds.max_bid(top) == ds.league.budget - (ds.league.roster_size - 1)
    ds.draft(top, 30, mine=True)
    other = str(ds.values.iloc[1]["player_id"])
    # after one pick: remaining - (open - 1)
    assert ds.max_bid(other) == ds.my_remaining - (ds.my_open_slots - 1)


def test_cannot_exceed_position_cap(ds):
    qbs = ds.values[ds.values["position"] == "QB"]["player_id"].astype(str).tolist()
    cap = ds.league.cap("QB")
    for pid in qbs[:cap]:
        ds.draft(pid, 1, mine=True)
    extra = qbs[cap]
    assert ds.can_draft(extra) is False
    assert ds.max_bid(extra) == 0


def test_roster_full_blocks_further_drafts(ds):
    for pid in ds.values["player_id"].astype(str).tolist()[: ds.league.roster_size]:
        if ds.can_draft(pid):
            ds.draft(pid, 1, mine=True)
    assert ds.my_open_slots == 0
    nxt = ds.values["player_id"].astype(str).tolist()[ds.league.roster_size + 1]
    assert ds.max_bid(nxt) == 0


def test_overpay_deflates_remaining_expected_prices(ds):
    target = str(ds.values.iloc[0]["player_id"])
    before = ds.expected_prices()[ds.values.index[0]]
    # a low-value player bought for a lot removes money without removing value
    cheap = str(ds.values.sort_values("vorp").iloc[0]["player_id"])
    ds.draft(cheap, 40, mine=False)
    after = ds.expected_prices()[ds.values.index[0]]
    assert after < before                              # others get cheaper
    assert target == str(ds.values.iloc[0]["player_id"])


def test_recommendations_are_roster_aware(ds):
    # Draft my QB; recommendations should stop pushing QB and prefer RB/WR/TE.
    qb = str(ds.values[ds.values["position"] == "QB"].iloc[0]["player_id"])
    ds.draft(qb, 20, mine=True)
    recs = ds.recommendations(top_n=10)
    assert not recs.empty
    assert "QB" not in set(recs["position"])           # QB starter already filled
    # nothing recommended is drafted or unaffordable
    assert (recs["max_bid"] >= 1).all()
    assert recs["player_id"].astype(str).map(lambda i: not ds.is_drafted(i)).all()
    assert (recs["suggested_bid"] <= recs["max_bid"]).all()


def test_open_slots_track_flex_and_bench(ds):
    league = ds.league
    # fill both dedicated RB slots, next RB should count toward FLEX
    rbs = ds.values[ds.values["position"] == "RB"]["player_id"].astype(str).tolist()
    for pid in rbs[: league.starters["RB"]]:
        ds.draft(pid, 1, mine=True)
    assert ds.open_slots()["RB"] == 0
    assert ds.open_slots()["FLEX"] == league.starters["FLEX"]
    ds.draft(rbs[league.starters["RB"]], 1, mine=True)   # extra RB -> FLEX
    assert ds.open_slots()["FLEX"] == league.starters["FLEX"] - 1


def test_persistence_roundtrip(ds):
    ds.draft(str(ds.values.iloc[0]["player_id"]), 15, mine=True)
    ds.draft(str(ds.values.iloc[1]["player_id"]), 22, mine=False)
    data = ds.to_dict()
    fresh = DraftState(ds.league, ds.values)
    fresh.load_dict(data)
    assert fresh.picks == ds.picks
    assert fresh.my_spent == ds.my_spent
