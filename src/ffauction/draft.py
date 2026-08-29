"""Live draft state: who's gone, my roster, max bids and recommendations.

``DraftState`` wraps a valued player table (from :func:`valuation.compute_values`)
and the picks made so far. It answers the three questions the app is built
around:

* **Expected $** -- the live, inflation-adjusted market price (via
  :func:`valuation.live_expected_prices`).
* **Max bid** -- the most *I* can pay for a player and still legally fill my
  roster (leaving $1 for every other open slot), respecting ESPN position caps.
* **Recommendations** -- who to target next given my roster needs, budget and
  the value left on the board.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import valuation
from .league import FLEX_ELIGIBLE, POSITIONS, LeagueSettings


@dataclass
class Pick:
    player_id: str
    price: int
    mine: bool


@dataclass
class DraftState:
    league: LeagueSettings
    values: pd.DataFrame                       # output of valuation.compute_values
    picks: Dict[str, Pick] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def draft(self, player_id: str, price: int, mine: bool) -> None:
        self.picks[str(player_id)] = Pick(str(player_id), int(price), bool(mine))

    def undo(self, player_id: str) -> None:
        self.picks.pop(str(player_id), None)

    def reset(self) -> None:
        self.picks.clear()

    def is_drafted(self, player_id: str) -> bool:
        return str(player_id) in self.picks

    # ------------------------------------------------------------------ #
    # Budgets
    # ------------------------------------------------------------------ #
    @property
    def total_spent(self) -> int:
        return sum(p.price for p in self.picks.values())

    @property
    def my_spent(self) -> int:
        return sum(p.price for p in self.picks.values() if p.mine)

    @property
    def my_remaining(self) -> int:
        return self.league.budget - self.my_spent

    @property
    def my_player_ids(self) -> List[str]:
        return [pid for pid, p in self.picks.items() if p.mine]

    @property
    def my_open_slots(self) -> int:
        return self.league.roster_size - len(self.my_player_ids)

    def my_roster(self) -> pd.DataFrame:
        ids = set(self.my_player_ids)
        r = self.values[self.values["player_id"].astype(str).isin(ids)].copy()
        r["price"] = r["player_id"].astype(str).map(lambda i: self.picks[i].price)
        return r.sort_values("price", ascending=False)

    def my_position_counts(self) -> Counter:
        roster = self.my_roster()
        return Counter(roster["position"].tolist())

    # ------------------------------------------------------------------ #
    # Roster construction (which starter slots are still open?)
    # ------------------------------------------------------------------ #
    def open_slots(self) -> Dict[str, int]:
        """Open **starter** slots by category plus ``FLEX`` and ``BENCH``.

        Players are seated greedily: dedicated starter slots first, then FLEX
        for RB/WR/TE overflow, then the bench.
        """
        counts = self.my_position_counts()
        starters = self.league.starters
        open_ded, overflow = {}, {}
        for pos in POSITIONS:
            req = starters.get(pos, 0)
            have = counts.get(pos, 0)
            open_ded[pos] = max(0, req - have)
            overflow[pos] = max(0, have - req)

        flex_req = starters.get("FLEX", 0)
        flex_avail = sum(overflow[p] for p in FLEX_ELIGIBLE)
        flex_filled = min(flex_req, flex_avail)
        open_flex = flex_req - flex_filled

        seated = sum(min(counts.get(p, 0), starters.get(p, 0)) for p in POSITIONS) + flex_filled
        bench_used = max(0, len(self.my_player_ids) - seated)
        bench_open = max(0, self.league.bench - bench_used)

        result = {pos: open_ded[pos] for pos in POSITIONS}
        result["FLEX"] = open_flex
        result["BENCH"] = bench_open
        return result

    def open_starter_positions(self) -> List[str]:
        """Positions with an unfilled dedicated starter slot (RB/WR/TE also
        listed while a FLEX remains)."""
        openings = self.open_slots()
        needs = [pos for pos in POSITIONS if openings.get(pos, 0) > 0]
        if openings.get("FLEX", 0) > 0:
            for pos in FLEX_ELIGIBLE:
                if pos not in needs:
                    needs.append(pos)
        return needs

    # ------------------------------------------------------------------ #
    # Draftability & max bid
    # ------------------------------------------------------------------ #
    def can_draft(self, player_id: str, mine: bool = True) -> bool:
        if self.is_drafted(player_id):
            return False
        if not mine:
            return True
        if self.my_open_slots <= 0:
            return False
        row = self._row(player_id)
        if row is None:
            return False
        pos = row["position"]
        if self.my_position_counts().get(pos, 0) >= self.league.cap(pos):
            return False
        return True

    def max_bid(self, player_id: str) -> int:
        """Most I can bid on this player and still fill every remaining slot at
        $1. Returns 0 if the player can't legally join my roster."""
        if not self.can_draft(player_id, mine=True):
            return 0
        # Bidding on one slot; keep $1 for each of the OTHER open slots.
        return max(1, self.my_remaining - (self.my_open_slots - 1))

    # ------------------------------------------------------------------ #
    # Board & prices
    # ------------------------------------------------------------------ #
    def expected_prices(self) -> pd.Series:
        return valuation.live_expected_prices(
            self.values, self.league, self.picks.keys(), self.total_spent
        )

    def board(self, available_only: bool = True) -> pd.DataFrame:
        """The full player table decorated with live Expected $, Max Bid, draft
        status and (if drafted) the price and team."""
        b = self.values.copy()
        b["expected_dollar"] = self.expected_prices().astype(int)
        ids = b["player_id"].astype(str)
        b["drafted"] = ids.isin(self.picks)
        b["draft_price"] = ids.map(lambda i: self.picks[i].price if i in self.picks else np.nan)
        b["drafted_by"] = ids.map(
            lambda i: ("Me" if self.picks[i].mine else "Other") if i in self.picks else ""
        )
        # Vectorised Max Bid: the same budget-based ceiling for every player I
        # can legally add, 0 for the rest (drafted / roster full / cap reached).
        base = self.max_bid_any()
        counts = self.my_position_counts()
        under_cap = b["position"].map(lambda p: counts.get(p, 0) < self.league.cap(p))
        can = (~b["drafted"]) & (self.my_open_slots > 0) & under_cap
        b["max_bid"] = np.where(can, base, 0).astype(int)
        if available_only:
            b = b[~b["drafted"]]
        return b

    # ------------------------------------------------------------------ #
    # Recommendations
    # ------------------------------------------------------------------ #
    def recommendations(self, top_n: int = 12) -> pd.DataFrame:
        """Rank available, draftable players as targets for *my* roster.

        Score blends roster **need** (open starter slot, scaled by how few
        startable options remain), **value** (optimal minus expected, plus raw
        VORP) and **tier scarcity** (few left in the player's tier)."""
        board = self.board(available_only=True)
        # max_bid >= 1 means the player can legally join my roster (respects
        # position caps and a full roster) and I can afford at least $1.
        board = board[board["max_bid"] >= 1]
        if board.empty:
            return board.assign(rec_score=[], suggested_bid=[], why=[])

        openings = self.open_slots()
        starter_needs = self.open_starter_positions()

        # startable-quality (positive VORP) options still available per position
        avail_pos_supply = (
            board[board["vorp"] > 0].groupby("position")["player_id"].count().to_dict()
        )
        # players remaining in each (position, tier) bucket
        tier_supply = board.groupby(["position", "tier"])["player_id"].count().to_dict()

        rows = []
        for _, r in board.iterrows():
            pos = r["position"]
            # --- need ---
            if openings.get(pos, 0) > 0:
                need = 1.0
            elif pos in FLEX_ELIGIBLE and openings.get("FLEX", 0) > 0:
                need = 0.7
            elif openings.get("BENCH", 0) > 0:
                need = 0.3
            else:
                need = 0.0
            supply = avail_pos_supply.get(pos, 1)
            urgency = 1.0 / np.sqrt(max(1, supply))          # scarcer -> more urgent
            need_score = need * (0.6 + 0.4 * urgency) if pos in starter_needs or need >= 0.3 else need * 0.3

            # --- value ---
            surplus = float(r["optimal_dollar"] - r["expected_dollar"])   # bargain if > 0
            vorp = float(max(r["vorp"], 0.0))

            # --- tier scarcity ---
            left_in_tier = tier_supply.get((pos, r["tier"]), 1)
            scarce = 1.0 / max(1, left_in_tier)

            # affordability: gentle penalty if it will cost more than I can pay
            afford = 1.0 if r["expected_dollar"] <= max(1, r["max_bid"]) else 0.5

            rows.append({
                "need_score": need_score, "surplus": surplus, "vorp": vorp,
                "scarce": scarce, "afford": afford,
            })
        comp = pd.DataFrame(rows, index=board.index)

        score = (
            0.42 * need_score_norm(comp["need_score"])
            + 0.24 * _norm(comp["vorp"])
            + 0.16 * _norm(comp["surplus"])
            + 0.10 * comp["scarce"]
            + 0.08 * _norm(comp["afford"])
        ) * comp["afford"]
        board = board.assign(rec_score=(score * 100).round(1).clip(upper=100))

        board["suggested_bid"] = np.minimum(
            board["max_bid"], np.maximum(1, board["expected_dollar"])
        ).astype(int)
        board["why"] = [
            self._rationale(r, openings, tier_supply) for _, r in board.iterrows()
        ]
        return board.sort_values("rec_score", ascending=False).head(top_n)

    def max_bid_any(self) -> int:
        """The single largest bid I could make on anyone right now."""
        if self.my_open_slots <= 0:
            return 0
        return max(1, self.my_remaining - (self.my_open_slots - 1))

    def _rationale(self, r, openings, tier_supply) -> str:
        pos = r["position"]
        bits = []
        if openings.get(pos, 0) > 0:
            bits.append(f"fills {pos} starter")
        elif pos in FLEX_ELIGIBLE and openings.get("FLEX", 0) > 0:
            bits.append("fills FLEX")
        else:
            bits.append(f"{pos} depth")
        surplus = int(r["optimal_dollar"] - r["expected_dollar"])
        if surplus >= 2:
            bits.append(f"value (worth ${int(r['optimal_dollar'])} vs ${int(r['expected_dollar'])})")
        left = tier_supply.get((pos, r["tier"]), 1)
        if left <= 2:
            bits.append(f"only {left} left in {pos} tier {int(r['tier'])}")
        return "; ".join(bits)

    # ------------------------------------------------------------------ #
    # Helpers & persistence
    # ------------------------------------------------------------------ #
    def _row(self, player_id: str):
        m = self.values[self.values["player_id"].astype(str) == str(player_id)]
        return None if m.empty else m.iloc[0]

    def to_dict(self) -> dict:
        return {
            "picks": {pid: {"price": p.price, "mine": p.mine} for pid, p in self.picks.items()},
        }

    def load_dict(self, data: dict) -> None:
        self.picks = {
            str(pid): Pick(str(pid), int(v["price"]), bool(v["mine"]))
            for pid, v in (data.get("picks") or {}).items()
        }


def _norm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def need_score_norm(s: pd.Series) -> pd.Series:
    # need already lives in [0, 1]; keep as-is but guard empties
    return s.astype(float).clip(0.0, 1.0)
