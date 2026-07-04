"""Monte-Carlo simulator for the 2026 World Cup knockout stage.

The current state is reconstructed from the live results data with no hard-coded
bracket: a match is a *knockout* match when it is the 4th-or-later game for both
teams, its winner (penalty-shootout aware) advances, and the teams that have not
lost are still alive. From that frontier the simulator plays out the remaining
rounds thousands of times and reports, for every surviving team, its probability
of reaching each round and lifting the trophy.

Because a flat results file does not encode who-plays-who in future rounds, the
bracket order is taken from ``config/wc2026.yaml`` if given, otherwise the alive
teams are seeded into a standard single-elimination bracket by Elo rating.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from .data import load_matches, load_shootouts, wc2026_matches
from .paths import WC2026_CONFIG_PATH

GROUP_MATCHES_PER_TEAM = 3
ROUND_NAMES = {
    64: "Round of 64", 32: "Round of 32", 16: "Round of 16",
    8: "Quarter-final", 4: "Semi-final", 2: "Final",
}


# --------------------------------------------------------------------------- #
# State reconstruction
# --------------------------------------------------------------------------- #
def decide_winner(home, away, hs, as_, shootouts: pd.DataFrame) -> str:
    if hs > as_:
        return home
    if as_ > hs:
        return away
    row = shootouts[shootouts["home_team"].isin([home, away])
                    & shootouts["away_team"].isin([home, away])]
    if len(row):
        return row["winner"].iloc[0]
    return home  # unknown shootout: arbitrary but deterministic


def reconstruct_state(matches: pd.DataFrame | None = None,
                      shootouts: pd.DataFrame | None = None) -> dict:
    """Derive alive/eliminated teams and completed knockout results from data."""
    if matches is None:
        matches = load_matches()
    if shootouts is None:
        shootouts = load_shootouts()
    wc = wc2026_matches(matches).sort_values("date").reset_index(drop=True)
    so = shootouts[shootouts["date"].dt.year == 2026]

    order: dict[str, int] = defaultdict(int)
    knockout = []
    group_rows = []
    for r in wc.itertuples(index=False):
        hn, an = order[r.home_team], order[r.away_team]
        if hn >= GROUP_MATCHES_PER_TEAM and an >= GROUP_MATCHES_PER_TEAM:
            knockout.append(r)
        else:
            group_rows.append(r)
        order[r.home_team] += 1
        order[r.away_team] += 1

    all_teams = set(order)
    knockout_teams = {r.home_team for r in knockout} | {r.away_team for r in knockout}
    eliminated = all_teams - knockout_teams  # went out in the group stage
    alive = set(knockout_teams)
    completed = []
    for r in knockout:
        w = decide_winner(r.home_team, r.away_team, r.home_score, r.away_score, so)
        loser = r.away_team if w == r.home_team else r.home_team
        alive.discard(loser)
        eliminated.add(loser)
        completed.append({"home": r.home_team, "away": r.away_team,
                          "home_score": int(r.home_score), "away_score": int(r.away_score),
                          "winner": w})

    return {
        "alive": sorted(alive),
        "eliminated": sorted(eliminated),
        "completed_knockout": completed,
        "n_group_matches": len(group_rows),
        "n_knockout_matches": len(knockout),
        "next_round": ROUND_NAMES.get(len(alive), f"{len(alive)} teams"),
        "groups": _derive_groups(group_rows),
    }


def _derive_groups(group_rows: list) -> dict:
    """Connected-components grouping from group-stage matches, with standings."""
    adj = defaultdict(set)
    teams = set()
    for r in group_rows:
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)
        teams.update([r.home_team, r.away_team])
    seen, comps = set(), []
    for t in sorted(teams):
        if t in seen:
            continue
        stack, comp = [t], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        comps.append(sorted(comp))

    pts = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
    for r in group_rows:
        h, a = pts[r.home_team], pts[r.away_team]
        h["gf"] += r.home_score; h["ga"] += r.away_score
        a["gf"] += r.away_score; a["ga"] += r.home_score
        if r.home_score > r.away_score:
            h["pts"] += 3
        elif r.away_score > r.home_score:
            a["pts"] += 3
        else:
            h["pts"] += 1; a["pts"] += 1

    groups = {}
    for i, comp in enumerate(sorted(comps, key=lambda c: c[0])):
        label = chr(ord("A") + i) if i < 26 else f"G{i}"
        standing = sorted(
            [{"team": t, **pts[t], "gd": pts[t]["gf"] - pts[t]["ga"]} for t in comp],
            key=lambda d: (d["pts"], d["gd"], d["gf"]), reverse=True,
        )
        groups[label] = standing
    return groups


# --------------------------------------------------------------------------- #
# Bracket seeding
# --------------------------------------------------------------------------- #
def seed_order(n: int) -> list[int]:
    """Standard single-elimination bracket order of seeds 1..n (n a power of 2)."""
    order = [1]
    while len(order) < n:
        m = len(order) * 2
        order = [x for a in order for x in (a, m + 1 - a)]
    return order


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class TournamentSimulator:
    def __init__(self, predictor, config_path=WC2026_CONFIG_PATH,
                 matches: pd.DataFrame | None = None):
        self.predictor = predictor
        self.config = {}
        try:
            self.config = yaml.safe_load(open(config_path)) or {}
        except FileNotFoundError:
            pass
        self.state = reconstruct_state(matches)
        self.bracket = self._build_bracket()
        self._adv_cache: dict[tuple[str, str], float] = {}

    # -- bracket ------------------------------------------------------- #
    def _build_bracket(self) -> list[str]:
        alive = self.state["alive"]
        n = len(alive)
        if n < 2 or (n & (n - 1)) != 0:
            return alive  # not a clean power-of-two frontier; simulate as-is
        order = self.config.get("knockout_order") or []
        if order and set(order) == set(alive) and len(order) == n:
            return list(order)
        # Elo-seed: strongest seed 1, etc.
        ranked = [t for t in self.predictor.rankings()["team"] if t in set(alive)]
        ranked += [t for t in alive if t not in ranked]
        seeds = seed_order(n)
        return [ranked[s - 1] for s in seeds]

    # -- match advance probability (cached) ---------------------------- #
    def advance_prob(self, home: str, away: str) -> float:
        """P(``home`` advances) at a neutral knockout venue, shootout-aware."""
        key = (home, away)
        if key in self._adv_cache:
            return self._adv_cache[key]
        p = self.predictor.predict(home, away, neutral=True)
        denom = p.p_home + p.p_away
        p_home_shootout = p.p_home / denom if denom > 0 else 0.5
        adv = p.p_home + p.p_draw * p_home_shootout
        self._adv_cache[key] = adv
        self._adv_cache[(away, home)] = 1.0 - adv
        return adv

    # -- simulation ---------------------------------------------------- #
    def run(self, n_sims: int | None = None, seed: int | None = None) -> pd.DataFrame:
        n_sims = int(n_sims or self.config.get("simulations", 20000))
        seed = self.config.get("seed", 0) if seed is None else seed
        rng = np.random.default_rng(seed)
        bracket = self.bracket
        size = len(bracket)
        if size < 2:
            raise ValueError("Need at least two alive teams to simulate.")

        reach = defaultdict(lambda: defaultdict(int))  # team -> round_size -> count
        champ = defaultdict(int)
        # Precompute the round sequence sizes (e.g. 16,8,4,2).
        sizes = []
        s = size
        while s >= 2:
            sizes.append(s)
            s //= 2

        idx = {t: i for i, t in enumerate(bracket)}
        for _ in range(n_sims):
            cur = list(range(size))  # indices into bracket
            for rs in sizes:
                for t in cur:
                    reach[bracket[t]][rs] += 1
                nxt = []
                for i in range(0, len(cur), 2):
                    a, b = cur[i], cur[i + 1]
                    pa = self.advance_prob(bracket[a], bracket[b])
                    nxt.append(a if rng.random() < pa else b)
                cur = nxt
            champ[bracket[cur[0]]] += 1

        rows = []
        for t in bracket:
            row = {"team": t, "champion": champ[t] / n_sims}
            for rs in sizes:
                row[f"reach_{rs}"] = reach[t][rs] / n_sims
            rows.append(row)
        out = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
        out.attrs["sizes"] = sizes
        out.attrs["round_names"] = {rs: ROUND_NAMES.get(rs, f"{rs}") for rs in sizes}
        out.attrs["n_sims"] = n_sims
        return out
