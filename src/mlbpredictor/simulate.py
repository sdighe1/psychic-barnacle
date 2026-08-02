"""Plate-appearance Monte-Carlo game simulator.

A base-out state machine plays out full games PA-by-PA. Each batter's outcome is
drawn from a precomputed matchup distribution (vs the opposing starter, then the
bullpen once the starter is pulled). Runners carry their identity so we credit runs
and RBIs and produce per-player stat lines — the raw material for props — alongside
each team's run total. Aggregating many simulations gives the win probability, the
run/total/run-line distributions and prop over/under probabilities, all internally
consistent because they come from one engine.

Advancement is a compact, standard model (forced walks; singles/doubles advance
runners with tuned extra-base probabilities; GIDP and sac flies on outs). The few
constants are tuned so a league-average game reproduces ~4.5 runs/team.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matchup import matchup_probs, scale_offense
from .retrosheet import PA_OUTCOMES

# Outcome indices.
I_1B, I_2B, I_3B, I_HR, I_BB, I_HBP, I_SO, I_OUT = range(8)
_TOTAL_BASES = np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=float)
_IS_HIT = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)

# Advancement constants (tuned so a league-average game reproduces ~4.5 R/team).
P_2ND_SCORES_ON_1B = 0.62
P_1ST_TO_3RD_ON_1B = 0.30
P_1ST_SCORES_ON_2B = 0.45
P_GIDP = 0.10
P_SAC_FLY = 0.30
P_OUT_ADVANCE = 0.26      # a "productive out" nudges each runner up one base


def precompute_matchups(lineup_vecs, starter_vec, bullpen_vec, league_vec,
                        park_factor: float = 1.0, tto_factors=(1.0,)):
    """Return ``(vs_starter[9], vs_bullpen[9])`` cumulative outcome distributions.

    ``vs_starter[i]`` is a list of cumulative dists — one per *time through the order*
    (``tto_factors`` scales the batter's offense as the starter tires). ``vs_bullpen[i]``
    is a single cumulative dist.
    """
    vs_sp, vs_bp = [], []
    for b in lineup_vecs:
        base_sp = matchup_probs(b, starter_vec, league_vec, park_factor)
        vs_sp.append([np.cumsum(scale_offense(base_sp, f)) for f in tto_factors])
        vs_bp.append(np.cumsum(matchup_probs(b, bullpen_vec, league_vec, park_factor)))
    return vs_sp, vs_bp


@dataclass
class TeamPack:
    """One team's batting inputs against the opponent's pitching."""
    vs_sp: list          # 9 lists of per-time-through-order cumulative dists vs the starter
    vs_bp: list          # 9 cumulative dists vs opposing bullpen
    starter_max_batters: int = 27


def _sample(cum: np.ndarray, u: float) -> int:
    return int(np.searchsorted(cum, u, side="right"))


class _BoxScore:
    """Per-sim accumulators for one team's batters and its two pitchers."""
    __slots__ = ("bat", "runs", "sp_out", "sp_k", "sp_bb", "sp_h", "sp_r",
                 "bp_r", "bp_k")

    def __init__(self):
        # bat[pos] = [1B,2B,3B,HR,BB,HBP,SO,OUT, R, RBI]
        self.bat = np.zeros((9, 10), dtype=np.int32)
        self.runs = 0
        self.sp_out = 0; self.sp_k = 0; self.sp_bb = 0; self.sp_h = 0; self.sp_r = 0
        self.bp_r = 0; self.bp_k = 0


def _play_half(pack: TeamPack, order_idx: int, bf_before: int, box: _BoxScore,
               rng: np.random.Generator, ghost: bool) -> int:
    """Simulate one half-inning; mutate ``box``; return the next batting index.

    Bases ``b1/b2/b3`` hold the batting-order position of the runner (or -1). The
    advancement is collision-free: lead runners are resolved before trailing ones.
    """
    use_bp = bf_before >= pack.starter_max_batters
    outs = 0
    b1 = b2 = b3 = -1
    if ghost:                                   # extra-innings runner on 2nd
        b2 = (order_idx - 1) % 9
    us = rng.random(128)
    up = 0
    faced = bf_before                           # starter's cumulative batters faced
    while outs < 3:
        if up >= len(us) - 5:
            us = rng.random(128); up = 0
        if use_bp:
            cum = pack.vs_bp[order_idx]
        else:                                   # pick the time-through-order tier
            tiers = pack.vs_sp[order_idx]
            cum = tiers[min(faced // 9, len(tiers) - 1)]
        o = _sample(cum, us[up]); up += 1
        faced += 1
        box.bat[order_idx, o] += 1
        if o == I_SO:
            if use_bp:
                box.bp_k += 1
            else:
                box.sp_k += 1
        elif o == I_BB and not use_bp:
            box.sp_bb += 1
        if o < I_BB and not use_bp:             # a hit (1B/2B/3B/HR)
            box.sp_h += 1

        scored = 0
        outs_added = 0

        if o == I_SO:
            outs_added = 1
        elif o == I_BB or o == I_HBP:           # forced advance only
            if b1 == -1:
                b1 = order_idx
            elif b2 == -1:
                b2 = b1; b1 = order_idx
            elif b3 == -1:
                b3 = b2; b2 = b1; b1 = order_idx
            else:
                box.bat[b3, 8] += 1; scored += 1
                b3 = b2; b2 = b1; b1 = order_idx
        elif o == I_1B:
            r1, r2, r3 = b1, b2, b3
            b1 = b2 = b3 = -1
            if r3 != -1:
                box.bat[r3, 8] += 1; scored += 1
            if r2 != -1:
                if us[up] < P_2ND_SCORES_ON_1B:
                    box.bat[r2, 8] += 1; scored += 1
                else:
                    b3 = r2
                up += 1
            if r1 != -1:
                if b3 == -1 and us[up] < P_1ST_TO_3RD_ON_1B:
                    b3 = r1
                else:
                    b2 = r1
                up += 1
            b1 = order_idx
        elif o == I_2B:
            r1, r2, r3 = b1, b2, b3
            b1 = b2 = b3 = -1
            if r3 != -1:
                box.bat[r3, 8] += 1; scored += 1
            if r2 != -1:
                box.bat[r2, 8] += 1; scored += 1
            if r1 != -1:
                if us[up] < P_1ST_SCORES_ON_2B:
                    box.bat[r1, 8] += 1; scored += 1
                else:
                    b3 = r1
                up += 1
            b2 = order_idx
        elif o == I_3B:
            for r in (b1, b2, b3):
                if r != -1:
                    box.bat[r, 8] += 1; scored += 1
            b1 = b2 = -1; b3 = order_idx
        elif o == I_HR:
            for r in (b1, b2, b3):
                if r != -1:
                    box.bat[r, 8] += 1; scored += 1
            box.bat[order_idx, 8] += 1; scored += 1
            b1 = b2 = b3 = -1
        else:                                    # OUT in play
            r_dp = us[up]; r_sf = us[up + 1]; r_adv = us[up + 2]; up += 3
            if outs < 2 and b1 != -1 and r_dp < P_GIDP:
                outs_added = 2; b1 = -1                     # ground into double play
            elif outs < 2 and b3 != -1 and r_sf < P_SAC_FLY:
                box.bat[b3, 8] += 1; scored += 1; b3 = -1; outs_added = 1  # sac fly
            else:
                outs_added = 1
                if outs < 2 and r_adv < P_OUT_ADVANCE:      # productive out
                    if b3 != -1:
                        box.bat[b3, 8] += 1; scored += 1; b3 = -1
                    if b2 != -1:
                        b3 = b2; b2 = -1
                    if b1 != -1:
                        b2 = b1; b1 = -1

        outs += outs_added
        if not use_bp:
            box.sp_out += outs_added
        if scored:
            box.bat[order_idx, 9] += scored        # RBI to the batter
            box.runs += scored
            if use_bp:
                box.bp_r += scored
            else:
                box.sp_r += scored
        order_idx = (order_idx + 1) % 9
    return order_idx


@dataclass
class GameSimResult:
    away_runs: np.ndarray          # (N,)
    home_runs: np.ndarray          # (N,)
    away_box: np.ndarray           # (N, 9, 10)
    home_box: np.ndarray           # (N, 9, 10)
    away_pitch: dict               # arrays keyed by stat for away starter/bullpen
    home_pitch: dict
    away_runs_f5: np.ndarray = None   # runs through 5 innings (first-5 market)
    home_runs_f5: np.ndarray = None

    # -- headline markets -------------------------------------------------- #
    def p_home_win(self) -> float:
        diff = self.home_runs - self.away_runs
        wins = (diff > 0).sum()
        ties = (diff == 0).sum()          # (shouldn't happen: extra innings resolve)
        return float((wins + 0.5 * ties) / len(diff))

    def exp_runs(self) -> tuple[float, float]:
        return float(self.away_runs.mean()), float(self.home_runs.mean())

    def total(self) -> np.ndarray:
        return self.away_runs + self.home_runs

    def p_over(self, line: float) -> float:
        t = self.total()
        return float(((t > line).sum() + 0.5 * (t == line).sum()) / len(t))

    def p_home_cover(self, line: float = -1.5) -> float:
        """P(home margin + line > 0), e.g. home -1.5 run line."""
        margin = self.home_runs - self.away_runs
        return float((margin + line > 0).mean())

    def most_likely_score(self) -> tuple[int, int]:
        a = int(np.round(self.away_runs.mean()))
        h = int(np.round(self.home_runs.mean()))
        return a, h


def simulate_game(away: TeamPack, home: TeamPack, n_sims: int = 5000,
                  ghost_runner: bool = True, seed: int = 0,
                  max_extra: int = 6) -> GameSimResult:
    """Monte-Carlo a full game ``n_sims`` times; return distributions and box lines."""
    rng = np.random.default_rng(seed)
    away_runs = np.zeros(n_sims, dtype=np.int32)
    home_runs = np.zeros(n_sims, dtype=np.int32)
    away_box = np.zeros((n_sims, 9, 10), dtype=np.int32)
    home_box = np.zeros((n_sims, 9, 10), dtype=np.int32)
    away_f5 = np.zeros(n_sims, dtype=np.int32)
    home_f5 = np.zeros(n_sims, dtype=np.int32)
    ap = {k: np.zeros(n_sims, dtype=np.int32) for k in ("sp_k", "sp_bb", "sp_h", "sp_r", "sp_out", "bp_k", "bp_r")}
    hp = {k: np.zeros(n_sims, dtype=np.int32) for k in ("sp_k", "sp_bb", "sp_h", "sp_r", "sp_out", "bp_k", "bp_r")}

    for s in range(n_sims):
        abox, hbox = _BoxScore(), _BoxScore()
        a_idx = h_idx = 0
        a_bf = h_bf = 0
        inning = 0
        while True:
            inning += 1
            ghost = ghost_runner and inning >= 10
            # Top: away bats vs home pitching.
            a_idx = _play_half(away, a_idx, a_bf, abox, rng, ghost)
            a_bf = abox.bat[:, :8].sum()
            # Bottom: home bats (skip if home already leads after 9).
            if inning >= 9 and hbox.runs > abox.runs:
                break
            h_idx = _play_half(home, h_idx, h_bf, hbox, rng, ghost)
            h_bf = hbox.bat[:, :8].sum()
            if inning == 5:                        # first-5-innings snapshot
                away_f5[s] = abox.runs
                home_f5[s] = hbox.runs
            if inning >= 9 and hbox.runs != abox.runs:
                break
            if inning >= 9 + max_extra:       # safety cap
                if hbox.runs == abox.runs:    # break tie by a coin flip
                    if rng.random() < 0.5:
                        hbox.runs += 1
                    else:
                        abox.runs += 1
                break
        away_runs[s] = abox.runs
        home_runs[s] = hbox.runs
        away_box[s] = abox.bat
        home_box[s] = hbox.bat
        for d, box in ((ap, abox), (hp, hbox)):
            d["sp_k"][s] = box.sp_k; d["sp_bb"][s] = box.sp_bb; d["sp_h"][s] = box.sp_h
            d["sp_r"][s] = box.sp_r; d["bp_k"][s] = box.bp_k; d["bp_r"][s] = box.bp_r
            d["sp_out"][s] = min(box.sp_out, 27)

    return GameSimResult(away_runs, home_runs, away_box, home_box, ap, hp,
                         away_runs_f5=away_f5, home_runs_f5=home_f5)
