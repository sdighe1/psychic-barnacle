"""Specific-reliever bullpen: fatigue policy, quality/fatigue weighting, live parsing."""
import numpy as np

import mlbpredictor.livedata as ld
from mlbpredictor.bullpen import (fatigue_multiplier, live_bullpen_vectors,
                                  team_bullpen_vector)
from mlbpredictor.livedata import (fetch_bullpen_usage, parse_boxscore_pitchers,
                                   parse_roster_pitchers)
from mlbpredictor.offense import woba
from mlbpredictor.retrosheet import PA_OUTCOMES

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()
SO, HR = PA_OUTCOMES.index("SO"), PA_OUTCOMES.index("HR")

CFG = {"quality_temp": 0.06, "min_relievers": 3, "lookback_days": 2,
       "b2b_penalty": 0.15, "day1_penalty": 0.60, "day2_penalty": 0.90,
       "starter_gs_frac": 0.5}


def _good(): v = LEAGUE.copy(); v[SO] *= 1.8; v[:4] *= 0.6; return v / v.sum()   # misses bats
def _bad():  v = LEAGUE.copy(); v[HR] *= 2.0; v[SO] *= 0.6; return v / v.sum()   # hittable


class FakePS:
    """Minimal ProjectionSystem stand-in for the bullpen blender."""
    def __init__(self, pit, season_bull):
        self._pit, self._bull = pit, season_bull
    def known_pitcher(self, rid): return rid in self._pit
    def pitcher(self, rid, vs=None): return self._pit[rid]
    def bullpen(self, team, vs=None): return self._bull


# ------------------------------- fatigue ------------------------------ #
def test_fatigue_multiplier_tiers():
    assert fatigue_multiplier([], CFG) == 1.0                 # rested
    assert fatigue_multiplier([3], CFG) == 1.0               # 3 days ago -> fresh
    assert fatigue_multiplier([2], CFG) == 0.90              # 2 days ago only
    assert fatigue_multiplier([1], CFG) == 0.60              # yesterday only
    assert fatigue_multiplier([1, 2], CFG) == 0.15           # back-to-back -> likely down


# --------------------------- quality weighting ------------------------ #
def test_quality_weight_favors_better_arms():
    pit = {"good1": _good(), "good2": _good(), "ace": _good(), "mop": _bad()}
    ps = FakePS(pit, LEAGUE)
    vec = team_bullpen_vector(ps, "AAA", list(pit), cfg=CFG)
    assert abs(vec.sum() - 1.0) < 1e-9
    # A pen that is mostly good arms allows a lower wOBA than a naive equal blend
    # that includes the mop-up man at full weight.
    equal = np.mean([pit[k] for k in pit], axis=0)
    assert woba(vec) < woba(equal)


def test_fatigue_downweights_the_rested_ace_out():
    # One elite arm + two average; when the ace is gassed (back-to-back) the blended
    # pen should be worse (higher wOBA) than when the ace is rested.
    pit = {"ace": _good(), "avg1": LEAGUE.copy(), "avg2": LEAGUE.copy()}
    ps = FakePS(pit, LEAGUE)
    rested = team_bullpen_vector(ps, "AAA", list(pit), cfg=CFG)
    gassed = team_bullpen_vector(ps, "AAA", list(pit),
                                 appearances={"ace": {1, 2}}, cfg=CFG)
    assert woba(gassed) > woba(rested)


def test_falls_back_to_season_aggregate_when_thin():
    pit = {"only": _good()}
    sentinel = LEAGUE.copy(); sentinel[HR] += 0.01; sentinel /= sentinel.sum()
    ps = FakePS(pit, sentinel)
    # Fewer than min_relievers known -> season aggregate (the sentinel) is returned.
    vec = team_bullpen_vector(ps, "AAA", ["only", "unknown1", "unknown2"], cfg=CFG)
    assert np.allclose(vec, sentinel)


def test_live_bullpen_vectors_maps_each_team():
    pit = {f"r{i}": _good() for i in range(4)}
    ps = FakePS(pit, LEAGUE)
    usage = {"AAA": {"relievers": list(pit), "appearances": {}},
             "BBB": {"relievers": ["r0"], "appearances": {}}}   # thin -> aggregate
    out = live_bullpen_vectors(ps, usage, cfg=CFG)
    assert set(out) == {"AAA", "BBB"}
    assert woba(out["AAA"]) < woba(LEAGUE)          # real blend
    assert np.allclose(out["BBB"], LEAGUE)          # fell back


# ----------------------------- live parsing --------------------------- #
def test_parse_roster_pitchers(monkeypatch):
    monkeypatch.setattr(ld, "retro_for_mlbam",
                        lambda i: {11: "aaaa001", 12: "bbbb001"}.get(int(i)) if i else None)
    payload = {"roster": [
        {"person": {"id": 11}, "position": {"abbreviation": "P", "type": "Pitcher"}},
        {"person": {"id": 12}, "position": {"abbreviation": "P", "type": "Pitcher"}},
        {"person": {"id": 99}, "position": {"abbreviation": "SS", "type": "Infielder"}},
    ]}
    assert parse_roster_pitchers(payload) == ["aaaa001", "bbbb001"]


def test_parse_boxscore_pitchers(monkeypatch):
    monkeypatch.setattr(ld, "retro_for_mlbam",
                        lambda i: {21: "strt001", 22: "relf001"}.get(int(i)) if i else None)
    payload = {"teams": {"home": {
        "team": {"id": 147},                          # NYA -> NYA
        "pitchers": [21, 22],
        "players": {
            "ID21": {"person": {"id": 21}, "stats": {"pitching": {"gamesStarted": 1}}},
            "ID22": {"person": {"id": 22}, "stats": {"pitching": {"gamesStarted": 0}}},
        }}, "away": {}}}
    got = parse_boxscore_pitchers(payload)
    assert ("NYA", "strt001", True) in got and ("NYA", "relf001", False) in got


def test_fetch_bullpen_usage_assembles(monkeypatch):
    # Stub the three statsapi calls; assert relievers + fatigue days assemble correctly.
    monkeypatch.setattr(ld, "fetch_pitcher_roles", lambda season: {"sp1": 0.9, "rp1": 0.0, "rp2": 0.0})
    monkeypatch.setattr(ld, "fetch_recent_appearances", lambda date, lb: {"rp1": {1}})
    monkeypatch.setattr(ld, "fetch_team_relievers",
                        lambda tc, date, roles, gs: ["rp1", "rp2"] if tc == "AAA" else [])
    usage = fetch_bullpen_usage("2025-08-01", ["AAA", "BBB"], cfg=CFG)
    assert usage["AAA"]["relievers"] == ["rp1", "rp2"]
    assert usage["AAA"]["appearances"]["rp1"] == {1}
    assert usage["AAA"]["appearances"]["rp2"] == set()        # no recent outing
    assert usage["BBB"]["relievers"] == []
