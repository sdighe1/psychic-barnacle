"""Build a team's bullpen PA-outcome vector from its *available* relievers.

The season aggregate (:meth:`ProjectionSystem.bullpen`) blends every reliever who
threw for a team all year — including arms since traded, injured or demoted — and
ignores who is actually rested tonight. When a live active roster + recent-usage map
is available (statsapi), we instead blend the **specific** available relievers'
projections, weighting each by projected quality (better arms absorb more of a game's
relief innings) and **down-weighting arms that pitched in the last day or two**
(fatigue). With no roster — offline, or on any failure — we transparently fall back to
the season aggregate, so nothing here can make a prediction fail.

Everything in this module is pure (no I/O); :mod:`mlbpredictor.livedata` fetches the
roster and recent appearances that feed it.
"""
from __future__ import annotations

import numpy as np

from .config import load_config
from .offense import woba
from .retrosheet import PA_OUTCOMES

_N = len(PA_OUTCOMES)


def _cfg(cfg: dict | None) -> dict:
    return cfg if cfg is not None else load_config().get("bullpen", {})


def fatigue_multiplier(days_ago, cfg: dict | None = None) -> float:
    """Availability multiplier for a reliever from the days-ago it recently pitched.

    ``days_ago`` is an iterable of integer day offsets from the game date
    (``1`` = yesterday). Pitching on back-to-back days (1 *and* 2 ago) means the arm
    is likely unavailable today (heaviest penalty); a single recent outing is a lighter
    penalty; three-plus days of rest is full availability (multiplier ``1.0``).
    """
    cfg = _cfg(cfg)
    days = set(days_ago or ())
    d1, d2 = 1 in days, 2 in days
    if d1 and d2:
        return float(cfg.get("b2b_penalty", 0.15))
    if d1:
        return float(cfg.get("day1_penalty", 0.60))
    if d2:
        return float(cfg.get("day2_penalty", 0.90))
    return 1.0


def team_bullpen_vector(ps, team: str, reliever_ids, appearances: dict | None = None,
                        vs: str | None = None, cfg: dict | None = None) -> np.ndarray:
    """Quality- and fatigue-weighted blend of a team's available relievers.

    ``reliever_ids`` — Retrosheet ids of the relievers available today.
    ``appearances`` — ``{reliever_id: iterable_of_days_ago}`` recent outings (fatigue).
    ``vs`` — batter effective hand for a platoon-resolved blend (usually ``None``; a
    bullpen is mixed-handed). Falls back to ``ps.bullpen(team, vs)`` whenever fewer than
    ``bullpen.min_relievers`` of the ids have real projections.
    """
    cfg = _cfg(cfg)
    appearances = appearances or {}
    min_rel = int(cfg.get("min_relievers", 3))
    temp = max(float(cfg.get("quality_temp", 0.06)), 1e-6)

    vecs, logq, fat = [], [], []
    for rid in reliever_ids or []:
        if not ps.known_pitcher(rid):
            continue
        v = ps.pitcher(rid, vs=vs)
        vecs.append(v)
        logq.append(-woba(v) / temp)              # lower wOBA-against ⇒ better ⇒ more usage
        fat.append(fatigue_multiplier(appearances.get(rid, ()), cfg))
    if len(vecs) < min_rel:
        return ps.bullpen(team, vs=vs)

    logq = np.asarray(logq)
    w = np.exp(logq - logq.max()) * np.asarray(fat)     # quality softmax × availability
    s = w.sum()
    if s <= 0:
        return ps.bullpen(team, vs=vs)
    blended = (np.asarray(vecs) * (w / s)[:, None]).sum(axis=0)
    bs = blended.sum()
    return blended / bs if bs > 0 else ps.bullpen(team, vs=vs)


def live_bullpen_vectors(ps, usage: dict, cfg: dict | None = None) -> dict[str, np.ndarray]:
    """Map each team to its specific-reliever bullpen vector from a ``usage`` dict.

    ``usage`` is what :func:`mlbpredictor.livedata.fetch_bullpen_usage` returns —
    ``{team: {"relievers": [...], "appearances": {...}}}``. Teams whose live pen is too
    thin silently fall back to the season aggregate inside :func:`team_bullpen_vector`.
    """
    cfg = _cfg(cfg)
    out = {}
    for team, u in (usage or {}).items():
        out[team] = team_bullpen_vector(ps, team, u.get("relievers", []),
                                        u.get("appearances", {}), cfg=cfg)
    return out
