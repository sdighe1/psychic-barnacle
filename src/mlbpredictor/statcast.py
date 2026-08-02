"""Statcast expected-stats ("de-luck") adjustment for projections — optional.

Baseball Savant publishes *expected* statistics (xwOBA, xBA, xSLG) derived from the
launch angle / exit velocity of every batted ball. Because they price contact quality
rather than where the ball happened to land, they strip much of the BABIP luck that
contaminates a single season's actual results. We use the gap between a player's actual
and expected wOBA as a small, regressed **luck correction** on their Marcel projection:
a hitter whose xwOBA far exceeds his wOBA was unlucky and is nudged up; a pitcher who
allowed a lower xwOBA than his actual wOBA was unlucky and is nudged toward stinginess.

This needs ``baseballsavant.mlb.com`` reachable, so it is **off by default**
(``statcast.enabled``) and fails soft: when disabled, unreachable, or unparseable,
:func:`multipliers_for_season` returns empty maps and projections are untouched.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .config import load_config
from .ids import retro_for_mlbam
from .net import cached_text
from .offense import WOBA_WEIGHTS

# Only batted-ball hit outcomes carry BABIP/HR luck; K/BB/HBP are held as actuals
# (exactly what xwOBA does), so the correction rescales 1B/2B/3B/HR and absorbs the
# difference in OUT.
_HIT = slice(0, 4)      # 1B, 2B, 3B, HR
_BBHBP = slice(4, 6)    # BB, HBP


def _sc_cfg() -> dict:
    return load_config().get("statcast", {})


def statcast_url(kind: str, year: int) -> str:
    base = _sc_cfg().get("base_url",
                         "https://baseballsavant.mlb.com/leaderboard/expected_statistics")
    return f"{base}?type={kind}&year={year}&position=&team=&min=q&csv=true"


def _pick(cols: list[str], *cands: str) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for cand in cands:
        if cand in low:
            return low[cand]
    return None


def load_expected(kind: str, year: int, refresh: bool = False) -> pd.DataFrame | None:
    """Fetch a Savant expected-stats CSV → ``retro_id, pa, woba, est_woba`` (or ``None``).

    Tolerant of column order/naming; maps Savant ``player_id`` (MLBAM) to Retrosheet ids.
    """
    text = cached_text(statcast_url(kind, year), f"statcast_{kind}_{year}.csv", refresh=refresh)
    if not text:
        return None
    try:
        raw = pd.read_csv(io.StringIO(text))
    except (ValueError, pd.errors.ParserError):
        return None
    cols = list(raw.columns)
    c_id = _pick(cols, "player_id", "mlbam_id", "mlb_id")
    c_pa = _pick(cols, "pa", "plate_appearances")
    c_woba = _pick(cols, "woba")
    c_est = _pick(cols, "est_woba", "xwoba", "expected_woba")
    if not all((c_id, c_pa, c_woba, c_est)):
        return None
    df = raw[[c_id, c_pa, c_woba, c_est]].copy()
    df.columns = ["mlbam", "pa", "woba", "est_woba"]
    df["retro_id"] = df["mlbam"].map(lambda i: retro_for_mlbam(i) if pd.notna(i) else None)
    df = df.dropna(subset=["retro_id"])
    for c in ("pa", "woba", "est_woba"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["pa", "woba", "est_woba"])
    return df[["retro_id", "pa", "woba", "est_woba"]].reset_index(drop=True)


def luck_multipliers(df: pd.DataFrame, regress_pa: float, clip: float) -> dict[str, float]:
    """``{retro_id: wOBA multiplier}`` from actual→expected, regressed by PA and clipped.

    The luck *delta* (est−actual) is shrunk toward zero at low PA, then expressed as a
    proportional multiplier on wOBA and capped at ``±clip``.
    """
    out: dict[str, float] = {}
    for r in df.itertuples(index=False):
        woba = float(r.woba)
        if woba <= 1e-6 or float(r.pa) <= 0:
            continue
        delta = (float(r.est_woba) - woba) * (float(r.pa) / (float(r.pa) + regress_pa))
        mult = 1.0 + delta / woba
        out[r.retro_id] = float(np.clip(mult, 1.0 - clip, 1.0 + clip))
    return out


def multipliers_for_season(year: int) -> tuple[dict[str, float], dict[str, float]]:
    """``(batter_mults, pitcher_mults)`` for ``year`` — ``({}, {})`` when off/unavailable."""
    cfg = _sc_cfg()
    if not cfg.get("enabled", False):
        return {}, {}
    reg = float(cfg.get("regress_pa", 200.0))
    clip = float(cfg.get("clip", 0.15))
    try:
        bat_df = load_expected("batter", year)
        pit_df = load_expected("pitcher", year)
    except Exception:                       # never let a data hiccup break training
        return {}, {}
    bat = luck_multipliers(bat_df, reg, clip) if bat_df is not None else {}
    pit = luck_multipliers(pit_df, reg, clip) if pit_df is not None else {}
    return bat, pit


def apply_luck(vec: np.ndarray, mult: float | None) -> np.ndarray:
    """Rescale a PA-outcome vector so its wOBA moves by ``mult`` (hits only, OUT absorbs).

    Returns ``vec`` unchanged for ``None``/≈1.0 multipliers. Works for both batters
    (scale hits up when unlucky) and pitchers-allowed (scale down when unlucky).
    """
    if mult is None or abs(mult - 1.0) < 1e-6:
        return vec
    w = WOBA_WEIGHTS
    hit_c = float(np.dot(w[_HIT], vec[_HIT]))
    oth_c = float(np.dot(w[_BBHBP], vec[_BBHBP]))
    woba_old = hit_c + oth_c
    if hit_c <= 1e-9 or woba_old <= 1e-9:
        return vec
    f = float(np.clip((mult * woba_old - oth_c) / hit_c, 0.5, 1.8))
    adj = vec.copy()
    adj[_HIT] *= f
    adj[7] = max(1e-6, 1.0 - adj[0:7].sum())        # OUT absorbs the change
    s = adj.sum()
    return adj / s if s > 0 else vec
