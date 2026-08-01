"""Closing-line value (CLV) and market-comparison metrics.

The strongest test of a betting model is not raw accuracy but how it stacks up
against the **closing line** — the sharpest price the market offers. This module is
source-agnostic: give it a table of games with the model's win probability, the
actual result, and the closing (and optionally opening) moneylines, and it reports:

* **Sharpness vs. the market** — model log-loss / Brier next to the *no-vig* closing
  line's. If the model is not below the market, it is not beating the close (the
  honest, usually-humbling headline).
* **+EV bet selection & ROI** — flagging games where the model's edge over the
  no-vig close clears a threshold, then the realized hit-rate and return if you had
  bet those at the closing price.
* **Beat-the-close CLV** — when opening lines are supplied, how often and by how much
  the line moved toward the model's side (positive CLV predicts long-term profit even
  before you know outcomes).

All functions are pure (no I/O), so they unit-test against synthetic odds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Odds conversions
# --------------------------------------------------------------------------- #
def american_to_prob(odds) -> float:
    """Implied win probability (with vig) from American odds."""
    o = _parse_american(odds)
    if o is None:
        return float("nan")
    return (-o) / (-o + 100) if o < 0 else 100 / (o + 100)


def american_to_decimal(odds) -> float:
    """Decimal payout multiplier from American odds (e.g. -110 → 1.909)."""
    o = _parse_american(odds)
    if o is None:
        return float("nan")
    return 1 + (100 / (-o)) if o < 0 else 1 + (o / 100)


def _parse_american(odds):
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    if isinstance(odds, str):
        s = odds.strip().upper().replace("+", "")
        if s in ("EVEN", "EV", "PK", "PICK", "100"):
            return 100.0
        try:
            return float(s)
        except ValueError:
            return None
    return float(odds)


def no_vig_two_way(p_home: float, p_away: float) -> tuple[float, float]:
    """Remove the bookmaker margin from a two-way market → fair probabilities."""
    s = p_home + p_away
    if s <= 0:
        return float("nan"), float("nan")
    return p_home / s, p_away / s


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def normalize_odds_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an odds table's join keys: teams → Retrosheet codes, date → ISO."""
    from .livedata import resolve_team
    df = df.copy()
    df["home_team"] = df["home_team"].astype(str).map(resolve_team)
    df["away_team"] = df["away_team"].astype(str).map(resolve_team)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["date", "home_team", "away_team"])


def clv_report(df: pd.DataFrame, edge_threshold: float = 0.02) -> dict:
    """Compute CLV / market-comparison metrics.

    ``df`` needs: ``model_p_home``, ``actual_home_win`` (0/1), ``close_home_ml``,
    ``close_away_ml``. Optionally ``open_home_ml`` / ``open_away_ml`` enable the
    beat-the-close CLV. Bets are flagged where the model's probability on a side
    exceeds the no-vig closing probability by ``edge_threshold``.
    """
    df = df.dropna(subset=["model_p_home", "actual_home_win",
                           "close_home_ml", "close_away_ml"]).copy()
    if df.empty:
        return {"n_games": 0}

    imp_h = df["close_home_ml"].map(american_to_prob).to_numpy()
    imp_a = df["close_away_ml"].map(american_to_prob).to_numpy()
    novig_h = imp_h / (imp_h + imp_a)
    novig_a = 1 - novig_h
    mp_h = df["model_p_home"].to_numpy(dtype=float)
    y = df["actual_home_win"].to_numpy(dtype=float)

    market = {"avg_hold": round(float(np.mean(imp_h + imp_a - 1)), 4),
              "log_loss": round(_log_loss(novig_h, y), 4),
              "brier": round(_brier(novig_h, y), 4)}
    model = {"log_loss": round(_log_loss(mp_h, y), 4),
             "brier": round(_brier(mp_h, y), 4)}
    model["beats_market_logloss"] = bool(model["log_loss"] < market["log_loss"])
    fav_agreement = round(float(np.mean((mp_h > 0.5) == (novig_h > 0.5))), 4)

    # +EV bet selection vs the no-vig close.
    edge_h = mp_h - novig_h
    edge_a = (1 - mp_h) - novig_a
    bet_home = (edge_h >= edge_threshold) & (edge_h >= edge_a)
    bet_away = (edge_a >= edge_threshold) & (edge_a > edge_h)
    dec_h = df["close_home_ml"].map(american_to_decimal).to_numpy()
    dec_a = df["close_away_ml"].map(american_to_decimal).to_numpy()

    profit = np.where(bet_home, np.where(y == 1, dec_h - 1, -1.0), 0.0) \
        + np.where(bet_away, np.where(y == 0, dec_a - 1, -1.0), 0.0)
    n_bets = int(bet_home.sum() + bet_away.sum())
    wins = int((bet_home & (y == 1)).sum() + (bet_away & (y == 0)).sum())
    staked = float(n_bets)
    betting = {
        "edge_threshold": edge_threshold, "n_bets": n_bets,
        "n_home_bets": int(bet_home.sum()), "n_away_bets": int(bet_away.sum()),
        "hit_rate": round(wins / n_bets, 4) if n_bets else None,
        "roi": round(float(profit.sum()) / staked, 4) if n_bets else None,
        "profit_units": round(float(profit.sum()), 2),
        "avg_model_edge": round(float(np.mean(np.maximum(edge_h, edge_a)[bet_home | bet_away])), 4)
        if n_bets else None,
    }

    report = {"n_games": int(len(df)), "market": market, "model": model,
              "favorite_agreement": fav_agreement, "betting": betting}

    # Beat-the-close CLV (requires opening lines).
    if {"open_home_ml", "open_away_ml"}.issubset(df.columns) and n_bets:
        odh = df["open_home_ml"].map(american_to_decimal).to_numpy()
        oda = df["open_away_ml"].map(american_to_decimal).to_numpy()
        # CLV% = price we'd get at open / price at close − 1 (positive = beat the close).
        clv_pct = np.where(bet_home, odh / dec_h - 1.0, 0.0) \
            + np.where(bet_away, oda / dec_a - 1.0, 0.0)
        mask = bet_home | bet_away
        report["clv"] = {
            "beat_close_rate": round(float(np.mean(clv_pct[mask] > 0)), 4),
            "avg_clv_pct": round(float(np.mean(clv_pct[mask])) * 100, 3),
        }
    return report


def format_report(rep: dict) -> str:
    """Human-readable summary of :func:`clv_report`."""
    if not rep.get("n_games"):
        return "No games with odds to evaluate."
    m, mk, b = rep["model"], rep["market"], rep["betting"]
    lines = [
        f"Games with odds: {rep['n_games']}   (avg book hold {mk['avg_hold']:.1%})",
        f"Sharpness (log-loss, lower=sharper):  model {m['log_loss']}  vs  "
        f"market close {mk['log_loss']}  →  "
        + ("MODEL BEATS THE CLOSE" if m["beats_market_logloss"]
           else "market is sharper (expected)"),
        f"Favorite agreement with the close: {rep['favorite_agreement']:.1%}",
        f"+EV plays vs close (edge ≥ {b['edge_threshold']:.0%}): {b['n_bets']} bets "
        f"({b['n_home_bets']} home / {b['n_away_bets']} away)",
    ]
    if b["n_bets"]:
        lines.append(f"  hit rate {b['hit_rate']:.1%}   ROI {b['roi']:+.1%}   "
                     f"({b['profit_units']:+.2f}u on {b['n_bets']}u staked)")
    if "clv" in rep:
        c = rep["clv"]
        lines.append(f"Beat-the-close: {c['beat_close_rate']:.1%} of bets got CLV, "
                     f"avg {c['avg_clv_pct']:+.2f}%")
    return "\n".join(lines)
