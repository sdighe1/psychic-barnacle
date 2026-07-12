"""MLB game prediction model (betting-oriented).

A plate-appearance Monte-Carlo simulator, driven by Marcel-style player
projections and an odds-ratio (log5) batter-vs-pitcher matchup engine, wrapped in
a calibrated ensemble with an Elo baseline and a negative-binomial run model.
Produces a game's outcome, score, team box stats and per-player prop lines, plus
fair moneyline / run-line / totals prices.

See ``scripts/train_mlb.py`` (build + backtest) and ``scripts/predict_today.py``
(the morning slate run). Data is sourced offline from the Chadwick Bureau's
Retrosheet mirror; live daily slates come from ``statsapi.mlb.com`` when reachable.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
