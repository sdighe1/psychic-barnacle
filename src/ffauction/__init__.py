"""Fantasy football auction-draft assistant.

A small toolkit that turns NFL projections into auction dollar values and drives
a live draft board: rankings, an *optimal* (model) price, a live *expected*
(market, inflation-adjusted) price, the *max* you can bid, and roster-aware
recommendations. ESPN scoring (Standard / Half-PPR / Full-PPR).
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
