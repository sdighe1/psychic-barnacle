"""FantasyPros expert-consensus rankings (the best reachable draft signal).

This environment's egress policy blocks live provider sites (fantasypros.com,
ESPN, Sleeper all return 403 policy denials), but the **DynastyProcess/ffverse**
GitHub mirror -- the same source ``nflreadr::load_ff_rankings()`` uses -- is
reachable and current. ``db_fpecr.parquet`` there carries FantasyPros Expert
Consensus Rankings (ECR) refreshed daily through the offseason.

We take the latest scrape's **redraft positional** ranks (``ecr_type == 'rp'``)
for QB/RB/WR/TE/K/DST. These are rankings only (no projected points); the caller
fuses this ordering with realistic point magnitudes (see
:func:`projections.anchor_to_rankings`).
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import pandas as pd

from .league import normalize_position

FP_ECR_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr.parquet"
)
_WANTED_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
_NEEDED_COLUMNS = ["player", "pos", "team", "ecr", "ecr_type", "scrape_date"]


def parse_ecr(raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Reduce the full ECR history to the latest **redraft positional** slice.

    Returns ``(ranks, scrape_date)`` where ``ranks`` has columns
    ``player, position, team, ecr`` (one row per player, best rank kept).
    """
    df = raw[[c for c in _NEEDED_COLUMNS if c in raw.columns]].copy()
    latest = df["scrape_date"].max()
    cur = df[(df["scrape_date"] == latest) & (df["ecr_type"] == "rp")].copy()
    cur["position"] = cur["pos"].map(normalize_position)
    cur = cur[cur["position"].isin(_WANTED_POSITIONS)]
    cur["ecr"] = pd.to_numeric(cur["ecr"], errors="coerce")
    cur = cur.dropna(subset=["ecr"])
    # one row per player/position, keep the strongest (lowest) rank
    cur = cur.sort_values("ecr").drop_duplicates(["player", "position"])
    ranks = cur[["player", "position", "team", "ecr"]].reset_index(drop=True)
    return ranks, str(latest)


def load_fantasypros_ecr(url: str = FP_ECR_URL, timeout: int = 90) -> Tuple[pd.DataFrame, str]:
    """Fetch and parse the current FantasyPros ECR. Raises on any failure so the
    build can fall back to the model-only projection."""
    import requests

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "ffauction/0.1"})
    resp.raise_for_status()
    raw = pd.read_parquet(io.BytesIO(resp.content), columns=_NEEDED_COLUMNS)
    return parse_ecr(raw)


def try_load_fantasypros_ecr(url: str = FP_ECR_URL) -> Optional[Tuple[pd.DataFrame, str]]:
    """Best-effort variant: returns ``None`` instead of raising when the mirror
    is unreachable (blocked host, network error, format change)."""
    try:
        return load_fantasypros_ecr(url)
    except Exception:
        return None
