"""Render the self-contained web app (Artifact) from committed projections.

    python scripts/build_web_app.py

Reads ``outputs/projections.csv`` + ``outputs/meta.json``, computes each player's
fantasy points for all three ESPN formats with :mod:`ffauction.scoring` (so the
web numbers match the Python app exactly), embeds a compact JSON payload into
``web/auction_app.template.html`` and writes ``web/auction_app.html`` -- a single
self-contained page that runs the whole auction tool client-side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ffauction import scoring  # noqa: E402
from ffauction.paths import META_PATH, PROJECTIONS_PATH  # noqa: E402

TEMPLATE = REPO / "web" / "auction_app.template.html"
OUTPUT = REPO / "web" / "auction_app.html"
PLACEHOLDER = "__EMBEDDED_JSON__"
FMT_KEY = {"standard": "std", "half_ppr": "half", "ppr": "ppr"}


def build_payload() -> dict:
    proj = pd.read_csv(PROJECTIONS_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    points = {k: scoring.project_points(proj, fmt) for fmt, k in FMT_KEY.items()}

    players = []
    for i, r in proj.iterrows():
        players.append({
            "id": str(r["player_id"]),
            "player": r["player"],
            "pos": r["position"],
            "team": (str(r["team"]) if pd.notna(r["team"]) else ""),
            "ecr": (round(float(r["ecr"]), 2) if pd.notna(r.get("ecr")) else None),
            "std": round(float(points["std"].iloc[i]), 1),
            "half": round(float(points["half"].iloc[i]), 1),
            "ppr": round(float(points["ppr"].iloc[i]), 1),
        })
    keep = ("ranking_source", "fp_scrape_date", "target_season", "data_through_season", "n_players")
    return {"players": players, "meta": {k: meta.get(k) for k in keep}}


def main() -> int:
    payload = build_payload()
    template = TEMPLATE.read_text()
    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER} not found in {TEMPLATE}")
    html = template.replace(PLACEHOLDER, json.dumps(payload, separators=(",", ":")))
    OUTPUT.write_text(html)
    print(f"Wrote {OUTPUT}  ({len(html):,} bytes, {len(payload['players'])} players, "
          f"source={payload['meta'].get('ranking_source')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
