"""Build the committed projection artifact from nflverse history.

    python scripts/build_projections.py

Fetches recent seasonal NFL stats, builds a data-driven baseline projection for
the upcoming season, appends the curated K / D-ST baseline, backtests the model
against the most recent completed season, and writes:

    outputs/projections.csv   one projected stat line per player
    outputs/meta.json         data vintage + backtest accuracy (the model card)

Optionally blends in provider projection CSVs (accuracy-weighted) if you pass
``--provider name=path.csv`` one or more times.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ffauction import accuracy, data, projections, providers  # noqa: E402
from ffauction.paths import KDST_BASELINE_PATH, META_PATH, PROJECTIONS_PATH  # noqa: E402
from ffauction.scoring import OVERRIDE_COLUMN, STAT_COLUMNS  # noqa: E402

FULL_COLUMNS = (
    ["player_id", "player", "position", "team", "age", "proj_games"]
    + STAT_COLUMNS + [OVERRIDE_COLUMN, "source"]
)


def load_kdst() -> pd.DataFrame:
    """Curated kicker / defense baseline -> projection schema (points override)."""
    raw = pd.read_csv(KDST_BASELINE_PATH)
    out = pd.DataFrame({
        "player_id": raw["position"].str.upper() + "_" + raw["team"].str.upper(),
        "player": raw["player"], "position": raw["position"], "team": raw["team"],
        "age": np.nan, "proj_games": 17.0,
    })
    for c in STAT_COLUMNS:
        out[c] = 0.0
    out[OVERRIDE_COLUMN] = raw["proj_points"].astype(float)
    out["source"] = "kdst_baseline"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", action="append", default=[], metavar="name=path.csv",
                    help="blend a provider projection CSV (accuracy-weighted); repeatable")
    ap.add_argument("--seasons", type=int, default=3, help="seasons of history for the baseline")
    args = ap.parse_args()

    print("Locating latest available NFL season from nflverse ...")
    latest = data.latest_available_season()
    target = latest + 1
    window = list(range(latest, latest - args.seasons, -1))
    load_years = list(range(latest, latest - args.seasons - 1, -1))  # +1 older for backtest
    print(f"  latest season with data: {latest}  ->  projecting {target}")
    print(f"  loading seasons: {load_years}")

    history = data.load_player_seasons(load_years)
    print(f"  loaded {len(history):,} player-seasons ({history['player_id'].nunique():,} players)")

    print("Building baseline projection ...")
    base = projections.build_projections(history[history["season"].isin(window)], target_season=target)

    sources = {"history": base}
    weights = {"history": 1.0}
    # Optional provider blends.
    for spec in args.provider:
        name, _, path = spec.partition("=")
        prov = providers.normalize_provider_frame(pd.read_csv(path), source=name)
        prov = providers.match_to_baseline(prov, base)
        prov = prov[prov["player_id"].notna() & (prov["player_id"] != "")]
        sources[name] = prov
        print(f"  + provider '{name}': {len(prov)} matched players from {path}")

    if len(sources) > 1:
        print("Backtesting each source for accuracy weights ...")
        mae = {"history": accuracy.backtest_model(history, latest).get("mae", np.nan)}
        for name in sources:
            if name != "history":
                mae[name] = mae["history"]  # providers: default to baseline error absent history
        weights = accuracy.accuracy_weights(mae)
        skill = accuracy.weighted_consensus(sources, weights)
    else:
        skill = base

    full = pd.concat([skill, load_kdst()], ignore_index=True)
    for c in FULL_COLUMNS:
        if c not in full.columns:
            full[c] = np.nan
    full = full[FULL_COLUMNS]
    full.to_csv(PROJECTIONS_PATH, index=False)
    print(f"Wrote {PROJECTIONS_PATH}  ({len(full)} players)")

    print(f"Backtesting model on {latest} (trained on earlier seasons) ...")
    metrics = accuracy.backtest_model(history, latest)

    meta = {
        "generated": date.today().isoformat(),
        "data_through_season": latest,
        "target_season": target,
        "seasons_used": window,
        "sources": list(sources),
        "source_weights": {k: round(v, 3) for k, v in weights.items()},
        "n_players": int(len(full)),
        "accuracy": metrics,
        "scoring_note": "ESPN Standard/Half-PPR/Full-PPR (chosen in-app)",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {META_PATH}")
    if metrics:
        print(f"  backtest {latest}: MAE {metrics['mae']} pts, rank-corr {metrics['rank_corr']} "
              f"(n={metrics['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
