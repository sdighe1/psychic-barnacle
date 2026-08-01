"""Backtest the model against the closing line (CLV).

Joins the model's out-of-sample test-season predictions (written by
``scripts/train_mlb.py`` to ``outputs/backtest_predictions.csv``) with a historical
odds file, and reports how the model stacks up against the market's closing line:
sharpness (log-loss vs. the no-vig close), favorite agreement, +EV bet ROI, and —
when opening lines are present — beat-the-close CLV.

Odds CSV schema (header row; teams as abbreviations or Retrosheet codes; American
odds; date ISO or YYYYMMDD)::

    date,home_team,away_team,close_home_ml,close_away_ml[,open_home_ml,open_away_ml]
    2025-04-04,LAD,SD,-145,+122,-130,+114

Run:  ``python scripts/backtest_clv.py --odds config/odds.example.csv``

Historical odds are not reachable from this sandbox (odds hosts are blocked), so
supply a CSV (e.g. a SportsbookReviewsOnline export) or run where the internet /
an odds API is available. ``config/odds.example.csv`` is a tiny synthetic sample
that only demonstrates the pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbpredictor.clv import clv_report, format_report, normalize_odds_frame  # noqa: E402
from mlbpredictor.paths import BACKTEST_PREDICTIONS_PATH, CLV_REPORT_PATH, CONFIG_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the model vs. the closing line (CLV).")
    ap.add_argument("--odds", default=str(CONFIG_DIR / "odds.example.csv"),
                    help="historical odds CSV (see the schema in this script's docstring)")
    ap.add_argument("--predictions", default=str(BACKTEST_PREDICTIONS_PATH))
    ap.add_argument("--edge", type=float, default=0.02, help="min model edge vs no-vig close to bet")
    args = ap.parse_args()

    if not Path(args.predictions).exists():
        raise SystemExit(f"No predictions at {args.predictions}. Run `python scripts/train_mlb.py` first.")
    preds = pd.read_csv(args.predictions, dtype={"home_team": str, "away_team": str})
    preds["date"] = pd.to_datetime(preds["date"]).dt.strftime("%Y-%m-%d")

    if not Path(args.odds).exists():
        raise SystemExit(f"Odds file not found: {args.odds}")
    odds = normalize_odds_frame(pd.read_csv(args.odds, dtype=str))

    merged = preds.merge(odds, on=["date", "home_team", "away_team"], how="inner")
    merged = merged.drop_duplicates(subset=["date", "home_team", "away_team"])
    print(f"Model predictions: {len(preds):,} games · odds rows: {len(odds):,} · "
          f"matched: {len(merged):,}")
    if merged.empty:
        raise SystemExit(
            "No games matched. Check the odds file covers the model's test season and uses "
            "matching team codes/dates. (Model test season is written by train_mlb.py.)")

    rep = clv_report(merged, edge_threshold=args.edge)
    print("\n" + format_report(rep))
    with open(CLV_REPORT_PATH, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nWrote {CLV_REPORT_PATH.name}.")
    print("\nNote: beating the closing line is very hard; a market-sharper log-loss or a "
          "negative ROI is the expected, honest result for most public models.")


if __name__ == "__main__":
    main()
