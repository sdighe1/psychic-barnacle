"""The morning run: predict every game on today's slate.

Gets the slate from the MLB Stats API when reachable, otherwise from a manual
slate file (``--slate``, default ``config/slate.yaml`` then ``config/slate.example.yaml``).
For each game it runs the simulator and writes ``outputs/predictions_<date>.json``
plus ``outputs/predictions_latest.json``, and prints a readable table.

Run:  ``python scripts/predict_today.py``            (auto-fetch or default slate)
      ``python scripts/predict_today.py --slate config/slate.example.yaml --date 2026-07-12``
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date as date_cls
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbpredictor.livedata import get_slate                          # noqa: E402
from mlbpredictor.paths import (CONFIG_DIR, MODEL_PATH,              # noqa: E402
                                PREDICTIONS_LATEST_PATH, predictions_path)
from mlbpredictor.predict import Predictor                          # noqa: E402


def _default_slate() -> Path | None:
    for name in ("slate.yaml", "slate.example.yaml"):
        p = CONFIG_DIR / name
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict today's MLB slate.")
    ap.add_argument("--date", default=date_cls.today().isoformat(), help="YYYY-MM-DD")
    ap.add_argument("--slate", default=None, help="manual slate YAML (fallback)")
    ap.add_argument("--n-sims", type=int, default=None, help="simulations per game")
    ap.add_argument("--no-live", action="store_true", help="skip the statsapi fetch")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python scripts/train_mlb.py` first.")
    predictor = Predictor.load()

    slate_path = Path(args.slate) if args.slate else _default_slate()
    games = get_slate(args.date, slate_path=slate_path, predictor=predictor,
                      prefer_live=not args.no_live)
    if not games:
        raise SystemExit(
            "No games found. Enable statsapi.mlb.com in the network policy, or fill in "
            f"a slate file (see {CONFIG_DIR / 'slate.example.yaml'}).")

    source = games[0].source
    print(f"Slate for {args.date} — {len(games)} games (source: {source}), "
          f"model trained through {predictor.trained_through}\n")

    predictions = []
    header = f"{'Away':>4} @ {'Home':<4} {'Fav':>4} {'Win%':>6} {'Fair':>7}  {'Proj':>7} {'Total':>6} {'O/U':>5}"
    print(header); print("-" * len(header))
    for g in games:
        if len(g.home_lineup) < 9 or len(g.away_lineup) < 9 or not g.home_sp or not g.away_sp:
            print(f"  (skipping {g.away_team} @ {g.home_team}: incomplete lineup/starter)")
            continue
        pred = predictor.predict_game(
            g.home_team, g.away_team, g.home_sp, g.away_sp,
            g.home_lineup, g.away_lineup, park=g.park, date=args.date,
            n_sims=args.n_sims)
        d = pred.to_dict()
        predictions.append(d)
        a, h = d["score"]["projected_away"], d["score"]["projected_home"]
        fav = d["moneyline"]["favorite"]
        favp = d["moneyline"]["p_home"] if fav == g.home_team else d["moneyline"]["p_away"]
        fair = d["moneyline"]["fair_home"] if fav == g.home_team else d["moneyline"]["fair_away"]
        print(f"{g.away_team:>4} @ {g.home_team:<4} {fav:>4} {favp*100:>5.1f}% {fair:>+7d}  "
              f"{a:>3}-{h:<3} {d['total']['line']:>6.1f} {'O' if d['total']['p_over']>=0.5 else 'U'}"
              f"{max(d['total']['p_over'],d['total']['p_under'])*100:>4.0f}%")

    out = {
        "date": args.date, "generated": date_cls.today().isoformat(),
        "source": source, "trained_through": predictor.trained_through,
        "n_games": len(predictions), "games": predictions,
    }
    for path in (predictions_path(args.date.replace("-", "")), PREDICTIONS_LATEST_PATH):
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nWrote {predictions_path(args.date.replace('-', '')).name} "
          f"and {PREDICTIONS_LATEST_PATH.name} ({len(predictions)} games).")
    print("\nDisclaimer: statistical estimates for research/entertainment — not betting advice.")


if __name__ == "__main__":
    main()
