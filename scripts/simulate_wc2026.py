"""Simulate the remaining 2026 World Cup and write the dashboard data file.

Loads the trained model, reconstructs the live tournament state, Monte-Carlo
simulates the rest of the bracket, and writes everything the dashboard needs to
``outputs/predictions_2026.json``.

Run (after ``scripts/train.py``):  ``python scripts/simulate_wc2026.py``
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcpredictor.paths import METRICS_PATH, PREDICTIONS_2026_PATH  # noqa: E402
from wcpredictor.predict import Predictor                          # noqa: E402
from wcpredictor.tournament import TournamentSimulator             # noqa: E402


def main() -> None:
    t0 = time.time()
    predictor = Predictor.load()
    sim = TournamentSimulator(predictor)
    state = sim.state
    print(f"Alive: {len(state['alive'])} teams — next round: {state['next_round']}")
    print("  " + ", ".join(state["alive"]))

    odds = sim.run()
    sizes = odds.attrs["sizes"]
    round_names = odds.attrs["round_names"]
    print(f"\nTitle odds (from {odds.attrs['n_sims']:,} simulations):")
    for r in odds.itertuples():
        print(f"  {r.team:<16} champion {r.champion*100:5.1f}%")

    # Current-round fixtures (adjacent bracket pairs) with full predictions.
    bracket = sim.bracket
    fixtures = []
    for i in range(0, len(bracket) - 1, 2):
        h, a = bracket[i], bracket[i + 1]
        fixtures.append({"home": h, "away": a,
                         "prediction": predictor.predict(h, a, neutral=True).to_dict()})

    metrics = json.load(open(METRICS_PATH)) if METRICS_PATH.exists() else {}
    payload = {
        "generated": str(date.today()),
        "trained_through": predictor.trained_through,
        "n_sims": odds.attrs["n_sims"],
        "state": state,
        "bracket": bracket,
        "round_names": {str(k): v for k, v in round_names.items()},
        "sizes": sizes,
        "title_odds": json.loads(odds.to_json(orient="records")),
        "current_round": state["next_round"],
        "current_round_fixtures": fixtures,
        "weights": metrics.get("weights", {}),
        "backtest": metrics.get("backtest", {}),
        "world_cup_backtest": metrics.get("world_cup", {}),
    }
    with open(PREDICTIONS_2026_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {PREDICTIONS_2026_PATH.name}. Total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
