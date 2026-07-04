"""Train, backtest, and persist the World Cup prediction model.

Pipeline
--------
1. Load + clean the full international match history and build as-of features.
2. **Backtest** with a temporal split (train < 2016, validate 2016-17, test 2018+
   which contains the 2018 & 2022 World Cups). Every component model and the
   ensemble is scored on the untouched test set; the ensemble is expected to win.
3. **Fit the deployed model** on *all* data (so ratings are current through the
   ongoing 2026 tournament) and pickle it to ``outputs/model.joblib``.
4. Write ``outputs/metrics.json`` and ``outputs/calibration.png``.

Run:  ``python scripts/train.py``
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcpredictor.backtest import evaluate, reliability_data          # noqa: E402
from wcpredictor.data import load_matches                            # noqa: E402
from wcpredictor.features import FeatureBuilder                      # noqa: E402
from wcpredictor.models.ensemble import EnsembleModel                # noqa: E402
from wcpredictor.paths import CALIBRATION_PLOT_PATH, METRICS_PATH    # noqa: E402
from wcpredictor.predict import Predictor, build_members, fit_members  # noqa: E402
from wcpredictor.viz import save_calibration_plot                    # noqa: E402

CORE_END = pd.Timestamp("2016-01-01")
VAL_END = pd.Timestamp("2018-01-01")
DEPLOY_BLEND_FROM = pd.Timestamp("2023-01-01")
PRETTY = {"dixon_coles": "Dixon-Coles", "gboost": "Gradient Boosting",
          "elo_logistic": "Elo (logistic)", "base_rate": "Base rate", "ensemble": "ENSEMBLE"}


def _table(rows: dict) -> str:
    hdr = f"{'model':<18}{'RPS':>8}{'log-loss':>10}{'Brier':>8}{'acc':>8}{'exact':>8}{'n':>8}"
    lines = [hdr, "-" * len(hdr)]
    for name, m in rows.items():
        lines.append(f"{PRETTY.get(name, name):<18}{m['rps']:>8.4f}{m['log_loss']:>10.4f}"
                     f"{m['brier']:>8.4f}{m['accuracy']:>8.3f}{m['exact_score']:>8.3f}{m['n']:>8}")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    print("Loading data and building features ...")
    matches = load_matches()
    fb = FeatureBuilder()
    feat = fb.fit_transform(matches)
    print(f"  {len(feat):,} matches, through {feat['date'].max().date()}  ({time.time()-t0:.1f}s)")

    core = feat[feat["date"] < CORE_END]
    val = feat[(feat["date"] >= CORE_END) & (feat["date"] < VAL_END)]
    test = feat[feat["date"] >= VAL_END]

    # ---------------- Backtest ---------------- #
    print(f"\nBacktest: train<{CORE_END.date()}  val<{VAL_END.date()}  "
          f"test>= {VAL_END.date()} ({len(test):,} matches) ...")
    members_bt = fit_members(build_members(), core)
    ens_bt = EnsembleModel(members_bt).fit_blend(val)

    rows = {}
    for name, model in members_bt.items():
        rows[name] = evaluate(model.predict_frame(test), test)
    ens_frame = ens_bt.predict_frame(test)
    rows["ensemble"] = evaluate(ens_frame, test)
    print("\n" + _table(rows))
    print(f"\nEnsemble weights: "
          + ", ".join(f"{PRETTY.get(k,k)} {v:.2f}" for k, v in ens_bt.weight_map.items())
          + f"  | temperature {ens_bt.temperature:.2f}")

    # World-Cup-only slice of the test period (2018 & 2022 finals).
    test_wc = test[test["tournament"] == "FIFA World Cup"]
    wc_metrics = evaluate(ens_bt.predict_frame(test_wc), test_wc) if len(test_wc) else {}
    if wc_metrics:
        print(f"\nEnsemble on World Cup matches only (2018 & 2022, n={wc_metrics['n']}): "
              f"RPS {wc_metrics['rps']:.4f}, log-loss {wc_metrics['log_loss']:.4f}, "
              f"accuracy {wc_metrics['accuracy']:.3f}")

    calib = reliability_data(ens_frame, test)
    save_calibration_plot(calib, CALIBRATION_PLOT_PATH)

    # ---------------- Deployed model (all data) ---------------- #
    print("\nFitting deployed model on all data ...")
    members = fit_members(build_members(), feat)
    ens = EnsembleModel(members).fit_blend(feat[feat["date"] >= DEPLOY_BLEND_FROM])
    predictor = Predictor(fb, ens,
                          metrics={"backtest": rows, "world_cup": wc_metrics,
                                   "weights": ens.weight_map, "temperature": ens.temperature,
                                   "calibration": calib},
                          trained_through=str(feat["date"].max().date()))
    predictor.save()

    with open(METRICS_PATH, "w") as f:
        json.dump(predictor.metrics, f, indent=2)
    print(f"Saved model, {METRICS_PATH.name}, {CALIBRATION_PLOT_PATH.name}.  "
          f"Total {time.time()-t0:.1f}s")

    # Sanity check.
    print("\nSanity predictions:")
    for h, a in [("Spain", "France"), ("Brazil", "Argentina"), ("France", "Canada")]:
        print("  " + predictor.predict(h, a, neutral=True).summary())


if __name__ == "__main__":
    main()
