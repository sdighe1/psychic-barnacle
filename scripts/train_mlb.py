"""Train, backtest and persist the MLB prediction model.

Pipeline
--------
1. Load offline data (Retrosheet game logs + event-derived rate aggregates) and
   build as-of, leak-free game features.
2. **Backtest** on a temporal split (train on the earliest season(s), validate on
   the next, test on the most recent). Every component and the ensemble is scored
   on the untouched test season: moneyline log-loss / Brier / accuracy /
   calibration, plus total-runs MAE/RMSE.
3. **Fit the deployed model** on all available data and pickle it to
   ``outputs/mlb_model.joblib``; write ``outputs/mlb_metrics.json`` and
   ``outputs/mlb_calibration.png``.

Run:  ``python scripts/train_mlb.py``
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbpredictor import backtest                                     # noqa: E402
from mlbpredictor.config import load_config                           # noqa: E402
from mlbpredictor.data import load_game_logs, load_rate_aggregates   # noqa: E402
from mlbpredictor.features import GameFeatureBuilder                  # noqa: E402
from mlbpredictor.models.baselines import EloLogistic, HomeBaseRate  # noqa: E402
from mlbpredictor.models.ensemble import EnsembleModel               # noqa: E402
from mlbpredictor.models.gboost import GBoostModel                   # noqa: E402
from mlbpredictor.models.rundist import RunDistModel                 # noqa: E402
from mlbpredictor.paths import (BACKTEST_PREDICTIONS_PATH,           # noqa: E402
                                CALIBRATION_PLOT_PATH, METRICS_PATH)
from mlbpredictor.predict import Predictor                           # noqa: E402
from mlbpredictor.viz import save_calibration_plot                   # noqa: E402

PRETTY = {"rundist": "Run model (NB)", "elo_logistic": "Elo (logistic)",
          "gboost": "Gradient boosting", "base_rate": "Home base rate", "ensemble": "ENSEMBLE"}


def _fit_members(train: pd.DataFrame) -> dict:
    rd = RunDistModel().fit_dispersion(
        np.concatenate([train["home_score"].to_numpy(), train["away_score"].to_numpy()]))
    return {
        "rundist": rd,
        "elo_logistic": EloLogistic().fit(train),
        "gboost": GBoostModel().fit(train),
        "base_rate": HomeBaseRate().fit(train),
    }


def _member_p_home(name: str, model, feat: pd.DataFrame) -> np.ndarray:
    if name == "rundist":
        return model.predict_frame(feat["exp_home_runs"].to_numpy(),
                                   feat["exp_away_runs"].to_numpy())["p_home_win"].to_numpy()
    return model.predict_p_home(feat)


def _ml_table(rows: dict) -> str:
    hdr = f"{'model':<20}{'log-loss':>10}{'Brier':>9}{'accuracy':>10}{'n':>7}"
    out = [hdr, "-" * len(hdr)]
    for name, m in rows.items():
        out.append(f"{PRETTY.get(name, name):<20}{m['log_loss']:>10.4f}"
                   f"{m['brier']:>9.4f}{m['accuracy']:>10.4f}{m['n']:>7}")
    return "\n".join(out)


def main() -> None:
    t0 = time.time()
    print("Loading data and building features ...")
    gl = load_game_logs()
    agg = load_rate_aggregates()
    fb = GameFeatureBuilder()
    feat = fb.fit_transform(gl, agg)
    seasons = sorted(int(s) for s in feat["season"].unique())
    print(f"  {len(feat):,} games with features; seasons {seasons}  ({time.time()-t0:.1f}s)")
    if len(seasons) < 3:
        raise SystemExit("Need at least 3 feature-seasons for a train/val/test backtest.")

    test_s, val_s = seasons[-1], seasons[-2]
    train = feat[feat["season"] <= seasons[-3]]
    if train.empty:                                # exactly 3 seasons
        train = feat[feat["season"] == seasons[0]]
    val = feat[feat["season"] == val_s]
    test = feat[feat["season"] == test_s]
    print(f"\nBacktest split — train {sorted(train.season.unique())} "
          f"({len(train):,}) | val {val_s} ({len(val):,}) | test {test_s} ({len(test):,})")

    members = _fit_members(train)
    ens = EnsembleModel(members["rundist"], members).fit_blend(val)

    y = test["home_win"].to_numpy()
    rows = {}
    for name, model in members.items():
        rows[name] = backtest.evaluate_moneyline(_member_p_home(name, model, test), y)
    ens_p = ens.predict_p_home(test)
    rows["ensemble"] = backtest.evaluate_moneyline(ens_p, y)

    # Persist the out-of-sample test-season predictions for the CLV backtest to join
    # against (scripts/backtest_clv.py).
    pd.DataFrame({
        "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "home_team": test["home_team"].to_numpy(),
        "away_team": test["away_team"].to_numpy(),
        "model_p_home": ens_p,
        "actual_home_win": y,
    }).to_csv(BACKTEST_PREDICTIONS_PATH, index=False)
    print("\nMoneyline (lower log-loss/Brier better; home base rate = no skill):")
    print(_ml_table(rows))

    ens_frame = ens.predict_frame(test)
    totals = backtest.evaluate_totals(ens_frame["exp_total"].to_numpy(), test["total"].to_numpy())
    side = backtest.evaluate_side_runs(
        ens_frame["exp_home_runs"].to_numpy(), test["home_score"].to_numpy(),
        ens_frame["exp_away_runs"].to_numpy(), test["away_score"].to_numpy())
    print(f"\nScore accuracy (test {test_s}):  total runs MAE {totals['runs_mae']} / "
          f"RMSE {totals['runs_rmse']}  (pred {totals['mean_pred']} vs actual {totals['mean_actual']})")
    print(f"  per-team runs MAE {side['team_runs_mae']}")

    # ---- interval calibration + confidence-band reliability ---- #
    levels = load_config()["intervals"]["levels"]
    coverage = backtest.evaluate_total_intervals(
        members["rundist"], test["exp_home_runs"].to_numpy(), test["exp_away_runs"].to_numpy(),
        test["total"].to_numpy(), levels=levels)
    member_spread = np.vstack(list(ens.member_probs_for(test).values())).std(axis=0)
    conf_bands = backtest.accuracy_by_confidence(ens_p, member_spread, y)
    print("\nInterval calibration (should match the nominal level):  "
          + ", ".join(f"{k.replace('coverage_','')}% → {v:.0%}" for k, v in coverage.items()))
    print("Accuracy by confidence band (High should beat Low):  "
          + ", ".join(f"{b} {m['accuracy']:.1%} (n={m['n']})" for b, m in conf_bands.items()))

    print(f"\nEnsemble weights: "
          + ", ".join(f"{PRETTY.get(k,k)} {v:.2f}" for k, v in ens.weight_map.items())
          + f"  | temperature {ens.temperature:.2f}")

    calib = backtest.reliability(ens_p, y)
    save_calibration_plot(calib, CALIBRATION_PLOT_PATH)

    # ---------------- Deployed model (all data) ---------------- #
    print("\nFitting deployed model on all data ...")
    members_all = _fit_members(feat)
    ens_all = EnsembleModel(members_all["rundist"], members_all).fit_blend(feat)
    predictor = Predictor(
        fb, ens_all,
        metrics={
            "backtest": {"split": {"train": sorted(int(s) for s in train.season.unique()),
                                   "val": val_s, "test": test_s},
                         "moneyline": rows, "totals": totals, "team_runs": side,
                         "interval_coverage": coverage, "confidence_bands": conf_bands},
            "calibration": calib,
            "weights": ens.weight_map, "temperature": ens.temperature,
            "deploy_weights": ens_all.weight_map,
        },
        trained_through=str(feat["date"].max().date()),
    )
    predictor.save()
    with open(METRICS_PATH, "w") as f:
        json.dump(predictor.metrics, f, indent=2)
    print(f"Saved model + {METRICS_PATH.name} + {CALIBRATION_PLOT_PATH.name}. "
          f"Total {time.time()-t0:.1f}s")

    # ---------------- Sanity predictions ---------------- #
    print("\nSanity predictions (from real recent lineups):")
    recent = feat[feat["season"] == test_s].tail(3)
    for r in recent.itertuples(index=False):
        pred = predictor.predict_game(
            r.home_team, r.away_team, r.home_sp, r.away_sp,
            list(r.home_lineup), list(r.away_lineup), park=r.park, n_sims=3000)
        print("  " + pred.summary())


if __name__ == "__main__":
    main()
