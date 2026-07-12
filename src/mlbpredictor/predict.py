"""High-level prediction API: :class:`Predictor`.

Bundles the fitted feature builder (Elo + deployed projections + park factors), the
calibrated ensemble and the run model. For a game it computes the calibrated
moneyline, runs the plate-appearance simulator from the deployed projections, and
reconciles the two by importance-weighting the simulations to the calibrated win
probability — then returns a :class:`~mlbpredictor.prediction.GamePrediction` with
the score, totals, run line, first-five and per-player props.

Pickles cleanly via joblib to ``outputs/mlb_model.joblib``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .features import GameFeatureBuilder
from .ids import name_for
from .models.ensemble import EnsembleModel
from .paths import MODEL_PATH
from .prediction import GamePrediction
from .simulate import TeamPack, precompute_matchups, simulate_game

_CLIP = 1e-3


def _importance_weights(res, p_home_cal: float) -> np.ndarray:
    """Reweight sims so the (weighted) home-win rate equals ``p_home_cal``."""
    home_win = (res.home_runs > res.away_runs).astype(float)
    ties = res.home_runs == res.away_runs
    home_win[ties] = 0.5
    p_sim = float(np.clip(home_win.mean(), _CLIP, 1 - _CLIP))
    p_cal = float(np.clip(p_home_cal, _CLIP, 1 - _CLIP))
    w = np.where(home_win > 0.5, p_cal / p_sim, (1 - p_cal) / (1 - p_sim))
    w[ties] = 1.0                                   # leave the rare exact ties neutral
    return w * (len(w) / w.sum())


@dataclass
class Predictor:
    fb: GameFeatureBuilder
    ens: EnsembleModel
    metrics: dict = field(default_factory=dict)
    trained_through: str = ""

    # ------------------------------------------------------------------ #
    def predict_game(self, home_team: str, away_team: str, home_sp: str, away_sp: str,
                     home_lineup: list[str], away_lineup: list[str],
                     park: str | None = None, date=None,
                     n_sims: int | None = None, seed: int | None = None) -> GamePrediction:
        cfg = load_config()["simulation"]
        n_sims = n_sims or int(cfg["n_sims"])
        seed = int(cfg["random_seed"]) if seed is None else seed

        feat = self.fb.match_features(home_team, away_team, home_sp, away_sp,
                                      home_lineup, away_lineup, park=park, date=date)
        p_home = float(np.clip(self.ens.predict_p_home(feat)[0], _CLIP, 1 - _CLIP))

        ps = self.fb.deploy_proj
        pf = self.fb.park.factor(park)
        lg = ps.league_bat
        home_vecs = [ps.batter(i) for i in home_lineup]
        away_vecs = [ps.batter(i) for i in away_lineup]
        home_sp_v, away_sp_v = ps.pitcher(home_sp), ps.pitcher(away_sp)
        home_bp, away_bp = ps.bullpen(home_team), ps.bullpen(away_team)
        smax = int(cfg["starter_max_batters"])

        # away bats vs home pitching; home bats vs away pitching
        away_pack = TeamPack(*precompute_matchups(away_vecs, home_sp_v, home_bp, lg, pf), smax)
        home_pack = TeamPack(*precompute_matchups(home_vecs, away_sp_v, away_bp, lg, pf), smax)
        res = simulate_game(away_pack, home_pack, n_sims=n_sims,
                            ghost_runner=bool(cfg["extra_innings_ghost_runner"]), seed=seed)

        weights = _importance_weights(res, p_home)
        return GamePrediction(
            home_team=home_team, away_team=away_team, p_home=p_home, p_away=1 - p_home,
            sim=res, weights=weights,
            home_batters=[(i, name_for(i)) for i in home_lineup],
            away_batters=[(i, name_for(i)) for i in away_lineup],
            home_sp=(home_sp, name_for(home_sp)), away_sp=(away_sp, name_for(away_sp)),
            park=park, date=str(date) if date is not None else None,
        )

    def teams(self) -> list[str]:
        return self.fb.teams()

    def elo_rankings(self, top: int | None = None) -> pd.DataFrame:
        return self.fb.elo.rankings(top)

    # ------------------------------------------------------------------ #
    def save(self, path=MODEL_PATH) -> None:
        self.fb.slim()                              # drop per-season backtest state
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH) -> "Predictor":
        return joblib.load(path)
