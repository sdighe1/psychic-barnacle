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
from .ids import effective_bats, name_for, throws_of
from .models.ensemble import EnsembleModel
from .paths import MODEL_PATH
from .prediction import GamePrediction
from .simulate import TeamPack, precompute_matchups, simulate_game

_CLIP = 1e-3


def _platoon_pairs(ps, lineup, starter_id, opp_team):
    """Per-batter ``(batter_vec, pitcher_vec)`` pairs vs the starter (platoon-resolved)
    and vs the bullpen (platoon-neutral, since a bullpen is mixed-handed)."""
    pt = throws_of(starter_id)
    sp = [(ps.batter(b, vs=pt), ps.pitcher(starter_id, vs=effective_bats(b, pt))) for b in lineup]
    bp = [(ps.batter(b), ps.bullpen(opp_team)) for b in lineup]
    return sp, bp


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
                     park: str | None = None, date=None, lineup_confirmed: bool = False,
                     n_sims: int | None = None, seed: int | None = None) -> GamePrediction:
        full_cfg = load_config()
        cfg = full_cfg["simulation"]
        levels = tuple(full_cfg["intervals"]["levels"])
        n_sims = n_sims or int(cfg["n_sims"])
        seed = int(cfg["random_seed"]) if seed is None else seed

        feat = self.fb.match_features(home_team, away_team, home_sp, away_sp,
                                      home_lineup, away_lineup, park=park, date=date)
        p_home = float(np.clip(self.ens.predict_p_home(feat)[0], _CLIP, 1 - _CLIP))
        member_probs = {n: float(v[0]) for n, v in self.ens.member_probs_for(feat).items()}

        ps = self.fb.deploy_proj
        pf = self.fb.park.factor(park)
        lg = ps.league_bat
        smax = int(cfg["starter_max_batters"])
        tto = tuple(cfg.get("tto_factors", (1.0,)))

        # away bats vs home pitching; home bats vs away pitching (platoon-resolved)
        a_sp, a_bp = _platoon_pairs(ps, away_lineup, home_sp, home_team)
        h_sp, h_bp = _platoon_pairs(ps, home_lineup, away_sp, away_team)
        away_pack = TeamPack(*precompute_matchups(a_sp, a_bp, lg, pf, tto_factors=tto), smax)
        home_pack = TeamPack(*precompute_matchups(h_sp, h_bp, lg, pf, tto_factors=tto), smax)
        res = simulate_game(away_pack, home_pack, n_sims=n_sims,
                            ghost_runner=bool(cfg["extra_innings_ghost_runner"]), seed=seed)

        weights = _importance_weights(res, p_home)
        roster_coverage = self._roster_coverage(ps, home_lineup, away_lineup, home_sp, away_sp)
        return GamePrediction(
            home_team=home_team, away_team=away_team, p_home=p_home, p_away=1 - p_home,
            sim=res, weights=weights,
            home_batters=[(i, name_for(i)) for i in home_lineup],
            away_batters=[(i, name_for(i)) for i in away_lineup],
            home_sp=(home_sp, name_for(home_sp)), away_sp=(away_sp, name_for(away_sp)),
            park=park, date=str(date) if date is not None else None,
            member_probs=member_probs, lineup_confirmed=bool(lineup_confirmed),
            roster_coverage=roster_coverage, interval_levels=levels,
        )

    @staticmethod
    def _roster_coverage(ps, home_lineup, away_lineup, home_sp, away_sp) -> float:
        """Fraction of the 18 batters + 2 starters that have real projections."""
        known = sum(ps.known_batter(i) for i in list(home_lineup) + list(away_lineup))
        known += ps.known_pitcher(home_sp) + ps.known_pitcher(away_sp)
        total = len(home_lineup) + len(away_lineup) + 2
        return known / total if total else 1.0

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
