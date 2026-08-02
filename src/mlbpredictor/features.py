"""As-of, leak-free game features tying the pieces together.

For every game we compute, using only information available that morning:

* **Elo** and Elo difference (pre-game, sequential);
* each side's **expected runs** (lineup vs opposing starter+bullpen, park-adjusted)
  from season projections fit on *earlier* seasons only;
* **starter quality** (projected wOBA allowed), **park factor**, and **rest days**.

Projections are keyed by season and fit as-of that season, so a game in year *Y*
is scored with projections built from years ``< Y`` — no look-ahead. The same code
builds a single feature row for a hypothetical/today's game from the deployed state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import statcast
from .offense import OffenseModel, woba
from .park import ParkFactors
from .projections import ProjectionSystem
from .ratings import EloRatingSystem

FEATURES = [
    "elo_diff", "exp_home_runs", "exp_away_runs", "exp_run_diff",
    "home_sp_quality", "away_sp_quality", "park_factor",
    "home_rest", "away_rest",
]
DEFAULT_REST = 3.0
MAX_REST = 10.0


def _add_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``home_rest``/``away_rest`` = days since each team's previous game."""
    long = pd.concat([
        pd.DataFrame({"mid": np.arange(len(df)), "team": df["home_team"].values,
                      "date": df["date"].values, "side": "home"}),
        pd.DataFrame({"mid": np.arange(len(df)), "team": df["away_team"].values,
                      "date": df["date"].values, "side": "away"}),
    ], ignore_index=True).sort_values(["team", "date"], kind="mergesort")
    long["rest"] = long.groupby("team")["date"].diff().dt.days
    long["rest"] = long["rest"].fillna(DEFAULT_REST).clip(1, MAX_REST)
    hp = long[long.side == "home"].set_index("mid")["rest"]
    ap = long[long.side == "away"].set_index("mid")["rest"]
    df = df.copy()
    df["home_rest"] = hp.reindex(range(len(df))).values
    df["away_rest"] = ap.reindex(range(len(df))).values
    return df


class GameFeatureBuilder:
    def __init__(self, sp_share: float = 0.62):
        self.sp_share = sp_share
        self.elo = EloRatingSystem()
        self.park = ParkFactors()
        self.proj_by_season: dict[int, ProjectionSystem] = {}
        self.offense_by_season: dict[int, OffenseModel] = {}
        self.deploy_season = 0
        self.deploy_proj: ProjectionSystem | None = None
        self.deploy_offense: OffenseModel | None = None
        self.default_lineups: dict[str, list] = {}    # team -> most recent 9-man order
        self.default_starter: dict[str, str] = {}      # team -> most recent starter
        self.default_park: dict[str, str] = {}         # team -> most recent home park

    # ------------------------------------------------------------------ #
    def _expected(self, row, ps: ProjectionSystem, off: OffenseModel):
        pf = self.park.factor(row.park)
        eh = off.expected_runs_ids(ps, list(row.home_lineup), row.away_sp, row.away_team, pf)
        ea = off.expected_runs_ids(ps, list(row.away_lineup), row.home_sp, row.home_team, pf)
        return eh, ea, woba(ps.pitcher(row.home_sp)), woba(ps.pitcher(row.away_sp)), pf

    def _build_season_models(self, batting, pitching, bullpen, lg_rpg_by_season, seasons):
        for Y in seasons:
            lb, lp = statcast.multipliers_for_season(int(Y) - 1)   # {} unless Savant enabled
            try:
                ps = ProjectionSystem().fit(batting, pitching, bullpen, ref_season=int(Y),
                                            luck_bat=lb, luck_pit=lp)
            except ValueError:
                continue
            self.proj_by_season[int(Y)] = ps
            self.offense_by_season[int(Y)] = OffenseModel(
                ps.league_bat, float(lg_rpg_by_season.get(Y, 4.5)), self.sp_share)

    # ------------------------------------------------------------------ #
    def fit_transform(self, game_logs: pd.DataFrame, rate_aggs: dict) -> pd.DataFrame:
        df = self.elo.fit_transform(game_logs)          # + home_elo/away_elo/elo_diff
        self.park.fit(game_logs)
        df = _add_rest(df)

        batting, pitching, bullpen = rate_aggs["batting"], rate_aggs["pitching"], rate_aggs["bullpen"]
        lg_rpg = ((df.groupby("season")["home_score"].mean()
                   + df.groupby("season")["away_score"].mean()) / 2).to_dict()
        seasons = sorted(df["season"].unique())
        self._build_season_models(batting, pitching, bullpen, lg_rpg, seasons)

        eh = np.full(len(df), np.nan); ea = np.full(len(df), np.nan)
        hq = np.full(len(df), np.nan); aq = np.full(len(df), np.nan)
        pf = np.ones(len(df))
        for i, row in enumerate(df.itertuples(index=False)):
            Y = int(row.season)
            ps = self.proj_by_season.get(Y)
            if ps is None:
                continue
            eh[i], ea[i], hq[i], aq[i], pf[i] = self._expected(row, ps, self.offense_by_season[Y])

        df["exp_home_runs"] = eh
        df["exp_away_runs"] = ea
        df["exp_run_diff"] = eh - ea
        df["home_sp_quality"] = hq
        df["away_sp_quality"] = aq
        df["park_factor"] = pf
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["total"] = df["home_score"] + df["away_score"]
        df = df.dropna(subset=["exp_home_runs"]).reset_index(drop=True)

        # Deployed (as-of next season) projection for live inference.
        self.deploy_season = int(max(seasons)) + 1
        lb, lp = statcast.multipliers_for_season(self.deploy_season - 1)
        try:
            self.deploy_proj = ProjectionSystem().fit(batting, pitching, bullpen,
                                                      ref_season=self.deploy_season,
                                                      luck_bat=lb, luck_pit=lp)
        except ValueError:
            self.deploy_proj = self.proj_by_season[max(self.proj_by_season)]
        recent_rpg = float(np.mean([lg_rpg[s] for s in seasons[-2:]]))
        self.deploy_offense = OffenseModel(self.deploy_proj.league_bat, recent_rpg, self.sp_share)
        self._store_default_lineups(game_logs)
        return df

    def _store_default_lineups(self, game_logs: pd.DataFrame) -> None:
        """Each team's most recent starting order + starter (fallback for live slates)."""
        rows = []
        for lin_col, sp_col, team_col in (("home_lineup", "home_sp", "home_team"),
                                          ("away_lineup", "away_sp", "away_team")):
            part = game_logs[["date", team_col, lin_col, sp_col]].rename(
                columns={team_col: "team", lin_col: "lineup", sp_col: "sp"})
            rows.append(part)
        allg = pd.concat(rows, ignore_index=True).sort_values("date")
        for team, sub in allg.groupby("team"):
            last = sub.iloc[-1]
            self.default_lineups[team] = list(last["lineup"])
            self.default_starter[team] = last["sp"]
        # Most recent home park per team.
        for team, sub in game_logs.sort_values("date").groupby("home_team"):
            self.default_park[team] = sub.iloc[-1]["park"]

    # ------------------------------------------------------------------ #
    def slim(self) -> "GameFeatureBuilder":
        """Drop the per-season projection state used only during backtesting.

        Inference needs only the deployed projection/offense, Elo and park factors,
        so clearing these keeps the pickled model small.
        """
        self.proj_by_season = {}
        self.offense_by_season = {}
        return self

    def teams(self) -> list[str]:
        return sorted(self.elo.ratings_)

    def match_features(self, home_team, away_team, home_sp, away_sp,
                       home_lineup, away_lineup, park=None,
                       date: pd.Timestamp | None = None) -> pd.DataFrame:
        """Feature row for one hypothetical/today's game, from deployed state."""
        ps, off = self.deploy_proj, self.deploy_offense
        pf = self.park.factor(park)
        eh = off.expected_runs_ids(ps, list(home_lineup), away_sp, away_team, pf)
        ea = off.expected_runs_ids(ps, list(away_lineup), home_sp, home_team, pf)
        row = {
            "elo_diff": self.elo.rating(home_team) - self.elo.rating(away_team),
            "exp_home_runs": eh, "exp_away_runs": ea, "exp_run_diff": eh - ea,
            "home_sp_quality": woba(ps.pitcher(home_sp)), "away_sp_quality": woba(ps.pitcher(away_sp)),
            "park_factor": pf, "home_rest": DEFAULT_REST, "away_rest": DEFAULT_REST,
        }
        return pd.DataFrame([row], columns=FEATURES)
