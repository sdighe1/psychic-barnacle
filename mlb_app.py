"""MLB game prediction dashboard (Streamlit).

    streamlit run mlb_app.py

Shows the day's predicted slate (``outputs/predictions_latest.json``), an interactive
game explorer backed by the trained model (``outputs/mlb_model.joblib``), and a model
card with the backtest metrics (``outputs/mlb_metrics.json``). Build the artifacts
with ``python scripts/train_mlb.py`` and refresh the slate with
``python scripts/predict_today.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mlbpredictor.ids import name_for  # noqa: E402
from mlbpredictor.paths import (CALIBRATION_PLOT_PATH, METRICS_PATH,  # noqa: E402
                                MODEL_PATH, PREDICTIONS_LATEST_PATH)
from mlbpredictor.predict import Predictor  # noqa: E402

st.set_page_config(page_title="MLB Predictor", page_icon="⚾", layout="wide")


@st.cache_resource
def get_predictor():
    if not MODEL_PATH.exists():
        return None
    try:
        return Predictor.load()
    except Exception as exc:
        st.warning(f"Could not load the model ({exc}). Run `python scripts/train_mlb.py`.")
        return None


@st.cache_data
def get_slate() -> dict | None:
    if PREDICTIONS_LATEST_PATH.exists():
        return json.load(open(PREDICTIONS_LATEST_PATH))
    return None


@st.cache_data
def get_metrics() -> dict | None:
    if METRICS_PATH.exists():
        return json.load(open(METRICS_PATH))
    return None


def _pct(x) -> str:
    return f"{x*100:.1f}%"


def _odds(x) -> str:
    return f"{x:+d}"


predictor = get_predictor()
slate = get_slate()
metrics = get_metrics()

st.title("⚾ MLB Game Predictor")
if predictor is None and slate is None:
    st.info("No artifacts yet. Run `python scripts/train_mlb.py` then "
            "`python scripts/predict_today.py`.")
    st.stop()

trained = predictor.trained_through if predictor else (slate or {}).get("trained_through", "")
st.caption(f"Model trained through **{trained}** · fair prices for research/entertainment — "
           "not betting advice.")

tab_slate, tab_explore, tab_model = st.tabs(
    ["📅 Today's Slate", "🔎 Game Explorer", "📊 Model Card"])


# --------------------------------------------------------------------------- #
# Slate
# --------------------------------------------------------------------------- #
def render_game(g: dict):
    ml, sc, tot, rl = g["moneyline"], g["score"], g["total"], g["run_line"]
    st.markdown(f"### {g['away_team']} {sc['projected_away']} — "
                f"{sc['projected_home']} {g['home_team']}")
    c1, c2, c3, c4 = st.columns(4)
    fav = ml["favorite"]
    favp = ml["p_home"] if fav == g["home_team"] else ml["p_away"]
    fair = ml["fair_home"] if fav == g["home_team"] else ml["fair_away"]
    c1.metric(f"Moneyline — {fav}", _pct(favp), _odds(fair))
    c2.metric("Total", f"{tot['line']}", f"O {_pct(tot['p_over'])} / U {_pct(tot['p_under'])}")
    c3.metric(f"Run line {g['home_team']} -1.5", _pct(rl["home_-1.5"]))
    f5 = g["first_5_innings"]
    c4.metric("First 5 innings", f"{fav if favp>0.5 else ''}",
              f"H {_pct(f5['p_home'])} / A {_pct(f5['p_away'])}")
    st.caption(f"Starters — {g['away_team']}: {g['starters']['away']} · "
               f"{g['home_team']}: {g['starters']['home']}  ·  park {g.get('park','?')}")

    with st.expander("Pitcher & batter props"):
        st.markdown("**Starting pitchers**")
        st.dataframe(pd.DataFrame(g["pitcher_props"])[
            ["player", "proj_k", "k_line", "p_over_k", "proj_ip",
             "proj_hits_allowed", "proj_earned_runs"]],
            hide_index=True, use_container_width=True)
        bc1, bc2 = st.columns(2)
        for col, team, key in ((bc1, g["away_team"], "away_batter_props"),
                               (bc2, g["home_team"], "home_batter_props")):
            with col:
                st.markdown(f"**{team} batters**")
                df = pd.DataFrame(g[key])[
                    ["order", "player", "proj_hits", "p_1plus_hit", "proj_tb", "proj_hr"]]
                st.dataframe(df, hide_index=True, use_container_width=True)


with tab_slate:
    if not slate:
        st.info("No slate yet. Run `python scripts/predict_today.py`.")
    else:
        st.subheader(f"{slate.get('date','')} · {slate.get('n_games',0)} games "
                     f"· source: {slate.get('source','')}")
        rows = []
        for g in slate["games"]:
            ml, tot = g["moneyline"], g["total"]
            fav = ml["favorite"]
            favp = ml["p_home"] if fav == g["home_team"] else ml["p_away"]
            fair = ml["fair_home"] if fav == g["home_team"] else ml["fair_away"]
            rows.append({
                "Game": f"{g['away_team']} @ {g['home_team']}",
                "Favorite": fav, "Win%": _pct(favp), "Fair": _odds(fair),
                "Proj": f"{g['score']['projected_away']}-{g['score']['projected_home']}",
                "Total": tot["line"], "Over%": _pct(tot["p_over"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.divider()
        for g in slate["games"]:
            render_game(g)
            st.divider()


# --------------------------------------------------------------------------- #
# Game Explorer
# --------------------------------------------------------------------------- #
with tab_explore:
    if predictor is None:
        st.info("Game explorer needs the trained model. Run `python scripts/train_mlb.py`.")
    else:
        teams = predictor.teams()
        fb = predictor.fb
        c1, c2, c3 = st.columns([3, 3, 2])
        away = c1.selectbox("Away team", teams, index=teams.index("SDN") if "SDN" in teams else 0)
        home = c2.selectbox("Home team", teams, index=teams.index("LAN") if "LAN" in teams else 1)
        n_sims = c3.select_slider("Simulations", [2000, 5000, 10000], value=5000)
        d_asp = fb.default_starter.get(away, "")
        d_hsp = fb.default_starter.get(home, "")
        c4, c5 = st.columns(2)
        asp = c4.text_input(f"{away} starter (id or name)", value=d_asp)
        hsp = c5.text_input(f"{home} starter (id or name)", value=d_hsp)

        if home == away:
            st.warning("Pick two different teams.")
        elif st.button("Predict", type="primary"):
            from mlbpredictor.livedata import resolve_player
            hl = fb.default_lineups.get(home, [])
            al = fb.default_lineups.get(away, [])
            if len(hl) < 9 or len(al) < 9:
                st.error("No stored lineup for one of these teams.")
            else:
                pred = predictor.predict_game(
                    home, away, resolve_player(hsp) or d_hsp, resolve_player(asp) or d_asp,
                    hl, al, park=fb.default_park.get(home), n_sims=int(n_sims))
                render_game(pred.to_dict())
                st.caption("Lineups default to each team's most recent starting order.")


# --------------------------------------------------------------------------- #
# Model Card
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("How accurate is it?")
    st.markdown(
        "Temporal-split backtest: components are fit on an earlier season, the ensemble "
        "is calibrated on the next, and everything is scored on the **untouched most-recent "
        "season**. Lower **log-loss** / **Brier** is better; the home base rate is the "
        "no-skill floor. No public model reliably *beats* the closing line — the goal is "
        "**well-calibrated fair prices** you can line-shop against.")
    bt = (metrics or {}).get("backtest", {})
    if bt:
        split = bt.get("split", {})
        st.caption(f"train {split.get('train')} · validate {split.get('val')} · test {split.get('test')}")
        pretty = {"rundist": "Run model (NB)", "elo_logistic": "Elo (logistic)",
                  "gboost": "Gradient boosting", "base_rate": "Home base rate", "ensemble": "Ensemble"}
        rows = [{"Model": pretty.get(k, k), "Log-loss": m["log_loss"], "Brier": m["brier"],
                 "Accuracy": _pct(m["accuracy"]), "n": m["n"]}
                for k, m in bt.get("moneyline", {}).items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        tot = bt.get("totals", {})
        if tot:
            st.markdown(f"**Total runs:** MAE {tot['runs_mae']} / RMSE {tot['runs_rmse']} "
                        f"(predicted mean {tot['mean_pred']} vs actual {tot['mean_actual']}); "
                        f"per-team runs MAE {bt.get('team_runs',{}).get('team_runs_mae','?')}.")
    weights = (metrics or {}).get("weights", {})
    if weights:
        st.markdown("**Ensemble weights (backtest):** "
                    + " · ".join(f"{k} {v:.0%}" for k, v in weights.items() if v > 0.001)
                    + f" · temperature {(metrics or {}).get('temperature', 1):.2f}")
    if CALIBRATION_PLOT_PATH.exists():
        st.subheader("Calibration")
        st.image(str(CALIBRATION_PLOT_PATH), width=430)
    st.subheader("Method")
    st.markdown(
        "- **Data:** Retrosheet game logs + event files (Chadwick Bureau mirror), offline.\n"
        "- **Projections:** Marcel-style, regressed, park/age-adjusted player rates.\n"
        "- **Matchup:** odds-ratio (log5) batter-vs-pitcher.\n"
        "- **Simulator:** plate-appearance Monte-Carlo → runs, box lines, props.\n"
        "- **Ensemble:** Elo + run model + gradient boosting, log-loss-optimal + temperature.\n"
        "- **Reconciliation:** the simulation is reweighted to the calibrated moneyline, "
        "so the score, totals and props always agree with the headline odds.")
