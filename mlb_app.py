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
from mlbpredictor.viz import confidence_badge_md, total_distribution_figure  # noqa: E402

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


def _ci(range_dict: dict, level: int = 90) -> str:
    r = (range_dict or {}).get(f"range_{level}")
    return f"{r[0]}–{r[1]}" if r else "—"


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
def render_game(g: dict, pred=None):
    ml, sc, tot, rl = g["moneyline"], g["score"], g["total"], g["run_line"]
    conf = g.get("confidence", {})
    level = conf.get("level", "—")
    st.markdown(
        f"### {g['away_team']} {sc['projected_away']} — {sc['projected_home']} {g['home_team']} "
        f"&nbsp; {confidence_badge_md(level)}", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    fav = ml["favorite"]
    favp = ml["p_home"] if fav == g["home_team"] else ml["p_away"]
    fair = ml["fair_home"] if fav == g["home_team"] else ml["fair_away"]
    ci = ml.get("p_home_ci")
    c1.metric(f"Moneyline — {fav}", _pct(favp), _odds(fair))
    if ci:
        c1.caption(f"win-prob range {_pct(ci[0])}–{_pct(ci[1])}")
    c2.metric("Total", f"{tot['line']}", f"O {_pct(tot['p_over'])} / U {_pct(tot['p_under'])}")
    c2.caption(f"90% range {_ci(tot, 90)} runs")
    c3.metric(f"Score {g['away_team']}–{g['home_team']}",
              f"{sc['projected_away']}–{sc['projected_home']}")
    c3.caption(f"90%: {_ci(sc.get('away_range',{}))} / {_ci(sc.get('home_range',{}))}")
    f5 = g["first_5_innings"]
    c4.metric("First 5 innings", "", f"H {_pct(f5['p_home'])} / A {_pct(f5['p_away'])}")
    st.caption(f"Run line {g['home_team']} -1.5: {_pct(rl['home_-1.5'])} · "
               f"{g['away_team']} -1.5: {_pct(rl['away_-1.5'])}  |  "
               f"Starters — {g['away_team']}: {g['starters']['away']} · "
               f"{g['home_team']}: {g['starters']['home']}  ·  park {g.get('park','?')}")

    comp = conf.get("components", {})
    dq = g.get("data_quality", {})
    with st.expander(f"Why confidence = {level}?"):
        st.markdown(
            f"**Score {conf.get('score','?')}** — a blend of: "
            f"decisiveness {comp.get('decisiveness','?')}, "
            f"component agreement {comp.get('agreement','?')}, "
            f"input quality {comp.get('input_quality','?')}, "
            f"simulation stability {comp.get('stability','?')}.")
        st.caption(
            f"Lineups {'confirmed' if dq.get('lineup_confirmed') else 'projected (defaulted)'} · "
            f"roster coverage {_pct(dq.get('roster_coverage',1))} · "
            f"effective sims {dq.get('n_eff','?'):,}/{dq.get('n_sims','?'):,}. "
            "Numeric ranges are 50%/90% credible intervals from the simulation.")

    if pred is not None:
        st.pyplot(total_distribution_figure(
            pred.sim.total(), pred.weights, tot["line"],
            interval=tot.get("range_90")), use_container_width=True)

    with st.expander("Pitcher & batter props (with 90% ranges)"):
        st.markdown("**Starting pitchers**")
        prows = [{"Pitcher": p["player"], "K": p["proj_k"], "K 90%": _ci(p.get("k_range", {})),
                  "Line": p["k_line"], "Over%": _pct(p["p_over_k"]), "IP": p["proj_ip"],
                  "H": p["proj_hits_allowed"], "ER": p["proj_earned_runs"]}
                 for p in g["pitcher_props"]]
        st.dataframe(pd.DataFrame(prows), hide_index=True, use_container_width=True)
        bc1, bc2 = st.columns(2)
        for col, team, key in ((bc1, g["away_team"], "away_batter_props"),
                               (bc2, g["home_team"], "home_batter_props")):
            with col:
                st.markdown(f"**{team} batters**")
                rows = [{"#": b["order"], "Batter": b["player"], "H": b["proj_hits"],
                         "H 90%": _ci(b.get("hits_range", {})), "1+H%": _pct(b["p_1plus_hit"]),
                         "TB": b["proj_tb"], "TB 90%": _ci(b.get("tb_range", {})),
                         "HR%": _pct(b["p_hr"])} for b in g[key]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


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
                "Conf": g.get("confidence", {}).get("level", "—"),
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
                render_game(pred.to_dict(), pred=pred)
                st.caption("Lineups default to each team's most recent starting order "
                           "(so confidence is capped at 'projected' input quality).")


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
        cov = bt.get("interval_coverage", {})
        if cov:
            st.markdown("**Interval calibration** (a 50% / 90% interval should contain the "
                        "actual total that often): "
                        + " · ".join(f"{k.replace('coverage_','')}% → {_pct(v)}"
                                     for k, v in cov.items()))
        bands = bt.get("confidence_bands", {})
        if bands:
            st.markdown("**Moneyline accuracy by confidence band** — validates the label is "
                        "meaningful (High should beat Low):")
            st.dataframe(pd.DataFrame([
                {"Confidence": b, "Accuracy": _pct(m["accuracy"]), "Games": m["n"]}
                for b, m in bands.items()]), hide_index=True, use_container_width=True)
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
