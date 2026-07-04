"""World Cup 2026 prediction dashboard (Streamlit).

    streamlit run app.py

Loads the trained model (``outputs/model.joblib``) for live match predictions and
the precomputed tournament simulation (``outputs/predictions_2026.json``) for the
title-odds and bracket views. Run ``scripts/train.py`` then
``scripts/simulate_wc2026.py`` first to generate those artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wcpredictor.paths import CALIBRATION_PLOT_PATH, MODEL_PATH, PREDICTIONS_2026_PATH  # noqa: E402
from wcpredictor.predict import Predictor  # noqa: E402
from wcpredictor.viz import champion_bar_figure, scoreline_heatmap_figure, wdl_bar_figure  # noqa: E402

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")


@st.cache_resource
def get_predictor():
    """Load the trained model; return None if it is missing or unreadable."""
    if not MODEL_PATH.exists():
        return None
    try:
        return Predictor.load()
    except Exception as exc:  # e.g. library-version mismatch on the pickle
        st.warning(f"Could not load the trained model ({exc}). "
                   "Run `python scripts/train.py` to rebuild it.")
        return None


@st.cache_data
def get_predictions() -> dict | None:
    if PREDICTIONS_2026_PATH.exists():
        return json.load(open(PREDICTIONS_2026_PATH))
    return None


def _pct(x) -> str:
    return f"{x*100:.1f}%"


predictor = get_predictor()
data = get_predictions()

if predictor is None and data is None:
    st.title("🏆 World Cup 2026 — Match Predictor")
    st.info("No artifacts found yet. Run `python scripts/train.py` then "
            "`python scripts/simulate_wc2026.py` to build the model and simulation.")
    st.stop()

st.title("🏆 World Cup 2026 — Match Predictor")
trained = predictor.trained_through if predictor else (data or {}).get("trained_through", "")
sub = f"Model trained through **{trained}**"
if data:
    sub += f" · {data.get('n_sims', 0):,} tournament simulations · generated {data.get('generated','')}"
st.caption(sub)

tab_match, tab_odds, tab_bracket, tab_model = st.tabs(
    ["🎯 Match Predictor", "🏆 Title Odds", "🗺️ Bracket & Results", "📊 Model Card"])

# --------------------------------------------------------------------------- #
# Match predictor
# --------------------------------------------------------------------------- #
with tab_match:
  if predictor is None:
    st.info("Live match prediction needs the trained model. Run `python scripts/train.py`.")
  else:
    teams = predictor.teams()
    alive = data["state"]["alive"] if data else []
    default_home = "Spain" if "Spain" in teams else teams[0]
    default_away = "France" if "France" in teams else teams[1]
    c1, c2, c3 = st.columns([3, 3, 2])
    home = c1.selectbox("Home / Team A", teams, index=teams.index(default_home))
    away = c2.selectbox("Away / Team B", teams, index=teams.index(default_away))
    neutral = c3.checkbox("Neutral venue", value=True,
                          help="World Cup matches are at neutral venues (except hosts at home).")

    if home == away:
        st.warning("Pick two different teams.")
    else:
        pred = predictor.predict(home, away, neutral=neutral)
        i, j = pred.most_likely_score()
        left, right = st.columns([5, 4])
        with left:
            st.markdown(f"### {home} &nbsp; {i} – {j} &nbsp; {away}")
            fav = pred.favorite
            fav_name = home if fav == "home" else away if fav == "away" else "Draw"
            st.markdown(
                f"**Confidence: {pred.confidence_label}** ({_pct(pred.confidence)}) · "
                f"favourite: **{fav_name}** · "
                f"expected goals {pred.exp_home_goals:.2f} – {pred.exp_away_goals:.2f}")
            st.pyplot(wdl_bar_figure(pred.p_home, pred.p_draw, pred.p_away, home, away), use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric(f"{home} win", _pct(pred.p_home))
            m2.metric("Draw", _pct(pred.p_draw))
            m3.metric(f"{away} win", _pct(pred.p_away))

            st.markdown("**Most likely scorelines**")
            rows = [{"Score": f"{a}–{b}", "Probability": _pct(p)} for a, b, p in pred.top_scorelines(5)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with right:
            st.pyplot(scoreline_heatmap_figure(pred.score_matrix, home, away), use_container_width=True)

# --------------------------------------------------------------------------- #
# Title odds
# --------------------------------------------------------------------------- #
with tab_odds:
    if not data:
        st.info("Run `python scripts/simulate_wc2026.py` to generate tournament odds.")
    else:
        odds = pd.DataFrame(data["title_odds"])
        st.subheader(f"Championship probability — {data['current_round']} onward")
        left, right = st.columns([3, 4])
        with left:
            st.pyplot(champion_bar_figure(odds["team"].tolist(), odds["champion"].tolist()),
                      use_container_width=True)
        with right:
            sizes = data["sizes"]
            names = data["round_names"]
            show = odds.copy()
            disp = pd.DataFrame({"Team": show["team"], "Champion": show["champion"].map(_pct)})
            for rs in sizes[1:]:  # skip the current round (trivially 100%)
                col = f"reach_{rs}"
                if col in show:
                    disp[f"Reach {names[str(rs)]}"] = show[col].map(_pct)
            st.dataframe(disp, hide_index=True, use_container_width=True, height=560)
        st.caption("Estimated by Monte-Carlo simulation of the remaining bracket. "
                   "Bracket order comes from `config/wc2026.yaml` (Elo-seeded by default — "
                   "set the real pairings there for a bracket-exact forecast).")

# --------------------------------------------------------------------------- #
# Bracket & results
# --------------------------------------------------------------------------- #
with tab_bracket:
    if not data:
        st.info("Run `python scripts/simulate_wc2026.py` to populate this view.")
    else:
        state = data["state"]
        st.subheader(f"Still alive — {state['next_round']} ({len(state['alive'])} teams)")
        st.write(" · ".join(f"**{t}**" for t in state["alive"]))

        st.subheader(f"Projected {data['current_round']} fixtures")
        for fx in data["current_round_fixtures"]:
            p = fx["prediction"]
            si, sj = p["projected_score"]
            cols = st.columns([4, 2, 3])
            cols[0].markdown(f"**{fx['home']} {si} – {sj} {fx['away']}**")
            cols[1].markdown(f"{p['confidence_label']} ({_pct(p['confidence'])})")
            cols[2].markdown(
                f"<span style='color:#2563eb'>{_pct(p['p_home'])}</span> / "
                f"<span style='color:#9ca3af'>{_pct(p['p_draw'])}</span> / "
                f"<span style='color:#f59e0b'>{_pct(p['p_away'])}</span>",
                unsafe_allow_html=True)

        with st.expander("Group stage standings"):
            groups = state.get("groups", {})
            gcols = st.columns(3)
            for k, (label, standing) in enumerate(groups.items()):
                with gcols[k % 3]:
                    st.markdown(f"**Group {label}**")
                    tbl = pd.DataFrame(standing)[["team", "pts", "gd", "gf"]]
                    tbl.columns = ["Team", "Pts", "GD", "GF"]
                    st.dataframe(tbl, hide_index=True, use_container_width=True)

        with st.expander("Completed knockout results"):
            kn = pd.DataFrame(state["completed_knockout"])
            if len(kn):
                kn["Result"] = kn.apply(
                    lambda r: f"{r['home']} {r['home_score']}–{r['away_score']} {r['away']}  →  {r['winner']}",
                    axis=1)
                st.dataframe(kn[["Result"]], hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("How accurate is it?")
    st.markdown(
        "Backtest on an untouched test set (matches from **2018 onward**, including the "
        "2018 & 2022 World Cups). Components are trained only on earlier data; the "
        "**ensemble** blends them. Lower **RPS** / **log-loss** / **Brier** is better.")
    bt = (data or {}).get("backtest") or {}
    if bt:
        pretty = {"dixon_coles": "Dixon-Coles", "gboost": "Gradient Boosting",
                  "elo_logistic": "Elo (logistic)", "base_rate": "Base rate", "ensemble": "Ensemble"}
        rows = [{"Model": pretty.get(k, k), "RPS": m["rps"], "Log-loss": m["log_loss"],
                 "Brier": m["brier"], "Accuracy": _pct(m["accuracy"]),
                 "Exact score": _pct(m["exact_score"]), "n": m["n"]}
                for k, m in bt.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    wc = (data or {}).get("world_cup_backtest") or {}
    if wc:
        st.markdown(f"**On World Cup matches only** (2018 & 2022, n={wc['n']}): "
                    f"RPS {wc['rps']}, log-loss {wc['log_loss']}, accuracy {_pct(wc['accuracy'])}.")

    weights = (data or {}).get("weights") or predictor.weight_map
    if weights:
        st.markdown("**Ensemble weights:** " +
                    " · ".join(f"{k} {v:.0%}" for k, v in weights.items() if v > 0.001))

    if CALIBRATION_PLOT_PATH.exists():
        st.subheader("Calibration")
        st.image(str(CALIBRATION_PLOT_PATH), width=430)

    st.subheader("Method")
    st.markdown(
        "- **Data:** ~49k internationals (1872→today), auto-updated.\n"
        "- **Ratings:** custom Elo (margin-of-victory + match-importance weighted).\n"
        "- **Scorelines:** time-weighted **Dixon-Coles** bivariate Poisson.\n"
        "- **Outcomes:** gradient boosting on Elo, form, rest & venue features.\n"
        "- **Ensemble:** log-loss-optimal blend + temperature calibration.\n"
        "- **Tournament:** Monte-Carlo of the remaining bracket (shootout-aware).")
