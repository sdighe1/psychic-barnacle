"""Fantasy football auction-draft assistant (Streamlit).

    streamlit run app.py

Rankings, an optimal (model) price, a live expected (market, inflation-adjusted)
price and your max bid for every player -- plus one-click "draft" check-off and
recommendations that adapt to the roster you have already built. ESPN scoring
(Standard / Half-PPR / Full-PPR).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ffauction import providers, scoring, valuation  # noqa: E402
from ffauction.draft import DraftState  # noqa: E402
from ffauction.league import POSITIONS, LeagueSettings, load_league  # noqa: E402
from ffauction.paths import DRAFT_STATE_PATH, META_PATH, PROJECTIONS_PATH  # noqa: E402

st.set_page_config(page_title="Auction Draft Assistant", page_icon="🏈", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_base_projections() -> pd.DataFrame:
    return pd.read_csv(PROJECTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def get_projections() -> pd.DataFrame:
    """Base projections, with any imported provider overrides applied."""
    base = load_base_projections().copy()
    custom = st.session_state.get("custom_proj")
    if custom is None:
        return base
    base = base.set_index("player_id")
    over = custom.set_index("player_id")
    cols = [c for c in over.columns if c in base.columns]
    base.update(over[cols])                       # override matched players
    new = over[~over.index.isin(base.index)]      # append brand-new players
    if len(new):
        base = pd.concat([base, new[[c for c in base.columns if c in new.columns]]])
    return base.reset_index()


# --------------------------------------------------------------------------- #
# Draft-state helpers (session backed)
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    st.session_state.setdefault("picks", {})
    st.session_state.setdefault("custom_proj", None)


def do_draft(player_id: str, price: int, mine: bool) -> None:
    st.session_state.picks[str(player_id)] = {"price": int(price), "mine": bool(mine)}
    _autosave()


def do_undo(player_id: str) -> None:
    st.session_state.picks.pop(str(player_id), None)
    _autosave()


def do_reset() -> None:
    st.session_state.picks = {}
    _autosave()


def _autosave() -> None:
    try:
        DRAFT_STATE_PATH.write_text(json.dumps({"picks": st.session_state.picks}))
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Sidebar: league settings, budget, import, save/load
# --------------------------------------------------------------------------- #
def sidebar_settings(defaults: LeagueSettings) -> LeagueSettings:
    st.sidebar.header("⚙️ League settings")
    fmt_options = list(scoring.FORMATS)
    fmt = st.sidebar.selectbox(
        "Scoring format", fmt_options,
        index=fmt_options.index(defaults.scoring_format),
        format_func=lambda f: scoring.FORMAT_LABELS[f],
    )
    c1, c2 = st.sidebar.columns(2)
    teams = c1.number_input("Teams", min_value=4, max_value=16, value=defaults.teams, step=1)
    budget = c2.number_input("Budget $", min_value=50, max_value=1000, value=defaults.budget, step=10)

    with st.sidebar.expander("Roster & caps (edit config/league.yaml)"):
        starters = " · ".join(f"{k} {v}" for k, v in defaults.starters.items())
        st.caption(f"**Starters:** {starters}")
        st.caption(f"**Bench:** {defaults.bench}  →  **{defaults.roster_size} draftable spots**")
        caps = " · ".join(f"{k}≤{v}" for k, v in defaults.position_max.items())
        st.caption(f"**Position caps:** {caps}")

    return defaults.with_updates(scoring_format=fmt, teams=int(teams), budget=int(budget))


def sidebar_import(base_proj: pd.DataFrame) -> None:
    st.sidebar.header("📥 Projections")
    up = st.sidebar.file_uploader(
        "Import a projections CSV (FantasyPros / PFF / ESPN / your own)",
        type=["csv"], key="proj_upload",
        help="Auto-detects player/pos/team and stat or points columns. Matched "
             "players override the baseline; new players are added.",
    )
    if up is not None and st.sidebar.button("Apply import", use_container_width=True):
        try:
            prov = providers.normalize_provider_frame(pd.read_csv(up))
            prov = providers.match_to_baseline(prov, base_proj)
            prov["player_id"] = prov["player_id"].where(
                prov["player_id"].astype(str).str.len() > 0, "IMP_" + prov["name_key"]
            ).fillna("IMP_" + prov["name_key"])
            st.session_state.custom_proj = prov
            st.sidebar.success(f"Imported {len(prov)} players.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Could not import: {exc}")
    if st.session_state.get("custom_proj") is not None:
        if st.sidebar.button("Clear imported projections", use_container_width=True):
            st.session_state.custom_proj = None
            st.rerun()


def sidebar_saveload() -> None:
    st.sidebar.header("💾 Draft")
    st.sidebar.download_button(
        "Download draft state", data=json.dumps({"picks": st.session_state.picks}, indent=2),
        file_name="draft_state.json", mime="application/json", use_container_width=True,
    )
    up = st.sidebar.file_uploader("Load draft state", type=["json"], key="draft_upload")
    if up is not None and st.sidebar.button("Load", use_container_width=True):
        try:
            st.session_state.picks = {
                str(k): {"price": int(v["price"]), "mine": bool(v["mine"])}
                for k, v in (json.load(up).get("picks") or {}).items()
            }
            _autosave()
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Bad file: {exc}")
    if st.sidebar.button("♻️ Reset draft", use_container_width=True):
        do_reset()
        st.rerun()


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
DOLLAR_COLS = {
    "optimal_dollar": "Optimal $", "expected_dollar": "Expected $", "max_bid": "Max bid",
    "draft_price": "Paid $", "suggested_bid": "Suggest $",
}
DISPLAY_RENAME = {
    "overall_rank": "#", "player": "Player", "position": "Pos", "team": "Tm",
    "tier": "Tier", "proj_points": "Proj", "pos_rank": "PosRk", "why": "Why",
    "rec_score": "Score", "drafted_by": "By", **DOLLAR_COLS,
}


def _fmt_board(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    show = df[cols].rename(columns=DISPLAY_RENAME)
    return show


def budget_banner(ds: DraftState) -> None:
    league = ds.league
    open_slots = ds.open_slots()
    open_total = ds.my_open_slots
    per_slot = ds.my_remaining / open_total if open_total else 0
    c = st.columns([1, 1, 1, 1, 3])
    c[0].metric("My budget left", f"${ds.my_remaining}", f"-${ds.my_spent} spent")
    c[1].metric("Roster", f"{len(ds.my_player_ids)}/{league.roster_size}")
    c[2].metric("Max bid now", f"${ds.max_bid_any()}")
    c[3].metric("Avg $/open slot", f"${per_slot:.0f}")
    needs = [f"{k}×{v}" for k, v in open_slots.items() if v and k != "BENCH"]
    bench = open_slots.get("BENCH", 0)
    with c[4]:
        st.caption("**Open starter slots**")
        st.write(("  ".join(f"`{n}`" for n in needs) or "_all starters filled_")
                 + (f"  +{bench} bench" if bench else ""))


def draft_board_tab(ds: DraftState) -> None:
    budget_banner(ds)
    st.divider()

    # --- log a pick ---------------------------------------------------------
    board_all = ds.board(available_only=False)
    avail = board_all[~board_all["drafted"]].copy()
    st.subheader("✔️ Log a pick")
    with st.form("log_pick", clear_on_submit=True):
        fc = st.columns([4, 1, 2, 1])
        options = avail.sort_values("overall_rank")["player_id"].astype(str).tolist()
        labels = {
            str(r.player_id): f"{r.player} ({r.position}-{r.team})  ·  exp ${int(r.expected_dollar)} · max ${int(r.max_bid)}"
            for r in avail.itertuples()
        }
        pid = fc[0].selectbox("Player", options, format_func=lambda i: labels.get(i, i), index=None,
                              placeholder="Search a player…")
        default_price = int(avail.set_index(avail["player_id"].astype(str)).loc[pid, "expected_dollar"]) if pid else 1
        default_price = min(int(ds.league.budget), max(1, default_price))
        price = fc[1].number_input("Price $", min_value=1, max_value=int(ds.league.budget), value=default_price)
        who = fc[2].radio("Drafted by", ["My team", "Another team"], horizontal=True)
        submitted = fc[3].form_submit_button("Draft ✔", use_container_width=True)
    if submitted and pid:
        mine = who == "My team"
        if mine and ds.max_bid(pid) < price:
            st.warning(f"That's above your max bid (${ds.max_bid(pid)}). Logged anyway.")
        do_draft(pid, price, mine)
        st.rerun()

    # --- filters ------------------------------------------------------------
    st.subheader("📋 Draft board")
    f = st.columns([2, 3, 1, 1])
    pos_filter = f[0].multiselect("Positions", list(POSITIONS), default=list(POSITIONS))
    search = f[1].text_input("Search player", "")
    only_need = f[2].checkbox("My needs", value=False, help="Only positions I still need to start.")
    show_drafted = f[3].checkbox("Show drafted", value=False)

    view = board_all if show_drafted else avail
    view = view[view["position"].isin(pos_filter)]
    if search:
        view = view[view["player"].str.contains(search, case=False, na=False)]
    if only_need:
        needs = set(ds.open_starter_positions())
        view = view[view["position"].isin(needs)]

    cols = ["overall_rank", "player", "position", "team", "tier", "proj_points",
            "optimal_dollar", "expected_dollar", "max_bid"]
    if show_drafted:
        cols += ["drafted_by", "draft_price"]
    st.dataframe(
        _fmt_board(view.sort_values("overall_rank"), cols),
        hide_index=True, use_container_width=True, height=520,
        column_config={
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Optimal $": st.column_config.NumberColumn(format="$%d"),
            "Expected $": st.column_config.NumberColumn(format="$%d"),
            "Max bid": st.column_config.NumberColumn(format="$%d"),
            "Paid $": st.column_config.NumberColumn(format="$%d"),
        },
    )
    st.caption("**Optimal $** = model value (VORP). **Expected $** = live market price, "
               "adjusts for draft inflation as players go off the board. **Max bid** = the "
               "most you can pay and still fill every roster spot ($1 minimum each).")


def recommendations_tab(ds: DraftState) -> None:
    budget_banner(ds)
    st.divider()
    if ds.my_open_slots <= 0:
        st.success("Your roster is full. 🎉")
        return
    needs = ds.open_starter_positions()
    st.markdown(
        f"**Targets for your roster** — ${ds.my_remaining} left for {ds.my_open_slots} spots "
        f"(~${ds.my_remaining / max(1, ds.my_open_slots):.0f}/slot). "
        + (f"Still need to start: {', '.join(needs)}." if needs else "All starters filled — building depth/upside.")
    )
    recs = ds.recommendations(top_n=15)
    if recs.empty:
        st.info("No affordable targets — you may be out of budget or roster space.")
        return
    cols = ["player", "position", "team", "tier", "proj_points",
            "optimal_dollar", "expected_dollar", "suggested_bid", "max_bid", "rec_score", "why"]
    st.dataframe(
        _fmt_board(recs, cols), hide_index=True, use_container_width=True, height=560,
        column_config={
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Optimal $": st.column_config.NumberColumn(format="$%d"),
            "Expected $": st.column_config.NumberColumn(format="$%d"),
            "Suggest $": st.column_config.NumberColumn(format="$%d"),
            "Max bid": st.column_config.NumberColumn(format="$%d"),
            "Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
        },
    )
    st.caption("Ranked by roster need (open starter slots, scaled by scarcity), value "
               "(optimal vs market) and tier scarcity. **Suggest $** is a sensible bid, "
               "capped at your max.")


def my_team_tab(ds: DraftState) -> None:
    budget_banner(ds)
    st.divider()
    roster = ds.my_roster()
    if roster.empty:
        st.info("No players yet. Log your picks on the Draft Board tab.")
        return
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Roster")
        show = roster[["player", "position", "team", "tier", "proj_points", "price"]].rename(
            columns={**DISPLAY_RENAME, "price": "Paid $"})
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"Proj": st.column_config.NumberColumn(format="%.1f"),
                                    "Paid $": st.column_config.NumberColumn(format="$%d")})
        undo_id = st.selectbox("Undo a pick", roster["player_id"].astype(str).tolist(),
                               format_func=lambda i: roster.set_index(roster["player_id"].astype(str)).loc[i, "player"],
                               index=None, placeholder="Select a player to undo…")
        if undo_id and st.button("Undo pick"):
            do_undo(undo_id)
            st.rerun()
    with right:
        st.subheader("Slots")
        for slot, n in ds.open_slots().items():
            filled = ds.league.starters.get(slot, ds.league.bench if slot == "BENCH" else 0)
            st.write(f"**{slot}**: {filled - n}/{filled} filled" if filled else f"**{slot}**: {n} open")
        st.metric("Projected starting points", f"{_starter_points(ds):.0f}")


def _starter_points(ds: DraftState) -> float:
    """Sum of my best legal starting lineup's projected points."""
    roster = ds.my_roster().sort_values("proj_points", ascending=False)
    from ffauction.league import FLEX_ELIGIBLE
    need = dict(ds.league.starters)
    total, flex_pool = 0.0, []
    used = set()
    for pos in ("QB", "RB", "WR", "TE", "DST", "K"):
        got = roster[roster["position"] == pos].head(need.get(pos, 0))
        total += got["proj_points"].sum()
        used.update(got.index)
    for _, r in roster.iterrows():
        if r.name not in used and r["position"] in FLEX_ELIGIBLE:
            flex_pool.append(r["proj_points"])
    total += sum(sorted(flex_pool, reverse=True)[: need.get("FLEX", 0)])
    return total


def draft_log_tab(ds: DraftState) -> None:
    board_all = ds.board(available_only=False)
    drafted = board_all[board_all["drafted"]].copy().sort_values("draft_price", ascending=False)
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"All picks ({len(drafted)})")
        if drafted.empty:
            st.info("No picks logged yet.")
        else:
            show = drafted[["player", "position", "team", "draft_price", "drafted_by",
                            "optimal_dollar"]].rename(columns=DISPLAY_RENAME)
            st.dataframe(show, hide_index=True, use_container_width=True, height=460,
                         column_config={"Paid $": st.column_config.NumberColumn(format="$%d"),
                                        "Optimal $": st.column_config.NumberColumn(format="$%d")})
    with right:
        st.subheader("Position scarcity")
        avail = board_all[~board_all["drafted"]]
        rows = []
        for pos in POSITIONS:
            startable = int((avail[avail["position"] == pos]["vorp"] > 0).sum())
            rows.append({"Pos": pos, "Startable left": startable,
                         "Total left": int((avail["position"] == pos).sum())})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("“Startable” = players still projected above replacement level.")


def model_card_tab(meta: dict) -> None:
    st.subheader("Projections & accuracy")
    if not meta:
        st.info("Run `python scripts/build_projections.py` to generate projections.")
        return
    c = st.columns(3)
    c[0].metric("Data through", f"{meta.get('data_through_season','?')} season")
    c[1].metric("Projecting", f"{meta.get('target_season','?')} season")
    c[2].metric("Players", meta.get("n_players", "?"))
    acc = meta.get("accuracy") or {}
    if acc:
        st.markdown(
            f"**Backtest** (projecting the {acc.get('test_season')} season from earlier "
            f"data, top {acc.get('n')} players): MAE **{acc.get('mae')}** pts · "
            f"rank-corr **{acc.get('rank_corr')}**.")
        pm = acc.get("pos_mae") or {}
        if pm:
            st.caption("Per-position MAE: " + " · ".join(f"{k} {v}" for k, v in pm.items()))
    st.markdown(
        f"**Sources:** {', '.join(meta.get('sources', []))} "
        f"(weights: {meta.get('source_weights', {})}). Built from open **nflverse** data; "
        "seasons " + ", ".join(map(str, meta.get("seasons_used", []))) + ".")
    st.info("This is a data-driven **baseline**. For your live draft, import current-season "
            "projections from your most-trusted source (sidebar → Projections) — matched "
            "players are overridden and new players added. Kicker/D-ST are a curated baseline.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()
    defaults = load_league()
    meta = load_meta()

    league = sidebar_settings(defaults)
    base_proj = load_base_projections()
    sidebar_import(base_proj)
    sidebar_saveload()

    proj = get_projections()
    values = valuation.compute_values(proj, league)
    ds = DraftState(league, values)
    ds.load_dict({"picks": st.session_state.picks})

    st.title("🏈 Auction Draft Assistant")
    st.caption(
        f"{scoring.FORMAT_LABELS[league.scoring_format]} · {league.teams} teams · "
        f"${league.budget} budget · {league.roster_size} roster spots · "
        f"baseline from NFL data through {meta.get('data_through_season', '?')}"
        + ("  ·  📥 imported projections active" if st.session_state.get("custom_proj") is not None else "")
    )

    tabs = st.tabs(["📋 Draft Board", "🎯 Recommendations", "🧑‍🤝‍🧑 My Team",
                    "📜 Draft Log", "📊 Model Card"])
    with tabs[0]:
        draft_board_tab(ds)
    with tabs[1]:
        recommendations_tab(ds)
    with tabs[2]:
        my_team_tab(ds)
    with tabs[3]:
        draft_log_tab(ds)
    with tabs[4]:
        model_card_tab(meta)


main()
