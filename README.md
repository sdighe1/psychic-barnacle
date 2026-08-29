# 🏈 Fantasy Football Auction Draft Assistant

An interactive **Streamlit** app for a fantasy football **auction** draft. For
every player it shows a **ranking**, an **optimal price** (model value), a live
**expected price** (market value that adjusts for draft inflation) and your
**max bid** — then lets you **check off drafted players** to keep the board
clean and **recommends who to target next** based on the roster you've already
built. Uses **ESPN scoring** (defaults to **Half-PPR**; Standard and Full-PPR
are one click away).

## What it does

- **Value every player** — Value-Based Drafting (VORP) converts projected points
  into auction dollars that sum to the league's money pool, with proper **FLEX**
  handling for RB/WR/TE.
- **Optimal vs. Expected vs. Max** —
  - **Optimal $**: what a player is worth (model value).
  - **Expected $**: what they'll actually cost *right now* — recomputed live from
    the money and value still on the board (auction **inflation**).
  - **Max bid**: the most you can pay and still legally fill every roster spot.
- **One-click draft check-off** — log any pick (yours or another team's) with its
  price; drafted players drop off the board, your budget and roster update, and
  every remaining price re-inflates.
- **Roster-aware recommendations** — targets ranked by your open starter slots
  (scaled by positional scarcity), value (optimal vs. market) and tier scarcity,
  each with a suggested bid and a one-line reason.
- **Your team, live** — budget left, average $/open slot, open starter slots,
  projected starting-lineup points, and a full draft log with position scarcity.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The projection artifact (`outputs/projections.csv`) is committed, so the app
runs immediately after cloning. During your draft: pick your scoring format and
league size in the sidebar, then log picks on the **Draft Board** tab and watch
the **Recommendations** tab adapt to your roster.

## League settings

Defaults match a standard ESPN league and live in `config/league.yaml` (also
editable in the sidebar):

| Setting | Default |
|---|---|
| Teams / budget | 10 / $200 |
| Starters | QB, RB, RB, WR, WR, TE, **FLEX**, D/ST, K |
| Bench | 4 → **13 draftable spots** (IR is not drafted) |
| Position caps | QB 4 · RB 8 · WR 8 · TE 3 · D/ST 3 · K 3 |
| Scoring | Half-PPR (Standard / Full-PPR selectable) |

Fantasy points are computed from each player's projected **stat line**, so
switching scoring format re-prices the whole board instantly.

## Where the projections come from

The baseline is **built from open NFL data** (nflverse via `nfl_data_py`): recent
seasonal stats are turned into recency-weighted per-game rates, regressed toward
the positional mean, adjusted by a position/age curve, and scaled by projected
games. Kicker and D/ST — which aren't in the offensive feed and go for ~$1 in
auctions anyway — come from a small curated baseline (`data/kdst_baseline.csv`).

Rebuild any time (uses the latest seasons available upstream):

```bash
python scripts/build_projections.py
```

This writes `outputs/projections.csv` and `outputs/meta.json` (data vintage +
a **backtest**: it re-projects the most recent completed season from earlier data
and scores it — see the app's **Model Card** tab).

### Bring your own / the most accurate provider

A model is only a baseline. To draft on the **most accurate available numbers**,
import a projection CSV from your most-trusted source — FantasyPros' consensus
(historically among the most accurate), PFF, ESPN, or your own — via the
sidebar (**Projections → Import**). Columns are auto-detected; matched players
override the baseline and new players are added. You can also blend multiple
sources into an **accuracy-weighted consensus** at build time:

```bash
python scripts/build_projections.py --provider fantasypros=fp.csv --provider pff=pff.csv
```

Each source is weighted by its backtested accuracy (lower error → more weight).

## How it works

1. **Data** (`src/ffauction/data.py`) — seasonal NFL stats + rosters from
   nflverse, normalised to a stat-line schema.
2. **Projections** (`projections.py`) — recency-weighted, regressed, age-adjusted
   baseline for the upcoming season.
3. **Providers & accuracy** (`providers.py`, `accuracy.py`) — import/normalise
   external projections and blend sources weighted by backtested accuracy.
4. **Scoring** (`scoring.py`) — ESPN Standard / Half-PPR / Full-PPR from stat lines.
5. **Valuation** (`valuation.py`) — replacement levels (+FLEX), VORP, optimal
   dollars, live inflation-adjusted expected prices, tiers.
6. **Draft** (`draft.py`) — draft state, roster/slot logic with caps, max bid,
   recommendations.
7. **App** (`app.py`) — the Streamlit dashboard.

## Project structure

```
app.py                       Streamlit auction dashboard
config/league.yaml           editable league / scoring / roster config
data/kdst_baseline.csv       curated kicker & D/ST baseline
scripts/build_projections.py rebuild projections (+ optional provider blend)
src/ffauction/               scoring, valuation, draft, projections, data, ...
outputs/projections.csv      committed baseline projections (app runs on clone)
outputs/meta.json            data vintage + backtest accuracy
tests/                       pytest unit + app-smoke tests
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notes & credits

- NFL data: [nflverse](https://github.com/nflverse) via
  [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) (open data). In this
  environment the freshest season available upstream is used automatically; the
  **Model Card** tab shows the exact vintage.
- Projections are a data-driven **baseline** for analysis and entertainment, not
  betting advice — import current-season numbers for your live draft.
