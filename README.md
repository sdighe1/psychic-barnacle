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

The board uses the **best draft signal reachable from this environment**:
**FantasyPros expert-consensus rankings (ECR)** — historically among the most
accurate free sources, and current for the upcoming season (rookies, new teams
and injuries included). Because FantasyPros publishes *rankings*, not points, we
fuse two things:

- **Ordering** — FantasyPros redraft positional ECR, pulled from the open
  DynastyProcess/ffverse GitHub mirror (the same source `nflreadr::load_ff_rankings()`
  uses), refreshed daily through the offseason.
- **Point magnitudes** — a model built from open NFL history (nflverse via
  `nfl_data_py`): recency-weighted per-game rates, regressed to the positional
  mean, age-adjusted, scaled by projected games — transplanted onto the consensus
  order at each positional rank.

The result honors the best-available ranking *and* has realistic, format-flexible
point spreads (Standard/Half/PPR all re-score live). Kicker and D/ST — which go
for ~$1 in auctions — use a curated baseline (`data/kdst_baseline.csv`) placed in
FantasyPros order, and are held at $1 so real money flows to skill players.

Rebuild any time:

```bash
python scripts/build_projections.py                 # FantasyPros-anchored (default)
python scripts/build_projections.py --no-fantasypros # nflverse model only
```

This writes `outputs/projections.csv` and `outputs/meta.json` (ranking source +
scrape date + a magnitude-model backtest — see the app's **Model Card** tab).

> **Network note.** This environment's egress policy blocks live provider sites
> (fantasypros.com, ESPN, Sleeper return 403 policy denials), so the app can't
> pull live auction values / AAV directly — it uses the reachable FantasyPros
> **GitHub mirror** instead. An org admin can allowlist those hosts if you want
> live AAV; otherwise, import a downloaded export (below).

### Bring your own / an even more specific source

Import a projection CSV from your most-trusted source — a FantasyPros/PFF/ESPN
export or your own — via the sidebar (**Projections → Import**). Columns are
auto-detected; matched players override, new players are added. You can also blend
multiple sources into an **accuracy-weighted consensus** at build time:

```bash
python scripts/build_projections.py --provider fantasypros=fp.csv --provider pff=pff.csv
```

Each source is weighted by its backtested accuracy (lower error → more weight).

## How it works

1. **Data** (`src/ffauction/data.py`) — seasonal NFL stats + rosters from
   nflverse, normalised to a stat-line schema.
2. **Rankings** (`fantasypros.py`) — current FantasyPros expert-consensus ranks
   from the reachable GitHub mirror.
3. **Projections** (`projections.py`) — recency-weighted, regressed, age-adjusted
   magnitude model, fused onto the FantasyPros consensus order (`anchor_to_rankings`).
4. **Providers & accuracy** (`providers.py`, `accuracy.py`) — import/normalise
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
scripts/build_projections.py rebuild projections (FantasyPros-anchored + blend)
src/ffauction/               scoring, valuation, draft, projections, fantasypros, ...
outputs/projections.csv      committed projections (app runs on clone)
outputs/meta.json            ranking source + scrape date + backtest accuracy
tests/                       pytest unit + app-smoke tests
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notes & credits

- Rankings: **FantasyPros** expert consensus, via the open
  [DynastyProcess](https://github.com/dynastyprocess/data) / ffverse GitHub mirror
  (the source `nflreadr::load_ff_rankings()` uses). Appropriate for a personal
  draft tool; the **Model Card** tab shows the scrape date.
- NFL data: [nflverse](https://github.com/nflverse) via
  [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) (open data), used for
  point magnitudes and the backtest.
- Projections are for analysis and entertainment, not betting advice. Auction
  market values (AAV) are model-derived (live AAV hosts are blocked here) — import
  a provider export for exact market numbers.
