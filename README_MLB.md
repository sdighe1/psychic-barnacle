# ⚾ MLB Game Predictor

A betting-oriented model that, run each **morning once projected starting pitchers &
lineups are announced**, predicts every MLB game's **outcome, score, statistics and
player props**, and reports them as **fair prices** (moneyline, run line, totals,
first-five-innings). It ships an interactive **Streamlit dashboard**.

The engine is a **plate-appearance Monte-Carlo simulator** driven by Marcel-style
player projections and an odds-ratio (log5) batter-vs-pitcher matchup, wrapped in a
**calibrated ensemble** (team Elo + a negative-binomial run model + gradient boosting).
It is trained and backtested entirely on **open, offline data** (the Chadwick Bureau's
Retrosheet mirror), so it builds with no paid feeds.

> Statistical estimates for research and entertainment — **not betting advice**. No
> public model reliably *beats* the closing line; the goal here is **well-calibrated
> fair prices** you can line-shop against, backed by an honest backtest.

## What it predicts, per game

- **Moneyline** — win probability + fair American odds for each side.
- **Score & totals** — projected runs for each team, the total-runs distribution and
  over/under at any line.
- **Run line** — cover probabilities for ±1.5.
- **First five innings (F5)** — the starter-driven market.
- **Player props** — per-pitcher projected strikeouts / innings / hits / earned runs,
  and per-batter hits / total bases / home runs / runs / RBI, each with over/under
  probabilities.

All of it comes from **one simulation**, reweighted to the calibrated moneyline, so the
score, the totals and the props never contradict the headline odds.

## How accurate is it?

Temporal-split backtest — components fit on **2022**, ensemble calibrated on **2023**,
everything scored on the **untouched 2025 season** (n = 2,430). Lower log-loss / Brier is
better; the home base rate is the no-skill floor.

| Model | Log-loss ↓ | Brier ↓ | Accuracy ↑ |
|---|---|---|---|
| Home base rate (no skill) | 0.6897 | 0.2483 | 54.3% |
| Gradient boosting | 0.6950 | 0.2505 | 54.9% |
| Run model (negative binomial) | 0.6847 | 0.2458 | 55.9% |
| Elo (logistic) | 0.6801 | 0.2437 | 55.9% |
| **Ensemble (shipped)** | **0.6805** | **0.2438** | **56.4%** |

**Score:** total-runs MAE 3.61 / RMSE 4.55, with a predicted mean of 9.04 vs 8.90 actual —
i.e. well-calibrated on run environment (the MAE is mostly irreducible run variance).
The ensemble is well-calibrated (temperature ≈ 1.0; see the Model Card tab). These are
honest MLB numbers: strong single-game models top out near 56–58%, roughly the market's
own ceiling.

## Quick start

```bash
pip install -r requirements.txt

# Build the model (downloads open data on first run, then caches): ~30–90s.
python scripts/train_mlb.py            # backtest + save outputs/mlb_model.joblib + metrics

# Predict a slate. Auto-fetches from the MLB Stats API if allowed (see below),
# otherwise reads a manual slate file you fill in each morning:
python scripts/predict_today.py --slate config/slate.example.yaml

streamlit run mlb_app.py               # dashboard: slate, game explorer, model card
```

The trained artifacts under `outputs/` are committed, so the dashboard works right after
cloning; the scripts are only needed to refresh.

## Enabling live auto-fetch (probable pitchers & lineups)

The morning run wants today's **probable pitchers and lineups**, which come from the free
**MLB Stats API** (`statsapi.mlb.com`). Some managed environments block it via their
**network policy** (chosen when the environment was created — see
[the docs](https://code.claude.com/docs/en/claude-code-on-the-web)). To turn auto-fetch on:

> **Allow the host `statsapi.mlb.com`** in the environment's network policy. That single
> host covers the schedule, probable pitchers, posted lineups, current-season player stats
> and final boxscores.

`scripts/predict_today.py` auto-detects reachability: if `statsapi.mlb.com` is allowed it
pulls the live slate; otherwise it transparently falls back to your **manual slate file**
(`config/slate.yaml`), and everything else (training, backtesting, the dashboard) already
works offline from GitHub-hosted data.

### Manual slate format

Fill in `config/slate.yaml` (see `config/slate.example.yaml`) once the probables are out.
Teams accept abbreviations (`LAD`, `NYY`, `SF`) or Retrosheet codes; pitchers/batters
accept a Retrosheet id **or** a full name. Lineups are optional — omit them and the model
uses each team's most recent starting order:

```yaml
games:
  - home: LAD
    away: SD
    home_sp: "Clayton Kershaw"    # or a retro id like kersc001
    away_sp: darvy001
    # home_lineup / away_lineup optional (9 ids or names)
```

## How it works

1. **Data** (`data.py`, `retrosheet.py`, `ids.py`) — Retrosheet **game logs** (scores,
   starters, lineups, park) and **event files** (play-by-play → per-player PA-outcome
   rates, park factors) from the Chadwick Bureau's GitHub mirror, joined to live MLBAM ids
   through the Chadwick **register**.
2. **Projections** (`projections.py`, `park.py`) — Marcel-style rate vectors over
   `{1B,2B,3B,HR,BB,HBP,SO,OUT}` for each batter, starter and team bullpen: recent seasons
   weighted, regressed to league by real sample size, park/age-adjusted, computed *as of*
   the season (leak-free).
3. **Matchup** (`matchup.py`) — the odds-ratio / log5 blend of batter × pitcher ÷ league.
4. **Simulator** (`simulate.py`) — a base-out state machine plays games out PA-by-PA
   (starter→bullpen hook, walk-offs, extra-innings ghost runner), yielding run
   distributions, team box lines and per-player stat lines.
5. **Ensemble** (`ratings.py`, `offense.py`, `models/`) — team Elo, a fast negative-binomial
   run model and gradient boosting are blended (log-loss-optimal weights + temperature) into
   a calibrated moneyline; the simulator is then **reweighted** to that probability so all
   outputs agree (`predict.py`).

## Project structure

```
mlb_app.py                     Streamlit dashboard (slate / explorer / model card)
config/mlb.yaml                seasons, projection & simulation settings
config/slate.example.yaml      manual slate template
scripts/train_mlb.py           backtest + fit + save model & metrics
scripts/predict_today.py       the morning slate run -> outputs/predictions_*.json
src/mlbpredictor/              data, projections, matchup, simulate, ratings, models/, predict
tests/                         pytest unit + integration tests
outputs/                       committed model + metrics + calibration + latest predictions
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Data & credits

Data: [Chadwick Bureau / Retrosheet](https://github.com/chadwickbureau/retrosheet) and
[Chadwick Bureau / register](https://github.com/chadwickbureau/register) (open data).
The information used here was obtained free of charge from and is copyrighted by Retrosheet
(www.retrosheet.org). Modelling builds on standard sabermetric methods (Marcel projections,
Tango's odds-ratio matchup, wOBA linear weights, 538-style Elo). **Not betting advice.**
