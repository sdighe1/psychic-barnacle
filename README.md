> **This repo contains two independent sports-prediction models:**
> the **World Cup 2026 match predictor** (below) and an **⚾ MLB game predictor**
> for daily betting markets (outcome, score, statistics & player props) —
> see **[README_MLB.md](README_MLB.md)** (`streamlit run mlb_app.py`).

# 🏆 World Cup 2026 Match Predictor

A football-specific machine-learning model that predicts international match
outcomes — **projected scoreline + win/draw/loss probabilities + a confidence
level** — and Monte-Carlo simulates the **2026 FIFA World Cup** to produce
championship odds. It ships with an interactive **Streamlit dashboard**.

The model is trained on ~49,000 internationals (1872 → today, auto-updated) and
its accuracy is proven by an out-of-sample backtest that includes the 2018 and
2022 World Cups.

<!-- Screenshots: run the dashboard (below) to view. -->

## What it does

- **Predict any match** — pick two teams, get the projected score, a scoreline
  probability heatmap, W/D/L probabilities, and a confidence band.
- **2026 title odds** — for every surviving team, the probability of reaching
  each remaining round and winning the trophy (Monte-Carlo over the bracket).
- **Live bracket & results** — the current tournament state is reconstructed
  automatically from the results data (who's out, who's alive, group standings).
- **Model card** — the full backtest metrics table and a calibration curve.

## How accurate is it?

Backtest on an **untouched test set** (all matches from 2018 onward; component
models see only earlier data). Lower RPS / log-loss / Brier is better.

| Model | RPS ↓ | Log-loss ↓ | Brier ↓ | Accuracy ↑ |
|---|---|---|---|---|
| Base rate (no skill) | 0.228 | 1.051 | 0.634 | 47.8% |
| Elo (logistic) | 0.170 | 0.874 | 0.514 | 60.2% |
| Gradient boosting | 0.171 | 0.873 | 0.513 | 60.1% |
| Dixon-Coles | 0.180 | 0.903 | 0.532 | 58.4% |
| **Ensemble (shipped)** | **0.169** | **0.869** | **0.510** | **60.4%** |

On **World Cup matches only** (2018 & 2022, n=216) the ensemble scores RPS 0.191,
log-loss 0.953, accuracy 58.3%. The ensemble is well-calibrated (see the Model
Card tab). *(Numbers regenerate whenever you re-run training on fresh data.)*

## Quick start

```bash
pip install -r requirements.txt      # numpy, pandas, scipy, scikit-learn, streamlit, …

# (optional) rebuild the model + simulation from the latest data:
python scripts/train.py              # downloads data, backtests, trains, saves outputs/model.joblib
python scripts/simulate_wc2026.py    # simulates the 2026 bracket -> outputs/predictions_2026.json

streamlit run app.py                 # launch the dashboard
```

The trained artifacts under `outputs/` are committed, so `streamlit run app.py`
works immediately after cloning — the two scripts are only needed to refresh
against newer results.

## How it works

1. **Data** (`src/wcpredictor/data.py`) — Mart Jürisoo's open
   [*international results*](https://github.com/martj42/international_results)
   dataset (results + shootouts), cleaned with a small team-name lineage map and
   a tournament-importance weighting.
2. **Elo ratings** (`elo.py`) — a custom, sequential (leak-free) Elo with
   margin-of-victory and match-importance K-scaling and home advantage.
3. **Features** (`features.py`) — all computed strictly *as-of* the match: Elo
   and Elo difference, rolling recent form (goals & points), rest days, venue and
   importance flags.
4. **Models** (`models/`)
   - **Dixon-Coles** time-weighted bivariate Poisson → the full scoreline matrix
     (projected score & score confidence).
   - **Gradient boosting** on the engineered features → outcome & goal counts.
   - **Elo-logistic** and **base-rate** baselines.
   - **Ensemble** — a log-loss-optimal blend + temperature calibration. The final
     scoreline is the Dixon-Coles matrix rescaled to the ensemble's (more
     accurate) W/D/L probabilities, so the score and the odds always agree.
5. **Tournament simulator** (`tournament.py`) — reconstructs the live knockout
   state from the data (shootout-aware) and Monte-Carlo plays out the remaining
   bracket thousands of times.

## "Confidence level" — what it means

Each prediction reports:

- **W/D/L probabilities** and the **most-likely outcome**;
- a **confidence band** — High (>60%), Medium (45-60%), Low (<45%) — from the top
  outcome probability;
- the **most-likely exact scoreline** and its probability, plus the top few
  scorelines and expected goals.

## The 2026 bracket

A flat results file doesn't encode who-plays-who in *future* rounds, so the
simulator seeds the surviving teams into a standard bracket by Elo rating by
default. To forecast the **exact** official bracket, list the real pairings under
`knockout_order` in `config/wc2026.yaml`. Completed matches are always taken from
the live data regardless.

## Project structure

```
app.py                      Streamlit dashboard
config/wc2026.yaml          editable 2026 bracket / host config
scripts/train.py            train + backtest + save model & metrics
scripts/simulate_wc2026.py  simulate the tournament -> predictions JSON
src/wcpredictor/            data, elo, features, models/, predict, tournament, backtest, viz
tests/                      pytest unit + integration tests
outputs/                    committed model + metrics + predictions + calibration plot
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Data & credits

Match data: [martj42/international_results](https://github.com/martj42/international_results)
(open data). Modelling builds on Dixon & Coles (1997) and the World Football Elo
Ratings scheme. This is a statistical model for entertainment and analysis — not
betting advice.
