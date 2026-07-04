"""wcpredictor — a football (soccer) match outcome & scoreline predictor.

The package is organised in three layers:

1. Data & features   -> :mod:`wcpredictor.data`, :mod:`wcpredictor.elo`,
                        :mod:`wcpredictor.features`
2. Prediction models -> :mod:`wcpredictor.models` (dixon_coles, gboost,
                        baselines, ensemble) + :mod:`wcpredictor.predict`
3. Tournament sim    -> :mod:`wcpredictor.tournament`

The public entry point most callers want is :func:`wcpredictor.predict.load_predictor`,
which returns a fitted model exposing ``predict_match(home, away, neutral=...)``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
