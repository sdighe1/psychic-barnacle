"""Prediction models for wcpredictor.

Every model exposes a common shape so they can be compared head-to-head in the
backtest and blended in the ensemble:

- ``fit(matches, ...)``
- ``predict_match(home, away, neutral=False, ...) -> Prediction``

See :class:`wcpredictor.predict.Prediction` for the unified output container.
"""
