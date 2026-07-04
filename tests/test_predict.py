import numpy as np
import pytest

from wcpredictor.paths import MODEL_PATH
from wcpredictor.predict import Predictor

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="model not trained yet (run scripts/train.py)")


def test_predictor_predict():
    p = Predictor.load()
    pred = p.predict("Spain", "France", neutral=True)
    assert abs((pred.p_home + pred.p_draw + pred.p_away) - 1.0) < 1e-6
    assert abs(pred.score_matrix.sum() - 1.0) < 1e-6
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.confidence_label in {"Low", "Medium", "High"}


def test_scoreline_consistent_with_outcome():
    # The rescaled matrix's win/draw/loss regions match the headline probs.
    p = Predictor.load()
    pred = p.predict("Brazil", "Argentina", neutral=True)
    M = pred.score_matrix
    home_region = np.tril(M, -1).sum()
    away_region = np.triu(M, 1).sum()
    draw_region = np.trace(M)
    assert abs(home_region - pred.p_home) < 1e-6
    assert abs(draw_region - pred.p_draw) < 1e-6
    assert abs(away_region - pred.p_away) < 1e-6
