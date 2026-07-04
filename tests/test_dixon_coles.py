import numpy as np

from wcpredictor.models.dixon_coles import DixonColesModel, dc_region_probs, dc_score_matrix


def test_score_matrix_normalised():
    M = dc_score_matrix(1.6, 1.1, -0.05, max_goals=10)
    assert M.shape == (11, 11)
    assert abs(M.sum() - 1.0) < 1e-9
    assert (M >= 0).all()


def test_region_probs_sum_to_one():
    lam = np.array([1.5, 2.0, 0.7])
    mu = np.array([1.1, 0.9, 2.2])
    ph, pd_, pa = dc_region_probs(lam, mu, rho=-0.05)
    total = ph + pd_ + pa
    assert np.allclose(total, 1.0, atol=1e-9)
    assert (ph >= 0).all() and (pd_ >= 0).all() and (pa >= 0).all()


def test_predict_match_strong_vs_weak():
    dc = DixonColesModel()
    dc.attack_ = {"Strong": 0.7, "Weak": -0.7}
    dc.defense_ = {"Strong": -0.4, "Weak": 0.4}
    dc.home_adv_ = 0.2
    dc.rho_ = -0.05
    dc.teams_ = ["Strong", "Weak"]
    p = dc.predict_match("Strong", "Weak", neutral=True)
    assert abs((p.p_home + p.p_draw + p.p_away) - 1.0) < 1e-9
    assert abs(p.score_matrix.sum() - 1.0) < 1e-9
    assert p.p_home > p.p_away
    i, j = p.most_likely_score()
    assert i >= j  # strong team projected to score at least as many
