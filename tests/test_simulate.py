"""Monte-Carlo simulator sanity: run environment, discrimination, valid markets."""
import numpy as np

from mlbpredictor.simulate import precompute_matchups, TeamPack, simulate_game

LEAGUE = np.array([0.14, 0.045, 0.004, 0.032, 0.086, 0.011, 0.227, 0.455])
LEAGUE = LEAGUE / LEAGUE.sum()


def _pack(lineup, starter, bullpen, pf=1.0):
    sp_pairs = [(b, starter) for b in lineup]
    bp_pairs = [(b, bullpen) for b in lineup]
    sp, bp = precompute_matchups(sp_pairs, bp_pairs, LEAGUE, pf)
    return TeamPack(sp, bp, 27)


def _league_pack():
    return _pack([LEAGUE] * 9, LEAGUE, LEAGUE)


def test_league_average_run_environment():
    p = _league_pack()
    res = simulate_game(p, p, n_sims=3000, seed=1)
    a, h = res.exp_runs()
    assert 4.0 <= (a + h) / 2 <= 5.1               # MLB is ~4.5 runs/team
    assert 8.0 <= res.total().mean() <= 10.0


def test_identical_teams_are_a_cointoss():
    p = _league_pack()
    res = simulate_game(p, p, n_sims=4000, seed=2)
    assert 0.45 <= res.p_home_win() <= 0.55


def test_markets_are_valid_probabilities():
    p = _league_pack()
    res = simulate_game(p, p, n_sims=2000, seed=3)
    for x in (res.p_home_win(), res.p_over(8.5), res.p_home_cover(-1.5)):
        assert 0.0 <= x <= 1.0
    # no unresolved ties should leak into the record
    assert (res.home_runs == res.away_runs).mean() < 0.02


def test_better_lineup_scores_more_and_wins():
    strong = LEAGUE.copy(); strong[:4] *= 1.4; strong[6] *= 0.8; strong /= strong.sum()
    ace = LEAGUE.copy(); ace[:4] *= 0.8; ace[6] *= 1.25; ace /= ace.sum()
    strong_team = _pack([strong] * 9, LEAGUE, LEAGUE)     # strong bats vs avg pitching
    weak_team = _pack([LEAGUE] * 9, ace, ace)             # avg bats vs an ace
    res = simulate_game(strong_team, weak_team, n_sims=3000, seed=4)  # away=strong
    away_runs, home_runs = res.exp_runs()
    assert away_runs > home_runs + 1.0
    assert res.p_home_win() < 0.4                          # away (strong) favored


def test_tto_penalty_increases_offense_late():
    import numpy as np
    pairs = [(LEAGUE, LEAGUE)] * 9
    vs_sp, _ = precompute_matchups(pairs, pairs, LEAGUE, 1.0, tto_factors=(0.9, 1.0, 1.15))
    pmf1 = np.diff(vs_sp[0][0], prepend=0)      # 1st time through
    pmf3 = np.diff(vs_sp[0][2], prepend=0)      # 3rd time through
    assert pmf3[7] < pmf1[7]                     # fewer OUTs the 3rd time
    assert pmf3[3] > pmf1[3]                     # more HR the 3rd time
    # centered factors keep the league run environment intact
    sp, bp = precompute_matchups(pairs, pairs, LEAGUE, 1.0, tto_factors=(0.97, 1.0, 1.05))
    res = simulate_game(TeamPack(sp, bp, 27), TeamPack(sp, bp, 27), n_sims=3000, seed=2)
    a, h = res.exp_runs()
    assert 4.0 <= (a + h) / 2 <= 5.1


def test_ace_records_more_strikeouts():
    ace = LEAGUE.copy(); ace[6] *= 1.6; ace[:4] *= 0.7; ace /= ace.sum()
    ace_team = _pack([LEAGUE] * 9, ace, LEAGUE)
    avg_team = _pack([LEAGUE] * 9, LEAGUE, LEAGUE)
    res = simulate_game(avg_team, ace_team, n_sims=2000, seed=5)   # home has the ace
    assert res.home_pitch["sp_k"].mean() > res.away_pitch["sp_k"].mean() + 1.5
