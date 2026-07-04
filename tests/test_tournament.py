import pandas as pd

from wcpredictor.tournament import decide_winner, reconstruct_state, seed_order


def test_seed_order():
    assert seed_order(2) == [1, 2]
    assert seed_order(4) == [1, 4, 2, 3]
    assert seed_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    s = seed_order(16)
    assert len(s) == 16 and sorted(s) == list(range(1, 17))


def test_decide_winner():
    so = pd.DataFrame({"home_team": ["X"], "away_team": ["Y"], "winner": ["Y"]})
    assert decide_winner("X", "Y", 2, 1, so) == "X"
    assert decide_winner("X", "Y", 0, 3, so) == "Y"
    # A drawn knockout is resolved by the shootout table.
    assert decide_winner("X", "Y", 1, 1, so) == "Y"


def test_reconstruct_state_live():
    state = reconstruct_state()
    alive, elim = set(state["alive"]), set(state["eliminated"])
    assert len(alive) >= 1
    assert alive.isdisjoint(elim)
    # Each completed knockout has a winner among its two teams.
    for m in state["completed_knockout"]:
        assert m["winner"] in (m["home"], m["away"])
