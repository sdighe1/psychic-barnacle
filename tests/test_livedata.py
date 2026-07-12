"""Live-data layer: team/player resolution, statsapi parsing, manual slate."""
import mlbpredictor.livedata as ld
from mlbpredictor.livedata import (GameInput, load_slate_file, parse_statsapi_schedule,
                                   resolve_player, resolve_team)


def test_resolve_team():
    assert resolve_team("LAD") == "LAN"
    assert resolve_team("SF") == "SFN"
    assert resolve_team("NYY") == "NYA"
    assert resolve_team("WSH") == "WAS"
    assert resolve_team("SFN") == "SFN"        # already a retro code


def test_resolve_player_retro_id_passthrough():
    assert resolve_player("acunr001") == "acunr001"   # no register lookup needed
    assert resolve_player("kersc001") == "kersc001"
    assert resolve_player("") is None


def test_parse_statsapi_schedule(monkeypatch):
    # Avoid the register (network); map the ids/names deterministically.
    monkeypatch.setattr(ld, "retro_for_mlbam",
                        lambda i: {545361: "troum001", 660271: "ohtas001"}.get(int(i)) if i else None)
    monkeypatch.setattr(ld, "retro_for_name", lambda n: None)
    payload = {"dates": [{"games": [{
        "gamePk": 1,
        "teams": {
            "home": {"team": {"id": 119}, "probablePitcher": {"id": 660271, "fullName": "S O"}},
            "away": {"team": {"id": 108}, "probablePitcher": {"id": 545361, "fullName": "M T"}},
        },
        "lineups": {"homePlayers": [{"id": 660271}], "awayPlayers": [{"id": 545361}]},
    }]}]}
    games = parse_statsapi_schedule(payload)
    assert len(games) == 1
    g = games[0]
    assert g.home_team == "LAN" and g.away_team == "ANA"
    assert g.home_sp == "ohtas001" and g.away_sp == "troum001"
    assert g.home_lineup == ["ohtas001"] and g.source == "statsapi"


def test_load_slate_file(tmp_path):
    p = tmp_path / "slate.yaml"
    p.write_text(
        "games:\n"
        "  - home: LAD\n    away: SD\n    home_sp: kersc001\n    away_sp: darvy001\n"
        "  - home: NYY\n    away: BOS\n    home_sp: coleg001\n    away_sp: crawk001\n"
        "    home_lineup: [judga001, sotoj001, stang001, torrg001, rizza001,\n"
        "                  lemad001, volpa001, wella001, cabro002]\n")
    games = load_slate_file(p)
    assert len(games) == 2
    assert games[0].home_team == "LAN" and games[0].away_team == "SDN"
    assert games[0].home_sp == "kersc001"
    assert len(games[1].home_lineup) == 9


def test_fill_defaults_uses_predictor(monkeypatch):
    class FakeFB:
        default_lineups = {"LAN": [f"p{i}" for i in range(9)]}
        default_starter = {"SDN": "searj001"}
        default_park = {"LAN": "LOS03"}

    class FakePred:
        fb = FakeFB()

    g = GameInput(home_team="LAN", away_team="SDN", home_sp="kersc001", away_sp="")
    g = ld.fill_defaults(g, FakePred())
    assert len(g.home_lineup) == 9          # filled from defaults
    assert g.away_sp == "searj001"          # filled starter
    assert g.park == "LOS03"
