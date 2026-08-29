"""The Streamlit app loads and survives a couple of drafted picks (headless)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app() -> AppTest:
    return AppTest.from_file(str(REPO / "app.py"), default_timeout=120)


@pytest.mark.skipif(not (REPO / "outputs" / "projections.csv").exists(),
                    reason="projections artifact not built")
def test_app_loads_clean():
    at = _app().run()
    assert not at.exception
    assert at.title[0].value.startswith("🏈")
    assert len(at.tabs) == 5


@pytest.mark.skipif(not (REPO / "outputs" / "projections.csv").exists(),
                    reason="projections artifact not built")
def test_app_handles_drafted_state():
    proj = pd.read_csv(REPO / "outputs" / "projections.csv")
    ids = proj["player_id"].astype(str).tolist()
    at = _app()
    at.session_state["picks"] = {ids[0]: {"price": 50, "mine": True},
                                 ids[1]: {"price": 40, "mine": False}}
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "My budget left" in labels
