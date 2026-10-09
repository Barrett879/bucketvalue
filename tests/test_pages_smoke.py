"""Every page renders without an exception (Streamlit AppTest, offline).

The live layer is replaced with the committed cdn fixtures, so this runs with
no network. A syntax error, a bad import, or an exception on a normal path
fails here instead of in front of a visitor (with showErrorDetails="type"
the visitor would only see a generic error)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from bucketlib import live
from bucketlib.sources import cdn

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "cdn"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    sched = cdn.parse_schedule(json.loads((FIX / "schedule_2026-27_trim.json").read_text()))
    roster = cdn.parse_player_index(json.loads((FIX / "playerIndex_trim.json").read_text()))
    monkeypatch.setattr(live, "schedule", lambda: live.Live(sched, 0.0))
    monkeypatch.setattr(live, "roster", lambda: live.Live(roster, 0.0))
    monkeypatch.setattr(live, "odds_on", lambda d: live.Live({("BOS", "DET"): {"home_line": -1.5, "total": 220.5, "provider": "DraftKings"}}, 0.0))


def run(path, **params):
    at = AppTest.from_file(str(REPO / path), default_timeout=60)
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home_opening_night():
    at = run("app.py", date="2026-10-20")
    html = " ".join(m.value for m in at.markdown)
    assert "BOS" in html and "DET" in html and "DET -1.5" in html


def test_home_empty_date_and_preseason():
    run("app.py", date="2026-08-01")
    run("app.py", date="2026-10-09")


def test_game_page_and_bad_params():
    run("pages/Game.py", date="2026-10-20", away="BOS", home="DET")
    at = run("pages/Game.py", date="2026-10-20", away="<script>", home="DET")
    assert "<script>" not in " ".join(m.value for m in at.markdown)


def test_about():
    run("pages/About.py")
