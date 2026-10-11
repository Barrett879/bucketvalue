"""Docs facts an operator acts on, checked against production code.

The opening-night runbook (BUCKETVALUE_PROJECT_INSTRUCTIONS.md, section 4) and
docs/data_notes.md state tip and freeze times that decide when the Oct 20 run
must finish. They are checked here against the committed cdn schedule payload
(fetched 2026-10-08) read through cdn.parse_schedule, pipeline.state_games and
pipeline.frozen_ids, so a wrong clock in the docs, or a change to the freeze
lead the docs do not follow, fails the suite. data_notes once said the opener
tipped at 7pm ET; it tips at 3pm ET, so the first freeze point is 2:55pm ET.
"""
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from bucketlib import pipeline as P
from bucketlib.sources import cdn
from bucketlib.util import ET

ROOT = Path(__file__).resolve().parent.parent
OPENER = dt.date(2026, 10, 20)
CLOCK = r"(\d{1,2})(?::(\d{2}))?\s?(am|pm) ET"


def _slate() -> pd.DataFrame:
    payload = json.loads((ROOT / "tests" / "fixtures" / "cdn" / "schedule_2026-27_trim.json").read_text())
    return P.state_games(cdn.parse_schedule(payload), OPENER)


def _hm(h: str, m: str | None, ampm: str) -> tuple[int, int]:
    return int(h) % 12 + (12 if ampm == "pm" else 0), int(m or 0)


def test_data_notes_opener_slate_matches_the_schedule():
    text = " ".join((ROOT / "docs" / "data_notes.md").read_text().split())
    m = re.search(r"opener (\d{4}-\d{2}-\d{2}) \(([^)]*)\)", text)
    assert m, "data_notes.md no longer states the opener slate"
    assert dt.date.fromisoformat(m.group(1)) == OPENER
    documented = [(a, h, _hm(*c)) for a, h, *c in
                  re.findall(r"([A-Z]{2,3})@([A-Z]{2,3}) " + CLOCK, m.group(2))]
    slate = _slate()
    tips = pd.to_datetime(slate["tip_utc"], utc=True).dt.tz_convert(ET)
    actual = [(r.away, r.home, (t.hour, t.minute)) for r, t in zip(slate.itertuples(), tips)]
    assert actual == [("BOS", "DET", (15, 0)), ("PHI", "NYK", (19, 0)), ("OKC", "SAS", (21, 30))]
    assert documented == actual


def test_runbook_first_freeze_point_is_where_the_pipeline_freezes():
    text = " ".join((ROOT / "BUCKETVALUE_PROJECT_INSTRUCTIONS.md").read_text().split())
    m = re.search(r"Opening-night runbook:.*?first freeze point is " + CLOCK + r" Oct 20", text)
    assert m, "the opening-night runbook does not state the first freeze point"
    h, mi = _hm(*m.groups())
    point = dt.datetime(2026, 10, 20, h, mi, tzinfo=ET)
    slate = _slate()
    first = slate["game_id"].iloc[0]
    assert P.frozen_ids(slate, point - dt.timedelta(minutes=1)) == set()
    assert P.frozen_ids(slate, point) == {first}
