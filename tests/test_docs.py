"""Docs facts an operator acts on, checked against production code.

The opening-night runbook (BUCKETVALUE_PROJECT_INSTRUCTIONS.md, section 4) and
docs/data_notes.md state tip and freeze times that decide when the Oct 20 run
must finish. They are checked here against the committed cdn schedule payload
(fetched 2026-10-08) read through cdn.parse_schedule, pipeline.state_games and
pipeline.frozen_ids, so a wrong clock in the docs, or a change to the freeze
lead the docs do not follow, fails the suite. data_notes once said the opener
tipped at 7pm ET; it tips at 3pm ET, so the first freeze point is 2:55pm ET.

docs/runbook.md section 8 says BUCKETVALUE_LOG must be exactly a level name and
has no fallback; that is checked against bucketlib.cache itself, loaded in a
fresh interpreter (build 7's review, R7-2).
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
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


# What a new process does with BUCKETVALUE_LOG: bucketlib.cache loaded once per
# value, with the logger as a fresh process has it (no handler, no level).
_LOG_CHILD = r"""
import importlib, json, logging, os, sys
sys.path.insert(0, sys.argv[1])
out = []
for value in json.loads(sys.argv[2]):
    log = logging.getLogger("bucketvalue")
    for h in list(log.handlers):
        log.removeHandler(h)
    log.setLevel(logging.NOTSET)
    sys.modules.pop("bucketlib.cache", None)
    os.environ["BUCKETVALUE_LOG"] = value
    try:
        importlib.import_module("bucketlib.cache")
        first = logging.getLevelName(log.level)
    except ValueError as e:
        first = "ValueError: %s" % e
        importlib.import_module("bucketlib.cache")
    out.append([value, first, logging.getLevelName(log.getEffectiveLevel())])
print(json.dumps(out))
"""


def test_runbook_bucketvalue_log_names_are_what_cache_accepts_and_it_has_no_fallback():
    """docs/runbook.md section 8 (build 7's review, R7-2): BUCKETVALUE_LOG
    must be exactly a level name and has no fallback, because bucketlib.cache
    hands it to logging's setLevel as the package first loads. Checked
    against the real module in a fresh interpreter, which loads it once per
    value as a new process would: each name the runbook's settings table
    lists loads at that level, in upper and in lower case; each bad value
    the runbook names raises ValueError (Unknown level) on that first load,
    and the next load in the same process works, with the log at WARNING
    (the module attached its handler before the raise)."""
    text = " ".join((ROOT / "docs" / "runbook.md").read_text().split())
    row = re.search(r"\| `BUCKETVALUE_LOG` \| `WARNING` \| ([^|]*) \|", text)
    assert row, "the runbook's settings table has no BUCKETVALUE_LOG row"
    names = re.findall(r"`([A-Z]+)`", row.group(1))
    assert names == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    m = re.search(r"`BUCKETVALUE_LOG` is the other: it has no fallback\. Anything but a level "
                  r"name, such as (.+?), raises `ValueError: Unknown level`", text)
    assert m, "the runbook no longer says that BUCKETVALUE_LOG has no fallback"
    bad = re.findall(r'`"([^"`]*)"`', m.group(1))
    assert bad == ["INFO ", "INF0", "verbose", ""]
    values = names + [n.lower() for n in names] + bad
    env = {k: v for k, v in os.environ.items() if k != "BUCKETVALUE_LOG"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-c", _LOG_CHILD, str(ROOT), json.dumps(values)],
                       capture_output=True, text=True, timeout=180, cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stderr[-800:]
    got = {value: (first, effective) for value, first, effective in json.loads(r.stdout)}
    for n in names:
        assert got[n] == got[n.lower()] == (n, n), n
    for b in bad:
        assert got[b] == ("ValueError: Unknown level: %r" % b.upper(), "WARNING"), b
