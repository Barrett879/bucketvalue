"""ET dates and season labels: the bugs these pin were live on the siblings."""
import datetime as dt
import re
from pathlib import Path

import pytest

from bucketlib import util

REPO = Path(__file__).resolve().parent.parent


def test_evening_tip_is_the_same_et_date_even_past_utc_midnight():
    # 2026-11-10 7:00pm ET is 2026-11-11 00:00 UTC (EST, UTC-5).
    assert util.et_date_of("2026-11-11T00:00:00Z") == dt.date(2026, 11, 10)
    # Under daylight time an 8pm ET tip is also past UTC midnight.
    assert util.et_date_of("2026-10-21T00:00:00Z") == dt.date(2026, 10, 20)


def test_naive_datetimes_are_rejected_not_guessed():
    with pytest.raises(ValueError):
        util.to_et(dt.datetime(2026, 11, 10, 19, 0))


def test_season_labels():
    assert util.season_for_date("2026-10-20") == "2026-27"
    assert util.season_for_date("2027-06-15") == "2026-27"
    assert util.season_for_date("2026-07-31") == "2025-26"
    assert util.season_end_year("2026-27") == 2027
    assert util.season_str(2027) == "2026-27"
    assert util.seasons_between("2014-15", "2016-17") == ["2014-15", "2015-16", "2016-17"]
    with pytest.raises(ValueError):
        util.season_end_year("2026-28")


def _local_clock_calls(src: str) -> list[int]:
    """Line numbers of date.today(), datetime.today(), argless datetime.now()
    and time.localtime() CALLS (comments and docstrings do not count)."""
    import ast
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr, base = node.func.attr, node.func.value
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            if (attr == "today" and name in ("date", "datetime")) or \
               (attr == "now" and name == "datetime" and not node.args and not node.keywords) or \
               (attr == "localtime" and name == "time"):
                out.append(node.lineno)
    return out


def test_no_server_local_clock_anywhere():
    """These read the server's zone, which is UTC on GitHub Actions and
    Streamlit Cloud. DiamondValue's evening refresh landed on the wrong game
    date for months because of exactly this. Use util.today_et()/now_et()."""
    offenders = []
    for p in REPO.rglob("*.py"):
        if ".venv" in p.parts:
            continue
        offenders += [f"{p.relative_to(REPO)}:{n}" for n in _local_clock_calls(p.read_text())]
    assert not offenders, offenders


def test_clock_check_catches_the_bug():
    """Mutation check: the detector must fire on the real patterns."""
    bad = "import datetime as dt\nimport time\nx = dt.date.today()\ny = dt.datetime.now()\nz = time.localtime()\n"
    assert _local_clock_calls(bad) == [3, 4, 5]
    assert _local_clock_calls("import datetime as dt\nx = dt.datetime.now(dt.timezone.utc)\n") == []
