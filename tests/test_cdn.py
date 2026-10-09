"""cdn schedule + player index parsing, on trimmed real payloads (2026-10-08)."""
import datetime as dt
import json
from pathlib import Path

import pytest

from bucketlib.sources import cdn

FIX = Path(__file__).parent / "fixtures" / "cdn"


def sched():
    return cdn.parse_schedule(json.loads((FIX / "schedule_2026-27_trim.json").read_text()))


def test_opener_tip_is_3pm_et_not_the_wall_clock_misread_as_utc():
    df = sched()
    g = df[df.game_id == "0022600001"].iloc[0]
    assert (g.away, g.home) == ("BOS", "DET")
    assert g.tip_utc.isoformat() == "2026-10-20T19:00:00+00:00"
    assert (g.tip_et.hour, g.tip_et.minute) == (15, 0)
    assert g.game_date == dt.date(2026, 10, 20)
    assert g.game_type == "RS"


def test_late_tip_keeps_its_et_date():
    df = sched()
    late = df[(df.game_type == "RS") & (df.tip_utc.map(lambda t: t.date()) == dt.date(2026, 10, 21))]
    assert len(late) >= 1
    assert (late.game_date == dt.date(2026, 10, 20)).all()


def test_non_nba_preseason_opponent_dropped_and_counted():
    df = sched()
    dropped = df.attrs["dropped_non_nba"]
    assert len(dropped) == 1 and dropped[0] not in set(df.game_id)
    assert df.home.isin(["NON_NBA"]).sum() == 0


def test_tbd_placeholder_and_cup_final():
    df = sched()
    tbd = df[df.is_tbd]
    assert len(tbd) >= 1 and tbd.home.isna().all()
    assert (df.game_id == "0062600001").any()
    assert df.loc[df.game_id == "0062600001", "game_type"].iloc[0] == "CUP_FINAL"


def test_date_mismatch_raises():
    payload = json.loads((FIX / "schedule_2026-27_trim.json").read_text())
    g = payload["leagueSchedule"]["gameDates"][0]["games"][0]
    g["gameDateEst"] = "1999-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        cdn.parse_schedule(payload)


def test_player_index_two_way_and_names():
    df = cdn.parse_player_index(json.loads((FIX / "playerIndex_trim.json").read_text()))
    assert df.person_id.is_unique and df.team.notna().all()
    assert df.two_way.sum() >= 1
    jokic = df[df.person_id == 203999].iloc[0]
    assert jokic["name"] == "Nikola Jokić" and jokic.team == "DEN"
