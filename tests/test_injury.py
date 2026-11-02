"""Injury-report PDFs: every trap the 2026-10-08 research found, pinned on
real (unmodified) PDFs from ak-static.cms.nba.com saved under
tests/fixtures/injury/. Offline except the one @network test."""
import datetime as dt
import functools
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import http
from bucketlib.names import NameIndex
from bucketlib.sources import injury as I
from bucketlib.util import ET, UTC

FIX = Path(__file__).resolve().parent / "fixtures" / "injury"
REPO = Path(__file__).resolve().parent.parent
FIXTURES = sorted(p.name for p in FIX.glob("Injury-Report_*.pdf"))


@functools.lru_cache(maxsize=None)
def load(stem: str) -> I.ParsedReport:
    return I.parse_report((FIX / f"Injury-Report_{stem}.pdf").read_bytes())


def names(day) -> list[str]:
    return [s.filename.split("_", 2)[-1].removesuffix(".pdf") for s in I.slots_for_date(day)]


def et(*a) -> dt.datetime:
    return dt.datetime(*a, tzinfo=ET)


def test_fixture_budget_stays_small():
    assert sum(p.stat().st_size for p in FIX.iterdir()) < 1_500_000
    assert len(FIXTURES) >= 10


# ── (a) URLs and eras ────────────────────────────────────────────────────────
def test_archive_starts_2018_12_17_with_three_slots():
    assert names("2018-12-16") == []
    assert names("2018-12-17") == ["01PM", "05PM", "08PM"]
    assert I.slots_for_date("2018-12-17")[1].url == (
        "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2018-12-17_05PM.pdf")


def test_bubble_restart_used_other_slots():
    # Verified 2020-08-10: exactly 11AM/02PM/05PM; 01PM and 08PM were 403.
    assert names("2020-08-10") == ["11AM", "02PM", "05PM"]
    assert set(names("2020-07-20")) == {"11AM", "01PM", "02PM", "05PM", "08PM"}  # margin: both
    assert names("2020-12-23") == ["01PM", "05PM", "08PM"]


def test_hourly_era_twelve_hour_clock():
    n = names("2022-01-15")
    assert len(n) == 24 and n[0] == "12AM" and "12PM" in n
    assert not any(x.startswith("00") for x in n)


def test_spring_forward_slots_do_not_exist():
    # 2023-03-12_02AM and 2026-03-08_02_00AM/02_30AM returned 403: the hour
    # never happened. Stepping in UTC never generates it.
    assert "02AM" not in names("2023-03-12") and len(names("2023-03-12")) == 23
    q = names("2026-03-08")
    assert not any(x.startswith("02_") and x.endswith("AM") for x in q) and len(q) == 92
    assert "01_45AM" in q and "03_00AM" in q


def test_fall_back_hour_is_one_object_named_once():
    n = names("2023-11-05")
    assert len(n) == len(set(n)) == 24
    one_am = next(s for s in I.slots_for_date("2023-11-05") if s.filename.endswith("_01AM.pdf"))
    assert one_am.nominal_et.utcoffset() == dt.timedelta(hours=-4)  # the EDT pass (LM 05:30 GMT)


def test_quarter_hour_switch_tries_both_formats_near_the_boundary():
    assert "05PM" in names("2025-12-20") and "05_00PM" not in names("2025-12-20")
    both = names("2025-12-22")
    assert "12AM" in both and "10_00AM" in both and "05PM" in both and "05_00PM" in both
    assert "05PM" not in names("2025-12-24") and len(names("2026-01-15")) == 96


def test_slot_filename_round_trip_and_rejects():
    for name in ("Injury-Report_2023-03-12_12AM.pdf", "Injury-Report_2023-03-12_12PM.pdf",
                 "Injury-Report_2026-01-15_12_45PM.pdf", "Injury-Report_2026-10-07_09_45AM.pdf"):
        s = I.parse_slot_filename(name)
        assert I.slot_filename(s.nominal_et, s.quarter_format) == name
    assert I.parse_slot_filename("Injury-Report_2026-10-07_10_45AM.pdf").nominal_et == et(2026, 10, 7, 10, 45)
    assert I.parse_slot_filename("Injury-Report_2023-03-12_12AM.pdf").nominal_et.hour == 0
    for bad in ("Injury-Report_2026-10-07_00AM.pdf", "Injury-Report_2026-10-07_10_40AM.pdf",
                "Injury-Report_2026-10-07_13PM.pdf", "report.pdf"):
        with pytest.raises(ValueError):
            I.parse_slot_filename(bad)


def test_walk_back_crosses_days_and_dst():
    first = [s.filename for s in I.slots_descending(et(2019, 1, 15, 12, 55), inclusive=False)][:2]
    assert first == ["Injury-Report_2019-01-14_08PM.pdf", "Injury-Report_2019-01-14_05PM.pdf"]
    walk = [s for s in I.slots_descending(et(2026, 3, 8, 3, 15))][:3]
    assert [s.filename[-11:] for s in walk] == ["03_15AM.pdf", "03_00AM.pdf", "01_45AM.pdf"]
    # 15 real minutes. (Subtracting the ET values directly would say 1:15:
    # Python treats two datetimes sharing a tzinfo as naive wall clocks.)
    assert walk[1].nominal_et.astimezone(UTC) - walk[2].nominal_et.astimezone(UTC) == \
        dt.timedelta(minutes=15)


# ── (b) fetching and the raw store ───────────────────────────────────────────
def test_fetch_report_demands_a_whole_pdf(monkeypatch):
    seen = {}
    body = {"b": b"%PDF-1.4\n...\n%%EOF\n"}

    def fake(url, **kw):
        seen.update(kw, url=url)
        return body["b"]
    monkeypatch.setattr(I.http, "fetch", fake)
    assert I.fetch_report("u") == body["b"] and seen["expect"] == "pdf"
    assert seen["retries"] == 2 and seen["timeout"] == 30.0
    I.fetch_report("u", timeout=5, retries=0)
    assert seen["retries"] == 0 and seen["timeout"] == 5
    # A truncated body still starts %PDF: http's own check passes it.
    body["b"] = (FIX / "Injury-Report_2022-01-15_05PM.pdf").read_bytes()[:30_000]
    with pytest.raises(http.BadPayload, match="truncated"):
        I.fetch_report("u")


def test_parse_refuses_an_s3_xml_body_saved_as_pdf():
    xml = b"<?xml version='1.0'?><Error><Code>AccessDenied</Code></Error>"
    with pytest.raises(ValueError, match="not a PDF"):
        I.parse_report(xml)


def _old_slot():
    return I.parse_slot_filename("Injury-Report_2024-01-15_05PM.pdf")


def test_store_absent_is_none_and_checkpointed(tmp_path):
    calls = []

    def absent(url):
        calls.append(url)
        raise http.Absent(url, "HTTP 403 S3 AccessDenied")
    store = I.ReportStore(tmp_path, fetcher=absent)
    assert store.get(_old_slot()) is None
    store.flush()
    again = I.ReportStore(tmp_path, fetcher=absent)  # a resumed run
    assert again.get(_old_slot()) is None and len(calls) == 1
    assert again.stats["absent_cached"] == 1


def test_store_does_not_remember_a_recent_absence(tmp_path):
    calls = []

    def absent(url):
        calls.append(url)
        raise http.Absent(url, "403")
    recent = I.Slot(I.now_et().replace(second=0, microsecond=0), "Injury-Report_x.pdf")
    store = I.ReportStore(tmp_path, fetcher=absent)
    assert store.get_bytes(recent) is None and store.get_bytes(recent) is None
    assert len(calls) == 2  # may still be published


def test_store_blocked_is_never_absent(tmp_path):
    def blocked(url):
        raise http.Blocked(url, "HTTP 403 text/html: Akamai edge block")
    with pytest.raises(http.Blocked):
        I.ReportStore(tmp_path, fetcher=blocked).get(_old_slot())


def test_store_caches_pdf_and_enforces_budget(tmp_path):
    pdf = (FIX / "Injury-Report_2022-01-15_05PM.pdf").read_bytes()
    calls = []

    def ok(url):
        calls.append(url)
        return pdf
    store = I.ReportStore(tmp_path, fetcher=ok, budget=1)
    rep = store.get(_old_slot())
    assert rep.meta["layout"] == 2 and (tmp_path / "2023-24" / _old_slot().filename).exists()
    assert I.ReportStore(tmp_path, fetcher=ok).get(_old_slot()) is not None and len(calls) == 1
    with pytest.raises(RuntimeError, match="budget"):
        store.get(I.parse_slot_filename("Injury-Report_2024-01-15_06PM.pdf"))


# ── (c) parsing: the three layouts ───────────────────────────────────────────
def test_layout1_2018_12_20():
    meta, rows, team = load("2018-12-20_05PM")
    assert meta["layout"] == 1 and meta["creator"] is None and "iTextSharp" in meta["producer"]
    assert meta["report_ts_et"] == et(2018, 12, 20, 17, 30) and meta["n_pages"] == 4
    assert len(rows) == 93 and rows.status.value_counts().to_dict() == {
        "Out": 72, "Questionable": 18, "Probable": 3}
    assert sorted(rows.game_date.unique()) == [dt.date(2018, 12, 20), dt.date(2018, 12, 21)]
    r = rows.set_index("player_raw")
    assert r.loc["Ennis III, James", "reason_raw"] == "Injury/Illness - Right Hamstring Strain"
    assert r.loc["Edwards, Vincent", "reason_cat"] == "g_league"           # 'G League Team' / '-'
    assert r.loc["Broekhoff, Ryan", "prev_status"] == "Available"
    assert r.loc["Anthony, Carmelo", "reason_cat"] == "not_with_team"
    assert set(team.kind) == {"not_yet_submitted"} and len(team) == 4


def test_layout1_wrapped_team_name_and_reason():
    # extract_text() prints 'SAC@MIN Minnesota\nTimberwolves' and splits
    # Murray's reason around his status; positions keep both whole.
    meta, rows, team = load("2018-12-17_05PM")
    mn = rows[rows.matchup == "SAC@MIN"]
    assert set(mn.team) == {"MIN", "SAC"} and (mn.team == "MIN").sum() == 5  # incl. page 2
    r = rows.set_index("player_raw")
    assert r.loc["Murray, Dejounte", "reason_raw"] == (
        "Injury/Illness - Torn right anterior cruciate ligament")
    assert r.loc["Oubre Jr., Kelly", "reason_cat"] == "trade_pending"
    assert r.loc["Robinson III, Glenn", "team"] == "DET"
    k = dict(zip(zip(team.matchup, team.team), team.kind))
    assert k[("POR@LAC", "POR")] == "all_players_available"
    assert k[("CLE@IND", "CLE")] == "not_yet_submitted"


def test_layout1_team_cell_on_a_two_line_row():
    # 'Milwaukee Bucks' sits on the TOP line of Giannis' two-line row, 4pt
    # above his status: the merged team cell is top-aligned, rows are centred.
    meta, rows, _ = load("2019-01-15_05PM")
    r = rows.set_index("player_raw")
    assert (r.loc["Antetokounmpo, Giannis", ["matchup", "team"]] == ["MIA@MIL", "MIL"]).all()
    assert r.loc["Antetokounmpo, Giannis", "reason_raw"].endswith("soreness/Left hip contusion")
    assert (r.loc["Baynes, Aron", ["matchup", "team"]] == ["TOR@BOS", "BOS"]).all()
    assert r.loc["Baynes, Aron", "game_date"] == dt.date(2019, 1, 16)


def test_layout2_2020_01_15():
    meta, rows, team = load("2020-01-15_05PM")
    assert meta["layout"] == 2 and meta["creator"] is None
    assert meta["report_ts_et"] == et(2020, 1, 15, 17, 30)
    assert len(rows) == 120 and rows.prev_status.isna().all()
    assert rows.status.value_counts().to_dict() == {
        "Out": 90, "Questionable": 14, "Probable": 13, "Available": 2, "Doubtful": 1}
    assert len(team) == 6


def test_bubble_report_is_on_the_hour():
    meta, rows, _ = load("2020-08-10_02PM")
    assert meta["report_ts_et"] == et(2020, 8, 10, 14, 0)
    bkn = rows[rows.matchup == "BKN@ORL"]
    assert bkn.game_time_et.iloc[0] == dt.time(13, 0) and bkn.game_time_raw.iloc[0] == "01:00 (ET)"
    assert (rows[rows.matchup == "OKC@PHX"].game_time_et == dt.time(14, 30)).all()


def test_title_time_not_filename_time():
    # File '..._05PM' holds the 5:30 PM report. Choosing by filename would
    # take the 7PM file (published 7:30) for a 7:00 PM tip.
    meta, rows, _ = load("2022-01-15_05PM")
    assert meta["title"] == "Injury Report: 01/15/22 05:30 PM"
    assert meta["report_ts_et"] == et(2022, 1, 15, 17, 30)
    assert abs(meta["created_et"] - meta["report_ts_et"]) < dt.timedelta(minutes=2)
    r = rows.set_index("player_raw")
    assert r.loc["McCollum, CJ", "reason_cat"] == "reconditioning"
    assert r.loc["Beal, Bradley", "reason_cat"] == "health_protocols"
    assert r.loc["Pokusevski, Aleksej", "reason_cat"] == "unspecified"  # reason '-'
    assert r.loc["Irving, Kyrie", "reason_cat"] == "other"               # 'Ineligible To Play'


def test_gembox_2023_24_reason_split_across_pages():
    # The research prototype's 1-2% mislabels ('Contusion Injury/Illness',
    # 'issue G League') were page-break tails glued to the next page's first
    # row. They belong to the last row of the previous page.
    meta, rows, _ = load("2023-12-01_05PM")
    assert meta["layout"] == 3 and "GemBox" in meta["creator"] and meta["n_pages"] == 9
    assert len(rows) == 126 and meta["n_carried_cells"] == 4
    assert meta["n_fallback_cells"] == 0
    for reason in rows.reason_raw:
        head, _, rest = reason.partition(" - ")
        assert not reason[:1].islower(), reason
        assert "Injury/Illness" not in rest and "G League" not in rest, reason
    for tail in ("issue", "Contusion", "Management", "Maintenance"):
        assert rows.reason_raw.str.endswith(tail).any()


def test_gembox_under_the_hourly_filename_keys_curry_and_horford():
    meta, rows, team = load("2025-12-21_05PM")
    assert meta["layout"] == 3 and meta["report_ts_et"] == et(2025, 12, 21, 17, 45)
    r = rows.set_index("player_raw")
    for p in ("Curry, Seth", "Horford, Al"):
        assert (r.loc[p, ["game_date", "matchup", "team"]] ==
                [dt.date(2025, 12, 22), "ORL@GSW", "GSW"]).all(), p
    assert ("ORL@GSW", "ORL") in set(zip(team.matchup, team.team))


def test_2026_01_15_quarter_hour_report():
    meta, rows, team = load("2026-01-15_05_00PM")
    assert meta["report_ts_et"] == et(2026, 1, 15, 17, 0) and meta["n_pages"] == 9
    assert len(rows) == 147
    assert rows.status.value_counts().to_dict() == {
        "Out": 105, "Questionable": 17, "Probable": 12, "Available": 7, "Doubtful": 6}
    edey = rows[rows.player_raw == "Edey, Zach"].iloc[0]
    assert (edey.matchup, edey.team, edey.status, edey.game_time_et) == (
        "MEM@ORL", "MEM", "Out", dt.time(14, 0))
    assert edey.reason_raw == "Injury/Illness - Left Ankle; Stress Reaction"
    # line-wrap hyphenation rejoined, ' - ' separators kept
    for p in ("Hayes-Davis, Nigel", "Prosper, Olivier-Maxence", "Lively II, Dereck",
              "Jones Garcia, David"):
        assert p in set(rows.player_raw), p
    assert not rows.reason_raw.str.contains(r"\w- \w").any()
    assert (rows.reason_raw == "G League - Two-Way").sum() > 10
    assert dt.date(2026, 1, 16) in set(rows.game_date)  # the 5 PM report lists tomorrow
    assert (rows.reason_cat.str.startswith("g_league")).sum() > 30


def test_spurs_stay_under_den_sas_when_denver_has_not_submitted():
    meta, rows, team = load("2026-03-12_12_00PM")
    sas = rows[rows.team == "SAS"]
    assert len(sas) == 6 and set(sas.matchup) == {"DEN@SAS"}
    assert {"Wembanyama, Victor", "Barnes, Harrison", "Ingram, Harrison"} <= set(sas.player_raw)
    assert ("DEN@SAS", "DEN", "not_yet_submitted") in set(zip(team.matchup, team.team, team.kind))
    # Barnes' row is split over pages 5/6; 'Impingement' comes back to him.
    assert sas.set_index("player_raw").loc["Barnes, Harrison", "reason_raw"] == (
        "Injury/Illness - Left Ankle; Impingement")
    assert not rows.reason_raw.str.contains("NOT YET SUBMITTED").any()
    assert not rows.matchup.isin(["DAL@MEM"]).any() or not (
        rows[rows.matchup == "DAL@MEM"].team == "SAS").any()


def test_all_teams_not_yet_submitted_does_not_crash():
    meta, rows, team = load("2026-07-10_05_00PM")  # Summer League
    assert rows.empty and list(rows.columns) == I.ROW_COLUMNS
    assert len(team) == 32 and set(team.kind) == {"not_yet_submitted"}
    assert team.groupby("matchup").size().eq(2).all()


def test_qa_file_parses_but_is_flagged_by_time():
    meta, rows, team = load("2026-10-07_10_45AM")
    assert meta["report_ts_et"] == et(2026, 10, 7, 10, 58)  # filename says 10:45
    assert set(rows.matchup) == {"BOS@DET"} and len(rows) == 7
    assert set(team.matchup) == {"GSW@OKC"}
    assert rows.set_index("player_raw").loc["Collins, John", "reason_cat"] == "reconditioning"


@pytest.mark.parametrize("name", FIXTURES)
def test_invariants_on_every_fixture(name):
    meta, rows, team = I.parse_report((FIX / name).read_bytes())
    assert meta["n_fallback_cells"] == 0
    for df in (rows, team):
        assert ((df.team == df.away) | (df.team == df.home)).all()
        assert (df.matchup == df.away + "@" + df.home).all()
    assert rows.status.isin(I.STATUSES).all()
    assert rows.reason_cat.isin(I.REASON_CATS).all()
    assert not rows.duplicated(["game_date", "matchup", "team", "player_raw"]).any()
    assert rows.player_raw.str.contains(", ").all()
    assert (rows.report_ts_et == meta["report_ts_et"]).all()


def test_unknown_status_is_loud(monkeypatch):
    monkeypatch.setattr(I, "STATUSES", ("Out", "Doubtful", "Questionable", "Available"))
    with pytest.raises(ValueError, match="unknown status 'Probable'"):
        I.parse_report((FIX / "Injury-Report_2026-01-15_05_00PM.pdf").read_bytes())


def test_unknown_team_is_loud(monkeypatch):
    real = I.teams.canonical
    monkeypatch.setattr(I.teams, "canonical",
                        lambda v: None if "Timberwolves" in str(v) else real(v))
    with pytest.raises(ValueError, match="unknown team"):
        I.parse_report((FIX / "Injury-Report_2018-12-17_05PM.pdf").read_bytes())


def test_mutation_without_team_level_rows_fails_loudly(monkeypatch):
    """Mutation check: if NOT YET SUBMITTED lines are not treated as rows, the
    DEN@SAS change is lost (the prototype's bug). The parser must then
    refuse, not quietly key the Spurs to the previous game."""
    monkeypatch.setattr(I, "_TEAM_PHRASES", {})
    with pytest.raises(ValueError):
        I.parse_report((FIX / "Injury-Report_2026-03-12_12_00PM.pdf").read_bytes())


def test_game_time_twelve_hour_clock():
    assert I.parse_game_time("07:00 (ET)") == dt.time(19, 0)
    assert I.parse_game_time("02:00 (ET)") == dt.time(14, 0)
    assert I.parse_game_time("12:00 (ET)") == dt.time(12, 0)
    assert I.parse_game_time("12:30 (ET)") == dt.time(12, 30)
    assert I.parse_game_time("TBD") is None
    with pytest.raises(ValueError):
        I.parse_game_time("13:00 (ET)")


# ── (d) reason taxonomy ──────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,cat", [
    ("Injury/Illness - Left Ankle; Stress Reaction", "injury_illness"),
    ("G League - Two-Way", "g_league_two_way"),
    ("G League - Two- Way", "g_league_two_way"),
    ("G League - On Assignment", "g_league_assignment"),
    ("G League Team", "g_league"),
    ("Not With Team - Self Isolating", "not_with_team"),
    ("Personal Reasons", "personal"),
    ("League Suspension", "suspension"),
    ("Team Suspension", "suspension"),
    ("Coach's Decision", "coach_decision"),
    ("Coach’s Decision -xyz", "coach_decision"),
    ("Rest", "rest"),
    ("Rest - asdfe", "rest"),
    ("Injury/Illness - Rest", "rest"),
    ("Injury/Illness - Load Management, Return from Achilles Injury", "rest"),
    ("Health and Safety Protocols", "health_protocols"),
    ("Concussion Protocol", "concussion_protocol"),
    ("Injury/Illness - N/A; Concussion", "concussion_protocol"),
    ("Trade Pending", "trade_pending"),
    ("Return to Competition Reconditioning", "reconditioning"),
    # Defect 10: reconditioning under other leads, and 2018-19 'G League Team - x'.
    ("Injury/Illness - N/a; Return to Competition Reconditioning/Back Soreness", "reconditioning"),
    ("Not with Team - Return to Competition Reconditioning", "reconditioning"),
    ("Coach's Decision - Post Health and Safety Protocols Reconditioning", "reconditioning"),
    ("Injury/Illness - Illness; Reconditioning", "reconditioning"),
    ("G League Team - Left Ankle Surgery", "g_league"),
    ("G League Team - NBA G League Assignment (Two Way)", "g_league_two_way"),
    ("G League - Unknown Detail", "g_league"),
    ("-", "unspecified"),
    ("", "unspecified"),
    ("Ineligible To Play", "other"),
    ("Something New", "other"),
])
def test_reason_category(raw, cat):
    assert I.reason_category(raw) == cat
    assert cat in I.REASON_CATS


def test_reason_lead():
    assert I.reason_lead("Injury/Illness - Left Ankle; Sprain") == "Injury/Illness"
    assert I.reason_lead("Ineligible To Play") == "Ineligible To Play"


# ── (e) schedule validation ──────────────────────────────────────────────────
def sched(*games) -> pd.DataFrame:
    rows = []
    for gid, day, away, home, hh, mm in games:
        d = dt.date.fromisoformat(day)
        tip = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET).astimezone(UTC)
        rows.append(dict(game_id=gid, game_date=d, away=away, home=home, tip_utc=pd.Timestamp(tip)))
    return pd.DataFrame(rows)


# The real 2026-10-07 slate per ESPN (research): no BOS@DET, no GSW@OKC.
OCT7 = sched(("0012600031", "2026-10-07", "MIN", "IND", 19, 0),
             ("0012600032", "2026-10-07", "ORL", "MEM", 20, 0),
             ("0012600033", "2026-10-07", "MIL", "OKC", 20, 0),
             ("0012600034", "2026-10-07", "PHX", "CHI", 20, 0),
             ("0012600035", "2026-10-07", "GSW", "POR", 22, 0))


def test_validation_drops_the_qa_file():
    _, rows, team = load("2026-10-07_10_45AM")
    kept, dropped = I.validate_rows(rows, OCT7)
    assert kept.empty and len(dropped) == 7 and set(dropped.drop_reason) == {"not_in_schedule"}
    tk, td = I.validate_rows(team, OCT7)
    assert tk.empty and len(td) == 2


def test_validation_drops_summer_league():
    _, rows, team = load("2026-07-10_05_00PM")
    kept, dropped = I.validate_rows(team, OCT7)
    assert kept.empty and len(dropped) == 32
    k2, d2 = I.validate_rows(rows, OCT7)
    assert k2.empty and d2.empty


def test_validation_attaches_game_id_and_flags_swaps_and_times():
    _, rows, _ = load("2026-01-15_05_00PM")
    s = sched(("0022500601", "2026-01-15", "MEM", "ORL", 14, 0),   # Berlin, 2 PM
              ("0022500602", "2026-01-15", "DET", "PHX", 19, 1),   # reversed; 8:01-style tip
              ("0022500603", "2026-01-15", "UTA", "DAL", 19, 30))  # report says 08:30
    kept, dropped = I.validate_rows(rows, s)
    assert len(kept) + len(dropped) == len(rows)
    assert set(kept.game_id) == {"0022500601", "0022500602", "0022500603"}
    edey = kept[kept.player_raw == "Edey, Zach"].iloc[0]
    assert edey.game_id == "0022500601" and not edey.swapped and not edey.tip_mismatch
    assert kept[kept.game_id == "0022500602"].swapped.all()
    assert not kept[kept.game_id == "0022500602"].tip_mismatch.any()  # 1 min is not a mismatch
    assert kept[kept.game_id == "0022500603"].tip_mismatch.all()
    assert dt.date(2026, 1, 16) in set(dropped.game_date)


def test_validation_refuses_a_bad_schedule():
    _, rows, _ = load("2026-01-15_05_00PM")
    with pytest.raises(ValueError, match="missing columns"):
        I.validate_rows(rows, OCT7.drop(columns=["tip_utc"]))
    with pytest.raises(ValueError, match="duplicate"):
        I.validate_rows(rows, pd.concat([OCT7, OCT7]))


# ── (f) report selection ─────────────────────────────────────────────────────
def fake_report(ts: dt.datetime, title="x") -> I.ParsedReport:
    return I.ParsedReport({"report_ts_et": ts, "title": title},
                          pd.DataFrame(columns=I.ROW_COLUMNS),
                          pd.DataFrame(columns=I.TEAM_ROW_COLUMNS))


def getter_from(published: dict[str, dt.datetime], log: list | None = None):
    def get(slot):
        if log is not None:
            log.append(slot.filename)
        ts = published.get(slot.filename)
        return fake_report(ts, slot.filename) if ts else None
    return get


def test_selection_uses_title_time_not_filename():
    # Hourly era: file 'HH' holds the HH:30 report.
    pub = {f"Injury-Report_2022-01-15_{h:02d}PM.pdf": et(2022, 1, 15, 12 + h, 30) for h in range(1, 12)}
    got = I.select_reports_for_date("2022-01-15", [et(2022, 1, 15, 19, 0), et(2022, 1, 15, 19, 30),
                                                   et(2022, 1, 15, 20, 0)], getter=getter_from(pub))
    pick = {t.strftime("%H:%M"): s.slot.filename[-8:-4] for t, s in got.items()}
    # 7:00 tip: '06PM' (18:30). 7:30 tip: '07PM' says 19:30, not < 19:25 -> '06PM'.
    assert pick == {"19:00": "06PM", "19:30": "06PM", "20:00": "07PM"}
    assert all(s.report.meta["report_ts_et"] < t - I.TIP_LEAD for t, s in got.items())


def test_selection_on_real_fixture_bytes(tmp_path):
    # The 05PM file says 5:30: valid for a 6:00 PM tip, not for a 5:30 PM tip.
    store = I.ReportStore(tmp_path, offline=True)
    p = tmp_path / "2021-22" / "Injury-Report_2022-01-15_05PM.pdf"
    p.parent.mkdir(parents=True)
    p.write_bytes((FIX / p.name).read_bytes())
    got = I.select_reports_for_date("2022-01-15", ["2022-01-15T23:00:00Z", "2022-01-15T22:30:00Z"],
                                    getter=store.get, max_back=dt.timedelta(hours=3))
    by = {t.strftime("%H:%M"): s for t, s in got.items()}
    assert by["18:00"].slot.filename.endswith("_05PM.pdf")
    assert by["17:30"] is None


def test_selection_walks_into_the_previous_evening():
    pub = {"Injury-Report_2019-01-14_08PM.pdf": et(2019, 1, 14, 20, 30)}
    log = []
    got = I.select_reports_for_date("2019-01-15", [et(2019, 1, 15, 13, 0)], getter=getter_from(pub, log))
    assert got[et(2019, 1, 15, 13, 0)].slot.filename == "Injury-Report_2019-01-14_08PM.pdf"
    assert log == ["Injury-Report_2019-01-14_08PM.pdf"]


def test_selection_none_is_unknown_not_an_error():
    got = I.select_reports_for_date("2026-02-16", [et(2026, 2, 16, 19, 0)], getter=getter_from({}),
                                    max_back=dt.timedelta(hours=2))
    assert got == {et(2026, 2, 16, 19, 0): None}


def test_selection_accept_can_veto_a_qa_file():
    pub = {"Injury-Report_2026-01-15_06_45PM.pdf": et(2026, 1, 15, 18, 45),
           "Injury-Report_2026-01-15_06_30PM.pdf": et(2026, 1, 15, 18, 30)}
    got = I.select_reports_for_date("2026-01-15", [et(2026, 1, 15, 19, 0)], getter=getter_from(pub),
                                    accept=lambda r: not r.meta["title"].endswith("06_45PM.pdf"))
    assert got[et(2026, 1, 15, 19, 0)].slot.filename.endswith("06_30PM.pdf")


def test_selection_rejects_a_tip_on_another_date():
    with pytest.raises(ValueError):
        # 06:00Z on the 16th is 1:00 AM ET on the 16th (03:30Z would still be
        # 10:30 PM on the 15th: the UTC date is not the game date).
        I.select_reports_for_date("2026-01-15", ["2026-01-16T06:00:00Z"], getter=getter_from({}))
    assert I.select_reports_for_date("2026-01-15", ["2026-01-16T03:30:00Z"], getter=getter_from({}),
                                     max_back=dt.timedelta(hours=1)) == {et(2026, 1, 15, 22, 30): None}


def test_latest_report_walks_back_a_capped_number_of_slot_times():
    pub = {"Injury-Report_2026-01-15_05_00PM.pdf": et(2026, 1, 15, 17, 0)}
    now = et(2026, 1, 15, 19, 7)
    assert I.latest_report(now, getter=getter_from(pub), max_slots=8) is None  # 19:00..17:15
    sel = I.latest_report(now, getter=getter_from(pub), max_slots=9)
    assert sel.slot.filename.endswith("05_00PM.pdf") and sel.n_probed == 9


def test_latest_report_cap_counts_times_not_files_across_formats():
    # On 2025-12-22 each time has an hourly AND a quarter name.
    log = []
    I.latest_report(et(2025, 12, 22, 12, 0), getter=getter_from({}, log), max_slots=2)
    assert log == ["Injury-Report_2025-12-22_12_00PM.pdf", "Injury-Report_2025-12-22_12PM.pdf",
                   "Injury-Report_2025-12-22_11_45AM.pdf"]


def test_latest_report_blocked_propagates():
    def blocked(slot):
        raise http.Blocked(slot.url, "Akamai")
    with pytest.raises(http.Blocked):
        I.latest_report(et(2026, 1, 15, 19, 0), getter=blocked)


# ── (g) name resolution ──────────────────────────────────────────────────────
ROSTER = [  # (display name, person_id, team): a small fixture roster; ids are test values
    ("Yang Hansen", 1642905, "POR"), ("Zach Edey", 1641744, "MEM"),
    ("Dereck Lively II", 1641726, "DAL"), ("David Jones Garcia", 1642357, "SAS"),
    ("Nigel Hayes-Davis", 1628502, "PHX"), ("Jaylin Williams", 1631119, "OKC"),
    ("Jalen Williams", 1631114, "OKC"), ("Olivier-Maxence Prosper", 1641765, "MEM"),
]


def roster_index() -> NameIndex:
    idx = NameIndex(aliases={})
    for name, pid, team in ROSTER:
        idx.add(name, pid, team=team)
    return idx


def test_resolve_scoped_both_orders_and_counted_misses():
    rows = pd.DataFrame({"team": ["POR", "DAL", "SAS", "PHX", "MEM", "OKC", "OKC", "BOS", "BOS"],
                         "player_raw": ["Yang, Hansen", "Lively II, Dereck", "Jones Garcia, David",
                                        "Hayes-Davis, Nigel", "Prosper, Olivier-Maxence",
                                        "Williams, Jaylin", "Williams, Jay", "Nobody, Some",
                                        "Nobody, Some"]})
    for index in (roster_index(), {t: roster_index() for t in ("POR", "DAL", "SAS", "PHX", "MEM", "OKC")}):
        res, unres = I.resolve(rows, index)
        ids = dict(zip(res.player_raw, res.person_id))
        assert ids["Yang, Hansen"] == 1642905           # family-name-first player
        assert ids["Lively II, Dereck"] == 1641726 and ids["Jones Garcia, David"] == 1642357
        assert ids["Hayes-Davis, Nigel"] == 1628502 and ids["Prosper, Olivier-Maxence"] == 1641765
        assert ids["Williams, Jaylin"] == 1631119
        # names.NameIndex's prefix fallback binds 'Jay' to Jaylin (the only
        # OKC Williams whose given name starts 'jay'); it is resolved but
        # labelled, so an audit can list every non-exact match.
        hows = dict(zip(res.player_raw, res.match_how))
        assert hows["Williams, Jay"] == "prefix" and hows["Williams, Jaylin"] == "exact"
        assert len(res) + unres.n_rows.sum() == len(rows)
        miss = dict(zip(unres.player_raw, unres.n_rows))
        assert miss == {"Nobody, Some": 2}
        assert res.person_id.dtype == "int64"


def test_resolve_on_parsed_rows():
    _, rows, _ = load("2026-01-15_05_00PM")
    res, unres = I.resolve(rows, roster_index())
    assert set(res.player_raw) >= {"Edey, Zach", "Lively II, Dereck", "Hayes-Davis, Nigel"}
    assert len(res) + unres.n_rows.sum() == len(rows) and len(unres) > 100


def test_suffix_between_two_commas_on_the_2026_11_02_report():
    """The report as published prints CLE's two-way Ernest Udeh Jr. as 'Udeh,
    Jr., Ernest' (the playerIndex: person 1643640, last name 'Udeh, Jr.');
    its other suffixed names are 'Lively II, Dereck' style. Split at the first
    comma he resolved to nobody, so his Out never reached his row (he read as
    not listed on a team that filed) and the bot rebuilt him INACTIVE_UNLISTED."""
    from bucketlib import livectx as L
    udeh, gid = 1643640, "0022600158"                       # OKC@CLE, 7:00 PM ET
    rep = load("2026-11-02_02_15AM")
    rows = rep.rows[rep.rows["player_raw"].str.count(",") >= 2]
    assert rows[["matchup", "team", "player_raw", "status", "reason_cat"]].values.tolist() == \
        [["OKC@CLE", "CLE", "Udeh, Jr., Ernest", "Out", "g_league_two_way"]]
    idx = NameIndex(aliases={})
    idx.add("Ernest Udeh, Jr.", udeh, team="CLE")
    res, unres = I.resolve(rows, idx)
    assert (res.person_id.tolist(), res.match_how.tolist(), len(unres)) == ([udeh], ["exact"], 0)
    # The bot's and the app's path: context.context_for and box._report_people.
    roster = pd.DataFrame({"game_id": [gid], "team": ["CLE"], "person_id": [udeh],
                           "player_name": ["Ernest Udeh, Jr."]})
    ip = L.resolve_names(rows.assign(game_id=gid), roster)
    assert ip["person_id"].tolist() == [udeh] and ip.attrs["unresolved"] == 0
    me = pd.DataFrame({"game_id": [gid], "team": ["CLE"], "person_id": [udeh],
                       "tip_utc": [pd.Timestamp("2026-11-03T00:00Z")]})
    teams = pd.DataFrame({"game_id": [gid], "team": ["CLE"], "state": ["listed"],
                          "report_ts_et": [rep.meta["report_ts_et"]],
                          "cutoff_utc": [pd.Timestamp("2026-11-02T23:45Z")]})
    c = L.build_context(me, ip, teams).iloc[0]
    assert (c["inj_out"], c["inj_reason_gleague"], c["inj_not_listed"]) == (1.0, 1.0, 0.0)


# ── script: offline end to end ───────────────────────────────────────────────
def _script():
    spec = importlib.util.spec_from_file_location("build_injury_reports",
                                                  REPO / "scripts" / "build_injury_reports.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_helpers():
    m = _script()
    assert m.parse_seasons("2018-19:2020-21") == ["2018-19", "2019-20", "2020-21"]
    assert m.parse_seasons("2024-25") == ["2024-25"]
    d = list(range(100))
    s = m.sample_dates(d, 5)
    assert s[0] == 0 and s[-1] == 99 and len(s) == 5


def fixture_store(root: Path, served: dict[str, bytes], **kw) -> I.ReportStore:
    """A store whose 'network' serves `served` by filename and answers Absent
    (the S3 403) for every other slot, like the real bucket."""
    def fetch(url):
        name = url.rsplit("/", 1)[-1]
        if name in served:
            return served[name]
        raise http.Absent(url, "HTTP 403 S3 AccessDenied")
    return I.ReportStore(root, fetcher=fetch, **kw)


def fx(name: str) -> bytes:
    return (FIX / name).read_bytes()


def test_build_season(tmp_path, monkeypatch):
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    store = fixture_store(tmp_path / "raw" / "injury",
                          {"Injury-Report_2026-01-15_05_00PM.pdf": fx("Injury-Report_2026-01-15_05_00PM.pdf")})
    s = sched(("0022500601", "2026-01-15", "MEM", "ORL", 14, 0),    # before 5 PM: no report
              ("0022500602", "2026-01-15", "PHX", "DET", 19, 0),
              ("0022500603", "2026-01-15", "NYK", "GSW", 22, 0),
              ("0022500604", "2026-01-15", "LAL", "BOS", 19, 30),   # not in the report
              ("0012500099", "2026-01-15", "SAC", "LAC", 22, 30))   # a preseason id: skipped
    args = type("A", (), dict(dates="", sample=0, keep_going=False))()
    cov = m.build_season("2025-26", s, store, args)
    rows = pd.read_parquet(tmp_path / "injury_reports_2025-26_v1.parquet")
    tdf = pd.read_parquet(tmp_path / "injury_report_teams_2025-26_v1.parquet")
    assert set(rows.game_id) == {"0022500602", "0022500603"}
    assert (rows.report_file == "Injury-Report_2026-01-15_05_00PM.pdf").all()
    assert {"report_ts_et", "game_id", "team", "player_raw", "status", "reason_cat",
            "reason_raw"} <= set(rows.columns)
    st = dict(zip(zip(tdf.game_id, tdf.team), tdf.state))
    assert st[("0022500601", "MEM")] == "no_report" and st[("0022500602", "PHX")] == "listed"
    assert set(tdf.state) <= set(m.TEAM_STATES)
    assert len(tdf) == 8 and cov["games_no_report"] == 1 and cov["games_skipped_type"] == 1
    # The 5 PM report is real but never lists LAL@BOS: that game is 'not in
    # the report', not a 30-hour walk to some older report.
    assert st[("0022500604", "LAL")] == "game_not_in_report" and cov["games_not_in_report"] == 1
    assert (tdf[tdf.game_id == "0022500604"].report_file == "Injury-Report_2026-01-15_05_00PM.pdf").all()
    assert cov["schedule_drops"]["rows"] > 0 and cov["fetch"]["offline_miss"] == 0
    assert not cov["partial"] and cov["parse_failures"] == {}


# ── live ─────────────────────────────────────────────────────────────────────
@pytest.mark.network
def test_live_first_archived_report_and_absent():
    b = I.fetch_report(I.parse_slot_filename("Injury-Report_2018-12-17_05PM.pdf").url)
    assert I.parse_report(b).meta["report_ts_et"] == et(2018, 12, 17, 17, 30)
    with pytest.raises(http.Absent):
        I.fetch_report(I.parse_slot_filename("Injury-Report_2018-12-16_05PM.pdf").url)


def test_build_season_other_lead_never_overwrites_the_canonical_table(tmp_path, monkeypatch):
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    store = fixture_store(tmp_path / "raw" / "injury",
                          {"Injury-Report_2026-01-15_05_00PM.pdf": fx("Injury-Report_2026-01-15_05_00PM.pdf")})
    s = sched(("0022500602", "2026-01-15", "PHX", "DET", 19, 0),   # 4:00 PM cutoff: none
              ("0022500603", "2026-01-15", "NYK", "GSW", 22, 0))   # 7:00 PM cutoff: 5 PM report
    args = type("A", (), dict(dates="", sample=0, keep_going=False, lead_minutes=180))()
    cov = m.build_season("2025-26", s, store, args)
    assert not (tmp_path / "injury_reports_2025-26_v1.parquet").exists()
    rows = pd.read_parquet(tmp_path / "injury_reports_lead180_2025-26_v1.parquet")
    assert set(rows.game_id) == {"0022500603"} and cov["games_no_report"] == 1


# ── fixes after the adversarial review (2026-10-08) ──────────────────────────
# Each test below names the review defect it pins; each was run against the
# pre-fix code (or with the fix reverted) and failed there.

# Defect 1: a wrapped team name on a one-line row. 'Minnesota Timberwolves'
# wraps to two lines centred on the row, and the row's time and matchup sit on
# the team name's FIRST line, 4pt above the status.
def test_wrapped_team_name_on_a_team_level_row():
    meta, rows, team = load("2018-12-22_01PM")
    k = {(t.matchup, t.team): (t.kind, t.game_time_raw, t.game_date) for t in team.itertuples()}
    assert k[("MIN@OKC", "MIN")] == ("not_yet_submitted", "08:00 (ET)", dt.date(2018, 12, 23))
    assert k[("MIN@OKC", "OKC")][0] == "not_yet_submitted"
    assert meta["n_wrapped_group_cells"] == 1 and meta["n_carried_cells"] == 0


def test_wrapped_team_name_on_a_player_row():
    # The reviewer's live case: 2019-11-01_05PM raised on MIN@WAS.
    meta, rows, _ = load("2019-11-01_05PM")
    g = rows[rows.matchup == "MIN@WAS"]
    assert (g.team == "MIN").sum() == 6 and (g.team == "WAS").sum() == 7
    assert set(g.game_time_et) == {dt.time(20, 0)}
    assert {"Towns, Karl-Anthony", "Bates-Diop, Keita"} <= set(g[g.team == "MIN"].player_raw)


def test_wrapped_team_name_on_the_first_row_of_a_page():
    # The MIN row opens page 3; its matchup sits 4pt above the page's first
    # status. Trying the page-edge carry before the overlap rule sends
    # 'SAS@MIN' to the last row on page 2 (a MIA row): the reviewer's
    # direction alone would have mis-keyed 3 of the 116 failing files.
    meta, rows, team = load("2019-01-17_01PM")
    k = {(t.matchup, t.team, t.page) for t in team.itertuples()}
    assert {("SAS@MIN", "MIN", 3), ("SAS@MIN", "SAS", 3)} <= k
    assert meta["n_carried_cells"] == 0 and meta["n_wrapped_group_cells"] == 1
    assert not rows.matchup.eq("SAS@MIN").any()


# Defect 2: three iText column orders between 2019-11-15 and 2019-12-17.
def test_layout4_reason_before_status():
    meta, rows, _ = load("2019-11-15_05PM")
    assert meta["layout"] == 4 and meta["columns"][5:] == ["reason", "status", "prev_status"]
    r = rows.set_index("player_raw")
    assert (r.loc["Bacon, Dwayne", ["matchup", "team", "status", "reason_raw"]] ==
            ["DET@CHA", "CHA", "Probable", "Injury/Illness - Right Knee Soreness"]).all()
    # A reason wrapped around its status comes back whole, and in order.
    assert r.loc["Batum, Nicolas", "reason_raw"] == "Injury/Illness - Fractured Third Finger, Left Hand"
    assert r.loc["Franks, Robert", "reason_cat"] == "g_league_two_way"
    assert rows.prev_status.value_counts().to_dict() == {
        "Available": 6, "Out": 6, "Questionable": 3, "Probable": 2, "Doubtful": 1}


def test_layout5_status_reason_previous_status():
    meta, rows, _ = load("2019-11-20_05PM")
    assert meta["layout"] == 5 and meta["columns"][5:] == ["status", "reason", "prev_status"]
    r = rows.set_index("player_raw")
    assert (r.loc["Barrett, RJ", ["status", "reason_raw", "prev_status"]] ==
            ["Available", "Injury/Illness - Left left thumb; sprain", "Probable"]).all()
    assert r.loc["Bullock, Reggie", "reason_raw"] == "Injury/Illness - Neck; cervical disc herniation"
    assert rows.prev_reason.isna().all()


def test_layout6_previous_reason():
    meta, rows, _ = load("2019-12-17_05PM")
    assert meta["layout"] == 6 and meta["columns"][-1] == "prev_reason"
    r = rows.set_index("player_raw")
    assert r.loc["Oladipo, Victor", "reason_raw"] == "Injury/Illness - Right Knee; Rehab"
    assert r.loc["Batum, Nicolas", "prev_reason"] == "Injury/Illness - Left Hand; Discomfort"
    assert r.loc["Oladipo, Victor", "prev_reason"] is None   # '-' is no reason
    assert rows.prev_reason.notna().sum() == 5


def test_every_layout_keeps_the_previous_columns_apart():
    # Layout 2 and GemBox have no Previous columns; nothing may leak into them.
    for stem in ("2020-01-15_05PM", "2023-12-01_05PM"):
        _, rows, _ = load(stem)
        assert rows.prev_status.isna().all() and rows.prev_reason.isna().all()


# Defect 3: All-Star weekend 'UNK@UNK' / 'Non-NBA Team' rows.
def test_all_star_placeholder_rows_are_excluded_and_counted():
    meta, rows, team = load("2019-02-14_01PM")
    assert sorted(rows.matchup.unique()) == ["CHA@ORL", "NYK@ATL", "OKC@NOP"] and len(rows) == 19
    assert team.empty and meta["n_excluded_team_rows"] == 4 and meta["n_excluded_rows"] == 0
    assert all("UNK@UNK" in e and "2019-02-15" in e for e in meta["excluded"])


def test_only_the_exact_placeholder_is_excluded(monkeypatch):
    # An NBA-looking team under UNK@UNK, or a UNK beside a real code, raises.
    monkeypatch.setattr(I, "_NON_NBA_TEAM", "SOMETHING ELSE")
    with pytest.raises(ValueError, match="under UNK@UNK"):
        load.__wrapped__("2019-02-14_01PM")
    monkeypatch.undo()
    monkeypatch.setattr(I, "_NON_NBA_CODE", "XXX")
    with pytest.raises(ValueError, match="unknown team code in matchup 'UNK@UNK'"):
        load.__wrapped__("2019-02-14_01PM")


# Defect 15: a page that yields no text, or a footer that disagrees.
def _pages_with(monkeypatch, swap):
    import pypdf
    real = I._page_frags
    seen = []

    def fake(page):
        seen.append(page)
        return swap(len(seen) - 1, page, real)
    monkeypatch.setattr(I, "_page_frags", fake)
    return pypdf


def test_a_page_without_text_is_loud(monkeypatch):
    _pages_with(monkeypatch, lambda i, page, real: [] if i == 1 else real(page))
    with pytest.raises(ValueError, match="page 2 of 4 has no extractable text"):
        load.__wrapped__("2018-12-20_05PM")


def test_footer_must_match_the_page(monkeypatch):
    import pypdf
    other = pypdf.PdfReader(str(FIX / "Injury-Report_2018-12-20_05PM.pdf")).pages[2]
    _pages_with(monkeypatch, lambda i, page, real: real(other) if i == 1 else real(page))
    with pytest.raises(ValueError, match="page 2 of 4: footer says 'Page 3 of 4'"):
        load.__wrapped__("2018-12-20_05PM")


# Defect 9c/9d: the guards the original suite never exercised.
def test_team_outside_its_matchup_is_loud(monkeypatch):
    real = I.teams.canonical
    # Full names only (the matchup's codes are 3 letters): DET's rows land on BOS.
    monkeypatch.setattr(I.teams, "canonical",
                        lambda v: "BOS" if str(v) == "Detroit Pistons" else real(v))
    with pytest.raises(ValueError, match="team BOS is not in matchup MIL@DET"):
        load.__wrapped__("2018-12-17_05PM")


def test_report_level_duplicates_are_loud():
    _, rows, team = load("2026-03-12_12_00PM")
    with pytest.raises(ValueError, match="duplicate player rows"):
        I._check_tables(pd.concat([rows, rows.iloc[[0]]]), team)
    with pytest.raises(ValueError, match="duplicate team-level rows"):
        I._check_tables(rows, pd.concat([team, team.iloc[[0]]]))
    sas = rows[rows.team == "SAS"].iloc[[0]]
    nys = team[team.team == "DEN"].assign(team="SAS")
    with pytest.raises(ValueError, match="NOT YET SUBMITTED and listing players"):
        I._check_tables(sas, nys)
    I._check_tables(rows, team)   # the real report passes


# Defect 9e/9f: the selection boundary and the DST cutoff.
def test_a_report_titled_exactly_at_the_cutoff_is_too_late():
    pub = {"Injury-Report_2026-01-15_06_45PM.pdf": et(2026, 1, 15, 18, 55),
           "Injury-Report_2026-01-15_06_30PM.pdf": et(2026, 1, 15, 18, 30)}
    got = I.select_reports_for_date("2026-01-15", [et(2026, 1, 15, 19, 0)], getter=getter_from(pub))
    assert got[et(2026, 1, 15, 19, 0)].slot.filename.endswith("06_30PM.pdf")


def test_cutoff_is_computed_in_utc_across_dst():
    # Tip 7:00 PM EDT on 2026-03-08 (DST began 2 AM). Twenty hours earlier is
    # 10:00 PM EST on the 7th; wall-clock arithmetic says 11:00 PM and would
    # take the report titled 10:30 PM.
    pub = {"Injury-Report_2026-03-07_10_30PM.pdf": et(2026, 3, 7, 22, 30),
           "Injury-Report_2026-03-07_09_45PM.pdf": et(2026, 3, 7, 21, 45)}
    got = I.select_reports_for_date("2026-03-08", [et(2026, 3, 8, 19, 0)], getter=getter_from(pub),
                                    lead=dt.timedelta(hours=20))
    assert got[et(2026, 3, 8, 19, 0)].slot.filename.endswith("09_45PM.pdf")


def test_negative_lead_is_refused():
    with pytest.raises(ValueError, match="negative lead"):
        I.select_reports_for_date("2026-01-15", [et(2026, 1, 15, 19, 0)], getter=getter_from({}),
                                  lead=dt.timedelta(minutes=-5))


# Defect 6: latest_report must not return a report titled after `now`.
def test_latest_report_never_returns_a_report_titled_after_now():
    pub = {"Injury-Report_2025-02-12_07PM.pdf": et(2025, 2, 12, 19, 30),
           "Injury-Report_2025-02-12_06PM.pdf": et(2025, 2, 12, 18, 30)}
    sel = I.latest_report(et(2025, 2, 12, 19, 10), getter=getter_from(pub))
    assert sel.slot.filename.endswith("_06PM.pdf")
    sel = I.latest_report(et(2025, 2, 12, 19, 30), getter=getter_from(pub))   # at = published
    assert sel.slot.filename.endswith("_07PM.pdf")


def test_latest_report_accept_walks_past_a_qa_file():
    pub = {"Injury-Report_2026-01-15_06_45PM.pdf": et(2026, 1, 15, 18, 45),
           "Injury-Report_2026-01-15_06_30PM.pdf": et(2026, 1, 15, 18, 30)}
    sel = I.latest_report(et(2026, 1, 15, 18, 50), getter=getter_from(pub),
                          accept=lambda r: not r.meta["title"].endswith("06_45PM.pdf"))
    assert sel.slot.filename.endswith("06_30PM.pdf")


# Defects 5/9g: the report-level accept.
def test_lists_nba_game_reads_player_rows_and_team_rows():
    _, qa_rows, qa_team = load("2026-10-07_10_45AM")
    assert not I.lists_nba_game(I.ParsedReport({}, qa_rows, qa_team), OCT7)
    _, sl_rows, sl_team = load("2026-07-10_05_00PM")
    assert not I.lists_nba_game(I.ParsedReport({}, sl_rows, sl_team), OCT7)
    # A report whose only content for tonight is NOT YET SUBMITTED is real.
    _, rows, team = load("2026-03-12_12_00PM")
    s = sched(("0022500999", "2026-03-13", "CLE", "DAL", 19, 30))
    assert I.lists_nba_game(I.ParsedReport({}, rows.iloc[0:0], team), s)
    assert not I.lists_nba_game(I.ParsedReport({}, rows.iloc[0:0], team.iloc[0:0]), s)


def test_report_before_is_the_selection_rule():
    pub = {"Injury-Report_2022-01-15_06PM.pdf": et(2022, 1, 15, 18, 30)}
    sel = I.report_before(et(2022, 1, 15, 18, 55), getter=getter_from(pub))
    assert sel.slot.filename.endswith("_06PM.pdf") and sel.n_probed == 1
    assert I.report_before(et(2022, 1, 15, 18, 30), getter=getter_from(pub),
                           max_back=dt.timedelta(hours=1)) is None


# Defects 4/14/13: the store.
def test_a_parse_failure_is_parsed_once_and_reraised(tmp_path, monkeypatch):
    calls = []
    real = I.parse_report

    def counting(data):
        calls.append(1)
        raise ValueError("synthetic layout change")
    store = fixture_store(tmp_path, {"Injury-Report_2024-01-15_05PM.pdf": fx("Injury-Report_2022-01-15_05PM.pdf")})
    monkeypatch.setattr(I, "parse_report", counting)
    for _ in range(3):
        with pytest.raises(ValueError, match="synthetic"):
            store.get(_old_slot())
    assert len(calls) == 1 and store.stats["parse_failed"] == 1 and store.stats["parsed"] == 0
    assert store.parse_seconds > 0 and _old_slot().filename in store.failed
    monkeypatch.setattr(I, "parse_report", real)


def test_a_truncated_pdf_is_never_cached(tmp_path):
    full = fx("Injury-Report_2022-01-15_05PM.pdf")
    store = fixture_store(tmp_path, {"Injury-Report_2024-01-15_05PM.pdf": full[:int(len(full) * 0.9)]})
    with pytest.raises(http.BadPayload):
        store.get(_old_slot())
    assert not store.path(_old_slot()).exists()


def test_a_truncated_cached_pdf_is_fetched_again(tmp_path):
    full = fx("Injury-Report_2022-01-15_05PM.pdf")
    store = fixture_store(tmp_path, {"Injury-Report_2024-01-15_05PM.pdf": full})
    p = store.path(_old_slot())
    p.parent.mkdir(parents=True)
    p.write_bytes(full[:20_000])        # left by a run before the check existed
    assert store.get(_old_slot()).meta["layout"] == 2
    assert store.stats["cache_corrupt"] == 1 and store.stats["live"] == 1
    assert p.read_bytes() == full


def test_a_memory_store_writes_nothing(tmp_path):
    store = fixture_store(tmp_path, {"Injury-Report_2024-01-15_05PM.pdf": fx("Injury-Report_2022-01-15_05PM.pdf")},
                          persist=False)
    assert store.get(_old_slot()) is not None
    assert store.get(I.parse_slot_filename("Injury-Report_2024-01-15_06PM.pdf")) is None
    store.flush()
    assert list(tmp_path.iterdir()) == []
    # Absence is still remembered in memory (the 48h rule): one live probe.
    assert store.get(I.parse_slot_filename("Injury-Report_2024-01-15_06PM.pdf")) is None
    assert store.stats["live"] == 2 and store.stats["absent_cached"] == 1


def test_app_stores_never_write_the_shared_checkout(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen.update(kw)
        raise http.Absent(url, "403")
    monkeypatch.setattr(I.http, "fetch", fake)
    live = I.live_store(timeout=5.0)
    assert live.persist is False and live.get(_old_slot()) is None
    assert seen["retries"] == 0 and seen["timeout"] == 5.0
    monkeypatch.setattr(I, "READ_ONLY", True)
    monkeypatch.setattr(I, "_default_store", None)
    assert I.default_store().persist is False


# Defect 11: an ambiguous name is reported as ambiguous.
def test_resolve_reports_ambiguous_not_none():
    idx = roster_index()
    idx.add("Jalen Williams", 9999901, team="OKC")    # a second OKC Jalen Williams
    rows = pd.DataFrame({"team": ["OKC", "OKC"], "player_raw": ["Williams, Jalen", "Nobody, Some"]})
    res, unres = I.resolve(rows, idx)
    assert res.empty
    assert dict(zip(unres.player_raw, unres.match_how)) == {"Williams, Jalen": "ambiguous",
                                                            "Nobody, Some": "none"}


# ── script: the review's defects 4, 5, 7, 8, 9a, 9b, 12 ──────────────────────
JAN15 = sched(("0022500602", "2026-01-15", "PHX", "DET", 19, 0),
              ("0022500603", "2026-01-15", "NYK", "GSW", 22, 0))


def _args(**kw):
    base = dict(dates="", sample=0, keep_going=False, lead_minutes=5)
    base.update(kw)
    return type("A", (), base)()


def test_keep_going_never_swaps_in_an_older_report(tmp_path, monkeypatch):
    # The 06_00PM file does not parse; the older 05_00PM file does. The game
    # must say 'report_unparsed' (naming the file), not quietly use 5 PM.
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    good = fx("Injury-Report_2026-01-15_05_00PM.pdf")
    bad = good + b"%unparseable\n"
    real = I.parse_report
    monkeypatch.setattr(I, "parse_report",
                        lambda d: (_ for _ in ()).throw(ValueError("synthetic")) if d.endswith(b"%unparseable\n")
                        else real(d))
    store = fixture_store(tmp_path / "raw" / "injury", {"Injury-Report_2026-01-15_06_00PM.pdf": bad,
                                                        "Injury-Report_2026-01-15_05_00PM.pdf": good})
    cov = m.build_season("2025-26", JAN15, store, _args(keep_going=True))
    tdf = pd.read_parquet(tmp_path / "injury_report_teams_2025-26_v1.parquet")
    assert set(tdf.state) == {"report_unparsed"}
    assert set(tdf.report_file) == {"Injury-Report_2026-01-15_06_00PM.pdf"}
    assert pd.read_parquet(tmp_path / "injury_reports_2025-26_v1.parquet").empty
    # Two tip groups hit the same bad file: one failure, parsed once.
    assert list(cov["parse_failures"]) == ["Injury-Report_2026-01-15_06_00PM.pdf"]
    assert cov["fetch"]["parse_failed"] == 1 and cov["games_report_unparsed"] == 2
    with pytest.raises(ValueError, match="synthetic"):   # without --keep-going it stops
        m.build_season("2025-26", JAN15, fixture_store(tmp_path / "r2", {
            "Injury-Report_2026-01-15_06_00PM.pdf": bad}), _args())


def test_keep_going_does_not_swallow_blocked_or_the_budget(tmp_path, monkeypatch):
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)

    def blocked(url):
        raise http.Blocked(url, "HTTP 403 text/html: Akamai edge block")
    with pytest.raises(http.Blocked):
        m.build_season("2025-26", JAN15, I.ReportStore(tmp_path / "a", fetcher=blocked),
                       _args(keep_going=True))
    with pytest.raises(RuntimeError, match="budget"):
        m.build_season("2025-26", JAN15, fixture_store(tmp_path / "b", {}, budget=3),
                       _args(keep_going=True))
    assert not list(tmp_path.glob("*.parquet"))


def test_a_qa_file_is_vetoed_by_the_report_level_accept(tmp_path, monkeypatch):
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    store = fixture_store(tmp_path / "raw" / "injury",
                          {"Injury-Report_2026-10-07_10_45AM.pdf": fx("Injury-Report_2026-10-07_10_45AM.pdf")})
    args = _args(include_preseason=True)
    cov = m.build_season("2026-27", OCT7.iloc[[0]], store, args)
    assert cov["reports_vetoed_not_nba"] == {"Injury Report: 10/07/26 10:58 AM": 1}
    tdf = pd.read_parquet(tmp_path / "injury_report_teams_2026-27_v1.parquet")
    assert set(tdf.state) == {"no_report"}


def test_offline_misses_never_reach_the_canonical_table(tmp_path, monkeypatch):
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    store = I.ReportStore(tmp_path / "raw" / "injury", offline=True)
    dst = store.path(I.parse_slot_filename("Injury-Report_2026-01-15_05_00PM.pdf"))
    dst.parent.mkdir(parents=True)
    dst.write_bytes(fx(dst.name))
    cov = m.build_season("2025-26", JAN15, store, _args())
    assert cov["partial"] and cov["partial_reason"] == "offline misses"
    assert cov["fetch"]["offline_miss"] > 0
    assert not list(tmp_path.glob("injury_report*.parquet"))
    assert (tmp_path / "raw" / "injury" / "2025-26" / "_offline_partial_injury_reports.parquet").exists()


def test_every_row_is_asserted_to_precede_tip(tmp_path, monkeypatch):
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    rep = I.parse_report(fx("Injury-Report_2026-01-15_05_00PM.pdf"))   # titled 5:00 PM
    slot = I.parse_slot_filename("Injury-Report_2026-01-15_05_00PM.pdf")
    # A selection regression: hand the 5 PM report to a 4:30 PM tip.
    monkeypatch.setattr(m.I, "select_reports_for_date",
                        lambda day, tips, **kw: {t: I.Selection(slot, rep, 1) for t in tips})
    early = sched(("0022500602", "2026-01-15", "PHX", "DET", 16, 30))
    with pytest.raises(ValueError, match="not titled before tip"):
        m.build_season("2025-26", early, I.ReportStore(tmp_path, offline=True), _args())
    assert not list(tmp_path.glob("*.parquet"))


def test_check_before_tip_covers_team_rows():
    m = _script()
    tip = pd.Timestamp("2026-01-15T00:00:00Z")
    rows = pd.DataFrame({"game_id": [], "report_ts_et": pd.Series([], dtype="datetime64[ns, UTC]"),
                         "tip_utc": pd.Series([], dtype="datetime64[ns, UTC]")})
    tdf = pd.DataFrame({"game_id": ["g"], "report_ts_et": [tip - pd.Timedelta(minutes=4)],
                        "tip_utc": [tip]})
    with pytest.raises(ValueError, match="team rows"):
        m.check_before_tip(rows, tdf, dt.timedelta(minutes=5))
    m.check_before_tip(rows, tdf, dt.timedelta(minutes=3))


def test_negative_lead_minutes_is_refused(tmp_path, monkeypatch):
    m = _script()
    with pytest.raises(m.argparse.ArgumentTypeError):
        m._lead_minutes("-5")
    with pytest.raises(SystemExit):
        m.main(["--seasons", "2025-26", "--lead-minutes", "-5"])
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    # No modelled game to select for (a preseason id): nothing downstream
    # would refuse, and a '_lead-5' table would be written.
    pre = sched(("0012500001", "2026-01-15", "PHX", "DET", 19, 0))
    with pytest.raises(ValueError, match="negative lead"):
        m.build_season("2025-26", pre, I.ReportStore(tmp_path, offline=True), _args(lead_minutes=-5))
    assert not list(tmp_path.rglob("*.parquet"))


def _main_env(tmp_path, monkeypatch):
    from bucketlib import cache as C
    m = _script()
    monkeypatch.setattr(m, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(I, "CACHE_DIR", tmp_path)
    C.write_parquet(JAN15, tmp_path / "schedule_2025-26_v1.parquet")
    return m


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    m = _main_env(tmp_path, monkeypatch)

    def blocked(url, **kw):
        raise http.Blocked(url, "HTTP 403 text/html: Akamai edge block")
    monkeypatch.setattr(I.http, "fetch", blocked)
    for extra in ([], ["--keep-going"]):
        assert m.main(["--seasons", "2025-26", *extra]) == 3
    assert not list(tmp_path.glob("injury_report*.parquet"))

    def absent(url, **kw):
        raise http.Absent(url, "HTTP 403 S3 AccessDenied")
    monkeypatch.setattr(I.http, "fetch", absent)
    assert m.main(["--seasons", "2025-26", "--budget", "3"]) == 4

    dst = tmp_path / "raw" / "injury" / "2025-26" / "Injury-Report_2026-01-15_05_00PM.pdf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(fx(dst.name))
    assert m.main(["--seasons", "2025-26", "--offline"]) == 5
    assert not (tmp_path / "injury_reports_2025-26_v1.parquet").exists()
    assert m.main(["--seasons", "2025-26"]) == 0          # online: the 403s are real absences
    assert (tmp_path / "injury_reports_2025-26_v1.parquet").exists()


def test_flush_merges_with_a_concurrent_run(tmp_path):
    # The horizon backfill and this one share the store directory.
    def absent(url):
        raise http.Absent(url, "403")
    a, b = I.ReportStore(tmp_path, fetcher=absent), I.ReportStore(tmp_path, fetcher=absent)
    s1 = I.parse_slot_filename("Injury-Report_2024-01-15_05PM.pdf")
    s2 = I.parse_slot_filename("Injury-Report_2024-01-15_06PM.pdf")
    assert a.get(s1) is None and b.get(s2) is None
    a.flush()
    b.flush()
    c = I.ReportStore(tmp_path, fetcher=absent)
    assert c.known_absent(s1) and c.known_absent(s2)
