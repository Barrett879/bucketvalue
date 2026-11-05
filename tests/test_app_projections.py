"""The page layer's projection helpers (bucketlib/store.py, the wiring in
bucketlib/props_ui.py), on the real serve fixture (tests/fixtures/serve).

Every check calls production code; where a property is "the same as", the
other side is production code too (pipeline.live_projections,
context.display_status, simprov.medians, props.reliability_for). Inputs are
perturbed where a rule has an edge (a minute either side of a threshold, a
flag flipped), so a broken rule fails here rather than reading as a pass on
the fixture's particular rows.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import context as CTX
from bucketlib import dist as D
from bucketlib import http as HTTP
from bucketlib import live as LV
from bucketlib import models as M
from bucketlib import pipeline as P
from bucketlib import props, props_ui, simprov, store
from bucketlib.cache import MODELS_DIR
from bucketlib.sources import injury as INJ
from bucketlib.util import ET

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402

REAL_INJURY_REPORT = LV.injury_report      # before any test replaces it


@pytest.fixture(scope="module")
def mdl():
    try:
        return M.load()
    except M.ModelMismatch as e:
        pytest.skip(f"production models not loadable here: {e}")


@pytest.fixture(scope="module")
def proj() -> pd.DataFrame:
    return H.load().proj


@pytest.fixture(scope="module")
def params() -> D.SimParams:
    return D.SimParams.load(MODELS_DIR / store.SIM_PARAMS_FILE)


def _one(proj, gid, team):
    return proj[(proj["game_id"] == gid) & (proj["team"] == team)].reset_index(drop=True)


def _pid(proj, name) -> int:
    return int(proj.loc[proj["player_name"] == name, "person_id"].iloc[0])


# ── team_view ────────────────────────────────────────────────────────────────
def test_team_view_sections_order_and_numbers(proj, params):
    rows = _one(proj, H.EARLY, "DET")
    med = simprov.medians(proj, params, store.MEDIAN_STATS)
    v = store.team_view(rows, med)
    assert list(v.columns) == store.VIEW_COLS and len(v) == len(rows)
    sec = v["section"].tolist()
    assert sec == sorted(sec, key=["main", "deep", "out"].index)        # main, deep, out
    by = rows.set_index("person_id")
    for s in ("main", "deep"):
        part = v[v["section"] == s]
        assert part["min"].is_monotonic_decreasing, s
        assert ((part["min"] >= store.DEEP_BENCH_MIN) == (s == "main")).all()
    out = v[v["out"]]
    assert set(out["person_id"]) == set(by.index[by["inj_status"] == "Out"])
    assert out[[k for k, _ in store.STAT_COLS]].isna().all().all()     # Out: no numbers
    m = med.set_index(["person_id", "game_id"])
    played = v[~v["out"]]
    for s in store.MEDIAN_STATS:
        want = m.loc[list(zip(played["person_id"], played["game_id"])), f"med_{s}"].to_numpy()
        assert np.array_equal(played[s].to_numpy(), want, equal_nan=True), s
    assert np.array_equal(played["min"].to_numpy(), by.loc[played["person_id"], "min"].to_numpy())
    # FS is not a median: PrizePicks' formula on the parts printed in the row.
    parts = list(props.FANTASY_WEIGHTS)
    printed = played[parts].map(lambda x: float(f"{x:.1f}"))
    want = (printed["pts"] + 1.2 * printed["reb"] + 1.5 * printed["ast"] + 3 * printed["stl"]
            + 3 * printed["blk"] - printed["tov"])
    np.testing.assert_allclose(played["fantasy"].to_numpy(), want.to_numpy(), atol=1e-9)
    assert played["fantasy"].notna().all() and "fantasy" not in store.MEDIAN_STATS


def test_deep_bench_is_under_ten_projected_minutes(proj):
    rows = _one(proj, H.EARLY, "DET")
    healthy = rows[rows["inj_status"].isna()].sort_values("min").iloc[0]
    for m, want in ((store.DEEP_BENCH_MIN - 0.01, "deep"), (store.DEEP_BENCH_MIN, "main")):
        r = rows.copy()
        r.loc[r["person_id"] == healthy["person_id"], "min"] = m
        v = store.team_view(r).set_index("person_id")
        assert v.loc[healthy["person_id"], "section"] == want, m


def test_pplay_shows_for_listed_players_or_below_ninety_percent(proj):
    rows = _one(proj, H.EARLY, "DET")
    pid = int(rows[rows["inj_status"].isna()].sort_values("min").iloc[-1]["person_id"])

    def pplay(**change):
        r = rows.copy()
        for c, val in change.items():
            r.loc[r["person_id"] == pid, c] = val
        return store.team_view(r).set_index("person_id").loc[pid, "pplay"]
    assert pplay(p_play=0.9) == ""                       # not listed, 90%: hidden
    assert pplay(p_play=0.8999) == "90%"                 # below 90%: shown
    assert pplay(p_play=0.97, inj_status="Probable") == "97%"     # listed: shown
    assert pplay(p_play=0.999, inj_status="Probable") == ">99%"
    assert pplay(p_play=0.004, inj_status="Questionable") == "<1%"
    assert pplay(p_play=0.01, inj_status="Out") == ""    # Out: no numbers at all


def test_badges_come_from_display_status_and_unknown_is_labelled(proj):
    det, den = _one(proj, H.EARLY, "DET"), _one(proj, H.LATE, "DEN")
    v = store.team_view(det).set_index("person_id")
    shown = CTX.display_status(det).set_index("person_id")
    want = shown["status"].where(shown["status"].notna(), None)
    assert v["status"].where(v["status"].notna(), None).to_dict() == want.to_dict()
    assert {"Two-way", "Out", "Available"} <= set(v["status"].dropna())
    # A roster two-way flag badges a player the report does not list.
    pid = int(det[det["inj_status"].isna()].iloc[0]["person_id"])
    flagged = store.team_view(det.assign(two_way=det["person_id"] == pid)).set_index("person_id")
    assert flagged.loc[pid, "status"] == "Two-way"
    # Denver had not filed: every row unknown, labelled, never blank.
    vd = store.team_view(den)
    assert (vd["status"] == CTX.NOT_FILED).all()
    assert store.team_report_label(den) == CTX.NOT_FILED
    assert store.team_report_label(det) is None
    gone = den.assign(team_state="no_report")
    assert store.team_report_label(gone) == CTX.NO_REPORT
    assert 'class="bv-st unk"' in store.status_badge(CTX.NO_REPORT)
    html = store.render_projection_table("DEN", vd, team_note=store.team_report_label(den))
    assert html.count('class="bv-st unk"') == len(vd) and CTX.NOT_FILED in html


def test_names_and_reasons_from_the_feeds_render_escaped(proj):
    """Player names (cdn) and injury reasons (PDF text) are third-party
    input: a tag in either renders as text, never as markup."""
    rows = _one(proj, H.EARLY, "PHI")
    maxey, edge = _pid(proj, "Tyrese Maxey"), _pid(proj, "VJ Edgecombe")
    r = rows.assign(
        player_name=np.where(rows["person_id"] == edge, "<b>VJ</b> Edgecombe", rows["player_name"]),
        inj_reason=np.where(rows["person_id"] == maxey, "Injury/Illness - <i>Finger</i>",
                            rows["inj_reason"]))
    v = store.team_view(r)
    assert v.set_index("person_id").loc[maxey, "out"]           # still an Out row with its reason
    html = store.render_projection_table("PHI", v)
    assert "<b>VJ</b>" not in html and "&lt;b&gt;VJ&lt;/b&gt; Edgecombe" in html
    assert "<i>Finger</i>" not in html and "&lt;i&gt;Finger&lt;/i&gt;" in html


def test_projection_table_collapses_the_deep_bench_and_sinks_out_rows(proj):
    v = store.team_view(_one(proj, H.EARLY, "DET"))
    html = store.render_projection_table("DET", v)
    deep = html[html.index('<details class="bv-deep">'):html.index("</details>")]
    assert deep.count("data-pid") == (v["section"] == "deep").sum() > 0
    out = html[html.index('class="bv-pt-out"'):]
    assert out.count("data-pid") == (v["section"] == "out").sum() > 0
    assert html.index("<details") < html.index('class="bv-pt-out"')


# ── The slate ────────────────────────────────────────────────────────────────
def test_key_absences_names_listed_out_players_at_fifteen_minutes(proj):
    got = store.key_absences(proj)
    assert got[H.EARLY] == '<div class="bv-slate-sub bv-out"><b>Out:</b> Maxey, Oubre Jr. +5</div>'
    assert H.LATE not in got
    early = proj[proj["game_id"] == H.EARLY]
    maxey, george = _pid(proj, "Tyrese Maxey"), _pid(proj, "Paul George")
    tolu = _pid(proj, "Tolu Smith")                     # Out: "G League - Two-Way"

    def names(frame):
        return store.key_absences(frame).get(H.EARLY, "")
    assert "Maxey" not in names(early.assign(min=np.where(early["person_id"] == maxey, 14.99, early["min"])))
    assert "+4" in names(early.assign(min=np.where(early["person_id"] == maxey, 14.99, early["min"])))
    # Two-way and G League are excluded even at big minutes; so is a mark-out.
    assert "Smith" not in names(early.assign(min=np.where(early["person_id"] == tolu, 40.0, early["min"])))
    # A G League ASSIGNMENT (badge "G League", no two-way flag) is excluded too;
    # the same row with an injury reason is named, so the reason is what decides.
    big = early.assign(min=np.where(early["person_id"] == tolu, 40.0, early["min"]))

    def tolu_reason(reason):
        return big.assign(inj_reason=np.where(big["person_id"] == tolu, reason, big["inj_reason"]))
    assignment = tolu_reason("G League - On Assignment")
    assert not assignment["two_way"].any()
    assert "G League" in set(CTX.display_status(assignment)["status"])
    assert "Smith" not in names(assignment)
    assert "Smith" in names(tolu_reason("Injury/Illness - Left Ankle; Sprain"))
    assert "Maxey" not in names(early.assign(two_way=early["person_id"] == maxey))
    marked = early.assign(inj_reason=np.where(early["person_id"] == maxey, CTX.MARKED_REASON,
                                              early["inj_reason"]))
    assert "Maxey" not in names(marked)
    # A Questionable player is not Out.
    q = early.assign(inj_status=np.where(early["person_id"] == george, "Questionable", early["inj_status"]))
    assert "+4" in names(q)


def test_reports_filed_counts_teams_with_a_known_state(proj):
    assert store.reports_filed(proj) == (3, 4)                      # Denver had not filed
    assert store.reports_filed(proj.assign(team_state="no_report")) == (0, 4)
    assert store.reports_filed(proj.iloc[0:0]) is None


def test_card_subs_say_why_a_withheld_game_has_no_projection(proj):
    early = proj[proj["game_id"] == H.EARLY]
    served = store.Served(early, P.Info(date=H.DATE, source="live", withheld=[H.LATE]), H.MORNING)
    subs = store.card_subs(served)
    assert subs[H.LATE] == f'<div class="bv-slate-sub">{P.NO_LINE}</div>'
    assert "Maxey" in subs[H.EARLY]                           # the key absences stay
    assert store.card_subs(store.Served(early, P.Info(date=H.DATE), H.MORNING)).keys() == {H.EARLY}


def test_void_ids_are_exactly_the_frames_out_rows(monkeypatch, tmp_path, proj, mdl):
    """What the served frame reads Out, nothing else: a mark the live layer
    applied reads Out (so it is void), a mark it did not apply is not."""
    out = set(proj.loc[proj["inj_status"] == "Out", "person_id"].astype(int))
    assert props_ui.void_ids(proj) == out and len(out) >= 10
    assert props_ui.void_ids(None) == set() and props_ui.void_ids(proj.iloc[0:0]) == set()
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    cade = _pid(proj, "Cade Cunningham")
    marked, _ = P.live_projections(H.DATE, H.MORNING, marked_out={cade}, report=w.report,
                                   odds=w.odds, models=mdl, schedule=w.sched)
    assert props_ui.void_ids(marked) == out | {cade}
    tipped, info = P.live_projections(H.DATE, H.EVENING, marked_out={cade}, report=w.report,
                                      odds=w.odds, models=mdl, schedule=w.sched)
    assert H.EARLY in info.frozen and cade not in props_ui.void_ids(tipped)


# ── ?out= ────────────────────────────────────────────────────────────────────
def test_out_param_parses_junk_safely_and_round_trips():
    assert store.parse_out("203999, 1630595,abc,-3,1630595,1.5,") == (203999, 1630595)
    assert store.parse_out(None) == () and store.parse_out("") == ()
    assert len(store.parse_out(",".join(str(i) for i in range(100)))) == store.MAX_MARKS
    assert store.out_param([3, 1, 3]) == "1,3" and store.out_param([]) is None
    assert store.parse_out(store.out_param([5, 2])) == (2, 5)


# ── The loader ───────────────────────────────────────────────────────────────
def test_loader_is_live_projections_for_the_pages_inputs(monkeypatch, tmp_path, mdl):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    served = store.load_projections(H.DATE)
    # The loader hands live_projections the live layer's Live objects.
    want, info = P.live_projections(H.DATE, H.MORNING, report=LV.Live(w.report, 0.0),
                                    odds=LV.Live(w.odds, 0.0), models=mdl, schedule=w.sched)
    pd.testing.assert_frame_equal(served.frame, want, check_exact=True)
    assert served.info == info and served.info.source == "live" and served.past == frozenset()
    cade = _pid(w.proj, "Cade Cunningham")
    marked = store.load_projections(H.DATE, [cade])
    want, _ = P.live_projections(H.DATE, H.MORNING, marked_out={cade}, report=w.report,
                                 odds=w.odds, models=mdl, schedule=w.sched)
    pd.testing.assert_frame_equal(marked.frame, want, check_exact=True)
    assert marked.info.adjusted and marked.marks == (cade,)


def test_loader_cache_keys_on_the_report_the_lines_and_the_freeze(monkeypatch, tmp_path, mdl):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    first = store.load_projections(H.DATE)
    # A later report: the key changes, the frame is recomputed with it.
    title = dt.datetime(2026, 3, 12, 13, 0, tzinfo=ET)
    later = INJ.ParsedReport({**w.report.meta, "report_ts_et": title},
                             w.report.rows.assign(report_ts_et=title),
                             w.report.team_rows.assign(report_ts_et=title))
    monkeypatch.setattr(LV, "injury_report", lambda sched=None: LV.Live(later, 0.0))
    H.pin(monkeypatch, dt.datetime(2026, 3, 12, 13, 5, tzinfo=ET))
    second = store.load_projections(H.DATE)
    assert second.info.report_ts_et == title != first.info.report_ts_et
    # New lines: recomputed with them.
    moved = {k: {**v, "home_line": v["home_line"] - 1.0} for k, v in w.odds.items()}
    monkeypatch.setattr(LV, "odds_on", lambda d: LV.Live(moved, 0.0))
    third = store.load_projections(H.DATE)
    assert (third.frame["home_line"].to_numpy() == second.frame["home_line"].to_numpy() - 1.0).all()
    # Crossing a freeze point inside the TTL: the early game is served committed.
    H.pin(monkeypatch, dt.datetime(2026, 3, 12, 18, 54, tzinfo=ET))
    before = store.load_projections(H.DATE)
    H.pin(monkeypatch, dt.datetime(2026, 3, 12, 18, 56, tzinfo=ET))
    after = store.load_projections(H.DATE)
    assert before.info.frozen == [] and after.info.frozen == [H.EARLY]
    assert after.past == frozenset({H.EARLY})
    pd.testing.assert_frame_equal(after.game(H.EARLY).drop(columns="frozen").reset_index(drop=True),
                                  w.proj[w.proj["game_id"] == H.EARLY].drop(columns="frozen")
                                  .reset_index(drop=True), check_exact=True)


def _count_computes(monkeypatch) -> list:
    calls = []
    real = P.live_projections

    def counting(*a, **k):
        calls.append(sorted(k.get("marked_out") or ()))
        return real(*a, **k)
    monkeypatch.setattr(P, "live_projections", counting)
    return calls


def test_loader_keys_on_content_not_on_how_the_feed_answered(monkeypatch, tmp_path, mdl):
    """Through the REAL live._cached: the first read of the report and the
    lines is source "live", every read inside the TTL is "memory", and a
    failed refresh serves the same report stale. None of that is new input,
    so the date is computed once."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    monkeypatch.setattr(LV, "_good", {})
    monkeypatch.setattr(LV, "_fail", {})
    monkeypatch.setattr(LV, "injury_report", REAL_INJURY_REPORT)
    fetches = []

    def latest(now, getter=None, max_slots=None, accept=None):
        fetches.append(now)
        return INJ.Selection(None, w.report, 1)
    monkeypatch.setattr(INJ, "latest_report", latest)
    monkeypatch.setattr(LV, "odds_on", lambda d: LV._cached(f"odds:{d.isoformat()}", 300, 90,
                                                            lambda: w.odds))
    calls = _count_computes(monkeypatch)
    first = store.load_projections(H.DATE)
    assert LV.injury_report(w.sched).source == "memory"
    assert LV.odds_on(H.DATE).source == "memory"
    again = [store.load_projections(H.DATE) for _ in range(2)]
    # The report's TTL runs out and the refresh fails: the same report, stale.
    LV._good[LV.INJURY_KEY] = (time.time() - 10 * LV.INJURY_TTL, w.report)

    def down(*a, **k):
        raise HTTP.FetchError("injury", "down")
    monkeypatch.setattr(INJ, "latest_report", down)
    stale = LV.injury_report(w.sched)
    assert stale.stale and stale.source == "memory" and stale.value is w.report
    again.append(store.load_projections(H.DATE))
    assert len(fetches) == 1 and calls == [[]], calls
    for s in again:
        pd.testing.assert_frame_equal(s.frame, first.frame, check_exact=True)


def test_marks_naming_nobody_or_a_listed_out_player_cost_nothing(monkeypatch, tmp_path, mdl):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    calls = _count_computes(monkeypatch)
    plain = store.load_projections(H.DATE)
    maxey = _pid(w.proj, "Tyrese Maxey")                    # listed Out, "Right Finger"
    for marks in ([1], [2, 3], [999999999], [maxey], [maxey, 5]):
        s = store.load_projections(H.DATE, marks)
        assert s.marks == () and s.applied == () and not s.info.adjusted, marks
        pd.testing.assert_frame_equal(s.frame, plain.frame, check_exact=True)
    assert calls == [[]], calls
    # Maxey keeps his own reason and his place on the card's Out line.
    row = plain.frame[plain.frame["person_id"] == maxey].iloc[0]
    assert "Right Finger" in row["inj_reason"]
    assert "Maxey" in store.card_subs(s)[H.EARLY]
    # A real mark next to the junk: only it reaches the key and the frame.
    cade = _pid(w.proj, "Cade Cunningham")
    s = store.load_projections(H.DATE, [cade, maxey, 7])
    assert s.marks == (cade,) and s.applied == (cade,) and calls == [[], [cade]]
    again = store.load_projections(H.DATE, [cade])
    pd.testing.assert_frame_equal(again.frame, s.frame, check_exact=True)
    assert len(calls) == 2


def test_projected_ids_are_the_pipelines_state_games(proj):
    w = H.load()
    assert store.projected_ids(w.sched, H.DATE) == {H.EARLY, H.LATE}
    assert store.projected_ids(w.sched, H.DATE) == set(P.state_games(w.sched, H.DATE)["game_id"])
    po = w.sched.assign(game_type=np.where(w.sched["game_id"] == H.LATE, "PO", w.sched["game_type"]))
    assert store.projected_ids(po, H.DATE) == {H.EARLY}
    assert store.projected_ids(None, H.DATE) == set()


def test_an_unpublished_date_fetches_nothing(monkeypatch, tmp_path):
    H.install(monkeypatch, tmp_path, H.MORNING)

    def boom(*a, **k):
        raise AssertionError("a date with nothing published must not fetch")
    monkeypatch.setattr(LV, "injury_report", boom)
    monkeypatch.setattr(LV, "odds_on", boom)
    served = store.load_projections(dt.date(2026, 3, 13))
    assert served.frame.empty and served.info.reason == P.NOT_PUBLISHED


def test_after_every_tip_the_loader_fetches_nothing(monkeypatch, tmp_path):
    w = H.install(monkeypatch, tmp_path, dt.datetime(2026, 3, 12, 23, 0, tzinfo=ET))

    def boom(*a, **k):
        raise AssertionError("every game is frozen: nothing to fetch")
    monkeypatch.setattr(LV, "injury_report", boom)
    monkeypatch.setattr(LV, "odds_on", boom)
    served = store.load_projections(H.DATE)
    assert served.info.frozen == sorted([H.EARLY, H.LATE])
    pd.testing.assert_frame_equal(served.frame.drop(columns="frozen"),
                                  w.proj.drop(columns="frozen"), check_exact=True)


def test_table_medians_and_the_board_share_one_provider(monkeypatch, tmp_path, params):
    """One simulation per frame: the Game tables' medians build the provider,
    the board (on an equal copy of the frame) reuses it."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    monkeypatch.setattr(store, "sim_params", lambda: params)
    monkeypatch.setattr(simprov, "_PROVIDERS", OrderedDict())     # nothing shared yet
    built = []
    real = simprov.SimProvider.__init__

    def counting(self, *a, **k):
        built.append(self)
        real(self, *a, **k)
    monkeypatch.setattr(simprov.SimProvider, "__init__", counting)
    frame = w.proj
    med = store.table_medians(frame)
    view = props_ui.date_board(H.paste(w), H.DATE, frame.copy(), params, w.roster)
    assert len(built) == 1 and len(view.shown) > 0
    assert med.equals(built[0].medians(store.MEDIAN_STATS))
    # The line's model value is the table's median for that player and stat.
    m = med.set_index(["person_id", "game_id"])
    for r in view.shown.to_dict(orient="records"):
        if r["stat"] in store.MEDIAN_STATS:
            assert r["model"] == m.loc[(r["person_id"], r["game_id"]), f"med_{r['stat']}"], r
    # A Fantasy Score line's model value is the table's FS for that player:
    # one number, the formula on the medians the page prints.
    tv = pd.concat([store.team_view(g, med) for _, g in frame.groupby(["game_id", "team"])])
    tv = tv[~tv["out"]].set_index(["person_id", "game_id"])
    lines = pd.DataFrame([{"name": n, "team": t, "position": "", "stat_type": "Fantasy Score",
                           "line": 30.5, "start_time": None, "direction": "both",
                           "odds_type": "standard", "opp": None, "is_combo": False, "league": None,
                           "source": "board", "person_id": int(pid), "game_id": str(gid),
                           "match": "matched"}
                          for (pid, gid), n, t in zip(tv.index, tv["name"], frame.set_index(
                              ["person_id", "game_id"]).loc[tv.index, "team"])])
    table, meta = props.compare(lines, simprov.provider_for(frame, params), frame)
    assert len(table) == len(tv) and meta["unprojected"] == 0, meta["unprojected_why"]
    for r in table.to_dict(orient="records"):
        assert r["model"] == tv.loc[(r["person_id"], r["game_id"]), "fantasy"], r


# ── Notices ──────────────────────────────────────────────────────────────────
def test_stale_notes_never_render_the_raw_error(proj):
    info = P.Info(date=H.DATE, source="committed",
                  stale="FileNotFoundError: [Errno 2] /Users/someone/secret/state/x.parquet")
    served = store.Served(proj, info, H.MORNING)
    page = " ".join(store.projection_notes(served, H.EARLY, today=H.DATE))
    assert "/Users" not in page and "Errno" not in page
    assert store.stale_reason(info.stale) in page
    assert store.stale_reason("NoReport: no injury report at all") == "no injury report could be read"
    assert store.stale_reason("RuntimeError: lines feed failed (x)") == "the lines feed did not answer"


def test_frozen_and_provisional_notes(proj):
    frozen = P.Info(date=H.DATE, source="live", frozen=[H.EARLY])
    page = " ".join(store.projection_notes(store.Served(proj, frozen, H.EVENING,
                                                        past=frozenset({H.EARLY})), H.EARLY))
    assert "Frozen at tip" in page
    prov = proj.assign(provisional=True)
    info = P.Info(date=H.DATE + dt.timedelta(days=1), source="live", provisional=True)
    page = " ".join(store.projection_notes(store.Served(prov, info, H.MORNING), H.EARLY,
                                           today=H.DATE))
    assert "Provisional" in page and "future date" in page
    assert not store.projection_notes(store.Served(proj, P.Info(date=H.DATE, source="live"),
                                                   H.MORNING), H.EARLY, today=H.DATE)


# ── About ────────────────────────────────────────────────────────────────────
def test_trust_table_is_the_boards_reliability(monkeypatch):
    t = store.trust_table().set_index("stat")
    assert list(t.index[t["source"] == "model"]) == list(simprov.GATED)
    assert set(t.index[t["source"] == "baseline"]) == set(simprov.BASELINE)
    for s in simprov.GATED:
        rel = props.reliability_for(s)
        assert (t.loc[s, "tier"], t.loc[s, "edge"]) == (rel["tier"], rel["edge"]), s
    # A combo shows its weakest component's measurement and names it.
    assert t.loc["pra", "via"] in props.tier_components("pra")[1:]
    via = props.reliability_for(t.loc["pra", "via"])
    assert (via["tier"], via["edge"]) == (t.loc["pra", "tier"], t.loc["pra", "edge"])
    assert t.loc["pra", "own_tier"] == "strong" and t.loc["pra", "tier"] != "strong"
    assert t.loc["pts", "via"] is None
    assert (t.loc[list(simprov.BASELINE), "tier"] == "unmeasured").all()
    html = store.render_trust_table(store.trust_table())
    assert html.count("<tr><td") == len(t)
    assert round(store.breakeven(2), 1) == 57.7
    assert props.BREAKEVEN_LO <= store.breakeven(2) <= props.BREAKEVEN_HI
    caveats = " ".join(store.trust_caveats())
    assert "PROXY" in caveats and "2023-24" in caveats


# ── Audit fixes, 2026-10-09 ──────────────────────────────────────────────────
def test_the_slate_trust_line_is_generated_from_the_boards_tiers(monkeypatch):
    """DiamondValue's "most predictable stat" line, generated from the tiers
    the board ranks by, never typed: change the table and the line follows."""
    t = store.trust_table()
    model = t[t["source"] == "model"]
    line = store.trust_line()
    strong = list(model.loc[model["tier"] == "strong", "label"])
    weak = list(model.loc[model["tier"] == "weak", "label"])
    assert strong and weak
    for lab in strong + weak:
        assert lab in line
    for lab in model.loc[model["tier"] == "moderate", "label"]:
        assert lab not in line, lab
    monkeypatch.setattr(store, "trust_table", lambda: pd.DataFrame(
        [{"label": "Rebounds", "tier": "strong", "source": "model"},
         {"label": "Assists", "tier": "strong", "source": "model"},
         {"label": "Steals", "tier": "moderate", "source": "model"},
         {"label": "FT Made", "tier": "unmeasured", "source": "baseline"}]))
    got = store.trust_line()
    assert "Rebounds and Assists are the board's most reliable stats" in got
    assert "weakest" not in got and "Steals" not in got and "FT Made" not in got
    monkeypatch.setattr(store, "trust_table", lambda: pd.DataFrame(
        [{"label": "Steals", "tier": "moderate", "source": "model"}]))
    assert store.trust_line() == ""

    def gone():
        raise FileNotFoundError("data/stat_reliability.csv")
    monkeypatch.setattr(store, "trust_table", gone)
    assert store.trust_line() == ""


def _visible(markup: str) -> str:
    import re
    from html import unescape
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def test_date_board_without_sim_params_is_the_explained_failure(monkeypatch, tmp_path):
    """sim_params() failed: the board says it could not be built, and never
    files the visitor's lines as 'no projection: internal error'."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = H.paste(w)
    v = props_ui.date_board(token, H.DATE, w.proj, None, w.roster)
    assert v.meta.get("error") is True and v.meta["posted"] == 11
    assert v.matched == 0 and "unprojected" not in v.meta


def test_stale_with_no_rows_says_why_once(proj):
    """Features committed, no projections, the live layer down: no 'Showing
    the published projections' over an empty page; one notice that says
    why the live update could not run."""
    info = P.Info(date=H.DATE, source="none", reason=P.NOT_PUBLISHED,
                  stale="RuntimeError: injury: live fetch failed (Blocked) and nothing is cached")
    served = store.Served(proj.iloc[0:0], info, H.MORNING)
    for gid in (H.EARLY, None):
        assert "Showing the published projections" not in " ".join(
            store.projection_notes(served, gid, today=H.DATE))
    for msg in (store.empty_notice(served, H.EARLY),
                store.empty_notice(served, game_ids={H.EARLY, H.LATE})):
        text = _visible(msg)
        assert "not up yet, and the live update could not run" in text
        assert "the injury report could not be fetched" in text
    # With rows, the stale notice is still there.
    with_rows = store.Served(proj, info, H.MORNING)
    assert "Showing the published projections" in " ".join(store.projection_notes(with_rows, H.EARLY))


def test_a_tipped_game_without_a_projection_promises_none(monkeypatch, tmp_path, proj):
    info = P.Info(date=H.DATE, source="none", reason=P.NOT_PUBLISHED, unpublished=[H.EARLY, H.LATE])
    served = store.Served(proj.iloc[0:0], info, H.EVENING, past=frozenset({H.EARLY, H.LATE}))
    game = _visible(store.empty_notice(served, H.EARLY))
    assert "No projection was published before tip" in game and "publishes them once" not in game
    slate = _visible(store.empty_notice(served, game_ids={H.EARLY, H.LATE}))
    assert "No projections were published for this date before tip" in slate
    assert "publishes them once" not in slate
    # One game still ahead: the date's promise stands for it.
    part = store.Served(proj.iloc[0:0], info, H.EVENING, past=frozenset({H.EARLY}))
    assert "publishes them once" in _visible(store.empty_notice(part, game_ids={H.EARLY, H.LATE}))
    # The loader knows which games tipped even for a date with nothing committed.
    H.install(monkeypatch, tmp_path, dt.datetime(2026, 3, 12, 23, 0, tzinfo=ET))
    P.features_path(H.DATE).unlink()
    P.projections_path(H.DATE).unlink()
    s = store.load_projections(H.DATE)
    assert s.frame.empty and s.past == frozenset({H.EARLY, H.LATE})


def test_the_frozen_notice_says_when_and_with_which_report(proj):
    rows = proj[proj["game_id"] == H.EARLY]
    assert store.frozen_caption(rows, H.DATE) == \
        "Frozen at tip: projected 12:20 PM ET with the 12:00 PM ET report"
    later = dt.datetime(2026, 3, 12, 18, 45, tzinfo=ET)
    info = P.Info(date=H.DATE, source="live", frozen=[H.EARLY], report_ts_et=later)
    page = _visible(" ".join(store.projection_notes(
        store.Served(proj, info, H.EVENING, past=frozenset({H.EARLY})), H.EARLY)))
    assert "Frozen at tip." in page
    assert "published at 12:20 PM ET with the 12:00 PM ET injury report" in page
    assert "latest read is 6:45 PM ET" in page and "are not reflected" in page
    same = P.Info(date=H.DATE, source="live", frozen=[H.EARLY],
                  report_ts_et=dt.datetime(2026, 3, 12, 12, 0, tzinfo=ET))
    page = _visible(" ".join(store.projection_notes(
        store.Served(proj, same, H.EVENING, past=frozenset({H.EARLY})), H.EARLY)))
    assert "not reflected" not in page


def test_a_started_game_shows_the_line_its_projection_used(proj):
    """ESPN drops the odds at tip: the header, the card and the masthead
    count read the frozen rows' own line for a started game, labelled; an
    untipped game without a line stays 'not posted'."""
    w = H.load()
    dropped = {k: v for k, v in w.odds.items() if k != ("PHI", "DET")}
    got = store.posted_lines(dropped, w.sched, proj, started={H.EARLY})
    phi = got[("PHI", "DET")]
    assert (phi["home_line"], phi["total"], phi["pre_tip"]) == (-6.5, 219.5, True)
    assert ("PHI", "DET") not in store.posted_lines(dropped, w.sched, proj, started=())
    assert store.lines_posted(got, w.sched) == 2 and store.lines_posted(dropped, w.sched) == 1
    facts = _visible("".join(store.market_facts(phi, "PHI", "DET")))
    assert "DET -6.5" in facts and "219.5" in facts and store.PRE_TIP in facts
    assert "PHI 106.5" in facts and "DET 113.0" in facts
    assert "Unavailable" in _visible("".join(store.market_facts(None, "PHI", "DET", unavailable=True)))
    assert "Not posted" in _visible("".join(store.market_facts(None, "PHI", "DET")))
    # The live line, when there is one, is never replaced.
    assert store.posted_lines(w.odds, w.sched, proj.assign(home_line=1.0), {H.EARLY}) == w.odds


def test_a_tipped_game_without_rows_shows_its_line_from_the_committed_snapshots(
        monkeypatch, tmp_path, proj):
    """A started game with no projection rows (withheld, never published, not
    projected) and no live line shows the bot's latest pre-tip capture,
    labelled; a game with rows keeps its rows' line; an untipped game and a
    live line are never replaced. The snapshots are read from the one odds
    path (state.odds_path)."""
    w = H.install(monkeypatch, tmp_path, H.EVENING)
    t0 = pd.Timestamp("2026-03-12 16:00", tz="UTC")
    snaps = pd.DataFrame({"away": ["PHI", "PHI", "DEN"], "home": ["DET", "DET", "SAS"],
                          "home_line": [-5.0, -7.0, 3.5], "total": [218.0, 220.0, 231.0],
                          "captured_at": [t0, t0 + pd.Timedelta(hours=2), t0]})
    assert store.committed_odds(H.DATE) is None
    from bucketlib import cache as C
    from bucketlib import state as S
    C.write_parquet(snaps, S.odds_path(H.DATE))
    got = store.committed_odds(H.DATE)
    pd.testing.assert_frame_equal(got, snaps)
    none = {}
    no_rows = proj[proj["game_id"] != H.EARLY]
    phi = store.posted_lines(none, w.sched, no_rows, {H.EARLY}, snapshots=got)[("PHI", "DET")]
    assert (phi["home_line"], phi["total"], phi["pre_tip"]) == (-7.0, 220.0, True)   # the latest capture
    # No frame at all (a game the site does not project): the snapshot still.
    assert store.posted_lines(none, w.sched, None, {H.EARLY}, snapshots=got)[("PHI", "DET")] == phi
    # Rows win over the snapshot; untipped games and live lines are untouched.
    assert store.posted_lines(none, w.sched, proj, {H.EARLY}, snapshots=got)[("PHI", "DET")][
        "home_line"] == -6.5
    assert store.posted_lines(none, w.sched, no_rows, (), snapshots=got) == {}
    assert store.posted_lines(w.odds, w.sched, no_rows, {H.EARLY, H.LATE}, snapshots=got) == w.odds
    assert ("PHI", "DET") not in store.posted_lines(none, w.sched, no_rows, {H.EARLY})


def test_the_game_notes_name_the_games_own_gaps_only(proj):
    """Provisional is per game: on /Game a game provisional for its own gaps
    says earlier games of its teams are missing (a gap can be a game still
    in progress, so never 'final'); a game on the same date without gaps
    says nothing of the kind. The slate says it for the date."""
    gap = "0022500941"
    info = P.Info(date=H.DATE, source="live", provisional=True, completeness=[gap],
                  completeness_by_game={H.EARLY: [], H.LATE: [gap]})
    frame = proj.assign(provisional=proj["game_id"] == H.LATE)
    served = store.Served(frame, info, H.MORNING)
    late = _visible(" ".join(store.projection_notes(served, H.LATE, today=H.DATE)))
    assert "Provisional." in late
    assert "Earlier game(s) of these teams are not in the box scores yet" in late
    assert "final game" not in late
    early = _visible(" ".join(store.projection_notes(served, H.EARLY, today=H.DATE)))
    assert "Provisional" not in early and "Earlier game(s)" not in early
    # Rows provisional for another reason (the features stamp): no gap claimed.
    stamp = P.Info(date=H.DATE, source="live", provisional=True, completeness=[gap],
                   completeness_by_game={H.EARLY: [], H.LATE: [gap]}, notes=[P.STALE_FEATURES])
    both = _visible(" ".join(store.projection_notes(
        store.Served(proj.assign(provisional=True), stamp, H.MORNING), H.EARLY, today=H.DATE)))
    assert "built before the latest finals" in both and "Earlier game(s)" not in both
    slate = _visible(" ".join(store.projection_notes(served, today=H.DATE)))
    assert "Earlier game(s) of teams playing on this date are not in the box scores yet" in slate
    # A game with no rows (withheld: no line yet) reads its OWN flag
    # (Info.provisional_by_game), never the date's, which LATE's gap sets.
    own = {H.EARLY: False, H.LATE: True}
    held = P.Info(date=H.DATE, source="live", provisional=True, completeness=[gap],
                  completeness_by_game={H.EARLY: [], H.LATE: [gap]}, provisional_by_game=own,
                  withheld=[H.EARLY])
    no_early = store.Served(frame[frame["game_id"] != H.EARLY], held, H.MORNING)
    assert "Provisional" not in _visible(" ".join(
        store.projection_notes(no_early, H.EARLY, today=H.DATE)))
    held_late = P.Info(date=H.DATE, source="live", provisional=True, completeness=[gap],
                       completeness_by_game={H.EARLY: [], H.LATE: [gap]}, provisional_by_game=own,
                       withheld=[H.LATE])
    no_late = _visible(" ".join(store.projection_notes(
        store.Served(frame[frame["game_id"] != H.LATE], held_late, H.MORNING), H.LATE,
        today=H.DATE)))
    assert "Provisional." in no_late and "Earlier game(s) of these teams" in no_late
    # No flag of its own (past its freeze point, unpublished): nothing to call provisional.
    gone = P.Info(date=H.DATE, source="live", provisional=True, completeness=[gap],
                  completeness_by_game={H.EARLY: [], H.LATE: [gap]}, unpublished=[H.EARLY])
    assert "Provisional" not in _visible(" ".join(
        store.projection_notes(store.Served(frame[frame["game_id"] != H.EARLY], gone, H.MORNING),
                               H.EARLY, today=H.DATE)))


def test_an_unexpected_live_failure_reads_as_an_error_not_a_feed(proj):
    why = f"{P.LIVE_FRAME_FAILED}: TypeError: injury table has no column x"
    assert store.stale_reason(why) == "the live update hit an error"
    info = P.Info(date=H.DATE, source="committed", stale=why)
    page = _visible(" ".join(store.projection_notes(store.Served(proj, info, H.MORNING), H.EARLY)))
    assert "the live update hit an error" in page and "TypeError" not in page


def _tbd(gid="0062600010", **kw):
    return pd.Series({"game_id": gid, "away": None, "home": None, "game_type": "RS",
                      "tip_utc": pd.Timestamp("2026-12-08T05:00:00Z"), "status": "scheduled",
                      "status_text": "TBD", "time_tbd": True, "label": "Emirates NBA Cup",
                      "subtype": "in-season-knockout", "postponed": False, **kw})


def test_cup_placeholders_render_as_tbd_cards_never_at_midnight():
    d = dt.date(2026, 12, 8)
    card = store.render_slate_card(_tbd(), d, {})
    assert card.startswith('<div class="bv-slate-card') and store.TBD_TIME in card
    assert "12:00 AM" not in card and "Emirates NBA Cup" in card
    two = pd.DataFrame([_tbd(), _tbd("0062600011")])
    mast = _visible(store.render_masthead(d, two, 0))
    assert "TBD First tip" in mast and "12:00 AM" not in mast
    slate = _visible(store.render_slate(two, d, {}))
    assert slate.startswith(store.TBD_TIME) and "12:00 AM" not in slate
    # A timed game beside a placeholder: its tip is the first tip, and the
    # placeholder is grouped last.
    timed = _tbd("0022600300", away="BOS", home="DET", status_text="7:00 pm ET", time_tbd=False,
                 tip_utc=pd.Timestamp("2026-12-09T00:00:00Z"))
    both = pd.DataFrame([_tbd(), timed])
    assert "7:00 PM ET First tip" in _visible(store.render_masthead(d, both, 0))
    s = _visible(store.render_slate(both, d, {}))
    assert s.index("7:00 PM ET") < s.index(store.TBD_TIME)
    # The card panel and stripe are on the unlinked placeholder too.
    from bucketlib import theme
    import re
    assert re.search(r"(?m)^\s*\.bv-slate-card \{\s*position: relative", theme.COMMON_CSS)
    assert re.search(r"(?m)^\s*\.bv-slate-card::before \{", theme.COMMON_CSS)


def test_postponed_and_upcoming_cards():
    w = H.load()
    row = w.sched[w.sched["game_id"] == H.LATE].iloc[0]
    up = store.render_slate_card(row, H.DATE, {}, w.odds[("DEN", "SAS")], "upcoming")
    assert '<span class="bv-slate-tag upcoming">Upcoming</span>' in up and "SAS -3.5" in up
    ppd = store.render_slate_card(row, H.DATE, {}, w.odds[("DEN", "SAS")], "postponed", b2b=("DEN",))
    assert ">Postponed</span>" in ppd and "SAS -3.5" not in ppd and "B2B" not in ppd
    hdr = _visible(store.render_matchup_header(row.copy(), "postponed"))
    assert hdr.startswith("Postponed")
    games = w.sched.assign(postponed=w.sched["game_id"] == H.LATE)
    assert list(store.active_games(games)["game_id"]) == [H.EARLY]
    assert "1 Games" in _visible(store.render_masthead(H.DATE, store.active_games(games), 1))


def test_the_preseason_masthead_counts_preseason_games_and_no_lines():
    w = H.load()
    pre = w.sched.assign(game_type="PRE")
    mast = _visible(store.render_masthead(H.DATE, pre, None, games_label="Preseason games"))
    assert "2 Preseason games" in mast and "Lines posted" not in mast and "0 Games" not in mast
    assert "Lines posted" not in _visible(store.render_masthead(H.DATE, pre.iloc[0:0], None))


def test_key_absences_read_season_minutes_per_game(proj):
    """Spec 5: 'Out: X' names 15+ MPG players. A 17-MPG player projected at
    13 if he plays is named; a 12-MPG player projected at 16 is not; with no
    MPG for a player the projection's minutes decide."""
    early = proj[proj["game_id"] == H.EARLY]
    maxey, oubre = _pid(proj, "Tyrese Maxey"), _pid(proj, "Kelly Oubre Jr.")
    f = early.assign(min=np.where(early["person_id"] == maxey, 13.0,
                                  np.where(early["person_id"] == oubre, 16.0, early["min"])))
    mpg = {(maxey, H.EARLY): 17.0, (oubre, H.EARLY): 12.0}
    got = store.key_absences(f, mpg=mpg, show=20)[H.EARLY]
    assert "Maxey" in got and "Oubre" not in got
    plain = store.key_absences(f, show=20)[H.EARLY]         # no MPG known: projected minutes
    assert "Maxey" not in plain and "Oubre" in plain
    # season_mpg reads this season, else last season, from the committed features.
    feats = H.load().feats
    m = store.season_mpg(feats)
    row = feats[feats["person_id"] == maxey].iloc[0]
    assert m[(maxey, H.EARLY)] == pytest.approx(row["min_std_s"])
    rookie = feats[feats["min_std_s"].isna() & feats["p1_mpg"].notna()]
    for _, r in rookie.iterrows():
        assert m[(int(r["person_id"]), str(r["game_id"]))] == pytest.approx(r["p1_mpg"])


def test_projected_starters_and_starters_then_bench(proj):
    feats = H.load().feats
    st_ = store.projected_starters(proj, feats)
    det = _one(proj, H.EARLY, "DET")
    mine = {p for p, g in st_ if g == H.EARLY and p in set(det["person_id"])}
    assert len(mine) == store.STARTERS
    by = det.set_index("person_id")
    assert all(by.loc[p, "inj_status"] != "Out" and by.loc[p, "p_play"] >= 0.5 for p in mine)
    v = store.team_view(det, starters=st_)
    main = v[v["section"] == "main"]
    assert main["starter"].tolist() == sorted(main["starter"], reverse=True)       # starters first
    for flag in (True, False):
        assert main.loc[main["starter"] == flag, "min"].is_monotonic_decreasing
    assert store.render_projection_table("DET", v).count('class="bv-sd"') == store.STARTERS
    # A bench player projected above every starter still sits after them.
    big = int(v.loc[~v["starter"] & ~v["out"]].iloc[0]["person_id"])
    v2 = store.team_view(det.assign(min=np.where(det["person_id"] == big, 47.0, det["min"])), starters=st_)
    main2 = v2[v2["section"] == "main"]
    assert int(main2.iloc[store.STARTERS]["person_id"]) == big and main2.iloc[:store.STARTERS]["starter"].all()
    # The share of starts decides: a bench player who started every game
    # comes in; an Out player never does, whatever his share.
    bench = int(v.loc[~v["starter"] & ~v["out"]].sort_values("min").iloc[-1]["person_id"])
    out_p = int(det.loc[det["inj_status"] == "Out", "person_id"].iloc[0])
    f2 = feats.assign(start_share_s=np.where(feats["person_id"].isin([bench, out_p]), 1.01,
                                             feats["start_share_s"]))
    st2 = store.projected_starters(proj, f2)
    assert (bench, H.EARLY) in st2 and (out_p, H.EARLY) not in st2
    # No starts history yet (opening night): projected minutes alone.
    st3 = store.projected_starters(proj, feats.assign(start_share_s=np.nan))
    top5 = det[(det["inj_status"] != "Out") & (det["p_play"] >= 0.5)].nlargest(5, "min")["person_id"]
    assert {p for p, g in st3 if g == H.EARLY and p in set(det["person_id"])} == set(top5)


def test_rest_and_back_to_back_from_the_committed_features():
    feats = H.load().feats
    rest = store.rest_by_game(feats)
    assert store.b2b_by_game(rest) == {H.LATE: ["DEN"]}
    assert _visible(store.rest_fact(rest[H.LATE], "DEN", "SAS")) == \
        "Rest DEN Back-to-back / SAS 1 day off"
    assert store.rest_fact(None, "DEN", "SAS") is None
    w = H.load()
    row = w.sched[w.sched["game_id"] == H.LATE].iloc[0]
    card = store.render_slate_card(row, H.DATE, {}, b2b=tuple(store.b2b_by_game(rest)[H.LATE]))
    assert ">DEN B2B</span>" in card
    slate = store.render_slate(w.sched, H.DATE, {}, b2b=store.b2b_by_game(rest))
    assert slate.count("B2B</span>") == 1


def _box():
    from bucketlib.sources import cdnbox
    return LV.Box(H.EARLY, True, cdnbox.player_lines(H.box_payload()))


def test_after_tip_lines_show_what_happened_never_a_push(proj):
    cade, duren, drummond = (_pid(proj, n) for n in ("Cade Cunningham", "Jalen Duren", "Andre Drummond"))
    recs = [{"person_id": cade, "stat": "pts", "line": 25.5, "lean": "Less"},
            {"person_id": duren, "stat": "reb", "line": 10.0, "lean": "More"},
            {"person_id": drummond, "stat": "reb", "line": 6.5, "lean": "More"},
            {"person_id": cade, "stat": "pra", "line": 30.5, "lean": "Less"}]
    out = store.line_actuals(recs, store.actuals_from(_box()))
    assert (out[0]["actual"], out[0]["settle"]) == (8.0, None)
    assert (out[1]["actual"], out[1]["settle"]) == (10.0, "exact")
    assert (out[2]["actual"], out[2]["settle"]) == (None, "void")
    assert out[3]["actual"] == 26.0                                  # 8 + 5 + 13
    html = props_ui.render_player_lines(out)
    text = _visible(html)
    assert "Did not play, pick voided" in text and "10: landed on the line" in text
    assert "push" not in html.lower() and "Final" in text
    # Live, not final: nobody is voided yet, and nothing is settled.
    live_box = LV.Box(H.EARLY, False, _box().lines)
    live_out = store.line_actuals(recs, store.actuals_from(live_box))
    assert all(r["settle"] is None for r in live_out)
    assert "So far" in _visible(props_ui.render_player_lines(live_out))
    # A missing component is no value, never a zero.
    assert store.actual_of("pra", {"pts": 8, "reb": 5}) is None
    assert props_ui.actual_text("pra", None) == "—"
    # Before tip: no actual column at all.
    assert "has-actual" not in props_ui.render_player_lines(store.line_actuals(recs, None))


def test_after_tip_the_tables_show_each_players_box_line(proj):
    act = store.actuals_from(_box())
    phi = store.team_view(_one(proj, H.EARLY, "PHI"))
    html = store.render_projection_table("PHI", phi, actuals=act)
    import re
    acts = re.findall(r'<tr class="act">(.*?)</tr>', html, re.S)
    text = [_visible(a) for a in acts]
    edge = [t for t in text if t.startswith("Final 29.2 10 5 0")]
    assert edge, text[:3]                                   # VJ Edgecombe: 29.2 min, 10 pts
    assert sum("Did not play" in t for t in text) == 1      # Drummond (Questionable) only
    # Listed-Out players get no extra row; their Out row already says it.
    assert len(acts) == int((~phi["out"]).sum())
    # Live, not final: a player not in yet gets no row at all.
    live_html = store.render_projection_table("PHI", phi, actuals=store.actuals_from(
        LV.Box(H.EARLY, False, _box().lines)))
    assert "Did not play" not in live_html and "So far" in live_html


def test_the_player_search_lists_the_slate_and_draws_his_game_row(monkeypatch, tmp_path, params):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    monkeypatch.setattr(store, "sim_params", lambda: params)
    opts = store.player_options(w.proj)
    assert len(opts) == w.proj["person_id"].nunique()
    assert [n for _, n in opts] == sorted(n for _, n in opts) or opts == sorted(
        opts, key=lambda o: (o[1].rsplit(", ", 1)[0], o[0]))
    jokic = _pid(w.proj, "Nikola Jokić")
    html = store.player_panel(w.proj, jokic, H.DATE, {"board": "tok123", "theme": "dark"})
    assert "Nikola Joki" in html and 'href="/Game?date=2026-03-12&amp;away=DEN&amp;home=SAS' in html
    assert "board=tok123" in html and "Open DEN at SAS" in html
    # His numbers are the Game page's: MIN the mean, the medians his own.
    med = simprov.medians(w.proj, params, store.MEDIAN_STATS).set_index(["person_id", "game_id"])
    row = w.proj.set_index("person_id").loc[jokic]
    from bucketlib.util import fmt_num
    assert f">{fmt_num(row['min'], 1)}</td>" in html
    assert f">{fmt_num(med.loc[(jokic, H.LATE), 'med_pts'], 1)}</td>" in html


def test_roster_statuses_badge_a_game_without_projections(monkeypatch, tmp_path):
    """DEN at SAS withheld for its line: the roster is badged from the live
    report through the projections' own context path."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    row = w.sched[w.sched["game_id"] == H.LATE].iloc[0]
    st_ = store.roster_statuses(row, w.sched, w.roster).set_index("person_id")
    assert st_.loc[_pid(w.proj, "Victor Wembanyama"), "status"] == "Questionable"
    ingram = st_.loc[_pid(w.proj, "Harrison Ingram")]
    assert bool(ingram["out"]) and ingram["status"] == "Two-way"
    den = st_[st_["team"] == "DEN"]
    assert (den["status"] == CTX.NOT_FILED).all() and len(den)
    # The report could not be read: no statuses (the page says so).
    def down(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(LV, "injury_report", down)
    assert store.roster_statuses(row, w.sched, w.roster) is None


def test_the_loader_reads_the_report_and_the_lines_at_once(monkeypatch, tmp_path):
    """A cold page waits for the slower of the two live reads, not their sum
    (spec 4: a ~5 s budget)."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    spans = {}

    def slow(name, value):
        def f(*a, **k):
            t0 = time.monotonic()
            time.sleep(0.4)
            spans[name] = (t0, time.monotonic())
            return LV.Live(value, 0.0)
        return f
    monkeypatch.setattr(LV, "injury_report", slow("report", w.report))
    monkeypatch.setattr(LV, "odds_on", slow("odds", w.odds))
    store.load_projections(H.DATE)
    (a0, a1), (b0, b1) = spans["report"], spans["odds"]
    assert a0 < b1 and b0 < a1, spans                       # the two waits overlap
