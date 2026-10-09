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
