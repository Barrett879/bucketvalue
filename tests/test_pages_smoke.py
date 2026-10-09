"""Every page renders without an exception (Streamlit AppTest, offline), and
the projections reach the pages as the spec says.

The live layer is replaced with committed fixtures, so this runs with no
network. A syntax error, a bad import, or an exception on a normal path
fails here instead of in front of a visitor (with showErrorDetails="type"
the visitor would only see a generic error).

The projection tests serve tests/fixtures/serve (two real games of
2026-03-12, the real noon injury report, the production models; see its
make_fixture.py) through serve_harness.install, and compare what the pages
draw with what PRODUCTION code computes for the same inputs
(pipeline.live_projections, simprov.medians, props_ui.date_board)."""
import json
import re
import sys
from html import unescape
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from bucketlib import cache as C
from bucketlib import context as CTX
from bucketlib import dist as D
from bucketlib import live
from bucketlib import models as M
from bucketlib import pipeline as P
from bucketlib import props, props_ui, simprov, store
from bucketlib import state as S
from bucketlib.cache import MODELS_DIR
from bucketlib.sources import cdn
from bucketlib.util import SENTINEL, fmt_num

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "cdn"
sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402

DATE = H.DATE.isoformat()
REAL_STATE = S.STATE_DIR          # the checkout's own state/, before any test moves it


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    sched = cdn.parse_schedule(json.loads((FIX / "schedule_2026-27_trim.json").read_text()))
    roster = cdn.parse_player_index(json.loads((FIX / "playerIndex_trim.json").read_text()))
    monkeypatch.setattr(live, "schedule", lambda: live.Live(sched, 0.0))
    monkeypatch.setattr(live, "roster", lambda: live.Live(roster, 0.0))
    monkeypatch.setattr(live, "odds_on", lambda d: live.Live({("BOS", "DET"): {"home_line": -1.5, "total": 220.5, "provider": "DraftKings"}}, 0.0))
    # Never the repo's own state/: what the bot has committed must not change
    # what these tests see.
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "empty_state")
    store._live_frame.clear()


@pytest.fixture(scope="module")
def mdl():
    try:
        return M.load()
    except M.ModelMismatch as e:
        pytest.skip(f"production models not loadable here: {e}")


def run(path, **params):
    at = AppTest.from_file(str(REPO / path), default_timeout=90)
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def html(at) -> str:
    return " ".join(m.value for m in at.markdown)


def texts(at) -> list[str]:
    """Every string the page shows: markdown and messages by value, widget
    and expander labels by label."""
    out = []
    for kind in ("markdown", "caption", "success", "info", "warning", "error"):
        out += [str(e.value) for e in getattr(at, kind)]
    for kind in ("expander", "multiselect", "button", "toggle", "selectbox", "text_area"):
        out += [str(e.label) for e in getattr(at, kind)]
    return out


ROW = re.compile(r'<tr(?: class="([^"]*)")? data-pid="(\d+)">(.*?)</tr>', re.S)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def visible(markup: str) -> str:
    """The text a reader sees: every tag dropped WITH its attributes (a
    title= tooltip is not on screen), entities decoded."""
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def pid_of(name: str) -> int:
    proj = H.load().proj
    return int(proj.loc[proj["player_name"] == name, "person_id"].iloc[0])


def facts(at) -> dict[str, str]:
    """{label: visible value} of the Game header's facts."""
    return {lab: visible(val) for lab, val in re.findall(
        r'<div class="bv-fact-lab">(.*?)</div><div class="bv-fact-val">(.*?)</div>', html(at))}


def player_lines(at) -> dict[str, list[tuple[str, str]]]:
    """{player: [(model, line)]} from the Game page's per-player expanders."""
    out = {}
    for e in at.expander:
        if ": " in e.label and e.label.split(": ")[1].split(" ")[0].isdigit():
            body = " ".join(m.value for m in e.markdown)
            out[e.label.split(": ")[0]] = re.findall(
                r'<span class="num">([^<]*)</span><span class="num line">([^<]*)</span>', body)
    return out


LEDGER = re.compile(r'<div class="bv-ledger-row"><span class="nm">([^<]*)<i>.*?'
                    r'<span class="num">([^<]*)</span><span class="num line">([^<]*)</span>', re.S)


def table_rows(at) -> dict[int, dict]:
    """{person_id: {out, cells}} from the projection tables on the page.
    Cells after badge and player: P(play), MIN and the nine medians; an Out
    row has one cell, its reason."""
    rows = {}
    for cls, pid, body in ROW.findall(html(at)):
        cells = CELL.findall(body)
        rows[int(pid)] = {"out": "out" in (cls or "").split(), "cells": cells[2:]}
    return rows


def expected(mdl, at_clock, marks=()):
    """The frame and medians production computes for the page's inputs."""
    w = H.load()
    frame, info = P.live_projections(H.DATE, at_clock, marked_out=set(marks), report=w.report,
                                     odds=w.odds, models=mdl, schedule=w.sched)
    med = simprov.medians(frame, D.SimParams.load(MODELS_DIR / store.SIM_PARAMS_FILE),
                          store.MEDIAN_STATS)
    return frame, info, med


def game(**params):
    return run("pages/Game.py", date=DATE, **params)


# ── The pages render ─────────────────────────────────────────────────────────
def test_home_opening_night():
    at = run("app.py", date="2026-10-20")
    page = html(at)
    assert "BOS" in page and "DET" in page and "DET -1.5" in page
    # A regular-season date the bot has not published says so.
    assert "Projections not published yet for this date" in page


def test_home_empty_date_and_preseason():
    run("app.py", date="2026-08-01")
    run("app.py", date="2026-10-09")
    at = run("app.py", date="2026-10-03")         # the fixture's preseason game
    assert "Preseason: 1 game, no projections" in html(at)


def test_game_page_and_bad_params():
    run("pages/Game.py", date="2026-10-20", away="BOS", home="DET")
    at = run("pages/Game.py", date="2026-10-20", away="<script>", home="DET")
    assert "<script>" not in " ".join(m.value for m in at.markdown)


def test_about():
    run("pages/About.py")


def test_the_committed_state_renders(monkeypatch):
    """The workflow runs this file right after the bot writes state/: the
    slate and every game page of the newest committed dates must render from
    the files as written. The live report and lines are forced down, so the
    pages serve the committed projections and nothing touches the network."""
    dates = sorted(p.parent.name for p in (REAL_STATE / "serve").glob("*/projections_v1.parquet"))
    if not dates:
        pytest.skip("the bot has committed no projections yet")
    monkeypatch.setattr(S, "STATE_DIR", REAL_STATE)
    sched = C.read_parquet(S.current_path("schedule"))
    roster = C.read_parquet(S.current_path("roster"))
    assert sched is not None and roster is not None, "projections committed without schedule/roster"

    def down(*a, **k):
        raise RuntimeError("offline in the smoke test")
    monkeypatch.setattr(live, "schedule", lambda: live.Live(sched, 0.0))
    monkeypatch.setattr(live, "roster", lambda: live.Live(roster, 0.0))
    monkeypatch.setattr(live, "injury_report", down)
    monkeypatch.setattr(live, "odds_on", lambda d: live.Live({}, None, stale=True, source="none",
                                                             error="offline"))
    for d in dates[-2:]:
        run("app.py", date=d)
        proj = C.read_parquet(REAL_STATE / "serve" / d / "projections_v1.parquet")
        for gid in sorted(set(proj["game_id"])):
            g = sched[sched["game_id"] == gid].iloc[0]
            at = run("pages/Game.py", date=d, away=g["away"], home=g["home"])
            assert 'data-pid="' in html(at), (d, gid)


# ── /Game: the tables ────────────────────────────────────────────────────────
def test_game_page_renders_the_projected_numbers(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    at = game(away="PHI", home="DET")
    frame, info, med = expected(mdl, H.MORNING)
    assert info.source == "live"
    rows = table_rows(at)
    mine = frame[frame["game_id"] == H.EARLY].set_index("person_id")
    m = med.set_index(["person_id", "game_id"])
    assert set(rows) == set(mine.index)
    played = [p for p in mine.index if mine.loc[p, "inj_status"] != "Out"]
    assert len(played) > 20
    for pid in played:
        cells = rows[pid]["cells"]
        assert not rows[pid]["out"] and len(cells) == 1 + len(store.STAT_COLS)
        assert cells[1] == fmt_num(mine.loc[pid, "min"], 1), pid           # MIN is the mean
        for i, s in enumerate(store.MEDIAN_STATS, start=2):
            assert cells[i] == fmt_num(m.loc[(pid, H.EARLY), f"med_{s}"], 1), (pid, s)
    assert sum(c != SENTINEL for p in played for c in rows[p]["cells"][1:]) > 20 * 10
    page = html(at)
    assert "Median if he plays" in page and "Injury report" in page and "12:00 PM ET" in page


def test_pplay_only_for_listed_players_or_below_ninety(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    frame, _, _ = expected(mdl, H.MORNING)
    seen = {"listed": 0, "low": 0, "blank": 0}
    for away, home, gid in (("PHI", "DET", H.EARLY), ("DEN", "SAS", H.LATE)):
        rows = table_rows(game(away=away, home=home))
        f = frame[frame["game_id"] == gid].set_index("person_id")
        for pid, r in rows.items():
            if r["out"]:
                continue
            listed = pd.notna(f.loc[pid, "inj_status"])
            low = f.loc[pid, "p_play"] < 0.90
            assert (r["cells"][0] != "") == (listed or low), (pid, r["cells"][0])
            seen["listed" if listed else "low" if low else "blank"] += 1
    # The fixture exercises all three: Drummond (Questionable), Lowry (54%,
    # not listed) and a healthy starter.
    assert all(n > 0 for n in seen.values()), seen


def test_an_out_player_has_no_numbers_and_shows_his_reason(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    rows = table_rows(game(away="PHI", home="DET"))
    frame, _, _ = expected(mdl, H.MORNING)
    out = frame[(frame["game_id"] == H.EARLY) & (frame["inj_status"] == "Out")]
    assert len(out) >= 8
    for _, r in out.iterrows():
        got = rows[int(r["person_id"])]
        assert got["out"] and len(got["cells"]) == 1, r["player_name"]
        assert not re.search(r"\d", unescape(got["cells"][0])), got["cells"]  # no numbers
    maxey = rows[int(out.loc[out["player_name"] == "Tyrese Maxey", "person_id"].iloc[0])]
    assert "Right Finger" in maxey["cells"][0]
    # In each team's table every Out row sits below every row with numbers.
    order = [int(p) for _, p, _ in ROW.findall(html(game(away="PHI", home="DET")))]
    team = dict(zip(frame["person_id"].astype(int), frame["team"]))
    for t in ("PHI", "DET"):
        mine = [p for p in order if team[p] == t]
        flags = [rows[p]["out"] for p in mine]
        assert any(flags) and flags == sorted(flags), t


def test_unknown_report_state_is_never_drawn_healthy(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    at = game(away="DEN", home="SAS")
    page = html(at)
    den = H.load().proj
    den = den[(den["game_id"] == H.LATE) & (den["team"] == "DEN")]
    assert CTX.NOT_FILED in page
    for pid in den["person_id"]:
        assert re.search(rf'data-pid="{int(pid)}"><td class="l"><span class="bv-st unk"', page), pid
    # What the reader SEES, not what an attribute says: the caption chip's
    # text reads "Report not filed", and every Denver badge shows a label.
    cap = re.search(r'<div class="bv-pt-cap"><span class="ab">DEN</span>(.*?)</div>', page, re.S)
    assert cap and CTX.NOT_FILED in visible(cap.group(1)), cap
    rows = dict((int(p), b) for _, p, b in ROW.findall(page))
    for pid in den["person_id"]:
        badge = visible(CELL.findall(rows[int(pid)])[0])
        assert badge and badge not in ("Avail", "Prob"), (pid, badge)
    sas = visible(re.search(r'<div class="bv-pt-cap"><span class="ab">SAS</span>(.*?)</div>',
                            page, re.S).group(1))
    assert CTX.NOT_FILED not in sas and CTX.NO_REPORT not in sas


# ── /Game: mark-outs ─────────────────────────────────────────────────────────
def test_out_param_adds_the_banner_and_moves_a_teammates_minutes(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    plain = table_rows(game(away="PHI", home="DET"))
    proj = H.load().proj
    cade = int(proj.loc[proj["player_name"] == "Cade Cunningham", "person_id"].iloc[0])
    at = game(away="PHI", home="DET", out=str(cade))
    page = html(at)
    assert "Your adjustments, not the published projection" in page
    marked = table_rows(at)
    assert marked[cade]["out"] and CTX.MARKED_REASON in marked[cade]["cells"][0]
    # Every number of every played row is the MARKED frame's: MIN its mean,
    # the nine medians its simulation (not the unmarked frame's medians next
    # to recomputed minutes).
    frame, info, med = expected(mdl, H.MORNING, marks=[cade])
    assert info.adjusted
    mine = frame[frame["game_id"] == H.EARLY].set_index("person_id")
    m = med.set_index(["person_id", "game_id"])
    assert set(marked) == set(mine.index)
    for pid, r in marked.items():
        if mine.loc[pid, "inj_status"] == "Out":
            assert r["out"] and len(r["cells"]) == 1, pid
            continue
        assert r["cells"][1] == fmt_num(mine.loc[pid, "min"], 1), pid
        for i, s in enumerate(store.MEDIAN_STATS, start=2):
            assert r["cells"][i] == fmt_num(m.loc[(pid, H.EARLY), f"med_{s}"], 1), (pid, s)
    det = proj[(proj["game_id"] == H.EARLY) & (proj["team"] == "DET") & (proj["inj_status"] != "Out")]
    mates = [int(p) for p in det["person_id"] if int(p) != cade]
    gain = sum(float(marked[p]["cells"][1]) - float(plain[p]["cells"][1]) for p in mates)
    assert gain > 3.0
    # The other team is untouched.
    phi = proj[(proj["game_id"] == H.EARLY) & (proj["team"] == "PHI")]
    assert all(marked[int(p)] == plain[int(p)] for p in phi["person_id"])
    # The picker was seeded from the URL; removing the name clears ?out=.
    ms = at.multiselect(key="bv_out_DET")
    assert ms.value == [cade]
    ms.unselect(cade).run()
    assert not at.exception and "out" not in at.query_params
    assert "Your adjustments" not in html(at)
    assert table_rows(at) == plain


def test_marking_out_through_the_picker_writes_the_url(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    at = game(away="PHI", home="DET")
    opts = at.multiselect(key="bv_out_PHI").options
    proj = H.load().proj
    edge = int(proj.loc[proj["player_name"] == "VJ Edgecombe", "person_id"].iloc[0])
    at.multiselect(key="bv_out_PHI").select(edge).run()
    assert at.query_params["out"] == [str(edge)] or at.query_params["out"] == str(edge)
    assert "Your adjustments, not the published projection" in html(at)
    # Stable options: the mark moved minutes, the option list did not move.
    assert at.multiselect(key="bv_out_PHI").options == opts


def test_a_frozen_game_has_no_mark_out_control_and_the_committed_numbers(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.EVENING)
    proj = H.load().proj
    at = game(away="PHI", home="DET", out=str(int(proj["person_id"].iloc[0])))
    assert len(at.multiselect) == 0
    page = html(at)
    assert "Frozen at tip" in page and "Your adjustments" not in page
    rows = table_rows(at)
    early = proj[proj["game_id"] == H.EARLY].set_index("person_id")
    for pid, r in rows.items():
        if not r["out"]:
            assert r["cells"][1] == fmt_num(early.loc[pid, "min"], 1)
    # The untipped game of the same evening keeps its control.
    assert len(game(away="DEN", home="SAS").multiselect) == 2


def test_a_mark_on_a_listed_out_player_changes_nothing(monkeypatch, tmp_path, mdl):
    """The picker offers every player (its options never change), but a
    player the report already lists Out moves no number when marked: no
    banner, his own reason kept, his place on the card's Out line kept."""
    H.install(monkeypatch, tmp_path, H.MORNING)
    plain = game(away="PHI", home="DET")
    maxey = pid_of("Tyrese Maxey")
    at = game(away="PHI", home="DET", out=str(maxey))
    assert "Your adjustments" not in html(at)
    assert table_rows(at) == table_rows(plain)
    assert "Right Finger" in table_rows(at)[maxey]["cells"][0]
    assert any("Already listed Out" in c.value and "Tyrese Maxey" in c.value for c in at.caption)
    assert at.multiselect(key="bv_out_PHI").value == [maxey]
    slate = run("app.py", date=DATE, out=str(maxey))
    assert "<b>Out:</b> Maxey, Oubre Jr. +5" in cards(slate)["PHI at DET"]
    assert "Your adjustments" not in html(slate)


def test_the_injury_report_fact_claims_only_what_was_read(monkeypatch, tmp_path, mdl):
    # Nothing published for the date: no report was read, so no fact at all.
    at = run("pages/Game.py", date="2026-10-20", away="BOS", home="DET")
    assert "Injury report" not in facts(at) and "Not filed yet" not in html(at)
    # A withheld game (no line, nothing committed for it): no rows, but the
    # live layer did read a report, and the fact names it.
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    C.write_parquet(w.proj[w.proj["game_id"] != H.LATE], P.projections_path(H.DATE))
    lines = {k: v for k, v in w.odds.items() if k != ("DEN", "SAS")}
    monkeypatch.setattr(live, "odds_on", lambda d: live.Live(lines if d == H.DATE else {}, 0.0))
    store._live_frame.clear()
    at = game(away="DEN", home="SAS")
    assert P.NO_LINE in html(at) and 'data-pid="' not in html(at)
    assert facts(at)["Injury report"] == "12:00 PM ET"
    # A projected game keeps its report time (the rows' own).
    assert facts(game(away="PHI", home="DET"))["Injury report"] == "12:00 PM ET"


# ── / and the board ──────────────────────────────────────────────────────────
CARD = re.compile(r'<a class="bv-slate-card" href="([^"]*)".*?</a>', re.S)
PILL = re.compile(r'<span class="bv-slate-lines"><b>(\d+)</b>')


def cards(at) -> dict[str, str]:
    """{'PHI at DET': card html}."""
    out = {}
    for m in CARD.finditer(html(at)):
        q = dict(re.findall(r"(away|home)=([A-Z]{3})", m.group(1)))
        out[f"{q['away']} at {q['home']}"] = m.group(0)
    return out


def test_slate_cards_show_key_absences(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    c = cards(run("app.py", date=DATE))
    assert "<b>Out:</b> Maxey, Oubre Jr. +5" in c["PHI at DET"]
    assert "Out:" not in c["DEN at SAS"]              # only two-way players under 15 minutes


def test_board_renders_and_the_card_pill_equals_the_game_page_count(monkeypatch, tmp_path, mdl):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = H.paste(w)
    slate = run("app.py", date=DATE, board=token)
    assert "Model vs the board" in html(slate)
    c = cards(slate)
    for (away, home), gid in ((("PHI", "DET"), H.EARLY), (("DEN", "SAS"), H.LATE)):
        pill = PILL.search(c[f"{away} at {home}"])
        page = game(away=away, home=home, board=token)
        n_rows = html(page).count('class="bv-pl-row"')
        assert pill is not None and int(pill.group(1)) == n_rows > 0, (away, home)
        assert f"{n_rows} of " in html(page)
    # Tyrese Maxey is listed Out: his line is void, never a row.
    at = game(away="PHI", home="DET", board=token)
    assert "1 void" in html(at)
    labels = [e.label for e in at.expander]
    assert any(lab.startswith("Cade Cunningham") for lab in labels)
    assert not any(lab.startswith("Tyrese Maxey") for lab in labels)


def test_the_board_under_out_reads_the_marked_frame(monkeypatch, tmp_path, mdl):
    """With Cunningham marked out, Daniss Jenkins's points median crosses his
    7.5 line (hidden as Even without the mark): the line is shown, and its
    model value is the MARKED frame's median, on the Game page (where it is
    also his PTS cell) and in the slate's board."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = H.paste(w)
    cade, jenkins = pid_of("Cade Cunningham"), pid_of("Daniss Jenkins")
    _, _, med = expected(mdl, H.MORNING, marks=[cade])
    _, _, med0 = expected(mdl, H.MORNING)
    want = med.set_index(["person_id", "game_id"]).loc[(jenkins, H.EARLY), "med_pts"]
    before = med0.set_index(["person_id", "game_id"]).loc[(jenkins, H.EARLY), "med_pts"]
    assert round(want, 1) != round(before, 1)            # the fixture exercises the move
    model = visible(props_ui._fmt_model({"model": want, "line": 7.5}))
    page = game(away="PHI", home="DET", board=token, out=str(cade))
    lines = player_lines(page)
    assert lines.get("Daniss Jenkins") == [(model, "7.5")], lines
    assert table_rows(page)[jenkins]["cells"][2] == fmt_num(want, 1)
    assert "Cade Cunningham" not in lines                 # marked out: void
    led = {n: (m, ln) for n, m, ln in LEDGER.findall(html(run("app.py", date=DATE, board=token,
                                                                 out=str(cade))))}
    assert led.get("Daniss Jenkins") == (model, "7.5"), led
    assert "Cade Cunningham" not in led
    plain = {n for n, _, _ in LEDGER.findall(html(run("app.py", date=DATE, board=token)))}
    assert "Daniss Jenkins" not in plain and "Cade Cunningham" in plain


def test_marks_the_frame_did_not_apply_never_void_a_line(monkeypatch, tmp_path, mdl):
    """A frozen game, and the committed stand-in, ignore ?out=: the page
    shows the player with numbers, so his lines are not void and the card
    pill and the line count do not move."""
    w = H.install(monkeypatch, tmp_path, H.EVENING)
    token = H.paste(w)
    cade = pid_of("Cade Cunningham")

    def view(**out):
        page = game(away="PHI", home="DET", board=token, **out)
        pill = PILL.search(cards(run("app.py", date=DATE, board=token, **out))["PHI at DET"])
        count = re.search(r"\d+ of \d+ lines shown \([^)]*\)", html(page)).group(0)
        return (count, pill.group(1) if pill else None, sorted(player_lines(page)),
                table_rows(page)[cade], "Your adjustments" in html(page))
    frozen = view()
    assert "Cade Cunningham" in frozen[2] and not frozen[3]["out"] and not frozen[4]
    assert view(out=str(cade)) == frozen
    # The report cannot be read: the committed projections stand in, and the
    # marks are not applied there either.
    def down(*a, **k):
        raise RuntimeError("the report is down")
    monkeypatch.setattr(live, "injury_report", down)
    H.pin(monkeypatch, H.MORNING)
    store._live_frame.clear()
    stand_in = view()
    assert "Cade Cunningham" in stand_in[2]
    assert view(out=str(cade)) == stand_in
    assert P.MARKS_NOT_APPLIED in html(game(away="PHI", home="DET", out=str(cade)))


def test_slate_banner_reports_tile_and_a_date_step_drops_the_marks(monkeypatch, tmp_path, mdl):
    H.install(monkeypatch, tmp_path, H.MORNING)
    cade = str(pid_of("Cade Cunningham"))
    plain = run("app.py", date=DATE)
    assert "Your adjustments" not in html(plain)
    # Denver had not filed at noon: three of the four teams.
    assert re.search(r'<div class="bv-sum-num">3/4</div><div class="bv-sum-lab">Injury reports '
                     r"filed</div>", html(plain))
    at = run("app.py", date=DATE, out=cade)
    assert "Your adjustments, not the published projection." in html(at)
    assert "1 player marked out" in html(at)
    assert f"out={cade}" in cards(at)["PHI at DET"]                    # the cards carry them
    at.button(key="bv_day_next").click().run()
    assert not at.exception and "out" not in at.query_params
    assert at.query_params["date"] in ("2026-03-13", ["2026-03-13"])


# ── /About ───────────────────────────────────────────────────────────────────
def test_about_renders_the_trust_table_from_the_reliability_file():
    page = html(run("pages/About.py"))
    for s in simprov.GATED:
        rel = props.reliability_for(s)
        row = re.search(rf"<tr><td class=\"l pl\">{re.escape(props.STAT_LABEL[s])}</td>(.*?)</tr>", page)
        assert row, s
        assert f">{props.TIER_LABEL[rel['tier']]}</span>" in row.group(1), s
        assert f"<td>{rel['edge']:+.1f}</td>" in row.group(1), s
    assert "57.7%" in page and "PROXY" in page and "weakest component" in page
    assert "The board is at its ceiling." in page
    assert "already prices most of what a model knows" in page
    # The freeze point is the pipeline's, and a provisional projection never freezes.
    mins = int(P.FREEZE_LEAD.total_seconds() // 60)
    text = visible(page)
    assert f"freeze {mins} minutes before tip-off" in text and "never frozen" in text
    assert "freeze at tip-off" not in text


def test_about_nav_carries_the_board_the_marks_and_the_date(monkeypatch, tmp_path):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = H.paste(w)
    at = run("pages/About.py", board=token, out="1630595,junk", date=DATE)
    nav = re.findall(r'href="([^"]*)"', re.search(r'<div class="top-nav">(.*?)</div>',
                                                  html(at), re.S).group(1))
    assert len(nav) >= 2
    for href in nav:
        assert f"board={token}" in href and "out=1630595" in href and f"date={DATE}" in href, href
    # Junk is not carried, and nothing from the URL reaches an href unchecked.
    bad = run("pages/About.py", board="nope", out="abc", date='"><script>x</script>')
    nav = re.findall(r'href="([^"]*)"', re.search(r'<div class="top-nav">(.*?)</div>',
                                                  html(bad), re.S).group(1))
    assert nav and not any(k in h for h in nav for k in ("board=", "out=", "date=", "script"))


# ── House style ──────────────────────────────────────────────────────────────
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def _dash_ok(text: str) -> bool:
    """Every em dash is the sentinel: a whole text node, never inside words,
    sentences or attributes."""
    nodes = re.split(r"<[^>]*>", text)
    return (all(n.strip() == SENTINEL for n in nodes if SENTINEL in n)
            and text.count(SENTINEL) == sum(n.count(SENTINEL) for n in nodes))


def test_no_em_dash_outside_the_sentinel_and_no_emoji(monkeypatch, tmp_path, mdl):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = H.paste(w)
    proj = H.load().proj
    cade = str(int(proj.loc[proj["player_name"] == "Cade Cunningham", "person_id"].iloc[0]))
    pages = [run("app.py", date=DATE, board=token, out=cade),
             game(away="PHI", home="DET", board=token, out=cade),
             game(away="DEN", home="SAS", board=token),
             run("pages/About.py")]
    H.pin(monkeypatch, H.EVENING)
    store._live_frame.clear()
    pages.append(game(away="PHI", home="DET"))
    for at in pages:
        for t in texts(at):
            assert _dash_ok(t), [n for n in re.split(r"<[^>]*>", t) if SENTINEL in n][:3]
            assert not EMOJI.search(t), EMOJI.findall(t)[:3]
