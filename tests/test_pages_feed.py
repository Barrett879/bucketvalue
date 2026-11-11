"""The pages with the licensed odds feed (design 2.8, 8.2 and 14 step 17;
layout B, amendments A3 to A9; build amendments N1 to N3), through Streamlit's
AppTest, offline.

The world is the serve fixture's 2026-03-12 (tests/fixtures/serve, a fixed past
date) through serve_harness.install, with every clock the pages read pinned.
serve_harness.install_feed turns the feed on: a fake key built at run time
("K" * 40), and lines bought by the REAL live walk through the REAL oddsapi
parser from the CONSTRUCTED fixtures in tests/fixtures/oddsapi (PROVENANCE.txt:
invented lines on that slate's real players; PrizePicks has posted PHI at DET
only), served from live.feed_view. The pages then run the REAL
store.feed_start, feed_result, feed_tables and feed_info, props.feed_board and
props_ui's renderers. The fake vendor is reachable only while the harness
buys; conftest's block stands for the pages.

What a page draws is compared with what production computes for the same
inputs (props_ui.date_board on store.feed_tables of the served feed), as
tests/test_pages_smoke.py does for the paste, and with the fixtures' own
lines, read from tests/fixtures/oddsapi by hand: on points (the shipped
market), PHI at DET's Cade Cunningham is 25.5 at all five books, Jalen Duren
13.5 everywhere but Underdog, which has no line for him, and Tobias Harris
12.5 at DraftKings alone.

With no key the pages must be exactly today's: the real feed_start gives the
same page, element for element, as a feed stubbed off.
"""
from __future__ import annotations

import re
import sys
import threading
from html import unescape
from pathlib import Path

import pytest

from bucketlib import http, live, props, props_ui, store
from bucketlib.util import SENTINEL, UTC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import test_live_feed as LF  # noqa: E402  (the fake key, the fake vendor, the event ids)
import test_pages_smoke as PS  # noqa: E402  (the page runner and the house-style checks)
import test_props_feed as PF  # noqa: E402  (the constructed payloads, and edited copies)

DATE = H.DATE.isoformat()
KEY = LF.KEY                                  # "K" * 40, built at run time
EARLY, LATE = H.EARLY, H.LATE                 # PHI at DET 7:00 PM ET, DEN at SAS 9:00 PM ET
GAMES = ((("PHI", "DET"), EARLY), (("DEN", "SAS"), LATE))
et = LF.et                                    # an ET instant on 2026-03-12


# -- The world -------------------------------------------------------------------------------
def feed_world(monkeypatch, tmp_path, at=H.MORNING, **feed):
    """The serve fixture's world with the odds feed on (the page clock at
    `at`; install_feed's own defaults: one pull of both games at 12:20 PM)."""
    w = H.install(monkeypatch, tmp_path, at)
    return w, H.install_feed(monkeypatch, **feed)


def paste(monkeypatch, w, at=H.MORNING) -> str:
    """The serve fixture's board text through the real paste path, pasted at
    `at`: the board's paste time is pinned too (props._wall), so no time on
    any page comes from the real clock."""
    monkeypatch.setattr(props, "_wall", lambda: at.astimezone(UTC))
    return H.paste(w, at)


def empty(payload: dict) -> dict:
    """An event-odds answer with no book in it (nothing posted yet: free)."""
    return {**payload, "bookmakers": []}


def slate(**q):
    return PS.run("app.py", date=DATE, **q)


def game(away, home, **q):
    return PS.run("pages/Game.py", date=DATE, away=away, home=home, **q)


def html(at) -> str:
    return PS.html(at)


def text(at) -> str:
    """What a reader sees: every block and message joined, the style blocks
    left out."""
    return " ".join(PS.visible(t) for t in PS.texts(at) if not t.lstrip().startswith("<style>"))


def pills(at):
    """The book pills (amendment A3), or None when the page has none."""
    found = [b for b in at.button_group if b.key == props_ui.BOOK_KEY]
    return found[0] if found else None


def picked(at) -> str:
    return pills(at).value


def dump(at) -> str:
    """Every element of the page, in order, as its proto: what a browser
    would be sent. The keeper's per-session request id is masked."""
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            out.append(f"{type(node).__name__}: {proto}")
        kids = getattr(node, "children", None)
        if isinstance(kids, dict):
            for k in sorted(kids):
                walk(kids[k])
    walk(at._tree)
    page = "\n".join(out)
    try:
        nonce = at.session_state[props_ui.KEEP_NONCE]
    except (KeyError, AttributeError):
        nonce = None
    return page.replace(nonce, "<nonce>") if isinstance(nonce, str) and nonce else page


def order(at) -> list[tuple[str, str]]:
    """The page's elements in render order: ("md", markdown) or ("pills", key),
    the style blocks left out."""
    out = []

    def walk(node):
        kind = type(node).__name__
        if kind == "Markdown":
            if not str(node.value).lstrip().startswith("<style>"):
                out.append(("md", str(node.value)))
        elif kind == "ButtonGroup":
            out.append(("pills", node.key))
        kids = getattr(node, "children", None)
        if isinstance(kids, dict):
            for k in sorted(kids):
                walk(kids[k])
    walk(at._tree)
    return out


WINDOW = '<div class="bv-window"><span class="lab">Model vs the board</span>'
LEDGER_ROW = re.compile(r'<div class="bv-ledger-row"><span class="nm">([^<]*)<i>.*?'
                        r'<span class="num line">([^<]*)</span>.*?'
                        r'<span class="lean ?([^"]*)">([^<]*)</span>'
                        r'(?:<span class="ob">(.*?)</span>)?</div>', re.S)
LINE_ROW = re.compile(r'<div class="bv-pl-row">.*?<span class="num line">([^<]*)</span>.*?'
                      r'<span class="lean ?([^"]*)">([^<]*)</span>.*?'
                      r'(?:<span class="ob">(.*?)</span>)?</div>', re.S)
LEDGER_HEAD = re.compile(r'<div class="bv-ledger-head">.*?<span class="num">Model</span>'
                         r'<span class="num">([^<]*)</span>', re.S)
LINES_HEAD = re.compile(r'<div class="bv-pl-head">.*?<span class="num">Model</span>'
                        r'<span class="num">([^<]*)</span>', re.S)


def ledger(at) -> dict[str, dict]:
    """{player: {line, cls, lean, ob}} from the slate's board ledger."""
    return {unescape(n): {"line": ln, "cls": cls, "lean": lean, "ob": ob}
            for n, ln, cls, lean, ob in LEDGER_ROW.findall(html(at))}


def ledger_head(at) -> str:
    m = LEDGER_HEAD.search(html(at))
    return m.group(1) if m else ""


def player_lines(at) -> dict[str, list[dict]]:
    """{player: [{line, cls, lean, ob}]} from the Game page's expanders."""
    out = {}
    for e in at.expander:
        m = re.match(r"(.+): (\d+) lines?$", str(e.label))
        if m:
            body = " ".join(str(x.value) for x in e.markdown)
            out[m.group(1).replace("\\", "")] = [
                {"line": ln, "cls": cls, "lean": lean, "ob": ob}
                for ln, cls, lean, ob in LINE_ROW.findall(body)]
    return out


def pill_counts(at) -> dict[str, int]:
    """{game_id: the card pill's count} (0 for a card without one)."""
    cards = PS.cards(at)
    out = {}
    for (away, home), gid in GAMES:
        m = PS.PILL.search(cards[f"{away} at {home}"])
        out[gid] = int(m.group(1)) if m else 0
    return out


def page_rows(at) -> int:
    return html(at).count('class="bv-pl-row"')


def production(book: str, token=None):
    """What production's date_board shows for the page's own inputs: the
    feed the pages are served, the date's served frame, the simulator
    parameters and the roster, compared by store.feed_tables."""
    fl = live.feed_view(H.DATE)
    served = store.load_projections(H.DATE)
    params = store.sim_params()
    roster = H.load().roster
    ft = store.feed_tables(H.DATE, fl, served, params, roster)
    return props_ui.date_board(token, H.DATE, served.frame, params, roster, feed=ft,
                               feed_info=store.feed_info(fl, H.DATE), selected=book)


def hrefs(at) -> list[str]:
    return [unescape(h) for h in PS.HREF.findall(html(at))]


# -- With no key: today's pages ---------------------------------------------------------------
def _forced_off(monkeypatch):
    """The feed stubbed off by name: what every page did before the feed."""
    monkeypatch.setattr(store, "feed_on", lambda environ=None: False)
    monkeypatch.setattr(store, "feed_start",
                        lambda view_date, sched: store.FeedTicket(view_date, "off"))


def test_with_no_key_the_real_feed_start_gives_the_same_pages_as_the_feed_off_m9(
        monkeypatch, tmp_path):
    """Design 8.2: no key means the real feed_start is "off" before any feed
    code runs, so the slate, both Game pages and About are, element for
    element, what a feed stubbed off gives, a ?book= included, and nothing
    reaches the vendor (here a fake that records any call)."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = paste(monkeypatch, w)
    vendor = LF.Vendor()
    monkeypatch.setattr(http, "vendor_get", vendor)
    views = [("app.py", {}), ("app.py", {"board": token}), ("app.py", {"book": "draftkings"}),
             ("pages/Game.py", {"away": "PHI", "home": "DET"}),
             ("pages/Game.py", {"away": "PHI", "home": "DET", "board": token}),
             ("pages/Game.py", {"away": "DEN", "home": "SAS", "board": token,
                                "book": "fanduel"}),
             ("pages/About.py", {"board": token, "book": "betmgm"})]

    def pages():
        return [dump(PS.run(path, date=DATE, **q)) for path, q in views]
    real = pages()
    with monkeypatch.context() as m:
        _forced_off(m)
        off = pages()
    assert real == off
    assert vendor.calls == [] and live.ANCHOR_NAME not in sys.modules
    # A ?book= is never read with the feed off: the same page as without it.
    assert real[2] == real[0]
    for page in real:
        assert "book=" not in page and "bv_book" not in page and "Other books" not in page


def test_with_no_key_a_failing_work_order_still_shows_the_no_lines_prompts_m9(
        monkeypatch, tmp_path):
    """M9: every piece of the feed raises, and with no key none of it runs:
    the pages keep today's prompts (tests/test_pages_smoke.py:519-520), and
    no feed phrase ("down" least of all) appears."""
    H.install(monkeypatch, tmp_path, H.MORNING)

    def boom(*a, **k):
        raise AssertionError("feed code ran with no key")
    for name in ("feed_work", "_work_order", "feed_tables", "feed_info", "feed_loading_watch"):
        monkeypatch.setattr(store, name, boom)
    for name in ("feed_kick", "feed_view", "prop_lines_memory", "feed_anchor", "ensure_refresher"):
        monkeypatch.setattr(live, name, boom)
    page = html(slate())
    assert props_ui.NO_LINES_SLATE in page and props_ui.NO_LINES_SLATE_FEED not in page
    den = html(game("DEN", "SAS"))
    assert props_ui.NO_LINES_GAME in den and props_ui.NO_LINES_GAME_FEED not in den
    for markup in (page, den):
        assert props_ui.FEED_DOWN not in markup and "odds feed" not in markup


# -- The pills (amendment A3) ------------------------------------------------------------------
def test_the_pills_show_only_with_the_feed_on_and_lines_on_the_date(monkeypatch, tmp_path):
    H.install(monkeypatch, tmp_path, H.MORNING)
    for at in (slate(), slate(book="draftkings"), game("PHI", "DET", book="draftkings"),
               game("DEN", "SAS")):
        assert pills(at) is None                                  # the feed off: none, ever
    H.install_feed(monkeypatch)
    at = slate()
    bg = pills(at)
    assert bg is not None and bg.value == "prizepicks"
    assert bg.options == ["PrizePicks", "Underdog", "DraftKings", "FanDuel", "BetMGM"]
    assert list(bg.proto.default) == [] and not at.warning       # seeded, never a default=
    for (away, home), _gid in GAMES:
        assert picked(game(away, home)) == "prizepicks"


def test_on_the_slate_the_pills_sit_after_the_feed_notice_right_above_the_board(
        monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    seq = order(slate())
    at_pills = seq.index(("pills", props_ui.BOOK_KEY))
    assert WINDOW in seq[at_pills + 1][1]                           # directly above the window
    notice = next(i for i, (k, v) in enumerate(seq) if k == "md" and "Lines as of 12:20 PM ET" in v)
    assert notice < at_pills
    # The cards come before the pills, and the board's own filter after them.
    last_card = max(i for i, (k, v) in enumerate(seq) if k == "md" and "bv-slate-card" in v)
    assert last_card < notice and all(k != "pills" for k, _v in seq[:at_pills])


def test_the_pages_draw_the_pills_outside_every_fragment():
    """Amendment A3: a pick is a full rerun, so the cards follow it. Each page
    calls render_book_pills at its own top level, never inside a function (a
    fragment is a decorated function), and render_book_pills is no fragment."""
    import ast
    for rel in ("app.py", "pages/Game.py"):
        tree = ast.parse((PS.REPO / rel).read_text())
        calls = []

        def visit(node, inside):
            for child in ast.iter_child_nodes(node):
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "render_book_pills"):
                    calls.append(inside)
                visit(child, inside or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                          ast.Lambda)))
        visit(tree, False)
        assert calls == [False], (rel, calls)
    src = ast.parse((PS.REPO / "bucketlib" / "props_ui.py").read_text())
    (fn,) = [n for n in ast.walk(src) if isinstance(n, ast.FunctionDef) and n.name == "render_book_pills"]
    assert fn.decorator_list == []


def test_on_the_game_page_the_pills_sit_right_above_the_lines_note(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    for (away, home), _gid in GAMES:
        seq = order(game(away, home))
        at_pills = seq.index(("pills", props_ui.BOOK_KEY))
        nxt = PS.visible(seq[at_pills + 1][1])
        assert nxt.startswith("Lines on this game:" if away == "PHI"
                              else "No PrizePicks lines for this game yet."), nxt


def test_no_lines_for_the_date_means_no_pills_and_the_feeds_own_prompt(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, payloads={LF.PHI_EID: empty(PF.phi_det()),
                                               LF.DEN_EID: empty(PF.den_sas())})
    at = slate()
    assert pills(at) is None and WINDOW not in html(at)
    assert props_ui.NO_LINES_SLATE_FEED in html(at) and props_ui.NO_LINES_SLATE not in html(at)
    den = game("DEN", "SAS")
    assert pills(den) is None and props_ui.NO_LINES_GAME_FEED in html(den)


def test_a_game_with_no_lines_has_no_pills_while_the_slate_has_them(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, payloads={LF.PHI_EID: empty(PF.phi_det())})
    at = slate(book="draftkings")
    assert picked(at) == "draftkings"
    phi = game("PHI", "DET", book="draftkings")
    assert pills(phi) is None and 'class="bv-pl-row"' not in html(phi)
    assert props_ui.NO_LINES_GAME_FEED in html(phi)                # upcoming: the feed's prompt
    assert pills(game("DEN", "SAS", book="draftkings")) is not None


# -- ?book= -------------------------------------------------------------------------------------
def test_book_draftkings_in_the_url_picks_draftkings_on_first_load(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    at = slate(book="draftkings")
    assert picked(at) == "draftkings" and ledger_head(at) == "DK line"
    assert pill_counts(at)[LATE] > 0                     # DraftKings posted DEN at SAS
    g = game("DEN", "SAS", book="draftkings")
    assert picked(g) == "draftkings" and LINES_HEAD.search(html(g)).group(1) == "DK line"


@pytest.mark.parametrize("junk, nowhere", [("<script>alert(1)</script>", True),
                                           ("DraftKings", False),   # a title, never a key
                                           ("pinnacle", True), ("fanduel\"><b>x</b>", True)])
def test_a_junk_book_gives_prizepicks_and_is_never_echoed(monkeypatch, tmp_path, junk, nowhere):
    """A junk ?book= is dropped from the URL, gives PrizePicks and is never
    carried by a link; a value that is not a book's own words is nowhere on
    the page at all."""
    feed_world(monkeypatch, tmp_path)
    pages = [slate(book=junk), game("PHI", "DET", book=junk),
             PS.run("pages/About.py", book=junk, date=DATE)]
    for at in pages[:2]:
        assert picked(at) == "prizepicks" and ledger_head(at) in ("PP line", "")
    for at in pages:
        assert "book" not in at.query_params and not any("book=" in h for h in hrefs(at))
        assert "book=" + junk not in dump(at)
        if nowhere:
            assert junk not in dump(at) and junk not in text(at)


def test_with_the_feed_off_a_book_in_the_url_changes_nothing(monkeypatch, tmp_path):
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = paste(monkeypatch, w)
    for path, q in (("app.py", {"board": token}),
                    ("pages/Game.py", {"away": "PHI", "home": "DET", "board": token})):
        plain = PS.run(path, date=DATE, **q)
        with_book = PS.run(path, date=DATE, book="draftkings", **q)
        assert dump(with_book) == dump(plain)
        assert with_book.query_params["book"] in ("draftkings", ["draftkings"])   # never read
        assert props_ui.BOOK_KEY not in with_book.session_state


# -- A pick ---------------------------------------------------------------------------------------
def test_a_pick_changes_the_header_the_rows_and_the_cards(monkeypatch, tmp_path):
    """Amendments A3, A4 and A7: the pills sit below the cards, yet a pick
    is read before the board is built, so the cards already count the picked
    book's shown lines, which are production's for the same inputs."""
    feed_world(monkeypatch, tmp_path)
    at = slate()
    want = {b: props_ui.line_counts_by_game(production(b).shown) for b in ("prizepicks", "draftkings")}
    assert ledger_head(at) == "PP line" and pill_counts(at) == {EARLY: want["prizepicks"][EARLY],
                                                                 LATE: 0}
    pp = ledger(at)
    assert "Tobias Harris" not in pp and all(r["lean"] in ("More", "Less") for r in pp.values())
    at.button_group(key=props_ui.BOOK_KEY).set_value("draftkings").run()
    assert not at.exception and at.query_params["book"] in ("draftkings", ["draftkings"])
    assert ledger_head(at) == "DK line" and picked(at) == "draftkings"
    dk = ledger(at)
    assert dk["Tobias Harris"]["line"] == "12.5"                       # DraftKings' line alone
    assert {"Over", "Under"} >= {r["lean"] for r in dk.values()}        # a sportsbook's words (m8)
    assert pill_counts(at) == {EARLY: want["draftkings"][EARLY], LATE: want["draftkings"][LATE]}
    assert pill_counts(at) != {EARLY: want["prizepicks"][EARLY], LATE: 0}
    # The cards' pill says the board's lines, not "your" lines (design 2.7).
    assert re.search(r'title="\d+ lines shown for this game"', html(at))
    assert "of your lines" not in html(at)
    # Biggest discrepancies names the book (amendment A8).
    assert re.search(r"\d+ standard DraftKings lines?, largest gap first", html(at))


def test_the_slates_count_equals_the_game_pages_count_for_every_book(monkeypatch, tmp_path):
    """The card pill, the Game page's rows, its expanders and its teal counts
    agree for every pick (amendment A7; tests/test_pages_smoke.py:595-603)."""
    feed_world(monkeypatch, tmp_path)
    for book in props.BOOK_ORDER:
        counts = pill_counts(slate(book=book))
        assert counts == {gid: props_ui.line_counts_by_game(production(book).shown).get(gid, 0)
                          for _t, gid in GAMES}, book
        for (away, home), gid in GAMES:
            g = game(away, home, book=book)
            assert page_rows(g) == counts[gid], (book, away)
            opened = sum(len(v) for v in player_lines(g).values())
            teal = sum(int(n) for n in re.findall(r'<span class="bv-nl"[^>]*>(\d+)</span>', html(g)))
            assert opened == teal == counts[gid], (book, away)
            if counts[gid]:
                assert LINES_HEAD.search(html(g)).group(1) == f"{props.BOOK_CODE[book]} line"


def test_a_pick_survives_a_rerun_and_only_a_url_change_reseeds_it(monkeypatch, tmp_path):
    """The pills follow ?book= only when the URL CHANGED since the last run
    (memory note streamlit-keyed-widget-ignores-default). A pill pick writes
    ?book= in its on_change before the script reads the URL, and survives
    reruns: another widget's, a date with no lines (where Streamlit drops the
    pills' state) and back. A pick the session holds while the URL still
    names the book it was last reconciled with must survive a rerun too: a
    seed on every run, or whenever URL and pick disagree, throws it away.
    Only a changed URL (back or forward) replaces the pick."""
    feed_world(monkeypatch, tmp_path)
    at = slate()
    at.button_group(key=props_ui.BOOK_KEY).set_value("fanduel").run()
    assert picked(at) == "fanduel" and ledger_head(at) == "FD line"
    assert at.query_params["book"] in ("fanduel", ["fanduel"])
    at.run()
    assert picked(at) == "fanduel"
    at.button(key="bv_day_next").click().run()                    # Mar 13: no lines, no pills
    assert not at.exception and pills(at) is None
    at.button(key="bv_day_prev").click().run()
    assert not at.exception and picked(at) == "fanduel" and ledger_head(at) == "FD line"
    at.session_state[props_ui.BOOK_KEY] = "betmgm"                # the URL still says fanduel
    at.run()
    assert not at.exception and picked(at) == "betmgm" and ledger_head(at) == "MGM line"
    at.query_params["book"] = "underdog"                            # back or forward
    at.run()
    assert picked(at) == "underdog" and ledger_head(at) == "UD line"


# -- The Other books cell (amendment A5) ------------------------------------------------------
ARROWS = (chr(0x2191), chr(0x2193), chr(0x25B2), chr(0x25BC), "&uarr;", "&darr;", "&#8593;",
          "&#8595;")


def _plain_cell(ob: str) -> None:
    """The cell holds codes and numbers only: no lean, colour, arrow, gap or
    price."""
    assert "class=" not in ob and "style=" not in ob, ob
    assert not any(a in ob for a in ARROWS), ob
    assert not re.search(r"(?<![\d.])[+-]\d", PS.visible(ob)), ob      # no gap, no price


def test_the_other_books_cell_lists_the_other_four_in_book_order(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    at = slate(book="draftkings")
    rows = ledger(at)
    assert PS.visible(rows["Cade Cunningham"]["ob"]) == "PP 25.5, UD 25.5, FD 25.5, MGM 25.5"
    assert PS.visible(rows["Jalen Duren"]["ob"]) == "PP 13.5, FD 13.5, MGM 13.5"   # no UD line
    assert rows["Tobias Harris"]["ob"] == SENTINEL                     # none: the whole node
    for r in rows.values():
        _plain_cell(r["ob"])
    assert ('<abbr title="PrizePicks">PP</abbr> 25.5, <abbr title="Underdog">UD</abbr> 25.5'
            in rows["Cade Cunningham"]["ob"])
    page = html(at)
    assert '<span class="ob">Other books</span>' in page
    assert props_ui.books_legend_html() in page
    assert PS.visible(props_ui.books_legend_html()) == (
        "PP PrizePicks, UD Underdog, DK DraftKings, FD FanDuel, MGM BetMGM.")
    # With PrizePicks picked, PrizePicks is the row and the cell is the other four.
    pp = ledger(slate())
    assert PS.visible(pp["Cade Cunningham"]["ob"]) == "UD 25.5, DK 25.5, FD 25.5, MGM 25.5"
    assert PS.visible(pp["Jalen Duren"]["ob"]) == "DK 13.5, FD 13.5, MGM 13.5"


def test_the_game_page_lists_all_five_books_per_player_a7(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    lines = player_lines(game("PHI", "DET", book="draftkings"))
    (cade,) = lines["Cade Cunningham"]
    assert cade["line"] == "25.5" and cade["lean"] in ("Over", "Under")
    assert PS.visible(cade["ob"]) == "PP 25.5, UD 25.5, FD 25.5, MGM 25.5"
    (harris,) = lines["Tobias Harris"]
    assert harris["ob"] == SENTINEL
    for recs in lines.values():
        for r in recs:
            _plain_cell(r["ob"])


def test_a_books_older_line_shows_its_own_time_m23(monkeypatch, tmp_path):
    """The evening refresh at 6:30 PM came back without FanDuel, so its 8:55
    AM lines stand (stale beats empty) and say how old they are; the books
    the refresh brought need no time."""
    feed_world(monkeypatch, tmp_path, at=et(18, 40),
               pulls=[(et(8, 55), ()), (et(18, 30), ("fanduel",))])
    rows = ledger(slate(book="draftkings"))
    assert PS.visible(rows["Cade Cunningham"]["ob"]) == "PP 25.5, UD 25.5, FD 25.5 (8:55 AM), MGM 25.5"
    lines = player_lines(game("PHI", "DET", book="draftkings"))
    assert PS.visible(lines["Cade Cunningham"][0]["ob"]) == (
        "PP 25.5, UD 25.5, FD 25.5 (8:55 AM), MGM 25.5")


# -- A game the picked book has not posted (amendment A6) ----------------------------------
def test_a_game_the_picked_book_has_not_posted_shows_no_rows_and_says_so(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    at = slate()                                                   # PrizePicks
    words = text(at)
    assert ("No PrizePicks lines yet for DEN at SAS. Pick another book to see theirs." in words)
    # Only PrizePicks' own rows, all of them PHI at DET's: production's shown
    # rows for the pick, one per player, at PrizePicks' numbers.
    want = production("prizepicks").shown
    assert set(want["game_id"].astype(str)) == {EARLY}
    rows = ledger(at)
    assert set(rows) == set(want["player"])
    assert {n: r["line"] for n, r in rows.items()} == {
        "Cade Cunningham": "25.5", "Jalen Duren": "13.5", "VJ Edgecombe": "17.5"}
    assert pill_counts(at)[LATE] == 0
    g = game("DEN", "SAS")
    assert pills(g) is not None and 'class="bv-pl-row"' not in html(g)
    assert "No PrizePicks lines for this game yet. Pick another book to see theirs." in text(g)
    assert "Lines on this game" not in html(g) and 'class="bv-nl"' not in html(g)
    assert props_ui.NO_LINES_GAME_FEED not in html(g) and "PP line" not in html(g)
    assert player_lines(g) == {}


def test_the_notice_names_two_games_and_shows_when_nothing_else_is_drawn_m7(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, payloads={LF.PHI_EID: PF.without(PF.phi_det(), "prizepicks")})
    at = slate()
    assert WINDOW not in html(at) and "Biggest discrepancies" not in html(at)
    assert ("No PrizePicks lines yet for PHI at DET and DEN at SAS. Pick another book to see theirs."
            in text(at))
    assert pills(at) is not None and props_ui.NO_LINES_SLATE_FEED not in html(at)
    at.button_group(key=props_ui.BOOK_KEY).set_value("underdog").run()
    assert WINDOW in html(at) and "No Underdog lines yet" not in html(at)


# -- The links keep the pick ---------------------------------------------------------------------
def test_the_links_between_the_slate_and_the_game_page_keep_the_book(monkeypatch, tmp_path):
    w, _f = feed_world(monkeypatch, tmp_path)
    token = paste(monkeypatch, w)
    s = slate(book="draftkings", board=token)
    links = hrefs(s)
    assert any(h.startswith("/Game?") for h in links) and any(h.startswith("/About") for h in links)
    assert all("book=draftkings" in h and f"board={token}" in h for h in links), links
    (share,) = [c.value for c in s.code]
    assert "book=draftkings" in share
    g = game("PHI", "DET", book="draftkings", board=token)
    back = re.search(r'<a class="bv-back" href="([^"]*)"', html(g)).group(1)
    assert "book=draftkings" in unescape(back)
    assert all("book=draftkings" in h for h in hrefs(g))
    # A pick rewrites every link on the same run; PrizePicks rides as no param.
    s.button_group(key=props_ui.BOOK_KEY).set_value("fanduel").run()
    assert all("book=fanduel" in h for h in hrefs(s))
    s.button_group(key=props_ui.BOOK_KEY).set_value("prizepicks").run()
    assert "book" not in s.query_params and not any("book=" in h for h in hrefs(s))
    # The rosters down (no paste popover, whose new token rebuilds the links):
    # the links the page minted first carry the pick too.
    with monkeypatch.context() as m:
        def no_roster():
            raise RuntimeError("roster: offline in the tests")
        m.setattr(live, "roster", no_roster)
        bare = slate(book="betmgm")
        assert any(h.startswith("/Game?") for h in hrefs(bare))
        assert all("book=betmgm" in h for h in hrefs(bare))
    # The feed off: a ?book= is never carried.
    monkeypatch.delenv(live.KEY_ENV)
    off = slate(book="draftkings", board=token)
    assert pills(off) is None and not any("book=" in h for h in hrefs(off))


# -- The feed's phrases (design 6; build amendment N1) ---------------------------------------
NO_FEED_WORDS = ("Odds API", "the-odds-api", "credit", "remaining", "OUT_OF_USAGE", "INVALID_KEY",
                 "x-requests", "quota")


def _no_internals(at) -> None:
    words = text(at)
    for w in NO_FEED_WORDS:
        assert w not in words, w


@pytest.mark.parametrize("state, posted, slate_says, game_says", [
    ("stale", True,
     "Lines as of 12:20 PM ET. The odds feed did not answer just now, so the board shows the last "
     "lines it had.",
     "Lines as of 12:20 PM ET. The odds feed did not answer just now, so the board shows the last "
     "lines it had."),
    ("paused", True,
     "Lines as of 12:20 PM ET. The board is not refreshing these lines for now.",
     "Lines as of 12:20 PM ET. The board is not refreshing these lines for now."),
    ("down", False, PS.visible(props_ui.FEED_DOWN), PS.visible(props_ui.FEED_DOWN)),
    ("paused", False, PS.visible(props_ui.FEED_PAUSED_EMPTY), PS.visible(props_ui.FEED_PAUSED_EMPTY)),
])
def test_the_down_paused_and_stale_phrases(monkeypatch, tmp_path, state, posted, slate_says,
                                           game_says):
    payloads = None if posted else {LF.PHI_EID: empty(PF.phi_det()), LF.DEN_EID: empty(PF.den_sas())}
    feed_world(monkeypatch, tmp_path, payloads=payloads, state=state)
    at = slate()
    assert slate_says in text(at)
    g = game("PHI", "DET")
    assert game_says in text(g)
    for page in (at, g):
        _no_internals(page)
        # With no lines the phrase is the prompt: the feed's "fills from the
        # odds feed" prompt would contradict it.
        assert props_ui.NO_LINES_SLATE_FEED not in html(page)
        assert props_ui.NO_LINES_GAME_FEED not in html(page)
    assert (WINDOW in html(at)) is posted


def test_a_morning_restart_whose_re_buys_the_floor_refuses_reads_paused_f4(monkeypatch, tmp_path):
    """Build 3's review, F4, on the pages, through the REAL walk and the
    REAL view (no state forced): tests/test_live_feed.py's short morning
    (100 left and 100 used, a period that was spending, 139 other games
    waiting for their morning pull) in a ledger born at 08:55, a restart.
    Both first pulls are re-buys the floor refuses (B11), so the feed holds
    no line today, and it reads "paused" (design 1.3 and 6). The slate and
    both Game pages say there are no lines from the odds feed today (the
    PAUSED phrase, with the paste instruction), never that the board fills
    from the odds feed from about 9 AM. The same morning in a ledger born at
    08:00, which saw it, buys both games (B1)."""
    _w, f = feed_world(monkeypatch, tmp_path, state=None, balance=(100, 100), horizon=LF.HORIZON,
                       pulls=[(et(8, 55), ())])
    assert f.vendor.odds_calls() == [] and f.mem["budget"]["mode"] == "short"
    assert {e["held"] for e in f.mem["games"].values()} == {"floor"} and len(f.mem["games"]) == 2
    said = PS.visible(props_ui.FEED_PAUSED_EMPTY)
    s = slate()
    assert said in text(s) and props_ui.NO_LINES_SLATE_FEED not in html(s)
    assert pills(s) is None and WINDOW not in html(s)
    _no_internals(s)
    for (away, home), _gid in GAMES:
        g = game(away, home)
        assert said in text(g) and props_ui.NO_LINES_GAME_FEED not in html(g), away
        _no_internals(g)
    _w, f = feed_world(monkeypatch, tmp_path / "early", state=None, balance=(100, 100),
                       horizon=LF.HORIZON, pulls=[(et(8, 0), ()), (et(8, 55), ())])
    assert [c.event_id for c in f.vendor.odds_calls()] == [LF.PHI_EID, LF.DEN_EID]
    s = slate()
    assert "Lines as of 8:55 AM ET." in text(s) and WINDOW in html(s) and said not in text(s)


def test_the_loading_line_shows_once_while_games_load(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, loading=2)
    line = "Lines for 2 games are still loading; this board fills in by itself."
    at = slate()
    assert text(at).count(line) == 1 and at.session_state[store.WATCH_RUNS_KEY] == 1
    seq = order(at)
    watch = next(i for i, (k, v) in enumerate(seq) if k == "md" and line in PS.visible(v))
    last_card = max(i for i, (k, v) in enumerate(seq) if k == "md" and "bv-slate-card" in v)
    assert watch == last_card + 1                                  # right after the cards
    # The Game page counts its own game only (build 4's review, L3-4).
    one = "Lines for 1 game are still loading; this board fills in by itself."
    assert text(game("PHI", "DET")).count(one) == 1 and line not in text(game("PHI", "DET"))
    # Loading with no lines yet: the line, and no "No lines on this page yet" beside it.
    feed_world(monkeypatch, tmp_path / "idle", loading=2,
               payloads={LF.PHI_EID: empty(PF.phi_det()), LF.DEN_EID: empty(PF.den_sas())})
    idle = slate()
    assert text(idle).count(line) == 1 and props_ui.NO_LINES_SLATE_FEED not in html(idle)


def test_the_newer_of_a_paste_and_the_feed_shows_and_the_board_says_which(monkeypatch, tmp_path):
    """M5 on the page: the feed's PrizePicks line for Cunningham's points is
    26.5 (pulled at 8:55 AM); the visitor's paste says 25.5. Pasted at 12:20
    PM, the paste is newer and its line shows; pasted at 8:00 AM (a kept
    copy restored), the feed's line shows. Either way the notice says which."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    H.install_feed(monkeypatch, payloads={LF.PHI_EID: PF.moved(PF.phi_det(), "prizepicks",
                                                                "Cade Cunningham", 26.5)},
                   pulls=[(et(8, 55), ())])
    plain = ledger(slate())
    assert plain["Cade Cunningham"]["line"] == "26.5"
    newer = slate(board=paste(monkeypatch, w))
    assert ledger(newer)["Cade Cunningham"]["line"] == "25.5"
    assert re.search(r"Your newer, typed, Demon or Goblin PrizePicks lines are shown for \d+ "
                     r"players? and stats?\.", text(newer))                  # build amendment B28
    # The paste's own DEN at SAS lines are PrizePicks lines for that game: no A6 notice for it.
    assert "No PrizePicks lines yet" not in text(newer)
    older = slate(board=paste(monkeypatch, w, et(8, 0)))
    assert ledger(older)["Cade Cunningham"]["line"] == "26.5"
    assert re.search(r"Your pasted PrizePicks lines from 8:00 AM are older than the odds feed's "
                     r"\(8:55 AM\), so the feed's line is shown for \d+ players? and stats?\.",
                     text(older)), text(older)


# -- Lines the board cannot compare (build 3's review, F1) -----------------------------------------
FAILED_SLATE = ("The comparison could not be built for this date. The odds feed's lines are still "
                "in memory; try reloading the page.")
FAILED_GAME = ("The comparison with the board's lines could not be built just now. The odds feed's "
               "lines are still in memory; try reloading the page.")


def _no_comparison(monkeypatch, how: str) -> None:
    """props.compare_books raises, or the simulator parameters do not load
    (store.sim_params raises, as app.py and Game.py catch it)."""
    def boom(*a, **k):
        raise KeyError("vendor-said-xyzzy")
    if how == "compare_books":
        monkeypatch.setattr(props, "compare_books", boom)
    else:
        monkeypatch.setattr(store, "sim_params", boom)


@pytest.mark.parametrize("how", ["compare_books", "sim_params"])
def test_lines_the_board_cannot_compare_are_explained_never_promised_f1(monkeypatch, tmp_path, how):
    """Build 3's review, F1. The feed holds lines for both games, but the
    comparison cannot be built. The slate keeps the feed's stamp and its
    board says the comparison failed and the feed's lines are still in
    memory (design 2.6's words); both Game pages say so in theirs. No page
    promises that the board fills from the odds feed from about 9 AM, no
    page has pills (no book has a board to pick), no card counts a line, and
    no error text shows. With a paste, the notices say the paste is kept."""
    w, _f = feed_world(monkeypatch, tmp_path)
    token = paste(monkeypatch, w)
    _no_comparison(monkeypatch, how)
    s = slate()
    words = text(s)
    assert "Lines as of 12:20 PM ET." in words and FAILED_SLATE in words, words
    assert props_ui.NO_LINES_SLATE_FEED not in html(s) and pills(s) is None
    assert pill_counts(s) == {EARLY: 0, LATE: 0} and "xyzzy" not in dump(s)
    assert WINDOW in html(s) and "Biggest discrepancies" not in html(s)
    for (away, home), _gid in GAMES:
        g = game(away, home)
        assert FAILED_GAME in text(g), away
        assert props_ui.NO_LINES_GAME_FEED not in html(g) and pills(g) is None
        assert 'class="bv-pl-row"' not in html(g) and "xyzzy" not in dump(g)
    if how == "sim_params":
        # No simulator parameters: the paste cannot be compared either, so the
        # notices say the paste is kept.
        assert ("The comparison could not be built for this date. Your pasted lines are still "
                "saved; try reloading the page.") in text(slate(board=token))
        assert ("The comparison with the board's lines could not be built just now. Your pasted "
                "lines are still saved; try reloading the page.") in text(game("PHI", "DET",
                                                                               board=token))
    else:
        # Build amendment B17: with a paste for the date, the paste's own board
        # shows under one note (test_a_paste_still_shows_when_the_feeds_comparison_fails_b17).
        assert props_ui.PASTE_ONLY_NOTE in text(slate(board=token))
        assert props_ui.PASTE_ONLY_NOTE in text(game("PHI", "DET", board=token))


def test_the_prompts_promise_lines_only_while_the_feed_holds_none_f1(monkeypatch, tmp_path):
    """The feed's prompts say lines load from the odds feed from about 9 AM:
    only while the feed holds no line for the date (the slate) or for the
    game (the Game page). Here the feed holds lines and, by a fault, no
    tables come back at all (store.feed_tables gives None): no board, and no
    promise either. With no line held for PHI at DET while DEN at SAS has
    some, PHI at DET's page still prompts, DEN at SAS's and the slate do
    not."""
    feed_world(monkeypatch, tmp_path)
    monkeypatch.setattr(store, "feed_tables", lambda *a, **k: None)
    s = slate()
    assert WINDOW not in html(s) and props_ui.NO_LINES_SLATE_FEED not in html(s)
    for (away, home), _gid in GAMES:
        assert props_ui.NO_LINES_GAME_FEED not in html(game(away, home)), away
    feed_world(monkeypatch, tmp_path / "phi", payloads={LF.PHI_EID: empty(PF.phi_det())})
    assert props_ui.NO_LINES_GAME_FEED in html(game("PHI", "DET"))
    assert props_ui.NO_LINES_GAME_FEED not in html(game("DEN", "SAS"))
    assert props_ui.NO_LINES_SLATE_FEED not in html(slate())


# -- After tip (m22) --------------------------------------------------------------------------------
def test_after_tip_the_game_page_says_its_lines_are_the_pre_tip_ones(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, at=H.EVENING, pulls=[(et(8, 55), ()), (et(18, 30), ())])
    g = game("PHI", "DET")                                         # tipped at 7:00 PM
    words = text(g)
    assert re.search(r"Lines on this game: \d+ of \d+ lines shown \([^)]*\)\. Pre-tip lines, as of "
                     r"6:30 PM ET\.", words), words
    d = game("DEN", "SAS", book="draftkings")                      # 9:00 PM: still ahead
    assert "Lines as of 6:30 PM ET." in text(d) and "Pre-tip" not in text(d)


def test_a_restart_after_tip_says_the_pre_tip_lines_were_not_kept_m22(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, at=et(19, 10), pulls=[(et(19, 10), ())])
    g = game("PHI", "DET")
    assert PS.visible(props_ui.after_tip_note_html("lost_in_restart")) in text(g)
    assert ("The odds feed's pre-tip lines for this game were not kept. The site restarted after "
            "its last update. A pasted board still shows here." in text(g))
    assert pills(g) is None and props_ui.NO_LINES_GAME_FEED not in html(g)
    assert pills(game("DEN", "SAS")) is not None                   # bought at the restart


def test_a_tipped_game_whose_books_posted_nothing_says_so(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, at=et(19, 10), payloads={LF.PHI_EID: empty(PF.phi_det())},
               pulls=[(et(8, 55), ())])
    words = text(game("PHI", "DET"))
    assert "No lines were posted for this game before tip." in words
    assert "were not kept" not in words
    # Before tip the same game gets the feed's prompt instead.
    feed_world(monkeypatch, tmp_path / "noon", payloads={LF.PHI_EID: empty(PF.phi_det())})
    noon = game("PHI", "DET")
    assert props_ui.NO_LINES_GAME_FEED in html(noon) and "before tip" not in text(noon)


# -- About (design 9.5) ---------------------------------------------------------------------------
SOURCES = ("Player lines on the board come from DraftKings, FanDuel, BetMGM, PrizePicks and "
           "Underdog through The Odds API, a licensed odds feed, refreshed through the day until "
           "shortly before tip; you can also paste your own PrizePicks board.")
BREAKEVEN = ("A sportsbook line is different: a side priced at -110 needs about 52.4% to break "
             "even, and the price sets the figure. Underdog's payouts are not modelled here, so no "
             "break-even is shown for its lines. No price is shown for any book.")   # B27


def test_about_says_where_the_player_lines_come_from_only_with_the_feed_on(monkeypatch, tmp_path):
    H.install(monkeypatch, tmp_path, H.MORNING)
    off = text(PS.run("pages/About.py"))
    assert SOURCES not in off and BREAKEVEN not in off and "odds feed" not in off
    H.install_feed(monkeypatch)
    on = text(PS.run("pages/About.py"))
    assert ("Game lines from ESPN (DraftKings). " + SOURCES + " Availability from the NBA's" in on)
    assert "An edge of a few points over the more common side is a long way from that. " + BREAKEVEN in on
    assert "indicative" not in on and on.count("No price is shown for any book.") == 1   # B27
    assert f"{100 * props.american_breakeven(-110):.1f}%" == "52.4%"
    for word in ("layout", "grid", "fallback", "Layout"):
        assert word not in on


# -- K6: fixed sentences, never an error's text ------------------------------------------------
LEAK = "vendor-said-xyzzy /srv/app/secret.py"


def test_the_outage_notices_never_print_the_errors_text_k6(monkeypatch, tmp_path):
    w = H.install(monkeypatch, tmp_path, H.MORNING)

    def down():
        raise RuntimeError(f"schedule: live fetch failed ({LEAK})")
    with monkeypatch.context() as m:
        m.setattr(live, "schedule", down)
        s = slate()
        assert ("The NBA schedule feed is unreachable and nothing is cached. BucketValue cannot "
                "draw a slate without it. This is a data-source outage, not an empty day. Reload "
                "in a minute." in text(s))
        g = game("PHI", "DET")
        assert "The schedule feed is unreachable." in text(g)
        for at in (s, g):
            assert "xyzzy" not in dump(at) and "/srv/" not in dump(at) and "Details" not in text(at)
    # The rosters fail after the page's first read: a play-in game draws them.
    import numpy as np
    sched = w.sched.assign(game_type=np.where(w.sched["game_id"] == LATE, "PI", w.sched["game_type"]))
    monkeypatch.setattr(live, "schedule", lambda: live.Live(sched, 0.0))
    reads = {"n": 0}

    def roster():
        reads["n"] += 1
        if reads["n"] > 1:
            raise RuntimeError(f"roster: down ({LEAK})")
        return live.Live(w.roster, 0.0)
    monkeypatch.setattr(live, "roster", roster)
    store._live_frame.clear()
    g = game("DEN", "SAS")
    assert "Rosters are unavailable." in text(g) and "xyzzy" not in dump(g)


# -- House style on the feed's pages (tests/test_pages_smoke.py's checks) ---------------------
def test_no_em_dash_outside_the_sentinel_no_emoji_and_no_key_on_the_feed_pages(monkeypatch, tmp_path):
    w, _f = feed_world(monkeypatch, tmp_path, at=et(18, 40),
                       pulls=[(et(8, 55), ()), (et(18, 30), ("fanduel",))])
    token = paste(monkeypatch, w, et(18, 40))
    cade = str(PS.pid_of("Cade Cunningham"))
    pages = [slate(board=token, out=cade), slate(book="draftkings"),
             game("PHI", "DET", board=token, out=cade), game("PHI", "DET", book="betmgm"),
             game("DEN", "SAS"), game("DEN", "SAS", book="underdog"),
             PS.run("pages/About.py", book="fanduel")]
    pages[1].button_group(key=props_ui.BOOK_KEY).set_value("fanduel").run()
    H.pin(monkeypatch, H.EVENING)
    store._live_frame.clear()
    pages += [game("PHI", "DET"), slate(book="underdog")]
    assert any(SENTINEL in m.value for at in pages for m in at.markdown)   # the cell's sentinel
    for at in pages:
        assert not at.exception
        for t in PS.texts(at):
            assert PS._dash_ok(t), [n for n in re.split(r"<[^>]*>", t) if SENTINEL in n][:3]
            assert not PS.EMOJI.search(t), PS.EMOJI.findall(t)[:3]
        assert KEY not in dump(at) and KEY[:20] not in " ".join(PS.texts(at))
        assert KEY not in repr(dict(at.query_params))
        state = at.session_state.filtered_state
        assert state and all(KEY not in repr(v) for v in state.values())
    # The pages never reached the feed anchor or started the refresher: the
    # lines came from the harness's own ledger.
    assert live.ANCHOR_NAME not in sys.modules
    assert not any(t.name == live.REFRESHER_THREAD for t in threading.enumerate())


# -- Build 4's fix pass: build amendments B17 and B18, review L3-1, L3-2, L3-4, L3-5 ----------
def test_a_paste_still_shows_when_the_feeds_comparison_fails_b17(monkeypatch, tmp_path):
    """Build amendment B17. The feed holds lines whose comparison failed and
    the visitor has a paste for the date: the paste's own board, drawn
    exactly as with the feed off (the same ledger, header, cards and Game
    page rows as the same pages with the feed stubbed off), with one note
    above it and no pills. Without a paste, the failure notice as built."""
    w, _f = feed_world(monkeypatch, tmp_path)
    token = paste(monkeypatch, w)
    with monkeypatch.context() as m:
        _forced_off(m)
        off_slate, off_game = slate(board=token), game("PHI", "DET", board=token)
    _no_comparison(monkeypatch, "compare_books")
    s = slate(board=token)
    words = text(s)
    assert props_ui.PASTE_ONLY_NOTE in words and FAILED_SLATE not in words
    assert ledger(s) == ledger(off_slate) and ledger(s) and ledger_head(s) == "Line"
    assert pill_counts(s) == pill_counts(off_slate) and sum(pill_counts(s).values()) > 0
    assert pills(s) is None and "Lines as of" not in words and "Other books" not in html(s)
    seq = order(s)
    note = next(i for i, (k, v) in enumerate(seq) if k == "md" and props_ui.PASTE_ONLY_NOTE
                in PS.visible(v))
    assert WINDOW in seq[note + 1][1]                                # right above the board
    assert words.count(props_ui.PASTE_ONLY_NOTE) == 1
    g = game("PHI", "DET", board=token)
    gw = text(g)
    assert props_ui.PASTE_ONLY_NOTE in gw and "Your lines on this game:" in gw
    assert player_lines(g) == player_lines(off_game) and player_lines(g)
    assert pills(g) is None and FAILED_GAME not in gw
    # Without a paste the failure notice stays as built.
    bare = text(slate())
    assert FAILED_SLATE in bare and props_ui.PASTE_ONLY_NOTE not in bare


def test_after_tip_a_game_the_feed_never_loaded_says_so_b18(monkeypatch, tmp_path):
    """Build amendment B18. After tip, "No lines were posted for this game
    before tip." only when its books really posted none; a game whose pulls
    all failed (never an answer with lines), one the budget held, or a
    paused day reads "The odds feed did not load lines for this game before
    tip."."""
    not_loaded = "The odds feed did not load lines for this game before tip."
    posted_none = "No lines were posted for this game before tip."
    # PHI at DET's only pull answered junk (a billed failure); DEN at SAS's landed.
    feed_world(monkeypatch, tmp_path, at=et(19, 10), payloads={LF.PHI_EID: {"bookmakers": 5}},
               pulls=[(et(8, 55), ())])
    words = text(game("PHI", "DET"))
    assert not_loaded in words and posted_none not in words
    # The budget held DEN at SAS all day (one credit above the reserve: PHI at
    # DET's first pull took it), read after its 9:00 PM tip.
    feed_world(monkeypatch, tmp_path / "held", at=et(21, 10), balance=(26, 0),
               pulls=[(et(8, 55), ()), (et(12, 0), ())])
    held = text(game("DEN", "SAS"))
    assert not_loaded in held and posted_none not in held
    # Books that posted nothing still say so.
    feed_world(monkeypatch, tmp_path / "none", at=et(19, 10),
               payloads={LF.PHI_EID: empty(PF.phi_det())}, pulls=[(et(8, 55), ())])
    none = text(game("PHI", "DET"))
    assert posted_none in none and not_loaded not in none


def test_the_stamp_is_the_picked_books_own_fetch_time_l3_1(monkeypatch, tmp_path):
    """Build 4's review, L3-1. A refresh at 6:30 PM left FanDuel out, so its
    8:55 AM lines stand (stale beats empty). With FanDuel picked the slate
    and the Game page say its lines are as of 8:55 AM, never the game's
    newer pull; with DraftKings picked, 6:30 PM. DEN at SAS posted nothing
    here, so the slate's stamp is PHI at DET's alone."""
    feed_world(monkeypatch, tmp_path, at=et(18, 40), payloads={LF.DEN_EID: empty(PF.den_sas())},
               pulls=[(et(8, 55), ()), (et(18, 30), ("fanduel",))])
    for book, when in (("fanduel", "8:55 AM"), ("draftkings", "6:30 PM")):
        s = text(slate(book=book))
        assert f"Lines as of {when} ET." in s, (book, s)
        g = text(game("PHI", "DET", book=book))
        assert re.search(r"Lines on this game: \d+ of \d+ lines shown \([^)]*\)\. Lines as of "
                         + re.escape(when) + r" ET\.", g), (book, g)
    assert "Lines as of 6:30 PM ET." not in text(slate(book="fanduel"))
    # The default board, PrizePicks, after PrizePicks dropped out of the refresh.
    feed_world(monkeypatch, tmp_path / "pp", at=et(18, 40),
               payloads={LF.DEN_EID: empty(PF.den_sas())},
               pulls=[(et(8, 55), ()), (et(18, 30), ("prizepicks",))])
    assert "Lines as of 8:55 AM ET." in text(slate())
    assert "Lines as of 6:30 PM ET." in text(slate(book="betmgm"))


def test_a_game_only_the_paste_posted_draws_the_pills_with_another_book_picked_l3_2(
        monkeypatch, tmp_path):
    """Build 4's review, L3-2 (amendment A6: the Game page of a game the
    picked book has not posted shows the pills and the note). DEN at SAS
    answered empty, so only the visitor's paste has its lines: with
    DraftKings picked its page says no DraftKings lines yet and points at
    the pills, which are there; picking PrizePicks brings the paste's rows
    back on the same page."""
    w, _f = feed_world(monkeypatch, tmp_path, payloads={LF.DEN_EID: empty(PF.den_sas())})
    token = paste(monkeypatch, w)
    d = game("DEN", "SAS", board=token, book="draftkings")
    assert pills(d) is not None and picked(d) == "draftkings"
    assert "No DraftKings lines for this game yet. Pick another book to see theirs." in text(d)
    assert page_rows(d) == 0
    d.button_group(key=props_ui.BOOK_KEY).set_value("prizepicks").run()
    assert not d.exception and page_rows(d) > 0 and pills(d) is not None
    # No paste: no line any pick can show on DEN at SAS, so no pills there.
    assert pills(game("DEN", "SAS", book="draftkings")) is None


def test_the_game_page_counts_its_own_games_load_only_l3_4(monkeypatch, tmp_path):
    """Build 4's review, L3-4: DEN at SAS loading is not PHI at DET's
    business. The loading line, and the watch that reruns the page, show on
    the page of the game that loads only, counting that game."""
    feed_world(monkeypatch, tmp_path, loading=1, loading_games=(LATE,))
    one = "Lines for 1 game are still loading; this board fills in by itself."
    den = game("DEN", "SAS")
    assert text(den).count(one) == 1 and den.session_state[store.WATCH_RUNS_KEY] == 1
    phi = game("PHI", "DET")
    assert "still loading" not in text(phi)
    assert store.WATCH_RUNS_KEY not in phi.session_state
    assert text(slate()).count(one) == 1                           # the slate counts the date's


def test_the_game_page_builds_the_other_books_from_its_own_games_rows_l3_5(monkeypatch, tmp_path):
    """Build 4's review, L3-5: each player's expander gets this game's
    context rows only, never the whole date's (whose index cost grew as
    players times the date's lines), and the cells are the same as from the
    whole date's context."""
    feed_world(monkeypatch, tmp_path)
    real = props_ui.render_player_lines_block
    whole = production("draftkings").context          # DraftKings posted both games
    seen = []

    def spy(rows, context=None, feed=None):
        seen.append(set(context["game_id"].astype(str)) if context is not None else None)
        assert props_ui.render_player_lines(rows, context, feed) == \
            props_ui.render_player_lines(rows, whole, feed)
        return real(rows, context, feed)
    monkeypatch.setattr(props_ui, "render_player_lines_block", spy)
    g = game("PHI", "DET", book="draftkings")
    assert seen and all(x == {EARLY} for x in seen), seen
    assert set(whole["game_id"].astype(str)) == {EARLY, LATE}
    assert player_lines(g)


# -- Build 6, stage S12: build amendments B24 and B25 on the pages --------------------------
PICK = "Pick another book to see theirs."


def test_the_a6_lines_name_no_direction_on_either_page_b24(monkeypatch, tmp_path):
    """Build amendment B24 (build 4's review, L3-3, which pinned the slate's
    "Pick another book above to see theirs." sitting above the pills it
    pointed up to). The sentence names no direction, so it is true on the
    slate, where amendment A3 puts the feed's notice above the pills, and
    on the Game page, whose note sits under them; the order is A3's,
    unchanged."""
    feed_world(monkeypatch, tmp_path)
    s, g = slate(), game("DEN", "SAS")
    seq = order(s)
    a6 = next(i for i, (k, v) in enumerate(seq) if k == "md" and PICK in PS.visible(v))
    assert a6 < seq.index(("pills", props_ui.BOOK_KEY))
    assert "No PrizePicks lines yet for DEN at SAS. " + PICK in text(s)
    gq = order(g)
    note = next(i for i, (k, v) in enumerate(gq) if k == "md" and PICK in PS.visible(v))
    assert gq.index(("pills", props_ui.BOOK_KEY)) < note
    assert "No PrizePicks lines for this game yet. " + PICK in text(g)
    for at in (s, g):
        assert "book above" not in text(at) and "book below" not in text(at)


def test_after_tip_and_for_a_book_the_site_does_not_buy_the_a6_lines_say_so_b25(monkeypatch, tmp_path):
    """Build amendment B25 (build 4's review, L3-6, which pinned A6's "yet"
    after tip and for a book never bought). PrizePicks has posted neither
    game here (PHI at DET's payload without it). At 7:10 PM PHI at DET has
    tipped and DEN at SAS has not: the slate's A6 list names DEN at SAS
    alone, PHI at DET's Game page says the odds feed did not load PrizePicks
    lines for it before tip (build amendment B29's sentence, in place of
    B25's "No PrizePicks lines were posted"), and DEN at SAS's still says
    "yet". Once both have tipped (9:30 PM) the slate names no game, and both
    Game pages say before tip. With BUCKETVALUE_ODDS_BOOKS leaving BetMGM out
    (the pills still show all five), a BetMGM pick says its lines are not
    loaded on this site, on the slate and on both Game pages, before tip and
    after, never "yet"."""
    after = f"The odds feed did not load PrizePicks lines for this game before tip. {PICK}"   # B29
    payloads = {LF.PHI_EID: PF.without(PF.phi_det(), "prizepicks")}
    feed_world(monkeypatch, tmp_path, at=et(19, 10), payloads=payloads)
    s = text(slate())
    assert f"No PrizePicks lines yet for DEN at SAS. {PICK}" in s and "PHI at DET and" not in s
    phi, den = text(game("PHI", "DET")), text(game("DEN", "SAS"))
    assert after in phi and "were posted" not in phi
    assert "lines for this game yet" not in phi
    assert f"No PrizePicks lines for this game yet. {PICK}" in den and "before tip" not in den
    feed_world(monkeypatch, tmp_path / "late", at=et(21, 30), payloads=payloads)
    assert "No PrizePicks lines yet" not in text(slate())
    for away, home in (("PHI", "DET"), ("DEN", "SAS")):
        assert after in text(game(away, home)), away
    narrow = {"BUCKETVALUE_ODDS_BOOKS": "draftkings,fanduel,prizepicks,underdog"}
    gone = f"BetMGM lines are not loaded on this site. {PICK}"
    for i, at in enumerate((H.MORNING, et(21, 30))):
        feed_world(monkeypatch, tmp_path / f"narrow{i}", at=at, environ=narrow)
        s = slate(book="betmgm")
        assert gone in text(s) and "No BetMGM lines" not in text(s), at
        assert pills(s).options == ["PrizePicks", "Underdog", "DraftKings", "FanDuel", "BetMGM"]
        for away, home in (("PHI", "DET"), ("DEN", "SAS")):
            g = text(game(away, home, book="betmgm"))
            assert gone in g and "No BetMGM lines" not in g, (at, away)
        if at == H.MORNING:          # a book the site buys keeps A6's own sentence
            assert f"No PrizePicks lines yet for DEN at SAS. {PICK}" in text(slate())


# -- Build 6's fix pass: the review of stage S12 (S12-R2 pinned, S12-R3 by B29) -------------
def test_after_a_cold_start_re_buy_the_game_page_says_the_feed_did_not_load_prizepicks_s12_r2_b29(
        monkeypatch, tmp_path):
    """Build 6's review, S12-R2, pinned as built (build amendment B33), and
    S12-R3, settled by build amendment B29. A restart at 10:00 ET in a period
    that was spending (the brake applies; P = 2), the harness's walks at
    10:00, noon, 6:30 PM and 8:30 PM. PrizePicks has not posted DEN at SAS
    at 10:00 and posts it from noon, but the 10:00 re-buy was the game's last
    pull (B21), so the site never asks again. At 8:50 PM the slate,
    PrizePicks picked, says PrizePicks has no lines yet for DEN at SAS (R2);
    at 9:30 PM, after its tip, DEN at SAS's Game page says the odds feed did
    not load PrizePicks lines for it before tip, which is true, where B25's
    sentence said none were posted, though PrizePicks had posted them since
    noon (R3)."""
    pulls = [(et(10, 0), ("prizepicks",)), (et(12, 0), ()), (et(18, 30), ()), (et(20, 30), ())]
    den = LF._with_prizepicks(LF._load("event_odds_den_sas_constructed.json"))
    world = dict(state=None, pulls=pulls, balance=(350, 150), horizon=LF.HORIZON,
                 payloads={LF.DEN_EID: den})
    _w, feed = feed_world(monkeypatch, tmp_path, at=et(20, 50), **world)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [LF.PHI_EID, LF.DEN_EID]
    assert feed.mem["brake_day"] is not None and feed.mem["games"][LF.DEN_EID]["pulls"] == 2
    assert f"No PrizePicks lines yet for DEN at SAS. {PICK}" in text(slate())
    feed_world(monkeypatch, tmp_path / "late", at=et(21, 30), **world)
    page = text(game("DEN", "SAS"))
    assert f"The odds feed did not load PrizePicks lines for this game before tip. {PICK}" in page
    assert "were posted" not in page and "lines for this game yet" not in page


# -- Build 7, stage S13: build amendment B30 on the Game page ----------------------------------
def _lines_text(at, player: str) -> str:
    """What a reader sees in `player`'s expander on the Game page ('' when he
    has none)."""
    for e in at.expander:
        if re.match(re.escape(player) + r": \d+ lines?$", str(e.label).replace("\\", "")):
            return PS.visible(" ".join(str(x.value) for x in e.markdown))
    return ""


def test_after_the_final_a_player_who_did_not_play_reads_voided_on_prizepicks_only_b30(
        monkeypatch, tmp_path):
    """Build amendment B30 on the Game page, through the real box score
    (serve_box: PHI at DET's final, in which Andre Drummond, Questionable at
    noon, did not play) and the real feed path. PrizePicks, Underdog and
    DraftKings each post a points line for him (an edited copy of the
    CONSTRUCTED payload). With PrizePicks picked his settled line reads "Did
    not play, pick voided"; with Underdog picked it reads "Did not play" and
    claims no void, as with DraftKings, since Underdog's own rule decides
    (B26). His line is shown each time, with the final's actual column."""
    payload = PF.phi_det()
    for book in ("prizepicks", "underdog", "draftkings"):
        payload = PF.added(payload, book, "Andre Drummond", 9.5)
    feed_world(monkeypatch, tmp_path, at=H.EVENING, payloads={LF.PHI_EID: payload})
    H.serve_box(monkeypatch)
    for book, note in (("prizepicks", "Did not play, pick voided"), ("underdog", "Did not play"),
                       ("draftkings", "Did not play")):
        g = game("PHI", "DET", book=book)
        assert picked(g) == book
        words = _lines_text(g, "Andre Drummond")
        assert "Final" in words and note in words, (book, words)
        assert ("pick voided" in words) == (book == "prizepicks"), (book, words)
