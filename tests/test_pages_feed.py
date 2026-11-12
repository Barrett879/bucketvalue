"""The pages with the licensed odds feed (design 2.8, 8.2 and 14 step 17;
layout C, amendments C6 to C14; build amendments N1 to N3), through
Streamlit's AppTest, offline.

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
12.5 at DraftKings alone. The grid is read back with tests/grid_parse.py,
which checks every cell against its own column.

With no key the pages must be exactly today's: the real feed_start gives the
same page, element for element, as a feed stubbed off.
"""
from __future__ import annotations

import copy
import re
import sys
import threading
from html import unescape
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import http, live, props, props_ui, store, theme
from bucketlib.util import SENTINEL, UTC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import grid_parse as GP  # noqa: E402  (layout C's grid read back, column by column)
import test_live_feed as LF  # noqa: E402  (the fake key, the fake vendor, the event ids)
import test_pages_smoke as PS  # noqa: E402  (the page runner and the house-style checks)
import test_props_feed as PF  # noqa: E402  (the constructed payloads, and edited copies)

DATE = H.DATE.isoformat()
KEY = LF.KEY                                  # "K" * 40, built at run time
EARLY, LATE = H.EARLY, H.LATE                 # PHI at DET 7:00 PM ET, DEN at SAS 9:00 PM ET
GAMES = ((("PHI", "DET"), EARLY), (("DEN", "SAS"), LATE))
et = LF.et                                    # an ET instant on 2026-03-12
CODES = ["DK", "FD", "MGM", "PP", "UD"]       # the grid's columns, the mockup's order
LABEL_STAT = {label: stat for stat, label in props.STAT_LABEL.items()}


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


def book_pills(at) -> list:
    """Layout B's book pills (key "bv_book"), which no page draws any more."""
    return [b for b in at.button_group if b.key == "bv_book"]


GRID_MARK = '<div class="bv-grid-wrap"'      # a grid's own markup (BOOKS_CSS names the class too)


def grid_drawn(at) -> bool:
    """Whether the page draws a grid: the grid's own markup, never a mere
    "bv-grid" in a page whose BOOKS_CSS names the class."""
    return GRID_MARK in html(at)


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


def production(token=None):
    """What production's date_board shows for the page's own inputs: the
    feed the pages are served, the date's served frame, the simulator
    parameters and the roster, compared by store.feed_tables."""
    fl = live.feed_view(H.DATE)
    served = store.load_projections(H.DATE)
    params = store.sim_params()
    roster = H.load().roster
    ft = store.feed_tables(H.DATE, fl, served, params, roster)
    return props_ui.date_board(token, H.DATE, served.frame, params, roster, feed=ft,
                               feed_info=store.feed_info(fl, H.DATE))


def hrefs(at) -> list[str]:
    return [unescape(h) for h in PS.HREF.findall(html(at))]


def by_key(view) -> dict:
    """{(player, stat): {book: (line row, shown)}} of a board's shown and
    hidden lines: what each grid cell must show."""
    out: dict = {}
    for frame, shown in ((view.shown, True), (view.hidden, False)):
        for r in frame.to_dict(orient="records"):
            out.setdefault((r["player"], r["stat"]), {})[r["book"]] = (r, shown)
    return out


def check_cells(row: dict, books: dict) -> None:
    """A parsed grid row's five cells against production's lines for its key:
    a shown line over or under by its own lean, a hidden one plain, a book
    with no line the sentinel; each at its own book's number."""
    for book in props.BOOK_ORDER:
        c = row["cells"][props.BOOK_CODE[book]]
        if book not in books:
            assert (c["cls"], c["number"], c["arrow"]) == ("none", SENTINEL, None), (
                row["player"], book)
            continue
        rec, shown = books[book]
        want = {"More": "over", "Less": "under"}.get(rec["lean"]) if shown else "plain"
        arrow = {"over": "&#8593;", "under": "&#8595;"}.get(want)
        assert (c["cls"], c["number"], c["arrow"]) == (want, f"{float(rec['line']):g}", arrow), (
            row["player"], book)
        if shown:
            word = props.side_words(book)[0 if rec["lean"] == "More" else 1]
            assert c["sr"] == f" leans {word}", (row["player"], book)


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


# -- The grid (layout C, amendments C6 and C7) -----------------------------------------------
def test_the_grid_shows_only_with_the_feed_on_and_lines_on_the_date(monkeypatch, tmp_path):
    H.install(monkeypatch, tmp_path, H.MORNING)
    for at in (slate(), slate(book="draftkings"), game("PHI", "DET", book="draftkings"),
               game("DEN", "SAS")):
        assert not grid_drawn(at) and not book_pills(at)        # the feed off: none, ever
    H.install_feed(monkeypatch)
    for at in (slate(), slate(book="draftkings"), game("PHI", "DET"), game("DEN", "SAS")):
        assert GP.grid_heads(html(at)) == CODES
        assert not book_pills(at) and not at.warning


def test_on_the_slate_the_feed_notice_sits_right_above_the_board(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    seq = order(slate())
    window = next(i for i, (k, v) in enumerate(seq) if k == "md" and WINDOW in v)
    notice = PS.visible(seq[window - 1][1])
    assert notice == "Lines as of 12:20 PM ET. No PrizePicks lines yet for DEN at SAS.", notice
    # The cards come before the notice, and no picker sits above the board.
    last_card = max(i for i, (k, v) in enumerate(seq) if k == "md" and "bv-slate-card" in v)
    assert last_card < window - 1 and all(k != "pills" for k, _v in seq[:window])


def test_no_page_draws_book_pills_or_reads_book():
    """Layout C (amendments C1 and C14): no page draws book pills or reads
    or writes a ?book=, in code (the AST, so comments and docstrings that
    name the old parameter do not count)."""
    import ast
    pages = ["app.py", *sorted(f"pages/{p.name}" for p in (PS.REPO / "pages").glob("*.py"))]
    assert {"app.py", "pages/Game.py", "pages/About.py"} <= set(pages)

    def const(node):
        return node.value if isinstance(node, ast.Constant) else None
    for rel in pages + ["bucketlib/store.py", "bucketlib/props_ui.py"]:
        tree = ast.parse((PS.REPO / rel).read_text())
        names = ({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
                 | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
        for gone in ("render_book_pills", "seed_book_from_url", "selected_book", "feed_games",
                     "BOOK_PARAM", "BOOK_KEY", "BOOK_SEEN"):
            assert gone not in names, (rel, gone)
        qp = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Attribute) and n.value.attr == "query_params"
              for t in n.targets if isinstance(t, ast.Name)}

        def is_qp(node):
            return ((isinstance(node, ast.Attribute) and node.attr == "query_params")
                    or (isinstance(node, ast.Name) and node.id in qp))
        for node in ast.walk(tree):
            bad = ((isinstance(node, ast.Subscript) and is_qp(node.value)
                    and const(node.slice) == "book")
                   or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and is_qp(node.func.value) and node.args and const(node.args[0]) == "book")
                   or (isinstance(node, ast.Compare) and const(node.left) == "book"
                       and any(is_qp(c) for c in node.comparators)))
            assert not bad, (rel, getattr(node, "lineno", None))


def test_on_the_game_page_the_lines_note_then_the_missing_books_note(monkeypatch, tmp_path):
    """Every book posted PHI at DET: its page has the "Lines on this game"
    note and no books-missing note. PrizePicks has not posted DEN at SAS:
    the note comes right after "Lines on this game", once."""
    feed_world(monkeypatch, tmp_path)
    phi = [PS.visible(v) for k, v in order(game("PHI", "DET")) if k == "md"]
    assert any(v.startswith("Lines on this game:") for v in phi)
    assert not any("lines for this game yet" in v or "not loaded on this site" in v for v in phi)
    den = [PS.visible(v) for k, v in order(game("DEN", "SAS")) if k == "md"]
    at = next(i for i, v in enumerate(den) if v.startswith("Lines on this game:"))
    assert den[at + 1] == "No PrizePicks lines for this game yet."
    assert den.count("No PrizePicks lines for this game yet.") == 1


def test_no_lines_for_the_date_means_no_pills_and_the_feeds_own_prompt(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, payloads={LF.PHI_EID: empty(PF.phi_det()),
                                               LF.DEN_EID: empty(PF.den_sas())})
    at = slate()
    assert not grid_drawn(at) and WINDOW not in html(at)
    assert props_ui.NO_LINES_SLATE_FEED in html(at) and props_ui.NO_LINES_SLATE not in html(at)
    den = game("DEN", "SAS")
    assert not grid_drawn(den) and props_ui.NO_LINES_GAME_FEED in html(den)


def test_a_game_with_no_lines_has_no_grid_while_the_slate_has_one(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, payloads={LF.PHI_EID: empty(PF.phi_det())})
    assert GP.grid(slate())                                        # DEN at SAS's lines
    phi = game("PHI", "DET")
    assert not grid_drawn(phi) and 'class="bv-pl-row"' not in html(phi)
    assert props_ui.NO_LINES_GAME_FEED in html(phi)                # upcoming: the feed's prompt
    books = "|".join(props.BOOK_TITLE[b] for b in props.BOOK_ORDER)
    assert not re.search(rf"No (?:{books})", text(phi))            # no book has posted it
    assert grid_drawn(game("DEN", "SAS"))


# -- ?book= -------------------------------------------------------------------------------------
def test_an_old_book_link_loads_and_changes_nothing(monkeypatch, tmp_path):
    """A ?book= from a layout B link still loads, and the page is the page
    without it, element for element (amendment C14)."""
    feed_world(monkeypatch, tmp_path)
    assert dump(slate(book="draftkings")) == dump(slate())
    for (away, home), _gid in GAMES:
        assert dump(game(away, home, book="draftkings")) == dump(game(away, home))
    assert dump(PS.run("pages/About.py", book="betmgm", date=DATE)) == dump(
        PS.run("pages/About.py", date=DATE))


@pytest.mark.parametrize("junk, nowhere", [("<script>alert(1)</script>", True),
                                           ("DraftKings", False),   # a title, never a key
                                           ("pinnacle", True), ("fanduel\"><b>x</b>", True)])
def test_a_junk_book_changes_nothing_and_is_never_echoed(monkeypatch, tmp_path, junk, nowhere):
    """A junk ?book= is never read: the pages equal the pages without it,
    no link or share link carries it, and a value that is not a book's own
    words is nowhere on the page. It stays in the visitor's own URL bar,
    which nothing reads (C1R-16(d))."""
    feed_world(monkeypatch, tmp_path)
    pairs = [(slate(book=junk), slate()), (game("PHI", "DET", book=junk), game("PHI", "DET")),
             (PS.run("pages/About.py", book=junk, date=DATE), PS.run("pages/About.py", date=DATE))]
    for at, plain in pairs:
        assert dump(at) == dump(plain)
        assert not any("book=" in h for h in hrefs(at)) and "book=" not in dump(at)
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
        assert "bv_book" not in with_book.session_state


# -- Every book at once ---------------------------------------------------------------------------
def test_the_grid_shows_every_book_and_the_cards_count_its_shown_lines(monkeypatch, tmp_path):
    """Amendments C6 and C8. The page world holds keys whose shown lines lean
    More at one book and Less at another (C1R-06), so a cell's arrow taken
    from its row's lead line could not pass. With Show all every row and
    cell is production's; the cards count every book's shown lines, and
    Biggest discrepancies counts its merged rows (C11)."""
    feed_world(monkeypatch, tmp_path)
    prod = production()
    lines = by_key(prod)
    both = [k for k, books in lines.items()
            if {"More", "Less"} <= {r["lean"] for r, shown in books.values() if shown}]
    assert both
    at = slate()
    assert GP.grid_heads(html(at)) == CODES
    rows = GP.grid(at, show_all=True)
    grid = props.grid_rows(prod.shown, prod.hidden)
    assert [(r["player"], r["stat"]) for r in rows] == [
        (g["player"], props.STAT_LABEL[g["stat"]]) for g in grid.to_dict(orient="records")]
    for r in rows:
        check_cells(r, lines[(r["player"], LABEL_STAT[r["stat"]])])
    assert any({"over", "under"} <= {c["cls"] for c in r["cells"].values()} for r in rows)
    cells = {r["player"]: r["cells"] for r in rows if r["stat"] == "Points"}
    assert [c["number"] for c in cells["Cade Cunningham"].values()] == ["25.5"] * 5
    assert cells["Jalen Duren"]["UD"]["cls"] == "none"
    assert [c["cls"] for code, c in cells["Tobias Harris"].items() if code != "DK"] == ["none"] * 4
    counts = pill_counts(at)
    assert counts == {gid: props_ui.line_counts_by_game(prod.shown).get(gid, 0) for _t, gid in GAMES}
    assert sum(counts.values()) == len(prod.shown) > len(rows)
    titles = [int(n) for n in re.findall(r'title="(\d+) lines shown for this game"', html(at))]
    assert sorted(titles) == sorted(n for n in counts.values() if n)
    n = len(props_ui.merge_books(props_ui.discrepancies(prod.shown)))
    assert f'<span class="meta">{n} standard lines, largest gap first</span>' in html(at)
    strip = next(m for m in (x.value for x in at.markdown) if '<div class="bv-gap-strip">' in m)
    assert strip.count('class="bv-gap-chip') == strip.count('<span class="gc-book">') == min(
        n, props_ui.DISCREPANCY_CHIPS)
    # Each Game page's section counts its own game's merged rows, a code in every chip.
    for (away, home), gid in GAMES:
        mine = prod.shown[prod.shown["game_id"].astype(str) == gid]
        n = len(props_ui.merge_books(props_ui.discrepancies(mine)))
        g = game(away, home)
        assert n > 1 and (f'<span class="meta">{n} standard lines, largest gap first</span>'
                          f'<span class="lock">{away} at {home}</span>') in html(g), away
        strip = next(m for m in (x.value for x in g.markdown) if '<div class="bv-gap-strip">' in m)
        assert strip.count('class="bv-gap-chip') == strip.count('<span class="gc-book">') == min(
            n, props_ui.DISCREPANCY_CHIPS), away


def test_each_players_cells_on_the_game_page_equal_the_slates(monkeypatch, tmp_path):
    """C1R-04: every (player, stat, book) cell of both Game pages' grids is
    the slate's cell with Show all on: class, number, hidden text, title and
    badges. Jalen Duren's BetMGM 13.5 (Under only, the model leaning Over) is
    the plain "not offered" cell with its tag on both pages, never the
    sentinel: his hidden lines reach his grid."""
    feed_world(monkeypatch, tmp_path)
    slate_rows = {(r["player"], r["stat"]): r["cells"] for r in GP.grid(slate(), show_all=True)}
    seen = 0
    for (away, home), _gid in GAMES:
        for player, rows in GP.player_grid(game(away, home)).items():
            for r in rows:
                assert r["cells"] == slate_rows[(player, r["stat"])], (player, r["stat"])
                seen += 1
    assert seen == len(slate_rows) > 0
    mgm = slate_rows[("Jalen Duren", "Points")]["MGM"]
    assert (mgm["cls"], mgm["number"], mgm["sr"]) == ("plain", "13.5", " leans Over, not offered")
    assert ">Under only</span>" in mgm["badges"]


def test_a_drawn_hidden_pick_is_never_called_not_shown_on_the_game_page(monkeypatch, tmp_path):
    """C1R-05, the page half, with test_pages_smoke's construction: a typed
    pick on the model's own side, re-added with the line offering only the
    other side, so the line is hidden. Cade Cunningham's points have shown
    lines at the other four books, so his PrizePicks pick is drawn (with its
    tag) and the "Lines on this game" note never calls it not shown, and says
    a hidden line is drawn. A pasted line of a stat the feed does not buy (its
    player's only line for that stat) is never drawn, and the note still
    names it. The slate's hidden note names both, as before."""
    w, _f = feed_world(monkeypatch, tmp_path)
    token = paste(monkeypatch, w)
    prod = production(token)
    rows = pd.concat([prod.shown, prod.hidden], ignore_index=True)
    cade = PS.pid_of("Cade Cunningham")
    board = props.get_board(token)

    def typed(token, rec):
        mine = board[(board["person_id"] == rec["person_id"]) & (board["stat"] == rec["stat"])]
        assert len(mine) == 1 and float(mine["line"].iloc[0]) == float(rec["line"])
        side = str(rec["lean"]).lower()
        return props.add_to_board(token, mine.assign(pick=side,
                                                     direction="less" if side == "more" else "more"))
    (pts,) = [r for r in rows.to_dict(orient="records") if r["person_id"] == cade
              and r["stat"] == "pts" and r["book"] == "prizepicks"]
    other = [r for r in rows.to_dict(orient="records") if r["origin"] == "pasted"
             and r["stat"] != "pts" and r["person_id"] != cade and str(r["game_id"]) == EARLY
             and r["lean"] in ("More", "Less")][0]
    assert pts["lean"] in ("More", "Less")
    token = typed(typed(token, pts), other)
    page = game("PHI", "DET", board=token)
    (cade_pts,) = [r for r in GP.player_grid(page)["Cade Cunningham"] if r["stat"] == "Points"]
    pp = cade_pts["cells"]["PP"]
    assert pp["cls"] == "plain" and ">Pick: " in pp["badges"] and pp["sr"].endswith(", not offered")
    assert sum(c["cls"] in ("over", "under") for c in cade_pts["cells"].values()) == 4
    (note,) = [PS.visible(m.value) for m in page.markdown
               if PS.visible(m.value).startswith("Lines on this game:")]
    named = (f"{other['player']} {props.STAT_LABEL[other['stat']]} (picked {other['lean']}; the "
             f"model leans {other['lean']}, a side this line does not offer)")
    assert props_ui.HIDDEN_DRAWN in note and "Cade Cunningham" not in note
    assert note.endswith(f"Not shown here: the pasted pick {named}."), note
    words = text(slate(board=token))
    assert "The pasted picks among them: " in words and named in words
    assert "Cade Cunningham Points (picked " in words


def test_a_game_whose_posted_lines_all_went_unmatched_still_names_the_book_missing_it(
        monkeypatch, tmp_path):
    """C1R-15: every book's DEN at SAS lines carry names nobody on the roster
    has (PR04's edit, applied to all of them), and PrizePicks has not posted
    the game. The page has no grid and no "Lines on this game" note, and
    still names the book missing it, once, as a note of its own; the
    feed's "no lines yet" prompt is not drawn beside it."""
    late = copy.deepcopy(PF.den_sas())
    for bm in late["bookmakers"]:
        for m in bm["markets"]:
            for oc in m["outcomes"]:
                oc["description"] = "Zzyzx " + oc["description"].split()[-1] + " Qq"
    feed_world(monkeypatch, tmp_path, payloads={LF.DEN_EID: late})
    g = game("DEN", "SAS")
    assert not grid_drawn(g) and "Lines on this game" not in text(g)
    notes = [m.value for m in g.markdown
             if m.value == '<div class="bv-note"><b>No PrizePicks lines for this game yet.</b></div>']
    assert len(notes) == 1 and text(g).count("No PrizePicks lines for this game yet.") == 1
    assert props_ui.NO_LINES_GAME_FEED not in html(g)


def test_the_slates_count_equals_the_game_pages_count(monkeypatch, tmp_path):
    """Per game the card count, the "Lines on this game" count, the
    expanders' counts, the teal counts and the arrowed cells of the page's
    grids are one number: every book's shown lines (amendment C8;
    tests/test_pages_smoke.py:595-603)."""
    feed_world(monkeypatch, tmp_path)
    counts = pill_counts(slate())
    assert counts == {gid: props_ui.line_counts_by_game(production().shown).get(gid, 0)
                      for _t, gid in GAMES}
    assert all(counts.values())
    for (away, home), gid in GAMES:
        g = game(away, home)
        noted = re.search(r"Lines on this game: (\d+) of \d+ lines shown", text(g))
        opened = sum(int(m.group(1)) for m in (re.match(r".+: (\d+) lines?$", str(e.label))
                                               for e in g.expander) if m)
        teal = sum(int(n) for n in re.findall(r'<span class="bv-nl"[^>]*>(\d+)</span>', html(g)))
        arrows = sum(c["cls"] in ("over", "under") for rows in GP.player_grid(g).values()
                     for r in rows for c in r["cells"].values())
        assert int(noted.group(1)) == opened == teal == arrows == counts[gid], (away, opened, teal)


def test_the_default_view_is_one_row_per_player_ranked_by_the_strongest_line(monkeypatch, tmp_path):
    """Without Show all: each player's best-ranked row, in the grid's own
    order (grid_rows: tier, then the strongest shown line's confidence), and
    the caption counts production's own players."""
    feed_world(monkeypatch, tmp_path)
    at = slate()
    prod = production()
    grid = props.grid_rows(prod.shown, prod.hidden)
    top, more = props_ui.top_per_player(grid, props_ui.TOP_N)
    rows = GP.grid(at)
    assert [(r["player"], r["stat"]) for r in rows] == [
        (t["player"], props.STAT_LABEL[t["stat"]]) for t in top.to_dict(orient="records")]
    assert len({r["player"] for r in rows}) == len(rows) > 1
    assert [r["more"] for r in rows] == [more.get(int(p), 0) for p in top["person_id"]]
    n = grid["person_id"].nunique()
    assert props_ui.GRID_CAPTION.format(k=min(props_ui.TOP_N, n), n=n) in [
        str(c.value) for c in at.caption]


def test_the_grids_live_inside_their_keyed_containers(monkeypatch, tmp_path):
    """Amendment C13's guard is anchored on two keyed containers, drawn only
    with the feed's rows: the slate's grid inside bv_board_grid, the Game
    page's grids inside bv_game_lines."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    token = paste(monkeypatch, w)
    for at in (slate(board=token), game("PHI", "DET", board=token)):
        assert GP.keyed_markdown(at, "bv_board_grid") is None
        assert GP.keyed_markdown(at, "bv_game_lines") is None
    H.install_feed(monkeypatch)
    s, g = slate(), game("PHI", "DET")
    (grid_md,) = [m.value for m in s.markdown if 'aria-label="Lines by book"' in m.value]
    assert grid_md in GP.keyed_markdown(s, "bv_board_grid")
    inside = GP.keyed_markdown(g, "bv_game_lines")
    mine = [m.value for m in g.markdown if GP.PLAYER_LABEL + '"' in m.value]
    assert mine and all(m in inside for m in mine)
    assert GP.keyed_markdown(s, "bv_game_lines") is None and GP.keyed_markdown(g, "bv_board_grid") is None


def test_the_dark_toggle_changes_the_tokens_not_the_grid(monkeypatch, tmp_path):
    """The grid is drawn in theme tokens: the Dark toggle changes the token
    block (THEME_LIGHT_CSS only on the light page), never the grid."""
    feed_world(monkeypatch, tmp_path)
    dark, light = slate(theme="dark"), slate(theme="light")

    def grids(at):
        return [m.value for m in at.markdown if m.value.startswith(GRID_MARK)]
    assert grids(dark) == grids(light) and len(grids(dark)) == 1
    light_css = theme.THEME_LIGHT_CSS.strip()
    assert light_css in [m.value for m in light.markdown]
    assert light_css not in [m.value for m in dark.markdown]


def test_a_rerun_and_a_date_step_keep_the_grid(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path)
    at = slate()
    first = GP.grid(at)
    assert first
    at.run()
    assert not at.exception and GP.grid(at) == first
    at.button(key="bv_day_next").click().run()                    # Mar 13: no lines, no grid
    assert not at.exception and not grid_drawn(at)
    at.button(key="bv_day_prev").click().run()
    assert not at.exception and GP.grid(at) == first


def test_a_book_that_has_not_posted_a_game_shows_the_sentinel_and_the_notice_names_it(
        monkeypatch, tmp_path):
    """PrizePicks has not posted DEN at SAS: its column is the sentinel on
    that game's rows, on the slate and on the game's page, and the notes
    name it; never another book's line under its header."""
    feed_world(monkeypatch, tmp_path)
    at = slate()
    assert "No PrizePicks lines yet for DEN at SAS." in text(at)
    late = {(r["player"], r["stat"]) for r in production().shown.to_dict(orient="records")
            if str(r["game_id"]) == LATE}
    rows = [r for r in GP.grid(at, show_all=True) if (r["player"], LABEL_STAT[r["stat"]]) in late]
    assert rows and all(r["cells"]["PP"]["cls"] == "none" for r in rows)
    g = game("DEN", "SAS")
    grids = GP.player_grid(g)
    assert grids and all(r["cells"]["PP"]["number"] == SENTINEL for rs in grids.values() for r in rs)
    assert "No PrizePicks lines for this game yet." in text(g) and "Lines on this game" in text(g)
    assert props_ui.NO_LINES_GAME_FEED not in html(g)
    assert "Pick another" not in text(g) + text(at)


def test_the_notice_names_two_games_and_shows_when_nothing_else_is_drawn_m7(monkeypatch, tmp_path):
    """PrizePicks posted neither game: the notice names both, above the
    board, which still draws the other books' lines (its PP column all
    sentinels)."""
    feed_world(monkeypatch, tmp_path, payloads={LF.PHI_EID: PF.without(PF.phi_det(), "prizepicks")})
    at = slate()
    words = text(at)
    assert "No PrizePicks lines yet for PHI at DET and DEN at SAS." in words
    assert "Pick another" not in words and props_ui.NO_LINES_SLATE_FEED not in html(at)
    seq = order(at)
    window = next(i for i, (k, v) in enumerate(seq) if k == "md" and WINDOW in v)
    notice = next(i for i, (k, v) in enumerate(seq) if k == "md"
                  and "No PrizePicks lines yet" in PS.visible(v))
    assert notice < window
    rows = GP.grid(at, show_all=True)
    assert rows and all(r["cells"]["PP"]["cls"] == "none" for r in rows)


# -- No link carries a book ---------------------------------------------------------------------
def test_no_link_carries_a_book(monkeypatch, tmp_path):
    w, _f = feed_world(monkeypatch, tmp_path)
    token = paste(monkeypatch, w)
    s = slate(book="draftkings", board=token)
    links = hrefs(s)
    assert any(h.startswith("/Game?") for h in links) and any(h.startswith("/About") for h in links)
    assert all(f"board={token}" in h and "book=" not in h for h in links), links
    (share,) = [c.value for c in s.code]
    assert f"board={token}" in share and "book=" not in share
    g = game("PHI", "DET", book="draftkings", board=token)
    back = unescape(re.search(r'<a class="bv-back" href="([^"]*)"', html(g)).group(1))
    assert f"board={token}" in back and "book=" not in back
    assert all("book=" not in h for h in hrefs(g)) and all("book=" not in c.value for c in g.code)
    # The rosters down (no paste popover, whose new token rebuilds the links):
    # the links the page minted first carry no book either.
    with monkeypatch.context() as m:
        def no_roster():
            raise RuntimeError("roster: offline in the tests")
        m.setattr(live, "roster", no_roster)
        bare = slate(book="betmgm")
        assert any(h.startswith("/Game?") for h in hrefs(bare))
        assert not any("book=" in h for h in hrefs(bare))
    # The feed off: a ?book= is never carried.
    monkeypatch.delenv(live.KEY_ENV)
    off = slate(book="draftkings", board=token)
    assert not grid_drawn(off) and not any("book=" in h for h in hrefs(off))


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
    assert not grid_drawn(s) and WINDOW not in html(s)
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
    PM, the paste is newer and its line shows in the PP column; pasted at
    8:00 AM (a kept copy restored), the feed's line shows. Either way the
    notice says which."""
    w = H.install(monkeypatch, tmp_path, H.MORNING)
    H.install_feed(monkeypatch, payloads={LF.PHI_EID: PF.moved(PF.phi_det(), "prizepicks",
                                                                "Cade Cunningham", 26.5)},
                   pulls=[(et(8, 55), ())])

    def pp_number(at):
        (row,) = [r for r in GP.grid(at, show_all=True)
                  if (r["player"], r["stat"]) == ("Cade Cunningham", "Points")]
        return row["cells"]["PP"]["number"]
    assert pp_number(slate()) == "26.5"
    newer = slate(board=paste(monkeypatch, w))
    assert pp_number(newer) == "25.5"
    assert re.search(r"Your newer, typed, Demon or Goblin PrizePicks lines are shown for \d+ "
                     r"players? and stats?\.", text(newer))                  # build amendment B28
    # The paste's own DEN at SAS lines are PrizePicks lines for that game: no notice for it.
    assert "No PrizePicks lines yet" not in text(newer)
    older = slate(board=paste(monkeypatch, w, et(8, 0)))
    assert pp_number(older) == "26.5"
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
    page draws a grid, no card counts a line, and no error text shows.
    With a paste, the notices say the paste is kept."""
    w, _f = feed_world(monkeypatch, tmp_path)
    token = paste(monkeypatch, w)
    _no_comparison(monkeypatch, how)
    s = slate()
    words = text(s)
    assert "Lines as of 12:20 PM ET." in words and FAILED_SLATE in words, words
    assert props_ui.NO_LINES_SLATE_FEED not in html(s) and not grid_drawn(s)
    assert pill_counts(s) == {EARLY: 0, LATE: 0} and "xyzzy" not in dump(s)
    assert WINDOW in html(s) and "Biggest discrepancies" not in html(s)
    for (away, home), _gid in GAMES:
        g = game(away, home)
        assert FAILED_GAME in text(g), away
        assert props_ui.NO_LINES_GAME_FEED not in html(g) and not grid_drawn(g)
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
    d = game("DEN", "SAS")                                         # 9:00 PM: still ahead
    assert "Lines as of 6:30 PM ET." in text(d) and "Pre-tip" not in text(d)


def test_a_restart_after_tip_says_the_pre_tip_lines_were_not_kept_m22(monkeypatch, tmp_path):
    feed_world(monkeypatch, tmp_path, at=et(19, 10), pulls=[(et(19, 10), ())])
    g = game("PHI", "DET")
    assert PS.visible(props_ui.after_tip_note_html("lost_in_restart")) in text(g)
    assert ("The odds feed's pre-tip lines for this game were not kept. The site restarted after "
            "its last update. A pasted board still shows here." in text(g))
    assert not grid_drawn(g) and props_ui.NO_LINES_GAME_FEED not in html(g)
    assert grid_drawn(game("DEN", "SAS"))                   # bought at the restart


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
    H.pin(monkeypatch, H.EVENING)
    store._live_frame.clear()
    pages += [game("PHI", "DET"), slate(book="underdog")]
    assert any(SENTINEL in m.value for at in pages for m in at.markdown)   # the grid's sentinel
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
    above it and no grid. Without a paste, the failure notice as built."""
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
    assert not grid_drawn(s) and "Lines as of" not in words and "Other books" not in html(s)
    seq = order(s)
    note = next(i for i, (k, v) in enumerate(seq) if k == "md" and props_ui.PASTE_ONLY_NOTE
                in PS.visible(v))
    assert WINDOW in seq[note + 1][1]                                # right above the board
    assert words.count(props_ui.PASTE_ONLY_NOTE) == 1
    g = game("PHI", "DET", board=token)
    gw = text(g)
    assert props_ui.PASTE_ONLY_NOTE in gw and "Your lines on this game:" in gw
    assert player_lines(g) == player_lines(off_game) and player_lines(g)
    assert not grid_drawn(g) and FAILED_GAME not in gw
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


def test_the_stamp_is_the_oldest_books_own_fetch_time_l3_1(monkeypatch, tmp_path):
    """Amendment C10 (build 4's review, L3-1). A refresh at 6:30 PM left
    FanDuel out, so its 8:55 AM lines stand (stale beats empty): the slate
    and PHI at DET's page say the lines are as of 8:55 AM, the oldest book's
    own time, never the newer pull's; FanDuel's cells say their own time in
    their titles, DraftKings' none. DEN at SAS posted nothing here, so the
    slate's stamp is PHI at DET's alone. Likewise with PrizePicks left out."""
    feed_world(monkeypatch, tmp_path, at=et(18, 40), payloads={LF.DEN_EID: empty(PF.den_sas())},
               pulls=[(et(8, 55), ()), (et(18, 30), ("fanduel",))])
    s = slate()
    assert "Lines as of 8:55 AM ET." in text(s) and "Lines as of 6:30 PM ET." not in text(s)
    g = game("PHI", "DET")
    assert re.search(r"Lines on this game: \d+ of \d+ lines shown \([^)]*\)\. Lines as of 8:55 AM "
                     r"ET\.", text(g)), text(g)
    assert "Lines as of 6:30 PM ET." not in text(g)
    for rows in (GP.grid(s, show_all=True), [r for rs in GP.player_grid(g).values() for r in rs]):
        fd = [r["cells"]["FD"] for r in rows if r["cells"]["FD"]["cls"] != "none"]
        dk = [r["cells"]["DK"] for r in rows if r["cells"]["DK"]["cls"] != "none"]
        assert fd and all(c["title"].endswith(" Line as of 8:55 AM ET.") for c in fd)
        assert dk and not any("Line as of" in c["title"] for c in dk)
    feed_world(monkeypatch, tmp_path / "pp", at=et(18, 40),
               payloads={LF.DEN_EID: empty(PF.den_sas())},
               pulls=[(et(8, 55), ()), (et(18, 30), ("prizepicks",))])
    s = slate()
    assert "Lines as of 8:55 AM ET." in text(s) and "Lines as of 6:30 PM ET." not in text(s)
    rows = GP.grid(s, show_all=True)
    pp = [r["cells"]["PP"] for r in rows if r["cells"]["PP"]["cls"] != "none"]
    assert pp and all(c["title"].endswith(" Line as of 8:55 AM ET.") for c in pp)
    assert not any("Line as of" in r["cells"]["DK"]["title"] for r in rows
                   if r["cells"]["DK"]["cls"] != "none")


def test_a_game_only_the_paste_posted_shows_its_prizepicks_column_l3_2(monkeypatch, tmp_path):
    """Build 4's review, L3-2, in layout C. DEN at SAS answered empty, so
    only the visitor's paste has its lines: the grid on its page shows them
    in the PP column, every other column the sentinel, and the note names
    the four books with no line for it. Without a paste, no grid there."""
    w, _f = feed_world(monkeypatch, tmp_path, payloads={LF.DEN_EID: empty(PF.den_sas())})
    token = paste(monkeypatch, w)
    d = game("DEN", "SAS", board=token)
    late = [r for r in production(token).shown.to_dict(orient="records")
            if str(r["game_id"]) == LATE]
    want = {(r["player"], r["stat"]): f"{float(r['line']):g}" for r in late}
    assert want and {(r["book"], r["origin"]) for r in late} == {("prizepicks", "pasted")}
    got = {}
    for player, rows in GP.player_grid(d).items():
        for r in rows:
            assert [r["cells"][c]["cls"] for c in ("DK", "FD", "MGM", "UD")] == ["none"] * 4
            pp = r["cells"]["PP"]
            assert pp["cls"] in ("over", "under") and pp["title"].startswith(
                f"PrizePicks {pp['number']} (pasted): ")
            got[(player, LABEL_STAT[r["stat"]])] = pp["number"]
    assert got == want
    assert "No DraftKings, FanDuel, BetMGM or Underdog lines for this game yet." in text(d)
    assert not grid_drawn(game("DEN", "SAS"))


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


def test_each_players_grid_gets_his_own_lines_only_l3_5(monkeypatch, tmp_path):
    """Build 4's review, L3-5, in layout C: each player's expander gets his
    own lines in this game only, shown and hidden, as the lists
    store.line_actuals hands over, never the whole date's."""
    feed_world(monkeypatch, tmp_path)
    real = props_ui.render_player_lines_block
    seen = []

    def spy(rows, context=None, feed=None):
        recs = list(rows) + list(context or [])
        seen.append(({int(r["person_id"]) for r in recs}, {str(r["game_id"]) for r in recs},
                     type(rows), type(context), len(context or [])))
        return real(rows, context, feed)
    monkeypatch.setattr(props_ui, "render_player_lines_block", spy)
    g = game("PHI", "DET")
    assert seen and all(len(pids) == 1 and gids == {EARLY} for pids, gids, *_ in seen), seen
    assert all(r is list and c is list for _p, _g, r, c, _n in seen)
    assert any(n for *_, n in seen)                     # some player's hidden lines reach his grid
    assert GP.player_grid(g)


def test_the_missing_books_lines_never_point_at_a_picker_b24(monkeypatch, tmp_path):
    """Build amendment B24 in layout C: there is no picker to point at, so no
    sentence says "Pick another book", or "above" or "below" a book; the
    slate and DEN at SAS's page name the book that has not posted it."""
    feed_world(monkeypatch, tmp_path)
    s, g = slate(), game("DEN", "SAS")
    assert "No PrizePicks lines yet for DEN at SAS." in text(s)
    assert "No PrizePicks lines for this game yet." in text(g)
    for at in (s, g):
        for bad in ("Pick another book", "book above", "book below"):
            assert bad not in text(at), bad


def test_after_tip_and_for_a_book_the_site_does_not_buy_the_a6_lines_say_so_b25(monkeypatch, tmp_path):
    """Build amendment B25 (build 4's review, L3-6, which pinned A6's "yet"
    after tip and for a book never bought), in layout C. PrizePicks has
    posted neither game here (PHI at DET's payload without it). At 7:10 PM
    PHI at DET has tipped and DEN at SAS has not: the slate's note names DEN
    at SAS alone, PHI at DET's Game page says the odds feed did not load
    PrizePicks lines for it before tip (build amendment B29), and DEN at
    SAS's still says "yet". Once both have tipped (9:30 PM) the slate names
    no game, and both Game pages say before tip. With BUCKETVALUE_ODDS_BOOKS
    leaving BetMGM out, the grid keeps all five columns, BetMGM's cells all
    the sentinel, and BetMGM's lines are said not to be loaded on this site,
    on the slate and on both Game pages, before tip and after, never "yet"."""
    after = "The odds feed did not load PrizePicks lines for this game before tip."   # B29
    payloads = {LF.PHI_EID: PF.without(PF.phi_det(), "prizepicks")}
    feed_world(monkeypatch, tmp_path, at=et(19, 10), payloads=payloads)
    s = text(slate())
    assert "No PrizePicks lines yet for DEN at SAS." in s and "PHI at DET and" not in s
    phi, den = text(game("PHI", "DET")), text(game("DEN", "SAS"))
    assert after in phi and "were posted" not in phi
    assert "lines for this game yet" not in phi
    assert "No PrizePicks lines for this game yet." in den and "before tip" not in den
    feed_world(monkeypatch, tmp_path / "late", at=et(21, 30), payloads=payloads)
    assert "No PrizePicks lines yet" not in text(slate())
    for away, home in (("PHI", "DET"), ("DEN", "SAS")):
        assert after in text(game(away, home)), away
    narrow = {"BUCKETVALUE_ODDS_BOOKS": "draftkings,fanduel,prizepicks,underdog"}
    gone = "BetMGM lines are not loaded on this site."
    for i, at in enumerate((H.MORNING, et(21, 30))):
        feed_world(monkeypatch, tmp_path / f"narrow{i}", at=at, environ=narrow)
        s = slate()
        assert gone in text(s) and "No BetMGM lines" not in text(s), at
        assert GP.grid_heads(html(s)) == CODES
        assert all(r["cells"]["MGM"]["cls"] == "none" for r in GP.grid(s, show_all=True))
        for away, home in (("PHI", "DET"), ("DEN", "SAS")):
            g = text(game(away, home))
            assert gone in g and "No BetMGM lines" not in g, (at, away)
        if at == H.MORNING:          # a book the site buys keeps its own sentence
            assert "No PrizePicks lines yet for DEN at SAS." in text(slate())


# -- Build 6's fix pass: the review of stage S12 (S12-R2 pinned, S12-R3 by B29) -------------
def test_after_a_cold_start_re_buy_the_game_page_says_the_feed_did_not_load_prizepicks_s12_r2_b29(
        monkeypatch, tmp_path):
    """Build 6's review, S12-R2, pinned as built (build amendment B33), and
    S12-R3, settled by build amendment B29. A restart at 10:00 ET in a period
    that was spending (the brake applies; P = 2), the harness's walks at
    10:00, noon, 6:30 PM and 8:30 PM. PrizePicks has not posted DEN at SAS
    at 10:00 and posts it from noon, but the 10:00 re-buy was the game's last
    pull (B21), so the site never asks again. At 8:50 PM the slate says
    PrizePicks has no lines yet for DEN at SAS (R2); at 9:30 PM, after its
    tip, DEN at SAS's Game page says the odds feed did not load PrizePicks
    lines for it before tip, which is true, where B25's sentence said none
    were posted, though PrizePicks had posted them since noon (R3)."""
    pulls = [(et(10, 0), ("prizepicks",)), (et(12, 0), ()), (et(18, 30), ()), (et(20, 30), ())]
    den = LF._with_prizepicks(LF._load("event_odds_den_sas_constructed.json"))
    world = dict(state=None, pulls=pulls, balance=(350, 150), horizon=LF.HORIZON,
                 payloads={LF.DEN_EID: den})
    _w, feed = feed_world(monkeypatch, tmp_path, at=et(20, 50), **world)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [LF.PHI_EID, LF.DEN_EID]
    assert feed.mem["brake_day"] is not None and feed.mem["games"][LF.DEN_EID]["pulls"] == 2
    assert "No PrizePicks lines yet for DEN at SAS." in text(slate())
    feed_world(monkeypatch, tmp_path / "late", at=et(21, 30), **world)
    page = text(game("DEN", "SAS"))
    assert "The odds feed did not load PrizePicks lines for this game before tip." in page
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
    noon, did not play) and the real feed path, in layout C's per-player
    grid. PrizePicks, Underdog and DraftKings each post a points line for him
    (an edited copy of the CONSTRUCTED payload): his row reads "PP: Did not
    play, pick voided. DK, UD: Did not play", "pick voided" once, so no book
    but PrizePicks is said to void; with DraftKings and Underdog only, "Did
    not play"; with PrizePicks only, "Did not play, pick voided". His line is
    shown each time, with the final's actual column."""
    for books, note in ((("prizepicks", "underdog", "draftkings"),
                         "PP: Did not play, pick voided. DK, UD: Did not play"),
                        (("underdog", "draftkings"), "Did not play"),
                        (("prizepicks",), "Did not play, pick voided")):
        payload = PF.phi_det()
        for book in books:
            payload = PF.added(payload, book, "Andre Drummond", 9.5)
        feed_world(monkeypatch, tmp_path / "-".join(books), at=H.EVENING,
                   payloads={LF.PHI_EID: payload})
        H.serve_box(monkeypatch)
        g = game("PHI", "DET")
        words = _lines_text(g, "Andre Drummond")
        assert "Final" in words and note in words, (books, words)
        assert words.count("pick voided") == ("prizepicks" in books), (books, words)
        (row,) = GP.player_grid(g)["Andre Drummond"]
        assert [code for code, c in row["cells"].items() if c["cls"] != "none"] == [
            props.BOOK_CODE[b] for b in props.BOOK_ORDER if b in books]


# -- Build 8's fix pass: the review of stages S14 and S15 (R8-01 to R8-04) --------------------
def test_the_boards_game_and_stat_filters_act_on_the_grids_rows_r8_01(monkeypatch, tmp_path):
    """D3's view controls with the feed on (build 8's review, R8-01), on the
    real slate with the serve fixture's paste (the feed buys points only;
    the paste adds PrizePicks' other stats, so some players have several
    rows). The default view is each player's best-ranked row with "+N more"
    for his others, as production's top_per_player of its grid rows; with
    Show all on, every row, then the game selectbox leaves exactly that
    game's rows and the stat family pills exactly that family's, each in the
    grid's own order: production's grid rows filtered the same way, never a
    list typed into the test. "All games" and "All" bring every row back."""
    w, _f = feed_world(monkeypatch, tmp_path)
    token = paste(monkeypatch, w)
    s = slate(board=token)
    prod = production(token)
    table = props.grid_rows(prod.shown, prod.hidden)
    grid = table.to_dict(orient="records")

    def keys(recs):
        return [(r["player"], props.STAT_LABEL[r["stat"]]) for r in recs]

    def drawn(at):
        return [(r["player"], r["stat"]) for r in GP.grid(at)]
    top, more = props_ui.top_per_player(table, props_ui.TOP_N)
    assert drawn(s) == keys(top.to_dict(orient="records")) and len(top) < len(grid)
    assert [r["more"] for r in GP.grid(s)] == [more.get(int(p), 0) for p in top["person_id"]]
    assert any(more.values())
    every = keys(grid)
    assert [(r["player"], r["stat"]) for r in GP.grid(s, show_all=True)] == every
    late = keys(r for r in grid if str(r["game_id"]) == LATE)
    rebs = keys(r for r in grid if props.STAT_FAMILY.get(r["stat"]) == "Rebounds")
    assert 0 < len(late) < len(every) and 0 < len(rebs) < len(every)
    sel = s.selectbox(key="bv_board_game")
    assert list(sel.options) == ["All games", "PHI at DET", "DEN at SAS"]
    sel.set_value("DEN at SAS").run()
    assert drawn(s) == late
    assert {r["matchup"] for r in GP.grid(s)} <= {"DEN vs SAS", "SAS vs DEN"}
    s.selectbox(key="bv_board_game").set_value("All games").run()
    assert drawn(s) == every
    (fam,) = [b for b in s.button_group if b.key == "bv_board_fam"]
    fam.set_value("Rebounds").run()
    assert drawn(s) == rebs and {stat for _p, stat in rebs} == {"Rebounds"}
    (fam,) = [b for b in s.button_group if b.key == "bv_board_fam"]
    fam.set_value("All").run()
    assert drawn(s) == every and not s.exception


def _moved_one_sided(payload: dict, book: str, player: str, point: float) -> dict:
    """A copy of `payload` with `player`'s one-sided points line at `book`
    (a single outcome) moved to `point`, still offering that side alone."""
    p = copy.deepcopy(payload)
    found = [oc for bm in p["bookmakers"] if bm["key"] == book for mk in bm["markets"]
             if mk["key"] == "player_points" for oc in mk["outcomes"]
             if oc["description"] == player]
    assert len(found) == 1, (book, player, len(found))
    found[0]["point"] = point
    return p


def test_after_the_final_a_hidden_line_he_landed_on_is_named_in_his_grid_r8_02(monkeypatch, tmp_path):
    """Amendment C7 (build 8's review, R8-02): the Game page passes each
    player's hidden lines through store.line_actuals too, so an exact
    landing on a hidden line is named. Jalen Duren scores 14 in the served
    PHI at DET final (serve_box). His BetMGM line, Under only, is moved to
    14 (an edited copy of the CONSTRUCTED payload), so it stays hidden while
    the model leans Over, and his grid draws it as its number alone beside
    his shown 13.5s at DraftKings, FanDuel and PrizePicks. None of those is
    14, so his Points row's note, "14: landed on the line (MGM)", can come
    only from the hidden line's own settle."""
    payload = _moved_one_sided(PF.phi_det(), "betmgm", "Jalen Duren", 14.0)
    feed_world(monkeypatch, tmp_path, at=H.EVENING, payloads={LF.PHI_EID: payload})
    H.serve_box(monkeypatch)
    g = game("PHI", "DET")
    (pts,) = [r for r in GP.player_grid(g)["Jalen Duren"] if r["stat"] == "Points"]
    mgm = pts["cells"]["MGM"]
    assert (mgm["cls"], mgm["number"], mgm["sr"]) == ("plain", "14", " leans Over, not offered")
    assert ">Under only</span>" in mgm["badges"]
    for code in ("DK", "FD", "PP"):
        assert (pts["cells"][code]["number"], pts["cells"][code]["cls"] in ("over", "under")) == (
            "13.5", True), code
    assert pts["actual"] == "14" and "Final" in _lines_text(g, "Jalen Duren")
    assert GP.player_notes(g)["Jalen Duren"] == ["14: landed on the line (MGM)"]


def test_the_game_pages_note_says_a_hidden_line_is_drawn_only_when_one_is_r8_03(monkeypatch, tmp_path):
    """Amendment C8 (build 8's review, R8-03): the Game page's "Lines on this
    game" note carries HIDDEN_DRAWN only when one of the game's hidden lines
    is drawn, never because the game merely has hidden lines (in the page
    world both games draw one, so the two never differed). Without BetMGM's
    PHI at DET lines (Jalen Duren's and VJ Edgecombe's, that game's drawn
    hidden lines), PHI at DET's hidden lines (Daniss Jenkins' and Quentin
    Grimes' points, with no shown line at any book) are all undrawn: its
    note counts them and has no HIDDEN_DRAWN. DEN at SAS's, whose drawn
    hidden lines are untouched, keeps it."""
    feed_world(monkeypatch, tmp_path, payloads={LF.PHI_EID: PF.without(PF.phi_det(), "betmgm")})
    prod = production()
    hidden = {}
    for gid, drawn_here in ((EARLY, False), (LATE, True)):
        sh = prod.shown[prod.shown["game_id"].astype(str) == gid]
        hi = prod.hidden[prod.hidden["game_id"].astype(str) == gid]
        d, u = props.split_hidden(sh, hi)
        assert len(hi) == len(d) + len(u) > 0 and bool(len(d)) is drawn_here, gid
        hidden[gid] = len(hi)
    for (away, home), gid in GAMES:
        g = game(away, home)
        (note,) = [PS.visible(m.value) for m in g.markdown
                   if PS.visible(m.value).startswith("Lines on this game:")]
        assert f"({hidden[gid]} hidden / " in note, (away, note)
        assert (props_ui.HIDDEN_DRAWN in note) is (gid == LATE), (away, note)
        assert (props_ui.HIDDEN_DRAWN in text(g)) is (gid == LATE), away


RUNBOOK_BOOKS = ("The board keeps all five columns: a book left out shows the dash on every row "
                 "(PrizePicks' column still shows a visitor's pasted lines), and while it has no "
                 "line for a game another book has, the note reads `<Book> lines are not loaded on "
                 "this site.`")


def paste_teams(monkeypatch, w, teams, at=H.MORNING) -> str:
    """The part of the serve fixture's board text whose players play for one
    of `teams`, through the real paste path (props_ui.ingest), pasted at
    `at` (props._wall pinned, as paste() pins it)."""
    monkeypatch.setattr(props, "_wall", lambda: at.astimezone(UTC))
    head, *body = w.board_text.splitlines()
    blocks = [body[i:i + 8] for i in range(0, len(body), 8)]
    assert len(body) % 8 == 0 and all(b[-2:] == ["Less", "More"] for b in blocks)
    keep = [b for b in blocks if b[1].split(" - ")[0] in teams]
    token, report = props_ui.ingest("\n".join([head, *(x for b in keep for x in b)]), None,
                                    w.roster, w.sched, now=at, dates=[H.DATE])
    assert token and 0 < report["comparable"] == len(keep) < len(blocks), report
    return token


def test_the_runbooks_books_setting_row_is_what_the_pages_do_r8_04(monkeypatch, tmp_path):
    """docs/runbook.md's BUCKETVALUE_ODDS_BOOKS row (build 8's review, R8-04:
    it said a book left out "shows the dash everywhere", false for
    PrizePicks once a visitor has pasted), checked against the pages with
    PrizePicks left out of the setting. With no paste: five columns, every
    PP cell the dash, and the row's not-loaded sentence on the slate and on
    both Game pages. With PHI at DET's part of the serve fixture's board
    pasted: PP holds those pasted lines, each titled with B28's word, on PHI
    at DET's rows only, and the sentence stays on the slate and on DEN at
    SAS's page (a game other books have and PrizePicks has no line for),
    not on PHI at DET's. With the whole board pasted no game lacks a
    PrizePicks line, and the slate says no such thing."""
    doc = " ".join((HERE.parent / "docs" / "runbook.md").read_text().split())
    row = re.search(r"\| `BUCKETVALUE_ODDS_BOOKS` \| [^|]* \| [^|]* \| ([^|]*) \|", doc)
    assert row, "the runbook's settings table has no BUCKETVALUE_ODDS_BOOKS row"
    assert row.group(1).endswith(RUNBOOK_BOOKS), row.group(1)
    words = re.search(r"the note reads `<Book> ([^`]*)`$", row.group(1)).group(1)
    gone = f"{props.BOOK_TITLE['prizepicks']} {words}"
    assert gone == "PrizePicks lines are not loaded on this site."
    narrow = {"BUCKETVALUE_ODDS_BOOKS": "draftkings,fanduel,betmgm,underdog"}
    w, _f = feed_world(monkeypatch, tmp_path, environ=narrow)
    s = slate()
    assert GP.grid_heads(html(s)) == CODES and gone in text(s)
    rows = GP.grid(s, show_all=True)
    assert rows and all(r["cells"]["PP"]["cls"] == "none" for r in rows)
    for away, home in (("PHI", "DET"), ("DEN", "SAS")):
        assert gone in text(game(away, home)), away
    token = paste_teams(monkeypatch, w, ("PHI", "DET"))
    s = slate(board=token)
    pp = [r for r in GP.grid(s, show_all=True) if r["cells"]["PP"]["cls"] != "none"]
    assert pp and {r["matchup"] for r in pp} <= {"PHI vs DET", "DET vs PHI"}
    for r in pp:
        assert re.match(r"PrizePicks [\d.]+ \((pasted|typed|Demon|Goblin)\): ",
                        r["cells"]["PP"]["title"]), r["cells"]["PP"]
    assert gone in text(s)
    assert gone not in text(game("PHI", "DET", board=token))
    assert gone in text(game("DEN", "SAS", board=token))
    s = slate(board=paste(monkeypatch, w))
    assert gone not in text(s)
    pasted = {frozenset(r["matchup"].split(" vs ")) for r in GP.grid(s, show_all=True)
              if r["cells"]["PP"]["cls"] != "none"}
    assert pasted == {frozenset(teams) for teams, _gid in GAMES}


# -- Round P1 (2026-11-12): a past date read from memory, every game final -----------------------
NEXT_NIGHT = et(1, 4, day=13)          # 01:04 ET on Mar 13: the page's ET date is the next day
FINAL_HEAD = '<th class="num" scope="col">Final</th>'


def test_a_past_date_read_from_memory_with_every_game_final_shows_its_lines_p1(
        monkeypatch, tmp_path):
    """The view that went empty on the live site after a code push (round P1,
    2026-11-12: the evening's slate after midnight ET), which no page test
    drew before. At 01:04 ET on Mar 13 the page's ET date is the next day, so
    store.feed_start gives "memory" for 2026-03-12 and store.feed_result reads
    live.prop_lines_memory: no kick, no refresher, no wait. The schedule
    calls both games final, with their scores as cdn sends them: PHI at
    DET's real final score (the box payload's) and a CONSTRUCTED one for DEN
    at SAS, whose result is not in the fixtures.

    The slate's cards count the feed's shown lines, production's for the
    same inputs, under the feed's stamp with its date. Each Game page draws
    its players' grids inside bv_game_lines, its counts the card's, under the
    pre-tip stamp; PHI at DET's grids, with its real final box (serve_box),
    carry the Final column with every row's actual, and DEN at SAS's (no box
    served) carry none."""
    w, feed = feed_world(monkeypatch, tmp_path, at=NEXT_NIGHT)
    box = H.box_payload()["game"]
    scores = {EARLY: (box["awayTeam"]["score"], box["homeTeam"]["score"]),
              LATE: (112, 118)}                                      # CONSTRUCTED
    final = w.sched.assign(status="final", status_text="Final",
                           away_score=[scores[g][0] for g in w.sched["game_id"]],
                           home_score=[scores[g][1] for g in w.sched["game_id"]])
    monkeypatch.setattr(live, "schedule", lambda: live.Live(final, 0.0))
    H.serve_box(monkeypatch)
    reads, other = [], []
    monkeypatch.setattr(live, "prop_lines_memory",
                        lambda date, *a, **k: reads.append(date) or feed.view(date))
    monkeypatch.setattr(live, "feed_view",
                        lambda *a, **k: other.append("feed_view") or feed.view(*a, **k))
    monkeypatch.setattr(live, "feed_kick", lambda *a, **k: other.append("feed_kick"))
    monkeypatch.setattr(live, "ensure_refresher", lambda *a, **k: other.append("ensure_refresher"))
    at = slate()
    assert reads == [H.DATE] and other == []
    cards = PS.cards(at)
    for (away, home), gid in GAMES:
        card = cards[f"{away} at {home}"]
        assert '<span class="bv-slate-tag final">Final</span>' in card, away
        shown = re.search(r'<div class="bv-slate-final"><span[^>]*>(\d+)</span> - '
                          r'<span[^>]*>(\d+)</span></div>', card)
        assert shown and shown.groups() == tuple(str(s) for s in scores[gid]), away
    assert "Lines as of 12:20 PM ET, Mar 12." in text(at) and GP.grid(at)
    prod = production()
    counts = pill_counts(at)
    assert counts == {gid: props_ui.line_counts_by_game(prod.shown).get(gid, 0) for _t, gid in GAMES}
    assert all(counts.values()) and sum(counts.values()) == len(prod.shown)
    for (away, home), gid in GAMES:
        reads.clear()
        other.clear()
        g = game(away, home)
        assert reads == [H.DATE] and other == [], away
        grids = GP.player_grid(g)
        inside = GP.keyed_markdown(g, "bv_game_lines")
        mine = [m.value for m in g.markdown if GP.PLAYER_LABEL + '"' in m.value]
        assert grids and inside is not None and mine and all(m in inside for m in mine), away
        words = text(g)
        noted = re.search(r"Lines on this game: (\d+) of \d+ lines shown \([^)]*\)\. Pre-tip lines, "
                          r"as of 12:20 PM ET, Mar 12\.", words)
        opened = sum(int(m.group(1)) for m in (re.match(r".+: (\d+) lines?$", str(e.label))
                                               for e in g.expander) if m)
        assert noted and int(noted.group(1)) == opened == counts[gid], (away, words)
        actuals = [r["actual"] for rows in grids.values() for r in rows]
        if gid == EARLY:
            assert sum(m.count(FINAL_HEAD) for m in mine) == len(mine), away
            assert None not in actuals and SENTINEL not in actuals, away
        else:
            assert not any(FINAL_HEAD in m for m in mine) and set(actuals) == {None}, away
