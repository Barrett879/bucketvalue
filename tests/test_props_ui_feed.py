"""bucketlib.props_ui with the licensed odds feed (design 2.6 and 14 step 16;
layout C: amendments C5 to C13; build amendments N1 to N3): BoardView's
fields, date_board's feed path, the grid (one column per book, each cell the
book's line with the model's lean at it, the per-player grid, the legend and
the CSS), the side words, the feed's notices (feed_notice_html and the
books-missing notes), Biggest discrepancies naming the books, and the
per-source wording, as HTML unit tests (AppTest where a widget is involved).

With the feed off every function must give exactly the HTML it gave before
the feed: tests/fixtures/props_ui/feed_off_golden.json holds what the pre-S8
props_ui rendered for tests/ui_rows.py's rows (captured from build 2's tree
before props_ui.py was edited), and it is compared byte for byte.

The rows are hand-built (tests/ui_rows.py) or, end to end, the serve
fixture's 2026-03-12 slate with the CONSTRUCTED vendor fixtures through the
real parser (tests/test_props_feed.py's helpers) and the real paste path.
The grid is read back with tests/grid_parse.py, which checks every cell
against its own column. Every time is a fixed past instant passed in, never
a clock: a guard test makes props_ui's own clock raise while the feed's text
is built.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from html import unescape
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from bucketlib import dist as D
from bucketlib import props, props_ui, simprov, theme
from bucketlib.util import ET, SENTINEL, esc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import grid_parse as GP  # noqa: E402  (layout C's grid read back, column by column)
import test_props_feed as PF  # noqa: E402  (the feed frame through the real parser)
import ui_rows as U  # noqa: E402

GOLDEN = json.loads((HERE / "fixtures" / "props_ui" / "feed_off_golden.json").read_text())
DAY = dt.date(2026, 3, 12)
EARLY, LATE = U.EARLY, U.LATE
NINE, NOON, LAST_NIGHT = U.NINE, U.NOON, U.LAST_NIGHT
LABELS = {EARLY: "PHI at DET", LATE: "DEN at SAS"}       # the schedule's own orientation (N2)
NAN = float("nan")
SPORTSBOOKS = ("draftkings", "fanduel", "betmgm")
UP, DOWN = "&#8593;", "&#8595;"                          # the arrows' entities, as drawn


def visible(markup: str) -> str:
    """What a reader sees: tags dropped with their attributes, entities decoded."""
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def dash_ok(text: str) -> bool:
    """tests/test_pages_smoke.py's rule: an em dash only as a whole text node."""
    nodes = re.split(r"<[^>]*>", text)
    return (all(n.strip() == SENTINEL for n in nodes if SENTINEL in n)
            and text.count(SENTINEL) == sum(n.count(SENTINEL) for n in nodes))


# -- Hand-built feed rows ------------------------------------------------------------------
def feed_row(book: str, line: float, at=NOON, **kw) -> dict:
    """One compare() row of the odds feed: a book's own line for Ann Able's
    points on PHI at DET, fetched at `at`. Sportsbook rows carry prices,
    which nothing may show (amendment A1)."""
    dfs = book in props.DFS_BOOKS
    base = dict(book=book, book_title=props.BOOK_TITLE[book], origin="feed", fetched_at=at,
                last_update=at, line=line, price_over=NAN if dfs else -115.0,
                price_under=NAN if dfs else -105.0)
    base.update(kw)
    return U.row(**base)


def ref(book="draftkings", line=25.5, **kw) -> pd.DataFrame:
    return U.frame([feed_row(book, line, **kw)])


def lines(*specs) -> pd.DataFrame:
    """Ann Able's points at several books: each spec (book, line) or (book,
    line, {feed_row's keywords}), every one a feed_row."""
    return U.frame([feed_row(s[0], s[1], **(s[2] if len(s) > 2 else {})) for s in specs])


def grid(shown, hidden=None) -> pd.DataFrame:
    """The grid's rows (layout C, amendment C4) of shown and hidden lines."""
    return props.grid_rows(shown, hidden)


def facts(**kw) -> dict:
    out = {"state": "ok", "asof": NINE, "loading": 0, "generation": 2, "lines": 40,
           "per_game": {EARLY: {"asof": NOON, "books": {}, "pretip": False},
                        LATE: {"asof": NINE, "books": {}, "pretip": False}},
           "report": {}, "today": DAY}
    out.update(kw)
    return out


def drawn(shown, hidden=None, feed=None) -> list[dict]:
    """The slate grid's rows as a reader sees them (tests/grid_parse.py)."""
    return GP.rows(props_ui.ledger_html(grid(shown, hidden), feed=feed if feed is not None
                                        else facts()))


def cells_of(html: str) -> list[str]:
    """Every <td class="bk"> of a grid, as drawn."""
    return re.findall(r'<td class="bk">.*?</td>', html, re.S)


# -- The feed off: today's HTML, byte for byte -------------------------------------------
def test_the_feed_off_gives_todays_html_byte_for_byte():
    """Every renderer, called as the pages called it before the feed, gives
    the HTML the pre-feed props_ui gave (the golden file), PROPS_CSS and the
    no-lines prompts included."""
    now = U.feed_off_outputs(props_ui)
    assert set(now) == set(GOLDEN["functions"])
    for name, html in GOLDEN["functions"].items():
        assert now[name] == html, name


def test_no_feed_passed_explicitly_is_the_same_html():
    rows = U.pasted_rows()
    shown, hidden, thin, void = U.split(rows)
    g = GOLDEN["functions"]
    assert props_ui.ledger_html(rows, {1001: 2}, None) == g["ledger"]
    assert props_ui.render_player_lines(rows, None, None) == g["player_lines"]
    assert "\n".join(props_ui.notes_html(U.meta_full(), shown, hidden, thin, void, None)) == g["notes"]
    assert props_ui.discrepancies_window(U.gap_rows(), "March 12", None) == g["discrepancies_window"]
    assert props_ui.discrepancies_html(U.gap_rows(), "March 12", None) == g["discrepancies"]
    assert props_ui.none_matched_html(3, "March 12", False) == g["none_matched"]
    # Without a feed no cell, no header code, no legend, no books CSS, no grid.
    for html in (g["ledger"], g["player_lines"]):
        assert 'class="ob"' not in html and "has-books" not in html and "Other books" not in html
        assert "bv-grid" not in html and "bv-cell" not in html


BOARD_SCRIPT = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(HERE)!r})
import ui_rows as U
from bucketlib import props_ui
rows = U.pasted_rows()
shown, hidden, thin, void = U.split(rows)
props_ui.render_board(shown, hidden, thin, void, "March 12", meta=U.meta_full(),
                      game_labels={{U.EARLY: "PHI at DET", U.LATE: "DEN at SAS"}})
props_ui.render_discrepancies(U.gap_rows(), "March 12")
props_ui.render_player_lines_block(rows)
"""


def _page(script: str, **query) -> AppTest:
    at = AppTest.from_string(script, default_timeout=60)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_the_feed_off_board_page_is_todays_byte_for_byte():
    """render_board, render_discrepancies and render_player_lines_block on a
    Streamlit page: every markdown, success and caption element as before."""
    at = _page(BOARD_SCRIPT)
    got = {kind: [str(e.value) for e in getattr(at, kind)] for kind in GOLDEN["page"]}
    assert got == GOLDEN["page"]
    assert props_ui.BOOKS_CSS.strip() not in got["markdown"]
    assert props_ui.PROPS_CSS.strip() in got["markdown"]


@pytest.fixture(scope="module")
def world():
    return H.load()


@pytest.fixture(scope="module")
def params():
    return D.SimParams()


@pytest.fixture()
def tokens():
    made: list = []
    yield made
    for token in made:
        props.clear_board(token)


def _paste(world, tokens, at) -> str:
    """The serve fixture's board text through the real paste path, kept on a
    token board pasted at `at`."""
    token = props.add_to_board(None, PF.paste_lines(world, at.astimezone(ET)), pasted_utc=at)
    tokens.append(token)
    return token


def test_date_board_without_a_feed_is_todays_function(world, params, tokens):
    token = _paste(world, tokens, U.NINE)
    view = props_ui.date_board(token, DAY, world.proj, params, world.roster)
    want = props_ui.board_view(token, DAY, simprov.provider_for(world.proj, params), world.proj,
                               world.roster, props_ui.void_ids(world.proj))
    for a, b in ((view.shown, want.shown), (view.hidden, want.hidden), (view.thin, want.thin),
                 (view.void, want.void), (view.lines, want.lines)):
        pd.testing.assert_frame_equal(a, b)
    assert view.meta.keys() == want.meta.keys() and view.meta["posted"] == want.meta["posted"] == 11
    assert view.feed is None and view.books_by_game == {} and view.missing == {}
    assert (view.paste_won, view.feed_won, view.paste_times) == (0, 0, {})


# -- Layout C: the grid (amendments C6, C7, C13) ---------------------------------------------
def test_the_grid_has_one_column_per_book_in_the_mockups_order():
    html = props_ui.ledger_html(grid(lines(("draftkings", 24.5))), feed=facts())
    assert html.startswith('<div class="bv-grid-wrap" tabindex="0" role="region" '
                           'aria-label="Lines by book"><table class="bv-grid"><thead>')
    assert html.endswith("</tbody></table></div>")
    assert GP.heads(html) == [("DK", "DraftKings"), ("FD", "FanDuel"), ("MGM", "BetMGM"),
                              ("PP", "PrizePicks"), ("UD", "Underdog")]
    assert GP.heads(html) == [(props.BOOK_CODE[b], props.BOOK_TITLE[b]) for b in props.BOOK_ORDER]
    assert ('<thead><tr><th class="pl" scope="col">Player</th><th class="st" scope="col">Stat</th>'
            '<th class="num" scope="col">Model</th><th class="bk" scope="col">'
            '<abbr title="DraftKings">DK</abbr></th>') in html
    for word in (">Gap<", ">Lean<", "Other books", "bv-ledger", "\n"):
        assert word not in html, word
    # Plain shown lines handed to the ledger go through grid_rows: the same grid.
    assert props_ui.ledger_html(lines(("draftkings", 24.5)), feed=facts()) == html
    (row,) = GP.rows(html)
    assert (row["player"], row["matchup"], row["stat"], row["model"]) == (
        "Ann Able", "DET vs PHI", "Points", "26.3")


MOCKUP = (   # the mockup's three rows: (player, model, ((book, line, lean) in DK FD MGM PP UD))
    ("Player A", 26.1, (("draftkings", 24.5, "More"), ("fanduel", 25.5, "More"),
                        ("betmgm", 25.5, "More"), ("prizepicks", 24.5, "More"),
                        ("underdog", 24.5, "More"))),
    ("Player B", 17.2, (("draftkings", 17.5, "Less"), ("fanduel", 16.5, "More"),
                        ("betmgm", 17.5, "Less"), ("prizepicks", 18.5, "Less"),
                        ("underdog", 17.5, "Less"))),
    ("Player C", 21.0, (("draftkings", 21.5, "Less"), ("fanduel", 21.5, "Less"),
                        ("betmgm", 20.5, "More"), ("prizepicks", 20.5, "More"),
                        ("underdog", 21.5, "Less"))),
)


def test_the_mockups_three_rows_cell_by_cell():
    """C1R-06: the mockup's rows, every line shown with its own lean. Rows B
    and C lean both ways, so a cell that took its lean from the row's lead
    line, or a body in another order than the header's, fails here."""
    rows = []
    for i, (player, model, books) in enumerate(MOCKUP):
        for book, line, lean in books:
            rows.append(feed_row(book, line, person_id=5001 + i, player=player, model=model,
                                 gap=model - line, lean=lean, confidence=0.3 - 0.1 * i))
    got = drawn(U.frame(rows))
    assert [(r["player"], r["model"]) for r in got] == [("Player A", "26.1"), ("Player B", "17.2"),
                                                        ("Player C", "21.0")]
    for row, (_player, _model, books) in zip(got, MOCKUP):
        assert list(row["cells"]) == ["DK", "FD", "MGM", "PP", "UD"]
        for (book, line, lean), (code, c) in zip(books, row["cells"].items()):
            assert code == props.BOOK_CODE[book]
            word = props.side_words(book)[0 if lean == "More" else 1]
            assert (c["cls"], c["arrow"], c["sr"], c["number"]) == (
                "over" if lean == "More" else "under", UP if lean == "More" else DOWN,
                f" leans {word}", f"{line:g}"), (row["player"], code)
            assert c["title"] == f"{props.BOOK_TITLE[book]} {line:g}: the model leans {word}."
    assert got[1]["cells"]["DK"]["title"] == "DraftKings 17.5: the model leans Under."
    assert got[1]["cells"]["FD"]["title"] == "FanDuel 16.5: the model leans Over."
    assert got[2]["cells"]["PP"]["sr"] == " leans More" and got[2]["cells"]["UD"]["sr"] == " leans Less"


def test_a_cell_is_the_line_and_the_models_lean_at_it():
    """Amendment C6's exact markup: shown More and Less with the arrow and
    its text, a hidden Even line and a hidden line whose model side is not
    offered as the number alone, and the sentinel for no line; no price."""
    shown = lines(("draftkings", 24.5), ("fanduel", 26.5, {"lean": "Less", "gap": -0.2}))
    hidden = lines(("betmgm", 25.5, {"lean": "Even", "gap": 0.8}),
                   ("prizepicks", 23.5, {"sides": "less", "gap": 2.8, "origin": "feed"}))
    html = props_ui.ledger_html(grid(shown, hidden), feed=facts())
    assert cells_of(html) == [
        '<td class="bk"><span class="bv-cell over" title="DraftKings 24.5: the model leans Over.">'
        '24.5<span class="ar" aria-hidden="true">&#8593;</span><span class="bv-sr"> leans Over'
        "</span></span></td>",
        '<td class="bk"><span class="bv-cell under" title="FanDuel 26.5: the model leans Under.">'
        '26.5<span class="ar" aria-hidden="true">&#8595;</span><span class="bv-sr"> leans Under'
        "</span></span></td>",
        '<td class="bk"><span class="bv-cell plain" title="BetMGM 25.5: no lean at this line.">'
        '25.5<span class="bv-sr"> no lean</span></span></td>',
        '<td class="bk"><span class="bv-cell plain" title="PrizePicks 23.5: the model leans More, '
        'a side this line does not offer.">23.5<span class="bv-sr"> leans More, not offered</span>'
        '</span><span class="bv-pp-tag" title="This line offers Less only">Less only</span></td>',
        f'<td class="bk"><span class="bv-cell none">{SENTINEL}<span class="bv-sr"> no Underdog '
        "line</span></span></td>"]
    # Amendment C6's own example: BetMGM 13.5 Under only while the model leans More.
    duren = U.frame([feed_row("draftkings", 13.5, person_id=1101, player="Jay Dee"),
                     feed_row("betmgm", 13.5, person_id=1101, player="Jay Dee", sides="less")])
    html = props_ui.ledger_html(grid(duren.iloc[[0]], duren.iloc[[1]]), feed=facts())
    assert cells_of(html)[2] == (
        '<td class="bk"><span class="bv-cell plain" title="BetMGM 13.5: the model leans Over, a side '
        'this line does not offer.">13.5<span class="bv-sr"> leans Over, not offered</span></span>'
        '<span class="bv-pp-tag" title="This line offers Under only">Under only</span></td>')
    # The arrows only on over and under cells; no price anywhere.
    page = props_ui.ledger_html(grid(shown, hidden), feed=facts())
    for c in cells_of(page):
        assert ('class="ar"' in c) == ('bv-cell over' in c or 'bv-cell under' in c), c
    assert page.count(UP) == 1 and page.count(DOWN) == 1
    for bad in ("-115", "-105", "price", "&uarr;", "&darr;", chr(0x2191), chr(0x2193)):
        assert bad not in page, bad


def test_a_book_with_no_line_gives_the_whole_node_sentinel():
    """The cell holds the sentinel as a whole text node and its visually
    hidden "no <Book> line", its own column's title, and nothing else: no
    number, arrow, badge or title (C1R-11)."""
    for book in props.BOOK_ORDER:
        other = "fanduel" if book == "draftkings" else "draftkings"
        html = props_ui.ledger_html(grid(lines((other, 24.5))), feed=facts())
        (row,) = GP.rows(html)
        c = row["cells"][props.BOOK_CODE[book]]
        assert (c["cls"], c["number"], c["title"], c["arrow"], c["badges"]) == (
            "none", SENTINEL, None, None, ""), book
        assert c["sr"] == f" no {props.BOOK_TITLE[book]} line"
        assert dash_ok(html) and html.count(SENTINEL) == 4
    sentinel = (f'<td class="bk"><span class="bv-cell none">{SENTINEL}<span class="bv-sr"> no '
                "BetMGM line</span></span></td>")
    for rec in (None, {}, {"line": NAN, "lean": "More", "shown": True},
                {"line": None, "lean": "Less", "shown": True}, "junk"):
        assert props_ui._grid_cell("betmgm", rec, {}, DAY) == sentinel, rec
    assert dash_ok(sentinel) and visible(sentinel) == f"{SENTINEL} no BetMGM line"


def test_a_cells_title_names_our_own_book_title_never_the_vendors():
    """Book names come from props.BOOK_TITLE and BOOK_CODE: a line's vendor
    title never shows, and a book outside the five is never a column nor a
    cell, shown or hidden."""
    shown = U.frame([feed_row("underdog", 24.5, book_title="Underdog Fantasy <b>"),
                     U.row(book="evilbook", book_title="<script>x</script>", origin="feed",
                           fetched_at=NOON, line=25.5, confidence=0.5)])
    hidden = U.frame([U.row(book="sixthbook", book_title="Sixth <i>", origin="feed",
                            fetched_at=NOON, line=26.5, lean="Even")])
    html = props_ui.ledger_html(grid(shown, hidden), feed=facts())
    (row,) = GP.rows(html)
    assert row["cells"]["UD"]["title"] == "Underdog 24.5: the model leans More."
    assert [c["cls"] for c in row["cells"].values()] == ["none", "none", "none", "none", "over"]
    assert GP.grid_heads(html) == ["DK", "FD", "MGM", "PP", "UD"]
    for bad in ("Underdog Fantasy", "evilbook", "script", "sixthbook", "Sixth", "&lt;b&gt;"):
        assert bad not in html, bad


def test_a_cells_title_says_its_own_time_when_over_a_minute_older_m23():
    """m23 and amendment C10: a feed line over LINE_TIME_AGE (one minute)
    older than its game's latest pull says its own time in its title only.
    The latest pull is the page's per_game "asof", else the newest fetch
    among the rows' cells. A pasted line never shows a time."""
    def fd_title(at, feed):
        rows = drawn(lines(("draftkings", 25.5), ("fanduel", 25.5, {"fetched_at": at})), feed=feed)
        return rows[0]["cells"]["FD"]["title"]
    plain = "FanDuel 25.5: the model leans Over."
    minute = pd.Timedelta(minutes=1)
    assert props_ui.LINE_TIME_AGE == minute
    assert fd_title(NOON - minute, facts()) == plain
    assert fd_title(NOON - minute - pd.Timedelta(seconds=1), facts()) == plain + " Line as of 11:58 AM ET."
    assert fd_title(NINE, facts()) == plain + " Line as of 9:02 AM ET."
    assert fd_title(LAST_NIGHT, facts()) == plain + " Line as of 9:41 PM ET, Mar 11."
    # No per_game facts: DraftKings' noon fetch is the latest.
    assert fd_title(NINE, {"today": DAY}) == plain + " Line as of 9:02 AM ET."
    assert fd_title(NINE, {"today": DAY, "per_game": {EARLY: {"asof": NINE}}}) == plain
    assert fd_title(None, facts()) == plain                          # no fetch time: no time
    pp = drawn(lines(("draftkings", 25.5), ("prizepicks", 24.5, {"origin": "pasted",
                                                                  "fetched_at": NINE})))
    assert pp[0]["cells"]["PP"]["title"] == "PrizePicks 24.5 (pasted): the model leans More."
    # The time is in the title only: never on screen.
    html = props_ui.ledger_html(grid(lines(("draftkings", 25.5), ("fanduel", 25.5,
                                                                  {"fetched_at": NINE}))),
                                feed=facts())
    assert "9:02" not in visible(html) and "9:02 AM" in html


def test_a_pasted_prizepicks_line_keeps_todays_badges_in_its_cell():
    """D5 and build amendment B28: a PrizePicks line from the paste keeps its
    badges in its own cell (the typed pick, Demon, Goblin), its title carries
    B28's word after the number, and a plain pasted line and a feed line
    carry no badge. The void-risk tag only on a shown PrizePicks More."""
    def pp(shown=True, **kw):
        line = U.row(book="prizepicks", origin="pasted", line=24.5, **kw)
        dk = feed_row("draftkings", 25.5)
        rows = drawn(U.frame([dk, line])) if shown else drawn(U.frame([dk]), U.frame([line]))
        return rows[0]["cells"]["PP"]
    typed = pp(pick="more")
    assert typed["title"] == "PrizePicks 24.5 (typed): the model leans More."
    assert typed["badges"] == ('<span class="bv-pp-tag pick" title="The pick in the pasted list: '
                               'More. The model leans More.">Pick: More</span>')
    demon = pp(odds_type="demon")
    assert demon["title"].startswith("PrizePicks 24.5 (Demon): ") and ">Demon</span>" in demon["badges"]
    goblin = pp(odds_type="goblin")
    assert goblin["title"].startswith("PrizePicks 24.5 (Goblin): ") and ">Goblin</span>" in goblin["badges"]
    plain = pp()
    assert (plain["title"], plain["badges"]) == ("PrizePicks 24.5 (pasted): the model leans More.", "")
    fed = drawn(lines(("draftkings", 25.5), ("prizepicks", 24.5)))[0]["cells"]["PP"]
    assert (fed["title"], fed["badges"]) == ("PrizePicks 24.5: the model leans More.", "")
    # Amendment C6's example: a plain pasted PrizePicks 24.5, shown, Less.
    less = U.frame([feed_row("draftkings", 25.5), U.row(book="prizepicks", origin="pasted",
                                                        line=24.5, lean="Less", gap=-0.6)])
    html = props_ui.ledger_html(grid(less), feed=facts())
    assert cells_of(html)[3] == (
        '<td class="bk"><span class="bv-cell under" title="PrizePicks 24.5 (pasted): the model leans '
        'Less.">24.5<span class="ar" aria-hidden="true">&#8595;</span><span class="bv-sr"> leans '
        "Less</span></span></td>")
    # Reboot's void-risk tag: a shown PrizePicks More only.
    bench = dict(void_note=True, mu_min=15.0)
    assert ">void risk</span>" in pp(**bench)["badges"]
    assert "void risk" not in pp(shown=False, sides="less", **bench)["badges"]
    ud = drawn(lines(("draftkings", 25.5), ("underdog", 24.5, bench)))[0]["cells"]["UD"]
    assert "void risk" not in ud["badges"]


def test_a_hidden_pick_beside_a_shown_line_is_drawn_and_never_called_not_shown():
    """C1R-05, the unit half. A typed pick on an Even PrizePicks line, in a
    row with a shown DraftKings line, is drawn (its cell plain, "no lean",
    with its "Pick: More" tag), so the Game page's "Not shown here" must not
    name it: props.split_hidden puts it in the drawn part. The same pick on
    a key with no shown line is never drawn, and is named."""
    shown = lines(("draftkings", 25.5))
    pick = U.frame([U.row(book="prizepicks", origin="pasted", line=24.5, lean="Even",
                          pick="more", confidence=0.001)])
    c = drawn(shown, pick)[0]["cells"]["PP"]
    assert (c["cls"], c["sr"], c["arrow"]) == ("plain", " no lean", None)
    assert ">Pick: More</span>" in c["badges"]
    on, off = props.split_hidden(shown, pick)
    assert len(on) == 1 and off.empty
    assert props_ui.hidden_picks_text(off, game_page=True) == ""
    alone = U.frame([U.row(book="prizepicks", origin="pasted", line=24.5, lean="Even",
                           pick="more", person_id=1009, player="Lone Pick")])
    on, off = props.split_hidden(shown, alone)
    assert on.empty and len(off) == 1
    assert props_ui.hidden_picks_text(off, game_page=True) == (
        " Not shown here: the pasted pick Lone Pick Points (picked More; the model has no lean on it).")


def test_the_model_cell_takes_a_second_decimal_when_it_would_print_a_lines_number():
    """_fmt_model's rule for a grid row: two decimals when one decimal would
    print the line of a SHOWN cell back beside its arrow; a hidden line at
    that number does not count."""
    near = {"model": 24.96, "lean": "Less", "gap": -0.04}
    assert drawn(lines(("draftkings", 25.0, near)))[0]["model"] == "24.96"
    assert drawn(lines(("draftkings", 24.5, {"model": 24.96})),
                 lines(("fanduel", 25.0, {**near, "lean": "Even"})))[0]["model"] == "25.0"
    assert drawn(lines(("draftkings", 24.5, {"model": 24.96}),
                       ("betmgm", 25.0, near)))[0]["model"] == "24.96"
    html = props_ui.ledger_html(grid(lines(("draftkings", 24.5, {"model": NAN}))), feed=facts())
    assert f'<td class="num">{SENTINEL}</td>' in html and dash_ok(html)


def test_the_per_player_grid_and_its_settled_notes():
    """Amendment C7: the Game page's grid per player, Stat pinned (no Player
    column: the expander names him), Final or So far after tip, and the
    settled notes by the books with a line in the row."""
    shown = [feed_row("draftkings", 24.5), feed_row("prizepicks", 24.5)]
    html = props_ui.render_player_lines(shown, None, facts())
    assert html.startswith(
        '<div class="bv-grid-wrap" tabindex="0" role="region" aria-label="Ann Able: lines by book">'
        '<table class="bv-grid bv-grid-pl"><thead><tr><th class="st" scope="col">Stat</th>'
        '<th class="num" scope="col">Model</th><th class="bk" scope="col"><abbr title="DraftKings">'
        "DK</abbr></th>")
    assert '<tr><th class="st" scope="row">Points<span class="bv-tier' in html
    assert "Player" not in html and "Final" not in html and "So far" not in html
    (row,) = GP.rows(html)
    assert row["actual"] is None and [c["cls"] for c in row["cells"].values()] == [
        "over", "none", "none", "over", "none"]
    # The same cell as the slate's for the same line.
    slate = props_ui.ledger_html(grid(U.frame(shown)), feed=facts())
    assert cells_of(html) == cells_of(slate)
    assert props_ui.render_player_lines([], None, facts()) == ""
    assert props_ui.render_player_lines(None, None, facts()) == ""
    tag = 'aria-label="Bo &quot;B&quot; &lt;Baker&gt; &amp; Co: lines by book"'
    assert tag in props_ui.render_player_lines([feed_row("draftkings", 24.5, player='Bo "B" <Baker> & Co')],
                                               None, facts())

    def settled(specs, hidden=(), final=True):
        recs = [{**feed_row(b, x), "actual": a, "settle": s, "final": final} for b, x, a, s in specs]
        hid = [{**feed_row(b, x, lean="Even"), "actual": a, "settle": s, "final": final}
               for b, x, a, s in hidden]
        return props_ui.render_player_lines(recs, hid, facts())

    def notes(specs, hidden=()):
        return GP.tables(settled(specs, hidden))[0]["notes"]
    live = settled([("draftkings", 24.5, 12.0, None)], final=False)
    assert 'class="bv-grid bv-grid-pl has-actual"' in live
    assert '<th class="num" scope="col">So far</th></tr></thead>' in live
    assert GP.rows(live)[0]["actual"] == "12" and '<tr class="note">' not in live
    final = settled([("draftkings", 24.5, 27.0, None)])
    assert '<th class="num" scope="col">Final</th>' in final and GP.rows(final)[0]["actual"] == "27"
    gone = None, "void"
    assert notes([("prizepicks", 24.5, *gone)]) == [(8, "Did not play, pick voided")]
    assert notes([("draftkings", 24.5, *gone), ("underdog", 24.5, *gone)]) == [(8, "Did not play")]
    assert notes([("underdog", 24.5, *gone), ("prizepicks", 24.5, *gone),
                  ("draftkings", 24.5, *gone)]) == [
        (8, "PP: Did not play, pick voided. DK, UD: Did not play")]
    # His hidden line is in the row too: its book is named.
    assert notes([("draftkings", 24.5, *gone)], [("prizepicks", 23.5, *gone)]) == [
        (8, "PP: Did not play, pick voided. DK: Did not play")]
    assert notes([("draftkings", 25.0, 25.0, "exact"), ("fanduel", 25.0, 25.0, "exact"),
                  ("prizepicks", 24.5, 25.0, None)]) == [(8, "25: landed on the line (DK, FD)")]
    assert notes([("draftkings", 24.5, 27.0, None)]) == []
    assert settled([("prizepicks", 24.5, *gone)]).count("pick voided") == 1


def css_rules(css: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """(enclosing at-rule or "", selector, [(property, value)]) for every rule
    of a <style> block, comments dropped, whitespace normalised."""
    text = re.sub(r"/\*.*?\*/", "", css, flags=re.S).replace("<style>", "").replace("</style>", "")
    out, stack, buf = [], [], ""
    for ch in text:
        if ch == "{":
            stack.append(" ".join(buf.split()))
            buf = ""
        elif ch == "}":
            head = stack.pop()
            if buf.strip():
                decls = [tuple(" ".join(x.split()) for x in d.split(":", 1))
                         for d in buf.split(";") if d.strip()]
                out.append((stack[-1] if stack else "", head, decls))
            buf = ""
        else:
            buf += ch
    assert not stack and not buf.strip(), (stack, buf)
    return out


def test_the_grid_css_scrolls_sideways_pins_the_player_column_and_follows_the_theme():
    """Amendment C13, read from BOOKS_CSS's own rules (selector, declarations),
    never a substring of the block."""
    rules = css_rules(props_ui.BOOKS_CSS)

    def decls(selector, media=""):
        found = [d for m, s, d in rules if s == selector and m == media]
        assert len(found) == 1, (selector, media, len(found))
        return found[0]

    wrap = dict(decls(".bv-grid-wrap"))
    assert wrap["overflow-x"] == "auto" and wrap["position"] == "relative"      # C1R-03
    assert wrap["max-width"] == "100%"
    table = dict(decls(".bv-grid-wrap table.bv-grid"))                          # C1R-08: (0,2,1)
    assert table["margin"] == "0" and table["border-collapse"] == "separate"
    assert table["border-spacing"] == "0"
    assert dict(decls("table.bv-grid"))["min-width"] == "36rem"
    cell = decls("table.bv-grid th, table.bv-grid td")
    props_in_order = [p for p, _v in cell]
    assert ("border", "0") in cell and props_in_order.index("border") < props_in_order.index("border-bottom")
    for selector in ("table.bv-grid .pl", "table.bv-grid.bv-grid-pl .st"):
        d = dict(decls(selector))
        assert (d["position"], d["left"]) == ("sticky", "0"), selector
        assert d["background"] == "var(--panel-solid)" and d["box-shadow"] == "1px 0 0 var(--panel-line)"
    assert dict(decls("table.bv-grid .pl .nm, table.bv-grid .pl i"))["max-width"] == "12rem"
    assert dict(decls("table.bv-grid .pl .nm, table.bv-grid .pl i", "@media (max-width: 760px)"))[
        "max-width"] == "8.5rem"
    for _m, selector, d in rules:                                               # C1R-09
        if any(part.strip().endswith(".pl") for part in selector.split(",")):
            assert "max-width" not in dict(d), selector
    phone = [s for m, s, _d in rules if m == "@media (max-width: 760px)"]
    assert "table.bv-grid" in phone and "table.bv-grid .pl" in phone and ".bv-cell" in phone
    guards = [(s, dict(d)) for m, s, d in rules if ":has(.bv-grid-wrap)" in s]
    assert len(guards) == 1
    selector, d = guards[0]
    assert selector.split(", ") == [
        '[data-testid="stVerticalBlock"].st-key-bv_board_grid [data-testid="stElementContainer"]'
        ":has(.bv-grid-wrap)",
        '[data-testid="stVerticalBlock"].st-key-bv_game_lines [data-testid="stElementContainer"]'
        ":has(.bv-grid-wrap)"]
    assert d == {"min-width": "0", "max-width": "100%"}
    for cls, token in (("over", "good"), ("under", "bad")):
        d = dict(decls(f".bv-cell.{cls}"))
        assert d == {"color": f"var(--value-{token})", "background": f"var(--tint-{token})"}
    sr = dict(decls(".bv-sr"))
    assert sr["position"] == "absolute" and sr["clip"] == "rect(0, 0, 0, 0)"
    # Theme tokens only: each defined in both of theme.py's token blocks.
    used = set(re.findall(r"var\(--([a-z0-9-]+)\)", props_ui.BOOKS_CSS))
    assert used >= {"panel-solid", "panel-line", "value-good", "tint-good", "value-bad", "tint-bad"}
    for name in used:
        for block in (theme.THEME_BASE_CSS, theme.THEME_LIGHT_CSS):
            assert re.search(rf"--{name}\s*:", block), name
    colour = re.compile(r"^(?:\d*\.?\d+(?:px|rem|em|%)?|none|solid|0)$")
    for _m, selector, d in rules:
        for prop, value in d:
            assert "#" not in value and "rgb" not in value and "hsl" not in value, (selector, prop)
            if prop in ("color", "background", "border", "border-bottom", "box-shadow"):
                rest = re.sub(r"var\(--[a-z0-9-]+\)", "", value).split()
                assert all(colour.match(t) for t in rest), (selector, prop, value)
    assert all(ord(ch) < 128 for ch in props_ui.BOOKS_CSS)
    # The feed-off CSS is untouched: the feed-off golden's and its page's.
    assert props_ui.PROPS_CSS == GOLDEN["functions"]["props_css"]
    assert props_ui.DISCREPANCY_CSS.strip() in GOLDEN["page"]["markdown"]


def test_the_legend_line_under_the_grid():
    assert props_ui.GRID_LEGEND == (
        "Up arrow: model leans More (Over) at that line. Down arrow: Less (Under). No arrow: no "
        "lean there, or the line does not offer that side. A dash: no line from that book.")
    assert props_ui.grid_legend_html() == ('<div class="bv-grid-legend">' + props_ui.GRID_LEGEND
                                           + "</div>")
    assert esc(props_ui.GRID_LEGEND) == props_ui.GRID_LEGEND
    assert props_ui.GRID_CAPTION.format(k=13, n=13) == (
        "Top 13 of 13 players, one row each (their best-ranked stat). Show all lists every stat.")


def test_the_hidden_note_says_a_hidden_line_beside_a_shown_one_is_drawn():
    """C1R-17: with the feed on, the hidden note says a hidden line in a row
    with a shown line is drawn as its number alone, right after "There is no
    pick to show on those.", and only when such a line exists. The feed-off
    board never says it (its golden is unchanged)."""
    assert props_ui.HIDDEN_DRAWN == ("A hidden line in a row with a shown line is drawn as its "
                                     "number alone, with no arrow.")
    shown = lines(("draftkings", 24.5))
    beside = lines(("fanduel", 25.5, {"lean": "Even"}))
    alone = U.frame([feed_row("fanduel", 25.5, lean="Even", person_id=1009, player="Lone Line")])
    meta = {"posted": 2, "matched": 2}

    def note(hidden, feed):
        found = [visible(n) for n in props_ui.notes_html(meta, shown, hidden, None, None, feed)
                 if " hidden:" in visible(n)]
        assert len(found) == 1
        return found[0]
    said = note(beside, facts())
    assert said == ("1 line hidden: 1 sits on a coin flip with no lean. There is no pick to show "
                    "on those. " + props_ui.HIDDEN_DRAWN)
    assert props_ui.HIDDEN_DRAWN not in note(alone, facts())
    assert props_ui.HIDDEN_DRAWN not in note(beside, None)
    picked = beside.assign(pick="more", origin="pasted", book="prizepicks")
    assert note(picked, facts()).endswith(
        "There is no pick to show on those. " + props_ui.HIDDEN_DRAWN + " The pasted pick among "
        "them: Ann Able Points (picked More; the model has no lean on it).")


# -- The side words (amendment A4) and per-book rules -------------------------------------------
def test_side_words_are_over_and_under_on_a_sportsbook_and_more_and_less_on_pickem():
    """m8: the words follow the cell's book; the stored lean stays More, Less
    or Even, and the cell's class over, under or plain."""
    def lean_cell(book, lean, sides="both"):
        shown = lean in ("More", "Less") and sides in ("both", lean.lower())
        mine = lines((book, 24.5, {"lean": lean, "sides": sides}))
        other = lines(("fanduel" if book == "draftkings" else "draftkings", 25.5))
        rows = drawn(pd.concat([other, mine], ignore_index=True)) if shown else drawn(other, mine)
        c = rows[0]["cells"][props.BOOK_CODE[book]]
        return (c["cls"], c["sr"]), c
    for book in SPORTSBOOKS:
        assert lean_cell(book, "More")[0] == ("over", " leans Over")
        assert lean_cell(book, "Less")[0] == ("under", " leans Under")
        assert lean_cell(book, "Even")[0] == ("plain", " no lean")
        assert lean_cell(book, "More")[1]["title"].endswith(": the model leans Over.")
        assert lean_cell(book, "Less", "more")[0] == ("plain", " leans Under, not offered")
        assert '<span class="bv-pp-tag" title="This line offers Over only">Over only</span>' in \
            lean_cell(book, "More", "more")[1]["badges"]
        assert '<span class="bv-pp-tag" title="This line offers Under only">Under only</span>' in \
            lean_cell(book, "Less", "less")[1]["badges"]
    for book in ("prizepicks", "underdog"):
        assert lean_cell(book, "More")[0] == ("over", " leans More")
        assert lean_cell(book, "Less")[0] == ("under", " leans Less")
        assert lean_cell(book, "Less", "more")[0] == ("plain", " leans Less, not offered")
        assert "More only</span>" in lean_cell(book, "More", "more")[1]["badges"]
    # The discrepancy chips and their table say the same words.
    rows = U.frame([feed_row("draftkings", 20.5, person_id=3000 + i, player=f"P{i}", model=22.5 + i,
                             gap=2.0 + i, lean="More") for i in range(8)])
    strip = props_ui.discrepancy_strip_html(props_ui.discrepancies(rows))
    assert strip.count('<span class="gc-lean">Over</span>') == props_ui.DISCREPANCY_CHIPS
    assert strip.count('<td class="l lean over">Over</td>') == 2
    assert "More" not in visible(strip)


def test_per_book_rules_the_tie_the_void_risk_and_a_player_who_did_not_play():
    """design 2.6: the tie tooltip keeps PrizePicks' words on pick'em rows and
    says the book's own rule on a sportsbook row (in the Stat cell with the
    feed off, in the line's own cell's title with it); the void-risk tag
    (Reboot) is PrizePicks' only; a settled sportsbook row reads 'Did not
    play', and since build amendment B30 so does a settled Underdog row
    (PrizePicks' keeps VOID_TEXT)."""
    tie = dict(tie_possible=True, p_push=0.12)
    dk = props_ui._stat_cell(feed_row("draftkings", 6.0, **tie))
    assert ('title="Chance of landing exactly on 6: 12 in 100. On a sportsbook line the book&#x27;s '
            'own rule decides an exact landing."') in dk
    for book in ("prizepicks", "underdog"):
        assert "That lowers the payout tier; it is not a push." in \
            props_ui._stat_cell(feed_row(book, 6.0, **tie))
    assert drawn(lines(("draftkings", 6.0, tie)))[0]["cells"]["DK"]["title"] == (
        "DraftKings 6: the model leans Over. Chance of landing exactly on 6: 12 in 100. On a "
        "sportsbook line the book's own rule decides an exact landing.")
    assert drawn(lines(("underdog", 6.0, tie)))[0]["cells"]["UD"]["title"] == (
        "Underdog 6: the model leans More. Chance of landing exactly on 6: 12 in 100. That lowers "
        "the payout tier; it is not a push.")
    bench = dict(lean="More", void_note=True, mu_min=15.0)
    assert props_ui._void_tag(feed_row("prizepicks", 6.0, **bench))
    for book in ("underdog", "draftkings"):
        assert props_ui._void_tag(feed_row(book, 6.0, **bench)) == ""
    gone = {"actual": None, "settle": "void", "final": True}
    for book in ("draftkings", "underdog"):                         # B30: Underdog's too
        assert '<div class="bv-pl-note">Did not play</div>' in \
            props_ui.render_player_lines([{**feed_row(book, 6.0), **gone}], None, facts())
    assert f'<div class="bv-pl-note">{props_ui.VOID_TEXT}</div>' in props_ui.render_player_lines(
        [{**feed_row("prizepicks", 6.0), **gone}], None, facts())


# -- Biggest discrepancies (amendments A8 and C11) ---------------------------------------------
def test_the_discrepancies_title_names_the_book_with_the_feed():
    rows = props_ui.discrepancies(U.gap_rows())
    win = props_ui.discrepancies_window(rows, "March 12", book_label="DraftKings")
    assert '<span class="meta">9 standard DraftKings lines, largest gap first</span>' in win
    one = props_ui.discrepancies_window(rows.head(1), "PHI at DET", "PrizePicks")
    assert '<span class="meta">1 standard PrizePicks line, largest gap first</span>' in one
    assert props_ui.discrepancies_window(rows, "March 12") == \
        props_ui.discrepancies_window(rows, "March 12", None)
    assert "9 standard lines, largest gap first" in props_ui.discrepancies_window(rows, "March 12")
    assert "9 standard BetMGM lines" in props_ui.discrepancies_html(U.gap_rows(), "March 12", "BetMGM")
    at = _page(f"""
import sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(HERE)!r})
import ui_rows as U
from bucketlib import props_ui
props_ui.render_discrepancies(U.gap_rows(), "March 12", book_label="FanDuel")
""")
    assert any("9 standard FanDuel lines, largest gap first" in m.value for m in at.markdown)


def test_discrepancy_chips_name_their_book_with_the_feed():
    """Amendment C11: with `books` each chip names the books posting its line
    after the line, as codes in <abbr>s of our titles, and the '+N more'
    table gains a Book column; without, every function is today's."""
    g = GOLDEN["functions"]
    rows = props_ui.discrepancies(U.gap_rows())
    assert props_ui.discrepancy_strip_html(rows, False) == g["discrepancy_strip"]
    assert props_ui.discrepancies_html(U.gap_rows(), "March 12", None, False) == g["discrepancies"]
    strip = props_ui.discrepancy_strip_html(rows, True)
    pp = '<abbr title="PrizePicks">PP</abbr>'
    assert strip.count(f'<span class="gc-book">{pp}</span> &middot; ') == props_ui.DISCREPANCY_CHIPS
    assert '<th>Line</th><th class="l">Book</th><th>Gap</th>' in strip
    assert strip.count(f'</td><td class="l">{pp}</td><td>') == len(rows) - props_ui.DISCREPANCY_CHIPS
    two = U.frame([feed_row("fanduel", 20.5, person_id=4001, player="Pat Two", model=23.5, gap=3.0),
                   feed_row("draftkings", 20.5, person_id=4001, player="Pat Two", model=23.5, gap=3.0)])
    chip = props_ui.discrepancy_strip_html(props_ui.discrepancies(two), True)
    assert ('<span class="gc-nums"><span class="num">23.5</span> <span class="gc-vs">vs</span> '
            '<span class="num line">20.5</span> <span class="gc-book"><abbr title="DraftKings">DK'
            '</abbr> <abbr title="FanDuel">FD</abbr></span> &middot; <span class="num diff over">'
            '+3.0</span> <span class="gc-lean">Over</span></span>') in chip
    assert chip.count('class="bv-gap-chip') == 1
    assert props_ui.discrepancies_html(two, "March 12", books=True).startswith(
        '<div class="bv-window"><span class="lab">Biggest discrepancies</span><span class="meta">1 '
        "standard line, largest gap first</span>")
    at = _page(f"""
import sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(HERE)!r})
import test_props_ui_feed as T
import ui_rows as U
from bucketlib import props_ui
two = U.frame([T.feed_row("fanduel", 20.5, model=23.5, gap=3.0),
               T.feed_row("draftkings", 20.5, model=23.5, gap=3.0)])
props_ui.render_discrepancies(two, "March 12", books=True)
""")
    md = [m.value for m in at.markdown]
    assert props_ui.BOOKS_CSS.strip() in md and props_ui.DISCREPANCY_CSS.strip() in md
    assert any('<span class="gc-book"><abbr title="DraftKings">DK</abbr> <abbr title="FanDuel">FD'
               "</abbr></span>" in m for m in md)
    assert any("1 standard line, largest gap first" in m for m in md)


def test_lines_at_one_number_share_one_discrepancy_row(world):
    """C1R-07 on test_props_feed's board (MeanDispersionProvider, no paste):
    the 36 standard lines merge to 28 rows, one per player, stat, line and
    side word, naming every book that posts it; the title counts the rows.
    Lines of one key at different numbers, and an Over line and a More line
    at one number, stay apart; the codes come in BOOK_ORDER."""
    fb = props.feed_board(PF.tables(world, props.MeanDispersionProvider(world.proj)), None, None,
                          None)
    shown = props.board_rows(fb.lines, props_ui.void_ids(world.proj))[0]
    ranked = props_ui.discrepancies(shown)
    merged = props_ui.merge_books(ranked)
    assert (len(ranked), len(merged)) == (36, 28)
    assert sum(len(b) for b in merged[props_ui.GAP_BOOKS]) == len(ranked)
    assert [(int(r["person_id"]), r["stat"], r["line"], r[props_ui.GAP_BOOKS], props_ui._lean_text(r))
            for r in merged.head(props_ui.DISCREPANCY_CHIPS).to_dict(orient="records")] == [
        (PF.DUREN, "pts", 13.5, ("draftkings", "fanduel"), "Over"),
        (PF.DUREN, "pts", 13.5, ("prizepicks",), "More"),
        (PF.JOKIC, "pts", 29.5, ("betmgm",), "Under"),
        (PF.CUNNINGHAM, "pts", 25.5, ("draftkings", "fanduel", "betmgm"), "Under"),
        (PF.CUNNINGHAM, "pts", 25.5, ("prizepicks", "underdog"), "Less"),
        (PF.CUNNINGHAM, "ast", 8.5, ("prizepicks",), "More")]
    strip = props_ui.discrepancy_strip_html(merged, True)
    assert [visible(c) for c in re.findall(r'<span class="gc-book">(.*?)</span>', strip)][:6] == [
        "DK FD", "PP", "MGM", "DK FD MGM", "PP UD", "PP"]
    assert props_ui.discrepancy_strip_html(ranked, True) == strip       # merged on the way in
    page = props_ui.discrepancies_html(shown, "March 12", books=True)
    assert "28 standard lines, largest gap first" in page
    assert "36 standard lines" in props_ui.discrepancies_html(shown, "March 12")
    # Hand-built: one key's lines arriving Underdog first.
    rows = U.frame([feed_row("underdog", 24.5, gap=1.8), feed_row("draftkings", 24.5, gap=1.8),
                    feed_row("fanduel", 25.5, gap=0.8), feed_row("prizepicks", 24.5, gap=1.8)])
    m = props_ui.merge_books(props_ui.discrepancies(rows))
    assert [(r["book"], r["line"], r[props_ui.GAP_BOOKS]) for r in m.to_dict(orient="records")] == [
        ("underdog", 24.5, ("prizepicks", "underdog")), ("draftkings", 24.5, ("draftkings",)),
        ("fanduel", 25.5, ("fanduel",))]
    keyless = U.frame([feed_row("draftkings", 24.5, game_date=None),
                       feed_row("fanduel", 24.5, game_date=None)])
    assert len(props_ui.merge_books(keyless)) == 2
    empty = props_ui.merge_books(props_ui.discrepancies(rows.iloc[:0]))
    assert empty.empty and props_ui.GAP_BOOKS in empty.columns


# -- The feed's notices (design 2.6, 6; C9, C10; N1 to N3) -------------------------------------
def test_the_books_missing_note_names_at_most_two_games_then_and_n_more():
    """Amendment C9 (C15 items 7 and 8): each sentence in its own <b>, one
    space between, all in one bv-note on the slate; books whose remaining
    games are the same share a sentence; different lists come in BOOK_ORDER
    of their first book; at most two games named, then "and N more"."""
    more = {**LABELS, "g3": "IND at PHX", "g4": "MIA at BOS"}

    def note(missing, labels=LABELS, **kw):
        return props_ui.books_missing_html(missing, labels, **kw)
    assert note({"prizepicks": [LATE]}) == (
        '<div class="bv-note"><b>No PrizePicks lines yet for DEN at SAS.</b></div>')
    assert visible(note({"prizepicks": [EARLY, LATE]})) == (
        "No PrizePicks lines yet for PHI at DET and DEN at SAS.")
    assert note({"prizepicks": [LATE, EARLY, "g3"], "underdog": [LATE, EARLY, "g3"]}, more) == (
        '<div class="bv-note"><b>No PrizePicks or Underdog lines yet for DEN at SAS and 2 more '
        "games.</b></div>")
    assert visible(note({b: [LATE] for b in ("underdog", "betmgm", "fanduel", "draftkings")})) == (
        "No DraftKings, FanDuel, BetMGM or Underdog lines yet for DEN at SAS.")
    assert note({"underdog": [LATE], "prizepicks": [EARLY], "fanduel": [LATE]}) == (
        '<div class="bv-note"><b>No FanDuel or Underdog lines yet for DEN at SAS.</b> '
        "<b>No PrizePicks lines yet for PHI at DET.</b></div>")
    assert visible(note({"draftkings": [LATE, EARLY, "g3", "g4"]}, more)) == (
        "No DraftKings lines yet for DEN at SAS and 3 more games.")
    for n in range(1, 6):
        gids = [LATE, EARLY, "g3", "g4", "g5"][:n]
        said = visible(note({"prizepicks": gids}, more))
        assert sum(said.count(x) for x in more.values()) <= 2, n
    assert visible(note({"prizepicks": ["g9"]})) == "No PrizePicks lines yet for 1 game."
    assert visible(note({"prizepicks": [LATE, "g9"]})) == (
        "No PrizePicks lines yet for DEN at SAS and 1 more game.")
    html = note({"prizepicks": [LATE]}, {LATE: "<b>X</b>"})
    assert "&lt;b&gt;X&lt;/b&gt;" in html and "<b>X</b>" not in html
    for nothing in ({}, None, {"prizepicks": []}, {b: [] for b in props.BOOK_ORDER},
                    {"<script>": [LATE], "bet365": [LATE]}, {"prizepicks": 5}, "junk"):
        assert note(nothing) == "", nothing
    game = props_ui.books_missing_game_html
    assert game(["prizepicks"]) == "<b>No PrizePicks lines for this game yet.</b>"
    assert game(("underdog", "betmgm", "fanduel", "draftkings")) == (
        "<b>No DraftKings, FanDuel, BetMGM or Underdog lines for this game yet.</b>")
    assert game([]) == game(None) == game(["bet365", "x<b>"]) == ""
    for html in (note({"prizepicks": [LATE]}), game(["prizepicks"]), game(["betmgm"], tipped=True)):
        assert "Pick another" not in html and " above" not in html and " below" not in html


def test_the_notice_names_the_schedules_games_never_the_vendors_teams_n2():
    """N2: the games come from the page's labels (the schedule's away at
    home), in the order the board gives (tip order); nothing of the
    vendor's event teams is read."""
    info = facts(missing={"prizepicks": [LATE, EARLY]}, report={"event_home": "DEN"})
    html = props_ui.feed_notice_html(info, DAY, LABELS)
    assert "No PrizePicks lines yet for DEN at SAS and PHI at DET." in visible(html)


def test_the_feeds_state_phrases_n1():
    """design 6's phrases: stale and paused with lines quote the as-of
    stamp; with no lines stale and down read DOWN and paused its no-lines
    phrase (N1); ok shows the stamp; idle and off say nothing."""
    def say(**kw):
        return visible(props_ui.feed_notice_html(facts(**kw), DAY))
    assert say() == "Lines as of 9:02 AM ET."
    assert say(asof=LAST_NIGHT) == "Lines as of 9:41 PM ET, Mar 11."
    assert say(state="stale") == ("Lines as of 9:02 AM ET. The odds feed did not answer just now, "
                                  "so the board shows the last lines it had.")
    down = ("Lines are unavailable right now. The odds feed did not answer. To compare a PrizePicks "
            "board, paste it with Update lines in the top bar.")
    assert say(state="down", asof=None, lines=0) == down
    assert say(state="stale", asof=None, lines=0) == down
    assert say(state="paused") == "Lines as of 9:02 AM ET. The board is not refreshing these lines for now."
    assert say(state="paused", asof=None, lines=0) == (
        "No lines from the odds feed today. To compare a PrizePicks board, paste it with Update "
        "lines in the top bar.")
    assert say(state="idle", asof=None, lines=0) == ""
    assert props_ui.feed_notice_html(facts(state="off"), DAY) == ""
    assert props_ui.feed_notice_html(None, DAY) == props_ui.feed_notice_html({}, DAY) == ""
    assert '<div class="bv-notice"><b>Lines are unavailable' in props_ui.feed_notice_html(
        facts(state="down", asof=None, lines=0), DAY)


def test_the_loading_line_and_the_game_scope():
    assert visible(props_ui.feed_notice_html(facts(loading=2), DAY)) == (
        "Lines as of 9:02 AM ET. Lines for 2 games are still loading; this board fills in by itself.")
    assert "1 game are still loading" in props_ui.feed_notice_html(facts(loading=1), DAY)
    assert "loading" not in props_ui.feed_notice_html(facts(loading=2), DAY, with_loading=False)
    assert props_ui.loading_line_html(0) == ""
    # The Game page: that game's own latest pull, its pre-tip copy after tip,
    # and none of the date's other lines.
    info = facts(missing={"prizepicks": [LATE]}, feed_won=3,
                 per_game={EARLY: {"asof": NOON, "pretip": False}, LATE: {"asof": NINE, "pretip": True}})
    assert visible(props_ui.feed_notice_html(info, DAY, LABELS, game_id=EARLY)) == \
        "Lines as of 12:00 PM ET."
    assert props_ui.lines_asof_text(info, DAY, LATE) == "Pre-tip lines, as of 9:02 AM ET."
    assert props_ui.lines_asof_text(info, DAY) == "Lines as of 9:02 AM ET."
    assert props_ui.lines_asof_text(info, DAY, "nope") == ""
    assert props_ui.lines_asof_text(None) == ""


def test_the_stamp_is_the_oldest_books_own_time_l3_1():
    """Amendment C10 (build 4's review, L3-1): the stamp is the OLDEST own
    fetch time among the books with feed lines, for one game or the date, so
    a refresh that left FanDuel out never stamps its 8:55 AM lines 6:30 PM.
    Only BOOK_ORDER's books and readable times count; with no book's time
    given, the feed's own "asof"."""
    six30, eight55 = pd.Timestamp("2026-03-12T22:30:00Z"), pd.Timestamp("2026-03-12T12:55:00Z")
    info = facts(per_game={EARLY: {"asof": six30, "pretip": False,
                                   "books": {"draftkings": six30, "fanduel": eight55}},
                           LATE: {"asof": six30, "pretip": False, "books": {"draftkings": six30}}})
    assert props_ui.lines_asof_text(info, DAY, EARLY) == "Lines as of 8:55 AM ET."
    assert props_ui.lines_asof_text(info, DAY) == "Lines as of 8:55 AM ET."
    assert props_ui.lines_asof_text(info, DAY, LATE) == "Lines as of 6:30 PM ET."
    assert visible(props_ui.feed_notice_html(info, DAY, LABELS)) == "Lines as of 8:55 AM ET."
    assert visible(props_ui.feed_notice_html({**info, "state": "stale"}, DAY, LABELS,
                                             game_id=EARLY)).startswith("Lines as of 8:55 AM ET.")
    odd = facts(per_game={EARLY: {"asof": six30, "books": {"bet365": eight55, "fanduel": "junk",
                                                           "draftkings": six30}}})
    assert props_ui.lines_asof_text(odd, DAY, EARLY) == "Lines as of 6:30 PM ET."
    none = facts(asof=NINE, per_game={EARLY: {"asof": NOON, "books": {}}})
    assert props_ui.lines_asof_text(none, DAY, EARLY) == "Lines as of 12:00 PM ET."
    assert props_ui.lines_asof_text(none, DAY) == "Lines as of 9:02 AM ET."


def test_the_merge_line_quotes_the_feeds_oldest_winning_fetch_n3():
    times = {"pasted_utc": LAST_NIGHT, "feed_oldest": NINE, "feed_newest": NOON}
    said = visible(props_ui.feed_notice_html(facts(feed_won=14, paste_won=3, paste_times=times), DAY))
    assert ("Your pasted PrizePicks lines from 9:41 PM, Mar 11 are older than the odds feed's "
            "(9:02 AM), so the feed's line is shown for 14 players and stats. Your newer, typed, "
            "Demon or Goblin lines are shown for 3.") in said            # build amendment B28
    one = visible(props_ui.feed_notice_html(facts(feed_won=1, paste_times=times), DAY))
    assert "shown for 1 player and stat." in one and "newer, typed" not in one
    mine = visible(props_ui.feed_notice_html(facts(paste_won=2, paste_times=times), DAY))
    assert ("Your newer, typed, Demon or Goblin PrizePicks lines are shown for 2 players and "
            "stats.") in mine
    assert "newer or typed" not in said + one + mine
    assert "pasted" not in visible(props_ui.feed_notice_html(facts(), DAY))


def test_the_lines_a_book_took_down_are_named_with_our_titles():
    report = {"pulled": [("draftkings", "Jalen Duren", "Rebounds"), ["fanduel", "A <b>", "Points"],
                         ("vendorbook", "Cade Cunningham", "Assists")], "n_pulled": 4}
    html = props_ui.feed_notice_html(facts(report=report), DAY)
    assert ("<b>4 lines taken down by their book since the last update:</b> Jalen Duren Rebounds "
            "(DraftKings), A &lt;b&gt; Points (FanDuel), Cade Cunningham Assists, and others.") in html
    assert "vendorbook" not in html


def test_the_feeds_text_never_reads_props_uis_clock(monkeypatch, world, params, tokens):
    """m1: every feed text is built from the stamps and the `today` the page
    passes. props_ui's own clock raises here while all of it is built."""
    def no_clock():
        raise AssertionError("props_ui read its clock for feed text")
    monkeypatch.setattr(props_ui, "now_utc", no_clock)
    monkeypatch.setattr(props_ui, "today_et", no_clock)
    times = {"pasted_utc": LAST_NIGHT, "feed_oldest": NINE, "feed_newest": NOON}
    info = facts(missing={"prizepicks": [LATE]}, feed_won=2, paste_won=1, paste_times=times,
                 loading=1, report={"pulled": [("draftkings", "X", "Points")], "n_pulled": 1})
    for state in ("ok", "stale", "paused", "down", "idle"):
        props_ui.feed_notice_html({**info, "state": state}, DAY, LABELS)
        props_ui.feed_notice_html({**info, "state": state}, DAY, LABELS, game_id=EARLY)
    shown = lines(("draftkings", 25.5), ("fanduel", 25.5, {"fetched_at": LAST_NIGHT}),
                  ("prizepicks", 24.5, {"origin": "pasted", "pick": "more", "fetched_at": None}))
    hidden = lines(("betmgm", 25.5, {"lean": "Even"}))
    assert "Line as of 9:41 PM ET, Mar 11." in props_ui.ledger_html(grid(shown, hidden), feed=info)
    props_ui.render_player_lines(shown, hidden, info)
    settled = [{**r, "actual": None, "settle": "void", "final": True}
               for r in shown.to_dict(orient="records")]
    assert "pick voided" in props_ui.render_player_lines(settled, hidden, info)
    props_ui.notes_html({"posted": 1, "matched": 1}, shown, hidden, None, None, info)
    assert props_ui.books_missing_html(info["missing"], LABELS, tipped={EARLY})
    assert props_ui.books_missing_game_html(["prizepicks"], tipped=True)
    ft = PF.tables(world, simprov.provider_for(world.proj, params))
    token = _paste(world, tokens, NINE)
    view = props_ui.date_board(token, DAY, world.proj, params, world.roster, feed=ft,
                               feed_info=info)
    assert not view.meta.get("error") and len(view.shown)


# -- date_board with the feed (layout C, end to end) -----------------------------------------
@pytest.fixture(scope="module")
def feed_tables(world, params):
    return PF.tables(world, simprov.provider_for(world.proj, params))


def test_the_board_is_every_books_lines_balanced_and_counted(world, params, feed_tables):
    """The serve fixture's slate through the real parser (test_props_feed's
    counts: 53 lines, PrizePicks' 9 all on PHI at DET). The board holds every
    book's lines, the ledger balances over all of them, and the page's facts
    are kept beside the board's own."""
    view = props_ui.date_board(None, DAY, world.proj, params, world.roster, feed=feed_tables,
                               feed_info=facts())
    rows = pd.concat([view.shown, view.hidden, view.thin, view.void])
    assert set(rows["book"]) == set(props.BOOK_ORDER)
    assert {b: int((rows["book"] == b).sum()) for b in props.BOOK_ORDER} == PF.BOOK_ROWS
    assert view.meta["posted"] == sum(PF.BOOK_ROWS.values()) == 53 and not view.meta.get("error")
    assert props.ledger(view.meta, view.shown, view.hidden, view.thin, view.void)["balanced"]
    assert view.feed["counts"] == {"feed": 53, "pasted": 0}
    want = {"draftkings": [], "fanduel": [], "betmgm": [], "prizepicks": [LATE], "underdog": []}
    assert view.missing == view.feed["missing"] == want
    assert view.books_by_game == {EARLY: props.BOOK_ORDER,
                                  LATE: ("draftkings", "fanduel", "betmgm", "underdog")}
    assert view.feed["state"] == "ok" and view.feed["today"] == DAY       # the page's facts kept
    assert view.feed["lines"] == 40 and "book" not in view.feed
    assert props_ui.line_counts_by_game(view.shown) == \
        {str(g): n for g, n in view.shown.groupby("game_id").size().items()}
    rows_of = props.grid_rows(view.shown, view.hidden)
    assert set(zip(rows_of["person_id"], rows_of["stat"])) == set(zip(view.shown["person_id"],
                                                                     view.shown["stat"]))


def test_the_paste_merges_into_prizepicks_newest_wins_and_the_board_says_so(
        world, params, feed_tables, tokens):
    """M5 through date_board: a paste at 11 AM is newer than the 9 AM feed,
    so it wins the 6 shared keys; one from last night loses them. The ledger
    counts every book's lines on file by where they came from."""
    eleven = pd.Timestamp("2026-03-12T15:00:00Z")
    late = _paste(world, tokens, eleven)
    view = props_ui.date_board(late, DAY, world.proj, params, world.roster, feed=feed_tables,
                               feed_info=facts())
    assert (view.paste_won, view.feed_won) == (len(PF.SHARED), 0)
    assert view.meta["posted"] == 53 + PF.PASTED - len(PF.SHARED) == 58
    assert view.feed["counts"] == {"feed": 47, "pasted": PF.PASTED}
    assert view.missing["prizepicks"] == [] and view.feed["pasted"] == PF.PASTED
    note = props_ui.notes_html(view.meta, view.shown, view.hidden, view.thin, view.void, view.feed)[0]
    assert visible(note).startswith(f"58 lines on file for this date (47 from the odds feed, "
                                    f"{PF.PASTED} pasted): ")
    old = _paste(world, tokens, LAST_NIGHT)
    view = props_ui.date_board(old, DAY, world.proj, params, world.roster, feed=feed_tables,
                               feed_info=facts())
    assert (view.paste_won, view.feed_won) == (0, len(PF.SHARED))
    assert view.feed["counts"] == {"feed": 53, "pasted": PF.PASTED - len(PF.SHARED)}
    assert view.paste_times["pasted_utc"] == LAST_NIGHT and view.paste_times["feed_oldest"] is not None
    said = visible(props_ui.feed_notice_html(view.feed, DAY, LABELS))
    assert ("Your pasted PrizePicks lines from 9:41 PM, Mar 11 are older than the odds feed's "
            "(9:00 AM), so the feed's line is shown for 6 players and stats.") in said
    # PrizePicks' rows on the board hold both origins; the paste's time is the meta's.
    rows = pd.concat([view.shown, view.hidden, view.thin, view.void])
    assert set(rows.loc[rows["book"] == "prizepicks", "origin"]) == {"feed", "pasted"}
    assert pd.Timestamp(view.meta["pasted_utc"]) == LAST_NIGHT


def test_a_feed_board_that_fails_is_explained_with_the_feeds_words(
        world, params, feed_tables, tokens, monkeypatch):
    def boom(*a, **k):
        raise ValueError("vendor-said-xyzzy")
    monkeypatch.setattr(props, "feed_board", boom)
    for token, want in ((None, "The odds feed's lines are still in memory"),
                        (_paste(world, tokens, NINE), "Your pasted lines are still saved")):
        view = props_ui.date_board(token, DAY, world.proj, params, world.roster, feed=feed_tables,
                                   feed_info=facts())
        assert view.meta["error"] and view.meta["posted"] == 53 + (PF.PASTED if token else 0)
        assert "book" not in view.feed and props_ui._kept_text(view.feed) == want
    no_params = props_ui.date_board(None, DAY, world.proj, None, world.roster, feed=feed_tables,
                                    feed_info=facts())
    assert no_params.meta == {"posted": 53, "error": True}
    bare = props_ui.date_board(None, DAY, world.proj, params, world.roster, feed=feed_tables)
    assert bare.feed == {"missing": {}, "paste_won": 0, "feed_won": 0, "paste_times": {},
                         "pasted": 0, "counts": {"feed": 53, "pasted": 0}, "paste_unmatched": [],
                         "paste_ambiguous": []}


def test_tables_that_could_not_be_built_are_the_explained_empty_board_f1(
        world, params, feed_tables, tokens):
    """Build 3's review, F1. store.feed_tables marks the tables failed when
    the feed holds lines it could not compare: props.FeedTables(failed=True),
    no table and no meta. date_board then gives the explained empty board at
    once, with or without the simulator parameters, never a board built from
    nothing: it counts every line the feed holds for the date (the facts' 40
    "lines") and the paste's, and says whose lines are kept. Failed tables
    that carried a meta count the facts' lines too."""
    failed = props.FeedTables(failed=True)
    assert (failed.tables, failed.metas, failed.failed) == ({}, {}, True)
    assert props.FeedTables().failed is False and feed_tables.failed is False
    token = _paste(world, tokens, NINE)
    for tok, kept in ((None, "The odds feed's lines are still in memory"),
                      (token, "Your pasted lines are still saved")):
        for p in (params, None):
            if tok and p is not None:
                continue     # a paste that can be compared is the board (B17, the next test)
            view = props_ui.date_board(tok, DAY, world.proj, p, world.roster, feed=failed,
                                       feed_info=facts())
            pasted = PF.PASTED if tok else 0
            assert view.meta == {"posted": 40 + pasted, "error": True}, p
            assert view.matched == 0 and view.missing == {} and view.feed["missing"] == {}
            assert view.feed["state"] == "ok" and "book" not in view.feed
            assert view.feed["counts"] == {"feed": 40, "pasted": pasted}
            assert props_ui._kept_text(view.feed) == kept
    with_meta = props.FeedTables(metas=dict(feed_tables.metas), failed=True)
    view = props_ui.date_board(None, DAY, world.proj, params, world.roster, feed=with_meta,
                               feed_info=facts())
    assert view.meta == {"posted": 40, "error": True} and view.books_by_game == {}


def test_a_paste_is_the_board_when_the_feeds_tables_could_not_be_compared_b17(
        world, params, tokens):
    """Build amendment B17. The feed holds lines it could not compare
    (props.FeedTables.failed) and the visitor has a paste for the date: the
    paste's own board, exactly as with the feed off (date_board with no
    feed), marked paste_only (the page puts B17's note above it). Without a
    paste, or with no simulator parameters (the paste cannot be compared
    either), the explained empty board as built (the test above)."""
    failed = props.FeedTables(failed=True)
    token = _paste(world, tokens, NINE)
    off = props_ui.date_board(token, DAY, world.proj, params, world.roster)
    assert not off.paste_only and off.matched == PF.PASTED and off.feed is None
    view = props_ui.date_board(token, DAY, world.proj, params, world.roster, feed=failed,
                               feed_info=facts())
    assert view.paste_only and view.feed is None and not view.meta.get("error")
    for got, want in zip((view.shown, view.hidden, view.thin, view.void),
                         (off.shown, off.hidden, off.thin, off.void)):
        pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True))
    assert set(view.meta) == set(off.meta) and view.books_by_game == {} and view.missing == {}
    for k, v in off.meta.items():
        if isinstance(v, pd.DataFrame):
            pd.testing.assert_frame_equal(view.meta[k], v)
        else:
            assert view.meta[k] == v, k
    assert props_ui.paste_only_note_html() == (
        '<div class="bv-note">The odds feed\'s lines could not be compared just now, so your '
        'pasted lines are shown.</div>')
    # No paste (no token, or a paste with no line for the date): as built.
    view = props_ui.date_board(None, DAY, world.proj, params, world.roster, feed=failed,
                               feed_info=facts())
    assert not view.paste_only and view.meta.get("error")
    other_day = props_ui.date_board(token, DAY + dt.timedelta(days=1), world.proj, params,
                                    world.roster, feed=failed, feed_info=facts())
    assert not other_day.paste_only and other_day.meta.get("error")


def test_the_break_even_note_reads_the_boards_book_when_no_line_is_shown_l3_8():
    """Build 4's review, L3-8: a DraftKings board whose lines are all hidden
    (or unmatched) shows no row, and its reading note used to fall back to
    PrizePicks' Power Play range. It reads the books of the board's own
    lines now (amendment C8: the hidden, thin and void ones); with no book
    (the feed off) the wording is today's."""
    hidden = ref("draftkings")
    empty = hidden.iloc[0:0]
    meta = {"posted": 1, "matched": 1}
    dk = props_ui.notes_html(meta, empty, hidden, empty, empty, facts())[-1]
    assert "A sportsbook side at -110 needs about 52.4% to break even." in dk
    assert "54.7%" not in dk
    ud = props_ui.notes_html(meta, empty, ref("underdog"), empty, empty, facts())[-1]
    assert "Underdog's payouts are not modelled here" in ud and "54.7%" not in ud
    pp = props_ui.notes_html(meta, empty, ref("prizepicks"), empty, empty, facts())[-1]
    assert "a leg has to hit 54.7% to 58.5% just to break even." in pp
    void = props_ui.notes_html(meta, empty, empty, empty, ref("betmgm"), facts())[-1]
    assert "52.4%" in void and "54.7%" not in void
    assert props_ui.breakeven_text(empty) == props_ui.breakeven_text(None) == (
        ": a leg has to hit 54.7% to 58.5% just to break even.")
    assert props_ui.breakeven_text(empty, "fanduel") == (
        ". A sportsbook side at -110 needs about 52.4% to break even.")
    assert props_ui.breakeven_text(empty, "nope") == props_ui.breakeven_text(empty)
    assert props_ui.breakeven_text(empty, ("underdog", "fanduel")) == (
        ". Underdog's payouts are not modelled here, so no break-even is shown for its lines. A "
        "sportsbook side at -110 needs about 52.4% to break even.")
    assert props_ui.breakeven_text(empty, ("nope", 5)) == props_ui.breakeven_text(empty, 5) == \
        props_ui.breakeven_text(empty)


# -- The per-source wording with the feed (design 2.6, m8) -----------------------------------
def test_the_notes_follow_where_each_line_came_from():
    rows = U.frame([feed_row("draftkings", 24.5)])
    void = U.frame([feed_row("draftkings", 8.5, player="Vee Void", person_id=1009)])
    meta = {"posted": 4, "matched": 2, "unmatched": 2, "ambiguous": 0, "n_refused": 0,
            "unprojected": 0, "unmatched_names": ["Feed Spelling", "Paste Typo"],
            "ambiguous_names": ["Two Names", "Jalen Williams"]}
    info = facts(counts={"feed": 3, "pasted": 1}, paste_unmatched=["Paste Typo"],
                 paste_ambiguous=["Jalen Williams"])
    notes = [visible(n) for n in props_ui.notes_html(meta, rows, None, None, void, info)]
    assert notes[0] == "4 lines on file for this date (3 from the odds feed, 1 pasted): 2 matched, 2 unmatched."
    assert ("Not matched to a current roster: Feed Spelling. The odds feed spells these differently "
            "from the NBA roster; they are added to the name list as they are found.") in notes
    assert ("Not matched to a current roster: Paste Typo. Check the spelling, or the player may not "
            "be on an NBA roster yet.") in notes
    assert "More than one player by that name: Two Names." in notes
    assert ("More than one player by that name: Jalen Williams. A paste that carries the team (the "
            "board's TEAM - POS line) resolves it.") in notes
    assert ("1 line void: Vee Void listed out or marked out; a sportsbook's own rule decides what "
            "happens to a bet on a player who does not play.") in notes
    assert notes[-1].endswith("not as an edge. A sportsbook side at -110 needs about 52.4% to "
                              "break even. Not betting advice.")
    one = props_ui.notes_html({"posted": 1, "matched": 1}, rows, None, None, None,
                              facts(counts={"feed": 1, "pasted": 0}))[0]
    assert visible(one) == "1 line on file for this date (from the odds feed): 1 matched."
    pasted = props_ui.notes_html({"posted": 2, "matched": 2}, ref("prizepicks", 24.5), None, None,
                                 None, facts(counts={"feed": 0, "pasted": 2}))[0]
    assert visible(pasted).startswith("2 lines on file for this date (pasted): ")


def test_the_break_even_note_is_written_per_book_on_screen(monkeypatch):
    pp = props_ui.breakeven_text(ref("prizepicks", 24.5))
    assert pp == ": a leg has to hit 54.7% to 58.5% just to break even."
    assert props_ui.breakeven_text(ref("underdog", 24.5)) == (
        ". Underdog's payouts are not modelled here, so no break-even is shown for its lines.")
    assert props_ui.breakeven_text(ref("betmgm", 24.5)) == (
        ". A sportsbook side at -110 needs about 52.4% to break even.")
    both = pd.concat([ref("prizepicks", 24.5), ref("fanduel", 24.5)], ignore_index=True)
    assert props_ui.breakeven_text(both) == (
        ". A PrizePicks leg has to hit 54.7% to 58.5% just to break even. A sportsbook side at "
        "-110 needs about 52.4% to break even.")
    assert props_ui.breakeven_text(None) == pp
    monkeypatch.setattr(props_ui, "RG_LINE", "Gamble Responsibly.")
    last = props_ui.notes_html({"posted": 1, "matched": 1}, ref(), None, None, None, facts())[-1]
    assert visible(last).endswith("Not betting advice. Gamble Responsibly.")


def test_the_prizepicks_footnotes_show_only_for_prizepicks_rows():
    demon = dict(odds_type="demon", lean="More")
    for book in ("underdog", "draftkings"):
        rows = ref(book, 24.5, **demon)
        assert props_ui.odds_footnote(rows) == "" and props_ui.reboot_note(rows) == ""
    rows = ref("prizepicks", 24.5, **demon)
    assert "Demons on this board" in props_ui.odds_footnote(rows) and props_ui.reboot_note(rows)


def test_the_prompts_and_the_none_matched_line_with_the_feed():
    assert props_ui.NO_LINES_SLATE_FEED == (
        "<b>No lines on this page yet.</b> On game days the board fills from the odds feed from "
        "about 9 AM ET. You can also paste a PrizePicks board with <b>Update lines</b> in the top bar.")
    assert props_ui.NO_LINES_GAME_FEED == (
        "<b>No lines for this game yet.</b> On game days they load from the odds feed from about "
        "9 AM ET, and each player's lines open under his team. You can also paste a PrizePicks "
        "board with <b>Update lines</b> in the top bar.")
    assert "<b>3 lines on file for March 12, but none matched" in props_ui.none_matched_html(
        3, "March 12", True)
    assert props_ui.after_tip_note_html("lost_in_restart") == (
        "<b>The odds feed's pre-tip lines for this game were not kept.</b> The site restarted "
        "after its last update. A pasted board still shows here.")
    assert props_ui.after_tip_note_html("none_posted") == \
        "<b>No lines were posted for this game before tip.</b>"


FEED_BOARD_SCRIPT = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(HERE)!r})
import test_props_ui_feed as T
from bucketlib import props_ui
shown = T.lines(("draftkings", 25.5), ("fanduel", 25.5, {{"fetched_at": T.NINE}}),
                ("prizepicks", 27.5, {{"lean": "Less", "gap": -1.2}}))
hidden = T.lines(("betmgm", 25.5, {{"lean": "Even", "gap": 0.8}}))
info = T.facts()
props_ui.render_board(shown, hidden, None, None, "March 12", meta={{"posted": 4, "matched": 4}},
                      feed=info)
props_ui.render_board(shown.iloc[:0], None, None, None, "PHI at DET",
                      meta={{"posted": 2, "error": True}}, key="bv_b2", feed=info)
props_ui.render_player_lines_block(shown, hidden, info)
"""


def test_the_feed_board_renders_the_grid_the_legend_and_the_feeds_words():
    """render_board with the feed: the grid inside the keyed container
    bv_board_grid (BOOKS_CSS's guard), the legend line right after it, the
    grid's caption, the error board's words; the per-player grid; every
    markdown ASCII with the sentinel only as a whole node."""
    at = _page(FEED_BOARD_SCRIPT)
    md = [m.value for m in at.markdown]
    assert md.count(props_ui.BOOKS_CSS.strip()) == 3 and md.count(props_ui.PROPS_CSS.strip()) == 3
    (slate,) = [m for m in md if m.startswith('<div class="bv-grid-wrap" tabindex="0" role="region" '
                                              'aria-label="Lines by book">')]
    inside = GP.keyed_markdown(at, "bv_board_grid")
    assert inside is not None and slate in inside and props_ui.grid_legend_html() not in inside
    assert GP.keyed_markdown(at, "bv_b2_grid") is None              # the error board draws no grid
    assert md.index(props_ui.grid_legend_html()) == md.index(slate) + 1
    (row,) = GP.rows(slate)
    assert [c["cls"] for c in row["cells"].values()] == ["over", "over", "plain", "under", "none"]
    assert row["cells"]["FD"]["title"].endswith("Line as of 9:02 AM ET.")
    assert [str(c.value) for c in at.caption] == [props_ui.GRID_CAPTION.format(k=1, n=1)]
    assert any("The odds feed's lines are still in memory; try reloading the page." in m for m in md)
    (player,) = [m for m in md if 'aria-label="Ann Able: lines by book"' in m]
    assert GP.rows(player)[0]["cells"] == row["cells"]
    for m in md:
        assert dash_ok(m) and all(ord(ch) < 128 or ch == SENTINEL for ch in m)


# -- The paste caption (design 2.6) -----------------------------------------------------------
CAPTION = f"""
import sys, datetime as dt
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(HERE / "fixtures" / "serve")!r})
import serve_harness as H
from bucketlib import props_ui
w = H.load()
props_ui.render_update_lines_popover(dt.date(2026, 3, 12), w.sched, w.roster, now=H.MORNING,
                                     dates=[H.DATE], feed_on={{feed_on}})
"""
LAST_SENTENCE = ("Nothing here fetches from PrizePicks: their API blocks server requests and "
                 "their terms prohibit automated access.")


def _caption(feed_on: bool) -> str:
    at = _page(CAPTION.replace("{feed_on}", str(feed_on)))
    (text,) = [str(c.value) for c in at.caption if LAST_SENTENCE in str(c.value)]
    return text


def test_the_paste_caption_says_the_board_comes_from_the_feed_only_with_the_feed_on():
    """design 2.6: with the feed on, the feed's two sentences go just before
    the caption's last sentence, which stays; with it off, today's caption."""
    on, off = _caption(True), _caption(False)
    assert on.endswith(props_ui.FEED_CAPTION + LAST_SENTENCE)
    # Build amendment B28: the pasted lines that win are named, as in the merge line.
    assert props_ui.FEED_CAPTION == (
        "PrizePicks lines on the board come from a licensed odds feed; pasting is the backup. "
        "For the same player and stat, the feed's line is shown unless yours is newer, typed, "
        "Demon or Goblin. ")
    assert on.replace(props_ui.FEED_CAPTION, "") == off
    assert "licensed odds feed" not in off and off.endswith(
        "Clear all removes them from your link and from this browser. " + LAST_SENTENCE)


# -- Build 6, stage S12: build amendments B24 to B28 ------------------------------------------
def test_the_void_note_names_each_books_own_rule_b26():
    """Build amendment B26 (build 4's review, L3-7, which pinned Underdog's
    board giving PrizePicks' sentence). On Underdog rows the void note says
    that Underdog's own rule decides what happens to a pick on a player who
    does not play; PrizePicks rows keep today's sentence (the feed-off golden
    holds it byte for byte) and sportsbook rows design 2.6's: one note per
    kind of book on screen, PrizePicks first, Underdog's naming no other
    book."""
    def notes(*books):
        void = U.frame([feed_row(b, 8.5, player=f"Vee {b}", person_id=2000 + i)
                        for i, b in enumerate(books)])
        return [visible(n) for n in props_ui._void_note(void)]
    assert notes("underdog") == [
        "1 line void: Vee underdog listed out or marked out; Underdog's own rule decides what "
        "happens to a pick on a player who does not play."]
    assert notes("prizepicks") == [
        "1 line void: Vee prizepicks listed out or marked out. PrizePicks removes a pick on a "
        "player who does not play a second."]
    for book in ("draftkings", "fanduel", "betmgm"):
        assert notes(book) == [
            f"1 line void: Vee {book} listed out or marked out; a sportsbook's own rule decides "
            "what happens to a bet on a player who does not play."], book
    mixed = notes("betmgm", "underdog", "prizepicks", "draftkings")
    assert [n.split(":")[0] for n in mixed] == ["1 line void", "1 line void", "2 lines void"]
    assert "Vee prizepicks" in mixed[0] and "PrizePicks removes" in mixed[0]
    assert "Vee underdog" in mixed[1] and "Underdog's own rule" in mixed[1]
    assert "PrizePicks" not in mixed[1] and "sportsbook" not in mixed[1]
    assert mixed[2].startswith("2 lines void: Vee betmgm, Vee draftkings ")
    pasted = U.frame([U.row(book=None, player="Old Paste", void_note=True)])   # a row before the feed
    assert "PrizePicks removes a pick" in visible(props_ui._void_note(pasted)[0])


def test_the_books_missing_lines_after_tip_and_for_a_book_the_site_does_not_buy_b24_b25():
    """Build amendments B24 and B25 in layout C (amendment C9). The slate's
    note names only games that have not tipped (the page passes the tipped
    ones) and says nothing once all have; a book BUCKETVALUE_ODDS_BOOKS
    leaves out (not in feed_info's "books") reads that its lines are not
    loaded on this site, tipped or not, several sharing one sentence with
    "and"; on both pages the loaded books' sentence comes first. Nothing
    points at a picker or names a direction."""
    four = ("draftkings", "fanduel", "prizepicks", "underdog")         # BetMGM left out
    note = props_ui.books_missing_html
    assert visible(note({"prizepicks": [EARLY, LATE]}, LABELS, tipped={EARLY})) == (
        "No PrizePicks lines yet for DEN at SAS.")
    assert visible(note({"prizepicks": [EARLY, LATE]}, LABELS, tipped=[LATE])) == (
        "No PrizePicks lines yet for PHI at DET.")
    assert note({"prizepicks": [EARLY, LATE]}, LABELS, tipped=(EARLY, LATE)) == ""
    gone = '<div class="bv-note"><b>BetMGM lines are not loaded on this site.</b></div>'
    assert note({"betmgm": [EARLY, LATE]}, LABELS, loaded=four) == gone
    assert note({"betmgm": [EARLY, LATE]}, LABELS, loaded=four, tipped={EARLY, LATE}) == gone
    assert note({"betmgm": [], "draftkings": []}, LABELS, loaded=four) == ""
    assert note({"betmgm": [EARLY], "underdog": [EARLY]}, LABELS,
                loaded=("draftkings", "fanduel", "prizepicks")) == (
        '<div class="bv-note"><b>BetMGM and Underdog lines are not loaded on this site.</b></div>')
    mixed = {"betmgm": [EARLY, LATE], "prizepicks": [LATE]}          # amendment C9's mixed case
    assert note(mixed, LABELS, loaded=four) == (
        '<div class="bv-note"><b>No PrizePicks lines yet for DEN at SAS.</b> <b>BetMGM lines are '
        "not loaded on this site.</b></div>")
    game = props_ui.books_missing_game_html
    assert f'<div class="bv-note">{game(["betmgm", "prizepicks"], loaded=four)}</div>' == (
        '<div class="bv-note"><b>No PrizePicks lines for this game yet.</b> <b>BetMGM lines are not '
        "loaded on this site.</b></div>")
    assert f'<div class="bv-note">{game(["prizepicks", "betmgm"], tipped=True, loaded=four)}</div>' == (
        '<div class="bv-note"><b>The odds feed did not load PrizePicks lines for this game before '
        "tip.</b> <b>BetMGM lines are not loaded on this site.</b></div>")
    assert game(["betmgm"], loaded=four) == game(["betmgm"], tipped=True, loaded=four) == (
        "<b>BetMGM lines are not loaded on this site.</b>")
    assert game(["betmgm", "underdog"], loaded=("draftkings", "fanduel", "prizepicks")) == (
        "<b>BetMGM and Underdog lines are not loaded on this site.</b>")
    info = facts(missing={"prizepicks": [EARLY, LATE]})
    assert "No PrizePicks lines yet for DEN at SAS." in visible(
        props_ui.feed_notice_html(info, DAY, LABELS, tipped={EARLY}))
    assert "No PrizePicks lines yet" not in props_ui.feed_notice_html(info, DAY, LABELS,
                                                                      tipped={EARLY, LATE})
    narrow = facts(missing={"betmgm": [EARLY, LATE]}, books=four)
    assert "BetMGM lines are not loaded on this site." in visible(
        props_ui.feed_notice_html(narrow, DAY, LABELS))
    for text in (note({"prizepicks": [LATE]}, LABELS), game(["prizepicks"]),
                 game(["prizepicks"], tipped=True), gone):
        for bad in (" above", " below", "Pick another"):
            assert bad not in text, (bad, text)


# -- Build 7, stage S13: build amendments B29 and B30 -----------------------------------------
def test_after_tip_a_book_with_no_lines_reads_the_feed_did_not_load_them_b29():
    """Build amendment B29 (build 6's review, S12-R3) in layout C. After its
    game's tip the Game page's note for the books with no lines for it says
    the odds feed did not load their lines before tip, for every book and
    for several together: true whether a book posted none or the site had
    stopped asking for it. Before tip the "yet" sentence stands; a book
    BUCKETVALUE_ODDS_BOOKS leaves out keeps B25's not-loaded sentence,
    tipped or not; a junk book key is dropped, never echoed; and B18's
    game-level sentences (a game with no feed lines at all) are unchanged."""
    game = props_ui.books_missing_game_html
    for book in props.BOOK_ORDER:
        title = props.BOOK_TITLE[book]
        assert game([book], tipped=True) == (
            f"<b>The odds feed did not load {title} lines for this game before tip.</b>"), book
        assert "were posted" not in game([book], tipped=True), book
        assert game([book]) == game([book], tipped=False) == (
            f"<b>No {title} lines for this game yet.</b>"), book
        others = tuple(b for b in props.BOOK_ORDER if b != book)
        for tipped in (False, True):
            assert game([book], tipped=tipped, loaded=others) == (
                f"<b>{title} lines are not loaded on this site.</b>"), (book, tipped)
        assert game([book], tipped=True, loaded=props.BOOK_ORDER) == game([book], tipped=True), book
    assert game(["underdog", "prizepicks"], tipped=True) == (
        "<b>The odds feed did not load PrizePicks or Underdog lines for this game before tip.</b>")
    assert game(list(reversed(props.BOOK_ORDER)), tipped=True) == (
        "<b>The odds feed did not load DraftKings, FanDuel, BetMGM, PrizePicks or Underdog lines "
        "for this game before tip.</b>")
    assert game(["x<b>", "prizepicks"], tipped=True) == game(["prizepicks"], tipped=True)
    assert game(["x<b>"], tipped=True) == game("x<b>", tipped=True) == ""
    assert game("prizepicks") == game(["prizepicks"])
    assert props_ui.after_tip_note_html("none_posted") == (
        "<b>No lines were posted for this game before tip.</b>")
    for reason in ("not_yet", "held", "paused", None):
        assert props_ui.after_tip_note_html(reason) == (
            "<b>The odds feed did not load lines for this game before tip.</b>"), reason


def test_the_settled_note_claims_a_void_on_prizepicks_rows_only_b30():
    """Build amendment B30 (build 6's review, open question 3). B26 leaves
    what happens to an Underdog pick on a player who does not play to
    Underdog's own rule, so a settled Underdog line's note claims no void:
    it reads DNP_TEXT, "Did not play", as a sportsbook line's does. A
    PrizePicks line keeps VOID_TEXT, "Did not play, pick voided", and so
    does a row with no book (a pasted line, as every line was before the
    feed: the feed-off golden holds that HTML byte for byte). With the feed
    the lines go through the per-player grid, whose note names each book's
    rule in one row (amendment C7) and the books an exact landing was on."""
    assert (props_ui.VOID_TEXT, props_ui.DNP_TEXT) == ("Did not play, pick voided", "Did not play")
    gone = {"actual": None, "settle": "void", "final": True}
    exact = {"actual": 6.0, "settle": "exact", "final": True}

    def notes(rows, feed=True, settled=gone):
        recs = [{**r, **settled} for r in (rows if isinstance(rows, list) else [rows])]
        html = props_ui.render_player_lines(recs, None, facts() if feed else None)
        return [visible(n) for n in re.findall(r'<div class="bv-pl-note">(.*?)</div>', html)]
    assert notes(feed_row("underdog", 6.0)) == ["Did not play"]
    assert notes(feed_row("prizepicks", 6.0)) == ["Did not play, pick voided"]
    no_book = [U.row(book=None), U.row(book=" "), {k: v for k, v in U.row().items() if k != "book"}]
    for row in no_book:
        assert notes(row, feed=False) == ["Did not play, pick voided"], row.get("book", "no key")
    for book in SPORTSBOOKS:
        assert notes(feed_row(book, 6.0)) == ["Did not play"], book
    for book in props.BOOK_ORDER:
        assert notes(feed_row(book, 6.0), settled=exact) == [
            f"6: landed on the line ({props.BOOK_CODE[book]})"], book
    assert notes(U.row(book=None, line=6.0), feed=False, settled=exact) == ["6: landed on the line"]
    both = notes([feed_row("prizepicks", 6.0), feed_row("draftkings", 6.0)])
    assert both == ["PP: Did not play, pick voided. DK: Did not play"]
    assert " ".join(both).count("pick voided") == 1
    note = props_ui._settle_note
    assert note({"book": "underdog", "settle": "void"}) == '<div class="bv-pl-note">Did not play</div>'
    for row in ({"book": "prizepicks", "settle": "void"}, {"settle": "void"}):
        assert note(row) == '<div class="bv-pl-note">Did not play, pick voided</div>', row
    assert note({"book": "underdog", "settle": None}) == note({"settle": None}) == ""


# -- Build 8's fix pass: the review of stages S14 and S15 (R8-01) ------------------------------
TOP_SCRIPT = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(HERE)!r})
import test_props_ui_feed as T
import ui_rows as U
from bucketlib import props_ui
rows = [T.feed_row("draftkings", 20.5, person_id=9001 + i, player=f"P{{i:02d}} Test",
                   confidence=0.30 - 0.01 * i) for i in range(27)]
rows += [T.feed_row("fanduel", 7.5, person_id=9001, player="P00 Test", stat=stat,
                    stat_label=label, confidence=0.05)
         for stat, label in (("reb", "Rebounds"), ("ast", "Assists"))]
props_ui.render_board(U.frame(rows), None, None, None, "March 12",
                      meta={{"posted": len(rows), "matched": len(rows)}}, feed=T.facts())
"""


def test_with_the_feed_the_board_shows_its_top_players_one_row_each_and_show_all_every_row_r8_01():
    """D3 with the feed on (build 8's review, R8-01: the page world's feed
    buys points only, 13 players with one row each, so it never reaches the
    top 25's cut). 27 players have one DraftKings points line each, P00 the
    strongest and P26 the weakest, and P00 also has FanDuel rebounds and
    assists lines that rank below every points line. The default view is
    the top 25 players, one row each (his best-ranked row, in rank order),
    P00's row carrying "+2 more", under the grid's caption for 25 of 27
    players. Show all draws all 29 rows, with no "+N more" and no caption,
    and the stat family pills, with the feed on as with it off, leave only
    that family's rows."""
    at = _page(TOP_SCRIPT)
    rows = GP.grid(at)
    assert [r["player"] for r in rows] == [f"P{i:02d} Test" for i in range(25)]
    assert [r["stat"] for r in rows] == ["Points"] * 25
    assert [r["more"] for r in rows] == [2] + [0] * 24
    assert [str(c.value) for c in at.caption] == [props_ui.GRID_CAPTION.format(k=25, n=27)]
    every = GP.grid(at, show_all=True)
    assert len(every) == 29 and len({r["player"] for r in every}) == 27
    assert sorted(r["stat"] for r in every if r["player"] == "P00 Test") == [
        "Assists", "Points", "Rebounds"]
    assert all(r["more"] == 0 for r in every) and not at.caption
    (fam,) = [b for b in at.button_group if b.key == "bv_board_fam"]
    assert list(fam.options) == ["All", "Points", "Rebounds", "Assists"]
    fam.set_value("Rebounds").run()
    assert not at.exception, [e.value for e in at.exception]
    assert [(r["player"], r["stat"]) for r in GP.grid(at)] == [("P00 Test", "Rebounds")]
