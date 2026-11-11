"""bucketlib.props_ui with the licensed odds feed (design 2.6 and 14 step 16;
layout B: amendments A1 and A3 to A8; build amendments N1 to N3): BoardView's
new fields, date_board's feed path, the book pills, the Other books cell and
its legend, the "<CODE> line" header and the side words, the feed's notices
(feed_notice_html and the A6 notes), Biggest discrepancies naming the book,
and the per-source wording, as HTML unit tests (AppTest where a widget is
involved).

With the feed off every function must give exactly the HTML it gave before
the feed: tests/fixtures/props_ui/feed_off_golden.json holds what the pre-S8
props_ui rendered for tests/ui_rows.py's rows (captured from build 2's tree
before props_ui.py was edited), and it is compared byte for byte.

The rows are hand-built (tests/ui_rows.py) or, end to end, the serve
fixture's 2026-03-12 slate with the CONSTRUCTED vendor fixtures through the
real parser (tests/test_props_feed.py's helpers) and the real paste path.
Every time is a fixed past instant passed in, never a clock: a guard test
makes props_ui's own clock raise while the feed's text is built.
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
from bucketlib import props, props_ui, simprov
from bucketlib.util import ET, SENTINEL

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import test_props_feed as PF  # noqa: E402  (the feed frame through the real parser)
import ui_rows as U  # noqa: E402

GOLDEN = json.loads((HERE / "fixtures" / "props_ui" / "feed_off_golden.json").read_text())
DAY = dt.date(2026, 3, 12)
EARLY, LATE = U.EARLY, U.LATE
NINE, NOON, LAST_NIGHT = U.NINE, U.NOON, U.LAST_NIGHT
LABELS = {EARLY: "PHI at DET", LATE: "DEN at SAS"}       # the schedule's own orientation (N2)
NAN = float("nan")


def visible(markup: str) -> str:
    """What a reader sees: tags dropped with their attributes, entities decoded."""
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", markup)).split())


def cells(html: str) -> list[str]:
    """The Other books cell of each ledger or line row (never the header's)."""
    return re.findall(r'<div class="bv-(?:ledger|pl)-row">.*?<span class="ob">(.*?)</span></div>', html)


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


def ctx(*specs) -> pd.DataFrame:
    """Context rows: (book, line, fetched_at) for the reference row's key,
    each with a lean, a gap and (sportsbooks) a price of its own."""
    return U.frame([feed_row(b, x, at=t, lean="Less", gap=-0.4, model=x - 0.4)
                    for b, x, t in specs])


FOUR = ctx(("prizepicks", 24.5, NOON), ("underdog", 24.5, NOON), ("fanduel", 25.5, NINE),
           ("betmgm", 25.5, NOON))


def facts(book="draftkings", **kw) -> dict:
    out = {"book": book, "state": "ok", "asof": NINE, "loading": 0, "generation": 2, "lines": 40,
           "per_game": {EARLY: {"asof": NOON, "books": {}, "pretip": False},
                        LATE: {"asof": NINE, "books": {}, "pretip": False}},
           "report": {}, "today": DAY}
    out.update(kw)
    return out


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
    assert props_ui.ledger_html(rows, {1001: 2}, None, None) == g["ledger"]
    assert props_ui.render_player_lines(rows, None, None) == g["player_lines"]
    assert "\n".join(props_ui.notes_html(U.meta_full(), shown, hidden, thin, void, None)) == g["notes"]
    assert props_ui.discrepancies_window(U.gap_rows(), "March 12", None) == g["discrepancies_window"]
    assert props_ui.discrepancies_html(U.gap_rows(), "March 12", None) == g["discrepancies"]
    assert props_ui.none_matched_html(3, "March 12", False) == g["none_matched"]
    # Without a feed no cell, no header code, no legend, no books CSS.
    for html in (g["ledger"], g["player_lines"]):
        assert 'class="ob"' not in html and "has-books" not in html and "Other books" not in html


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
    assert view.feed is None and view.context.empty and view.ref_by_game == {}
    assert (view.missing_games, view.paste_won, view.feed_won, view.paste_times) == ([], 0, 0, {})


# -- The Other books cell (amendment A5) ------------------------------------------------
def test_the_other_books_cell_lists_the_other_four_in_book_order():
    """DraftKings picked: PrizePicks, Underdog, FanDuel and BetMGM, in the
    pills' order, each as its code and number; FanDuel's line is three hours
    older than the game's latest pull, so it shows its own time (m23)."""
    html = props_ui.ledger_html(ref(), context=FOUR, feed=facts())
    (cell,) = cells(html)
    assert cell == ('<abbr title="PrizePicks">PP</abbr> 24.5, <abbr title="Underdog">UD</abbr> 24.5, '
                    '<abbr title="FanDuel">FD</abbr> 25.5 (9:02 AM), '
                    '<abbr title="BetMGM">MGM</abbr> 25.5')
    assert visible(cell) == "PP 24.5, UD 24.5, FD 25.5 (9:02 AM), MGM 25.5"
    # PrizePicks picked: the four others start with Underdog, never PrizePicks itself.
    pp = ctx(("underdog", 24.5, NOON), ("draftkings", 25.5, NOON), ("fanduel", 25.5, NOON),
             ("betmgm", 26.5, NOON))
    html = props_ui.ledger_html(ref("prizepicks", 24.5), context=pp, feed=facts("prizepicks"))
    assert visible(cells(html)[0]) == "UD 24.5, DK 25.5, FD 25.5, MGM 26.5"


def test_a_book_with_no_line_is_left_out_and_none_gives_the_whole_node_sentinel():
    two = FOUR[FOUR["book"].isin(["underdog", "betmgm"])]
    assert visible(cells(props_ui.ledger_html(ref(), context=two, feed=facts()))[0]) == \
        "UD 24.5, MGM 25.5"
    for context in (None, FOUR.iloc[:0], FOUR.assign(person_id=9999), FOUR.assign(stat="reb"),
                    FOUR.assign(game_date="2026-03-13")):
        html = props_ui.ledger_html(ref(), context=context, feed=facts())
        assert cells(html) == [SENTINEL] and dash_ok(html)
    # The row's own book in the context (it cannot be "other") is left out too.
    own = pd.concat([FOUR, ctx(("draftkings", 26.5, NOON))], ignore_index=True)
    assert "DK" not in visible(cells(props_ui.ledger_html(ref(), context=own, feed=facts()))[0])


def test_a_books_own_time_shows_when_its_line_is_over_a_minute_older():
    """m23: against the game's latest pull (the page's per_game facts, else
    the newest fetch among the board's rows of that game); a minute is not
    over a minute; a stamp from another ET day carries its date."""
    def fd_at(at, feed):
        c = ctx(("fanduel", 25.5, at))
        return visible(cells(props_ui.ledger_html(ref(), context=c, feed=feed))[0])
    minute = pd.Timedelta(minutes=1)
    assert fd_at(NOON - minute, facts()) == "FD 25.5"
    assert fd_at(NOON - minute - pd.Timedelta(seconds=1), facts()) == "FD 25.5 (11:58 AM)"
    assert fd_at(NINE, facts()) == "FD 25.5 (9:02 AM)"
    assert fd_at(LAST_NIGHT, facts()) == "FD 25.5 (9:41 PM, Mar 11)"
    # No per_game facts: the reference row's own fetch (noon) is the latest.
    assert fd_at(NINE, {"book": "draftkings", "today": DAY}) == "FD 25.5 (9:02 AM)"
    assert fd_at(NINE, {"book": "draftkings", "today": DAY, "per_game": {EARLY: {"asof": NINE}}}) == \
        "FD 25.5"
    # A context row with no fetch time shows no time.
    assert fd_at(None, facts()) == "FD 25.5"


def test_the_cell_carries_no_lean_colour_arrow_gap_or_price():
    """Amendment A5: only the code and the line. The context rows have a
    lean, a gap, a model value and a price each, and none of it shows."""
    html = props_ui.ledger_html(ref(), context=FOUR, feed=facts())
    (cell,) = cells(html)
    assert re.findall(r'<span class="(ob[^"]*)">', html) == ["ob", "ob"]     # the header's, the row's
    assert re.findall(r"<(\w+)", cell) == ["abbr"] * 4                 # nothing but the codes
    assert re.findall(r'<abbr title="([^"]*)">', cell) == ["PrizePicks", "Underdog", "FanDuel", "BetMGM"]
    text = re.sub(r'title="[^"]*"', "", cell).lower()        # the books' names said once
    for bad in ("over", "under", "more", "less", "class=", "style", "color", "lean", "diff",
                "&uarr;", "&darr;", "&#8593;", "&#8595;", "&#9650;", "&#9660;", "-115", "-105",
                "-0.4", "+", "25.1", "24.1", "%"):
        assert bad not in text, bad
    assert all(ord(ch) < 128 for ch in cell)


def test_the_cell_names_books_from_our_own_titles_never_the_vendors():
    """Book names render from props.BOOK_TITLE and BOOK_CODE: a row's vendor
    title is never shown, and a book outside the five is left out."""
    c = U.frame([feed_row("underdog", 24.5, book_title="Underdog Fantasy <b>"),
                 U.row(book="evilbook", book_title="<script>x</script>", origin="feed",
                       fetched_at=NOON, line=25.5)])
    cell = cells(props_ui.ledger_html(ref(), context=c, feed=facts()))[0]
    assert cell == '<abbr title="Underdog">UD</abbr> 24.5'


def test_the_ledger_and_the_game_rows_get_the_cell_its_header_and_the_legend():
    html = props_ui.ledger_html(ref(), context=FOUR, feed=facts())
    assert html.startswith('<div class="bv-ledger has-books"><div class="bv-ledger-head">')
    assert ('<span class="num">DK line</span><span class="num">Gap</span><span class="lean">Lean</span>'
            '<span class="ob">Other books</span></div>') in html
    row = re.search(r'<div class="bv-ledger-row">(.*?)</div>', html).group(1)
    assert re.findall(r'<span class="(nm|st|num|num line|num diff over|lean over|ob)"', row) == \
        ["nm", "st", "num", "num line", "num diff over", "lean over", "ob"]
    lines = props_ui.render_player_lines(ref(), FOUR, facts())
    assert lines.startswith('<div class="bv-pl has-books"><div class="bv-pl-head">')
    assert "Other books" in lines and visible(cells(lines)[0]).startswith("PP 24.5, UD 24.5")
    after = props_ui.render_player_lines([{**feed_row("draftkings", 25.5), "actual": 27.0,
                                           "settle": None, "final": True}], FOUR, facts())
    assert after.startswith('<div class="bv-pl has-actual has-books">')
    assert re.search(r'<span class="num act">27</span><span class="ob">', after)
    assert props_ui.books_legend_html() == ('<div class="bv-ob-legend">PP PrizePicks, UD Underdog, '
                                            'DK DraftKings, FD FanDuel, MGM BetMGM.</div>')


def test_the_cells_css_wraps_it_to_its_own_line_on_a_phone():
    css = props_ui.BOOKS_CSS
    phone = css[css.index("@media (max-width: 760px)"):]
    assert ".bv-ledger .ob, .bv-pl .ob { grid-column: 1 / -1; }" in phone
    assert ".bv-ledger.has-books .bv-ledger-row" in css and ".bv-pl.has-books .bv-pl-row" in css
    assert props_ui.PROPS_CSS == GOLDEN["functions"]["props_css"]      # today's CSS, untouched
    assert all(ord(ch) < 128 for ch in css)


# -- The header code and the side words (amendment A4) -----------------------------------------
@pytest.mark.parametrize("book, code", [("prizepicks", "PP"), ("underdog", "UD"),
                                        ("draftkings", "DK"), ("fanduel", "FD"),
                                        ("betmgm", "MGM")])
def test_the_line_header_names_the_picked_books_code(book, code):
    rows = ref(book, 24.5)
    assert f'<span class="num">{code} line</span>' in props_ui.ledger_html(rows, feed=facts(book))
    assert f'<span class="num">{code} line</span>' in props_ui.render_player_lines(rows, None, facts(book))
    assert '<span class="num">Line</span>' in props_ui.ledger_html(rows)
    assert '<span class="num">Line</span>' in props_ui.render_player_lines(rows)


def test_a_junk_book_in_the_facts_is_prizepicks():
    assert '<span class="num">PP line</span>' in props_ui.ledger_html(ref(), feed={"book": "x<b>"})
    assert props_ui.selected_book(False) == "prizepicks"


def test_side_words_are_over_and_under_on_a_sportsbook_and_more_and_less_on_pickem():
    """m8: the words follow the row's book; the stored lean and its colour
    class stay More, Less or Even."""
    def lean_cell(book, lean, sides="both"):
        html = props_ui.ledger_html(ref(book, 24.5, lean=lean, sides=sides), feed=facts(book))
        row = html.split('<div class="bv-ledger-row">', 1)[1]
        return re.search(r'<span class="lean ?([a-z]*)">([^<]*)</span>', row).groups(), row
    for book in ("draftkings", "fanduel", "betmgm"):
        assert lean_cell(book, "More")[0] == ("over", "Over")
        assert lean_cell(book, "Less")[0] == ("under", "Under")
        assert lean_cell(book, "Even")[0] == ("", "Even")
        assert '<span class="bv-pp-tag" title="This line offers Over only">Over only</span>' in \
            lean_cell(book, "More", "more")[1]
        assert '<span class="bv-pp-tag" title="This line offers Under only">Under only</span>' in \
            lean_cell(book, "Less", "less")[1]
    for book in ("prizepicks", "underdog"):
        assert lean_cell(book, "More")[0] == ("over", "More")
        assert lean_cell(book, "Less")[0] == ("under", "Less")
        assert "More only</span>" in lean_cell(book, "More", "more")[1]
    # The discrepancy chips and their table say the same words.
    rows = U.frame([feed_row("draftkings", 20.5, person_id=3000 + i, player=f"P{i}", model=22.5 + i,
                             gap=2.0 + i, lean="More") for i in range(8)])
    strip = props_ui.discrepancy_strip_html(props_ui.discrepancies(rows))
    assert strip.count('<span class="gc-lean">Over</span>') == props_ui.DISCREPANCY_CHIPS
    assert strip.count('<td class="l lean over">Over</td>') == 2
    assert "More" not in visible(strip)


def test_per_book_rules_the_tie_the_void_risk_and_a_player_who_did_not_play():
    """design 2.6: the tie tooltip keeps PrizePicks' words on pick'em rows and
    says the book's own rule on a sportsbook row; the void-risk tag (Reboot)
    is PrizePicks' only; a settled sportsbook row reads 'Did not play', and
    since build amendment B30 so does a settled Underdog row (PrizePicks'
    keeps VOID_TEXT)."""
    tie = dict(tie_possible=True, p_push=0.12)
    dk = props_ui._stat_cell(feed_row("draftkings", 6.0, **tie))
    assert ('title="Chance of landing exactly on 6: 12 in 100. On a sportsbook line the book&#x27;s '
            'own rule decides an exact landing."') in dk
    for book in ("prizepicks", "underdog"):
        assert "That lowers the payout tier; it is not a push." in \
            props_ui._stat_cell(feed_row(book, 6.0, **tie))
    bench = dict(lean="More", void_note=True, mu_min=15.0)
    assert props_ui._void_tag(feed_row("prizepicks", 6.0, **bench))
    for book in ("underdog", "draftkings"):
        assert props_ui._void_tag(feed_row(book, 6.0, **bench)) == ""
    gone = {"actual": None, "settle": "void", "final": True}
    for book in ("draftkings", "underdog"):                         # B30: Underdog's too
        assert '<div class="bv-pl-note">Did not play</div>' in \
            props_ui.render_player_lines([{**feed_row(book, 6.0), **gone}], None, facts(book))
    assert f'<div class="bv-pl-note">{props_ui.VOID_TEXT}</div>' in props_ui.render_player_lines(
        [{**feed_row("prizepicks", 6.0), **gone}], None, facts("prizepicks"))


# -- Biggest discrepancies (amendment A8) --------------------------------------------------
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


# -- The feed's notices (design 2.6, 6; A6; N1 to N3) ---------------------------------------
def test_amendment_a6_names_at_most_two_games_then_and_n_more():
    def said(gids, book="prizepicks", labels=LABELS):
        return visible(props_ui.book_missing_html(book, gids, labels))
    more = {**LABELS, "g3": "IND at PHX", "g4": "MIA at BOS"}
    assert said([LATE]) == "No PrizePicks lines yet for DEN at SAS. Pick another book to see theirs."
    assert said([EARLY, LATE]) == ("No PrizePicks lines yet for PHI at DET and DEN at SAS. Pick "
                                   "another book to see theirs.")
    assert said([LATE, EARLY, "g3"], labels=more) == ("No PrizePicks lines yet for DEN at SAS and 2 "
                                                     "more games. Pick another book to see theirs.")
    assert said([LATE, EARLY, "g3", "g4"], "draftkings", more).startswith(
        "No DraftKings lines yet for DEN at SAS and 3 more games.")
    for n in range(1, 6):
        gids = [LATE, EARLY, "g3", "g4", "g5"][:n]
        named = sum(said(gids, labels=more).count(x) for x in more.values())
        assert named <= 2, n
    assert said(["g9"], labels=LABELS) == "No PrizePicks lines yet for 1 game. Pick another book to see theirs."
    assert said([LATE, "g9"]).startswith("No PrizePicks lines yet for DEN at SAS and 1 more game.")
    assert props_ui.book_missing_html("prizepicks", [], LABELS) == ""
    html = props_ui.book_missing_html("prizepicks", [LATE], {LATE: "<b>X</b>"})
    assert "&lt;b&gt;X&lt;/b&gt;" in html and "<b>X</b>" not in html
    assert props_ui.book_missing_game_html("prizepicks") == (
        "<b>No PrizePicks lines for this game yet.</b> Pick another book to see theirs.")
    assert props_ui.book_missing_game_html("betmgm").startswith("<b>No BetMGM lines for this game")


def test_the_notice_names_the_schedules_games_never_the_vendors_teams_n2():
    """N2: the games come from the page's labels (the schedule's away at
    home), in the order the board gives (tip order); nothing of the
    vendor's event teams is read."""
    info = facts("prizepicks", missing_games=[LATE, EARLY], report={"event_home": "DEN"})
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
    info = facts(missing_games=[LATE], feed_won=3,
                 per_game={EARLY: {"asof": NOON, "pretip": False}, LATE: {"asof": NINE, "pretip": True}})
    assert visible(props_ui.feed_notice_html(info, DAY, LABELS, game_id=EARLY)) == \
        "Lines as of 12:00 PM ET."
    assert props_ui.lines_asof_text(info, DAY, LATE) == "Pre-tip lines, as of 9:02 AM ET."
    assert props_ui.lines_asof_text(info, DAY) == "Lines as of 9:02 AM ET."
    assert props_ui.lines_asof_text(info, DAY, "nope") == ""
    assert props_ui.lines_asof_text(None) == ""


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
    info = facts(missing_games=[LATE], feed_won=2, paste_won=1, paste_times=times, loading=1,
                 report={"pulled": [("draftkings", "X", "Points")], "n_pulled": 1})
    for state in ("ok", "stale", "paused", "down", "idle"):
        props_ui.feed_notice_html({**info, "state": state}, DAY, LABELS)
        props_ui.feed_notice_html({**info, "state": state}, DAY, LABELS, game_id=EARLY)
    props_ui.ledger_html(ref(), context=FOUR, feed=info)
    props_ui.render_player_lines(ref(), FOUR, info)
    props_ui.notes_html({"posted": 1, "matched": 1}, ref(), None, None, None, info)
    ft = PF.tables(world, simprov.provider_for(world.proj, params))
    token = _paste(world, tokens, NINE)
    view = props_ui.date_board(token, DAY, world.proj, params, world.roster, feed=ft,
                               feed_info=info, selected="draftkings")
    assert not view.meta.get("error") and len(view.shown)


# -- date_board with the feed (layout B, end to end) -----------------------------------------
@pytest.fixture(scope="module")
def feed_tables(world, params):
    return PF.tables(world, simprov.provider_for(world.proj, params))


def test_the_board_is_the_picked_books_rows_with_the_others_as_context(world, params, feed_tables):
    """The serve fixture's slate through the real parser (test_props_feed's
    counts: PrizePicks 9 lines, all on PHI at DET; DraftKings 10 and 11).
    Each pick's rows are its own book's only; every other book's rows for
    those keys are context; the ledger balances; the header names the book."""
    for book in props.BOOK_ORDER:
        view = props_ui.date_board(None, DAY, world.proj, params, world.roster, feed=feed_tables,
                                   feed_info=facts(book), selected=book)
        rows = pd.concat([view.shown, view.hidden, view.thin, view.void])
        assert set(rows["book"]) == {book}, book
        assert view.meta["posted"] == PF.BOOK_ROWS[book] and not view.meta.get("error")
        assert props.ledger(view.meta, view.shown, view.hidden, view.thin, view.void)["balanced"]
        assert book not in set(view.context["book"]) and set(view.context["origin"]) == {"feed"}
        assert view.feed["book"] == book and view.feed["counts"] == {"feed": PF.BOOK_ROWS[book],
                                                                     "pasted": 0}
        assert view.feed["state"] == "ok" and view.feed["today"] == DAY       # the page's facts kept
        assert view.ref_by_game == {g: book for g in (EARLY, LATE) if (book, g) in PF.POSTED}
        assert view.missing_games == ([LATE] if book == "prizepicks" else [])
        assert f'<span class="num">{props.BOOK_CODE[book]} line</span>' in \
            props_ui.ledger_html(view.shown, context=view.context, feed=view.feed)
    assert props_ui.line_counts_by_game(view.shown) == \
        {str(g): n for g, n in view.shown.groupby("game_id").size().items()}


def test_the_paste_merges_into_prizepicks_newest_wins_and_the_board_says_so(
        world, params, feed_tables, tokens):
    """M5 through date_board: a paste at 11 AM is newer than the 9 AM feed,
    so it wins the 6 shared keys; one from last night loses them. The ledger
    counts the lines on file by where they came from."""
    eleven = pd.Timestamp("2026-03-12T15:00:00Z")
    late = _paste(world, tokens, eleven)
    view = props_ui.date_board(late, DAY, world.proj, params, world.roster, feed=feed_tables,
                               feed_info=facts("prizepicks"), selected="prizepicks")
    assert (view.paste_won, view.feed_won) == (len(PF.SHARED), 0)
    assert view.meta["posted"] == 9 + PF.PASTED - len(PF.SHARED)
    assert view.feed["counts"] == {"feed": 9 - len(PF.SHARED), "pasted": PF.PASTED}
    assert view.missing_games == [] and view.feed["pasted"] == PF.PASTED
    note = props_ui.notes_html(view.meta, view.shown, view.hidden, view.thin, view.void, view.feed)[0]
    assert visible(note).startswith(f"14 PrizePicks lines on file for this date (3 from the odds "
                                    f"feed, {PF.PASTED} pasted): ")
    old = _paste(world, tokens, LAST_NIGHT)
    view = props_ui.date_board(old, DAY, world.proj, params, world.roster, feed=feed_tables,
                               feed_info=facts("prizepicks"), selected="prizepicks")
    assert (view.paste_won, view.feed_won) == (0, len(PF.SHARED))
    assert view.feed["counts"] == {"feed": 9, "pasted": PF.PASTED - len(PF.SHARED)}
    assert view.paste_times["pasted_utc"] == LAST_NIGHT and view.paste_times["feed_oldest"] is not None
    said = visible(props_ui.feed_notice_html(view.feed, DAY, LABELS))
    assert ("Your pasted PrizePicks lines from 9:41 PM, Mar 11 are older than the odds feed's "
            "(9:00 AM), so the feed's line is shown for 6 players and stats.") in said
    # With DraftKings picked the merged PrizePicks rows are context.
    dk = props_ui.date_board(old, DAY, world.proj, params, world.roster, feed=feed_tables,
                             feed_info=facts("draftkings"), selected="draftkings")
    pp = dk.context[dk.context["book"] == "prizepicks"]
    assert set(pp["origin"]) == {"feed", "pasted"} and dk.feed["counts"]["pasted"] == 0
    assert dk.meta.get("pasted_utc") is None


def test_a_feed_board_that_fails_is_explained_with_the_feeds_words(
        world, params, feed_tables, tokens, monkeypatch):
    def boom(*a, **k):
        raise ValueError("vendor-said-xyzzy")
    monkeypatch.setattr(props, "feed_board", boom)
    for token, want in ((None, "The odds feed's lines are still in memory"),
                        (_paste(world, tokens, NINE), "Your pasted lines are still saved")):
        view = props_ui.date_board(token, DAY, world.proj, params, world.roster, feed=feed_tables,
                                   feed_info=facts("prizepicks"), selected="prizepicks")
        assert view.meta["error"] and view.meta["posted"] == 9 + (PF.PASTED if token else 0)
        assert view.feed["book"] == "prizepicks" and props_ui._kept_text(view.feed) == want
    no_params = props_ui.date_board(None, DAY, world.proj, None, world.roster, feed=feed_tables,
                                    feed_info=facts("draftkings"), selected="draftkings")
    assert no_params.meta == {"posted": 21, "error": True}
    junk = props_ui.date_board(None, DAY, world.proj, params, world.roster, feed=feed_tables,
                               selected="<b>nope")
    assert junk.feed == {"book": "prizepicks", "missing_games": [], "paste_won": 0, "feed_won": 0,
                         "paste_times": {}, "pasted": 0, "counts": {"feed": 9, "pasted": 0},
                         "paste_unmatched": [], "paste_ambiguous": []}


def test_tables_that_could_not_be_built_are_the_explained_empty_board_f1(
        world, params, feed_tables, tokens):
    """Build 3's review, F1. store.feed_tables marks the tables failed when
    the feed holds lines it could not compare: props.FeedTables(failed=True),
    no table and no meta. date_board then gives the explained empty board at
    once, for every pick and with or without the simulator parameters, never
    a board built from nothing: it counts every line the feed holds for the
    date (the facts' 40 "lines") and, with PrizePicks picked, the paste's,
    and says whose lines are kept. No book has a board to pick, so the pills
    have no game, even for failed tables that carried a meta."""
    failed = props.FeedTables(failed=True)
    assert (failed.tables, failed.metas, failed.failed) == ({}, {}, True)
    assert props.FeedTables().failed is False and feed_tables.failed is False
    assert props_ui.feed_games(failed) == set()
    assert props_ui.feed_games(props.FeedTables(metas=dict(feed_tables.metas), failed=True)) == set()
    token = _paste(world, tokens, NINE)
    for tok, kept in ((None, "The odds feed's lines are still in memory"),
                      (token, "Your pasted lines are still saved")):
        for book in props.BOOK_ORDER:
            for p in (params, None):
                if tok and p is not None:
                    continue     # a paste that can be compared is the board (B17, the next test)
                view = props_ui.date_board(tok, DAY, world.proj, p, world.roster, feed=failed,
                                           feed_info=facts(book), selected=book)
                pasted = PF.PASTED if tok and book == "prizepicks" else 0
                assert view.meta == {"posted": 40 + pasted, "error": True}, (book, p)
                assert view.matched == 0 and view.context.empty and view.missing_games == []
                assert view.feed["book"] == book and view.feed["state"] == "ok"
                assert view.feed["counts"] == {"feed": 40, "pasted": pasted}
                assert props_ui._kept_text(view.feed) == kept


def test_a_paste_is_the_board_when_the_feeds_tables_could_not_be_compared_b17(
        world, params, tokens):
    """Build amendment B17. The feed holds lines it could not compare
    (props.FeedTables.failed) and the visitor has a paste for the date: the
    paste's own board, exactly as with the feed off (date_board with no
    feed), for every pick, marked paste_only (the page puts B17's note above
    it and draws no pills). Without a paste, or with no simulator parameters
    (the paste cannot be compared either), the explained empty board as
    built (the test above)."""
    failed = props.FeedTables(failed=True)
    token = _paste(world, tokens, NINE)
    off = props_ui.date_board(token, DAY, world.proj, params, world.roster)
    assert not off.paste_only and off.matched == PF.PASTED and off.feed is None
    for book in props.BOOK_ORDER:
        view = props_ui.date_board(token, DAY, world.proj, params, world.roster, feed=failed,
                                   feed_info=facts(book), selected=book)
        assert view.paste_only and view.feed is None and not view.meta.get("error"), book
        for got, want in zip((view.shown, view.hidden, view.thin, view.void),
                             (off.shown, off.hidden, off.thin, off.void)):
            pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True))
        assert set(view.meta) == set(off.meta) and view.context.empty and view.missing_games == []
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
                               feed_info=facts("prizepicks"), selected="prizepicks")
    assert not view.paste_only and view.meta.get("error")
    other_day = props_ui.date_board(token, DAY + dt.timedelta(days=1), world.proj, params,
                                    world.roster, feed=failed, feed_info=facts("prizepicks"),
                                    selected="prizepicks")
    assert not other_day.paste_only and other_day.meta.get("error")


def test_the_break_even_note_reads_the_boards_book_when_no_line_is_shown_l3_8():
    """Build 4's review, L3-8: a DraftKings board whose lines are all hidden
    (or unmatched) shows no row, and its reading note used to fall back to
    PrizePicks' Power Play range. It reads the board's own book now; with no
    book (the feed off) the wording is today's."""
    hidden = ref("draftkings")
    empty = hidden.iloc[0:0]
    meta = {"posted": 1, "matched": 1}
    dk = props_ui.notes_html(meta, empty, hidden, empty, empty, facts("draftkings"))[-1]
    assert "A sportsbook side at -110 needs about 52.4% to break even." in dk
    assert "54.7%" not in dk
    ud = props_ui.notes_html(meta, empty, ref("underdog"), empty, empty, facts("underdog"))[-1]
    assert "Underdog's payouts are not modelled here" in ud and "54.7%" not in ud
    pp = props_ui.notes_html(meta, empty, ref("prizepicks"), empty, empty, facts("prizepicks"))[-1]
    assert "a leg has to hit 54.7% to 58.5% just to break even." in pp
    assert props_ui.breakeven_text(empty) == props_ui.breakeven_text(None) == (
        ": a leg has to hit 54.7% to 58.5% just to break even.")
    assert props_ui.breakeven_text(empty, "fanduel") == (
        ". A sportsbook side at -110 needs about 52.4% to break even.")
    assert props_ui.breakeven_text(empty, "nope") == props_ui.breakeven_text(empty)


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
    assert notes[0] == "4 DraftKings lines on file for this date (3 from the odds feed, 1 pasted): 2 matched, 2 unmatched."
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
    assert visible(one) == "1 DraftKings line on file for this date (from the odds feed): 1 matched."
    pasted = props_ui.notes_html({"posted": 2, "matched": 2}, ref("prizepicks", 24.5), None, None,
                                 None, facts("prizepicks", counts={"feed": 0, "pasted": 2}))[0]
    assert visible(pasted).startswith("2 PrizePicks lines on file for this date (pasted): ")


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
import pandas as pd
import ui_rows as U
import test_props_ui_feed as T
from bucketlib import props_ui
rows = T.ref()
info = T.facts()
props_ui.render_board(rows, None, None, None, "March 12", meta={{"posted": 1, "matched": 1}},
                      context=T.FOUR, feed=info)
props_ui.render_board(rows.iloc[:0], None, None, None, "PHI at DET",
                      meta={{"posted": 2, "error": True}}, key="bv_b2", feed=info)
props_ui.render_player_lines_block(rows, T.FOUR, info)
"""


def test_the_feed_board_renders_the_cell_the_legend_and_the_feeds_words():
    at = _page(FEED_BOARD_SCRIPT)
    md = [m.value for m in at.markdown]
    assert md.count(props_ui.BOOKS_CSS.strip()) == 3 and md.count(props_ui.PROPS_CSS.strip()) == 3
    ledger = next(m for m in md if m.startswith('<div class="bv-ledger has-books">'))
    assert "DK line" in ledger and visible(cells(ledger)[0]) == \
        "PP 24.5, UD 24.5, FD 25.5 (9:02 AM), MGM 25.5"
    assert props_ui.books_legend_html() in md
    assert md.index(props_ui.books_legend_html()) == md.index(ledger) + 1
    assert any("The odds feed's lines are still in memory; try reloading the page." in m for m in md)
    assert any(m.startswith('<div class="bv-pl has-books">') for m in md)
    for m in md:
        assert dash_ok(m) and all(ord(ch) < 128 or ch == SENTINEL for ch in m)


# -- The book pills (amendment A3) ----------------------------------------------------------
PILLS = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
import streamlit as st
from bucketlib import props_ui
FEED_ON = {{feed_on}}
props_ui.seed_book_from_url(FEED_ON)
st.markdown("picked " + props_ui.selected_book(FEED_ON))
if FEED_ON and st.query_params.get("hide") != "1":       # "1": a date with no feed rows
    props_ui.render_book_pills()
"""


def _pills(feed_on=True, **query) -> AppTest:
    return _page(PILLS.replace("{feed_on}", str(feed_on)), **query)


def _picked(at) -> str:
    return next(m.value for m in at.markdown if m.value.startswith("picked ")).split(" ", 1)[1]


def test_the_games_with_feed_lines_say_where_the_pills_show(feed_tables):
    """Amendment A3: the pills show only with feed rows for the date (or the
    game). Both games have lines in the fixtures (PrizePicks only on PHI at
    DET); with the feed off there are none."""
    assert props_ui.feed_games(feed_tables) == {EARLY, LATE}
    only = props.FeedTables({k: t for k, t in feed_tables.tables.items() if k[1] == EARLY},
                            {k: m for k, m in feed_tables.metas.items() if k[1] == EARLY})
    assert props_ui.feed_games(only) == {EARLY}
    assert props_ui.feed_games(None) == props_ui.feed_games(props.FeedTables()) == set()


def test_the_pills_are_the_five_books_in_order_seeded_never_given_a_default():
    at = _pills()
    bg = at.button_group(key=props_ui.BOOK_KEY)
    assert bg.options == [props.BOOK_TITLE[b] for b in props.BOOK_ORDER]
    assert bg.options == ["PrizePicks", "Underdog", "DraftKings", "FanDuel", "BetMGM"]
    assert list(bg.proto.default) == [] and bg.value == "prizepicks" and _picked(at) == "prizepicks"
    assert not at.warning and not at.exception and dict(at.query_params) == {}


def test_book_draftkings_in_the_url_picks_draftkings_on_first_load():
    at = _pills(book="draftkings")
    assert _picked(at) == "draftkings" and at.button_group(key=props_ui.BOOK_KEY).value == "draftkings"
    assert at.session_state[props_ui.BOOK_SEEN] == "draftkings"


def test_a_junk_book_gives_prizepicks_and_is_dropped_never_echoed():
    for junk in ("<script>x</script>", "DraftKings", "pinnacle", ""):
        at = _pills(book=junk)
        assert _picked(at) == "prizepicks", junk
        assert "book" not in at.query_params or not junk, junk
        assert all(junk not in m.value for m in at.markdown if junk), junk


def test_with_the_feed_off_a_book_in_the_url_changes_nothing():
    at = _pills(False, book="draftkings")
    assert _picked(at) == "prizepicks"
    assert props_ui.BOOK_KEY not in at.session_state and props_ui.BOOK_SEEN not in at.session_state
    assert dict(at.query_params) == {"book": ["draftkings"]}            # untouched, never read
    assert not at.button_group


def test_a_pick_writes_the_url_and_a_url_change_reseeds_only_when_it_changes():
    at = _pills()
    at.button_group(key=props_ui.BOOK_KEY).set_value("fanduel").run()
    assert _picked(at) == "fanduel" and dict(at.query_params) == {"book": ["fanduel"]}
    assert at.session_state[props_ui.BOOK_SEEN] == "fanduel"
    at.run()                                                   # the URL did not change: the pick stays
    assert _picked(at) == "fanduel"
    at.query_params["book"] = "betmgm"                          # back or forward: the URL changed
    at.run()
    assert _picked(at) == "betmgm" and at.button_group(key=props_ui.BOOK_KEY).value == "betmgm"
    at.button_group(key=props_ui.BOOK_KEY).set_value("prizepicks").run()
    assert _picked(at) == "prizepicks" and dict(at.query_params) == {}
    assert at.session_state[props_ui.BOOK_SEEN] is None


def test_a_dropped_widget_state_comes_back_from_the_url():
    """Streamlit drops the state of pills a run did not draw (a date with no
    feed rows): the next run seeds the pick again from a valid ?book=, so the
    board and the URL agree. With no ?book= it is PrizePicks."""
    at = _pills(book="underdog")
    at.query_params["hide"] = "1"
    at.run()
    assert _picked(at) == "underdog" and props_ui.BOOK_KEY not in at.session_state
    del at.query_params["hide"]
    at.run()
    assert _picked(at) == "underdog" and at.button_group(key=props_ui.BOOK_KEY).value == "underdog"
    at.button_group(key=props_ui.BOOK_KEY).set_value("prizepicks").run()
    at.query_params["hide"] = "1"
    at.run()
    del at.query_params["hide"]
    at.run()
    assert _picked(at) == "prizepicks" and dict(at.query_params) == {}


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


def test_a_pasted_prizepicks_line_in_other_books_shows_its_badge_word_b28():
    """Build amendment B28 (build 4's review, L3-11, which pinned a typed
    pasted line reading as a plain PrizePicks line). With DraftKings picked,
    a PrizePicks line from the visitor's paste shows its badge word after
    the number, in place of a time: "(typed)" for a typed pick (first, as
    the merge reads them), "(Demon)" or "(Goblin)", else "(pasted)". The
    feed's own PrizePicks line keeps the time rule, on the slate and on the
    Game page alike; the cell stays plain text."""
    rows = ref("draftkings", 25.5)

    def cell(context, page=False):
        render = props_ui.render_player_lines if page else props_ui.ledger_html
        html = render(rows, context, facts("draftkings")) if page else \
            render(rows, None, context, facts("draftkings"))
        (c,) = cells(html)
        assert "class=" not in c and "style=" not in c
        return visible(c)

    def paste(**kw):
        return U.frame([U.row(book="prizepicks", origin="pasted", line=27.5, **kw),
                        feed_row("underdog", 25.5, lean="Less", gap=-0.4)])
    for kw, word in (({"pick": "more"}, "typed"), ({"pick": "less", "odds_type": "demon"}, "typed"),
                     ({"odds_type": "demon"}, "Demon"), ({"odds_type": "Goblin"}, "Goblin"),
                     ({}, "pasted")):
        for page in (False, True):
            assert cell(paste(**kw), page) == f"PP 27.5 ({word}), UD 25.5", (kw, page)
    fed = U.frame([feed_row("prizepicks", 27.5, at=NINE), feed_row("underdog", 25.5)])
    assert cell(fed) == "PP 27.5 (9:02 AM), UD 25.5"                   # the feed's: its own time
    same = U.frame([feed_row("prizepicks", 27.5), feed_row("underdog", 25.5)])
    assert cell(same) == "PP 27.5, UD 25.5"
    assert props_ui.PASTE_WINNERS == "newer, typed, Demon or Goblin"


def test_a6_after_tip_and_for_a_book_the_site_does_not_buy_b24_b25():
    """Build amendments B24 and B25 (build 4's review, L3-3 and L3-6). The A6
    sentences name no direction: "Pick another book to see theirs." The
    slate's list names only games that have not tipped (the page passes the
    tipped ones), and says nothing once all have; the Game page's note says
    after its game's tip that the odds feed did not load the book's lines
    before tip (build amendment B29, in place of B25's "No PrizePicks lines
    were posted"). A book BUCKETVALUE_ODDS_BOOKS leaves out (not in
    feed_info's "books") reads that its lines are not loaded on this site,
    on both pages and tipped or not, in place of the "yet" sentences.
    feed_notice_html hands the page's tipped games and the facts' books
    on."""
    def said(gids, book="prizepicks", **kw):
        return visible(props_ui.book_missing_html(book, gids, LABELS, **kw))
    pick = "Pick another book to see theirs."
    assert said([EARLY, LATE], tipped={EARLY}) == f"No PrizePicks lines yet for DEN at SAS. {pick}"
    assert said([EARLY, LATE], tipped=[LATE]) == f"No PrizePicks lines yet for PHI at DET. {pick}"
    assert said([EARLY, LATE], tipped=(EARLY, LATE)) == ""
    assert said([EARLY, LATE]) == f"No PrizePicks lines yet for PHI at DET and DEN at SAS. {pick}"
    four = ("draftkings", "fanduel", "prizepicks", "underdog")
    gone = f"BetMGM lines are not loaded on this site. {pick}"
    assert said([EARLY, LATE], "betmgm", loaded=four) == gone
    assert said([EARLY, LATE], "betmgm", loaded=four, tipped={EARLY, LATE}) == gone
    assert said([EARLY], "draftkings", loaded=four).startswith("No DraftKings lines yet")
    assert said([], "betmgm", loaded=four) == ""
    game = props_ui.book_missing_game_html
    assert game("prizepicks", tipped=True) == (                       # B29
        f"<b>The odds feed did not load PrizePicks lines for this game before tip.</b> {pick}")
    assert game("betmgm", loaded=four) == game("betmgm", tipped=True, loaded=four) == (
        f"<b>BetMGM lines are not loaded on this site.</b> {pick}")
    assert game("betmgm") == f"<b>No BetMGM lines for this game yet.</b> {pick}"
    info = facts("prizepicks", missing_games=[EARLY, LATE])
    assert f"No PrizePicks lines yet for DEN at SAS. {pick}" in visible(
        props_ui.feed_notice_html(info, DAY, LABELS, tipped={EARLY}))
    assert "No PrizePicks lines yet" not in props_ui.feed_notice_html(info, DAY, LABELS,
                                                                      tipped={EARLY, LATE})
    narrow = facts("betmgm", missing_games=[EARLY, LATE], books=four)
    assert gone in visible(props_ui.feed_notice_html(narrow, DAY, LABELS))
    for text in (said([LATE]), game("prizepicks"), game("prizepicks", tipped=True), gone):
        assert " above" not in text and " below" not in text


# -- Build 7, stage S13: build amendments B29 and B30 -----------------------------------------
def test_after_tip_a_picked_book_with_no_lines_reads_the_feed_did_not_load_them_b29():
    """Build amendment B29 (build 6's review, S12-R3). After its game's tip
    the Game page's note for a picked book with no lines for the game says
    the odds feed did not load that book's lines before tip, for every book:
    true whether the book posted none or the site had stopped asking for it.
    B25's "No <Book> lines were posted for this game before tip." is gone.
    Before tip the "yet" sentence stands; a book BUCKETVALUE_ODDS_BOOKS
    leaves out keeps B25's not-loaded sentence, tipped or not; a junk book is
    PrizePicks, never echoed; and B18's game-level sentences (a game with no
    feed lines at all) are unchanged."""
    pick = props_ui.PICK_ANOTHER
    game = props_ui.book_missing_game_html
    for book in props.BOOK_ORDER:
        title = props.BOOK_TITLE[book]
        assert game(book, tipped=True) == (
            f"<b>The odds feed did not load {title} lines for this game before tip.</b> {pick}"), book
        assert "were posted" not in game(book, tipped=True), book
        assert game(book) == game(book, tipped=False) == (
            f"<b>No {title} lines for this game yet.</b> {pick}"), book
        others = tuple(b for b in props.BOOK_ORDER if b != book)
        for tipped in (False, True):
            assert game(book, tipped=tipped, loaded=others) == (
                f"<b>{title} lines are not loaded on this site.</b> {pick}"), (book, tipped)
        assert game(book, tipped=True, loaded=props.BOOK_ORDER) == game(book, tipped=True), book
    assert game("x<b>", tipped=True) == game("prizepicks", tipped=True)
    assert props_ui.after_tip_note_html("none_posted") == (
        "<b>No lines were posted for this game before tip.</b>")
    for reason in ("not_yet", "held", "paused", None):
        assert props_ui.after_tip_note_html(reason) == (
            "<b>The odds feed did not load lines for this game before tip.</b>"), reason


def test_the_settled_note_claims_a_void_on_prizepicks_rows_only_b30():
    """Build amendment B30 (build 6's review, open question 3). B26 leaves
    what happens to an Underdog pick on a player who does not play to
    Underdog's own rule, so a settled Underdog row's note claims no void: it
    reads DNP_TEXT, "Did not play", as a sportsbook row's does. A PrizePicks
    row keeps VOID_TEXT, "Did not play, pick voided", and so does a row with
    no book (a pasted line, as every line was before the feed: the feed-off
    golden holds that HTML byte for byte). A value on the line reads
    EXACT_TEXT whatever the book."""
    assert (props_ui.VOID_TEXT, props_ui.DNP_TEXT) == ("Did not play, pick voided", "Did not play")
    gone = {"actual": None, "settle": "void", "final": True}
    exact = {"actual": 6.0, "settle": "exact", "final": True}

    def notes(row, book=None, settled=gone):
        html = props_ui.render_player_lines([{**row, **settled}], None,
                                            facts(book) if book else None)
        return [visible(n) for n in re.findall(r'<div class="bv-pl-note">(.*?)</div>', html)]
    assert notes(feed_row("underdog", 6.0), "underdog") == ["Did not play"]
    assert notes(feed_row("prizepicks", 6.0), "prizepicks") == ["Did not play, pick voided"]
    no_book = [U.row(book=None), U.row(book=" "), {k: v for k, v in U.row().items() if k != "book"}]
    for row in no_book:
        assert notes(row) == ["Did not play, pick voided"], row.get("book", "no key")
    for book in ("draftkings", "fanduel", "betmgm"):
        assert notes(feed_row(book, 6.0), book) == ["Did not play"], book
    for book in props.BOOK_ORDER:
        assert notes(feed_row(book, 6.0), book, exact) == ["6: landed on the line"], book
    assert notes(U.row(book=None, line=6.0), settled=exact) == ["6: landed on the line"]
    note = props_ui._settle_note
    assert note({"book": "underdog", "settle": "void"}) == '<div class="bv-pl-note">Did not play</div>'
    for row in ({"book": "prizepicks", "settle": "void"}, {"settle": "void"}):
        assert note(row) == '<div class="bv-pl-note">Did not play, pick voided</div>', row
    assert note({"book": "underdog", "settle": None}) == note({"settle": None}) == ""
