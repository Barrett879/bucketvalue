"""The Avg column (amendment C22; Barrett, 2026-11-27: "put an average of the
book lines next to our predicted amount"). In both layout C grids, right
after Model: the mean of the row's book lines, every cell's line, shown or
hidden, with PrizePicks Goblin and Demon lines left out (they sit off the
market on purpose); one decimal; the sentinel when no line counts; never an
arrow or a lean class (spec rule 15). Every expected number below is worked
out by hand from the fixture's literal lines, never by calling the code under
test. The grids are read back with tests/grid_parse.py, which also checks on
every grid any test parses that Avg sits right after Model.
"""
from __future__ import annotations

import re

from bucketlib import props_ui
from bucketlib.util import SENTINEL

import grid_parse as GP
import test_props_ui_feed as UF  # the hand-built feed rows, facts and grid helpers
import ui_rows as U


def slate(shown, hidden=None) -> dict:
    """The slate grid's one row for Ann Able's points."""
    (row,) = UF.drawn(shown, hidden)
    return row


def pasted(line, **kw):
    """A pasted PrizePicks line for Ann Able's points (ui_rows' default row)."""
    return U.frame([U.row(line=line, **kw)])


def head_labels(markup: str) -> list[str]:
    head = re.search(r"<thead><tr>(.*?)</tr></thead>", markup, re.S).group(1)
    return re.findall(r'<th class="[^"]*" scope="col">(?:<abbr title="[^"]*">)?([^<]*)', head)


def test_avg_sits_right_after_model_on_the_slate_grid():
    html = props_ui.ledger_html(UF.grid(UF.lines(("draftkings", 24.5))), feed=UF.facts())
    assert head_labels(html) == ["Player", "Stat", "Model", "Avg", "DK", "FD", "MGM", "PP", "UD"]
    assert ('<th class="num avg" scope="col"><abbr title="Average of the book lines">'
            "Avg</abbr></th>") in html


def test_the_avg_is_the_mean_of_every_books_line():
    # (24.5 + 25.5 + 24.5 + 25.0) / 4 = 24.875, printed 24.9
    row = slate(UF.lines(("draftkings", 24.5), ("fanduel", 25.5), ("betmgm", 24.5), ("underdog", 25.0)))
    assert row["avg"] == "24.9"
    assert row["avg_title"] == "Average of 4 book lines: DK 24.5, FD 25.5, MGM 24.5, UD 25."
    assert row["avg_sr"] == " average of 4 book lines"


def test_the_mean_not_the_median_and_not_the_lead_line():
    # median 24.5 and the lead (DraftKings) line 24.5; the mean is (24.5 + 24.5 + 27.5) / 3 = 25.5
    row = slate(UF.lines(("draftkings", 24.5), ("fanduel", 24.5), ("betmgm", 27.5)))
    assert row["avg"] == "25.5"


def test_one_decimal_always():
    assert slate(UF.lines(("draftkings", 24.5), ("fanduel", 25.0)))["avg"] == "24.8"   # 24.75
    assert slate(UF.lines(("draftkings", 24.5), ("fanduel", 25.5),
                          ("underdog", 25.0)))["avg"] == "25.0"                        # 25.0, not 25
    one = slate(UF.lines(("fanduel", 31.5)))
    assert (one["avg"], one["avg_title"], one["avg_sr"]) == (
        "31.5", "Average of 1 book line: FD 31.5.", " average of 1 book line")


def test_a_tie_rounds_up_the_way_a_reader_rounds():
    # (24 + 24.5) / 2 = 24.25: a reader writes 24.3 (float formatting prints 24.2)
    row = slate(UF.lines(("draftkings", 24.0), ("fanduel", 24.5)))
    assert (row["avg"], row["avg_title"]) == ("24.3", "Average of 2 book lines: DK 24, FD 24.5.")
    # (24.5 + 24.5 + 25.0 + 25.0) / 4 = 24.75 prints 24.8
    assert slate(UF.lines(("draftkings", 24.5), ("fanduel", 24.5), ("betmgm", 25.0),
                          ("underdog", 25.0)))["avg"] == "24.8"
    # Ten lines (the feed's ten books): seven 24s and three 24.5s sum to 241.5, a mean of
    # exactly 24.15, which binary cannot hold (float formatting prints 24.1). The slate
    # fixture draws five book columns, so this one calls the cell's formatter directly.
    assert props_ui._avg_text([24.0] * 7 + [24.5] * 3) == "24.2"
    assert props_ui._avg_text([24.0] * 9 + [24.5]) == "24.1"      # 24.05 rounds up to 24.1
    assert props_ui._avg_text([31.5]) == "31.5"
    assert props_ui._avg_text([25.0]) == "25.0"


def test_a_hidden_line_counts_because_it_is_still_that_books_number():
    row = slate(UF.lines(("draftkings", 24.5)), UF.lines(("fanduel", 25.5, {"lean": "Even"})))
    assert row["cells"]["FD"]["cls"] == "plain"        # the hidden line, drawn as its number alone
    assert row["avg"] == "25.0"                         # (24.5 + 25.5) / 2


def test_goblin_and_demon_lines_are_left_out_and_named():
    gob = slate(UF.lines(("draftkings", 24.5)), pasted(21.5, odds_type="goblin", lean="Even"))
    assert gob["cells"]["PP"]["number"] == "21.5"       # still drawn in its column
    assert gob["avg"] == "24.5"
    assert gob["avg_title"] == "Average of 1 book line: DK 24.5. PP Goblin 21.5 left out."
    demon = slate(UF.lines(("draftkings", 24.5), ("fanduel", 25.5)), pasted(28.5, odds_type="demon",
                                                                            lean="Even"))
    assert demon["avg"] == "25.0"
    assert demon["avg_title"] == (
        "Average of 2 book lines: DK 24.5, FD 25.5. PP Demon 28.5 left out.")


def test_a_row_with_no_standard_line_shows_the_sentinel():
    row = slate(pasted(21.5, odds_type="goblin"))
    assert row["avg"] == SENTINEL
    assert row["avg_title"] == "No standard book line to average. PP Goblin 21.5 left out."
    assert row["avg_sr"] == " no book average"


def test_a_pasted_prizepicks_line_and_a_typed_pick_count_like_any_cell():
    assert slate(UF.lines(("draftkings", 24.5)), pasted(23.5, lean="Even"))["avg"] == "24.0"
    typed = slate(UF.lines(("draftkings", 24.5)), pasted(25.5, lean="Even", pick="more"))
    assert typed["avg"] == "25.0"


def test_the_avg_cell_is_never_a_lean():
    html = props_ui.ledger_html(UF.grid(UF.lines(("draftkings", 24.5, {"lean": "More"}),
                                                 ("fanduel", 25.5, {"lean": "Less"}))),
                                feed=UF.facts())
    # The class is exactly "num avg" (no over, under or plain), and no arrow or lean span.
    (cell,) = re.findall(r'<td class="num avg" title="[^"]*">.*?</td>', html, re.S)
    assert "bv-cell" not in cell and "&#8593;" not in cell and "&#8595;" not in cell
    assert cell.count("<span") == 1 and '<span class="bv-sr">' in cell
    assert UF.dash_ok(props_ui.ledger_html(UF.grid(pasted(21.5, odds_type="goblin")), feed=UF.facts()))


def test_the_game_page_grid_has_the_same_avg_and_its_notes_span_every_column():
    shown = [UF.feed_row("draftkings", 24.5), UF.feed_row("fanduel", 25.5)]
    html = props_ui.render_player_lines(shown, None, UF.facts())
    assert head_labels(html) == ["Stat", "Model", "Avg", "DK", "FD", "MGM", "PP", "UD"]
    (row,) = GP.rows(html)
    assert (row["avg"], row["avg_title"]) == ("25.0", "Average of 2 book lines: DK 24.5, FD 25.5.")
    settled = [{**UF.feed_row("draftkings", 24.5), "actual": None, "settle": "void", "final": True}]
    done = props_ui.render_player_lines(settled, None, UF.facts())
    labels = head_labels(done)
    assert labels == ["Stat", "Model", "Avg", "DK", "FD", "MGM", "PP", "UD", "Final"]
    (table,) = GP.tables(done)
    assert [n for n, _note in table["notes"]] == [len(labels)]
    assert table["rows"][0]["avg"] == "24.5"


def test_the_legend_says_what_avg_is():
    assert "Avg: the average of the book lines in the row, leaving out PrizePicks Goblin and " \
           "Demon lines." in props_ui.GRID_LEGEND
