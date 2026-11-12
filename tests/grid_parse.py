"""Read layout C's grid back from its HTML, for the tests (not a test module).

With the odds feed on, props_ui draws the board as a grid (amendments C6 and
C7): one column per book, each cell a <td class="bk"> holding that book's line
with the model's lean at it. These helpers read it the way a reader would:
the header's book columns (code and title), and each row's player, matchup,
stat, model, actual and cells, each cell {"cls", "title", "number", "arrow",
"sr", "badges"} (the title and the visually hidden text unescaped; the arrow
and the badges as their markup), and the notes under a per-player grid's
settled rows.

Every cell is checked against its own column while it is parsed (amendment
C1V, C1R-06): a cell with a line has a title that begins with its column
header's title ("DraftKings ..." under DK), and a dash cell's hidden text
names that same title (" no DraftKings line"). A mismatch raises, so every
test that parses a grid checks the column order for free.
"""
from __future__ import annotations

import re
from html import unescape

SHOW_ALL = "bv_board_all"          # the slate board's Show all toggle (props_ui.render_board)
WRAP = re.compile(r'<div class="bv-grid-wrap" tabindex="0" role="region" aria-label="([^"]*)">'
                  r'<table class="(bv-grid[^"]*)"><thead><tr>(.*?)</tr></thead><tbody>(.*?)'
                  r"</tbody></table></div>", re.S)
HEAD = re.compile(r'<th class="bk" scope="col"><abbr title="([^"]*)">([^<]*)</abbr></th>')
ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
NOTE = re.compile(r'<tr class="note"><td colspan="(\d+)"><div class="bv-pl-note">(.*?)</div>'
                  r"</td></tr>", re.S)
CELL = re.compile(r'<td class="bk">(.*?)</td>', re.S)
CELL_BODY = re.compile(r'<span class="bv-cell (over|under|plain|none)"(?: title="([^"]*)")?>'
                       r'([^<]*)(?:<span class="ar" aria-hidden="true">([^<]*)</span>)?'
                       r'<span class="bv-sr">([^<]*)</span></span>(.*)', re.S)
SLATE_LEAD = re.compile(r'<th class="pl" scope="row"><span class="nm">(.*?)</span><i>(.*?)</i></th>'
                        r'<td class="st">(.*?)</td>', re.S)
PLAYER_LEAD = re.compile(r'<th class="st" scope="row">(.*?)</th>', re.S)
MODEL = re.compile(r'<td class="num">([^<]*)</td>')
ACTUAL = re.compile(r'<td class="num act">([^<]*)</td>')
MORE = re.compile(r'<b class="bv-pp-more">\+(\d+) more</b>')


class GridMismatch(AssertionError):
    """A grid whose cells do not belong to their columns, or whose markup is
    not a grid cell's."""


def heads(markup: str) -> list[tuple[str, str]]:
    """[(code, title)] of the first grid's book columns, in order."""
    m = WRAP.search(markup)
    return [(unescape(code), unescape(title)) for title, code in HEAD.findall(m.group(3))] if m else []


def grid_heads(markup: str) -> list[str]:
    """The first grid's column codes, in order: ["DK", "FD", "MGM", "PP", "UD"]."""
    return [code for code, _title in heads(markup)]


def cell(markup: str, code: str = "", title: str = "") -> dict:
    """One <td class="bk">'s content as a dict, checked against its column
    (`code`, `title`) when they are given."""
    m = CELL_BODY.fullmatch(markup)
    if not m:
        raise GridMismatch(f"not a grid cell under {code or '?'}: {markup[:160]!r}")
    cls, tip, number, arrow, sr, badges = m.groups()
    out = {"cls": cls, "title": unescape(tip) if tip is not None else None, "number": number,
           "arrow": arrow, "sr": unescape(sr), "badges": badges}
    if title:
        if cls == "none" and out["sr"] != f" no {title} line":
            raise GridMismatch(f"the {code} column's dash names another book: {out['sr']!r}")
        if cls != "none" and not (out["title"] or "").startswith(title + " "):
            raise GridMismatch(f"the {code} column holds another book's line: {out['title']!r}")
    return out


def _row(markup: str, columns: list[tuple[str, str]]) -> dict:
    lead = SLATE_LEAD.match(markup)
    if lead:
        name, under, stat = lead.groups()
        more = MORE.search(under)
        row = {"player": unescape(name), "matchup": unescape(MORE.sub("", under)),
               "more": int(more.group(1)) if more else 0}
    else:
        lead = PLAYER_LEAD.match(markup)
        if not lead:
            raise GridMismatch(f"not a grid row: {markup[:160]!r}")
        (stat,) = lead.groups()
        row = {"player": None, "matchup": None, "more": 0}
    rest = markup[lead.end():]
    model = MODEL.match(rest)
    if not model:
        raise GridMismatch(f"a grid row without its Model cell: {rest[:160]!r}")
    found = CELL.findall(rest)
    if len(found) != len(columns):
        raise GridMismatch(f"{len(found)} book cells under {len(columns)} book columns")
    actual = ACTUAL.search(rest)
    row.update(stat=unescape(stat.split("<", 1)[0]), stat_html=stat, model=model.group(1),
               actual=actual.group(1) if actual else None,
               cells={code: cell(c, code, title) for (code, title), c in zip(columns, found)})
    return row


def tables(markup: str) -> list[dict]:
    """Every grid in `markup`: {"label" (the wrapper's aria-label), "cls" (the
    table's classes), "heads" ([(code, title)]), "rows", "notes" ([(colspan,
    note)])}, each row as _row reads it."""
    out = []
    for label, cls, head, body in WRAP.findall(markup):
        columns = [(unescape(code), unescape(title)) for title, code in HEAD.findall(head)]
        out.append({"label": unescape(label), "cls": cls, "heads": columns,
                    "rows": [_row(r, columns) for r in ROW.findall(body)],
                    "notes": [(int(n), unescape(x)) for n, x in NOTE.findall(body)]})
    return out


def rows(markup: str) -> list[dict]:
    """The rows of every grid in `markup`, in order."""
    return [r for t in tables(markup) for r in t["rows"]]


def grid(at, show_all: bool = False) -> list[dict]:
    """The slate's grid rows on an AppTest page ([] when it draws no grid),
    after turning the board's Show all on when asked (a rerun of `at`)."""
    if show_all:
        at.toggle(key=SHOW_ALL).set_value(True).run()
        assert not at.exception, [e.value for e in at.exception]
    found = [t for m in at.markdown for t in tables(str(m.value)) if t["label"] == "Lines by book"]
    if len(found) > 1:
        raise GridMismatch(f"{len(found)} slate grids on one page")
    return found[0]["rows"] if found else []


PLAYER_LABEL = ": lines by book"     # a per-player grid's aria-label: "<player>: lines by book"


def _player_tables(at):
    """(player, grid) for each per-player grid in the Game page's expanders
    ("Name: N lines"), the player read from the grid's own aria-label."""
    for e in at.expander:
        if re.match(r"(.+): (\d+) lines?$", str(e.label)):
            for t in tables(" ".join(str(x.value) for x in e.markdown)):
                if not t["label"].endswith(PLAYER_LABEL):
                    raise GridMismatch(f"a per-player grid without its player: {t['label']!r}")
                yield t["label"][:-len(PLAYER_LABEL)], t


def player_grid(at) -> dict[str, list[dict]]:
    """{player: his grid's rows} from the Game page's expanders ("Name: N
    lines"), keyed by the player each grid names in its own aria-label (the
    board's spelling, the slate's Player column's)."""
    out: dict = {}
    for player, t in _player_tables(at):
        out.setdefault(player, []).extend(t["rows"])
    return out


def player_notes(at) -> dict[str, list[str]]:
    """{player: the notes under his grid's settled rows, in order} from the
    Game page's expanders, keyed as player_grid keys them (a player whose
    grid has no note maps to [])."""
    out: dict = {}
    for player, t in _player_tables(at):
        out.setdefault(player, []).extend(note for _span, note in t["notes"])
    return out


def keyed_markdown(at, key: str) -> list[str] | None:
    """The markdown values inside the Block whose element id ends with
    "-<key>" (st.container(key=...)), in render order; None when the page
    has no such block. Element ids hash the script's path, so only their
    tail (the key) is read."""
    hits: list = []

    def collect(node, into):
        if type(node).__name__ == "Markdown":
            into.append(str(node.value))
        kids = getattr(node, "children", None)
        if isinstance(kids, dict):
            for k in sorted(kids):
                collect(kids[k], into)

    def walk(node):
        proto = getattr(node, "proto", None)
        if type(node).__name__ == "Block" and str(getattr(proto, "id", "")).endswith("-" + key):
            found: list = []
            collect(node, found)
            hits.append(found)
            return
        kids = getattr(node, "children", None)
        if isinstance(kids, dict):
            for k in sorted(kids):
                walk(kids[k])
    walk(at._tree)
    if len(hits) > 1:
        raise GridMismatch(f"{len(hits)} blocks keyed {key}")
    return hits[0] if hits else None
