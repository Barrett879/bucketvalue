"""Player lines on the page: the nav-bar paste popover, the slate board (pasted
PrizePicks lines and the licensed odds feed's five books), the per-player rows
on the game page, and the card pill counts.

Deliberately NOT a page (sibling precedent: DiamondValue started with a props
tab and folded it inline). Ported from GridironValue's props_ui and
DiamondValue's paste flow, with BucketValue's two structural differences:

1. THE BOARD LIVES IN THE PROCESS, KEYED BY ?board=. Every internal link here
   is a full page load, a fresh Streamlit session, so st.session_state cannot
   carry a paste from the slate to a game page. props.new_board / add_to_board
   keep it in an in-process LRU under an unguessable token, and every minted
   URL carries the token (keep_params). Never on the server's disk, never
   frozen; seen only by whoever holds the link. Since 2026-11-05 the visitor's
   OWN browser also keeps a copy (board_keeper, localStorage), so a reload, a
   bare link or a server restart brings the lines back (Barrett: "I have to
   reenter them every time I refresh").
2. ONE DEFINITION. board_view() runs compare() and props.board_rows() once;
   the card pills, the board ledger and the per-player rows all read its four
   frames, so a pill can never promise more lines than the game page shows.

Informational model-versus-market readout, never a wager prompt: the gap is a
difference in the stat's own units, never a percentage column or a dollar
value. Every string from a paste, a feed or a query param goes through esc().
The accuracy record is LOCAL ONLY and is not imported or called from here.

THE LICENSED ODDS FEED (design 2.6; layout C, amendments C1 to C13). With a
key configured the page hands date_board the feed's compared tables
(props.FeedTables) and the feed's plain facts (store.feed_info); nothing here
reaches the feed, which lives in the live layer. The board then shows every
book at once: one row per player and stat (props.grid_rows), one column per
book in the order DK, FD, MGM, PP, UD, each cell that book's line with the
model's lean at it (an arrow and its text equivalent on a shown line, the
number alone on a hidden one, the whole-node sentinel where the book has no
line), and the visitor's pasted PrizePicks lines in the PP column under the
newest-line rule. Every feed text is built from the stamps and the `today`
the page passes, never from this module's clock. With the feed off
(feed=None everywhere) every function here returns exactly the HTML it
returned before the feed.
"""
from __future__ import annotations

import hashlib
import math
import secrets
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from . import props, simprov
from .cache import logger
from .util import SENTINEL, esc, fmt_tip, now_utc, to_et, today_et

PASTE_KEY = "bv_pp_paste"
REPORT_KEY = "bv_pp_report"
TOAST_KEY = "bv_pp_toast"
EXPIRED_KEY = "bv_pp_expired"
TOP_N = 25

# A responsible-gambling line, appended to the break-even note when set
# (design 15 item 4: none by default).
RG_LINE = ""
# A feed line more than this older than its game's latest pull says its own
# time in its grid cell's title (design m23; layout C, amendment C10).
LINE_TIME_AGE = pd.Timedelta(minutes=1)
# The visitor's pasted PrizePicks lines that win the merge with the feed's
# (props._paste_wins): a newer one, and whatever its time a typed pick, a
# Demon or a Goblin. The merge line and the paste caption name them so (build
# amendment B28).
PASTE_WINNERS = "newer, typed, Demon or Goblin"

# -- The browser's copy of the visitor's board (props.saved_board) -------------
# One hidden st.components.v2 component (no iframe: its JS runs in the page and
# reads and writes ONE localStorage key of this site's own origin). The page
# talks to it through `data`; it answers through one-shot triggers, read from
# the mount's return value on the run they arrive with (inside an
# on_*_change callback the value is NOT readable in Streamlit 1.59: measured).
#
# board_keeper runs at the TOP of a page, before the nav and every widget, so a
# board it restores is on the link for that whole run: no st.rerun(), which
# mid-script would drop the state of every widget registered later in the
# page (review 2026-11-05).
#   save    after a paste that made a NEW board (never for a board only
#           opened from someone else's link), and again after a restore so
#           the copy follows the fresh token. Resent until acknowledged
#           ("saved") or refused ("failed": storage blocked or full).
#   clear   after Clear all, naming the board on screen by a hash of its
#           token (never the token), plus the copy this session restored it
#           from: the browser removes its copy only when the copy is that
#           board's, so Clear all on a friend's link leaves the visitor's own
#           copy alone. Resent until acknowledged.
#   want    while the link carries no live board: one id per session (and a
#           new one each time a board dies mid-session), answered once per id
#           and only when the browser holds a copy ("restore"), so a visitor
#           without one costs nothing.
# An answer counts only while its request is outstanding: Streamlit lets a
# hand-built client hold any value in the component's state, which the result
# merges over the one-shot triggers.
KEEPER_KEY = "bv_board_keeper"      # the component's key (Streamlit owns this state entry)
KEEP_SAVE = "bv_keep_save"          # pending save payload
KEEP_CLEAR = "bv_keep_clear"        # pending clear {"id", "owner"}
KEEP_SENT = "bv_keep_sent"          # runs a pending save or clear has been sent
KEEP_RESEND = 4                     # give up on an unacknowledged write after this many runs
KEEP_DONE = "bv_keep_done"          # the browser answered this session's restore request
KEEP_NONCE = "bv_keep_nonce"        # this session's id for its restore request
KEEP_FAILED = "bv_keep_failed"      # the browser would not keep the board
KEEP_RESTORED = "bv_keep_restored"  # lines brought back (for the toast)
KEEP_ADOPTED = "bv_keep_adopted"    # owner of the copy this session restored
KEEPER_STORAGE_KEY = "bucketvalue.board.v1"
KEEPER_JS = """
// BucketValue keeps the visitor's pasted board in THIS browser so a reload
// brings it back (props_ui.board_keeper). One storage key of this site's own
// origin; no network. Each id is handled once per page load, because
// Streamlit calls this again whenever `data` changes. Clear runs before
// restore, so a Clear all can never be undone by a pending restore request.
const KEY = 'bucketvalue.board.v1';
const seen = new Set();
function storage() {
  try { return window.localStorage; } catch (e) { return null; }
}
function readCopy(store) {
  try {
    const raw = store ? store.getItem(KEY) : null;
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return { unreadable: true };
  }
}
export default function(component) {
  const { data, setTriggerValue } = component;
  const d = data || {};
  const store = storage();
  if (d.clear && d.clear.id && !seen.has('c:' + d.clear.id)) {
    seen.add('c:' + d.clear.id);
    const copy = readCopy(store);
    const owner = (copy && copy.owner !== undefined) ? copy.owner : null;
    const owners = Array.isArray(d.clear.owners) ? d.clear.owners : [];
    try {
      if (store && copy && (copy.unreadable || owners.indexOf(owner) >= 0)) {
        store.removeItem(KEY);
      }
    } catch (e) { /* nothing to remove */ }
    setTriggerValue('cleared', d.clear.id);
  }
  if (d.save && d.save.id && !seen.has('s:' + d.save.id)) {
    seen.add('s:' + d.save.id);
    try {
      if (!store) { throw new Error('no storage'); }
      store.setItem(KEY, JSON.stringify(d.save));
      setTriggerValue('saved', d.save.id);
    } catch (e) {
      // A failed write must not leave an OLDER copy to come back later.
      try { if (store) { store.removeItem(KEY); } } catch (e2) { /* blocked */ }
      setTriggerValue('failed', String((e && e.name) || 'Error').slice(0, 60));
    }
  }
  if (d.want && !seen.has('w:' + d.want)) {
    seen.add('w:' + d.want);
    const copy = readCopy(store);
    if (copy) { setTriggerValue('restore', copy); }
  }
}
"""

# Additions to the theme for this module only. Tokens come from theme.py.
PROPS_CSS = """
<style>
.bv-pp-more { font-style: normal; font-weight: 700; color: var(--accent-teal);
    margin-left: 0.35rem; }
.bv-pp-tag { display: inline-block; margin-left: 0.35rem; padding: 0.02rem 0.34rem;
    border-radius: 999px; font-size: 0.54rem; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase; vertical-align: middle;
    border: 1px solid var(--panel-line); color: var(--fg-4); cursor: help; }
.bv-pp-tag.void { border-color: var(--amber); color: var(--amber); }
.bv-pp-tag.pick { border-color: var(--accent-teal); color: var(--accent-teal); }
.bv-pp-ledger-note { color: var(--fg-5); font-size: 0.74rem; margin: 0.1rem 0 0.4rem; }
</style>
"""

# The grid of five books (layout C, amendment C13), injected beside PROPS_CSS
# only on a board that carries the odds feed's rows: a page with the feed off
# keeps exactly the markup it had before the feed.
BOOKS_CSS = """
<style>
/* The grid of five books (layout C, amendment C13), drawn only on a board that
   carries the odds feed's rows, so a page with the feed off keeps exactly the
   markup it had before the feed. Theme tokens only, so the Dark toggle
   recolours it. The wrapper scrolls sideways on a phone; the Player column
   (the Game page's Stat column) stays pinned. The wrapper is positioned, so
   the visually hidden text inside it is clipped and scrolled by it and can
   never widen the page. */
.bv-grid-wrap { position: relative; overflow-x: auto; overflow-y: hidden; max-width: 100%;
    margin: 0.2rem 0 0.6rem; background: var(--panel-solid); border: 1px solid var(--panel-line);
    border-radius: 12px; -webkit-overflow-scrolling: touch; }
table.bv-grid { width: 100%; min-width: 36rem; font-variant-numeric: tabular-nums;
    font-size: 0.85rem; }
/* Streamlit's markdown styles every table (collapsed borders, a bottom margin,
   a border round every cell); these win on specificity, not on source order. */
.bv-grid-wrap table.bv-grid { margin: 0; border-collapse: separate; border-spacing: 0; }
table.bv-grid.bv-grid-pl { min-width: 32rem; font-size: 0.78rem; }
table.bv-grid th, table.bv-grid td { padding: 0.42rem 0.55rem; text-align: right;
    vertical-align: middle; white-space: nowrap; border: 0;
    border-bottom: 1px solid var(--hairline-soft); }
table.bv-grid tbody tr:last-child th, table.bv-grid tbody tr:last-child td { border-bottom: none; }
table.bv-grid thead th { font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--fg-4);
    background: var(--panel-2); border-bottom: 1px solid var(--panel-line); }
table.bv-grid tbody th { font-weight: 400; }
table.bv-grid .pl, table.bv-grid .st { text-align: left; white-space: normal; }
table.bv-grid .pl { position: sticky; left: 0; z-index: 1; min-width: 9rem;
    background: var(--panel-solid); box-shadow: 1px 0 0 var(--panel-line); }
table.bv-grid .pl .nm, table.bv-grid .pl i { max-width: 12rem; overflow-wrap: anywhere; }
table.bv-grid.bv-grid-pl .st { position: sticky; left: 0; z-index: 1; min-width: 7rem;
    background: var(--panel-solid); box-shadow: 1px 0 0 var(--panel-line); }
table.bv-grid thead th.pl, table.bv-grid.bv-grid-pl thead th.st { z-index: 2;
    background: var(--panel-2); }
table.bv-grid .pl .nm { display: block; color: var(--fg-1); font-weight: 600; line-height: 1.25; }
table.bv-grid .pl i { display: block; font-style: normal; font-size: 0.66rem; color: var(--fg-5);
    font-weight: 500; }
table.bv-grid td.st, table.bv-grid tbody th.st { color: var(--fg-3); font-size: 0.78rem; }
table.bv-grid .num { color: var(--fg-2); }
table.bv-grid .num.act { color: var(--fg-1); font-weight: 700; }
table.bv-grid .bk { text-align: center; }
table.bv-grid abbr { text-decoration: none; border: 0; cursor: help; }
.bv-cell { display: inline-block; min-width: 3.4rem; padding: 0.12rem 0.4rem; border-radius: 6px;
    text-align: center; font-weight: 600; color: var(--fg-3); }
.bv-cell.over { color: var(--value-good); background: var(--tint-good); }
.bv-cell.under { color: var(--value-bad); background: var(--tint-bad); }
.bv-cell.none { color: var(--fg-5); font-weight: 500; }
.bv-cell .ar { margin-left: 0.2rem; font-weight: 700; }
.bv-sr { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
    clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
table.bv-grid tr.note td { text-align: left; white-space: normal; padding-top: 0; }
.bv-grid-legend { color: var(--fg-5); font-size: 0.72rem; margin: 0.1rem 0 0.5rem; }
.bv-gap-chip .gc-book { color: var(--fg-4); font-size: 0.7rem; font-weight: 700; }
.bv-gap-chip abbr, table.bv-gap-table abbr { text-decoration: none; border: 0; cursor: help; }
/* The Streamlit element that holds a grid never widens with it: found by what
   it contains, inside the two keyed containers the pages draw around the
   grids, never by a path through Streamlit's wrappers. */
[data-testid="stVerticalBlock"].st-key-bv_board_grid [data-testid="stElementContainer"]:has(.bv-grid-wrap),
[data-testid="stVerticalBlock"].st-key-bv_game_lines [data-testid="stElementContainer"]:has(.bv-grid-wrap) {
    min-width: 0; max-width: 100%; }
@media (max-width: 760px) {
    table.bv-grid { font-size: 0.8rem; }
    table.bv-grid th, table.bv-grid td { padding: 0.38rem 0.4rem; }
    table.bv-grid .pl { min-width: 7rem; }
    table.bv-grid .pl .nm, table.bv-grid .pl i { max-width: 8.5rem; }
    table.bv-grid.bv-grid-pl .st { min-width: 5.5rem; }
    .bv-cell { min-width: 3rem; padding: 0.1rem 0.3rem; }
}
</style>
"""


def _inject_css(books: bool = False) -> None:
    st.markdown(PROPS_CSS, unsafe_allow_html=True)
    if books:
        st.markdown(BOOKS_CSS, unsafe_allow_html=True)


def _notice(body_html: str) -> str:
    return f'<div class="bv-notice">{body_html}</div>'


def _note(body_html: str) -> str:
    return f'<div class="bv-note">{body_html}</div>'


def _fmt_date(d) -> str:
    """'Wed Oct 21' from an ISO date or a date."""
    try:
        ts = pd.Timestamp(d)
    except (TypeError, ValueError):
        return esc(d)
    return f"{ts.strftime('%a')} {ts.strftime('%b')} {ts.day}"


_MD_SPECIAL = set("\\`*_{}[]()<>#+-.!|~$:")


def md_escape(text) -> str:
    """Backslash-escape Markdown so pasted text shown through st.info /
    st.success (which render Markdown) stays literal: no links, no emphasis,
    no HTML. esc() is for unsafe_allow_html blocks; this is for Markdown ones."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in str(text or ""))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def paste_time_text(meta) -> str:
    """'Last paste 7:05 PM ET, Thu Nov 5.' from meta['pasted_utc'] (when the
    visitor's board was last written), or '' when unknown. Always with the
    date: an 18-hour board can outlive the day it was pasted on, and lines
    move, so the reader needs to see how old they are."""
    t = (meta or {}).get("pasted_utc")
    if t is None:
        return ""
    try:
        et = to_et(t)
    except (TypeError, ValueError):
        return ""
    return f"Last paste {fmt_tip(et)}, {_fmt_date(et)}."


NO_LINES_SLATE = ("<b>No PrizePicks lines on this page.</b> Paste a board with "
                  "<b>Update lines</b> in the top bar: the model's comparison and the "
                  "biggest discrepancies appear here.")
NO_LINES_GAME = ("<b>No PrizePicks lines on this page.</b> Paste a board with "
                 "<b>Update lines</b> in the top bar: each player's lines open under his "
                 "team, and the biggest discrepancies appear here.")
# The same prompts with the odds feed on (design 2.6): the board fills by
# itself, and the paste box stays the backup.
NO_LINES_SLATE_FEED = ("<b>No lines on this page yet.</b> On game days the board fills from "
                       "the odds feed from about 9 AM ET. You can also paste a PrizePicks "
                       "board with <b>Update lines</b> in the top bar.")
NO_LINES_GAME_FEED = ("<b>No lines for this game yet.</b> On game days they load from the "
                      "odds feed from about 9 AM ET, and each player's lines open under his "
                      "team. You can also paste a PrizePicks board with <b>Update lines</b> "
                      "in the top bar.")


# -- The token ----------------------------------------------------------------
def board_token() -> str | None:
    """This visitor's board token from ?board=, or None when absent, malformed
    or expired. Validated before it touches the store."""
    raw = st.query_params.get("board")
    if raw and props.valid_token(raw) and props.get_board(raw) is not None:
        return raw
    return None


def keep_params(token: str | None = None) -> dict:
    """View state every minted internal URL must carry so a paste survives the
    full page load: {'board': token} when the visitor has a board."""
    token = token if token is not None else board_token()
    return {"board": token} if token else {}


# -- One definition of what the board shows -----------------------------------
@dataclass
class BoardView:
    """What the board shows for one ET date (board_view, date_board).

    shown, hidden, thin, void   props.board_rows of the board's rows: with the
                  feed on, EVERY book's lines (layout C, amendment C5), line
                  by line, which count and colour; props.grid_rows makes the
                  grid's rows from shown and hidden.
    meta          the ledger's counts, balanced over those four.
    lines         the visitor's pasted lines for the date.
    With the feed on (date_board's `feed`), props.feed_board's facts too:
    books_by_game {game_id: the books with lines for it, in BOOK_ORDER}.
    missing       {book: the game ids, in tip order, it has not posted while
                  another book (or the paste) has}; empty with the feed off.
    feed          the feed's plain facts (store.feed_info) plus this board's:
                  "missing", "paste_won", "feed_won", "paste_times", "pasted"
                  (pasted lines for the date), "counts" ({"feed": F,
                  "pasted": P} lines on file) and the paste's own unmatched
                  and ambiguous names. None with the feed off: then every
                  renderer gives today's HTML.
    paste_won, feed_won, paste_times   the PrizePicks merge (props.FeedBoard).
    paste_only    build amendment B17: the feed's lines could not be compared
                  (props.FeedTables.failed) and the visitor has a paste for the
                  date, so this is the paste's own board exactly as with the
                  feed off (feed None); the page puts PASTE_ONLY_NOTE above it.
    """
    shown: pd.DataFrame
    hidden: pd.DataFrame
    thin: pd.DataFrame
    void: pd.DataFrame
    meta: dict = field(default_factory=dict)
    lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    books_by_game: dict = field(default_factory=dict)    # props.FeedBoard's
    missing: dict = field(default_factory=dict)          # props.FeedBoard's
    feed: dict | None = None
    paste_won: int = 0
    feed_won: int = 0
    paste_times: dict = field(default_factory=dict)
    paste_only: bool = False

    @property
    def matched(self) -> int:
        return len(self.shown) + len(self.hidden) + len(self.thin) + len(self.void)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=props.TABLE_COLUMNS)


def board_view(token, date, provider, projections, roster=None, void_ids=()) -> BoardView:
    """compare() + board_rows() for one ET date, once. NEVER raises: a failure
    renders as an explained empty board, never as a traceback on a public page.
    Everything is inside the try, the date lookup included (a bad date raised
    from outside it before the audit)."""
    lines = pd.DataFrame()
    try:
        board = props.get_board(token) if token else None
        lines = props.lines_for_date(board, date)
        table, meta = props.compare(lines, provider, projections, roster)
        shown, hidden, thin, void = props.board_rows(table, void_ids)
        info = props.board_info(token) if token else None
        meta = {**meta, "pasted_utc": info.get("pasted_utc") if info else None}
    except Exception as e:  # noqa: BLE001
        logger.warning("board_view failed (%s: %s)", type(e).__name__, e)
        meta = {"posted": int(len(lines)), "error": True}
        return BoardView(_empty(), _empty(), _empty(), _empty(), meta, lines)
    return BoardView(shown, hidden, thin, void, meta, lines)


def void_ids(projections: pd.DataFrame | None) -> set[int]:
    """The players whose picks PrizePicks voids, for board_rows: exactly the
    rows the served frame reads Out (the report, and any mark-out the live
    layer APPLIED, which reads Out too). Never the raw ?out= marks: on a
    frozen game or the committed stand-in the marks are not applied, the
    page shows the player with numbers, and voiding his lines there made the
    board and the card pill contradict the page."""
    if projections is None or not len(projections) or "inj_status" not in projections.columns:
        return set()
    return props.void_id_set(projections.loc[projections["inj_status"] == "Out", "person_id"].tolist())


def _failed_view(token, date) -> BoardView:
    """The explained empty board (render_board: "The comparison could not be
    built for this date. Your lines are still saved"), never a ledger that
    blames the visitor's lines for a server-side failure."""
    try:
        posted = len(props.lines_for_date(props.get_board(token) if token else None, date))
    except Exception:  # noqa: BLE001
        posted = 0
    return BoardView(_empty(), _empty(), _empty(), _empty(), {"posted": posted, "error": True})


def date_board(token, date, projections: pd.DataFrame | None, params, roster=None,
               feed=None, feed_info=None) -> BoardView:
    """board_view for one ET date with the simulator provider. The provider is
    simprov.provider_for(projections, params), the one the Game tables' medians
    read, so a line's model value and the table's median come from one
    simulation; the void set is void_ids(projections). Never raises.

    `params` None (the simulator parameters did not load) is the failed view
    at once: a provider built on it prices nothing, and every line would read
    as "no projection: internal error".

    `feed` is the odds feed's compared tables for the date (props.FeedTables,
    store.feed_tables) and `feed_info` its plain facts (store.feed_info).
    With feed=None this is the function it was before the feed. With a feed,
    the paste is compared as today and props.feed_board makes the board of
    every book (layout C: the paste merged into PrizePicks' rows, newest
    wins), all inside one try, and the view carries the feed's facts
    (BoardView)."""
    if feed is not None:
        return _feed_board_view(token, date, projections, params, roster, feed, feed_info)
    if params is None:
        logger.warning("date_board: no simulator parameters")
        return _failed_view(token, date)
    try:
        provider = simprov.provider_for(projections, params)
    except Exception as e:  # noqa: BLE001 - a broken frame is an explained empty board
        logger.warning("date_board: no provider (%s: %s)", type(e).__name__, e)
        return _failed_view(token, date)
    return board_view(token, date, provider, projections, roster, void_ids(projections))


def _feed_lines(feed) -> int:
    """The feed lines every book of BOOK_ORDER posted for the date (every
    game): the tables' metas count each one, compared or not."""
    metas = feed.metas if isinstance(feed, props.FeedTables) else {}
    return sum(int(m.get("posted", 0) or 0) for (b, _gid), m in metas.items()
               if b in props.BOOK_ORDER and isinstance(m, dict))


def _failed_feed_view(token, date, feed, facts: dict) -> BoardView:
    """The explained empty board with the feed on: render_board says the
    comparison could not be built, and whose lines are still kept (the
    visitor's pasted ones, or the odds feed's in memory). The lines counted
    are the visitor's pasted lines for the date and the feed's: every line
    it holds for the date (the facts' "lines") when the tables could not be
    built at all (props.FeedTables.failed), else every line its tables'
    metas count (amendment C5)."""
    try:
        pasted = len(props.lines_for_date(props.get_board(token) if token else None, date))
    except Exception:  # noqa: BLE001
        pasted = 0
    feed_part = (int(facts.get("lines", 0) or 0) if getattr(feed, "failed", False)
                 else _feed_lines(feed))
    facts.update(missing={}, paste_won=0, feed_won=0, paste_times={}, pasted=pasted,
                 counts={"feed": feed_part, "pasted": pasted}, paste_unmatched=[],
                 paste_ambiguous=[])
    return BoardView(_empty(), _empty(), _empty(), _empty(),
                     {"posted": feed_part + pasted, "error": True}, feed=facts)


# Build amendment B17: above the paste's own board when the feed's lines could
# not be compared (one bv-note, PASTE_ONLY_NOTE).
PASTE_ONLY_NOTE = ("The odds feed's lines could not be compared just now, so your pasted lines "
                   "are shown.")


def paste_only_note_html() -> str:
    """B17's one note, drawn by the page above a paste_only board."""
    return _note(PASTE_ONLY_NOTE)


def _paste_board(token, date, projections, params, roster) -> BoardView | None:
    """Build amendment B17: tables that could not be compared, and a paste
    for the date. The paste's own board, exactly as with the feed off
    (date_board with no feed), marked paste_only. None with no pasted line
    for the date, or when the paste's own board could not be built either:
    the failure notice as built explains it then."""
    if not token:
        return None
    try:
        if not len(props.lines_for_date(props.get_board(token), date)):
            return None
    except Exception:  # noqa: BLE001 - no paste to show: the failure notice as built
        return None
    view = date_board(token, date, projections, params, roster)
    if view.meta.get("error"):
        return None
    view.paste_only = True
    return view


def _feed_board_view(token, date, projections, params, roster, feed, feed_info) -> BoardView:
    """date_board with the odds feed on (layout C, amendment C5): the
    visitor's paste compared as today, then props.feed_board for every book,
    and board_rows on its lines, all inside one try. Tables marked failed
    (store.feed_tables could not compare the feed's lines) are the paste's
    own board when the visitor has a paste for the date (B17), else the
    explained empty board at once, as missing simulator parameters are."""
    facts = dict(feed_info) if isinstance(feed_info, dict) else {}
    if params is None:
        logger.warning("date_board: no simulator parameters")
        return _failed_feed_view(token, date, feed, facts)
    if getattr(feed, "failed", False):
        pasted = _paste_board(token, date, projections, params, roster)
        return pasted if pasted is not None else _failed_feed_view(token, date, feed, facts)
    lines = pd.DataFrame()
    try:
        provider = simprov.provider_for(projections, params)
        board = props.get_board(token) if token else None
        lines = props.lines_for_date(board, date)
        table, pmeta = props.compare(lines, provider, projections, roster)
        info = props.board_info(token) if token else None
        fb = props.feed_board(feed, table, pmeta, info.get("pasted_utc") if info else None)
        shown, hidden, thin, void = props.board_rows(fb.lines, void_ids(projections))
    except Exception as e:  # noqa: BLE001 - an explained empty board, never a traceback
        logger.warning("date_board: the feed board failed (%s)", type(e).__name__)
        return _failed_feed_view(token, date, feed, facts)
    meta = dict(fb.meta)
    on_file = int(meta.get("posted", 0) or 0)
    # The pasted lines still on file: the paste's own, less the keys the
    # feed's newer line won (props.feed_board's merge, M5).
    from_paste = min(on_file, max(0, int(pmeta.get("posted", 0) or 0) - int(fb.feed_won)))
    facts.update(missing={b: list(g) for b, g in fb.missing.items()},
                 paste_won=int(fb.paste_won), feed_won=int(fb.feed_won),
                 paste_times=dict(fb.paste_times), pasted=int(len(lines)),
                 counts={"feed": on_file - from_paste, "pasted": from_paste},
                 paste_unmatched=list(pmeta.get("unmatched_names") or []),
                 paste_ambiguous=list(pmeta.get("ambiguous_names") or []))
    return BoardView(shown, hidden, thin, void, meta, lines,
                     books_by_game=dict(fb.books_by_game),
                     missing={b: list(g) for b, g in fb.missing.items()}, feed=facts,
                     paste_won=int(fb.paste_won), feed_won=int(fb.feed_won),
                     paste_times=dict(fb.paste_times))


def line_counts_by_game(shown: pd.DataFrame) -> dict:
    """{game_id: SHOWN lines}. Feeds the micro-pill on each slate card. Counts
    shown rows only, so the card promises exactly what the game page shows."""
    if shown is None or shown.empty or "game_id" not in shown.columns:
        return {}
    return {str(g): int(n) for g, n in shown.groupby("game_id").size().items()}


def rows_by_person(shown: pd.DataFrame) -> dict:
    """{person_id: [row dicts]}, ranked order kept, for the game page's
    expandable player rows."""
    out: dict[int, list] = {}
    if shown is None or shown.empty:
        return out
    for rec in shown.to_dict(orient="records"):
        out.setdefault(int(rec["person_id"]), []).append(rec)
    return out


# -- The paste flow -----------------------------------------------------------
def ingest(text, token, roster, schedule, now=None, dates=None) -> tuple[str | None, dict]:
    """Parse, prepare and add a paste to the visitor's board. Pure apart from
    the in-process store, so it is testable without a Streamlit runtime.
    Returns (token holding the board, report). Never raises."""
    report: dict = {"error": False, "nonempty": bool(str(text or "").strip())}
    try:
        lines, rep = props.prepare_paste(text, roster, schedule, now=now, dates=dates)
        report.update(rep)
        if lines is not None and not lines.empty:
            token = props.add_to_board(token, lines)
            info = props.board_info(token) or {}
            report["on_board"] = int(info.get("rows", 0))
            report["dropped_by_cap"] = int(info.get("dropped", 0))
            # Lines this paste's merge held back as suspected quarter or half
            # lines sitting far below a full-game line already on the board
            # (props.collapse_alt_lines): counted, never loaded.
            if "period_held" in info:
                report["period_held"] = int(info.get("period_held") or 0)
                report["period_held_by_date"] = dict(info.get("period_held_by_date") or {})
                # Of those, the ones held below a typed pick's number alone,
                # whose message names the pick (review 4, A8, B5).
                report["period_held_pick"] = int(info.get("period_held_pick") or 0)
                report["period_held_pick_names"] = list(
                    info.get("period_held_pick_names") or [])[:8]
            # Lines the merge set aside for a typed pick at another number
            # (props.collapse_alt_lines, review 3, R8): counted, never loaded.
            # The paste's own were never counted as comparable.
            merged = int(info.get("pick_kept") or 0)
            report["pick_kept_merge"] = merged
            report["pick_kept_by_date"] = dict(info.get("pick_kept_by_date") or {})
            report["pick_kept"] = int(report.get("pick_kept", 0) or 0) + merged
            names = list(report.get("pick_kept_names") or []) + \
                list(info.get("pick_kept_names") or [])
            report["pick_kept_names"] = list(dict.fromkeys(names))[:8]
    except Exception as e:  # noqa: BLE001 - never a traceback on the page
        logger.warning("paste ingest failed (%s: %s)", type(e).__name__, e)
        report["error"] = True
    return token, report


def _on_add(view_date, schedule, roster, now, dates) -> None:
    text = st.session_state.get(PASTE_KEY) or ""
    before = board_token()
    token, report = ingest(text, before, roster, schedule, now=now, dates=dates)
    report["view_date"] = str(pd.Timestamp(view_date).date())
    st.session_state[REPORT_KEY] = report
    if token and token != before and props.get_board(token) is not None:
        st.query_params["board"] = token
        _queue_save(token)
    n = toast_count(report)
    if n:
        st.session_state[TOAST_KEY] = n
    # Apply each paste ONCE. Left in the box, it would be re-read on every
    # rerun, and a Clear all would be undone by the next script body.
    st.session_state[PASTE_KEY] = ""


def _on_clear() -> None:
    """Visitor-only: drops the board token from THIS visitor's link and this
    browser's copy (board_keeper). The board itself is not deleted: anyone
    else holding the link (it rides in every URL, Share included) keeps it,
    and it ages out of the in-process store. Nothing on the server's disk is
    touched."""
    token = board_token()
    st.session_state[PASTE_KEY] = ""
    st.session_state.pop(REPORT_KEY, None)
    st.session_state.pop(TOAST_KEY, None)
    st.session_state[KEEP_FAILED] = False
    if "board" in st.query_params:
        del st.query_params["board"]
    # ...and this browser's copy, if the copy is this board's, or the next
    # load would bring the lines back.
    st.session_state[KEEP_SAVE] = None
    if token is not None:
        # This board's copy: saved under this token, or the copy this session
        # restored it from (its re-save may not have reached the browser yet).
        owners = [_owner(token)]
        adopted = st.session_state.get(KEEP_ADOPTED)
        if isinstance(adopted, str):
            owners.append(adopted)
        st.session_state[KEEP_CLEAR] = {"id": secrets.token_hex(8), "owners": owners}
        st.session_state[KEEP_SENT] = 0


def _empty_reason(report: dict) -> str:
    p = report.get("parse") or {}
    if p.get("too_large"):
        return ("That paste is too large to read. Copy one feed page or one "
                "board at a time.")
    if p.get("skipped_noname"):
        return (f"That JSON has {p['skipped_noname']} lines but no player names "
                "(its included player list is missing), so nothing was added. "
                "Re-open the feed and copy the WHOLE page, or paste the board "
                "text instead.")
    if p.get("skipped_live"):
        return ("Every square in that paste is a live, in-game line, and those "
                "are not priced. Paste the pre-game board.")
    if p.get("skipped_badline"):
        return "None of the lines in that paste had a readable number."
    if p.get("json_shape") == props.EMPTY_FEED:
        return ("That is the PrizePicks feed, but it has no lines on it right now. "
                "Lines post through the day; try again later.")
    if p.get("json_shape"):
        return ("That is JSON, but not the PrizePicks projections feed. Open the "
                "feed link above and copy the whole page.")
    if p.get("typed_noname"):
        one = int(p["typed_noname"]) == 1
        return (f"{_plural(p['typed_noname'], 'line')} in that paste "
                f"{'reads like a pick but names' if one else 'read like picks but name'} "
                "no player in full, so nothing was added. Write each player's first and last "
                "name, as in Miles Bridges over 6.5 Rebounds + Assists.")
    if p.get("typed_unread"):
        one = int(p["typed_unread"]) == 1
        return (f"{_plural(p['typed_unread'], 'line')} in that paste "
                f"{'looks like a pick but holds' if one else 'look like picks but hold'} "
                "more than one number, so nothing was added. Write one pick per line with "
                "its line as the only number, as in Miles Bridges over 6.5 Rebounds + Assists.")
    if p.get("typed_unclear"):
        one = int(p["typed_unclear"]) == 1
        reads = ("reads like a pick, but which side it takes" if one
                 else "read like picks, but which side they take")
        return (f"{_plural(p['typed_unclear'], 'line')} in that paste {reads} "
                "could not be told (a Yes or No that may belong to a note), so nothing was "
                "added. Write the side you take, as in Jalen Brunson under 25.5 points.")
    if p.get("typed_recap"):
        one = int(p["typed_recap"]) == 1
        return (f"{_plural(p['typed_recap'], 'line')} in that paste "
                f"{'reads like a result' if one else 'read like results'}, not "
                f"{'a pick' if one else 'picks'} (a word such as cashed, missed or won), so "
                "nothing was added. Paste picks for games still to come, one per line, as in "
                "Miles Bridges over 6.5 Rebounds + Assists.")
    return ("Nothing readable in that paste. Paste the feed JSON, the board text, or a "
            "list with one pick per line, such as Miles Bridges over 6.5 Rebounds + "
            "Assists.")


def toast_count(report: dict | None) -> int:
    """Lines the toast may announce: the ones that can reach a board at all
    (a mapped stat, a matched player, an upcoming game), less those the merge
    held back as period lines or set aside for a typed pick. Never `kept`,
    which counts refused, unrecognised and unmatched lines too (audit
    2026-10-09)."""
    if not report or report.get("error"):
        return 0
    held = int(report.get("period_held", 0) or 0)
    aside = int(report.get("pick_kept_merge", 0) or 0)
    return max(0, int(report.get("comparable", 0) or 0) - held - aside)


_WHY_WORDS = {"not priced": "not priced (period props, Dunks, two-player squares and the like)",
              "other board": "from another board",
              "unrecognised stat": "a stat we do not recognise",
              "not matched": "not matched to a current roster",
              "not projected": "for a game the site does not project"}


def _why_not(nc: dict) -> str:
    """'2 not priced (...); 1 from another board' for the report's
    not-comparable counts. A reason props adds before it has words here is
    still said (by its key), never a KeyError inside the paste panel."""
    bits = [f"{int(nc[k])} {_WHY_WORDS.get(k, k)}" for k in props.NOT_COMPARABLE if nc.get(k)]
    return "; ".join(bits)


def report_messages(report: dict | None) -> list[tuple[str, str]]:
    """[(kind, text)] for a paste report; kind is success, info or warning.
    Plain text, rendered by st.success / st.info / st.warning (no HTML).

    The green banner counts COMPARABLE lines only: a matched player, a mapped
    stat and an upcoming game. Lines that can never show are explained, never
    counted as loaded."""
    if not report:
        return []
    out: list[tuple[str, str]] = []
    if report.get("error"):
        return [("warning", "That paste could not be read, so nothing was added.")]
    kept = int(report.get("kept", 0) or 0)
    unrouted = int(report.get("unrouted", 0) or 0)
    comparable = toast_count(report)
    nc = report.get("not_comparable") or {}
    why = _why_not(nc)
    if report.get("nonempty") and kept <= 0:
        return [("warning", _empty_reason(report))]
    view = report.get("view_date")
    held_by = report.get("period_held_by_date") or {}
    kept_by = report.get("pick_kept_by_date") or {}
    by_date = {d: int(n) - int(held_by.get(d, 0) or 0) - int(kept_by.get(d, 0) or 0)
               for d, n in (report.get("comparable_by_date") or {}).items()}
    by_date = {d: n for d, n in by_date.items() if n > 0}
    here = int(by_date.get(view, 0)) if view else 0
    held = int(report.get("period_held", 0) or 0)
    below = min(held, int(report.get("period_held_pick", 0) or 0))
    wrong = report.get("wrong_board")
    if held - below:
        one = held - below == 1
        out.append(("warning", f"{_plural(held - below, 'line')} sat far below the full-game "
                    f"line already on your board, so {'it was' if one else 'they were'} held "
                    "back: that looks like a quarter or half board. Paste the full-game board."))
    if below:
        # Below a typed pick's number with no board line to go by: a quarter
        # square, or the full-game board under a Demon typed above it. Which
        # it is cannot be told, so the message says both (review 4, A8, B5).
        names = ", ".join(md_escape(x) for x in (report.get("period_held_pick_names") or []))
        one = below == 1
        if one:
            said = (f"1 line sat far below the pick you typed for the same player and stat"
                    f"{f' ({names})' if names else ''}, so it was held back and the board keeps "
                    "your pick. If that paste is the full-game board, type the pick again at "
                    "the board's number to compare it.")
        else:
            # Each held line under its own pick (review 5, A20).
            said = (f"{below} lines sat far below the picks you typed for the same players "
                    f"and stats{f' ({names})' if names else ''}, so they were held back and the "
                    "board keeps your picks. If that paste is the full-game board, type those "
                    "picks again at the board's numbers to compare them.")
        out.append(("warning", said))
    if comparable <= 0 and wrong:
        out.append(("warning", f"That is the {md_escape(wrong)} board, not the full-game NBA "
                    "board, so nothing in it can be compared. Open the NBA board (or the "
                    "feed link above) and copy that."))
    aside = int(report.get("pick_kept", 0) or 0)
    also = None                    # the other lines' reasons, said after the set-aside ones
    if comparable > 0:
        if here:
            out.append(("success", f"{_plural(comparable, 'line')} loaded. {here} "
                        f"{'is' if here == 1 else 'are'} for {_fmt_date(view)}: see Model vs "
                        "the board on the slate and on each game page."))
        else:
            out.append(("success", f"{_plural(comparable, 'line')} loaded."))
        if why:
            out.append(("info", f"Also in that paste and not compared: {why}."))
    elif report.get("nonempty") and not wrong and (held or aside):
        # Every comparable line was held back or set aside: their message
        # says why, and the paste's other lines are still explained
        # (review 4, B11).
        also = f"Also in that paste and not compared: {why}." if why else None
    elif report.get("nonempty") and not wrong:
        head = f"None of the {_plural(kept, 'line')} in that paste can be compared"
        out.append(("warning", f"{head}: {why}." if why else f"{head}."))
    if aside:
        # A board line a typed pick outranks at another number (review 3, R8).
        names = ", ".join(md_escape(x) for x in (report.get("pick_kept_names") or []))
        one = aside == 1
        out.append(("info" if comparable > 0 else "warning",
                    f"{_plural(aside, 'line')} in that paste {'was' if one else 'were'} set "
                    "aside for a pick you typed for the same player and stat at another "
                    f"number ({names}). The board keeps your pick's number; type the pick "
                    "again at the board's number to compare that line instead."))
    if also:
        out.append(("info", also))
    for d, n in sorted(by_date.items()):
        if d != view:
            out.append(("info", f"{_plural(int(n), 'line')} saved for {_fmt_date(d)}, "
                        "step the date to see them."))
    tipped = min(int(report.get("tipped", 0) or 0), unrouted)
    if tipped:
        names = ", ".join(md_escape(x) for x in (report.get("tipped_names") or []))
        one = tipped == 1
        out.append(("info", f"{_plural(tipped, 'line')} {'is' if one else 'are'} for a game "
                    f"that has already started ({names}), so {'it' if one else 'they'} "
                    "cannot be compared. A line has to be pasted before its game tips."))
    if unrouted - tipped:
        late = set(report.get("tipped_names") or []) if tipped else set()
        rest = [x for x in (report.get("unrouted_names") or []) if x not in late]
        names = ", ".join(md_escape(x) for x in rest)
        # A third reason since typed lists (review 3, R6): a matchup written
        # on the list ("@ Heat", "Knicks vs Nets") that the schedule does
        # not have coming up.
        out.append(("info", f"{_plural(unrouted - tipped, 'line')} could not be placed on an "
                    f"upcoming game{f' ({names})' if names else ''}. A name with no team, "
                    "a player whose game has already tipped, or a matchup that is not on "
                    "the upcoming schedule cannot be filed."))
    dropped = int(report.get("dropped_by_cap", 0) or 0)
    if dropped:
        out.append(("warning", f"Your board is at its limit of {props.BOARD_MAX_ROWS} lines, "
                    f"so the {dropped} oldest were dropped. Clear all and paste one slate "
                    "at a time."))
    p = report.get("parse") or {}
    if p.get("skipped_badline"):
        out.append(("info", f"{_plural(p['skipped_badline'], 'line')} had no readable "
                    "number and were skipped."))
    if p.get("skipped_live"):
        out.append(("info", f"{_plural(p['skipped_live'], 'live in-game square')} "
                    "skipped. Only pre-game lines are compared."))
    if p.get("skipped_noname"):
        out.append(("info", f"{_plural(p['skipped_noname'], 'line')} carried no player "
                    "name (a team code where the name should be) and were skipped."))
    if p.get("typed_noname"):
        one = int(p["typed_noname"]) == 1
        out.append(("info", f"{_plural(p['typed_noname'], 'line')} "
                    f"{'reads like a pick but names' if one else 'read like picks but name'} "
                    "no player in full (first and last name), so "
                    f"{'it was' if one else 'they were'} skipped."))
    if p.get("typed_unread"):
        one = int(p["typed_unread"]) == 1
        out.append(("info", f"{_plural(p['typed_unread'], 'line')} "
                    f"{'looks like a pick but holds' if one else 'look like picks but hold'} "
                    "more than one number, so "
                    f"{'it was' if one else 'they were'} skipped. Write one pick per line "
                    "with its line as the only number."))
    if p.get("typed_unclear"):
        # "No, I like Jalen Brunson over 25.5 points", "over 25.5 points, no
        # for real": a Yes or No that may be the answer or a note's, never
        # guessed (review 6, B5, B6).
        one = int(p["typed_unclear"]) == 1
        reads = ("reads like a pick, but which side it takes" if one
                 else "read like picks, but which side they take")
        out.append(("info", f"{_plural(p['typed_unclear'], 'line')} {reads} "
                    "could not be told (a Yes or No that may belong to a note), so "
                    f"{'it was' if one else 'they were'} skipped. Write the side you take, "
                    "as in Jalen Brunson under 25.5 points."))
    n = min(int(p.get("typed_period_label") or 0), int(nc.get("not priced", 0) or 0))
    if n:
        # A label, heading or name line naming a period refuses every pick
        # under it, which is the safe side; the report says so, since the
        # picks may be full-game ones under a remark (review 6, B12). Never
        # more of them than the paste's own "not priced" count, and the
        # first minutes said too (review 7, A13).
        one = n == 1
        out.append(("info", f"{_plural(n, 'line')} sat under a line that names a quarter, a "
                    "half or the first minutes (such as 1Q props), so "
                    f"{'it was' if one else 'they were'} read as "
                    f"{'a pick' if one else 'picks'} for that period, which is not priced. "
                    f"If {'it is' if one else 'they are'} for the full game, take the period off "
                    "that line and paste again."))
    if p.get("typed_recap"):
        # "Jalen Brunson o25.5 pts - cashed": a result, never filed as a pick
        # for the next game (review 4, A7, B7). Wherever the word sat, and with
        # advice only for a pick wrongly taken as a result (review 5, B12).
        one = int(p["typed_recap"]) == 1
        out.append(("info", f"{_plural(p['typed_recap'], 'line')} "
                    f"{'reads like a result' if one else 'read like results'}, not "
                    f"{'a pick' if one else 'picks'} (a word such as cashed, missed or won), "
                    f"so {'it was' if one else 'they were'} skipped. "
                    f"{'If it is' if one else 'If any is'} a pick for a game still to come, "
                    "type it without that word to compare it."))
    if p.get("truncated"):
        out.append(("info", "The paste was cut short at its size limit, so some of it "
                    "was not read. Paste one board or one feed page at a time."))
    return out


def render_update_lines_popover(view_date, schedule, roster, now=None, dates=None,
                                in_nav_container: bool = True, feed_on: bool = False) -> str | None:
    """The nav bar's 'Update lines' popover. Call right after render_nav.
    Returns the visitor's board token (or None) for the page's minted URLs.

    in_nav_container=False when the page has already opened the keyed
    `bv_nav_actions` container itself (to put Share beside it): opening a keyed
    container twice is a duplicate-key error. `feed_on`: the odds feed is on,
    and the caption says that the paste is the backup (design 2.6)."""
    raw = st.query_params.get("board")
    if raw and board_token() is None:
        # A token that no longer resolves: expired, or the server restarted.
        st.session_state[EXPIRED_KEY] = True
        del st.query_params["board"]
    if in_nav_container:
        with st.container(key="bv_nav_actions"):
            with st.popover("Update lines"):
                _paste_panel(view_date, schedule, roster, now, dates, feed_on)
    else:
        with st.popover("Update lines"):
            _paste_panel(view_date, schedule, roster, now, dates, feed_on)
    token = board_token()
    placed = st.session_state.pop(TOAST_KEY, None)
    if placed:
        st.toast(f"{_plural(int(placed), 'line')} loaded")
    back = st.session_state.pop(KEEP_RESTORED, None)
    if back:
        st.toast(f"Your {_plural(int(back), 'saved line')} are back" if int(back) != 1
                 else "Your saved line is back")
    return token


def _owner(token) -> str:
    """The board a browser copy belongs to: a short hash of its token, so the
    copy never holds the token itself."""
    return hashlib.sha256(str(token).encode()).hexdigest()[:16]


def _queue_save(token) -> None:
    """Send `token`'s board to this browser (resent until acknowledged). A
    board of past games only has nothing to keep."""
    info = props.board_info(token) or {}
    payload = props.saved_board(props.get_board(token), info.get("pasted_utc"),
                                secrets.token_hex(8), today_et(), owner=_owner(token))
    st.session_state[KEEP_SAVE] = payload
    st.session_state[KEEP_SENT] = 0
    st.session_state[KEEP_FAILED] = False


def _rearm_restore() -> None:
    """A board died mid-session (a restart, a deploy's purge, the 18-hour
    limit): ask the browser again, under a NEW id (the JS answers each id
    once per page load)."""
    st.session_state[KEEP_DONE] = False
    st.session_state[KEEP_NONCE] = secrets.token_hex(6)


def _comparable(lines: pd.DataFrame) -> int:
    """The lines that can show on a board (what the paste toast counts)."""
    if lines is None or lines.empty:
        return 0
    ok = lines["kind"].astype(str).eq("ok") & lines["match"].astype(str).eq("matched")
    return int((ok & lines["game_date"].notna()).sum())


def _noop() -> None:
    """A trigger reaches the mount's result only when an on_<event>_change
    callable is passed; the page reads the answer from the result itself."""


def _keeper_answers(res, roster, token) -> str | None:
    """Act on the browser's answers on this run; returns the link's board
    token afterwards (a restore puts a fresh one on it). Each answer counts
    only while the request it answers is outstanding."""
    pend = st.session_state.get(KEEP_SAVE)
    saved = res.get("saved")
    if saved and isinstance(pend, dict) and saved == pend.get("id"):
        st.session_state[KEEP_SAVE] = None
    clear = st.session_state.get(KEEP_CLEAR)
    cleared = res.get("cleared")
    if cleared and isinstance(clear, dict) and cleared == clear.get("id"):
        st.session_state[KEEP_CLEAR] = None
    if res.get("failed") and isinstance(pend, dict):
        st.session_state[KEEP_SAVE] = None
        st.session_state[KEEP_FAILED] = True
    copy = res.get("restore")
    if copy is None or st.session_state.get(KEEP_DONE) or token is not None:
        return token
    st.session_state[KEEP_DONE] = True
    got = props.restore_saved(copy, roster, today_et(), now_utc())
    owner = copy.get("owner") if isinstance(copy, dict) else None
    owner = owner if isinstance(owner, str) and len(owner) <= 64 else None
    if got is None:                     # nothing usable left: drop it from the browser
        st.session_state[KEEP_CLEAR] = {"id": secrets.token_hex(8), "owners": [owner]}
        st.session_state[KEEP_SENT] = 0
        return None
    lines, pasted, _dropped = got
    token = props.add_to_board(None, lines, pasted_utc=pasted)
    st.query_params["board"] = token
    st.session_state[KEEP_RESTORED] = _comparable(lines)
    st.session_state[KEEP_ADOPTED] = owner
    _queue_save(token)                  # the copy follows the fresh token
    return token


def keeper_data(token) -> dict | None:
    """What the keeper tells the browser on this run (and the run counter
    that bounds the resends). Separate from the mount so a test can read it."""
    data = {}
    for key, field_ in ((KEEP_SAVE, "save"), (KEEP_CLEAR, "clear")):
        pend = st.session_state.get(key)
        if pend:
            data[field_] = pend
    if data:
        sent = int(st.session_state.get(KEEP_SENT) or 0) + 1
        st.session_state[KEEP_SENT] = sent
        if sent > KEEP_RESEND:          # never acknowledged: stop resending
            st.session_state[KEEP_SAVE] = None
            st.session_state[KEEP_CLEAR] = None
            data = {}
    if token is None and not st.session_state.get(KEEP_DONE):
        data["want"] = st.session_state.setdefault(KEEP_NONCE, secrets.token_hex(6))
    return data or None


def board_keeper(roster) -> str | None:
    """Mount the browser keeper. Pages call it at the TOP, after the roster
    loads and before the nav or any widget, on EVERY run (an unmounted
    component loses its state). Returns the link's live board token, with a
    restored board already on the link. A token the server no longer knows
    is dropped here and the restore re-armed. Hidden by theme CSS
    (.st-key-bv_board_keeper); registered inside the run, as AppTest needs."""
    raw = st.query_params.get("board")
    token = board_token()
    if raw and token is None:
        st.session_state[EXPIRED_KEY] = True
        del st.query_params["board"]
        _rearm_restore()
    keeper = st.components.v2.component("bv_board_keeper", js=KEEPER_JS)
    res = keeper(key=KEEPER_KEY, data=keeper_data(token), on_saved_change=_noop,
                 on_cleared_change=_noop, on_failed_change=_noop, on_restore_change=_noop)
    if roster is None:
        return token
    return _keeper_answers(res, roster, token)


# With the odds feed on, the paste caption says so before its last sentence
# (design 2.6): the board's PrizePicks lines come from the licensed feed, and
# the paste merges in by the newest-line rule (props.feed_board, M5), under
# which a typed pick, a Demon or a Goblin of the visitor's always wins (build
# amendment B28: the caption names all three, as the merge line does).
FEED_CAPTION = ("PrizePicks lines on the board come from a licensed odds feed; pasting is the "
                "backup. For the same player and stat, the feed's line is shown unless yours is "
                f"{PASTE_WINNERS}. ")


def _paste_panel(view_date, schedule, roster, now, dates, feed_on: bool = False) -> None:
    if st.session_state.pop(EXPIRED_KEY, False):
        st.info("Your earlier lines are no longer on this link (the site keeps a pasted "
                "board for about 18 hours, and a restart clears it). If this browser "
                "kept a copy they come back on their own; otherwise paste again.")
    if st.session_state.get(KEEP_FAILED):
        st.caption("This browser would not keep a copy of your lines (its storage is blocked "
                   "or full), so opening the site without this link, or after a restart, "
                   "will need a fresh paste.")
    with st.container(border=True):
        st.markdown("**Copy the PrizePicks board and paste it here**")
        st.markdown(
            f'1. Open the <a href="{esc(props.FEED_URL)}" target="_blank" '
            'rel="noopener noreferrer"><b>PrizePicks NBA feed</b></a> in a new tab, '
            "or the board page itself. The feed looks like a wall of code; that IS "
            "the whole board. <b>Turn any VPN off first</b>: PrizePicks rejects VPN "
            "and datacenter addresses and shows an error instead of the board.<br>"
            "2. Select all and copy.<br>"
            "3. Paste below and click <b>Add these lines</b>. Each line lands on the "
            "date of its own game, so an evening copy of tomorrow's board files "
            "itself under tomorrow.<br>"
            "Your own list works too, one pick per line, such as <i>Miles Bridges over "
            "6.5 Rebounds + Assists</i>. A team on a line of its own (<i>Suns:</i>) says "
            "whose players follow, and the Over or Under you wrote shows beside the model's "
            "lean as your pick.",
            unsafe_allow_html=True)
    st.text_area("Paste the feed JSON, the board text or your own list", height=150,
                 key=PASTE_KEY,
                 placeholder=("Paste the copied feed page here, the board text:\n"
                              "Nikola Jokic\nDEN - C\n27.5\nPoints\nLess\nMore\n"
                              "or your own list:\nMiles Bridges over 6.5 Rebounds + Assists"))
    token = board_token()
    has_own = token is not None or bool((st.session_state.get(PASTE_KEY) or "").strip())
    c_add, c_clear = st.columns([2.4, 1], gap="small")
    with c_add:
        st.button("Add these lines", type="primary", key="bv_pp_add", width="stretch",
                  on_click=_on_add, args=(view_date, schedule, roster, now, dates))
    with c_clear:
        st.button("Clear all", key="bv_pp_clear", width="stretch", on_click=_on_clear,
                  disabled=not has_own,
                  help="Remove the lines you pasted. Nobody else's board is touched.")
    show = {"success": st.success, "info": st.info, "warning": st.warning}
    for kind, text in report_messages(st.session_state.get(REPORT_KEY)):
        show.get(kind, st.info)(text)
    board = props.get_board(token) if token is not None else None
    if board is not None:
        st.caption(f"{_plural(len(board), 'line')} on your board across every date.")
    st.caption(
        "A paste ADDS to your board, so stat tabs can come in one at a time, and "
        "alt-line ladders collapse to one line per player and stat. When you add "
        "lines, this browser keeps a copy of your board, so they come back when you "
        "reload or return, until the day of their games has passed. They are never "
        "saved on our server and never enter any accuracy record, but anyone you send "
        "the link to sees them. Clear all removes them from your link and from this "
        "browser. " + (FEED_CAPTION if feed_on else "") + "Nothing "
        "here fetches from PrizePicks: their API blocks server requests and their "
        "terms prohibit automated access.")


# -- Cells --------------------------------------------------------------------
_TIER_CLASS = {"strong": "t-strong", "moderate": "t-mod", "weak": "t-weak",
               "none": "t-none", "negative": "t-neg", "unmeasured": "t-unk"}
_SRC_BADGE = {
    "baseline": '<span class="bv-src base" title="Served from a baseline, not a '
                'model that passed its gate">baseline</span>',
    "model_low_confidence": '<span class="bv-src low">low conf</span>',
    "composite": '<span class="bv-src comp">composed</span>',
}


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _tier_badge(tier, edge) -> str:
    """How much this stat's lean is worth, MEASURED. 'unmeasured' renders
    quietly (outlined, faint): it is the honest state before validation."""
    t = str(tier or "unmeasured").lower()
    cls = _TIER_CLASS.get(t, "t-unk")
    label = props.TIER_LABEL.get(t, "unmeasured")
    tip = (f"Measured side-picking edge {float(edge):+.1f} points against always "
           "taking the more common side, out of sample." if _finite(edge)
           else "Side-picking accuracy not measured for this stat yet.")
    return f'<span class="bv-tier {cls}" title="{esc(tip)}">{esc(label)}</span>'


def _odds_badge(odds_type) -> str:
    """Demons and Goblins are deliberately SHIFTED lines, so a big gap against
    one is manufactured by the line rather than evidence the market is wrong."""
    o = str(odds_type or "standard").lower()
    if o == "demon":
        return ('<span class="bv-odds demon" title="Demon: set harder than the '
                'standard line, pays more">Demon</span>')
    if o == "goblin":
        return ('<span class="bv-odds goblin" title="Goblin: set easier than the '
                'standard line, pays less">Goblin</span>')
    return ""


def _row_book(r) -> str:
    """The book of a row: its own `book`, else PrizePicks (a pasted line, and
    every row before the feed, is a PrizePicks line)."""
    b = r.get("book") if hasattr(r, "get") else None
    return b.strip().lower() if isinstance(b, str) and b.strip() else props.PASTE_BOOK


def _is_dfs(r) -> bool:
    return _row_book(r) in props.DFS_BOOKS


def _side_word(r, side: str) -> str:
    """The row's own word for a side (design 2.6 m8, amendment A4): More or
    Less on a pick'em row (PrizePicks, Underdog), Over or Under on a
    sportsbook row. The stored lean stays More, Less or Even."""
    more, less = props.side_words(_row_book(r))
    return more if side == "More" else less if side == "Less" else side


def _lean_text(r) -> str:
    """The Lean cell's word, escaped."""
    lean = r.get("lean")
    return esc(_side_word(r, lean) if lean in ("More", "Less") else lean)


def _sides_tag(sides, book=None) -> str:
    """'More only' / 'Less only' on a pick'em row, 'Over only' / 'Under only'
    on a sportsbook row (`book`: the row's book; None is PrizePicks)."""
    s = str(sides or "both").lower()
    if s not in ("more", "less"):
        return ""
    word = _side_word({"book": book}, "More" if s == "more" else "Less")
    return f'<span class="bv-pp-tag" title="This line offers {word} only">{word} only</span>'


def _pick_tag(r) -> str:
    """The side a typed list named ("Pick: More"), beside the model's lean and
    never in place of it: the lean is still the row's colour and its own
    column. Worded for anyone who opens the board, since a shared link shows
    the sharer's picks to someone who did not make them (review 2, I12)."""
    side = props.pick_of(r.get("pick"))
    if side is None:
        return ""
    word = "More" if side == "more" else "Less"
    lean = r.get("lean")
    said = (f"The model leans {lean}." if lean in ("More", "Less")
            else "The model has no lean on this line.")
    tip = f"The pick in the pasted list: {word}. {said}"
    return f'<span class="bv-pp-tag pick" title="{esc(tip)}">Pick: {word}</span>'


def _void_tag(r) -> str:
    """Reboot: PrizePicks voids a More pick if the player leaves in the first
    half and does not return. Nothing measures how often by role yet (an exit
    model needs play-by-play, v1.1), so a More lean with props' `void_note`
    (a bench role) gets the words and never a number. A provider that does
    carry a measured `void_risk` (the hook for that model) gets its number.
    PrizePicks rows only: Reboot is PrizePicks' rule (design 2.6)."""
    if r.get("lean") != "More" or _row_book(r) != props.PASTE_BOOK:
        return ""
    if _finite(r.get("void_risk")):
        pct = round(100 * float(r["void_risk"]))
        tip = (f"PrizePicks voids a More pick if he leaves in the first half and does "
               f"not return. Estimated chance for his role: {pct} in 100. Not priced "
               "into the lean.")
    elif r.get("void_note"):
        tip = ("PrizePicks voids a More pick if he leaves in the first half and does not "
               "return. A bench role carries more of that risk. Not measured yet, and not "
               "priced into the lean.")
    else:
        return ""
    return f'<span class="bv-pp-tag void" title="{esc(tip)}">void risk</span>'


def _tie_text(r) -> str:
    """The chance a line lands exactly on its number, or '' when it cannot
    (the Stat cell's title, and a grid cell's: amendment C6). A pick'em book
    lowers the payout tier on an exact landing; on a sportsbook line the
    book's own rule decides (RC G11, UNVERIFIED). Plain text, unescaped."""
    if not (r.get("tie_possible") and _finite(r.get("p_push")) and _finite(r.get("line"))):
        return ""
    rule = ("That lowers the payout tier; it is not a push." if _is_dfs(r)
            else "On a sportsbook line the book's own rule decides an exact landing.")
    return ("Chance of landing exactly on %g: %d in 100. %s"
            % (float(r["line"]), round(100 * float(r["p_push"])), rule))


def _stat_label(r) -> str:
    """The stat's label (STAT_LABEL, else the row's own), unescaped."""
    stat = r.get("stat")
    return props.STAT_LABEL.get(stat) or str(r.get("stat_label") or stat or "")


def _stat_cell(r) -> str:
    text = _tie_text(r)
    tip = f' title="{esc(text)}"' if text else ""
    return (f'<span class="st"{tip}>{esc(_stat_label(r))}'
            f'{_tier_badge(r.get("tier"), r.get("tier_edge"))}'
            f'{_odds_badge(r.get("odds_type"))}{_sides_tag(r.get("sides"), _row_book(r))}'
            f'{_SRC_BADGE.get(str(r.get("source") or ""), "")}{_void_tag(r)}'
            f'{_pick_tag(r)}</span>')


def _lean_cls(lean) -> str:
    """One colour source for every cell in a row, so a row cannot contradict
    itself (GridironValue coloured the gap from one number and the lean from
    another)."""
    return "over" if lean == "More" else "under" if lean == "Less" else ""


def _fmt_model(r) -> str:
    """The mid-p median. One more decimal when rounding would print the line's
    own number back beside a decided lean ('5.0 vs 5.0, More')."""
    if not _finite(r.get("model")):
        return f'<span class="num">{SENTINEL}</span>'
    val = float(r["model"])
    line = r.get("line")
    if _finite(line) and round(val, 1) == round(float(line), 1):
        return f'<span class="num">{val:.2f}</span>'
    return f'<span class="num">{val:.1f}</span>'


def _fmt_line(r) -> str:
    if not _finite(r.get("line")):
        return f'<span class="num line">{SENTINEL}</span>'
    return f'<span class="num line">{float(r["line"]):g}</span>'


def _fmt_gap(r) -> str:
    """Median minus line, in the stat's own units. Never a percentage: 52%
    dresses a coin flip in a precise-looking number. Safe to show because the
    model value is the mid-p median, whose sign agrees with the lean on every
    whole and half line (tests/test_pmf.py)."""
    cls = _lean_cls(r.get("lean"))
    d = r.get("gap")
    if not _finite(d):
        return f'<span class="num diff">{SENTINEL}</span>'
    d = float(d)
    if cls == "":
        return f'<span class="num diff">{d:+.1f}</span>' if round(d, 1) else \
            '<span class="num diff">0.0</span>'
    v = round(d, 1)
    if v == 0:
        fine = round(d, 2)
        if fine:
            return f'<span class="num diff {cls}">{fine:+.2f}</span>'
        return f'<span class="num diff {cls}">&lt;0.01</span>'
    return f'<span class="num diff {cls}">{v:+.1f}</span>'


def _matchup(r) -> str:
    team, opp = r.get("team"), r.get("opp")
    if team and opp and not (isinstance(opp, float) and math.isnan(opp)):
        return f"{esc(team)} vs {esc(opp)}"
    return esc(team or "")


# -- Keys, stamps and clock text for the feed's rows ---------------------------
def _day_key(v) -> str | None:
    """A game date as 'YYYY-MM-DD' (an ISO string, a date or a Timestamp)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()[:10]
    return s if len(s) == 10 else None


def _pid(v) -> int | None:
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


def _key_of(r) -> tuple | None:
    """(game date, person_id, stat): the board's key for a player's line
    (props.feed_board's merge key, by which merge_books groups lines)."""
    day, pid, stat = _day_key(r.get("game_date")), _pid(r.get("person_id")), r.get("stat")
    if day is None or pid is None or not isinstance(stat, str):
        return None
    return day, pid, stat


def _stamp_of(v) -> pd.Timestamp | None:
    """An aware UTC Timestamp, or None for a missing, naive or unreadable one."""
    if v is None:
        return None
    try:
        ts = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts) or ts.tzinfo is None:
        return None
    return ts.tz_convert("UTC")


def _as_day(v):
    """An ET calendar date from a date or an ISO date string, else None."""
    if v is None:
        return None
    try:
        return pd.Timestamp(str(v)[:10]).date()
    except (TypeError, ValueError):
        return None


def _clock_text(t, today, zone: bool = True) -> str:
    """'9:02 AM ET' ('9:02 AM' without the zone) from a stamp, with ', Mar 11'
    when its ET date is not `today` (store._stamp's rule): text built from
    the stamps the page passes, never from a clock read here (m1)."""
    ts = _stamp_of(t)
    if ts is None:
        return ""
    et = to_et(ts.to_pydatetime())
    text = fmt_tip(et)
    if not zone and text.endswith(" ET"):
        text = text[:-3]
    day = _as_day(today)
    return text + ("" if day is not None and et.date() == day else f", {et.strftime('%b')} {et.day}")


def _paste_word(rec) -> str:
    """The badge word of a pasted PrizePicks line, which its grid cell's
    title carries after the number (build amendment B28; layout C,
    amendment C6), in the order the merge reads them (props._paste_wins): a
    typed pick, then a Demon or a Goblin, else a plain pasted line."""
    if props.pick_of(rec.get("pick")) is not None:
        return "typed"
    kind = str(rec.get("odds_type") or "").lower()
    if kind in ("demon", "goblin"):
        return kind.capitalize()
    return "pasted"


def ledger_html(rows: pd.DataFrame, more: dict | None = None, feed: dict | None = None) -> str:
    """The board's ledger as HTML. Every value escaped. With the odds feed
    (`feed`, BoardView.feed) the board is layout C's grid instead
    (_grid_html: one column per book), and rows handed in without the grid's
    cells (plain shown lines) go through props.grid_rows first."""
    if feed is not None:
        return _grid_html(rows, more, feed)
    more = more or {}
    out = []
    for r in rows.to_dict(orient="records"):
        n_more = int(more.get(int(r["person_id"]), 0)) if r.get("person_id") is not None else 0
        more_html = f'<b class="bv-pp-more">+{n_more} more</b>' if n_more else ""
        out.append(
            '<div class="bv-ledger-row">'
            f'<span class="nm">{esc(r.get("player"))}<i>{_matchup(r)}{more_html}</i></span>'
            f"{_stat_cell(r)}{_fmt_model(r)}{_fmt_line(r)}{_fmt_gap(r)}"
            f'<span class="lean {_lean_cls(r.get("lean"))}">{_lean_text(r)}</span></div>')
    head = ('<div class="bv-ledger-head"><span class="nm">Player</span>'
            '<span class="st">Stat</span><span class="num">Model</span>'
            '<span class="num">Line</span><span class="num">Gap</span>'
            '<span class="lean">Lean</span></div>')
    return f'<div class="bv-ledger">{head}{"".join(out)}</div>'


# What happened to a line, after tip (store.line_actuals fills `actual` and
# `settle`). PrizePicks voids a pick on a player who does not play; a value on
# the line is neither a win nor a loss, and is never called a push.
VOID_TEXT = "Did not play, pick voided"
# On an Underdog or a sportsbook row, whose own rule decides the pick or the
# bet (design 2.6; build amendments B26 and B30), the note claims no void.
DNP_TEXT = "Did not play"
EXACT_TEXT = "landed on the line"


def actual_text(stat, value) -> str:
    """An actual as the pages print it (the line rows here and the Game
    table's box-score rows): minutes and Fantasy Score to a tenth, counts
    whole; the sentinel when there is none yet (not in the game yet, or a
    component missing: never zero-filled)."""
    if not _finite(value):
        return SENTINEL
    return f"{float(value):.1f}" if stat in ("min", "fantasy") else f"{float(value):g}"


def _actual_text(r) -> str:
    return actual_text(r.get("stat"), r.get("actual"))


def _fmt_actual(r) -> str:
    return f'<span class="num act">{esc(_actual_text(r))}</span>'


def _settle_note(r) -> str:
    """The full-width note under a settled line, or ''. PrizePicks voids a
    pick on a player who did not play: VOID_TEXT on a PrizePicks row, and
    on a row with no book (a pasted line, as before the feed, so the HTML
    with the feed off is unchanged). Every other book's own rule decides
    what happens to a pick or a bet on him (Underdog's: build amendments
    B26 and B30; a sportsbook's: design 2.6), so on its rows the note only
    says he did not play (DNP_TEXT)."""
    s = r.get("settle")
    if s == "void":
        void = _row_book(r) == props.PASTE_BOOK          # PrizePicks, or no book at all
        return f'<div class="bv-pl-note">{VOID_TEXT if void else DNP_TEXT}</div>'
    if s == "exact":
        return f'<div class="bv-pl-note">{esc(_actual_text(r))}: {EXACT_TEXT}</div>'
    return ""


def render_player_lines(rows, context=None, feed: dict | None = None) -> str:
    """One player's SHOWN lines, for his expandable row on the game page.
    `rows` is a list of row dicts (rows_by_person) or a DataFrame. After tip
    (rows carrying `actual`, from store.line_actuals) an Actual column, and
    a note under a settled line: for a player who did not play VOID_TEXT on
    a PrizePicks row, DNP_TEXT on any other book's (build amendment B30),
    EXACT_TEXT for a value on the line. With the odds feed (`feed`,
    BoardView.feed) it is layout C's per-player grid instead
    (_player_grid_html, amendment C7): `context` then holds his HIDDEN lines
    in this game, a list of row dicts or a frame, drawn as the number alone
    beside his shown ones."""
    if feed is not None:
        return _player_grid_html(rows, context, feed)
    if rows is None:
        return ""
    recs = rows.to_dict(orient="records") if isinstance(rows, pd.DataFrame) else list(rows)
    if not recs:
        return ""
    after_tip = any("actual" in r for r in recs)
    body = []
    for r in recs:
        body.append(
            '<div class="bv-pl-row">'
            f"{_stat_cell(r)}{_fmt_model(r)}{_fmt_line(r)}{_fmt_gap(r)}"
            f'<span class="lean {_lean_cls(r.get("lean"))}">{_lean_text(r)}</span>'
            + (_fmt_actual(r) if after_tip else "") + "</div>" + _settle_note(r))
    final = any(r.get("final") for r in recs)
    act_head = (f'<span class="num">{"Final" if final else "So far"}</span>' if after_tip else "")
    head = ('<div class="bv-pl-head"><span class="st">Stat</span>'
            '<span class="num">Model</span><span class="num">Line</span>'
            f'<span class="num">Gap</span><span class="lean">Lean</span>{act_head}</div>')
    cls = "bv-pl has-actual" if after_tip else "bv-pl"
    return f'<div class="{cls}">{head}{"".join(body)}</div>'


# -- Layout C: the grid (amendments C6, C7 and C13) -----------------------------
# One column per book, in props.BOOK_ORDER (DK, FD, MGM, PP, UD). Each cell is
# that book's line with the model's lean at it, decided as the feed-off Lean
# column decides it (props.side_probabilities and lean_from against that
# book's own line, in compare()): on a SHOWN line an arrow and its text
# equivalent (up for More or Over on a green tint, down for Less or Under on
# a red one), on a hidden line the number alone (no lean, or a side the line
# does not offer), and the whole-node sentinel where the book has no line,
# so colour is never the only cue. No price is shown for any book, and every
# book's name comes from props.BOOK_TITLE and BOOK_CODE, never a vendor's.
UP = "&#8593;"           # the arrows, HTML entities inserted raw (never through esc)
DOWN = "&#8595;"
GRID_CAPTION = ("Top {k} of {n} players, one row each (their best-ranked stat). Show all "
                "lists every stat.")
GRID_LEGEND = ("Up arrow: model leans More (Over) at that line. Down arrow: Less (Under). "
               "No arrow: no lean there, or the line does not offer that side. "
               "A dash: no line from that book.")
# The one sentence the hidden notes gain with the feed on when a hidden line
# is drawn beside a shown one (amendment C8).
HIDDEN_DRAWN = ("A hidden line in a row with a shown line is drawn as its number alone, "
                "with no arrow.")


def grid_legend_html() -> str:
    """The one legend line under the slate's grid (amendment C6)."""
    return f'<div class="bv-grid-legend">{esc(GRID_LEGEND)}</div>'


def _grid_head(book: str) -> str:
    """A book's column header: its code, its title in an <abbr>."""
    return (f'<th class="bk" scope="col"><abbr title="{esc(props.BOOK_TITLE[book])}">'
            f'{esc(props.BOOK_CODE[book])}</abbr></th>')


def _grid_stat(r) -> str:
    """The grid's Stat cell content: the label, the tier badge and the source
    badge. The per-line badges sit in each line's own cell."""
    return (f'{esc(_stat_label(r))}{_tier_badge(r.get("tier"), r.get("tier_edge"))}'
            f'{_SRC_BADGE.get(str(r.get("source") or ""), "")}')


def _grid_model(r) -> str:
    """The Model cell: one decimal, two when one decimal would print the line
    of any SHOWN cell of the row back beside its arrow (_fmt_model's rule
    for one line); the whole-node sentinel when there is no model value."""
    if not _finite(r.get("model")):
        return f'<td class="num">{SENTINEL}</td>'
    val = float(r["model"])
    cells = r.get(props.GRID_CELLS) or {}
    lines = [c.get("line") for c in cells.values() if c.get("shown") is True]
    if any(_finite(x) and round(val, 1) == round(float(x), 1) for x in lines):
        return f'<td class="num">{val:.2f}</td>'
    return f'<td class="num">{val:.1f}</td>'


def _grid_latest(rows: pd.DataFrame, feed: dict | None) -> dict:
    """{game_id: the game's latest pull}: the facts' per_game "asof" where the
    page gives it, else the newest fetched_at among the rows' cells (m23)."""
    latest: dict = {}
    if isinstance(rows, pd.DataFrame) and props.GRID_CELLS in rows.columns:
        for cells in rows[props.GRID_CELLS]:
            for rec in (cells or {}).values():
                at = _stamp_of(rec.get("fetched_at"))
                gid = rec.get("game_id")
                if at is not None and isinstance(gid, str) and (gid not in latest or at > latest[gid]):
                    latest[gid] = at
    for gid, facts in ((feed or {}).get("per_game") or {}).items():
        at = _stamp_of(facts.get("asof")) if isinstance(facts, dict) else None
        if at is not None:
            latest[str(gid)] = at
    return latest


def _grid_cell(book: str, rec, latest: dict, today) -> str:
    """One book's cell of a grid row (amendment C6), built from its column's
    own book and that book's record for the row (`rec`, None when the book
    has no line for it), never from the row's lead line. The class reads the
    record's `shown` flag AND its lean: over and under only on a shown line.
    The title names the book, the line (with B28's word on a pasted
    PrizePicks line) and the model's lean, then the chance of an exact
    landing when there is one, then, on a feed line over LINE_TIME_AGE older
    than its game's latest pull, the line's own time (m23, C10)."""
    title = props.BOOK_TITLE[book]
    if not isinstance(rec, dict) or not _finite(rec.get("line")):
        return (f'<td class="bk"><span class="bv-cell none">{SENTINEL}'
                f'<span class="bv-sr"> no {esc(title)} line</span></span></td>')
    n = f"{float(rec['line']):g}"
    lean = rec.get("lean")
    side = _side_word({"book": book}, lean)
    shown = rec.get("shown") is True
    if shown and lean == "More":
        cls, mark, said, what = "over", UP, f" leans {side}", f"the model leans {side}"
    elif shown and lean == "Less":
        cls, mark, said, what = "under", DOWN, f" leans {side}", f"the model leans {side}"
    elif lean in ("More", "Less"):
        cls, mark, said = "plain", "", f" leans {side}, not offered"
        what = f"the model leans {side}, a side this line does not offer"
    else:
        cls, mark, said, what = "plain", "", " no lean", "no lean at this line"
    pasted = rec.get("origin") == "pasted"
    word = f" ({_paste_word(rec)})" if pasted and book == props.PASTE_BOOK else ""
    tip = f"{title} {n}{word}: {what}."
    tie = _tie_text(rec)
    if tie:
        tip += f" {tie}"
    at, last = _stamp_of(rec.get("fetched_at")), latest.get(rec.get("game_id"))
    if not pasted and at is not None and last is not None and last - at > LINE_TIME_AGE:
        tip += f" Line as of {_clock_text(at, today)}."
    arrow = f'<span class="ar" aria-hidden="true">{mark}</span>' if mark else ""
    badges = (_odds_badge(rec.get("odds_type")) + _sides_tag(rec.get("sides"), book)
              + (_void_tag(rec) if shown else "") + _pick_tag(rec))
    return (f'<td class="bk"><span class="bv-cell {cls}" title="{esc(tip)}">{esc(n)}{arrow}'
            f'<span class="bv-sr">{esc(said)}</span></span>{badges}</td>')


def _grid_rows_of(rows) -> pd.DataFrame:
    """Rows carrying the grid's cells: grid rows as they are, anything else
    (plain shown lines, a list of row dicts) through props.grid_rows, so no
    caller can get a row whose own line is drawn as the sentinel."""
    if isinstance(rows, pd.DataFrame) and props.GRID_CELLS in rows.columns:
        return rows
    return props.grid_rows(rows)


def _grid_html(rows, more: dict | None, feed: dict) -> str:
    """The slate's grid (amendment C6): Player (the name, the matchup and
    "+N more" under it), Stat, Model, then one column per book. The header's
    book columns and every row's cells come from the same loop over the same
    books, so a column's header and its cells never disagree."""
    rows = _grid_rows_of(rows)
    more = more or {}
    books = props.BOOK_ORDER
    latest, today = _grid_latest(rows, feed), feed.get("today")
    head = ('<div class="bv-grid-wrap" tabindex="0" role="region" aria-label="Lines by book">'
            '<table class="bv-grid"><thead><tr><th class="pl" scope="col">Player</th>'
            '<th class="st" scope="col">Stat</th><th class="num" scope="col">Model</th>'
            + "".join(_grid_head(b) for b in books) + "</tr></thead><tbody>")
    body = []
    for r in rows.to_dict(orient="records"):
        cells = r.get(props.GRID_CELLS) or {}
        n_more = int(more.get(int(r["person_id"]), 0)) if r.get("person_id") is not None else 0
        more_html = f'<b class="bv-pp-more">+{n_more} more</b>' if n_more else ""
        body.append(f'<tr><th class="pl" scope="row"><span class="nm">{esc(r.get("player"))}</span>'
                    f'<i>{_matchup(r)}{more_html}</i></th><td class="st">{_grid_stat(r)}</td>'
                    f"{_grid_model(r)}"
                    + "".join(_grid_cell(b, cells.get(b), latest, today) for b in books)
                    + "</tr>")
    return head + "".join(body) + "</tbody></table></div>"


def _grid_settle_note(r) -> str:
    """The note under a settled row of the per-player grid (amendment C7),
    escaped, or ''. For a player who did not play (a cell settled "void"),
    by the books with a line in the row: PrizePicks voids the pick
    (VOID_TEXT), every other book's own rule decides (DNP_TEXT; build
    amendments B26 and B30), and a row with both names each, PrizePicks
    first: "PP: Did not play, pick voided. DK, UD: Did not play". So no book
    but PrizePicks is said to void. For a value on the line, the books
    whose line it was: "25: landed on the line (DK, FD)"."""
    cells = r.get(props.GRID_CELLS) or {}
    books = [b for b in props.BOOK_ORDER if b in cells]
    if any(cells[b].get("settle") == "void" for b in books):
        others = [b for b in books if b != props.PASTE_BOOK]
        if props.PASTE_BOOK not in books:
            return DNP_TEXT
        if not others:
            return VOID_TEXT
        codes = ", ".join(esc(props.BOOK_CODE[b]) for b in others)
        return f"{esc(props.BOOK_CODE[props.PASTE_BOOK])}: {VOID_TEXT}. {codes}: {DNP_TEXT}"
    exact = [b for b in books if cells[b].get("settle") == "exact"]
    if exact:
        codes = ", ".join(esc(props.BOOK_CODE[b]) for b in exact)
        return f"{esc(_actual_text(cells[exact[0]]))}: {EXACT_TEXT} ({codes})"
    return ""


def _player_grid_html(rows, context, feed: dict) -> str:
    """One player's grid on the Game page (amendment C7): one row per stat
    with a shown line (props.grid_rows of his shown `rows` and his hidden
    `context`), Stat (the pinned column here: the expander names him), Model
    and one column per book, each cell the slate's cell for the same line
    (_grid_cell); after tip the Final or So far column and the settled
    notes (_grid_settle_note). '' when he has no shown line."""
    grid = props.grid_rows(rows, context)
    if grid.empty:
        return ""
    recs = grid.to_dict(orient="records")
    after_tip = any("actual" in r for r in recs)
    final = any(r.get("final") for r in recs)
    books = props.BOOK_ORDER
    latest, today = _grid_latest(grid, feed), feed.get("today")
    act_head = (f'<th class="num" scope="col">{"Final" if final else "So far"}</th>'
                if after_tip else "")
    head = (f'<div class="bv-grid-wrap" tabindex="0" role="region" '
            f'aria-label="{esc(recs[0].get("player"))}: lines by book">'
            f'<table class="bv-grid bv-grid-pl{" has-actual" if after_tip else ""}"><thead><tr>'
            '<th class="st" scope="col">Stat</th><th class="num" scope="col">Model</th>'
            + "".join(_grid_head(b) for b in books) + act_head + "</tr></thead><tbody>")
    span = 2 + len(books) + (1 if after_tip else 0)
    body = []
    for r in recs:
        cells = r.get(props.GRID_CELLS) or {}
        body.append(f'<tr><th class="st" scope="row">{_grid_stat(r)}</th>{_grid_model(r)}'
                    + "".join(_grid_cell(b, cells.get(b), latest, today) for b in books)
                    + (f'<td class="num act">{esc(_actual_text(r))}</td>' if after_tip else "")
                    + "</tr>")
        note = _grid_settle_note(r)
        if note:
            body.append(f'<tr class="note"><td colspan="{span}"><div class="bv-pl-note">{note}'
                        "</div></td></tr>")
    return head + "".join(body) + "</tbody></table></div>"


def top_per_player(table: pd.DataFrame, n: int = TOP_N) -> tuple[pd.DataFrame, dict]:
    """The best-ranked row for each of the first `n` players, plus
    {person_id: how many more lines that player has}. `table` is ranked."""
    if table is None or table.empty:
        return table, {}
    counts = table.groupby("person_id").size().to_dict()
    first = table.drop_duplicates("person_id", keep="first").head(n)
    more = {int(p): int(counts[p]) - 1 for p in first["person_id"]}
    return first.reset_index(drop=True), more


# -- Notes --------------------------------------------------------------------
def count_header(shown, hidden, thin, void) -> str:
    """'N of M lines shown (H hidden / T thin / V void)'."""
    n = [len(x) if x is not None else 0 for x in (shown, hidden, thin, void)]
    return (f"{n[0]} of {sum(n)} lines shown ({n[1]} hidden / {n[2]} thin / "
            f"{n[3]} void)")


def _pp_rows(table):
    """The PrizePicks rows of a board table (a row with no book is one)."""
    if table is None or not len(table) or "book" not in table.columns:
        return table
    return table[[_row_book({"book": b}) == props.PASTE_BOOK for b in table["book"]]]


def odds_footnote(table) -> str:
    """Shown only when a Demon or Goblin is on screen. Dated, and honest that
    the Less rollout has not been confirmed for the NBA. PrizePicks rows only
    (the odds feed carries no Demons or Goblins)."""
    if table is None or "odds_type" not in getattr(table, "columns", []):
        return ""
    table = _pp_rows(table)
    kinds = {str(o).lower() for o in table["odds_type"]} & {"demon", "goblin"}
    if not kinds:
        return ""
    names = " and ".join(sorted(k.capitalize() + "s" for k in kinds))
    return _note(
        f"<b>{names} on this board.</b> Those lines are deliberately shifted off the "
        "standard number: a Goblin is set easier and pays less, a Demon is set "
        "harder and pays more. A large gap against one is manufactured by the line, "
        "not evidence that the market is wrong. They were More-only until 21 August "
        "2026, when PrizePicks began allowing Less on selected MLB and WNBA stat "
        "types (NFL followed in September). No PrizePicks page names the NBA yet, so "
        "a basketball Demon or Goblin may still offer More only. Which sides a line "
        "sells is read per line from your paste, and a row whose lean is not on "
        "offer is left out.")


def hidden_picks_text(hidden, game_page: bool = False) -> str:
    """The typed picks among the hidden rows, or ''. The board hides a coin
    flip and a lean the line does not offer; a reader looking for a pick
    there is told which it was, and why, said of the model's lean, since the
    pick itself may be the side the line offers (review 3, R9). On the slate
    it follows the hidden-lines note (' The pasted pick among them: Miles
    Bridges Rebs+Asts (picked More; the model has no lean on it).'); the
    Game page has no such note, so it says the pick is not shown (review 2,
    I11). Escaped HTML."""
    if hidden is None or not len(hidden) or "pick" not in hidden:
        return ""
    mine = hidden[hidden["pick"].map(props.pick_of).notna()]
    if mine.empty:
        return ""
    named = ", ".join(
        f"{esc(r.get('player'))} {esc(props.STAT_LABEL.get(r.get('stat'), r.get('stat') or ''))}"
        f" (picked {'More' if props.pick_of(r.get('pick')) == 'more' else 'Less'}; "
        f"{_hidden_why(r)})"
        for r in mine.head(6).to_dict(orient="records"))
    more = ", and others" if len(mine) > 6 else ""
    one = len(mine) == 1
    if game_page:
        return f" Not shown here: the pasted {'pick' if one else 'picks'} {named}{more}."
    return f" The pasted {'pick' if one else 'picks'} among them: {named}{more}."


def _hidden_why(r) -> str:
    """Why a typed pick's row is hidden, said of the model's lean: on a
    More-only square a More pick is playable, and the model's Less is what
    the line does not offer."""
    lean = r.get("lean")
    if lean in ("More", "Less"):
        return f"the model leans {lean}, a side this line does not offer"
    return "the model has no lean on it"


def _names(items, others: bool = True) -> str:
    """At most 12 names, escaped, then ', and others' (when `others`)."""
    items = list(items)
    more = ", and others" if others and len(items) > 12 else ""
    return ", ".join(esc(x) for x in items[:12]) + more


def _ledger_line(led: dict, meta: dict, parts: list, feed: dict | None) -> str:
    """The ledger's first line. With the odds feed: every book's lines on
    file and where they came from (design 2.6, m8; layout C, amendment C8),
    "58 lines on file for this date (47 from the odds feed, 11 pasted)"."""
    when = paste_time_text(meta)
    stamp = f"{' ' + esc(when) if when else ''}"
    if feed is None:
        return (f'<div class="bv-pp-ledger-note">{led["posted"]} lines pasted for this '
                f'date: {", ".join(parts)}.{stamp}</div>')
    counts = feed.get("counts") or {}
    f_n, p_n = int(counts.get("feed", 0) or 0), int(counts.get("pasted", 0) or 0)
    src = (f" ({f_n} from the odds feed, {p_n} pasted)" if f_n and p_n
           else " (from the odds feed)" if f_n else " (pasted)" if p_n else "")
    return (f'<div class="bv-pp-ledger-note">{esc(_plural(led["posted"], "line"))} on '
            f'file for this date{src}: {", ".join(parts)}.{stamp}</div>')


# The void note's rule, per kind of book (design 2.6; build amendment B26):
# PrizePicks removes the pick (today's sentence), Underdog's own rule decides
# (it is not taken to follow PrizePicks), and so does a sportsbook's.
_VOID_RULES = (
    ("prizepicks", "listed out or marked out. PrizePicks removes a pick on a player who does not "
                   "play a second."),
    ("underdog", "listed out or marked out; Underdog's own rule decides what happens to a pick "
                 "on a player who does not play."),
    ("sportsbook", "listed out or marked out; a sportsbook's own rule decides what happens to a "
                   "bet on a player who does not play."),
)


def _void_kind(r) -> str:
    book = _row_book(r)
    return book if book in props.DFS_BOOKS else "sportsbook"


def _void_note(void) -> list[str]:
    """The void note, one per kind of book on screen: PrizePicks removes a
    pick on a player who does not play; Underdog's own rule decides what
    happens to a pick on him (build amendment B26), and a sportsbook's own
    rule what happens to a bet on him (design 2.6)."""
    if void is None or not len(void):
        return []
    recs = void.to_dict(orient="records")
    out = []
    for kind, rule in _VOID_RULES:
        mine = [r for r in recs if _void_kind(r) == kind]
        if not mine:
            continue
        names = sorted({str(r.get("player")) for r in mine})
        out.append(_note(f"<b>{_plural(len(mine), 'line')} void:</b> "
                         f"{esc(', '.join(names[:6]))} {rule}"))
    return out


def breakeven_text(shown, book=None) -> str:
    """The break-even sentence for the books on screen (design 7). PrizePicks'
    Power Play range alone is today's wording; Underdog's payouts are not
    modelled; a sportsbook side gets the -110 example. No price is shown
    for any book (amendment A1). With no row shown, the board's own books
    say which (`book`: a book key, or a collection of them, such as the
    books of a feed board's hidden, thin and void lines: amendment C8), so
    a sportsbook board whose lines are all withheld never gets PrizePicks'
    range (build 4's review, L3-8); keys outside BOOK_ORDER are ignored,
    and with no book left (the feed off), today's wording."""
    books = {_row_book(r) for r in shown.to_dict(orient="records")} \
        if shown is not None and len(shown) else set()
    if not books:
        given = (book,) if book is None or isinstance(book, str) else book
        try:
            books = {b for b in given if isinstance(b, str) and b in props.BOOK_ORDER}
        except TypeError:
            books = set()
    pp = (f"a leg has to hit {props.BREAKEVEN_LO:.1f}% to {props.BREAKEVEN_HI:.1f}% just to "
          "break even.")
    if books <= {props.PASTE_BOOK}:
        return f": {pp}"
    said = []
    if props.PASTE_BOOK in books:
        said.append(f"A PrizePicks {pp[2:]}")
    if "underdog" in books:
        said.append("Underdog's payouts are not modelled here, so no break-even is shown for "
                    "its lines.")
    if books - props.DFS_BOOKS:
        even = 100 * props.american_breakeven(-110)
        said.append(f"A sportsbook side at -110 needs about {even:.1f}% to break even.")
    return ". " + " ".join(said)


def _books_of(*frames) -> tuple:
    """The books of the lines in `frames`, in BOOK_ORDER (a row with no book
    is a PrizePicks line)."""
    seen = {_row_book(r) for f in frames if f is not None and len(f)
            for r in f.to_dict(orient="records")}
    return tuple(b for b in props.BOOK_ORDER if b in seen)


def notes_html(meta: dict, shown, hidden, thin, void, feed: dict | None = None) -> list[str]:
    """Every explanation the board owes the reader, as escaped HTML blocks.
    With the odds feed (`feed`, BoardView.feed) the words follow where each
    line came from (design 2.6, m8): every book's lines on file, the names
    the feed spells differently, the paste-only hints; the hidden note says
    a hidden line beside a shown one is drawn as its number alone when one
    is (HIDDEN_DRAWN, amendment C8), and with no line shown the break-even
    sentence reads the books of the board's own lines."""
    meta = meta or {}
    out: list[str] = []
    led = props.ledger(meta, shown, hidden, thin, void)
    parts = [f"{led['matched']} matched"]
    for key, word in (("unmatched", "unmatched"), ("ambiguous", "ambiguous"),
                      ("n_refused", "not priced"), ("unprojected", "no projection")):
        if led[key]:
            parts.append(f"{led[key]} {word}")
    out.append(_ledger_line(led, meta, parts, feed))
    pasted = set(feed.get("paste_unmatched") or []) if feed is not None else None
    names = list(meta.get("unmatched_names") or [])
    fed = [x for x in names if pasted is not None and x not in pasted]
    mine = [x for x in names if pasted is None or x in pasted]
    if fed:
        out.append(_note(f"<b>Not matched to a current roster:</b> {_names(fed)}. The odds "
                         "feed spells these differently from the NBA roster; they are added "
                         "to the name list as they are found."))
    if mine:
        out.append(_note(f"<b>Not matched to a current roster:</b> {_names(mine)}"
                         + ". Check the spelling, or the player may not be on an NBA "
                         "roster yet."))
    pasted = set(feed.get("paste_ambiguous") or []) if feed is not None else None
    names = list(meta.get("ambiguous_names") or [])
    fed = [x for x in names if pasted is not None and x not in pasted]
    mine = [x for x in names if pasted is None or x in pasted]
    if fed:
        out.append(_note(f"<b>More than one player by that name:</b> {_names(fed, False)}."))
    if mine:
        out.append(_note(f"<b>More than one player by that name:</b> {_names(mine, False)}. "
                         "A paste that carries the team (the board's TEAM - POS line) "
                         "resolves it."))
    refused = meta.get("refused") or {}
    if refused:
        out.append(_note("<b>Not priced:</b> " + "; ".join(
            f"{esc(k)} ({int(v)}): {esc(meta.get('refused_why', {}).get(k, ''))}"
            for k, v in sorted(refused.items())) + "."))
    unmapped = meta.get("unmapped") or {}
    if unmapped:
        out.append(_note("<b>Stats we do not recognise:</b> " + ", ".join(
            f"{esc(k)} ({int(v)})" for k, v in sorted(unmapped.items())) + "."))
    if meta.get("unprojected_why"):
        out.append(_note("<b>No projection:</b> " + "; ".join(
            f"{esc(k)} ({int(v)})" for k, v in sorted(meta["unprojected_why"].items()))
            + "."))
    nonexact = meta.get("nonexact") or []
    if nonexact:
        pairs = ", ".join(f"{esc(a)} as {esc(b)}" for a, b, _how in nonexact[:8])
        out.append(_note(f"<b>Matched by nickname, partial name or another team:</b> "
                         f"{pairs}."))
    if thin is not None and len(thin):
        names = sorted({str(x) for x in thin["player"]})
        gp = pd.to_numeric(thin["games_prior"], errors="coerce") if "games_prior" in thin \
            else pd.Series([float("nan")] * len(thin))
        unknown = int(gp.isna().sum())
        why = (f"Fewer than {props.MIN_GAMES} prior games, so the projection is close to "
               "a role base rate, and ranking would put exactly those rows on top.")
        if unknown:
            why += (f" {unknown} of them carry no prior-game count in the projection, "
                    "and an unknown history is treated as a thin one.")
        out.append(_note(
            f"<b>{_plural(len(thin), 'line')} withheld for thin history</b> "
            f"({esc(', '.join(names[:4]))}{', and others' if len(names) > 4 else ''}). "
            + why))
    if hidden is not None and len(hidden):
        even = int((hidden["lean"] == "Even").sum()) if "lean" in hidden else 0
        off = len(hidden) - even
        bits = []
        if off:
            bits.append(f"{off} {'leans' if off == 1 else 'lean'} a side the line does not offer")
        if even:
            bits.append(f"{even} {'sits' if even == 1 else 'sit'} on a coin flip with no lean")
        drawn = (f" {esc(HIDDEN_DRAWN)}" if feed is not None
                 and len(props.split_hidden(shown, hidden)[0]) else "")
        out.append(_note(f"<b>{_plural(len(hidden), 'line')} hidden:</b> "
                         + " and ".join(bits) + ". There is no pick to show on those."
                         + drawn + hidden_picks_text(hidden)))
    out += _void_note(void)
    foot = odds_footnote(shown)
    if foot:
        out.append(foot)
    reboot = reboot_note(shown)
    if reboot:
        out.append(reboot)
    if shown is not None and len(shown) and shown["stat"].isin(props.COMBO_STATS).any():
        out.append(_note(
            "Combination lines (Pts+Rebs+Asts, Blks+Stls and the rest) show their "
            "own median, which is not the sum of the component medians. Fantasy "
            "Score is the exception: PrizePicks' formula on the medians shown for "
            "its parts, the same number as the FS column on the game page."))
    out.append(_note(
        "Model values are distribution medians for a player who plays, and the "
        "gap is median minus line in the stat's own units. Rows are ranked by "
        "measured reliability, then by how far the chance sits from a coin flip, "
        "never by the size of the gap. The posted line already carries almost "
        "everything a model knows, so read a gap as where the model disagrees, not "
        f"as an edge{breakeven_text(shown, _books_of(hidden, thin, void) if feed is not None else None)} "
        "Not betting advice."
        + (f" {esc(RG_LINE)}" if RG_LINE else "")))
    return out


def reboot_note(shown) -> str:
    """The Reboot footnote, when a PrizePicks More lean is on screen (the void
    tags on bench roles point at it), else ''. Reboot is PrizePicks' rule."""
    shown = _pp_rows(shown)
    if shown is None or not len(shown) or "lean" not in shown or not (shown["lean"] == "More").any():
        return ""
    return _note(
        "<b>Reboot.</b> PrizePicks voids a More pick on a full-game line if the "
        "player leaves in the first half and does not return, unless he already "
        "cleared it. The model does not price that yet, so More leans on "
        "bench roles carry more void risk than the lean suggests.")


# -- The board ----------------------------------------------------------------
def _game_label(gid, sub: pd.DataFrame, labels: dict | None) -> str:
    if labels and gid in labels:
        return str(labels[gid])
    t = sorted({str(x) for x in sub["team"].dropna()} | {str(x) for x in sub["opp"].dropna()})
    return " / ".join(t[:2]) if t else str(gid)


def window_html(shown, hidden, thin, void, scope_label) -> str:
    """The board's title bar. The scope label is escaped (a pure function, so
    the escaping is tested rather than assumed)."""
    return ('<div class="bv-window"><span class="lab">Model vs the board</span>'
            f'<span class="meta">{esc(count_header(shown, hidden, thin, void))}</span>'
            f'<span class="lock">{esc(scope_label)}</span></div>')


def none_matched_html(posted: int, scope_label, feed: bool = False) -> str:
    """The board with lines but none matched. With the odds feed's lines
    counted (`feed`), they are lines on file, not saved ones."""
    kept = "on file" if feed else "saved"
    return _notice(f"<b>{_plural(int(posted), 'line')} {kept} for {esc(scope_label)}, but "
                   "none matched a projected player and stat.</b> The reasons are below.")


@st.fragment
def _board_fragment(shown: pd.DataFrame, key: str, game_labels: dict | None,
                    feed: dict | None = None) -> None:
    """The board's filters and rows. `shown` is the board's shown lines with
    the feed off, and with it the grid's rows (props.grid_rows, amendment
    C6), which hold the same stats and games, so the options are the same
    sets in the same fixed order either way."""
    fams_here = set(shown["stat"].map(props.STAT_FAMILY).dropna())
    options = ["All"] + [f for f in props.FAMILY_ORDER if f in fams_here]
    # Options built from the WHOLE shown set in a fixed order, so they never
    # reorder between reruns (a keyed widget's id is hashed from its options).
    games = sorted({str(g) for g in shown["game_id"].dropna()})
    labels = {g: _game_label(g, shown[shown["game_id"].astype(str) == g], game_labels)
              for g in games}
    c_fam, c_game, c_all = st.columns([3.2, 1.6, 0.9], gap="small")
    with c_fam:
        fam = st.pills("Stat family", options, selection_mode="single", default="All",
                       key=f"{key}_fam", label_visibility="collapsed")
    pick_game = None
    with c_game:
        if len(games) > 1:
            choice = st.selectbox("Game", ["All games"] + [labels[g] for g in games],
                                  key=f"{key}_game", label_visibility="collapsed")
            pick_game = next((g for g in games if labels[g] == choice), None)
    with c_all:
        show_all = st.toggle("Show all", key=f"{key}_all")
    t = shown
    if fam and fam != "All":
        t = t[t["stat"].map(props.STAT_FAMILY) == fam]
    if pick_game:
        t = t[t["game_id"].astype(str) == pick_game]
    if t.empty:
        st.markdown(_note("No lines in that filter."), unsafe_allow_html=True)
        return
    if show_all:
        rows, more = t.reset_index(drop=True), {}
    else:
        rows, more = top_per_player(t, TOP_N)
    st.markdown(ledger_html(rows, more, feed), unsafe_allow_html=True)
    if not show_all:
        n_players = t["person_id"].nunique()
        if feed is None:
            st.caption(f"Top {min(TOP_N, n_players)} of {n_players} players, one row each "
                       "(their best-ranked line). Show all lists every line.")
        else:
            st.caption(GRID_CAPTION.format(k=min(TOP_N, n_players), n=n_players))


def _kept_text(feed: dict | None) -> str:
    """Whose lines survive a comparison that could not be built (design 2.6)."""
    if feed is None:
        return "Your lines are still saved"
    if int(feed.get("pasted", 0) or 0):
        return "Your pasted lines are still saved"
    return "The odds feed's lines are still in memory"


def render_board(shown, hidden, thin, void, scope_label: str, meta: dict | None = None,
                 key: str = "bv_board", game_labels: dict | None = None,
                 feed: dict | None = None) -> int:
    """'Model vs the board' for a slate (or one game). Returns lines shown.

    Renders nothing when nothing was pasted for this scope. A board that has
    lines but shows none EXPLAINS why rather than going silent. With the odds
    feed (`feed` = BoardView.feed) the rows are layout C's grid
    (props.grid_rows of the shown and hidden lines: one row per player and
    stat, one column per book), drawn inside st.container(key=f"{key}_grid")
    (BOOKS_CSS's guard finds it), with GRID_LEGEND's line under it; the page
    renders feed_notice_html above this, so its notice shows even when this
    renders nothing (m7).
    """
    meta = meta or {}
    shown = shown if shown is not None else _empty()
    n = [len(x) if x is not None else 0 for x in (shown, hidden, thin, void)]
    matched = sum(n)
    posted = int(meta.get("posted", matched) or 0)
    if posted == 0 and matched == 0:
        return 0
    _inject_css(books=feed is not None)
    st.markdown(window_html(shown, hidden, thin, void, scope_label), unsafe_allow_html=True)
    if meta.get("error"):
        st.markdown(_notice("<b>The comparison could not be built for this date.</b> "
                            f"{_kept_text(feed)}; try reloading the page."),
                    unsafe_allow_html=True)
        return 0
    period = int(meta.get("period_withheld", 0) or 0) if meta.get("period_board_suspect") else 0
    if n[0]:
        st.success(f"{_plural(posted, 'line')} on file for "
                   f"{md_escape(scope_label)}, {n[0]} shown.")
    elif matched == 0 and not period:
        st.markdown(none_matched_html(posted, scope_label, feed is not None),
                    unsafe_allow_html=True)
    elif matched:
        st.markdown(_notice(
            f"<b>All {matched} matched lines are withheld.</b> The reasons are below."),
            unsafe_allow_html=True)
    if period:
        # props.compare refuses these rows (they never reach a pill, the
        # ledger or the record): say so, never "none matched".
        st.markdown(_notice(
            f"<b>{_plural(period, 'line')} withheld: this looks like a quarter or half "
            "board.</b> Those lines sit far below the full-game projections. Period props "
            "are not priced; paste the full-game board."), unsafe_allow_html=True)
    if n[0]:
        if feed is None:
            _board_fragment(shown, key, game_labels)
        else:
            with st.container(key=f"{key}_grid"):            # "bv_board_grid": BOOKS_CSS's guard
                _board_fragment(props.grid_rows(shown, hidden), key, game_labels, feed)
            st.markdown(grid_legend_html(), unsafe_allow_html=True)
    for block in notes_html(meta, shown, hidden, thin, void, feed):
        st.markdown(block, unsafe_allow_html=True)
    return n[0]


# -- The odds feed's notices (design 2.6, 6; amendments C9, C10, N1 to N3) ----
# The page renders feed_notice_html ABOVE render_board, so a notice shows even
# when the board draws nothing (m7). `info` is store.feed_info (plain values:
# state, asof, loading, per_game, report, today, lines) or a BoardView's
# `feed`, which adds the board's own facts (the games each book has not
# posted, the paste merge). Every time in it is a stamp the page passed,
# and `today` is the page's: nothing here reads a clock (m1). Visitors never
# see a balance, a credit, an error code or the vendor's name.
FEED_DOWN = ("<b>Lines are unavailable right now.</b> The odds feed did not answer. To "
             "compare a PrizePicks board, paste it with <b>Update lines</b> in the top bar.")
FEED_PAUSED_EMPTY = ("<b>No lines from the odds feed today.</b> To compare a PrizePicks "
                     "board, paste it with <b>Update lines</b> in the top bar.")
NAMED_GAMES = 2         # the books-missing note names at most this many games, then "and N more"


def _asof_of(info: dict, game_id=None):
    """The time the shown lines are as of: the OLDEST own fetch time among
    the books with feed lines (the facts' per_game "books", {book:
    fetched_at}, books of BOOK_ORDER only), over that game's books, or over
    every game's books for the date (layout C, amendment C10; build 4's
    review, L3-1). So no book's lines are ever stamped with a later pull's
    time. Where no book's own time is given, the feed's own stamps: one
    game's latest pull ("asof"), or the date's oldest latest pull."""
    own = []
    for gid, facts in (info.get("per_game") or {}).items():
        if game_id is not None and str(gid) != str(game_id):
            continue
        books = facts.get("books") if isinstance(facts, dict) else None
        for book, at in (books.items() if isinstance(books, dict) else ()):
            t = _stamp_of(at) if book in props.BOOK_ORDER else None
            if t is not None:
                own.append(t)
    if own:
        return min(own)
    if game_id is None:
        return _stamp_of(info.get("asof"))
    facts = (info.get("per_game") or {}).get(str(game_id))
    return _stamp_of(facts.get("asof")) if isinstance(facts, dict) else None


def _has_lines(info: dict, game_id=None) -> bool:
    if game_id is not None:
        return _asof_of(info, game_id) is not None
    return int(info.get("lines", 0) or 0) > 0 or _asof_of(info) is not None


def lines_asof_text(info, today=None, game_id=None) -> str:
    """'Lines as of 9:02 AM ET.' (', Mar 11' when not `today`), or for a game
    whose lines are its pre-tip copies 'Pre-tip lines, as of 6:52 PM ET.';
    '' when there are no lines. `game_id` scopes it to one game (the Game
    page's note), else the date's oldest latest pull."""
    info = info if isinstance(info, dict) else {}
    today = today if today is not None else info.get("today")
    at = _asof_of(info, game_id)
    if at is None:
        return ""
    facts = (info.get("per_game") or {}).get(str(game_id)) if game_id is not None else None
    if isinstance(facts, dict) and facts.get("pretip"):
        return f"Pre-tip lines, as of {_clock_text(at, today)}."
    return f"Lines as of {_clock_text(at, today)}."


def loading_line_html(n: int) -> str:
    """The loading line (design 1.3), or '' with nothing loading."""
    n = int(n or 0)
    if n <= 0:
        return ""
    return _note(f"Lines for {_plural(n, 'game')} are still loading; this board fills in by "
                 "itself.")


def _state_html(info: dict, today, game_id=None) -> str:
    """The feed's state phrase (design 6): stale, down, paused, or the
    'as of' line; '' when idle (no lines and nothing wrong, N1)."""
    state = info.get("state")
    at = _asof_of(info, game_id)
    when = _clock_text(at, today) if at is not None else ""
    lines = _has_lines(info, game_id)
    if state == "stale" and lines:
        return _notice(f"<b>Lines as of {esc(when)}.</b> The odds feed did not answer just now, "
                       "so the board shows the last lines it had.")
    if state in ("stale", "down"):
        return _notice(FEED_DOWN)
    if state == "paused":
        if lines:
            return _notice(f"<b>Lines as of {esc(when)}.</b> The board is not refreshing these "
                           "lines for now.")
        return _notice(FEED_PAUSED_EMPTY)
    if state == "ok" and lines:
        return _note(esc(lines_asof_text(info, today, game_id)))
    return ""


def _merge_html(info: dict, today) -> str:
    """The paste merge (M5, N3): which line the board shows where the
    visitor's pasted PrizePicks line and the feed's meet. The feed's time is
    the OLDEST fetch of the feed lines that won (the safe one to quote). A
    pasted line wins when it is newer, or whatever its time when it is a
    typed pick, a Demon or a Goblin, and the line names all four (B28)."""
    feed_won, paste_won = int(info.get("feed_won", 0) or 0), int(info.get("paste_won", 0) or 0)
    if not (feed_won or paste_won):
        return ""
    times = info.get("paste_times") or {}
    if feed_won:
        pasted = _clock_text(times.get("pasted_utc"), today, zone=False)
        theirs = _clock_text(times.get("feed_oldest"), today, zone=False)
        text = (f"Your pasted PrizePicks lines{' from ' + pasted if pasted else ''} are older "
                f"than the odds feed's{' (' + theirs + ')' if theirs else ''}, so the feed's "
                f"line is shown for {_player_stats(feed_won)}.")
        if paste_won:
            text += f" Your {PASTE_WINNERS} lines are shown for {paste_won}."
        return _note(esc(text))
    return _note(esc(f"Your {PASTE_WINNERS} PrizePicks lines are shown for "
                     f"{_player_stats(paste_won)}."))


def _player_stats(n: int) -> str:
    return f"{n} {'player and stat' if n == 1 else 'players and stats'}"


def _games_text(gids, labels: dict | None) -> str:
    """At most NAMED_GAMES games by name (the schedule's own 'DEN at SAS',
    away at home, N2), then 'and N more games'."""
    gids = [str(g) for g in gids]
    names = [str(labels[g]) for g in gids if labels and g in labels]
    n = len(gids)
    if names and len(names) == n and n <= NAMED_GAMES:
        return " and ".join(esc(x) for x in names)
    if names:
        rest = n - 1
        return f"{esc(names[0])} and {rest} more {'game' if rest == 1 else 'games'}"
    return f"{n} {'game' if n == 1 else 'games'}"


def _is_loaded(book: str, loaded) -> bool:
    """Whether the site buys `book`: `loaded` is the books
    BUCKETVALUE_ODDS_BOOKS names (feed_info's "books"); None, when the facts
    do not say, is all five (the default setting)."""
    if loaded is None:
        return True
    try:
        return book in {str(b) for b in ((loaded,) if isinstance(loaded, str) else loaded)}
    except TypeError:
        return True


def _book_list(books, word: str) -> str:
    """Our titles of `books` (BOOK_ORDER keys), escaped, joined in BOOK_ORDER:
    "PrizePicks", "PrizePicks or Underdog", "DraftKings, FanDuel, BetMGM or
    Underdog" (`word` "or" or "and")."""
    want = set(books)
    titles = [esc(props.BOOK_TITLE[b]) for b in props.BOOK_ORDER if b in want]
    if len(titles) < 2:
        return "".join(titles)
    return f"{', '.join(titles[:-1])} {word} {titles[-1]}"


def _or_list(books) -> str:
    return _book_list(books, "or")


def _and_list(books) -> str:
    return _book_list(books, "and")


def _not_loaded_html(books) -> str:
    return f"<b>{_and_list(books)} lines are not loaded on this site.</b>"


def books_missing_html(missing, labels: dict | None = None, *, tipped=(), loaded=None) -> str:
    """The slate's note on the books that have not posted some games (layout
    C, amendment C9), inside feed_notice_html; the grid shows the sentinel
    in those books' columns on those games' rows. `missing` is
    BoardView.feed's {book: game ids in tip order}; keys outside BOOK_ORDER
    are dropped, never echoed. A book the site does not buy (not in
    `loaded`, feed_info's books: BUCKETVALUE_ODDS_BOOKS, build amendment B25)
    goes to the not-loaded sentence, tipped or not. Every other book names
    only its games that have not tipped (B25: `tipped` holds the ids the
    page's schedule calls live or final), says nothing with none left, and
    shares one sentence with every book whose remaining games are the same,
    the groups in BOOK_ORDER of their first book, at most two games named,
    then "and N more" (_games_text). Each sentence in its own <b>, joined by
    one space, all inside one bv-note; '' when there is none."""
    missing = missing if isinstance(missing, dict) else {}
    done = {str(g) for g in (tipped or ())}
    groups: dict = {}
    gone = []
    for book in props.BOOK_ORDER:
        try:
            games = [str(g) for g in (missing.get(book) or ())]
        except TypeError:
            games = []
        if not games:
            continue
        if not _is_loaded(book, loaded):
            gone.append(book)
            continue
        left = tuple(g for g in games if g not in done)
        if left:
            groups.setdefault(left, []).append(book)
    said = [f"<b>No {_or_list(books)} lines yet for {_games_text(games, labels)}.</b>"
            for games, books in groups.items()]
    if gone:
        said.append(_not_loaded_html(gone))
    return _note(" ".join(said)) if said else ""


def books_missing_game_html(books, *, tipped: bool = False, loaded=None) -> str:
    """The Game page's note on the books with no line for its game (layout C,
    amendment C9; the page wraps it in a bv-note and draws it after its own
    notes, rows or none). `books` are those books (keys outside BOOK_ORDER
    dropped, never echoed). The books the site buys share one sentence
    first: before tip "No PrizePicks lines for this game yet.", after the
    game's tip (`tipped`, the page's state live or final) that the odds feed
    did not load their lines before tip, which is true whether a book posted
    none or the site had stopped asking for it (build amendment B29). Then
    the books BUCKETVALUE_ODDS_BOOKS leaves out (not in `loaded`), tipped or
    not: their lines are not loaded on this site (B25). Each sentence in
    its own <b>, joined by one space; '' when `books` is empty."""
    try:
        given = {str(b) for b in ((books,) if isinstance(books, str) else (books or ()))}
    except TypeError:
        given = set()
    mine = [b for b in props.BOOK_ORDER if b in given]
    bought = [b for b in mine if _is_loaded(b, loaded)]
    gone = [b for b in mine if not _is_loaded(b, loaded)]
    said = []
    if bought and tipped:
        said.append(f"<b>The odds feed did not load {_or_list(bought)} lines for this game "
                    "before tip.</b>")
    elif bought:
        said.append(f"<b>No {_or_list(bought)} lines for this game yet.</b>")
    if gone:
        said.append(_not_loaded_html(gone))
    return " ".join(said)


AFTER_TIP_NONE_POSTED = "<b>No lines were posted for this game before tip.</b>"
AFTER_TIP_NOT_LOADED = "<b>The odds feed did not load lines for this game before tip.</b>"


def after_tip_note_html(reason) -> str:
    """The Game page's note for a tipped game with no feed lines (m22, build
    amendment B18): "No lines were posted" only when its books really posted
    none (reason none_posted); lost_in_restart its own note; a game the
    budget held, a paused day or a game never pulled (any other reason)
    says the odds feed did not load its lines."""
    if reason == "lost_in_restart":
        return ("<b>The odds feed's pre-tip lines for this game were not kept.</b> The site "
                "restarted after its last update. A pasted board still shows here.")
    if reason == "none_posted":
        return AFTER_TIP_NONE_POSTED
    return AFTER_TIP_NOT_LOADED


def _pulled_html(info: dict) -> str:
    """Lines a book took down since its last update (design 6): up to the
    report's eight, named by player, stat and our own book title."""
    report = info.get("report") or {}
    pulled = [p for p in (report.get("pulled") or []) if isinstance(p, (list, tuple)) and len(p) == 3]
    n = max(int(report.get("n_pulled", 0) or 0), len(pulled))
    if not n:
        return ""
    named = []
    for book, name, stat in pulled:
        title = props.BOOK_TITLE.get(book)
        named.append(f"{esc(name)} {esc(stat)}" + (f" ({esc(title)})" if title else ""))
    more = ", and others" if n > len(named) else ""
    return _note(f"<b>{_plural(n, 'line')} taken down by their book since the last update:</b> "
                 f"{', '.join(named)}{more}.")


def feed_notice_html(info, today=None, labels: dict | None = None, *, game_id=None,
                     with_loading: bool = True, tipped=()) -> str:
    """Everything the odds feed owes the reader above the board (design 2.6,
    6), as escaped HTML; '' with the feed off or nothing to say.

      the state      stale, down or paused (design 6's phrases; N1), else
                     'Lines as of 9:02 AM ET.' while there are lines;
      loading        'Lines for 2 games are still loading ...' (with_loading:
                     pass False when the page draws store.feed_loading_watch,
                     which shows the line itself);
      the merge      which line shows where the paste and the feed meet;
      books missing  the games each book has not posted (books_missing_html,
                     amendment C9), named from `labels` ({game_id: 'DEN at
                     SAS'}, the schedule's orientation, N2), at most two, then
                     'and N more': only those not in `tipped` (the game ids
                     the page's schedule calls live or final, build amendment
                     B25), and for a book the info's "books" (feed_info's:
                     the books the site buys) leave out, that its lines are
                     not loaded here;
      taken down     lines a book pulled since its last update.

    `today` is the page's date (', Mar 11' marks an older stamp); `game_id`
    scopes the state line to one game (the Game page, whose own note says
    which books have not posted it, books_missing_game_html) and leaves out
    the date's merge, books-missing and taken-down lines."""
    info = info if isinstance(info, dict) else {}
    if not info or info.get("state") in (None, "off"):
        return ""
    today = today if today is not None else info.get("today")
    parts = [_state_html(info, today, game_id)]
    if with_loading:
        parts.append(loading_line_html(info.get("loading", 0)))
    if game_id is None:
        parts.append(_merge_html(info, today))
        parts.append(books_missing_html(info.get("missing"), labels, tipped=tipped,
                                        loaded=info.get("books")))
        parts.append(_pulled_html(info))
    return "".join(p for p in parts if p)


# -- Biggest discrepancies (DiamondValue's biggest-gaps strip, 2026-11-05) ----
# Barrett asked for DiamondValue's section at the bottom: the lines ranked by
# the gap between our number and the posted line, in each stat's own units,
# as chips with a bar for the size of the gap, then the rest on demand. It
# reads the board's SHOWN rows only (a lean on an offered side, an informed
# history, not void), so it never puts forward a line the board withholds,
# and of those the STANDARD lines only: a Goblin sits below the market's own
# number and a Demon above it on purpose, so their gap is mostly that offset
# (the alt-line collapse keeps one only when no standard line was posted).
# The board itself still ranks by measured reliability, and the section says
# why: a big gap is as often our miss as the line's.
DISCREPANCY_CHIPS = 6           # DiamondValue's strip
DISCREPANCY_MIN_GAP = 0.005     # a gap that rounds to nothing is not a discrepancy
DISCREPANCY_MORE = 50           # rows under "+N more"; the board lists every line
DISCREPANCY_CSS = """
<style>
.bv-gap-strip { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.35rem 0 0.6rem; }
.bv-gap-chip { display: flex; flex-direction: column; gap: 0.15rem; min-width: 150px;
    flex: 0 1 auto; padding: 0.5rem 0.7rem; background: var(--panel);
    border: 1px solid var(--panel-line); border-left: 3px solid var(--panel-line);
    border-radius: 10px; box-shadow: var(--shadow-card); }
.bv-gap-chip.over { border-left-color: var(--value-good); }
.bv-gap-chip.under { border-left-color: var(--value-bad); }
.bv-gap-chip .gc-player { font-weight: 600; font-size: 0.9rem; color: var(--fg-1);
    line-height: 1.15; }
.bv-gap-chip .gc-stat { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--fg-5); }
.bv-gap-chip .gc-nums { font-size: 0.82rem; color: var(--fg-2);
    font-variant-numeric: tabular-nums; }
.bv-gap-chip .gc-vs { color: var(--fg-5); font-size: 0.72rem; padding: 0 0.1rem; }
.bv-gap-chip.over .gc-lean { color: var(--value-good); font-weight: 700; }
.bv-gap-chip.under .gc-lean { color: var(--value-bad); font-weight: 700; }
.bv-gap-chip .gc-bar { position: relative; height: 4px; border-radius: 999px;
    background: var(--hairline); margin-top: 0.3rem; overflow: hidden; }
.bv-gap-chip .gc-bar > i { position: absolute; left: 0; top: 0; bottom: 0; display: block;
    border-radius: 999px; }
.bv-gap-chip.over .gc-bar > i { background: var(--value-good); }
.bv-gap-chip.under .gc-bar > i { background: var(--value-bad); }
details.bv-gap-more > summary { cursor: pointer; color: var(--fg-4); font-size: 0.8rem;
    font-weight: 600; margin: 0.1rem 0 0.4rem; }
table.bv-gap-table { width: 100%; border-collapse: collapse; font-size: 0.8rem;
    font-variant-numeric: tabular-nums; }
table.bv-gap-table td, table.bv-gap-table th { padding: 0.25rem 0.4rem;
    border-bottom: 1px solid var(--hairline); text-align: right; }
table.bv-gap-table td.l, table.bv-gap-table th.l { text-align: left; }
table.bv-gap-table th { color: var(--fg-5); font-weight: 600; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.04em; }
table.bv-gap-table .diff.over, table.bv-gap-table .lean.over { color: var(--value-good); }
table.bv-gap-table .diff.under, table.bv-gap-table .lean.under { color: var(--value-bad); }
table.bv-gap-table .lean { font-weight: 700; }
.bv-gap-chip.over .diff { color: var(--value-good); font-weight: 700; }
.bv-gap-chip.under .diff { color: var(--value-bad); font-weight: 700; }
.bv-gap-cap { color: var(--fg-5); font-size: 0.74rem; margin: 0.35rem 0 0.2rem; }
.bv-gap-scroll { overflow-x: auto; }
</style>
"""


def discrepancies(shown) -> pd.DataFrame:
    """The board's shown STANDARD lines ranked by the absolute gap between
    our number and the line (`gap`, in the stat's own units), largest first;
    ties keep the board's own order. Gaps under DISCREPANCY_MIN_GAP, and
    Goblin and Demon lines (set off the market on purpose), are left out."""
    if shown is None or shown.empty or "gap" not in shown.columns:
        return _empty()
    t = shown.reset_index(drop=True)
    g = pd.to_numeric(t["gap"], errors="coerce").abs()
    keep = g >= DISCREPANCY_MIN_GAP
    if "odds_type" in t.columns:
        keep &= t["odds_type"].map(lambda o: str(o or "standard").strip().lower()) == "standard"
    order = g[keep].sort_values(ascending=False, kind="mergesort").index
    return t.loc[order].reset_index(drop=True)


# Biggest discrepancies with the odds feed (layout C, amendment C11): the lines
# of one player and stat that sit at one number with one side word are one row
# naming every book that posts it.
GAP_BOOKS = "books"           # merge_books' column: the books posting the row's line


def _book_rank(book) -> int:
    return props.BOOK_ORDER.index(book) if book in props.BOOK_ORDER else len(props.BOOK_ORDER)


def merge_books(rows) -> pd.DataFrame:
    """Biggest discrepancies with the odds feed (amendment C11): the rows of
    discrepancies() that share a (game_date, person_id, stat), a line and
    a side word (_lean_text, so Over and More stay apart) become the FIRST
    of them, carrying GAP_BOOKS: every such row's book, in BOOK_ORDER. A
    key's model is the same at every book and the lean is read at the
    line, so the merged rows share their gap and lean (verified in C1V: no
    shown key and line has two leans): nothing is chosen between books,
    and there is no best line. Order kept: largest gap first. A row whose
    _key_of is None, or whose line is not a number, is never merged with
    another, and different lines of one key stay separate rows."""
    rows = rows if isinstance(rows, pd.DataFrame) else _empty()
    out, books, index = [], [], {}
    for rec in rows.to_dict(orient="records"):
        key = _key_of(rec)
        k = ((key, round(float(rec["line"]), 4), _lean_text(rec))
             if key is not None and _finite(rec.get("line")) else None)
        if k is not None and k in index:
            books[index[k]].append(_row_book(rec))
            continue
        if k is not None:
            index[k] = len(out)
        out.append(rec)
        books.append([_row_book(rec)])
    merged = pd.DataFrame(out, columns=list(rows.columns)) if out \
        else pd.DataFrame(columns=list(rows.columns))
    return merged.assign(**{GAP_BOOKS: [tuple(sorted(set(b), key=_book_rank)) for b in books]})


def _gap_codes(r) -> str:
    """The books posting a discrepancy row's line, each its code in an
    <abbr> of its title, in BOOK_ORDER, joined by one space."""
    have = set(r.get(GAP_BOOKS) or ())
    return " ".join(f'<abbr title="{esc(props.BOOK_TITLE[b])}">{esc(props.BOOK_CODE[b])}</abbr>'
                    for b in props.BOOK_ORDER if b in have)


def _gap_chip(r, top: float, books: bool = False) -> str:
    cls = _lean_cls(r.get("lean"))
    width = max(4, int(round(abs(float(r["gap"])) / top * 100))) if top > 0 else 4
    book_html = f' <span class="gc-book">{_gap_codes(r)}</span>' if books else ""
    return (f'<div class="bv-gap-chip {cls}">'
            f'<span class="gc-player">{esc(r.get("player"))}</span>'
            f'<span class="gc-stat">{esc(r.get("stat_label"))}</span>'
            f'<span class="gc-nums">{_fmt_model(r)} <span class="gc-vs">vs</span> {_fmt_line(r)}'
            f'{book_html} &middot; {_fmt_gap(r)} <span class="gc-lean">{_lean_text(r)}</span></span>'
            f'<span class="gc-bar"><i style="width:{width}%"></i></span></div>')


def discrepancy_note(picker: bool) -> str:
    """The section's note (HTML-escaped). `picker`: the family filter shows,
    so the note can point at it."""
    pick = "; pick a stat family to compare like with like" if picker else ""
    return _note(esc(
        "Our number minus the posted line, in each stat's own units, so combined stats like "
        f"Pts+Rebs+Asts and Fantasy Score tend to lead{pick}. Standard lines only: a Goblin "
        "or Demon line is set off the market on purpose, so its gap is mostly the offset. A "
        "big gap is as often our miss as the line's, which is why the board ranks lines by "
        "measured reliability instead. Informational, not a wager recommendation."))


def discrepancies_window(rows, scope_label, book_label=None) -> str:
    """The section's title bar, styled like the board's; the count is
    len(rows), so with the odds feed (merge_books' rows, amendment C11) it
    counts rows, each one line at one or more books. `book_label` names a
    book in the count ("6 standard PrizePicks lines", amendment A8), an
    option the pages no longer pass; None keeps the title it had before the
    feed."""
    what = "standard line" if book_label is None else f"standard {book_label} line"
    return ('<div class="bv-window"><span class="lab">Biggest discrepancies</span>'
            f'<span class="meta">{esc(_plural(len(rows), what))}, largest gap '
            f'first</span><span class="lock">{esc(scope_label)}</span></div>')


def discrepancy_strip_html(rows, books: bool = False) -> str:
    """Chips for the first DISCREPANCY_CHIPS of `rows` (already ranked by
    discrepancies(), and possibly filtered), each with a bar sized to its gap
    against the first chip's, then the next DISCREPANCY_MORE in a table under
    '+N more'. Empty string for no rows. With `books` (the odds feed on,
    amendment C11) each chip names the books posting its line after the line
    and the table gains a Book column; rows without GAP_BOOKS go through
    merge_books first, so no chip lacks its books."""
    if rows is None or len(rows) == 0:
        return ""
    if books and GAP_BOOKS not in rows.columns:
        rows = merge_books(rows)
    chips = rows.head(DISCREPANCY_CHIPS)
    top = float(pd.to_numeric(chips["gap"], errors="coerce").abs().max())
    strip = "".join(_gap_chip(r, top, books) for r in chips.to_dict(orient="records"))
    rest = rows.iloc[len(chips):]
    more = ""
    if len(rest):
        listed = rest.head(DISCREPANCY_MORE)
        body = "".join(
            f'<tr><td class="l">{esc(r.get("player"))}</td><td class="l">{esc(r.get("stat_label"))}</td>'
            f"<td>{_fmt_model(r)}</td><td>{_fmt_line(r)}</td>"
            + (f'<td class="l">{_gap_codes(r)}</td>' if books else "")
            + f"<td>{_fmt_gap(r)}</td>"
            f'<td class="l lean {_lean_cls(r.get("lean"))}">{_lean_text(r)}</td></tr>'
            for r in listed.to_dict(orient="records"))
        cap = (f'<div class="bv-gap-cap">The next {len(listed)} of {len(rest)}, largest gap '
               "first. The board lists every line.</div>" if len(rest) > len(listed) else "")
        more = (f'<details class="bv-gap-more"><summary>+{len(rest)} more</summary>'
                '<div class="bv-gap-scroll"><table class="bv-gap-table"><thead><tr>'
                '<th class="l">Player</th><th class="l">Stat</th><th>Ours</th><th>Line</th>'
                + ('<th class="l">Book</th>' if books else "")
                + '<th>Gap</th><th class="l">Lean</th>'
                f"</tr></thead><tbody>{body}</tbody></table></div>{cap}</details>")
    return f'<div class="bv-gap-strip">{strip}</div>{more}'


def _ranked_gaps(shown, books: bool) -> pd.DataFrame:
    """discrepancies(shown), and with `books` its rows merged per line
    (merge_books)."""
    rows = discrepancies(shown)
    return merge_books(rows) if books and not rows.empty else rows


def discrepancies_html(shown, scope_label, book_label=None, books: bool = False) -> str:
    """The whole section for every family (the unfiltered view): title bar,
    strip, '+N more' and the note. Empty string when the board shows no
    standard line with a gap. `book_label` as discrepancies_window; `books`
    (the odds feed on) merges the lines that share a player, stat, line and
    side word into one row naming each book (merge_books), and the title
    counts those rows."""
    rows = _ranked_gaps(shown, books)
    if rows.empty:
        return ""
    return (discrepancies_window(rows, scope_label, book_label)
            + discrepancy_strip_html(rows, books)
            + discrepancy_note(len(discrepancy_families(rows)) > 2))


def discrepancy_families(rows) -> list[str]:
    """'All' and the stat families present, in the board's fixed order (a
    keyed widget's id is hashed from its options, so they never reorder)."""
    fams = set(rows["stat"].map(props.STAT_FAMILY).dropna()) if len(rows) else set()
    return ["All"] + [f for f in props.FAMILY_ORDER if f in fams]


def in_family(rows, fam) -> pd.DataFrame:
    """`rows` narrowed to one stat family, order kept; 'All' or None is all."""
    if not fam or fam == "All":
        return rows
    return rows[rows["stat"].map(props.STAT_FAMILY) == fam].reset_index(drop=True)


@st.fragment
def _discrepancy_fragment(rows: pd.DataFrame, key: str, books: bool = False) -> None:
    options = discrepancy_families(rows)
    fam = "All"
    if len(options) > 2:            # one family: nothing to pick
        fam = st.pills("Stat family", options, selection_mode="single", default="All",
                       key=f"{key}_fam", label_visibility="collapsed")
    st.markdown(discrepancy_strip_html(in_family(rows, fam), books), unsafe_allow_html=True)


def render_discrepancies(shown, scope_label, key: str = "bv_gaps", book_label=None,
                         books: bool = False) -> int:
    """'Biggest discrepancies' (DiamondValue's strip) under a board, with a
    stat-family filter when the lines span more than one family. Renders
    nothing when there is nothing to rank. Returns how many rows it ranks.
    With the odds feed (`books`, amendment C11) `shown` is every book's
    shown lines, merged per line and side word (merge_books) into rows that
    name each book, which the title counts, and BOOKS_CSS is injected (its
    .gc-book rule); `book_label` stays an option (amendment A8), which the
    pages no longer pass. The family filter's options are built from the
    rows, which hold the same stats merged or not."""
    rows = _ranked_gaps(shown, books)
    if rows.empty:
        return 0
    st.markdown(DISCREPANCY_CSS, unsafe_allow_html=True)
    if books:
        st.markdown(BOOKS_CSS, unsafe_allow_html=True)
    st.markdown(discrepancies_window(rows, scope_label, book_label), unsafe_allow_html=True)
    _discrepancy_fragment(rows, key, books)
    st.markdown(discrepancy_note(len(discrepancy_families(rows)) > 2), unsafe_allow_html=True)
    return len(rows)


def render_player_lines_block(rows, context=None, feed: dict | None = None) -> None:
    """Convenience: render one player's lines with the module CSS. With the
    odds feed (`feed`, BoardView.feed; `context` his hidden lines in this
    game) it is his grid, one row per stat and one column per book, and
    BOOKS_CSS is injected too (render_player_lines, amendment C7)."""
    html = render_player_lines(rows, context, feed)
    if html:
        _inject_css(books=feed is not None)
        st.markdown(html, unsafe_allow_html=True)
