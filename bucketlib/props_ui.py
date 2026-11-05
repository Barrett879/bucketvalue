"""PrizePicks on the page: the nav-bar paste popover, the slate board, the
per-player rows on the game page, and the card pill counts.

Deliberately NOT a page (sibling precedent: DiamondValue started with a props
tab and folded it inline). Ported from GridironValue's props_ui and
DiamondValue's paste flow, with BucketValue's two structural differences:

1. THE BOARD LIVES IN THE PROCESS, KEYED BY ?board=. Every internal link here
   is a full page load, a fresh Streamlit session, so st.session_state cannot
   carry a paste from the slate to a game page. props.new_board / add_to_board
   keep it in an in-process LRU under an unguessable token, and every minted
   URL carries the token (keep_params). Never on disk, never frozen, never seen
   by another visitor.
2. ONE DEFINITION. board_view() runs compare() and props.board_rows() once;
   the card pills, the board ledger and the per-player rows all read its four
   frames, so a pill can never promise more lines than the game page shows.

Informational model-versus-market readout, never a wager prompt: the gap is a
difference in the stat's own units, never a percentage column or a dollar
value. Every string from a paste, a feed or a query param goes through esc().
The accuracy record is LOCAL ONLY and is not imported or called from here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from . import props, simprov
from .cache import logger
from .util import SENTINEL, esc

PASTE_KEY = "bv_pp_paste"
REPORT_KEY = "bv_pp_report"
TOAST_KEY = "bv_pp_toast"
EXPIRED_KEY = "bv_pp_expired"
TOP_N = 25

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
.bv-pp-ledger-note { color: var(--fg-5); font-size: 0.74rem; margin: 0.1rem 0 0.4rem; }
</style>
"""


def _inject_css() -> None:
    st.markdown(PROPS_CSS, unsafe_allow_html=True)


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


# ── The token ────────────────────────────────────────────────────────────────
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


# ── One definition of what the board shows ───────────────────────────────────
@dataclass
class BoardView:
    shown: pd.DataFrame
    hidden: pd.DataFrame
    thin: pd.DataFrame
    void: pd.DataFrame
    meta: dict = field(default_factory=dict)
    lines: pd.DataFrame = field(default_factory=pd.DataFrame)

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


def date_board(token, date, projections: pd.DataFrame | None, params, roster=None) -> BoardView:
    """board_view for one ET date with the simulator provider. The provider is
    simprov.provider_for(projections, params), the one the Game tables' medians
    read, so a line's model value and the table's median come from one
    simulation; the void set is void_ids(projections). Never raises.

    `params` None (the simulator parameters did not load) is the failed view
    at once: a provider built on it prices nothing, and every line would read
    as "no projection: internal error"."""
    if params is None:
        logger.warning("date_board: no simulator parameters")
        return _failed_view(token, date)
    try:
        provider = simprov.provider_for(projections, params)
    except Exception as e:  # noqa: BLE001 - a broken frame is an explained empty board
        logger.warning("date_board: no provider (%s: %s)", type(e).__name__, e)
        return _failed_view(token, date)
    return board_view(token, date, provider, projections, roster, void_ids(projections))


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


# ── The paste flow ───────────────────────────────────────────────────────────
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
    except Exception as e:  # noqa: BLE001 - never a traceback on the page
        logger.warning("paste ingest failed (%s: %s)", type(e).__name__, e)
        report["error"] = True
    return token, report


def _on_add(view_date, schedule, roster, now, dates) -> None:
    text = st.session_state.get(PASTE_KEY) or ""
    token = board_token()
    token, report = ingest(text, token, roster, schedule, now=now, dates=dates)
    report["view_date"] = str(pd.Timestamp(view_date).date())
    st.session_state[REPORT_KEY] = report
    if token and props.get_board(token) is not None:
        st.query_params["board"] = token
    n = toast_count(report)
    if n:
        st.session_state[TOAST_KEY] = n
    # Apply each paste ONCE. Left in the box, it would be re-read on every
    # rerun, and a Clear all would be undone by the next script body.
    st.session_state[PASTE_KEY] = ""


def _on_clear() -> None:
    """Visitor-only: drops the board token from THIS visitor's link. The
    board itself is not deleted: anyone else holding the link (it rides in
    every URL, Share included) keeps it, and it ages out of the in-process
    store. Nothing on disk is touched."""
    st.session_state[PASTE_KEY] = ""
    st.session_state.pop(REPORT_KEY, None)
    st.session_state.pop(TOAST_KEY, None)
    if "board" in st.query_params:
        del st.query_params["board"]


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
    return "Nothing readable in that paste. Paste the feed JSON or the board text."


def toast_count(report: dict | None) -> int:
    """Lines the toast may announce: the ones that can reach a board at all
    (a mapped stat, a matched player, an upcoming game). Never `kept`, which
    counts refused, unrecognised and unmatched lines too (audit 2026-10-09)."""
    if not report or report.get("error"):
        return 0
    held = int(report.get("period_held", 0) or 0)
    return max(0, int(report.get("comparable", 0) or 0) - held)


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
    by_date = {d: int(n) - int(held_by.get(d, 0) or 0)
               for d, n in (report.get("comparable_by_date") or {}).items()}
    by_date = {d: n for d, n in by_date.items() if n > 0}
    here = int(by_date.get(view, 0)) if view else 0
    held = int(report.get("period_held", 0) or 0)
    wrong = report.get("wrong_board")
    if held:
        out.append(("warning", f"{_plural(held, 'line')} sat far below the full-game line "
                    "already on your board, so they were held back: that looks like a "
                    "quarter or half board. Paste the full-game board."))
    if comparable <= 0 and wrong:
        out.append(("warning", f"That is the {md_escape(wrong)} board, not the full-game NBA "
                    "board, so nothing in it can be compared. Open the NBA board (or the "
                    "feed link above) and copy that."))
    if comparable > 0:
        if here:
            out.append(("success", f"{_plural(comparable, 'line')} loaded. {here} are for "
                        f"{_fmt_date(view)}: see Model vs the board on the slate and on "
                        "each game page."))
        else:
            out.append(("success", f"{_plural(comparable, 'line')} loaded."))
        if why:
            out.append(("info", f"Also in that paste and not compared: {why}."))
    elif report.get("nonempty") and not wrong and not held:
        head = f"None of the {_plural(kept, 'line')} in that paste can be compared"
        out.append(("warning", f"{head}: {why}." if why else f"{head}."))
    for d, n in sorted(by_date.items()):
        if d != view:
            out.append(("info", f"{_plural(int(n), 'line')} saved for {_fmt_date(d)}, "
                        "step the date to see them."))
    if unrouted:
        names = ", ".join(md_escape(x) for x in (report.get("unrouted_names") or []))
        out.append(("info", f"{_plural(unrouted, 'line')} could not be placed on an "
                    f"upcoming game ({names}). A name with no team, or a player whose "
                    "game has already tipped, cannot be filed."))
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
    if p.get("truncated"):
        out.append(("info", "The paste was cut short at its size limit, so some of it "
                    "was not read. Paste one board or one feed page at a time."))
    return out


def render_update_lines_popover(view_date, schedule, roster, now=None, dates=None,
                                in_nav_container: bool = True) -> str | None:
    """The nav bar's 'Update lines' popover. Call right after render_nav.
    Returns the visitor's board token (or None) for the page's minted URLs.

    in_nav_container=False when the page has already opened the keyed
    `bv_nav_actions` container itself (to put Share beside it): opening a keyed
    container twice is a duplicate-key error."""
    raw = st.query_params.get("board")
    if raw and board_token() is None:
        # A token that no longer resolves: expired, or the server restarted.
        st.session_state[EXPIRED_KEY] = True
        del st.query_params["board"]
    if in_nav_container:
        with st.container(key="bv_nav_actions"):
            with st.popover("Update lines"):
                _paste_panel(view_date, schedule, roster, now, dates)
    else:
        with st.popover("Update lines"):
            _paste_panel(view_date, schedule, roster, now, dates)
    token = board_token()
    placed = st.session_state.pop(TOAST_KEY, None)
    if placed:
        st.toast(f"{_plural(int(placed), 'line')} loaded")
    return token


def _paste_panel(view_date, schedule, roster, now, dates) -> None:
    if st.session_state.pop(EXPIRED_KEY, False):
        st.info("Your earlier lines have expired (a pasted board is kept for about "
                "18 hours, and a site restart clears it). Paste again to bring "
                "them back.")
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
            "itself under tomorrow.",
            unsafe_allow_html=True)
    st.text_area("Paste the feed JSON or the board text", height=150, key=PASTE_KEY,
                 placeholder=("Paste the copied feed page here, or the board text:\n"
                              "Nikola Jokic\nDEN - C\n27.5\nPoints\nLess\nMore"))
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
        "alt-line ladders collapse to one line per player and stat. Your board "
        "lives only in this visit's link: it is never saved to disk and never "
        "enters any accuracy record, but anyone you send the link to sees it. "
        "Clear all takes it off your link only. Nothing here fetches from "
        "PrizePicks: their API blocks server requests and their terms prohibit "
        "automated access.")


# ── Cells ────────────────────────────────────────────────────────────────────
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


def _sides_tag(sides) -> str:
    s = str(sides or "both").lower()
    if s == "more":
        return '<span class="bv-pp-tag" title="This line offers More only">More only</span>'
    if s == "less":
        return '<span class="bv-pp-tag" title="This line offers Less only">Less only</span>'
    return ""


def _void_tag(r) -> str:
    """Reboot: PrizePicks voids a More pick if the player leaves in the first
    half and does not return. Nothing measures how often by role yet (an exit
    model needs play-by-play, v1.1), so a More lean with props' `void_note`
    (a bench role) gets the words and never a number. A provider that does
    carry a measured `void_risk` (the hook for that model) gets its number."""
    if r.get("lean") != "More":
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


def _stat_cell(r) -> str:
    stat = r.get("stat")
    label = props.STAT_LABEL.get(stat) or str(r.get("stat_label") or stat or "")
    tip = ""
    if r.get("tie_possible") and _finite(r.get("p_push")) and _finite(r.get("line")):
        text = ("Chance of landing exactly on %g: %d in 100. That lowers the payout "
                "tier; it is not a push." % (float(r["line"]), round(100 * float(r["p_push"]))))
        tip = f' title="{esc(text)}"'
    return (f'<span class="st"{tip}>{esc(label)}'
            f'{_tier_badge(r.get("tier"), r.get("tier_edge"))}'
            f'{_odds_badge(r.get("odds_type"))}{_sides_tag(r.get("sides"))}'
            f'{_SRC_BADGE.get(str(r.get("source") or ""), "")}{_void_tag(r)}</span>')


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


def ledger_html(rows: pd.DataFrame, more: dict | None = None) -> str:
    """The board's ledger as HTML. Every value escaped."""
    more = more or {}
    out = []
    for r in rows.to_dict(orient="records"):
        n_more = int(more.get(int(r["person_id"]), 0)) if r.get("person_id") is not None else 0
        more_html = f'<b class="bv-pp-more">+{n_more} more</b>' if n_more else ""
        out.append(
            '<div class="bv-ledger-row">'
            f'<span class="nm">{esc(r.get("player"))}<i>{_matchup(r)}{more_html}</i></span>'
            f"{_stat_cell(r)}{_fmt_model(r)}{_fmt_line(r)}{_fmt_gap(r)}"
            f'<span class="lean {_lean_cls(r.get("lean"))}">{esc(r.get("lean"))}</span>'
            "</div>")
    head = ('<div class="bv-ledger-head"><span class="nm">Player</span>'
            '<span class="st">Stat</span><span class="num">Model</span>'
            '<span class="num">Line</span><span class="num">Gap</span>'
            '<span class="lean">Lean</span></div>')
    return f'<div class="bv-ledger">{head}{"".join(out)}</div>'


# What happened to a line, after tip (store.line_actuals fills `actual` and
# `settle`). PrizePicks voids a pick on a player who does not play; a value on
# the line is neither a win nor a loss, and is never called a push.
VOID_TEXT = "Did not play, pick voided"
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
    """The full-width note under a settled line, or ''."""
    s = r.get("settle")
    if s == "void":
        return f'<div class="bv-pl-note">{VOID_TEXT}</div>'
    if s == "exact":
        return f'<div class="bv-pl-note">{esc(_actual_text(r))}: {EXACT_TEXT}</div>'
    return ""


def render_player_lines(rows) -> str:
    """One player's SHOWN lines, for his expandable row on the game page.
    `rows` is a list of row dicts (rows_by_person) or a DataFrame. After tip
    (rows carrying `actual`, from store.line_actuals) an Actual column, and
    a note under a settled line: VOID_TEXT for a player who did not play,
    EXACT_TEXT for a value on the line."""
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
            f'<span class="lean {_lean_cls(r.get("lean"))}">{esc(r.get("lean"))}</span>'
            + (_fmt_actual(r) if after_tip else "") + "</div>" + _settle_note(r))
    final = any(r.get("final") for r in recs)
    act_head = (f'<span class="num">{"Final" if final else "So far"}</span>' if after_tip else "")
    head = ('<div class="bv-pl-head"><span class="st">Stat</span>'
            '<span class="num">Model</span><span class="num">Line</span>'
            f'<span class="num">Gap</span><span class="lean">Lean</span>{act_head}</div>')
    cls = "bv-pl has-actual" if after_tip else "bv-pl"
    return f'<div class="{cls}">{head}{"".join(body)}</div>'


def top_per_player(table: pd.DataFrame, n: int = TOP_N) -> tuple[pd.DataFrame, dict]:
    """The best-ranked row for each of the first `n` players, plus
    {person_id: how many more lines that player has}. `table` is ranked."""
    if table is None or table.empty:
        return table, {}
    counts = table.groupby("person_id").size().to_dict()
    first = table.drop_duplicates("person_id", keep="first").head(n)
    more = {int(p): int(counts[p]) - 1 for p in first["person_id"]}
    return first.reset_index(drop=True), more


# ── Notes ────────────────────────────────────────────────────────────────────
def count_header(shown, hidden, thin, void) -> str:
    """'N of M lines shown (H hidden / T thin / V void)'."""
    n = [len(x) if x is not None else 0 for x in (shown, hidden, thin, void)]
    return (f"{n[0]} of {sum(n)} lines shown ({n[1]} hidden / {n[2]} thin / "
            f"{n[3]} void)")


def odds_footnote(table) -> str:
    """Shown only when a Demon or Goblin is on screen. Dated, and honest that
    the Less rollout has not been confirmed for the NBA."""
    if table is None or "odds_type" not in getattr(table, "columns", []):
        return ""
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


def notes_html(meta: dict, shown, hidden, thin, void) -> list[str]:
    """Every explanation the board owes the reader, as escaped HTML blocks."""
    meta = meta or {}
    out: list[str] = []
    led = props.ledger(meta, shown, hidden, thin, void)
    parts = [f"{led['matched']} matched"]
    for key, word in (("unmatched", "unmatched"), ("ambiguous", "ambiguous"),
                      ("n_refused", "not priced"), ("unprojected", "no projection")):
        if led[key]:
            parts.append(f"{led[key]} {word}")
    out.append(f'<div class="bv-pp-ledger-note">{led["posted"]} lines pasted for this '
               f'date: {", ".join(parts)}.</div>')
    if meta.get("unmatched_names"):
        names = ", ".join(esc(x) for x in meta["unmatched_names"][:12])
        out.append(_note(f"<b>Not matched to a current roster:</b> {names}"
                         + (", and others" if len(meta["unmatched_names"]) > 12 else "")
                         + ". Check the spelling, or the player may not be on an NBA "
                         "roster yet."))
    if meta.get("ambiguous_names"):
        names = ", ".join(esc(x) for x in meta["ambiguous_names"][:12])
        out.append(_note(f"<b>More than one player by that name:</b> {names}. A paste "
                         "that carries the team (the board's TEAM - POS line) resolves "
                         "it."))
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
            bits.append(f"{off} lean a side the line does not offer")
        if even:
            bits.append(f"{even} sit on a coin flip with no lean")
        out.append(_note(f"<b>{_plural(len(hidden), 'line')} hidden:</b> "
                         + " and ".join(bits) + ". There is no pick to show on those."))
    if void is not None and len(void):
        names = sorted({str(x) for x in void["player"]})
        out.append(_note(
            f"<b>{_plural(len(void), 'line')} void:</b> {esc(', '.join(names[:6]))} "
            "listed out or marked out. PrizePicks removes a pick on a player who "
            "does not play a second."))
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
        f"as an edge: a leg has to hit {props.BREAKEVEN_LO:.1f}% to "
        f"{props.BREAKEVEN_HI:.1f}% just to break even. Not betting advice."))
    return out


def reboot_note(shown) -> str:
    """The Reboot footnote, when a More lean is on screen (the void tags on
    bench roles point at it), else ''."""
    if shown is None or not len(shown) or "lean" not in shown or not (shown["lean"] == "More").any():
        return ""
    return _note(
        "<b>Reboot.</b> PrizePicks voids a More pick on a full-game line if the "
        "player leaves in the first half and does not return, unless he already "
        "cleared it. The model does not price that yet, so More leans on "
        "bench roles carry more void risk than the lean suggests.")


# ── The board ────────────────────────────────────────────────────────────────
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


def none_matched_html(posted: int, scope_label) -> str:
    return _notice(f"<b>{_plural(int(posted), 'line')} saved for {esc(scope_label)}, but "
                   "none matched a projected player and stat.</b> The reasons are below.")


@st.fragment
def _board_fragment(shown: pd.DataFrame, key: str, game_labels: dict | None) -> None:
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
    st.markdown(ledger_html(rows, more), unsafe_allow_html=True)
    if not show_all:
        n_players = t["person_id"].nunique()
        st.caption(f"Top {min(TOP_N, n_players)} of {n_players} players, one row each "
                   "(their best-ranked line). Show all lists every line.")


def render_board(shown, hidden, thin, void, scope_label: str, meta: dict | None = None,
                 key: str = "bv_board", game_labels: dict | None = None) -> int:
    """'Model vs the board' for a slate (or one game). Returns lines shown.

    Renders nothing when nothing was pasted for this scope. A board that has
    lines but shows none EXPLAINS why rather than going silent.
    """
    meta = meta or {}
    shown = shown if shown is not None else _empty()
    n = [len(x) if x is not None else 0 for x in (shown, hidden, thin, void)]
    matched = sum(n)
    posted = int(meta.get("posted", matched) or 0)
    if posted == 0 and matched == 0:
        return 0
    _inject_css()
    st.markdown(window_html(shown, hidden, thin, void, scope_label), unsafe_allow_html=True)
    if meta.get("error"):
        st.markdown(_notice("<b>The comparison could not be built for this date.</b> "
                            "Your lines are still saved; try reloading the page."),
                    unsafe_allow_html=True)
        return 0
    period = int(meta.get("period_withheld", 0) or 0) if meta.get("period_board_suspect") else 0
    if n[0]:
        st.success(f"{_plural(posted, 'PrizePicks line')} on file for "
                   f"{md_escape(scope_label)}, {n[0]} shown.")
    elif matched == 0 and not period:
        st.markdown(none_matched_html(posted, scope_label), unsafe_allow_html=True)
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
        _board_fragment(shown, key, game_labels)
    for block in notes_html(meta, shown, hidden, thin, void):
        st.markdown(block, unsafe_allow_html=True)
    return n[0]


# ── Biggest discrepancies (DiamondValue's biggest-gaps strip, 2026-11-05) ────
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


def _gap_chip(r, top: float) -> str:
    cls = _lean_cls(r.get("lean"))
    width = max(4, int(round(abs(float(r["gap"])) / top * 100))) if top > 0 else 4
    return (f'<div class="bv-gap-chip {cls}">'
            f'<span class="gc-player">{esc(r.get("player"))}</span>'
            f'<span class="gc-stat">{esc(r.get("stat_label"))}</span>'
            f'<span class="gc-nums">{_fmt_model(r)} <span class="gc-vs">vs</span> {_fmt_line(r)}'
            f' &middot; {_fmt_gap(r)} <span class="gc-lean">{esc(r.get("lean"))}</span></span>'
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


def discrepancies_window(rows, scope_label) -> str:
    """The section's title bar, styled like the board's."""
    return ('<div class="bv-window"><span class="lab">Biggest discrepancies</span>'
            f'<span class="meta">{esc(_plural(len(rows), "standard line"))}, largest gap '
            f'first</span><span class="lock">{esc(scope_label)}</span></div>')


def discrepancy_strip_html(rows) -> str:
    """Chips for the first DISCREPANCY_CHIPS of `rows` (already ranked by
    discrepancies(), and possibly filtered), each with a bar sized to its gap
    against the first chip's, then the next DISCREPANCY_MORE in a table under
    '+N more'. Empty string for no rows."""
    if rows is None or len(rows) == 0:
        return ""
    chips = rows.head(DISCREPANCY_CHIPS)
    top = float(pd.to_numeric(chips["gap"], errors="coerce").abs().max())
    strip = "".join(_gap_chip(r, top) for r in chips.to_dict(orient="records"))
    rest = rows.iloc[len(chips):]
    more = ""
    if len(rest):
        listed = rest.head(DISCREPANCY_MORE)
        body = "".join(
            f'<tr><td class="l">{esc(r.get("player"))}</td><td class="l">{esc(r.get("stat_label"))}</td>'
            f"<td>{_fmt_model(r)}</td><td>{_fmt_line(r)}</td><td>{_fmt_gap(r)}</td>"
            f'<td class="l lean {_lean_cls(r.get("lean"))}">{esc(r.get("lean"))}</td></tr>'
            for r in listed.to_dict(orient="records"))
        cap = (f'<div class="bv-gap-cap">The next {len(listed)} of {len(rest)}, largest gap '
               "first. The board lists every line.</div>" if len(rest) > len(listed) else "")
        more = (f'<details class="bv-gap-more"><summary>+{len(rest)} more</summary>'
                '<div class="bv-gap-scroll"><table class="bv-gap-table"><thead><tr>'
                '<th class="l">Player</th><th class="l">Stat</th><th>Ours</th><th>Line</th>'
                '<th>Gap</th><th class="l">Lean</th>'
                f"</tr></thead><tbody>{body}</tbody></table></div>{cap}</details>")
    return f'<div class="bv-gap-strip">{strip}</div>{more}'


def discrepancies_html(shown, scope_label) -> str:
    """The whole section for every family (the unfiltered view): title bar,
    strip, '+N more' and the note. Empty string when the board shows no
    standard line with a gap."""
    rows = discrepancies(shown)
    if rows.empty:
        return ""
    return (discrepancies_window(rows, scope_label) + discrepancy_strip_html(rows)
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
def _discrepancy_fragment(rows: pd.DataFrame, key: str) -> None:
    options = discrepancy_families(rows)
    fam = "All"
    if len(options) > 2:            # one family: nothing to pick
        fam = st.pills("Stat family", options, selection_mode="single", default="All",
                       key=f"{key}_fam", label_visibility="collapsed")
    st.markdown(discrepancy_strip_html(in_family(rows, fam)), unsafe_allow_html=True)


def render_discrepancies(shown, scope_label, key: str = "bv_gaps") -> int:
    """'Biggest discrepancies' (DiamondValue's strip) under a board, with a
    stat-family filter when the lines span more than one family. Renders
    nothing when there is nothing to rank. Returns how many lines it ranks."""
    rows = discrepancies(shown)
    if rows.empty:
        return 0
    st.markdown(DISCREPANCY_CSS, unsafe_allow_html=True)
    st.markdown(discrepancies_window(rows, scope_label), unsafe_allow_html=True)
    _discrepancy_fragment(rows, key)
    st.markdown(discrepancy_note(len(discrepancy_families(rows)) > 2), unsafe_allow_html=True)
    return len(rows)


def render_player_lines_block(rows) -> None:
    """Convenience: render one player's lines with the module CSS."""
    html = render_player_lines(rows)
    if html:
        _inject_css()
        st.markdown(html, unsafe_allow_html=True)
