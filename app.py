"""BucketValue home: the slate for one ET date.

Pick a date, see every game grouped by tip time, open one for its page. The
schedule and rosters come from cdn.nba.com through bucketlib.live (short budget,
circuit breaker, stale-beats-empty with a visible notice). Projections come
from store.load_projections, the same cached frame the Game page reads, so a
card's line-count pill (SHOWN rows of the board only) equals the count on that
game's page, and "Model vs the board" uses the same simulator provider as the
Game tables.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import live, pipeline, props, props_ui, store  # noqa: E402
from bucketlib.theme import SITE_NAME, render_footer, render_nav, render_page_chrome  # noqa: E402
from bucketlib.util import esc, today_et  # noqa: E402

st.set_page_config(page_title=SITE_NAME, page_icon="static/favicon.svg", layout="wide")
render_page_chrome()

# ── Schedule ─────────────────────────────────────────────────────────────────
try:
    SCHED_LIVE = live.schedule()
except RuntimeError as e:
    render_nav("Home")
    st.markdown(store.render_notice(
        "<b>The NBA schedule feed is unreachable and nothing is cached.</b> "
        f"{SITE_NAME} cannot draw a slate without it. This is a data-source "
        f"outage, not an empty day. Details: {esc(e)}"), unsafe_allow_html=True)
    render_footer()
    st.stop()
SCHED = SCHED_LIVE.value
SEASON_DATES = sorted(SCHED["game_date"].unique())

# ── View date: seeded from ?date= only when the URL CHANGES ──────────────────
# A keyed widget ignores its default after the first render, and a sync that
# fires whenever URL and widget DISAGREE throws away the user's pick on the
# rerun before the URL catches up. So the last reconciled URL value is kept in
# its own key and the sync fires only when the URL itself changed.
def _parse_date(raw):
    try:
        return dt.date.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None


url_date = st.query_params.get("date")
if url_date != st.session_state.get("_url_date_seen"):
    parsed = _parse_date(url_date)
    st.session_state["view_date"] = parsed or st.session_state.get("view_date") or live.default_view_date()
    st.session_state["_url_date_seen"] = url_date
st.session_state.setdefault("view_date", live.default_view_date())


def _step(delta: int):
    st.session_state["view_date"] = st.session_state["view_date"] + dt.timedelta(days=delta)


def _jump(d: dt.date):
    st.session_state["view_date"] = d


view = st.session_state["view_date"]
if st.query_params.get("date") != view.isoformat():
    if st.query_params.get("date") is not None and "out" in st.query_params:
        # Mark-outs belong to the date they were made on: stepping the date
        # drops them rather than marking the same players out tomorrow.
        del st.query_params["out"]
    st.query_params["date"] = view.isoformat()
    st.session_state["_url_date_seen"] = view.isoformat()
MARKS = store.parse_out(st.query_params.get("out"))

try:
    ROSTER = live.roster().value
except RuntimeError:
    ROSTER = None
# Every internal link is a full page load (a new Streamlit session), so a
# visitor's pasted board travels as ?board=<token> on every minted URL.
TOKEN = props_ui.board_token()
keep = {"theme": "dark" if st.session_state.get("theme_dark") else "light",
        **store.view_params(TOKEN, MARKS)}
render_nav("Home", keep={"date": view.isoformat(), **store.view_params(TOKEN, MARKS)})
if ROSTER is not None:
    # Board text carries no dates: lines route to each player's game today or
    # tomorrow (all PrizePicks posts), plus the date on screen. Tipped games are
    # never a destination, so a past date on screen adds nothing.
    _route_dates = sorted({today_et(), today_et() + dt.timedelta(days=1), view})
    TOKEN = props_ui.render_update_lines_popover(view, SCHED, ROSTER, dates=_route_dates) or TOKEN
    keep.update(store.view_params(TOKEN, MARKS))

# ── Date bar ─────────────────────────────────────────────────────────────────
today = today_et()
with st.container(key="bv_datebar"):
    c_prev, c_next, c_date, c_y, c_t, c_tm, _ = st.columns([0.05, 0.05, 0.17, 0.11, 0.1, 0.12, 0.4])
    with c_prev:
        st.button("‹", key="bv_day_prev", on_click=_step, args=(-1,), help="Previous day")
    with c_next:
        st.button("›", key="bv_day_next", on_click=_step, args=(1,), help="Next day")
    with c_date:
        st.date_input("Date", key="view_date", format="YYYY-MM-DD")
    with c_y:
        st.button("Yesterday", key="bv_day_y", on_click=_jump, args=(today - dt.timedelta(days=1),))
    with c_t:
        st.button("Today", key="bv_day_today", on_click=_jump, args=(today,), disabled=(view == today))
    with c_tm:
        st.button("Tomorrow", key="bv_day_tm", on_click=_jump, args=(today + dt.timedelta(days=1),))
st.markdown('<div class="bv-bar-rule"></div>', unsafe_allow_html=True)

if SCHED_LIVE.stale:
    st.markdown(store.render_notice(
        f"<b>Showing a saved schedule from {SCHED_LIVE.age_min:.0f} minutes ago.</b> "
        "The live NBA feed did not answer just now, so scores and statuses may lag."),
        unsafe_allow_html=True)

# ── Slate ────────────────────────────────────────────────────────────────────
games = live.games_on(SCHED, view)
nba = games[games["game_type"].isin(["RS", "PI", "PO", "CUP_FINAL"])] if len(games) else games
pre = games[games["game_type"] == "PRE"] if len(games) else games

# One cached frame per (date, report, lines, mark-outs): the Game page reads
# the same one, so the cards, the board and the tables cannot disagree.
served = store.load_projections(view, MARKS) if store.projected_ids(SCHED, view) else None
FRAME = served.frame if served is not None else None
BOARD = None
if TOKEN and FRAME is not None and len(FRAME):
    try:
        _params = store.sim_params()
    except (OSError, ValueError, KeyError):
        _params = None
    BOARD = props_ui.date_board(TOKEN, view, FRAME, _params, ROSTER)

_odds_for_mast = live.odds_on(view).value if len(nba) else {}
_inj_note = ""
_inj = store.injury_fact(FRAME, served.info) if FRAME is not None and len(FRAME) else None
if _inj:
    _inj_note = f"Injury report: {_inj[0]}{f' ({esc(_inj[1])})' if _inj[1] else ''}."
st.markdown(store.render_masthead(view, nba, len(_odds_for_mast), injury_note=_inj_note,
                                  reports=store.reports_filed(FRAME)), unsafe_allow_html=True)

if len(nba):
    states = {r["game_id"]: live.game_state(r) for _, r in nba.iterrows()}
    odds = live.odds_on(view)
    by_game = {r["game_id"]: odds.value.get((r["away"], r["home"])) for _, r in nba.iterrows()}
    if served is not None and FRAME.empty:
        if served.info.withheld and served.info.reason is None:
            st.markdown(store.render_notice(
                f"<b>{esc(pipeline.NO_LINE)}.</b> The models were trained with the spread and "
                "total, so a game is projected once both are posted."), unsafe_allow_html=True)
        else:
            st.markdown(store.render_notice(
                "<b>Projections not published yet for this date.</b> The bot publishes them "
                "once the history features, the injury report and the lines are in."),
                unsafe_allow_html=True)
    elif served is not None:
        if served.applied:
            # Counted from the rows the live layer marked out: a mark on a
            # tipped game or on the committed stand-in moved nothing.
            _clear = store.page_url("/", {k: v for k, v in keep.items() if k != "out"},
                                    date=view.isoformat())
            st.markdown(store.adjusted_banner(len(served.applied), _clear), unsafe_allow_html=True)
        for block in store.projection_notes(served, today=today):
            st.markdown(block, unsafe_allow_html=True)
    st.markdown(store.render_slate(
        nba, view, keep, odds_by_game=by_game, states=states,
        line_counts=props_ui.line_counts_by_game(BOARD.shown) if BOARD is not None else None,
        subs=store.card_subs(served)), unsafe_allow_html=True)
    if odds.stale and odds.source == "none":
        st.markdown(store.render_notice(
            "<b>Game lines are unavailable right now.</b> The ESPN feed did not answer, "
            "so cards show no spread or total. This is a feed outage, not missing games."),
            unsafe_allow_html=True)
elif len(pre):
    st.markdown(store.render_notice(
        f"<b>Preseason: {len(pre)} game{'s' if len(pre) != 1 else ''}, no projections.</b> "
        "Preseason minutes are not regular-season minutes (2025-26 starters averaged "
        "half their usual minutes in this preseason), so the site does not project them. "
        "The regular season opens October 20."), unsafe_allow_html=True)
    states = {r["game_id"]: live.game_state(r) for _, r in pre.iterrows()}
    st.markdown(store.render_slate(pre, view, keep, states=states), unsafe_allow_html=True)
else:
    upcoming = [d for d in SEASON_DATES if d > view]
    nxt = (f" The next slate is <b>{upcoming[0].strftime('%A, %B %-d')}</b>."
           if upcoming else "")
    st.markdown(store.render_notice(f"<b>No NBA games on {view.strftime('%A, %B %-d')}.</b>{nxt}"),
                unsafe_allow_html=True)

# ── Model vs the board (only when the visitor pasted lines for this date) ────
if BOARD is not None:
    _labels = {str(r["game_id"]): f"{r['away']} at {r['home']}" for _, r in nba.iterrows()}
    props_ui.render_board(BOARD.shown, BOARD.hidden, BOARD.thin, BOARD.void,
                          scope_label=view.strftime("%B %-d"), meta=BOARD.meta,
                          game_labels=_labels)
elif TOKEN:
    board = props.get_board(TOKEN)
    todays = props.lines_for_date(board, view) if board is not None else None
    n = 0 if todays is None else len(todays)
    if n:
        st.markdown(store.render_notice(
            f"<b>{n} PrizePicks lines saved for {view.strftime('%B %-d')}.</b> The model "
            "comparison appears here once projections are published for this date; your "
            "board stays attached to this link until then."), unsafe_allow_html=True)

render_footer()
