"""BucketValue home: the slate for one ET date.

Pick a date, see every game grouped by tip time, open one for its page. The
schedule and rosters come from cdn.nba.com through bucketlib.live (short budget,
circuit breaker, stale-beats-empty with a visible notice). Projections come
from store.load_projections, the same cached frame the Game page reads, so a
card's line-count pill (SHOWN rows of the board only) equals the count on that
game's page, and "Model vs the board" uses the same simulator provider as the
Game tables.

The licensed odds feed (design 2.8; layout C, amendments C6 to C14). With a
key configured, store.feed_start kicks the live layer's walk before the
projection load, so the two overlap, and store.feed_result reads the lines
after it. The board then shows every book at once: one row per player and
stat, one column per book (DK, FD, MGM, PP, UD), each cell that book's line
with the model's lean at it, and the cards count every book's shown lines.
No ?book= is read or carried (an old link still loads). With no key none of
it runs and the page is what it was.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Before any bucketlib import: load the code on disk, never a stale package
# left in this process by an earlier deploy (freshen.py).
sys.modules.pop("freshen", None)
import freshen  # noqa: E402

freshen.ensure()

from bucketlib import live, props, props_ui, store  # noqa: E402
from bucketlib.theme import (SITE_NAME, render_footer, render_nav, render_page_chrome,  # noqa: E402
                             render_share)
from bucketlib.util import esc, today_et  # noqa: E402

st.set_page_config(page_title=SITE_NAME, page_icon="static/favicon.svg", layout="wide")
render_page_chrome()

# -- Schedule -----------------------------------------------------------------
try:
    SCHED_LIVE = live.schedule()
except RuntimeError:
    render_nav("Home")
    # A fixed sentence, never the error's text (RC K6): it can carry a host or
    # a server path.
    st.markdown(store.render_notice(
        "<b>The NBA schedule feed is unreachable and nothing is cached.</b> "
        f"{SITE_NAME} cannot draw a slate without it. This is a data-source "
        "outage, not an empty day. Reload in a minute."), unsafe_allow_html=True)
    render_footer()
    st.stop()
SCHED = SCHED_LIVE.value
SEASON_DATES = sorted(SCHED["game_date"].unique())

# -- View date: seeded from ?date= only when the URL CHANGES ------------------
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
    if st.query_params.get("date") is not None:
        # Mark-outs and a searched player belong to the date they were made
        # on: stepping the date drops them rather than carrying them over.
        for k in ("out", "player"):
            if k in st.query_params:
                del st.query_params[k]
    st.query_params["date"] = view.isoformat()
    st.session_state["_url_date_seen"] = view.isoformat()
MARKS = store.parse_out(st.query_params.get("out"))

# -- The licensed odds feed (design 2.8) --------------------------------------
# Kicked here, before the projection load below, so the walk runs while the
# projections load (M6 item 4); with no key it is "off" before any feed code
# runs (M9). FEED_ON tells the paste popover to say where the board's lines
# come from.
FEED_TICKET = store.feed_start(view, SCHED)
FEED_ON = FEED_TICKET.mode != "off"

try:
    ROSTER = live.roster().value
except RuntimeError:
    ROSTER = None
# Every internal link is a full page load (a new Streamlit session), so a
# visitor's pasted board travels as ?board=<token> on every minted URL. The
# keeper runs first, before the nav and every widget: a board this browser
# kept comes back onto the link for this whole run (props_ui.board_keeper).
TOKEN = props_ui.board_keeper(ROSTER)
keep = {"theme": "dark" if st.session_state.get("theme_dark") else "light",
        **store.view_params(TOKEN, MARKS)}
render_nav("Home", keep={"date": view.isoformat(), **store.view_params(TOKEN, MARKS)})
with st.container(key="bv_nav_actions"):
    if ROSTER is not None:
        # Board text carries no dates: lines route to each player's game
        # today or tomorrow (all PrizePicks posts), plus the date on screen.
        # Tipped games are never a destination, so a past date adds nothing.
        _route_dates = sorted({today_et(), today_et() + dt.timedelta(days=1), view})
        TOKEN = props_ui.render_update_lines_popover(view, store.mark_projected(SCHED, _route_dates),
                                                     ROSTER, dates=_route_dates,
                                                     in_nav_container=False,
                                                     feed_on=FEED_ON) or TOKEN
        keep.update(store.view_params(TOKEN, MARKS))
    render_share(store.page_url("/", keep, date=view.isoformat(),
                                player=st.query_params.get("player")))
FOOT_KEEP = {"date": view.isoformat(), **store.view_params(TOKEN, MARKS)}

# -- Date bar -----------------------------------------------------------------
today = today_et()
with st.container(key="bv_datebar"):
    c_prev, c_next, c_date, c_y, c_t, c_tm, _ = st.columns([0.05, 0.05, 0.17, 0.11, 0.1, 0.12, 0.4])
    with c_prev:
        st.button(chr(0x2039), key="bv_day_prev", on_click=_step, args=(-1,), help="Previous day")
    with c_next:
        st.button(chr(0x203A), key="bv_day_next", on_click=_step, args=(1,), help="Next day")
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

# -- Slate --------------------------------------------------------------------
games = live.games_on(SCHED, view)
nba = games[games["game_type"].isin(["RS", "PI", "PO", "CUP_FINAL"])] if len(games) else games
pre = games[games["game_type"] == "PRE"] if len(games) else games

# One cached frame per (date, report, lines, mark-outs): the Game page reads
# the same one, so the cards, the board and the tables cannot disagree.
IDS = store.projected_ids(SCHED, view)
served = store.load_projections(view, MARKS) if IDS else None
FRAME = served.frame if served is not None else None
FEATS = store.display_features(view) if IDS else None
# The feed's lines for the date, read once the projection load is done: every
# book's lines are the board, with or without a paste (design 2.8).
FEED = store.feed_result(FEED_TICKET)
FEED_LIVE = FEED.state != "off"
_params = None
if TOKEN or FEED_LIVE:
    try:
        _params = store.sim_params()
    except Exception:  # noqa: BLE001 - date_board explains a board it cannot build
        _params = None
FT = (store.feed_tables(view, FEED, served, _params, ROSTER)
      if FEED_LIVE and FRAME is not None and len(FRAME) else None)
BOARD = None
# Tables marked failed (the feed holds lines it could not compare) still make
# a board: date_board's explained empty one, never the prompt that says the
# board fills from the odds feed.
if (TOKEN or (FT is not None and (FT.tables or FT.failed))) and FRAME is not None and len(FRAME):
    BOARD = props_ui.date_board(TOKEN, view, FRAME, _params, ROSTER, feed=FT,
                                feed_info=store.feed_info(FEED, today) if FT is not None else None)
# The board carries the feed's rows: it is the grid of every book, and the
# cards count its shown lines, every book's (amendment C8).
FEED_BOARD = BOARD is not None and BOARD.feed is not None

states = {r["game_id"]: live.game_state(r) for _, r in nba.iterrows()} if len(nba) else {}
odds = live.odds_on(view) if len(nba) else None
# ESPN drops a game's line at tip: a started game shows the line its frozen
# projection was made with, or without projection rows the bot's last
# pre-tip capture (store.committed_odds) (display only).
started = {g for g, s in states.items() if s in ("live", "final")} | set(served.past if served else ())
posted = store.posted_lines(odds.value if odds is not None else {}, nba, FRAME, started,
                            snapshots=store.committed_odds(view) if started else None)
active = store.active_games(nba)

_inj_note = ""
_inj = store.injury_fact(FRAME, served.info) if FRAME is not None and len(FRAME) else None
if _inj:
    _inj_note = f"Injury report: {_inj[0]}{f' ({esc(_inj[1])})' if _inj[1] else ''}."
if len(nba):
    st.markdown(store.render_masthead(view, active, store.lines_posted(posted, active),
                                      injury_note=_inj_note, reports=store.reports_filed(FRAME)),
                unsafe_allow_html=True)
elif len(pre):
    # Preseason: the games and their first tip; no lines are read.
    st.markdown(store.render_masthead(view, pre, None, games_label="Preseason games"),
                unsafe_allow_html=True)
else:
    st.markdown(store.render_masthead(view, nba, None), unsafe_allow_html=True)


# -- Player search (?player=, seeded only when the URL changes) ---------------
def _player_search(frame, date: dt.date, keep: dict, feats) -> None:
    opts = store.player_options(frame)
    labels = dict(opts)
    url_player = st.query_params.get("player")
    if url_player != st.session_state.get("_url_player_seen") \
            or st.session_state.get("_player_date") != date:
        try:
            pid = int(str(url_player))
        except (TypeError, ValueError):
            pid = None
        st.session_state["bv_player"] = pid if pid in labels else None
        st.session_state["_url_player_seen"] = url_player
        st.session_state["_player_date"] = date

    def _on_pick() -> None:
        pid = st.session_state.get("bv_player")
        if pid is None:
            if "player" in st.query_params:
                del st.query_params["player"]
        else:
            st.query_params["player"] = str(pid)
        st.session_state["_url_player_seen"] = st.query_params.get("player")
    st.selectbox("Find a player", [p for p, _ in opts], key="bv_player",
                 format_func=lambda p: labels.get(p, str(p)), on_change=_on_pick,
                 placeholder="Find a player on this slate", label_visibility="collapsed")
    pid = st.session_state.get("bv_player")
    if pid in labels:
        st.markdown(store.player_panel(frame, pid, date, keep, feats), unsafe_allow_html=True)


if FRAME is not None and len(FRAME):
    _player_search(FRAME, view, keep, FEATS)

if len(nba):
    if served is not None and FRAME.empty:
        st.markdown(store.empty_notice(served, game_ids=IDS), unsafe_allow_html=True)
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
        nba, view, keep, odds_by_game={r["game_id"]: posted.get((r["away"], r["home"]))
                                       for _, r in nba.iterrows()},
        states=states,
        line_counts=props_ui.line_counts_by_game(BOARD.shown) if BOARD is not None else None,
        subs=store.card_subs(served, mpg=store.season_mpg(FEATS)),
        b2b=store.b2b_by_game(store.rest_by_game(FEATS)), feed=FEED_BOARD), unsafe_allow_html=True)
    if FEED.loading:
        # Games still loading: the watch shows the loading line and reruns the
        # page once their lines land (design 2.7, M6 item 3).
        store.feed_loading_watch(view.isoformat(), FEED.generation)
    if FRAME is not None and len(FRAME):
        _trust = store.trust_line(store.page_url("/About", keep, date=view.isoformat()))
        if _trust:
            st.markdown(f'<div class="bv-note">{_trust}</div>', unsafe_allow_html=True)
    if odds is not None and odds.stale and odds.source == "none":
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
    pre_states = {r["game_id"]: live.game_state(r) for _, r in pre.iterrows()}
    st.markdown(store.render_slate(pre, view, keep, states=pre_states), unsafe_allow_html=True)
else:
    upcoming = [d for d in SEASON_DATES if d > view]
    nxt = (f" The next slate is <b>{upcoming[0].strftime('%A, %B %-d')}</b>."
           if upcoming else "")
    st.markdown(store.render_notice(f"<b>No NBA games on {view.strftime('%A, %B %-d')}.</b>{nxt}"),
                unsafe_allow_html=True)

# -- Model vs the board (a pasted board, the odds feed, or both) --------------
_labels = {str(r["game_id"]): f"{r['away']} at {r['home']}" for _, r in nba.iterrows()}
if FEED_LIVE and len(nba):
    # The feed's notice goes right above the board, so it shows even when the
    # board draws nothing (m7): its state, the paste merge, the games each
    # book has not posted (named the schedule's way, N2; only those not yet
    # tipped, by the schedule's own state, build amendment B25; amendment C9)
    # and the lines taken down. The loading line is the watch's while it runs.
    if BOARD is not None and BOARD.paste_only:
        # Build amendment B17: the feed's lines could not be compared, so the
        # visitor's paste is the board, exactly as with the feed off, under
        # one note.
        st.markdown(props_ui.paste_only_note_html(), unsafe_allow_html=True)
    else:
        _tipped = {str(g) for g, s in states.items() if s in ("live", "final")}
        _notice = props_ui.feed_notice_html(BOARD.feed if FEED_BOARD else store.feed_info(FEED, today),
                                            today, _labels, with_loading=not FEED.loading,
                                            tipped=_tipped)
        if _notice:
            st.markdown(_notice, unsafe_allow_html=True)
if BOARD is not None:
    props_ui.render_board(BOARD.shown, BOARD.hidden, BOARD.thin, BOARD.void,
                          scope_label=view.strftime("%B %-d"), meta=BOARD.meta,
                          game_labels=_labels, feed=BOARD.feed)
    # DiamondValue's biggest-gaps strip, as its own section at the bottom. With
    # the feed's rows each chip names the books posting its line (amendment C11).
    props_ui.render_discrepancies(BOARD.shown, view.strftime("%B %-d"), books=FEED_BOARD)
elif TOKEN:
    board = props.get_board(TOKEN)
    todays = props.lines_for_date(board, view) if board is not None else None
    n = 0 if todays is None else len(todays)
    if n:
        st.markdown(store.render_notice(
            f"<b>{n} {'line' if n == 1 else 'lines'} saved for {view.strftime('%B %-d')}.</b> "
            "The model comparison appears here once projections are published for this "
            "date; your board stays attached to this link until then."), unsafe_allow_html=True)
elif ROSTER is not None and FRAME is not None and len(FRAME) \
        and any(s == "upcoming" for s in states.values()):
    # DiamondValue's empty-board prompt: a paste could still land on this date.
    # With the feed on it says the board fills by itself, and only while the
    # feed holds no line for the date (lines it holds are the board's to show
    # or explain); while the feed is down, paused or loading its own notice
    # above says what to do instead.
    if not FEED_LIVE:
        st.markdown(f'<div class="bv-note">{props_ui.NO_LINES_SLATE}</div>', unsafe_allow_html=True)
    elif FEED.state in ("idle", "ok") and not FEED.loading and not len(FEED.value):
        st.markdown(f'<div class="bv-note">{props_ui.NO_LINES_SLATE_FEED}</div>',
                    unsafe_allow_html=True)

render_footer(FOOT_KEEP)
