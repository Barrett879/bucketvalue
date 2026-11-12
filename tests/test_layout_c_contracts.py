"""Layout C's contracts (build 8; amendments C1, C16 and C17.2): what the
change from one book at a time to every book at once must leave alone, and
what it must take away.

Left alone, pinned as recorded at 5bbb11e (layout B's last commit): the
tracker's API (the local accuracy tracker in ~/BucketValueAccuracy calls
these names, so their signatures and results are a contract), and the odds
feed's memory shape (the buying side is unchanged by a presentation change).
Taken away: layout B's controls (the book pills, the ?book= parameter, the
Other books cell, the picked book's notices), which no name, field or
keyword may keep.

Nothing here reaches a network or reads a clock: the one instant used is a
fixed past one, passed in.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import inspect

import pandas as pd
import pytest

from bucketlib import dist, live, pipeline, props, props_ui, simprov, state, store, util

FIXED = dt.datetime(2026, 3, 12, 13, 0, tzinfo=dt.timezone.utc)      # a fixed past instant


def test_the_trackers_api_is_unchanged():
    signatures = {
        "pipeline.live_projections": (pipeline.live_projections,
                                      "(date, now, marked_out=None, *, report=None, odds=None, "
                                      "models: 'M.Models | None' = None, schedule: "
                                      "'pd.DataFrame | None' = None) -> 'tuple[pd.DataFrame, Info]'"),
        "pipeline.season_for_date": (pipeline.season_for_date, "(d: 'dt.date | str') -> 'str'"),
        "simprov.provider_for": (simprov.provider_for,
                                 "(projections: 'pd.DataFrame | None', params: 'D.SimParams', "
                                 "n: 'int' = 3000) -> 'SimProvider'"),
        "props.fantasy_shown": (props.fantasy_shown, "(medians) -> 'float | None'"),
        "props.side_probabilities": (props.side_probabilities,
                                     "(p_lt: 'float', p_eq: 'float', p_gt: 'float', p_void_more: "
                                     "'float | None' = None) -> 'tuple[float, float]'"),
        "props.lean_from": (props.lean_from, "(q_more: 'float', band: 'float' = 0.01) -> 'str'"),
        "state.season_partitions": (state.season_partitions,
                                    "(season: 'str') -> 'dict[str, pd.DataFrame]'"),
        "state.ingested_ids": (state.ingested_ids, "(season: 'str') -> 'set[str]'"),
    }
    for name, (fn, want) in signatures.items():
        assert str(inspect.signature(fn)) == want, name
    assert pipeline.season_for_date is util.season_for_date
    assert (props.EVEN_BAND, props.SHOWN_PLACES) == (0.01, 1)
    assert (props.BREAKEVEN_LO, props.BREAKEVEN_HI) == (54.7, 58.5)
    assert [f.name for f in dataclasses.fields(dist.SimParams)] == ["min_resid", "beta", "gamma_k"]
    # Results, exactly. The function divides by p_gt + p_lt, which is
    # 0.8999999999999999 here, so the pair is (0.6666666666666667,
    # 0.33333333333333337); (0.6 / 0.9, 0.3 / 0.9) is NOT equal and must not
    # be written (amendment C1V, C1R-01).
    assert props.side_probabilities(0.3, 0.1, 0.6) == (0.6 / (0.6 + 0.3), 0.3 / (0.6 + 0.3))
    assert props.side_probabilities(0, 1, 0) == (0.5, 0.5)
    assert [props.lean_from(q) for q in (0.52, 0.511, 0.51, 0.5, 0.49, 0.489, 0.48)] == [
        "More", "More", "Even", "Even", "Even", "Less", "Less"]
    assert props.fantasy_shown({"pts": 20, "reb": 10, "ast": 5, "stl": 1, "blk": 1,
                                "tov": 3}) == 42.5


def test_the_feed_memory_shape_is_unchanged():
    """Hard rule 9 of build 8: the feed's buying, memory and ledger are not
    touched by layout C."""
    assert live.FEED_MEMORY_VERSION == 2
    mem = live.new_feed_memory(FIXED, False)
    assert sorted(mem) == [
        "born_at", "brake_day", "brake_used", "braked", "breaker_day", "budget", "calls",
        "event_locks", "events", "events_fail", "futures", "games", "generation", "had_ledger",
        "lock", "queued", "streak", "v", "walk", "warned", "work"]
    assert sorted(mem["budget"]) == ["auth", "auth_until", "last", "mode", "pending", "remaining",
                                     "seen_at", "used"]
    entry = live._new_entry("0022500953", "2026-03-12", FIXED, FIXED)
    assert sorted(entry) == [
        "books_at", "braked", "empty_next", "fail_until", "fails", "first_at", "frame",
        "free_failed", "game_date", "game_id", "held", "last_at", "n_pulled", "pp_missing",
        "pp_next", "pulled", "pulls", "tip_utc", "vendor_tip"]
    assert live._LEDGER_KEYS == frozenset(mem)
    assert live._valid_ledger(mem) is True


# Amendment C1's removed names in props_ui ("not rebuilt anywhere"), and the
# one renamed (OTHER_BOOKS_AGE is LINE_TIME_AGE).
LAYOUT_B_NAMES = ("BOOK_KEY", "BOOK_SEEN", "BOOK_PARAM", "_url_book", "seed_book_from_url",
                  "_on_book", "render_book_pills", "selected_book", "_book_of", "_book_lines",
                  "feed_games", "_line_head", "_OtherBooks", "books_legend_html", "PICK_ANOTHER",
                  "book_missing_html", "book_missing_game_html", "_own_times", "OTHER_BOOKS_AGE")


def test_layout_bs_controls_are_gone():
    for name in LAYOUT_B_NAMES:
        assert not hasattr(props_ui, name), name
    assert props_ui.LINE_TIME_AGE == pd.Timedelta(minutes=1)
    view = {f.name for f in dataclasses.fields(props_ui.BoardView)}
    assert not view & {"context", "ref_by_game", "missing_games"}
    assert {"books_by_game", "missing"} <= view
    assert [f.name for f in dataclasses.fields(props.FeedBoard)] == [
        "lines", "meta", "books_by_game", "missing", "paste_won", "feed_won", "paste_times"]
    for fn, gone in ((props.feed_board, "selected"), (props_ui.date_board, "selected"),
                     (props_ui._feed_board_view, "selected"), (store.view_params, "book"),
                     (props_ui.ledger_html, "context"), (props_ui.render_board, "context")):
        assert gone not in inspect.signature(fn).parameters, (fn.__name__, gone)
    with pytest.raises(TypeError):
        props.feed_board(props.FeedTables(), None, None, None, selected="prizepicks")
    with pytest.raises(TypeError):
        props_ui.date_board(None, FIXED.date(), None, None, selected="prizepicks")
    with pytest.raises(TypeError):
        store.view_params("tok", (), book="draftkings")
    with pytest.raises(TypeError):
        store.view_params("tok", (), "draftkings")
