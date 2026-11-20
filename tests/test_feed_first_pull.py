"""The empty-board prompts promise the odds feed's first-pull hour (2026-11-19).

Barrett asked for the lines at 8 AM instead of 9 AM. The first pull is the
setting BUCKETVALUE_ODDS_FIRST_PULL_ET ("08:50" by default; "07:55" gives the
lines by 8 AM), and the pages' prompts said "from about 9 AM ET" in fixed
words. They now say the setting's time to the nearest hour
(props_ui.feed_hour), read on every view (store.feed_first_pull), and the
default's words are exactly the old ones. The pages themselves are in
tests/test_pages_feed.py (test_the_prompts_promise_the_first_pull_settings_hour).
"""
from __future__ import annotations

import datetime as dt
import logging

import pytest

from bucketlib import props_ui, store
from bucketlib.sources import oddsapi as O

SETTING = "BUCKETVALUE_ODDS_FIRST_PULL_ET"

# The prompts as they read before the hour followed the setting (0293ddc..f4a29e5).
OLD_SLATE = ("<b>No lines on this page yet.</b> On game days the board fills from "
             "the odds feed from about 9 AM ET. You can also paste a PrizePicks "
             "board with <b>Update lines</b> in the top bar.")
OLD_GAME = ("<b>No lines for this game yet.</b> On game days they load from the "
            "odds feed from about 9 AM ET, and each player's lines open under his "
            "team. You can also paste a PrizePicks board with <b>Update lines</b> "
            "in the top bar.")


@pytest.mark.parametrize("hms, said", [
    ((8, 50), "9 AM"),          # the default
    ((7, 55), "8 AM"),          # lines by 8 AM
    ((8, 0), "8 AM"),           # on the hour: that hour
    ((8, 0, 1), "9 AM"),        # any later second: the next hour, never earlier
    ((8, 1), "9 AM"),
    ((8, 29), "9 AM"),
    ((11, 45), "12 PM"),        # noon
    ((12, 0), "12 PM"),
    ((12, 10), "1 PM"),
    ((13, 5), "2 PM"),
    ((23, 40), "12 AM"),        # midnight
    ((0, 0), "12 AM"),
    ((0, 10), "1 AM"),
    ((6, 0), "6 AM"),
])
def test_the_hour_is_the_first_pull_time_rounded_up(hms, said):
    assert props_ui.feed_hour(dt.time(*hms)) == said


def test_the_promise_is_never_earlier_than_the_first_pull():
    for h in range(24):
        for m in (0, 1, 29, 30, 59):
            said = props_ui.feed_hour(dt.time(h, m))
            n, half = said.split()
            promised = (int(n) % 12) + (12 if half == "PM" else 0)
            gap = (promised - h) % 24 * 60 - m
            assert 0 <= gap < 60, (h, m, said)


@pytest.mark.parametrize("given", [None, "07:55", 7.55, dt.datetime(2026, 11, 20, 7, 55)])
def test_anything_but_a_time_reads_as_the_default_setting(given):
    """A datetime is not a datetime.time either (it is a date subclass): the
    default, never an hour taken from a full timestamp."""
    assert O.read_settings({}).first_pull_et == dt.time(8, 50)
    assert props_ui.FEED_FIRST_PULL_DEFAULT == O.read_settings({}).first_pull_et
    assert props_ui.feed_hour(given) == "9 AM"


def test_the_default_prompts_are_the_words_the_site_has_always_shown():
    assert props_ui.NO_LINES_SLATE_FEED == OLD_SLATE
    assert props_ui.NO_LINES_GAME_FEED == OLD_GAME
    assert props_ui.no_lines_slate_feed() == OLD_SLATE == props_ui.no_lines_slate_feed(None)
    assert props_ui.no_lines_game_feed() == OLD_GAME == props_ui.no_lines_game_feed(None)
    assert props_ui.no_lines_slate_feed(dt.time(8, 50)) == OLD_SLATE
    assert props_ui.no_lines_game_feed(dt.time(8, 50)) == OLD_GAME


def test_the_prompts_follow_the_setting():
    for prompt in (props_ui.no_lines_slate_feed(dt.time(7, 55)),
                   props_ui.no_lines_game_feed(dt.time(7, 55))):
        assert "from about 8 AM ET" in prompt and "9 AM" not in prompt
    assert props_ui.no_lines_slate_feed(dt.time(7, 55)) == OLD_SLATE.replace("9 AM", "8 AM")
    assert props_ui.no_lines_game_feed(dt.time(7, 55)) == OLD_GAME.replace("9 AM", "8 AM")


def test_the_store_reads_the_setting_on_every_call(monkeypatch):
    assert store.feed_first_pull({}) == dt.time(8, 50)
    assert store.feed_first_pull({SETTING: "07:55"}) == dt.time(7, 55)
    assert store.feed_first_pull({SETTING: " 07:55 "}) == dt.time(7, 55)
    assert store.feed_first_pull({SETTING: "7:5x"}) == dt.time(8, 50)      # unreadable: the default
    assert store.feed_first_pull({SETTING: "25:00"}) == dt.time(8, 50)
    # None is the process environment, read on this call, not at import.
    monkeypatch.delenv(SETTING, raising=False)
    assert store.feed_first_pull() == dt.time(8, 50)
    monkeypatch.setenv(SETTING, "07:55")
    assert store.feed_first_pull() == dt.time(7, 55)
    monkeypatch.setenv(SETTING, "08:10")
    assert store.feed_first_pull() == dt.time(8, 10)


def test_a_failed_read_is_none_one_warning_and_the_default_hour(monkeypatch, caplog):
    def boom(env):
        raise RuntimeError("never in the page")
    monkeypatch.setattr(store.O, "read_settings", boom)
    with caplog.at_level(logging.WARNING, logger="bucketvalue"):
        got = store.feed_first_pull({SETTING: "07:55"})
    assert got is None
    lines = [r.getMessage() for r in caplog.records if r.name == "bucketvalue"]
    assert lines == ["oddsapi: first-pull time unreadable (RuntimeError)"]
    assert "never in the page" not in caplog.text
    monkeypatch.undo()                     # the prompts' own default read needs the real reader
    assert props_ui.no_lines_slate_feed(got) == OLD_SLATE


def test_the_prompts_read_no_environment_themselves(monkeypatch):
    """With no time given, the prompts are the DEFAULT's, whatever the
    process environment holds: they read no environment (the constants are
    built the same way, at import), and a view's own hour comes only from
    store.feed_first_pull. So a page that forgot to pass it would say 9 AM
    under a 07:55 setting, which the page tests catch."""
    monkeypatch.setenv(SETTING, "07:55")
    assert props_ui.feed_hour() == "9 AM"
    assert props_ui.no_lines_slate_feed() == OLD_SLATE
    assert props_ui.no_lines_game_feed() == OLD_GAME
    assert props_ui.no_lines_slate_feed(store.feed_first_pull()) == OLD_SLATE.replace("9 AM", "8 AM")
