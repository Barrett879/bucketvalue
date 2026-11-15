"""The slow-build line (round BKS, 2026-11-15; docs/runbook.md section 9).

When the page's frame recompute (the live_projections call in
store._live_frame) or its board build (store._tables_now) takes
store.SLOW_BUILD_S seconds or more, the "bucketvalue" log gets ONE WARNING
line of fixed words and numbers: the step, its seconds to one decimal,
os.cpu_count(), the cgroup CPU limit in CPUs (or none) and the OpenMP thread
count the process would use outside the recompute's one-thread limit. It
holds no key, URL, path, setting value, player or game. The facts are read
only when the line is about to be logged; reading the cgroup files never
raises; the line never costs the page; and the timing changes no cached
value and no cache key.

Every duration comes from a pinned stand-in for store._build_clock: its reads
alternate between 0 and the step's seconds, so each timed step takes exactly
those seconds. Two tests pin what each step's seconds cover instead, with a
stand-in clock that only the stand-in parts of the step move, each by its own
seconds. Nothing sleeps and nothing reads a real clock. The cgroup
files are written in tmp_path. The frame steps use a stand-in recompute; the
board builds run on test_props_feed's world (the serve fixture's 2026-03-12
slate, the CONSTRUCTED odds fixtures through the real parser) with the cheap
baseline provider, as test_store_feed's cache tests do. No network.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from bucketlib import dist as D
from bucketlib import live
from bucketlib import pipeline as P
from bucketlib import props, simprov, store
from bucketlib.util import ET, UTC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import test_props_feed as PF  # noqa: E402  (the feed frame through the real parser)

DAY = H.DATE                                               # 2026-03-12
NOON = dt.datetime(2026, 3, 12, 12, 0, tzinfo=ET)
LINE = re.compile(r"slow build: step=(frame|board) seconds=(\d+\.\d) cpu_count=(\d+|none) "
                  r"cgroup_cpus=(\d+\.\d\d|none) openmp_threads=(\d+|none)")


class Clock:
    """store._build_clock pinned: odd reads give 0.0 and even reads `took`,
    so a step that reads it at its start and its end took exactly `took`."""

    def __init__(self, took: float):
        self.took, self.reads = float(took), 0

    def __call__(self) -> float:
        self.reads += 1
        return 0.0 if self.reads % 2 else self.took


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.WARNING, logger="bucketvalue")
    return caplog


@pytest.fixture(autouse=True)
def _no_cached_builds():
    """No cached frame or tables passes into or out of a test here."""
    store._live_frame.clear()
    store._feed_tables.clear()
    yield
    store._live_frame.clear()
    store._feed_tables.clear()


@pytest.fixture(scope="module")
def world():
    return H.load()


@pytest.fixture(scope="module")
def feed_frame(world):
    return PF.feed(world)


@pytest.fixture
def cheap(monkeypatch):
    """The baseline provider in place of the simulator: the board's timing is
    under test here, not the simulation."""
    monkeypatch.setattr(simprov, "provider_for",
                        lambda frame, params: props.MeanDispersionProvider(frame))


def _slow_lines(caplog) -> list[str]:
    out = []
    for r in caplog.records:
        if r.name == "bucketvalue" and r.getMessage().startswith("slow build"):
            assert r.levelno == logging.WARNING
            out.append(r.getMessage())
    return out


def _openmp_now() -> str:
    """threadpoolctl's OpenMP count for this thread, as the line spells it."""
    counts = [int(i["num_threads"]) for i in threadpool_info() if i.get("user_api") == "openmp"]
    return str(max(counts)) if counts else "none"


def _frame(monkeypatch, took: float, recompute=None):
    """store._live_frame's body (the cache bypassed) with a stand-in
    recompute that takes `took` seconds on the pinned clock."""
    monkeypatch.setattr(store, "_build_clock", Clock(took))
    monkeypatch.setattr(store, "models", lambda: None)
    monkeypatch.setattr(P, "live_projections", recompute or (lambda *a, **k: ("frame", "info")))
    return store._live_frame.__wrapped__(DAY.isoformat(), ("bks",), ("bks",), (), (),
                                         NOON.astimezone(UTC), None, None, None)


def _board(monkeypatch, world, feed_frame, took: float) -> props.FeedTables:
    """store._tables_now on the fixtures' feed, taking `took` seconds."""
    monkeypatch.setattr(store, "_build_clock", Clock(took))
    return store._tables_now(feed_frame, world.proj, D.SimParams(), world.roster)


# -- The threshold ---------------------------------------------------------------------
def test_the_threshold_is_twenty_seconds():
    assert store.SLOW_BUILD_S == 20.0


def test_a_frame_recompute_logs_one_line_from_the_threshold_and_none_below(monkeypatch, logs):
    for took in (0.0, 1.5, store.SLOW_BUILD_S - 0.05):
        assert _frame(monkeypatch, took) == ("frame", "info")
    assert _slow_lines(logs) == []
    for took in (store.SLOW_BUILD_S, 368.96, 375.04):
        logs.clear()
        assert _frame(monkeypatch, took) == ("frame", "info")
        (line,) = _slow_lines(logs)
        m = LINE.fullmatch(line)
        assert m, line
        assert (m.group(1), m.group(2)) == ("frame", f"{took:.1f}")


def test_a_board_build_logs_one_line_from_the_threshold_and_none_below(
        monkeypatch, logs, world, feed_frame, cheap):
    for took in (0.0, store.SLOW_BUILD_S - 0.05):
        assert len(_board(monkeypatch, world, feed_frame, took).tables) == 9
    assert _slow_lines(logs) == []
    for took in (store.SLOW_BUILD_S, 312.44):
        logs.clear()
        assert len(_board(monkeypatch, world, feed_frame, took).tables) == 9
        (line,) = _slow_lines(logs)
        m = LINE.fullmatch(line)
        assert m, line
        assert (m.group(1), m.group(2)) == ("board", f"{took:.1f}")


# -- What each step's seconds cover ------------------------------------------------------
class Ticking:
    """store._build_clock pinned to a clock that only the stand-in parts of a
    step move: takes(seconds, fn) is fn, adding its seconds when it runs."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def takes(self, seconds: float, fn):
        def run(*a, **k):
            self.now += seconds
            return fn(*a, **k)
        return run


def test_the_board_seconds_cover_the_feed_the_simulation_and_the_compares(
        monkeypatch, logs, world, feed_frame):
    """prepare_feed takes 7 s, the simulator's provider 9 s and compare_books
    11 s: the line says 27.0. No part alone reaches 20 s and no two parts
    make 27, so a board timing that leaves one out logs nothing or another
    number."""
    clock = Ticking()
    monkeypatch.setattr(store, "_build_clock", clock)
    monkeypatch.setattr(props, "prepare_feed", clock.takes(7.0, props.prepare_feed))
    monkeypatch.setattr(simprov, "provider_for",
                        clock.takes(9.0, lambda frame, params: props.MeanDispersionProvider(frame)))
    monkeypatch.setattr(props, "compare_books", clock.takes(11.0, props.compare_books))
    ft = store._tables_now(feed_frame, world.proj, D.SimParams(), world.roster)
    assert len(ft.tables) == 9 and clock.now == 27.0
    (line,) = _slow_lines(logs)
    assert LINE.fullmatch(line).group(1, 2) == ("board", "27.0"), line


def test_the_frame_seconds_cover_the_recompute_and_not_the_model_load(monkeypatch, logs):
    """The model load takes 100 s and live_projections 25 s: the line says
    25.0. The load is store.models(), cached once per process, so it is not
    the recompute's time."""
    clock = Ticking()
    monkeypatch.setattr(store, "_build_clock", clock)
    monkeypatch.setattr(store, "models", clock.takes(100.0, lambda: None))
    monkeypatch.setattr(P, "live_projections", clock.takes(25.0, lambda *a, **k: ("frame", "info")))
    assert store._live_frame.__wrapped__(DAY.isoformat(), ("bks", "span"), ("bks",), (), (),
                                         NOON.astimezone(UTC), None, None, None) == ("frame", "info")
    assert clock.now == 125.0
    (line,) = _slow_lines(logs)
    assert LINE.fullmatch(line).group(1, 2) == ("frame", "25.0"), line


# -- The line's content ----------------------------------------------------------------
def test_the_line_is_fixed_words_and_numbers_only(monkeypatch, logs, tmp_path, world, feed_frame,
                                                  cheap):
    """Each field is the process's: os.cpu_count(), the cgroup limit read from
    store.CGROUP_DIR, and threadpoolctl's OpenMP count. Nothing of the slate
    reaches it: no game id, no player, no book, no path."""
    (tmp_path / "cpu.max").write_text("150000 100000\n")
    monkeypatch.setattr(store, "CGROUP_DIR", tmp_path)
    _frame(monkeypatch, 400.0)
    _board(monkeypatch, world, feed_frame, 400.0)
    frame_line, board_line = _slow_lines(logs)
    cpus = os.cpu_count()
    for line, step in ((frame_line, "frame"), (board_line, "board")):
        m = LINE.fullmatch(line)
        assert m, line
        assert m.groups() == (step, "400.0", "none" if cpus is None else str(cpus), "1.50",
                              _openmp_now())
    names = (set(world.roster["name"]) | set(world.proj["game_id"].astype(str))
             | set(PF.FIXTURE_BOOKS) | {"/", "http", str(tmp_path)})
    for line in (frame_line, board_line):
        assert not [n for n in names if n and n in line], line
        assert line.isascii() and "\n" not in line


def test_the_openmp_count_in_the_line_is_the_one_outside_the_limit(monkeypatch, logs):
    """The recompute runs on one thread, and the line reports the count the
    process uses everywhere else: it is logged after the limit is lifted.
    The count is the OpenMP runtime's own (threadpoolctl's), not the one
    scikit-learn caps at the physical cores: with this thread's count set
    above the CPUs, as OMP_NUM_THREADS can set it, the line says so."""
    inside = []

    def recompute(*a, **k):
        inside.append(_openmp_now())
        return "frame", "info"
    outside = _openmp_now()
    _frame(monkeypatch, 400.0, recompute)
    (line,) = _slow_lines(logs)
    assert LINE.fullmatch(line).group(5) == outside
    assert inside == ["1"]
    logs.clear()
    above = 2 * (os.cpu_count() or 1) + 2
    with threadpool_limits(limits=above, user_api="openmp"):
        _frame(monkeypatch, 400.0, recompute)
        raised = _openmp_now()
    (line,) = _slow_lines(logs)
    assert LINE.fullmatch(line).group(5) == raised != "1"
    assert inside == ["1", "1"] and _openmp_now() == outside


# -- The cgroup limit ------------------------------------------------------------------
_ARABIC_TWO = chr(0x0662)       # a digit outside ASCII: int() would read it, the line must not


@pytest.mark.parametrize("files, want", [
    ({}, "none"),                                                       # no cgroup files at all
    ({"cpu.max": "max 100000\n"}, "none"),                              # v2: no limit
    ({"cpu.max": "200000 100000\n"}, "2.00"),                           # v2: two CPUs
    ({"cpu.max": "50000 100000\n"}, "0.50"),
    ({"cpu.max": "150000 100000"}, "1.50"),
    ({"cpu.max": "  400000\t100000 \n"}, "4.00"),
    ({"cpu/cpu.cfs_quota_us": "300000\n", "cpu/cpu.cfs_period_us": "100000\n"}, "3.00"),   # v1
    ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, "none"),      # v1: none
    ({"cpu/cpu.cfs_quota_us": "300000\n"}, "none"),                     # v1 without its period
    ({"cpu/cpu.cfs_period_us": "100000\n"}, "none"),                    # v1 without its quota
    ({"cpu.max": "max 100000\n", "cpu/cpu.cfs_quota_us": "300000\n",
      "cpu/cpu.cfs_period_us": "100000\n"}, "none"),                    # cpu.max decides when there
    ({"cpu.max": ""}, "none"),
    ({"cpu.max": "200000"}, "none"),
    ({"cpu.max": "max"}, "none"),
    ({"cpu.max": "200000 100000 7"}, "none"),
    ({"cpu.max": "0 100000"}, "none"),
    ({"cpu.max": "200000 0"}, "none"),
    ({"cpu.max": "-200000 100000"}, "none"),
    ({"cpu.max": "+200000 100000"}, "none"),
    ({"cpu.max": "2e5 1e5"}, "none"),
    ({"cpu.max": "200000.0 100000"}, "none"),
    ({"cpu.max": "2_00000 100000"}, "none"),
    ({"cpu.max": "nan 100000"}, "none"),
    ({"cpu.max": "inf 100000"}, "none"),
    ({"cpu.max": "9" * 19 + " 100000"}, "none"),
    ({"cpu.max": "200000 100000" + " " * 60 + "junk"}, "none"),        # past 64 bytes: no cgroup file
    ({"cpu.max": "200000 100000" + " " * 51}, "2.00"),                  # 64 bytes exactly
    ({"cpu.max": _ARABIC_TWO + "00000 100000"}, "none"),
    ({"cpu.max": bytes([0xFF, 0xFE, 0x20, 0x31])}, "none"),
    ({"cpu/cpu.cfs_quota_us": "abc", "cpu/cpu.cfs_period_us": "100000"}, "none"),
    ({"cpu/cpu.cfs_quota_us": "300000", "cpu/cpu.cfs_period_us": ""}, "none"),
])
def test_the_cgroup_limit_reads_as_cpus_or_none(tmp_path, files, want):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    assert store._cgroup_cpus(tmp_path) == want


def test_unreadable_cgroup_files_read_as_none_and_never_raise(tmp_path, monkeypatch):
    good = tmp_path / "good"
    good.mkdir()
    (good / "cpu.max").write_text("200000 100000\n")
    assert store._cgroup_cpus(good) == "2.00"                  # it reads when it can
    assert re.fullmatch(r"\d+\.\d\d|none", store._cgroup_cpus())   # this host's, whatever it is
    folder = tmp_path / "folder"
    (folder / "cpu.max").mkdir(parents=True)                   # a directory where the file is
    assert store._cgroup_cpus(folder) == "none"
    v1 = tmp_path / "v1"
    (v1 / "cpu" / "cpu.cfs_quota_us").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert store._cgroup_cpus(v1) == "none"
    afile = tmp_path / "afile"
    afile.write_text("200000 100000\n")
    assert store._cgroup_cpus(afile) == "none"                 # a root that is a file
    assert store._cgroup_cpus(tmp_path / ("bad" + chr(0) + "name")) == "none"
    assert store._cgroup_cpus(tmp_path / "missing") == "none"

    def refused(*a, **k):
        raise PermissionError(13, "Permission denied")
    with monkeypatch.context() as mp:
        mp.setattr(store, "open", refused, raising=False)      # store's own open only
        assert store._cgroup_cpus(good) == "none"

    def broken(path):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(store, "_cgroup_text", broken)
    assert store._cgroup_cpus(good) == "none"


# -- Read only when the line is logged ---------------------------------------------------
def test_the_facts_are_read_only_when_the_line_is_logged(monkeypatch, logs, tmp_path, world,
                                                         feed_frame, cheap):
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    monkeypatch.setattr(store, "CGROUP_DIR", tmp_path)
    reads = []
    real_limit, real_text, real_info = store._cgroup_cpus, store._cgroup_text, store.threadpool_info
    monkeypatch.setattr(store, "_cgroup_cpus", lambda root=None: reads.append("limit") or real_limit(root))
    monkeypatch.setattr(store, "_cgroup_text", lambda p: reads.append("file") or real_text(p))
    monkeypatch.setattr(store, "threadpool_info", lambda: reads.append("openmp") or real_info())
    _frame(monkeypatch, store.SLOW_BUILD_S - 0.05)
    _board(monkeypatch, world, feed_frame, 1.0)
    assert reads == [] and _slow_lines(logs) == []
    # A log that takes no WARNING line (BUCKETVALUE_LOG=ERROR): nothing read, nothing said.
    logs.set_level(logging.ERROR, logger="bucketvalue")
    _frame(monkeypatch, 400.0)
    _board(monkeypatch, world, feed_frame, 400.0)
    assert reads == [] and _slow_lines(logs) == []
    logs.set_level(logging.WARNING, logger="bucketvalue")
    _frame(monkeypatch, 400.0)
    assert reads == ["limit", "file", "openmp"]
    (line,) = _slow_lines(logs)
    assert LINE.fullmatch(line).group(4) == "2.00"


# -- The page is never worse for it ------------------------------------------------------
def test_a_slow_build_that_raises_still_logs_its_line_and_raises_as_before(
        monkeypatch, logs, world, feed_frame, cheap):
    def fails(*a, **k):
        raise RuntimeError("the recompute failed")
    with pytest.raises(RuntimeError, match="the recompute failed"):
        _frame(monkeypatch, 400.0, fails)

    def boom(*a, **k):
        raise KeyError("compare failed")
    monkeypatch.setattr(props, "compare_books", boom)
    with pytest.raises(KeyError, match="compare failed"):
        _board(monkeypatch, world, feed_frame, 400.0)
    assert [LINE.fullmatch(x).group(1) for x in _slow_lines(logs)] == ["frame", "board"]


def test_a_line_that_cannot_be_made_never_costs_the_page(monkeypatch, logs, world, feed_frame,
                                                         cheap):
    monkeypatch.setattr(store, "_cgroup_cpus", lambda root=None: 1 / 0)
    assert _frame(monkeypatch, 400.0) == ("frame", "info")
    assert len(_board(monkeypatch, world, feed_frame, 400.0).tables) == 9
    assert _slow_lines(logs) == []
    monkeypatch.setattr(store, "_cgroup_cpus", lambda root=None: "none")

    def no_log(*a, **k):
        raise ValueError("the log failed")
    monkeypatch.setattr(store.C.logger, "warning", no_log)
    assert _frame(monkeypatch, 400.0) == ("frame", "info")
    for odd in (float("nan"), None, "slow"):
        assert store._note_if_slow("frame", odd) is None
    assert _slow_lines(logs) == []


# -- No cached value or key moves --------------------------------------------------------
def test_the_timing_changes_no_cached_frame_or_its_key(monkeypatch, logs):
    """The cached recompute (st.cache_data) on a slow clock: the value is the
    recompute's own, the same key is a hit (no recompute, no second line),
    and another key computes again."""
    frame = pd.DataFrame({"person_id": [203999, 1630595], "pts": [27.25, 25.5]})
    calls = []

    def recompute(day, now, **k):
        calls.append((day, now))
        return frame, "info"
    monkeypatch.setattr(store, "_build_clock", Clock(400.0))
    monkeypatch.setattr(store, "models", lambda: None)
    monkeypatch.setattr(P, "live_projections", recompute)
    kw = dict(_now=NOON.astimezone(UTC), _report=None, _odds=None, _sched=None)
    key = (DAY.isoformat(), ("report", "12:00"), ("lines",), (), ())
    first = store._live_frame(*key, **kw)
    again = store._live_frame(*key, **kw)
    other = store._live_frame(DAY.isoformat(), ("report", "12:15"), ("lines",), (), (), **kw)
    assert calls == [(DAY, NOON.astimezone(UTC))] * 2
    for got in (first, again, other):
        assert isinstance(got, tuple) and len(got) == 2 and got[1] == "info"
        pd.testing.assert_frame_equal(got[0], frame)
    assert [LINE.fullmatch(x).group(1) for x in _slow_lines(logs)] == ["frame", "frame"]


def test_the_timing_changes_no_cached_tables_or_their_key(monkeypatch, logs, world, feed_frame,
                                                          cheap):
    """The cached board (st.cache_resource through store.feed_tables) on a slow
    clock: the tables are those of a build that logged nothing, the same
    inputs are a hit (the very same object, built once, one line), and a new
    ledger generation builds again."""
    params = D.SimParams()
    want = _board(monkeypatch, world, feed_frame, 0.0)
    assert _slow_lines(logs) == []
    built = []
    real = props.prepare_feed
    monkeypatch.setattr(props, "prepare_feed", lambda *a, **k: built.append(1) or real(*a, **k))
    monkeypatch.setattr(store, "_build_clock", Clock(400.0))
    served = types.SimpleNamespace(frame=world.proj)

    def feed(generation):
        return live.FeedLive(state="ok", value=feed_frame, asof=NOON.astimezone(UTC), per_game={},
                             generation=generation)
    first = store.feed_tables(DAY, feed(3), served, params, world.roster)
    again = store.feed_tables(DAY, feed(3), served, params, world.roster)
    assert type(first) is props.FeedTables and again is first and len(built) == 1
    assert set(first.tables) == set(want.tables) == set(first.metas) == set(want.metas)
    assert first.report == want.report and not first.failed and not want.failed
    for k, table in want.tables.items():
        pd.testing.assert_frame_equal(first.tables[k], table)
        assert set(first.metas[k]) == set(want.metas[k])
        for field, value in want.metas[k].items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(first.metas[k][field], value)
            else:
                assert first.metas[k][field] == value, (k, field)
    newer = store.feed_tables(DAY, feed(4), served, params, world.roster)
    assert newer is not first and len(built) == 2
    assert [LINE.fullmatch(x).group(1) for x in _slow_lines(logs)] == ["board", "board"]
