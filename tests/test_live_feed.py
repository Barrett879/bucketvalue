"""bucketlib.live's side of the licensed odds feed (The Odds API): the key
reader (design 2.4.3, build amendment B7), the feed anchor and its ledger
(2.4.1, 2.4.2), the walk, the kick and the view (2.4.4), with the credit
governor wired per build amendments B1 to B4, B11 (a first pull the ledger
cannot tell from a re-buy is floor-checked) and B12 (a game whose pulls were
all billed failures is retried once its backoff allows), and the refresher
(2.4.6, M6; B15, today's order written before today's files).
The anchor check through store.feed_start is a later step's (B10: tested here
through feed_kick directly).

The refresher never runs on its own here: conftest sets
BUCKETVALUE_ODDS_REFRESHER "0" for the session. refresher_tick is called
directly with the synchronous pool and a pinned now; the loop is driven on its
own stop Event with a tiny tick (monkeypatch); ensure_refresher is turned on
through monkeypatch, and each such test stops and joins its thread. Every
test's teardown checks that no refresher thread is left alive.

The key reader, read on every call:
  - the environment key (BUCKETVALUE_ODDS_API_KEY) first, then a key file
    (BUCKETVALUE_ODDS_API_KEY_FILE), with "~" expanded (m12), and no default
    path, ever;
  - `environ=None` means the process environment, and a mapping given is read
    as it is, so an EMPTY test mapping is no key (m11);
  - only a regular file is read, without blocking (B7): a FIFO, a device or a
    folder gives None at once;
  - a key file that cannot be read gives None and one WARNING with a fixed
    code (the exception's TYPE, or "not a regular file"): never the path, the
    message or the key.

The feed tests run the real live functions with only the vendor and the
clocks replaced: a fake http.vendor_get (installed on bucketlib.http, which
live looks up at call time) answers from the hand-built fixtures in
tests/fixtures/oddsapi/ and records every call; live.now_utc is pinned and
live.time faked as in tests/test_live.py; each test has a fresh ledger and a
synchronous pool (SyncPool), so no task outlives its test. The tests that
need calls in flight use a real two-worker pool, shut down before they end
(where a test must order two pulls, the vendor holds one call on an Event,
never a sleep), or a held pool that the test releases. The date is the
serve fixture's 2026-03-12, a fixed past date. The fake keys are built at
run time. Nothing here reaches a network.
"""
from __future__ import annotations

import ast
import concurrent.futures as cf
import copy
import datetime as dt
import json
import logging
import os
import re
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import http as H
from bucketlib import live
from bucketlib import oddsbudget as OB
from bucketlib.sources import oddsapi as O
from bucketlib.util import ET, UTC

KEY = "K" * 40          # a fake key, built at run time
OTHER = "J" * 40        # a second fake key, to tell the two sources apart
UNREADABLE = "oddsapi: key file unreadable"


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.DEBUG, logger="bucketvalue")
    return caplog


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A fake home folder, and a working directory with nothing in it, so a
    "~" path that was NOT expanded cannot be found relative to the cwd."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return h


def _key_file(folder, data: bytes, name: str = "odds_api_key"):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(data)
    p.chmod(0o600)
    return p


def _assert_no_key_logged(caplog, *keys):
    for k in keys:
        assert k not in caplog.text
        for r in caplog.records:
            assert k not in r.getMessage() and k not in repr(r.args)


def test_the_two_settings_have_the_designs_names():
    assert live.KEY_ENV == "BUCKETVALUE_ODDS_API_KEY"
    assert live.KEY_FILE_ENV == "BUCKETVALUE_ODDS_API_KEY_FILE"


def test_the_environment_key_wins_over_the_key_file(tmp_path, logs):
    path = _key_file(tmp_path, (OTHER + "\n").encode())
    env = {live.KEY_ENV: KEY, live.KEY_FILE_ENV: str(path)}
    assert live._read_key(env) == KEY
    assert live._read_key({live.KEY_ENV: "  " + KEY + " \n", live.KEY_FILE_ENV: str(path)}) == KEY
    # A blank environment key falls through to the file.
    assert live._read_key({live.KEY_ENV: "   ", live.KEY_FILE_ENV: str(path)}) == OTHER
    assert live._read_key({live.KEY_ENV: "", live.KEY_FILE_ENV: str(path)}) == OTHER
    assert logs.records == []
    _assert_no_key_logged(logs, KEY, OTHER)


def test_an_empty_mapping_is_no_key_even_with_a_key_in_the_process_environment(
        monkeypatch, tmp_path, logs):
    """m11: `environ or os.environ` would read the real environment when a
    test passes {}. A mapping is read as it is; only None means os.environ."""
    path = _key_file(tmp_path, (OTHER + "\n").encode())
    monkeypatch.setenv(live.KEY_ENV, KEY)
    monkeypatch.setenv(live.KEY_FILE_ENV, str(path))
    assert live._read_key({}) is None
    assert live._read_key({"UNRELATED": "1"}) is None
    assert live._read_key() == KEY
    assert live._read_key(None) == KEY
    monkeypatch.delenv(live.KEY_ENV)
    assert live._read_key() == OTHER          # the process environment's key file
    assert live._read_key({}) is None
    assert logs.records == []


def test_a_tilde_in_the_key_file_path_is_expanded(home, logs):
    """m12: a value set without a shell (a launchd plist, a .env line) keeps
    a literal "~", which open() alone would look for under the cwd."""
    _key_file(home / ".config" / "bucketvalue", (KEY + "\n").encode())
    assert live._read_key({live.KEY_FILE_ENV: "~/.config/bucketvalue/odds_api_key"}) == KEY
    assert live._read_key({live.KEY_FILE_ENV: "  ~/.config/bucketvalue/odds_api_key \n"}) == KEY
    assert logs.records == []
    _assert_no_key_logged(logs, KEY)


def test_the_key_file_gives_its_first_line_stripped(tmp_path):
    for data in ((KEY + "\n").encode(), (KEY + "\r\n").encode(), ("\n  " + KEY + "  \nsecond\n").encode(),
                 (KEY + "\n" + "X" * 500).encode(), KEY.encode()):
        assert live._read_key({live.KEY_FILE_ENV: str(_key_file(tmp_path, data))}) == KEY, data[:60]


def test_no_key_file_setting_means_no_key_and_no_default_path(home, logs):
    """The conventional place (design 4.3) is never read unless it is named."""
    _key_file(home / ".config" / "bucketvalue", (KEY + "\n").encode())
    assert live._read_key({}) is None
    assert live._read_key({live.KEY_FILE_ENV: ""}) is None
    assert live._read_key({live.KEY_FILE_ENV: "   "}) is None
    assert live._read_key({live.KEY_ENV: "  ", live.KEY_FILE_ENV: " \n"}) is None
    assert logs.records == []


def _missing(tmp_path):
    return tmp_path / "nowhere" / "odds_api_key"


def _a_folder(tmp_path):
    d = tmp_path / "a_folder"
    d.mkdir()
    return d


def _not_ascii(tmp_path):
    return _key_file(tmp_path, KEY.encode() + bytes([0xC3, 0xA9]) + b"\n")


def _empty(tmp_path):
    return _key_file(tmp_path, b"")


def _blank(tmp_path):
    return _key_file(tmp_path, b"  \n\n \t\n")


@pytest.mark.parametrize("make, kind", [
    (_missing, "FileNotFoundError"),
    (_a_folder, "not a regular file"),           # B7: refused by fstat, never opened as a file
    (_not_ascii, "UnicodeDecodeError"),
    (_empty, "IndexError"),
    (_blank, "IndexError"),
])
def test_an_unreadable_key_file_gives_none_and_logs_only_the_exception_type(
        tmp_path, logs, make, kind):
    path = make(tmp_path)
    assert live._read_key({live.KEY_FILE_ENV: str(path)}) is None
    recs = [r for r in logs.records if r.name == "bucketvalue"]
    assert [r.getMessage() for r in recs] == [f"{UNREADABLE} ({kind})"]
    assert recs[0].levelno == logging.WARNING
    assert recs[0].exc_info is None and recs[0].stack_info is None      # no traceback
    assert str(path) not in logs.text and "odds_api_key" not in logs.text and "a_folder" not in logs.text
    assert "Errno" not in logs.text and "codec" not in logs.text and "position" not in logs.text
    _assert_no_key_logged(logs, KEY)


def test_the_key_is_read_again_on_every_call(monkeypatch, tmp_path):
    """RC K3, G1: never cached, so a rotated key takes effect without a
    restart, and a deleted one stops the feed."""
    monkeypatch.setenv(live.KEY_ENV, KEY)
    assert live._read_key() == KEY
    monkeypatch.setenv(live.KEY_ENV, OTHER)
    assert live._read_key() == OTHER
    monkeypatch.delenv(live.KEY_ENV)
    assert live._read_key() is None
    path = _key_file(tmp_path, (KEY + "\n").encode())
    env = {live.KEY_FILE_ENV: str(path)}
    assert live._read_key(env) == KEY
    path.write_bytes((OTHER + "\n").encode())
    assert live._read_key(env) == OTHER
    path.unlink()
    assert live._read_key(env) is None


def test_the_conftest_block_leaves_no_key_in_this_process():
    """Every test starts with no vendor key at all (tests/conftest.py)."""
    assert not os.environ.get(live.KEY_ENV) and not os.environ.get(live.KEY_FILE_ENV)
    assert live._read_key() is None


# -- B7: the key reader never blocks ------------------------------------------
def test_a_fifo_key_file_returns_none_at_once_and_never_blocks(tmp_path, logs):
    """B7: open() of a FIFO for reading waits for a writer, so a FIFO named by
    BUCKETVALUE_ODDS_API_KEY_FILE would hang a page view for good. The reader
    opens without blocking and refuses what fstat says is not a regular file.
    The call runs in a thread joined with a 2 s timeout, so a regression fails
    here instead of hanging the suite, and the FIFO is unblocked in a finally,
    so no thread is left behind."""
    fifo = tmp_path / "odds_api_key"
    os.mkfifo(fifo)
    out: dict = {}
    t = threading.Thread(target=lambda: out.setdefault("key", live._read_key({live.KEY_FILE_ENV: str(fifo)})),
                         daemon=True)
    t.start()
    try:
        t.join(2.0)
        hung = t.is_alive()
    finally:
        if t.is_alive():                       # a regression: let the blocked open() go
            try:
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass
            t.join(2.0)
    assert not hung, "reading a FIFO key file blocked"
    assert out == {"key": None}
    recs = [r for r in logs.records if r.name == "bucketvalue"]
    assert [r.getMessage() for r in recs] == [f"{UNREADABLE} (not a regular file)"]
    assert str(fifo) not in logs.text and "odds_api_key" not in logs.text
    _assert_no_key_logged(logs, KEY)


def test_a_device_or_a_folder_is_refused_and_a_regular_file_still_read(tmp_path, logs):
    """B7: a character device (/dev/null, as /dev/stdin often is) and a folder
    are not regular files: None, one fixed-code WARNING each, nothing read. A
    regular file, and a symlink to one, still give the key."""
    folder = tmp_path / "keys"
    folder.mkdir()
    for path in ("/dev/null", str(folder)):
        assert live._read_key({live.KEY_FILE_ENV: path}) is None, path
    msgs = [r.getMessage() for r in logs.records if r.name == "bucketvalue"]
    assert msgs == [f"{UNREADABLE} (not a regular file)"] * 2
    assert "/dev/null" not in logs.text and str(folder) not in logs.text
    real = _key_file(tmp_path / "real", (KEY + "\n").encode())
    link = tmp_path / "link"
    link.symlink_to(real)
    assert live._read_key({live.KEY_FILE_ENV: str(real)}) == KEY
    assert live._read_key({live.KEY_FILE_ENV: str(link)}) == KEY
    assert len(logs.records) == 2
    _assert_no_key_logged(logs, KEY)


def test_the_key_reader_reads_at_most_its_cap_from_a_regular_file(tmp_path):
    """B7: at most KEY_READ_BYTES of the file are read: a first line longer
    than that is cut there, never read on."""
    assert live.KEY_READ_BYTES == 200
    long = _key_file(tmp_path, b"A" * 500 + b"\n")
    assert live._read_key({live.KEY_FILE_ENV: str(long)}) == "A" * 200


def test_the_module_docstring_says_which_file_live_reads():
    """B7: live.py's module docstring stays true: it writes nothing to disk
    and reads one local file, the key file, only when the setting names it."""
    doc = " ".join(live.__doc__.split())
    assert "Nothing here writes to disk" in doc and "touches disk" not in doc
    assert "BUCKETVALUE_ODDS_API_KEY_FILE names one" in doc and "never on Cloud" in doc


# -- The feed: fixtures -----------------------------------------------------------
FIX = Path(__file__).parent / "fixtures" / "oddsapi"
DATE = dt.date(2026, 3, 12)                              # the serve fixture's date (EDT, UTC-4)
PHI_DET, DEN_SAS = "0022500953", "0022500958"            # 7:00 PM and 9:00 PM ET
PHI_EID, DEN_EID = "c4f1a9e2b7d3058f6a1e9c2b4d7f0953", "e7b2d9f4a1c6083e5b9d2f7a4c1e0958"
PHI_TIP = dt.datetime(2026, 3, 12, 23, 0, tzinfo=UTC)
DEN_TIP = dt.datetime(2026, 3, 13, 1, 0, tzinfo=UTC)
FIVE = O.read_settings({}).books                         # the five, in the settings' order
PTS = ("player_points",)
MARK = "vendor-said-xyzzy"                               # vendor text that must never surface


def et(hh: int, mm: int = 0, ss: int = 0, day: int = 12) -> dt.datetime:
    return dt.datetime(2026, 3, day, hh, mm, ss, tzinfo=ET)


MORNING = et(8, 55)
MORNING_UTC = MORNING.astimezone(UTC)
MINUTE = dt.timedelta(minutes=1)

TODAY = [{"game_id": PHI_DET, "game_date": "2026-03-12", "tip_utc": "2026-03-12T23:00:00+00:00",
          "tbd": False, "away": "PHI", "home": "DET"},
         {"game_id": DEN_SAS, "game_date": "2026-03-12", "tip_utc": "2026-03-13T01:00:00+00:00",
          "tbd": False, "away": "DEN", "home": "SAS"}]


def _horizon() -> list[dict]:
    """The work order's horizon as build amendment B2 has it: the period's
    projected games from its opening day (March's window opens on Feb 28) to
    the horizon end (08:00 ET on Apr 1), tipped ones too. Seven games a day
    at 7:30 PM ET, ids invented; on Mar 12 two of them are the real PHI at
    DET and DEN at SAS. 224 games, 91 of them dated through Mar 12."""
    out, d = [], dt.date(2026, 2, 28)
    while d <= dt.date(2026, 3, 31):
        tip = dt.datetime.combine(d, dt.time(19, 30), tzinfo=ET).astimezone(UTC).isoformat()
        rows = [{"game_id": f"h{d:%m%d}{i}", "game_date": d.isoformat(), "tip_utc": tip, "tbd": False}
                for i in range(7)]
        if d == DATE:
            rows[:2] = [{k: g[k] for k in ("game_id", "game_date", "tip_utc", "tbd")} for g in TODAY]
        out += rows
        d += dt.timedelta(days=1)
    return out


HORIZON = _horizon()
UNPUBLISHED = {g["game_id"] for g in HORIZON if g["game_date"] == "2026-03-12"} - {PHI_DET, DEN_SAS}


def work_order(games=(PHI_DET, DEN_SAS), date="2026-03-12") -> dict:
    """A work order as design 2.4.5 writes it (plain data only)."""
    return {"date": date, "written_at": "2026-03-12T12:50:00+00:00",
            "games": [dict(g) for g in TODAY if g["game_id"] in games],
            "horizon": [dict(g) for g in HORIZON], "serve_files": {}, "busiest": 236}


def _load(name: str):
    return json.loads((FIX / name).read_text())


@dataclass
class Call:
    what: str                  # "events" or "odds"
    event_id: str | None
    books: tuple
    markets: tuple
    keyed: bool                # the key in params["apiKey"] and secrets, and nowhere else


class Vendor:
    """http.vendor_get's stand-in. It answers /events from
    events_constructed.json and a game's odds from its constructed payload,
    cut to the books and markets asked for, as the vendor does; it bills the
    unique markets it returns (an answer with no bookmaker entry is free) and
    sends the x-requests-* headers; it records every call, never the key.
    `script` holds answers to give first, in order: an exception to raise,
    or a callable (call, vendor) -> VendorResponse. `drop` names books to
    leave out of odds answers; `sleep` delays every odds answer."""

    def __init__(self, remaining=350, used=150):
        self.remaining, self.used = remaining, used
        self.calls: list[Call] = []
        self.events = _load("events_constructed.json")
        self.odds = {PHI_EID: _load("event_odds_phi_det_constructed.json"),
                     DEN_EID: _load("event_odds_den_sas_constructed.json")}
        self.script: dict = {"events": [], "odds": []}
        self.drop: set = set()
        self.sleep = 0.0
        self.lock = threading.Lock()
        self.odds_started = threading.Event()

    def __call__(self, host, path, params, *, timeout, secrets=()):
        assert host == O.HOST and timeout == live.ODDS_TIMEOUT
        rest = dict(params)
        keyed = (rest.pop("apiKey", None) == KEY and tuple(secrets) == (KEY,)
                 and KEY not in path and KEY not in repr(rest))
        if path == O.EVENTS_PATH:
            call = Call("events", None, (), (), keyed)
        else:
            call = Call("odds", path.split("/")[-2], tuple(rest["bookmakers"].split(",")),
                        tuple(rest["markets"].split(",")), keyed)
        with self.lock:
            self.calls.append(call)
            scripted = self.script[call.what].pop(0) if self.script[call.what] else None
        if call.what == "odds":
            self.odds_started.set()
            if self.sleep:
                time.sleep(self.sleep)
        if isinstance(scripted, BaseException):
            raise scripted
        if scripted is not None:
            return scripted(call, self)
        return self.answer(call)

    def answer(self, call: Call) -> H.VendorResponse:
        if call.what == "events":
            return self.response(200, copy.deepcopy(self.events), last=0)
        body = copy.deepcopy(self.odds[call.event_id])
        body["bookmakers"] = [dict(b, markets=[m for m in b["markets"] if m["key"] in call.markets])
                              for b in body["bookmakers"]
                              if b["key"] in call.books and b["key"] not in self.drop]
        body["bookmakers"] = [b for b in body["bookmakers"] if b["markets"]]
        return self.response(200, body, last=len({m["key"] for b in body["bookmakers"]
                                                  for m in b["markets"]}))

    def response(self, status, body, *, last, snippet="") -> H.VendorResponse:
        with self.lock:
            self.remaining -= last
            self.used += last
            headers = {"x-requests-remaining": str(self.remaining), "x-requests-used": str(self.used),
                       "x-requests-last": str(last)}
        return H.VendorResponse(status=status, headers=headers, body=body, snippet=snippet, nbytes=99)

    def odds_calls(self, event_id=None) -> list[Call]:
        return [c for c in self.calls if c.what == "odds" and event_id in (None, c.event_id)]

    def events_calls(self) -> list[Call]:
        return [c for c in self.calls if c.what == "events"]


def answer_with(status, *, last=0, body=None, snippet=""):
    return lambda call, vendor: vendor.response(status, body, last=last, snippet=snippet)


class SyncPool:
    """A synchronous stand-in for ThreadPoolExecutor's submit (M10): the task
    runs at once and its Future comes back finished, so nothing outlives the
    test or reaches the next test's fakes."""

    def __init__(self):
        self.names: list[str] = []

    def submit(self, fn, *args, **kwargs):
        self.names.append(fn.__name__)
        fut = cf.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:                       # what a worker thread does
            fut.set_exception(e)
        return fut


class HeldPool(SyncPool):
    """Holds every task until the test releases it: work in flight with no
    real seconds (design 8.2's held pool)."""

    def __init__(self):
        super().__init__()
        self.held: list = []

    def submit(self, fn, *args, **kwargs):
        self.names.append(fn.__name__)
        fut = cf.Future()
        self.held.append((fut, fn, args, kwargs))
        return fut

    def release(self, n=None) -> int:
        """Run the held tasks, first in first out (`n` of them, or all)."""
        ran = 0
        while self.held and (n is None or ran < n):
            self._run(self.held.pop(0))
            ran += 1
        return ran

    def run(self, name: str, nth: int = 0) -> None:
        """Run one held task by its function's name (the nth such)."""
        picks = [i for i, (_f, fn, _a, _k) in enumerate(self.held) if fn.__name__ == name]
        self._run(self.held.pop(picks[nth]))

    @staticmethod
    def _run(task) -> None:
        fut, fn, args, kwargs = task
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:
            fut.set_exception(e)


class CountingPool:
    """A real pool's submit, counted by task name."""

    def __init__(self, inner):
        self.inner, self.names, self.lock = inner, [], threading.Lock()

    def submit(self, fn, *args, **kwargs):
        with self.lock:
            self.names.append(fn.__name__)
        return self.inner.submit(fn, *args, **kwargs)


@pytest.fixture
def feed(monkeypatch, caplog):
    """A fresh ledger born at 08:55 ET on Mar 12, the fake vendor, live's
    clocks pinned (now_utc) and faked (time, as tests/test_live.py:55-56), a
    synchronous pool, and the anchor reset before and after."""
    live.reset_feed_anchor_for_tests()
    clock = [1_000_000.0]
    monkeypatch.setattr(live, "time", types.SimpleNamespace(time=lambda: clock[0],
                                                            monotonic=lambda: clock[0]))
    f = types.SimpleNamespace(at=[MORNING_UTC], clock=clock)
    monkeypatch.setattr(live, "now_utc", lambda: f.at[0])
    f.vendor = Vendor()
    monkeypatch.setattr(H, "vendor_get", f.vendor)
    f.mem = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    f.pool = SyncPool()
    f.env = {live.KEY_ENV: KEY}
    f.work = work_order()
    caplog.set_level(logging.DEBUG, logger="bucketvalue")
    f.logs = caplog

    def kick(at=None, *, work=None, env=None, mem=None, pool=None):
        if at is not None:
            f.at[0] = at.astimezone(UTC)
        live.feed_kick(f.work if work is None else work, mem=f.mem if mem is None else mem,
                       environ=f.env if env is None else env, pool=f.pool if pool is None else pool)

    def view(date=DATE, *, mem=None, wait_s=0):
        return live.feed_view(date, mem=f.mem if mem is None else mem, wait_s=wait_s)

    f.kick, f.view = kick, view
    yield f
    live.reset_feed_anchor_for_tests()


@pytest.fixture
def fresh_anchor():
    """No anchor before the test, and none after it, whatever the test did."""
    live.reset_feed_anchor_for_tests()
    yield
    live.reset_feed_anchor_for_tests()


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "bucketvalue"]


def _plain(mem: dict) -> dict:
    """The ledger's fields as plain comparable copies (the lock left out)."""
    out = {}
    for k, v in mem.items():
        if k == "lock":
            continue
        out[k] = dict(v) if isinstance(v, dict) else sorted(v) if isinstance(v, set) else \
            list(v) if isinstance(v, list) else v
    return out


def _text(obj, depth=0) -> str:
    """Everything an object holds, as text: dicts, lists, tuples, sets, frames
    (every cell and attr), module dicts; for the key search."""
    if depth > 8:
        return ""
    if isinstance(obj, pd.DataFrame):
        return obj.to_string() + repr(obj.attrs) + repr(list(obj.columns))
    if isinstance(obj, dict):
        return " ".join(_text(k, depth + 1) + " " + _text(v, depth + 1) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return " ".join(_text(v, depth + 1) for v in obj)
    if isinstance(obj, types.ModuleType):
        return _text(vars(obj), depth + 1)
    return repr(obj)


def _rows(fl, game_id):
    v = fl.value
    return v[v["game_id"] == game_id].sort_values(["book", "market", "name"]).reset_index(drop=True)


# -- No key, the kill switch, the key per call -------------------------------------
def test_no_key_means_no_call_and_no_memory_change_and_an_empty_mapping_is_no_key(feed, monkeypatch):
    """m11: with a key in the PROCESS environment, a mapping given is still
    read as it is, so {} is no key: no call, no task, no ledger change and no
    anchor made."""
    monkeypatch.setenv(live.KEY_ENV, KEY)
    before = _plain(feed.mem)
    for env in ({}, {"UNRELATED": "1"}, {live.KEY_ENV: "   "}):
        feed.kick(env=env)
        assert live.prop_lines(feed.work, mem=feed.mem, environ=env, pool=feed.pool).state == "off"
    assert feed.vendor.calls == [] and feed.pool.names == []
    assert _plain(feed.mem) == before and feed.mem["work"] is None
    assert live.ANCHOR_NAME not in sys.modules
    monkeypatch.delenv(live.KEY_ENV)
    live.feed_kick(feed.work, environ=None, pool=feed.pool)    # no key anywhere: not even the anchor
    assert live.prop_lines(feed.work, pool=feed.pool).state == "off"
    assert live.ANCHOR_NAME not in sys.modules and feed.vendor.calls == []


def test_a_tilde_key_file_path_is_expanded_for_the_walk_m12(feed, monkeypatch, tmp_path):
    home = tmp_path / "home"
    _key_file(home / ".config" / "bucketvalue", (KEY + "\n").encode())
    monkeypatch.setenv("HOME", str(home))
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    feed.kick(env={live.KEY_FILE_ENV: "~/.config/bucketvalue/odds_api_key"})
    assert [c.what for c in feed.vendor.calls] == ["events", "odds", "odds"]
    assert all(c.keyed for c in feed.vendor.calls)


@pytest.mark.parametrize("value", ["0", "off", "maybe"])
def test_the_kill_switch_stops_every_call_the_free_one_too(feed, value):
    """"0" stops every vendor call, events included; an unreadable value is
    OFF too (build amendment B6)."""
    env = {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_ENABLED": value}
    feed.kick(env=env)
    assert live.prop_lines(feed.work, mem=feed.mem, environ=env, pool=feed.pool).state == "off"
    assert feed.vendor.calls == [] and feed.mem["work"] is None and feed.pool.names == []


@pytest.mark.parametrize("change", ["switch", "key"])
def test_the_switch_and_the_key_are_read_again_before_each_paid_call(feed, change):
    """3.6 item 8: a switch turned off, or a key taken away, between the walk's
    free call and its pulls stops the pulls; nothing is left pending."""
    env = dict(feed.env)

    def events_then_change(call, vendor):
        if change == "switch":
            env["BUCKETVALUE_ODDS_ENABLED"] = "0"
        else:
            del env[live.KEY_ENV]
        return vendor.answer(call)
    feed.vendor.script["events"] = [events_then_change]
    feed.kick(env=env)
    assert [c.what for c in feed.vendor.calls] == ["events"]
    assert feed.mem["budget"]["pending"] == 0 and not feed.mem["queued"] and not feed.mem["futures"]


# -- Events: free, cached, matched -------------------------------------------------
def test_the_free_events_call_learns_the_balance_and_is_cached_ten_minutes(feed):
    """The balance is known before any credit is spent (RC B2); the free call
    is cached ODDS_EVENTS_TTL (10 minutes) and spends nothing."""
    assert live.ODDS_EVENTS_TTL == 600
    feed.kick(at=et(8, 0))                                   # before 08:50: nothing is due
    assert [c.what for c in feed.vendor.calls] == ["events"] and feed.vendor.calls[0].keyed
    b = feed.mem["budget"]
    assert (b["remaining"], b["used"], b["last"], b["seen_at"]) == (350, 150, 0, et(8, 0))
    for m in (1, 5, 9):
        feed.kick(at=et(8, m))
    assert len(feed.vendor.events_calls()) == 1
    feed.kick(at=et(8, 10))
    assert len(feed.vendor.events_calls()) == 2
    assert feed.vendor.odds_calls() == [] and (feed.vendor.used, b["used"]) == (150, 150)
    got = feed.view()
    assert (got.state, got.value.empty, got.loading) == ("idle", True, 0)
    assert {gid: g["reason"] for gid, g in got.per_game.items()} == {
        PHI_DET: "not_yet", DEN_SAS: "not_yet", **{gid: "not_published" for gid in UNPUBLISHED}}
    assert got.report["unmatched_events"] == 7 and got.report["unknown_team"] == []


def test_the_morning_walk_buys_each_of_todays_games_once_in_tip_order(feed):
    feed.kick()
    v = feed.vendor
    assert [c.what for c in v.calls] == ["events", "odds", "odds"]
    assert [c.event_id for c in v.odds_calls()] == [PHI_EID, DEN_EID]         # 7 PM, then 9 PM
    assert all(c.books == FIVE and c.markets == PTS and c.keyed for c in v.odds_calls())
    got = feed.view()
    assert (got.state, got.loading, got.asof, got.generation) == ("ok", 0, MORNING_UTC, 2)
    assert list(got.value.columns) == O.FEED_COLUMNS
    assert set(got.value["game_id"]) == {PHI_DET, DEN_SAS} and set(got.value["game_date"]) == {"2026-03-12"}
    assert got.value.groupby("game_id").size().to_dict() == {PHI_DET: 25, DEN_SAS: 20}
    tips = got.value.groupby("game_id")["tip_utc"].first()
    assert (tips[PHI_DET], tips[DEN_SAS]) == (pd.Timestamp(PHI_TIP), pd.Timestamp(DEN_TIP))
    phi, den = got.per_game[PHI_DET], got.per_game[DEN_SAS]
    assert phi == {"asof": MORNING_UTC, "books": {b: MORNING_UTC for b in O.BOOKS}, "pretip": False,
                   "pp_missing": False, "reason": None}
    assert set(den["books"]) == set(O.BOOKS) - {"prizepicks"} and den["pp_missing"]
    b = feed.mem["budget"]
    assert (b["remaining"], b["used"], b["last"], b["pending"], b["mode"]) == (348, 152, 1, 0, "full")
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["first_at"], e["last_at"], e["fails"]) == (1, MORNING_UTC, MORNING_UTC, 0)
    assert feed.mem["calls"] == [MORNING_UTC, MORNING_UTC] and feed.mem["streak"] == 0
    assert "oddsapi: walk pulled 2 game(s), cost 2, remaining 348" in _messages(feed.logs)
    assert feed.pool.names == ["_walk", "_pull_game", "_pull_game"]
    feed.kick(at=et(9, 0))                                   # nothing more is due yet
    assert len(v.odds_calls()) == 2


# -- Single flight (M8) ---------------------------------------------------------------
def test_two_kicks_together_buy_each_game_once_and_both_views_see_the_lines(feed):
    """M8 and design 1.2's one walk at a time: two visitors kick together on
    a real two-worker pool while each paid call takes 0.2 s, and a third
    kicks while the calls are in flight. One events call, one paid call per
    game, every game queued once, and both views see both games' lines."""
    feed.vendor.sleep = 0.2
    inner = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-oddsapi")
    pool = CountingPool(inner)
    seen: list = [None, None]
    start = threading.Barrier(2)

    def visitor(i):
        start.wait(timeout=2)
        seen[i] = live.prop_lines(feed.work, mem=feed.mem, environ=feed.env, pool=pool, wait_s=3.0)
    threads = [threading.Thread(target=visitor, args=(i,), daemon=True) for i in range(2)]
    try:
        for t in threads:
            t.start()
        assert feed.vendor.odds_started.wait(2.0)
        live.feed_kick(feed.work, mem=feed.mem, environ=feed.env, pool=pool)   # while in flight
        for t in threads:
            t.join(5.0)
    finally:
        inner.shutdown(wait=True)
    assert not any(t.is_alive() for t in threads)
    assert len(feed.vendor.events_calls()) == 1
    assert sorted(c.event_id for c in feed.vendor.odds_calls()) == sorted([PHI_EID, DEN_EID])
    assert pool.names.count("_pull_game") == 2                    # never queued again
    for got in seen:
        assert got.state == "ok" and got.loading == 0
        assert set(got.value["game_id"]) == {PHI_DET, DEN_SAS}
    assert feed.mem["queued"] == set() and feed.mem["futures"] == {} and feed.mem["budget"]["pending"] == 0


def test_two_pulls_of_one_game_at_once_make_one_call_and_a_later_one_none(feed):
    """MUTATION, single flight (design 8.2): two tasks for the same game run
    together on a real two-worker pool while the paid call takes 0.2 s. The
    second finds the game's lock taken and returns at once, without waiting
    and without a call; a third, after both, is refused by the fresh due_kind
    (the game has had its morning pull). A no-op event lock buys the game
    twice; a pull without the re-check buys it a third time."""
    feed.kick(at=et(8, 0))                                   # events known; nothing due before 08:50
    assert feed.vendor.odds_calls() == []
    feed.at[0] = MORNING_UTC
    feed.vendor.sleep = 0.2
    pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-oddsapi")
    try:
        both = [pool.submit(live._pull_game, feed.mem, PHI_EID, feed.env) for _ in range(2)]
        done, _ = cf.wait(both, timeout=5.0)
        later = pool.submit(live._pull_game, feed.mem, PHI_EID, feed.env).result(timeout=5.0)
    finally:
        pool.shutdown(wait=True)
    assert len(done) == 2
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID]
    assert sorted((f.result() for f in both), key=lambda r: r is None) == [{"billed": True, "cost": 1}, None]
    assert later is None
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], feed.mem["budget"]["pending"], feed.mem["budget"]["remaining"]) == (1, 0, 349)


# -- Pulls in flight see each other's reservation (review R2) ---------------------------
def _overlapped(feed, env, held_eid: str, other_eid: str):
    """Two pulls in flight at once on a real two-worker pool: the first
    game's paid call is held at the vendor until the second game's pull has
    taken its ticket, or been refused one, and returned. Both results."""
    go = threading.Event()

    def held(call, vendor):
        go.wait(2.0)
        return vendor.answer(call)
    feed.vendor.script["odds"] = [held]
    feed.vendor.odds_started.clear()
    pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-oddsapi")
    try:
        first = pool.submit(live._pull_game, feed.mem, held_eid, env)
        assert feed.vendor.odds_started.wait(2.0)             # its ticket is out, its call in flight
        second = pool.submit(live._pull_game, feed.mem, other_eid, env).result(timeout=2.0)
        go.set()
        return first.result(timeout=2.0), second
    finally:
        go.set()
        pool.shutdown(wait=True)


def test_a_pull_in_flight_holds_its_cost_so_another_cannot_cross_the_reserve_r2(feed):
    """Review R2 (3.3, 3.6 item 4): a ticket reserves its cost in `pending`
    until its call settles, so pulls in flight at once see each other. One
    credit above a reserve of 300 (301 left, 0 used): PHI at DET's first pull
    is held at the vendor while DEN at SAS's pull asks for its ticket, which
    is refused, since what PHI's ticket reserved leaves nothing above the
    reserve. One paid call, and the balance never under the reserve. Without
    the reservation both are bought: 299 left, under the reserve, and 2
    used, over the day's ceiling of 1."""
    env = {**feed.env, "BUCKETVALUE_ODDS_RESERVE": "300"}
    feed.vendor.remaining, feed.vendor.used = 301, 0
    feed.kick(at=et(8, 0), env=env)                          # the balance known; nothing due before 08:50
    feed.at[0] = MORNING_UTC
    first, second = _overlapped(feed, env, PHI_EID, DEN_EID)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID]
    assert (first, second) == ({"billed": True, "cost": 1}, None)
    b = feed.mem["budget"]
    assert (b["remaining"], b["used"], b["pending"], feed.vendor.remaining) == (300, 1, 0, 300)
    assert feed.mem["games"][DEN_EID]["pulls"] == 0


def test_a_pull_in_flight_holds_its_cost_so_another_cannot_pass_the_ceiling_r2(feed):
    """R2 at the spending ceiling (3.6 item 6: used + pending + cost): 300 left
    and 410 used make today's ceiling 411, so one more call fits. PHI at
    DET's is held in flight while DEN at SAS asks. The plan is full and the
    reserve far off, so only may_spend's count of what is pending can refuse
    it: one paid call, and `used` ends at the ceiling, never over it."""
    feed.vendor.remaining, feed.vendor.used = 300, 410
    feed.kick(at=et(8, 0))
    feed.at[0] = MORNING_UTC
    with feed.mem["lock"]:
        plan = live._budget(feed.mem, O.read_settings(feed.env), MORNING_UTC).plan
    assert (plan.mode, plan.ceiling, plan.reserve) == ("full", 411, 36)          # the premise
    first, second = _overlapped(feed, feed.env, PHI_EID, DEN_EID)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID]
    assert (first, second) == ({"billed": True, "cost": 1}, None)
    assert (feed.mem["budget"]["used"], feed.mem["budget"]["pending"]) == (411, 0)
    assert "oddsapi: spending ceiling reached for today" in _messages(feed.logs)


def test_a_pull_in_flight_counts_in_the_plan_another_pull_reads_r2(feed):
    """R2 in the plan (3.3: B = remaining - S - pending): 305 left and 150 used
    give P = 2, and after both morning pulls (303 left, 2 pulls made) P is
    still 2, at its margin. With PHI at DET's evening refresh in flight the
    plan counts that cost and P is 1, so DEN at SAS's PrizePicks top-up, due
    at 18:30 by itself, is not due while the refresh runs: one paid call.
    Once the refresh has settled, the top-up is due again."""
    feed.vendor.remaining = 305
    feed.kick()                                              # 08:55: both first pulls
    assert len(feed.vendor.odds_calls()) == 2 and feed.mem["games"][DEN_EID]["pp_missing"]
    feed.at[0] = et(18, 30).astimezone(UTC)
    first, second = _overlapped(feed, feed.env, PHI_EID, DEN_EID)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID, PHI_EID]
    assert (first, second) == ({"billed": True, "cost": 1}, None)
    s = O.read_settings(feed.env)
    with feed.mem["lock"]:
        plan = live._budget(feed.mem, s, feed.at[0]).plan
        slot = live._slot_of(feed.mem["games"][DEN_EID], DEN_EID)
    assert (plan.pulls_per_game, OB.due_kind(slot, plan, s, feed.at[0], "2026-03-12")) == (2, "pp")


# -- The tip guard (part 2) ----------------------------------------------------------
def test_no_paid_call_inside_the_tip_guard_even_when_due_kind_says_first(feed, monkeypatch):
    """MUTATION, tip guard part 2 (part 1 is tests/test_oddsbudget.py): with
    due_kind forced to "first", the walk queues the game and the pull asks
    the governor with the game's tip and the configured guard. At tip minus
    4 minutes there are zero paid calls, and none at the guard's own instant
    or after tip; a minute outside the guard the same forced kind is bought,
    so the zero is the guard's."""
    monkeypatch.setattr(OB, "due_kind", lambda *a, **k: "first")
    work = work_order(games=(PHI_DET,))
    for at in (PHI_TIP - 4 * MINUTE, PHI_TIP - 5 * MINUTE, PHI_TIP, PHI_TIP + 60 * MINUTE):
        feed.kick(at=at, work=work)
    assert feed.vendor.odds_calls() == [] and feed.vendor.events_calls()
    assert feed.mem["budget"]["pending"] == 0 and not feed.mem["queued"] and feed.mem["calls"] == []
    outside = live.new_feed_memory(PHI_TIP - 6 * MINUTE, had_ledger=False)
    feed.kick(at=PHI_TIP - 6 * MINUTE, work=work, mem=outside)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID]


# -- Billing (B2, M7) -------------------------------------------------------------------
# Two more of the slate's real games (tests/fixtures/pipeline/schedule.parquet),
# with events in events_constructed.json; no odds payload is needed for them.
IND_PHX, ORL_WAS = "0022500954", "0022500955"
IND_EID, ORL_EID = "1b8e3f6a9c2d5074e1b8a3f6c9d20954", "9d3a6c1f4b7e2058d9a3c6f1b4e70955"
MORE = [{"game_id": IND_PHX, "game_date": "2026-03-12", "tip_utc": "2026-03-12T23:00:00+00:00",
         "tbd": False, "away": "PHX", "home": "IND"},
        {"game_id": ORL_WAS, "game_date": "2026-03-12", "tip_utc": "2026-03-12T23:00:00+00:00",
         "tbd": False, "away": "WAS", "home": "ORL"}]


def test_billed_junk_all_day_costs_exactly_three_calls(feed):
    """B2, RP case E: every paid call bills 1 credit and answers {} (JSON of
    the wrong shape). At 08:50 four games are due, three of them at 7 PM.
    Their three billed failures in a row trip the day breaker, and the fresh
    due_kind in the fourth game's pull reads it, so the walk that tripped it
    buys nothing more. A view every 5 minutes all day: exactly 3 paid calls,
    one WARNING, and the free events call goes on."""
    feed.vendor.script["odds"] = [answer_with(200, last=1, body={})] * 50
    feed.work["games"] += [dict(g) for g in MORE]
    t = et(8, 50)
    while t <= et(23, 55):
        feed.kick(at=t)
        t += 5 * MINUTE
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, IND_EID, ORL_EID]
    assert feed.mem["games"][DEN_EID]["pulls"] == 0                 # queued at 08:50, never bought
    assert (feed.mem["streak"], feed.mem["breaker_day"], feed.vendor.used) == (3, DATE, 153)
    assert len(feed.vendor.events_calls()) > 10
    breaker = [m for m in _messages(feed.logs) if "billed failures in a row" in m]
    assert breaker == ["oddsapi: 3 billed failures in a row; paid calls stopped until tomorrow "
                       "(parser counters: unknown_team 1)"]
    got = feed.view()
    assert (got.state, got.value.empty) == ("paused", True)
    assert "oddsapi event odds failed: bad_payload unknown_team" in _messages(feed.logs)


def test_a_timeout_counts_as_a_pull_and_backs_the_game_off(feed):
    """3.5: a timeout may have been billed: it counts as a pull (the worst
    case comes off the balance, with no headers to say) and backs the game
    off 90 s, but no answer came back, so it does not grow the streak (build
    amendment B22: the day breaker is for billed junk). Its pulls so far
    brought no lines, so once the 90 s are over it is due again at once
    (build amendment B12), not at the evening refresh; that pull lands, and
    P = 2 is then spent."""
    work = work_order(games=(PHI_DET,))
    feed.vendor.script["odds"] = [H.VendorTransportError("timeout")]
    feed.kick(work=work)
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"], e["fail_until"], e["last_at"]) == (
        1, 1, MORNING_UTC + dt.timedelta(seconds=90), MORNING_UTC)
    b = feed.mem["budget"]
    assert (b["remaining"], b["used"], b["pending"], feed.mem["streak"]) == (349, 151, 0, 0)
    assert "oddsapi event odds failed: timeout" in _messages(feed.logs)
    assert feed.view().state == "down"
    feed.kick(at=MORNING + dt.timedelta(seconds=89), work=work)          # still backing off
    assert len(feed.vendor.odds_calls()) == 1
    feed.kick(at=MORNING + dt.timedelta(seconds=90), work=work)          # B12: due again at once
    assert len(feed.vendor.odds_calls()) == 2
    assert (e["pulls"], e["fails"], e["fail_until"], feed.mem["streak"]) == (2, 0, None, 0)
    assert feed.view().state == "ok"
    for at in (et(18, 29), et(18, 30), et(18, 50)):                     # P = 2 spent: no evening pull
        feed.kick(at=at, work=work)
    assert len(feed.vendor.odds_calls()) == 2


def _timed_odds(feed, monkeypatch) -> list:
    """The pinned clock at each paid (odds) call, in order: the fake vendor
    records the calls, and this their instants."""
    when: list = []

    def timed(host, path, params, **kw):
        if path != O.EVENTS_PATH:
            when.append(feed.at[0])
        return feed.vendor(host, path, params, **kw)
    monkeypatch.setattr(H, "vendor_get", timed)
    return when


# Kicks every 5 minutes from 08:50 to 23:55 ET on Mar 12.
DAY_KICKS = [et(8, 50) + k * 5 * MINUTE for k in range(182)]


def _utc_times(*hms) -> list:
    return [et(*t).astimezone(UTC) for t in hms]


def test_a_timeout_at_0850_is_retried_after_its_backoff_and_a_game_with_lines_keeps_its_spacing_b12(
        feed, monkeypatch):
    """Build amendment B12. Billed failures count as pulls (3.5), so before
    it a timeout on a game's 08:50 pull left it without lines until its
    pre-tip refresh, and Barrett looks around 9 AM. A ledger born at 08:00,
    450 left and 150 used: P = 3 (420 above the reserve of 30 for 140
    games). At 08:50 PHI at DET times out (a pull, a 90 s backoff) while DEN
    at SAS lands lines. PHI at DET's pulls brought no lines, so it is due
    again the moment its backoff is over, 08:51:30, and that pull lands them;
    from there it keeps its spacing: its third and last pull at 6:30 PM, tip
    minus 30 minutes. DEN at SAS has lines, so its refresh that times out at
    2:40 PM is retried at its next spaced time, 8:30 PM, never at its
    backoff's end. Kicks every 5 minutes all day and at both backoffs' ends:
    six paid calls, no game past P, nothing after tip. (The PrizePicks-only
    polls are off here, BUCKETVALUE_ODDS_PP_TOPUP "0", so that every call is
    a full pull: DEN at SAS has no PrizePicks line.)"""
    when = _timed_odds(feed, monkeypatch)
    feed.env = {**feed.env, "BUCKETVALUE_ODDS_PP_TOPUP": "0"}
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.vendor.remaining = 450
    feed.vendor.script["odds"] = [H.VendorTransportError("timeout")]
    feed.kick(at=et(8, 50))
    phi, den = feed.mem["games"][PHI_EID], feed.mem["games"][DEN_EID]
    assert (phi["pulls"], phi["fails"], phi["fail_until"], phi["frame"]) == (
        1, 1, et(8, 51, 30).astimezone(UTC), None)
    assert (den["pulls"], den["fails"], feed.mem["streak"]) == (1, 0, 0) and den["frame"] is not None
    with feed.mem["lock"]:
        plan = live._budget(feed.mem, O.read_settings(feed.env), feed.at[0]).plan
    assert (plan.mode, plan.pulls_per_game, plan.reserve) == ("full", 3, 30)
    got = feed.view()
    assert got.state == "stale" and _rows(got, PHI_DET).empty
    feed.kick(at=et(8, 51, 29))                              # still backing off
    assert len(when) == 2
    feed.kick(at=et(8, 51, 30))                              # due again at once, and it lands
    assert (phi["pulls"], phi["fails"], phi["fail_until"]) == (2, 0, None)
    got = feed.view()
    assert got.state == "ok" and not _rows(got, PHI_DET).empty
    for at in sorted(set(DAY_KICKS[1:]) | {et(14, 41, 30)}):
        if at == et(14, 40):
            feed.vendor.script["odds"] = [H.VendorTransportError("timeout")]
        feed.kick(at=at)
        if at == et(14, 40):
            assert (den["pulls"], den["fails"], den["fail_until"]) == (2, 1, et(14, 41, 30).astimezone(UTC))
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID] * 3
    assert when == _utc_times((8, 50), (8, 50), (8, 51, 30), (14, 40), (18, 30), (20, 30))
    assert [feed.mem["games"][eid]["pulls"] for eid in (PHI_EID, DEN_EID)] == [3, 3]


def test_a_game_that_keeps_failing_never_passes_p_and_the_day_breaker_still_stops_the_day_b12(
        feed, monkeypatch):
    """B12's bounds. P = 2 (350 left, 150 used), a ledger born at 08:00. PHI
    at DET times out at 08:50 and again at its retry at 08:51:30, with DEN at
    SAS's success between them: two pulls, P, and none more all day, though
    it never has lines. DEN at SAS keeps its evening refresh. Then a morning
    on which every call answers billed junk (a timeout grows no streak since
    build amendment B22; an answer that came back, billed and of no use,
    does): PHI at DET, DEN at SAS and PHI at DET's retry are three billed
    failures in a row, so the day breaker stops paid calls, and DEN at SAS's
    retry, due at the same instant, is never made: three paid calls all
    day. (The PrizePicks-only polls are off, as in the test above.)"""
    when = _timed_odds(feed, monkeypatch)
    feed.env = {**feed.env, "BUCKETVALUE_ODDS_PP_TOPUP": "0"}
    timeout = H.VendorTransportError("timeout")
    kicks = sorted(set(DAY_KICKS) | {et(8, 51, 30)})
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.vendor.script["odds"] = [timeout, lambda call, vendor: vendor.answer(call), timeout]
    for at in kicks:
        feed.kick(at=at)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID, PHI_EID, DEN_EID]
    assert when == _utc_times((8, 50), (8, 50), (8, 51, 30), (20, 30))
    phi = feed.mem["games"][PHI_EID]
    assert (phi["pulls"], phi["fails"], phi["frame"], feed.mem["breaker_day"]) == (2, 2, None, None)
    got = feed.view()
    assert got.state == "stale" and _rows(got, PHI_DET).empty and not _rows(got, DEN_SAS).empty
    feed.vendor.remaining, feed.vendor.used = 350, 150
    feed.vendor.script["odds"] = [answer_with(200, last=1, body={})] * 3
    n = len(when)
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    for at in kicks:
        feed.kick(at=at)
    assert [c.event_id for c in feed.vendor.odds_calls()[4:]] == [PHI_EID, DEN_EID, PHI_EID]
    assert when[n:] == _utc_times((8, 50), (8, 50), (8, 51, 30))
    assert (feed.mem["streak"], feed.mem["breaker_day"]) == (3, DATE)
    assert [feed.mem["games"][eid]["pulls"] for eid in (PHI_EID, DEN_EID)] == [2, 1]
    assert ("oddsapi: 3 billed failures in a row; paid calls stopped until tomorrow (parser counters: "
            "unknown_team 1)") in _messages(feed.logs)


def test_a_connection_failure_with_nothing_received_is_no_pull_and_backs_off(feed):
    """3.5: nothing received means nothing billed: no pull, no streak, the
    balance untouched; the game still waits out its 90 s backoff."""
    feed.vendor.script["odds"] = [H.VendorTransportError("connect", 0)]
    work = work_order(games=(PHI_DET,))
    feed.kick(work=work)
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"], feed.mem["streak"], feed.mem["budget"]["remaining"]) == (0, 1, 0, 350)
    feed.kick(at=MORNING + dt.timedelta(seconds=89), work=work)
    assert len(feed.vendor.odds_calls()) == 1
    feed.kick(at=MORNING + dt.timedelta(seconds=90), work=work)
    assert len(feed.vendor.odds_calls()) == 2 and (e["pulls"], e["fails"]) == (1, 0)
    assert "oddsapi event odds failed: server connect" in _messages(feed.logs)


def test_a_parser_exception_leaves_pending_at_zero_and_is_logged_by_type(feed, monkeypatch):
    """M7: an exception between the call and its settling (here the parser's)
    gives `pending` back in the finally, and is logged by its type only; the
    answer came, so the call counts as billed (3.5's safe side)."""
    def broken(*a, **k):
        raise RuntimeError(MARK + " " + KEY)
    monkeypatch.setattr(O, "parse_event_odds", broken)
    feed.kick(work=work_order(games=(PHI_DET,)))
    assert len(feed.vendor.odds_calls()) == 1
    b = feed.mem["budget"]
    assert (b["pending"], b["remaining"], b["used"]) == (0, 349, 151)
    assert feed.mem["queued"] == set() and feed.mem["futures"] == {}
    assert "oddsapi: worker failed (RuntimeError)" in _messages(feed.logs)
    assert MARK not in feed.logs.text and KEY not in feed.logs.text
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"], e["frame"]) == (1, 1, None)


class _Escape(BaseException):
    """What no `except Exception` catches (the test block raises one too)."""


def test_a_task_that_raises_past_every_except_is_logged_by_type_and_taken_off_the_queue_m7(feed):
    feed.vendor.script["odds"] = [_Escape()]
    feed.kick(work=work_order(games=(PHI_DET,)))
    assert feed.mem["budget"]["pending"] == 0
    assert feed.mem["queued"] == set() and feed.mem["futures"] == {}
    assert "oddsapi: worker failed (_Escape)" in _messages(feed.logs)
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"]) == (0, 1)                  # no answer came: not billed
    feed.kick(at=MORNING + 2 * MINUTE, work=work_order(games=(PHI_DET,)))   # queued again, and bought
    assert len(feed.vendor.odds_calls()) == 2 and e["pulls"] == 1


def _raises(*a, **k):
    raise RuntimeError(MARK + " " + KEY)


def test_a_raise_inside_the_settling_still_counts_the_call_and_backs_the_game_off_r1(feed, monkeypatch):
    """Review R1 (3.5: every call that may have been billed counts as a pull;
    6: a worker that raised counts as a failed call). merge_refresh raises
    inside the settling, after the answer came, so the call is billed: one
    pull and one failure, a 90 s backoff, the streak grown, the vendor's
    counters taken once and `pending` back to 0. A kick inside the backoff
    buys nothing, so a game is never bought on every walk. Its pulls brought
    no lines, so once the backoff is over it is due again at once (build
    amendment B12): PHI at DET's retry at 08:57 fails the same way, the third
    billed failure in a row, and the day breaker stops the day (RP case E):
    three calls in all, kicks a minute apart then every 5 minutes all day."""
    monkeypatch.setattr(O, "merge_refresh", _raises)
    for m in range(2):
        feed.kick(at=MORNING + m * MINUTE)                   # 08:55, then 08:56 inside the backoff
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    for eid in (PHI_EID, DEN_EID):
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["fails"], e["fail_until"], e["first_at"], e["last_at"], e["frame"]) == (
            1, 1, MORNING_UTC + dt.timedelta(seconds=90), MORNING_UTC, MORNING_UTC, None), eid
    b = feed.mem["budget"]
    assert (b["pending"], b["remaining"], b["used"], feed.mem["streak"], feed.vendor.used) == (
        0, 348, 152, 2, 152)
    assert feed.mem["queued"] == set() and feed.mem["futures"] == {} and feed.mem["generation"] == 0
    assert _messages(feed.logs).count("oddsapi: worker failed (RuntimeError)") == 2
    assert MARK not in feed.logs.text and KEY not in feed.logs.text
    t = MORNING + 2 * MINUTE
    while t <= et(23, 55):
        feed.kick(at=t)
        t += MINUTE if t < MORNING + 15 * MINUTE else 5 * MINUTE
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID, PHI_EID]   # 08:57
    phi = feed.mem["games"][PHI_EID]
    assert (phi["pulls"], phi["fails"], phi["last_at"]) == (2, 2, (MORNING + 2 * MINUTE).astimezone(UTC))
    assert feed.mem["games"][DEN_EID]["pulls"] == 1                     # its retry: the breaker's
    assert (feed.mem["streak"], feed.mem["breaker_day"], feed.mem["games"][PHI_EID]["pulls"]) == (3, DATE, 2)
    assert ("oddsapi: 3 billed failures in a row; paid calls stopped until tomorrow (parser counters: "
            "worker)") in _messages(feed.logs)


def test_a_raise_inside_the_settling_takes_the_calls_cost_off_the_balance_once_r1(feed, monkeypatch):
    """R1, the counters: an answer with no x-requests-* headers, then a raise
    inside the settling. The call's worst-case cost (1) comes off the balance
    once, not once for the half-done settling and again for the failure."""
    def bare(call, vendor):
        r = vendor.answer(call)
        return H.VendorResponse(status=r.status, headers={}, body=r.body, snippet="", nbytes=r.nbytes)
    feed.vendor.script["odds"] = [bare]
    monkeypatch.setattr(O, "merge_refresh", _raises)
    feed.kick(work=work_order(games=(PHI_DET,)))
    b, e = feed.mem["budget"], feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"], feed.mem["streak"]) == (1, 1, 1)
    assert (b["remaining"], b["used"], b["last"], b["pending"]) == (349, 151, 0, 0)


def test_a_settling_that_raises_half_way_is_undone_then_counted_once_r1(feed, monkeypatch):
    """R1, half done: the success path raises after it has counted the pull
    and stored the merged frame (here in _books_at). That half-done settling
    is undone, and the call is settled once, as a billed failure: one pull,
    not two; no frame and no generation bump from the half-stored merge (the
    lines a failed call leaves are the previous ones); one failure, the
    streak grown."""
    monkeypatch.setattr(live, "_books_at", _raises)
    feed.kick(work=work_order(games=(PHI_DET,)))
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"], e["fail_until"], e["frame"], e["books_at"], e["pulled"],
            e["pp_missing"]) == (1, 1, MORNING_UTC + dt.timedelta(seconds=90), None, {}, [], False)
    b = feed.mem["budget"]
    assert (feed.mem["streak"], feed.mem["generation"], b["remaining"], b["used"], b["pending"]) == (
        1, 0, 349, 151, 0)
    assert "oddsapi: worker failed (RuntimeError)" in _messages(feed.logs)
    assert MARK not in feed.logs.text and KEY not in feed.logs.text


def test_a_timeout_whose_settling_raises_still_counts_as_billed_r1(feed, monkeypatch):
    """R1 with no answer at all: a timeout may have been billed (3.5), so when
    its settling raises (here the failure's own bookkeeping, once), the
    broken call still counts as billed: one pull, one failure and the
    worst-case cost off the balance once. No answer came back, so the
    streak does not grow (build amendment B22)."""
    real, billed = live._failed, []

    def once(*a, **k):
        billed.append(a[2])
        if len(billed) == 1:
            raise RuntimeError(MARK)
        return real(*a, **k)
    monkeypatch.setattr(live, "_failed", once)
    feed.vendor.script["odds"] = [H.VendorTransportError("timeout")]
    feed.kick(work=work_order(games=(PHI_DET,)))
    e, b = feed.mem["games"][PHI_EID], feed.mem["budget"]
    assert billed == [True, True]
    assert (e["pulls"], e["fails"], feed.mem["streak"], b["remaining"], b["used"], b["pending"]) == (
        1, 1, 0, 349, 151, 0)


def test_an_empty_answer_is_free_not_a_pull_and_retried_in_ten_minutes(feed):
    def empty(call, vendor):
        return vendor.response(200, {**copy.deepcopy(vendor.odds[call.event_id]), "bookmakers": []}, last=0)
    feed.vendor.script["odds"] = [empty]
    work = work_order(games=(PHI_DET,))
    feed.kick(work=work)
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"], e["empty_next"]) == (0, 0, MORNING_UTC + OB.EMPTY_RETRY)
    assert (feed.mem["budget"]["remaining"], feed.mem["streak"]) == (350, 0)
    got = feed.view()
    assert got.state == "idle" and got.per_game[PHI_DET]["reason"] == "none_posted"
    feed.kick(at=MORNING + 9 * MINUTE, work=work)
    assert len(feed.vendor.odds_calls()) == 1
    feed.kick(at=MORNING + 10 * MINUTE, work=work)
    assert len(feed.vendor.odds_calls()) == 2 and e["pulls"] == 1


def test_a_404_ends_the_games_pulls_for_the_day(feed):
    feed.vendor.script["odds"] = [answer_with(404, snippet='{"error_code": "EVENT_NOT_FOUND"}')]
    work = work_order(games=(PHI_DET,))
    feed.kick(work=work)
    e = feed.mem["games"][PHI_EID]
    assert e["fail_until"] == et(0, 0, day=13).astimezone(UTC) and e["pulls"] == 0
    assert "oddsapi event odds: EVENT_NOT_FOUND; the game's pulls end for today" in _messages(feed.logs)
    for at in (et(9, 30), et(13, 0), et(18, 30)):
        feed.kick(at=at, work=work)
    assert len(feed.vendor.odds_calls()) == 1


# -- Stale beats empty, per game and per book; pre-tip lines ------------------------
def test_a_failed_refresh_keeps_the_games_lines_stale_beats_empty(feed):
    feed.kick()
    morning = _rows(feed.view(), PHI_DET)
    feed.vendor.script["odds"] = [answer_with(503, snippet=MARK)]
    feed.kick(at=et(18, 30))                                     # PHI's evening refresh fails
    got = feed.view()
    assert got.state == "stale" and got.per_game[PHI_DET]["asof"] == MORNING_UTC
    pd.testing.assert_frame_equal(_rows(got, PHI_DET), morning)
    assert "oddsapi event odds failed: server HTTP_503" in _messages(feed.logs)
    assert MARK not in feed.logs.text and MARK not in _text(feed.mem)
    feed.kick(at=et(18, 32))                                     # its backoff over: retried
    got = feed.view()
    assert got.state == "ok" and got.per_game[PHI_DET]["asof"] == et(18, 32)


def test_an_absent_book_keeps_its_own_older_lines_and_time_m23(feed):
    feed.kick()
    feed.vendor.drop = {"fanduel"}
    feed.kick(at=et(18, 30))
    got = feed.view()
    phi = got.per_game[PHI_DET]
    assert phi["books"]["fanduel"] == MORNING_UTC
    assert {b: t for b, t in phi["books"].items() if b != "fanduel"} == {
        b: et(18, 30) for b in O.BOOKS if b != "fanduel"}
    assert phi["asof"] == et(18, 30) and got.asof == MORNING_UTC     # DEN's last pull was the morning's
    fd = _rows(got, PHI_DET)
    assert len(fd[fd["book"] == "fanduel"]) == 5 and set(fd[fd["book"] == "fanduel"]["fetched_at"]) == {
        pd.Timestamp(MORNING_UTC)}


def test_pre_tip_lines_are_kept_after_tip_and_never_bought_again(feed):
    feed.kick()
    feed.kick(at=et(18, 30))                                     # PHI's evening refresh
    n = len(feed.vendor.odds_calls(PHI_EID))
    assert n == 2
    for at in (PHI_TIP - 4 * MINUTE, PHI_TIP, PHI_TIP + 30 * MINUTE, PHI_TIP + 3 * 60 * MINUTE):
        feed.kick(at=at)
    assert len(feed.vendor.odds_calls(PHI_EID)) == n
    got = feed.view()
    assert got.per_game[PHI_DET]["pretip"] and got.per_game[PHI_DET]["asof"] == et(18, 30)
    assert len(_rows(got, PHI_DET)) == 25 and got.per_game[PHI_DET]["reason"] is None


# -- The auth states (3.8) -----------------------------------------------------------------
def test_out_of_credits_stops_paid_calls_but_not_the_events_call(feed):
    feed.vendor.script["odds"] = [answer_with(401, snippet='{"error_code": "OUT_OF_USAGE_CREDITS"}')]
    work = work_order(games=(PHI_DET,))
    feed.kick(work=work)
    b = feed.mem["budget"]
    assert (b["auth"], b["auth_until"]) == ("out_of_credits", None)
    assert ("oddsapi: 401 OUT_OF_USAGE_CREDITS; paid calls stopped until the credits reset"
            in _messages(feed.logs))
    for m in range(5, 60, 5):
        feed.kick(at=MORNING + m * MINUTE, work=work)
    assert len(feed.vendor.odds_calls()) == 1 and len(feed.vendor.events_calls()) == 6
    assert feed.view().state == "paused"
    feed.vendor.used, feed.vendor.remaining = 0, 500             # the vendor's reset
    feed.kick(at=MORNING + 60 * MINUTE, work=work)
    assert b["auth"] == "ok" and len(feed.vendor.odds_calls()) == 2
    assert "oddsapi: credits reset outside the 1st; the period rule needs a change" in _messages(feed.logs)


def test_a_bad_key_stops_every_call_for_30_minutes_then_one_free_call_checks(feed):
    feed.vendor.script["events"] = [answer_with(401, snippet='{"error_code": "INVALID_KEY"}')]
    feed.kick()
    assert [c.what for c in feed.vendor.calls] == ["events"]
    b = feed.mem["budget"]
    assert (b["auth"], b["auth_until"]) == ("bad_key", MORNING_UTC + live.AUTH_WAIT)
    assert "oddsapi: 401 INVALID_KEY; all calls stopped for 30 min" in _messages(feed.logs)
    for m in (5, 10, 15, 20, 25, 29):
        feed.kick(at=MORNING + m * MINUTE)
    assert len(feed.vendor.calls) == 1
    got = feed.view()
    assert got.state == "paused" and {g["reason"] for g in got.per_game.values()} >= {"paused"}
    feed.kick(at=MORNING + 30 * MINUTE)                          # one free call: the key works now
    assert [c.what for c in feed.vendor.calls] == ["events", "events", "odds", "odds"]
    assert b["auth"] == "ok" and feed.view().state == "ok"


def test_the_check_after_a_bad_key_backs_off_when_the_vendor_is_down(feed):
    """3.8 with 2.4.4 step 2: the one free call after the 30 minutes finds the
    vendor down; it backs off like any events failure (90 s, then doubling)
    instead of calling again on every view."""
    feed.vendor.script["events"] = [answer_with(401, snippet='{"error_code": "INVALID_KEY"}'),
                                    answer_with(503), answer_with(503)]
    feed.kick()
    feed.kick(at=MORNING + 30 * MINUTE)
    assert len(feed.vendor.events_calls()) == 2
    for s in (10, 30, 60, 89):
        feed.kick(at=MORNING + 30 * MINUTE + dt.timedelta(seconds=s))
    assert len(feed.vendor.events_calls()) == 2
    feed.kick(at=MORNING + 31 * MINUTE + dt.timedelta(seconds=30))
    assert len(feed.vendor.events_calls()) == 3 and feed.mem["events_fail"][2] == 2
    feed.kick(at=MORNING + 35 * MINUTE)                          # 3 minutes on: the key works
    assert [c.what for c in feed.vendor.calls][-3:] == ["events", "odds", "odds"]
    assert feed.mem["budget"]["auth"] == "ok"
    assert "oddsapi events failed: server HTTP_503" in _messages(feed.logs)


def test_a_paid_call_refused_as_a_bad_key_stops_the_free_call_too(feed):
    feed.vendor.script["odds"] = [answer_with(401, snippet="no code here")]
    feed.kick()
    assert [c.what for c in feed.vendor.calls] == ["events", "odds"]       # the second game waits
    assert feed.mem["budget"]["auth"] == "bad_key"
    assert "oddsapi: 401 HTTP_401; all calls stopped for 30 min" in _messages(feed.logs)
    feed.kick(at=MORNING + 15 * MINUTE)
    assert len(feed.vendor.calls) == 2


# -- Failures become fixed states and fixed log lines ----------------------------------------
FAILURES = {
    "422": (answer_with(422, snippet='{"error_code": "INVALID_MARKET", "message": "' + MARK + '"}'),
            "bad_request INVALID_MARKET"),
    "429": (answer_with(429, snippet='{"error_code": "EXCEEDED_FREQ_LIMIT", "m": "' + MARK + '"}'),
            "rate_limited EXCEEDED_FREQ_LIMIT"),
    "503": (answer_with(503, snippet=MARK), "server HTTP_503"),
    "timeout": (H.VendorTransportError("timeout"), "timeout"),
    "oversize": (H.VendorTransportError("oversize", 4_000_001), "oversize"),
    "connect": (H.VendorTransportError("connect", 0), "server connect"),
    "not json": (answer_with(200, last=1, body=None), "bad_payload"),
    "a list": (answer_with(200, last=1, body=[MARK]), "bad_payload"),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_a_failed_call_becomes_a_fixed_state_and_a_fixed_log_line(feed, name):
    """422, 429, 5xx, timeouts and bad JSON: DOWN with no lines in memory,
    STALE with some; the log line names our kind and code only, and no
    vendor text reaches the log, the ledger or the view."""
    answer, words = FAILURES[name]
    feed.vendor.script["odds"] = [answer]
    feed.kick(work=work_order(games=(PHI_DET,)))
    got = feed.view()
    assert (got.state, got.value.empty) == ("down", True)
    assert f"oddsapi event odds failed: {words}" in _messages(feed.logs)
    other = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.vendor.script["odds"] = [answer]                        # PHI fails, DEN lands
    feed.kick(mem=other)
    got = feed.view(mem=other)
    assert got.state == "stale" and set(got.value["game_id"]) == {DEN_SAS}
    for blob in (feed.logs.text, _text(feed.mem), _text(other), repr(got), _text(got.per_game)):
        assert MARK not in blob


# -- In flight, the rate cap, other dates ---------------------------------------------------
def test_a_slow_walk_reports_loading_then_lands(feed):
    """The held pool: the walk and its calls wait until the test releases
    them. A kick while the walk waits starts no second walk (one at a time,
    design 1.2); queued games count as loading; a kick while the calls wait
    queues nothing again (M8); the lines land once the calls run."""
    pool = HeldPool()
    feed.kick(pool=pool)
    feed.kick(pool=pool)                                         # one walk at a time: no second
    assert pool.names == ["_walk"]
    got = feed.view()
    assert (got.state, got.loading, feed.vendor.calls) == ("idle", 0, [])
    pool.run("_walk")                                            # the walk: events, two calls queued
    got = feed.view()
    assert (got.state, got.loading, got.value.empty) == ("idle", 2, True)
    assert feed.mem["queued"] == {PHI_EID, DEN_EID}
    feed.kick(pool=pool)
    pool.run("_walk")                                            # the second walk queues nothing
    assert [fn.__name__ for _f, fn, _a, _k in pool.held] == ["_pull_game", "_pull_game"]
    pool.release()
    got = feed.view()
    assert (got.state, got.loading) == ("ok", 0) and set(got.value["game_id"]) == {PHI_DET, DEN_SAS}
    assert pool.names == ["_walk", "_pull_game", "_pull_game", "_walk"]


def test_a_game_already_queued_is_never_queued_again_m8(feed):
    """M8: one game's call waits in the pool while the other's has landed.
    A kick then finds a free place under the queue's cap (today's games), yet
    sends no second task for the waiting game: a duplicate would hold a
    worker even though its event lock keeps it from calling."""
    pool = HeldPool()
    feed.kick(pool=pool)
    pool.run("_walk")                                            # the walk: two calls held
    assert [args[1] for _f, fn, args, _k in pool.held] == [PHI_EID, DEN_EID]
    pool.run("_pull_game", 1)                                    # DEN at SAS's call runs alone
    assert feed.mem["queued"] == {PHI_EID} and len(pool.held) == 1
    feed.kick(pool=pool)
    pool.run("_walk")                                            # the second walk, PHI still waiting
    assert [fn.__name__ for _f, fn, _a, _k in pool.held] == ["_pull_game"]
    assert pool.names.count("_pull_game") == 2 and pool.names.count("_walk") == 2
    pool.release()
    assert [c.event_id for c in feed.vendor.odds_calls()] == [DEN_EID, PHI_EID]
    assert feed.mem["queued"] == set()


def test_the_rate_cap_holds_paid_calls_to_the_setting_per_ten_minutes(feed):
    """BUCKETVALUE_ODDS_MAX_CALLS_10MIN, the call times kept in the ledger."""
    env = {**feed.env, "BUCKETVALUE_ODDS_MAX_CALLS_10MIN": "1"}
    feed.kick(env=env)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID]
    assert feed.mem["calls"] == [MORNING_UTC]
    assert "oddsapi: paid-call rate cap reached; the rest wait their turn" in _messages(feed.logs)
    feed.kick(at=MORNING + dt.timedelta(minutes=9, seconds=59), env=env)
    assert len(feed.vendor.odds_calls()) == 1
    feed.kick(at=MORNING + 10 * MINUTE, env=env)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]


def test_another_date_never_fetches(feed):
    """Today ET only (1.1): memory of any date is read without a call, and a
    work order whose games are not today's buys nothing, events included."""
    for d in (dt.date(2026, 3, 11), dt.date(2026, 3, 13), DATE):
        assert live.prop_lines_memory(d, mem=feed.mem).value.empty
    tomorrow = [dict(g, game_date="2026-03-13") for g in TODAY]
    feed.kick(work={**work_order(), "games": tomorrow})
    feed.kick(work=work_order(date="2026-03-11", games=()))
    assert feed.vendor.calls == []
    feed.kick()
    assert live.prop_lines_memory(dt.date(2026, 3, 13), mem=feed.mem).value.empty
    assert not live.prop_lines_memory(DATE, mem=feed.mem).value.empty
    assert len(feed.vendor.calls) == 3


def test_a_restart_past_a_tip_says_lost_in_restart_and_re_buys_the_rest_at_once_m22_b21(feed):
    """m22 and the brake (3.4): a ledger born at 7:10 PM ET, after PHI at
    DET's tip guard, holds no frame for it, so its reason is lost_in_restart.
    The period was spending (150 used against 84 earlier games), so DEN at
    SAS counts as pulled at the birth. With no lines in this ledger, P = 2
    and its last refresh time (8:30 PM) still ahead, it keeps one re-buy at
    once (build amendment B21; the brake alone left it to that refresh), so
    its lines show at once, and nothing more is bought for it."""
    born = PHI_TIP + 10 * MINUTE
    feed.mem = live.new_feed_memory(born, had_ledger=False)
    feed.kick(at=born)
    assert [c.what for c in feed.vendor.calls] == ["events", "odds"]
    assert [(c.event_id, c.books, c.markets) for c in feed.vendor.odds_calls()] == [(DEN_EID, FIVE, PTS)]
    den = feed.mem["games"][DEN_EID]
    assert (den["pulls"], den["first_at"], den["last_at"], den["braked"], feed.mem["braked"]) == (
        2, born, born, True, True)
    got = feed.view()
    assert got.per_game[PHI_DET]["reason"] == "lost_in_restart" and not _rows(got, DEN_SAS).empty
    for at in (DEN_TIP - 31 * MINUTE, DEN_TIP - 30 * MINUTE, DEN_TIP - 10 * MINUTE):
        feed.kick(at=at)
    assert len(feed.vendor.odds_calls()) == 1
    got = feed.view()
    assert got.per_game[PHI_DET]["reason"] == "lost_in_restart" and not _rows(got, DEN_SAS).empty


@pytest.mark.parametrize("listed", ["none", "phi_only"])
def test_the_brake_also_covers_todays_games_matched_after_it_was_decided_r5(feed, listed):
    """Review R5 (3.4: the brake treats EACH of today's untipped games as
    pulled once, at the ledger's birth). A ledger born at 13:00 ET in a
    spending period (150 used against 84 earlier games). Its first events
    answer lists none of today's games, or PHI at DET alone, so the brake is
    decided before they are all matched. The games matched ten minutes later
    are braked too: counted as pulled at the birth, so each has the brake's
    pull and its one re-buy (build amendment B21: P = 2, no lines yet), two
    pulls, P, and no evening pull. (Unbraked, a game matched at 13:10 would
    be bought once then and again at tip minus 30 minutes.)"""
    born = et(13, 0)
    feed.mem = live.new_feed_memory(born, had_ledger=False)
    first = [] if listed == "none" else [e for e in feed.vendor.events if e["id"] == PHI_EID]
    feed.vendor.script["events"] = [answer_with(200, body=first)]
    feed.kick(at=born)
    assert feed.mem["braked"] and feed.mem["brake_day"] == DATE
    assert set(feed.mem["games"]) == ({PHI_EID} if first else set())
    assert [c.event_id for c in feed.vendor.odds_calls()] == ([PHI_EID] if first else [])  # B21
    feed.kick(at=et(13, 10))                                 # the events again: both games matched
    assert len(feed.vendor.events_calls()) == 2
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    for eid in (PHI_EID, DEN_EID):
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["first_at"], e["braked"]) == (2, born, True), eid
    for at in (et(18, 29), et(18, 30), et(20, 30)):
        feed.kick(at=at)
    assert len(feed.vendor.odds_calls()) == 2
    assert feed.mem["brake_day"] == DATE


def test_the_brake_stays_on_the_day_it_was_decided_r5(feed):
    """R5's other side: the brake's decision is kept for its own ET day only.
    A ledger braked on Mar 12 (born 13:00, the period spending) and still
    alive the next morning has lost no pull of Mar 13's games, so a game of
    Mar 13 matched then is not braked. A brake day that is not a date makes
    the ledger malformed (m25)."""
    born = et(13, 0)
    feed.mem = live.new_feed_memory(born, had_ledger=False)
    feed.kick(at=born)
    phi = feed.mem["games"][PHI_EID]
    assert feed.mem["brake_day"] == DATE and (phi["pulls"], phi["braked"]) == (2, True)   # B21's re-buy
    s = O.read_settings(feed.env)
    now = et(8, 55, day=13).astimezone(UTC)
    later = live._new_entry("0022500999", "2026-03-13", et(19, 0, day=13).astimezone(UTC), None)
    with feed.mem["lock"]:
        view = live._budget(feed.mem, s, now)
        braked = live._brake(feed.mem, s, view, [("f" * 32, later)], dt.date(2026, 3, 13), now)
    assert (braked, later["pulls"], later["first_at"], later["braked"], feed.mem["brake_day"]) == (
        0, 0, None, False, DATE)
    assert live._valid_ledger(feed.mem)
    assert not live._valid_ledger({**feed.mem, "brake_day": "2026-03-12"})


# A horizon in which Mar 12 falls late in its period: 91 of its 119 games are
# dated through Mar 12, so today's spending ceiling is A - S (review R4).
LATE_HORIZON = [g for g in HORIZON if g["game_date"] <= "2026-03-16"]


def test_a_morning_crash_loop_is_bounded_by_the_floor_once_its_re_buys_are_checked_b11(feed):
    """Build amendment B11 (review R4). Before it, B1 never checked the
    morning-pull floor for a first pull, and a fresh ledger cannot tell a
    re-buy from a first pull, so every restart before the brake's window
    ends (first-pull time + 60 minutes, 09:50 ET) bought today's games again,
    bounded only by the reserve: 15 restarts from 08:50 to 09:46, with 60
    left and 440 used (A 500, S 25) and the ceiling at A - S, made 30 paid
    calls and left 30 credits, 5 above the reserve, for the horizon's 33
    other games' morning pulls. Now a ledger born after the first-pull time
    in a period that was already spending (442 used against 84 earlier
    games) asks the governor with a "rebuy", which the floor guards as design
    3.6 wrote it. The ledger born at 08:50 has seen the morning, so its two
    pulls are first pulls (B1) and leave 58. Each of the 14 later ledgers'
    re-buys is refused by the floor, 32 above the reserve against 34 other
    games waiting (the premise below): the loop costs 2 credits, not 30, and
    the horizon's other games keep their morning pulls. A restart at 09:51
    is braked and buys nothing."""
    work = {**work_order(), "horizon": [dict(g) for g in LATE_HORIZON]}
    s = O.read_settings(feed.env)
    horizon, t2 = live._horizon_of(work), et(8, 54).astimezone(UTC)
    period = OB.period_of(t2, 442, 500, s.reserve, len(s.markets), horizon, busiest=236)
    plan = OB.plan(58, 442, 0, s, horizon, {}, period, "ok", 236, now=t2, period_games=horizon)
    waiting = OB.waiting_first(horizon, {}, t2, period, PHI_DET)
    before = OB.games_before(horizon, period, DATE)
    first = OB.first_pull_time(DATE, s.first_pull_et)
    guards = dict(now=t2, tip=PHI_TIP, guard=s.tip_guard, auth="ok", calls=[], max_calls=s.max_calls_10min)
    assert (len(horizon), plan.ceiling, plan.reserve, waiting, before) == (119, 475, 25, 34, 84)
    assert OB.first_kind(et(8, 50), False, 440, 1, before, first) == "first"     # saw the morning
    assert OB.first_kind(t2, False, 442, 1, before, first) == "rebuy"            # 884 >= 84: spending
    assert OB.may_spend(58, 442, 0, 1, plan, waiting, 1, kind="rebuy", **guards) == "floor"
    assert OB.may_spend(58, 442, 0, 1, plan, waiting, 1, kind="first", **guards) is None
    feed.vendor.remaining, feed.vendor.used = 60, 440
    t, left, bought = et(8, 50), [], []
    for _ in range(15):
        feed.mem = live.new_feed_memory(t, had_ledger=False)          # a restart: a new process's ledger
        n = len(feed.vendor.odds_calls())
        feed.kick(at=t, work=work)
        left.append(feed.mem["budget"]["remaining"])
        bought.append(len(feed.vendor.odds_calls()) - n)
        assert feed.mem["budget"]["pending"] == 0 and not feed.mem["queued"]
        t += 4 * MINUTE
    assert bought == [2] + [0] * 14
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    assert left == [58] * 15 and (feed.vendor.remaining, feed.vendor.used) == (58, 442)
    assert feed.mem["brake_used"] == 442
    assert [feed.mem["games"][eid]["pulls"] for eid in (PHI_EID, DEN_EID)] == [0, 0]
    born = et(9, 51)
    feed.mem = live.new_feed_memory(born, had_ledger=False)
    feed.kick(at=born, work=work)
    assert len(feed.vendor.odds_calls()) == 2 and feed.mem["braked"]
    assert [feed.mem["games"][eid]["first_at"] for eid in (PHI_EID, DEN_EID)] == [born, born]


def test_the_prizepicks_top_up_polls_free_every_ten_minutes_then_counts_once_posted_m4(feed, monkeypatch):
    """3.4, M4 and B3: DEN at SAS came back without PrizePicks. Every 10
    minutes the walk asks for PrizePicks alone, free while it has nothing;
    the first answer with lines is billed, counts as a pull and touches
    PrizePicks' rows only. merge_refresh is told, on every call, the books
    and the markets that call asked for (B3)."""
    merges = []
    real = O.merge_refresh

    def spy(prev, new, books_requested, **kw):
        merges.append((tuple(books_requested), kw.get("markets_requested")))
        return real(prev, new, books_requested, **kw)
    monkeypatch.setattr(O, "merge_refresh", spy)
    feed.kick()
    assert merges == [(FIVE, PTS), (FIVE, PTS)]
    den = feed.mem["games"][DEN_EID]
    assert (den["pp_missing"], den["pp_next"]) == (True, MORNING_UTC + OB.PP_RETRY)
    feed.kick(at=MORNING + 9 * MINUTE)
    assert len(feed.vendor.odds_calls()) == 2
    feed.kick(at=MORNING + 10 * MINUTE)
    top = feed.vendor.odds_calls()[-1]
    assert (top.event_id, top.books, top.markets) == (DEN_EID, ("prizepicks",), PTS)
    assert (den["pulls"], den["pp_next"], feed.mem["budget"]["remaining"]) == (
        1, MORNING_UTC + 20 * MINUTE, 348)                       # empty: free, and not a pull
    posted = copy.deepcopy(next(b for b in feed.vendor.odds[DEN_EID]["bookmakers"] if b["key"] == "draftkings"))
    posted.update(key="prizepicks", title="PrizePicks")
    for m in posted["markets"]:
        for oc in m["outcomes"]:
            oc["price"] = -137
    feed.vendor.odds[DEN_EID]["bookmakers"].append(posted)
    feed.kick(at=MORNING + 20 * MINUTE)
    assert merges[-1] == (("prizepicks",), PTS)
    assert (den["pulls"], den["pp_missing"], den["pp_next"], feed.mem["budget"]["remaining"]) == (
        2, False, None, 347)
    got = feed.view()
    assert got.per_game[DEN_SAS]["books"]["prizepicks"] == MORNING_UTC + 20 * MINUTE
    assert {b: t for b, t in got.per_game[DEN_SAS]["books"].items() if b != "prizepicks"} == {
        b: MORNING_UTC for b in ("draftkings", "fanduel", "betmgm", "underdog")}
    n = len(feed.vendor.odds_calls(DEN_EID))
    for at in (et(12, 0), et(20, 30), et(20, 50)):               # P = 2 is spent: no evening pull
        feed.kick(at=at)
    assert len(feed.vendor.odds_calls(DEN_EID)) == n


def test_a_first_pull_asks_what_first_pull_markets_gives_and_a_refresh_the_plans_markets_b1(
        feed, monkeypatch):
    """B1 wired in: a first pull asks the markets oddsbudget.first_pull_markets
    returns, given the balance, what is pending and the OTHER games still
    waiting for a first pull; a refresh asks the plan's markets. The ledger
    is born at 08:00, so it saw the morning and its pulls are first pulls
    (born at 08:55 in this spending period they would be B11 re-buys, which
    ask the core markets only: build amendment B20)."""
    seen = []

    def chosen(plan, settings, remaining, pending, waiting):
        seen.append((tuple(plan.markets), remaining, pending, waiting))
        return ("player_rebounds",)
    monkeypatch.setattr(OB, "first_pull_markets", chosen)
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.kick()
    assert [c.markets for c in feed.vendor.odds_calls()] == [("player_rebounds",)] * 2
    assert seen == [(PTS, 350, 0, 139), (PTS, 349, 0, 138)]
    feed.kick(at=et(18, 30))
    assert feed.vendor.odds_calls()[-1].markets == PTS and len(seen) == 2


def test_a_short_morning_buys_todays_first_pulls_and_the_floor_still_refuses_a_refresh_b1(
        feed, monkeypatch):
    """Review R3, build amendment B1 wired through live, for a ledger born at
    08:00 ET: before the games' 08:50 first-pull time, so it has seen their
    whole morning and its pulls are first pulls (B11). 100 left and 100 used
    on Mar 12 (the reserve 25, 140 games to plan) put the plan in short mode,
    P = 1. The morning-pull floor would refuse each of today's first pulls (74
    above the reserve, 139 other games waiting: the premise below), and B1
    does not check it for a first pull, so the walk buys both, in tip order.
    The same morning in a ledger born at 08:55, after the first-pull time in
    a period that was spending (100 used against 84 earlier games), is a
    pair of re-buys, which the floor refuses (B11). A refresh or a
    PrizePicks top-up of the same games, forced due, is still refused by the
    floor: live hands may_spend each pull's own kind."""
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.vendor.remaining, feed.vendor.used = 100, 100
    s = O.read_settings(feed.env)
    feed.kick(at=et(8, 0))                                   # the balance known; nothing due before 08:50
    feed.at[0] = MORNING_UTC
    with feed.mem["lock"]:
        view = live._budget(feed.mem, s, MORNING_UTC)
    waiting = OB.waiting_first(view.horizon, view.slots, MORNING_UTC, view.period, PHI_DET)
    before = OB.games_before(view.horizon, view.period, DATE)
    first = OB.first_pull_time(DATE, s.first_pull_et)
    guards = dict(now=MORNING_UTC, tip=PHI_TIP, guard=s.tip_guard, auth="ok", calls=[],
                  max_calls=s.max_calls_10min)
    assert (view.plan.mode, view.plan.pulls_per_game, view.plan.reserve, waiting) == ("short", 1, 25, 139)
    assert OB.may_spend(100, 100, 0, 1, view.plan, waiting, 1, kind="refresh", **guards) == "floor"
    assert OB.may_spend(100, 100, 0, 1, view.plan, waiting, 1, kind="rebuy", **guards) == "floor"
    assert OB.may_spend(100, 100, 0, 1, view.plan, waiting, 1, kind="first", **guards) is None
    assert (feed.mem["brake_used"], before) == (100, 84)
    assert OB.first_kind(et(8, 0), False, 100, 1, before, first) == "first"
    assert OB.first_kind(MORNING, False, 100, 1, before, first) == "rebuy"
    late = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.kick(mem=late)                                      # born 08:55: two re-buys, both refused
    assert feed.vendor.odds_calls() == [] and late["brake_used"] == 100
    assert [late["games"][eid]["pulls"] for eid in (PHI_EID, DEN_EID)] == [0, 0]
    feed.kick()
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]          # in tip order
    assert (feed.mem["budget"]["remaining"], feed.mem["budget"]["mode"]) == (98, "short")
    for at, kind in ((et(9, 30), "refresh"), (et(9, 45), "pp")):
        monkeypatch.setattr(OB, "due_kind", lambda *a, _kind=kind, **k: _kind)
        feed.kick(at=at)
    assert len(feed.vendor.odds_calls()) == 2 and feed.mem["budget"]["pending"] == 0


def test_a_ledger_born_at_0900_in_a_period_that_was_not_spending_buys_first_pulls_b11(feed):
    """Build amendment B11's other side. A ledger born at 09:00 ET, after the
    08:50 first-pull time, in a period that was NOT spending before it (41
    used against 84 earlier games: 82 < 84), has nothing an earlier process
    could have bought, so its pulls are first pulls and skip the floor (B1).
    Barrett's reserve of 400 (BUCKETVALUE_ODDS_RESERVE) leaves 59 of the 459
    above it for 140 games: short mode, P = 1, where the floor would refuse a
    re-buy (58 above the reserve against 139 games waiting: the premise).
    Both games are bought, in tip order. The usage test reads the `used` the
    ledger's brake was decided on, so its own first pull, which takes `used`
    to 42 (84, the line), never turns the second game's into a re-buy. With
    42 used before the ledger (84 >= 84: spending) the same morning is two
    re-buys, both refused."""
    env = {**feed.env, "BUCKETVALUE_ODDS_RESERVE": "400"}
    s = O.read_settings(env)
    nine = et(9, 0)
    t = nine.astimezone(UTC)
    horizon = live._horizon_of(feed.work)
    period = OB.period_of(t, 41, 500, s.reserve, len(s.markets), horizon, busiest=236)
    plan = OB.plan(459, 41, 0, s, horizon, {}, period, "ok", 236, now=t, period_games=horizon)
    waiting = OB.waiting_first(horizon, {}, t, period, PHI_DET)
    before = OB.games_before(horizon, period, DATE)
    first = OB.first_pull_time(DATE, s.first_pull_et)
    guards = dict(now=t, tip=PHI_TIP, guard=s.tip_guard, auth="ok", calls=[], max_calls=s.max_calls_10min)
    assert (plan.mode, plan.pulls_per_game, plan.reserve, waiting, before) == ("short", 1, 400, 139, 84)
    assert OB.may_spend(459, 41, 0, 1, plan, waiting, 1, kind="rebuy", **guards) == "floor"
    assert OB.may_spend(459, 41, 0, 1, plan, waiting, 1, kind="first", **guards) is None
    assert OB.first_kind(t, False, 41, 1, before, first) == "first"
    assert OB.first_kind(t, False, 42, 1, before, first) == "rebuy"
    for used, bought in ((41, [PHI_EID, DEN_EID]), (42, [])):
        feed.vendor.remaining, feed.vendor.used = 500 - used, used
        n = len(feed.vendor.odds_calls())
        feed.mem = live.new_feed_memory(nine, had_ledger=False)
        feed.kick(at=nine, env=env)
        assert [c.event_id for c in feed.vendor.odds_calls()[n:]] == bought, used
        assert feed.mem["brake_used"] == used and not feed.mem["brake_day"]
        assert (feed.mem["budget"]["mode"], feed.mem["budget"]["pending"]) == ("short", 0)
    assert len(feed.vendor.odds_calls()) == 2


def test_a_morning_re_buy_the_floor_refuses_holds_the_game_and_today_reads_paused_f4(feed):
    """Build 3's review, F4 (design 1.3 and 6, "the reserve, floor or
    ceiling reached" is PAUSED; N1). The short morning above in a ledger born
    at 08:55 (a restart, or Cloud waking the app): both of today's first
    pulls are re-buys, which the floor refuses (B11), all day, since 139
    other games keep waiting. Each game is held by the floor, and today's
    view reads "paused" (reason "paused"), not "idle" with "not_yet": the
    pages then say there are no lines from the feed today, never that the
    board fills by itself. Read after midnight, when Mar 12 is no longer
    today, the same ledger is not paused (nothing is bought for that date
    any more). Once a call goes ahead (the balance rises: the floor holds, a
    re-buy is bought) the hold is gone and the view is "ok"."""
    feed.vendor.remaining, feed.vendor.used = 100, 100
    late = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    for at in (MORNING, et(9, 30), et(12, 20)):
        feed.kick(at=at, mem=late)
    assert feed.vendor.odds_calls() == [] and late["budget"]["mode"] == "short"
    assert [late["games"][eid]["held"] for eid in (PHI_EID, DEN_EID)] == ["floor", "floor"]
    got = feed.view(mem=late)
    assert (got.state, got.value.empty, got.loading) == ("paused", True, 0)
    assert {gid: got.per_game[gid]["reason"] for gid in (PHI_DET, DEN_SAS)} == {
        PHI_DET: "paused", DEN_SAS: "paused"}
    after = live.feed_view(DATE, mem=late, wait_s=0, now=et(1, 0, day=13))
    # Never pulled, so never "none_posted" (build amendment B18: that is for a
    # game whose books posted none); the Game page says the feed did not load it.
    assert (after.state, after.per_game[PHI_DET]["reason"]) == ("idle", "not_yet")
    feed.vendor.remaining = 500                                  # the floor holds at 500 left
    feed.kick(at=et(12, 31), mem=late)                           # the events call learns it
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    assert [late["games"][eid]["held"] for eid in (PHI_EID, DEN_EID)] == [None, None]
    got = feed.view(mem=late)
    assert got.state == "ok" and {gid: got.per_game[gid]["reason"] for gid in (PHI_DET, DEN_SAS)} == {
        PHI_DET: None, DEN_SAS: None}


def test_the_ceiling_holds_a_game_too_and_the_rate_cap_never_does_f4(feed):
    """F4's other budget holds. 300 left and 410 used make today's ceiling 411
    (the R2 premise above): PHI at DET's first pull reaches it, and DEN at
    SAS's is refused by the ceiling, held, so today reads "paused" with lines
    ("The board is not refreshing these lines for now"), and DEN at SAS's
    reason is "paused". The rate cap only spaces calls: one call per 10
    minutes refuses DEN at SAS's pull just as surely, yet holds nothing, and
    today stays "ok" (the rest wait their turn). Each hold is a fixed code of
    the governor's refusals."""
    assert live.HOLDS == ("reserve", "floor", "ceiling") and set(live.HOLDS) <= set(OB.REFUSALS)
    feed.vendor.remaining, feed.vendor.used = 300, 410
    feed.kick()
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID]
    assert [feed.mem["games"][eid]["held"] for eid in (PHI_EID, DEN_EID)] == [None, "ceiling"]
    got = feed.view()
    assert (got.state, got.per_game[DEN_SAS]["reason"]) == ("paused", "paused")
    assert set(got.value["game_id"]) == {PHI_DET}
    capped = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.vendor.remaining, feed.vendor.used = 350, 150
    feed.kick(mem=capped, env={**feed.env, "BUCKETVALUE_ODDS_MAX_CALLS_10MIN": "1"})
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, PHI_EID]
    assert [capped["games"][eid]["held"] for eid in (PHI_EID, DEN_EID)] == [None, None]
    got = feed.view(mem=capped)
    assert (got.state, got.per_game[DEN_SAS]["reason"]) == ("ok", "not_yet")


def test_no_reset_by_midday_on_the_1st_is_logged_once(feed):
    """3.2: by 12:00 ET on the 1st, a `used` that still fails the window's own
    test says the credits did not reset: one WARNING that day."""
    april = [{"game_id": f"a{i}", "game_date": d, "tip_utc": f"{d}T23:30:00+00:00", "tbd": False}
             for d in ("2026-03-31", "2026-04-01") for i in range(5)]
    work = {"date": "2026-04-01", "games": [{**april[5], "home": "DET", "away": "PHI"}],
            "horizon": april, "busiest": 236}
    feed.vendor.used, feed.vendor.remaining = 470, 30
    for at in (dt.datetime(2026, 4, 1, 11, 55, tzinfo=ET), dt.datetime(2026, 4, 1, 12, 5, tzinfo=ET),
               dt.datetime(2026, 4, 1, 13, 5, tzinfo=ET)):
        feed.kick(at=at, work=work)
    said = [m for m in _messages(feed.logs) if "did not reset" in m]
    assert said == ["oddsapi: credits did not reset by midday on the 1st; check the vendor dashboard"]
    other = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.vendor.used, feed.vendor.remaining = 3, 497
    feed.kick(at=dt.datetime(2026, 4, 1, 12, 30, tzinfo=ET), work=work, mem=other)
    assert len([m for m in _messages(feed.logs) if "did not reset" in m]) == 1


# -- The key, nowhere ------------------------------------------------------------------------
def test_the_key_appears_nowhere_in_memory_results_or_logs(feed):
    """5.2 row 15: a fake key, read per call, reaches only vendor_get's
    params and secrets. A walk, a failure whose body echoes the key, the
    top-up polls and a view: the key is in no ledger field, no frame, no
    FeedLive and no log line, and the anchor holds none of it either."""
    feed.vendor.script["odds"] = [answer_with(422, snippet="apiKey=" + KEY + " INVALID_MARKET")]
    mem = live.feed_anchor()
    for at in (MORNING, et(9, 5), et(9, 15), et(18, 30)):
        feed.kick(at=at, mem=mem)
    got = live.prop_lines(feed.work, mem=mem, environ=feed.env, pool=feed.pool)
    assert not got.value.empty and all(c.keyed for c in feed.vendor.calls)
    blobs = {"ledger": _text(mem), "anchor": _text(sys.modules[live.ANCHOR_NAME]), "view": repr(got),
             "rows": got.value.to_string(), "per_game": _text(got.per_game), "report": _text(got.report),
             "logs": feed.logs.text}
    for name, blob in blobs.items():
        assert KEY not in blob and "apiKey" not in blob, name


# -- The anchor (2.4.1, B1; B10) ---------------------------------------------------------------
def test_the_anchor_survives_a_streamlit_cache_clear_between_two_kicks_b10(feed):
    """B10, design 8.2's anchor check made through feed_kick: clear both of
    Streamlit's caches between two kicks; the second makes zero extra paid
    calls, the ledger is the same object and the rate cap's times survive."""
    import streamlit as st
    mem = live.feed_anchor()
    live.feed_kick(feed.work, environ=feed.env, pool=feed.pool)          # mem=None: the anchor's
    assert len(feed.vendor.odds_calls()) == 2 and mem["calls"] == [MORNING_UTC, MORNING_UTC]
    times = list(mem["calls"])
    st.cache_data.clear()
    st.cache_resource.clear()
    feed.at[0] = MORNING_UTC + MINUTE
    assert live.feed_anchor() is mem
    live.feed_kick(feed.work, environ=feed.env, pool=feed.pool)
    assert len(feed.vendor.odds_calls()) == 2 and mem["calls"] == times
    assert live.feed_view(DATE, wait_s=0).generation == 2


def test_the_anchor_is_outside_bucketlib_has_no_file_and_survives_a_freshen_purge(
        fresh_anchor, monkeypatch, caplog):
    """2.4.1: freshen's own purge, sprung for real, pops every bucketlib
    module and not the anchor; Streamlit's watcher finds no file for it (its
    own get_module_paths); it is made once and logged once."""
    import freshen
    from streamlit.watcher.local_sources_watcher import get_module_paths
    caplog.set_level(logging.INFO, logger="bucketvalue")
    mem = live.feed_anchor()
    mod = sys.modules[live.ANCHOR_NAME]
    assert not live.ANCHOR_NAME.startswith(freshen.PKG) and "." not in live.ANCHOR_NAME
    assert not hasattr(mod, "__file__") and mod.__spec__ is None and not hasattr(mod, "__path__")
    assert get_module_paths(mod) == set()
    monkeypatch.setattr(freshen, "code_quick", lambda: (("changed", 0, 0),))
    monkeypatch.setattr(freshen, "code_digest", lambda: "changed")
    before = {n: m for n, m in sys.modules.items() if n == freshen.PKG or n.startswith(freshen.PKG + ".")}
    try:
        assert freshen.ensure() is True
        assert not any(n == freshen.PKG or n.startswith(freshen.PKG + ".") for n in sys.modules)
        assert sys.modules[live.ANCHOR_NAME] is mod and mod.state is mem
    finally:
        sys.modules.update(before)
    assert live.feed_anchor() is mem and mod.births == 1
    assert _messages(caplog) == ["oddsapi: feed memory created (process start)"]
    live.reset_feed_anchor_for_tests()
    assert live.ANCHOR_NAME not in sys.modules                   # the test reset leaves nothing


def test_a_version_bump_replaces_the_ledger_and_logs_it(fresh_anchor, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="bucketvalue")
    first = live.feed_anchor()
    first["generation"] = 7
    assert live.feed_anchor() is first and first["had_ledger"] is False
    monkeypatch.setattr(live, "FEED_MEMORY_VERSION", live.FEED_MEMORY_VERSION + 1)
    second = live.feed_anchor()
    assert second is not first and second["v"] == live.FEED_MEMORY_VERSION
    assert (second["had_ledger"], second["generation"], sys.modules[live.ANCHOR_NAME].births) == (True, 0, 2)
    assert _messages(caplog) == ["oddsapi: feed memory created (process start)",
                                 "oddsapi: feed memory created (replaced: version change)"]


def test_a_malformed_entry_counts_as_absent_and_a_malformed_ledger_is_replaced_m25(feed):
    feed.kick(at=et(8, 0))
    feed.mem["games"][PHI_EID]["pulls"] = "one"
    assert live._slot_of(feed.mem["games"][PHI_EID], PHI_EID) is None
    assert feed.view().per_game[PHI_DET]["reason"] == "not_yet"           # absent, never an error
    feed.kick(at=MORNING)
    feed.kick(at=MORNING + MINUTE)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    assert feed.mem["games"][PHI_EID]["pulls"] == 1
    assert _messages(feed.logs).count("oddsapi: a malformed game entry counted as absent") == 1
    live.reset_feed_anchor_for_tests()
    for field, bad in (("games", []), ("streak", True), ("budget", {"auth": "ok"}), ("v", 1)):
        mem = live.feed_anchor()
        mem[field] = bad
        assert live.feed_anchor() is not mem, field
    good = live.feed_anchor()
    good["budget"]["auth"] = "maybe"
    assert live.feed_anchor() is not good


def test_the_ledger_holds_plain_data_and_stdlib_objects_only_b4(feed):
    """B4: after a day's walks the ledger holds no bucketlib instance or
    function (no Slot, Plan, Period, Settings, FeedLive ...)."""
    for at in (MORNING, et(9, 5), et(18, 30), et(20, 30)):
        feed.kick(at=at)
    bad = []

    def walk(obj, where, depth=0):
        if depth > 10:
            return
        mod = getattr(type(obj), "__module__", "") or ""
        if mod.startswith("bucketlib") or callable(obj) and getattr(obj, "__module__", "").startswith("bucketlib"):
            bad.append(where)
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{where}.{k}", depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for i, v in enumerate(obj):
                walk(v, f"{where}[{i}]", depth + 1)
    walk(feed.mem, "mem")
    assert bad == []
    assert live._valid_ledger(feed.mem)


# -- The refresher (2.4.6, M6) -------------------------------------------------------------------
SERVE = Path(__file__).parent / "fixtures" / "serve"     # the serve fixture's committed 2026-03-12 files
EVENING = et(21, 30, day=11)                              # a view the evening before, 9:30 PM ET
TEAMS = {g["game_id"]: (g["away"], g["home"]) for g in TODAY}
# Two games of 2026-03-11, ids and pairings invented. The vendor fixture lists
# no event that day, so the evening view's walk learns the balance and buys nothing.
ELEVENTH = [{"game_id": "0022500940", "game_date": "2026-03-11",
             "tip_utc": "2026-03-11T23:00:00+00:00", "tbd": False, "away": "NYK", "home": "BOS"},
            {"game_id": "0022500941", "game_date": "2026-03-11",
             "tip_utc": "2026-03-12T00:00:00+00:00", "tbd": False, "away": "CHA", "home": "MIA"}]


def evening_order(serve, *, teams=True, written=EVENING) -> dict:
    """The work order a view at 9:30 PM ET on Mar 11 leaves (2.4.5): that
    day's buyable games, the period's horizon (B2), and Mar 12's committed
    files, projections first (`serve`). The horizon's Mar 12 rows of PHI at
    DET and DEN at SAS carry away and home, since the walk matches vendor
    events by the date and the team pair; teams=False gives the bare row
    DESIGN 2.4.5 lists (game_id, game_date, tip_utc, tbd)."""
    horizon = []
    for g in HORIZON:
        row = dict(g)
        if teams and g["game_id"] in TEAMS:
            row["away"], row["home"] = TEAMS[g["game_id"]]
        horizon.append(row)
    return {"date": "2026-03-11", "written_at": written.astimezone(UTC).isoformat(),
            "games": [dict(g) for g in ELEVENTH], "horizon": horizon,
            "serve_files": {"2026-03-12": [str(p) for p in serve]}, "busiest": 236}


def _ids_file(path: Path, ids) -> Path:
    """A committed serve file with the given game ids, and a second column
    the refresher must not read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": list(ids), "person_id": list(range(len(ids)))}).to_parquet(path, index=False)
    return path


def _evening(feed, order) -> None:
    """The process has run since the evening: a ledger born at 8 PM ET on
    Mar 11, and the 9:30 PM view's walk, which made the free events call
    (the balance) and bought nothing."""
    feed.mem = live.new_feed_memory(et(20, 0, day=11), had_ledger=False)
    n = len(feed.vendor.calls)
    feed.kick(at=EVENING, work=order)
    assert [c.what for c in feed.vendor.calls[n:]] == ["events"]
    assert feed.mem["budget"]["remaining"] is not None and feed.mem["work"] is order


def _tick(feed, at, *, env=None, mem=None, pool=None) -> str:
    """One refresher tick at `at`, called directly (2.4.6): the fixture's
    ledger, key and synchronous pool, with live's clock pinned there too
    (the view reads it)."""
    feed.at[0] = at.astimezone(UTC)
    return live.refresher_tick(feed.mem if mem is None else mem,
                               environ=feed.env if env is None else env,
                               pool=feed.pool if pool is None else pool, now=at)


def _refresher_threads() -> list:
    return [t for t in threading.enumerate() if t.name == live.REFRESHER_THREAD and t.is_alive()]


@pytest.fixture(autouse=True)
def _no_refresher_thread_left():
    """No test in this file leaves a refresher thread running (M10),
    checked at its teardown. One left behind is stopped here (the anchor
    reset sets its stop Event, and a loop whose anchor is gone returns at
    its next wake), and the test fails."""
    yield
    left = _refresher_threads()
    if left:
        live.reset_feed_anchor_for_tests()
        for t in left:
            t.join(2.0)
        pytest.fail(f"{len(left)} refresher thread(s) outlived the test", pytrace=False)


class CountingEvent(threading.Event):
    """A loop's own stop Event that counts the loop's waits, so a test can
    see it go round without sleeping on a guess."""

    def __init__(self):
        super().__init__()
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        return super().wait(timeout)


def _until(cond, timeout: float = 2.0) -> bool:
    """Poll `cond` every 2 ms for at most `timeout` seconds."""
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            return False
        time.sleep(0.002)
    return True


@pytest.fixture
def loops(monkeypatch):
    """Refresher loops the test drives (2.4.6): real threads running
    live._refresher_loop, each on its own stop Event, with REFRESHER_TICK_S
    made tiny and live's clock pinned (feed_anchor stamps the ledger it
    makes); the anchor fresh before and after. At teardown no loop may be
    alive (the test stops and joins its own); one left is stopped here and
    fails the test."""
    live.reset_feed_anchor_for_tests()
    monkeypatch.setattr(live, "REFRESHER_TICK_S", 0.005)
    monkeypatch.setattr(live, "now_utc", lambda: MORNING_UTC)
    started: list = []

    def start(version: int):
        stop = CountingEvent()
        t = threading.Thread(target=live._refresher_loop, args=(stop, version),
                             name=live.REFRESHER_THREAD, daemon=True)
        started.append((t, stop))
        t.start()
        return t, stop
    yield start
    left = _refresher_threads()
    for _t, stop in started:
        stop.set()
    live.reset_feed_anchor_for_tests()
    for t in [t for t, _s in started] + left:
        t.join(2.0)
    assert not left, "a refresher loop outlived its test"


@pytest.fixture
def refresher_on(monkeypatch):
    """The feed and the refresher switched on in the process environment
    through monkeypatch (conftest has the refresher off for the session),
    the anchor fresh before and after. The real tick waits REFRESHER_TICK_S
    (60 s) before its first tick, and a stand-in that never reaches the
    vendor takes its place anyway. At teardown no refresher thread may be
    alive: the test stops and joins its own."""
    live.reset_feed_anchor_for_tests()
    monkeypatch.setenv(live.KEY_ENV, KEY)
    monkeypatch.setenv("BUCKETVALUE_ODDS_REFRESHER", "1")
    monkeypatch.setattr(live, "refresher_tick", lambda mem, **k: "idle")
    yield
    left = _refresher_threads()
    live.reset_feed_anchor_for_tests()
    for t in left:
        t.join(2.0)
    assert not left, "a refresher thread outlived its test"


def _stop_refresher(anchor) -> None:
    """Stop and join the anchor's refresher: the test's own last act."""
    anchor.refresher_stop.set()
    anchor.refresher.join(2.0)


def test_at_0850_the_refresher_buys_from_the_evenings_work_order_with_no_view(feed):
    """M6 and 2.4.5: a view at 9:30 PM ET on Mar 11 left the work order, and
    nobody visits on the morning of Mar 12. The ticks overnight and up to
    08:49 make no call. At 08:50 the tick takes Mar 12's horizon rows whose
    ids are in the date's committed projections file (the serve fixture's:
    PHI at DET and DEN at SAS, not the period's five other games that day)
    and kicks the page's walk with them: the free events call, then each
    game once, in tip order. The ledger's order is still the evening's (its
    date and written_at), with Mar 12's games beside Mar 11's."""
    order = evening_order([SERVE / "projections.parquet", SERVE / "features.parquet"])
    _evening(feed, order)
    for at in (et(0, 0), et(4, 30), et(8, 49, 59)):
        assert _tick(feed, at) == "idle", at
    assert len(feed.vendor.calls) == 1
    assert _tick(feed, et(8, 50)) == "kicked"
    assert [c.what for c in feed.vendor.calls] == ["events", "events", "odds", "odds"]
    assert [(c.event_id, c.books, c.markets) for c in feed.vendor.odds_calls()] == [
        (PHI_EID, FIVE, PTS), (DEN_EID, FIVE, PTS)]
    assert feed.pool.names == ["_walk", "_walk", "_pull_game", "_pull_game"]
    got = feed.view()
    assert (got.state, got.asof) == ("ok", et(8, 50).astimezone(UTC))
    assert set(got.value["game_id"]) == {PHI_DET, DEN_SAS}
    w = feed.mem["work"]
    assert (w["date"], w["written_at"], w["horizon"], w["busiest"]) == (
        "2026-03-11", order["written_at"], order["horizon"], 236)
    assert [g["game_id"] for g in w["games"]] == [g["game_id"] for g in ELEVENTH] + [PHI_DET, DEN_SAS]
    assert {g["game_id"]: (g["away"], g["home"]) for g in w["games"][2:]} == TEAMS
    assert order["games"] == ELEVENTH and live._valid_ledger(feed.mem)     # the evening's own order untouched
    assert _tick(feed, et(8, 51)) == "idle" and len(feed.vendor.calls) == 4


def test_todays_order_written_before_todays_files_buys_at_0850_once_they_exist_b15(feed, tmp_path):
    """Build amendment B15 (S7 Q3). A view at 6:30 AM ET on Mar 12, before
    the bot published the day's files, left an order dated today with no
    buyable game (store's order holds only the games in the date's committed
    file, m26), and that view's walk made no call at all. Before B15 the
    refresher waited for the next view. Now, when today's order holds no
    game of today, the tick reads the order's horizon rows dated today and
    today's serve_files, exactly as for an order from the evening before:
    "no_work" while the files are missing, "idle" once they exist but before
    08:50, and at 08:50 it kicks the walk, which makes the free events call
    and buys each published game once, in tip order. The ledger's order
    keeps its own date and written_at; the order the view left is
    untouched."""
    day = tmp_path / "serve" / "2026-03-12"
    proj, feat = day / "projections_v1.parquet", day / "features_v1.parquet"
    early = et(6, 30)
    order = {**evening_order([proj, feat], written=early), "date": "2026-03-12", "games": []}
    feed.mem = live.new_feed_memory(et(6, 0), had_ledger=False)
    feed.kick(at=early, work=order)
    assert feed.vendor.calls == [] and feed.mem["work"] is order       # nothing buyable: no call
    for at in (et(7, 0), et(8, 30)):
        assert _tick(feed, at) == "no_work", at
    _ids_file(proj, [PHI_DET, DEN_SAS])                                # the bot publishes the day
    assert _tick(feed, et(8, 45)) == "idle" and feed.vendor.calls == []
    assert _tick(feed, et(8, 50)) == "kicked"
    assert [c.what for c in feed.vendor.calls] == ["events", "odds", "odds"]
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    w = feed.mem["work"]
    assert (w["date"], w["written_at"], w["horizon"]) == ("2026-03-12", order["written_at"],
                                                         order["horizon"])
    assert [g["game_id"] for g in w["games"]] == [PHI_DET, DEN_SAS] and order["games"] == []
    got = feed.view()
    assert got.state == "ok" and set(got.value["game_id"]) == {PHI_DET, DEN_SAS}
    assert _tick(feed, et(8, 51)) == "idle" and len(feed.vendor.odds_calls()) == 2


@pytest.mark.parametrize("files, bought", [
    ((), []),                                    # neither file: the date is not published
    (("features",), [DEN_EID]),                  # no projections file: the features file decides
    (("projections", "features"), [PHI_EID]),    # the projections file comes first
], ids=["unpublished", "features_only", "projections_first"])
def test_the_refresher_skips_an_unpublished_date_and_games_missing_from_its_file_m26(
        feed, tmp_path, files, bought):
    """m26: only a published date's games with rows are bought. The date's
    projections file decides, or its features file when there is none (the
    test store.load_projections makes); a game whose id is not in that file
    is never bought, all day; with neither file no tick makes a call, the
    free events call included."""
    day = tmp_path / "serve" / "2026-03-12"
    proj, feat = day / "projections_v1.parquet", day / "features_v1.parquet"
    if "projections" in files:
        _ids_file(proj, [PHI_DET])
    if "features" in files:
        _ids_file(feat, [DEN_SAS])
    _evening(feed, evening_order([proj, feat]))
    words = [_tick(feed, at) for at in (et(8, 50), et(9, 30), et(12, 0), et(18, 30), et(20, 30))]
    assert words[0] == ("kicked" if bought else "no_work")
    assert {c.event_id for c in feed.vendor.odds_calls()} == set(bought)
    if not bought:
        assert words == ["no_work"] * 5 and len(feed.vendor.calls) == 1   # the evening's events call only


def test_the_refresher_reads_only_the_game_id_column_once_per_file_version(feed, tmp_path, monkeypatch):
    """2.4.5: of the committed file only the game_id column is read, and an
    unchanged file (path, mtime_ns, size) is read once however many ticks
    see it. A rewritten one is read again, so a game the bot publishes later
    in the morning is bought; the features file behind it is never read."""
    proj = _ids_file(tmp_path / "projections_v1.parquet", [PHI_DET])
    feat = _ids_file(tmp_path / "features_v1.parquet", [PHI_DET, DEN_SAS])
    _evening(feed, evening_order([proj, feat]))
    reads = []
    real = pd.read_parquet

    def spy(path, *a, **k):
        reads.append((str(path), k.get("columns")))
        return real(path, *a, **k)
    monkeypatch.setattr(pd, "read_parquet", spy)
    for at in (et(7, 0), et(7, 1), et(7, 2)):
        assert _tick(feed, at) == "idle"
    assert reads == [(str(proj), ["game_id"])]
    _ids_file(proj, [PHI_DET, DEN_SAS])                       # the bot's next commit adds DEN at SAS
    info = proj.stat()
    os.utime(proj, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    assert _tick(feed, et(8, 50)) == "kicked"
    assert reads == [(str(proj), ["game_id"])] * 2
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    assert _tick(feed, et(8, 51)) == "idle" and len(reads) == 2


UNREADABLE_FILE = re.compile(r"oddsapi: today's published file is unreadable \((\w+)\); "
                             r"the refresher waits for a view")


def test_an_unreadable_published_file_buys_nothing_and_logs_its_type_once_a_day(feed, tmp_path):
    """2.4.5: the first of today's files that exists decides. When it cannot
    be read (a corrupt file, or a folder in its place, which is refused
    unopened), the tick buys nothing and waits for a view, the features file
    behind it unread, with one WARNING a day naming the exception's type
    only: never the path or the file's bytes."""
    proj = tmp_path / "projections_v1.parquet"
    proj.write_bytes(b"not a parquet file " + MARK.encode())
    feat = _ids_file(tmp_path / "features_v1.parquet", [PHI_DET, DEN_SAS])
    _evening(feed, evening_order([proj, feat]))
    for at in (et(8, 50), et(9, 0), et(12, 0)):
        assert _tick(feed, at) == "no_work", at
    said = [m for m in _messages(feed.logs) if "published file" in m]
    assert len(said) == 1 and UNREADABLE_FILE.fullmatch(said[0]), said
    proj.unlink()
    proj.mkdir()                                                # a folder where the file should be
    _evening(feed, evening_order([proj, feat]))
    assert _tick(feed, et(8, 50)) == "no_work"
    said = [m for m in _messages(feed.logs) if "published file" in m]
    assert said[1:] == ["oddsapi: today's published file is unreadable (not a regular file); "
                        "the refresher waits for a view"]
    assert feed.vendor.odds_calls() == [] and len(feed.vendor.events_calls()) == 2   # the two evenings'
    assert str(tmp_path) not in feed.logs.text and MARK not in feed.logs.text


def test_the_refresher_remembers_a_bounded_number_of_files(fresh_anchor, tmp_path):
    """The memory of committed files is bounded (_IDS_KEEP), the files read
    most recently kept, and the test reset empties it."""
    paths = [str(_ids_file(tmp_path / f"projections_{i}.parquet", [f"00225009{i:02d}"]))
             for i in range(live._IDS_KEEP + 3)]
    for p in paths:
        assert live._published_ids([p]) == (frozenset({f"00225009{paths.index(p):02d}"}), "")
    assert list(live._ids_seen) == paths[-live._IDS_KEEP:]
    live.reset_feed_anchor_for_tests()
    assert live._ids_seen == {}


def test_the_refresher_makes_no_call_when_nothing_is_due(feed):
    """MUTATION (design 2.4.6): a tick kicks only when a game of today is
    due, and an idle tick makes no network call, the free events call
    included. PHI at DET alone, the balance learned by a view at 08:00:
    idle to 08:49, the morning pull bought at 08:50, idle to the evening
    refresh at 6:30 PM (P = 2), bought then, and idle after it, inside the
    tip guard and after tip. The two kicks are the control, so the idle
    ticks are idle because nothing is due. A tick that kicks when nothing
    is due fails here."""
    work = work_order(games=(PHI_DET,))
    feed.kick(at=et(8, 0), work=work)
    assert [c.what for c in feed.vendor.calls] == ["events"]
    for at in (et(8, 1), et(8, 20), et(8, 49, 59)):
        assert _tick(feed, at) == "idle", at
    assert len(feed.vendor.calls) == 1
    assert _tick(feed, et(8, 50)) == "kicked"
    assert [c.what for c in feed.vendor.calls] == ["events", "events", "odds"]
    for at in (et(8, 51), et(9, 0), et(9, 30), et(12, 0), et(15, 30), et(18, 29, 59)):
        assert _tick(feed, at) == "idle", at
    assert len(feed.vendor.calls) == 3
    assert _tick(feed, et(18, 30)) == "kicked"
    assert [c.what for c in feed.vendor.calls] == ["events", "events", "odds", "events", "odds"]
    for at in (et(18, 31), et(18, 45), et(18, 55), et(19, 0), et(23, 30)):
        assert _tick(feed, at) == "idle", at
    assert len(feed.vendor.calls) == 5
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, PHI_EID]


def test_with_the_balance_unknown_the_refresher_kicks_from_0850_to_the_tip_guard(feed):
    """2.4.6: with the balance not known yet (here the 08:00 view's free
    events call timed out: no answer, so no usage headers), every tick from
    08:50 kicks, and the walk's free call learns the balance and buys,
    never more often than its backoff (90 s, then doubling) allows. Before
    08:50, and for a game inside its tip guard or after tip, the tick is
    idle and makes no call."""
    feed.vendor.script["events"] = [H.VendorTransportError("timeout"), H.VendorTransportError("timeout")]
    feed.kick(at=et(8, 0))
    b = feed.mem["budget"]
    assert b["remaining"] is None and len(feed.vendor.calls) == 1
    assert _tick(feed, et(8, 49, 59)) == "idle" and len(feed.vendor.calls) == 1
    assert _tick(feed, et(8, 50)) == "kicked"                    # the backoff is over: one free call, failing
    assert [c.what for c in feed.vendor.calls] == ["events", "events"]
    for s in (30, 90, 179):                                       # its second failure backs off 3 minutes
        assert _tick(feed, et(8, 50) + dt.timedelta(seconds=s)) == "kicked"
    assert len(feed.vendor.calls) == 2
    assert _tick(feed, et(8, 53)) == "kicked"
    assert [c.what for c in feed.vendor.calls] == ["events"] * 3 + ["odds"] * 2
    assert (b["remaining"], b["used"]) == (348, 152)
    assert _tick(feed, et(8, 54)) == "idle" and len(feed.vendor.calls) == 5
    late = live.new_feed_memory(et(8, 0), had_ledger=False)      # born before 09:50: no brake (3.4)
    late["work"] = work_order(games=(PHI_DET,))                  # as a view's kick stores it; no balance
    for at in (PHI_TIP - 5 * MINUTE, PHI_TIP - 4 * MINUTE, PHI_TIP + 30 * MINUTE):
        assert _tick(feed, at, mem=late) == "idle", at
    assert len(feed.vendor.calls) == 5
    assert _tick(feed, PHI_TIP - 6 * MINUTE, mem=late) == "kicked"            # outside the guard: the control
    assert [c.what for c in feed.vendor.calls[5:]] == ["events", "odds"]


def test_a_tick_reads_the_key_and_the_settings_on_every_call(feed, monkeypatch):
    """RC K3, G1: "off" with no key, the kill switch off (an unreadable value
    too, B6) or BUCKETVALUE_ODDS_REFRESHER off, conftest's "0" in the process
    environment included, each read on this call: no call, no task and no
    change to the ledger. With the key and both switches on, it kicks."""
    feed.mem["work"] = work_order()
    before = _plain(feed.mem)
    monkeypatch.setenv(live.KEY_ENV, KEY)
    assert live.refresher_tick(feed.mem, pool=feed.pool, now=MORNING) == "off"     # environ None
    for env in ({}, {live.KEY_ENV: "   "}, {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_ENABLED": "0"},
                {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_ENABLED": "maybe"},
                {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_REFRESHER": "0"},
                {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_REFRESHER": "off"}):
        assert _tick(feed, MORNING, env=env) == "off", env
    assert feed.vendor.calls == [] and feed.pool.names == [] and _plain(feed.mem) == before
    with pytest.raises(ValueError):                              # a naive clock is refused, never guessed
        live.refresher_tick(feed.mem, environ=feed.env, pool=feed.pool, now=dt.datetime(2026, 3, 12, 8, 55))
    assert _tick(feed, MORNING) == "kicked" and len(feed.vendor.odds_calls()) == 2


def test_no_order_an_unreadable_or_older_than_36_hours_order_or_no_game_today_is_no_work(feed):
    """2.4.5: the refresher waits for a view when there is no work order,
    when its written_at is missing, unreadable or more than 36 hours before
    now, or when it holds no game of today, and makes no call. An order
    exactly 36 hours old is still used."""
    proj = [SERVE / "projections.parquet"]
    assert _tick(feed, MORNING) == "no_work"                     # a new ledger holds no order
    limit = MORNING_UTC - live.ORDER_MAX_AGE
    for written in ("", "yesterday", None, 17, (limit - dt.timedelta(seconds=1)).isoformat()):
        order = evening_order(proj)
        order["written_at"] = written
        feed.mem["work"] = order
        assert _tick(feed, MORNING) == "no_work", written
    feed.mem["work"] = {k: v for k, v in evening_order(proj).items() if k != "written_at"}
    assert _tick(feed, MORNING) == "no_work"
    feed.mem["work"] = {**work_order(), "games": [dict(g, game_date="2026-03-13") for g in TODAY]}
    assert _tick(feed, MORNING) == "no_work"                     # today's order, no game of today
    feed.mem["work"] = {**evening_order(proj), "horizon": [g for g in HORIZON if g["game_date"] != "2026-03-12"]}
    assert _tick(feed, MORNING) == "no_work"                     # the evening's order, no row of today
    assert feed.vendor.calls == []
    feed.mem["work"] = evening_order(proj, written=limit)
    assert _tick(feed, MORNING) == "kicked" and len(feed.vendor.odds_calls()) == 2


def test_horizon_rows_without_teams_are_not_kicked_for_and_say_so_once_a_day(feed):
    """The walk matches vendor events by the date and the unordered team
    pair, so a horizon row without away and home (the bare shape DESIGN
    2.4.5 lists) can never be bought. The tick leaves such rows out, makes
    no call when nothing else is left, and logs one WARNING a day that the
    refresher waits for a view; a row of the same order with its teams is
    still bought."""
    _evening(feed, evening_order([SERVE / "projections.parquet"], teams=False))
    for at in (et(8, 50), et(9, 0), et(12, 0)):
        assert _tick(feed, at) == "no_work", at
    assert len(feed.vendor.calls) == 1
    said = [m for m in _messages(feed.logs) if "no teams" in m]
    assert said == ["oddsapi: 2 game(s) of today have no teams in the work order's horizon; "
                    "the refresher waits for a view"]
    order = evening_order([SERVE / "projections.parquet"])
    phi = next(r for r in order["horizon"] if r["game_id"] == PHI_DET)
    del phi["away"], phi["home"]
    _evening(feed, order)
    assert _tick(feed, et(8, 50)) == "kicked"
    assert [c.event_id for c in feed.vendor.odds_calls()] == [DEN_EID]
    assert [m for m in _messages(feed.logs) if "no teams" in m][1:] == [
        "oddsapi: 1 game(s) of today have no teams in the work order's horizon; "
        "the refresher waits for a view"]


def test_a_game_whose_call_is_already_queued_is_not_kicked_for_again(feed):
    """M8 in the refresher: while a game's call waits in the pool, the tick
    does not count it as due, so it starts no walk for it; once the calls
    land, nothing is due either."""
    pool = HeldPool()
    feed.kick(pool=pool)
    pool.run("_walk")                                            # events; both calls queued and held
    assert feed.mem["queued"] == {PHI_EID, DEN_EID}
    assert _tick(feed, MORNING + MINUTE, pool=pool) == "idle"
    pool.release(1)                                              # PHI at DET's call lands; DEN at SAS waits
    assert _tick(feed, MORNING + 2 * MINUTE, pool=pool) == "idle"
    pool.release()
    assert _tick(feed, MORNING + 3 * MINUTE, pool=pool) == "idle"
    assert pool.names == ["_walk", "_pull_game", "_pull_game"]
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]


def test_an_old_refresher_loop_exits_on_a_version_change(loops, monkeypatch):
    """2.4.6: a loop returns at its next wake once the anchor names another
    REFRESHER_VERSION (a newer loop took over), its own stop Event never
    set, and the same once the anchor is gone. Until then each tick hands
    the anchor's ledger to live.refresher_tick."""
    seen: list = []
    monkeypatch.setattr(live, "refresher_tick", lambda mem, **k: seen.append(mem) or "idle")
    anchor = live._anchor_module()
    anchor.refresher_version = live.REFRESHER_VERSION
    t, stop = loops(live.REFRESHER_VERSION)
    assert _until(lambda: len(seen) >= 2)
    mem = live.feed_anchor()
    assert all(m is mem for m in list(seen))
    anchor.refresher_version = live.REFRESHER_VERSION + 1        # a newer loop took over
    t.join(2.0)
    assert not t.is_alive() and not stop.is_set()
    ticks = len(seen)
    newer, newer_stop = loops(live.REFRESHER_VERSION + 1)
    assert _until(lambda: len(seen) >= ticks + 2)
    sys.modules.pop(live.ANCHOR_NAME)                            # the anchor is gone
    newer.join(2.0)
    assert not newer.is_alive() and not newer_stop.is_set()


def test_the_refresher_loop_ticks_the_current_live_module_after_a_purge(loops, monkeypatch, caplog):
    """2.4.6: after a code push the running loop is OLD code, so each tick
    looks up bucketlib.live in sys.modules and calls THAT module's
    refresher_tick on its own feed_anchor(). A new module under the name
    gets the next tick; while the name is purged and not re-imported, the
    loop goes round without a tick and carries on; a tick that raises is
    logged by its type only, and the loop goes on."""
    caplog.set_level(logging.DEBUG, logger="bucketvalue")
    calls: list = []

    def ours(mem, **k):
        calls.append("ours")
        if calls.count("ours") == 1:
            raise RuntimeError(MARK + " " + KEY)
        return "idle"
    monkeypatch.setattr(live, "refresher_tick", ours)
    ledger = {"a": "the pushed module's ledger"}
    pushed = types.SimpleNamespace(refresher_tick=lambda mem, **k: calls.append(("pushed", mem)) or "idle",
                                   feed_anchor=lambda: ledger)
    live._anchor_module().refresher_version = live.REFRESHER_VERSION
    t, stop = loops(live.REFRESHER_VERSION)
    try:
        assert _until(lambda: calls.count("ours") >= 2)           # the failed tick did not stop it
        sys.modules["bucketlib.live"] = pushed                    # a push: the name is a new module
        assert _until(lambda: ("pushed", ledger) in calls)
        del sys.modules["bucketlib.live"]                         # purged, not re-imported yet
        w = stop.waits
        assert _until(lambda: stop.waits >= w + 1)                # a tick in flight has ended
        n = len(calls)
        assert _until(lambda: stop.waits >= w + 3)                # two rounds with no module
        assert len(calls) == n and t.is_alive()
        sys.modules["bucketlib.live"] = live                      # re-imported
        k = calls.count("ours")
        assert _until(lambda: calls.count("ours") > k)
    finally:
        sys.modules["bucketlib.live"] = live
        stop.set()
        t.join(2.0)
    assert not t.is_alive()
    assert [m for m in _messages(caplog) if "refresher" in m] == [
        "oddsapi: refresher tick failed (RuntimeError)"]
    assert MARK not in caplog.text and KEY not in caplog.text


def test_the_refresher_never_starts_with_the_setting_off_the_switch_off_or_no_key(fresh_anchor, monkeypatch):
    """2.4.6 and 4.1: conftest sets BUCKETVALUE_ODDS_REFRESHER "0" for the
    session, so with a key in the process environment ensure_refresher()
    starts nothing; nor with the setting off in a mapping, the kill switch
    off, or no key. None of them so much as makes the anchor: with no key,
    nothing changes (M9)."""
    assert os.environ.get("BUCKETVALUE_ODDS_REFRESHER") == "0"
    monkeypatch.setenv(live.KEY_ENV, KEY)
    live.ensure_refresher()
    for env in ({live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_REFRESHER": "0"},
                {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_REFRESHER": "no"},
                {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_ENABLED": "0"},
                {live.KEY_ENV: KEY, "BUCKETVALUE_ODDS_ENABLED": "maybe"},
                {"BUCKETVALUE_ODDS_REFRESHER": "1"}, {}):
        live.ensure_refresher(env)
    assert _refresher_threads() == [] and live.ANCHOR_NAME not in sys.modules


def test_the_refresher_starts_once_per_process_and_a_dead_one_is_restarted(refresher_on, caplog):
    """2.4.6: with the feed and the refresher on, ensure_refresher() starts
    one daemon thread; the thread, its stop Event and its version live in
    the anchor, as stdlib objects and an int (B4), with no key. Later calls
    leave it alone. Once it has stopped, the next call starts a new one."""
    caplog.set_level(logging.INFO, logger="bucketvalue")
    live.ensure_refresher()
    anchor = sys.modules[live.ANCHOR_NAME]
    first, stop = anchor.refresher, anchor.refresher_stop
    try:
        assert type(first) is threading.Thread and type(stop) is threading.Event
        assert first.is_alive() and first.daemon and first.name == live.REFRESHER_THREAD
        assert anchor.refresher_version == live.REFRESHER_VERSION and not stop.is_set()
        assert KEY not in _text(anchor)
        for _ in range(3):
            live.ensure_refresher()
        assert anchor.refresher is first and _refresher_threads() == [first]
        stop.set()                                                # the loop stops, as if it had died
        first.join(2.0)
        assert not first.is_alive()
        live.ensure_refresher()
        second = anchor.refresher
        assert second is not first and second.is_alive() and _refresher_threads() == [second]
        assert anchor.refresher_stop is not stop and not anchor.refresher_stop.is_set()
    finally:
        _stop_refresher(anchor)
    assert _refresher_threads() == []
    assert [m for m in _messages(caplog) if "refresher" in m] == [
        "oddsapi: refresher started", "oddsapi: refresher restarted (the last loop had stopped)"]


def test_a_refresher_version_change_stops_the_old_loop_and_starts_a_new_one(refresher_on, monkeypatch):
    """2.4.6: when REFRESHER_VERSION changes, the new code sets the old
    loop's stop Event, so it ends at once, and starts a new loop that
    carries the new version; the next call leaves that one alone."""
    live.ensure_refresher()
    anchor = sys.modules[live.ANCHOR_NAME]
    old, old_stop = anchor.refresher, anchor.refresher_stop
    monkeypatch.setattr(live, "REFRESHER_VERSION", live.REFRESHER_VERSION + 1)
    live.ensure_refresher()
    try:
        assert old_stop.is_set()
        old.join(2.0)
        assert not old.is_alive()
        new = anchor.refresher
        assert new is not old and new.is_alive() and anchor.refresher_version == live.REFRESHER_VERSION
        live.ensure_refresher()
        assert anchor.refresher is new and _refresher_threads() == [new]
    finally:
        _stop_refresher(anchor)
        old_stop.set()
        old.join(2.0)
    assert _refresher_threads() == []


def test_a_refresher_that_cannot_start_logs_its_type_and_never_raises(refresher_on, monkeypatch, caplog):
    """ensure_refresher runs on a page view (store.feed_start): whatever goes
    wrong is logged by its type only and never reaches the page."""
    def broken():
        raise RuntimeError(MARK + " " + KEY)
    monkeypatch.setattr(live, "_anchor_module", broken)
    live.ensure_refresher()
    assert _refresher_threads() == []
    assert [m for m in _messages(caplog) if "refresher" in m] == [
        "oddsapi: refresher start failed (RuntimeError)"]
    assert MARK not in caplog.text and KEY not in caplog.text


def test_the_refresher_never_touches_streamlit_the_store_or_the_feed_tables():
    """2.4.6: only live functions run off the script thread. live.py imports
    no streamlit, store, props or props_ui at all, and no refresher function
    names st, a Streamlit cache, session state or a feed-table builder."""
    tree = ast.parse(Path(live.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {(node.module or "").split(".")[0]} | {a.name for a in node.names}
    assert not imported & {"streamlit", "store", "props", "props_ui", "pmf", "app"}
    names = {"ensure_refresher", "_refresher_loop", "refresher_tick", "_tick_order", "_tick_due",
             "_read_game_ids", "_published_ids"}
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in found} == names
    for fn in found:
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert not used & {"st", "streamlit", "store", "props"}, fn.name
        assert not attrs & {"cache_data", "cache_resource", "session_state", "prepare_feed",
                            "compare_books", "feed_tables"}, fn.name
    assert live.TICK_WORDS == ("off", "no_work", "idle", "kicked")


# -- Build 4's fix pass: build amendments B18 to B20, review L1-1, L2-1, L2-2, L2-4, L3-10,
# and the mutation campaign's survivors (LV01 to LV24) ---------------------------------------
THREE = ("player_points", "player_rebounds", "player_assists")
THREE_ENV = {"BUCKETVALUE_ODDS_MARKETS": ",".join(THREE)}


def _with_prizepicks(payload: dict) -> dict:
    """DEN at SAS's payload once PrizePicks has posted it: DraftKings' points
    lines as PrizePicks', at the indicative -137."""
    posted = copy.deepcopy(next(b for b in payload["bookmakers"] if b["key"] == "draftkings"))
    posted.update(key="prizepicks", title="PrizePicks")
    posted["markets"] = [m for m in posted["markets"] if m["key"] == "player_points"]
    for m in posted["markets"]:
        for oc in m["outcomes"]:
            oc["price"] = -137
    return {**payload, "bookmakers": payload["bookmakers"] + [posted]}


def test_a_rebuy_asks_the_core_markets_only_b20(feed):
    """Build amendment B20. With three markets configured and a balance
    whose plan is "full", the default ledger (born at 08:55 in a period
    that was spending) cannot tell its first pulls from re-buys of what the
    process before it bought (B11): each asks the core markets only, points.
    The same morning in a ledger born at 08:00, which saw it, its first
    pulls ask the plan's three markets (B1: the floor holds at that cost)."""
    feed.env = {**feed.env, **THREE_ENV}
    feed.vendor.remaining = 5000
    feed.kick()
    with feed.mem["lock"]:
        plan = live._budget(feed.mem, O.read_settings(feed.env), MORNING_UTC).plan
    assert (plan.mode, plan.markets) == ("full", THREE)                    # the premise
    assert [c.markets for c in feed.vendor.odds_calls()] == [PTS, PTS]
    assert feed.mem["budget"]["pending"] == 0
    saw = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.kick(mem=saw)
    assert [c.markets for c in feed.vendor.odds_calls()][2:] == [THREE, THREE]


def test_a_budget_hold_is_logged_once_a_day_per_guard_b19(feed):
    """Build amendment B19. The short morning of the F4 test above: both of
    today's re-buys are refused by the floor at every kick; one fixed
    WARNING says so for the day. Each guard (the reserve, the floor, the
    ceiling) has its own line, at most once per ET day; a refusal that is
    no budget hold (the rate cap) has none."""
    feed.vendor.remaining, feed.vendor.used = 100, 100
    late = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    for at in (MORNING, et(9, 30), et(12, 20), et(18, 0)):
        feed.kick(at=at, mem=late)
    held = [m for m in _messages(feed.logs) if "budget held" in m]
    assert held == ["oddsapi: budget held a pull today (floor)"]
    rec = next(r for r in feed.logs.records if "budget held" in r.getMessage())
    assert rec.levelno == logging.WARNING
    mem = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    with mem["lock"]:
        for code, at in (("reserve", et(9, 0)), ("ceiling", et(9, 1)), ("reserve", et(23, 59)),
                         ("rate_cap", et(10, 0)), ("ceiling", et(0, 1, day=13)),
                         ("floor", et(0, 2, day=13)), ("floor", et(22, 0, day=13))):
            live._refused(mem, code, at.astimezone(UTC))
    held = [m for m in _messages(feed.logs) if "budget held" in m][1:]
    assert held == ["oddsapi: budget held a pull today (reserve)",
                    "oddsapi: budget held a pull today (ceiling)",
                    "oddsapi: budget held a pull today (ceiling)",
                    "oddsapi: budget held a pull today (floor)"]
    assert KEY not in feed.logs.text


def test_after_tip_a_game_never_answered_with_lines_is_not_none_posted_b18(feed):
    """Build amendment B18 in the view: "none_posted" only for a game whose
    last answer had no bookmaker entry. A game whose only pull failed reads
    "not_yet" after tip (the Game page: the odds feed did not load it), one
    whose books answered empty "none_posted", and a held one "paused"."""
    feed.vendor.script["odds"] = [H.VendorTransportError("timeout"),
                                  lambda call, v: v.response(200, {**v.odds[call.event_id],
                                                                   "bookmakers": []}, last=0)]
    feed.kick()
    after = live.feed_view(DATE, mem=feed.mem, wait_s=0, now=et(21, 30))
    assert after.per_game[PHI_DET]["reason"] == "not_yet"        # timed out: never loaded
    assert after.per_game[DEN_SAS]["reason"] == "none_posted"    # its books posted nothing


def test_a_junk_tip_in_the_events_answer_costs_only_its_event_and_a_parse_that_raises_backs_off_l1_1(
        feed, monkeypatch):
    """Build 4's review, L1-1. The vendor lists a placeholder event whose
    commence_time is Go's zero time: the parser drops it, and the walk buys
    today's games as ever. And should the parser raise anyway, the events
    call is settled as a bad payload: the balance is learned, the call
    backs off (90 s), the view reads down, never a dead walk that calls
    /events again on every view."""
    feed.vendor.events = feed.vendor.events + [
        {**feed.vendor.events[0], "id": "f" * 32, "commence_time": "0001-01-01T00:00:00Z"},
        {**feed.vendor.events[0], "id": "e" * 32, "commence_time": "9999-12-31T23:00:00-12:00"}]
    feed.kick()
    assert [c.what for c in feed.vendor.calls] == ["events", "odds", "odds"]
    assert "walk failed" not in feed.logs.text

    def broken(*a, **k):
        raise OverflowError(MARK)
    monkeypatch.setattr(O, "parse_events", broken)
    mem = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.kick(mem=mem)
    assert [c.what for c in feed.vendor.calls][3:] == ["events"]
    b = mem["budget"]
    assert (b["remaining"], b["used"]) == (348, 152)                 # the balance is learned
    assert mem["events"] is None and mem["events_fail"][1:] == ("bad_payload", 1)
    assert mem["events_fail"][0] == MORNING_UTC + dt.timedelta(seconds=90)
    assert "oddsapi events failed: bad_payload" in _messages(feed.logs)
    assert "walk failed" not in feed.logs.text and MARK not in feed.logs.text
    assert feed.view(mem=mem).state == "down"
    feed.kick(at=MORNING + dt.timedelta(seconds=60), mem=mem)        # backing off: no call
    assert len(feed.vendor.events_calls()) == 2


def test_timed_out_top_ups_hours_apart_cost_the_game_nothing_and_never_trip_the_breaker_l2_1(feed):
    """Build 4's review, L2-1, with the PrizePicks top-ups ON (the default).
    DEN at SAS came back without PrizePicks at 08:55 (P = 2), so it polls
    PrizePicks alone for free. Three of those polls time out, hours apart,
    with clean empty answers between. A timeout of a call expected to be
    free still takes its worst case off the balance, but it is no pull, so
    the game keeps its top-up; and no timeout grows the streak (build
    amendment B22; a clean answer, an empty one too, ends a run of billed
    junk: the next test), so the day breaker never trips. PrizePicks posts
    late in the morning and its line lands; PHI at DET's evening refresh
    comes as planned."""
    feed.kick()
    den = feed.mem["games"][DEN_EID]
    assert (den["pulls"], den["pp_missing"]) == (1, True)
    timeout = H.VendorTransportError("timeout")
    for hh in (9, 10, 11):
        feed.vendor.script["odds"] = [timeout]
        feed.kick(at=et(hh, 5))
        assert feed.vendor.odds_calls()[-1].books == ("prizepicks",)
        assert (den["pulls"], feed.mem["streak"]) == (1, 0), hh       # no pull; no streak (B22)
        assert den["fail_until"] == et(hh, 5).astimezone(UTC) + OB.backoff(den["fails"]), hh
        feed.kick(at=et(hh, 15))                                      # PrizePicks: nothing yet
        assert (den["pulls"], feed.mem["streak"]) == (1, 0), hh
    assert feed.mem["breaker_day"] is None
    assert feed.mem["budget"]["remaining"] == 348 - 3                 # each timeout's worst case
    feed.vendor.odds[DEN_EID] = _with_prizepicks(feed.vendor.odds[DEN_EID])
    feed.kick(at=et(11, 25))
    assert (den["pulls"], den["pp_missing"]) == (2, False)
    assert "prizepicks" in set(den["frame"]["book"])
    feed.kick(at=et(18, 30))
    assert feed.vendor.odds_calls()[-1].event_id == PHI_EID and feed.mem["streak"] == 0


def test_a_pull_after_an_empty_answer_that_times_out_is_no_pull_l2_1(feed):
    """L2-1's other free call: a first pull whose last answer was empty (no
    book had posted) times out. It is no pull; after its 90 s backoff the
    first pull is due again, and lands."""
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    work = work_order(games=(PHI_DET,))

    def empty(call, vendor):
        return vendor.response(200, {**copy.deepcopy(vendor.odds[call.event_id]), "bookmakers": []},
                               last=0)
    feed.vendor.script["odds"] = [empty, H.VendorTransportError("timeout")]
    feed.kick(at=et(8, 50), work=work)
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["empty_next"]) == (0, et(9, 0).astimezone(UTC))
    feed.kick(at=et(9, 0), work=work)
    assert (e["pulls"], e["fails"], e["first_at"], feed.mem["streak"]) == (0, 1, None, 0)   # B22
    assert feed.mem["budget"]["remaining"] == 349                       # its worst case
    feed.kick(at=et(9, 1, 30), work=work)
    assert len(feed.vendor.odds_calls()) == 3 and (e["pulls"], e["fails"]) == (1, 0)
    assert e["first_at"] == et(9, 1, 30).astimezone(UTC)


def _spend_kinds(monkeypatch) -> list:
    """(kind, true_first) of every oddsbudget.may_spend call live makes: the
    governor's own function, recorded on its way through."""
    seen: list = []
    real = OB.may_spend

    def spy(*a, **k):
        seen.append((k["kind"], k.get("true_first", False)))
        return real(*a, **k)
    monkeypatch.setattr(OB, "may_spend", spy)
    return seen


def _plan_now(mem: dict, env: dict, at: dt.datetime) -> OB.Plan:
    with mem["lock"]:
        return live._budget(mem, O.read_settings(env), at.astimezone(UTC)).plan


def test_a_cold_start_after_0950_in_a_spending_period_re_buys_todays_games_at_once_b21(
        feed, monkeypatch):
    """Build amendment B21 (build 4's review, L2-2, which pinned the brake
    leaving today's games without lines until tip minus 30 minutes). A
    ledger born at 10:00 in a period that was spending (150 used against 84
    earlier games), whatever its predecessor did: the brake still counts
    today's games as pulled at its birth, and logs it as a WARNING with its
    count (the default log level shows it). Each braked game with no lines
    here keeps one re-buy at once, while P >= 2 (here 2) and before its last
    refresh time: both are bought in the 10:00 walk, in tip order, as
    re-buys (the floor and the ceiling checked, the core markets), and the
    view reads ok with their lines. That re-buy is each game's second pull,
    so P is spent: a kick every 5 minutes all day buys neither again, the
    evening refresh included."""
    kinds = _spend_kinds(monkeypatch)
    born = et(10, 0)
    feed.mem = live.new_feed_memory(born, had_ledger=False)
    feed.kick(at=born)
    assert [c.what for c in feed.vendor.calls] == ["events", "odds", "odds"]
    assert [(c.event_id, c.books, c.markets) for c in feed.vendor.odds_calls()] == [
        (PHI_EID, FIVE, PTS), (DEN_EID, FIVE, PTS)]
    assert kinds == [("rebuy", False), ("rebuy", False)]
    rec = next(r for r in feed.logs.records if "memory-loss brake" in r.getMessage())
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == ("oddsapi: memory-loss brake: 2 game(s) of today counted as pulled "
                                "at the ledger's birth")
    assert _plan_now(feed.mem, feed.env, born).pulls_per_game == 2
    got = feed.view()
    assert got.state == "ok" and set(got.value["game_id"]) == {PHI_DET, DEN_SAS}
    for eid in (PHI_EID, DEN_EID):
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["braked"], e["first_at"], e["last_at"]) == (
            2, True, born.astimezone(UTC), born.astimezone(UTC)), eid
    for at in [t for t in DAY_KICKS if t > born]:
        feed.kick(at=at)
    assert len(feed.vendor.odds_calls()) == 2                       # none bought twice in this ledger


def test_a_braked_games_re_buy_asks_the_core_markets_only_b21(feed):
    """B21 with B20: with three markets configured and a balance whose plan
    is "full" (5,000 left), the braked games' re-buys at 10:00 ask the core
    markets, points, never the plan's three."""
    feed.env = {**feed.env, **THREE_ENV}
    feed.vendor.remaining = 5000
    born = et(10, 0)
    feed.mem = live.new_feed_memory(born, had_ledger=False)
    feed.kick(at=born)
    plan = _plan_now(feed.mem, feed.env, born)
    assert (plan.mode, plan.markets) == ("full", THREE) and plan.pulls_per_game >= 2   # the premise
    assert [(c.event_id, c.markets) for c in feed.vendor.odds_calls()] == [(PHI_EID, PTS), (DEN_EID, PTS)]
    assert all(feed.mem["games"][eid]["braked"] for eid in (PHI_EID, DEN_EID))


def test_with_p_1_a_cold_start_after_0950_buys_nothing_more_b21(feed):
    """B21's P >= 2. 250 left and 100 used (A 350, S 25): the plan gives each
    game one pull (P = 1), the period was spending (200 >= 84), and the brake
    counts today's games as pulled at the 10:00 birth. That assumed pull is
    each game's only one, so a kick every 5 minutes all day buys nothing: no
    re-buy, and no evening refresh. The view says not yet."""
    feed.vendor.remaining, feed.vendor.used = 250, 100
    born = et(10, 0)
    feed.mem = live.new_feed_memory(born, had_ledger=False)
    feed.kick(at=born)
    plan = _plan_now(feed.mem, feed.env, born)
    assert (plan.mode, plan.pulls_per_game) == ("full", 1)                         # the premise
    assert feed.mem["brake_day"] == DATE
    assert all(feed.mem["games"][eid]["braked"] for eid in (PHI_EID, DEN_EID))
    for at in [t for t in DAY_KICKS if t > born]:
        feed.kick(at=at)
    assert feed.vendor.odds_calls() == []
    got = live.feed_view(DATE, mem=feed.mem, wait_s=0, now=et(12, 0))
    assert (got.state, got.value.empty) == ("idle", True)
    assert {gid: got.per_game[gid]["reason"] for gid in (PHI_DET, DEN_SAS)} == {
        PHI_DET: "not_yet", DEN_SAS: "not_yet"}


def test_a_crash_loop_after_0950_re_buys_each_game_once_a_restart_until_p_is_one_b21(feed):
    """B21's bound in live: restarts every 10 minutes from 10:00 to 14:00
    (25 new ledgers, each a process start in a period that was spending).
    Each restart re-buys each of today's games at most once (the brake's
    pull and the re-buy are its two pulls), and only while the plan still
    gives each game two pulls: the re-buys spend what the plan holds, so
    after the 24th restart (48 credits, 350 left falling to 302) P is 1 and
    the 25th buys nothing. Every re-buy is floor-checked, and the floor's
    promise holds throughout: the balance above the reserve still covers
    the morning pull of every game of the horizon waiting for one (the floor
    never has to refuse here: P under 2 stops the loop first). The ceiling
    holds too."""
    t, bought, kept = et(10, 0), [], []
    s = O.read_settings(feed.env)
    for _ in range(25):
        feed.mem = live.new_feed_memory(t, had_ledger=False)       # a restart: a new process's ledger
        n = len(feed.vendor.odds_calls())
        feed.kick(at=t)
        bought.append(len(feed.vendor.odds_calls()) - n)
        b = feed.mem["budget"]
        with feed.mem["lock"]:
            view = live._budget(feed.mem, s, t.astimezone(UTC))
            waiting = OB.waiting_first(view.horizon, view.slots, t.astimezone(UTC), view.period)
        kept.append((b["remaining"] - view.plan.reserve >= len(s.core) * waiting,
                     b["used"] <= view.plan.ceiling))
        assert all(feed.mem["games"][eid]["pulls"] <= 2 for eid in (PHI_EID, DEN_EID))
        t += 10 * MINUTE
    assert bought == [2] * 24 + [0]
    assert (feed.vendor.remaining, feed.vendor.used) == (302, 198)
    assert all(floor and ceiling for floor, ceiling in kept)
    assert _plan_now(feed.mem, feed.env, t).pulls_per_game == 1


def test_an_entry_from_before_b21_reads_as_not_braked_and_a_bad_flag_is_malformed_b21():
    """B21's flag on a game's ledger entry: new entries say False; an entry
    without it (a ledger kept across a code push, which no version check
    sees) reads as not braked and stays valid, so it is neither dropped nor
    bought again; anything but a bool makes the entry malformed (m25)."""
    entry = live._new_entry(PHI_DET, "2026-03-12", PHI_TIP, PHI_TIP)
    assert entry["braked"] is False and live._slot_of(entry, PHI_EID).braked is False
    old = {k: v for k, v in entry.items() if k != "braked"}
    assert live._slot_of(old, PHI_EID) is not None and live._slot_of(old, PHI_EID).braked is False
    assert live._slot_of({**entry, "braked": True}, PHI_EID).braked is True
    for bad in (None, 1, "True"):
        assert live._slot_of({**entry, "braked": bad}, PHI_EID) is None, bad


def test_three_timeouts_in_one_0850_batch_leave_the_breaker_off_and_every_game_is_retried_b22(feed):
    """Build amendment B22 (build 4's review, L2-1's residual). At 08:50 four
    games are due (a ledger born at 08:00), three of them at 7 PM, and their
    three calls time out. Each timeout still takes its worst case off the
    balance, counts as a pull (3.5) and backs its game off 90 s, but no
    answer came back, so none grows the streak: the day breaker stays off,
    and DEN at SAS's call, the fourth, goes ahead in the same walk. Each
    timed-out game's pulls brought no lines, so at 08:51:30, its backoff's
    end, it is retried (build amendment B12), and PHI at DET's lines land.
    (Before B22 the three timeouts tripped the breaker: no fourth call, no
    retry and no lines all day.)"""
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.work["games"] += [dict(g) for g in MORE]
    timeout = H.VendorTransportError("timeout")

    def nothing_yet(call, vendor):
        event = next(e for e in vendor.events if e["id"] == call.event_id)
        return vendor.response(200, {**copy.deepcopy(event), "bookmakers": []}, last=0)
    feed.vendor.script["odds"] = [timeout] * 3
    feed.kick(at=et(8, 50))
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, IND_EID, ORL_EID, DEN_EID]
    assert (feed.mem["streak"], feed.mem["breaker_day"]) == (0, None)
    for eid in (PHI_EID, IND_EID, ORL_EID):
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["fails"], e["fail_until"], e["frame"]) == (
            1, 1, et(8, 51, 30).astimezone(UTC), None), eid
    assert feed.mem["games"][DEN_EID]["frame"] is not None
    assert feed.mem["budget"]["remaining"] == 350 - 3                     # three worst cases
    feed.vendor.script["odds"] = [lambda call, vendor: vendor.answer(call), nothing_yet, nothing_yet]
    feed.kick(at=et(8, 51, 29))                                         # still backing off
    assert len(feed.vendor.odds_calls()) == 4
    feed.kick(at=et(8, 51, 30))
    assert [c.event_id for c in feed.vendor.odds_calls()[4:]] == [PHI_EID, IND_EID, ORL_EID]
    assert (feed.mem["streak"], feed.mem["breaker_day"]) == (0, None)
    phi = feed.mem["games"][PHI_EID]
    assert (phi["pulls"], phi["fails"]) == (2, 0) and phi["frame"] is not None
    assert set(feed.view().value["game_id"]) == {PHI_DET, DEN_SAS}


def test_a_clean_empty_answer_ends_a_run_of_billed_junk_l2_1_b22(feed):
    """Build 4's review, L2-1, under B22: the streak counts billed junk, an
    answer that came back, billed, and of no use; a clean answer, an empty
    one too, ends a run of it. PHI at DET answers billed junk at 08:55 (the
    streak 1, seen as DEN at SAS's call goes out); DEN at SAS answers that
    nothing is posted yet (free), which ends the run."""
    seen = []

    def empty_after_junk(call, vendor):
        with feed.mem["lock"]:
            seen.append(feed.mem["streak"])
        return vendor.response(200, {**copy.deepcopy(vendor.odds[call.event_id]), "bookmakers": []},
                               last=0)
    feed.vendor.script["odds"] = [answer_with(200, last=1, body={}), empty_after_junk]
    feed.kick()
    assert seen == [1] and feed.mem["streak"] == 0 and feed.mem["breaker_day"] is None
    assert feed.mem["games"][DEN_EID]["empty_next"] == MORNING_UTC + OB.EMPTY_RETRY


def test_a_true_first_pull_skips_the_ceiling_and_every_other_pull_keeps_it_b23(feed, monkeypatch):
    """Build amendment B23 in live (build 4's review, L2-3). An outside drain
    has left 100 credits and 400 used on Mar 12, so today's spending ceiling
    (290) is passed. A ledger born at 08:00 saw the morning: its 08:50
    first pulls of both games are true first pulls, which skip the ceiling
    (and, B1, the floor), and are bought. A ledger born at 08:55 in that
    spending period cannot tell its pulls from re-buys (B11): no true first
    pull, and the floor holds them. And on a quiet account's go-live morning
    (20 used: the period was not spending) a ledger born at 08:55 makes the
    first pulls of first_kind's usage-test clause: PHI at DET's is bought,
    its answer shows an outside drain (400 used), and DEN at SAS's first
    pull is held by the ceiling, which still holds such a first pull."""
    kinds = _spend_kinds(monkeypatch)
    feed.vendor.remaining, feed.vendor.used = 100, 400
    saw = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.kick(at=et(8, 50), mem=saw)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    assert kinds == [("first", True), ("first", True)]
    plan = _plan_now(saw, feed.env, et(8, 50))
    assert (plan.mode, plan.ceiling, saw["budget"]["used"]) == ("short", 290, 402)
    late = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.kick(at=MORNING, mem=late)
    assert len(feed.vendor.odds_calls()) == 2 and kinds[2:] == [("rebuy", False), ("rebuy", False)]
    assert {e["held"] for e in late["games"].values()} == {"floor"}
    feed.vendor.remaining, feed.vendor.used = 480, 20

    def drained(call, vendor):
        vendor.remaining, vendor.used = 101, 399
        return vendor.answer(call)
    feed.vendor.script["odds"] = [drained]
    quiet = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.kick(at=MORNING, mem=quiet)
    assert [c.event_id for c in feed.vendor.odds_calls()[2:]] == [PHI_EID]
    assert kinds[4:] == [("first", False), ("first", False)]
    assert quiet["brake_used"] == 20 and quiet["budget"]["used"] == 400
    assert (quiet["games"][PHI_EID]["held"], quiet["games"][DEN_EID]["held"]) == (None, "ceiling")
    assert "oddsapi: budget held a pull today (ceiling)" in _messages(feed.logs)


def test_a_version_bump_in_the_morning_re_buys_todays_games_with_the_core_markets_l2_4_pinned(feed):
    """Build 4's review, L2-4, pinned as built (an open question for
    Barrett): a FEED_MEMORY_VERSION bump at 09:10 ET gives a ledger with
    had_ledger True, before the brake's window, so today's games are bought
    again as re-buys: floor-checked (B11), and with three markets
    configured they ask the core markets only (B20)."""
    feed.env = {**feed.env, **THREE_ENV}
    feed.vendor.remaining = 5000
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.kick(at=et(8, 50))
    assert [c.markets for c in feed.vendor.odds_calls()] == [THREE, THREE]
    bumped = live.new_feed_memory(et(9, 10), had_ledger=True)
    feed.kick(at=et(9, 10), mem=bumped)
    assert [(c.event_id, c.markets) for c in feed.vendor.odds_calls()][2:] == [
        (PHI_EID, PTS), (DEN_EID, PTS)]
    assert bumped["braked"] and bumped["brake_day"] is None


def test_the_taken_down_count_is_every_line_not_the_eight_named_l3_10(feed):
    """Build 4's review, L3-10: an evening refresh in which eleven points
    lines vanish across the five books (no book losing half its lines, so
    none is suspect). The view's report counts all eleven, names eight, and
    the slate's line says ", and others"."""
    feed.kick()
    gone = {"draftkings": {"Tyrese Maxey", "Duncan Robinson", "Quentin Grimes"},
            "fanduel": {"Tyrese Maxey", "Daniss Jenkins"}, "betmgm": {"Tyrese Maxey", "VJ Edgecombe"},
            "prizepicks": {"Tyrese Maxey", "Daniss Jenkins", "Quentin Grimes"},
            "underdog": {"Tyrese Maxey"}}
    p = copy.deepcopy(feed.vendor.odds[PHI_EID])
    for bm in p["bookmakers"]:
        for m in bm["markets"]:
            if m["key"] == "player_points":
                m["outcomes"] = [o for o in m["outcomes"] if o["description"] not in gone[bm["key"]]]
    feed.vendor.odds[PHI_EID] = p
    feed.kick(at=et(18, 30))
    got = feed.view()
    assert got.report["n_pulled"] == 11 and len(got.report["pulled"]) == 8
    assert feed.mem["games"][PHI_EID]["n_pulled"] == 11
    from bucketlib import props_ui
    html = props_ui._pulled_html({"report": got.report})
    assert "<b>11 lines taken down by their book since the last update:</b>" in html
    assert html.endswith(", and others.</div>")


def test_a_ticket_reserves_the_whole_cost_of_a_call_of_several_markets_lv01(feed):
    """Survivor LV01: while a three-market first pull is in flight, pending
    holds its three credits (what the next ticket reads), never one."""
    feed.env = {**feed.env, **THREE_ENV}
    feed.vendor.remaining = 5000
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    seen = []

    def held(call, vendor):
        with feed.mem["lock"]:
            seen.append(feed.mem["budget"]["pending"])
        return vendor.answer(call)
    feed.vendor.script["odds"] = [held]
    feed.kick()
    assert feed.vendor.odds_calls()[0].markets == THREE
    assert seen == [3] and feed.mem["budget"]["pending"] == 0


def test_a_connection_failure_after_bytes_arrived_counts_as_billed_lv05(feed):
    """Survivor LV05 (3.5's table): a connection failure that received body
    bytes may have been billed: a pull, and its worst case off the balance.
    It is a transport failure all the same, with no answer, so it grows no
    streak (build amendment B22)."""
    feed.vendor.script["odds"] = [H.VendorTransportError("connect", 120)]
    feed.kick(work=work_order(games=(PHI_DET,)))
    e = feed.mem["games"][PHI_EID]
    assert (e["pulls"], e["fails"], feed.mem["streak"]) == (1, 1, 0)
    assert (feed.mem["budget"]["remaining"], feed.mem["budget"]["used"]) == (349, 151)


def test_a_breaker_tripped_in_the_evening_stops_that_et_day_only_lv07(feed):
    """Survivor LV07: three billed failures in a row from 8:00 PM ET (after
    midnight UTC) trip the day breaker for that ET day, Mar 12, never the
    UTC date's: the next morning's pulls go ahead. (Billed junk answers: a
    timeout grows no streak since build amendment B22.)"""
    feed.vendor.remaining = 20000                                    # P = 48: refreshes every 15 min
    feed.kick()
    assert feed.mem["streak"] == 0
    feed.vendor.script["odds"] = [answer_with(200, last=1, body={})] * 3
    for at in (et(20, 0), et(20, 15), et(20, 30)):                   # DEN at SAS alone is still ahead
        feed.kick(at=at)
    assert feed.mem["streak"] == 3 and et(20, 30).astimezone(UTC).date() == dt.date(2026, 3, 13)
    assert feed.mem["breaker_day"] == DATE
    assert not OB.breaker_on(feed.mem["breaker_day"], et(8, 50, day=13))


def test_merge_refresh_hears_what_a_narrower_call_asked_for_lv09(feed):
    """Survivor LV09 (B3): a full first pull of three markets, then the plan
    drops to core (an outside drain) and PHI at DET's evening refresh asks
    points only, with one points line moved. The moved line lands, the
    rebounds and assists lines keep their first-pull time, and nothing is
    called taken down: merge_refresh judged only the markets the call
    asked for."""
    feed.env = {**feed.env, **THREE_ENV}
    feed.vendor.remaining = 5000
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.kick(at=et(8, 50))
    e = feed.mem["games"][PHI_EID]
    first = e["frame"]
    assert set(first["market"]) == set(THREE)
    feed.vendor.remaining, feed.vendor.used = 330, 170               # drained from outside
    p = copy.deepcopy(feed.vendor.odds[PHI_EID])
    for bm in p["bookmakers"]:
        for m in bm["markets"]:
            for oc in m["outcomes"]:
                if bm["key"] == "draftkings" and m["key"] == "player_points" \
                        and oc["description"] == "Cade Cunningham":
                    oc["point"] = 26.5
    feed.vendor.odds[PHI_EID] = p
    feed.kick(at=et(18, 30))
    assert feed.vendor.odds_calls()[-1].markets == PTS
    f = e["frame"]
    dk = f[(f["book"] == "draftkings") & (f["name"] == "Cade Cunningham")]
    assert dk[dk["market"] == "player_points"]["line"].tolist() == [26.5]
    others = f[f["market"] != "player_points"]
    assert len(others) == len(first[first["market"] != "player_points"]) > 0
    assert set(others["fetched_at"]) == {pd.Timestamp(et(8, 50).astimezone(UTC))}
    assert e["pulled"] == [] and e["n_pulled"] == 0


def test_a_ledger_with_a_field_it_should_not_have_is_replaced_lv13(fresh_anchor, caplog):
    """Survivor LV13 (m25): a ledger carrying an extra top-level field is not
    valid, and the anchor replaces it (and logs it)."""
    caplog.set_level(logging.INFO, logger="bucketvalue")
    stray = {**live.new_feed_memory(MORNING_UTC, False), "stray": 0}
    assert not live._valid_ledger(stray)
    assert live._valid_ledger(live.new_feed_memory(MORNING_UTC, False))
    live.feed_anchor()
    mod = sys.modules[live.ANCHOR_NAME]
    mod.state = stray
    again = live.feed_anchor()
    assert again is not stray and "stray" not in again and mod.births == 2
    assert "oddsapi: feed memory created (replaced: version change)" in caplog.text


def test_answers_landing_out_of_order_never_raise_the_balance_or_read_as_a_reset_lv20_lv21(feed):
    """Survivors LV20 and LV21: two calls in flight, the later one's answer
    (348 left, 152 used) settles first, then the earlier one's (349, 151).
    The balance keeps the lower remaining and the higher used, and the
    earlier answer is no reset: no WARNING, and an out-of-credits state
    stays (paid calls do not resume on it)."""
    s = O.read_settings({})
    mem = feed.mem
    with mem["lock"]:
        mem["budget"].update(remaining=350, used=150, auth="out_of_credits")
        live._take_usage(mem, O.Usage(348, 152, 1), billed=True, cost=1, now=MORNING_UTC, s=s)
        live._take_usage(mem, O.Usage(349, 151, 1), billed=True, cost=1, now=MORNING_UTC, s=s)
    b = mem["budget"]
    assert (b["remaining"], b["used"], b["auth"]) == (348, 152, "out_of_credits")
    assert "credits reset outside the 1st" not in feed.logs.text
    assert "the credits came back" not in feed.logs.text


def test_yesterdays_games_survive_the_walk_after_midnight_and_older_ones_go_lv22(feed):
    """Survivor LV22 (2.4.2): the prune drops games dated before YESTERDAY.
    At 00:30 ET the default view date is still yesterday's, so last night's
    games keep their lines; a game two days old goes."""
    feed.kick()
    old = live._new_entry("0022500900", "2026-03-11", None, None)
    feed.mem["games"]["d" * 32] = old
    feed.kick(at=et(0, 30, day=13))
    assert set(feed.mem["games"]) == {PHI_EID, DEN_EID}
    got = live.prop_lines_memory(DATE, mem=feed.mem, now=et(0, 30, day=13))
    assert set(got.value["game_id"]) == {PHI_DET, DEN_SAS}


def test_another_days_memory_is_never_stale_when_todays_events_call_fails_lv23(feed):
    """Survivor LV23: yesterday's lines read "ok" after midnight even while
    today's events call is backing off: nothing is bought for that day."""
    feed.kick()
    tomorrow = dict(TODAY[0], game_id="0022500999", game_date="2026-03-13")
    work = {**work_order(), "games": work_order()["games"] + [tomorrow]}
    feed.vendor.script["events"] = [H.VendorTransportError("timeout")]
    feed.kick(at=et(0, 30, day=13), work=work)
    assert feed.mem["events_fail"] is not None
    got = live.prop_lines_memory(DATE, mem=feed.mem, now=et(0, 30, day=13))
    assert got.state == "ok" and not got.value.empty


def test_an_empty_answer_with_no_usage_headers_costs_nothing_lv24(feed):
    """Survivor LV24 (3.5): an answer with no bookmaker entry is free, so
    with no usage headers to say otherwise nothing comes off the balance,
    and the walk has bought nothing to log."""
    def bare_empty(call, vendor):
        body = {**copy.deepcopy(vendor.odds[call.event_id]), "bookmakers": []}
        return H.VendorResponse(status=200, headers={}, body=body, snippet="", nbytes=99)
    feed.vendor.script["odds"] = [bare_empty]
    feed.kick(work=work_order(games=(PHI_DET,)))
    b = feed.mem["budget"]
    assert (b["remaining"], b["used"]) == (350, 150)
    assert feed.mem["games"][PHI_EID]["pulls"] == 0
    assert not any("walk pulled" in m for m in _messages(feed.logs))


# -- Build 6's fix pass: the review of stage S12 (S12-R1 to S12-R4) ---------------------------
def _billed_timeout(call, vendor):
    """The vendor bills the call (1 credit, points) and its answer comes after
    the timeout: the client sees a timeout and no headers."""
    with vendor.lock:
        vendor.remaining -= 1
        vendor.used += 1
    raise H.VendorTransportError("timeout")


def _nothing_posted(call, vendor):
    """No book has posted the game yet: an answer with no bookmaker entry,
    free (any listed event; no payload needed)."""
    event = next(e for e in vendor.events if e["id"] == call.event_id)
    return vendor.response(200, {**copy.deepcopy(event), "bookmakers": []}, last=0)


def _oversize(call, vendor):
    """The vendor bills the call (1 credit) and sends a body over the cap:
    the client stops the download at VENDOR_MAX_BYTES (http's "oversize")."""
    with vendor.lock:
        vendor.remaining -= 1
        vendor.used += 1
    raise H.VendorTransportError("oversize", H.VENDOR_MAX_BYTES + 1)


def _kick_every(feed, start: dt.datetime, end: dt.datetime, step=MINUTE, **kw) -> None:
    t = start
    while t <= end:
        feed.kick(at=t, **kw)
        t += step


def test_a_vendor_that_bills_and_times_out_after_an_empty_answer_meets_p_again_s12_r1(
        feed, monkeypatch):
    """Build 6's review, S12-R1. L2-1 counts no pull for a timeout of a call
    expected to be free (here a first pull after an empty answer), and B22
    lets no timeout grow the streak, so before this fix nothing but the
    backoff and the reserve bounded the retries: 62 billed calls for these
    four games in one day. Now the exemption holds once per clean answer. A
    ledger born at 08:00 (P = 2): at 08:50 every first pull answers that
    nothing is posted yet (free); from then on the vendor bills every odds
    call and its answer comes after the timeout. Each game: 09:00's call,
    expected to be free, is no pull (its worst case still comes off the
    balance) and spends the exemption; 09:02's (its backoff ends at
    09:01:30) is not taken as free, so it counts, and it is no true first
    pull (the ceiling is checked); B12 retries it at 09:05, which counts
    too, and P is spent. Kicks every minute to 9 PM: three billed calls a
    game, and the streak stays 0 (B22)."""
    when = _timed_odds(feed, monkeypatch)
    kinds = _spend_kinds(monkeypatch)
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.work["games"] += [dict(g) for g in MORE]
    feed.vendor.script["odds"] = [_nothing_posted] * 4 + [_billed_timeout] * 100
    _kick_every(feed, et(8, 50), et(21, 0))
    eids = (PHI_EID, IND_EID, ORL_EID, DEN_EID)
    assert [c.event_id for c in feed.vendor.odds_calls()] == list(eids) * 4
    assert when == [t for t in _utc_times((8, 50), (9, 0), (9, 2), (9, 5)) for _ in eids]
    assert kinds == [("first", True)] * 8 + [("first", False)] * 4 + [("refresh", False)] * 4
    for eid in eids:
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["fails"], e["free_failed"], e["frame"]) == (2, 3, True, None), eid
    b = feed.mem["budget"]
    assert (feed.vendor.used, b["remaining"], b["used"]) == (162, 338, 162)    # 12 billed
    assert (feed.mem["streak"], feed.mem["breaker_day"]) == (0, None)


def test_the_ceiling_holds_the_retry_of_a_true_first_pull_whose_free_call_timed_out_s12_r1(
        feed, monkeypatch):
    """S12-R1 with B23. An outside drain has left 100 credits and 400 used, so
    today's spending ceiling (290) is passed. A ledger born at 08:00: its
    08:50 first pulls are true first pulls and skip the ceiling (B23); both
    answer that nothing is posted yet. At 09:00 they are still true first
    pulls, expected to be free: bought, billed, timed out, no pull. Their
    retries may buy again what those calls bought, so they are no true
    first pulls, and the ceiling holds both games all day, with one
    WARNING: one billed call a game, where before this fix only the
    reserve stopped them (32 billed calls)."""
    kinds = _spend_kinds(monkeypatch)
    feed.vendor.remaining, feed.vendor.used = 100, 400
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.vendor.script["odds"] = [_nothing_posted] * 2 + [_billed_timeout] * 100
    _kick_every(feed, et(8, 50), et(21, 0), 5 * MINUTE)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID] * 2
    assert (feed.vendor.remaining, feed.vendor.used) == (98, 402)
    assert kinds[:4] == [("first", True)] * 4
    assert len(kinds) > 4 and set(kinds[4:]) == {("first", False)}
    for eid in (PHI_EID, DEN_EID):
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["fails"], e["free_failed"], e["held"]) == (0, 1, True, "ceiling"), eid
    assert [m for m in _messages(feed.logs) if "budget held" in m] == [
        "oddsapi: budget held a pull today (ceiling)"]


def test_a_braked_games_re_buy_after_an_empty_answer_counts_once_its_free_retry_is_spent_s12_r1(
        feed, monkeypatch):
    """S12-R1 with B21. A cold start at 10:00 in a period that was spending
    (P = 2): the brake counts both games as pulled at the ledger's birth,
    and each keeps one re-buy. Both re-buys answer that nothing is posted
    yet (free, no pull), so each is due again 10 minutes on; from then on
    the vendor bills every odds call and times out. 10:10's re-buy,
    expected to be free, is no pull; 10:12's is not taken as free and
    counts: each game's second pull, so no re-buy, top-up or refresh
    follows. Two billed re-buys a game, where before this fix a re-buy was
    bought on every backoff all day (30 billed calls)."""
    kinds = _spend_kinds(monkeypatch)
    feed.mem = live.new_feed_memory(et(10, 0), had_ledger=False)
    feed.vendor.script["odds"] = [_nothing_posted] * 2 + [_billed_timeout] * 100
    _kick_every(feed, et(10, 0), et(21, 0))
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID] * 3
    assert kinds == [("rebuy", False)] * 6 and feed.vendor.used == 154
    for eid in (PHI_EID, DEN_EID):
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["braked"], e["fails"], e["free_failed"]) == (2, True, 2, True), eid


def test_top_ups_that_time_out_with_no_clean_answer_between_meet_p_s12_r1(feed, monkeypatch):
    """S12-R1 on the PrizePicks-only top-up, a call L2-1 always expected to be
    free. DEN at SAS came back without PrizePicks at 08:55 (P = 2). From then
    on the vendor bills every odds call and its answer comes after the
    timeout. DEN at SAS's 09:05 top-up is no pull (L2-1); its next one, with
    no clean answer between, counts: its second pull, so its top-ups end,
    where before this fix it polled on every backoff all day (16 billed
    top-ups). PHI at DET's evening refresh times out and counts. Three
    billed calls after 08:55. (With a clean answer between, a timed-out
    top-up is no pull again: the L2-1 test above, hours apart.)"""
    kinds = _spend_kinds(monkeypatch)
    feed.kick()
    den = feed.mem["games"][DEN_EID]
    assert (den["pulls"], den["pp_missing"], den["free_failed"], feed.vendor.used) == (1, True, False, 152)
    feed.vendor.script["odds"] = [_billed_timeout] * 100
    _kick_every(feed, et(9, 0), et(21, 0))
    assert [(c.event_id, c.books) for c in feed.vendor.odds_calls()[2:]] == [
        (DEN_EID, ("prizepicks",)), (DEN_EID, ("prizepicks",)), (PHI_EID, FIVE)]
    assert kinds[2:] == [("pp", False), ("pp", False), ("refresh", False)]
    assert (den["pulls"], den["fails"], den["free_failed"]) == (2, 2, True)
    assert feed.mem["games"][PHI_EID]["pulls"] == 2
    assert (feed.vendor.used, feed.mem["streak"]) == (155, 0)


def test_a_failure_that_answers_never_gives_back_the_free_timeout_s12_r1(feed):
    """S12-R1's flag, `free_failed`, is cleared only by a clean answer for the
    game: nothing posted, or lines that landed. DEN at SAS alone, a ledger
    born at 08:00. 08:50: nothing posted. 09:00: a billed timeout, no pull
    (L2-1), the flag set. 09:02: a 429 that billed nothing comes back, an
    answer but no clean one: still no pull, and the flag stays, so 09:05's
    billed timeout counts."""
    work = work_order(games=(DEN_SAS,))
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.vendor.script["odds"] = [_nothing_posted, _billed_timeout,
                                  answer_with(429, last=0, snippet='{"error_code": "X"}'),
                                  _billed_timeout]
    e = None
    for at, want in ((et(8, 50), (0, 0, False)), (et(9, 0), (0, 1, True)), (et(9, 2), (0, 2, True)),
                     (et(9, 5), (1, 3, True))):
        feed.kick(at=at, work=work)
        e = feed.mem["games"][DEN_EID]
        assert (e["pulls"], e["fails"], e["free_failed"]) == want, at
    assert len(feed.vendor.odds_calls()) == 4 and feed.vendor.used == 152


def test_lines_that_land_give_back_the_free_timeout_s12_r1(feed):
    """S12-R1's flag cleared by lines that landed. DEN at SAS alone, a ledger
    born at 08:00. 08:50: nothing posted. 09:00: a billed timeout, no pull,
    the flag set. 09:02: lines land (none from PrizePicks), pull 1, the
    flag cleared. The PrizePicks-only top-up that times out at 09:12 is
    expected to be free again: no pull, the flag set; the next one, at
    09:14 with no clean answer since, counts (pull 2)."""
    work = work_order(games=(DEN_SAS,))
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.vendor.script["odds"] = [_nothing_posted, _billed_timeout,
                                  lambda call, vendor: vendor.answer(call),
                                  _billed_timeout, _billed_timeout]
    for at, want in ((et(8, 50), (0, False)), (et(9, 0), (0, True)), (et(9, 2), (1, False)),
                     (et(9, 12), (1, True)), (et(9, 14), (2, True))):
        feed.kick(at=at, work=work)
        e = feed.mem["games"][DEN_EID]
        assert (e["pulls"], e["free_failed"]) == want, at
    assert [c.books for c in feed.vendor.odds_calls()] == [FIVE] * 3 + [("prizepicks",)] * 2


def test_a_failure_that_cannot_have_been_billed_leaves_the_true_first_pull_alone_s12_r1(
        feed, monkeypatch):
    """S12-R1's ceiling rule touches only a failure that may have been
    billed. Past today's ceiling (100 left, 400 used), a ledger born at
    08:00, PHI at DET alone: 08:50 nothing posted; 09:00 and 09:01:30 no
    connection with nothing received (3.5: not billed, no pull), so the
    flag stays unset; the true first pull at 09:04:30 still skips the
    ceiling (B23), and its lines land. (Holding every retry after any
    failure would have held this game all day.)"""
    kinds = _spend_kinds(monkeypatch)
    feed.vendor.remaining, feed.vendor.used = 100, 400
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    work = work_order(games=(PHI_DET,))
    nothing = H.VendorTransportError("connect", 0)
    feed.vendor.script["odds"] = [_nothing_posted, nothing, nothing]
    for at in (et(8, 50), et(9, 0), et(9, 1, 30), et(9, 4, 30)):
        feed.kick(at=at, work=work)
    e = feed.mem["games"][PHI_EID]
    assert len(feed.vendor.odds_calls()) == 4 and kinds == [("first", True)] * 4
    assert (e["pulls"], e["fails"], e["free_failed"], e["held"]) == (1, 0, False, None)
    assert e["frame"] is not None and feed.vendor.used == 401


def test_an_entry_from_before_s12_r1_reads_its_flag_as_false_and_a_bad_flag_is_malformed():
    """S12-R1's flag on a game's ledger entry: new entries say False; an
    entry without it (a ledger kept across a code push, which no version
    check sees) stays valid and reads as False; anything but a bool makes
    the entry malformed (m25)."""
    entry = live._new_entry(PHI_DET, "2026-03-12", PHI_TIP, PHI_TIP)
    assert entry["free_failed"] is False and live._slot_of(entry, PHI_EID) is not None
    old = {k: v for k, v in entry.items() if k != "free_failed"}
    assert live._slot_of(old, PHI_EID) is not None
    assert live._slot_of({**entry, "free_failed": True}, PHI_EID) is not None
    for bad in (None, 1, "True"):
        assert live._slot_of({**entry, "free_failed": bad}, PHI_EID) is None, bad


def test_an_entry_without_the_flag_keeps_l2_1s_one_free_timeout_s12_r1(feed):
    """An entry written before S12-R1 (no `free_failed`) mid-morning: the
    pull after its empty answer, timed out and billed, is still no pull
    (L2-1), and the flag is then written, so the next one counts."""
    work = work_order(games=(PHI_DET,))
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.vendor.script["odds"] = [_nothing_posted, _billed_timeout, _billed_timeout]
    feed.kick(at=et(8, 50), work=work)
    e = feed.mem["games"][PHI_EID]
    del e["free_failed"]
    feed.kick(at=et(9, 0), work=work)
    assert (e["pulls"], e["fails"], e["free_failed"]) == (0, 1, True)
    feed.kick(at=et(9, 1, 30), work=work)
    assert (e["pulls"], e["fails"]) == (1, 2)


def test_oversize_answers_are_billed_junk_so_three_in_a_row_trip_the_breaker_s12_r4(feed):
    """Build 6's review, S12-R4. http's vendor_get raises "oversize" only from
    its body callback: the status line and the headers had come and the
    body was arriving when the client stopped it. An answer came back,
    assumed billed (3.5) and of no use (design 6 lists a body over 4 MB as
    billed junk), so it grows the streak, B22's rule for billed junk, as RP
    case E wants. Every odds call of the 08:50 batch of four games answers
    so: three calls, then the day breaker with one WARNING, and a view
    every 5 minutes all day buys nothing more (before this fix: eight
    billed calls, P a game, and no breaker)."""
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.work["games"] += [dict(g) for g in MORE]
    feed.vendor.script["odds"] = [_oversize] * 50
    _kick_every(feed, et(8, 50), et(23, 55), 5 * MINUTE)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, IND_EID, ORL_EID]
    assert (feed.mem["streak"], feed.mem["breaker_day"], feed.vendor.used) == (3, DATE, 153)
    assert [m for m in _messages(feed.logs) if "billed failures in a row" in m] == [
        "oddsapi: 3 billed failures in a row; paid calls stopped until tomorrow "
        "(parser counters: oversize)"]
    assert "oddsapi event odds failed: oversize" in _messages(feed.logs)


def test_an_oversize_answer_to_a_call_expected_to_be_free_is_a_pull_s12_r4(feed):
    """S12-R4 with L2-1: an oversize answer came back, so it is no timeout of
    a call expected to be free. DEN at SAS lacks PrizePicks at 08:55 (P =
    2); its 09:05 top-up answers oversize: a pull, its second, so its
    top-ups end, and a step of the streak."""
    feed.kick()
    den = feed.mem["games"][DEN_EID]
    feed.vendor.script["odds"] = [_oversize]
    feed.kick(at=et(9, 5))
    assert feed.vendor.odds_calls()[-1].books == ("prizepicks",)
    assert (den["pulls"], den["fails"], den["free_failed"], feed.mem["streak"]) == (2, 1, False, 1)


def test_an_oversize_answer_whose_settling_raises_still_grows_the_streak_s12_r4(feed, monkeypatch):
    """S12-R4 in the broken path (review R1): an oversize answer whose
    settling raises is settled once as a broken call that came back, so it
    is billed, a pull and a step of the streak."""
    real, answered = live._failed, []

    def once(*a, **k):
        answered.append(k["answered"])
        if len(answered) == 1:
            raise RuntimeError(MARK)
        return real(*a, **k)
    monkeypatch.setattr(live, "_failed", once)
    feed.vendor.script["odds"] = [_oversize]
    feed.kick(work=work_order(games=(PHI_DET,)))
    e, b = feed.mem["games"][PHI_EID], feed.mem["budget"]
    assert answered == [True, True]
    assert (e["pulls"], e["fails"], feed.mem["streak"], b["remaining"], b["used"]) == (1, 1, 1, 349, 151)
    assert MARK not in feed.logs.text


def test_a_cold_start_re_buy_on_p_2_is_the_games_last_pull_so_prizepicks_never_comes_s12_r2_pinned(
        feed):
    """Build 6's review, S12-R2, pinned as built (an open question for
    Barrett). B21 on the free plan (P = 2): a cold start at 10:00 in a period
    that was spending re-buys both games at once, and that re-buy is each
    game's second and last pull, so due_kind's `k >= P` gate also ends the
    PrizePicks-only top-ups and the evening refresh. DEN at SAS has no
    PrizePicks line at 10:00, and PrizePicks posts it from noon: DEN at
    SAS's 10:00 re-buy is its one call all day, and at 8:50 PM its lines
    still have no PrizePicks line (the board's default pill). The same day
    in a ledger born at 08:00 (no restart) gets the PrizePicks line by a
    top-up."""
    plain = _load("event_odds_den_sas_constructed.json")

    def day(born):
        feed.mem = live.new_feed_memory(born, had_ledger=False)
        feed.vendor.odds[DEN_EID] = plain
        n = len(feed.vendor.odds_calls(DEN_EID))
        for at in [t for t in DAY_KICKS if t >= born]:
            if at == et(12, 0):
                feed.vendor.odds[DEN_EID] = _with_prizepicks(plain)
            feed.kick(at=at)
        v = live.feed_view(DATE, mem=feed.mem, wait_s=0, now=et(20, 50)).value
        return feed.vendor.odds_calls(DEN_EID)[n:], set(v[v["game_id"] == DEN_SAS]["book"])
    calls, books = day(et(10, 0))
    assert [(c.books, c.markets) for c in calls] == [(FIVE, PTS)]
    assert books == {"draftkings", "fanduel", "betmgm", "underdog"}
    den = feed.mem["games"][DEN_EID]
    assert (den["pulls"], den["braked"], den["pp_missing"]) == (2, True, True)
    calls, books = day(et(8, 0))
    assert calls[0].books == FIVE and ("prizepicks",) in [c.books for c in calls[1:]]
    assert "prizepicks" in books


# -- Build 7, stage S13: build amendment B31 ---------------------------------------------------
CUT = 2048                                      # the body bytes that came before the cut


def _cut_off(call, vendor):
    """The vendor bills the call (1 credit, points) and its answer starts to
    come, then the network cuts the download off: http's kind "transport",
    with the bytes that arrived (design 3.5's connection failure after body
    bytes arrived; build amendment B31). No status line and no headers reach
    the caller."""
    with vendor.lock:
        vendor.remaining -= 1
        vendor.used += 1
    raise H.VendorTransportError("transport", CUT)


def test_three_downloads_the_network_cut_off_in_a_row_leave_the_breaker_off_and_are_retried_b31(feed):
    """Build amendment B31 (build 6's verifier, probe V-transport-answered). A
    download the network cut off is no answer: it takes its worst case off
    the balance, counts as a pull (3.5: it may have been billed), backs its
    game off, and never grows the streak (B22). At 08:50 four games are due
    (a ledger born at 08:00), three of them at 7 PM, and those three calls
    are cut off once their bytes have begun to arrive, each billed: the
    streak is 0 as each next call goes out and after the last, the day
    breaker stays off, and DEN at SAS's call, the fourth, goes ahead in the
    same walk. Each cut-off game's pulls brought no lines, so it is retried
    at 08:51:30, its 90 s backoff's end (B12), never before, and PHI at
    DET's lines land. The ledger's balance and the vendor's agree after each
    walk: every billed credit counted once. (Were a cut-off download an
    answer, the three would be billed junk in a row: the breaker, no fourth
    call and no retry all day.)"""
    feed.mem = live.new_feed_memory(et(8, 0), had_ledger=False)
    feed.work["games"] += [dict(g) for g in MORE]
    seen = []

    def watched(answer):
        def call(c, vendor):
            with feed.mem["lock"]:
                seen.append((c.event_id, feed.mem["streak"], feed.mem["breaker_day"]))
            return answer(c, vendor)
        return call
    feed.vendor.script["odds"] = [watched(_cut_off)] * 3 + [watched(lambda c, v: v.answer(c))]
    feed.kick(at=et(8, 50))
    assert seen == [(PHI_EID, 0, None), (IND_EID, 0, None), (ORL_EID, 0, None), (DEN_EID, 0, None)]
    assert (feed.mem["streak"], feed.mem["breaker_day"]) == (0, None)
    for eid in (PHI_EID, IND_EID, ORL_EID):
        e = feed.mem["games"][eid]
        assert (e["pulls"], e["fails"], e["fail_until"], e["frame"], e["free_failed"]) == (
            1, 1, et(8, 51, 30).astimezone(UTC), None, False), eid
    assert feed.mem["games"][DEN_EID]["frame"] is not None
    b = feed.mem["budget"]
    assert (b["remaining"], b["used"], b["pending"]) == (346, 154, 0)       # 3 worst cases and DEN's
    assert (feed.vendor.remaining, feed.vendor.used) == (346, 154)
    assert _messages(feed.logs).count("oddsapi event odds failed: server transport") == 1
    assert not [m for m in _messages(feed.logs) if "billed failures in a row" in m]
    feed.vendor.script["odds"] = [lambda c, v: v.answer(c), _nothing_posted, _nothing_posted]
    feed.kick(at=et(8, 51, 29))                                         # still backing off
    assert len(feed.vendor.odds_calls()) == 4
    feed.kick(at=et(8, 51, 30))
    assert [c.event_id for c in feed.vendor.odds_calls()[4:]] == [PHI_EID, IND_EID, ORL_EID]
    assert (feed.mem["streak"], feed.mem["breaker_day"]) == (0, None)
    phi = feed.mem["games"][PHI_EID]
    assert (phi["pulls"], phi["fails"]) == (2, 0) and phi["frame"] is not None
    assert set(feed.view().value["game_id"]) == {PHI_DET, DEN_SAS}
    assert (b["remaining"], b["used"]) == (feed.vendor.remaining, feed.vendor.used) == (345, 155)


def test_a_cut_off_download_of_a_call_expected_to_be_free_is_no_pull_and_grows_no_streak_b31(feed):
    """B31 with L2-1 (build 4's review) and S12-R1: a cut-off download is no
    answer, so when the call was expected to be free it counts no pull, once
    per clean answer, and grows no streak, though its worst case comes off
    the balance. DEN at SAS lacks PrizePicks at 08:55 (P = 2); its 09:05
    PrizePicks-only top-up is cut off: no pull, `free_failed` set, a 90 s
    backoff, the streak 0. Its next top-up, at 09:06:30 with no clean answer
    between, is cut off too: no longer taken as free, it counts, its second
    and last pull, and the streak is still 0. (Were a cut-off download an
    answer, the first would be a pull and a step of the streak, as an
    oversize answer is: the S12-R4 test above.)"""
    feed.kick()
    den = feed.mem["games"][DEN_EID]
    assert (den["pulls"], den["pp_missing"], den["free_failed"]) == (1, True, False)
    feed.vendor.script["odds"] = [_cut_off, _cut_off]
    feed.kick(at=et(9, 5))
    assert [c.books for c in feed.vendor.odds_calls()[2:]] == [("prizepicks",)]
    assert (den["pulls"], den["fails"], den["free_failed"], feed.mem["streak"]) == (1, 1, True, 0)
    assert den["fail_until"] == et(9, 6, 30).astimezone(UTC)
    b = feed.mem["budget"]
    assert (b["remaining"], b["used"]) == (feed.vendor.remaining, feed.vendor.used) == (347, 153)
    feed.kick(at=et(9, 6, 29))
    assert len(feed.vendor.odds_calls()) == 3
    feed.kick(at=et(9, 6, 30))
    assert [c.books for c in feed.vendor.odds_calls()[2:]] == [("prizepicks",)] * 2
    assert (den["pulls"], den["fails"], den["free_failed"], feed.mem["streak"]) == (2, 2, True, 0)
    assert feed.mem["breaker_day"] is None
    assert (b["remaining"], b["used"]) == (feed.vendor.remaining, feed.vendor.used) == (346, 154)


def test_a_cut_off_download_whose_settling_raises_is_billed_and_grows_no_streak_b31(feed, monkeypatch):
    """B31 in the broken path (review R1): a cut-off download whose settling
    raises is settled once as a broken call that got no answer: billed
    (bytes came, so it may have been), a pull and a backoff, and no step of
    the streak, where an oversize answer's is one (the S12-R4 test above)."""
    real, answered = live._failed, []

    def once(*a, **k):
        answered.append(k["answered"])
        if len(answered) == 1:
            raise RuntimeError(MARK)
        return real(*a, **k)
    monkeypatch.setattr(live, "_failed", once)
    feed.vendor.script["odds"] = [_cut_off]
    feed.kick(work=work_order(games=(PHI_DET,)))
    e, b = feed.mem["games"][PHI_EID], feed.mem["budget"]
    assert answered == [False, False]
    assert (e["pulls"], e["fails"], feed.mem["streak"], b["remaining"], b["used"], b["pending"]) == (
        1, 1, 0, 349, 151, 0)
    assert e["fail_until"] == MORNING_UTC + dt.timedelta(seconds=90)
    assert MARK not in feed.logs.text


# -- Build 7's review: what docs/runbook.md section 8 says, pinned as built ------------------
@pytest.mark.parametrize("local_born", [et(8, 30), et(20, 0, day=11)],
                         ids=["started_at_0830", "left_open_overnight"])
def test_a_second_process_running_at_0850_buys_todays_games_again_past_the_ceiling_r7_1(
        feed, monkeypatch, local_born):
    """Build 7's review, R7-1, pinned as built (B23 stands; docs/runbook.md
    section 8: "The ceiling", B23 and the local-run advice). The ceiling
    bounds every first pull of a process born after the game's 08:50, so a
    restart cannot push past it. But each process keeps its own ledger, and a
    second one already running at 08:50 makes TRUE first pulls, which skip
    the ceiling. 200 left and 400 used on Mar 12 (A 600, S 30): today's
    ceiling, 348, is already passed. The site's process (born at 03:00) and a
    local run with the key file (started at 08:30, or left open since the
    evening before) each buy both games at 08:50: four true first pulls, two
    of them the same games bought again, bounded only by the reserve. A
    restart at 08:55 in that spending period cannot tell its first pulls
    from re-buys (B11): the floor lets them through, and the ceiling holds
    them."""
    kinds = _spend_kinds(monkeypatch)
    feed.vendor.remaining, feed.vendor.used = 200, 400
    site = live.new_feed_memory(et(3, 0), had_ledger=False)
    local = live.new_feed_memory(local_born, had_ledger=False)
    feed.kick(at=et(8, 50), mem=site)
    feed.kick(at=et(8, 50), mem=local)
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID] * 2
    assert kinds == [("first", True)] * 4
    plan = _plan_now(local, feed.env, et(8, 50))
    assert (plan.mode, plan.reserve, plan.ceiling) == ("full", 30, 348)
    assert (local["budget"]["remaining"], local["budget"]["used"]) == (196, 404)     # past 348
    restart = live.new_feed_memory(MORNING_UTC, had_ledger=False)
    feed.kick(at=MORNING, mem=restart)
    assert len(feed.vendor.odds_calls()) == 4 and kinds[4:] == [("rebuy", False)] * 2
    assert {e["held"] for e in restart["games"].values()} == {"ceiling"}
    assert "oddsapi: budget held a pull today (ceiling)" in _messages(feed.logs)


def test_after_a_day_with_no_view_of_today_the_refresher_waits_for_a_view_r7_3(feed, tmp_path):
    """Build 7's review, R7-3 (docs/runbook.md section 8, "When it buys").
    The refresher buys only the games of the work order the last view of
    today wrote: store names that day's and the next day's committed files
    in its serve_files (tests/test_store_feed.py pins it), and live ignores
    an order more than ORDER_MAX_AGE (36 hours) old (pinned above). The last
    view here was of Mar 10's slate at 9:30 PM ET, and nobody opens Mar 11's.
    At 08:50 on Mar 12 that order is 35 h 20 min old, so the age rule is not
    what stops it: it names no file of Mar 12, so every tick is "no_work"
    and nothing is bought, not even the free events call (the evening
    before's order does buy at 08:50: the first refresher test above). A
    view of Mar 12 then buys both games at once, and leaves the refresher a
    new order to work from."""
    written = et(21, 30, day=10)
    order = {**evening_order([], written=written), "date": "2026-03-10",
             "games": [],                                       # Mar 10's games have tipped
             "serve_files": {d: [str(tmp_path / d / "projections_v1.parquet"),
                                 str(tmp_path / d / "features_v1.parquet")]
                             for d in ("2026-03-10", "2026-03-11")}}
    feed.mem = live.new_feed_memory(et(20, 0, day=10), had_ledger=False)
    feed.kick(at=written, work=order)
    assert feed.vendor.calls == [] and feed.mem["work"] is order
    assert et(9, 0) - written < live.ORDER_MAX_AGE                       # the premise: young enough
    for at in (et(8, 50), et(8, 51), et(9, 0)):
        assert _tick(feed, at) == "no_work", at
    assert feed.vendor.calls == []
    feed.kick(at=et(9, 5))                                               # the first view of Mar 12
    assert [c.event_id for c in feed.vendor.odds_calls()] == [PHI_EID, DEN_EID]
    assert feed.mem["work"]["date"] == "2026-03-12" and _tick(feed, et(9, 6)) == "idle"
