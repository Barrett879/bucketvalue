"""The page's recompute runs the model trees on ONE OpenMP thread (round BKS,
2026-11-15; docs/serving_design.md section 11): store._live_frame runs
pipeline.live_projections, and nothing else, inside store._one_openmp_thread
(threadpoolctl.threadpool_limits(limits=1, user_api="openmp")), so a busy or
oversized host cannot stall the page's 17 HistGradientBoosting models at
every one of their ~3,500 small parallel regions. The bot and the research
scripts call models.predict outside the page and keep their threads.

1. The recompute sees one OpenMP thread, through exactly that threadpoolctl
   call (OpenMP only: a BLAS library that runs its own threads keeps its
   count); the model load before it and the caller after it see the
   thread's own count, also when the recompute raises (0293ddc: the
   recompute saw the machine's default).
2. The limit is the recomputing thread's own: a worker thread recomputes on
   one thread while this thread keeps its count, and the worker is back at
   the default after.
3. models.predict gives the same bits on one thread and at the default, on
   a served date's committed features (tests/fixtures/pipeline: the nine
   games of 2026-03-12 as the bot commits their history features, with the
   real 12:00 PM ET report and made-up lines) and the production models, so
   the limit cannot move a number.
4. models.predict sets no limit of its own: each of its models predicts at
   its caller's count.
5. The limit never costs the page: threadpoolctl failing to set it or to
   lift it costs one WARNING line, the recompute's own error still raises
   as before, and the home page and the Game page draw what they draw with a
   working limit (before the round's fix the home page showed Streamlit's
   error box).

The counts are scikit-learn's own (_openmp_effective_n_threads, the count
the HistGradientBoosting predictors use). The process's count is read on a
NEW thread, which starts at the runtime's default in libgomp and libomp
alike, so a limit that earlier code left on the test's own thread fails the
tests that need threads instead of passing them as skips; they skip only
where the process has a single OpenMP thread, where there is nothing to
limit. In 1, 2 and the unit tests of 5 live_projections is a stand-in; the
pages of 5 run on the serve fixture (tests/fixtures/serve). Nothing reads a
clock or a network, and the date is the fixtures' fixed past one.
"""
from __future__ import annotations

import datetime as dt
import inspect
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from threadpoolctl import threadpool_info, threadpool_limits

from bucketlib import context as X
from bucketlib import features as F
from bucketlib import models as M
from bucketlib import pipeline as P
from bucketlib import store
from bucketlib.sources import injury as INJ
from bucketlib.util import ET, UTC

FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(FIX / "serve"))
import serve_harness as H  # noqa: E402

DAY = dt.date(2026, 3, 12)
NOON = dt.datetime(2026, 3, 12, 12, 0, tzinfo=ET)
MORNING = dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET)    # the 12:00 PM report is 20 minutes old
NOT_SET = "thread limit not set (RuntimeError): the recompute runs at the default count"
NOT_LIFTED = "thread limit not lifted (OSError)"


def _threads() -> int:
    return _openmp_effective_n_threads()


def _process_default() -> int:
    """The process's OpenMP count, read on a NEW thread: a thread starts at
    the runtime's default, whatever limit another thread holds or left on."""
    out = []
    t = threading.Thread(target=lambda: out.append(_threads()), daemon=True)
    t.start()
    t.join(30)
    assert out, "the counting thread did not finish"
    return out[0]


@pytest.fixture
def default_threads() -> int:
    """The process's OpenMP count with no limit, which this thread must run
    too; a skip where it is one (nothing to limit). A limit left on this
    thread by earlier code (one never lifted, or one set for the whole
    importing thread) fails here: it would read as one thread."""
    n = _process_default()
    if n < 2:
        pytest.skip("a single OpenMP thread here: nothing to limit")
    here = _threads()
    assert here == n, (f"this thread runs {here} OpenMP threads and a new thread {n}: "
                       "earlier code left a limit on this thread")
    return n


def _own_blas() -> dict:
    """{library: threads} for each BLAS library threadpoolctl controls that
    runs its own (pthreads) threads, as numpy's OpenBLAS does on Linux, where
    its count is the whole process's. On macOS threadpoolctl controls none
    (numpy uses Accelerate), so this is empty there."""
    return {i.get("filepath"): i.get("num_threads") for i in threadpool_info()
            if i.get("user_api") == "blas" and i.get("threading_layer") == "pthreads"}


def _recompute_in_live_frame(tag: str):
    """store._live_frame's own body (Streamlit's cache bypassed) for the
    fixtures' date; `tag` keeps one call's keys apart from another's."""
    return store._live_frame.__wrapped__(DAY.isoformat(), ("bks", tag), ("bks", tag), (), (),
                                         NOON.astimezone(UTC), None, None, None)


def _thread_lines(caplog) -> list[str]:
    out = []
    for r in caplog.records:
        if r.name == "bucketvalue" and r.getMessage().startswith("thread limit"):
            assert r.levelno == logging.WARNING
            out.append(r.getMessage())
    return out


# -- 1. One thread inside, the thread's own count outside -----------------------------
def test_the_page_recompute_runs_on_one_openmp_thread_and_the_count_comes_back(
        monkeypatch, default_threads):
    seen = {"model load": [], "recompute": [], "own blas": []}
    calls = []
    real_limits = store.threadpool_limits

    def limits(*a, **k):
        calls.append(dict(inspect.signature(real_limits).bind(*a, **k).arguments))
        return real_limits(*a, **k)

    def models():
        seen["model load"].append(_threads())

    def recompute(*a, **k):
        seen["recompute"].append(_threads())
        seen["own blas"].append(_own_blas())
        return "frame", "info"
    blas = _own_blas()
    monkeypatch.setattr(store, "threadpool_limits", limits)
    monkeypatch.setattr(store, "models", models)
    monkeypatch.setattr(P, "live_projections", recompute)
    assert _recompute_in_live_frame("returns") == ("frame", "info")
    assert seen == {"model load": [default_threads], "recompute": [1], "own blas": [blas]}
    assert _threads() == default_threads

    def fails(*a, **k):
        seen["recompute"].append(_threads())
        raise RuntimeError("the recompute failed")
    monkeypatch.setattr(P, "live_projections", fails)
    with pytest.raises(RuntimeError, match="the recompute failed"):
        _recompute_in_live_frame("raises")
    assert seen["recompute"] == [1, 1] and _threads() == default_threads
    assert calls == [{"limits": 1, "user_api": "openmp"}] * 2


# -- 2. This thread's limit only -------------------------------------------------------
def test_the_limit_is_the_recomputing_threads_own(monkeypatch, default_threads):
    """A worker thread runs the recompute and waits inside it, on one thread;
    meanwhile this thread runs its own default, and after its recompute the
    worker is back at the default too."""
    inside, release, out, counts = threading.Event(), threading.Event(), [], {}

    def recompute(*a, **k):
        counts["worker, inside"] = _threads()
        inside.set()
        assert release.wait(30)
        return "frame", "info"

    def work():
        out.append(_recompute_in_live_frame("worker"))
        counts["worker, after"] = _threads()
    monkeypatch.setattr(store, "models", lambda: None)
    monkeypatch.setattr(P, "live_projections", recompute)
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        assert inside.wait(30)
        counts["here, meanwhile"] = _threads()      # while the worker is inside its recompute
    finally:
        release.set()
        worker.join(30)
    assert not worker.is_alive() and out == [("frame", "info")]
    assert counts == {"worker, inside": 1, "here, meanwhile": default_threads,
                      "worker, after": default_threads}
    assert _threads() == default_threads


# -- 3. The numbers do not depend on the thread count -----------------------------------
def _served_rows() -> pd.DataFrame:
    """The model's input rows for the nine games of 2026-03-12, as
    pipeline.project builds them: the committed history features, made-up
    lines (historical lines are never committed), then context.context_for
    with the real 12:00 PM ET report at 12:20 PM ET."""
    feats = pd.read_parquet(FIX / "pipeline" / "features.parquet")
    sched = pd.read_parquet(FIX / "pipeline" / "schedule.parquet")
    games = P.state_games(sched, DAY)
    report = INJ.parse_report((FIX / "injury" / "Injury-Report_2026-03-12_12_00PM.pdf").read_bytes())
    n = len(games)
    lines = pd.DataFrame({"game_id": games["game_id"].to_numpy(),
                          "home_line": np.linspace(-9.5, 7.5, n), "total": np.linspace(214.5, 236.5, n)})
    names = feats[["person_id", "player_name"]].drop_duplicates()
    return X.context_for(F.apply_lines(feats, lines), games, sched, report, MORNING, names)


@pytest.fixture(scope="module")
def mdl() -> M.Models:
    return M.load()


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return _served_rows()


def test_models_predict_gives_the_same_bits_on_one_thread_and_at_the_default(
        default_threads, mdl, rows):
    """HistGradientBoosting sums each row's trees in one order at any thread
    count, so the frame the page recomputes on one thread is the frame it
    computed at the default: every output, bit for bit (a -0.0 or a NaN
    payload included), on today's frame size (the live 2026-11-15 frame had
    315 rows)."""
    assert len(rows) >= 300 and rows["game_id"].nunique() == 9
    at_default = M.predict(mdl, rows)
    with threadpool_limits(limits=1, user_api="openmp"):
        assert _threads() == 1
        on_one = M.predict(mdl, rows)
    assert _threads() == default_threads
    assert list(on_one.columns) == list(at_default.columns) == M.output_columns(mdl.stats)
    assert on_one.index.equals(at_default.index) and len(on_one) == len(rows)
    a = at_default.to_numpy(dtype="float64")
    b = on_one.to_numpy(dtype="float64")
    assert np.isfinite(a).all() and a.shape == b.shape and a.tobytes() == b.tobytes()


# -- 4. models.predict keeps its caller's threads ----------------------------------------
def test_models_predict_keeps_its_callers_threads(monkeypatch, default_threads, mdl, rows):
    """The limit is the page's alone. models.predict, which the bot and the
    research scripts call, sets none: every HistGradientBoosting model in it
    reads its caller's count when it predicts (each predict reads
    _openmp_effective_n_threads). Under the caller's own limit the same reads
    give 1, so a limit anywhere inside models.predict would show."""
    from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as HGB
    seen = []
    real = HGB._openmp_effective_n_threads

    def counted(*a, **k):
        n = real(*a, **k)
        seen.append(n)
        return n
    monkeypatch.setattr(HGB, "_openmp_effective_n_threads", counted)
    n_models = len(mdl.estimators())
    assert n_models == 2 + len(mdl.stats) >= 3
    M.predict(mdl, rows)
    assert len(seen) >= n_models and set(seen) == {default_threads}, seen
    seen.clear()
    with threadpool_limits(limits=1, user_api="openmp"):
        M.predict(mdl, rows)
    assert len(seen) >= n_models and set(seen) == {1}, seen
    assert _threads() == default_threads


# -- 5. The limit never costs the page ----------------------------------------------------
def _cannot_set(*a, **k):
    raise RuntimeError("threadpoolctl could not read the loaded libraries")


class _CannotLift:
    """threadpool_limits whose lifting raises, after it put the thread's
    count back."""

    def __init__(self, *a, **k):
        self.real = threadpool_limits(*a, **k)

    def restore_original_limits(self):
        self.real.restore_original_limits()
        raise OSError(5, "Input/output error")


def test_a_limit_that_cannot_be_set_costs_one_line_and_the_recompute_runs_at_the_default(
        monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="bucketvalue")
    default, seen = _process_default(), []
    monkeypatch.setattr(store, "threadpool_limits", _cannot_set)
    monkeypatch.setattr(store, "models", lambda: None)

    def recompute(*a, **k):
        seen.append(_threads())
        return "frame", "info"
    monkeypatch.setattr(P, "live_projections", recompute)
    assert _recompute_in_live_frame("not set") == ("frame", "info")
    assert seen == [default] and _threads() == default
    assert _thread_lines(caplog) == [NOT_SET]

    def fails(*a, **k):
        raise KeyError("the recompute failed")
    monkeypatch.setattr(P, "live_projections", fails)
    with pytest.raises(KeyError, match="the recompute failed"):
        _recompute_in_live_frame("not set, raises")
    assert _thread_lines(caplog) == [NOT_SET, NOT_SET] and _threads() == default


def test_a_limit_that_cannot_be_lifted_costs_one_line_and_never_hides_the_recomputes_error(
        monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="bucketvalue")
    default, seen = _process_default(), []
    monkeypatch.setattr(store, "threadpool_limits", _CannotLift)
    monkeypatch.setattr(store, "models", lambda: None)

    def recompute(*a, **k):
        seen.append(_threads())
        return "frame", "info"
    monkeypatch.setattr(P, "live_projections", recompute)
    assert _recompute_in_live_frame("not lifted") == ("frame", "info")
    assert seen == [1] and _threads() == default
    assert _thread_lines(caplog) == [NOT_LIFTED]

    def fails(*a, **k):
        raise KeyError("the recompute failed")
    monkeypatch.setattr(P, "live_projections", fails)
    with pytest.raises(KeyError, match="the recompute failed"):
        _recompute_in_live_frame("not lifted, raises")
    assert _thread_lines(caplog) == [NOT_LIFTED, NOT_LIFTED] and _threads() == default


@pytest.fixture
def no_cached_builds():
    store._live_frame.clear()
    store._feed_tables.clear()
    yield
    store._live_frame.clear()
    store._feed_tables.clear()


@pytest.mark.parametrize("page, params", [
    ("app.py", {}),
    ("pages/Game.py", {"away": "DEN", "home": "SAS"}),
])
def test_the_pages_draw_the_same_when_the_limit_cannot_be_set(monkeypatch, tmp_path, caplog,
                                                              no_cached_builds, page, params):
    """The whole page on the serve fixture at 12:20 PM ET (a live recompute
    with the production models), first with a working limit, set once for
    the one recompute, then with threadpoolctl raising: no exception, the
    one WARNING line, and every markdown element the same."""
    from streamlit.testing.v1 import AppTest
    caplog.set_level(logging.WARNING, logger="bucketvalue")
    H.install(monkeypatch, tmp_path, H.MORNING)
    real_limits, limits_set = store.threadpool_limits, []

    def limits(*a, **k):
        limits_set.append(1)
        return real_limits(*a, **k)

    def draw() -> list[str]:
        store._live_frame.clear()
        store._feed_tables.clear()
        at = AppTest.from_file(str(H.REPO / page), default_timeout=90)
        at.query_params["date"] = H.DATE.isoformat()
        for k, v in params.items():
            at.query_params[k] = v
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return [m.value for m in at.markdown]
    monkeypatch.setattr(store, "threadpool_limits", limits)
    want = draw()
    assert limits_set == [1] and _thread_lines(caplog) == [] and len(want) >= 5
    monkeypatch.setattr(store, "threadpool_limits", _cannot_set)
    got = draw()
    assert _thread_lines(caplog) == [NOT_SET]
    assert got == want
