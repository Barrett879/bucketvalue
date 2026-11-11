"""bucketlib.http.vendor_get: the one entry point for the licensed odds feed
(The Odds API). One attempt, an honest client, no redirect followed, its own
pacing, a size cap enforced during the download, a timeout floor, and no key,
URL or vendor exception text in anything it returns, raises or logs.

Nothing here reaches a network. conftest's session block replaces
http.vendor_get and http._vendor_session with a function that fails the test;
REAL and REAL_SESSION are the real ones, saved when this module is imported
(at collection, before that block applies). Every test that calls REAL first
installs a fake http._vendor_session (the `vendor` fixture), which plays
curl_cffi's Session.get. The fake key is built at run time. Every body below
is CONSTRUCTED by hand; none is a captured vendor answer.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import traceback
import types
from urllib.parse import urlencode

import pytest

from bucketlib import http as H   # first: on macOS it loads CoreFoundation before curl_cffi

from curl_cffi.curl import CURL_WRITEFUNC_ERROR, CurlError  # noqa: E402
from curl_cffi.requests import Headers  # noqa: E402
from curl_cffi.requests import exceptions as cx  # noqa: E402

REAL = H.vendor_get                 # saved before conftest's session block replaces it
REAL_SESSION = H._vendor_session

HOST = "api.the-odds-api.com"
EVENTS = "/v4/sports/basketball_nba/events"
KEY = "K" * 40                      # a fake key, built at run time
NEVER_SHOWN = (KEY, "apiKey=", "the-odds-api.com")
USAGE = {"X-Requests-Remaining": "475", "X-Requests-Used": "25", "X-Requests-Last": "1",
         "Content-Type": "application/json; charset=utf-8"}
USAGE_LOWER = {k.lower(): v for k, v in USAGE.items()}


class FakeResponse:
    """What matters of curl_cffi's Response: the status, the headers and the
    effective URL, which holds the key in its query as the real one does."""

    def __init__(self, status: int, headers: dict, url: str):
        self.status_code = status
        self.headers = Headers(headers)
        self.url = url


def _leaky(case: str, full: str) -> Exception:
    """What curl_cffi raises, as it reads: the full URL, key and all, in the
    text, and for its request exceptions the response (which knows the URL)
    attached."""
    rsp = FakeResponse(0, {}, full)
    if case == "timeout":
        return cx.Timeout(f"Failed to perform, curl: (28) Operation timed out. {full}", 28, rsp)
    if case == "connect":
        return cx.ConnectionError(f"Failed to perform, curl: (7) Could not connect. {full}", 7, rsp)
    if case == "dns":
        return cx.DNSError(f"Failed to perform, curl: (6) Could not resolve host. {full}", 6, rsp)
    if case == "setopt":            # a URL-option failure prints the URL with the key
        return CurlError(f"Failed to setopt CurlOpt.URL b'{full}', curl: (3) URL malformed", 3)
    return RuntimeError(f"unexpected failure for {full}")


KIND = {"timeout": "timeout", "connect": "connect", "dns": "connect", "setopt": "transport",
        "other": "transport"}


class FakeSession:
    """Plays curl_cffi's Session.get: it hands `chunks` to the content
    callback one at a time, as libcurl does, and when the callback returns
    CURL_WRITEFUNC_ERROR it stops the transfer the way curl_cffi reports it
    (CURLE_WRITE_ERROR, the URL in the text). Then it raises `fail` (a case
    of _leaky) if set, else answers `status` with `headers`."""

    def __init__(self):
        self.calls = []
        self.status, self.headers, self.chunks = 200, dict(USAGE), [b"[]"]
        self.fail = None
        self.honor_abort = True
        self.returned = []          # the callback's answer for each chunk handed over

    def get(self, url, **kw):
        self.calls.append((url, kw))
        full = url + "?" + urlencode(kw["params"])
        for chunk in self.chunks:
            rv = kw["content_callback"](chunk)
            self.returned.append(rv)
            if rv == CURL_WRITEFUNC_ERROR and self.honor_abort:
                raise cx.RequestException("Failed to perform, curl: (23) Failure writing output "
                                          f"to destination. {full}", 23, FakeResponse(0, {}, full))
        if self.fail is not None:
            raise _leaky(self.fail, full)
        return FakeResponse(self.status, self.headers, full)


@pytest.fixture
def vendor(monkeypatch):
    """A fake session in place of the blocked one, a fresh pacing record and a
    fresh thread-local, and a fake clock for the pacing (each sleep is
    recorded and moves the clock), so nothing reaches a network and no test
    waits real time."""
    fake = FakeSession()
    monkeypatch.setattr(H, "_vendor_session", lambda: fake)
    monkeypatch.setattr(H, "_vendor_last", {})
    monkeypatch.setattr(H, "_vendor_tls", threading.local())
    clock, fake.sleeps = [1000.0], []

    def sleep(s):
        fake.sleeps.append(s)
        clock[0] += s
    monkeypatch.setattr(H, "time", types.SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    return fake


def _params(**extra) -> dict:
    return {"apiKey": KEY, "dateFormat": "iso", **extra}


def _call(params=None, **kw):
    kw.setdefault("timeout", 4.0)
    kw.setdefault("secrets", (KEY,))
    return REAL(HOST, EVENTS, _params() if params is None else params, **kw)


# -- The blocks: no test reaches the vendor -----------------------------------
def test_the_real_entry_points_were_saved_before_the_session_block():
    """Otherwise every test below would exercise conftest's blocker."""
    assert (REAL.__module__, REAL.__name__) == ("bucketlib.http", "vendor_get")
    assert (REAL_SESSION.__module__, REAL_SESSION.__name__) == ("bucketlib.http", "_vendor_session")


def test_no_test_has_a_vendor_key_or_the_refresher():
    for k in ("BUCKETVALUE_ODDS_API_KEY", "BUCKETVALUE_ODDS_API_KEY_FILE", "THE_ODDS_API_KEY"):
        assert k not in os.environ, k
    assert os.environ.get("BUCKETVALUE_ODDS_REFRESHER") == "0"


def test_the_session_block_holds_through_a_function_monkeypatch_undo(monkeypatch, vendor_block):
    """tests/test_pages_smoke.py calls monkeypatch.undo() mid-suite. The
    vendor block lives on its own MonkeyPatch, so the undo cannot bring back
    the real entry point; and the real vendor_get cannot connect either,
    because its session is blocked too."""
    monkeypatch.undo()
    monkeypatch.setattr(H, "_vendor_last", {})
    assert H.vendor_get is not REAL and H._vendor_session is not REAL_SESSION
    with pytest.raises(vendor_block.error, match="a test reached the vendor network"):
        H.vendor_get(HOST, EVENTS, _params(), timeout=4.0, secrets=(KEY,))
    with pytest.raises(vendor_block.error, match="a test reached the vendor network"):
        _call()
    assert vendor_block.hits == 2


def test_the_block_cannot_be_swallowed_by_except_exception(vendor_block):
    """Review I1, F4. The probe's ask() and the feed's worker and walk catch
    Exception and log only its type (design 2.4.4). Had the block raised an
    Exception (it raised AssertionError), a test that forgot its stub would
    pass through them green; it is a BaseException, so it gets out."""
    assert issubclass(vendor_block.error, BaseException)
    assert not issubclass(vendor_block.error, Exception)

    def worker_body():
        try:
            H.vendor_get(HOST, EVENTS, _params(), timeout=4.0, secrets=(KEY,))
        except Exception as e:  # noqa: BLE001 - the shape of the code under test
            return f"worker failed ({type(e).__name__})"
        return "no error"

    with pytest.raises(vendor_block.error):
        worker_body()
    assert vendor_block.hits == 1


def test_a_block_hit_swallowed_in_a_worker_thread_is_still_counted(vendor_block):
    """A Future keeps any exception, a BaseException too, and a done-callback
    that logs its type would hide it. Every hit is counted, and a test that
    leaves one behind fails at its teardown (conftest); this one owns its
    hit, through `vendor_block`."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(H.vendor_get, HOST, EVENTS, _params(), timeout=4.0, secrets=(KEY,))
        err = fut.exception(timeout=10)
    assert isinstance(err, vendor_block.error) and vendor_block.hits == 1
    assert KEY not in str(err) and "apiKey" not in str(err)


# -- An honest client: one attempt, no impersonation, no redirect -------------
def test_the_allow_list_and_an_honest_user_agent(vendor):
    assert H.VENDOR_HOSTS == frozenset({"api.the-odds-api.com"})
    resp = _call()
    assert resp.status == 200 and resp.body == []
    (url, kw), = vendor.calls
    assert url == "https://" + HOST + EVENTS
    assert kw["headers"] == {"User-Agent": H.VENDOR_UA, "Accept": "application/json"}
    ua = kw["headers"]["User-Agent"]
    assert ua.startswith("BucketValue/")
    assert not any(b in ua for b in ("Mozilla", "Chrome", "Safari", "AppleWebKit", "Gecko"))
    # the call itself: no impersonation, no stream mode, nothing else
    assert set(kw) == {"params", "headers", "timeout", "allow_redirects", "content_callback"}
    assert kw["timeout"] == 4.0


def test_the_vendor_session_impersonates_nothing_and_retries_nothing(monkeypatch):
    """The real _vendor_session, on a recording stand-in for curl_cffi's
    Session: no `impersonate` (so no Chrome fingerprint), no retries, one
    session per thread, apart from fetch's Chrome session."""
    made = []

    class Recorder:
        def __init__(self, **kw):
            made.append(kw)
    monkeypatch.setattr(H, "cr", types.SimpleNamespace(Session=Recorder))
    monkeypatch.setattr(H, "_vendor_tls", threading.local())
    first = REAL_SESSION()
    assert REAL_SESSION() is first                  # reused on this thread
    other = []
    t = threading.Thread(target=lambda: other.append(REAL_SESSION()))
    t.start()
    t.join(timeout=5)
    assert other and other[0] is not first          # one per thread
    assert len(made) == 2
    for kw in made:
        assert kw.get("impersonate") is None and not {"ja3", "akamai", "extra_fp"} & set(kw)
        assert kw.get("retry", 0) == 0
    assert H._vendor_tls is not H._tls


def test_the_key_travels_only_in_params(vendor):
    p = _params()
    _call(p)
    assert p == _params()                           # the caller's dict is not touched
    (url, kw), = vendor.calls
    assert KEY not in url and "?" not in url
    assert kw["params"] == {"apiKey": KEY, "dateFormat": "iso"}
    assert KEY not in json.dumps(kw["headers"])
    for bad in (EVENTS + "?apiKey=" + KEY, "v4/sports", "", None):
        with pytest.raises(ValueError) as ei:
            REAL(HOST, bad, {}, timeout=4.0, secrets=(KEY,))
        assert str(ei.value) == "vendor path not allowed"
    assert len(vendor.calls) == 1


@pytest.mark.parametrize("status", [401, 403, 404, 422, 429, 500, 503])
def test_an_error_status_is_one_attempt_and_comes_back_as_an_answer(vendor, status):
    """fetch retries 401, 422 and 429; a paid feed never may."""
    vendor.status = status
    vendor.chunks = [b'{"message": "constructed", "error_code": "CONSTRUCTED_CODE"}']
    resp = _call()
    assert len(vendor.calls) == 1 and resp.status == status
    assert "CONSTRUCTED_CODE" in resp.snippet
    assert vendor.sleeps == []                      # and no backoff before or after it


def test_a_timeout_is_one_attempt_too(vendor):
    vendor.fail = "timeout"
    with pytest.raises(H.VendorTransportError) as ei:
        _call()
    assert ei.value.kind == "timeout"
    assert len(vendor.calls) == 1 and vendor.sleeps == []


def test_no_redirect_is_followed(vendor):
    """Following a 3xx could resend the query, key included, to another host."""
    vendor.status = 302
    vendor.headers = dict(USAGE, Location="https://elsewhere.example/v4/sports?apiKey=" + KEY)
    vendor.chunks = [b""]
    resp = _call()
    (url, kw), = vendor.calls
    assert kw["allow_redirects"] is False
    assert resp.status == 302 and resp.headers == USAGE_LOWER
    assert KEY not in repr(resp) and "elsewhere" not in repr(resp)


# -- Refused before anything is sent ------------------------------------------
def test_a_timeout_under_half_a_second_is_refused_before_anything_is_sent(vendor):
    """libcurl reads 0 ms as "wait forever", and curl_cffi truncates to whole
    milliseconds, so a tiny or derived timeout must never reach it."""
    for bad in (0, 0.0, 0.0004, 0.49, -1, None, "4", True, float("nan"), float("inf"), (2, 2)):
        with pytest.raises(ValueError) as ei:
            _call(timeout=bad)
        assert str(ei.value) == "vendor timeout too small", bad
    assert vendor.calls == [] and vendor.sleeps == [] and H._vendor_last == {}
    _call(timeout=0.5)
    _call(timeout=4)
    assert [kw["timeout"] for _, kw in vendor.calls] == [0.5, 4]


def test_a_host_outside_the_allow_list_is_refused(vendor):
    for bad in ("evil.example", "api.the-odds-api.com.evil.example", "API.THE-ODDS-API.COM",
                "stats.nba.com", "", None):
        with pytest.raises(ValueError) as ei:
            REAL(bad, EVENTS, _params(), timeout=4.0, secrets=(KEY,))
        assert str(ei.value) == "vendor host not allowed"
    assert vendor.calls == [] and H._vendor_last == {}


def test_a_none_param_is_refused_without_naming_it(vendor):
    """curl_cffi would send the literal string "None". The message names no
    parameter and no value."""
    with pytest.raises(H.BadRequest) as ei:
        _call(_params(eventIds=None))
    assert str(ei.value) == "BadRequest: None-valued param [vendor]"
    assert ei.value.url == "vendor" and "eventIds" not in str(ei.value)
    assert vendor.calls == []


# -- What an answer carries ---------------------------------------------------
def test_an_error_answer_carries_its_status_the_usage_headers_only_and_a_scrubbed_snippet(vendor):
    vendor.status = 401
    vendor.headers = dict(USAGE, **{"Set-Cookie": "session=constructed", "Server": "constructed",
                                    "Location": "/elsewhere", "X-Amzn-Trace-Id": "constructed"})
    body = {"message": "Usage credits are used up (constructed). The call was "
                       "/v4/sports/basketball_nba/events?apiKey=" + KEY + "&dateFormat=iso",
            "error_code": "OUT_OF_USAGE_CREDITS"}
    raw = json.dumps(body).encode()
    vendor.chunks = [raw[:7], raw[7:50], raw[50:]]
    resp = _call()
    assert resp.status == 401
    assert resp.headers == USAGE_LOWER              # the usage headers and content-type, nothing else
    assert "OUT_OF_USAGE_CREDITS" in resp.snippet   # the code survives, to be read
    assert "apiKey=[redacted]" in resp.snippet
    assert KEY not in resp.snippet and "K" * 4 not in resp.snippet and len(resp.snippet) <= 300
    assert resp.nbytes == len(raw)
    assert resp.body == body                        # parsed, whatever the status
    assert KEY not in repr(resp)


def test_a_200_has_no_snippet_and_a_body_that_is_not_json_is_none(vendor):
    vendor.chunks = [b'[{"id": "constructed", ', b'"home_team": "Detroit Pistons"}]']
    resp = _call()
    assert resp.status == 200 and resp.snippet == "" and resp.headers == USAGE_LOWER
    assert resp.body == [{"id": "constructed", "home_team": "Detroit Pistons"}]
    for raw in (b"<html>constructed</html>", b"", b'"a string"', b"12", b"null", b"[" * 100_000):
        vendor.chunks = [raw]
        resp = _call()
        assert resp.body is None and resp.nbytes == len(raw)


def test_the_response_holds_no_url_no_parameter_and_no_response_object(vendor):
    vendor.chunks = [b'{"ok": true}']
    resp = _call()
    assert [f.name for f in dataclasses.fields(resp)] == ["status", "headers", "body",
                                                          "snippet", "nbytes"]
    for text in (repr(resp), json.dumps(resp.headers), resp.snippet):
        assert KEY not in text and HOST not in text and "apiKey" not in text
    with pytest.raises(dataclasses.FrozenInstanceError):
        resp.status = 500


# -- The size cap holds DURING the download -----------------------------------
def test_an_oversize_body_is_aborted_during_the_download(vendor):
    """The content callback returns CURL_WRITEFUNC_ERROR on the chunk that
    passes VENDOR_MAX_BYTES, which curl_cffi hands straight back to libcurl,
    and libcurl stops the transfer there: the rest is never received.
    Mutation check: a cap checked after the download receives all 10 MB."""
    assert H.VENDOR_MAX_BYTES == 4_000_000
    mb = b"x" * 1_000_000
    vendor.chunks = [mb] * 10
    with pytest.raises(H.VendorTransportError) as ei:
        _call()
    err = ei.value
    assert vendor.returned == [len(mb)] * 4 + [CURL_WRITEFUNC_ERROR]
    assert err.kind == "oversize" and err.received == 5_000_000
    assert str(err) == "the vendor answer was larger than allowed"
    assert err.__cause__ is None and err.__context__ is None
    # exactly the cap is allowed
    vendor.chunks, vendor.returned = [mb] * 4, []
    resp = _call()
    assert resp.status == 200 and resp.nbytes == 4_000_000
    assert vendor.returned == [len(mb)] * 4


def test_an_oversize_body_is_refused_even_if_the_transfer_ran_on(vendor):
    """A transport that ignored the abort still never yields an answer."""
    vendor.honor_abort = False
    vendor.chunks = [b"x" * 3_000_000, b"x" * 3_000_000]
    with pytest.raises(H.VendorTransportError) as ei:
        _call()
    assert ei.value.kind == "oversize" and ei.value.received == 6_000_000


# -- Its own pacing -----------------------------------------------------------
def test_the_vendor_call_proceeds_while_another_thread_holds_the_shared_pace_lock(vendor):
    """`_pace` sleeps inside one lock shared by every host, so a vendor call
    behind it would wait out a cdn.nba.com pause. The vendor has its own.
    Mutation check: pacing the vendor on http._pace_lock fails this."""
    out, done = {}, threading.Event()

    def call():
        try:
            out["resp"] = _call()
        except BaseException as e:  # noqa: BLE001 - reported by the asserts below
            out["err"] = e
        finally:
            done.set()

    assert H._pace_lock.acquire(timeout=5)
    worker = threading.Thread(target=call, daemon=True)
    try:
        worker.start()
        finished = done.wait(timeout=5)
    finally:
        H._pace_lock.release()
        worker.join(timeout=5)
    assert finished, "the vendor call waited behind http._pace_lock"
    assert not worker.is_alive() and "err" not in out and out["resp"].status == 200


def test_the_vendor_paces_itself_and_never_touches_the_shared_pacing(vendor):
    shared_before = dict(H._last_call)
    _call()
    _call()
    assert vendor.sleeps == [pytest.approx(0.05)]   # the vendor's own gap, on the fake clock
    assert H._vendor_last == {HOST: pytest.approx(1000.05)}
    assert H._last_call == shared_before
    assert set(H._vendor_pace_locks) == set(H._VENDOR_GAP) == set(H.VENDOR_HOSTS)
    assert all(lock is not H._pace_lock for lock in H._vendor_pace_locks.values())


# -- Failures: fixed phrases, never the key, the URL or the cause -------------
@pytest.mark.parametrize("case", sorted(KIND))
def test_a_transport_error_carries_no_key_no_url_no_cause(vendor, caplog, case):
    """Mutation check, the scrubber's other half: the fake raises what
    curl_cffi raises, its text the full URL with apiKey= and the key, the
    response attached. None of it may reach the error's str or repr, its
    traceback or the log, and neither __cause__ nor __context__ may hold the
    original. Formatting `e` into the message, or raising inside the handler,
    fails this."""
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="bucketvalue")
    leak = _leaky(case, "https://" + HOST + EVENTS + "?" + urlencode(_params()))
    assert all(s in str(leak) for s in NEVER_SHOWN)     # the fake's text really carries them
    vendor.chunks = [b'{"partial": ']
    vendor.fail = case
    with pytest.raises(H.VendorTransportError) as ei:
        _call()
    err = ei.value
    shown = (str(err), repr(err), caplog.text, "".join(traceback.format_exception(err)))
    for text in shown:
        for s in NEVER_SHOWN:
            assert s not in text, (s, text)
    assert err.__cause__ is None and err.__context__ is None
    assert err.kind == KIND[case] and err.received == len(b'{"partial": ')
    assert str(err) == H.VendorTransportError.PHRASES[err.kind] and err.args == (str(err),)
    assert set(vars(err)) == {"kind", "received"}
    assert caplog.records == []
    assert H._vendor_tls.session is None            # the failed session is dropped


def test_a_failed_call_drops_this_threads_session_and_an_answer_keeps_it(vendor):
    H._vendor_tls.session = marker = object()
    vendor.status = 500
    _call()
    assert H._vendor_tls.session is marker
    vendor.fail = "connect"
    with pytest.raises(H.VendorTransportError):
        _call()
    assert H._vendor_tls.session is None


def test_failure_kinds_go_by_type_and_count_the_bytes_received(vendor):
    vendor.chunks = [b"ab", b"cde"]
    for case, kind in KIND.items():
        vendor.fail = case
        with pytest.raises(H.VendorTransportError) as ei:
            _call()
        assert (ei.value.kind, ei.value.received) == (kind, 5), case
    vendor.chunks, vendor.fail = [], "dns"          # nothing arrived at all
    with pytest.raises(H.VendorTransportError) as ei:
        _call()
    assert (ei.value.kind, ei.value.received) == ("connect", 0)


def test_the_transport_error_is_one_of_four_fixed_phrases():
    assert H.VendorTransportError.PHRASES == {
        "timeout": "the vendor did not answer in time",
        "connect": "the vendor could not be reached",
        "oversize": "the vendor answer was larger than allowed",
        "transport": "the vendor request failed"}
    e = H.VendorTransportError("not a kind " + KEY, 7)
    assert (e.kind, e.received, str(e)) == ("transport", 7, "the vendor request failed")
    assert KEY not in repr(e)


def test_vendor_get_logs_nothing(vendor, caplog):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="bucketvalue")
    _call()
    vendor.status = 429
    _call()
    vendor.fail = "timeout"
    with pytest.raises(H.VendorTransportError):
        _call()
    assert caplog.records == []


# -- The scrubber -------------------------------------------------------------
def test_scrub_removes_the_key_an_apikey_pair_and_long_tokens():
    assert H.scrub("a " + KEY + " b", (KEY,)) == "a [redacted] b"
    assert H.scrub("a " + KEY + " b") == "a [redacted] b"          # the long-token rule alone
    assert H.scrub("a " + KEY + " b", KEY) == "a [redacted] b"     # one secret passed bare
    short = "s3cr3tPw"                                              # 8 characters: used
    assert H.scrub("a " + short + " b", (short,)) == "a [redacted] b"
    assert H.scrub("a abc b", ("abc",)) == "a abc b"                # under 8: never a pattern
    assert H.scrub("/v4/sports/x/events?apiKey=abc123&dateFormat=iso") == \
        "/v4/sports/x/events?apiKey=[redacted]&dateFormat=iso"
    assert H.scrub("APIKEY: abc") == "APIKEY=[redacted]"
    assert H.scrub("api_key=abc api-key=def") == "api_key=[redacted] api-key=[redacted]"
    assert H.scrub("t" * 31) == "t" * 31 and H.scrub("t" * 32) == "[redacted]"
    assert H.scrub("OUT_OF_USAGE_CREDITS") == "OUT_OF_USAGE_CREDITS"
    assert H.scrub(None) == "" and H.scrub("  a \n\t b  ") == "a b"
    assert H.scrub(ValueError("boom " + KEY)) == "boom [redacted]"
    assert len(H.scrub("word " * 500)) == 300


def test_scrub_replaces_before_it_cuts_so_a_key_across_the_cut_never_shows():
    """Mutation check: cutting to 300 first would leave the key's first
    characters, which neither the exact key nor the long-token rule then
    matches."""
    text = "no key here. " * 22 + KEY + " tail" * 20
    assert text.index(KEY) < 300 < text.index(KEY) + len(KEY)
    for secrets in ((KEY,), ()):
        out = H.scrub(text, secrets)
        assert "K" * 4 not in out and len(out) <= 300


def test_vendor_get_scrubs_the_whole_error_body_before_it_cuts(vendor):
    """A key that straddles byte 400 of the body. Mutation check: scrubbing
    only the first 400 bytes would leave its first characters, and the run
    of spaces before it collapses, so they would show."""
    raw = b'{"message": "' + b" " * 380 + KEY.encode() + b'"}'
    assert raw.index(KEY.encode()) < 400 < raw.index(KEY.encode()) + len(KEY)
    vendor.status = 422
    vendor.chunks = [raw]
    resp = _call()
    assert "K" * 4 not in resp.snippet and "[redacted]" in resp.snippet


# -- Build 4's fix pass: review L1-2 and L1-5, and the mutation campaign's survivors ---------
def test_the_vendor_session_is_made_with_exactly_no_retries_and_nothing_else_l1_2(monkeypatch):
    """Build 4's review, L1-2: the real _vendor_session's Session kwargs are
    pinned exactly, so a session made with verify=False, a proxies mapping,
    a client cert or curl options (TLS off, or the query with its key sent
    through someone else) fails here, not only one that impersonates."""
    made = []

    class Recorder:
        def __init__(self, **kw):
            made.append(kw)
    monkeypatch.setattr(H, "cr", types.SimpleNamespace(Session=Recorder))
    monkeypatch.setattr(H, "_vendor_tls", threading.local())
    REAL_SESSION()
    t = threading.Thread(target=REAL_SESSION)
    t.start()
    t.join(timeout=5)
    assert made == [{"retry": 0}, {"retry": 0}]


class _Recording:
    """fetch's session stand-in: records each GET and answers a 200 JSON body."""

    def __init__(self):
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return types.SimpleNamespace(status_code=200, headers={"content-type": "application/json"},
                                     content=b'{"ok": true}')


def test_fetch_refuses_the_vendor_before_any_request_l1_5(monkeypatch):
    """Design 5.2 row 8 (build 4's review, L1-5): the vendor never goes
    through http.fetch, which retries billed 5xx answers, impersonates
    Chrome and copies exception text into Blocked and its log. fetch
    refuses the vendor's host, any name under its domain, in any case and
    with the root dot, with one fixed text and no request; another host
    is fetched as before."""
    rec = _Recording()
    monkeypatch.setattr(H, "_session", lambda: rec)
    monkeypatch.setattr(H, "_pace", lambda host: None)
    domain = HOST.split(".", 1)[1]
    for host in (HOST, HOST.upper(), HOST + ".", "eu." + HOST, domain, "x." + domain):
        with pytest.raises(ValueError) as ei:
            H.fetch("https://" + host + EVENTS, params={"apiKey": KEY}, retries=2, backoff=0)
        assert str(ei.value) == "a licensed vendor never goes through fetch", host
        assert KEY not in repr(ei.value)
    assert rec.calls == []
    assert H.fetch("https://cdn.nba.com/static/x.json", retries=0) == {"ok": True}
    assert H.fetch("https://evil-" + domain + "/x", retries=0) == {"ok": True}   # not the vendor's
    assert rec.calls == ["https://cdn.nba.com/static/x.json", "https://evil-" + domain + "/x"]


def test_the_kept_headers_are_scrubbed_too_ht01(vendor):
    """Survivor HT01: a kept header that carries the key (or any long token)
    comes back scrubbed, never as the vendor sent it."""
    vendor.status = 401
    vendor.headers = dict(USAGE, **{"Content-Type": "application/json; " + KEY,
                                    "X-Requests-Used": KEY})
    vendor.chunks = [b'{"error_code": "INVALID_KEY"}']
    resp = _call()
    assert KEY not in json.dumps(resp.headers)
    assert resp.headers["content-type"] == "application/json; [redacted]"
    assert resp.headers["x-requests-used"] == "[redacted]"


def test_a_name_that_merely_ends_like_the_vendors_host_is_refused_ht04(vendor):
    """Survivor HT04: the allow-list is an exact match, never a suffix one: a
    look-alike name or another of the vendor's names is refused with the
    fixed text, before any call or pacing record."""
    domain = HOST.split(".", 1)[1]
    for bad in ("evil-" + domain, "eu." + HOST, "x" + HOST, domain):
        with pytest.raises(ValueError) as ei:
            REAL(bad, EVENTS, _params(), timeout=4.0, secrets=(KEY,))
        assert str(ei.value) == "vendor host not allowed", bad
    assert vendor.calls == [] and H._vendor_last == {}


def test_a_transport_error_raised_while_the_caller_handles_another_hides_it_ht05(vendor):
    """Survivor HT05: raised `from None`, the error suppresses any context,
    so a caller that calls the vendor while handling an exception of its
    own never sees that exception printed in the error's traceback."""
    vendor.fail = "timeout"
    try:
        raise LookupError("the caller's own trouble " + KEY)
    except LookupError:
        with pytest.raises(H.VendorTransportError) as ei:
            _call()
    err = ei.value
    assert err.__suppress_context__ is True and err.__cause__ is None
    shown = "".join(traceback.format_exception(err))
    assert "the caller's own trouble" not in shown and KEY not in shown


def test_scrub_takes_one_secret_passed_bare_whatever_its_length_ht07():
    """Survivor HT07: a secret passed bare (not in a tuple) is one secret, so
    a short one (8 characters, too short for the long-token rule) is still
    removed, never read character by character."""
    assert H.scrub("a s3cr3tPw b", "s3cr3tPw") == "a [redacted] b"
    assert H.scrub("a s3cr3tPw b", ("s3cr3tPw",)) == "a [redacted] b"
