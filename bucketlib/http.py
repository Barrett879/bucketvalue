"""One HTTP client for every source BucketValue reads, with failure CLASSES.

Every source in this project has a way of failing that looks like "no data",
and the research pass (2026-10-08) found each of them live:

  - stats.nba.com does not refuse an unwelcome client, it HANGS: TLS completes
    and no byte ever arrives, ending in a read timeout. python-requests and
    curl with browser headers both hang; a curl_cffi session impersonating
    Chrome AND sending nba_api's header set answers in under a second. Both
    halves are required (dropping the headers hangs too).
  - cdn.nba.com has TWO different 403s. An Akamai HTML page ("Access Denied",
    errors.edgesuite.net) means the request was blocked, typically a missing
    Referer. An S3 XML body (<Code>AccessDenied</Code>) means the object does
    not exist (e.g. a liveData box score for a pre-2019-20 game).
  - ak-static.cms.nba.com (injury report PDFs) returns the S3 XML 403 for a
    report that is not published yet, a day with no games, AND a wrong filename
    format. `curl -o` happily saves that XML under a .pdf name.
  - curl_cffi serialises a None query param as the literal string "None"
    (requests dropped it), and stats.nba.com answers HTTP 400, which nba_api
    surfaces as KeyError('resultSet'). A 400 is OUR bug, never "missing data".
  - stats.nba.com legitimately returns 200 with full headers and ZERO rows for
    a season that has not started. That is data, not an outage.

So `fetch` never returns "nothing". It returns a parsed payload, or raises one
of the classes below, and callers decide what each class means for them.

The licensed odds feed (The Odds API) is the one exception: it goes through
`vendor_get` below, never `fetch`.
"""
from __future__ import annotations

import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

if sys.platform == "darwin":
    # macOS 27: curl_cffi's _wrapper.abi3.so looks CoreFoundation up in the
    # flat namespace and does not load it itself, so in a fresh interpreter
    # `from bucketlib import http` raised ImportError "symbol not found in
    # flat namespace '_CFRelease'" unless something (pandas) had loaded the
    # framework first. Load it, globally, before curl_cffi. Linux never runs
    # this branch.
    import ctypes
    try:
        ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
                    mode=ctypes.RTLD_GLOBAL)
    except OSError:      # no such framework: let curl_cffi's import say what is wrong
        pass

from curl_cffi import requests as cr  # noqa: E402
from curl_cffi.curl import CURL_WRITEFUNC_ERROR  # noqa: E402
from curl_cffi.requests import exceptions as cr_exc  # noqa: E402

from .cache import logger


# -- Failure classes ----------------------------------------------------------
class FetchError(Exception):
    """Base class. `url` and `detail` are always set."""

    def __init__(self, url: str, detail: str):
        super().__init__(f"{self.__class__.__name__}: {detail} [{url}]")
        self.url = url
        self.detail = detail


class Blocked(FetchError):
    """The host refused us: a hang/timeout, an Akamai HTML 403/503, or
    repeated 429/5xx. Never read this as "no games" or "nothing published"."""


class Absent(FetchError):
    """The object does not exist (S3 XML AccessDenied 403, or 404). For a
    report that may simply not be published yet, that is the expected answer."""


class BadRequest(FetchError):
    """HTTP 400: a malformed request on our side. Never retried."""


class BadPayload(FetchError):
    """200, but not the expected content (HTML where JSON was expected, an XML
    error saved as a PDF, unparseable JSON)."""


# -- Licensed vendors (The Odds API): one entry point, no retries ------------
# `fetch` is wrong for a paid feed: it retries 401, 422 and 429, drops the
# headers and the error bodies, copies str(e) into its errors and impersonates
# Chrome. `vendor_get` makes ONE attempt as an honest client, follows no
# redirect, and lets no URL, parameter or original exception out: the key
# travels only in `params`, and every failure is a fixed phrase.
VENDOR_HOSTS = frozenset({"api.the-odds-api.com"})
VENDOR_UA = "BucketValue/1.0 (+https://bucketvalue.streamlit.app)"
VENDOR_HEADERS = ("x-requests-remaining", "x-requests-used", "x-requests-last",
                  "content-type")
VENDOR_MAX_BYTES = 4_000_000      # one game, 12 markets x 5 books, is well under (UNVERIFIED size)
VENDOR_MIN_TIMEOUT = 0.5          # curl_cffi gives libcurl int(timeout * 1000) ms; 0 = no timeout
_VENDOR_GAP = {"api.the-odds-api.com": 0.05}   # 20 a second; the vendor allows 30
_vendor_pace_locks = {h: threading.Lock() for h in VENDOR_HOSTS}
_vendor_last: dict[str, float] = {}
_vendor_tls = threading.local()
_VENDOR_REQUEST_HEADERS = {"User-Agent": VENDOR_UA, "Accept": "application/json"}


@dataclass(frozen=True)
class VendorResponse:
    """A complete HTTP answer, whatever its status. It holds no URL, no
    parameter and no response object."""
    status: int                 # the HTTP status, whatever it is
    headers: dict               # lower-cased name -> str, only VENDOR_HEADERS that came back
    body: object = field(repr=False)   # parsed JSON (dict or list), or None when not JSON
    snippet: str                # non-200 only: scrub(whole body), then cut; for an error CODE
    nbytes: int                 # body bytes received


class VendorTransportError(Exception):
    """No complete HTTP response. str() is one fixed phrase. It never carries
    a URL, a parameter, a vendor message or the original exception."""
    PHRASES = {"timeout": "the vendor did not answer in time",
               "connect": "the vendor could not be reached",
               "oversize": "the vendor answer was larger than allowed",
               "transport": "the vendor request failed"}

    def __init__(self, kind: str, received: int = 0):
        self.kind = kind if kind in self.PHRASES else "transport"
        self.received = int(received)     # body bytes that arrived before the failure
        super().__init__(self.PHRASES[self.kind])


_KEY_PAIR = re.compile(r"""(?i)(api[_-]?key)\s*[=:]\s*[^&\s"'<>]+""")
_LONG_TOKEN = re.compile(r"[A-Za-z0-9]{32,}")   # the key is 40 characters (the vendor's schema)


def scrub(text, secrets: tuple[str, ...] = ()) -> str:
    """The one function that turns vendor text into a string that may leave
    the vendor path. It removes each given secret of 8 or more characters,
    any apiKey=... pair and any run of 32 or more letters and digits (an old,
    rotated key too; a 32-character event id goes with it, harmless in an
    error string), and only THEN collapses the whitespace and cuts to 300
    characters, so a key that straddles the cut is never left half shown."""
    try:
        s = "" if text is None else str(text)
    except Exception:  # noqa: BLE001 - a text that cannot be read is dropped
        s = ""
    if isinstance(secrets, str):                   # one secret passed bare
        secrets = (secrets,)
    for sec in secrets or ():
        if isinstance(sec, str) and len(sec) >= 8:
            s = s.replace(sec, "[redacted]")
    s = _KEY_PAIR.sub(r"\1=[redacted]", s)
    s = _LONG_TOKEN.sub("[redacted]", s)          # also hides an old, rotated key
    return " ".join(s.split())[:300]              # the cut comes last, after every replacement


def _vendor_session() -> cr.Session:
    """This thread's vendor session, apart from fetch's Chrome one: no
    `impersonate` (curl_cffi's default is None, so no browser fingerprint is
    sent) and no session-level retries."""
    s = getattr(_vendor_tls, "session", None)
    if s is None:
        s = cr.Session(retry=0)
        _vendor_tls.session = s
    return s


def _vendor_pace(host: str) -> None:
    """The vendor's own pacing: its own lock and gap. Never _pace_lock, which
    `_pace` holds while it sleeps for any host (a cdn.nba.com pause)."""
    with _vendor_pace_locks[host]:
        wait = _vendor_last.get(host, 0.0) + _VENDOR_GAP[host] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _vendor_last[host] = time.monotonic()


def _transport_kind(e: BaseException) -> str:
    """By type only. A curl_cffi exception's text can hold the URL, key
    included, so it is never read."""
    if isinstance(e, cr_exc.Timeout):           # libcurl's OPERATION_TIMEDOUT
        return "timeout"
    if isinstance(e, cr_exc.ConnectionError):   # DNSError included
        return "connect"
    return "transport"


def _vendor_headers(headers, secrets) -> dict:
    """Only VENDOR_HEADERS, by lower-cased name; an unreadable one is absent."""
    out = {}
    for name in VENDOR_HEADERS:
        try:
            v = headers.get(name)
        except Exception:  # noqa: BLE001
            v = None
        if v is not None:
            out[name] = scrub(v, secrets)
    return out


def vendor_get(host: str, path: str, params: dict, *, timeout: float,
               secrets: tuple[str, ...] = ()) -> VendorResponse:
    """GET https://<host><path> from a licensed vendor: ONE attempt.

    Returns a VendorResponse for any complete HTTP answer, 3xx, 4xx and 5xx
    included; no redirect is followed (one could resend the query, key and
    all, to another host). Raises VendorTransportError (a fixed phrase, its
    `kind` and the bytes `received`) when no complete answer arrived, and
    ValueError or BadRequest, with fixed texts, for a call refused before
    anything is sent. The body is capped at VENDOR_MAX_BYTES DURING the
    download. Callers pass the key in `params` and again in `secrets`, which
    are scrubbed out of everything kept. Nothing here logs."""
    if not isinstance(host, str) or host not in VENDOR_HOSTS:
        raise ValueError("vendor host not allowed")
    if not isinstance(path, str) or not path.startswith("/") or "?" in path:
        raise ValueError("vendor path not allowed")       # the key travels in params only
    params = dict(params or {})
    if any(v is None for v in params.values()):
        # curl_cffi would send the literal string "None". Name no parameter.
        raise BadRequest("vendor", "None-valued param")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout < VENDOR_MIN_TIMEOUT):
        raise ValueError("vendor timeout too small")
    _vendor_pace(host)
    session = _vendor_session()          # outside the try: a test's network block must surface
    sink = bytearray()
    oversize = []

    def _sink(chunk: bytes) -> int:      # libcurl's write function, called per chunk
        sink.extend(chunk)
        if len(sink) > VENDOR_MAX_BYTES:
            oversize.append(True)
            return CURL_WRITEFUNC_ERROR  # libcurl aborts the transfer here, mid-download
        return len(chunk)

    kind, status, headers = None, 0, {}
    try:
        r = session.get("https://" + host + path, params=params,
                        headers=dict(_VENDOR_REQUEST_HEADERS), timeout=timeout,
                        allow_redirects=False, content_callback=_sink)
    except Exception as e:  # noqa: BLE001 - only its type is read, never its text
        kind = _transport_kind(e)
    else:
        try:
            status, headers = int(r.status_code), _vendor_headers(r.headers, secrets)
        except Exception:  # noqa: BLE001 - an answer that cannot be read is no answer
            kind = "transport"
        del r                            # it knows its URL: never kept, returned or logged
    if oversize:
        kind = "oversize"
    if kind is not None:
        _vendor_tls.session = None       # a failed transfer can leave the session wedged
        # Raised OUTSIDE the except block, and from None: no __cause__ and no
        # __context__, so no traceback shows the curl_cffi exception (its request
        # object, or a URL-option failure that prints the URL with the key).
        raise VendorTransportError(kind, len(sink)) from None
    try:
        body = json.loads(bytes(sink))
    except (ValueError, RecursionError):
        body = None
    if not isinstance(body, (dict, list)):
        body = None
    snippet = "" if status == 200 else scrub(sink.decode("utf-8", "replace"), secrets)
    return VendorResponse(status=status, headers=headers, body=body, snippet=snippet,
                          nbytes=len(sink))


# -- Hosts --------------------------------------------------------------------
# nba_api 1.11.4's STATS_HEADERS, verbatim minus Host (curl_cffi sets it).
STATS_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Fetch-Dest": "empty",
}
CDN_HEADERS = {"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com"}

# Minimum spacing between requests to one host. stats.nba.com is the one that
# matters; the others are CDNs that showed no throttling at these rates.
_MIN_INTERVAL = {
    "stats.nba.com": 0.8,
    # 0.25s (4/s) drew an Akamai HTML 403 after ~860 consecutive box scores on
    # 2026-10-08 (cleared within minutes): a rate limit, not a block. 2/s.
    "cdn.nba.com": 0.5,
    "site.api.espn.com": 0.2,
    "sports.core.api.espn.com": 0.2,
    "site.web.api.espn.com": 0.2,
    "ak-static.cms.nba.com": 0.25,
}
_DEFAULT_INTERVAL = 0.5

_tls = threading.local()
_pace_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _session() -> cr.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = cr.Session(impersonate="chrome")
        _tls.session = s
    return s


def _pace(host: str) -> None:
    gap = _MIN_INTERVAL.get(host, _DEFAULT_INTERVAL)
    with _pace_lock:
        wait = _last_call.get(host, 0.0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def _headers_for(host: str) -> dict:
    if host == "stats.nba.com":
        return dict(STATS_HEADERS)
    if host.endswith("nba.com"):
        return dict(CDN_HEADERS)
    return {}


def classify(url: str, status: int, content_type: str, head: bytes) -> FetchError | None:
    """Map a non-200 response to a failure class (None for status 200)."""
    if status == 200:
        return None
    text = head[:600].decode("utf-8", "replace")
    if status == 400:
        return BadRequest(url, f"HTTP 400: {text[:200]}")
    if status == 404:
        return Absent(url, "HTTP 404")
    if status == 403:
        if "<Code>AccessDenied</Code>" in text or "xml" in content_type:
            return Absent(url, "HTTP 403 S3 AccessDenied (object does not exist)")
        return Blocked(url, f"HTTP 403 {content_type or 'unknown type'}: "
                            f"{'Akamai edge block' if 'edgesuite' in text else text[:120]}")
    if status in (429, 500, 502, 503, 504):
        return Blocked(url, f"HTTP {status}")
    return Blocked(url, f"unexpected HTTP {status}")


def _is_vendor_host(host: str) -> bool:
    """A licensed vendor's host, or any name under its domain (VENDOR_HOSTS'
    parent domains: a subdomain or the bare domain is the vendor's too), any
    case, with or without the root dot."""
    h = (host or "").strip().lower().rstrip(".")
    domains = {v.split(".", 1)[1] for v in VENDOR_HOSTS if "." in v}
    return h in VENDOR_HOSTS or any(h == d or h.endswith("." + d) for d in domains)


def fetch(url: str, *, params: dict | None = None, headers: dict | None = None,
          timeout: float = 30.0, retries: int = 2, backoff: float = 2.0,
          expect: str = "json"):
    """GET `url` and return the parsed payload.

    expect: "json" -> parsed object; "pdf" -> bytes starting %PDF;
            "text" -> str; "bytes" -> raw bytes.
    Raises Blocked / Absent / BadRequest / BadPayload. Absent and BadRequest
    are not retried; timeouts and 429/5xx/Akamai blocks are, with backoff.
    A licensed vendor's host (VENDOR_HOSTS, or a name under its domain) is
    refused with ValueError before any request: the vendor goes through
    vendor_get only (design 5.2 row 8).
    """
    host = urlparse(url).hostname or ""
    if _is_vendor_host(host):
        # Design 5.2 row 8 (build 4's review, L1-5): a licensed vendor never goes
        # through fetch, which retries billed calls, impersonates Chrome and
        # copies exception text (which can carry the URL and its key) into its
        # errors and its log. One fixed text, before any request.
        raise ValueError("a licensed vendor never goes through fetch")
    if params:
        bad = [k for k, v in params.items() if v is None]
        if bad:
            # curl_cffi would send the literal string "None" -> HTTP 400.
            raise BadRequest(url, f"None-valued params {bad}: pass '' or a value")
    hdrs = _headers_for(host)
    if headers:
        hdrs.update(headers)

    last: FetchError | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff * (2 ** (attempt - 1)))
        _pace(host)
        try:
            r = _session().get(url, params=params, headers=hdrs, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - curl_cffi raises its own types
            last = Blocked(url, f"{type(e).__name__}: {str(e)[:160]}")
            logger.info("fetch attempt %d failed: %s", attempt + 1, last)
            # A fresh session after a hang: a wedged connection can poison it.
            _tls.session = None
            continue
        ctype = (r.headers.get("content-type") or "").lower()
        err = classify(url, r.status_code, ctype, r.content[:600])
        if err is None:
            return _parse(url, r.content, ctype, expect)
        if isinstance(err, (Absent, BadRequest)):
            raise err
        last = err
        logger.info("fetch attempt %d failed: %s", attempt + 1, err)
    assert last is not None
    raise last


def _parse(url: str, body: bytes, ctype: str, expect: str):
    if expect == "bytes":
        return body
    if expect == "pdf":
        if not body.startswith(b"%PDF"):
            raise BadPayload(url, f"expected %PDF, got {body[:40]!r}")
        return body
    if expect == "text":
        return body.decode("utf-8", "replace")
    try:
        return json.loads(body)
    except ValueError:
        raise BadPayload(url, f"expected JSON ({ctype}), got {body[:80]!r}") from None


# -- stats.nba.com ------------------------------------------------------------
STATS_BASE = "https://stats.nba.com/stats/"


def stats_params(endpoint_cls, **kwargs) -> dict:
    """The full parameter dict nba_api would send for `endpoint_cls(**kwargs)`,
    built WITHOUT making its request, with None defaults turned into ''.

    MeasureType is the one None default that must be chosen explicitly
    (PlayerGameLogs/TeamGameLogs): '' would silently pick a default.
    """
    ep = endpoint_cls(get_request=False, **kwargs)
    params = dict(ep.parameters)
    if "MeasureType" in params and not params["MeasureType"]:
        raise ValueError(f"{endpoint_cls.__name__}: pass the measure type explicitly")
    return {k: ("" if v is None else v) for k, v in params.items()}


def stats_get(endpoint: str, params: dict, timeout: float = 45.0, retries: int = 3) -> dict:
    """Raw JSON from stats.nba.com/stats/<endpoint>."""
    return fetch(STATS_BASE + endpoint, params=params, timeout=timeout, retries=retries,
                 expect="json")


def stats_call(endpoint_cls, timeout: float = 45.0, retries: int = 3, **kwargs) -> dict:
    """Call an nba_api endpoint class through our client. Returns raw JSON."""
    return stats_get(endpoint_cls.endpoint, stats_params(endpoint_cls, **kwargs),
                     timeout=timeout, retries=retries)


def result_sets(payload: dict):
    """{name: DataFrame} for the classic resultSets / resultSet shapes.

    An empty rowSet with headers present is a legitimate empty frame (a season
    that has not started), returned with its columns.
    """
    import pandas as pd

    sets = payload.get("resultSets")
    if sets is None:
        rs = payload.get("resultSet")
        if rs is None:
            raise KeyError("payload has neither resultSets nor resultSet")
        sets = [rs] if isinstance(rs, dict) else rs
    if isinstance(sets, dict):  # some endpoints key the sets by name
        sets = [dict(v, name=k) for k, v in sets.items()]
    out = {}
    for s in sets:
        out[s.get("name", f"set{len(out)}")] = pd.DataFrame(s.get("rowSet", []),
                                                           columns=s.get("headers", []))
    return out
