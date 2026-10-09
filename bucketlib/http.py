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
"""
from __future__ import annotations

import json
import threading
import time
from urllib.parse import urlparse

from curl_cffi import requests as cr

from .cache import logger


# ── Failure classes ──────────────────────────────────────────────────────────
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


# ── Hosts ────────────────────────────────────────────────────────────────────
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


def fetch(url: str, *, params: dict | None = None, headers: dict | None = None,
          timeout: float = 30.0, retries: int = 2, backoff: float = 2.0,
          expect: str = "json"):
    """GET `url` and return the parsed payload.

    expect: "json" -> parsed object; "pdf" -> bytes starting %PDF;
            "text" -> str; "bytes" -> raw bytes.
    Raises Blocked / Absent / BadRequest / BadPayload. Absent and BadRequest
    are not retried; timeouts and 429/5xx/Akamai blocks are, with backoff.
    """
    host = urlparse(url).hostname or ""
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


# ── stats.nba.com ────────────────────────────────────────────────────────────
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
