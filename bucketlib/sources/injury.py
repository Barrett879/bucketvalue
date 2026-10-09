"""The NBA's official injury-report PDFs (ak-static.cms.nba.com/referee/injury/).

This is the pre-game Out / Doubtful / Questionable / Probable / Available
signal: training data for the availability model (December 2018 onward) and
the live status feed for serving. Every trap below was observed live in the
2026-10-08 research pass; each one has a test.

URLS AND ERAS. There is no archive index (official.nba.com lists only the
latest day and reuses last season's slug), so URLs are built directly. The
filename is an ET wall-clock time on a 12-hour clock ("12AM", "12PM"):
  2018-12-17 .. ~2021-10-14  Injury-Report_YYYY-MM-DD_HHAM.pdf, slots 01PM,
                             05PM, 08PM; the 2020 bubble restart (~07-30 ..
                             10-11) used 11AM, 02PM, 05PM instead.
  ~2021-10-15 .. 2025-12-22  hourly, same _HHAM|PM format (that day's 12AM is
                             the last old-format file).
  2025-12-22 (by 10AM) on    quarter-hourly, _HH_MMAM|PM, MM in 00/15/30/45.
The switch dates were bracketed, not pinned, so dates near each boundary try
both conventions. Candidate slots are stepped in UTC and converted to ET:
spring-forward slots do not exist and the fall-back 1AM has one object, so
decrementing wall-clock strings would probe names that never exist.

ABSENT IS NOT BLOCKED. A missing PDF is an S3 XML 403 (http.Absent), the same
response for "not published yet", "no games that day", "wrong filename format"
and "future". An Akamai HTML 403/503 (http.Blocked) means WE were refused and
must stop the run; reading it as "no report" would write an empty slate.

THE FILENAME TIME IS NOT THE REPORT TIME. In the hourly and 3-per-day eras the
file named HH holds the report from HH:30 (Dec 2025: HH:45); the bubble 11AM
and 02PM files are on the hour; a preseason "10_45AM" file says 10:58 AM.
Picking "the last report before tip" by filename takes the 7PM file for a
7:00 PM tip, published at 7:30, and leaks post-tip news into training. Report
time comes from the PDF title ("Injury Report: 01/15/22 05:30 PM") only.

LAYOUTS. Six column sets, one positional engine. After Game Date | Game
Time | Matchup | Team | Player Name, in date order:
  1. 2018-12-17 .. 2019-11-14 08PM (iTextSharp): Category | Reason |
     Current Status | Previous Status
  4. 2019-11-15 01PM .. 11-19 08PM: Reason | Current Status | Previous Status
  5. 2019-11-20 01PM .. 12-16 05PM: Current Status | Reason | Previous Status
  6. 2019-12-16 08PM .. 12-17 08PM: Current Status | Reason | Previous Status
     | Previous Reason
  2. 2019-12-18 01PM .. 2023-04 (iTextSharp, /Creator None): Current Status
     | Reason
(boundaries pinned live 2026-10-08 on both sides of each switch)
  3. GemBox.Document (2023-24 on): layout-2 columns, but extract_text() gives
     one WORD per line and layout mode reverses the rows.
(4-6 were found by the adversarial pass after 1-3 shipped: the builder had
sampled 2019-10-22 and 2020-01-15 only, and a month of 2019-20 raised
"unknown header". The numbers are ids, not an order.)
Plain extract_text() also wraps cells in the iTextSharp eras ("SAC@MIN
Minnesota\\nTimberwolves", a Reason continued on its own line), so every
layout is parsed by position: pypdf's visitor gives (x, y, text) per run,
columns come from the page-1 header x positions (later GemBox pages have no
header), each row is anchored on its Current Status cell, and every other
cell is assigned to the anchor it is centred on. The merged game/team cells
are top-aligned and can wrap, so a group cell belongs to the row its line
SPAN overlaps, and once placed it widens that row: in 2018-19 'Minnesota
Timberwolves' wraps to two lines centred on a one-line row, and the row's
time and matchup sit on the team name's FIRST line, 4pt above the status
(about 27% of 2018-19 files carry such a row). Cells split across a page
break are carried to the anchor on the neighbouring page. Game/matchup/team are forward-filled across rows
AND pages, and every team must be one of its matchup's two teams: the
research prototype keyed six Spurs rows to DAL@MEM on 2026-03-12 because it
only updated the matchup on player lines, and the DEN@SAS change sat on a
team-level "NOT YET SUBMITTED" line.

TEAM-LEVEL ROWS. "NOT YET SUBMITTED" (and 2018-19's "ALL PLAYERS AVAILABLE")
belong to a team and a game, never to a player. They are returned as their own
frame; a team that has not submitted has NO player rows, so a missing row then
means "unknown", not "healthy".

NON-NBA GAMES. All-Star weekend reports list the celebrity and rising-stars
games as 'UNK@UNK' / 'Non-NBA Team' (2019-02-14_01PM, 2020-02-13_05PM). Those
rows are excluded and COUNTED in meta (n_excluded_rows, n_excluded_team_rows,
excluded); raising on them threw away the real games on the same report. Only
the exact placeholder pair is excluded: one real team in a UNK matchup, or an
NBA team under UNK@UNK, still raises.

NOT EVERY PDF IS NBA DATA. QA files are published to the production bucket
(2026-10-01_12_45PM, 2026-10-07_09_45AM/10_45AM list a BOS@DET and GSW@OKC that
never existed, with reasons like "Rest - asdfe"), and Summer League reports
share the bucket, the naming and the team names. validate_rows() keeps only
rows whose (game_date, away, home) is a real NBA game.

READING THE ROWS. The 5 PM report also lists the NEXT day's games: key rows by
the Game Date column plus matchup, never by the report's date. About 30% of
'Out' rows are G League two-way/assignment players, not injuries: reason_cat
separates them. Game Time is '07:00 (ET)' with no AM/PM: PM is assumed except
12:xx (noon); the raw string is kept so the schedule can arbitrate.
"""
from __future__ import annotations

import datetime as dt
import functools
import io
import logging
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .. import http, teams
from ..cache import CACHE_DIR, READ_ONLY, read_json, write_json
from ..names import Match
from ..util import ET, UTC, now_et, now_utc, season_for_date, to_et

# pypdf logs a warning per odd xref in some iTextSharp files; the content is
# fine and the noise buries real problems in a backfill log.
logging.getLogger("pypdf").setLevel(logging.ERROR)
import pypdf  # noqa: E402

BASE_URL = "https://ak-static.cms.nba.com/referee/injury/"

STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")

# ── Eras (ET calendar dates) ────────────────────────────────────────────────
FIRST_REPORT_DATE = dt.date(2018, 12, 17)    # first file: 2018-12-17_05PM
# 2020 bubble restart: 11AM/02PM/05PM verified 07-30, 08-10, 09-15, 10-09 (01PM
# and 08PM 403 on those dates). The margins try both slot sets.
BUBBLE = (dt.date(2020, 7, 30), dt.date(2020, 10, 11))
BUBBLE_FUZZ = (dt.date(2020, 7, 15), dt.date(2020, 10, 31))
# Hourly began during the 2021 preseason (10-12_03PM 403, 10-15_03PM 200).
# Hourly names are a superset of the 3-per-day names, so starting early only
# costs a few extra 403 probes.
HOURLY_FROM = dt.date(2021, 10, 1)
# Quarter-hourly began during 2025-12-22 (12AM old format, 10_00AM new); probe
# both conventions on the surrounding days rather than hard-coding the hour.
QUARTER_FUZZ = (dt.date(2025, 12, 21), dt.date(2025, 12, 23))

_SLOTS_3_STD = (13, 17, 20)
_SLOTS_3_BUBBLE = (11, 14, 17)


# ── Slots and URLs ───────────────────────────────────────────────────────────
@dataclass(frozen=True, order=True)
class Slot:
    """One candidate filename. `nominal_et` is the time the FILENAME names; it
    is not the report time (see the module docstring) and is used only to
    order and bound the walk."""
    nominal_et: dt.datetime
    filename: str

    @property
    def url(self) -> str:
        return BASE_URL + self.filename

    @property
    def season(self) -> str:
        return season_for_date(self.nominal_et.date())

    @property
    def quarter_format(self) -> bool:
        return _QUARTER_RE.match(self.filename) is not None


_HOURLY_RE = re.compile(r"^Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(AM|PM)\.pdf$")
_QUARTER_RE = re.compile(r"^Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf$")


def slot_filename(t_et: dt.datetime, quarter: bool) -> str:
    """'Injury-Report_2026-01-15_05_00PM.pdf' / 'Injury-Report_2022-01-15_05PM.pdf'."""
    h12 = t_et.hour % 12 or 12
    ampm = "AM" if t_et.hour < 12 else "PM"
    if quarter:
        if t_et.minute % 15:
            raise ValueError(f"quarter-hour slot must be :00/:15/:30/:45, got {t_et}")
        return f"Injury-Report_{t_et:%Y-%m-%d}_{h12:02d}_{t_et.minute:02d}{ampm}.pdf"
    if t_et.minute:
        raise ValueError(f"hourly slot must be on the hour, got {t_et}")
    return f"Injury-Report_{t_et:%Y-%m-%d}_{h12:02d}{ampm}.pdf"


def parse_slot_filename(name: str) -> Slot:
    """Inverse of slot_filename. Raises ValueError on anything else."""
    name = name.rsplit("/", 1)[-1]
    m = _QUARTER_RE.match(name)
    if m:
        d, hh, mm, ap = m.groups()
    else:
        m = _HOURLY_RE.match(name)
        if not m:
            raise ValueError(f"not an injury-report filename: {name!r}")
        (d, hh, ap), mm = m.groups(), "00"
    h = int(hh)
    if not 1 <= h <= 12 or int(mm) % 15:
        raise ValueError(f"impossible slot time in {name!r}")
    h24 = (h % 12) + (12 if ap == "PM" else 0)
    day = dt.date.fromisoformat(d)
    return Slot(dt.datetime(day.year, day.month, day.day, h24, int(mm), tzinfo=ET), name)


def _wall_times(day: dt.date, step_minutes: int) -> Iterator[dt.datetime]:
    """Every ET wall time on `day` at `step_minutes`, stepped in UTC so the
    spring-forward hour is skipped and the repeated fall-back hour appears
    twice (callers de-duplicate by filename; the bucket keeps one object)."""
    start = dt.datetime(day.year, day.month, day.day, tzinfo=ET).astimezone(UTC)
    nxt = day + dt.timedelta(days=1)
    end = dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=ET).astimezone(UTC)
    t = start
    while t < end:
        yield t.astimezone(ET)
        t += dt.timedelta(minutes=step_minutes)


def slots_for_date(day: dt.date | str) -> list[Slot]:
    """Every candidate filename for one ET date, ascending by nominal time."""
    if isinstance(day, str):
        day = dt.date.fromisoformat(day)
    if day < FIRST_REPORT_DATE:
        return []
    out: list[Slot] = []
    if day < HOURLY_FROM:
        if BUBBLE[0] <= day <= BUBBLE[1]:
            hours = set(_SLOTS_3_BUBBLE)
        elif BUBBLE_FUZZ[0] <= day <= BUBBLE_FUZZ[1]:
            hours = set(_SLOTS_3_BUBBLE) | set(_SLOTS_3_STD)
        else:
            hours = set(_SLOTS_3_STD)
        for h in sorted(hours):
            t = dt.datetime(day.year, day.month, day.day, h, tzinfo=ET)
            out.append(Slot(t, slot_filename(t, quarter=False)))
        return out
    if day <= QUARTER_FUZZ[1]:
        out += [Slot(t, slot_filename(t, quarter=False)) for t in _wall_times(day, 60)]
    if day >= QUARTER_FUZZ[0]:
        out += [Slot(t, slot_filename(t, quarter=True)) for t in _wall_times(day, 15)]
    seen, uniq = set(), []
    for s in out:  # fall-back day: keep the first (EDT) 1AM, the one that exists
        if s.filename not in seen:
            seen.add(s.filename)
            uniq.append(s)
    return sorted(uniq)


def slots_descending(until_et: dt.datetime, inclusive: bool = True) -> Iterator[Slot]:
    """Candidate slots with nominal time <= until (or <), newest first, back
    to the first archived report. Crosses dates and eras."""
    until = to_et(until_et)
    until_u = until.astimezone(UTC)
    day = until.date()
    while day >= FIRST_REPORT_DATE:
        for s in reversed(slots_for_date(day)):
            t = s.nominal_et.astimezone(UTC)
            if t < until_u or (inclusive and t == until_u):
                yield s
        day -= dt.timedelta(days=1)


# ── Fetching ─────────────────────────────────────────────────────────────────
def fetch_report(url: str, *, timeout: float = 30.0, retries: int = 2) -> bytes:
    """PDF bytes. Raises http.Absent (S3 XML 403: not published / no games /
    wrong format), http.Blocked (Akamai HTML 403/503 or timeouts: stop the
    run), or http.BadPayload (a 200 that is not a whole PDF). The defaults
    suit a backfill; the app passes a short timeout and no retries (see
    live_store)."""
    data = http.fetch(url, expect="pdf", timeout=timeout, retries=retries)
    if not _pdf_complete(data):
        # A truncated body still starts '%PDF'. Cached, it would be re-read
        # (and fail) on every rerun and never re-fetched.
        raise http.BadPayload(url, f"PDF without a trailing %%EOF ({len(data)} bytes): truncated")
    return data


def _pdf_complete(data: bytes) -> bool:
    # Every archived report (4,693 files checked 2026-10-08) ends '%%EOF'
    # within its last 64 bytes; a truncated download does not.
    return data.startswith(b"%PDF") and b"%%EOF" in data[-1024:]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    # bucketlib.cache has only a private bytes writer; same tmp + os.replace
    # recipe so a killed backfill never leaves a truncated PDF behind.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}-{threading.get_ident()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Parsing ──────────────────────────────────────────────────────────────────
_GROUP_COLS = ("date", "time", "matchup", "team")
_HEAD = [("date", "Game Date"), ("time", "Game Time"), ("matchup", "Matchup"),
         ("team", "Team"), ("player", "Player Name")]
_STATUS, _REASON = ("status", "Current Status"), ("reason", "Reason")
_PREV_STATUS, _PREV_REASON = ("prev_status", "Previous Status"), ("prev_reason", "Previous Reason")
# Ids, not an order: see LAYOUTS in the module docstring for the date windows.
_LAYOUT_HEADERS = {
    1: _HEAD + [("category", "Category"), _REASON, _STATUS, _PREV_STATUS],
    2: _HEAD + [_STATUS, _REASON],
    4: _HEAD + [_REASON, _STATUS, _PREV_STATUS],
    5: _HEAD + [_STATUS, _REASON, _PREV_STATUS],
    6: _HEAD + [_STATUS, _REASON, _PREV_STATUS, _PREV_REASON],
}
# All-Star weekend placeholders (see NON-NBA GAMES above).
_NON_NBA_CODE = "UNK"
_NON_NBA_TEAM = "NON-NBA TEAM"
# Line pitch inside one wrapped cell. Rows are >= 18 (iText) / 22 (GemBox)
# apart and a wrapped team name in iText sits only 10 above the next row, so
# the cell-grouping threshold has to be the measured pitch, not a fraction of
# the row pitch.
_LINE_PITCH = {"itext": 8.0, "gembox": 14.04}
_TOL = 1.5                      # y tolerance for "same line" / "centred on"
_COL_SLACK = 4.0                # data x vs header x: iText -1, GemBox +1
_TEAM_PHRASES = {"NOT YET SUBMITTED": "not_yet_submitted",
                 "ALL PLAYERS AVAILABLE": "all_players_available"}
_TITLE_RE = re.compile(r"Injury Report:\s*(\d{2})/(\d{2})/(\d{2})\s+(\d{1,2}):(\d{2})\s*([AP]M)")
_MATCHUP_RE = re.compile(r"^([A-Z]{2,4})@([A-Z]{2,4})$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*\(ET\)$")
_DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
_HYPHEN_WRAP = re.compile(r"(?<=\w)- (?=\w)")   # 'Two- Way' -> 'Two-Way'; ' - ' kept
_FOOTER_RE = re.compile(r"\bPage\s+(\d+)\s+of\b(?:\s+(\d+))?")

ROW_COLUMNS = ["report_ts_et", "game_date", "game_time_et", "game_time_raw", "matchup",
               "away", "home", "team", "player_raw", "status", "reason_raw",
               "reason_cat", "prev_status", "prev_reason", "page"]
TEAM_ROW_COLUMNS = ["report_ts_et", "game_date", "game_time_et", "game_time_raw",
                    "matchup", "away", "home", "team", "kind", "page"]


class ParsedReport(NamedTuple):
    """meta: dict (report_ts_et, title, creator, producer, n_pages, layout
    (1-6, see LAYOUTS), columns, created_et, and parse diagnostics:
    n_excluded_rows / n_excluded_team_rows / excluded for the All-Star
    placeholders, n_carried_cells, n_wrapped_group_cells,
    n_pages_without_footer). rows: one per player listing (ROW_COLUMNS).
    team_rows: team-level NOT YET SUBMITTED / ALL PLAYERS AVAILABLE."""
    meta: dict
    rows: pd.DataFrame
    team_rows: pd.DataFrame


@dataclass
class _Frag:
    x: float
    y: float          # reading order: larger = further down the page
    text: str


@dataclass
class _Anchor:
    page: int
    y: float
    status: str | None = None
    team_kind: str | None = None
    cells: dict[str, list[tuple[int, float, str]]] = field(default_factory=dict)

    def add(self, col: str, page: int, y: float, text: str) -> None:
        self.cells.setdefault(col, []).append((page, y, text))

    def text(self, col: str) -> str:
        parts = [t for _, _, t in sorted(self.cells.get(col, []))]
        return _clean(" ".join(parts))


def _clean(s: str) -> str:
    s = " ".join(s.split())
    return _HYPHEN_WRAP.sub("-", s)


def _page_frags(page) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []

    def visit(text, cm, tm, font_dict, font_size):  # noqa: ARG001 - pypdf signature
        t = text.strip()
        if t:
            # tm[4], tm[5] are the run's origin in the producer's layout space:
            # landscape points for iTextSharp (cm only rotates the page) and
            # top-down points for GemBox (cm only flips y). Orientation is
            # settled from the title-vs-header order below, not assumed.
            out.append((float(tm[4]), float(tm[5]), t))

    page.extract_text(visitor_text=visit)
    return out


def _lines(frags: list[_Frag]) -> list[list[_Frag]]:
    """Cluster fragments into visual lines (same y within _TOL)."""
    out: list[list[_Frag]] = []
    for f in sorted(frags, key=lambda f: (f.y, f.x)):
        if out and abs(f.y - out[-1][0].y) <= _TOL:
            out[-1].append(f)
        else:
            out.append([f])
    return [sorted(line, key=lambda f: f.x) for line in out]


def _line_text(line: list[_Frag]) -> str:
    return " ".join(f.text for f in line)


def _parse_pdf_date(raw) -> dt.datetime | None:
    """PDF /CreationDate 'D:20181217173001-05'00'' -> aware ET datetime."""
    m = re.match(r"D:(\d{14})([+-])(\d{2})'?(\d{2})", str(raw or ""))
    if not m:
        return None
    base = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    off = dt.timedelta(hours=int(m.group(3)), minutes=int(m.group(4)))
    tz = dt.timezone(off if m.group(2) == "+" else -off)
    return base.replace(tzinfo=tz).astimezone(ET)


def parse_game_time(raw: str) -> dt.time | None:
    """'07:00 (ET)' -> 19:00; '12:00 (ET)' -> 12:00 (noon). The report prints
    a 12-hour clock with no AM/PM and NBA tips are afternoon/evening, so PM is
    assumed; a morning tip will disagree with the schedule (validate_rows
    flags it). None when the string is not a clock time at all."""
    m = _TIME_RE.match(raw.strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not 1 <= h <= 12 or mi > 59:
        raise ValueError(f"impossible game time {raw!r}")
    return dt.time(12 if h == 12 else h + 12, mi)


def _detect_header(line: list[_Frag]) -> tuple[int, dict[str, float]] | None:
    words = [(f.x, w) for f in line for w in f.text.split()]
    for layout, cols in _LAYOUT_HEADERS.items():
        expected = [w for _, label in cols for w in label.split()]
        if [w for _, w in words] == expected:
            xs, i = {}, 0
            for col, label in cols:
                xs[col] = words[i][0]
                i += len(label.split())
            return layout, xs
    return None


def _place_cell(hit: list[int], idxs: list[int], anchors: list[_Anchor], pi: int,
                first: float, last: float, text: str) -> int | None:
    """Anchor index for one cell; None when it cannot be placed."""
    if len(hit) > 1:
        raise ValueError(f"page {pi + 1}: text {text!r} fits {len(hit)} rows")
    if hit:
        return hit[0]
    if not idxs or last < anchors[idxs[0]].y:
        # Above the first row on this page: the tail of a row that began on
        # the previous page (Barnes' 'Impingement', 2026-03-12_12_00PM p6).
        prev = max((i for i, a in enumerate(anchors) if a.page < pi), default=None)
        if prev is None:
            raise ValueError(f"page {pi + 1}: text {text!r} before the first row")
        return prev
    if first > anchors[idxs[-1]].y:
        # Below the last row: the head of a row continued overleaf.
        nxt = next((i for i, a in enumerate(anchors) if a.page > pi), None)
        if nxt is None:
            raise ValueError(f"page {pi + 1}: text {text!r} after the last row")
        return nxt
    return None


def parse_report(pdf_bytes: bytes) -> ParsedReport:
    """Parse one injury-report PDF. Raises ValueError, with the page and y, on
    anything it does not understand (unknown header, status, team code,
    matchup, a team outside its matchup, duplicate rows, unassignable text)."""
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(f"not a PDF (starts {pdf_bytes[:40]!r}); an S3 403 XML body saved as .pdf?")
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        info = reader.metadata or {}
        n_pages = len(reader.pages)
    except pypdf.errors.PyPdfError as e:  # truncated download, broken xref
        raise ValueError(f"unreadable PDF: {e}") from e
    creator = info.get("/Creator")
    producer = info.get("/Producer")
    creator = None if creator is None else str(creator)
    producer = None if producer is None else str(producer)
    if "GemBox" in (creator or "") or "GemBox" in (producer or ""):
        family = "gembox"
    elif "iText" in (producer or ""):
        family = "itext"
    else:
        raise ValueError(f"unknown PDF producer creator={creator!r} producer={producer!r}")

    try:
        raw_pages = [_page_frags(p) for p in reader.pages]
    except pypdf.errors.PyPdfError as e:
        raise ValueError(f"unreadable PDF page: {e}") from e
    if not raw_pages or not raw_pages[0]:
        raise ValueError("page 1 has no text")

    # Orientation: the title sits above the header row in reading order.
    p0 = raw_pages[0]
    hdr_ty = next((ty for _, ty, t in p0 if t == "Matchup"), None)
    title_ty = next((ty for _, ty, t in p0 if t.startswith("Injury")
                     and any(abs(ty2 - ty) <= _TOL and "Report" in t2 for _, ty2, t2 in p0)), None)
    if hdr_ty is None or title_ty is None:
        raise ValueError("page 1: no 'Injury Report' title or no 'Matchup' header")
    ydown = title_ty < hdr_ty
    pages = [[_Frag(x, ty if ydown else -ty, t) for x, ty, t in fr] for fr in raw_pages]

    title_line = next(ln for ln in _lines(pages[0]) if _line_text(ln).startswith("Injury"))
    title = _line_text(title_line)
    m = _TITLE_RE.search(title)
    if not m:
        raise ValueError(f"unparseable title {title!r}")
    mo, da, yy, hh, mi, ap = m.groups()
    h24 = (int(hh) % 12) + (12 if ap == "PM" else 0)
    report_ts = dt.datetime(2000 + int(yy), int(mo), int(da), h24, int(mi), tzinfo=ET)

    layout = None
    xs: dict[str, float] = {}
    page_lines: list[list[list[_Frag]]] = []
    n_no_footer = 0
    for pi, frags in enumerate(pages):
        lines = _lines(frags)
        if not lines:
            # Every archived page carries at least the title; a page pypdf
            # reads no text from would drop its rows without a trace.
            raise ValueError(f"page {pi + 1} of {n_pages} has no extractable text")
        if not _line_text(lines[0]).startswith("Injury Report"):
            raise ValueError(f"page {pi + 1}: first line is not the title: {_line_text(lines[0])[:60]!r}")
        lines = lines[1:]
        if lines:
            hdr = _detect_header(lines[0])
            if hdr is not None:
                if layout is None:
                    layout, xs = hdr
                elif hdr[0] != layout:
                    raise ValueError(f"page {pi + 1}: header changes layout {layout}->{hdr[0]}")
                lines = lines[1:]
            elif pi == 0:
                raise ValueError(f"page 1: unknown header {_line_text(lines[0])!r}")
        # Footer 'Page N of M' (iText also prints a stray total below it). It
        # must agree with the page we are on and the file's page count, or a
        # page went missing between the footer's writer and pypdf.
        cut = None
        for i, ln in enumerate(lines):
            fm = _FOOTER_RE.search(_line_text(ln))
            if fm:
                cut = i
                if int(fm.group(1)) != pi + 1 or (fm.group(2) and int(fm.group(2)) != n_pages):
                    raise ValueError(f"page {pi + 1} of {n_pages}: footer says {fm.group(0)!r}")
                break
        if cut is None:
            n_no_footer += 1
        else:
            lines = lines[:cut]
        page_lines.append(lines)
    if layout is None:
        raise ValueError("no header row on page 1")
    if family == "gembox" and layout != 2:
        raise ValueError(f"GemBox PDF with iText column set {layout}: unknown layout")
    final_layout = 3 if family == "gembox" else layout

    cols_sorted = sorted(xs.items(), key=lambda kv: kv[1])
    starts = [(c, x - _COL_SLACK) for c, x in cols_sorted]

    def col_of(x: float) -> str:
        c = starts[0][0]
        for name, s in starts:
            if x >= s:
                c = name
        return c

    # ── Anchors (one per row) and per-column text on every line ──
    anchors: list[_Anchor] = []
    page_anchor_idx: list[list[int]] = []
    page_col_lines: list[dict[str, list[tuple[float, str]]]] = []
    for pi, lines in enumerate(page_lines):
        idxs: list[int] = []
        col_lines: dict[str, list[tuple[float, str]]] = {}
        for ln in lines:
            y = sum(f.y for f in ln) / len(ln)
            by_col: dict[str, list[_Frag]] = {}
            for f in ln:
                by_col.setdefault(col_of(f.x), []).append(f)
            texts = {c: " ".join(f.text for f in fs) for c, fs in by_col.items()}
            body = " ".join(texts[c] for c, _ in cols_sorted
                            if c in texts and c not in _GROUP_COLS)
            kind = _TEAM_PHRASES.get(" ".join(body.upper().split()))
            status = texts.get("status")
            if kind is not None:
                anchors.append(_Anchor(pi, y, team_kind=kind))
                idxs.append(len(anchors) - 1)
                texts = {c: t for c, t in texts.items() if c in _GROUP_COLS}
            elif status is not None:
                st = " ".join(status.split())
                if st not in STATUSES:
                    raise ValueError(f"page {pi + 1} y={y:.1f}: unknown status {st!r}")
                anchors.append(_Anchor(pi, y, status=st))
                idxs.append(len(anchors) - 1)
                texts.pop("status")
            for c, t in texts.items():
                col_lines.setdefault(c, []).append((y, t))
        page_anchor_idx.append(idxs)
        page_col_lines.append(col_lines)

    # ── Cells: consecutive lines of one column within one line pitch ──
    pitch = _LINE_PITCH[family]
    row_cells: list[tuple[int, str, list[tuple[float, str]]]] = []
    group_cells: list[tuple[int, str, list[tuple[float, str]]]] = []
    for pi, col_lines in enumerate(page_col_lines):
        for col, entries in col_lines.items():
            entries.sort()
            cells: list[list[tuple[float, str]]] = []
            for y, t in entries:
                if cells and y - cells[-1][-1][0] <= pitch + 1.0:
                    cells[-1].append((y, t))
                else:
                    cells.append([(y, t)])
            (group_cells if col in _GROUP_COLS else row_cells).extend((pi, col, c) for c in cells)

    unplaced: list[str] = []
    carried: list[str] = []
    n_wrapped_group = 0
    extent = {i: (a.y, a.y) for i, a in enumerate(anchors)}

    def put(target: int, pi: int, col: str, cell, hit) -> None:
        if target not in hit:
            carried.append(f"p{pi + 1} {col}={' '.join(t for _, t in cell)!r} -> "
                           f"p{anchors[target].page + 1} y={anchors[target].y:.1f}")
        for y, t in cell:
            anchors[target].add(col, pi, y, t)

    def edge_or_unplaced(pi: int, col: str, cell) -> None:
        first, last = cell[0][0], cell[-1][0]
        text = " ".join(t for _, t in cell)
        target = _place_cell([], page_anchor_idx[pi], anchors, pi, first, last, text)
        if target is None:
            unplaced.append(f"p{pi + 1} y={(first + last) / 2:.1f} {col}={text!r}")
        else:
            put(target, pi, col, cell, [])

    # Row cells (player, reason, ...) are vertically centred on their row, so
    # a cell belongs to the anchor at its centre; this also measures each
    # row's extent.
    for pi, col, cell in row_cells:
        first, last = cell[0][0], cell[-1][0]
        center = (first + last) / 2
        idxs = page_anchor_idx[pi]
        hit = [i for i in idxs if abs(anchors[i].y - center) <= _TOL]
        if not hit:
            edge_or_unplaced(pi, col, cell)
            continue
        target = _place_cell(hit, idxs, anchors, pi, first, last, " ".join(t for _, t in cell))
        lo, hi = extent[target]
        extent[target] = (min(lo, first), max(hi, last))
        put(target, pi, col, cell, hit)

    # Group cells (date, time, matchup, team) are top-aligned and may wrap:
    # one belongs to the row its [first, last] span overlaps, and once placed
    # it widens that row. Repeat until nothing more fits, because the time
    # and matchup on a wrapped team name's first line only overlap the row
    # after the team cell has widened it. Page-edge carries are tried LAST:
    # trying them first would carry a top-of-page matchup that sits 4pt above
    # the page's first status to the previous page.
    def overlaps(i: int, first: float, last: float) -> bool:
        return extent[i][0] - _TOL <= last and first <= extent[i][1] + _TOL

    pending = list(group_cells)
    while pending:
        rest = []
        for pi, col, cell in pending:
            first, last = cell[0][0], cell[-1][0]
            idxs = page_anchor_idx[pi]
            hit = [i for i in idxs if overlaps(i, first, last)]
            if len(hit) > 1:   # the top-aligned rule settles a tie
                hit = [i for i in hit if extent[i][0] - _TOL <= first <= extent[i][1] + _TOL]
            if not hit:
                rest.append((pi, col, cell))
                continue
            target = _place_cell(hit, idxs, anchors, pi, first, last, " ".join(t for _, t in cell))
            lo, hi = extent[target]
            if not lo - _TOL <= first <= hi + _TOL:
                n_wrapped_group += 1
            extent[target] = (min(lo, first), max(hi, last))
            put(target, pi, col, cell, hit)
        if len(rest) == len(pending):
            break
        pending = rest
    for pi, col, cell in pending:
        edge_or_unplaced(pi, col, cell)
    if unplaced:
        # A cell neither centred on a row, inside one, nor at a page edge:
        # a layout this parser does not understand. Guessing the nearest row
        # is how the prototype polluted reasons, so refuse.
        raise ValueError(f"{len(unplaced)} unplaceable cells: {unplaced[:5]}")

    # ── Forward-fill game context and build rows ──
    cur: dict[str, object] = {"date": None, "time": None, "time_raw": None, "matchup": None,
                              "away": None, "home": None, "team": None, "non_nba": False}
    rows, team_rows, excluded = [], [], []
    n_unparsed_times = n_no_comma = 0
    n_excluded_rows = n_excluded_team_rows = 0
    for a in anchors:
        where = f"page {a.page + 1} y={a.y:.1f}"
        g = {c: a.text(c) for c in _GROUP_COLS if a.cells.get(c)}
        if "date" in g:
            dm = _DATE_RE.match(g["date"])
            if not dm:
                raise ValueError(f"{where}: bad game date {g['date']!r}")
            cur["date"] = dt.date(int(dm.group(3)), int(dm.group(1)), int(dm.group(2)))
            if "matchup" not in g:
                raise ValueError(f"{where}: new game date without a matchup")
        if "time" in g:
            cur["time_raw"] = g["time"]
            cur["time"] = parse_game_time(g["time"])
            if cur["time"] is None:
                n_unparsed_times += 1
            if "matchup" not in g:
                raise ValueError(f"{where}: new game time without a matchup")
        if "matchup" in g:
            mm = _MATCHUP_RE.match(g["matchup"].replace(" ", ""))
            if not mm:
                raise ValueError(f"{where}: bad matchup {g['matchup']!r}")
            if mm.group(1) == mm.group(2) == _NON_NBA_CODE:
                cur.update(matchup=f"{_NON_NBA_CODE}@{_NON_NBA_CODE}", away=None, home=None,
                           team=None, non_nba=True)
            else:
                away, home = teams.canonical(mm.group(1)), teams.canonical(mm.group(2))
                if away is None or home is None or away == home:
                    raise ValueError(f"{where}: unknown team code in matchup {g['matchup']!r}")
                cur.update(matchup=f"{away}@{home}", away=away, home=home, non_nba=False)
            if "team" not in g:
                raise ValueError(f"{where}: new matchup {g['matchup']!r} without a team")
        if "team" in g:
            if cur["non_nba"]:
                # Only the placeholder team may sit under UNK@UNK: an NBA team
                # there means the matchup cell landed on the wrong row.
                if " ".join(g["team"].upper().split()) != _NON_NBA_TEAM:
                    raise ValueError(f"{where}: team {g['team']!r} under {cur['matchup']}")
            else:
                code = teams.canonical(g["team"])
                if code is None:
                    raise ValueError(f"{where}: unknown team {g['team']!r}")
                cur["team"] = code
        if cur["date"] is None or cur["matchup"] is None:
            raise ValueError(f"{where}: row before any game date/matchup/team")
        if cur["non_nba"]:
            what = a.team_kind or f"{a.text('player')!r} {a.status}"
            excluded.append(f"{where} {cur['date']} {cur['matchup']}: {what}")
            if a.team_kind is not None:
                n_excluded_team_rows += 1
            else:
                n_excluded_rows += 1
            continue
        if cur["team"] is None:
            raise ValueError(f"{where}: row before any game date/matchup/team")
        if cur["team"] not in (cur["away"], cur["home"]):
            # The research prototype's mis-key: a matchup change it never saw.
            raise ValueError(f"{where}: team {cur['team']} is not in matchup {cur['matchup']}")
        base = dict(report_ts_et=report_ts, game_date=cur["date"], game_time_et=cur["time"],
                    game_time_raw=cur["time_raw"], matchup=cur["matchup"], away=cur["away"],
                    home=cur["home"], team=cur["team"], page=a.page + 1)
        if a.team_kind is not None:
            extra = [c for c in a.cells if c not in _GROUP_COLS]
            if extra:
                raise ValueError(f"{where}: team-level row carries {extra}")
            team_rows.append(dict(base, kind=a.team_kind))
            continue
        player = re.sub(r"\s*,\s*", ", ", a.text("player")).strip()
        if not player:
            raise ValueError(f"{where}: status {a.status!r} with no player name")
        if "," not in player:
            n_no_comma += 1
        if "category" in xs:
            cat, why = a.text("category"), a.text("reason")
            reason = cat if why in ("", "-") else (f"{cat} - {why}" if cat else why)
        else:
            reason = a.text("reason")
        prev = prev_reason = None
        if "prev_status" in xs:
            prev = a.text("prev_status") or None
            if prev == "-":
                prev = None
            if prev is not None and prev not in STATUSES:
                raise ValueError(f"{where}: unknown previous status {prev!r}")
        if "prev_reason" in xs:
            prev_reason = a.text("prev_reason") or None
            if prev_reason == "-":
                prev_reason = None
        for phrase in _TEAM_PHRASES:
            if phrase in reason.upper() or phrase in player.upper():
                raise ValueError(f"{where}: team-level text leaked into {player!r}: {reason!r}")
        rows.append(dict(base, player_raw=player, status=a.status, reason_raw=reason,
                         reason_cat=reason_category(reason), prev_status=prev,
                         prev_reason=prev_reason))

    rows_df = pd.DataFrame(rows, columns=ROW_COLUMNS)
    team_df = pd.DataFrame(team_rows, columns=TEAM_ROW_COLUMNS)
    _check_tables(rows_df, team_df)

    meta = dict(report_ts_et=report_ts, title=title, creator=creator, producer=producer,
                n_pages=n_pages, layout=final_layout, columns=[c for c, _ in cols_sorted],
                created_et=_parse_pdf_date(info.get("/CreationDate")),
                n_rows=len(rows_df), n_team_rows=len(team_df),
                n_excluded_rows=n_excluded_rows, n_excluded_team_rows=n_excluded_team_rows,
                excluded=excluded,
                n_carried_cells=len(carried), carried_cells=carried,
                n_wrapped_group_cells=n_wrapped_group, n_pages_without_footer=n_no_footer,
                n_fallback_cells=0, fallback_cells=[],
                n_unparsed_times=n_unparsed_times, n_names_without_comma=n_no_comma)
    return ParsedReport(meta, rows_df, team_df)


def _check_tables(rows_df: pd.DataFrame, team_df: pd.DataFrame) -> None:
    """Report-level invariants. One player twice under one game is a parse
    error (two rows merged into one anchor's neighbour, or a page parsed
    twice), and a team that has not submitted cannot also list players."""
    dup = rows_df.duplicated(["game_date", "matchup", "team", "player_raw"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate player rows: {rows_df.loc[dup, ['matchup', 'player_raw']].values.tolist()}")
    tdup = team_df.duplicated(["game_date", "matchup", "team"], keep=False)
    if tdup.any():
        raise ValueError(f"duplicate team-level rows: {team_df.loc[tdup, ['matchup', 'team']].values.tolist()}")
    both = team_df.merge(rows_df, on=["game_date", "matchup", "team"])
    if len(both):
        raise ValueError(f"team both NOT YET SUBMITTED and listing players: {both.team.unique().tolist()}")


# ── Reason taxonomy ──────────────────────────────────────────────────────────
REASON_CATS = ("injury_illness", "g_league_two_way", "g_league_assignment", "g_league",
               "not_with_team", "personal", "suspension", "coach_decision", "rest",
               "health_protocols", "concussion_protocol", "trade_pending",
               "reconditioning", "unspecified", "other")

# Matched against a key with punctuation folded to spaces ("Coach's Decision"
# -> "coach s decision", "G League - Two-Way" -> "g league two way"), in order.
_REASON_RULES: list[tuple[re.Pattern, str]] = [
    # Every concussion goes through the league's return-to-play protocol;
    # printed both as a lead ('Concussion Protocol') and as an Injury/Illness
    # detail ('Injury/Illness - N/A; Concussion').
    (re.compile(r"\bconcussion"), "concussion_protocol"),
    # The ramp-up after an absence is filed under every lead: 'Injury/Illness
    # - N/a; Return to Competition Reconditioning/Back Soreness', 'Not with
    # Team - Return to Competition Reconditioning', "Coach's Decision - Post
    # Health and Safety Protocols Reconditioning" (530 such rows across the
    # 5,235 cached files, 2026-10-08). Matched anywhere, like concussion.
    (re.compile(r"\breconditioning\b"), "reconditioning"),
    # 2018-19 filed rest under Injury/Illness ('Injury/Illness - Rest',
    # 'Load Management, Return from Achilles Injury'): schedule-driven
    # absences, which an availability model must not read as injuries.
    (re.compile(r"^injury ?/? ?illness (rest|load management)\b"), "rest"),
    (re.compile(r"^injury ?/? ?illness\b"), "injury_illness"),
    # 2018-19 'G League Team - NBA G League Assignment (Two Way)'.
    (re.compile(r"^g league (team )?.*\btwo ?way\b"), "g_league_two_way"),
    (re.compile(r"^g league (on )?assignment\b"), "g_league_assignment"),
    # 2018-19 Category 'G League Team' did not say two-way or assignment, and
    # its detail can be an injury ('G League Team - Left Ankle Surgery'): the
    # player is with the G League team either way.
    (re.compile(r"^g league( team)?\b"), "g_league"),
    (re.compile(r"^not with team\b"), "not_with_team"),
    (re.compile(r"^personal\b"), "personal"),
    (re.compile(r"^((league|team|nba) )?suspen(sion|ded)\b"), "suspension"),
    (re.compile(r"^coach(e?s| s)? decision\b"), "coach_decision"),
    (re.compile(r"^rest\b"), "rest"),
    (re.compile(r"^(nba )?(covid|health (and |& )?safety)"), "health_protocols"),
    (re.compile(r"^trade pending\b"), "trade_pending"),
]


def _reason_key(reason_raw) -> str:
    s = str(reason_raw or "").lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9/&]+", " ", s).split())


def reason_category(reason_raw) -> str:
    """Map a Reason string to REASON_CATS. Unknown leads -> 'other' (callers
    count them); an empty or '-' reason -> 'unspecified'."""
    key = _reason_key(reason_raw)
    if not key:
        return "unspecified"
    for pat, cat in _REASON_RULES:
        if pat.search(key):
            return cat
    return "other"


def reason_lead(reason_raw) -> str:
    """The part before ' - ' / ';' ('Injury/Illness', 'G League'): what the
    coverage report lists for the 'other' bucket."""
    s = " ".join(str(reason_raw or "").split())
    return re.split(r" - |;", s, maxsplit=1)[0].strip()


# ── Schedule validation ──────────────────────────────────────────────────────
SCHEDULE_COLUMNS = ("game_id", "game_date", "away", "home", "tip_utc")
TIP_SLACK_MIN = 10


def _as_date(v) -> dt.date:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, pd.Timestamp):
        return v.date()
    return dt.date.fromisoformat(str(v)[:10])


def validate_rows(rows: pd.DataFrame, schedule_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(kept, dropped). Kept rows are on a real NBA game and gain game_id,
    tip_utc, swapped (the PDF had home/away reversed vs the schedule; same
    teams, same date, so the same game) and tip_mismatch (the printed time
    disagrees with the schedule, reading the 12-hour clock either way).
    Dropped rows (QA files, Summer League, anything off-schedule) are
    returned whole with drop_reason, never silently discarded. Works on the
    player frame and on the team-level frame alike."""
    missing = [c for c in SCHEDULE_COLUMNS if c not in schedule_df.columns]
    if missing:
        raise ValueError(f"schedule_df missing columns {missing}")
    sched = schedule_df.loc[:, list(SCHEDULE_COLUMNS)].copy()
    sched["game_date"] = sched["game_date"].map(_as_date)
    for c in ("away", "home"):
        bad = sched[c].map(teams.canonical).isna()
        if bad.any():
            raise ValueError(f"schedule has unknown team codes in {c}: {sched.loc[bad, c].unique()[:5]}")
        sched[c] = sched[c].map(teams.canonical)
    if sched.duplicated(["game_date", "away", "home"]).any():
        raise ValueError("schedule has duplicate (game_date, away, home)")
    sched["_pair"] = [frozenset(p) for p in zip(sched.away, sched.home)]
    if sched.duplicated(["game_date", "_pair"]).any():
        raise ValueError("schedule has the same two teams twice on one date")
    by_pair = {(d, p): (gid, a, tip) for d, p, gid, a, tip in
               zip(sched.game_date, sched._pair, sched.game_id, sched.away, sched.tip_utc)}

    out = rows.copy()
    if out.empty:
        kept = out.assign(game_id=pd.Series(dtype=object), tip_utc=pd.Series(dtype=object),
                          swapped=pd.Series(dtype=bool), tip_mismatch=pd.Series(dtype=bool))
        return kept, out.assign(drop_reason=pd.Series(dtype=object))
    gids, tips, swapped, ok = [], [], [], []
    for d, a, h in zip(out.game_date.map(_as_date), out.away, out.home):
        hit = by_pair.get((d, frozenset((a, h))))
        ok.append(hit is not None)
        gids.append(hit[0] if hit else None)
        tips.append(hit[2] if hit else None)
        swapped.append(bool(hit) and hit[1] != a)
    out["game_id"], out["tip_utc"], out["swapped"] = gids, tips, swapped
    mask = pd.Series(ok, index=out.index)
    kept = out[mask].copy()
    dropped = out[~mask].drop(columns=["game_id", "tip_utc", "swapped"]).assign(
        drop_reason="not_in_schedule")

    def _mismatch(t, tip) -> bool:
        # On a 12-hour dial (the PDF has no AM/PM), with 10 minutes of slack:
        # stats.nba.com's schedule carries 8:01 PM for an '08:00 (ET)' tip.
        if t is None or tip is None or (isinstance(t, float) and t != t):
            return False
        e = to_et(pd.Timestamp(tip).to_pydatetime() if not isinstance(tip, str) else tip)
        d = abs((t.hour % 12) * 60 + t.minute - ((e.hour % 12) * 60 + e.minute)) % 720
        return min(d, 720 - d) > TIP_SLACK_MIN

    kept["tip_mismatch"] = [_mismatch(t, tip) for t, tip in zip(kept.game_time_et, kept.tip_utc)]
    return kept, dropped


def lists_nba_game(rep: ParsedReport, schedule_df: pd.DataFrame) -> bool:
    """True when the report has at least one player row OR team-level row on
    a real NBA game in `schedule_df` (validate_rows). This is the report-level
    test for 'a real report': QA files (fake games) and Summer League files
    (Summer League games) fail it; a report whose only content is NOT YET
    SUBMITTED lines for tonight's games passes it."""
    kept, _ = validate_rows(rep.rows, schedule_df)
    if len(kept):
        return True
    tkept, _ = validate_rows(rep.team_rows, schedule_df)
    return len(tkept) > 0


# ── Raw store: on-disk PDFs + an index of known-absent slots ─────────────────
class ReportStore:
    """Resumable access to report PDFs.

    A present report is its PDF on disk (cache/raw/injury/<season>/<filename>).
    An absent slot is remembered in <season>/_absent.json.gz ONLY once its
    nominal time is `absent_final_hours` old: a slot from ten minutes ago may
    still be uploaded, a slot from last season will not. `budget` caps live
    fetches (a --sample run must stay inside its allowance).

    persist=False keeps everything in memory (no disk reads or writes): the
    app's store, since a READ_ONLY checkout is shared by every visitor.

    A PDF that fails to parse is remembered with its error: get() re-raises
    the same ValueError without re-reading the file, so a backfill counts and
    reports each bad file once (stats['parse_failed']), not once per tip.

    <season> is util.season_for_date of the SLOT's date (August 1 cutover),
    a storage key only: the 2020 bubble's October Finals PDFs sit under
    2020-21/. Raw PDFs are 14-100 KB; a season of selected reports is
    about 250 files (3-per-day era) to 1,400 (quarter-hour era, measured
    6.7 per game date), 10-120 MB."""

    def __init__(self, root: Path | None = None, *, fetcher: Callable[[str], bytes] = fetch_report,
                 budget: int | None = None, offline: bool = False, persist: bool = True,
                 absent_final_hours: float = 48.0, memo_size: int = 256):
        self.root = Path(root) if root is not None else CACHE_DIR / "raw" / "injury"
        self.fetcher = fetcher
        self.budget = budget
        self.offline = offline
        self.persist = persist
        self.absent_final = dt.timedelta(hours=absent_final_hours)
        self._absent: dict[str, dict[str, str]] = {}
        self._dirty: set[str] = set()
        self._memo: OrderedDict[str, ParsedReport] = OrderedDict()
        self._memo_size = memo_size
        self._bytes: dict[str, bytes] = {}          # persist=False only
        self.failed: dict[str, ValueError] = {}     # filename -> parse error
        self.stats = Counter()
        self.fetch_seconds = 0.0
        self.parse_seconds = 0.0

    def path(self, slot: Slot) -> Path:
        return self.root / slot.season / slot.filename

    def _absent_index(self, season: str) -> dict[str, str]:
        if season not in self._absent:
            data = read_json(self.root / season / "_absent.json.gz") if self.persist else None
            self._absent[season] = dict(data or {})
        return self._absent[season]

    def flush(self) -> None:
        """Checkpoint the absent index (called every 25 probes and at exit)."""
        if not self.persist:
            self._dirty.clear()
            return
        for season in sorted(self._dirty):
            # Merge with what is on disk: the horizon snapshot backfill and
            # this one share the store and run concurrently, and a plain
            # overwrite drops the other run's entries (costing re-probes).
            path = self.root / season / "_absent.json.gz"
            merged = dict(read_json(path) or {})
            merged.update(self._absent[season])
            self._absent[season] = merged
            write_json(path, merged, gz=True)
        self._dirty.clear()

    def known_absent(self, slot: Slot) -> bool:
        return slot.filename in self._absent_index(slot.season)

    def _cached(self, slot: Slot) -> bytes | None:
        if not self.persist:
            return self._bytes.get(slot.filename)
        p = self.path(slot)
        if not p.exists():
            return None
        data = p.read_bytes()
        if not _pdf_complete(data):
            # Written before fetch_report checked for truncation: re-fetch.
            self.stats["cache_corrupt"] += 1
            return None
        return data

    def get_bytes(self, slot: Slot) -> bytes | None:
        """PDF bytes, or None for an absent slot (offline: not on disk, counted
        in stats['offline_miss']). Blocked/BadPayload raise."""
        data = self._cached(slot)
        if data is not None:
            self.stats["cache_hit"] += 1
            return data
        if self.known_absent(slot):
            self.stats["absent_cached"] += 1
            return None
        if self.offline:
            self.stats["offline_miss"] += 1
            return None
        if self.budget is not None and self.stats["live"] >= self.budget:
            raise RuntimeError(f"live fetch budget of {self.budget} exhausted at {slot.filename}")
        self.stats["live"] += 1
        t0 = time.monotonic()
        try:
            data = self.fetcher(slot.url)
        except http.Absent:
            self.fetch_seconds += time.monotonic() - t0
            self.stats["absent_live"] += 1
            if now_utc() - slot.nominal_et.astimezone(UTC) >= self.absent_final:
                self._absent_index(slot.season)[slot.filename] = now_et().isoformat(timespec="seconds")
                self._dirty.add(slot.season)
                if self.stats["absent_live"] % 25 == 0:
                    self.flush()
            return None
        self.fetch_seconds += time.monotonic() - t0
        if not _pdf_complete(data):  # a custom fetcher that skipped the check
            raise http.BadPayload(slot.url, f"PDF without a trailing %%EOF ({len(data)} bytes)")
        self.stats["fetched"] += 1
        if self.persist:
            _atomic_write_bytes(self.path(slot), data)
        else:
            self._bytes[slot.filename] = data
        return data

    def get(self, slot: Slot) -> ParsedReport | None:
        """Parsed report, or None when the slot is absent. A file that does
        not parse raises ValueError, the same one on every later call."""
        if slot.filename in self._memo:
            self._memo.move_to_end(slot.filename)
            return self._memo[slot.filename]
        if slot.filename in self.failed:
            raise self.failed[slot.filename]
        data = self.get_bytes(slot)
        if data is None:
            return None
        t0 = time.monotonic()
        try:
            rep = parse_report(data)
        except ValueError as e:
            self.failed[slot.filename] = e
            self.stats["parse_failed"] += 1
            raise
        finally:
            self.parse_seconds += time.monotonic() - t0
        self.stats["parsed"] += 1
        self._memo[slot.filename] = rep
        if len(self._memo) > self._memo_size:
            self._memo.popitem(last=False)
        return rep


_default_store: ReportStore | None = None


def default_store() -> ReportStore:
    """The backfill's store. Under cache.READ_ONLY (a shared deployment) it
    keeps PDFs in memory instead of writing into the checkout; the app
    should still pass live_store().get, which also drops the retries."""
    global _default_store
    if _default_store is None:
        _default_store = ReportStore(persist=not READ_ONLY)
    return _default_store


def live_store(timeout: float = 8.0) -> ReportStore:
    """For the app: a short timeout, no retries (a dead host costs seconds,
    not the backfill client's retry loop), nothing written to disk."""
    return ReportStore(fetcher=functools.partial(fetch_report, timeout=timeout, retries=0),
                       persist=False)


# ── Report selection ─────────────────────────────────────────────────────────
class Selection(NamedTuple):
    slot: Slot
    report: ParsedReport
    n_probed: int


Getter = Callable[[Slot], "ParsedReport | None"]

TIP_LEAD = dt.timedelta(minutes=5)
MAX_BACK = dt.timedelta(hours=30)


def report_before(cutoff: dt.datetime, *, getter: Getter | None = None,
                  accept: Callable[[ParsedReport], bool] | None = None,
                  max_back: dt.timedelta = MAX_BACK) -> Selection | None:
    """The latest report whose TITLE time is strictly before `cutoff` (aware,
    any zone, or ISO), walking candidate slots newest-first from the last one
    named before the cutoff and at most `max_back` behind it. A report titled
    at or after the cutoff (the hourly-era '07PM' file holds the 7:30 report)
    is skipped and the walk continues; so is one `accept` vetoes. None means
    nothing within `max_back`: unknown, never 'everyone available'.

    This is THE selection rule; select_reports_for_date and any horizon
    snapshot should call it rather than re-implement it."""
    getter = getter or default_store().get
    # Arithmetic in UTC: Python subtracts two datetimes sharing a tzinfo as
    # naive wall clocks, which is an hour off across a DST change.
    cutoff_u = to_et(cutoff).astimezone(UTC)
    n = 0
    for slot in slots_descending(cutoff_u.astimezone(ET), inclusive=False):
        if slot.nominal_et.astimezone(UTC) < cutoff_u - max_back:
            return None
        n += 1
        rep = getter(slot)
        if (rep is not None and rep.meta["report_ts_et"].astimezone(UTC) < cutoff_u
                and (accept is None or accept(rep))):
            return Selection(slot, rep, n)
    return None


def select_reports_for_date(game_date: dt.date | str, tips: Iterable, *,
                            getter: Getter | None = None, lead: dt.timedelta = TIP_LEAD,
                            max_back: dt.timedelta = MAX_BACK,
                            accept: Callable[[ParsedReport], bool] | None = None,
                            ) -> dict[dt.datetime, Selection | None]:
    """For each distinct tip time (aware datetimes or ISO strings),
    report_before(tip - lead): the latest report whose TITLE time is strictly
    before tip - `lead`. The walk may cross into the previous day (a 1:00 PM
    tip in the 3-per-day era has nothing earlier that day; the 5 PM report
    lists next-day games).

    `accept` (optional) vetoes a report and continues the walk. The backfill
    passes a REPORT-level test (lists_nba_game: any real NBA game), so a QA
    file published between two real reports cannot become a game's report.
    A game-level test ('lists THIS game') would walk the whole `max_back`
    window for a game no report lists (BOS@NYK 2023-10-25 is in none of
    five) and hand it a stale report instead of 'not in the report'."""
    if lead < dt.timedelta(0):
        raise ValueError(f"negative lead {lead}: the report would be after tip")
    getter = getter or default_store().get
    if isinstance(game_date, str):
        game_date = dt.date.fromisoformat(game_date)
    out: dict[dt.datetime, Selection | None] = {}
    for tip in sorted({to_et(t) for t in tips}):
        if tip.date() != game_date:
            raise ValueError(f"tip {tip} is not on game date {game_date} (ET)")
        out[tip] = report_before(tip.astimezone(UTC) - lead, getter=getter, accept=accept,
                                 max_back=max_back)
    return out


def latest_report(now: dt.datetime | None = None, *, getter: Getter | None = None,
                  max_slots: int = 8, accept: Callable[[ParsedReport], bool] | None = None,
                  ) -> Selection | None:
    """Serving: the newest report TITLED at or before `now` (ET aware or ISO;
    default the current time), walking back at most `max_slots` distinct slot
    TIMES (near an era boundary both filename formats are tried per time, so
    the cap counts times, not files). None = nothing found = unknown; the
    caller must not overwrite the last good status table with it.

    A report titled after `now` is skipped: in a replay (now in the past) the
    hourly '07PM' file holds the 7:30 report, and returning it at 7:10 leaks
    twenty minutes of news. `accept` vetoes a report and continues the walk;
    pass lists_nba_game with the slate, because QA files are published to
    the same bucket and one in the newest slot would otherwise end the walk."""
    getter = getter or default_store().get
    now = to_et(now) if now is not None else now_et()
    now_u = now.astimezone(UTC)
    times_seen: set[dt.datetime] = set()
    n = 0
    for slot in slots_descending(now, inclusive=True):
        if slot.nominal_et not in times_seen:
            if len(times_seen) >= max_slots:
                break
            times_seen.add(slot.nominal_et)
        n += 1
        rep = getter(slot)
        if (rep is not None and rep.meta["report_ts_et"].astimezone(UTC) <= now_u
                and (accept is None or accept(rep))):
            return Selection(slot, rep, n)
    return None


# ── Name resolution ──────────────────────────────────────────────────────────
def _resolve_last_first(idx, last_first: str, team: str | None) -> Match:
    """names.NameIndex.resolve_last_first, except that an ambiguous order is
    reported as 'ambiguous' (proposed for the foundation; kept private here
    so the two cannot drift silently: same orders, same index calls)."""
    if "," not in last_first:
        return idx.get(last_first, team=team)
    last, first = [p.strip() for p in last_first.split(",", 1)]
    how = "none"
    for cand in (f"{first} {last}", f"{last} {first}"):
        m = idx.get(cand, team=team)
        if m.value is not None:
            return m
        if m.how == "ambiguous":
            how = "ambiguous"
    return Match(None, how)


def resolve(rows: pd.DataFrame, index_by_team) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach person_id to report rows. `index_by_team` is a names.NameIndex
    built with team=<tricode> on add(), or a {tricode: NameIndex} mapping
    whose indexes are ALSO built with team= (names.NameIndex drops its prefix
    fallback when asked for a team its entries were not tagged with).
    Each 'Last, First' is resolved scoped to the row's team, in both name
    orders (the report prints 'Yang, Hansen' for Yang Hansen).

    match_how is 'ambiguous' when either order hit two players, 'none' only
    when neither order found anyone: names.resolve_last_first reports both as
    'none', which hides the cases an alias or a roster fix would settle.

    Returns (resolved, unresolved): resolved = the matched rows plus person_id
    and match_how; unresolved = one line per (team, player_raw) that did not
    resolve, with n_rows. len(resolved) + unresolved.n_rows.sum() == len(rows),
    always: nothing is dropped silently."""
    ids, hows = [], []
    for name, team in zip(rows["player_raw"], rows["team"]):
        idx = index_by_team.get(team) if isinstance(index_by_team, Mapping) else index_by_team
        m = _resolve_last_first(idx, name, team) if idx is not None else Match(None, "none")
        ids.append(m.value)
        hows.append(m.how)
    out = rows.copy()
    out["person_id"] = ids
    out["match_how"] = hows
    hit = out["person_id"].notna()
    resolved = out[hit].copy()
    try:
        resolved["person_id"] = resolved["person_id"].astype("int64")
    except (TypeError, ValueError):
        pass  # non-integer ids (tests, custom indexes) stay as given
    miss = out[~hit]
    unresolved = (miss.groupby(["team", "player_raw", "match_how"], dropna=False)
                  .size().reset_index(name="n_rows")
                  .sort_values(["n_rows", "team", "player_raw"], ascending=[False, True, True])
                  .reset_index(drop=True))
    if len(resolved) + int(unresolved["n_rows"].sum()) != len(rows):
        raise AssertionError("resolve lost rows")
    return resolved, unresolved
