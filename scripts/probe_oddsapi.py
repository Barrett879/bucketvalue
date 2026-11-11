"""The first-key-day probe for The Odds API (design section 10).

LOCAL ONLY. Barrett runs it by hand from his development tree, at about
9:00 AM ET on a game day. It never runs in the bot or in GitHub Actions
(tests/test_feed_guards.py), and it refuses to run in Actions. The key sits in
a private file outside every repo and every iCloud folder (design 4.3):

    BUCKETVALUE_ODDS_API_KEY_FILE=~/.config/bucketvalue/odds_api_key .venv/bin/python scripts/probe_oddsapi.py

    --balance          one free /events call: remaining, used and allowance
    --evening          free /events calls with and without status=pre, while
                       games are on (INVALID_STATUS: the filter does not exist)
    --max-credits N    the most the full run may spend (default 16)
    --key-file PATH    the key file, instead of BUCKETVALUE_ODDS_API_KEY_FILE

It settles the facts only a real key can (report_feed.md part 4: the numbers
in brackets below) with the code the site runs: http.vendor_get for every
call, live._read_key for the key, oddsapi's request builders and parsers, and
bucketlib.names against the committed state/current/roster_v1.parquet. Paste
its output back: no line of it holds the key.

The steps, each with its worst case in credits per call. Game 1 and game 2
are the two earliest games step 3 read lines for (the earliest games when
none had any), since a game no book has posted shows the later steps nothing.
     1  /events, free: tip order [12a], quota headers [5a], the balance    0
     2  /participants: the 30 teams through teams.canonical [9a]           1
     3  event odds, player_points, the five books, no regions, up to 7
        untipped games, earliest first [1, 2, 3a, 6a, 10a, 13, 9b];
        a 422 MISSING_REGION is a FAIL (m5)                                1 a game
     4  game 1, player_points_alternate, prizepicks and underdog [6b, 10b] 1
     5  game 1, points, prizepicks and underdog, includeMultipliers [7]    1
     6  game 1, points, the five books plus regions=us [3b]                2
     7  game 1, bookmakers=draftkings,notabook [4]                         1
     8  game 1, markets=player_pointz [5b]                                 0
     9  an event id of 32 zeros [5c]                                       0
    10  /events with a made-up key of 40 zeros, free [5d]                  0
    11  game 2, points, commenceTimeFrom a year ahead [14]                 1
    12  game 1, points again, 2+ minutes after step 3 [8]                  1
    13  a game whose step-3 answer lacked PrizePicks, prizepicks alone     1
That is 16 with 7 games. Steps 8 and 9 count 0, because only returned
markets bill (docs 1602, 1639) and an unknown market or event returns none
(UNVERIFIED); every paid call still needs 1 credit of room, so the cap holds
even if they do bill. Before each paid call its room is added to what has
been spent, and the call is skipped if the sum would pass --max-credits. A
call's room is its worst case above times the most a game's odds call has
billed per credit of its own worst case so far (1 until one bills more, the
case step 3 exists to catch, fact 3a). So once a call shows the table is
low, every later paid call keeps room for that bill: only the one call that
first bills more than its room can pass the cap, by what it billed over its
room. After each call, x-requests-last is what counts (the call's room when
that header is missing or no answer came back). The free /events calls
(steps 1 and 10) always run: they spend nothing, past the cap or not.

PASS and FAIL are for what the site relies on: the quota headers, the team
names, a bookmakers-only call, 1 credit for one market, lines that parse, a
bad key's 401, the free PrizePicks-only poll. Every other fact is an INFO
line. Exit 0 when nothing FAILed, 1 on a FAIL, 2 when it refuses to run:
  - no --key-file and no BUCKETVALUE_ODDS_API_KEY_FILE (a key in the shell's
    BUCKETVALUE_ODDS_API_KEY is never read here);
  - a key file under ~/Desktop, ~/Documents or ~/Library/Mobile Documents (the
    iCloud folders), or inside a git checkout (a .git in any parent folder);
  - a key file that group or others can read, that is not a plain file, or
    that holds no key;
  - running in GitHub Actions.

It never prints the key, a URL or an exception's text: its lines hold fixed
words, numbers, our own codes (oddsapi.error_code) and vendor names cleaned
through http.scrub, and every line is cleaned again before it is printed. It
writes no file (bytecode included), never calls the event-markets endpoint
(1 credit every time, docs 1724), and buys no game within 5 minutes of tip.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import stat
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import pwd                          # the account's own home folder (POSIX; the probe runs on a Mac)
except ImportError:  # pragma: no cover - not a POSIX system: $HOME alone
    pwd = None

if __name__ == "__main__":
    sys.dont_write_bytecode = True      # a probe run writes no file at all, bytecode included

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bucketlib import http as H  # noqa: E402  (first: on macOS it loads CoreFoundation)
from bucketlib import live  # noqa: E402
from bucketlib import names as NM  # noqa: E402
from bucketlib import teams  # noqa: E402
from bucketlib.sources import oddsapi as O  # noqa: E402
from bucketlib.util import ET, UTC, now_utc, today_et  # noqa: E402

import pandas as pd  # noqa: E402

MAX_CREDITS = 16
MAX_GAMES = 7                          # step 3: up to 7 untipped games, earliest first
TIP_GUARD = dt.timedelta(minutes=5)    # no game is bought this close to its tip (the site's default)
STEP12_GAP_S = 120.0                   # step 12 runs 2+ minutes after step 3's first call
TIMEOUT = live.TIMEOUT                 # the site's fixed 4 s a call (design 2.4: ODDS_TIMEOUT = TIMEOUT)
POINTS = ("player_points",)
DFS = ("prizepicks", "underdog")
ALTERNATE = "player_points_alternate"
QUOTA = ("x-requests-remaining", "x-requests-used", "x-requests-last")
ROSTER = REPO / "state" / "current" / "roster_v1.parquet"
SCHEDULE = REPO / "state" / "current" / "schedule_v1.parquet"
ICLOUD = ("Desktop", "Documents", "Library/Mobile Documents")
FAKE_EVENT_ID = "0" * 32               # step 9
FAKE_KEY = "0" * 40                    # step 10: made up; the real key is never sent there
_SAMPLE = 8                            # names shown on one line, at most
# The only paths a call may take: the events list, the participants list and
# one game's odds. Anything else (the event-markets endpoint above all) is
# refused before it is sent.
_ALLOWED_PATH = re.compile(r"/v4/sports/basketball_nba/(?:events|participants|events/[A-Za-z0-9]{32}/odds)")
_ODDS_PATH = re.compile(r"/v4/sports/basketball_nba/events/[A-Za-z0-9]{32}/odds")   # one game's odds


# -- The steps and the budget -------------------------------------------------
@dataclass(frozen=True)
class Step:
    n: int
    what: str
    worst: int          # credits counted in the plan for one call (design section 10)
    paid: bool          # False only for the free /events calls


STEPS = (
    Step(1, "events", 0, False),
    Step(2, "participants", 1, True),
    Step(3, "points, the five books, no regions", 1, True),     # once a game, up to MAX_GAMES
    Step(4, "PrizePicks and Underdog alternates", 1, True),
    Step(5, "includeMultipliers", 1, True),
    Step(6, "the five books plus regions=us", 2, True),
    Step(7, "an unknown book key", 1, True),
    Step(8, "an unknown market key", 0, True),
    Step(9, "an unknown event id", 0, True),
    Step(10, "a made-up key", 0, False),
    Step(11, "commenceTimeFrom a year ahead", 1, True),
    Step(12, "points again, 2+ minutes later", 1, True),
    Step(13, "PrizePicks alone", 1, True),
)
STEP = {s.n: s for s in STEPS}


def planned_credits(games: int = MAX_GAMES) -> int:
    """The full run's worst case: step 3 once a game, every other step once."""
    return sum(s.worst * (games if s.n == 3 else 1) for s in STEPS)


def room_needed(step: Step, rate: int = 1) -> int:
    """The credits a call needs left under the cap: its worst case, at least
    1 for a paid endpoint (where a 0 is UNVERIFIED), times `rate`, the most
    a game's odds call has billed per credit of its own worst case so far."""
    return max(step.worst, 1) * max(1, int(rate)) if step.paid else 0


class Budget:
    """--max-credits. `spent` is what x-requests-last said each call cost, or
    the call's room when it did not say (a call refused before it was sent
    costs nothing). `rate` starts at 1 and rises when a game's odds call
    bills more than its worst case (learn), so later calls keep room for
    what the vendor really charges."""

    def __init__(self, cap: int):
        self.cap, self.spent, self.rate = int(cap), 0, 1

    def allows(self, need: int) -> bool:
        return self.spent + need <= self.cap

    def count(self, ans: "Answer", need: int) -> int:
        cost = ans.cost if ans.cost is not None else (0 if ans.kind == "refused" else need)
        self.spent += cost
        return cost

    def learn(self, step: Step, ans: "Answer") -> None:
        """A game's odds call billed `ans.cost`: the rate is the most billed
        per credit of a call's worst case, rounded up, and never falls."""
        if ans.cost is not None and ans.cost > 0:
            self.rate = max(self.rate, -(-int(ans.cost) // max(step.worst, 1)))


# -- One call -----------------------------------------------------------------
@dataclass
class Answer:
    """What one call left: our own codes and the vendor's numbers, never its
    text, a URL or the key."""
    status: int = 0           # 0: no complete answer came back
    code: str = ""            # oddsapi.error_code (our string); "" on a 200
    kind: str = ""            # oddsapi.failure_kind, a transport kind, or "refused"
    headers: tuple = ()       # the quota header NAMES that came back
    usage: object = None      # oddsapi.Usage
    body: object = None       # the parsed JSON, or None

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def cost(self) -> int | None:
        return None if self.usage is None else self.usage.last


def ask(key: str, path: str, params: dict, *, secrets: tuple = (), get=None) -> Answer:
    """One vendor call, through http.vendor_get (looked up now, so a test's
    stub is the one that runs) or `get`. It never raises, and it keeps none
    of the key, the URL or any exception's text."""
    if not isinstance(path, str) or not _ALLOWED_PATH.fullmatch(path):
        return Answer(kind="refused")
    fn = H.vendor_get if get is None else get
    query = dict(params)
    query["apiKey"] = key
    keep = tuple(s for s in (key, *secrets) if isinstance(s, str) and s)
    kind = None
    try:
        r = fn(O.HOST, path, query, timeout=TIMEOUT, secrets=keep)
    except H.VendorTransportError as e:
        kind = e.kind                          # one of four fixed words
    except (ValueError, H.BadRequest):
        kind = "refused"                       # vendor_get refused it before sending anything
    except Exception as e:  # noqa: BLE001 - only the type is kept, never the text
        kind = "error " + type(e).__name__
    if kind is not None:
        return Answer(kind=kind)
    headers = r.headers if isinstance(r.headers, dict) else {}
    status = r.status if isinstance(r.status, int) else 0
    code = "" if status == 200 else O.error_code(status, r.snippet)
    return Answer(status=status, code=code, kind=O.failure_kind(status, code),
                  headers=tuple(h for h in QUOTA if h in headers),
                  usage=O.usage_from_headers(headers), body=r.body)


def describe(ans: Answer | None) -> str:
    """One answer in our own words: the status, our code, x-requests-last."""
    if ans is None:
        return "not made"
    if not ans.status:
        return "no answer (refused before sending)" if ans.kind == "refused" else f"no answer ({ans.kind})"
    text = f"HTTP {ans.status}" + (f" {ans.code}" if ans.code else "")
    return text + (f"; x-requests-last {ans.cost}" if ans.cost is not None else "; no x-requests-last")


# -- Output ---------------------------------------------------------------------
_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://\S*")
_KEY_PAIR = re.compile(r"(?i)api[_-]?key\s*[=:]\s*\S+")
_LONG = re.compile(r"[A-Za-z0-9]{32,}")


def safe_line(text, secrets: tuple = ()) -> str:
    """The last cleaning before a line is printed: each secret, any URL, any
    apiKey pair and any run of 32 or more letters and digits (a key, an
    event id) go, and so does every character that is not printable."""
    s = str(text)
    for sec in secrets:
        if isinstance(sec, str) and len(sec) >= 8:
            s = s.replace(sec, "[redacted]")
    s = _URL.sub("[url]", s)
    s = _KEY_PAIR.sub("[redacted]", s)
    s = _LONG.sub("[redacted]", s)
    return "".join(ch for ch in s if ch.isprintable())


def vendor_text(value, secrets: tuple = (), n: int = 40) -> str:
    """A vendor string (a player or team name) fit for a line: http.scrub's
    cleaning, printable characters only, then cut."""
    if not isinstance(value, str):
        return "(not text)"
    s = "".join(ch for ch in H.scrub(value, secrets) if ch.isprintable())
    return s[:n] or "(blank)"


class Report:
    """PASS, FAIL and INFO lines, each made safe as it is written."""

    def __init__(self, secrets: tuple = (), write=print):
        self.secrets, self.write = tuple(secrets), write
        self.counts: Counter = Counter()

    def __call__(self, level: str, text: str, where: str | None = None) -> None:
        if level not in ("PASS", "FAIL", "INFO"):
            raise ValueError("a line is PASS, FAIL or INFO")
        self.counts[level] += 1
        self.write(safe_line(f"{level} {where}: {text}" if where else f"{level} {text}", self.secrets))

    @property
    def failed(self) -> bool:
        return self.counts["FAIL"] > 0


# -- The key file -----------------------------------------------------------------
def _under(path: Path, folder: Path) -> bool:
    """Is `path` inside `folder`? Case-blind, as macOS folders are."""
    p = [part.casefold() for part in path.parts]
    f = [part.casefold() for part in folder.parts]
    return len(p) > len(f) and p[:len(f)] == f


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path


def _account_home() -> str | None:
    """The account's own home folder, from the password database, so a run
    with $HOME pointed elsewhere still knows where the iCloud folders are."""
    if pwd is None:
        return None
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, OSError):
        return None


def _homes() -> list[Path]:
    """$HOME and the account's home, each as spelled and resolved."""
    out: list[Path] = []
    for h in (os.path.expanduser("~"), _account_home()):
        if isinstance(h, str) and h.strip():
            p = Path(os.path.abspath(h))
            out += [q for q in (p, _resolved(p)) if q not in out]
    return out


def _identity(path) -> tuple[int, int] | None:
    """A folder's (device, inode): the same for every spelling of it."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _in_icloud(places, homes) -> bool:
    """Is any of `places` inside an iCloud folder of any of `homes`? By
    spelling (case-blind), and by identity: a folder reached under another
    spelling, such as macOS's /System/Volumes/Data firmlink, which
    Path.resolve() keeps as written, is the same folder when its device and
    inode are."""
    if any(_under(p, h / folder) for p in places for h in homes for folder in ICLOUD):
        return True
    ids = {i for h in homes for folder in ICLOUD if (i := _identity(h / folder)) is not None}
    return bool(ids) and any(_identity(d) in ids for p in places for d in p.parents)


def key_file_refusal(path_text) -> str | None:
    """Why the probe will not read this key file, or None when it will. A
    fixed phrase, never the path or anything in the file. `~` is expanded,
    and both the path as given and the file a link points at are checked;
    the iCloud folders are compared by identity as well as by spelling."""
    text = path_text.strip() if isinstance(path_text, str) else ""
    if not text:
        return f"no key file: pass --key-file or set {live.KEY_FILE_ENV}"
    given = Path(os.path.abspath(os.path.expanduser(text)))
    places = (given, _resolved(given))
    if _in_icloud(places, _homes()):
        return ("the key file is in an iCloud folder (~/Desktop, ~/Documents or "
                "~/Library/Mobile Documents); keep it in ~/.config/bucketvalue (design 4.3)")
    if any(os.path.exists(d / ".git") for p in places for d in p.parents):
        return "the key file is inside a git checkout; keep it outside every repo (design 4.3)"
    try:
        st = os.stat(given)
    except OSError:
        return "the key file cannot be found or read"
    if not stat.S_ISREG(st.st_mode):
        return "the key file is not a plain file"
    if st.st_mode & (stat.S_IRGRP | stat.S_IROTH):
        return "group or others can read the key file; chmod 600 it (design 4.3)"
    return None


def read_key_file(path_text) -> str | None:
    """The key, from the key file only, through the site's own reader: the
    mapping names the file and nothing else, so a key exported in the shell
    (BUCKETVALUE_ODDS_API_KEY, which live._read_key would put first) is never
    the one used, and the file's checks always apply to the key sent."""
    return live._read_key({live.KEY_FILE_ENV: path_text if isinstance(path_text, str) else ""})


# -- Reading answers (pure) -----------------------------------------------------------
def _utc(value) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        t = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t.astimezone(UTC) if t.tzinfo is not None else None


def _bookmakers(body) -> list[dict]:
    bms = body.get("bookmakers") if isinstance(body, dict) else None
    return [b for b in bms if isinstance(b, dict)] if isinstance(bms, list) else []


def _outcomes(bm: dict, market: str) -> list[dict]:
    out = []
    for mk in bm.get("markets") if isinstance(bm.get("markets"), list) else ():
        if isinstance(mk, dict) and mk.get("key") == market and isinstance(mk.get("outcomes"), list):
            out += [oc for oc in mk["outcomes"] if isinstance(oc, dict)]
    return out


def number_text(value) -> str:
    """A price or a multiplier as a line shows it."""
    if value is None:
        return "null"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "(other)"
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def counts_text(c: Counter) -> str:
    return ", ".join(f"{k} x {n}" for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def event_order(body) -> tuple[int, bool]:
    """(events listed, whether their commence_time is in order) [12a]."""
    items = [ev for ev in body if isinstance(ev, dict)] if isinstance(body, list) else []
    times = [t for t in (_utc(ev.get("commence_time")) for ev in items) if t is not None]
    return len(items), times == sorted(times)


def started(body, now: dt.datetime) -> tuple[int, int]:
    """(events listed, how many have already started) [12b]."""
    items = [ev for ev in body if isinstance(ev, dict)] if isinstance(body, list) else []
    times = [_utc(ev.get("commence_time")) for ev in items]
    return len(items), sum(1 for t in times if t is not None and t <= now)


def map_participants(body) -> tuple[set, list]:
    """(the tricodes the names map to, the names teams.canonical does not know) [9a]."""
    known, unknown = set(), []
    for item in body if isinstance(body, list) else ():
        name = item.get("full_name") if isinstance(item, dict) else None
        code = teams.canonical(name) if isinstance(name, str) else None
        if code:
            known.add(code)
        else:
            unknown.append(name)
    return known, unknown


def book_rows(frame: pd.DataFrame) -> dict[str, int]:
    """Lines per book for one market: one line a player [1]."""
    got = frame["book"].value_counts().to_dict() if len(frame) else {}
    return {b: int(got.get(b, 0)) for b in O.BOOKS}


def dfs_prices(body, market: str = "player_points") -> dict[str, Counter]:
    """The raw prices on each DFS book's outcomes (the parser blanks them) [6a]."""
    out: dict[str, Counter] = {}
    for bm in _bookmakers(body):
        if bm.get("key") in O.US_DFS_BOOKS:
            c = out.setdefault(bm["key"], Counter())
            for oc in _outcomes(bm, market):
                c[number_text(oc.get("price"))] += 1
    return out


def side_counts(frame: pd.DataFrame) -> dict[str, Counter]:
    """Per book: lines with both sides, Over only and Under only [10a]."""
    words = {"both": "both sides", "more": "Over only", "less": "Under only"}
    out: dict[str, Counter] = {}
    for book, direction in zip(frame["book"], frame["direction"]):
        out.setdefault(book, Counter())[words.get(direction, str(direction))] += 1
    return out


def bookmaker_stamps(body) -> tuple[int, int]:
    """(entries with a bookmaker-level last_update, entries) [13]."""
    bms = _bookmakers(body)
    return sum(1 for b in bms if "last_update" in b), len(bms)


def roster_index(roster: pd.DataFrame, aliases: dict | None = None) -> tuple[NM.NameIndex, dict]:
    """The site's name matcher over the committed roster, and each player's team."""
    idx, team_of = NM.NameIndex(aliases), {}
    for name, pid, team in zip(roster["name"], roster["person_id"], roster["team"]):
        if isinstance(name, str) and name.strip():
            idx.add(name, pid, team=team)
            team_of[pid] = team
    return idx, team_of


def name_matches(frame: pd.DataFrame, idx: NM.NameIndex, team_of: dict, home: str,
                 away: str) -> dict[str, tuple[int, int, list]]:
    """Per book: (names matched to a player of this game's two teams, names
    matched to a player on another team, names not matched) [9b]."""
    out = {}
    for book in [b for b in O.BOOKS if b in set(frame["book"])]:
        matched, elsewhere, missing = 0, 0, []
        for name in frame.loc[frame["book"] == book, "name"]:
            m = idx.get(name, team=home)
            if m.value is None or team_of.get(m.value) not in (home, away):
                other = idx.get(name, team=away)
                if other.value is not None:
                    m = other
            if m.value is None:
                missing.append(name)
            elif team_of.get(m.value) in (home, away):
                matched += 1
            else:
                elsewhere += 1
        out[book] = (matched, elsewhere, missing)
    return out


def alt_summary(body, market: str = ALTERNATE) -> dict[str, dict]:
    """Per book: alternate outcomes, how many at +100, players, and rungs (the
    points a player is offered) at most and at the median [6b, 10b]."""
    out = {}
    for bm in _bookmakers(body):
        ocs = _outcomes(bm, market)
        if not ocs:
            continue
        rungs: dict[str, set] = {}
        for oc in ocs:
            point = oc.get("point")
            if (isinstance(oc.get("description"), str) and isinstance(point, (int, float))
                    and not isinstance(point, bool)):
                rungs.setdefault(oc["description"].casefold(), set()).add(oc["point"])
        sizes = sorted(len(v) for v in rungs.values())
        out[bm.get("key")] = {
            "outcomes": len(ocs),
            "plus100": sum(1 for oc in ocs if number_text(oc.get("price")) == "100"),
            "players": len(rungs), "most": sizes[-1] if sizes else 0,
            "median": number_text(statistics.median(sizes)) if sizes else "0"}
    return out


def multiplier_summary(body, market: str = "player_points") -> dict[str, Counter]:
    """Per book: what `multiplier` holds on its outcomes ("absent" when the
    field is not there) [7]."""
    out: dict[str, Counter] = {}
    for bm in _bookmakers(body):
        c = out.setdefault(bm.get("key"), Counter())
        for oc in _outcomes(bm, market):
            c["absent" if "multiplier" not in oc else number_text(oc["multiplier"])] += 1
    return out


def stamp_moves(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, tuple[int, int, int, int]]:
    """Per book, between two pulls of one game: (lines in both, lines that did
    not move, of those how many have a newer market last_update, lines that
    moved) [8]."""
    def keyed(df):
        return {(b, str(n).casefold(), m): (ln, lu) for b, n, m, ln, lu in
                zip(df["book"], df["name"], df["market"], df["line"], df["last_update"])}
    a, b = keyed(before), keyed(after)
    out: dict[str, list] = {}
    for k in sorted(set(a) & set(b)):
        (l0, u0), (l1, u1) = a[k], b[k]
        row = out.setdefault(k[0], [0, 0, 0, 0])
        row[0] += 1
        if l0 == l1:
            row[1] += 1
            if pd.notna(u0) and pd.notna(u1) and u1 > u0:
                row[2] += 1
        else:
            row[3] += 1
    return {k: tuple(v) for k, v in out.items()}


def field_names(body) -> str:
    """An error body's field NAMES, never their values [5b]."""
    if not isinstance(body, dict):
        return "(no JSON object)"
    names = sorted(re.sub(r"[^A-Za-z0-9_]", "", str(k))[:30] for k in body)
    return ", ".join(n for n in names[:10] if n) or "(none)"


def headers_text(ans: Answer) -> str:
    return ", ".join(ans.headers) or "none"


def books_answered(body) -> list[str]:
    return [vendor_text(b.get("key"), n=24) for b in _bookmakers(body)]


# -- Verdicts (pure) -----------------------------------------------------------
REGIONS_FAIL = "the vendor requires regions; see step 6 before setting oddsapi.SEND_REGIONS"


def odds_verdict(ans: Answer, frame: pd.DataFrame) -> tuple[str, str]:
    """Step 3, one game: what the site relies on in a one-market answer."""
    if ans.status == 422 and ans.code == "MISSING_REGION":
        return "FAIL", REGIONS_FAIL
    if not ans.ok:
        return "FAIL", describe(ans)
    if frame.attrs.get("unknown_team") or frame.attrs.get("bad_payload"):
        return "FAIL", "the answer did not parse (an unknown team or a bad shape)"
    if ans.cost is None:
        return "FAIL", "no x-requests-last on a 200 [5a]"
    if ans.cost > 1:
        return "FAIL", f"one market billed {ans.cost} credits [3a]"
    if frame.empty:
        entries = frame.attrs.get("bookmaker_entries", 0)
        if entries:
            return "FAIL", (f"{entries} books answered, but the parser kept no line: a shape it "
                            f"does not read (cost {ans.cost})")
        if ans.cost:
            return "FAIL", f"an answer with no book in it billed {ans.cost}; the docs say it is free [3a]"
        return "INFO", "no book has posted points yet; the empty answer was free [3a]"
    if ans.cost == 0:
        return "INFO", f"{len(frame)} lines, and x-requests-last says they cost 0 [3a]"
    return "PASS", f"{len(frame)} lines for 1 credit [3a]"


def regions_verdict(ans: Answer) -> str:
    """Step 6: what sending regions next to bookmakers does to the bill [3b]."""
    if not ans.ok:
        return describe(ans) + " [3b]"
    if not _bookmakers(ans.body):
        return f"an empty answer (cost {ans.cost}), which cannot show the bill; run it again later [3b]"
    if ans.cost == 1:
        return "x-requests-last 1: bookmakers alone sets the bill, so SEND_REGIONS is safe if step 3 needs it [3b]"
    if ans.cost == 2:
        return ("x-requests-last 2: regions bills a second region on top of bookmakers; keep "
                "SEND_REGIONS False unless step 3 FAILed on MISSING_REGION [3b]")
    return f"x-requests-last {ans.cost} [3b]"


def unknown_book_verdict(ans: Answer) -> str:
    if ans.status == 422 and ans.code == "INVALID_BOOKMAKERS":
        return "an unknown book key is refused: 422 INVALID_BOOKMAKERS [4]"
    if ans.ok:
        got = books_answered(ans.body)
        return (f"an unknown book key is dropped without a word; books that answered: "
                f"{', '.join(got) or 'none'}; cost {ans.cost} [4]")
    return describe(ans) + " [4]"


def bad_key_verdict(ans: Answer) -> tuple[str, str]:
    """Step 10: the site stops every call on a bad key's 401 (design 3.8),
    and reads OUT_OF_USAGE_CREDITS as a different 401."""
    if ans.status == 401 and ans.kind == "bad_key":
        return "PASS", (f"a bad key gets 401 {ans.code}, which the site reads as a bad key "
                        f"(design 3.8); usage headers: {headers_text(ans)} [5d]")
    if ans.status == 401:
        return "FAIL", f"a bad key gets 401 {ans.code}, which the site would read as {ans.kind} [5d]"
    return "FAIL", f"a bad key got {describe(ans)}, not a 401 [5d]"


def commence_verdict(ans: Answer) -> str:
    if not ans.ok:
        return f"{describe(ans)}: commenceTimeFrom is not accepted on event odds [14]"
    if not _bookmakers(ans.body):
        return f"commenceTimeFrom is honored on event odds: an empty answer, cost {ans.cost} [14]"
    return f"commenceTimeFrom is ignored on event odds: lines came back, cost {ans.cost} [14]"


def topup_verdict(ans: Answer, frame: pd.DataFrame) -> tuple[str, str]:
    """Step 13: the free PrizePicks-only polls of design 3.4."""
    if not ans.ok:
        return "INFO", describe(ans)
    if len(frame):
        return "INFO", f"PrizePicks has posted this game since step 3: {len(frame)} lines, cost {ans.cost}"
    if frame.attrs.get("bookmaker_entries"):
        return "INFO", f"PrizePicks answered with nothing the parser reads; cost {ans.cost}"
    if ans.cost is None:
        return "FAIL", "no x-requests-last on a 200 [5a]"
    if ans.cost == 0:
        return "PASS", "a PrizePicks-only poll is free while PrizePicks has nothing (design 3.4)"
    return "FAIL", (f"a PrizePicks-only answer with no line billed {ans.cost}: the top-up polls "
                    f"of design 3.4 are not free")


# -- The run --------------------------------------------------------------------
def fmt_et(t: dt.datetime) -> str:
    t = t.astimezone(ET)
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'} ET"


def label(g) -> str:
    return f"{g.away} at {g.home}"


def tip_far(g, now: dt.datetime) -> bool:
    """More than the tip guard before the game's tip: the only time it may be bought."""
    return g.commence_utc - now > TIP_GUARD


def untipped(events: pd.DataFrame, now: dt.datetime) -> list:
    """Today's events more than the tip guard before their tip, earliest first."""
    rows = [g for g in events.itertuples(index=False) if tip_far(g, now)]
    return sorted(rows, key=lambda g: g.commence_utc)


def first_two(with_lines: list, games: list) -> tuple:
    """Game 1 and game 2 of steps 4 to 12: the two earliest games step 3 read
    lines for, since a game nobody has posted shows nothing; else the
    earliest games. One game only: it is both."""
    pool = with_lines or games
    return pool[0], pool[1] if len(pool) > 1 else pool[0]


def odds_request(event_id: str, books: tuple, markets=POINTS, **extra) -> tuple[str, dict]:
    """(path, params) for one game's odds: the site's own request
    (oddsapi.event_odds_request), then the step's one change: a market list
    the site would refuse (a string here), or an extra parameter."""
    if isinstance(markets, str):
        path, params = O.event_odds_request(event_id, books, POINTS)
        params["markets"] = markets
    else:
        path, params = O.event_odds_request(event_id, books, markets)
    params.update(extra)
    return path, params


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - a missing file is said, never its text
        return None


def _parse(ans: Answer, books: tuple) -> pd.DataFrame:
    if not ans.ok:
        return O.empty_feed()
    return O.parse_event_odds(ans.body, books=books, markets=POINTS, fetched_at=now_utc())


class Probe:
    """One run: the key, the output, the budget and the clocks."""

    def __init__(self, key: str, out: Report, cap: int, *, get=None, now=None, today=None,
                 sleep=time.sleep, mono=time.monotonic, clock=now_utc):
        self.key, self.out, self.budget, self.get = key, out, Budget(cap), get
        self.now = now if now is not None else now_utc()
        self.today = today if today is not None else today_et()
        self.sleep, self.mono, self.clock = sleep, mono, clock
        self.remaining = None

    def can(self, n: int, where: str | None = None) -> bool:
        """A free step always may (it spends nothing, past the cap or not);
        a paid one when its room, at the rate seen so far, fits the cap."""
        if not STEP[n].paid:
            return True
        need = room_needed(STEP[n], self.budget.rate)
        if self.budget.allows(need):
            return True
        self.out("INFO", f"skipped: {need} more credit(s) would pass --max-credits "
                         f"{self.budget.cap} ({self.budget.spent} counted)", where or f"step {n}")
        return False

    def call(self, n: int, path: str, params: dict, *, key: str | None = None,
             where: str | None = None, game=None) -> Answer | None:
        """Step n's call, inside the budget and, for a game, outside its tip
        guard, read again now (step 12 runs minutes after the games were
        picked); says what it cost. A game's odds call that billed more than
        its worst case raises the room of every later paid call."""
        where = where or f"step {n}"
        if game is not None and not tip_far(game, self.clock()):
            self.out("INFO", f"skipped: {label(game)} tips within {TIP_GUARD.seconds // 60} minutes",
                     where)
            return None
        if not self.can(n, where):
            return None
        need = room_needed(STEP[n], self.budget.rate)
        ans = ask(self.key if key is None else key, path, params, secrets=(self.key,), get=self.get)
        cost = self.budget.count(ans, need)
        if isinstance(path, str) and _ODDS_PATH.fullmatch(path):
            self.budget.learn(STEP[n], ans)
        if ans.usage is not None and ans.usage.remaining is not None:
            self.remaining = ans.usage.remaining
        self.out("INFO", f"{STEP[n].what}: {describe(ans)}; counted {cost}", where)
        return ans


def step1(p: Probe) -> pd.DataFrame | None:
    where = "step 1"
    path, params = O.events_request()
    ans = p.call(1, path, params)
    if ans is None or not ans.ok:
        p.out("FAIL", f"the events list did not come back ({describe(ans)}); nothing else can run", where)
        return None
    missing = [h for h in QUOTA if h not in ans.headers]
    if missing:
        p.out("FAIL", f"quota headers missing on a 200: {', '.join(missing)} [5a]", where)
    else:
        p.out("PASS", "the quota headers came back on a 200 [5a]", where)
    u = ans.usage
    p.out("INFO", f"balance: remaining {u.remaining}, used {u.used}, allowance {u.allowance}", where)
    listed, in_order = event_order(ans.body)
    p.out("INFO", f"{listed} events listed, {'in' if in_order else 'NOT in'} commence_time order [12a]",
          where)
    events, stats = O.parse_events(ans.body, p.today)
    p.out("INFO", f"{stats['kept']} today ET; {stats['other_date']} on other dates, "
                  f"{stats['bad_event']} unreadable", where)
    if stats["unknown_team_events"]:
        names = ", ".join(vendor_text(n, p.out.secrets) for n in stats["unknown_team"])
        p.out("FAIL", f"{stats['unknown_team_events']} events name a team teams.canonical does not "
                      f"know: {names}", where)
    sched = _read_parquet(SCHEDULE)
    if sched is not None and len(events):
        games = sched[sched["game_date"].astype(str) == p.today.isoformat()]
        _, ms = O.match_events(events, games)
        p.out("INFO", f"matched to the committed schedule: {ms['matched']} of {len(games)} games; "
                      f"{ms['swapped']} with home and away swapped, {ms['event_no_game']} events with "
                      f"no game, {ms['game_no_event']} games with no event", where)
    return events


def step2(p: Probe) -> None:
    where = "step 2"
    ans = p.call(2, O.PARTICIPANTS_PATH, {})
    if ans is None:
        return
    if not ans.ok:
        p.out("FAIL", f"participants: {describe(ans)}", where)
        return
    known, unknown = map_participants(ans.body)
    missing = sorted(set(teams.CODES) - known)
    if missing:
        p.out("FAIL", f"{len(missing)} of the 30 teams have no name that maps: {', '.join(missing)} [9a]",
              where)
    else:
        p.out("PASS", "all 30 teams map through teams.canonical [9a]", where)
    if unknown:
        shown = ", ".join(vendor_text(n, p.out.secrets) for n in unknown[:_SAMPLE])
        p.out("INFO", f"{len(unknown)} participant names do not map (inactive ones may): {shown}", where)


def step3(p: Probe, games: list, idx, team_of: dict) -> dict:
    """Every game's points from the five books. What the later steps need is
    kept: each game's lines, cost and call time, the games step 3 read lines
    for (in tip order: steps 4 to 12 use the first two), and the games
    without PrizePicks (step 13), those where other books posted first."""
    where = "step 3"
    res = {"frames": {}, "costs": {}, "called_at": {}, "with_lines": [], "lacking_pp": []}
    posted, read, lacking = Counter(), 0, []
    for g in games:
        t0 = p.mono()
        ans = p.call(3, *odds_request(g.event_id, O.BOOKS), game=g)
        if ans is None:
            continue
        res["called_at"][g.event_id] = t0
        frame = _parse(ans, O.BOOKS)
        level, text = odds_verdict(ans, frame)
        p.out(level, f"{label(g)}, {fmt_et(g.commence_utc)}: {text}", where)
        if level == "FAIL" and text == REGIONS_FAIL:
            p.out("INFO", "the other games are not called: each would need regions too", where)
            break
        if not ans.ok or frame.attrs.get("unknown_team") or frame.attrs.get("bad_payload"):
            continue
        read += 1
        res["frames"][g.event_id], res["costs"][g.event_id] = frame, ans.cost
        rows = book_rows(frame)
        posted.update(b for b, n in rows.items() if n)
        if not rows["prizepicks"]:
            lacking.append((frame.empty, g))
        if frame.empty:
            continue
        res["with_lines"].append(g)
        p.out("INFO", f"{label(g)}: lines per book: " + ", ".join(f"{b} {n}" for b, n in rows.items())
              + " [1]", where)
        prices = dfs_prices(ans.body)
        if prices:
            p.out("INFO", f"{label(g)}: DFS prices: " + "; ".join(
                f"{b} {counts_text(c) or 'none'}" for b, c in prices.items()) + " [6a]", where)
        sides = side_counts(frame)
        if sides:
            p.out("INFO", f"{label(g)}: sides: " + "; ".join(
                f"{b} {counts_text(c)}" for b, c in sides.items()) + " [10a]", where)
        stamped, entries = bookmaker_stamps(ans.body)
        p.out("INFO", f"{label(g)}: a bookmaker-level last_update on {stamped} of {entries} books [13]",
              where)
        if idx is not None:
            for book, (ok, elsewhere, missing) in name_matches(frame, idx, team_of, g.home, g.away).items():
                text = (f"{label(g)}: names {book}: {ok} of {ok + elsewhere + len(missing)} matched "
                        f"in this game, {elsewhere} on another team")
                if missing:
                    text += "; not matched: " + ", ".join(vendor_text(x, p.out.secrets)
                                                          for x in missing[:_SAMPLE])
                p.out("INFO", text + " [9b]", where)
    if read:
        p.out("INFO", f"PrizePicks posted points in {posted['prizepicks']} of {read} games, Underdog "
                      f"in {posted['underdog']} of {read} [2]", where)
    res["lacking_pp"] = [g for _, g in sorted(lacking, key=lambda t: t[0])]   # stable: tip order kept
    return res


def step4(p: Probe, g1) -> None:
    ans = p.call(4, *odds_request(g1.event_id, DFS, markets=ALTERNATE), game=g1)
    if ans is not None and ans.ok:
        for book, s in alt_summary(ans.body).items():
            p.out("INFO", f"{vendor_text(book, n=24)}: {s['outcomes']} alternate outcomes, {s['plus100']} "
                          f"at +100 [6b]; rungs for {s['players']} player(s), up to {s['most']}, "
                          f"median {s['median']} [10b]", "step 4")


def step5(p: Probe, g1, res: dict) -> None:
    ans = p.call(5, *odds_request(g1.event_id, DFS, includeMultipliers="true"), game=g1)
    if ans is None:
        return
    if ans.status == 422 and ans.code == "INVALID_INCLUDE_MULTIPLIERS":
        p.out("INFO", "includeMultipliers is refused: 422 INVALID_INCLUDE_MULTIPLIERS [7]", "step 5")
    elif ans.ok:
        for book, c in multiplier_summary(ans.body).items():
            p.out("INFO", f"{vendor_text(book, n=24)} multiplier: {counts_text(c) or 'no outcome'} [7]",
                  "step 5")
        p.out("INFO", f"cost {ans.cost}; this game's step-3 call cost {res['costs'].get(g1.event_id)} [7]",
              "step 5")


def step6(p: Probe, g1) -> None:
    ans = p.call(6, *odds_request(g1.event_id, O.BOOKS, regions="us"), game=g1)
    if ans is not None:
        p.out("INFO", regions_verdict(ans), "step 6")


def step7(p: Probe, g1) -> None:
    ans = p.call(7, *odds_request(g1.event_id, ("draftkings",), bookmakers="draftkings,notabook"),
                 game=g1)
    if ans is not None:
        p.out("INFO", unknown_book_verdict(ans), "step 7")


def step8(p: Probe, g1) -> None:
    ans = p.call(8, *odds_request(g1.event_id, O.BOOKS, markets="player_pointz"), game=g1)
    if ans is not None:
        p.out("INFO", f"{describe(ans)}; error body fields: {field_names(ans.body)}; usage headers: "
                      f"{headers_text(ans)} [5b]", "step 8")


def step9(p: Probe) -> None:
    ans = p.call(9, *odds_request(FAKE_EVENT_ID, O.BOOKS))
    if ans is not None:
        p.out("INFO", f"{describe(ans)}; the site reads it as {ans.kind}; usage headers: "
                      f"{headers_text(ans)} [5c]", "step 9")


def step10(p: Probe) -> None:
    path, params = O.events_request()
    ans = p.call(10, path, params, key=FAKE_KEY)        # the real key is never sent here
    if ans is not None:
        level, text = bad_key_verdict(ans)
        p.out(level, text, "step 10")


def step11(p: Probe, g1, g2) -> None:
    ahead = (p.now + dt.timedelta(days=365)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ans = p.call(11, *odds_request(g2.event_id, O.BOOKS, commenceTimeFrom=ahead), game=g2)
    if ans is not None:
        p.out("INFO", ("game 1, as no second game had lines: " if g2 is g1 else "")
              + commence_verdict(ans), "step 11")


def step12(p: Probe, g1, res: dict) -> None:
    before = res["frames"].get(g1.event_id)
    if before is None or before.empty:
        p.out("INFO", "skipped: step 3 read no lines for game 1 to compare", "step 12")
        return
    if not p.can(12):
        return
    wait = STEP12_GAP_S - (p.mono() - res["called_at"][g1.event_id])
    if wait > 0:
        p.out("INFO", f"waiting {math.ceil(wait)} s, so that 2 minutes pass after step 3", "step 12")
        p.sleep(wait)
    ans = p.call(12, *odds_request(g1.event_id, O.BOOKS), game=g1)
    if ans is None or not ans.ok:
        return
    for book, (both, same, newer, moved) in stamp_moves(before, _parse(ans, O.BOOKS)).items():
        p.out("INFO", f"{book}: {both} lines in both pulls, {same} unchanged, a newer last_update on "
                      f"{newer} of those, {moved} moved [8]", "step 12")


def step13(p: Probe, res: dict) -> None:
    if not res["lacking_pp"]:
        p.out("INFO", "skipped: PrizePicks had points lines in every game step 3 read", "step 13")
        return
    g = res["lacking_pp"][0]
    ans = p.call(13, *odds_request(g.event_id, ("prizepicks",)), game=g)
    if ans is not None:
        level, text = topup_verdict(ans, _parse(ans, ("prizepicks",)))
        p.out(level, f"{label(g)}: {text}", "step 13")


def run_full(p: Probe) -> None:
    p.out("INFO", f"full run for {p.today.isoformat()} ET: at most {p.budget.cap} credits (the "
                  f"plan's worst case is {planned_credits()}); books {', '.join(O.BOOKS)}; "
                  f"SEND_REGIONS {O.SEND_REGIONS}")
    events = step1(p)
    if events is None:
        return
    step2(p)
    games = untipped(events, p.now)[:MAX_GAMES]
    if not games:
        p.out("INFO", "no untipped game today ET: steps 3 to 9 and 11 to 13 need one; run the probe "
                      "on a game day before the first tip", "step 3")
        step10(p)
        return
    roster = _read_parquet(ROSTER)
    if roster is None:
        p.out("INFO", "no committed roster: names are not matched", "step 3")
    idx, team_of = (None, {}) if roster is None else roster_index(roster)
    res = step3(p, games, idx, team_of)
    g1, g2 = first_two(res["with_lines"], games)
    p.out("INFO", f"game 1 is {label(g1)}" + ("" if g2 is g1 else f", game 2 is {label(g2)}")
          + (", the earliest with lines" if res["with_lines"] else
             "; no game had lines yet, so later steps may learn little"), "step 3")
    step4(p, g1)
    step5(p, g1, res)
    step6(p, g1)
    step7(p, g1)
    step8(p, g1)
    step9(p)
    step10(p)
    step11(p, g1, g2)
    step12(p, g1, res)
    step13(p, res)


def run_balance(p: Probe) -> None:
    path, params = O.events_request()
    ans = p.call(1, path, params, where="balance")
    if ans is None or not ans.ok:
        p.out("FAIL", f"the events list did not come back ({describe(ans)})", "balance")
        return
    u = ans.usage
    if u.remaining is None or u.used is None:
        p.out("FAIL", "the quota headers did not come back [5a]", "balance")
    p.out("INFO", f"remaining {u.remaining}, used {u.used}, allowance {u.allowance}", "balance")


def run_evening(p: Probe) -> None:
    path, params = O.events_request()
    plain = p.call(1, path, params, where="evening")
    if plain is None or not plain.ok:
        p.out("FAIL", f"the events list did not come back ({describe(plain)})", "evening")
        return
    listed, on = started(plain.body, p.now)
    p.out("INFO", f"without status=pre: {listed} events listed, {on} already started [12b]", "evening")
    if not on:
        p.out("INFO", "no game has started yet: run --evening while a game is on", "evening")
    pre = p.call(1, path, dict(params, status="pre"), where="evening")
    if pre is not None and pre.status == 422 and pre.code == "INVALID_STATUS":
        p.out("INFO", "status=pre is refused: 422 INVALID_STATUS, so the filter does not exist [12b]",
              "evening")
    elif pre is not None and pre.ok:
        listed2, on2 = started(pre.body, p.now)
        p.out("INFO", f"with status=pre: {listed2} events listed, {on2} already started [12b]", "evening")
    u = plain.usage
    p.out("INFO", f"remaining {u.remaining}, used {u.used}, allowance {u.allowance}", "evening")


# -- main -----------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="probe_oddsapi.py",
                                 description="The first-key-day probe for The Odds API (LOCAL ONLY).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--balance", action="store_true", help="one free /events call: the balance")
    mode.add_argument("--evening", action="store_true",
                      help="free /events calls with and without status=pre, while games are on")
    ap.add_argument("--max-credits", type=int, default=MAX_CREDITS,
                    help=f"the most the full run may spend (default {MAX_CREDITS})")
    ap.add_argument("--key-file", default=None, help=f"the key file (default: ${live.KEY_FILE_ENV})")
    return ap


def main(argv=None) -> int:
    """Exit 0 when nothing FAILed, 1 on a FAIL, 2 when the probe refuses to
    run. Every refusal comes before the key is read and before any call."""
    args = _parser().parse_args(argv)
    out = Report()
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        out("FAIL", "refused: the probe never runs in GitHub Actions (design 2.10)")
        return 2
    if args.max_credits < 0:
        out("FAIL", "refused: --max-credits cannot be below 0")
        return 2
    path_text = args.key_file if args.key_file is not None else os.environ.get(live.KEY_FILE_ENV, "")
    why = key_file_refusal(path_text)
    if why:
        out("FAIL", f"refused: {why}")
        return 2
    key = read_key_file(path_text)
    if not key:
        out("FAIL", "refused: the key file holds no key")
        return 2
    out = Report(secrets=(key,))
    if os.environ.get(live.KEY_ENV):
        out("INFO", f"{live.KEY_ENV} is set in this shell; the probe ignores it and reads the key file")
    p = Probe(key, out, args.max_credits)
    try:
        if args.balance:
            run_balance(p)
        elif args.evening:
            run_evening(p)
        else:
            run_full(p)
    except KeyboardInterrupt:
        out("INFO", "interrupted")
        return 1
    except Exception as e:  # noqa: BLE001 - only the type is shown, never the text
        out("FAIL", f"the probe stopped: {type(e).__name__}")
        return 1
    tail = f"; remaining {p.remaining}" if p.remaining is not None else ""
    out("INFO", f"done: {out.counts['PASS']} PASS, {out.counts['FAIL']} FAIL; counted "
                f"{p.budget.spent} of --max-credits {p.budget.cap}{tail}")
    return 1 if out.failed else 0


if __name__ == "__main__":
    sys.exit(main())
