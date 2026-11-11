"""Repo-wide guards for the licensed odds feed (The Odds API), which hold
whatever the feed code does (design 8.2, 2.10 and 5.2).

The repo is public, so every file a key could live in is git-ignored before
any such file exists, none is tracked, and no file a commit could carry holds
a key-shaped value (m14: tests build their fake key at run time).

The rest, file by file:
  - bucketlib/sources/oddsapi.py and scripts/probe_oddsapi.py import no
    network library. The regex style of tests/test_espn.py:1149-1152 and
    tests/test_statsnba.py:1020-1029, widened to `from urllib import x`,
    `from http import client` and `import os, socket` (the gap RF 1.12 found),
    with an AST pass beside it. The probe reaches the vendor only through
    bucketlib.http; oddsapi imports nothing of bucketlib's network side.
  - oddsapi does no I/O and reads no clock, banned by CALL SHAPE (`.today(`,
    `.now(`, ...), so parse_events' `today` PARAMETER passes (m12, m14; RC
    G14, T5).
  - bucketlib/oddsbudget.py, the governor (design step 8), is held to every
    rule oddsapi is from the moment the file exists: the lists below pick it
    up by themselves.
  - The vendor's host appears only in bucketlib/http.py, oddsapi and the
    probe, and nothing of the vendor is in the bot or in Actions (2.10).
  - No widget that offers rows for download (RC R3, G12), and
    showErrorDetails stays "type" (RC K6).

Every guard also runs on a laundered copy of what it guards and is seen to
fail there (tests/test_props.py's rule: a guard nobody has watched fail is not
known to work). No message here ever quotes a matched value: a key-shaped
match is reported by file and line only.
"""
import ast
import fnmatch
import io
import re
import subprocess
import tokenize
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SECRET_PATTERNS = (".streamlit/secrets.toml", ".env", ".env.*", "*.key", "*.secret")
SECRET_FILES = (".streamlit/secrets.toml", ".env", ".env.local", ".env.production",
                "odds_api.key", "bucketlib/odds.key", "the_odds_api.secret", "scripts/a.secret")


def _git(*args) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if r.returncode == 128:
        pytest.skip("not a git checkout")
    return r


def _is_secret_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path == ".streamlit/secrets.toml" or any(
        fnmatch.fnmatchcase(name, p) for p in (".env", ".env.*", "*.key", "*.secret"))


def test_the_secret_files_are_git_ignored():
    lines = [line.strip() for line in (REPO / ".gitignore").read_text().splitlines()]
    for pattern in SECRET_PATTERNS:
        assert pattern in lines, f".gitignore lost {pattern}"
    taken_back = [line for line in lines if line.startswith("!")
                  and any(s in line for s in ("secret", ".env", ".key"))]
    assert not taken_back, taken_back
    for path in SECRET_FILES:
        assert _is_secret_file(path), path
        r = _git("check-ignore", "-q", path)
        assert r.returncode == 0, f"{path} is not git-ignored"
    assert not _is_secret_file(".streamlit/config.toml")


def test_no_secret_file_is_tracked():
    tracked = _git("ls-files", "--cached").stdout.splitlines()
    assert ".streamlit/config.toml" in tracked        # the listing is real
    assert not [p for p in tracked if _is_secret_file(p)]


# -- The files each rule covers ---------------------------------------------
HTTP = "bucketlib/http.py"
ODDSAPI = "bucketlib/sources/oddsapi.py"
ODDSBUDGET = "bucketlib/oddsbudget.py"      # design step 8: guarded from the moment it exists
PROBE = "scripts/probe_oddsapi.py"


def _with_later(required: tuple, later: tuple) -> list[str]:
    """Every required file (a missing one fails its test) plus every later
    one that exists by now. Computed at collection, so a later file joins its
    rules on the first run after it lands, with no skip mark to remove."""
    return list(required) + [rel for rel in later if (REPO / rel).is_file()]


PURE = _with_later((ODDSAPI,), (ODDSBUDGET,))                 # no network, no I/O, no clock
NO_NETWORK = _with_later((ODDSAPI, PROBE), (ODDSBUDGET,))     # import no network library


def _source(rel: str) -> str:
    p = REPO / rel
    assert p.is_file(), f"{rel} is missing"
    return p.read_text(encoding="utf-8")


def _shippable() -> list[str]:
    """Every file a commit could carry: the tracked ones, and the untracked
    ones git does not ignore (so a new file is checked before its first
    commit). Ignored files (a local .env, a secrets.toml) are never read."""
    out = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard").stdout
    return sorted({p for p in out.split("\0") if p and (REPO / p).is_file()})


# -- Code, without its comments and strings ---------------------------------
_BLANKED = {tokenize.COMMENT, tokenize.STRING}
_BLANKED |= {getattr(tokenize, n) for n in ("FSTRING_MIDDLE", "TSTRING_MIDDLE") if hasattr(tokenize, n)}


def _code(src: str) -> str:
    """`src` with every comment and string literal blanked to spaces, line for
    line and column for column. A regex over it sees code only: a docstring
    that explains why nothing calls datetime.now() is not an offender, while
    code inside an f-string's braces still is."""
    rows = [list(line) for line in src.splitlines(keepends=True)]
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in _BLANKED:
            continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for r in range(r0, r1 + 1):
            row = rows[r - 1]
            for c in range(c0 if r == r0 else 0, min(c1 if r == r1 else len(row), len(row))):
                if row[c] not in "\r\n":
                    row[c] = " "
    return "".join("".join(row) for row in rows)


def _hits(rules: dict, code: str) -> list[str]:
    out = []
    for what, rx in rules.items():
        for m in re.finditer(rx, code, re.M):
            out.append(f"{what} at line {code.count(chr(10), 0, m.start()) + 1}")
    return out


# -- No network library -----------------------------------------------------
# tests/test_props.py's list for the props closure, the names a fetcher could
# hide behind included: subprocess (a curl), importlib, ctypes and the rest.
NETWORK_LIBS = frozenset({
    "requests", "urllib", "urllib3", "curl_cffi", "httpx", "aiohttp", "pycurl", "socket",
    "ssl", "websocket", "websockets", "http", "ftplib", "smtplib", "poplib", "imaplib",
    "nntplib", "telnetlib", "xmlrpc", "socketserver", "asyncio", "nba_api", "subprocess",
    "importlib", "ctypes", "multiprocessing", "webbrowser", "pty", "builtins", "selenium",
    "playwright", "mechanize"})
_LIBS = "|".join(sorted(NETWORK_LIBS, key=lambda s: (-len(s), s)))     # one fixed order
# `import x`, `import a, x`, `import x.y as z`, and either after a `;`.
_IMPORT_RE = (r"(?:^|;)[ \t]*import[ \t]+(?:[\w.]+(?:[ \t]+as[ \t]+\w+)?[ \t]*,[ \t]*)*(?:"
              + _LIBS + r")\b")
# `from x import y` and `from x.y import z`: `from urllib import request` and
# `from http import client` name the network module only as the PACKAGE.
_FROM_RE = r"(?:^|;)[ \t]*from[ \t]+(?:" + _LIBS + r")\b"
_DYNAMIC = {
    "an import by string": r"\b__import__\b|\bimport_module\b|\bimportlib\b",
    "exec, eval or compile": r"(?<![\w.])(?:exec|eval|compile)\s*\(",
    "a process spawn": r"\bos\s*\.\s*(?:system|popen|exec\w*|spawn\w*|fork\w*|posix_spawn\w*)\b",
    "sys.modules": r"\bsys\s*\.\s*modules\b",
}
# bucketlib's own network side, which no pure module may import, directly or
# by name: http (curl_cffi), live (every network read the app makes), store
# (it calls live), cache (it reads the environment and the disk at import).
FEED_NETWORK_SIDE = frozenset({"http", "live", "store", "cache"})


def _package_of(rel: str) -> list[str]:
    """bucketlib/sources/oddsapi.py -> ["bucketlib", "sources"]."""
    return rel[:-3].split("/")[:-1] if rel.endswith(".py") else []


def _imported_modules(src: str, rel: str) -> list[tuple[int, str]]:
    """(line, dotted name) for every import, relative ones resolved against
    `rel`'s package; `from p import a` yields p and p.a (a may be a module)."""
    out = []
    pkg = _package_of(rel)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - (node.level - 1)] if node.level - 1 <= len(pkg) else []
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod:
                out.append((node.lineno, mod))
            out += [(node.lineno, f"{mod}.{a.name}" if mod else a.name) for a in node.names]
    return out


def _network_imports(src: str, rel: str, pure: bool) -> list[str]:
    code = _code(src)
    bad = _hits({"a network library import": _IMPORT_RE,
                 "a network library from-import": _FROM_RE, **_DYNAMIC}, code)
    for line, mod in _imported_modules(src, rel):
        parts = mod.split(".")
        if parts[0] in NETWORK_LIBS:
            bad.append(f"imports {mod} at line {line} (AST)")
        if pure and parts[0] == "bucketlib" and len(parts) > 1 and parts[1] in FEED_NETWORK_SIDE:
            bad.append(f"imports bucketlib's network side ({mod}) at line {line}")
        if pure and parts[:2] == ["bucketlib", "sources"] and len(parts) > 2 and parts[2] != "oddsapi":
            bad.append(f"imports a source module that fetches ({mod}) at line {line}")
    return bad


@pytest.mark.parametrize("rel", NO_NETWORK)
def test_the_feeds_pure_modules_and_the_probe_import_no_network_library(rel):
    assert _network_imports(_source(rel), rel, pure=rel in PURE) == []


def _vendor_path_problems(src: str, rel: str) -> list[str]:
    """The probe tests the code the site runs: bucketlib.http.vendor_get,
    looked up as an attribute when a call is made (so a test's stub is the
    one that runs), never bound by name at import, and never http.fetch,
    which retries, keeps str(e) and impersonates Chrome (design 5.2 row 8)."""
    mods = {m for _, m in _imported_modules(src, rel)}
    attrs = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    bad = [] if "bucketlib.http" in mods else ["does not import bucketlib.http"]
    bad += [] if "vendor_get" in attrs else ["never names http.vendor_get"]
    bad += [f"binds {m} at import" for m in sorted(mods) if m.endswith(".vendor_get") or m.endswith(".fetch")]
    bad += ["names http.fetch"] if "fetch" in attrs else []
    return bad


def test_the_probe_reaches_the_vendor_through_bucketlib_http_vendor_get_only():
    assert _vendor_path_problems(_source(PROBE), PROBE) == []


@pytest.mark.parametrize("laundered", [
    "\nfrom bucketlib.http import vendor_get\n",
    "\nfrom bucketlib.http import fetch\n",
    "\ndef f(u):\n    return H.fetch(u)\n",
])
def test_the_vendor_path_guard_catches_a_bound_name_or_fetch(laundered):
    assert _vendor_path_problems(_source(PROBE) + laundered, PROBE), laundered


@pytest.mark.parametrize("laundered", [
    "\nimport requests\n",
    "\nfrom urllib import request\n",
    "\nfrom urllib.request import urlopen\n",
    "\nimport urllib.request as u\n",
    "\nimport urllib3\n",
    "\nimport os, socket\n",
    "\nimport json as j, socket as s\n",
    "\nfrom http import client\n",
    "\nimport http.client\n",
    "\nimport httpx\n",
    "\nfrom curl_cffi import requests as cr\n",
    "\nx = 1; import socket\n",
    "\ndef f():\n    import aiohttp\n    return aiohttp\n",
    "\nimport subprocess\n",
    "\ndef f():\n    return __import__('socket')\n",
    "\nimport importlib\ndef f():\n    return importlib.import_module('socket')\n",
    "\ndef f(c):\n    return eval(c)\n",
    "\nimport os\ndef f():\n    os.system('curl -s example.invalid')\n",
    "\nimport sys\ndef f():\n    return sys.modules['urllib.request']\n",
    "\nimport \\\n    socket\n",
])
@pytest.mark.parametrize("rel", [ODDSAPI, PROBE])
def test_the_network_guard_catches_a_laundered_import(rel, laundered):
    assert _network_imports(_source(rel) + laundered, rel, pure=rel in PURE), laundered


@pytest.mark.parametrize("laundered", [
    "\nfrom .. import http\n",
    "\nfrom ..http import vendor_get\n",
    "\nfrom .. import live\n",
    "\nfrom ..live import _read_key\n",
    "\nfrom bucketlib import store\n",
    "\nimport bucketlib.http as H\n",
    "\nfrom .. import cache\n",
    "\nfrom . import espn\n",
    "\nfrom bucketlib.sources.cdn import SCHEDULE_URL\n",
])
def test_a_pure_module_may_not_import_bucketlibs_network_side(laundered):
    assert _network_imports(_source(ODDSAPI) + laundered, ODDSAPI, pure=True), laundered


@pytest.mark.parametrize("widened", [
    "from urllib import request\n", "from http import client\n", "import os, socket\n",
    "import json as j, http.client as hc\n", "x = 1; import requests\n", "    import httpx\n",
])
def test_the_regex_style_alone_catches_the_widened_forms(widened):
    """RF 1.12: `from urllib import x` and its kin, by the regex alone."""
    assert _hits({"import": _IMPORT_RE, "from-import": _FROM_RE}, _code(widened)), widened


def test_the_ast_pass_alone_catches_what_a_line_regex_cannot():
    src = "import \\\n    socket\nfrom bucketlib import http as H\n"
    assert _hits({"import": _IMPORT_RE, "from-import": _FROM_RE}, _code(src)) == []
    assert [m for _, m in _imported_modules(src, PROBE)] == ["socket", "bucketlib", "bucketlib.http"]
    assert _network_imports(src, PROBE, pure=False) == ["imports socket at line 1 (AST)"]


def test_the_network_guard_reads_code_not_prose():
    """A docstring or comment that NAMES a library is not an import, and the
    probe's `from bucketlib import http` is the vendor path, not stdlib http."""
    prose = '\n"""Never import requests; from urllib import request is banned."""\n# import socket\n'
    assert _network_imports(_source(ODDSAPI) + prose, ODDSAPI, pure=True) == []
    assert _network_imports("from bucketlib import http as H\n", PROBE, pure=False) == []


# -- No I/O and no clock in the pure modules ------------------------------------
IO_SHAPES = {
    "open(": r"\bopen\s*\(",
    "read_text(": r"\.read_text\s*\(",
    "read_bytes(": r"\.read_bytes\s*\(",
    "write_text(": r"\.write_text\s*\(",
    "write_bytes(": r"\.write_bytes\s*\(",
    "os.environ": r"\bos\s*\.\s*environ\b",
    "getenv(": r"\bgetenv\s*\(",
    "Path(": r"\bPath\s*\(",
    "a pandas reader or writer": (r"\.(?:read|to)_(?:csv|parquet|json|pickle|feather|excel|hdf|"
                                  r"html|xml|sql|orc|stata|sas|spss|fwf|table|clipboard)\s*\("),
    "an I/O module": (r"(?:^|;)[ \t]*(?:import|from)[ \t]+(?:os|pathlib|io|shutil|tempfile|glob|"
                      r"pickle|shelve|sqlite3|gzip|bz2|lzma|zipfile|tarfile|codecs|csv)\b"),
}
CLOCK_SHAPES = {
    ".today(": r"\.today\s*\(",
    ".now(": r"\.now\s*\(",
    "utcnow(": r"\butcnow\s*\(",
    "time.time(": r"\btime\s*\.\s*time\s*\(",
    "time.monotonic(": r"\btime\s*\.\s*monotonic\s*\(",
    "Timestamp.now(": r"\bTimestamp\s*\.\s*now\s*\(",
    "util's clocks": r"\b(?:now_utc|now_et|today_et)\s*\(",
    "another clock": r"\b(?:perf_counter|perf_counter_ns|time_ns|monotonic_ns|process_time|thread_time)\s*\(",
    "the time module": r"(?:^|;)[ \t]*(?:import|from)[ \t]+time\b",
}
_CLOCK_WORDS = frozenset({"now", "today", "utcnow"})     # pd.Timestamp("now"), np.datetime64("today")


def _io_and_clock(src: str) -> list[str]:
    bad = _hits({**IO_SHAPES, **CLOCK_SHAPES}, _code(src))
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        and arg.value.strip().lower() in _CLOCK_WORDS):
                    bad.append(f"a clock read by string at line {node.lineno}")
    return bad


@pytest.mark.parametrize("rel", PURE)
def test_the_pure_modules_do_no_io_and_read_no_clock(rel):
    assert _io_and_clock(_source(rel)) == []


@pytest.mark.parametrize("laundered", [
    "\ndef f(p):\n    return open(p).read()\n",
    "\ndef f(p):\n    return p.read_text()\n",
    "\ndef f(p):\n    return p.read_bytes()\n",
    "\ndef f(p):\n    p.write_text('x')\n",
    "\nimport os\nKEY = os.environ.get('BUCKETVALUE_ODDS_API_KEY')\n",
    "\nimport os\nKEY = os.getenv('BUCKETVALUE_ODDS_API_KEY')\n",
    "\nfrom pathlib import Path\nP = Path('x')\n",
    "\ndef f():\n    return pd.read_parquet('x')\n",
    "\ndef f(df):\n    df.to_csv('x')\n",
    "\nimport gzip\n",
    "\ndef f():\n    return dt.date.today()\n",
    "\ndef f():\n    return dt.datetime.now(UTC)\n",
    "\ndef f():\n    return pd.Timestamp.now(tz='UTC')\n",
    "\ndef f():\n    return dt.datetime.utcnow()\n",
    "\nimport time\ndef f():\n    return time.time()\n",
    "\ndef f():\n    return time.monotonic()\n",
    "\nfrom ..util import now_utc\ndef f():\n    return now_utc()\n",
    "\ndef f():\n    return today_et()\n",
    "\ndef f():\n    return pd.Timestamp('now')\n",
    "\ndef f():\n    return pd.to_datetime('today', utc=True)\n",
    "\nfrom time import perf_counter\n",
])
def test_the_io_and_clock_guard_catches_a_laundered_read(laundered):
    assert _io_and_clock(_source(ODDSAPI) + laundered), laundered


def test_the_today_parameter_and_prose_pass_the_clock_guard():
    """m14: parse_events takes `today`, and a docstring may say why nothing
    calls datetime.now(); only call SHAPES in code are offenders."""
    ok = ('\ndef f(today, now):\n    """Never datetime.now() or open(path)."""\n'
          '    return today, now   # not date.today()\n')
    assert _io_and_clock(_source(ODDSAPI) + ok) == []
    assert re.search(r"def parse_events\(payload, today\b", _source(ODDSAPI))   # the parameter is there


# -- The vendor's host, and nothing of the vendor in the bot or Actions ---------
VENDOR_DOMAIN = b"the-odds-api"          # api.the-odds-api.com and every other spelling of it
HOST_HOMES = frozenset({HTTP, ODDSAPI, PROBE})


def _vendor_named(files) -> list[str]:
    """Files outside tests/ that name the vendor's domain, other than the
    three that may: http (the allow-list), oddsapi (HOST) and the probe."""
    return [rel for rel, data in files
            if rel not in HOST_HOMES and not rel.startswith("tests/") and VENDOR_DOMAIN in data.lower()]


def _read_all(rels) -> list[tuple[str, bytes]]:
    return [(rel, (REPO / rel).read_bytes()) for rel in rels]


def test_the_vendor_host_appears_only_in_http_oddsapi_and_the_probe():
    files = _shippable()
    assert {HTTP, ODDSAPI, PROBE} <= set(files)                  # the listing is real
    assert _vendor_named(_read_all(files)) == []
    for rel in (HTTP, ODDSAPI):
        assert b"api.the-odds-api.com" in (REPO / rel).read_bytes(), rel


def test_the_host_guard_catches_the_vendor_named_elsewhere():
    files = [("bucketlib/store.py", b"BASE = 'https://API.The-Odds-Api.com/v4'"),
             ("docs/runbook.md", b"see the-odds-api.com"), ("state/x.json", b"api.the-odds-api.com"),
             (ODDSAPI, b"api.the-odds-api.com"), ("tests/test_x.py", b"api.the-odds-api.com")]
    assert _vendor_named(files) == ["bucketlib/store.py", "docs/runbook.md", "state/x.json"]


BOT_FILES = ("scripts/daily_update.py", "scripts/commit_state.sh", "scripts/probe_hosts.py")
VENDOR_WORDS = ("odds-api", "oddsapi", "oddsbudget", "probe_oddsapi", "bucketvalue_odds_",
                "odds_api", "the odds api")


def _bot_and_actions() -> list[str]:
    wf = sorted(str(p.relative_to(REPO)) for p in (REPO / ".github" / "workflows").iterdir()
                if p.suffix in (".yml", ".yaml"))
    return wf + list(BOT_FILES)


def _vendor_words(files) -> list[str]:
    return [f"{rel}: {w}" for rel, text in files for w in VENDOR_WORDS if w in text.lower()]


def test_nothing_of_the_vendor_is_in_the_bot_or_in_actions():
    """design 2.10: the bot commits all of state/ and its Actions logs are
    public, so no workflow, daily_update.py, commit_state.sh or probe_hosts.py
    (whose output is committed) names the vendor, its modules, the probe or a
    BUCKETVALUE_ODDS_ variable."""
    rels = _bot_and_actions()
    assert {".github/workflows/daily.yml", ".github/workflows/tests.yml",
            ".github/workflows/probe.yml"} <= set(rels)
    files = [(rel, (REPO / rel).read_text(encoding="utf-8")) for rel in rels]
    assert _vendor_words(files) == []


def test_bv_healthcheck_url_stays_the_only_secret_in_actions():
    names = set()
    for rel in _bot_and_actions():
        if rel.startswith(".github/"):
            names |= set(re.findall(r"secrets\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)",
                                    (REPO / rel).read_text(encoding="utf-8")))
    assert names == {"BV_HEALTHCHECK_URL"}


@pytest.mark.parametrize("laundered", [
    "      - run: python scripts/probe_oddsapi.py --balance\n",
    "        env:\n          BUCKETVALUE_ODDS_API_KEY: ${{ secrets.BUCKETVALUE_ODDS_API_KEY }}\n",
    "    from bucketlib.sources import oddsapi\n",
    "    ('odds', 'https://api.the-odds-api.com/v4/sports', None, 'json'),\n",
    "# The Odds API lines\n",
])
def test_the_bot_guard_catches_the_vendor_in_a_workflow_or_the_bot(laundered):
    for rel in (".github/workflows/daily.yml", "scripts/daily_update.py", "scripts/probe_hosts.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert _vendor_words([(rel, text + laundered)]), (rel, laundered)


# -- Errors stay types on the page ------------------------------------------------
def test_show_error_details_stays_type():
    """RC K6: Streamlit 1.59's default is "full", and its old `false` maps to
    "stacktrace" (.streamlit/config.toml:36-42). With "type" a visitor sees an
    exception's type, never its text, which for the feed could name the URL."""
    cfg = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["showErrorDetails"] == "type"


# -- No widget that offers the rows for download ----------------------------------
# st.dataframe and st.data_editor carry a CSV download that cannot be turned
# off (RC G12); st.table, st.json and st.download_button hand rows over too
# (RC R3). st.write renders a DataFrame AS st.dataframe, so it is banned on
# Streamlit too. Odds render only through the HTML tables (st.markdown + esc).
WIDGETS = frozenset({"dataframe", "data_editor", "experimental_data_editor", "download_button"})
WIDGETS_ON_STREAMLIT = WIDGETS | {"table", "json", "write", "write_stream"}
WIDGETS_ON_A_CONTAINER = WIDGETS | {"table", "json"}     # col.table(df), tab.json(x)


def _download_widgets(src: str) -> list[str]:
    """Calls or references that would render a download-capable widget:
    st.<w>, a container's .<w>() (st.columns, st.tabs, st.sidebar ...),
    `from streamlit import <w>` and getattr(st, "<w>"). A non-Streamlit
    module's own .json() or .table() (json, pyarrow) passes."""
    tree = ast.parse(src)
    st_names, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                (st_names if a.name.split(".")[0] == "streamlit" else modules).add(bound)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if node.module.split(".")[0] == "streamlit" and a.name in WIDGETS_ON_STREAMLIT:
                    return [f"from streamlit import {a.name} at line {node.lineno}"]
                modules.add(a.asname or a.name)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Call, ast.Subscript)):
                root = root.func if isinstance(root, ast.Call) else root.value
            rname = root.id if isinstance(root, ast.Name) else None
            if node.attr in WIDGETS:
                bad.append(f".{node.attr} at line {node.lineno}")
            elif rname in st_names and node.attr in WIDGETS_ON_STREAMLIT:
                bad.append(f"{rname}...{node.attr} at line {node.lineno}")
            elif node.attr in WIDGETS_ON_A_CONTAINER and rname not in modules:
                bad.append(f"a container's .{node.attr} at line {node.lineno}")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and len(node.args) >= 2
              and isinstance(node.args[1], ast.Constant)
              and node.args[1].value in WIDGETS_ON_STREAMLIT):
            bad.append(f'getattr(..., "{node.args[1].value}") at line {node.lineno}')
    return bad


def _page_code() -> list[str]:
    return [rel for rel in _shippable() if rel.endswith(".py")
            and (rel.startswith("bucketlib/") or rel.startswith("pages/") or rel == "app.py")]


def test_no_widget_offers_the_rows_for_download():
    rels = _page_code()
    assert {"app.py", "pages/Game.py", "bucketlib/props_ui.py", "bucketlib/store.py"} <= set(rels)
    bad = [f"{rel}: {b}" for rel in rels for b in _download_widgets(_source(rel))]
    assert bad == [], bad


@pytest.mark.parametrize("laundered", [
    "\nst.dataframe(df)\n",
    "\nst.data_editor(df)\n",
    "\nst.table(df)\n",
    "\nst.json(rows)\n",
    "\nst.download_button('lines', data=b'')\n",
    "\nst.write(df)\n",
    "\nst.sidebar.dataframe(df)\n",
    "\na, b = st.columns(2)\na.table(df)\n",
    "\nwith st.expander('x'):\n    st.container().json(rows)\n",
    "\nimport streamlit\nstreamlit.dataframe(df)\n",
    "\nfrom streamlit import data_editor\n",
    "\ngetattr(st, 'json')(rows)\n",
    "\nst.experimental_data_editor(df)\n",
])
def test_the_widget_guard_catches_a_download_widget(laundered):
    for rel in ("app.py", "pages/Game.py", "bucketlib/props_ui.py"):
        assert _download_widgets(_source(rel) + laundered), (rel, laundered)


def test_the_widget_guard_passes_plain_code():
    ok = ("import json\nimport pyarrow as pa\nimport streamlit as st\n"
          "x = json.dumps({})\nt = pa.table({'a': [1]})\nst.markdown('<table></table>')\n"
          "def f(fh):\n    fh.write('x')\n")
    assert _download_widgets(ok) == []


# -- No key-shaped value in a file a commit could carry --------------------------
# m14 and design 5.2 row 13: an apiKey pair with a value of 20 or more letters
# and digits, or a valued BUCKETVALUE_ODDS_API_KEY (any *API_KEY name, any
# quoting: `= "..."`, `: "..."`, `"]= "..."`, `", "..."`). A placeholder such
# as "<the key>" or YOUR_API_KEY is not key-shaped. The fake keys the tests
# use are built at run time ("K" * 40), so no test file carries one.
_KEY_SHAPED = re.compile(rb"(?i)api[_-]?key[\"'\]]{0,3}\s*[=:,]\s*[\"']?[A-Za-z0-9]{20,}")


# A bare key file (build 4's review, L1-3): a file whose whole content is one
# run of 20 or more letters and digits, as `pbpaste > odds_api_key` writes it.
# A fixture's 32-character event ids sit inside JSON quotes, so they never match.
_BARE_KEY = re.compile(rb"\A\s*[A-Za-z0-9]{20,}\s*\Z")


def _key_shaped(files) -> list[str]:
    """`file:line` for each key-shaped value, and for a bare key file; never
    the value itself."""
    out = []
    for rel, data in files:
        if _BARE_KEY.match(data):
            out.append(f"{rel}:1")
            continue
        for m in _KEY_SHAPED.finditer(data):
            line = data.count(b"\n", 0, m.start()) + 1
            out.append(f"{rel}:{line}")
    return out


def test_no_file_a_commit_could_carry_holds_a_key_shaped_value():
    files = _shippable()
    assert ".gitignore" in files and "tests/test_vendor_http.py" in files      # the listing is real
    bad = _key_shaped(_read_all(files))
    assert bad == [], f"key-shaped value at {bad} (value not shown)"


def test_the_key_guard_catches_a_key_and_passes_the_placeholders():
    fake = ("A1b2" * 10).encode()                 # 40 letters and digits, built at run time
    caught = [b"https://example.invalid/v4/sports?apiKey=" + fake + b"&x=1",
              b'BUCKETVALUE_ODDS_API_KEY = "' + fake + b'"',
              b"BUCKETVALUE_ODDS_API_KEY=" + fake,
              b"export THE_ODDS_API_KEY=" + fake,
              b'os.environ["BUCKETVALUE_ODDS_API_KEY"] = "' + fake + b'"',
              b'monkeypatch.setenv("BUCKETVALUE_ODDS_API_KEY", "' + fake + b'")',
              b'params = {"apiKey": "' + fake + b'"}',
              b"api_key: '" + fake[:20] + b"'"]
    for i, data in enumerate(caught):
        assert _key_shaped([(f"f{i}", b"x\n" + data)]) == [f"f{i}:2"], data[:30]
    passed = [b'BUCKETVALUE_ODDS_API_KEY = "<the key>"', b"apiKey=YOUR_API_KEY", b'"apiKey="',
              b"apiKey={KEY}", b'KEY = "K" * 40', b"BUCKETVALUE_ODDS_API_KEY_FILE=~/.config/bucketvalue/x",
              b'monkeypatch.setenv("BUCKETVALUE_ODDS_API_KEY", KEY)', b"apiKey=" + fake[:19]]
    assert _key_shaped([(f"p{i}", d) for i, d in enumerate(passed)]) == []
    assert fake.decode() not in " ".join(_key_shaped([("f", caught[0])]))    # never the value


# -- Build 4's fix pass: review L1-3 -------------------------------------------------------
DESIGN_KEY_FILES = ("odds_api_key", "scripts/odds_api_key", "tests/fixtures/odds_api_key",
                    "my_odds_api_key.txt", "odds_api_key.bak")


def test_the_designs_key_file_name_is_git_ignored_wherever_it_lands_l1_3():
    """Design 4.3 names the key file odds_api_key (no extension). Run from
    the repo with a relative path, `pbpaste > odds_api_key` would write it
    into the checkout, where *.key never matched it: it is ignored by name,
    anywhere, and nothing so named is tracked."""
    lines = [line.strip() for line in (REPO / ".gitignore").read_text().splitlines()]
    assert "odds_api_key" in lines and "*odds_api_key*" in lines
    for path in DESIGN_KEY_FILES:
        assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not git-ignored"
    assert not [p for p in _shippable() if "odds_api_key" in p.rsplit("/", 1)[-1]]


def test_the_key_guard_catches_a_bare_key_file_l1_3():
    fake = ("A1b2" * 10).encode()                 # 40 letters and digits, built at run time
    for data in (fake, fake + b"\n", b"  " + fake + b"\r\n", fake[:20]):
        assert _key_shaped([("odds_api_key", data)]) == ["odds_api_key:1"], data[:8]
    passed = [b'{"id": "' + fake[:32] + b'"}', fake[:19], b"word " + fake, fake + b" word", b""]
    assert _key_shaped([(f"p{i}", d) for i, d in enumerate(passed)]) == []
    assert fake.decode() not in " ".join(_key_shaped([("odds_api_key", fake)]))
