"""scripts/probe_oddsapi.py, the first-key-day probe (design section 10).

It never runs against the vendor here. It is tested only through its refusal
paths, each of which must exit 2 before the key is read and before any call,
and through its pure helpers: the steps table and the budget, the one call
wrapper (given a fake in place of http.vendor_get), the line cleaner, the
readers of an answer and the verdicts. http.vendor_get is replaced by a
recorder in every test that reaches main(), and conftest's session block
stands behind it. Answers are built by hand or read from the CONSTRUCTED
fixtures in tests/fixtures/oddsapi; the fake key is built at run time.
"""
from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import http as H
from bucketlib import live
from bucketlib.sources import oddsapi as O
from bucketlib.util import UTC

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "scripts" / "probe_oddsapi.py"
FIX = REPO / "tests" / "fixtures" / "oddsapi"
KEY = "K" * 40                     # a fake key, built at run time
AT = dt.datetime(2026, 3, 12, 13, 0, tzinfo=UTC)     # the serve fixture's morning


def _load():
    spec = importlib.util.spec_from_file_location("probe_oddsapi", PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod       # dataclasses look their module up while a class is built
    spec.loader.exec_module(mod)
    return mod


PR = _load()


def _fixture(name: str):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def calls(monkeypatch):
    """http.vendor_get replaced by a recorder. Nothing here may reach it."""
    made = []

    def stub(*a, **k):
        made.append((a, k))
        raise AssertionError("the probe called the vendor")
    monkeypatch.setattr(H, "vendor_get", stub)
    return made


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A fake home folder, no vendor variable and no Actions marker, and a
    working directory with nothing in it. The fake home is the account's
    home too (the password database's, which the iCloud rule also reads),
    so no test looks at the real one."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setattr(PR, "_account_home", lambda: str(h))
    for k in ("GITHUB_ACTIONS", live.KEY_ENV, live.KEY_FILE_ENV):
        monkeypatch.delenv(k, raising=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return h


def _key_file(folder: Path, mode: int = 0o600, data: bytes | None = None,
              name: str = "odds_api_key") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes((KEY + "\n").encode() if data is None else data)
    p.chmod(mode)
    return p


def _run(capsys, *argv) -> tuple[int, str]:
    code = PR.main(list(argv))
    got = capsys.readouterr()
    text = got.out + got.err
    assert KEY not in text and "apiKey" not in text and "://" not in text
    return code, text


MODES = ([], ["--balance"], ["--evening"])


# -- Refusals: exit 2, nothing read, nothing called ---------------------------
@pytest.mark.parametrize("mode", MODES)
def test_no_key_file_is_a_refusal(home, calls, capsys, mode):
    code, text = _run(capsys, *mode)
    assert code == 2 and calls == []
    assert "FAIL refused: no key file: pass --key-file or set BUCKETVALUE_ODDS_API_KEY_FILE" in text
    code, text = _run(capsys, *mode, "--key-file", "  ")
    assert code == 2 and calls == [] and "no key file" in text


def test_the_shells_key_variable_is_never_read(home, calls, capsys, monkeypatch):
    """The probe reads a key FILE only: a key exported in the shell, with no
    key file named, still refuses."""
    monkeypatch.setenv(live.KEY_ENV, KEY)
    code, text = _run(capsys)
    assert code == 2 and calls == [] and "no key file" in text


@pytest.mark.parametrize("folder", ["Desktop", "Documents", "Library/Mobile Documents"])
@pytest.mark.parametrize("how", ["flag", "flag with ~", "environment"])
def test_a_key_file_in_an_icloud_folder_is_refused(home, calls, capsys, monkeypatch, folder, how):
    path = _key_file(home / folder / "keys")
    argv = []
    if how == "flag":
        argv = ["--key-file", str(path)]
    elif how == "flag with ~":
        argv = ["--key-file", "~/" + folder + "/keys/odds_api_key"]
    else:
        monkeypatch.setenv(live.KEY_FILE_ENV, str(path))
    for mode in MODES:
        code, text = _run(capsys, *mode, *argv)
        assert code == 2 and calls == []
        assert "refused: the key file is in an iCloud folder" in text


def test_the_icloud_rule_is_case_blind_and_follows_a_link(home, calls, capsys):
    """macOS folders are case-blind, and a link in ~/.config that points into
    ~/Desktop still puts the key in iCloud."""
    real = _key_file(home / "Desktop")
    assert PR.key_file_refusal("~/desktop/odds_api_key").startswith("the key file is in an iCloud folder")
    assert PR.key_file_refusal("~/DOCUMENTS/x").startswith("the key file is in an iCloud folder")
    link_dir = home / ".config" / "bucketvalue"
    link_dir.mkdir(parents=True)
    (link_dir / "odds_api_key").symlink_to(real)
    code, text = _run(capsys, "--key-file", str(link_dir / "odds_api_key"))
    assert code == 2 and calls == [] and "iCloud folder" in text


def test_the_icloud_rule_compares_folders_by_identity_not_by_spelling(home, calls, capsys, tmp_path,
                                                                       monkeypatch):
    """Review I1, F3. A spelling of ~/Desktop that Path.resolve() keeps as
    written (macOS's /System/Volumes/Data firmlink, played here by a folder
    link with resolve() made to keep it) is still ~/Desktop: the same device
    and inode. Compared by spelling alone, a 0600 key file there passed."""
    _key_file(home / "Desktop")
    alias = tmp_path / "another_spelling"
    alias.symlink_to(home / "Desktop", target_is_directory=True)
    monkeypatch.setattr(PR, "_resolved", lambda p: p)
    assert PR.key_file_refusal(str(alias / "odds_api_key")).startswith("the key file is in an iCloud folder")
    code, text = _run(capsys, "--key-file", str(alias / "odds_api_key"))
    assert code == 2 and calls == [] and "refused: the key file is in an iCloud folder" in text
    _key_file(home / ".config" / "bucketvalue")                         # the design's place still passes
    assert PR.key_file_refusal(str(home / ".config" / "bucketvalue" / "odds_api_key")) is None


def test_the_icloud_rule_sees_the_macos_data_volume_spelling(home, calls, capsys):
    """The real case on Barrett's Mac: /System/Volumes/Data + the path."""
    path = _key_file(home / "Desktop")
    alt = Path("/System/Volumes/Data" + os.path.realpath(path))
    if not (alt.exists() and os.path.samefile(alt, path)):
        pytest.skip("no /System/Volumes/Data firmlink here (not macOS)")
    assert PR._resolved(alt) == alt                     # resolve() keeps this spelling
    for mode in MODES:
        code, text = _run(capsys, *mode, "--key-file", str(alt))
        assert code == 2 and calls == [] and "refused: the key file is in an iCloud folder" in text


def test_the_icloud_rule_knows_the_accounts_own_home_when_home_is_moved(home, tmp_path, monkeypatch):
    """$HOME can point anywhere; the password database says where the
    account's iCloud folders really are, and both are checked."""
    account = tmp_path / "account"
    path = _key_file(account / "Documents" / "keys")
    monkeypatch.setattr(PR, "_account_home", lambda: str(account))
    assert PR.key_file_refusal(str(path)).startswith("the key file is in an iCloud folder")
    monkeypatch.setattr(PR, "_account_home", lambda: None)
    assert PR.key_file_refusal(str(path)) is None                       # $HOME alone cannot know it


def test_the_accounts_home_is_read_from_the_password_database(monkeypatch, tmp_path):
    pwd = pytest.importorskip("pwd")
    monkeypatch.setenv("HOME", str(tmp_path))
    account = pwd.getpwuid(os.getuid()).pw_dir
    assert PR._account_home() == account
    homes = PR._homes()
    assert Path(os.path.abspath(account)) in homes and Path(os.path.abspath(tmp_path)) in homes


@pytest.mark.parametrize("dot_git", ["folder", "file"])
def test_a_key_file_inside_a_git_checkout_is_refused(home, calls, capsys, tmp_path, dot_git):
    repo = tmp_path / "somerepo"
    repo.mkdir()
    if dot_git == "folder":
        (repo / ".git").mkdir()
    else:
        (repo / ".git").write_text("gitdir: elsewhere\n")       # a worktree's .git is a file
    path = _key_file(repo / "deep" / "er")
    for mode in MODES:
        code, text = _run(capsys, *mode, "--key-file", str(path))
        assert code == 2 and calls == []
        assert "refused: the key file is inside a git checkout" in text


@pytest.mark.parametrize("mode_bits", [0o640, 0o604, 0o644, 0o660, 0o666])
def test_a_key_file_group_or_others_can_read_is_refused(home, calls, capsys, mode_bits):
    path = _key_file(home / ".config" / "bucketvalue", mode=mode_bits)
    for mode in MODES:
        code, text = _run(capsys, *mode, "--key-file", str(path))
        assert code == 2 and calls == []
        assert "refused: group or others can read the key file; chmod 600 it" in text


def test_a_key_file_that_is_missing_or_not_a_plain_file_is_refused(home, calls, capsys):
    folder = home / ".config" / "bucketvalue"
    folder.mkdir(parents=True)
    code, text = _run(capsys, "--key-file", str(folder / "nothing_here"))
    assert code == 2 and calls == [] and "refused: the key file cannot be found or read" in text
    code, text = _run(capsys, "--key-file", str(folder))
    assert code == 2 and calls == [] and "refused: the key file is not a plain file" in text


@pytest.mark.parametrize("data", [b"", b"\n\n  \n"])
def test_a_key_file_with_no_key_in_it_is_refused(home, calls, capsys, data):
    path = _key_file(home / ".config" / "bucketvalue", data=data)
    code, text = _run(capsys, "--key-file", str(path))
    assert code == 2 and calls == [] and "refused: the key file holds no key" in text


def test_the_probe_refuses_to_run_in_github_actions(home, calls, capsys, monkeypatch):
    path = _key_file(home / ".config" / "bucketvalue")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    for mode in MODES:
        code, text = _run(capsys, *mode, "--key-file", str(path))
        assert code == 2 and calls == []
        assert "refused: the probe never runs in GitHub Actions" in text


def test_a_negative_cap_and_two_modes_are_refused(home, calls, capsys):
    path = _key_file(home / ".config" / "bucketvalue")
    code, text = _run(capsys, "--key-file", str(path), "--max-credits", "-1")
    assert code == 2 and calls == [] and "refused: --max-credits" in text
    with pytest.raises(SystemExit) as e:
        PR.main(["--key-file", str(path), "--balance", "--evening"])
    assert e.value.code == 2 and calls == []
    capsys.readouterr()


def test_a_good_key_file_passes_every_check(home):
    """The design's place (4.3): ~/.config/bucketvalue/odds_api_key, mode 600."""
    _key_file(home / ".config" / "bucketvalue")
    assert PR.key_file_refusal("~/.config/bucketvalue/odds_api_key") is None
    assert PR.key_file_refusal(str(home / ".config" / "bucketvalue" / "odds_api_key")) is None


def test_the_key_comes_from_the_checked_file_never_from_the_shell(home, monkeypatch):
    """live._read_key puts BUCKETVALUE_ODDS_API_KEY first; the probe hands it
    a mapping that names the file only, so the key sent is the file's."""
    other = "J" * 40
    _key_file(home / ".config" / "bucketvalue")
    monkeypatch.setenv(live.KEY_ENV, other)
    assert PR.read_key_file("~/.config/bucketvalue/odds_api_key") == KEY
    assert PR.read_key_file("") is None and PR.read_key_file(None) is None


# -- The steps table and the budget ------------------------------------------
def test_the_steps_are_the_designs_table():
    assert [s.n for s in PR.STEPS] == list(range(1, 14))
    assert [s.worst for s in PR.STEPS] == [0, 1, 1, 1, 1, 2, 1, 0, 0, 0, 1, 1, 1]
    assert [s.n for s in PR.STEPS if not s.paid] == [1, 10]           # the free /events calls
    assert PR.MAX_CREDITS == 16 and PR.MAX_GAMES == 7
    assert PR.planned_credits() == 16                    # "Worst case 16 credits" (section 10)
    assert PR.planned_credits(1) == 10 and PR.planned_credits(0) == 9
    assert [PR.room_needed(s) for s in PR.STEPS] == [0, 1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1]


def _ans(status=200, last=None, kind="", remaining=None, used=None, body=None, code=""):
    usage = None if status == 0 else O.Usage(remaining, used, last)
    return PR.Answer(status=status, code=code, kind=kind or ("ok" if status == 200 else ""),
                     usage=usage, body=body)


def test_the_budget_adds_each_calls_worst_case_before_it_is_made():
    b = PR.Budget(16)
    assert b.allows(16) and not b.allows(17)
    assert b.count(_ans(last=1), 1) == 1 and b.spent == 1
    assert b.count(_ans(last=0), 1) == 0 and b.spent == 1           # an empty answer is free
    assert b.count(_ans(last=None), 2) == 2 and b.spent == 3        # no header: the call's room
    assert b.count(_ans(status=0, kind="timeout"), 1) == 1 and b.spent == 4   # may have billed
    assert b.count(_ans(status=0, kind="refused"), 1) == 0 and b.spent == 4   # never sent
    assert b.allows(12) and not b.allows(13)


def test_the_cap_holds_even_when_a_step_counted_zero_bills():
    """Steps 8 and 9 count 0 (UNVERIFIED) but need 1 credit of room."""
    b = PR.Budget(16)
    b.spent = 15
    need = PR.room_needed(PR.STEP[8])
    assert need == 1 and b.allows(need)
    b.count(_ans(status=422, last=1, code="INVALID_MARKET"), need)    # it billed after all
    assert b.spent == 16 and not b.allows(PR.room_needed(PR.STEP[9]))
    assert b.allows(PR.room_needed(PR.STEP[10]))                       # a free call still runs


def test_a_call_that_would_pass_the_cap_is_never_made(calls):
    lines = []
    p = PR.Probe(KEY, PR.Report(secrets=(KEY,), write=lines.append), 0, now=AT,
                 today=dt.date(2026, 3, 12))
    path, params = PR.odds_request("a" * 32, O.BOOKS)
    assert p.call(3, path, params) is None
    assert calls == [] and p.budget.spent == 0
    assert lines == ["INFO step 3: skipped: 1 more credit(s) would pass --max-credits 0 (0 counted)"]
    p = PR.Probe(KEY, PR.Report(secrets=(KEY,), write=lines.append), 1, now=AT,
                 today=dt.date(2026, 3, 12))
    path, params = PR.odds_request("a" * 32, O.BOOKS, regions="us")
    assert p.call(6, path, params) is None and calls == []           # step 6 needs 2


def test_a_game_is_never_bought_inside_its_tip_guard_read_again_at_each_call(calls):
    """The games are picked at the start, but step 12 runs minutes later: the
    guard is read again before every call that names a game."""
    events, _ = O.parse_events(_fixture("events_constructed.json"), dt.date(2026, 3, 12))
    g = PR.untipped(events, AT)[0]                                   # tips 23:00Z
    assert PR.tip_far(g, g.commence_utc - dt.timedelta(minutes=6))
    assert not PR.tip_far(g, g.commence_utc - dt.timedelta(minutes=5))
    assert not PR.tip_far(g, g.commence_utc + dt.timedelta(minutes=1))
    lines, at = [], [g.commence_utc - dt.timedelta(minutes=4)]
    p = PR.Probe(KEY, PR.Report(secrets=(KEY,), write=lines.append), 16, now=AT,
                 today=dt.date(2026, 3, 12), clock=lambda: at[0])
    assert p.call(12, *PR.odds_request(g.event_id, O.BOOKS), game=g) is None
    assert calls == [] and p.budget.spent == 0
    assert lines == [f"INFO step 12: skipped: {PR.label(g)} tips within 5 minutes"]


# -- The one call wrapper -------------------------------------------------------
class FakeGet:
    """What http.vendor_get is to the probe: records each call and answers
    `then` (a VendorResponse) or raises it."""

    def __init__(self, then):
        self.then, self.seen = then, []

    def __call__(self, host, path, params, *, timeout, secrets=()):
        self.seen.append({"host": host, "path": path, "params": dict(params), "timeout": timeout,
                          "secrets": tuple(secrets)})
        if isinstance(self.then, BaseException):
            raise self.then
        return self.then


def _resp(status=200, body=None, snippet="", headers=None):
    return H.VendorResponse(status=status, headers={} if headers is None else headers, body=body,
                            snippet=snippet, nbytes=0)


def test_ask_sends_the_key_in_params_only_and_reads_the_answer():
    get = FakeGet(_resp(200, body=[], headers={"x-requests-remaining": "475", "x-requests-used": "25",
                                               "x-requests-last": "0"}))
    path, params = O.events_request()
    ans = PR.ask(KEY, path, params, get=get)
    (seen,) = get.seen
    assert seen["host"] == O.HOST == "api.the-odds-api.com" and seen["path"] == O.EVENTS_PATH
    assert seen["params"] == {"dateFormat": "iso", "apiKey": KEY}
    assert KEY in seen["secrets"] and seen["timeout"] == PR.TIMEOUT == live.TIMEOUT == 4.0
    assert "apiKey" not in params                          # the caller's params are not touched
    assert ans.ok and ans.cost == 0 and ans.usage.remaining == 475 and ans.usage.allowance == 500
    assert ans.headers == PR.QUOTA and ans.body == [] and ans.kind == "ok"


def test_ask_reads_an_error_as_our_codes_never_the_vendors_text():
    snippet = H.scrub(json.dumps({"message": "Invalid api key " + KEY, "error_code": "INVALID_KEY"}),
                      (KEY,))
    ans = PR.ask(KEY, O.EVENTS_PATH, {"dateFormat": "iso"}, get=FakeGet(_resp(401, snippet=snippet)))
    assert (ans.status, ans.code, ans.kind, ans.cost) == (401, "INVALID_KEY", "bad_key", None)
    assert ans.headers == () and KEY not in repr(ans) and "Invalid api key" not in repr(ans)
    out = PR.ask(KEY, O.EVENTS_PATH, {}, get=FakeGet(_resp(401, snippet='{"error_code": '
                                                           '"OUT_OF_USAGE_CREDITS"}')))
    assert (out.code, out.kind) == ("OUT_OF_USAGE_CREDITS", "out_of_credits")


@pytest.mark.parametrize("err, kind", [
    (H.VendorTransportError("timeout", 10), "timeout"),
    (H.VendorTransportError("oversize", 4_000_001), "oversize"),
    (ValueError("vendor timeout too small"), "refused"),
    (H.BadRequest("vendor", "None-valued param"), "refused"),
    (RuntimeError("GET https://api.the-odds-api.com/v4/sports?apiKey=" + KEY + " failed"),
     "error RuntimeError"),
    (KeyError("apiKey=" + KEY), "error KeyError"),
])
def test_ask_keeps_only_the_failures_kind_never_its_text(err, kind):
    ans = PR.ask(KEY, O.EVENTS_PATH, {"dateFormat": "iso"}, get=FakeGet(err))
    assert ans.status == 0 and ans.kind == kind and ans.usage is None and ans.body is None
    assert KEY not in repr(ans) and "://" not in repr(ans) and "apiKey" not in repr(ans)
    assert PR.describe(ans).startswith("no answer (")


@pytest.mark.parametrize("path", [
    O.EVENTS_PATH + "/" + "a" * 32 + "/markets",          # the event-markets endpoint: 1 credit, always
    O.EVENTS_PATH + "/" + "a" * 32 + "/markets/",
    "/v4/sports/basketball_nba/odds",
    "/v4/sports/upcoming/events",
    O.EVENTS_PATH + "/" + "a" * 31 + "/odds",
    O.EVENTS_PATH + "?apiKey=x",
    "https://" + O.HOST + O.EVENTS_PATH,
    None,
])
def test_ask_refuses_every_path_but_events_participants_and_one_games_odds(path):
    get = FakeGet(_resp(200, body=[]))
    ans = PR.ask(KEY, path, {}, get=get)
    assert get.seen == [] and ans.kind == "refused" and ans.status == 0
    assert PR.describe(ans) == "no answer (refused before sending)"


def test_every_path_the_steps_take_is_one_ask_allows():
    eid = "b" * 32
    paths = [O.events_request()[0], O.PARTICIPANTS_PATH]
    paths += [PR.odds_request(eid, books)[0] for books in (O.BOOKS, PR.DFS, ("prizepicks",))]
    paths += [PR.odds_request(PR.FAKE_EVENT_ID, O.BOOKS)[0]]
    for path in paths:
        get = FakeGet(_resp(200, body=[]))
        assert PR.ask(KEY, path, {}, get=get).ok and len(get.seen) == 1, path


def test_odds_request_is_the_sites_request_plus_one_change():
    eid = "c" * 32
    path, params = PR.odds_request(eid, O.BOOKS)
    assert (path, params) == O.event_odds_request(eid, O.BOOKS, ("player_points",))
    assert params["bookmakers"] == "draftkings,fanduel,betmgm,prizepicks,underdog"
    assert "regions" not in params and O.SEND_REGIONS is False
    assert PR.odds_request(eid, PR.DFS, markets=PR.ALTERNATE)[1]["markets"] == "player_points_alternate"
    assert PR.odds_request(eid, O.BOOKS, markets="player_pointz")[1]["markets"] == "player_pointz"
    assert PR.odds_request(eid, O.BOOKS, regions="us")[1]["regions"] == "us"
    assert PR.odds_request(eid, ("draftkings",), bookmakers="draftkings,notabook")[1]["bookmakers"] \
        == "draftkings,notabook"
    assert PR.odds_request(eid, PR.DFS, includeMultipliers="true")[1]["includeMultipliers"] == "true"
    assert "apiKey" not in params


def test_describe_says_status_code_and_cost_in_our_words():
    assert PR.describe(None) == "not made"
    assert PR.describe(_ans(200, last=1)) == "HTTP 200; x-requests-last 1"
    assert PR.describe(_ans(422, last=0, code="MISSING_REGION")) == "HTTP 422 MISSING_REGION; x-requests-last 0"
    assert PR.describe(_ans(200)) == "HTTP 200; no x-requests-last"
    assert PR.describe(_ans(status=0, kind="connect")) == "no answer (connect)"


def test_the_probes_ask_cannot_swallow_the_session_block(vendor_block):
    """ask() catches Exception and keeps only its type. conftest's block is
    a BaseException, so a call that reaches it fails the test instead of
    reading as "no answer (error ...)" (review I1, F4)."""
    with pytest.raises(vendor_block.error, match="a test reached the vendor network"):
        PR.ask(KEY, O.EVENTS_PATH, {"dateFormat": "iso"})          # get=None: http.vendor_get
    assert vendor_block.hits == 1


# -- A whole run against a fake vendor (review I1, F2) ---------------------------
class BillingVendor:
    """A whole run's vendor, played from the CONSTRUCTED fixtures: the events
    list, the participants, PHI at DET and DEN at SAS as their fixtures, and
    every other game with one hand-built DraftKings points line. It answers
    each step as design 10 expects (422 for the unknown market and book, 404
    for the made-up event, 401 for the made-up key, an empty answer for
    commenceTimeFrom), keeps only the books a call names, and bills `bill`
    credits for every game answer that holds a book: an empty one is free."""

    def __init__(self, bill: int):
        self.bill, self.left, self.calls = bill, 500, []        # a fresh free account
        self.events = _fixture("events_constructed.json")
        self.by_id = {e["id"]: e for e in self.events}
        self.odds = {p["id"]: p for p in (_fixture("event_odds_phi_det_constructed.json"),
                                          _fixture("event_odds_den_sas_constructed.json"))}

    def _usage(self, last: int) -> dict:
        self.left -= last
        return {"x-requests-remaining": str(self.left), "x-requests-used": str(500 - self.left),
                "x-requests-last": str(last)}

    def __call__(self, host, path, params, *, timeout, secrets=()):
        self.calls.append((path, dict(params)))
        assert host == O.HOST and timeout == PR.TIMEOUT and KEY in secrets
        if path == O.EVENTS_PATH:
            if params.get("apiKey") != KEY:
                return _resp(401, snippet='{"error_code": "INVALID_KEY"}')
            return _resp(200, body=self.events, headers=self._usage(0))
        if path == O.PARTICIPANTS_PATH:
            return _resp(200, body=_fixture("participants_constructed.json"), headers=self._usage(1))
        eid = path.split("/")[-2]
        if eid not in self.by_id:
            return _resp(404, snippet='{"error_code": "EVENT_NOT_FOUND"}', headers=self._usage(0))
        if params.get("markets") == "player_pointz":
            return _resp(422, snippet='{"error_code": "INVALID_MARKET"}', headers=self._usage(0))
        if "notabook" in params.get("bookmakers", ""):
            return _resp(422, snippet='{"error_code": "INVALID_BOOKMAKERS"}', headers=self._usage(0))
        body = json.loads(json.dumps(self.odds.get(eid) or {**self.by_id[eid], "bookmakers": [
            {"key": "draftkings", "title": "DraftKings", "markets": [{"key": "player_points", "outcomes": [
                {"name": "Over", "description": "Some Player", "price": -110, "point": 20.5},
                {"name": "Under", "description": "Some Player", "price": -110, "point": 20.5}]}]}]}))
        named = set(params.get("bookmakers", "").split(","))
        body["bookmakers"] = [] if "commenceTimeFrom" in params else [
            b for b in body["bookmakers"] if b["key"] in named]
        return _resp(200, body=body, headers=self._usage(self.bill if body["bookmakers"] else 0))


def _whole_run(bill: int, cap: int = PR.MAX_CREDITS):
    vendor, lines = BillingVendor(bill), []
    p = PR.Probe(KEY, PR.Report(secrets=(KEY,), write=lines.append), cap, get=vendor, now=AT,
                 today=dt.date(2026, 3, 12), sleep=lambda s: None, mono=lambda: 1e6, clock=lambda: AT)
    PR.run_full(p)
    assert not any(KEY in ln or "://" in ln or "apiKey" in ln for ln in lines)
    return p, vendor, lines


def test_a_whole_run_on_the_tables_bills_runs_every_step_inside_the_cap():
    """1 credit a game answer: step 2's 1, step 3's 7, then step 6's and step
    12's 1 each. Game 1 is PHX at IND (DraftKings only here), so the
    PrizePicks-and-Underdog steps 4 and 5 come back empty, and free."""
    p, vendor, lines = _whole_run(bill=1)
    assert "INFO step 3: game 1 is PHX at IND, game 2 is WAS at ORL, the earliest with lines" in lines
    assert p.budget.spent == 10 and p.budget.rate == 1
    assert not [ln for ln in lines if "would pass --max-credits" in ln]
    assert "PASS step 10: a bad key gets 401 INVALID_KEY" in "\n".join(lines)
    assert sum(1 for path, _ in vendor.calls if path.endswith("/odds")) == 7 + 9   # step 3, then 4 to 9, 11 to 13
    assert 500 - vendor.left == p.budget.spent


def test_a_vendor_that_bills_more_than_the_table_cannot_push_a_run_past_the_cap():
    """Fact 3a, which step 3 exists to test, wrong: every game answer bills 2.
    The first one shows it, and from then on every paid call needs room for
    2 credits a credit of its worst case: step 3 spends 1 + 7 x 2 = 15, and
    every later paid step is skipped. With the room fixed at the table's 1,
    step 4 was let in on the last credit and billed 2 (17 of 16), and then
    even the free step 10 was skipped."""
    p, vendor, lines = _whole_run(bill=2)
    text = "\n".join(lines)
    assert p.budget.spent == 15 <= p.budget.cap and p.budget.rate == 2
    assert 500 - vendor.left == 15                                       # what the vendor billed
    assert text.count("one market billed 2 credits [3a]") == 7
    assert "PASS step 10: a bad key gets 401 INVALID_KEY" in text           # a free step always runs
    for step, need in ((4, 2), (5, 2), (6, 4), (7, 2), (8, 2), (9, 2), (11, 2), (12, 2), (13, 2)):
        assert f"INFO step {step}: skipped: {need} more credit(s) would pass --max-credits 16" in text, step
    assert sum(1 for path, _ in vendor.calls if path.endswith("/odds")) == 7


def test_the_room_follows_the_most_a_game_call_billed_and_free_steps_always_run():
    b = PR.Budget(16)
    b.learn(PR.STEP[3], _ans(last=0))                       # an empty answer teaches nothing
    b.learn(PR.STEP[3], _ans(last=None))
    assert b.rate == 1
    b.learn(PR.STEP[6], _ans(last=2))                       # 2 for step 6's two credits: the table
    assert b.rate == 1
    b.learn(PR.STEP[3], _ans(last=3))
    b.learn(PR.STEP[4], _ans(last=1))                       # the rate never falls
    assert b.rate == 3
    assert [PR.room_needed(s, b.rate) for s in PR.STEPS] == [0, 3, 3, 3, 3, 6, 3, 3, 3, 0, 3, 3, 3]
    lines = []
    p = PR.Probe(KEY, PR.Report(secrets=(KEY,), write=lines.append), 16, now=AT,
                 today=dt.date(2026, 3, 12))
    p.budget.spent = 17                                     # past the cap: one call billed over its room
    assert p.can(1) and p.can(10) and not p.can(2)
    assert lines == ["INFO step 2: skipped: 1 more credit(s) would pass --max-credits 16 (17 counted)"]


# -- What is printed --------------------------------------------------------------
def test_a_line_never_carries_the_key_a_url_an_apikey_pair_or_a_long_token():
    dirty = ("x " + KEY + " https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey=" + KEY
             + " apikey: " + "Z" * 20 + " id " + "ab12" * 8 + " \x1b[31mred " + chr(7))
    clean = PR.safe_line(dirty, (KEY,))
    for bad in (KEY, "://", "the-odds-api", "apiKey", "apikey", "ab12" * 8, "\x1b", chr(7)):
        assert bad not in clean, bad
    assert clean.startswith("x [redacted] [url]") and "id [redacted]" in clean
    assert PR.safe_line("PASS step 2: all 30 teams map through teams.canonical [9a]") \
        == "PASS step 2: all 30 teams map through teams.canonical [9a]"


def test_a_vendor_name_is_cleaned_and_cut():
    assert PR.vendor_text("Nikola Jokic") == "Nikola Jokic"
    assert PR.vendor_text("  Cade \n Cunningham ") == "Cade Cunningham"
    assert PR.vendor_text("\x1b[2J" + "Cade " * 12) == ("[2J" + " ".join(["Cade"] * 12))[:40]
    assert PR.vendor_text("who " + KEY, (KEY,)) == "who [redacted]"
    assert PR.vendor_text("x" * 50) == "[redacted]"           # a key-shaped run, even unnamed
    assert PR.vendor_text(None) == "(not text)" and PR.vendor_text("   ") == "(blank)"


def test_the_report_writes_only_pass_fail_and_info_lines_made_safe():
    lines = []
    r = PR.Report(secrets=(KEY,), write=lines.append)
    r("PASS", "fine", "step 1")
    r("INFO", "the key is " + KEY)
    assert not r.failed
    r("FAIL", "bad", "step 3")
    assert r.failed and r.counts == Counter({"PASS": 1, "INFO": 1, "FAIL": 1})
    assert lines == ["PASS step 1: fine", "INFO the key is [redacted]", "FAIL step 3: bad"]
    with pytest.raises(ValueError):
        r("WARN", "x")


# -- Reading answers ------------------------------------------------------------
def _phi_det(**kw):
    payload = _fixture("event_odds_phi_det_constructed.json")
    return payload, O.parse_event_odds(payload, books=O.BOOKS, markets=PR.POINTS, fetched_at=AT, **kw)


def test_the_events_list_order_and_what_has_started():
    events = _fixture("events_constructed.json")
    assert PR.event_order(events) == (10, False)           # the fixture is not in tip order
    assert PR.event_order(sorted(events, key=lambda e: e["commence_time"])) == (10, True)
    assert PR.started(events, dt.datetime(2026, 3, 12, 23, 15, tzinfo=UTC)) == (10, 3)
    assert PR.started(events, dt.datetime(2026, 3, 12, 12, 0, tzinfo=UTC)) == (10, 0)
    assert PR.event_order({"message": "x"}) == (0, True) and PR.started(None, AT) == (0, 0)


def test_untipped_drops_games_inside_the_tip_guard_earliest_first():
    events, _ = O.parse_events(_fixture("events_constructed.json"), dt.date(2026, 3, 12))
    near = PR.untipped(events, dt.datetime(2026, 3, 12, 22, 56, tzinfo=UTC))      # 4 minutes to 7 PM ET
    assert [PR.label(g) for g in near] == ["BKN at ATL", "MIL at MIA", "DAL at MEM", "DEN at SAS",
                                           "BOS at OKC", "CHI at LAL"]
    early = PR.untipped(events, AT)
    assert len(early) == 9 and PR.label(early[0]) in ("PHI at DET", "PHX at IND", "WAS at ORL")
    assert PR.fmt_et(early[0].commence_utc) == "7:00 PM ET"
    assert PR.untipped(events, dt.datetime(2026, 3, 13, 5, 0, tzinfo=UTC)) == []


def test_game_1_and_game_2_are_the_earliest_games_step_3_read_lines_for():
    """A game nobody has posted at 9 AM shows steps 4 to 12 nothing."""
    a, b, c = "PHX at IND", "PHI at DET", "DEN at SAS"
    assert PR.first_two([b, c], [a, b, c]) == (b, c)
    assert PR.first_two([b], [a, b, c]) == (b, b)
    assert PR.first_two([], [a, b, c]) == (a, b)
    assert PR.first_two([], [a]) == (a, a)


def test_the_participants_map_to_all_30_teams():
    known, unknown = PR.map_participants(_fixture("participants_constructed.json"))
    assert len(known) == 30 and unknown == []
    extra = _fixture("participants_constructed.json") + [{"full_name": "Seattle SuperSonics"}, {"id": "x"}]
    known, unknown = PR.map_participants(extra)
    assert len(known) == 30 and unknown == ["Seattle SuperSonics", None]


def test_lines_per_book_sides_and_raw_dfs_prices():
    payload, frame = _phi_det()
    assert PR.book_rows(frame) == {"draftkings": 7, "fanduel": 5, "betmgm": 4, "prizepicks": 6,
                                   "underdog": 3}
    assert PR.book_rows(frame.iloc[:0]) == {b: 0 for b in O.BOOKS}
    sides = PR.side_counts(frame)
    assert sides["fanduel"] == Counter({"both sides": 4, "Over only": 1})
    assert sides["betmgm"] == Counter({"both sides": 2, "Under only": 2})
    assert sides["draftkings"] == Counter({"both sides": 7})
    prices = PR.dfs_prices(payload)                         # the parser blanks these to NaN
    assert prices == {"prizepicks": Counter({"-137": 12}), "underdog": Counter({"-137": 6})}
    assert frame.loc[frame["book"] == "prizepicks", "price_over"].isna().all()
    assert PR.counts_text(Counter({"-137": 12, "-120": 2})) == "-137 x 12, -120 x 2"


def test_the_bookmaker_level_stamp_is_counted():
    payload, _ = _phi_det()
    assert PR.bookmaker_stamps(payload) == (0, 5)
    payload["bookmakers"][0]["last_update"] = "2026-03-12T12:50:00Z"
    assert PR.bookmaker_stamps(payload) == (1, 5)
    assert PR.bookmaker_stamps({"bookmakers": "x"}) == (0, 0)


def test_names_match_inside_the_games_two_teams():
    _, frame = _phi_det()
    roster = pd.DataFrame([
        ("Cade Cunningham", 1, "DET"), ("Jalen Duren", 2, "DET"), ("Tyrese Maxey", 3, "PHI"),
        ("V.J. Edgecombe", 4, "PHI"), ("Daniss Jenkins", 5, "DET"), ("Tobias Harris", 6, "DET"),
        ("Duncan Robinson", 7, "MIA")], columns=["name", "person_id", "team"])
    idx, team_of = PR.roster_index(roster, aliases={})
    got = PR.name_matches(frame, idx, team_of, "DET", "PHI")
    assert got["draftkings"] == (5, 1, ["Quentin Grimes"])  # Robinson is on another team
    assert got["prizepicks"] == (5, 0, ["Quentin Grimes"])   # not on the roster at all
    assert got["underdog"] == (3, 0, [])
    assert list(got) == ["draftkings", "fanduel", "betmgm", "prizepicks", "underdog"]


def test_alternates_and_multipliers_are_summed_per_book():
    alt = {"bookmakers": [{"key": "prizepicks", "markets": [{"key": "player_points_alternate", "outcomes": [
        {"name": "Over", "description": "Cade Cunningham", "price": 100, "point": 29.5},
        {"name": "Over", "description": "Cade Cunningham", "price": -137, "point": 21.5},
        {"name": "Over", "description": "Cade Cunningham", "price": 100.0, "point": 32.5},
        {"name": "Over", "description": "Jalen Duren", "price": 100, "point": 16.5},
        {"name": "Over", "description": "Jalen Duren", "price": 100, "point": True}]}]},
        {"key": "underdog", "markets": [{"key": "player_points", "outcomes": []}]}]}
    assert PR.alt_summary(alt) == {"prizepicks": {"outcomes": 5, "plus100": 4, "players": 2, "most": 3,
                                                  "median": "2"}}
    mult = {"bookmakers": [{"key": "prizepicks", "markets": [{"key": "player_points", "outcomes": [
        {"name": "Over", "multiplier": None}, {"name": "Under", "multiplier": 1.5},
        {"name": "Over"}, {"name": "Under", "multiplier": "x2"}]}]}]}
    assert PR.multiplier_summary(mult) == {"prizepicks": Counter({"null": 1, "1.5": 1, "absent": 1,
                                                                  "(other)": 1})}


def test_stamp_moves_tells_a_moved_line_from_a_new_stamp():
    payload, before = _phi_det()
    later = json.loads(json.dumps(payload))
    for bm in later["bookmakers"]:
        for mk in bm["markets"]:
            mk["last_update"] = "2026-03-12T13:05:00Z"
    dk = later["bookmakers"][0]["markets"][0]["outcomes"]
    for oc in dk:
        if oc["description"] == "Jalen Duren":
            oc["point"] = 14.5
    after = O.parse_event_odds(later, books=O.BOOKS, markets=PR.POINTS, fetched_at=AT)
    moves = PR.stamp_moves(before, after)
    assert moves["draftkings"] == (7, 6, 6, 1)
    assert moves["prizepicks"] == (6, 6, 6, 0)
    assert PR.stamp_moves(before, before)["draftkings"] == (7, 7, 0, 0)


def test_an_error_body_shows_its_field_names_never_their_values():
    body = {"message": "Invalid market " + KEY, "error_code": "INVALID_MARKET", "details_url": "x",
            "<b>bad key</b>": 1}
    assert PR.field_names(body) == "bbadkeyb, details_url, error_code, message"
    assert PR.field_names([1]) == "(no JSON object)" and PR.field_names({}) == "(none)"


def test_number_text():
    assert [PR.number_text(v) for v in (-137, -137.0, 100, 1.5, None, True, "1", float("nan"))] == \
        ["-137", "-137", "100", "1.5", "null", "(other)", "(other)", "(other)"]


# -- Verdicts -------------------------------------------------------------------
def test_a_missing_region_is_a_fail_that_points_to_step_6():
    """m5: the schema marks regions required; a bookmakers-only call that the
    vendor refuses is the one finding that flips oddsapi.SEND_REGIONS."""
    level, text = PR.odds_verdict(_ans(422, last=0, code="MISSING_REGION"), O.empty_feed())
    assert (level, text) == ("FAIL", "the vendor requires regions; see step 6 before setting "
                                     "oddsapi.SEND_REGIONS")


def test_the_one_market_verdicts():
    payload, frame = _phi_det()
    assert PR.odds_verdict(_ans(200, last=1), frame) == ("PASS", "25 lines for 1 credit [3a]")
    assert PR.odds_verdict(_ans(200, last=2), frame)[0] == "FAIL"
    assert PR.odds_verdict(_ans(200, last=None), frame) == ("FAIL", "no x-requests-last on a 200 [5a]")
    assert PR.odds_verdict(_ans(200, last=0), frame)[0] == "INFO"
    assert PR.odds_verdict(_ans(500, last=None), frame)[0] == "FAIL"
    empty = O.parse_event_odds({**payload, "bookmakers": []}, books=O.BOOKS, markets=PR.POINTS,
                               fetched_at=AT)
    assert PR.odds_verdict(_ans(200, last=0), empty)[0] == "INFO"          # nothing posted, free
    assert PR.odds_verdict(_ans(200, last=1), empty)[0] == "FAIL"          # billed for nothing
    junk = O.parse_event_odds({**payload, "bookmakers": [{"key": "draftkings", "markets": [
        {"key": "player_points", "outcomes": [{"name": "Over", "price": "x"}]}]}]},
        books=O.BOOKS, markets=PR.POINTS, fetched_at=AT)
    level, text = PR.odds_verdict(_ans(200, last=1), junk)
    assert level == "FAIL" and "a shape it does not read" in text
    unknown = O.parse_event_odds({**payload, "home_team": "Seattle SuperSonics"}, books=O.BOOKS,
                                 markets=PR.POINTS, fetched_at=AT)
    assert PR.odds_verdict(_ans(200, last=1), unknown)[0] == "FAIL"


def test_the_regions_unknown_book_and_commence_verdicts():
    payload, _ = _phi_det()
    assert PR.regions_verdict(_ans(200, last=1, body=payload)).startswith(
        "x-requests-last 1: bookmakers alone sets the bill, so SEND_REGIONS is safe")
    assert PR.regions_verdict(_ans(200, last=2, body=payload)).startswith("x-requests-last 2: regions")
    assert PR.regions_verdict(_ans(200, last=0, body={"bookmakers": []})).startswith("an empty answer")
    assert PR.unknown_book_verdict(_ans(422, last=0, code="INVALID_BOOKMAKERS")) == \
        "an unknown book key is refused: 422 INVALID_BOOKMAKERS [4]"
    assert "dropped without a word; books that answered: draftkings" in \
        PR.unknown_book_verdict(_ans(200, last=1, body={"bookmakers": [{"key": "draftkings"}]}))
    assert PR.commence_verdict(_ans(200, last=0, body={"bookmakers": []})).startswith(
        "commenceTimeFrom is honored")
    assert PR.commence_verdict(_ans(200, last=1, body=payload)).startswith("commenceTimeFrom is ignored")
    assert "not accepted" in PR.commence_verdict(_ans(422, last=0, code="INVALID_COMMENCE_TIME_FROM"))


def test_a_bad_keys_401_is_a_pass_only_when_the_site_would_read_it_as_one():
    ok = PR.bad_key_verdict(_ans(401, code="INVALID_KEY", kind="bad_key"))
    assert ok[0] == "PASS" and "401 INVALID_KEY" in ok[1]
    assert PR.bad_key_verdict(_ans(401, code="OUT_OF_USAGE_CREDITS", kind="out_of_credits"))[0] == "FAIL"
    assert PR.bad_key_verdict(_ans(403, code="HTTP_403", kind="server"))[0] == "FAIL"
    assert PR.bad_key_verdict(_ans(status=0, kind="timeout"))[0] == "FAIL"


def test_the_prizepicks_only_poll_is_a_pass_only_when_an_empty_answer_is_free():
    payload, _ = _phi_det()
    empty = O.parse_event_odds({**payload, "bookmakers": []}, books=("prizepicks",), markets=PR.POINTS,
                               fetched_at=AT)
    assert PR.topup_verdict(_ans(200, last=0), empty)[0] == "PASS"
    assert PR.topup_verdict(_ans(200, last=1), empty)[0] == "FAIL"
    assert PR.topup_verdict(_ans(200, last=None), empty)[0] == "FAIL"
    posted = O.parse_event_odds(payload, books=("prizepicks",), markets=PR.POINTS, fetched_at=AT)
    level, text = PR.topup_verdict(_ans(200, last=1), posted)
    assert level == "INFO" and "6 lines" in text
    assert PR.topup_verdict(_ans(500), empty)[0] == "INFO"


# -- The probe's own source -----------------------------------------------------
SRC = PROBE.read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def test_the_probe_never_names_the_event_markets_endpoint():
    """It always bills 1 credit (docs 1724); the probe has no path to it."""
    assert "/markets" not in SRC
    assert not PR._ALLOWED_PATH.fullmatch(O.EVENTS_PATH + "/" + "d" * 32 + "/markets")


def test_the_probe_writes_no_file():
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
             for n in ast.walk(TREE) if isinstance(n, ast.Call)}
    for banned in ("open", "write_text", "write_bytes", "to_parquet", "to_csv", "to_json", "to_pickle",
                   "mkdir", "makedirs", "touch", "unlink", "remove", "rename", "rmtree", "FileHandler",
                   "basicConfig", "dump"):
        assert banned not in calls, banned
    mods = {a.name.split(".")[0] for n in ast.walk(TREE) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(TREE) if isinstance(n, ast.ImportFrom) and n.module}
    assert not mods & {"shutil", "tempfile", "pickle", "sqlite3", "logging"}
    # A probe run writes no bytecode either, and says so before bucketlib is imported.
    flag = SRC.index("sys.dont_write_bytecode = True")
    assert SRC.index('if __name__ == "__main__":') < flag < SRC.index("from bucketlib import")


def test_the_probe_reads_no_exception_text():
    """In every `except ... as e`, the exception is read only as type(e) or
    VendorTransportError's fixed `kind`: never str(e), repr(e), e.args or an
    f-string of it."""
    handlers = [n for n in ast.walk(TREE) if isinstance(n, ast.ExceptHandler) and n.name]
    assert handlers
    for h in handlers:
        parents = {}
        for node in ast.walk(h):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node
        for node in ast.walk(h):
            if isinstance(node, ast.Name) and node.id == h.name:
                up = parents.get(id(node))
                fine = (isinstance(up, ast.Call) and isinstance(up.func, ast.Name)
                        and up.func.id == "type") or (isinstance(up, ast.Attribute) and up.attr == "kind")
                assert fine, f"line {node.lineno} reads the exception {h.name}"


def test_the_probe_never_prints_but_through_the_report():
    printed = [n.lineno for n in ast.walk(TREE) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == "print"]
    assert printed == []
    assert re.search(r"def __init__\(self, secrets: tuple = \(\), write=print\)", SRC)


def test_the_probe_ships_ascii_only():
    assert all(ord(ch) < 128 for ch in SRC)
