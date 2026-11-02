"""Failure classification: each case below looked like 'no data' somewhere."""
import subprocess
import sys
from pathlib import Path

import pytest

from bucketlib import http as H

REPO = Path(__file__).resolve().parent.parent

AKAMAI = b"<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY>... errors.edgesuite.net/18.d6</BODY></HTML>"
S3 = b"<?xml version='1.0'?><Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"


def test_akamai_403_is_blocked_not_absent():
    assert isinstance(H.classify("u", 403, "text/html", AKAMAI), H.Blocked)


def test_s3_403_is_absent():
    assert isinstance(H.classify("u", 403, "application/xml", S3), H.Absent)


def test_400_is_our_bug():
    assert isinstance(H.classify("u", 400, "application/json", b'{"OpponentTeamID":["The value \'None\' is not valid"]}'), H.BadRequest)


def test_5xx_and_429_are_blocked():
    for s in (429, 500, 503):
        assert isinstance(H.classify("u", s, "", b""), H.Blocked)


def test_200_is_fine():
    assert H.classify("u", 200, "application/json", b"{}") is None


def test_none_params_refused_before_sending():
    # curl_cffi would send the literal string "None" and get an HTTP 400.
    with pytest.raises(H.BadRequest):
        H.fetch("https://stats.nba.com/stats/x", params={"OpponentTeamID": None})


def test_pdf_expectation_rejects_xml_body():
    with pytest.raises(H.BadPayload):
        H._parse("u", S3, "application/xml", "pdf")


def test_measure_type_must_be_explicit():
    from nba_api.stats.endpoints import playergamelogs
    with pytest.raises(ValueError):
        H.stats_params(playergamelogs.PlayerGameLogs, season_nullable="2024-25")
    p = H.stats_params(playergamelogs.PlayerGameLogs, season_nullable="2024-25",
                       measure_type_player_game_logs_nullable="Base")
    assert p["OpponentTeamID"] == "" and p["MeasureType"] == "Base"


def test_result_sets_keeps_columns_on_empty():
    rs = H.result_sets({"resultSets": [{"name": "X", "headers": ["A", "B"], "rowSet": []}]})
    assert list(rs["X"].columns) == ["A", "B"] and rs["X"].empty


@pytest.mark.skipif(sys.platform != "darwin", reason="the CoreFoundation load is macOS only")
def test_a_fresh_interpreter_imports_the_client():
    """macOS 27: curl_cffi's extension needs CoreFoundation loaded first, and
    nothing before bucketlib.http did that ("symbol not found in flat
    namespace '_CFRelease'" unless pandas happened to be imported first).
    A fresh interpreter that imports nothing else must import the client."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import bucketlib.http; print('ok')"
    r = subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, text=True,
                       timeout=180, cwd=str(REPO))
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-800:]
