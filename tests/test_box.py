"""Participation (who played / sat / was inactive / started). Every fixture is a
trimmed real payload saved 2026-10-08; each test pins a trap seen live."""
import copy
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bucketlib import cache, http
from bucketlib.sources import box as B
from bucketlib.sources import statsnba as S

FX = Path(__file__).resolve().parent / "fixtures" / "box"
REPO = Path(__file__).resolve().parent.parent


def fx(name):
    return json.loads((FX / name).read_text())


def cdn(gid):
    return fx(f"cdn_{gid}.json")


def v3(gid):
    return fx(f"v3box_{gid}.json"), fx(f"v3sum_{gid}.json")


def _players(payload, side):
    return payload["game"][side]["players"]


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point every cache path at a temp dir: no test touches the real cache.
    statsnba binds CACHE_DIR at import, so it is patched separately."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(S, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(S, "_fetched_this_run", set())
    return tmp_path


def on_date(monkeypatch, iso: str) -> None:
    """Freeze the ET game date the season state is judged from."""
    monkeypatch.setattr(B, "today_et", lambda: dt.date.fromisoformat(iso))


def no_stats(monkeypatch) -> None:
    """Any stats.nba.com call fails the test (it hangs from GitHub Actions)."""
    def boom(*a, **k):
        pytest.fail("stats.nba.com was called")
    monkeypatch.setattr(http, "stats_call", boom)
    monkeypatch.setattr(http, "stats_get", boom)


def stamp_of(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp()


def write_stamped(path: Path, payload, fetched_at: float) -> None:
    """A cached envelope with a chosen fetch stamp (cache.write_json stamps now)."""
    cache.write_json(path, {"fetched_at": fetched_at, "content_hash": "x", "data": payload}, gz=True)


# ── Game ids ─────────────────────────────────────────────────────────────────
def test_game_id_season_type_and_source():
    assert B.game_season("0021400500") == "2014-15"
    assert B.game_season("0042500405") == "2025-26"
    assert B.season_type_of("0051900111") == "PlayIn"
    assert B.season_type_of("0062300001") == "NBA Cup Final"
    assert B.default_source("0021800600") == "v3"
    assert B.default_source("0021900001") == "cdn"
    for bad in ("21400500", "0091400500", "1021400500", "0021400500x"):
        with pytest.raises(ValueError):
            B.game_season(bad)


def test_minute_parsers():
    assert B.iso_seconds("PT25M01.00S") == 1501
    assert B.iso_seconds("PT00M01.00S") == 1
    assert B.mmss_seconds("35:20") == 2120
    assert B.mmss_seconds("0:06") == 6
    for bad in ("", "25:01", "PT25M"):
        with pytest.raises(ValueError):
            B.iso_seconds(bad)
    for bad in ("", "PT25M01.00S", "35"):
        with pytest.raises(ValueError):
            B.mmss_seconds(bad)


def test_comment_prefixes():
    assert B.parse_comment("DNP - Coach's Decision") == ("DNP", "Coach's Decision")
    assert B.parse_comment("DND - Injury/Illness") == ("DND", "Injury/Illness")
    assert B.parse_comment("DND - Left Lower Leg Contusion") == ("DND", "Left Lower Leg Contusion")
    assert B.parse_comment("NWT - Strained Left Shoulder") == ("NWT", "Strained Left Shoulder")
    assert B.parse_comment("") == ("", "")
    # Some 2019-20 V3 payloads carry the cdn code instead of 'NWT - ...'.
    assert B.parse_comment("NWT_LEAGUE_SUSPENSION") == ("NWT", "LEAGUE_SUSPENSION")
    # A new prefix must be added on purpose, never filed as a reasonless DNP.
    for bad in ("Suspended - League", "DNPX - Rest", "NWTLEAGUE"):
        with pytest.raises(ValueError):
            B.parse_comment(bad)


def test_v3_game_with_cdn_style_comment_parses():
    # 0021900723: the Grizzlies' Jaren Jackson Jr. and Marko Guduric, suspended
    # for leaving the bench. The first full 2019-20 run failed this game.
    df, meta = B.parse_v3(*v3("0021900723"), "0021900723")
    sus = df[df.reason_code == "NWT"].set_index("player_name")
    assert set(sus.index) == {"Jaren Jackson Jr.", "Marko Guduric"}
    assert (sus.status == "dnp").all() and (sus.reason_text == "LEAGUE_SUSPENSION").all()
    assert (sus.absence_kind == "unavailable").all() and (sus.team == "MEM").all()


def test_unrecognised_v3_comment_names_the_game():
    box, summ = v3("0021900723")
    box["boxScoreTraditional"]["awayTeam"]["players"][-1]["comment"] = "SUSPENDED_LEAGUE"
    with pytest.raises(ValueError, match="0021900723: MEM .*unrecognised V3 comment"):
        B.parse_v3(box, summ, "0021900723")


# ── cdn ──────────────────────────────────────────────────────────────────────
def test_cdn_statuses_reasons_and_waived_row():
    df, meta = B.parse_cdn(cdn("0022300500"), "0022300500")
    assert list(df.columns) == B.COLUMNS
    assert df["person_id"].dtype == "int64" and df["played_seconds"].dtype == "int64"
    assert df.groupby(["team", "status"]).size().to_dict() == {
        ("NOP", "inactive"): 5, ("NOP", "played"): 12, ("SAC", "inactive"): 4, ("SAC", "played"): 14}
    assert (df.groupby("team")["starter"].sum() == 5).all()
    jta = df.set_index("player_name").loc["Juan Toscano-Anderson"]
    # Listed, but no longer on the roster: kept as inactive with its reason.
    assert (jta.status, jta.reason_code, jta.reason_text) == ("inactive", "INACTIVE_NOT_WITH_TEAM", "Waived")
    assert (df.loc[df.status == "played", "reason_code"] == "").all()
    assert meta == {**meta, "home": "SAC", "away": "NOP", "et_date": "2024-01-07", "periods": 4,
                    "ot": 0, "arena": "Golden 1 Center", "season_type": "Regular Season"}
    assert meta["tip_utc"] == pd.Timestamp("2024-01-07T23:00:00Z")


def test_cdn_2019_finalbox_missing_reasons_and_padded_names():
    p = cdn("0021900600")
    assert p["game"]["gameStatusText"] == "finalbox"   # status is read from the int only
    df, _ = B.parse_cdn(p, "0021900600")
    dnp = df[df.status == "dnp"]
    # 2019-20 coach's-decision DNPs carry no notPlayingReason: kept as ''.
    assert set(dnp.loc[dnp.reason_code == "", "player_name"]) == {
        "Marko Guduric", "Yuta Watanabe", "Chris Clemons", "Michael Frazier", "Thabo Sefolosha"}
    assert "Nene" in set(df.player_name)                  # payload says '  Nene'
    assert df.set_index("player_name").loc["Bruno Caboclo", "reason_code"] == "INACTIVE_COACH"


def test_cdn_finalbox_string_status_is_final():
    # 0021900718 (2019-20): gameStatus is the STRING 'finalbox' and there is
    # no regulationPeriods key, on a complete box (team minutes 240, points =
    # score). The first full 2019-20 run filed 89 such games as not final.
    p = cdn("0021900718")
    assert p["game"]["gameStatus"] == "finalbox" and "regulationPeriods" not in p["game"]
    df, meta = B.parse_cdn(p, "0021900718")
    assert (meta["home"], meta["away"], meta["periods"], meta["ot"]) == ("DEN", "UTA", 4, 0)
    assert (df.groupby("team")["starter"].sum() == 5).all()
    p["game"]["gameStatus"] = "live"                       # any other string is still not final
    with pytest.raises(B.NotFinal):
        B.parse_cdn(p, "0021900718")


def test_bubble_cdn_null_minutes_on_non_players_are_zero():
    # 0041900161 (2020 bubble playoffs): players who did not play carry
    # statistics.minutes = null instead of 'PT00M00.00S'.
    p = cdn("0041900161")
    nulls = {x["personId"] for s in ("homeTeam", "awayTeam") for x in _players(p, s)
             if x["statistics"]["minutes"] is None}
    assert len(nulls) == 13
    df, _ = B.parse_cdn(p, "0041900161")
    rows = df[df.person_id.isin(nulls)]
    assert (rows.status != "played").all() and (rows.played_seconds == 0).all()
    # A null on a player who DID play is still an error, and names the game.
    starter = _players(p, "homeTeam")[0]
    starter["statistics"]["minutes"] = None
    with pytest.raises(ValueError, match="0041900161: DEN .*unparseable ISO minutes None"):
        B.parse_cdn(p, "0041900161")


def test_cdn_double_overtime_team_seconds():
    df, meta = B.parse_cdn(cdn("0022500001"), "0022500001")
    assert (meta["periods"], meta["ot"]) == (6, 2)
    sums = df.groupby("team")["played_seconds"].sum()
    assert (sums.sub(B.REG_TEAM_SECONDS + 2 * B.OT_TEAM_SECONDS).abs() <= 1).all()


def _iso(secs: float) -> str:
    return f"PT{int(secs // 60):02d}M{secs % 60:05.2f}S"


def _shrink_bench_minutes(p, side, to_secs):
    """Cut one bench player's time to `to_secs` and give the rest to a starter,
    so the team total still adds up and only the cameo rule is under test."""
    bench = next(x for x in _players(p, side) if x["starter"] == "0" and x["played"] == "1")
    starter = next(x for x in _players(p, side) if x["starter"] == "1")
    moved = B.iso_seconds(bench["statistics"]["minutes"]) - to_secs
    bench["statistics"]["minutes"] = _iso(to_secs)
    starter["statistics"]["minutes"] = _iso(B.iso_seconds(starter["statistics"]["minutes"]) + moved)
    return bench["personId"]


def test_one_second_cameo_is_played_not_dnp():
    # 'PT00M01.00S' shows as MIN 0 in LeagueGameLog; it is still an appearance.
    p = cdn("0022300500")
    pid = _shrink_bench_minutes(p, "homeTeam", 1)
    row = B.parse_cdn(p, "0022300500")[0].set_index("person_id").loc[pid]
    assert (row.status, row.played_seconds) == ("played", 1)


def test_sub_second_bench_appearance_is_played_with_zero_seconds():
    p = cdn("0022300500")
    pid = _shrink_bench_minutes(p, "awayTeam", 0)
    row = B.parse_cdn(p, "0022300500")[0].set_index("person_id").loc[pid]
    assert (row.status, row.played_seconds, row.starter) == ("played", 0, False)


@pytest.mark.parametrize("mutate,needle", [
    (lambda p: _players(p, "homeTeam")[6].update(starter="1"), "6 starters"),
    (lambda p: _players(p, "homeTeam")[0]["statistics"].update(minutes="PT00M00.00S"), "starter"),
    (lambda p: _players(p, "homeTeam").pop(5), "player-seconds off"),
    (lambda p: _players(p, "awayTeam").append(copy.deepcopy(_players(p, "homeTeam")[8])), "listed twice"),
    (lambda p: _players(p, "homeTeam")[14].update(played="1"), "INACTIVE but played"),
    (lambda p: _players(p, "homeTeam")[14].update(status="SUSPENDED"), "status="),
    (lambda p: p["game"]["homeTeam"].update(teamTricode="XYZ"), "unknown or non-canonical"),
    (lambda p: p["game"]["homeTeam"].update(teamTricode="SA"), "unknown or non-canonical"),
    (lambda p: p["game"]["homeTeam"].update(teamId=1610612747), "team id 1610612747 is LAL"),
    (lambda p: _players(p, "homeTeam")[14].update(notPlayingReason="SUSPENDED_X"),
     "unrecognised cdn reason code"),
    (lambda p: p["game"]["awayTeam"].update(teamTricode="SAC", teamId=1610612758), "both SAC"),
    (lambda p: p["game"]["homeTeam"].update(players=[]), "no players"),
    (lambda p: p["game"].update(gameTimeUTC="2024-01-07T04:00:00Z"), "gameCode says"),
])
def test_cdn_invariants_raise_with_game_id(mutate, needle):
    p = cdn("0022300500")
    mutate(p)
    with pytest.raises(ValueError) as e:
        B.parse_cdn(p, "0022300500")
    assert needle in str(e.value) and "0022300500" in str(e.value)


def test_not_final_box_is_refused():
    p = cdn("0022300500")
    p["game"]["gameStatus"] = 2
    with pytest.raises(B.NotFinal):
        B.parse_cdn(p, "0022300500")


# ── V3 ───────────────────────────────────────────────────────────────────────
def test_v3_first_five_are_starters_even_when_bench_has_positions():
    box, summ = v3("0021400500")
    den = box["boxScoreTraditional"]["homeTeam"]["players"]
    assert sum(bool(p["position"]) for p in den) > 5      # bench tagged G/C/F in 2014-15
    df, meta = B.parse_v3(box, summ, "0021400500")
    starters = df[df.starter].groupby("team")["person_id"].apply(set).to_dict()
    assert starters["DEN"] == {p["personId"] for p in den[:5]}
    assert (meta["home"], meta["away"], meta["et_date"]) == ("DEN", "MEM", "2015-01-03")


def test_v3_first_five_rule_is_verified_against_starters_aggregate():
    box, summ = v3("0021400500")
    pl = box["boxScoreTraditional"]["homeTeam"]["players"]
    pl[4], pl[8] = pl[8], pl[4]                            # a bench player moved into the top 5
    with pytest.raises(ValueError, match="starters aggregate"):
        B.parse_v3(box, summ, "0021400500")


def test_v3_gameEt_is_eastern_with_a_fake_z():
    box, summ = v3("0021400500")
    s = summ["boxScoreSummary"]
    assert s["gameEt"] == "2015-01-03T21:00:00Z" and s["gameTimeUTC"] == "2015-01-04T02:00:00Z"
    _, meta = B.parse_v3(box, summ, "0021400500")
    assert meta["tip_utc"] == pd.Timestamp("2015-01-04T02:00:00Z")
    # A tip time in the wrong zone that lands on another ET date is caught.
    s["gameTimeUTC"] = "2015-01-03T03:00:00Z"
    with pytest.raises(ValueError, match="gameCode says"):
        B.parse_v3(box, summ, "0021400500")


def test_v3_dnp_comment_and_summary_inactives():
    df, _ = B.parse_v3(*v3("0021400500"), "0021400500")
    udrih = df.set_index("player_name").loc["Beno Udrih"]
    assert (udrih.status, udrih.reason_code, udrih.reason_text, udrih.played_seconds) == \
        ("dnp", "DND", "Left Lower Leg Contusion", 0)
    ina = df[df.status == "inactive"]
    assert set(ina.player_name) == {"JaVale McGee", "Randy Foye", "Danilo Gallinari", "Zach Randolph"}
    assert (ina.reason_code == "INACTIVE").all() and (ina.reason_text == "").all()
    assert df.set_index("player_name").loc["Randy Foye", "jersey"] == "4"   # payload ' 4'


def test_v3_double_overtime_periods_and_cameos():
    df, meta = B.parse_v3(*v3("0021400079"), "0021400079")
    assert (meta["periods"], meta["ot"]) == (6, 2)
    cam = df.set_index("player_name").loc[["Derrick Williams", "Ray McCallum"]]
    assert (cam.status == "played").all() and cam.played_seconds.tolist() == [6, 10]


def test_v3_played_row_with_dnp_comment_raises():
    box, summ = v3("0021400500")
    box["boxScoreTraditional"]["homeTeam"]["players"][9]["comment"] = "DNP - Coach's Decision"
    with pytest.raises(ValueError, match="but has comment"):
        B.parse_v3(box, summ, "0021400500")


def test_v3_inactive_also_listed_in_box_raises():
    box, summ = v3("0021400500")
    p = box["boxScoreTraditional"]["awayTeam"]["players"][7]
    summ["boxScoreSummary"]["awayTeam"]["inactives"].append(
        {"personId": p["personId"], "firstName": p["firstName"], "familyName": p["familyName"],
         "jerseyNum": ""})
    with pytest.raises(ValueError, match="listed twice"):
        B.parse_v3(box, summ, "0021400500")


def test_v3_missing_inactives_key_is_not_zero_inactives():
    box, summ = v3("0021400500")
    del summ["boxScoreSummary"]["homeTeam"]["inactives"]
    with pytest.raises(ValueError, match="no inactives key"):
        B.parse_v3(box, summ, "0021400500")


def test_v3_box_and_summary_must_agree_on_sides():
    box, summ = v3("0021400500")
    s = summ["boxScoreSummary"]
    s["homeTeam"], s["awayTeam"] = s["awayTeam"], s["homeTeam"]
    with pytest.raises(ValueError, match="summary says"):
        B.parse_v3(box, summ, "0021400500")


# ── Why a player did not play (status 'dnp' alone mixes these) ──────────────
def test_absence_kind_separates_coach_decisions_from_unavailable_players():
    df, _ = B.parse_cdn(cdn("0022500001"), "0022500001")
    dnp = df[df.status == "dnp"]
    # Two players DRESSED OUT injured (DND_INJURY) sit in 'dnp' next to five
    # healthy scratches; only absence_kind tells them apart.
    assert dict(zip(dnp.reason_code, dnp.absence_kind)) == {"": "coach", "DND_INJURY": "unavailable"}
    assert (dnp.reason_code == "DND_INJURY").sum() == 2 and (dnp.absence_kind == "coach").sum() == 5
    assert (df.loc[df.status == "played", "absence_kind"] == "").all()
    v, _ = B.parse_v3(*v3("0021400500"), "0021400500")
    k = v.set_index("player_name")["absence_kind"]
    assert k["Beno Udrih"] == "unavailable"                 # 'DND - Left Lower Leg Contusion'
    assert (v.loc[v.status == "inactive", "absence_kind"] == "unknown").all()
    w, _ = B.parse_v3(*v3("0021400079"), "0021400079")
    assert set(w.loc[w.status == "dnp", "absence_kind"]) == {"coach"}   # 'DNP - Coach's Decision'


@pytest.mark.parametrize("status,code,text,source,kind", [
    ("played", "", "", "cdn", ""),
    ("dnp", "", "", "cdn", "coach"),
    ("inactive", "", "", "cdn", "unknown"),
    ("dnp", "DNP_COACH", "", "cdn", "coach"),
    ("inactive", "INACTIVE_COACH", "", "cdn", "coach"),
    ("inactive", "INACTIVE_COACH", "Coach's Decision", "cdn", "coach"),
    # 2019-20 files injuries under INACTIVE_COACH: the description wins.
    ("inactive", "INACTIVE_COACH", "Left Knee; Soreness", "cdn", "unavailable"),
    ("dnp", "DND_INJURY", "Left Adductor Strain", "cdn", "unavailable"),
    ("dnp", "DNP_REST", "", "cdn", "unavailable"),
    ("inactive", "INACTIVE_GLEAGUE_TWOWAY", "", "cdn", "unavailable"),
    ("dnp", "DNP", "Coach's Decision", "v3", "coach"),
    ("dnp", "DND", "Coach's Decision", "v3", "coach"),
    ("dnp", "DNP", "Injury/Illness", "v3", "unavailable"),
    ("dnp", "NWT", "LEAGUE_SUSPENSION", "v3", "unavailable"),
    ("dnp", "", "", "v3", "unknown"),
    ("inactive", "INACTIVE", "", "v3", "unknown"),
])
def test_absence_kind_mapping(status, code, text, source, kind):
    assert B.absence_kind(status, code, text, source) == kind


@pytest.mark.parametrize("args", [
    ("dnp", "SUSPENDED_LEAGUE", "", "cdn"), ("dnp", "DNP", "", "cdn"),
    ("dnp", "XYZ", "", "v3"), ("benched", "", "", "cdn"), ("dnp", "", "", "espn"),
])
def test_absence_kind_refuses_codes_it_has_not_classified(args):
    with pytest.raises(ValueError):
        B.absence_kind(*args)


def _first(df, mask):
    return df.index[mask][0]


@pytest.mark.parametrize("mutate,needle", [
    # 5 seconds on a DNP row keeps the team total within tolerance: only the
    # non-player rule can see it.
    (lambda d: d.__setitem__("played_seconds", d.played_seconds.mask(
        d.index == _first(d, d.status == "dnp"), 5)), "non-player has seconds"),
    (lambda d: d.__setitem__("reason_code", d.reason_code.mask(
        d.index == _first(d, d.status == "played"), "DNP")), "carries a reason code"),
    (lambda d: d.__setitem__("absence_kind", d.absence_kind.mask(
        d.index == _first(d, d.status == "played"), "coach")), "absence_kind"),
    (lambda d: d.__setitem__("absence_kind", d.absence_kind.mask(
        d.index == _first(d, d.status == "dnp"), "")), "absence_kind"),
])
def test_participation_row_invariants(mutate, needle):
    df, meta = B.parse_cdn(cdn("0022500001"), "0022500001")
    mutate(df)
    with pytest.raises(ValueError, match=needle):
        B.check_participation(df, ot=meta["ot"])
    with pytest.raises(ValueError, match=needle):
        B.check_table(df, pd.DataFrame([meta], columns=B.META_COLUMNS))


def _two_games():
    parts, metas = zip(*(B.parse_cdn(cdn(g), g) for g in ("0022500001", "0022500147")))
    return pd.concat(parts, ignore_index=True), pd.DataFrame(list(metas), columns=B.META_COLUMNS)


@pytest.mark.parametrize("mutate,needle", [
    # A duplicated DNP row (0 seconds, no start) breaks nothing but the key.
    (lambda p, g: (pd.concat([p, p[p.status == "dnp"].head(1)], ignore_index=True), g), "not unique"),
    (lambda p, g: (p, pd.concat([g, g.head(1)], ignore_index=True)), "more than once"),
    (lambda p, g: (p[p.game_id != "0022500147"], g), "cover different games"),
    (lambda p, g: (p.assign(is_home=~p.is_home), g), "team / is_home"),
    (lambda p, g: (p.assign(source=p.source.mask(p.game_id == "0022500147", "v3")), g), "row source"),
    (lambda p, g: (p.drop(columns="absence_kind"), g), "older schema"),
])
def test_check_table_catches_what_per_game_checks_cannot(mutate, needle):
    part, games = _two_games()
    B.check_table(part, games, "2025-26")
    part, games = mutate(part.copy(), games.copy())
    with pytest.raises(ValueError, match=needle):
        B.check_table(part, games, "2025-26")


# ── cdn vs V3 on the same game (overlap era) ─────────────────────────────────
def test_cdn_and_v3_agree_on_an_overlap_game():
    a, ma = B.parse_cdn(cdn("0022300061"), "0022300061")
    b, mb = B.parse_v3(*v3("0022300061"), "0022300061")
    st = B.crosscheck_starters(a, b)
    assert len(st) == 2 and st.only_a.str.len().sum() == 0 and st.only_b.str.len().sum() == 0
    m = a.merge(b, on="person_id", suffixes=("_c", "_v"))
    assert len(m) == len(a) == len(b)
    assert (m.status_c == m.status_v).all()
    # Every row on the same side in both sources, inactives included.
    assert (m.team_c == m.team_v).all() and (m.is_home_c == m.is_home_v).all()
    # cdn ACTIVE + no reason is the coach's decision: V3 says so in words.
    dnp = m[m.status_c == "dnp"]
    assert len(dnp) == 3 and (dnp.reason_code_c == "").all()
    assert (dnp.reason_text_v == "Coach's Decision").all() and (dnp.absence_kind_c == "coach").all()
    # Sources round the clock differently: seconds agree to within one.
    assert (m.played_seconds_c - m.played_seconds_v).abs().max() <= 1
    # Names are source-dependent (cdn 'Nikola Jokic', V3 'Nikola Jokić'): join on ids.
    assert {"Nikola Jokic", "Nikola Jokić"} == set(m.loc[m.person_id == 203999, ["player_name_c", "player_name_v"]].iloc[0])
    assert {k: ma[k] for k in ("home", "away", "et_date", "periods")} == \
        {k: mb[k] for k in ("home", "away", "et_date", "periods")}


# ── LeagueGameLog cross-checks ───────────────────────────────────────────────
def test_played_set_equals_lgl_appearances_including_zero_minute_rows():
    df, _ = B.parse_v3(*v3("0021400079"), "0021400079")
    lgl = B.lgl_appearances(fx("lglP_2014-11-07.json"))
    assert set(lgl.loc[lgl.lgl_min == 0, "person_id"]) == set(
        df.loc[df.player_name.isin(["Derrick Williams", "Ray McCallum"]), "person_id"])
    res = B.crosscheck_appearances(df, lgl).iloc[0]
    assert (res.only_box, res.only_lgl, res.team_mismatch) == ([], [], [])
    # "played = minutes rounded > 0" would drop the cameo rows: the check sees it.
    wrong = df.copy()
    wrong.loc[wrong.played_seconds < 30, "status"] = "dnp"
    res = B.crosscheck_appearances(wrong, lgl).iloc[0]
    assert len(res.only_lgl) == 2
    # A player on the wrong side is a team mismatch, not agreement.
    pid = int(df.loc[df.status == "played", "person_id"].iloc[0])
    other = sorted(set(df.team) - {df.set_index("person_id").loc[pid, "team"]})[0]
    moved = lgl.copy()
    moved.loc[moved.person_id == pid, "team"] = other
    res = B.crosscheck_appearances(df, moved).iloc[0]
    assert (res.only_box, res.only_lgl, res.team_mismatch) == ([], [], [pid])


def test_parse_lgl_games_neutral_site_rows():
    g = B.parse_lgl_games(fx("lglT_2025-26_rs.json"), "2025-26", "Regular Season").set_index("game_id")
    # Mexico City: LGL prints 'DAL @ DET' AND 'DET @ DAL'.
    assert tuple(g.loc["0022500147", ["team_a", "team_b", "home", "away"]]) == ("DAL", "DET", "", "")
    assert tuple(g.loc["0022500001", ["home", "away", "game_date"]]) == ("OKC", "HOU", "2025-10-21")
    _, meta = B.parse_cdn(cdn("0022500147"), "0022500147")
    assert (meta["home"], meta["arena_city"]) == ("DET", "Mexico City")
    B.check_meta_against_lgl(meta, g.loc["0022500147"])          # neutral: teams + date only


def test_check_meta_against_lgl_catches_mismatch():
    g = B.parse_lgl_games(fx("lglT_2025-26_rs.json"), "2025-26", "Regular Season").set_index("game_id")
    _, meta = B.parse_cdn(cdn("0022500001"), "0022500001")
    B.check_meta_against_lgl(meta, g.loc["0022500001"])
    with pytest.raises(ValueError, match="game list home"):
        B.check_meta_against_lgl({**meta, "home": "HOU", "away": "OKC"}, g.loc["0022500001"])
    with pytest.raises(ValueError, match="game list says"):
        B.check_meta_against_lgl({**meta, "et_date": "2025-10-22"}, g.loc["0022500001"])


@pytest.mark.parametrize("mutate,needle", [
    (lambda rows, h: rows.append(list(rows[0])), "3 team rows"),
    (lambda rows, h: rows[0].__setitem__(h.index("TEAM_ABBREVIATION"), "XYZ"), "unknown team"),
    (lambda rows, h: rows[0].__setitem__(h.index("GAME_DATE"), "2025-10-22"), "disagree on the date"),
    (lambda rows, h: rows[0].__setitem__(h.index("MATCHUP"), "OKC v HOU"), "unparseable MATCHUP"),
])
def test_parse_lgl_games_invariants(mutate, needle):
    p = fx("lglT_2025-26_rs.json")
    rs = p["resultSets"][0]
    gi = rs["headers"].index("GAME_ID")
    rows = [r for r in rs["rowSet"] if r[gi] == "0022500001"]
    mutate(rows, rs["headers"])
    rs["rowSet"] = rows
    with pytest.raises(ValueError, match=needle):
        B.parse_lgl_games(p, "2025-26", "Regular Season")


def test_parse_lgl_games_rejects_wrong_season_type():
    with pytest.raises(ValueError, match="id says"):
        B.parse_lgl_games(fx("lglT_2025-26_rs.json"), "2025-26", "Playoffs")


# ── Fetching, caching, failure classes ───────────────────────────────────────
def test_fetch_cdn_caches_final_only_and_reuses(tmp_cache, monkeypatch):
    calls = []
    payload = cdn("0022300500")

    def fake(url, **kw):
        calls.append(url)
        return copy.deepcopy(payload)

    monkeypatch.setattr(http, "fetch", fake)
    B.fetch_cdn("0022300500")
    B.fetch_cdn("0022300500")
    assert len(calls) == 1 and B.raw_path("0022300500", "cdn").exists()
    assert B.raw_path("0022300500", "cdn").is_relative_to(tmp_cache)
    payload["game"]["gameStatus"] = 2
    payload["game"]["gameId"] = "0022300501"
    with pytest.raises(B.NotFinal):
        B.fetch_cdn("0022300501")
    assert not B.raw_path("0022300501", "cdn").exists()       # a partial box is never cached


def test_cdn_absent_falls_back_to_v3_and_is_remembered(tmp_cache, monkeypatch):
    calls = []

    def absent(url, **kw):
        calls.append(url)
        raise http.Absent(url, "HTTP 403 S3 AccessDenied (object does not exist)")

    monkeypatch.setattr(http, "fetch", absent)
    monkeypatch.setattr(B, "fetch_v3", lambda gid, refresh=False: v3(gid))
    df, meta = B.load_game("0051900111")                   # the real 2019-20 CDN hole
    assert meta["source"] == "v3" and set(df.source) == {"v3"}
    B.load_game("0051900111")
    assert len(calls) == 1                                 # the marker skips the re-probe
    with pytest.raises(http.Absent):
        B.load_game("0051900111", source="cdn")


def test_blocked_is_never_treated_as_absent(tmp_cache, monkeypatch):
    def blocked(url, **kw):
        raise http.Blocked(url, "HTTP 403 text/html: Akamai edge block")

    monkeypatch.setattr(http, "fetch", blocked)
    monkeypatch.setattr(B, "fetch_v3", lambda *a, **k: pytest.fail("fell back on a block"))
    with pytest.raises(http.Blocked):
        B.load_game("0022300500")
    assert not B.raw_path("0022300500", "cdnabsent").exists()


class _Resp:
    def __init__(self, status, ctype, body):
        self.status_code, self.headers, self.content = status, {"content-type": ctype}, body


@pytest.mark.parametrize("ctype,body,exc,marker", [
    ("application/xml", b"<?xml version='1.0'?><Error><Code>AccessDenied</Code>"
                        b"<Message>Access Denied</Message></Error>", http.Absent, True),
    ("text/html", b"<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD>"
                  b"<BODY>errors.edgesuite.net</BODY></HTML>", http.Blocked, False),
])
def test_cdn_403_kinds_through_fetch_cdn(tmp_cache, monkeypatch, ctype, body, exc, marker):
    """Through the real http.fetch: the S3 XML 403 (no such object) is
    remembered as a CDN hole; the Akamai HTML 403 is retried, raised as
    Blocked, and never recorded as absence."""
    calls = []

    class Sess:
        def get(self, url, **kw):
            calls.append(url)
            return _Resp(403, ctype, body)

    monkeypatch.setattr(http, "_session", lambda: Sess())
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    with pytest.raises(exc):
        B.fetch_cdn("0021800600")
    assert B.raw_path("0021800600", "cdnabsent").exists() is marker
    assert not B.raw_path("0021800600", "cdn").exists()
    assert len(calls) == (1 if marker else 3)          # Absent is final, Blocked is retried


def _fake_stats(monkeypatch, by_endpoint: dict, calls: list):
    def fake(cls, **kw):
        calls.append((cls.endpoint, kw.get("game_id")))
        return copy.deepcopy(by_endpoint[cls.endpoint])
    monkeypatch.setattr(http, "stats_call", fake)


def test_fetch_v3_caches_final_payloads_once(tmp_cache, monkeypatch):
    box, summ = v3("0021400500")
    calls = []
    _fake_stats(monkeypatch, {"boxscoresummaryv3": summ, "boxscoretraditionalv3": box}, calls)
    b, s = B.fetch_v3("0021400500")
    assert [c[0] for c in calls] == ["boxscoresummaryv3", "boxscoretraditionalv3"]
    assert B.raw_path("0021400500", "v3sum").exists() and B.raw_path("0021400500", "v3box").exists()
    B.fetch_v3("0021400500")
    assert len(calls) == 2                                   # second call is all cache


def test_fetch_v3_refuses_a_non_final_game_and_caches_nothing(tmp_cache, monkeypatch):
    box, summ = v3("0021400500")
    summ["boxScoreSummary"]["gameStatus"] = 2
    calls = []
    _fake_stats(monkeypatch, {"boxscoresummaryv3": summ, "boxscoretraditionalv3": box}, calls)
    with pytest.raises(B.NotFinal):
        B.fetch_v3("0021400500")
    assert calls == [("boxscoresummaryv3", "0021400500")]  # the box is never asked for
    assert not B.raw_path("0021400500", "v3sum").exists()
    assert not B.raw_path("0021400500", "v3box").exists()


def test_fetch_v3_box_for_another_game_is_bad_payload(tmp_cache, monkeypatch):
    box, summ = v3("0021400500")
    box["boxScoreTraditional"]["gameId"] = "0021400501"
    _fake_stats(monkeypatch, {"boxscoresummaryv3": summ, "boxscoretraditionalv3": box}, [])
    with pytest.raises(http.BadPayload):
        B.fetch_v3("0021400500")
    assert not B.raw_path("0021400500", "v3box").exists()


def test_v3_unknown_game_is_absent_not_bad_payload(tmp_cache, monkeypatch):
    # Live 2026-10-09: an id V3 does not know is HTTP 200 with an all-null
    # summary. As BadPayload it used to end a --keep-going run with exit 3.
    null = fx("v3sum_0021601300_null.json")
    assert null["boxScoreSummary"]["gameId"] is None and null["boxScoreSummary"]["gameStatus"] == 0
    calls = []
    _fake_stats(monkeypatch, {"boxscoresummaryv3": null, "boxscoretraditionalv3": {}}, calls)
    with pytest.raises(http.Absent, match="no such game"):
        B.fetch_v3("0021601300")
    assert len(calls) == 1 and not B.raw_path("0021601300", "v3sum").exists()
    # Through the CDN-absent fallback the game is Absent from every source.
    def s3_absent(url, **kw):
        raise http.Absent(url, "HTTP 403 S3 AccessDenied (object does not exist)")

    monkeypatch.setattr(http, "fetch", s3_absent)
    with pytest.raises(http.Absent):
        B.load_game("0021901300")


def test_cup_final_box_parses_and_matches_its_schedule_row():
    df, meta = B.parse_cdn(cdn("0062500001"), "0062500001")
    assert (meta["season_type"], meta["home"], meta["away"], meta["et_date"], meta["arena_city"]) == \
        ("NBA Cup Final", "NYK", "SAS", "2025-12-16", "Las Vegas")
    sched = B.schedule_final_games(S.parse_schedule(fx("sched_2025-26_trim.json"), "2025-26"), "2025-26")
    row = sched.set_index("game_id").loc["0062500001"]
    assert (row.season_type, row.home, row.away) == ("NBA Cup Final", "", "")   # neutral site
    B.check_meta_against_lgl(meta, row)


# ── The game list: live season, past season, provisional payloads ─────────────
def _cdn_schedule_as_of(today: str) -> dict:
    """The trimmed real 2025-26 schedule, as the cdn would serve it on `today`:
    games after that date are not final yet."""
    p = fx("sched_2025-26_trim.json")
    for gd in p["leagueSchedule"]["gameDates"]:
        d = dt.datetime.strptime(gd["gameDate"], "%m/%d/%Y %H:%M:%S").date()
        for g in gd["games"]:
            if d.isoformat() > today:
                g["gameStatus"], g["gameStatusText"] = 1, "7:00 pm ET"
    return p


def test_live_season_list_comes_from_the_cdn_schedule_with_no_stats_call(tmp_cache, monkeypatch):
    on_date(monkeypatch, "2026-01-23")                     # 2025-26 is live
    no_stats(monkeypatch)
    fetched = []

    def fake_fetch(url, **kw):
        assert url == B.CDN_SCHEDULE_URL
        fetched.append(url)
        return _cdn_schedule_as_of("2026-01-23")

    monkeypatch.setattr(http, "fetch", fake_fetch)
    games, info = B.season_game_list("2025-26")
    g = games.set_index("game_id")
    # Final RS games and the Cup Final; preseason and not-yet-final games out.
    assert list(g.index) == ["0022500001", "0022500147", "0022501230", "0062500001", "0022500632"]
    assert g.loc["0062500001", "season_type"] == "NBA Cup Final"
    assert tuple(g.loc["0022500147", ["home", "away"]]) == ("", "")      # Mexico City
    assert tuple(g.loc["0022500001", ["home", "away"]]) == ("OKC", "HOU")
    assert info["list_source"] == "cdn schedule" and info["list_final"] is False
    assert (tmp_cache / "raw/box/2025-26/cdnsched.json.gz").exists()
    B.season_game_list("2025-26")
    assert len(fetched) == 2                                # a live list is never reused


def test_live_season_before_the_cdn_schedule_rolls_over_is_empty(tmp_cache, monkeypatch):
    on_date(monkeypatch, "2026-08-10")                     # 2026-27 is live, schedule still 2025-26
    no_stats(monkeypatch)
    monkeypatch.setattr(http, "fetch", lambda url, **k: fx("sched_2025-26_trim.json"))
    games, info = B.season_game_list("2026-27")
    assert games.empty and list(games.columns) == B.LGL_GAME_COLUMNS
    assert "still 2025-26" in info["list_source"] and info["list_final"] is False


def _past_list(monkeypatch, sched_payload=None, rs=None):
    lgl = {"Regular Season": rs or fx("lglT_2025-26_rs.json"),
           "PlayIn": fx("lglT_2025-26_playin.json"), "Playoffs": fx("lglT_2025-26_po.json")}
    monkeypatch.setattr(B, "fetch_lgl", lambda season, st, kind="T", date=None, refresh=False: lgl[st])
    sp = sched_payload or fx("sched_2025-26_trim.json")
    monkeypatch.setattr(B, "league_schedule",
                        lambda season, refresh=False: S.parse_schedule(copy.deepcopy(sp), season))


def test_past_season_list_is_leaguegamelog_checked_against_schedule_plus_cup_final(tmp_cache, monkeypatch):
    on_date(monkeypatch, "2026-10-08")
    _past_list(monkeypatch)
    games, info = B.season_game_list("2025-26")
    g = games.set_index("game_id")
    assert sorted(g.index) == ["0022500001", "0022500147", "0022500632", "0022501230",
                               "0042500405", "0052500121", "0062500001"]
    # The Cup Final is in no LeagueGameLog; the schedule is the only way in.
    assert tuple(g.loc["0062500001", ["season_type", "game_date", "team_a", "team_b"]]) == \
        ("NBA Cup Final", "2025-12-16", "NYK", "SAS")
    assert info["list_final"] is True and info["list_source"] == "LeagueGameLog T + schedule"


def _sched_edit(fn):
    p = fx("sched_2025-26_trim.json")
    for gd in p["leagueSchedule"]["gameDates"]:
        for g in gd["games"]:
            fn(g)
    return p


@pytest.mark.parametrize("edit,needle", [
    # The schedule lists a final game LeagueGameLog does not (a truncated or
    # silent-empty list would look like this).
    (lambda g: g.update(gameStatus=1) if g["gameId"] == "0022500632" else None, "different final games"),
    (lambda g: g["homeTeam"].update(teamId=1610612762, teamTricode="UTA")
     if g["gameId"] == "0022500001" else None, "differ"),
])
def test_leaguegamelog_and_schedule_must_agree(tmp_cache, monkeypatch, edit, needle):
    on_date(monkeypatch, "2026-10-08")
    _past_list(monkeypatch, sched_payload=_sched_edit(edit))
    with pytest.raises(ValueError, match=needle):
        B.season_game_list("2025-26")


def test_past_season_without_a_final_cup_final_raises(tmp_cache, monkeypatch):
    on_date(monkeypatch, "2026-10-08")
    _past_list(monkeypatch, sched_payload=_sched_edit(
        lambda g: g.update(gameStatus=1) if g["gameId"] == "0062500001" else None))
    with pytest.raises(ValueError, match="NBA Cup Final"):
        B.season_game_list("2025-26")


def _empty_lgl():
    p = fx("lglT_2025-26_rs.json")
    p["resultSets"][0]["rowSet"] = []
    return p


@pytest.mark.parametrize("season,stype,ok", [
    ("2015-16", "Regular Season", False),   # a past season is never empty
    ("2015-16", "Playoffs", False),
    ("2015-16", "PlayIn", True),            # no play-in before 2019-20
    ("2021-22", "PlayIn", False),
    ("2026-27", "Regular Season", True),    # not started yet
])
def test_fetch_lgl_refuses_a_silent_empty_list_and_never_caches_it(tmp_cache, monkeypatch, season, stype, ok):
    on_date(monkeypatch, "2026-10-08")
    _fake_stats(monkeypatch, {"leaguegamelog": _empty_lgl()}, [])
    if ok:
        B.fetch_lgl(season, stype)
        assert B.lgl_raw_path(season, stype).exists()
    else:
        with pytest.raises(http.BadPayload, match="silent empty"):
            B.fetch_lgl(season, stype)
        assert not B.lgl_raw_path(season, stype).exists()


@pytest.mark.parametrize("kind,date", [("T", None), ("P", "2026-06-13")])
def test_fetch_lgl_refetches_a_payload_cached_while_its_season_was_live(tmp_cache, monkeypatch, kind, date):
    on_date(monkeypatch, "2026-10-08")
    calls = []
    _fake_stats(monkeypatch, {"leaguegamelog": fx("lglT_2025-26_rs.json")}, calls)
    p = B.lgl_raw_path("2025-26", "Playoffs", kind, date)
    # Fetched mid-Finals: later games (or a late game on that date) are missing.
    write_stamped(p, _empty_lgl(), stamp_of("2026-06-14T03:00:00"))
    assert B.fetch_lgl("2025-26", "Playoffs", kind, date)["resultSets"][0]["rowSet"]
    assert len(calls) == 1
    # Refetched after the season ended: settled, reused from now on.
    B.fetch_lgl("2025-26", "Playoffs", kind, date)
    assert len(calls) == 1


def test_settled_means_fetched_after_the_season_stopped_being_live():
    assert B.settled(stamp_of("2026-08-01T12:00:00"), "2025-26")
    assert not B.settled(stamp_of("2026-06-30T12:00:00"), "2025-26")
    assert not B.settled(None, "2025-26")


@pytest.mark.parametrize("fetched,expect_refresh", [("2026-03-01T12:00:00", True),
                                                    ("2026-09-01T12:00:00", False)])
def test_league_schedule_refetches_a_provisional_copy(tmp_cache, monkeypatch, fetched, expect_refresh):
    write_stamped(S.raw_path("2025-26", "schedule"), {"payload": {}}, stamp_of(fetched))
    seen = []
    monkeypatch.setattr(S, "schedule", lambda season, refresh=None: seen.append(refresh) or "df")
    assert B.league_schedule("2025-26") == "df"
    assert seen == [True if expect_refresh else False]


# ── Script: sampling, checkpoint, resume ─────────────────────────────────────
def _script():
    spec = importlib.util.spec_from_file_location("build_participation",
                                                  REPO / "scripts" / "build_participation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pick_sample_is_spread_and_deterministic():
    bp = _script()
    games = pd.DataFrame({"game_id": [f"00224{i:05d}" for i in range(1, 101)]})
    s = bp.pick_sample(games, 4)
    assert s == bp.pick_sample(games, 4) and s[0] == "0022400001" and s[-1] == "0022400100"
    assert bp.pick_sample(games, 500) == games.game_id.tolist()


LIST_INFO = {"list_source": "test list", "list_final": True, "list_fetched_at": 1.0}


def _rs_list():
    return B.parse_lgl_games(fx("lglT_2025-26_rs.json"), "2025-26", "Regular Season")


def _args(**kw):
    d = dict(sample=0, source="auto", reparse=False, refresh_games=False, keep_going=True,
             crosscheck=False)
    return SimpleNamespace(**{**d, **kw})


def _problems():
    return {"not_final": [], "absent": [], "failed": [], "crosscheck": []}


def _setup_script(monkeypatch, outcomes: dict, info=None):
    """The script with a fixture game list and load_game answering from
    `outcomes` (gid -> exception instance, or absent = parse the cdn fixture)."""
    bp = _script()
    monkeypatch.setattr(bp, "season_games", lambda season, refresh=False: (_rs_list(), dict(info or LIST_INFO)))
    seen = []

    def fake_load(gid, source):
        seen.append(gid)
        if gid in outcomes:
            raise outcomes[gid]
        return B.parse_cdn(cdn(gid), gid)

    monkeypatch.setattr(B, "load_game", fake_load)
    return bp, seen


def test_script_resumes_and_records_failures(tmp_cache, monkeypatch):
    bp, seen = _setup_script(monkeypatch, {"0022500632": ValueError("0022500632: synthetic invariant failure")})
    problems = _problems()
    bp.run_season("2025-26", _args(), ["0022500001", "0022500147", "0022500632"], problems)
    assert problems["failed"] == ["0022500632: 0022500632: synthetic invariant failure"]
    part, games, info = B.read_participation("2025-26")
    assert set(part.game_id) == set(games.game_id) == {"0022500001", "0022500147"}
    assert info["games_done"] == 2 and info["games_total"] == 4 and info["complete"] is False
    assert info["list_source"] == "test list"
    seen.clear()
    bp.run_season("2025-26", _args(), ["0022500001", "0022500147"], problems)
    assert seen == []                                      # done games are skipped
    # Without --keep-going the first broken invariant stops the run, after a checkpoint.
    with pytest.raises(ValueError, match="synthetic"):
        bp.run_season("2025-26", _args(keep_going=False), ["0022500632"], problems)
    assert set(pd.read_parquet(bp.part_path("2025-26")).game_id) == {"0022500001", "0022500147"}


@pytest.mark.parametrize("outcome,code", [
    (None, 0),
    (ValueError("0022500147: synthetic invariant failure"), 1),
    (http.Absent("u", "V3 has no such game"), 4),
    (B.NotFinal("0022500147: cdn gameStatus=2, not final"), 4),
    (http.Blocked("u", "HTTP 403 text/html: Akamai edge block"), 2),
    (http.BadPayload("u", "expected JSON"), 3),
])
def test_main_exit_codes(tmp_cache, monkeypatch, outcome, code):
    bp, seen = _setup_script(monkeypatch, {} if outcome is None else {"0022500147": outcome})
    argv = ["--games", "0022500001,0022500147", "--keep-going"]
    assert bp.main(argv) == code
    # The game parsed before the failure is on disk whatever the exit status.
    assert "0022500001" in set(pd.read_parquet(bp.part_path("2025-26")).game_id)


def test_blocked_stops_the_season_and_is_not_recorded_as_absent(tmp_cache, monkeypatch):
    bp, seen = _setup_script(monkeypatch, {"0022500147": http.Blocked("u", "hang")})
    problems = _problems()
    with pytest.raises(http.Blocked):
        bp.run_season("2025-26", _args(), ["0022500001", "0022500147", "0022501230"], problems)
    assert seen == ["0022500001", "0022500147"]            # nothing after the block
    assert problems == _problems()
    assert set(pd.read_parquet(bp.part_path("2025-26")).game_id) == {"0022500001"}


def test_a_failing_checkpoint_does_not_mask_blocked(tmp_cache, monkeypatch):
    bp, _ = _setup_script(monkeypatch, {"0022500147": http.Blocked("u", "hang")})

    def broken(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(bp, "checkpoint", broken)
    with pytest.raises(http.Blocked):
        bp.run_season("2025-26", _args(), ["0022500001", "0022500147"], _problems())
    with pytest.raises(OSError):                           # with nothing else wrong, it surfaces
        bp.run_season("2025-26", _args(), ["0022500001"], _problems())


def test_an_unchanged_table_is_not_rewritten(tmp_cache, monkeypatch):
    bp, _ = _setup_script(monkeypatch, {})
    bp.run_season("2025-26", _args(), ["0022500001"], _problems())
    before = {p: p.read_bytes() for p in (bp.part_path("2025-26"), bp.games_path("2025-26"))}
    # A later run with nothing new (only the list's fetch time differs) and a
    # Blocked run both leave the files byte-identical: no restamp, no churn.
    monkeypatch.setattr(bp, "season_games", lambda s, refresh=False: (_rs_list(), {**LIST_INFO, "list_fetched_at": 2.0}))
    bp.run_season("2025-26", _args(), ["0022500001"], _problems())
    def blocked(gid, source):
        raise http.Blocked("u", "hang")

    monkeypatch.setattr(B, "load_game", blocked)
    with pytest.raises(http.Blocked):
        bp.run_season("2025-26", _args(), ["0022500001", "0022500147"], _problems())
    assert {p: p.read_bytes() for p in before} == before
    # A real change (the list grew) is written.
    monkeypatch.setattr(bp, "season_games", lambda s, refresh=False: (
        pd.concat([_rs_list(), _rs_list().head(1).assign(game_id="0022500002")]), LIST_INFO))
    bp.run_season("2025-26", _args(), ["0022500001"], _problems())
    assert bp.games_path("2025-26").read_bytes() != before[bp.games_path("2025-26")]
    assert B.read_participation("2025-26")[2]["games_total"] == 5


def test_duplicate_game_ids_are_parsed_once(tmp_cache, monkeypatch):
    # '--games X,X' used to write 70 rows with 35 duplicates.
    bp, seen = _setup_script(monkeypatch, {})
    assert bp.main(["--games", "0022500001,0022500001"]) == 0
    assert seen == ["0022500001"]
    part = pd.read_parquet(bp.part_path("2025-26"))
    assert not part.duplicated(["game_id", "person_id"]).any() and len(part) == 35
    bp.run_season("2025-26", _args(reparse=True), ["0022500147", "0022500147"], _problems())
    assert seen == ["0022500001", "0022500147"]


def test_checkpoint_refuses_a_game_parsed_twice(tmp_cache, monkeypatch):
    # The per-game parse checks cannot see this; the season-level check must.
    bp = _script()
    df, meta = B.parse_cdn(cdn("0022500001"), "0022500001")
    with pytest.raises(ValueError, match="more than once|not unique"):
        bp.checkpoint("2025-26", _rs_list(), LIST_INFO, None, None, [df, df], [meta, meta])
    assert not bp.part_path("2025-26").exists()


def test_reparse_replaces_a_games_rows_wholesale(tmp_cache, monkeypatch):
    bp, _ = _setup_script(monkeypatch, {})
    bp.run_season("2025-26", _args(), ["0022500001", "0022500147"], _problems())
    df, meta = B.parse_cdn(cdn("0022500001"), "0022500001")
    df["player_name"] = df["player_name"].str.upper()     # what a parser fix would change
    monkeypatch.setattr(B, "load_game", lambda gid, src: (df.copy(), dict(meta)))
    bp.run_season("2025-26", _args(reparse=True), ["0022500001"], _problems())
    part = pd.read_parquet(bp.part_path("2025-26"))
    g1 = part[part.game_id == "0022500001"]
    assert len(g1) == len(df) and (g1.player_name == g1.player_name.str.upper()).all()
    assert set(part.game_id) == {"0022500001", "0022500147"}   # the other game is kept


def test_requested_v3_is_not_reported_as_a_cdn_hole(tmp_cache, monkeypatch, capsys):
    bp, _ = _setup_script(monkeypatch, {})
    df, meta = B.parse_cdn(cdn("0022500001"), "0022500001")
    monkeypatch.setattr(B, "load_game", lambda gid, src: (df.assign(source="v3"), {**meta, "source": "v3"}))
    bp.run_season("2025-26", _args(source="v3"), ["0022500001"], _problems())
    bp.report(["2025-26"], _problems())
    out = capsys.readouterr().out
    assert "CDN has no box score" not in out
    assert "served by the V3 fallback (CDN absent): 0" in out
    assert "by request (--source): 1" in out


def test_a_live_season_is_never_complete(tmp_cache, monkeypatch):
    games = _rs_list()
    live = {"list_source": "cdn schedule", "list_final": False, "list_fetched_at": 1.0}
    bp, _ = _setup_script(monkeypatch, {}, info=live)
    monkeypatch.setattr(bp, "season_games", lambda s, refresh=False: (games.head(2), dict(live)))
    bp.run_season("2025-26", _args(), None, _problems())
    info = B.read_participation("2025-26")[2]
    assert (info["games_done"], info["games_total"], info["complete"]) == (2, 2, False)
    monkeypatch.setattr(bp, "season_games", lambda s, refresh=False: (games.head(2), dict(LIST_INFO)))
    bp.run_season("2025-26", _args(), None, _problems())
    assert B.read_participation("2025-26")[2]["complete"] is True


def test_a_table_from_the_old_schema_is_reparsed_from_raw(tmp_cache, monkeypatch, capsys):
    bp, seen = _setup_script(monkeypatch, {})
    df, meta = B.parse_cdn(cdn("0022500001"), "0022500001")
    cache.write_parquet(df.drop(columns="absence_kind"), bp.part_path("2025-26"))
    cache.write_parquet(pd.DataFrame([meta], columns=B.META_COLUMNS), bp.games_path("2025-26"))
    with pytest.raises(ValueError, match="rebuild"):
        B.read_participation("2025-26")
    bp.run_season("2025-26", _args(), ["0022500147"], _problems())
    assert seen == ["0022500147", "0022500001"]
    assert "schema changed" in capsys.readouterr().out
    part, games, _ = B.read_participation("2025-26")
    assert list(part.columns) == B.COLUMNS and set(games.game_id) == {"0022500001", "0022500147"}


def test_crosscheck_skips_the_cup_final(tmp_cache, monkeypatch):
    bp = _script()
    monkeypatch.setattr(B, "fetch_lgl", lambda *a, **k: pytest.fail("no LeagueGameLog lists the Cup Final"))
    df, meta = B.parse_cdn(cdn("0062500001"), "0062500001")
    problems = _problems()
    bp.crosscheck("2025-26", df, pd.DataFrame([meta]), problems)
    assert problems == _problems()


# ── Live (opt-in: BUCKETVALUE_NETWORK=1; 2 cdn calls) ────────────────────────
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("BUCKETVALUE_NETWORK") != "1",
                    reason="live cdn.nba.com; set BUCKETVALUE_NETWORK=1")
def test_live_cdn_box_and_pre_2019_absence(tmp_cache):
    df, meta = B.load_game("0022300500", source="cdn")
    assert meta["home"] == "SAC" and (df.groupby("team")["starter"].sum() == 5).all()
    with pytest.raises(http.Absent):
        B.fetch_cdn("0021800600")


def test_cdn_frozen_midgame_falls_back_to_v3_for_a_past_season_only(monkeypatch):
    """The CDN archive holds some finished 2019-20 games frozen 'inprogress'
    (0021900877). Past season: use the V3 final. Live season: NotFinal stands."""
    from bucketlib.sources import box as B

    def frozen(gid, refresh=False):
        raise B.NotFinal(f"{gid}: cdn gameStatus='inprogress', not final")

    calls = []
    monkeypatch.setattr(B, "fetch_cdn", frozen)
    monkeypatch.setattr(B, "fetch_v3", lambda gid, refresh=False: calls.append(gid) or ("box", "sum"))
    monkeypatch.setattr(B, "parse_v3", lambda box, summ, gid: ("rows", {"source": "v3"}))
    assert B.load_game("0021900877") == ("rows", {"source": "v3"}) and calls == ["0021900877"]
    monkeypatch.setattr(B, "_is_live_game", lambda gid: True)
    import pytest
    with pytest.raises(B.NotFinal):
        B.load_game("0022600001")
