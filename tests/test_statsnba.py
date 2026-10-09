"""stats.nba.com bulk-endpoint parsers, fetch/cache policy and their traps.

Fixtures (tests/fixtures/statsnba/) are REAL payloads pulled live on
2026-10-08 through bucketlib.http, with the *_RANK columns dropped exactly as
the raw cache stores them, then trimmed:
  logs_2024-25_RS            5 whole games (all players, both teams): 0022400147
                             (Mexico City, neutral, both MATCHUPs '@'), 0022400330
                             (player minutes sum to 242.0, the 2.0 tolerance edge),
                             0022400341 (PGL/LGL disagree on Jokic's OREB),
                             0022401102 (2OT, a 0:00 Luka Garza row), 0022401201
                             (PGL's DD2 flag is stale for Jalen Suggs)
  logs_2022-23_RS_lgl_only   0022200140. PGL/LGL T/TGL live; LeagueGameLog P as
                             the research pull returned it at 19:04 ET, when it
                             still carried Dennis Schroder's all-zero 0:00 row
                             that PlayerGameLogs does not have (the cdn box score
                             lists him INACTIVE; a later pull no longer returns
                             the row at all: the endpoint is not stable)
  logs_2019-20_RS            0021900669 (Paris: vs./@ in the logs, isNeutral
                             false), 0021900969 (Atlanta, pre-hiatus) and
                             0021901231 (the first Orlando bubble game)
  empty_2026-27_RS           the pre-season state: HTTP 200, headers, zero rows
  schedule_<season>          scheduleleaguev2, a few games per season; 2023-24 has
                             the Cup semifinals and final in Las Vegas, Mexico City
                             and Paris (all isNeutral false) and a Spurs home game
                             in Austin
  playerindex_<season>       2026-27 (2+ per team) and 2024-25 (the past-season trap)
  roster_<season>_<team>     CommonTeamRoster, untrimmed; 2024-25 LAL is a PAST
                             season whose HOW_ACQUIRED is today's record
Every mutation test below breaks one real payload in one place and checks
that the parser refuses it.
"""
import copy
import datetime as dt
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import http
from bucketlib.sources import statsnba as S

FX = Path(__file__).resolve().parent / "fixtures" / "statsnba"
NAMES = ["pgl_Base_RS", "pgl_Advanced_RS", "lgl_P_RS", "lgl_T_RS", "tgl_Advanced_RS"]


def load(name: str):
    return json.loads(gzip.decompress((FX / name).read_bytes()))


@pytest.fixture(autouse=True)
def pinned_state(monkeypatch):
    """Every test sees 2026-10-08 ET (the live season is 2026-27, preseason)
    and a fresh process: no fetch memo, no stamps, no parsed schedules."""
    monkeypatch.setattr(S.util, "today_et", lambda: pd.Timestamp("2026-10-08").date())
    monkeypatch.setattr(S, "_fetched_this_run", set())
    monkeypatch.setattr(S, "_stamps", {})
    monkeypatch.setattr(S, "_schedule_memo", {})


def set_today(monkeypatch, day: str):
    monkeypatch.setattr(S.util, "today_et", lambda: pd.Timestamp(day).date())


@pytest.fixture
def logs25():
    return load("logs_2024-25_RS.json.gz")


def pg(d, season="2024-25", st="RS"):
    return S.parse_player_games(d["pgl_Base_RS"], d["pgl_Advanced_RS"], d["lgl_P_RS"],
                                d["lgl_T_RS"], season, st)


def tg(d, season="2024-25", st="RS"):
    return S.parse_team_games(d["lgl_T_RS"], d["tgl_Advanced_RS"], season, st)


def rows(payload, set_index=0):
    s = payload["resultSets"][set_index]
    return s["headers"], s["rowSet"]


def edit(payload, where: dict, **changes):
    """Set columns on every row matching `where` (column -> value)."""
    h, rs = rows(payload)
    n = 0
    for r in rs:
        if all(r[h.index(k)] == v for k, v in where.items()):
            for k, v in changes.items():
                r[h.index(k)] = v
            n += 1
    assert n, f"no row matched {where}"
    return n


# ── player_games ─────────────────────────────────────────────────────────────
def test_player_games_schema_and_invariants(logs25):
    df = pg(logs25)
    assert list(df.columns) == S.PLAYER_GAME_COLUMNS
    assert df["game_date"].dtype == "datetime64[ns]"
    assert df["person_id"].dtype == "int64" and df["sec"].dtype == "int64"
    assert df["home"].dtype == bool and df["neutral"].dtype == bool
    assert df["game_id"].str.len().eq(10).all()
    assert not df.duplicated(["person_id", "game_id"]).any()
    assert df.groupby(["game_id", "team"]).size().min() >= 5
    assert set(df["season"]) == {"2024-25"} and set(df["season_type"]) == {"RS"}
    r = df.attrs["report"]
    assert r["rows"] == 101 and r["games"] == 5 and r["lgl_only"] == 0
    # 0022400330 sums to exactly 242.0: inside the tolerance, reported.
    assert r["min_sum_max_abs_dev"] == 2.0
    # PGL wins, but the disagreement with LeagueGameLog is counted.
    assert r["pgl_vs_lgl_stat_disagreements"].get("oreb", 0) >= 1
    jok = df[(df.person_id == 203999) & (df.game_id == "0022400341")].iloc[0]
    assert (jok.oreb, jok.reb) == (8, 17)
    assert r["plus_minus_sum_violations"] == 0 and r["team_sum_mismatches"] == {}


def test_plus_minus_sum_is_checked(logs25):
    edit(logs25["pgl_Base_RS"], {"PLAYER_ID": 203999, "GAME_ID": "0022400341"}, PLUS_MINUS=40)
    assert pg(logs25).attrs["report"]["plus_minus_sum_violations"] == 1


def test_game_dates_are_parsed_et_dates(logs25):
    df = pg(logs25)
    g = df[df.game_id == "0022400147"]
    assert set(g["game_date"]) == {pd.Timestamp("2024-11-02")}
    assert (df["game_date"] == df["game_date"].dt.normalize()).all()


def test_fantasy_is_prizepicks_formula_and_matches_source(logs25):
    df = pg(logs25)
    want = df.pts + 1.2 * df.reb + 1.5 * df.ast + 3 * df.stl + 3 * df.blk - df.tov
    assert np.allclose(df.fantasy, want, atol=1e-9)
    assert df.attrs["report"]["fantasy_max_abs_err"] < 1e-6


def test_fantasy_mismatch_with_source_raises(logs25):
    edit(logs25["pgl_Base_RS"], {"PLAYER_ID": 203999, "GAME_ID": "0022400341"}, NBA_FANTASY_PTS=1.0)
    with pytest.raises(ValueError, match="fantasy"):
        pg(logs25)


def test_dd2_recomputed_not_copied(logs25):
    # PGL says DD2=0 for Suggs (32 pts, 10 reb after a correction).
    df = pg(logs25)
    suggs = df[df.player_name.str.contains("Suggs") & (df.game_id == "0022401201")].iloc[0]
    assert suggs.pts >= 10 and suggs.reb >= 10 and suggs.dd2 == 1
    assert df.attrs["report"]["source_dd2_disagreements"] == 1


def test_neutral_site_both_away(logs25):
    """Mexico City: MATCHUP is 'MIA @ WAS' AND 'WAS @ MIA'."""
    df = pg(logs25)
    g = df[df.game_id == "0022400147"]
    assert not g["home"].any() and g["neutral"].all()
    assert not df[df.game_id != "0022400147"]["neutral"].any()


def test_zero_minute_row_kept(logs25):
    df = pg(logs25)
    z = df[(df.game_id == "0022401102") & (df.sec == 0)]
    assert len(z) == 1 and z.iloc[0].src == "pgl"


def test_lgl_only_all_zero_row_excluded_and_counted():
    """PGL is the appearance authority: Schroder's all-zero LGL-only row is a
    player who did not play (cdn: INACTIVE), so it is excluded, never given a
    0.0 fantasy line, and it is counted and listed."""
    d = load("logs_2022-23_RS_lgl_only.json.gz")
    assert len(rows(d["pgl_Base_RS"])[1]) == 21 and len(rows(d["lgl_P_RS"])[1]) == 22
    df = pg(d, "2022-23")
    r = df.attrs["report"]
    assert r["lgl_only"] == 1 and r["lgl_only_excluded"] == 1
    assert r["lgl_only_rows"][0]["person_id"] == 203471
    assert 203471 not in set(df.person_id) and len(df) == 21
    assert set(df.src) == {"pgl"} and df.pfd.notna().all()


@pytest.mark.parametrize("change", [{"MIN": 3}, {"DREB": 1, "REB": 1}])
def test_lgl_only_row_with_minutes_or_stats_raises(change):
    """An LGL-only row WITH minutes or a stat would be an appearance PGL lacks;
    a rebound would not even move the points or minutes sums."""
    d = load("logs_2022-23_RS_lgl_only.json.gz")
    edit(d["lgl_P_RS"], {"PLAYER_ID": 203471}, **change)
    with pytest.raises(ValueError, match="appearance PGL lacks"):
        pg(d, "2022-23")


def test_minutes_sum_violation_raises_and_allowlist(logs25, monkeypatch):
    # Add 3 minutes to one player consistently in all three sources, so only the
    # team-game minutes invariant can catch it.
    key = {"PLAYER_ID": 203999, "GAME_ID": "0022400341"}
    h, rs = rows(logs25["pgl_Base_RS"])
    base_min = next(r[h.index("MIN")] for r in rs if r[h.index("PLAYER_ID")] == 203999
                    and r[h.index("GAME_ID")] == "0022400341")
    edit(logs25["pgl_Base_RS"], key, MIN=base_min + 3)
    edit(logs25["pgl_Advanced_RS"], key, MIN=round(base_min + 3, 1))
    edit(logs25["lgl_P_RS"], key, MIN=round(base_min + 3))
    with pytest.raises(ValueError, match="minutes sum"):
        pg(logs25)
    monkeypatch.setitem(S.KNOWN_MINUTES_EXCEPTIONS, "0022400341", "test: inspected box")
    assert pg(logs25).attrs["report"]["min_sum_known_exceptions"] == 1


def test_overtime_comes_from_team_log(logs25):
    """0022401102 is 2OT: 290 player-minutes, so ~290 must not raise."""
    df = pg(logs25)
    s = df[df.game_id == "0022401102"].groupby("team")["min"].sum()
    assert ((s - 290).abs() <= 2).all()


def _drop_rows(d, game_id, team_id, keep):
    """Remove all but `keep` players of one team-game from all three player
    sources (the same players everywhere: each payload has its own row order)."""
    h, rs = rows(d["pgl_Base_RS"])
    pids = [r[h.index("PLAYER_ID")] for r in rs
            if r[h.index("GAME_ID")] == game_id and r[h.index("TEAM_ID")] == team_id]
    drop = set(pids[keep:])
    assert drop
    for n in ("pgl_Base_RS", "pgl_Advanced_RS", "lgl_P_RS"):
        h, rs = rows(d[n])
        gi, pi = h.index("GAME_ID"), h.index("PLAYER_ID")
        rs[:] = [r for r in rs if not (r[gi] == game_id and r[pi] in drop)]


def test_fewer_than_five_players_raises(logs25):
    _drop_rows(logs25, "0022400147", 1610612748, keep=4)
    with pytest.raises(ValueError, match="fewer than 5"):
        pg(logs25)


def test_player_points_must_sum_to_team_score(logs25):
    # One extra made free throw on one PGL row, box identity and fantasy kept
    # consistent, minutes untouched: only the team-score invariant can see it.
    h, rs = rows(logs25["pgl_Base_RS"])
    r = next(r for r in rs if r[h.index("PLAYER_ID")] == 203999
             and r[h.index("GAME_ID")] == "0022400341")
    for c in ("FTM", "FTA", "PTS", "NBA_FANTASY_PTS"):
        r[h.index(c)] += 1
    with pytest.raises(ValueError, match="player points do not sum to the team score"):
        pg(logs25)


def test_duplicate_player_game_raises(logs25):
    h, rs = rows(logs25["pgl_Base_RS"])
    rs.append(list(rs[0]))
    with pytest.raises(ValueError, match="share a key"):
        pg(logs25)


def test_unknown_team_id_raises(logs25):
    for n in ("pgl_Base_RS", "pgl_Advanced_RS", "lgl_P_RS"):
        edit(logs25[n], {"TEAM_ID": 1610612748}, TEAM_ID=1610612799)
    with pytest.raises(ValueError, match="unknown TEAM_ID"):
        pg(logs25)


def test_matchup_must_start_with_own_team(logs25):
    edit(logs25["pgl_Base_RS"], {"PLAYER_ID": 203999, "GAME_ID": "0022400341"},
         MATCHUP="WAS @ DEN")
    with pytest.raises(ValueError, match="MATCHUP"):
        pg(logs25)


def test_pgl_lgl_team_disagreement_raises(logs25):
    edit(logs25["lgl_P_RS"], {"PLAYER_ID": 203999, "GAME_ID": "0022400341"},
         TEAM_ID=1610612764)
    with pytest.raises(ValueError, match="disagree on a player's team"):
        pg(logs25)


def test_impossible_box_values_raise(logs25):
    h, rs = rows(logs25["pgl_Base_RS"])
    r = next(r for r in rs if r[h.index("FGA")] > 0)
    r[h.index("FGM")] = r[h.index("FGA")] + 1
    with pytest.raises(ValueError):
        pg(logs25)


def test_game_id_prefix_must_match_season_type(logs25):
    with pytest.raises(ValueError, match="do not start|SEASON_ID"):
        pg(logs25, st="Playoffs")
    with pytest.raises(ValueError):
        pg(logs25, season="2023-24")


def test_pgl_date_format_is_explicit(logs25):
    # PGL dates are 'YYYY-MM-DDT00:00:00'; a bare LGL-style date is a format change.
    edit(logs25["pgl_Base_RS"], {"GAME_ID": "0022400147"}, GAME_DATE="2024-11-02")
    with pytest.raises(ValueError, match="GAME_DATE"):
        pg(logs25)


def test_advanced_row_without_base_raises(logs25):
    h, rs = rows(logs25["pgl_Base_RS"])
    rs.pop()
    with pytest.raises(ValueError, match="Advanced rows have no Base"):
        pg(logs25)


def test_missing_required_column_is_bad_payload(logs25):
    s = logs25["pgl_Base_RS"]["resultSets"][0]
    i = s["headers"].index("PFD")
    s["headers"].pop(i)
    for r in s["rowSet"]:
        r.pop(i)
    with pytest.raises(http.BadPayload, match="PFD"):
        pg(logs25)


def _set_min(d, pid, gid, value):
    """Set one player's minutes consistently in all three sources: Base exact,
    Advanced to 0.1, LeagueGameLog to the minute (how each source rounds)."""
    key = {"PLAYER_ID": pid, "GAME_ID": gid}
    edit(d["pgl_Base_RS"], key, MIN=value)
    edit(d["pgl_Advanced_RS"], key, MIN=round(value, 1))
    edit(d["lgl_P_RS"], key, MIN=round(value))


def test_player_minutes_longer_than_the_game_raises(logs25):
    # Jokic 38.8 -> 49.0 in a regulation game, Braun gives the same 10.2 back,
    # so the team-game minutes sum (the check that runs first) still holds.
    _set_min(logs25, 203999, "0022400341", 49.0)
    _set_min(logs25, 1631128, "0022400341", 37.783333 - (49.0 - 38.766667))
    with pytest.raises(ValueError, match="more minutes than the game lasted"):
        pg(logs25)


def test_lgl_min_must_be_pgl_min_rounded(logs25):
    edit(logs25["lgl_P_RS"], {"PLAYER_ID": 203999, "GAME_ID": "0022400341"}, MIN=42)
    with pytest.raises(ValueError, match="LGL MIN is not PGL MIN rounded"):
        pg(logs25)


def test_advanced_min_must_be_base_min_rounded(logs25):
    edit(logs25["pgl_Advanced_RS"], {"PLAYER_ID": 203999, "GAME_ID": "0022400341"}, MIN=39.9)
    with pytest.raises(ValueError, match="Advanced MIN is not Base MIN rounded"):
        pg(logs25)


def test_pgl_lgl_date_disagreement_raises(logs25):
    # Same game, a different date in LeagueGameLog for one player row only.
    edit(logs25["lgl_P_RS"], {"PLAYER_ID": 203999, "GAME_ID": "0022400341"},
         GAME_DATE="2024-11-12")
    with pytest.raises(ValueError, match="disagree on a game date"):
        pg(logs25)


def test_logs_neutral_from_bubble_and_schedule():
    """2019-20: the bubble game keeps its designated home side and is neutral
    from the logs alone; the Paris game is labelled vs./@ with no neutral flag
    anywhere in the logs, so only the schedule's arena city reveals it."""
    d = load("logs_2019-20_RS.json.gz")
    paris, atl, bubble = "0021900669", "0021900969", "0021901231"
    alone = S.parse_team_games(d["lgl_T_RS"], d["tgl_Advanced_RS"], "2019-20", "RS")
    g = alone.set_index(["game_id", "team"])
    assert g.loc[(bubble, "NOP"), "home"] and alone.loc[alone.game_id == bubble, "neutral"].all()
    assert not alone.loc[alone.game_id.isin([paris, atl]), "neutral"].any()
    assert alone.attrs["report"]["neutral_source"] == "logs only"
    sched = S.parse_schedule(load("schedule_2019-20.json.gz"), "2019-20")
    ng = frozenset(sched.loc[sched.is_neutral, "game_id"])
    t = S.parse_team_games(d["lgl_T_RS"], d["tgl_Advanced_RS"], "2019-20", "RS", neutral_games=ng)
    p = S.parse_player_games(d["pgl_Base_RS"], d["pgl_Advanced_RS"], d["lgl_P_RS"],
                             d["lgl_T_RS"], "2019-20", "RS", neutral_games=ng)
    for df in (t, p):
        assert df.loc[df.game_id.isin([paris, bubble]), "neutral"].all()
        assert not df.loc[df.game_id == atl, "neutral"].any()
    assert t.set_index(["game_id", "team"]).loc[(paris, "CHA"), "home"]  # designated side kept
    assert t.attrs["report"]["neutral_logs_not_schedule"] == []
    assert t.attrs["report"]["neutral_from_schedule_only"] == 2  # Paris + the bubble game


def _add_cancelled_game(d, gid="0022400999", pts=0, mins=0):
    """Two LGL T rows shaped like 0021201214 (BOS-IND 2013-04-16): MIN 0, no
    result, zeros everywhere, and no row in any other log."""
    h, rs = rows(d["lgl_T_RS"])
    base = next(r for r in rs if r[h.index("GAME_ID")] == "0022400330")
    for tid, abbr, matchup in ((1610612738, "BOS", "BOS vs. IND"), (1610612754, "IND", "IND @ BOS")):
        r = [0 if isinstance(v, (int, float)) and not isinstance(v, bool) else v for v in base]
        for k, v in {"TEAM_ID": tid, "TEAM_ABBREVIATION": abbr, "GAME_ID": gid,
                     "MATCHUP": matchup, "WL": None, "MIN": mins, "PTS": pts,
                     "SEASON_ID": base[h.index("SEASON_ID")],
                     "GAME_DATE": "2025-04-16"}.items():
            r[h.index(k)] = v
        rs.append(r)


def test_known_cancelled_game_excluded_and_counted(logs25, monkeypatch):
    _add_cancelled_game(logs25)
    with pytest.raises(ValueError, match="KNOWN_CANCELLED_GAMES"):
        tg(logs25)  # unlisted: the 240+25k check fails, with the hint
    monkeypatch.setitem(S.KNOWN_CANCELLED_GAMES, "0022400999", "test: never played")
    t, p = tg(logs25), pg(logs25)
    assert "0022400999" not in set(t.game_id) and len(t) == 10 and len(p) == 101
    assert t.attrs["report"]["cancelled_games_excluded"] == 1
    assert p.attrs["report"]["cancelled_games_excluded"] == 1


def test_listed_cancelled_game_with_a_result_raises(logs25, monkeypatch):
    monkeypatch.setitem(S.KNOWN_CANCELLED_GAMES, "0022400999", "test")
    _add_cancelled_game(logs25, pts=100)
    with pytest.raises(ValueError, match="has minutes or points"):
        tg(logs25)
    # A listed game with PLAYER rows was played: the list is wrong.
    d = load("logs_2024-25_RS.json.gz")
    _add_cancelled_game(d)
    h, rs = rows(d["pgl_Base_RS"])
    ghost = list(rs[0])
    ghost[h.index("GAME_ID")] = "0022400999"
    rs.append(ghost)
    with pytest.raises(ValueError, match="player rows for a game listed"):
        pg(d)


# ── team_games ───────────────────────────────────────────────────────────────
def test_team_games_mirror_and_ot(logs25):
    df = tg(logs25)
    assert list(df.columns) == S.TEAM_GAME_COLUMNS
    assert df.groupby("game_id").size().eq(2).all()
    a = df.set_index(["game_id", "team"])
    for (gid, team), r in a.iterrows():
        o = a.loc[(gid, r.opp)]
        assert o.opp == team and o.pts == r.opp_pts and o.margin == -r.margin
    assert set(df.loc[df.game_id == "0022401102", "ot_periods"]) == {2}
    assert df.loc[df.game_id == "0022400147", "neutral"].all()
    assert (df.loc[~df.neutral].groupby("game_id")["home"].sum() == 1).all()


def test_team_min_not_240_plus_25k_raises(logs25):
    edit(logs25["lgl_T_RS"], {"GAME_ID": "0022400330"}, MIN=250)
    with pytest.raises(ValueError, match="240\\+25k"):
        tg(logs25)


def test_team_game_missing_side_raises(logs25):
    h, rs = rows(logs25["lgl_T_RS"])
    rs.remove(next(r for r in rs if r[h.index("GAME_ID")] == "0022400330"))
    h2, rs2 = rows(logs25["tgl_Advanced_RS"])
    rs2.remove(next(r for r in rs2 if r[h2.index("GAME_ID")] == "0022400330"
                    and r[h2.index("TEAM_ID")] not in [x[h.index("TEAM_ID")] for x in rs
                                                       if x[h.index("GAME_ID")] == "0022400330"]))
    with pytest.raises(ValueError, match="exactly 2"):
        tg(logs25)


def test_team_advanced_must_match_lgl(logs25):
    h, rs = rows(logs25["tgl_Advanced_RS"])
    rs.pop()
    with pytest.raises(ValueError, match="only one of LGL T"):
        tg(logs25)


def test_team_game_minutes_disagree_with_ot_raises(logs25):
    edit(logs25["tgl_Advanced_RS"], {"GAME_ID": "0022401102"}, MIN=53)
    with pytest.raises(ValueError, match="overtime"):
        tg(logs25)


def test_wl_must_agree_with_score(logs25):
    h, rs = rows(logs25["lgl_T_RS"])
    r = next(r for r in rs if r[h.index("GAME_ID")] == "0022400330")
    r[h.index("WL")] = "L" if r[h.index("WL")] == "W" else "W"
    with pytest.raises(ValueError, match="WL disagrees"):
        tg(logs25)


def test_score_mirror_broken_raises(logs25):
    h, rs = rows(logs25["lgl_T_RS"])
    r = next(r for r in rs if r[h.index("GAME_ID")] == "0022400330")
    r[h.index("PLUS_MINUS")] += 1
    with pytest.raises(ValueError, match="PLUS_MINUS"):
        tg(logs25)


# ── empty seasons ────────────────────────────────────────────────────────────
def test_empty_current_season_is_data_not_outage():
    d = load("empty_2026-27_RS.json.gz")
    p = pg(d, "2026-27")
    t = tg(d, "2026-27")
    assert p.empty and list(p.columns) == S.PLAYER_GAME_COLUMNS
    assert t.empty and list(t.columns) == S.TEAM_GAME_COLUMNS
    assert p["person_id"].dtype == "int64" and p["game_date"].dtype == "datetime64[ns]"


def test_empty_allowed_only_before_the_first_scheduled_game(offline_cache, monkeypatch):
    """The fix for 'empty is fine all live season': empty is legitimate only
    through the ET date of the first scheduled game of that type."""
    _seed_raw(offline_cache, "2026-27", {"schedule": load("schedule_2026-27.json.gz")})
    # This process already refreshed the live schedule once (the module's
    # once-per-process rule); otherwise the default refresh=None refetches.
    S._fetched_this_run.add(S.raw_path("2026-27", "schedule"))
    # The fixture's first RS game is 2026-10-20; it lists no PI or PO game yet.
    assert S.first_game_date("2026-27", "RS") == dt.date(2026, 10, 20)
    assert S.empty_allowed("2026-27", "RS")                    # 10-08: preseason
    set_today(monkeypatch, "2026-10-20")
    assert S.empty_allowed("2026-27", "RS")                    # opening night, not final
    for day in ("2026-10-21", "2027-01-15"):
        set_today(monkeypatch, day)
        assert not S.empty_allowed("2026-27", "RS"), day
        assert not S.empty_allowed("2026-27", "Regular Season", refresh=False), day
        assert S.empty_allowed("2026-27", "Playoffs")          # none scheduled yet
    assert S.empty_allowed("2027-28", "Playoffs")              # a later season
    assert not S.empty_allowed("2024-25", "RS")
    assert not S.empty_allowed("2024-25", "PlayIn")
    assert S.empty_allowed("2017-18", "PlayIn")  # no play-in before 2019-20
    assert not S.empty_allowed("2019-20", "PI")  # the one bubble play-in game
    with pytest.raises(ValueError, match="outage"):
        S._guard_empty(pg(load("empty_2026-27_RS.json.gz"), "2026-27"), "2024-25", "RS", "x")


def _serve(monkeypatch, logs_payload, calls=None):
    """Fake stats.nba.com: the 2026-27 schedule fixture for scheduleleaguev2,
    `logs_payload` (a copy) for every log endpoint."""
    def fake_get(endpoint, params, timeout=45.0, retries=3):
        if calls is not None:
            calls.append(endpoint)
        if endpoint == "scheduleleaguev2":
            return load("schedule_2026-27.json.gz")
        return copy.deepcopy(logs_payload)
    monkeypatch.setattr(S.http, "stats_get", fake_get)


def test_empty_pull_in_january_raises_and_writes_nothing(offline_cache, monkeypatch, capsys):
    """The verifier's demo, 2027-01-15, every log endpoint answering empty:
    player_games/team_games returned 0 rows and build_logs exited 0 having
    written 0-row parquets."""
    set_today(monkeypatch, "2027-01-15")
    empty = load("empty_2026-27_RS.json.gz")["lgl_T_RS"]
    _serve(monkeypatch, empty)
    for fn in (S.player_games, S.team_games):
        with pytest.raises(http.BadPayload, match="0 rows where data must exist"):
            fn("2026-27", "RS")
    assert not list(offline_cache.rglob("*_RS.json.gz"))  # nothing cached
    assert _script("build_logs").main(["--seasons", "2026-27", "--types", "RS"]) == 1
    assert not S.tidy_path("player_games", "2026-27").exists()
    assert not S.tidy_path("team_games", "2026-27").exists()


def test_empty_answer_never_replaces_cached_rows(offline_cache, monkeypatch):
    """Preseason, so empty IS allowed, yet a cached payload with rows (an
    edge-cache glitch after a real pull) is never overwritten by it."""
    have = load("logs_2024-25_RS.json.gz")["lgl_T_RS"]
    _seed_raw(offline_cache, "2026-27", {"lgl_T_RS": have})
    _serve(monkeypatch, load("empty_2026-27_RS.json.gz")["lgl_T_RS"])
    with pytest.raises(http.BadPayload, match="would replace 10 cached rows"):
        S.fetch_lgl("2026-27", "RS", "T")
    S._fetched_this_run.clear()
    assert S.fetch_lgl("2026-27", "RS", "T", refresh=False) == have


def test_cached_empty_pull_read_after_opening_night_raises(offline_cache, monkeypatch):
    _seed_raw(offline_cache, "2026-27", {**load("empty_2026-27_RS.json.gz"),
                                         "schedule": load("schedule_2026-27.json.gz")})
    assert S.player_games("2026-27", "RS", refresh=False).empty     # preseason: legit
    set_today(monkeypatch, "2026-11-02")
    for fn in (S.player_games, S.team_games):
        with pytest.raises(ValueError, match="after the first scheduled RS game"):
            fn("2026-27", "RS", refresh=False)


def test_build_logs_keeps_both_tables_when_either_would_shrink(offline_cache, monkeypatch):
    """Guarded separately, one table could be replaced and the other kept,
    pairing player and team tables from different pulls."""
    bl = _script("build_logs")
    d = load("logs_2024-25_RS.json.gz")
    p_full, t_full = pg(d), tg(d)
    for table, df in (("player_games", p_full), ("team_games", t_full)):
        S.write_tidy(df, S.tidy_path(table, "2026-27"),
                     {"types": ["RS"], "reports": {}, "fetched_at": 1.0}, live=True)
    hashes = {t: S.tidy_meta(S.tidy_path(t, "2026-27"))["content_hash"]
              for t in ("player_games", "team_games")}
    p_new = p_full.copy()
    p_new.loc[0, "player_name"] = "changed"           # would rewrite player_games
    t_less = t_full[t_full.game_id != "0022400330"].copy()   # team_games shrinks
    p_new.attrs, t_less.attrs = dict(p_full.attrs), dict(t_full.attrs)
    monkeypatch.setattr(S, "player_games", lambda *a, **k: p_new)
    monkeypatch.setattr(S, "team_games", lambda *a, **k: t_less)
    rec = bl.build_season("2026-27", ["RS"], force=False)
    assert rec["status"] == "kept" and all("stale beats empty" in m for m in rec["notes"])
    assert hashes == {t: S.tidy_meta(S.tidy_path(t, "2026-27"))["content_hash"]
                      for t in ("player_games", "team_games")}


def test_one_empty_source_raises(logs25):
    empty = load("empty_2026-27_RS.json.gz")
    logs25["lgl_P_RS"] = copy.deepcopy(empty["lgl_P_RS"])
    with pytest.raises(ValueError, match="one source is empty"):
        pg(logs25)


def test_season_type_names():
    assert S.stype_code("Regular Season") == "RS"
    assert S.stype_code("PlayIn") == "PI" and S.stype_api("PI") == "PlayIn"
    assert S.stype_code("Playoffs") == "PO" and S.stype_code("PO") == "PO"
    with pytest.raises(ValueError):
        S.stype_code("Play In")  # an HTTP 400 on the real endpoint


# ── schedule ─────────────────────────────────────────────────────────────────
def test_schedule_2026_27_traps():
    df = S.parse_schedule(load("schedule_2026-27.json.gz"), "2026-27")
    r = df.attrs["report"]
    assert list(df.columns) == S.SCHEDULE_COLUMNS
    assert not df.duplicated("game_id").any()
    # London Lions dropped and counted, never mapped.
    assert r["dropped_non_nba"] == 1 and "0012600042" not in set(df.game_id)
    assert r["dropped_rows"][0]["non_nba"] == ["LON:50015"]
    # Cup knockout placeholders: TBD sides, unknown tip.
    tbd = df[df.is_tbd]
    assert len(tbd) == 7 and tbd.home.isna().all() and tbd.tip_utc.isna().all()
    # The Cup final is kept: rest features need it.
    cf = df[df.game_type == "CUP_FINAL"]
    assert len(cf) == 1 and cf.iloc[0].game_id == "0062600001" and cf.iloc[0].is_neutral
    # gameDate parsed, not string-compared ('01/01/2027' sorts before '10/03/2026').
    assert df.game_date.min() == pd.Timestamp("2026-10-03")
    assert (df.game_date == pd.Timestamp("2027-01-01")).any()


def test_schedule_tip_is_utc_and_est_fields_are_wall_clock():
    raw = load("schedule_2026-27.json.gz")
    df = S.parse_schedule(raw, "2026-27")
    opener = df[df.game_id == "0012600009"].iloc[0]  # MIA @ TOR, 7:00pm ET
    assert str(opener.tip_utc.tz) == "UTC"
    assert opener.tip_utc == pd.Timestamp("2026-10-03T23:00:00Z")
    g = next(x for gd in raw["leagueSchedule"]["gameDates"] for x in gd["games"]
             if x["gameId"] == "0012600009")
    # gameDateTimeEst carries 'Z' but is EASTERN wall clock: 19:00, not 23:00.
    assert g["gameDateTimeEst"] == "2026-10-03T19:00:00Z"
    ok = df.dropna(subset=["tip_utc"])
    et = ok.tip_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    assert (et == ok.game_date).all()


def test_schedule_national_tv_and_neutral():
    df = S.parse_schedule(load("schedule_2026-27.json.gz"), "2026-27")
    assert df.national_tv.notna().any()
    glob = df[df.subtype == "Global Games"]
    assert len(glob) and glob.is_neutral.any()


def test_schedule_2024_25_allstar_and_types():
    df = S.parse_schedule(load("schedule_2024-25.json.gz"), "2024-25")
    r = df.attrs["report"]
    assert r["dropped_allstar"] == 6 and r["dropped_non_nba"] == 1
    assert {"RS", "PI", "PO", "CUP_FINAL", "PRE"} <= set(df.game_type)
    assert df[df.game_type == "CUP_FINAL"].iloc[0][["home", "away"]].tolist() == ["OKC", "MIL"]


def test_schedule_2014_15_old_season_parses():
    df = S.parse_schedule(load("schedule_2014-15.json.gz"), "2014-15")
    assert set(df.game_type) == {"PRE", "RS", "PO"}
    assert df.attrs["report"]["cup_final_games"] == 0  # no Cup before 2023-24
    assert set(df.postponed_status) == {"A"}  # old seasons use 'A', new 'N'


def _sched_game(raw, gid):
    return next(x for gd in raw["leagueSchedule"]["gameDates"] for x in gd["games"]
                if x["gameId"] == gid)


def test_schedule_non_nba_team_in_regular_season_raises():
    raw = load("schedule_2026-27.json.gz")
    g = _sched_game(raw, "0022600001")
    g["awayTeam"]["teamId"], g["awayTeam"]["teamTricode"] = 50015, "LON"
    with pytest.raises(ValueError, match="non-NBA"):
        S.parse_schedule(raw, "2026-27")


def test_schedule_tricode_must_match_team_id():
    raw = load("schedule_2026-27.json.gz")
    g = _sched_game(raw, "0022600001")
    g["homeTeam"]["teamTricode"] = "XXX"
    with pytest.raises(ValueError, match="tricode"):
        S.parse_schedule(raw, "2026-27")


def test_schedule_tip_date_mismatch_raises():
    raw = load("schedule_2026-27.json.gz")
    _sched_game(raw, "0022600001")["gameDateTimeUTC"] = "2026-10-22T23:00:00Z"
    with pytest.raises(ValueError, match="gameDate says"):
        S.parse_schedule(raw, "2026-27")


def test_schedule_wrong_season_raises():
    with pytest.raises(ValueError, match="seasonYear"):
        S.parse_schedule(load("schedule_2026-27.json.gz"), "2025-26")


def test_schedule_missing_cup_final_raises():
    raw = load("schedule_2024-25.json.gz")
    for gd in raw["leagueSchedule"]["gameDates"]:
        gd["games"] = [g for g in gd["games"] if not g["gameId"].startswith("006")]
    with pytest.raises(ValueError, match="Cup final"):
        S.parse_schedule(raw, "2024-25")


def test_schedule_unknown_prefix_raises():
    raw = load("schedule_2026-27.json.gz")
    _sched_game(raw, "0022600001")["gameId"] = "0092600001"
    with pytest.raises(ValueError, match="prefix"):
        S.parse_schedule(raw, "2026-27")


def test_schedule_2019_20_bubble_and_paris_flagged_neutral():
    """The API says isNeutral=false for every Orlando bubble game AND for the
    Paris game (Jan 2020). The bubble is flagged by date, Paris by arena city
    (a CHA home game outside Charlotte)."""
    raw = load("schedule_2019-20.json.gz")
    assert not any(g["isNeutral"] for gd in raw["leagueSchedule"]["gameDates"] for g in gd["games"])
    df = S.parse_schedule(raw, "2019-20").set_index("game_id")
    assert not df.loc["0021900969", "is_neutral"]          # 2020-03-11, pre-hiatus
    assert df.loc["0021901231", "is_neutral"]               # 2020-07-30 seeding game
    assert df.loc["0051900111", "is_neutral"] and df.loc["0051900111", "game_type"] == "PI"
    assert df.loc["0041900111", "is_neutral"]
    assert df.loc["0021900669", "is_neutral"] and df.loc["0021900669", "arena_city"] == "Paris"
    # The bubble is not a "relocation": only Paris is listed as one.
    assert [r["game_id"] for r in df.attrs["report"]["relocated_games"]] == ["0021900669"]


def test_schedule_city_rule_flags_pre_2024_neutral_sites():
    """2023-24: the Cup semifinals and final in Las Vegas, Mexico City and
    Paris are all isNeutral=false in the source; the Spurs' Austin game is a
    HOME game (the source itself says so from 2024-25 on)."""
    raw = load("schedule_2023-24.json.gz")
    assert not any(g["isNeutral"] for gd in raw["leagueSchedule"]["gameDates"] for g in gd["games"])
    df = S.parse_schedule(raw, "2023-24").set_index("game_id")
    for gid in ("0022301229", "0022301230", "0062300001", "0022300172", "0022300527"):
        assert df.loc[gid, "is_neutral"], gid
    assert not df.loc["0022300965", "is_neutral"] and df.loc["0022300965", "arena_city"] == "Austin"
    assert not df.loc["0022300061", "is_neutral"]
    assert df.attrs["report"]["neutral_by_city_only"] == 5


def test_schedule_city_rule_agrees_with_source_where_it_labels():
    # 2024-25: Mexico City and the Las Vegas Cup final, both isNeutral=true.
    r = S.parse_schedule(load("schedule_2024-25.json.gz"), "2024-25").attrs["report"]
    assert len(r["relocated_games"]) == 2 and r["neutral_by_city_only"] == 0
    assert all(g["source_neutral"] for g in r["relocated_games"])


def test_schedule_new_city_spelling_is_caught(monkeypatch):
    raw = load("schedule_2023-24.json.gz")
    # Why the guard exists: one unknown spelling turns a home game neutral.
    _sched_game(raw, "0022300061")["arenaCity"] = "Denver, CO"
    assert S.parse_schedule(raw, "2023-24").set_index("game_id").loc["0022300061", "is_neutral"]
    # LAL has two relocated home games here (Las Vegas semifinal and final).
    monkeypatch.setattr(S, "MAX_RELOCATED_HOME_GAMES", 1)
    with pytest.raises(ValueError, match="LAL home games outside"):
        S.parse_schedule(load("schedule_2023-24.json.gz"), "2023-24")


def test_schedule_same_team_both_sides_raises():
    raw = load("schedule_2026-27.json.gz")
    g = _sched_game(raw, "0022600001")
    g["awayTeam"]["teamId"] = g["homeTeam"]["teamId"]
    g["awayTeam"]["teamTricode"] = g["homeTeam"]["teamTricode"]
    with pytest.raises(ValueError, match="same team on both sides"):
        S.parse_schedule(raw, "2026-27")


def test_schedule_game_date_est_must_match_game_date():
    raw = load("schedule_2026-27.json.gz")
    _sched_game(raw, "0022600001")["gameDateEst"] = "2026-10-21T00:00:00Z"
    with pytest.raises(ValueError, match="gameDateEst"):
        S.parse_schedule(raw, "2026-27")


def test_is_bubble():
    d = pd.Series(pd.to_datetime(["2020-03-11", "2020-07-30", "2020-10-11"]))
    assert S.is_bubble("2019-20", d).tolist() == [False, True, True]
    assert not S.is_bubble("2020-21", d).any()


# ── players ──────────────────────────────────────────────────────────────────
def test_parse_height():
    assert S.parse_height("6-10") == 82.0 and S.parse_height("7-4") == 88.0
    assert np.isnan(S.parse_height(None)) and np.isnan(S.parse_height(""))
    for bad in ("6-13", "610", "6'10\"", "9-0"):
        with pytest.raises(ValueError):
            S.parse_height(bad)


def test_player_index_current_season():
    df = S.parse_player_index(load("playerindex_2026-27.json.gz"), "2026-27")
    assert list(df.columns) == S.PLAYER_INDEX_COLUMNS
    assert df.team.nunique() == 30 and not df.person_id.duplicated().any()
    yang = df[df.person_id == 1642905].iloc[0]
    assert (yang.first_name, yang.last_name, yang.display_name) == ("Yang", "Hansen", "Yang Hansen")
    jok = df[df.display_name == "Nikola Jokić"].iloc[0]
    assert jok.height_in == 83.0 and jok.team == "DEN"
    assert df.two_way.dtype == bool and df.two_way.any()
    assert df.attrs["report"]["null_position"] == df.position.isna().sum() > 0


def test_player_index_past_season_is_not_a_roster():
    raw = load("playerindex_2024-25.json.gz")
    with pytest.raises(ValueError, match="TO_YEAR"):
        S.parse_player_index(raw, "2024-25")
    df = S.parse_player_index(raw, "2024-25", strict=False)
    r = df.attrs["report"]
    assert r["excluded_off_roster"] > 0 and r["kept"] + r["excluded_off_roster"] == r["raw_rows"]


def test_player_index_must_cover_30_teams():
    raw = load("playerindex_2026-27.json.gz")
    h, rs = rows(raw)
    rs[:] = [r for r in rs if r[h.index("TEAM_ABBREVIATION")] != "NYK"]
    with pytest.raises(ValueError, match="cover 29 teams, not 30"):
        S.parse_player_index(raw, "2026-27")


def test_player_index_null_two_way_status_raises():
    """SUPPLEMENTAL_STATUS is 0/1 on every rostered row; a null used to read
    as 'not two-way' in silence."""
    raw = load("playerindex_2026-27.json.gz")
    h, rs = rows(raw)
    rs[0][h.index("SUPPLEMENTAL_STATUS")] = None
    with pytest.raises(ValueError, match="SUPPLEMENTAL_STATUS"):
        S.parse_player_index(raw, "2026-27")


def test_player_index_empty_raises():
    # Why no row-count guard on the index table: an empty pull cannot parse.
    raw = load("playerindex_2026-27.json.gz")
    rows(raw)[1].clear()
    with pytest.raises(ValueError):
        S.parse_player_index(raw, "2026-27")


def test_player_index_bad_weight_raises():
    raw = load("playerindex_2026-27.json.gz")
    h, rs = rows(raw)
    rs[0][h.index("WEIGHT")] = "heavy"
    with pytest.raises(ValueError, match="non-numeric"):
        S.parse_player_index(raw, "2026-27")


def test_team_roster_current_and_old():
    nyk = S.parse_team_roster(load("roster_2026-27_NYK.json.gz"), "2026-27", "NYK")
    assert list(nyk.columns) == S.ROSTER_COLUMNS and set(nyk.team) == {"NYK"}
    assert nyk.birth_date.dtype == "datetime64[ns]" and nyk.birth_date.notna().all()
    akins = nyk[nyk.person_id == 1643022].iloc[0]
    assert akins.exp == 0 and akins.two_way == True  # noqa: E712 (nullable boolean)
    assert str(nyk.two_way.dtype) == "boolean"
    assert akins.how_acquired.startswith("Signed on")  # text, never a date
    sas = S.parse_team_roster(load("roster_2014-15_SAS.json.gz"), "2014-15", "SAS")
    kawhi = sas[sas.person_id == 202695].iloc[0]
    assert kawhi.birth_date == pd.Timestamp("1991-06-29") and kawhi.height_in == 79.0
    # Old seasons have no SUPPLEMENTAL_STATUS: unknown, not "not two-way".
    assert sas.two_way.isna().all() and sas.how_acquired.isna().all()


def test_past_season_roster_withholds_current_record_fields():
    """HOW_ACQUIRED is today's record, filled only for players still linked to
    the team (2024-25 LAL: 11 of 17, none for the six who left), so for a past
    season both it and the two-way flag are withheld and counted."""
    raw = load("roster_2024-25_LAL.json.gz")
    h, rs = rows(raw)
    assert sum(r[h.index("HOW_ACQUIRED")] is not None for r in rs) == 11
    lal = S.parse_team_roster(raw, "2024-25", "LAL")
    assert lal.how_acquired.isna().all() and lal.two_way.isna().all()
    rep = lal.attrs["report"]
    assert rep["how_acquired_withheld"] == 11 and not rep["point_in_time_fields"]
    nyk = S.parse_team_roster(load("roster_2026-27_NYK.json.gz"), "2026-27", "NYK")
    assert nyk.how_acquired.notna().all() and nyk.two_way.notna().all()
    assert nyk.attrs["report"]["how_acquired_withheld"] == 0


def test_team_roster_wrong_team_or_season_raises():
    raw = load("roster_2026-27_NYK.json.gz")
    with pytest.raises(ValueError, match="payload is for"):
        S.parse_team_roster(raw, "2026-27", "BOS")
    with pytest.raises(ValueError, match="SEASON"):
        S.parse_team_roster(raw, "2025-26", "NYK")


# ── fetch / raw cache policy (offline: stats_get is faked) ───────────────────
@pytest.fixture
def fake_net(tmp_path, monkeypatch):
    calls = []
    payloads = load("logs_2024-25_RS.json.gz")

    def fake_get(endpoint, params, timeout=45.0, retries=3):
        calls.append((endpoint, dict(params)))
        if endpoint == "scheduleleaguev2":
            return load("schedule_2026-27.json.gz")
        if params.get("Season") == "1999-00":
            raise http.Blocked("https://stats.nba.com/stats/" + endpoint, "read timeout")
        if params.get("Season") in ("2023-24", "2026-27") and params.get("SeasonType") == "Playoffs":
            return copy.deepcopy(load("empty_2026-27_RS.json.gz")["lgl_T_RS"])
        if params.get("Season") == "2000-01":
            return {"message": "An error has occurred."}
        # A raw payload WITH a rank column, to check compaction.
        p = copy.deepcopy(payloads["lgl_T_RS"])
        s = p["resultSets"][0]
        s["headers"] = s["headers"] + ["PTS_RANK"]
        s["rowSet"] = [r + [1] for r in s["rowSet"]]
        return p

    monkeypatch.setattr(S, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(S.http, "stats_get", fake_get)
    monkeypatch.setattr(S, "_fetched_this_run", set())
    monkeypatch.setattr(S.util, "today_et", lambda: pd.Timestamp("2026-10-08").date())
    return calls


def test_completed_season_fetched_once_and_compacted(fake_net, tmp_path):
    p1 = S.fetch_lgl("2024-25", "RS", "T")
    S._fetched_this_run.clear()  # a new process
    p2 = S.fetch_lgl("2024-25", "RS", "T")
    assert len(fake_net) == 1 and p1 == p2
    assert "PTS_RANK" not in p1["resultSets"][0]["headers"]
    path = tmp_path / "raw" / "statsnba" / "2024-25" / "lgl_T_RS.json.gz"
    env = json.loads(gzip.decompress(path.read_bytes()))
    assert env["fetched_at"] > 0 and env["data"]["params"]["SeasonType"] == "Regular Season"
    assert env["data"]["params"]["PlayerOrTeam"] == "T"


def test_live_season_refetched_once_per_process(fake_net):
    S.fetch_lgl("2026-27", "RS", "T")
    S.fetch_lgl("2026-27", "RS", "T")  # same process: memoised
    assert len(fake_net) == 1
    S._fetched_this_run.clear()  # next run: the current season grows, refetch
    S.fetch_lgl("2026-27", "RS", "T")
    assert len(fake_net) == 2
    S._fetched_this_run.clear()
    S.fetch_lgl("2026-27", "RS", "T", refresh=False)
    assert len(fake_net) == 2


def test_force_refresh_completed_season(fake_net):
    S.fetch_lgl("2024-25", "RS", "T")
    S._fetched_this_run.clear()
    S.fetch_lgl("2024-25", "RS", "T", refresh=True)
    assert len(fake_net) == 2


def test_blocked_propagates_and_writes_nothing(fake_net, tmp_path):
    with pytest.raises(http.Blocked, match="1999-00"):
        S.fetch_lgl("1999-00", "RS", "T")
    assert not list(tmp_path.rglob("*.gz"))


def test_empty_completed_season_is_not_cached(fake_net, tmp_path):
    """An empty answer for a finished season is an outage: raise, and do not
    leave it in the raw cache where a resumed run would trust it."""
    with pytest.raises(http.BadPayload, match="0 rows"):
        S.fetch_lgl("2023-24", "Playoffs", "T")
    assert not list(tmp_path.rglob("*.gz"))
    # The same empty answer for the live season, whose schedule lists no
    # playoff game yet, is the pre-season state: kept. Deciding that read
    # the schedule (one call), and only because the answer was empty.
    S.fetch_lgl("2026-27", "Playoffs", "T")
    assert len(list(tmp_path.rglob("lgl_T_PO.json.gz"))) == 1
    assert [c[0] for c in fake_net].count("scheduleleaguev2") == 1


def test_cache_only_never_fetches(offline_cache):
    """refresh=False (--rebuild) used to fetch whatever was missing."""
    with pytest.raises(S.CacheMiss, match="cache-only"):
        S.fetch_lgl("2024-25", "RS", "T", refresh=False)
    with pytest.raises(S.CacheMiss):
        S.player_games("2024-25", "RS", refresh=False)
    assert not list(offline_cache.rglob("*.gz"))


def test_no_play_in_calls_before_2019_20(offline_cache):
    """2017-18 PI cost 5 live calls that all returned 0 rows."""
    p, t = S.player_games("2017-18", "PlayIn"), S.team_games("2017-18", "PI")
    assert p.empty and list(p.columns) == S.PLAYER_GAME_COLUMNS
    assert t.empty and list(t.columns) == S.TEAM_GAME_COLUMNS
    assert "no call made" in p.attrs["report"]["skipped"]
    assert S.planned_log_calls(["2017-18"], ["PI"]) == 0
    assert S.planned_log_calls(["2017-18"], ["RS", "PI"]) == 6  # 5 logs + schedule


def test_unrecognised_payload_shape_not_cached(fake_net, tmp_path):
    with pytest.raises(http.BadPayload, match="shape"):
        S.fetch_lgl("2000-01", "RS", "T")
    assert not list(tmp_path.rglob("*.gz"))


def test_call_budget_stops_before_the_call(fake_net, monkeypatch):
    monkeypatch.setattr(S, "_calls", 0)
    S.set_call_budget(1)
    try:
        S.fetch_lgl("2024-25", "RS", "T")
        with pytest.raises(S.CallBudgetExceeded):
            S.fetch_lgl("2024-25", "RS", "P")
        assert len(fake_net) == 1
        S._fetched_this_run.clear()
        S.fetch_lgl("2024-25", "RS", "T")  # cached: free even at the budget
    finally:
        S.set_call_budget(None)


def test_planned_log_calls(offline_cache):
    assert S.planned_log_calls(["2024-25"], ["RS", "PO"]) == 11  # 10 logs + schedule
    _seed_raw(offline_cache, "2024-25", load("logs_2024-25_RS.json.gz"))
    assert S.planned_log_calls(["2024-25"], ["RS", "PO"]) == 6
    _seed_raw(offline_cache, "2024-25", {"schedule": load("schedule_2024-25.json.gz")})
    assert S.planned_log_calls(["2024-25"], ["RS", "PO"]) == 5
    assert S.planned_log_calls(["2024-25"], ["RS"], force=True) == 6
    assert S.planned_log_calls(["2026-27"], ["RS"]) == 6  # live: always refetched
    assert S.planned_log_calls(["2024-25", "2026-27"], ["RS", "PO"], rebuild=True) == 0


def test_measure_type_is_always_explicit(fake_net):
    S.fetch_pgl("2024-25", "PlayIn", "Advanced")
    ep, params = fake_net[-1]
    assert ep == "playergamelogs" and params["MeasureType"] == "Advanced"
    assert params["SeasonType"] == "PlayIn" and params["OpponentTeamID"] == ""
    assert all(v is not None for v in params.values())


def test_no_direct_http_imports():
    """All network access goes through bucketlib.http."""
    repo = Path(__file__).resolve().parent.parent
    import re
    pat = re.compile(r"^\s*(import|from)\s+(requests|urllib|curl_cffi|http\.client)\b", re.M)
    files = [repo / "bucketlib/sources/statsnba.py",
             *(repo / "scripts" / f for f in ("build_logs.py", "build_schedule.py", "build_players.py"))]
    for f in files:
        assert f.exists(), f
        assert not pat.search(f.read_text()), f.name


# ── live (opt-in: BUCKETVALUE_NETWORK=1) ─────────────────────────────────────
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("BUCKETVALUE_NETWORK") != "1",
                    reason="live stats.nba.com; set BUCKETVALUE_NETWORK=1")
def test_live_current_schedule_and_index():
    sch = S.schedule(S.current_season(), refresh=True)
    assert (sch.game_type == "RS").sum() >= 1200
    idx = S.player_index(S.current_season(), refresh=True)
    assert idx.team.nunique() == 30 and len(idx) >= 400


# ── script support ───────────────────────────────────────────────────────────
def _script(name):
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_season_range():
    assert S.parse_season_range("2014-15:2016-17") == ["2014-15", "2015-16", "2016-17"]
    assert S.parse_season_range("2014-15,2024-25:2025-26") == ["2014-15", "2024-25", "2025-26"]
    for bad in ("2014-16", "", "2016-17:2014-15"):
        with pytest.raises(ValueError):
            S.parse_season_range(bad)


def test_write_tidy_stale_beats_empty(tmp_path):
    p = tmp_path / "x_2026-27_v1.parquet"
    full = pd.DataFrame({"a": [1, 2, 3]})
    assert S.write_tidy(full, p, {"r": np.int64(3), "fetched_at": 5.0}, live=True)[0]
    ok, msg = S.write_tidy(full.iloc[:0], p, {"fetched_at": 6.0}, live=True)
    assert not ok and "stale beats empty" in msg and len(pd.read_parquet(p)) == 3
    assert S.write_tidy(full.iloc[:1], p, {"fetched_at": 7.0}, live=False)[0]  # --force rewrites


def test_write_tidy_stamps_the_input_age_not_the_write_time(tmp_path):
    """A --rebuild of week-old payloads is week-old data: fetched_at is the
    stamp the caller passes (the oldest raw input), built_at the write time."""
    p = tmp_path / "x_2024-25_v1.parquet"
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="fetched_at"):
        S.write_tidy(df, p, {}, live=False)
    S.write_tidy(df, p, {"fetched_at": 1_600_000_000.0}, live=False)
    m = S.tidy_meta(p)
    assert m["fetched_at"] == 1_600_000_000.0 and m["built_at"] > 1_700_000_000
    S.write_tidy(df, p, {"fetched_at": None}, live=False)  # unknown age: never fresh
    assert S.tidy_meta(p)["fetched_at"] is None


def _seed_raw(root: Path, season: str, payloads: dict, fetched_at: float | None = None):
    """Lay fixture payloads out exactly as _get_raw caches them (optionally
    with a chosen fetch stamp)."""
    from bucketlib.cache import write_json
    for name, payload in payloads.items():
        path = root / "raw" / "statsnba" / season / f"{name}.json.gz"
        write_json(path, {"endpoint": "fixture", "params": {}, "payload": payload},
                   stamped=True, gz=True)
        if fetched_at is not None:
            env = json.loads(gzip.decompress(path.read_bytes()))
            env["fetched_at"] = fetched_at
            path.write_bytes(gzip.compress(json.dumps(env).encode(), mtime=0))


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("network call in an offline test")
    monkeypatch.setattr(S, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(S.http, "stats_get", no_network)
    monkeypatch.setattr(S, "_fetched_this_run", set())
    monkeypatch.setattr(S.util, "today_et", lambda: pd.Timestamp("2026-10-08").date())
    return tmp_path


def test_build_logs_end_to_end_resumable(offline_cache, capsys):
    _seed_raw(offline_cache, "2024-25", load("logs_2024-25_RS.json.gz"))
    _seed_raw(offline_cache, "2024-25", {"schedule": load("schedule_2024-25.json.gz")})
    bl = _script("build_logs")
    assert bl.main(["--seasons", "2024-25", "--types", "RS"]) == 0
    out = capsys.readouterr().out
    assert "wrote player_games_2024-25_v1.parquet (101 rows)" in out and "COVERAGE" in out
    df = pd.read_parquet(S.tidy_path("player_games", "2024-25"))
    assert len(df) == 101 and list(df.columns) == S.PLAYER_GAME_COLUMNS
    assert df.loc[df.game_id == "0022400147", "neutral"].all()
    rep = S.tidy_meta(S.tidy_path("team_games", "2024-25"))["reports"]["RS"]["team"]
    assert rep["neutral_source"] == "logs+schedule" and rep["neutral_logs_not_schedule"] == []
    assert bl.main(["--seasons", "2024-25", "--types", "RS"]) == 0
    assert "done, skipped" in capsys.readouterr().out
    assert S.load_progress("build_logs")["2024-25"]["status"] == "done"


def test_build_logs_failure_is_reported_not_written(offline_cache, capsys):
    d = load("logs_2024-25_RS.json.gz")
    edit(d["lgl_T_RS"], {"GAME_ID": "0022400330"}, MIN=250)
    _seed_raw(offline_cache, "2024-25", {**d, "schedule": load("schedule_2024-25.json.gz")})
    assert _script("build_logs").main(["--seasons", "2024-25", "--types", "RS"]) == 1
    assert "FAILED (not written)" in capsys.readouterr().out
    assert not S.tidy_path("player_games", "2024-25").exists()


def test_build_logs_narrow_rebuild_keeps_other_types(offline_cache):
    bl = _script("build_logs")
    path = S.tidy_path("player_games", "2024-25")
    old = pd.DataFrame({"season_type": ["RS", "PO", "PO"], "x": [1, 2, 3]})
    S.write_tidy(old, path, {"types": ["RS", "PO"], "reports": {"RS": {}, "PO": {"k": 1}},
                             "fetched_at": 100.0}, live=False)
    new = pd.DataFrame({"season_type": ["RS", "RS"], "x": [7, 8]})
    df, meta = bl.merge_with_existing(path, [new], ["RS"], {"RS": {"k": 2}}, "2024-25", 200.0)
    assert df.season_type.tolist() == ["RS", "RS", "PO", "PO"] and df.x.tolist() == [7, 8, 2, 3]
    assert meta["types"] == ["RS", "PO"] and meta["reports"]["PO"] == {"k": 1}
    assert meta["fetched_at"] == 100.0  # the kept PO rows are the oldest data


def test_build_logs_rebuild_is_cache_only(offline_cache, capsys):
    """'--rebuild --max-calls 0' used to stop with 'live-call budget 0
    reached': it fetched every missing payload."""
    try:
        assert _script("build_logs").main(["--seasons", "2016-17", "--types", "RS",
                                           "--rebuild", "--max-calls", "0"]) == 1
    finally:
        S.set_call_budget(None)
    out = capsys.readouterr().out
    assert "at most 0 live calls planned" in out and "MISSING RAW" in out
    assert S.load_progress("build_logs")["2016-17"]["status"] == "missing_raw"


def test_build_logs_tables_carry_the_oldest_raw_stamp(offline_cache):
    d = load("logs_2024-25_RS.json.gz")
    _seed_raw(offline_cache, "2024-25", d, fetched_at=1_700_000_000.0)
    _seed_raw(offline_cache, "2024-25", {"pgl_Advanced_RS": d["pgl_Advanced_RS"]},
              fetched_at=1_600_000_000.0)
    _seed_raw(offline_cache, "2024-25", {"schedule": load("schedule_2024-25.json.gz")},
              fetched_at=1_650_000_000.0)
    assert _script("build_logs").main(["--seasons", "2024-25", "--types", "RS", "--rebuild"]) == 0
    for table in ("player_games", "team_games"):
        m = S.tidy_meta(S.tidy_path(table, "2024-25"))
        assert m["fetched_at"] == 1_600_000_000.0 and m["built_at"] > 1_700_000_000.0


def test_build_schedule_rebuild_is_cache_only_and_budget_is_not_a_block(offline_cache, capsys):
    bs = _script("build_schedule")
    try:
        assert bs.main(["--seasons", "2016-17", "--rebuild"]) == 1
        assert S.load_progress("build_schedule")["2016-17"]["status"] == "missing_raw"
        assert bs.main(["--seasons", "2016-17", "--max-calls", "0"]) == 1
        assert S.load_progress("build_schedule")["2016-17"]["status"] == "budget"
    finally:
        S.set_call_budget(None)
    _seed_raw(offline_cache, "2024-25", {"schedule": load("schedule_2024-25.json.gz")},
              fetched_at=1_650_000_000.0)
    assert bs.main(["--seasons", "2024-25", "--rebuild"]) == 0
    assert S.tidy_meta(S.tidy_path("schedule", "2024-25"))["fetched_at"] == 1_650_000_000.0


def test_build_players_survives_a_bad_cached_roster(offline_cache, capsys):
    """Step 3 caught only ValueError, so a BadPayload from a cached roster
    crashed the run before its coverage report."""
    _seed_raw(offline_cache, "2014-15", {"roster_SAS": {"message": "An error has occurred."}})
    rc = _script("build_players").main(["--seasons", "2014-15", "--teams", "SAS", "--no-index"])
    out = capsys.readouterr().out
    assert rc == 1 and "COVERAGE (team rosters)" in out and "BadPayload" in out


def test_build_players_spread_and_bio():
    bp = _script("build_players")
    pairs = [(s, t) for s in ("a", "b", "c") for t in range(30)]
    got = bp.spread(pairs, 3)
    assert got[0] == ("a", 0) and got[-1] == ("c", 29) and len(got) == 3
    r1 = S.parse_team_roster(load("roster_2014-15_SAS.json.gz"), "2014-15", "SAS")
    r2 = r1.copy()
    r2["season"] = "2015-16"
    r2.loc[r2.person_id == 202695, "birth_date"] = pd.Timestamp("1991-06-30")
    bio, rep = bp.build_bio(pd.concat([r1, r2, r1.assign(season="2016-17")]))
    assert rep["birth_date_conflicts"] == 1 and rep["persons"] == r1.person_id.nunique()
    kawhi = bio[bio.person_id == 202695].iloc[0]
    assert kawhi.birth_date == pd.Timestamp("1991-06-29")  # modal value kept
    assert (kawhi.first_season, kawhi.last_season, kawhi.team_seasons) == ("2014-15", "2016-17", 3)
