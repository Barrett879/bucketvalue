"""bucketlib.props.compare_books does each game's work once (round BKS,
2026-11-15; docs/serving_design.md section 11), and none of it moves a number.

compare_books cuts each game's projection rows and indexes them once
(props._projection_index); every book's compare() for that game shares the
index (compare's keyword-only `_index`). It compares the lines off a shallow
copy that carries no attrs, so pandas no longer deep-copies prepare_feed's
report into every frame and column cut from them. compare() reads each stat's
reliability once per call.

1. Each game's projection rows become records once per game, however many
   books posted it (0293ddc: once per (book, game), 9 times for 2 games).
2. The feed report is deep-copied at most twice per board build (0293ddc:
   390 times on these lines).
3. compare_books' tables and metas equal compare() run alone on each (book,
   game), as 0293ddc ran it: that pair's rows, report still on them, against
   that game's projection slice, over at least two stats. A second build on
   a moved frame is priced on that frame alone (no index outlives a build).
4. compare() never writes a projection row: rows handed out read-only give
   no row error and the same tables.
5. One compare() call holding lines of stats in different reliability tiers
   gives each row its own stat's tier, edge and note, and the next call reads
   the table afresh. Before this file only one test (in test_props_feed)
   caught a memo that gave every stat of a call one tier.
6. A game whose projection slice or index cannot be built costs only that
   game's (book, game) pairs, as a compare() that raises does: the slice and
   the index are built inside the per-(book, game) try.
7. Two projection rows for one (player, game): the last one prices his
   lines, alone and through the shared index alike (the rule moved from
   compare() into _projection_index unchanged).

The world is test_props_feed's: the serve fixture's real 2026-03-12 slate,
its roster and projections, and the CONSTRUCTED odds fixtures through the real
oddsapi parser (two games, five books, 53 lines, four markets). The 2 games
and 9 (book, game) pairs are test_props_feed's POSTED. No clock, no network.
"""
from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from bucketlib import props

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fixtures" / "serve"))
import serve_harness as H  # noqa: E402
import test_props_feed as PF  # noqa: E402  (the feed frame through the real parser)

EARLY, LATE = H.EARLY, H.LATE              # PHI at DET 7 PM ET, DEN at SAS 9 PM ET


@pytest.fixture(scope="module")
def world():
    return H.load()


@pytest.fixture(scope="module")
def provider(world):
    return props.MeanDispersionProvider(world.proj)


def _lines(world) -> pd.DataFrame:
    """The day's feed lines as the page compares them (prepare_feed), with
    prepare_feed's report on their attrs."""
    lines, report = props.prepare_feed(PF.feed(world), world.roster, world.proj)
    assert report["error"] is None and not lines.empty
    assert lines.attrs.get("feed_report")
    return lines


def _is_projection_frame(df: pd.DataFrame) -> bool:
    """A projection frame as compare() indexes it: person and game columns
    and no book column (the lines and the tables carry one)."""
    return "book" not in df.columns and {"person_id", "game_id"} <= set(df.columns)


def _same_meta(got: dict, want: dict) -> None:
    assert set(got) == set(want)
    for k, v in want.items():
        if isinstance(v, pd.DataFrame):
            pd.testing.assert_frame_equal(got[k], v)
        else:
            assert got[k] == v, k


# -- 1. Records once per game -----------------------------------------------------------
def test_each_games_projection_rows_become_records_once_per_game(world, provider, monkeypatch):
    """0293ddc turned the game's whole projection slice into records in every
    compare() call: 9 times for these 2 games. Now once per game, each time
    that game's own rows."""
    lines = _lines(world)
    assert lines.groupby(["book", "game_id"]).ngroups == 9 and lines["game_id"].nunique() == 2
    real = pd.DataFrame.to_dict
    cut = []

    def counting(self, *a, **k):
        if _is_projection_frame(self):
            cut.append(tuple(sorted(set(self["game_id"].astype(str)))))
        return real(self, *a, **k)
    with monkeypatch.context() as mp:
        mp.setattr(pd.DataFrame, "to_dict", counting)
        ft = props.compare_books(lines, provider, world.proj, world.roster)
    assert len(ft.tables) == 9
    assert sorted(cut) == [(EARLY,), (LATE,)], cut


# -- 2. The report is not copied per column ---------------------------------------------
class _CountingReport(dict):
    """A feed report that counts its deep copies."""
    copies = 0

    def __deepcopy__(self, memo):
        type(self).copies += 1
        return _CountingReport(copy.deepcopy(dict(self), memo))


def test_the_feed_report_is_deep_copied_at_most_twice_per_board(world, provider, monkeypatch):
    """pandas deep-copies a frame's non-empty attrs into every frame and
    column cut from it, so 0293ddc copied the report about 40 times per
    (book, game): 390 here. compare() reads no attrs. The report handed back
    is the one read first, and the caller's lines keep theirs."""
    lines = _lines(world)
    report = dict(lines.attrs["feed_report"])
    monkeypatch.setattr(_CountingReport, "copies", 0)
    lines.attrs = {"feed_report": _CountingReport(report)}
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    assert len(ft.tables) == 9
    assert ft.report == report and type(ft.report) is dict
    assert lines.attrs["feed_report"] == report
    assert _CountingReport.copies <= 2, _CountingReport.copies


# -- 3. The same tables as compare() alone ----------------------------------------------
def test_compare_books_equals_compare_alone_on_each_book_and_game(world, provider):
    """Each (book, game) as 0293ddc compared it: that pair's rows (prepare_feed's
    report still on them, as the old path passed them) against that game's
    projection slice, with compare()'s own index and no shared one. Every
    table is the same frame and every meta the same, field by field, over at
    least two stats, so each stat's own tier is exercised."""
    lines = _lines(world)
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    groups = lines.groupby(["book", "game_id"], sort=False).indices
    assert set(ft.tables) == set(ft.metas) == set(groups) and len(groups) == 9
    stats = set()
    for (book, gid), idx in groups.items():
        rows = lines.iloc[idx].reset_index(drop=True)
        assert rows.attrs.get("feed_report")
        table, meta = props.compare(rows, provider, props._game_projections(world.proj, gid),
                                    world.roster)
        assert len(table) == meta["posted"] > 0, (book, gid)
        pd.testing.assert_frame_equal(ft.tables[(book, gid)], table)
        _same_meta(ft.metas[(book, gid)], meta)
        stats |= set(table["stat"])
    assert len(stats) >= 2, stats


def test_each_board_build_indexes_its_own_projection_frame(world, provider):
    """Nothing of one board build reaches the next. After a build on the
    fixture's frame, a build on a frame whose PHI at DET means moved (as a
    new injury report or a mark-out moves them) prices every line on the new
    means, exactly as compare() alone does: each PHI at DET table moves, and
    no DEN at SAS table does."""
    lines = _lines(world)
    first = props.compare_books(lines, provider, world.proj, world.roster)
    moved = world.proj.copy()
    early = moved["game_id"].astype(str) == EARLY
    for stat in ("pts", "reb", "ast"):
        moved.loc[early, stat] = moved.loc[early, stat] + 1.5
    second = props.compare_books(lines, provider, moved, world.roster)
    groups = lines.groupby(["book", "game_id"], sort=False).indices
    changed = set()
    for (book, gid), idx in groups.items():
        rows = lines.iloc[idx].reset_index(drop=True)
        table, _ = props.compare(rows, provider, props._game_projections(moved, gid), world.roster)
        pd.testing.assert_frame_equal(second.tables[(book, gid)], table)
        if not first.tables[(book, gid)]["mean"].equals(table["mean"]):
            changed.add((book, gid))
    assert changed == {k for k in groups if k[1] == EARLY} and len(changed) == 5


# -- 4. Shared rows are read, never written ---------------------------------------------
def test_compare_never_writes_a_projection_row(world, provider, monkeypatch):
    """compare_books hands every book's compare() the same projection rows, so
    a write to one would leak into the next book's numbers. Every projection
    frame's rows come out of DataFrame.to_dict read-only
    (types.MappingProxyType): compare() built them there on 0293ddc, and
    props._projection_index does now, for the index compare_books shares. A
    write anywhere in compare() or what it calls would raise and count as a
    row error. None does, and the tables are the same."""
    lines = _lines(world)
    want = props.compare_books(lines, provider, world.proj, world.roster)
    real = pd.DataFrame.to_dict
    handed = []

    def read_only(self, *a, **k):
        out = real(self, *a, **k)
        if _is_projection_frame(self) and isinstance(out, list):
            handed.append(len(out))
            return [types.MappingProxyType(r) for r in out]
        return out
    with monkeypatch.context() as mp:
        mp.setattr(pd.DataFrame, "to_dict", read_only)
        got = props.compare_books(lines, provider, world.proj, world.roster)
    assert handed and sum(handed) >= len(world.proj)     # every game's rows went out read-only
    assert set(got.tables) == set(want.tables) and len(want.tables) == 9
    for key, table in want.tables.items():
        assert got.metas[key]["errors"] == 0, (key, got.metas[key]["unprojected_why"])
        assert len(table) == want.metas[key]["posted"]
        pd.testing.assert_frame_equal(got.tables[key], table)


# -- 5. Each stat's own tier, read afresh by each call ----------------------------------
def _reliability(tag: str, stats: list[str], tiers: tuple[str, ...]) -> pd.DataFrame:
    """A CONSTRUCTED reliability table over `stats`: tiers in turn from
    `tiers`, a different edge for each stat, and a note naming the table and
    the stat, so no two stats (and no two tables) share a reliability."""
    return pd.DataFrame({"stat": stats,
                         "tier": [tiers[i % len(tiers)] for i in range(len(stats))],
                         "side_edge_pts": [6.0 - 0.25 * i for i in range(len(stats))],
                         "note": [f"table {tag}: {s}" for s in stats]})


def test_each_row_carries_its_own_stats_tier_and_each_call_reads_the_table(
        world, provider, monkeypatch):
    """One compare() call on the (book, game) whose lines hold the most stats:
    every row's tier, edge and note are reliability_for of its own stat (the
    production rule: a combo takes its weakest part), and the call's rows hold
    more than one tier. A second call, after the table changed, gives the new
    table's values: the memo lives for one call."""
    lines = _lines(world)
    ft = props.compare_books(lines, provider, world.proj, world.roster)
    key = max(sorted(ft.tables), key=lambda k: ft.tables[k]["stat"].nunique())
    book, gid = key
    rows = lines[(lines["book"] == book) & (lines["game_id"] == gid)].reset_index(drop=True)
    game = props._game_projections(world.proj, gid)
    called = sorted(set(ft.tables[key]["stat"]))
    assert len(called) >= 2, (key, called)
    every = sorted(set(props.reliability_table()["stat"]) | set(called))
    for tag, tiers in (("A", ("strong", "moderate", "weak")), ("B", ("weak", "strong", "moderate"))):
        table = _reliability(tag, every, tiers)
        monkeypatch.setattr(props, "reliability_table", lambda table=table: table)
        want = {s: props.reliability_for(s) for s in called}
        assert len({w["tier"] for w in want.values()}) >= 2, want
        got, meta = props.compare(rows, provider, game, world.roster)
        assert len(got) == meta["posted"] == len(rows) and meta["errors"] == 0
        assert set(got["stat"]) == set(called) and got["tier"].nunique() >= 2
        for r in got.to_dict(orient="records"):
            w = want[r["stat"]]
            assert (r["tier"], r["tier_edge"], r["tier_note"]) == (w["tier"], w["edge"], w["note"]), \
                (tag, r["stat"], r["tier"], r["tier_note"])
            assert r["tier_note"] == f"table {tag}: {r['stat']}" or r["stat"] in props.COMBO_STATS


# -- 6. A game that cannot be indexed costs only its own pairs ---------------------------
@pytest.mark.parametrize("part", ["_game_projections", "_projection_index"])
def test_a_game_whose_slice_or_index_fails_costs_only_that_games_pairs(world, provider, monkeypatch,
                                                                      part):
    """DEN at SAS's projection slice (or its index) raises: each of its (book,
    game) pairs gets an empty table and every line counted as an internal
    error, the ledger balanced, and every PHI at DET pair is exactly what a
    normal build gives."""
    lines = _lines(world)
    want = props.compare_books(lines, provider, world.proj, world.roster)
    real = getattr(props, part)
    if part == "_game_projections":
        def broken(projections, gid):
            if str(gid) == LATE:
                raise RuntimeError("the slice failed")
            return real(projections, gid)
    else:
        def broken(gp):
            if set(gp["game_id"].astype(str)) == {LATE}:
                raise RuntimeError("the index failed")
            return real(gp)
    monkeypatch.setattr(props, part, broken)
    got = props.compare_books(lines, provider, world.proj, world.roster)
    late = {k for k in want.tables if k[1] == LATE}
    assert set(got.tables) == set(got.metas) == set(want.tables)
    assert 0 < len(late) < len(want.tables) == 9
    for key, table in want.tables.items():
        if key in late:
            posted = want.metas[key]["posted"]
            assert got.tables[key].empty and list(got.tables[key].columns) == list(props.TABLE_COLUMNS)
            _same_meta(got.metas[key], props._failed_meta(posted))
            assert got.metas[key]["errors"] == got.metas[key]["unprojected"] == posted > 0
        else:
            pd.testing.assert_frame_equal(got.tables[key], table)
            _same_meta(got.metas[key], want.metas[key])


# -- 7. The last of two rows for one player and game prices his lines --------------------
def test_a_duplicated_projection_row_prices_on_the_last_one(world, provider):
    """One player's row for his game is given twice, the copy's points mean 3
    higher. Every table equals the one built from a frame holding only the
    LAST of the two rows (the copy when it comes last, the original when it
    comes first), through compare_books' shared index and through compare()
    alone (its own index) alike, and his Points mean is 3 apart between the
    two orders."""
    lines = _lines(world)
    base = props.compare_books(lines, provider, world.proj, world.roster)
    key, rec = next((k, r) for k in sorted(base.tables)
                    for r in base.tables[k].to_dict(orient="records") if r["stat"] == "pts")
    pid, gid = int(rec["person_id"]), str(rec["game_id"])
    proj = world.proj
    mine = (proj["person_id"].astype("int64") == pid) & (proj["game_id"].astype(str) == gid)
    assert mine.sum() == 1
    copy_ = proj[mine].copy()
    copy_["pts"] = copy_["pts"] + 3.0
    moved = proj.copy()
    moved.loc[mine, "pts"] = copy_["pts"].to_numpy()
    book_rows = lines[(lines["book"] == key[0]) & (lines["game_id"] == key[1])].reset_index(drop=True)
    means = []
    for doubled, last_only in ((pd.concat([proj, copy_], ignore_index=True), moved),
                               (pd.concat([copy_, proj], ignore_index=True), proj)):
        got = props.compare_books(lines, provider, doubled, world.roster)
        want = props.compare_books(lines, provider, last_only, world.roster)
        assert set(got.tables) == set(want.tables)
        for k, table in want.tables.items():
            pd.testing.assert_frame_equal(got.tables[k], table)
        alone, _ = props.compare(book_rows, provider, props._game_projections(doubled, gid),
                                 world.roster)
        pd.testing.assert_frame_equal(alone, want.tables[key])
        his = want.tables[key]
        means.append(his.loc[(his["person_id"] == pid) & (his["stat"] == "pts"), "mean"].tolist())
    assert means[0] and len(means[0]) == len(means[1])
    assert [a - b for a, b in zip(*means)] == pytest.approx([3.0] * len(means[0]), abs=1e-6)
