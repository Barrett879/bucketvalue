"""The serve fixture installed as the app's world, for the page tests.

install() points bucketlib.state at a temp state dir holding what the bot
would have committed for 2026-03-12 (the two games' history features, stamped
as pipeline.write_features stamps them, and their projections), makes the
live layer answer from the fixture (schedule, rosters, the real 12:00 PM ET
injury report, the made-up lines) and pins every clock the pages read. The
box score is "unavailable" unless a test serves the real final box of PHI at
DET (serve_box) through the real live.box. No network, no repo state, no
writes outside the temp dir.

The licensed odds feed (design 8.3, m16): install() stubs the live layer's
feed calls (feed_kick, feed_view, prop_lines_memory) and drops the feed
anchor and the cached feed tables, so no page test kicks a walk or reads
another test's feed. With no key (conftest) store.feed_start is "off" before
any of them runs. install_feed() turns the feed on for a page test: a fake
key built at run time, and lines bought by the REAL live walk through the
REAL oddsapi parser from the CONSTRUCTED fixtures in tests/fixtures/oddsapi,
served from live.feed_view.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from bucketlib import cache as C
from bucketlib import http
from bucketlib import live as LV
from bucketlib import pipeline as P
from bucketlib import props_ui, store, util
from bucketlib import state as S
from bucketlib.sources import injury as INJ
from bucketlib.util import ET, UTC

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATE = dt.date(2026, 3, 12)
SEASON = "2025-26"
EARLY = "0022500953"     # PHI at DET, tip 7:00 PM ET: past its freeze point from 6:55
LATE = "0022500958"      # DEN at SAS, tip 9:00 PM ET; Denver had not filed at noon
MORNING = dt.datetime(2026, 3, 12, 12, 20, tzinfo=ET)   # the fixture's own compute time
EVENING = dt.datetime(2026, 3, 12, 19, 0, tzinfo=ET)    # PHI at DET frozen, DEN at SAS live


@dataclass(frozen=True)
class World:
    feats: pd.DataFrame
    sched: pd.DataFrame
    proj: pd.DataFrame
    report: object
    odds: dict
    roster: pd.DataFrame
    board_text: str


def roster_from(feats: pd.DataFrame) -> pd.DataFrame:
    """The fixture's players in cdn.parse_player_index's columns (the live
    roster the pages read for positions and paste resolution)."""
    r = feats.drop_duplicates("person_id")
    first = r["player_name"].str.split(" ", n=1).str[0]
    last = r["player_name"].str.split(" ", n=1).str[1].fillna("")
    return pd.DataFrame({"person_id": r["person_id"].astype("int64").to_numpy(),
                         "first_name": first.to_numpy(), "last_name": last.to_numpy(),
                         "team": r["team"].to_numpy(), "position": "",
                         "two_way": r["two_way"].astype(bool).to_numpy(),
                         "name": r["player_name"].to_numpy()})


@lru_cache(maxsize=1)
def load() -> World:
    feats = pd.read_parquet(HERE / "features.parquet")
    sched = pd.read_parquet(HERE / "schedule.parquet")
    proj = pd.read_parquet(HERE / "projections.parquet")
    report = INJ.parse_report((REPO / "tests" / "fixtures" / "injury"
                               / "Injury-Report_2026-03-12_12_00PM.pdf").read_bytes())
    lines = json.loads((HERE / "lines.json").read_text())
    odds = {(a, h): {**lines[g], "provider": "DraftKings"}
            for g, a, h in zip(sched["game_id"], sched["away"], sched["home"])}
    return World(feats, sched, proj, report, odds, roster_from(feats),
                 (HERE / "board.txt").read_text())


def box_payload() -> dict:
    """The real cdn liveData box score of PHI at DET (final, 2026-03-12)."""
    with gzip.open(HERE / f"box_{EARLY}.json.gz", "rt") as f:
        return json.load(f)


def no_box(game_id):
    raise RuntimeError(f"box:{game_id}: offline in the tests")


def serve_box(monkeypatch) -> None:
    """live.box answers through its REAL code (_cached, the final check,
    cdnbox.player_lines) from the fixture payload; any other game id is
    absent."""
    payload = box_payload()

    def fetch(url, **kw):
        if EARLY not in url:
            raise LV.H.Absent(url, "test: no box for this game")
        return payload
    monkeypatch.setattr(LV, "box", REAL_BOX)
    monkeypatch.setattr(LV.H, "fetch", fetch)
    monkeypatch.setattr(LV, "_good", {})
    monkeypatch.setattr(LV, "_fail", {})


REAL_BOX = LV.box
# The live layer's REAL feed functions, kept before install() stubs them:
# install_feed buys through the real kick and serves the real view.
REAL_FEED_KICK = LV.feed_kick
REAL_FEED_VIEW = LV.feed_view


def pin(monkeypatch, at: dt.datetime) -> None:
    """Every clock a page reads, at `at` (util's, and the names live and
    store imported from it). props_ui's own binding too: no feed text reads
    it (the page passes stamps and its `today`), and nothing else pinned it
    (design 8.3, m1)."""
    u, e = at.astimezone(UTC), at.astimezone(ET)
    for mod in (util, LV):
        monkeypatch.setattr(mod, "now_utc", lambda: u)
        monkeypatch.setattr(mod, "now_et", lambda: e)
    monkeypatch.setattr(store, "now_utc", lambda: u)
    monkeypatch.setattr(props_ui, "now_utc", lambda: u)


def install(monkeypatch, tmp_path: Path, at: dt.datetime = MORNING, report=None) -> World:
    w = load()
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(S, "SEED_DIR", tmp_path / "state" / "seed")
    C.write_parquet(w.feats[["person_id", "player_name"]].drop_duplicates(),
                    S.SEED_DIR / S.SEED_FILES["names"])
    P.write_features(w.feats, DATE, P.history_stamp(SEASON, DATE))
    C.write_parquet(w.proj, P.projections_path(DATE))
    rep = w.report if report is None else report
    monkeypatch.setattr(LV, "schedule", lambda: LV.Live(w.sched, 0.0))
    monkeypatch.setattr(LV, "roster", lambda: LV.Live(w.roster, 0.0))
    monkeypatch.setattr(LV, "odds_on", lambda d: LV.Live(w.odds if d == DATE else {}, 0.0))
    monkeypatch.setattr(LV, "injury_report", lambda sched=None: LV.Live(rep, 0.0))
    monkeypatch.setattr(LV, "box", no_box)
    # The odds feed (design 8.3): never a kick, and an off view, unless
    # install_feed turns it on; no ledger and no cached feed table passes from
    # one page test to the next (m16).
    monkeypatch.setattr(LV, "feed_kick", lambda *a, **k: None)
    monkeypatch.setattr(LV, "feed_view", lambda *a, **k: LV.FeedLive.off())
    monkeypatch.setattr(LV, "prop_lines_memory", lambda *a, **k: LV.FeedLive.off())
    pin(monkeypatch, at)
    LV.reset_feed_anchor_for_tests()
    store._live_frame.clear()
    store._feed_tables.clear()
    return w


@dataclass(frozen=True)
class Feed:
    """What install_feed set up: build 2's fake vendor (its calls recorded),
    the harness's own ledger, and the view the pages are served."""
    vendor: object
    mem: dict
    view: object


def install_feed(monkeypatch, payloads: dict | None = None, state: str | None = "ok",
                 at: dt.datetime = MORNING, *, pulls=None, loading: int | None = None,
                 environ: dict | None = None, balance: tuple | None = None,
                 horizon: list | None = None, loading_games: tuple | None = None) -> Feed:
    """The licensed odds feed ON for a page test, after install() (design 8.3).

    A fake key, built at run time ("K" * 40, tests/test_live_feed.py's), in
    the environment turns the feed on, so the pages run the REAL
    store.feed_start, feed_result, feed_tables and feed_info. The lines come
    from the REAL live walk (live.feed_kick) on a ledger of the harness's
    own: build 2's fake vendor answers /events and each game's odds from the
    CONSTRUCTED fixtures in tests/fixtures/oddsapi, or from `payloads`
    ({event_id: payload}), through the REAL oddsapi parser and merge, on the
    synchronous pool, for the REAL work order (store.feed_work). `pulls` is
    [(when, books the vendor leaves out)], the ledger born at the first; the
    default is one pull of every game at `at`. By default the vendor reports
    nothing used this period, so the memory-loss brake never defers a pull,
    and it is installed only while the harness buys: the pages never reach it
    (and live.feed_kick stays stubbed).

    For a budget that binds (the serve fixture's schedule holds the date's
    two games only, so nothing else waits for a credit): `balance` is the
    vendor's (remaining, used) in place of (350, 0), and `horizon` the work
    order's horizon rows (plain dicts, as live reads them) in place of the
    real order's own. `environ` holds feed settings (BUCKETVALUE_ODDS_*):
    the walk reads them, and so do the pages, which get them in the process
    environment beside the key, as on the site, where both read the same
    secrets (store.feed_info's books, build amendment B25).

    live.feed_view and live.prop_lines_memory then serve the REAL feed_view
    of that ledger with no wait, at the page's pinned clock (pre-tip copies
    and "lost_in_restart" follow it). `state` (None: as the view assembles
    it) and `loading` replace what the page reads, for the phrases a walk
    cannot be made to show on demand (design 6); with `loading` forced, the
    games loading (FeedLive.loading_games, which the Game page reads for its
    own game) are `loading_games`, or every game of the date's view."""
    import test_live_feed as LF       # build 2's fake vendor and synchronous pool (tests/)
    remaining, used = balance if balance is not None else (350, 0)
    vendor = LF.Vendor(remaining=remaining, used=used)
    if payloads is not None:
        vendor.odds.update(payloads)
    times = list(pulls) if pulls is not None else [(at, ())]
    env = {LV.KEY_ENV: LF.KEY, **(environ or {})}
    mem = LV.new_feed_memory(times[0][0].astimezone(UTC), had_ledger=False)
    work = store.feed_work(DATE, load().sched)
    if horizon is not None:
        work = {**work, "horizon": [dict(r) for r in horizon]}
    with monkeypatch.context() as buying:
        buying.setattr(http, "vendor_get", vendor)
        for when, left_out in times:
            vendor.drop = set(left_out)
            REAL_FEED_KICK(work, mem=mem, environ=env, pool=LF.SyncPool(), now=when.astimezone(UTC))
    vendor.drop = set()
    monkeypatch.setenv(LV.KEY_ENV, LF.KEY)
    for name, value in (environ or {}).items():       # the pages read the walk's settings too
        monkeypatch.setenv(name, value)

    def view(date=DATE, *a, **k) -> LV.FeedLive:
        got = REAL_FEED_VIEW(date, mem=mem, wait_s=0)
        if state is not None and got.state != state:
            got = dataclasses.replace(got, state=state)
        if loading is not None:
            games = tuple(got.per_game) if loading_games is None else tuple(loading_games)
            got = dataclasses.replace(got, loading=int(loading), loading_games=games)
        return got
    monkeypatch.setattr(LV, "feed_view", view)
    monkeypatch.setattr(LV, "prop_lines_memory", view)
    return Feed(vendor, mem, view)


def paste(w: World, at: dt.datetime = MORNING) -> str:
    """The board text through the real paste path (props_ui.ingest): the
    token a visitor's ?board= would carry."""
    token, report = props_ui.ingest(w.board_text, None, w.roster, w.sched, now=at, dates=[DATE])
    assert token and report["comparable"] == 11, report
    return token
