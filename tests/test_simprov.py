"""The simulator distribution provider (bucketlib/simprov.py).

Every check here runs PRODUCTION code: re-centring is checked against
props.shipped_mean (the function compare() re-centres on), the lean against
props.compare itself, medians against Pmf.median, the seed against dist.pmfs,
the baseline against props.MeanDispersionProvider, GATED against
scripts/prob_gate.py and the gate's own output files. Rows come from
cache/preds_eval_v1.parquet when it is present (out-of-sample model output,
mapped to the projection-frame column names), synthetic otherwise.
"""
from __future__ import annotations

import importlib
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bucketlib import cache as C
from bucketlib import dist as D
from bucketlib import props as P
from bucketlib import simprov as S

REPO = Path(__file__).resolve().parent.parent
EVAL = C.CACHE_DIR / "preds_eval_v1.parquet"
GID = "0022600005"
RATES = {"pts": 0.62, "reb": 0.24, "ast": 0.17, "fg3m": 0.07, "stl": 0.03, "blk": 0.02,
         "tov": 0.07, "ftm": 0.12, "fta": 0.15, "fgm": 0.22, "fga": 0.46, "fg3a": 0.19,
         "oreb": 0.05, "dreb": 0.19, "pf": 0.07}
MEANS = list(RATES) + ["pra", "pr", "pa", "ra", "stocks", "fantasy"]


@pytest.fixture(scope="module")
def params():
    return D.SimParams.load(C.MODELS_DIR / "sim_params_v1.json")


@pytest.fixture(autouse=True)
def _fresh_provider_cache():
    """provider_for() shares providers module-wide; each test starts empty so
    a call count or an identity check never depends on test order."""
    S._PROVIDERS.clear()
    yield
    S._PROVIDERS.clear()


def _compose(rec: dict) -> dict:
    """Combos and Fantasy from the per-stat means, as models.predict ships them."""
    for c in ("pra", "pr", "pa", "ra", "stocks", "fantasy"):
        rec[c] = sum(w * rec[k] for k, w in P.STAT_FORMULA[c].items())
    return rec


def synth(pid=203999, gid=GID, mu=31.0, **rates) -> dict:
    """One projection-frame row (serving design section 3 columns)."""
    r = dict(RATES)
    r.update(rates)
    rec = {"person_id": pid, "game_id": gid, "game_date": "2026-10-21", "player_name": "Test Player",
           "name": "Test Player", "team": "DEN", "opp": "DAL", "home": True, "p_play": 0.97,
           "games_prior": 300, "provisional": False, "model_version": "test", "mu_min": mu, "min": mu}
    for s, v in r.items():
        rec[f"rate_{s}"] = v
        rec[s] = v * mu
    return _compose(rec)


def _to_projection(ev: pd.DataFrame) -> pd.DataFrame:
    """An eval frame in projection-frame names: the ACTUAL outcomes (plain
    names, and min) are dropped, pred_<s> becomes <s>, min becomes mu_min."""
    actual = [c for c in MEANS + ["min"] if c in ev.columns]
    out = ev.drop(columns=actual).rename(columns={f"pred_{s}": s for s in MEANS})
    out["min"] = out["mu_min"]
    out["games_prior"] = 300
    out["provisional"] = False
    out["game_id"] = out["game_id"].astype(str)
    return out.reset_index(drop=True)


@pytest.fixture(scope="module")
def eval_rows() -> pd.DataFrame:
    """Raw eval rows (pred_<s> beside the actual <s>), spread over the
    minutes range, or a synthetic stand-in with the same layout."""
    if EVAL.exists():
        ev = C.read_parquet(EVAL)
        ev = ev[ev["played"] == True].sort_values("mu_min")  # noqa: E712
        ev = ev.iloc[np.linspace(0, len(ev) - 1, 40).astype(int)]
        return ev.drop_duplicates(["person_id", "game_id"]).reset_index(drop=True)
    recs = []
    for i, mu in enumerate(np.linspace(1.0, 38.0, 40)):
        rec = synth(pid=1000 + i, gid=f"00226{i:05d}", mu=float(mu))
        for s in MEANS:
            rec[f"pred_{s}"] = rec[s]
            rec[s] = rec[s] + 3.0          # the "actual" differs from the prediction
        rec["min"] = mu + 2.0
        recs.append(rec)
    return pd.DataFrame(recs)


@pytest.fixture(scope="module")
def projections(eval_rows) -> pd.DataFrame:
    return _to_projection(eval_rows)


def _close(a, b):
    """compare()'s own tolerance for 'already re-centred'."""
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def _same(p, q) -> bool:
    return p.offset == q.offset and p.step == q.step and np.array_equal(p.probs, q.probs)


# ── Coverage of what props can ask for ───────────────────────────────────────
def test_every_stat_props_can_route_is_simulated():
    assert set(P.STAT_FORMULA) <= set(S.STATS)
    assert set(P.STAT_MAP.values()) <= set(S.STATS)       # every label alias lands here


def test_is_a_dist_provider_labelled_model_only_on_gated_stats(params):
    prov = S.SimProvider(pd.DataFrame([synth()]), params)
    assert isinstance(prov, P.DistProvider)
    for s in S.STATS:
        assert prov.source(203999, GID, s) == ("model" if s in S.GATED else "baseline"), s
    assert prov.source(203999, GID, None) == "baseline"


# ── Only the probabilistically gated stats are 'model' ───────────────────────
def test_gated_is_exactly_what_the_probabilistic_gate_measured():
    """Spec 3.4: a target that has not passed the probabilistic gate ships a
    labelled baseline. GATED must move only with the gate's own outputs."""
    assert set(S.GATED) == set(_prob_gate().BOARD)
    assert set(P.reliability_table()["stat"]) == set(S.GATED)
    for f in ("prob_gate_v1.csv", "prob_gate_holdout_2025-26.csv"):
        g = pd.read_csv(REPO / "docs" / f)
        assert set(g.loc[g["dist"] == "sim", "stat"]) == set(S.GATED), f
    assert set(S.GATED) | set(S.BASELINE) == set(S.STATS)
    assert not set(S.GATED) & set(S.BASELINE)


def test_ungated_stats_are_props_own_baseline(params):
    rows = pd.DataFrame([synth(), synth(pid=2, mu=24.0, fta=0.31, ftm=0.24, oreb=0.11)])
    prov = S.SimProvider(rows, params)
    base = P.MeanDispersionProvider(rows)
    for pid in (203999, 2):
        for s in S.BASELINE:
            p = prov.pmf(pid, GID, s)
            assert p is not None and _same(p, base.pmf(pid, GID, s)), (pid, s)
            assert _close(p.mean(), P.shipped_mean(rows[rows.person_id == pid].iloc[0], s))


def test_an_ft_attempted_line_is_never_priced_from_the_even_only_simulator(params):
    """dist.simulate draws FTA as 2 x trips: served from it, a whole FTA line
    had P(push) = 0 and the same P(over) as the half line below it."""
    proj = pd.DataFrame([synth(pid=1)])
    prov = S.SimProvider(proj, params)
    specs = [(1, "FT Attempted", 4.5), (1, "FT Attempted", 5.0), (1, "FT Made", 3.0),
             (1, "Points", 19.5)]
    table, meta = P.compare(_lines(specs), prov, proj)
    assert meta["matched"] == len(specs), meta["unprojected_why"]
    t = table.set_index(["stat", "line"])
    assert t.loc[("fta", 5.0), "p_push"] > 0
    assert t.loc[("fta", 4.5), "p_over"] > t.loc[("fta", 5.0), "p_over"]
    assert t.loc[("ftm", 3.0), "p_push"] > 0
    assert set(t.loc[["fta", "ftm"], "source"]) == {"baseline"}
    assert t.loc[("pts", 19.5), "source"] == "model"


# ── Re-centring: E[pmf] equals the shipped mean ──────────────────────────────
def test_mean_equals_the_shipped_mean_for_every_board_stat(projections, params):
    prov = S.SimProvider(projections, params, n=1500)
    priced = 0
    for rec in projections.to_dict(orient="records"):
        for stat in P.STAT_FORMULA:
            target = P.shipped_mean(rec, stat)
            p = prov.pmf(rec["person_id"], rec["game_id"], stat)
            if rec["mu_min"] < S.MIN_MU_MIN:     # the lowest eval rows: below the floor
                assert p is None, (rec["person_id"], rec["mu_min"])
            elif target is not None and (stat == "fantasy" or target >= 0):
                priced += 1
                assert p is not None, (rec["person_id"], stat)
                assert _close(p.mean(), target), (rec["person_id"], stat, p.mean(), target)
            else:   # a negative count mean (FG3A above FGA) is not priced
                assert p is None, (rec["person_id"], stat, target)
    assert priced >= 0.9 * len(projections) * len(P.STAT_FORMULA)


def test_a_zero_mean_is_the_point_mass_at_zero(params):
    """dist.pmfs leaves a zero target alone; the provider settles it, so a
    rate of 0 still satisfies E[pmf] = shipped mean."""
    row = synth(blk=0.0)
    # The shipped mean wins even where the simulator disagrees: a zero mean
    # shipped beside a non-zero rate still draws blocks before re-centring.
    disagree = {**synth(pid=2), "blk": 0.0}
    disagree = _compose(disagree)
    prov = S.SimProvider(pd.DataFrame([row, disagree]), params)
    for pid, r in ((203999, row), (2, disagree)):
        blk = prov.pmf(pid, GID, "blk")
        assert blk is not None and len(blk) == 1 and blk.mean() == 0.0, pid
        assert blk.split(0.5) == (1.0, 0.0, 0.0)
        assert _close(prov.pmf(pid, GID, "stocks").mean(), r["stl"])


# ── Stable numbers ───────────────────────────────────────────────────────────
def test_identical_pmfs_across_two_instances_and_row_orders(projections, params):
    a = S.SimProvider(projections, params, n=1500)
    shuffled = projections.sample(frac=1.0, random_state=7).reset_index(drop=True)
    b = S.SimProvider(shuffled, params, n=1500)
    for pid, gid in a.keys():
        for stat in S.STATS:
            pa, pb = a.pmf(pid, gid, stat), b.pmf(pid, gid, stat)
            assert (pa is None) == (pb is None)
            if pa is not None:
                assert _same(pa, pb), (pid, stat)


def test_seed_is_crc32_of_person_and_game(params):
    """crc32, never hash() (salted per process) or a row position."""
    assert S.seed_of(2544, "0022300061") == zlib.crc32(b"2544:0022300061")
    row = synth(pid=2544, gid="0022300061")
    prov = S.SimProvider(pd.DataFrame([row]), params, n=1200)
    ref = D.pmfs(S.player_input(row, prefix=""), params, seed=zlib.crc32(b"2544:0022300061"),
                 stats=["pts", "pra", "fantasy"], n=1200)
    for stat in ("pts", "pra", "fantasy"):
        assert _same(prov.pmf(np.int64(2544), "0022300061", stat), ref[stat])


def test_one_simulation_per_player_game(params, monkeypatch):
    calls = []
    real = D.pmfs

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(D, "pmfs", counting)
    prov = S.SimProvider(pd.DataFrame([synth()]), params, n=500)
    for stat in S.STATS:
        prov.pmf(203999, GID, stat)
    prov.medians(["pts", "reb"])
    assert len(calls) == 1


# ── props.compare end to end ─────────────────────────────────────────────────
def _lines(specs, gid=GID):
    df = pd.DataFrame([{"name": "Test Player", "team": "DEN", "position": "C", "stat_type": label,
                        "line": line, "start_time": None, "direction": "both",
                        "odds_type": "standard", "opp": None, "is_combo": False, "league": None,
                        "source": "board", "person_id": pid, "game_id": gid}
                       for pid, label, line in specs])
    df["match"] = "matched"
    return df


def _check_lean(table):
    for r in table.itertuples():
        assert r.gap == pytest.approx(r.model - r.line)
        if r.lean == "More":
            assert r.model > r.line, (r.stat, r.line, r.model)
        elif r.lean == "Less":
            assert r.model < r.line, (r.stat, r.line, r.model)


def test_compare_end_to_end_whole_and_half_lines_both_sides(params):
    proj = pd.DataFrame([synth(pid=1), synth(pid=2, mu=24.0, reb=0.35, ast=0.08)])
    prov = S.SimProvider(proj, params)
    specs = [(1, "Points", 12.5), (1, "Points", 25.0), (1, "Points", 15.0), (1, "Points", 22.5),
             (2, "Rebounds", 4.0), (2, "Rebounds", 13.5), (2, "Rebounds", 4.5), (2, "Rebounds", 12.0),
             (1, "Pts+Rebs+Asts", 31.5), (2, "Fantasy Score", 30.0), (1, "3-PT Made", 2.5),
             (2, "Blks+Stls", 0.5)]
    table, meta = P.compare(_lines(specs), prov, proj)
    assert meta["matched"] == len(specs) and meta["unprojected"] == 0, meta["unprojected_why"]
    assert set(table["source"]) == {"model"}
    kinds = set()
    for r in table.itertuples():
        target = P.shipped_mean(proj[proj.person_id == r.person_id].iloc[0], r.stat)
        assert _close(r.mean, target), (r.stat, r.mean, target)
        p = prov.pmf(r.person_id, r.game_id, r.stat)
        assert (r.p_under, r.p_push, r.p_over) == p.split(r.line)
        assert r.model == p.median()
        whole = float(r.line).is_integer()
        assert (r.p_push > 0) == whole or r.stat == "fantasy"
        kinds.add((r.lean, "whole" if whole else "half"))
    _check_lean(table)
    assert {("More", "whole"), ("More", "half"), ("Less", "whole"), ("Less", "half")} <= kinds


def test_median_sign_equals_the_lean_on_every_whole_and_half_line(projections, params):
    """sign(median - line) == lean through compare(), sweeping each stat's
    lines across its support in half-point steps."""
    proj = projections.iloc[::5].reset_index(drop=True)
    prov = S.SimProvider(proj, params, n=1500)
    labels = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-PT Made",
              "pra": "Pts+Rebs+Asts", "stocks": "Blks+Stls", "tov": "Turnovers"}
    specs = []
    for pid, gid in prov.keys():
        for stat, label in labels.items():
            p = prov.pmf(pid, gid, stat)
            if p is None:
                continue
            lo, hi = p.support()
            specs += [(pid, label, float(x)) for x in np.arange(max(lo - 1, 0), hi + 1.5, 0.5)]
    lines = pd.concat([_lines([s for s in specs if s[0] == pid], gid) for pid, gid in prov.keys()],
                      ignore_index=True)
    table, meta = P.compare(lines, prov, proj)
    assert meta["matched"] == len(lines) and meta["errors"] == 0
    _check_lean(table)
    assert (table["lean"] == "More").any() and (table["lean"] == "Less").any()


# ── Medians ──────────────────────────────────────────────────────────────────
def test_medians_are_each_pmfs_mid_p_median(projections, params):
    stats = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra", "fantasy", "fta"]
    proj = pd.concat([projections, pd.DataFrame([synth(pid=77, mu=0.0)])], ignore_index=True)
    med = S.medians(proj, params, stats, n=1500)
    assert list(med.columns) == ["person_id", "game_id"] + [f"med_{s}" for s in stats]
    assert len(med) == len(proj.drop_duplicates(["person_id", "game_id"]))
    prov = S.SimProvider(proj, params, n=1500)
    for r in med.to_dict(orient="records"):
        for s in stats:
            p = prov.pmf(r["person_id"], r["game_id"], s)
            if p is None:
                assert math.isnan(r[f"med_{s}"])
            else:
                assert r[f"med_{s}"] == p.median()
    assert med.loc[med.person_id == 77, "med_pts"].isna().all()       # mu_min 0: not priced
    pd.testing.assert_frame_equal(med, prov.medians(stats))


def test_medians_reuse_the_boards_cached_pmfs(params, monkeypatch):
    """The contract signature medians(projections, params, stats) reads the
    SAME cached pmfs as the board: no second simulation per player."""
    calls = []
    real = D.pmfs

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(D, "pmfs", counting)
    gid = "0022600901"
    proj = pd.DataFrame([synth(pid=1, gid=gid), synth(pid=2, gid=gid, mu=20.0)])
    board = S.provider_for(proj, params, n=500)
    for pid in (1, 2):
        board.pmf(pid, gid, "pts")
    assert len(calls) == 2
    a = S.medians(proj.copy(), params, ["pts", "reb", "fta"], n=500)   # equal frame, new object
    b = S.medians(proj, params, ["pts", "reb", "fta"], n=500)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, board.medians(["pts", "reb", "fta"]))
    same_params = D.SimParams.load(C.MODELS_DIR / "sim_params_v1.json")   # equal content
    assert S.provider_for(proj, same_params, n=500) is board
    changed = proj.copy()
    changed.loc[0, "pts"] += 1.0
    assert S.provider_for(changed, params, n=500) is not board        # a read column changed
    assert S.provider_for(proj, params, n=600) is not board
    other = D.SimParams.load(C.MODELS_DIR / "sim_params_v1.json")
    other.beta = {**other.beta, "fga": other.beta["fga"] + 0.1}
    assert S.provider_for(proj, other, n=500) is not board


def test_the_provider_cache_is_bounded(params):
    frames = [pd.DataFrame([synth(pid=i)]) for i in range(S.PROVIDER_CACHE_SIZE + 2)]
    provs = [S.provider_for(f, params, n=300) for f in frames]
    assert len(S._PROVIDERS) == S.PROVIDER_CACHE_SIZE
    assert S.provider_for(frames[-1], params, n=300) is provs[-1]
    assert S.provider_for(frames[0], params, n=300) is not provs[0]     # evicted, rebuilt


def test_medians_refuse_an_unknown_stat(params):
    with pytest.raises(ValueError, match="3pm"):
        S.medians(pd.DataFrame([synth()]), params, ["pts", "3pm"])
    empty = S.medians(pd.DataFrame(), params, ["pts"])
    assert empty.empty and list(empty.columns) == ["person_id", "game_id", "med_pts"]


# ── Edge cases: None, never a crash ──────────────────────────────────────────
@pytest.mark.parametrize("mu", [0.0, 1e-9, 0.3, S.MIN_MU_MIN - 1e-6, -2.0])
def test_minutes_below_the_floor_are_not_priced(params, mu):
    prov = S.SimProvider(pd.DataFrame([synth(mu=mu)]), params)
    assert all(prov.pmf(203999, GID, s) is None for s in S.STATS)


def test_minutes_at_the_floor_are_priced_on_the_shipped_mean(params):
    row = synth(mu=S.MIN_MU_MIN)
    prov = S.SimProvider(pd.DataFrame([row]), params)
    for s in P.STAT_FORMULA:
        target = P.shipped_mean(row, s)
        assert _close(prov.pmf(203999, GID, s).mean(), target), s


@pytest.mark.parametrize("col", ["mu_min", "rate_fga", "rate_ftm", "rate_ast"])
def test_a_nan_simulator_input_returns_none(params, col, caplog):
    """A NaN VALUE is one bad row: that row is not priced, nothing raises."""
    row = synth()
    row[col] = float("nan")
    prov = S.SimProvider(pd.DataFrame([row, synth(pid=2)]), params)
    assert all(prov.pmf(203999, GID, s) is None for s in S.STATS)
    assert "cannot price" in caplog.text
    assert prov.pmf(2, GID, "pts") is not None


def test_a_missing_column_raises_naming_every_one(params):
    """A missing COLUMN is a contract break for the whole frame: it raises,
    never 'no distribution for this stat' on every line and NaN medians."""
    row = synth()
    for c in ("rate_fg3a", "mu_min", "fta"):
        del row[c]                                   # mu_min gone, only `min` kept
    with pytest.raises(S.MissingColumns) as e:
        S.SimProvider(pd.DataFrame([row]), params)
    assert e.value.missing == ["mu_min", "rate_fg3a", "fta"]
    assert all(c in str(e.value) for c in e.value.missing)
    with pytest.raises(S.MissingColumns):
        S.medians(pd.DataFrame([row]), params, ["pts"])
    for c in S.REQUIRED_COLUMNS:
        one = {k: v for k, v in synth().items() if k != c}
        with pytest.raises(S.MissingColumns, match=c):
            S.provider_for(pd.DataFrame([one]), params)
    S.SimProvider(pd.DataFrame(columns=["person_id"]), params)   # no rows: nothing to price


def test_input_columns_are_exactly_what_player_input_reads():
    row = synth()
    slim = {c: row[c] for c in S.INPUT_COLUMNS}
    S.player_input(slim, prefix="")              # nothing more is needed to simulate
    for c in S.INPUT_COLUMNS:
        with pytest.raises(ValueError, match=f"^{c}: missing"):
            S.player_input({k: v for k, v in slim.items() if k != c}, prefix="")


def test_a_nan_shipped_mean_drops_only_that_stat(params):
    row = synth()
    row["pts"] = float("nan")
    prov = S.SimProvider(pd.DataFrame([row]), params)
    assert prov.pmf(203999, GID, "pts") is None
    assert _close(prov.pmf(203999, GID, "reb").mean(), row["reb"])
    assert _close(prov.pmf(203999, GID, "pra").mean(), row["pra"])   # its own column


def test_fg3a_above_fga_refuses_the_two_point_and_fg_made_draws(params):
    """No 2-point attempt is ever drawn once the share clips at 1: FG2M became
    a two-point mass at maximum confidence and FGM piled onto its top atom,
    both with matching means. The simulator refuses them; the board gets the
    labelled baseline, never the degenerate pmf."""
    row = synth(fg3a=0.55, fga=0.40)
    inp = S.player_input(row, prefix="")
    assert inp.fg3a_share == 1.0
    priced = S.price(inp, params, S.seed_of(203999, GID), n=1500)
    assert not {"fg2a", "fg2m", "fgm"} & set(priced)
    for s in ("fga", "fg3a", "fg3m", "pts", "pra", "fantasy"):
        assert _close(priced[s].mean(), P.shipped_mean(row, s)), s
    ok = S.price(S.player_input(synth(), prefix=""), params, S.seed_of(203999, GID), n=1500)
    assert {"fg2a", "fg2m", "fgm"} <= set(ok)                # not refused when unclipped
    frame = pd.DataFrame([row])
    prov = S.SimProvider(frame, params)
    base = P.MeanDispersionProvider(frame)
    assert prov.pmf(203999, GID, "fg2a") is None             # FGA - FG3A < 0: nothing prices it
    for s in ("fg2m", "fgm"):
        assert prov.source(203999, GID, s) == "baseline"
        assert _same(prov.pmf(203999, GID, s), base.pmf(203999, GID, s)), s
    lo, _, hi = prov.pmf(203999, GID, "fg2m").split(3.5)
    assert 0 < hi < 1 and lo > 0


def test_a_capped_fg3a_never_serves_fg2a_as_a_certain_less(params):
    """models.predict caps rate_fg3a at rate_fga, so a capped row ships an
    FG2A mean of exactly 0, and the baseline prices a zero mean as the point
    mass at zero: a certain Less on every 2-PT Attempted line, wrong the first
    time he takes a two. The clipped share refuses FG2A on the baseline path
    too; FG2M, FGM and the unclipped row keep the baseline."""
    row = synth(fg3a=0.40, fga=0.40)
    assert P.shipped_mean(row, "fg2a") == 0.0
    frame = pd.DataFrame([row, synth(pid=2)])
    prov = S.SimProvider(frame, params)
    base = P.MeanDispersionProvider(frame)
    assert prov.pmf(203999, GID, "fg2a") is None
    assert base.pmf(203999, GID, "fg2a") is not None           # the baseline alone would price it
    for s in ("fg2m", "fgm", "fga", "fg3a"):
        assert _same(prov.pmf(203999, GID, s), base.pmf(203999, GID, s)), s
    assert _same(prov.pmf(2, GID, "fg2a"), base.pmf(2, GID, "fg2a"))
    med = prov.medians(["fg2a"]).set_index("person_id")["med_fg2a"]
    assert math.isnan(med[203999]) and med[2] > 0
    table, meta = P.compare(_lines([(203999, "2-PT Attempted", 0.5), (2, "2-PT Attempted", 4.5)]),
                            prov, frame)
    assert meta["matched"] == 1 and meta["unprojected"] == 1
    assert list(table["person_id"]) == [2], table
    assert meta["unprojected_why"] == {"no distribution for this stat": 1}


def test_ftm_at_or_above_fta_refuses_the_even_only_ftm_draw(params):
    """pft clips to 1, so every simulated FTM equals FTA (2 x trips): an FTM
    pmf with no odd count whose MEAN still matches."""
    for ftm in (0.15, 0.16):
        row = synth(ftm=ftm, fta=0.15)
        inp = S.player_input(row, prefix="")
        assert inp.pft == 1.0
        priced = S.price(inp, params, S.seed_of(203999, GID), n=1500)
        assert "ftm" not in priced, ftm
        for s in ("fta", "pts", "pra", "fantasy"):     # the rest of the row still prices
            assert _close(priced[s].mean(), P.shipped_mean(row, s)), (ftm, s)
        p = S.SimProvider(pd.DataFrame([row]), params).pmf(203999, GID, "ftm")
        odd = p.probs[(p.values % 2) == 1].sum()
        assert odd > 0.2 and _close(p.mean(), row["ftm"]), ftm
    ok = S.price(S.player_input(synth(), prefix=""), params, S.seed_of(203999, GID), n=1500)
    assert _close(ok["ftm"].mean(), synth()["ftm"])          # not refused when unclipped


def test_fg3m_at_its_fg3a_cap_refuses_the_perfect_night_draw(params):
    """models.predict caps rate_fg3m at rate_fg3a (and at rate_fgm). At the
    cap p3 is 1, so every simulated FG3M equals FG3A: a 3PM pmf that is the
    3PA pmf re-centred, with every three going in on every draw. The
    simulator refuses it; the rest of the row still prices, and an uncapped
    row keeps its 3PM."""
    row = synth(fg3m=0.19, fg3a=0.19)
    inp = S.player_input(row, prefix="")
    assert inp.p3 == 1.0
    assert "fg3m" in S._degenerate(inp)
    priced = S.price(inp, params, S.seed_of(203999, GID), n=1500, stats=S.GATED)
    assert "fg3m" not in priced
    for s in ("pts", "reb", "pra", "fantasy"):
        assert _close(priced[s].mean(), P.shipped_mean(row, s)), s
    frame = pd.DataFrame([row, synth(pid=2)])
    prov = S.SimProvider(frame, params)
    assert prov.pmf(203999, GID, "fg3m") is None and prov.pmf(2, GID, "fg3m") is not None
    table, meta = P.compare(_lines([(203999, "3-PT Made", 0.5), (2, "3-PT Made", 1.5)]), prov, frame)
    assert meta["matched"] == 1 and meta["unprojected"] == 1 and list(table["person_id"]) == [2]


def test_fg3m_at_its_fgm_cap_never_serves_fg2m_as_a_certain_less(params):
    """The other fg3m cap: rate_fg3m at rate_fgm makes every make a three, so
    FG2M's composed mean (FGM - FG3M) is exactly 0 and the baseline would
    price the point mass at zero, a certain Less on every 2-PT Made line.
    The raw models put fg3m above fgm on 8 rows of the training table, where
    that mean was negative and refused; the cap turns it into exactly 0, so
    it is refused by the p2 <= 0 rule now, on the baseline path too."""
    row = synth(fg3m=0.22, fgm=0.22, fg3a=0.25)
    assert P.shipped_mean(row, "fg2m") == 0.0
    inp = S.player_input(row, prefix="")
    assert inp.p2 == 0.0 and inp.p3 < 1.0
    assert "fg2m" in S._degenerate(inp) and "fg2m" in S._baseline_degenerate(inp)
    frame = pd.DataFrame([row, synth(pid=2)])
    prov = S.SimProvider(frame, params)
    base = P.MeanDispersionProvider(frame)
    assert base.pmf(203999, GID, "fg2m") is not None           # the baseline alone would price it
    assert prov.pmf(203999, GID, "fg2m") is None
    for s in ("fgm", "fga", "fg2a", "fg3a"):
        assert _same(prov.pmf(203999, GID, s), base.pmf(203999, GID, s)), s
    assert _same(prov.pmf(2, GID, "fg2m"), base.pmf(2, GID, "fg2m"))
    assert prov.pmf(203999, GID, "fg3m") is not None            # p3 < 1: still priced


def test_a_fantasy_mean_at_or_below_zero_is_priced(params):
    """Fantasy can be negative (a turnover is -1), so the count floor of 0
    must never apply to it."""
    row = synth(pts=0.0, reb=0.0, ast=0.0, stl=0.0, blk=0.0, tov=0.07)
    assert row["fantasy"] < 0
    pinned = {**synth(pid=2), "fantasy": -0.5}
    zero = {**synth(pid=3), "fantasy": 0.0}
    prov = S.SimProvider(pd.DataFrame([row, pinned, zero]), params)
    for pid, r in ((203999, row), (2, pinned), (3, zero)):
        p = prov.pmf(pid, GID, "fantasy")
        assert p is not None and _close(p.mean(), r["fantasy"]), (pid, r["fantasy"])
        assert prov.source(pid, GID, "fantasy") == "model"


def test_fg3m_above_fgm_refuses_fg2m(params):
    # fg3a 0.30 keeps p3 below 1: a 3PM above FG3A is its own refusal
    # (test_fg3m_at_its_fg3a_cap_refuses_the_perfect_night_draw).
    prov = S.SimProvider(pd.DataFrame([synth(fg3m=0.25, fgm=0.20, fg3a=0.30)]), params)
    assert prov.pmf(203999, GID, "fg2m") is None
    assert prov.pmf(203999, GID, "fg3m") is not None


@pytest.mark.parametrize("pid,gid,stat", [
    (999, GID, "pts"), (203999, "0022699999", "pts"), (None, GID, "pts"), (float("nan"), GID, "pts"),
    (203999, None, "pts"), (203999, GID, "min"), (203999, GID, "3pm"), (203999, GID, "Points"),
    (203999, GID, ""), (203999, GID, None), (203999, GID, 5)])
def test_unknown_player_game_or_stat_is_none(params, pid, gid, stat):
    assert S.SimProvider(pd.DataFrame([synth()]), params).pmf(pid, gid, stat) is None


def test_bad_identity_rows_are_skipped_and_empty_frames_work(params):
    rows = [synth(), {**synth(), "person_id": float("nan")}, {**synth(), "game_id": None}]
    prov = S.SimProvider(pd.DataFrame(rows), params)
    assert prov.keys() == [(203999, GID)]
    assert S.SimProvider(None, params).pmf(203999, GID, "pts") is None
    assert S.SimProvider(pd.DataFrame(), params).pmf(203999, GID, "pts") is None


def test_a_duplicate_row_keeps_the_last_as_compare_does(params):
    first, last = synth(), synth(pts=0.80, fta=0.30)
    prov = S.SimProvider(pd.DataFrame([first, last]), params)
    assert _close(prov.pmf(203999, GID, "pts").mean(), last["pts"])
    assert _close(prov.pmf(203999, GID, "fta").mean(), last["fta"])      # the baseline too


# ── player_input: moved from scripts/prob_gate.py, behaviour identical ───────
def _legacy_player_input(r):
    """FROZEN copy of scripts/prob_gate.py's player_input as it stood before
    the move (2026-10-09). It is the oracle for 'behaviour identical' and
    nothing else; never edit it to make a test pass."""
    rate = {s: max(float(r[f"rate_{s}"]), 1e-6) for s in ("fga", "fta", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf")}
    fga, fg3a, fgm, fg3m = (max(float(r[f"rate_{s}"]), 1e-9) for s in ("fga", "fg3a", "fgm", "fg3m"))
    fta, ftm = max(float(r["rate_fta"]), 1e-9), max(float(r["rate_ftm"]), 0.0)
    share3 = min(fg3a / fga, 1.0)
    p3 = min(fg3m / fg3a, 1.0)
    p2 = min(max(fgm - fg3m, 0.0) / max(fga - fg3a, 1e-9), 1.0)
    board = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra", "pr", "pa", "ra", "stocks", "fantasy"]
    shipped = {s: float(r[f"pred_{s}"]) for s in board}
    return D.PlayerInput(mu_min=float(r["mu_min"]), rate=rate, fg3a_share=share3, p3=p3, p2=p2,
                         pft=min(ftm / fta, 1.0), shipped_mean=shipped)


def _prob_gate():
    sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("prob_gate")


def test_prob_gate_uses_the_simprov_player_input():
    assert _prob_gate().player_input is S.player_input


def test_player_input_is_unchanged_for_prob_gate_rows(eval_rows, params):
    pg = _prob_gate()
    for i, (_, r) in enumerate(eval_rows.iterrows()):
        old, new = _legacy_player_input(r), pg.player_input(r)
        assert (old.mu_min, old.rate, old.fg3a_share, old.p3, old.p2, old.pft) == \
               (new.mu_min, new.rate, new.fg3a_share, new.p3, new.p2, new.pft)
        assert all(new.shipped_mean[s] == v for s, v in old.shipped_mean.items())
        if i % 8 == 0:
            a = D.pmfs(old, params, seed=i, stats=pg.BOARD, n=800)
            b = D.pmfs(new, params, seed=i, stats=pg.BOARD, n=800)
            assert all(_same(a[s], b[s]) for s in pg.BOARD)


def test_eval_rows_centre_on_the_prediction_never_the_actual(eval_rows):
    """An eval row holds the actual as <s> beside pred_<s>; reading <s> would
    centre every pmf on the outcome (a leak)."""
    r = eval_rows.iloc[len(eval_rows) // 2]
    inp = S.player_input(r)
    for s in ("pts", "reb", "pra", "fantasy"):
        assert inp.shipped_mean[s] == float(r[f"pred_{s}"])
    assert any(float(r[s]) != float(r[f"pred_{s}"]) for s in ("pts", "reb", "pra"))
    plain = S.player_input(r, prefix="")
    assert plain.shipped_mean["pts"] == float(r["pts"])       # explicit '' reads plain names
