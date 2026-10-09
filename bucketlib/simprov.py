"""The simulator as a props.DistProvider: projection rows in, re-centred pmfs out.

Serving design section 2. The props comparison (props.compare) and the Game
page tables both read distributions through this module, so a line on the
board and the median in the table come from ONE simulation per player-game.

WHICH STATS ARE 'model'
-----------------------
Only GATED, the stats the probabilistic gate measured (spec 3.4c:
scripts/prob_gate.py BOARD, docs/prob_gate_v1.csv and the 2025-26 holdout,
data/stat_reliability.csv), are served from the simulator and labelled
source 'model'. The other ten board stats (FT made and attempted, FG made and
attempted, 3-PT attempted, 2-PT made and attempted, offensive and defensive
rebounds, fouls) never sat that gate, and at least one is structurally wrong
in the simulator today: dist.simulate draws FTA as 2 x trips, so its FTA pmf
has no mass on any odd count while about a third of real games with a free
throw end on one. A mean test cannot see that (the mean is exact). Spec 3.4:
a target that has not passed the gate ships a LABELLED baseline, so BASELINE
stats come from props.MeanDispersionProvider and source() says 'baseline'.
Widen GATED only with a prob_gate run that covers the stat;
tests/test_simprov.py pins GATED to the gate's own outputs.

WHAT IT GUARANTEES
------------------
- Every pmf it returns has mean() equal to the shipped E[stat | plays]
  (props.shipped_mean of the projection row, the same function compare() uses)
  within compare()'s own tolerance, so compare() never has to re-centre it.
  dist.pmfs re-centres positive means; the boundary cases it leaves alone (a
  mean of exactly zero, a NEGATIVE count mean, a Fantasy mean at or below
  zero) are settled here with Pmf.recentre, the one re-centring implementation.
- Stable numbers. The seed is zlib.crc32("person_id:game_id"), never Python's
  hash() (salted per process) and never a row position (a re-sorted frame would
  shuffle every player's draws), so a Streamlit rerun, a second provider, or
  the bot and the app all show the same numbers for the same inputs.
- One dist.pmfs call per (person, game), every gated stat at once, cached as
  pmfs (never draws: the memory rule of spec 3.5). provider_for() shares one
  provider per frame, so the board and medians() read the same cached pmfs.

WHAT IT REFUSES
---------------
Raises MissingColumns (a contract break, once per frame): a non-empty
projection frame without a column it reads (REQUIRED_COLUMNS). A schema slip
upstream must not read as 'no distribution for this stat' on every line.

pmf -> None, never an exception:
- An unknown (person, game), or a stat props cannot ask for (minutes included).
- A row whose simulator inputs are not finite (NaN mu_min or rate): every stat.
- mu_min below MIN_MU_MIN: every stat. The minutes residual table was fitted
  on projections clipped at 0.5 (scripts/fit_dist.py), and the elasticity term
  rate * M**beta * mu**(1 - beta) grows without bound as mu -> 0 for beta > 1.
  A point mass at zero would read on the board as a certain Less.
- A stat whose shipped mean is missing or not finite, or NEGATIVE on a count
  (FG3A projected above FGA makes FG2A = FGA - FG3A negative, which no count
  distribution can have).
- From the simulator (price), a stat whose draws a CLIPPED split degenerates
  (see _degenerate): FTM when rate_ftm >= rate_fta, FG2M, FG2A and FGM when
  rate_fg3a >= rate_fga. Their re-centred means still match, which is exactly
  why a mean test cannot catch them.

A shipped mean of exactly zero on a count (a rate of 0) IS priced: the only
non-negative distribution with mean zero is the point mass at zero, so that is
what the re-centring returns.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import zlib
from collections import OrderedDict
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import dist as D
from . import props as P
from .cache import logger
from .pmf import Pmf

# Every stat this provider can hand compare(). tests/test_simprov.py pins
# that it covers every stat props.STAT_FORMULA can route to a provider.
STATS: tuple[str, ...] = tuple(D.COUNT_STATS + list(D.COMBOS) + ["fantasy"])
_STAT_SET = frozenset(STATS)

# The probabilistically gated stats (scripts/prob_gate.py BOARD), served from
# the simulator as 'model'. Everything else is the labelled baseline. See the
# module docstring; the test pins this to prob_gate.BOARD,
# data/stat_reliability.csv and both docs/prob_gate_*.csv files.
GATED: tuple[str, ...] = ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov",
                          "pra", "pr", "pa", "ra", "stocks", "fantasy")
_GATED_SET = frozenset(GATED)
BASELINE: tuple[str, ...] = tuple(s for s in STATS if s not in _GATED_SET)

# Below this many projected minutes (if he plays) the simulator has no fitted
# minutes to draw from; see the module docstring.
MIN_MU_MIN = 0.5

# The per-minute streams dist.simulate draws directly (Poisson with elasticity),
# and the made/attempt splits player_input derives from rate columns.
_STREAMS = ("fga", "fta", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf")
_SPLITS = ("fg3a", "fgm", "fg3m", "ftm")

# Columns a non-empty projection frame must carry (serving design section 3).
# INPUT_COLUMNS is exactly what player_input reads (pinned by a test); the
# per-stat means are the recentre targets. fg2m, fg2a and the combos need no
# column: props.shipped_mean composes them from these.
INPUT_COLUMNS: tuple[str, ...] = ("mu_min",) + tuple(f"rate_{s}" for s in _STREAMS + _SPLITS)
MEAN_COLUMNS: tuple[str, ...] = tuple(s for s in D.COUNT_STATS if s not in ("fg2m", "fg2a"))
REQUIRED_COLUMNS: tuple[str, ...] = ("person_id", "game_id") + INPUT_COLUMNS + MEAN_COLUMNS
# Every column a provider reads (the provider cache key hashes exactly these).
_READ_COLUMNS: tuple[str, ...] = ("person_id", "game_id") + INPUT_COLUMNS + STATS


class MissingColumns(KeyError):
    """A projection frame lacks columns the provider reads. Names EVERY absent
    column, as models.MissingFeatures does: a frame-level contract break
    raises once instead of pricing every row as 'no distribution'."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(f"{len(self.missing)} projection column(s) the simulator reads are "
                         f"absent from the frame: {', '.join(self.missing)}")

    def __str__(self) -> str:  # KeyError would print the message as a quoted repr
        return self.args[0]


# ── Projection row -> simulator input ────────────────────────────────────────
def _num(r, col: str) -> float:
    """A finite float from the row, or ValueError naming the column. A NaN
    rate used to flow into rng.poisson and fail there, far from its cause."""
    try:
        x = float(r[col])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{col}: missing or not a number ({e})") from None
    if not math.isfinite(x):
        raise ValueError(f"{col}: not finite ({x})")
    return x


def _mean_prefix(r) -> str:
    """Where the shipped means live: 'pred_' on an evaluation row, '' on a
    projection row. cache/preds_eval_v1.parquet carries the model's mean as
    pred_<s> BESIDE the actual outcome as plain <s>; reading <s> there would
    centre every pmf on the result itself, so pred_ wins whenever present."""
    return "pred_" if any(f"pred_{s}" in r for s in STATS) else ""


def player_input(r, prefix: str | None = None) -> D.PlayerInput:
    """The simulator's input for one player-game row.

    Rates are per minute (rate_<s>), shooting splits are made/attempt
    fractions of those rates, clipped into [0, 1] (FG3A above FGA becomes an
    all-threes share; FTM above FTA a perfect line). `prefix` says where the
    shipped means live: '' for the projection frame (serving design section
    3), 'pred_' for an evaluation frame, None to detect (scripts/prob_gate.py
    passes eval rows). Shipped means are props.shipped_mean of that view, so a
    stat without its own column (fg2m, fg2a) gets the exact linear combination
    of its components, the same number compare() re-centres on. Raises
    ValueError on a missing or non-finite mu_min or rate.
    """
    rate = {s: max(_num(r, f"rate_{s}"), 1e-6) for s in _STREAMS}
    fga, fg3a, fgm, fg3m = (max(_num(r, f"rate_{s}"), 1e-9) for s in ("fga", "fg3a", "fgm", "fg3m"))
    fta, ftm = max(_num(r, "rate_fta"), 1e-9), max(_num(r, "rate_ftm"), 0.0)
    share3 = min(fg3a / fga, 1.0)
    p3 = min(fg3m / fg3a, 1.0)
    p2 = min(max(fgm - fg3m, 0.0) / max(fga - fg3a, 1e-9), 1.0)
    pre = _mean_prefix(r) if prefix is None else prefix
    view = {s: r.get(pre + s) for s in STATS}
    shipped = {}
    for s in STATS:
        m = P.shipped_mean(view, s)
        if m is not None:
            shipped[s] = m
    return D.PlayerInput(mu_min=_num(r, "mu_min"), rate=rate, fg3a_share=share3, p3=p3, p2=p2,
                         pft=min(ftm / fta, 1.0), shipped_mean=shipped)


# ── Pricing one player-game ──────────────────────────────────────────────────
def seed_of(person_id: int, game_id: str) -> int:
    """Deterministic per (person, game); identical in every process."""
    return zlib.crc32(f"{int(person_id)}:{game_id}".encode())


def _matches(pmf: Pmf, target: float) -> bool:
    """compare()'s own test for 'already re-centred', so it never re-centres
    a pmf from this provider a second time."""
    return abs(pmf.mean() - target) <= 1e-9 * max(1.0, abs(target))


def _settle(pmf: Pmf | None, target, stat: str) -> Pmf | None:
    """`pmf` with its mean on the shipped `target`, or None when there is no
    usable target or a count cannot reach it. dist.pmfs leaves a mean <= 0
    alone: zero becomes the point mass at zero, a negative COUNT mean raises
    (below lower=0) and is not priced, and Fantasy (turnovers count -1) has no
    lower bound, so a Fantasy mean at or below zero is priced."""
    if pmf is None or target is None or not math.isfinite(target):
        return None
    if _matches(pmf, target):
        return pmf
    try:
        return pmf.recentre(target, lower=None if stat == "fantasy" else 0.0)
    except ValueError:
        return None


def _degenerate(inp: D.PlayerInput) -> frozenset[str]:
    """Stats a CLIPPED split makes the simulator draw degenerately. player_input
    stays as it is (prob_gate scores through it); the refusal lives here.
    - pft >= 1 (rate_ftm at or above rate_fta): every draw has FTM = FTA, and
      FTA is 2 x trips, so the FTM pmf has no mass on an odd count.
    - fg3a_share >= 1 (rate_fg3a at or above rate_fga): no 2-point attempt is
      ever drawn, so FG2M re-centres onto a two-point mass at maximum
      confidence, FGM piles its mass on its top atom, and FG2A's mean is <= 0.
    """
    out: set[str] = set()
    if inp.pft >= 1.0:
        out.add("ftm")
    if inp.fg3a_share >= 1.0:
        out.update(("fg2m", "fg2a", "fgm"))
    return frozenset(out)


# The one BASELINE stat a clipped FG3A share also breaks. FG2A's composed mean,
# FGA - FG3A, is <= 0 once the share clips, and exactly 0 at models.predict's
# rate_fg3a cap, where the baseline would price it as the point mass at zero:
# a certain Less on every 2-PT Attempted line. That zero is a cap, not a
# forecast. FG2M and FGM keep positive means and ordinary baseline pmfs.
_BASELINE_DEGENERATE = frozenset({"fg2a"})


def price(inp: D.PlayerInput, params: D.SimParams, seed: int, n: int = D.N_DRAWS,
          stats=STATS) -> dict[str, Pmf] | None:
    """Re-centred simulator pmfs for each of `stats` that can be priced, or
    None when the player cannot be (mu_min not finite or below MIN_MU_MIN).
    The draws do not depend on `stats` (dist.simulate draws every stream), so
    asking for fewer stats never changes the numbers of the rest."""
    if not (math.isfinite(inp.mu_min) and inp.mu_min >= MIN_MU_MIN):
        return None
    bad = _degenerate(inp)
    want = [s for s in stats if s not in bad]
    if not want:    # dist.pmfs reads an empty list as 'every stat'
        return {}
    raw = D.pmfs(inp, params, seed=seed, stats=want, n=n)
    out: dict[str, Pmf] = {}
    for s, p in raw.items():
        p = _settle(p, inp.shipped_mean.get(s), s)
        if p is not None:
            out[s] = p
    return out


def _key(person_id, game_id) -> tuple[int, str] | None:
    try:
        if person_id is None or game_id is None or pd.isna(person_id) or pd.isna(game_id):
            return None
        return int(person_id), str(game_id)
    except (TypeError, ValueError):
        return None


class SimProvider:
    """props.DistProvider backed by the simulator (GATED stats, source 'model')
    and props.MeanDispersionProvider (BASELINE stats, source 'baseline').

    SimProvider(projections, params, n): `projections` is the projection
    frame (serving design section 3: person_id, game_id, mu_min, rate_<s>,
    the shipped means as plain stat names); `params` is
    dist.SimParams.load(cache.MODELS_DIR / "sim_params_v1.json"). Raises
    MissingColumns when a non-empty frame lacks a REQUIRED_COLUMNS column. A
    duplicate (person_id, game_id) keeps the LAST row, as compare() does, so
    the pmf and the row compare() re-centres against are the same row.

    Callers that also need medians build it with provider_for(), so the board
    and the Game tables share one cache. Sharing across Streamlit sessions is
    safe: the worst a race does is price the same player twice, identically.
    """

    def __init__(self, projections: pd.DataFrame | None, params: D.SimParams, n: int = D.N_DRAWS):
        self._params = params
        self._n = int(n)
        self._rows: dict[tuple[int, str], dict] = {}
        frame = projections if projections is not None and not projections.empty else None
        if frame is not None:
            missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
            if missing:
                raise MissingColumns(missing)
            frame = frame[[c for c in _READ_COLUMNS if c in frame.columns]]
            for rec in frame.to_dict(orient="records"):
                k = _key(rec.get("person_id"), rec.get("game_id"))
                if k is not None:
                    self._rows[k] = rec
        # props' own baseline on the same rows (it also keeps the last duplicate).
        self._baseline = P.MeanDispersionProvider(frame)
        self._cache: dict[tuple[int, str], dict[str, Pmf] | None] = {}
        self._refused: dict[tuple[int, str], frozenset[str]] = {}

    def _pmfs(self, key: tuple[int, str]) -> dict[str, Pmf] | None:
        """The row's simulator pmfs, or None when the ROW cannot be priced."""
        if key in self._cache:
            return self._cache[key]
        out = None
        row = self._rows.get(key)
        if row is not None:
            try:
                inp = player_input(row, prefix="")
                self._refused[key] = _degenerate(inp) & _BASELINE_DEGENERATE
                out = price(inp, self._params, seed_of(*key), self._n, stats=GATED)
            except ValueError as e:
                logger.warning("simprov: cannot price %s in %s: %s", key[0], key[1], e)
        self._cache[key] = out
        return out

    def pmf(self, person_id, game_id, stat) -> Pmf | None:
        if not isinstance(stat, str) or stat not in _STAT_SET:
            return None
        key = _key(person_id, game_id)
        if key is None:
            return None
        pm = self._pmfs(key)
        if pm is None:
            # A row the simulator refuses (minutes below the floor, NaN
            # inputs) is refused for every stat, the baseline ones included.
            return None
        if stat in _GATED_SET:
            return pm.get(stat)
        if stat in self._refused.get(key, ()):
            return None
        return _settle(self._baseline.pmf(key[0], key[1], stat),
                       P.shipped_mean(self._rows[key], stat), stat)

    def source(self, person_id=None, game_id=None, stat=None) -> str:
        return "model" if isinstance(stat, str) and stat in _GATED_SET else "baseline"

    def keys(self) -> list[tuple[int, str]]:
        """(person_id, game_id) of every projection row, first-seen order."""
        return list(self._rows)

    def medians(self, stats) -> pd.DataFrame:
        """(person_id, game_id, med_<stat>) mid-p medians from this provider's
        cached pmfs (a BASELINE stat's median is the baseline's), NaN where a
        stat cannot be priced. An unknown stat name RAISES: a typo would
        otherwise render as a column of sentinels."""
        stats = list(stats)
        bad = [s for s in stats if s not in _STAT_SET]
        if bad:
            raise ValueError(f"not a simulated stat: {bad}")
        cols = ["person_id", "game_id"] + [f"med_{s}" for s in stats]
        recs = []
        for pid, gid in self._rows:
            rec = {"person_id": pid, "game_id": gid}
            for s in stats:
                p = self.pmf(pid, gid, s)
                rec[f"med_{s}"] = p.median() if p is not None else np.nan
            recs.append(rec)
        out = pd.DataFrame(recs, columns=cols)
        out["person_id"] = out["person_id"].astype("int64")
        out["game_id"] = out["game_id"].astype(str)
        return out


# ── One provider per frame ───────────────────────────────────────────────────
# medians(projections, params, stats) has the contract's signature but must
# read "the same cached pmfs" as the board, so providers are shared by CONTENT:
# an equal frame (a Streamlit rerun, a copy) gets the same provider, a frame
# that differs in any column the provider reads gets a new one. Bounded, so
# mark-out variants cannot grow memory without limit (a 400-player slate's
# pmfs are a few MB).
PROVIDER_CACHE_SIZE = 4
_PROVIDERS: "OrderedDict[tuple, SimProvider]" = OrderedDict()
_PROVIDERS_LOCK = threading.Lock()


def _cache_key(projections: pd.DataFrame | None, params: D.SimParams, n: int) -> tuple | None:
    """(frame digest, params digest, n), or None when the frame cannot be
    hashed (then the caller gets an uncached provider; numbers are the same)."""
    try:
        if projections is None or projections.empty:
            frame = "empty"
        else:
            cols = [c for c in _READ_COLUMNS if c in projections.columns]
            h = hashlib.sha1(repr(cols).encode())
            h.update(pd.util.hash_pandas_object(projections[cols], index=False).to_numpy().tobytes())
            frame = h.hexdigest()
        prm = hashlib.sha1(json.dumps(asdict(params), sort_keys=True, default=str).encode()).hexdigest()
    except (TypeError, ValueError) as e:
        logger.info("simprov: frame not hashable, provider not shared (%s)", type(e).__name__)
        return None
    return frame, prm, int(n)


def provider_for(projections: pd.DataFrame | None, params: D.SimParams,
                 n: int = D.N_DRAWS) -> SimProvider:
    """The shared SimProvider for this frame's content, params and n. Build the
    board's provider with this, and medians() reuses its pmfs."""
    key = _cache_key(projections, params, n)
    if key is None:
        return SimProvider(projections, params, n)
    with _PROVIDERS_LOCK:
        prov = _PROVIDERS.get(key)
        if prov is not None:
            _PROVIDERS.move_to_end(key)
            return prov
    prov = SimProvider(projections, params, n)   # a contract break raises; nothing cached
    with _PROVIDERS_LOCK:
        prov = _PROVIDERS.setdefault(key, prov)
        _PROVIDERS.move_to_end(key)
        while len(_PROVIDERS) > PROVIDER_CACHE_SIZE:
            _PROVIDERS.popitem(last=False)
    return prov


def medians(projections: pd.DataFrame | None, params: D.SimParams, stats,
            n: int = D.N_DRAWS) -> pd.DataFrame:
    """Mid-p medians for every projection row, from the same cached pmfs as
    the board's provider_for() provider (no second simulation)."""
    return provider_for(projections, params, n).medians(stats)
