"""Per-player discrete distributions by simulation (spec 3.5).

Why simulate instead of a negative binomial per stat:
  - Minutes drive everything and their error is skewed by role (starters'
    residuals are left-skewed: blowouts and foul trouble; bench residuals are
    right-skewed: garbage time and injuries ahead of them), with an overtime
    tail. An empirical residual table by projected-minutes bucket carries all of
    that for free.
  - Production does not scale proportionally with minutes WITHIN a game
    (measured: PTS/3PM/FTA scale ~1.2x, PF/TOV ~0.6-0.7x), so each stat carries
    an elasticity: E[Y | M] = r * mu_min * (M / mu_min) ** beta.
  - Shots must be coherent: FG3M <= FG3A <= FGA, FTM <= FTA, and PTS = 2*FG2M
    + 3*FG3M + FTM exactly. Drawn compositionally, points parity (bigs score in
    twos) and PTS overdispersion fall out mechanically.
  - Combos (PRA, P+R, P+A, R+A, Blks+Stls) and Fantasy Score come from the SAME
    draws, so their correlation through minutes is kept.
Every marginal is then RE-CENTRED on the shipped mean (Pmf.recentre), so the
distribution and the displayed projection can never disagree about the mean.

Parameters are FITTED on walk-forward predictions (scripts/fit_dist.py), never
on in-sample ones; `SimParams` holds them and is saved as JSON beside the models.
Memory: draws are per game, float32, a few thousand at most (ops review: a
15-game slate of 10k float64 draws would be ~0.6 GB in a 1 GB container).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .pmf import Pmf

COUNT_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "ftm", "fta", "fgm", "fga",
               "fg3a", "oreb", "dreb", "pf", "fg2m", "fg2a"]
COMBOS = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast"),
          "ra": ("reb", "ast"), "stocks": ("stl", "blk")}
FANTASY_W = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0, "tov": -1.0}
MIN_BUCKETS = (0.0, 8.0, 16.0, 24.0, 30.0, 35.0, 60.0)
N_DRAWS = 3000


@dataclass
class SimParams:
    # Empirical minutes residuals (actual - projected) per projected-minutes bucket.
    min_resid: dict = field(default_factory=dict)          # bucket index (str) -> list[float]
    beta: dict = field(default_factory=lambda: {          # minutes elasticity per stream
        "fga": 1.2, "fta": 1.2, "ast": 1.0, "oreb": 1.0, "dreb": 1.0, "stl": 1.0,
        "blk": 1.0, "tov": 0.7, "pf": 0.65})
    # Rate-uncertainty shape (higher = tighter). PLACEHOLDERS until the
    # walk-forward fit: chosen so a 35-minute regular reproduces the measured
    # unconditional var/mean (AST ~1.24, REB ~1.32) given ~15% minutes spread:
    # var/mean = 1 + lambda * (CV_minutes^2 + 1/k).
    gamma_k: dict = field(default_factory=lambda: {
        "fga": 60.0, "fta": 15.0, "ast": 80.0, "oreb": 40.0, "dreb": 60.0, "stl": 40.0,
        "blk": 15.0, "tov": 50.0, "pf": 60.0})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "SimParams":
        return cls(**json.loads(path.read_text()))


def bucket_of(mu_min: float) -> int:
    return int(np.clip(np.searchsorted(MIN_BUCKETS, mu_min, side="right") - 1, 0, len(MIN_BUCKETS) - 2))


def draw_minutes(mu_min: float, params: SimParams, rng: np.random.Generator, n: int) -> np.ndarray:
    res = params.min_resid.get(str(bucket_of(mu_min)))
    if not res:
        # No table yet: a symmetric spread proportional to role (labelled crude).
        r = rng.normal(0.0, max(2.0, 0.15 * mu_min), size=n)
    else:
        r = rng.choice(np.asarray(res, dtype=np.float32), size=n, replace=True)
    return np.clip(mu_min + r, 0.0, 58.0).astype(np.float32)


@dataclass
class PlayerInput:
    """Shipped means for one player-game, all CONDITIONAL ON PLAYING.
    rates are per minute; shooting splits are made/attempt fractions."""
    mu_min: float
    rate: dict            # per-minute means: fga, fta, ast, oreb, dreb, stl, blk, tov, pf
    fg3a_share: float     # FG3A / FGA
    p3: float             # FG3M / FG3A
    p2: float             # FG2M / FG2A
    pft: float            # FTM / FTA
    shipped_mean: dict    # stat -> E[stat | plays] the site displays (recentre target)


def _stream(name, base, M, mu, params, rng):
    """Poisson draw for one stream with elasticity and Gamma rate noise."""
    beta = params.beta.get(name, 1.0)
    k = params.gamma_k.get(name, 1e6)
    scale = np.where(mu > 0, (M / max(mu, 1e-6)) ** beta, 0.0)
    g = rng.gamma(k, 1.0 / k, size=M.shape).astype(np.float32)
    return rng.poisson(np.clip(base * mu * scale * g, 0, None)).astype(np.int32)


def simulate(inp: PlayerInput, params: SimParams, seed: int, n: int = N_DRAWS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    M = draw_minutes(inp.mu_min, params, rng, n)
    mu = inp.mu_min
    fga = _stream("fga", inp.rate["fga"], M, mu, params, rng)
    fg3a = rng.binomial(fga, np.clip(inp.fg3a_share, 0, 1))
    fg2a = fga - fg3a
    fg3m = rng.binomial(fg3a, np.clip(inp.p3, 0, 1))
    fg2m = rng.binomial(fg2a, np.clip(inp.p2, 0, 1))
    # Free throws come in trips of ~2: draw TRIPS, then attempts; that is what
    # makes FTA overdispersed (var/mean ~2 measured) rather than Poisson.
    trips = _stream("fta", inp.rate["fta"] / 2.0, M, mu, params, rng)
    fta = 2 * trips
    ftm = rng.binomial(fta, np.clip(inp.pft, 0, 1))
    oreb = _stream("oreb", inp.rate["oreb"], M, mu, params, rng)
    dreb = _stream("dreb", inp.rate["dreb"], M, mu, params, rng)
    out = {
        "min": M, "fga": fga, "fg3a": fg3a, "fg2a": fg2a, "fg3m": fg3m, "fg2m": fg2m,
        "fgm": fg2m + fg3m, "fta": fta, "ftm": ftm, "oreb": oreb, "dreb": dreb,
        "reb": oreb + dreb,
        "ast": _stream("ast", inp.rate["ast"], M, mu, params, rng),
        "stl": _stream("stl", inp.rate["stl"], M, mu, params, rng),
        "blk": _stream("blk", inp.rate["blk"], M, mu, params, rng),
        "tov": _stream("tov", inp.rate["tov"], M, mu, params, rng),
        # Six fouls disqualify: a capped draw is underdispersed, as measured.
        "pf": np.minimum(_stream("pf", inp.rate["pf"], M, mu, params, rng), 6),
    }
    out["pts"] = 2 * out["fg2m"] + 3 * out["fg3m"] + out["ftm"]
    for c, parts in COMBOS.items():
        out[c] = sum(out[p] for p in parts)
    out["fantasy"] = sum(w * out[s].astype(np.float32) for s, w in FANTASY_W.items())
    return out


def pmfs(inp: PlayerInput, params: SimParams, seed: int, stats=None, n: int = N_DRAWS) -> dict[str, Pmf]:
    """Re-centred marginal pmfs for each requested stat (default: every board stat)."""
    draws = simulate(inp, params, seed, n)
    want = stats or (COUNT_STATS + list(COMBOS) + ["fantasy"])
    out = {}
    for s in want:
        x = draws[s]
        step = 0.1 if s == "fantasy" else 1.0
        p = Pmf.from_samples(np.round(x, 1) if s == "fantasy" else x, step=step)
        target = inp.shipped_mean.get(s)
        if target is not None and np.isfinite(target) and target > 0:
            p = p.recentre(float(target), lower=None if s == "fantasy" else 0.0)
        out[s] = p
    return out
