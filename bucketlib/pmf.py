"""A discrete outcome distribution for one player-game stat.

Everything the props layer says about a line comes out of this class: the
chance of landing above, on, and below the number, the median that is shown
beside it, and the mean the distribution is re-centred on. It is deliberately
small and dependency-free (numpy only). scipy is in requirements.txt for the
model code, but a new import on the serving path has broken a Streamlit Cloud
build on a sibling site before, and nothing here needs it.

THE GRID IS REGULAR, AND THAT IS LOAD-BEARING
---------------------------------------------
A Pmf lives on a regular grid `offset + k * step` (step 1 for counts, 0.1 for
Fantasy Score), INCLUDING zero-probability points inside the support. The
displayed median is the mid-p median, which spreads each atom's mass evenly over
[v - step/2, v + step/2] and reads off where that piecewise-linear CDF crosses
one half:

    m = (v_k - step/2) + step * (0.5 - F(k-1)) / p(k)

which for step 1 is the spec's m = (k - 0.5) + (0.5 - F(k-1)) / p(k).

Why this number and not the ordinary median: the lean on a line L compares
P(Y > L) with P(Y < L), ties excluded. With atoms spread over half-step
intervals, the continuous CDF at a HALF line equals F(k) exactly, and at a
WHOLE line equals F(k-1) + p(k)/2, the mid-distribution value. Those are
exactly the quantities the lean compares, so m > L if and only if the lean is
More, on every whole and every half line. The ordinary median cannot do that on
whole lines: when the median atom IS the line, it shows a gap of zero beside a
decided lean. tests/test_pmf.py pins the agreement over thousands of random
distributions.

The agreement needs the regular grid. Dropping a zero-probability interior
point would widen its neighbours' intervals and break the half-line identity,
which is why the constructor refuses values that are not on one grid rather
than quietly accepting them. Lines between grid points that are neither whole
nor half (23.3 on an integer stat) are outside the guarantee; PrizePicks does
not post them.
"""
from __future__ import annotations

import math

import numpy as np

# How far (in grid units) a value may sit from its grid point and still be on
# the grid. Sim draws composed in float arithmetic (1.2 * REB) land within
# ~1e-13 of the grid; anything near 1e-6 is a different grid, not noise.
_GRID_TOL = 1e-6
# A line within this many grid units of a grid point IS that point: 23.5 / 0.1
# is 234.99999999999997 in floating point.
_TIE_TOL = 1e-6
# Bounds the work any one distribution can cost. A Fantasy Score pmf on the
# 0.1 grid spans ~1,500 points; a 1e6-point support is a bug, not a player.
MAX_SUPPORT = 200_000
# Tail mass below which the parametric constructors stop extending the grid.
_TAIL = 1e-14


def _infer_step(v: np.ndarray) -> float:
    """Smallest positive spacing between distinct values, tidied to 12
    significant figures so 0.09999999999999964 reads as 0.1."""
    if v.size < 2:
        return 1.0
    d = np.diff(np.unique(v))
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(f"{float(d.min()):.12g}")


class Pmf:
    """Probability mass on a regular grid.

    Pmf(values, probs, step=None). `values` need not be sorted or unique (they
    are sorted and merged); `probs` are non-negative and are normalised to sum
    to one. `step` is inferred from the smallest spacing when omitted. Raises
    ValueError on non-finite input, negative mass, zero total mass, or values
    that do not sit on one regular grid.
    """

    __slots__ = ("offset", "step", "probs")

    def __init__(self, values, probs, step: float | None = None):
        v = np.asarray(values, dtype=float).ravel()
        p = np.asarray(probs, dtype=float).ravel()
        if v.size == 0 or v.size != p.size:
            raise ValueError("values and probs must be non-empty and the same length")
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(p))):
            raise ValueError("values and probs must be finite")
        if np.any(p < 0):
            raise ValueError("negative probability")
        order = np.argsort(v, kind="stable")
        v, p = v[order], p[order]
        step = _infer_step(v) if step is None else float(step)
        if not (math.isfinite(step) and step > 0):
            raise ValueError(f"step must be a positive finite number, got {step!r}")
        offset = float(v[0])
        idx_f = (v - offset) / step
        idx = np.rint(idx_f)
        if np.any(np.abs(idx_f - idx) > _GRID_TOL):
            raise ValueError(f"values are not on one regular grid of step {step}")
        idx = idx.astype(np.int64)
        n = int(idx[-1]) + 1
        if n > MAX_SUPPORT:
            raise ValueError(f"support of {n} grid points exceeds {MAX_SUPPORT}")
        grid = np.bincount(idx, weights=p, minlength=n).astype(float)
        total = float(grid.sum())
        if not total > 0:
            raise ValueError("probabilities sum to zero")
        self.offset = offset
        self.step = step
        self.probs = grid / total

    # ── Construction ────────────────────────────────────────────────────────
    @classmethod
    def _from_grid(cls, offset: float, step: float, probs: np.ndarray) -> "Pmf":
        """Trusted fast path: probs already on the grid starting at offset."""
        p = np.asarray(probs, dtype=float)
        total = float(p.sum())
        if p.size == 0 or not total > 0 or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("invalid grid probabilities")
        if p.size > MAX_SUPPORT:
            raise ValueError(f"support of {p.size} grid points exceeds {MAX_SUPPORT}")
        obj = cls.__new__(cls)
        obj.offset = float(offset)
        obj.step = float(step)
        obj.probs = p / total
        return obj._trimmed()

    def _trimmed(self) -> "Pmf":
        """Drop exactly-zero mass at the two ENDS of the grid (never inside it,
        see the module docstring)."""
        nz = np.flatnonzero(self.probs > 0)
        lo, hi = int(nz[0]), int(nz[-1])
        if lo == 0 and hi == self.probs.size - 1:
            return self
        obj = Pmf.__new__(Pmf)
        obj.offset = self.offset + lo * self.step
        obj.step = self.step
        obj.probs = self.probs[lo:hi + 1].copy()
        return obj

    @classmethod
    def point(cls, value: float, step: float = 1.0) -> "Pmf":
        """All mass on one value."""
        if not math.isfinite(float(value)):
            raise ValueError("value must be finite")
        return cls._from_grid(float(value), step, np.array([1.0]))

    @classmethod
    def from_samples(cls, x, step: float = 1.0, weights=None) -> "Pmf":
        """Empirical pmf of simulated draws, binned to the nearest grid point.

        step=1.0 for counts, 0.1 for Fantasy Score (1.2 REB and 1.5 AST put
        every Fantasy outcome on the tenths). Non-finite draws RAISE: a NaN in
        a simulation is a bug upstream, and dropping it would quietly reweight
        the distribution.
        """
        a = np.asarray(x, dtype=float).ravel()
        if a.size == 0:
            raise ValueError("no samples")
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite sample")
        step = float(step)
        if not (math.isfinite(step) and step > 0):
            raise ValueError("step must be positive")
        idx = np.rint(a / step).astype(np.int64)
        lo = int(idx.min())
        if int(idx.max()) - lo + 1 > MAX_SUPPORT:
            raise ValueError("samples span too many grid points")
        w = None
        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != a.size or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("weights must be finite, non-negative and match x")
        counts = np.bincount(idx - lo, weights=w).astype(float)
        return cls._from_grid(lo * step, step, counts)

    @classmethod
    def from_poisson(cls, lam: float) -> "Pmf":
        """Poisson(lam) on the integers, truncated where the tail is negligible.
        Computed in log space with a cumulative log-factorial, so a large mean
        never underflows exp(-lam)."""
        lam = float(lam)
        if not math.isfinite(lam) or lam < 0:
            raise ValueError(f"Poisson mean must be finite and >= 0, got {lam!r}")
        if lam == 0:
            return cls.point(0.0)
        hi = int(math.ceil(lam + 12.0 * math.sqrt(lam) + 25.0))
        k = np.arange(hi + 1, dtype=float)
        log_fact = np.concatenate(([0.0], np.cumsum(np.log(k[1:]))))
        logp = -lam + k * math.log(lam) - log_fact
        p = np.exp(logp - logp.max())
        p[p < _TAIL * p.max()] = 0.0
        return cls._from_grid(0.0, 1.0, p)

    @classmethod
    def from_negbin(cls, mean: float, dispersion: float) -> "Pmf":
        """Negative binomial with the given mean and variance-to-mean ratio.

        `dispersion` is var / mean, the unit docs/data_notes.md measures in
        (PTS ~2.2, REB/AST/3PM ~1.0, PF ~0.8). As it falls to 1 this converges
        to Poisson, and at or below 1 it IS Poisson: a negative binomial cannot
        be under-dispersed, so PF's 0.8 is served slightly too wide rather than
        by a different family. Dependency-free: the pmf is built from the ratio
        p(k+1)/p(k) = (k + r)/(k + 1) * (1 - q) in log space.
        """
        mean, d = float(mean), float(dispersion)
        if not math.isfinite(mean) or mean < 0:
            raise ValueError(f"mean must be finite and >= 0, got {mean!r}")
        if not math.isfinite(d) or d <= 0:
            raise ValueError(f"dispersion must be finite and > 0, got {d!r}")
        if mean == 0:
            return cls.point(0.0)
        if d <= 1.0 + 1e-9:
            return cls.from_poisson(mean)
        r = mean / (d - 1.0)          # size
        q = 1.0 / d                   # success probability
        log1mq = math.log1p(-q)
        sd = math.sqrt(d * mean)
        hi = int(math.ceil(mean + 20.0 * sd + 30.0))
        while True:
            hi = min(hi, MAX_SUPPORT - 1)
            k = np.arange(hi, dtype=float)
            steps = np.log((k + r) / (k + 1.0)) + log1mq
            logp = np.concatenate(([r * math.log(q)], r * math.log(q) + np.cumsum(steps)))
            p = np.exp(logp - logp.max())
            if p[-1] < _TAIL * p.max() or hi >= MAX_SUPPORT - 1:
                break
            hi *= 2
        p[p < _TAIL * p.max()] = 0.0
        return cls._from_grid(0.0, 1.0, p)

    @classmethod
    def from_dict(cls, d: dict) -> "Pmf":
        """Inverse of to_dict. Cached pmfs are stored this way, never draws."""
        return cls._from_grid(float(d["offset"]), float(d["step"]),
                              np.asarray(d["probs"], dtype=float))

    def to_dict(self) -> dict:
        return {"offset": self.offset, "step": self.step,
                "probs": [float(x) for x in self.probs]}

    # ── Shape ───────────────────────────────────────────────────────────────
    @property
    def values(self) -> np.ndarray:
        return self.offset + self.step * np.arange(self.probs.size, dtype=float)

    def __len__(self) -> int:
        return int(self.probs.size)

    def __repr__(self) -> str:
        return (f"Pmf(n={self.probs.size}, step={self.step:g}, "
                f"mean={self.mean():.4g}, median={self.median():.4g})")

    def support(self) -> tuple[float, float]:
        """(smallest, largest) value carrying mass."""
        nz = np.flatnonzero(self.probs > 0)
        v = self.values
        return float(v[nz[0]]), float(v[nz[-1]])

    def mean(self) -> float:
        return float(np.dot(self.values, self.probs))

    def var(self) -> float:
        mu = self.mean()
        return float(np.dot((self.values - mu) ** 2, self.probs))

    def sd(self) -> float:
        return math.sqrt(max(self.var(), 0.0))

    def rescale(self, factor: float) -> "Pmf":
        """Multiply every value by `factor` (> 0). A pmf built in tenths of a
        Fantasy point becomes one in points with rescale(0.1)."""
        f = float(factor)
        if not (math.isfinite(f) and f > 0):
            raise ValueError("factor must be positive and finite")
        return Pmf._from_grid(self.offset * f, self.step * f, self.probs.copy())

    # ── Lines ───────────────────────────────────────────────────────────────
    def split(self, line: float) -> tuple[float, float, float]:
        """(P(Y < line), P(Y == line), P(Y > line)). Sums to one; computed
        from one cumulative sum so the three agree with the median."""
        line = float(line)
        if not math.isfinite(line):
            raise ValueError("line must be finite")
        n = self.probs.size
        c = np.cumsum(self.probs)
        total = float(c[-1])
        t = (line - self.offset) / self.step
        k = math.floor(t + 0.5)
        if abs(t - k) <= _TIE_TOL:
            i = int(k)
            if i < 0:
                return 0.0, 0.0, total
            if i >= n:
                return total, 0.0, 0.0
            lt = float(c[i - 1]) if i > 0 else 0.0
            eq = float(self.probs[i])
            return lt, eq, max(total - lt - eq, 0.0)
        i = int(math.floor(t)) + 1          # first grid index above the line
        if i <= 0:
            return 0.0, 0.0, total
        if i >= n:
            return total, 0.0, 0.0
        lt = float(c[i - 1])
        return lt, 0.0, max(total - lt, 0.0)

    def p_gt(self, line: float) -> float:
        return self.split(line)[2]

    def p_eq(self, line: float) -> float:
        return self.split(line)[1]

    def p_lt(self, line: float) -> float:
        return self.split(line)[0]

    def p_ge(self, line: float) -> float:
        lt, _eq, _gt = self.split(line)
        return max(1.0 - lt, 0.0)

    def p_le(self, line: float) -> float:
        _lt, _eq, gt = self.split(line)
        return max(1.0 - gt, 0.0)

    def cdf(self, x: float) -> float:
        """P(Y <= x)."""
        return self.p_le(x)

    def quantile(self, q: float) -> float:
        """Smallest grid value v with P(Y <= v) >= q, for q in [0, 1]."""
        q = float(q)
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0, 1]")
        c = np.cumsum(self.probs)
        i = int(np.searchsorted(c, q - 1e-12, side="left"))
        i = min(max(i, 0), self.probs.size - 1)
        return float(self.offset + i * self.step)

    def median(self) -> float:
        """The mid-p median (module docstring): where the CDF, with each atom
        spread over its half-step interval, crosses one half. Agrees with the
        lean's sign on every whole and half line."""
        c = np.cumsum(self.probs)
        i = int(np.searchsorted(c, 0.5, side="left"))
        i = min(i, self.probs.size - 1)
        f_prev = float(c[i - 1]) if i > 0 else 0.0
        p = float(self.probs[i])
        if p <= 0:                   # unreachable for a crossing atom; defensive
            return float(self.offset + i * self.step)
        v = self.offset + i * self.step
        return float(v - self.step / 2.0 + self.step * (0.5 - f_prev) / p)

    mid_p_median = median

    # ── Re-centring ─────────────────────────────────────────────────────────
    def recentre(self, target_mean: float, lower: float | None = None) -> "Pmf":
        """A pmf on the same grid whose mean() equals `target_mean`.

        Spec 3.5: every distribution is re-centred on the SHIPPED mean, so the
        number the model gate scored and the distribution the props layer reads
        cannot drift apart.

        Method: exponential tilting, p_i * exp(theta * v_i) renormalised, with
        theta found by bisection (the tilted mean is increasing in theta). Of
        all distributions on the same support with the target mean, this is the
        one closest to the original in KL divergence, so it moves the shape no
        more than the mean requires and never creates mass where there was none.
        A target at or outside the support's edge cannot be reached by tilting,
        so the grid is first translated by whole steps (piling anything below
        `lower` onto the first grid point at or above it), then tilted. A point
        mass is split between the two grid points around the target.
        """
        target = float(target_mean)
        if not math.isfinite(target):
            raise ValueError("target mean must be finite")
        if lower is not None and target < float(lower) - 1e-12:
            raise ValueError(f"target {target} is below the lower bound {lower}")
        tol = 1e-12 * max(1.0, abs(target), self.step)
        mu = self.mean()
        if abs(mu - target) <= tol:
            return Pmf._from_grid(self.offset, self.step, self.probs.copy())
        vlo, vhi = self.support()
        if vlo + tol < target < vhi - tol:
            return self._tilt(target)
        # Outside (or on the edge of) the support: translate, then tilt.
        k = int(round((target - mu) / self.step))
        shifted = Pmf._from_grid(self.offset + k * self.step, self.step, self.probs.copy())
        if lower is not None:
            shifted = shifted._floor_at(float(lower))
        vlo, vhi = shifted.support()
        if abs(shifted.mean() - target) <= tol:
            return shifted
        if vlo + tol < target < vhi - tol:
            return shifted._tilt(target)
        return Pmf._two_point(target, self.step, self.offset)

    def _floor_at(self, lower: float) -> "Pmf":
        """Pile all mass below `lower` onto the first grid point >= lower."""
        v = self.values
        below = v < lower - 1e-12
        if not below.any():
            return self
        j = int(math.ceil((lower - self.offset) / self.step - _GRID_TOL))
        if j >= self.probs.size:
            return Pmf.point(self.offset + j * self.step, self.step)
        p = self.probs.copy()
        moved = float(p[below].sum())
        p[below] = 0.0
        p[j] += moved
        return Pmf._from_grid(self.offset, self.step, p)

    @staticmethod
    def _two_point(target: float, step: float, offset: float) -> "Pmf":
        """Mass on the two grid points bracketing `target`, mean exactly target."""
        t = (target - offset) / step
        a = math.floor(t)
        w = t - a
        if w <= 1e-12:
            return Pmf._from_grid(offset + a * step, step, np.array([1.0]))
        return Pmf._from_grid(offset + a * step, step, np.array([1.0 - w, w]))

    def _tilt(self, target: float) -> "Pmf":
        mask = self.probs > 0
        v = self.values[mask]
        logp = np.log(self.probs[mask])
        scale = max(self.sd(), self.step)
        u = (v - target) / scale

        def tilted_mean(theta: float) -> float:
            w = logp + theta * u
            w = np.exp(w - w.max())
            return float(np.dot(w, u) / w.sum())

        g0 = tilted_mean(0.0)
        sign = 1.0 if g0 < 0 else -1.0
        lo, hi = 0.0, 1.0
        while sign * tilted_mean(sign * hi) < 0 and hi < 2.0 ** 80:
            lo, hi = hi, hi * 2.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if sign * tilted_mean(sign * mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-15 * max(1.0, hi):
                break
        theta = sign * (lo + hi) / 2.0
        w = logp + theta * u
        w = np.exp(w - w.max())
        p = np.zeros_like(self.probs)
        p[mask] = w / w.sum()
        return Pmf._from_grid(self.offset, self.step, p)
