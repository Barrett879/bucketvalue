"""bucketlib.pmf: the distribution every prop number comes out of.

The property tests pin the one fact the board's honesty rests on: the
displayed mid-p median sits on the same side of a whole or half line as the
lean computed from P(>) and P(<) with ties excluded. Random pmfs, seeded.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from bucketlib.pmf import Pmf

REPO = Path(__file__).resolve().parent.parent


def _random_pmf(rng, step=1.0, holes=True) -> Pmf:
    n = int(rng.integers(1, 40))
    p = rng.dirichlet(np.full(n, float(rng.choice([0.2, 1.0, 5.0]))))
    if holes and n > 3 and rng.random() < 0.5:
        p[rng.integers(1, n - 1, size=int(rng.integers(1, max(2, n // 3))))] = 0.0
    if p.sum() == 0:
        p[0] = 1.0
    offset = float(rng.integers(0, 30)) * step
    return Pmf(offset + step * np.arange(n), p, step=step)


def _lean(pmf: Pmf, line: float, band: float = 0.0) -> str:
    lt, _eq, gt = pmf.split(line)
    if gt + lt <= 0:
        return "Even"
    q = gt / (gt + lt)
    return "More" if q > 0.5 + band else "Less" if q < 0.5 - band else "Even"


# ── Construction ─────────────────────────────────────────────────────────────
def test_probabilities_are_normalised_and_duplicates_merged():
    p = Pmf([2, 0, 2, 1], [1, 1, 1, 1])
    assert list(p.values) == [0.0, 1.0, 2.0]
    assert p.probs.tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert p.probs.sum() == pytest.approx(1.0)


def test_zero_probability_interior_points_are_kept_on_the_grid():
    """Dropping them would widen the neighbours' half-step intervals and break
    the half-line agreement the module docstring explains."""
    p = Pmf([0, 3], [0.5, 0.5], step=1.0)
    assert len(p) == 4 and p.probs[1] == 0 and p.probs[2] == 0


@pytest.mark.parametrize("values,probs", [
    ([0, 1], [0.5, -0.1]), ([0, 1], [0, 0]), ([0, float("nan")], [0.5, 0.5]),
    ([0, 1], [0.5, float("inf")]), ([], []), ([0, 1, 2], [1, 1]),
])
def test_bad_input_raises(values, probs):
    with pytest.raises(ValueError):
        Pmf(values, probs)


def test_values_off_one_grid_raise():
    with pytest.raises(ValueError):
        Pmf([0.0, 1.0, 2.5], [1, 1, 1], step=1.0)


def test_from_samples_fantasy_grid_is_tenths():
    rng = np.random.default_rng(1)
    pts, reb, ast_ = rng.poisson(20, 5000), rng.poisson(7, 5000), rng.poisson(5, 5000)
    fantasy = pts + 1.2 * reb + 1.5 * ast_
    p = Pmf.from_samples(fantasy, step=0.1)
    assert p.step == pytest.approx(0.1)
    assert p.mean() == pytest.approx(fantasy.mean(), abs=1e-9)
    # every outcome is on the tenths, so a .5 line has no exact landing mass
    # unless a sample landed there, and the three parts always sum to one
    for line in (25.5, 30.0, 34.5, 41.0):
        assert sum(p.split(line)) == pytest.approx(1.0)
    assert p.p_eq(30.0) == pytest.approx(float(np.mean(np.isclose(fantasy, 30.0))), abs=1e-12)


def test_from_samples_rejects_nan():
    with pytest.raises(ValueError):
        Pmf.from_samples([1.0, float("nan")])


def test_poisson_moments_and_large_mean():
    for lam in (0.3, 3.5, 27.0, 600.0):
        p = Pmf.from_poisson(lam)
        assert p.mean() == pytest.approx(lam, rel=1e-6)
        assert p.var() == pytest.approx(lam, rel=1e-4)
    assert Pmf.from_poisson(0.0).values.tolist() == [0.0]


def test_poisson_matches_closed_form():
    p = Pmf.from_poisson(2.0)
    assert p.p_eq(0) == pytest.approx(math.exp(-2.0), rel=1e-9)
    assert p.p_eq(3) == pytest.approx(math.exp(-2.0) * 8 / 6, rel=1e-9)


def test_negbin_mean_and_dispersion():
    for mean, d in ((25.0, 2.2), (3.0, 1.5), (0.7, 3.0), (300.0, 24.0)):
        p = Pmf.from_negbin(mean, d)
        assert p.mean() == pytest.approx(mean, rel=1e-6)
        assert p.var() / p.mean() == pytest.approx(d, rel=1e-4)


def test_negbin_converges_to_poisson_and_underdispersion_falls_back():
    a, b = Pmf.from_negbin(5.0, 1.0 + 1e-7), Pmf.from_poisson(5.0)
    assert np.allclose(a.probs[:20], b.probs[:20], atol=1e-5)
    pf = Pmf.from_negbin(2.6, 0.8)          # PF is under-dispersed
    assert pf.var() == pytest.approx(2.6, rel=1e-4)


def test_negbin_is_wider_than_poisson():
    assert Pmf.from_negbin(2.0, 2.0).p_eq(0) > Pmf.from_poisson(2.0).p_eq(0)


def test_dict_round_trip_and_rescale():
    p = Pmf.from_negbin(300.0, 24.0).rescale(0.1)
    assert p.step == pytest.approx(0.1) and p.mean() == pytest.approx(30.0, rel=1e-6)
    q = Pmf.from_dict(p.to_dict())
    assert q.mean() == pytest.approx(p.mean()) and q.median() == pytest.approx(p.median())


# ── Lines ────────────────────────────────────────────────────────────────────
def test_split_on_whole_and_half_lines():
    p = Pmf([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
    assert p.split(2) == pytest.approx((0.3, 0.3, 0.4))
    assert p.split(1.5) == pytest.approx((0.3, 0.0, 0.7))
    assert p.split(-1) == pytest.approx((0.0, 0.0, 1.0))
    assert p.split(10) == pytest.approx((1.0, 0.0, 0.0))
    assert p.cdf(1) == pytest.approx(0.3)


def test_fantasy_half_line_float_noise_is_a_tie_on_the_grid():
    """23.5 / 0.1 is 234.99999999999997 in floating point; it is still the
    grid point 23.5."""
    p = Pmf.from_samples([23.4, 23.5, 23.5, 23.6], step=0.1)
    assert p.p_eq(23.5) == pytest.approx(0.5)


def test_split_rejects_a_non_finite_line():
    with pytest.raises(ValueError):
        Pmf.from_poisson(3).split(float("inf"))


def test_quantile():
    p = Pmf([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
    assert p.quantile(0.05) == 0 and p.quantile(0.3) == 1 and p.quantile(0.31) == 2
    assert p.quantile(1.0) == 3
    with pytest.raises(ValueError):
        p.quantile(1.5)


# ── The mid-p median ─────────────────────────────────────────────────────────
def test_mid_p_formula_by_hand():
    # F(1) = 0.3 < 0.5 <= F(2) = 0.6, so k = 2 and m = 1.5 + (0.5 - 0.3) / 0.3
    p = Pmf([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])
    assert p.median() == pytest.approx(1.5 + 0.2 / 0.3)


def test_mid_p_on_a_symmetric_pmf_is_the_centre():
    assert Pmf([4, 5, 6], [0.25, 0.5, 0.25]).median() == pytest.approx(5.0)
    assert Pmf.point(7).median() == pytest.approx(7.0)


@pytest.mark.parametrize("step", [1.0, 0.1])
def test_median_sign_agrees_with_the_lean_on_every_whole_and_half_line(step):
    """THE property. Thousands of random pmfs (with zero-mass holes), every
    whole and half line across and beyond the support, both grids."""
    rng = np.random.default_rng(20261020)
    checked = 0
    for _ in range(1500):
        p = _random_pmf(rng, step=step)
        m = p.median()
        lo, hi = p.offset - 2 * step, p.offset + (len(p) + 2) * step
        for k in np.arange(math.floor(lo / step) * 2, math.ceil(hi / step) * 2 + 1):
            line = round(k * step / 2, 10)           # whole AND half grid points
            lean = _lean(p, line)
            if lean == "More":
                assert m > line, (p.to_dict(), line, m)
            elif lean == "Less":
                assert m < line, (p.to_dict(), line, m)
            checked += 1
    assert checked > 50_000


def test_with_the_even_band_a_decided_lean_still_agrees():
    rng = np.random.default_rng(7)
    for _ in range(800):
        p = _random_pmf(rng)
        m = p.median()
        for line in np.arange(p.offset - 1, p.offset + len(p) + 1, 0.5):
            lean = _lean(p, float(line), band=0.01)
            if lean == "More":
                assert m > line
            elif lean == "Less":
                assert m < line


def test_the_ordinary_median_would_contradict_the_lean_on_a_whole_line():
    """Why the mid-p median and not the plain one: here the plain median IS the
    line (gap 0) while P(>) clearly beats P(<)."""
    p = Pmf([6, 7, 8], [0.1, 0.5, 0.4])
    assert p.quantile(0.5) == 7.0
    assert _lean(p, 7.0) == "More"
    assert p.median() > 7.0


# ── Re-centring ──────────────────────────────────────────────────────────────
def test_recentre_hits_the_target_mean_on_the_same_grid():
    rng = np.random.default_rng(11)
    for _ in range(300):
        p = _random_pmf(rng, holes=False)
        lo, hi = p.support()
        if hi - lo < 1:
            continue
        target = float(rng.uniform(lo + 0.05, hi - 0.05))
        r = p.recentre(target)
        assert r.mean() == pytest.approx(target, abs=1e-8 * max(1, target))
        assert r.step == p.step
        assert np.all(r.probs >= 0) and r.probs.sum() == pytest.approx(1.0)
        # same grid: every value of r is a grid point of p
        assert np.allclose((r.values - p.offset) / p.step,
                           np.rint((r.values - p.offset) / p.step))


def test_recentre_up_is_stochastically_larger():
    """Tilting toward a higher mean moves mass up at EVERY line, so no lean can
    flip the wrong way because of the re-centring."""
    p = Pmf.from_negbin(20.0, 2.2)
    up = p.recentre(22.0)
    for line in np.arange(0, 60, 0.5):
        assert up.p_gt(float(line)) >= p.p_gt(float(line)) - 1e-12


def test_recentre_outside_the_support_and_the_lower_bound():
    p = Pmf.from_poisson(0.3)
    r = p.recentre(40.0, lower=0.0)
    assert r.mean() == pytest.approx(40.0, rel=1e-9)
    assert r.support()[0] >= 0
    s = Pmf.from_poisson(9.0).recentre(0.4, lower=0.0)
    assert s.mean() == pytest.approx(0.4, rel=1e-9) and s.support()[0] >= 0
    with pytest.raises(ValueError):
        p.recentre(-1.0, lower=0.0)


def test_recentre_a_point_mass_splits_between_neighbours():
    r = Pmf.point(3).recentre(3.25)
    assert r.values.tolist() == [3.0, 4.0]
    assert r.probs.tolist() == pytest.approx([0.75, 0.25])


def test_recentre_identity_and_bad_target():
    p = Pmf.from_poisson(4.0)
    assert np.allclose(p.recentre(p.mean()).probs, p.probs)
    with pytest.raises(ValueError):
        p.recentre(float("nan"))


def test_pmf_module_imports_no_scipy():
    """Dependency-free on the serving path, as the spec requires."""
    tree = ast.parse((REPO / "bucketlib" / "pmf.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert "scipy" not in mods, mods
