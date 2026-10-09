"""The simulator's structural promises (parameters are fitted elsewhere)."""
import numpy as np

from bucketlib import dist as D


def star():
    return D.PlayerInput(
        mu_min=35.0,
        rate={"fga": 0.55, "fta": 0.22, "ast": 0.20, "oreb": 0.03, "dreb": 0.18, "stl": 0.03,
              "blk": 0.02, "tov": 0.08, "pf": 0.07},
        fg3a_share=0.40, p3=0.37, p2=0.55, pft=0.85,
        shipped_mean={"pts": 27.4, "reb": 7.1, "ast": 7.0, "pra": 41.5, "fantasy": 47.0})


def test_shots_are_coherent_and_points_compose():
    d = D.simulate(star(), D.SimParams(), seed=1)
    assert (d["fg3m"] <= d["fg3a"]).all() and (d["fg3a"] <= d["fga"]).all()
    assert (d["ftm"] <= d["fta"]).all() and (d["fta"] % 2 == 0).all()
    assert np.array_equal(d["pts"], 2 * d["fg2m"] + 3 * d["fg3m"] + d["ftm"])
    assert np.array_equal(d["reb"], d["oreb"] + d["dreb"])
    assert np.array_equal(d["pra"], d["pts"] + d["reb"] + d["ast"])
    assert (d["pf"] <= 6).all()


def test_marginals_recentred_on_the_shipped_mean():
    p = D.pmfs(star(), D.SimParams(), seed=2, stats=["pts", "reb", "pra", "fantasy"])
    for s, target in (("pts", 27.4), ("reb", 7.1), ("pra", 41.5), ("fantasy", 47.0)):
        assert abs(p[s].mean() - target) < 1e-6, s


def test_points_overdispersed_assists_near_poisson():
    d = D.simulate(star(), D.SimParams(), seed=3, n=20000)
    assert d["pts"].var() / d["pts"].mean() > 1.5
    assert 0.8 < d["ast"].var() / d["ast"].mean() < 1.6


def test_minutes_use_the_fitted_residual_table():
    params = D.SimParams(min_resid={str(D.bucket_of(35.0)): [-20.0, 0.0, 0.0, 0.0]})
    d = D.simulate(star(), params, seed=4, n=4000)
    share_blowout = np.mean(d["min"] < 20)
    assert 0.2 < share_blowout < 0.3      # a quarter of draws took the -20 residual


def test_deterministic_given_seed():
    a = D.simulate(star(), D.SimParams(), seed=9)
    b = D.simulate(star(), D.SimParams(), seed=9)
    assert all(np.array_equal(a[k], b[k]) for k in a)
