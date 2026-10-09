"""Walk-forward validation: does a model beat both baselines, on both folds?

    python scripts/validate_models.py [--variant logs_v0] [--targets min,pts,...]

Folds (spec 3.4): train <= 2022-23, VALIDATE 2023-24, TEST 2024-25. The 2025-26
holdout is NOT touched here (it is used once, at the end of the programme).

Variant logs_v0: events from the box logs alone (every played-game feature is
exact; availability/participation features are excluded because the logs hold
no absence rows). It answers the first question, whether point-in-time history
plus context beats the baselines at all, while the participation and injury
backfills finish.

Metrics, paired on the same rows (DiamondValue published +4.60% that was -0.07%
because two error columns were averaged over different rows):
  MAE, RMSE, and Poisson deviance for non-negative counts. Relative improvement
  vs each baseline with a 95% PLAYER-CLUSTER bootstrap interval (rows from one
  player are not independent). A target passes when the model beats BOTH
  baselines on BOTH folds with the interval above zero on its primary metric
  (RMSE for minutes and Fantasy, MAE otherwise).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import features as F  # noqa: E402
from bucketlib import train as T  # noqa: E402

TRAIN_FIRST, TRAIN_LAST, VALIDATE, TEST = "2015-16", "2022-23", "2023-24", "2024-25"
DEFAULT_TARGETS = ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "ftm", "fta",
                   "fgm", "fga", "fg3a", "oreb", "dreb", "pf", "pra", "pr", "pa", "ra",
                   "stocks", "fantasy"]
# Logs-only events carry no absence rows, so these are degenerate there.
LOGS_ONLY_EXCLUDE = ("miss_streak", "played_rate_l10", "last_absence_len", "games_since_return",
                     "events_s", "start_share_s", "started_last")
SQUARED = {"min", "fantasy"}


def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def cluster_ci(err_m, err_b, groups, reps=300, seed=0):
    """95% bootstrap CI of the relative improvement 1 - sum(err_m)/sum(err_b),
    resampling PLAYERS with replacement."""
    df = pd.DataFrame({"g": groups, "m": err_m, "b": err_b}).groupby("g")[["m", "b"]].sum()
    m, b = df["m"].to_numpy(), df["b"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(reps, len(m)))
    rel = 1 - m[idx].sum(1) / b[idx].sum(1)
    return float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))


def build_table(variant: str) -> pd.DataFrame:
    seasons_all = T.seasons("2014-15", "2025-26")
    logs = T.load_table("player_games", T.seasons("2011-12", "2025-26"))
    sched = T.load_table("schedule", T.seasons("2014-15", "2026-27"))
    tg = T.load_table("team_games", seasons_all)
    tg = tg[tg["season_type"].isin(["RS", "PI"])].copy()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    pri = F.prior_seasons(logs)
    ev = T.logs_events(logs[logs["season"].isin(seasons_all)], sched)
    t0 = time.time()
    df = F.build(ev, None, tg, T.schedule_dates(sched), pri)
    print(f"features: {df.shape} in {time.time() - t0:.0f}s")
    df = df.merge(logs[["game_id", "person_id", "season_type"]], on=["game_id", "person_id"], how="left")
    return T.add_derived_targets(df)


def fit_predict(train: pd.DataFrame, test_sets: dict, target: str, feats: list[str]):
    y = train[target].to_numpy(dtype=float)
    loss = "squared_error" if target in SQUARED else "poisson"
    m = HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.05,
                                      max_leaf_nodes=31, min_samples_leaf=200,
                                      l2_regularization=1.0, early_stopping=True,
                                      validation_fraction=0.1, random_state=0)
    m.fit(train[feats], y)
    return {k: m.predict(v[feats]) for k, v in test_sets.items()}, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="logs_v0")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = ap.parse_args()
    df = build_table(args.variant)
    rs = df[df["season_type"] == "RS"]
    train = rs[rs["season"].isin(T.seasons(TRAIN_FIRST, TRAIN_LAST))]
    folds = {"validate": rs[rs["season"] == VALIDATE], "test": rs[rs["season"] == TEST]}
    feats = T.feature_columns(df, exclude=LOGS_ONLY_EXCLUDE)
    print(f"{len(feats)} features; train {len(train)} rows; "
          + ", ".join(f"{k} {len(v)}" for k, v in folds.items()))
    rows = []
    for target in args.targets.split(","):
        t0 = time.time()
        preds, _ = fit_predict(train, folds, target, feats)
        league = train[target].mean()
        for fold, data in folds.items():
            y = data[target].to_numpy(dtype=float)
            p_m = preds[fold]
            p_b1 = T.baseline_std(data, target).fillna(league).to_numpy(dtype=float)
            p_b2 = T.baseline_rate(data, target).fillna(pd.Series(p_b1, index=data.index)).to_numpy(dtype=float)
            g = data["person_id"].to_numpy()
            primary = "rmse" if target in SQUARED else "mae"
            rec = {"target": target, "fold": fold, "n": len(y), "primary": primary}
            for name, p in (("model", p_m), ("b1_std", p_b1), ("b2_rate", p_b2)):
                rec[f"mae_{name}"] = float(np.mean(np.abs(y - p)))
                rec[f"rmse_{name}"] = float(np.sqrt(np.mean((y - p) ** 2)))
                if target not in SQUARED:
                    rec[f"dev_{name}"] = float(np.mean(poisson_dev(y, p)))
            for bname, p_b in (("b1", p_b1), ("b2", p_b2)):
                em = np.abs(y - p_m) if primary == "mae" else (y - p_m) ** 2
                eb = np.abs(y - p_b) if primary == "mae" else (y - p_b) ** 2
                lo, hi = cluster_ci(em, eb, g)
                rec[f"gain_vs_{bname}"] = 1 - em.sum() / eb.sum()
                rec[f"ci_vs_{bname}"] = f"[{lo:+.3%},{hi:+.3%}]"
                rec[f"win_vs_{bname}"] = lo > 0
            rows.append(rec)
        print(f"  {target:8s} {time.time() - t0:5.1f}s  "
              + "  ".join(f"{r['fold']}: vs b1 {r['gain_vs_b1']:+.2%} vs b2 {r['gain_vs_b2']:+.2%}"
                          for r in rows[-2:]))
    rep = pd.DataFrame(rows)
    gate = rep.groupby("target")[["win_vs_b1", "win_vs_b2"]].all().all(axis=1)
    rep["passes_gate"] = rep["target"].map(gate)
    out = Path("docs") / f"model_report_{args.variant}.csv"
    rep.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(rep.pivot_table(index="target", columns="fold", values=["gain_vs_b1", "gain_vs_b2"]).round(4).to_string())
    print("\nGATE (beats both baselines on both folds, CI above zero):")
    print(gate.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
