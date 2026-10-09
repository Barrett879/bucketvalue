"""What does projecting WITHOUT a betting line cost? The measurement behind
bucketlib.pipeline's missing-lines policy.

    python scripts/measure_missing_lines.py

Training rows almost always carry a line (2018-19..2025-26: 100% of regular-
season rows have home_line and total, but 36 rows of 2024-25), while tomorrow's
games often have none until the morning. A model that never saw a missing line
routes NaN to whichever side of each split held the most training rows, so the
cost is an empirical question, not a property of the model.

Protocol (walk-forward, gated hyperparameters from bucketlib.models, nothing
tuned): fit the availability model, the minutes model and the per-minute rate
models for PTS, REB, AST, STL, BLK and TOV (the six that compose PRA and
Fantasy) on the 2018-19..2023-24 regular-season rows of
cache/train_table_full_v1.parquet; predict the 2024-25 regular-season rows
twice, with the lines as recorded and with the five line columns (home_line,
total, team_line, implied_total, abs_line) blanked; compare on the SAME rows
(paired): minutes RMSE and bias on played rows, PTS / PRA / Fantasy RMSE and
bias on played rows, P(play) Brier and mean on every row. Also the bias by
spread bucket, because a missing line hides the blowout signal and an
aggregate can net a favourite's over-projection against an underdog's.

The policy rule (task, 2026-10-09): project without lines only if blanking
costs under 2% RMSE and under 1% bias on every one of those; otherwise
withhold line-less games. Prints the table; writes nothing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import models as M  # noqa: E402
from bucketlib import train as T  # noqa: E402

TABLE = C.CACHE_DIR / "train_table_full_v1.parquet"
TRAIN = T.seasons("2018-19", "2023-24")
TEST = "2024-25"
LINE_COLS = ["home_line", "total", "team_line", "implied_total", "abs_line"]
RATES = ["pts", "reb", "ast", "stl", "blk", "tov"]
RMSE_MAX, BIAS_MAX = 0.02, 0.01


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> int:
    t0 = time.time()
    names = pq.ParquetFile(TABLE).schema_arrow.names
    feats = T.feature_columns(pd.DataFrame(columns=names))
    missing = [c for c in LINE_COLS if c not in feats]
    if missing:
        raise KeyError(f"line columns {missing} are not features of {TABLE.name}")
    cols = ["game_id", "season", "horizon", "played", "min", *RATES, *feats]
    table = pd.read_parquet(TABLE, columns=list(dict.fromkeys(cols)))
    train = M.training_rows(table, TRAIN)
    test = M.training_rows(table, [TEST])
    del table
    cover = train[["home_line", "total"]].notna().all(axis=1).mean()
    print(f"{len(feats)} features; train {len(train)} rows ({cover:.4%} with a line), "
          f"test {len(test)} rows ({time.time() - t0:.0f}s)", flush=True)

    clf = M.availability_model().fit(train[feats], train["played"].astype(int))
    played = train[train["played"].eq(True)]
    mins = M.regressor("squared_error").fit(played[feats], played["min"])
    rates = {}
    for s in RATES:
        y, w = M.rate_target(played, s)
        rates[s] = M.regressor("poisson").fit(played[feats], y, sample_weight=w)
    print(f"fitted 2 + {len(RATES)} models ({time.time() - t0:.0f}s)", flush=True)

    blank = test.copy()
    blank[LINE_COLS] = np.nan

    def predict(X: pd.DataFrame) -> pd.DataFrame:
        out = {"p_play": clf.predict_proba(X[feats])[:, 1],
               "mu_min": np.clip(mins.predict(X[feats]), 0.0, M.MAX_MIN)}
        for s in RATES:
            out[s] = out["mu_min"] * rates[s].predict(X[feats])
        out["pra"] = out["pts"] + out["reb"] + out["ast"]
        out["fantasy"] = sum(w * out[s] for s, w in T.FANTASY_W.items())
        return pd.DataFrame(out, index=X.index)

    a, b = predict(test), predict(blank)
    pl = test["played"].eq(True).to_numpy()
    act = test.assign(pra=test["pts"] + test["reb"] + test["ast"],
                      fantasy=sum(w * test[s] for s, w in T.FANTASY_W.items()),
                      mu_min=test["min"])

    rows = []
    y = test["played"].astype(float).to_numpy()
    for label, p in (("lines", a), ("blank", b)):
        rows.append({"target": "p_play", "with": label, "n": len(y),
                     "rmse": float(np.sqrt(np.mean((y - p["p_play"].to_numpy()) ** 2))),
                     "mean_pred": float(p["p_play"].mean()), "mean_actual": float(y.mean())})
        for t in ("mu_min", "pts", "reb", "ast", "pra", "fantasy"):
            yy, pp = act[t].to_numpy(dtype=float)[pl], p[t].to_numpy()[pl]
            rows.append({"target": t, "with": label, "n": int(pl.sum()), "rmse": rmse(yy, pp),
                         "mean_pred": float(pp.mean()), "mean_actual": float(yy.mean())})
    rep = pd.DataFrame(rows)
    wide = rep.pivot(index="target", columns="with", values=["rmse", "mean_pred"])
    mean_actual = rep.groupby("target")["mean_actual"].first()
    summary = pd.DataFrame({
        "rmse_lines": wide[("rmse", "lines")], "rmse_blank": wide[("rmse", "blank")],
        "rmse_cost": wide[("rmse", "blank")] / wide[("rmse", "lines")] - 1,
        "bias_lines": (wide[("mean_pred", "lines")] - mean_actual) / mean_actual,
        "bias_blank": (wide[("mean_pred", "blank")] - mean_actual) / mean_actual,
        "bias_shift": (wide[("mean_pred", "blank")] - wide[("mean_pred", "lines")]) / mean_actual,
    }).loc[["p_play", "mu_min", "pts", "reb", "ast", "pra", "fantasy"]]
    print("\nPaired, 2024-25 regular season (rmse_cost and bias as fractions; p_play rmse = Brier^0.5)")
    print(summary.round(4).to_string())

    # Per horizon: the line-less case is mostly the evening-before and morning runs.
    by_h = []
    for h, idx in test.reset_index(drop=True).groupby("horizon").indices.items():
        m = pl[idx]
        for t in ("mu_min", "pts", "fantasy"):
            yy = act[t].to_numpy(dtype=float)[idx][m]
            pa_, pb_ = a[t].to_numpy()[idx][m], b[t].to_numpy()[idx][m]
            by_h.append({"horizon": h, "target": t, "n": int(m.sum()),
                         "rmse_cost": rmse(yy, pb_) / rmse(yy, pa_) - 1,
                         "bias_shift": float((pb_.mean() - pa_.mean()) / yy.mean())})
    print("\nBy horizon")
    print(pd.DataFrame(by_h).round(4).to_string(index=False))

    # By spread: favourites vs underdogs, where a missing line moves minutes.
    tl = test["team_line"].to_numpy()
    bucket = pd.cut(tl, [-np.inf, -10, -5, -0.01, 5, 10, np.inf],
                    labels=["fav 10+", "fav 5-10", "fav 0-5", "dog 0-5", "dog 5-10", "dog 10+"])
    by_s = []
    for lab in bucket.categories:
        m = pl & (bucket == lab)
        for t in ("mu_min", "pts"):
            yy = act[t].to_numpy(dtype=float)[m]
            by_s.append({"spread": lab, "target": t, "n": int(m.sum()),
                         "bias_lines": float((a[t].to_numpy()[m].mean() - yy.mean()) / yy.mean()),
                         "bias_blank": float((b[t].to_numpy()[m].mean() - yy.mean()) / yy.mean())})
    print("\nBy the team's spread (played rows)")
    print(pd.DataFrame(by_s).round(4).to_string(index=False))

    d = (b - a)[pl]
    print("\nPer-row |blank - lines| on played rows: "
          + ", ".join(f"{t} mean {d[t].abs().mean():.3f} p95 {d[t].abs().quantile(.95):.3f} "
                      f"max {d[t].abs().max():.3f}" for t in ("mu_min", "pts", "fantasy")))

    worst_rmse = float(summary["rmse_cost"].max())
    worst_bias = float(summary["bias_shift"].abs().max())
    ok = worst_rmse < RMSE_MAX and worst_bias < BIAS_MAX
    print(f"\nworst RMSE cost {worst_rmse:.4%} (limit {RMSE_MAX:.0%}), worst |bias shift| "
          f"{worst_bias:.4%} (limit {BIAS_MAX:.0%}): "
          f"{'PROJECT without lines (provisional)' if ok else 'WITHHOLD line-less games'}"
          f"  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
