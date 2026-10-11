"""How much does training/scoring on CLOSING lines flatter the models?

Closing lines already price late scratches; at a 5:30pm-the-day-before horizon
the site sees an earlier line. Same trained models, test folds scored twice:
with the closing line (as trained) and with the OPENING line swapped into every
line feature (open exists for 2023-24+ = both folds; features.apply_lines, the
one definition of the line features). The gap is the optimism attributable to
line timing (spec 3.2: "validation runs twice ... and reports the gap").

  Part 1, minutes (as first reported, 2026-10-09): the gated minutes regressor
    on played rows, RMSE per horizon, plus a no-lines model for reference.
    Rows without an open line keep their close (open coverage is printed).
  Part 2, availability and the board means (added 2026-10-09 after the
    leakage audit): the PRODUCTION structure, models.train on the gate's
    training window (2018-19..2022-23) and models.predict, on the rows of
    each fold with both opening values, so close and open score the same rows:
      P(play)      log loss and Brier on every rostered row (played, DNP,
                   inactive), per horizon
      PTS, Fantasy E[stat | plays] = minutes x rate, MAE and RMSE on played rows
    and the stratum where late news lives: rows whose spread moved 3+ points
    from open to close, at the two early horizons.
    Written to docs/open_vs_close_v1.csv.

    python scripts/exp_open_vs_close.py [--part 1|2|all]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bucketlib import cache as C  # noqa: E402
from bucketlib import features as F  # noqa: E402
from bucketlib import models as M  # noqa: E402
from bucketlib import train as T  # noqa: E402

LINE = ["home_line", "total", "team_line", "implied_total", "abs_line"]
FOLDS = ("2023-24", "2024-25")
HORIZONS = ["d1_1730", "d0_1100", "t_120", "t_15"]
EARLY = ["d1_1730", "d0_1100"]
TRAIN_WINDOW = T.seasons("2018-19", "2022-23")
BIG_MOVE = 3.0                      # spread points, open to close
MIN_STRATUM = 200
OUT = Path(__file__).resolve().parent.parent / "docs" / "open_vs_close_v1.csv"


def with_open_lines(rows: pd.DataFrame) -> pd.DataFrame:
    """`rows` with every line feature rebuilt from the OPENING spread and total
    (features.apply_lines); a row without an open value keeps its close."""
    lines = pd.DataFrame({"game_id": rows["game_id"],
                          "home_line": rows["open_home_line"].fillna(rows["home_line"]),
                          "total": rows["open_total"].fillna(rows["total"])}).drop_duplicates("game_id")
    out = F.apply_lines(rows.drop(columns=LINE).reset_index(drop=True), lines)
    out.index = rows.index
    return out


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1, random_state=0)


def part1(df: pd.DataFrame, lines: pd.DataFrame) -> None:
    df = df[df.played == True]  # noqa: E712
    feats = T.feature_columns(df)
    tr = df[df.season.isin(TRAIN_WINDOW)]
    m_close = hgb().fit(tr[feats], tr["min"])
    nol = [f for f in feats if f not in LINE]
    m_nol = hgb().fit(tr[nol], tr["min"])
    for fold in FOLDS:
        te = df[df.season == fold].merge(lines, on="game_id", how="left")
        op = with_open_lines(te)
        y = te["min"].to_numpy()
        for h in HORIZONS:
            k = (te.horizon == h).to_numpy()
            r = lambda p: float(np.sqrt(np.mean((y[k] - p[k]) ** 2)))  # noqa: E731
            rc, ro, rn = r(m_close.predict(te[feats])), r(m_close.predict(op[feats])), r(m_nol.predict(te[nol]))
            print(f"{fold} {h}: minutes RMSE close {rc:.3f}  open {ro:.3f}  (+{(ro / rc - 1) * 100:.2f}%)  "
                  f"no-lines model {rn:.3f}")
        print(f"  open line present for {te.open_home_line.notna().mean():.1%} of rows")


def _scores(y_play, p_close, p_open, played, preds) -> list[dict]:
    """One row per metric: close, open, gap %."""
    rows = [{"metric": "p_play_logloss", "n": len(y_play),
             "close": log_loss(y_play, p_close, labels=[0, 1]),
             "open": log_loss(y_play, p_open, labels=[0, 1])},
            {"metric": "p_play_brier", "n": len(y_play),
             "close": float(np.mean((y_play - p_close) ** 2)),
             "open": float(np.mean((y_play - p_open) ** 2))}]
    for stat in ("pts", "fantasy"):
        y = played[stat].to_numpy()
        c, o = preds[stat]
        rows += [{"metric": f"{stat}_mae", "n": len(y), "close": float(np.mean(np.abs(y - c))),
                  "open": float(np.mean(np.abs(y - o)))},
                 {"metric": f"{stat}_rmse", "n": len(y), "close": float(np.sqrt(np.mean((y - c) ** 2))),
                  "open": float(np.sqrt(np.mean((y - o) ** 2)))}]
    for r in rows:
        r["gap_pct"] = (r["open"] / r["close"] - 1) * 100
    return rows


def part2(df: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    models = M.train(df, seasons=TRAIN_WINDOW, log=print)
    out = []
    for fold in FOLDS:
        te = df[(df.season == fold) & df.game_id.str.startswith("002")].merge(lines, on="game_id", how="left")
        have = te.open_home_line.notna() & te.open_total.notna()
        print(f"{fold}: {have.mean():.1%} of rostered rows have both opening values; scored on those")
        te = te[have].reset_index(drop=True)
        op = with_open_lines(te)
        pc, po = M.predict(models, te), M.predict(models, op)
        moved = (te.home_line - te.open_home_line).abs() >= BIG_MOVE
        strata = [(h, "all", te.horizon == h) for h in HORIZONS]
        strata += [(h, f"spread moved {BIG_MOVE:g}+", (te.horizon == h) & moved) for h in EARLY]
        for h, stratum, k in strata:
            k = k.to_numpy()
            if k.sum() < MIN_STRATUM:
                continue
            y = te.loc[k, "played"].astype(int).to_numpy()
            pl = k & te["played"].eq(True).to_numpy()
            preds = {s: (pc.loc[pl, s].to_numpy(), po.loc[pl, s].to_numpy()) for s in ("pts", "fantasy")}
            for r in _scores(y, pc.loc[k, "p_play"].to_numpy(), po.loc[k, "p_play"].to_numpy(),
                             te[pl], preds):
                out.append({"fold": fold, "horizon": h, "stratum": stratum, **r})
    res = pd.DataFrame(out)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(res.round(4).to_string(index=False))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["1", "2", "all"], default="all")
    args = ap.parse_args()
    df = C.read_parquet(C.CACHE_DIR / "train_table_full_v1.parquet")
    df = df[df.game_id.str.startswith("002")]
    lines = T.load_table("game_lines", T.seasons(*FOLDS))[["game_id", "open_home_line", "open_total"]]
    if args.part in ("1", "all"):
        part1(df, lines)
    if args.part in ("2", "all"):
        res = part2(df, lines)
        res.to_csv(OUT, index=False, float_format="%.5f")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
