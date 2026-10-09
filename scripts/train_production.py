"""Train the PRODUCTION models and write models/{name}_v1.joblib plus
models/registry_v1.json.

    python scripts/train_production.py [--out models] [--check-rows 3000]

The configuration is the one that passed the ship gate (bucketlib.models.HGB;
scripts/train_eval.py made the gated predictions). The one change is the
window: every regular season 2018-19..2025-26 now trains the shipped model,
because the gate is spent and nothing is tuned on any season.

Before saving it runs models.check_nulls() on a sample of 2025-26 rows: the
registry's null-fraction reference must accept the very rows it was built from.
After saving it reloads through models.load() (registry, sha256 and scikit-learn
checks) and requires the loaded models to predict EXACTLY what the in-memory
ones do on that sample. A failure raises: do not commit the files.
Takes several minutes per model (17 models) and ~2-3 GB of memory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucketlib import cache as C  # noqa: E402
from bucketlib import models as M  # noqa: E402
import validate_full as VF  # noqa: E402


def round_trip_check(fitted: M.Models, out_dir: Path, sample: pd.DataFrame) -> None:
    """Loaded-from-disk predictions must equal in-memory ones bit for bit."""
    loaded = M.load(out_dir)
    pd.testing.assert_frame_equal(M.predict(loaded, sample), M.predict(fitted, sample),
                                  check_exact=True)


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(M.MODELS_DIR))
    ap.add_argument("--check-rows", type=int, default=3000)
    args = ap.parse_args()
    out_dir = Path(args.out)
    t0 = time.time()
    table = C.read_parquet(VF.TABLE)
    if table is None:
        raise FileNotFoundError(f"{VF.TABLE} is missing: run scripts/build_train_table.py")
    print(f"table {table.shape} read in {time.time() - t0:.0f}s", flush=True)

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    fitted = M.train(table, log=log)
    last = table[(table["season"] == M.TRAIN_SEASONS[-1]) & table["game_id"].str.startswith("002")]
    sample = last.sample(n=min(args.check_rows, len(last)), random_state=0)
    M.check_nulls(fitted, sample)
    log(f"null fractions OK on {len(sample)} {M.TRAIN_SEASONS[-1]} rows")
    reg = M.save(fitted, out_dir)
    round_trip_check(fitted, out_dir, sample)
    log(f"round trip OK on {len(sample)} {M.TRAIN_SEASONS[-1]} rows")

    pred = M.predict(fitted, sample)
    pl = sample["played"].astype(bool)
    print(f"in-sample sanity ({M.TRAIN_SEASONS[-1]} sample): mean p_play {pred['p_play'].mean():.3f} "
          f"vs played {pl.mean():.3f}; played rows mean min {pred.loc[pl, 'mu_min'].mean():.1f} "
          f"vs {sample.loc[pl, 'min'].mean():.1f}, pts {pred.loc[pl, 'pts'].mean():.2f} "
          f"vs {sample.loc[pl, 'pts'].mean():.2f}")
    print(f"model_version {reg['model_version']}; trained through {reg['trained_through']}; "
          f"{len(reg['features'])} features; "
          f"n_rows availability {reg['n_rows']['availability']}, played {reg['n_rows']['minutes']}")
    print("n_iter " + ", ".join(f"{k} {v}" for k, v in reg["n_iter"].items()))
    model_bytes = sum((out_dir / f).stat().st_size for f in reg["files"])
    print(f"model files {model_bytes / 1e6:.1f} MB; {out_dir} total {dir_size(out_dir) / 1e6:.1f} MB")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
