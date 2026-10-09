"""Build the committed serving seed: the training history, frozen for the bot.

    python scripts/build_serving_seed.py [--serves 2026-27] [--out state/seed]
    python scripts/build_serving_seed.py --override GAME_ID [GAME_ID ...]

The seed holds exactly the inputs scripts/build_train_table.py passes to
features.build, through the season before the one served (state.seed_inputs):
events (log-only warm-up 2014-15..2017-18 + participation events from 2018-19),
RS + PI team games, the prior-season rows of the served season only, schedule
dates and every name spelling. The bot adds the served season's own games from
its per-date partitions (state.history).

Run LOCALLY (it reads the cache/ backfill, which never reaches Actions), once
per season, after the previous season's tables are complete. A seed serves one
season: priors for 2027-28 need the 2026-27 logs, so the bot refuses a seed
built for another season (state.SeedMismatch). Every file is written with
if_changed=True, so rebuilding unchanged data leaves the commit empty.

--override commits final games the bot can never ingest (a cdn box score the
parser rejects, e.g. 0022500093, short 521 player-seconds) from the local
backfill instead of building the seed: state.override_from_backfill ->
state.write_override, under state/season/{season}/override/. Backfill the game
first (scripts/build_participation.py and build_logs.py reach stats.nba.com,
which the bot cannot); the bot reads the override like its own partitions.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bucketlib import state as S  # noqa: E402
from bucketlib.sources.box import game_season  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serves", default="2026-27", help="the season this seed serves")
    ap.add_argument("--out", default=str(S.SEED_DIR), help="output directory")
    ap.add_argument("--override", nargs="+", metavar="GAME_ID",
                    help="instead of the seed: commit these final games from the local backfill "
                         "as override partitions (games the bot cannot ingest)")
    args = ap.parse_args()
    t0 = time.time()
    if args.override:
        for gid in args.override:
            season = game_season(gid)
            paths = S.write_override(season, S.override_from_backfill(season, gid))
            print(f"override {gid} ({season}): {', '.join(p.name for p in paths)} in {paths[0].parent}")
        return 0
    out = Path(args.out)
    m = S.build_seed(args.serves, out)
    print(f"seed serves {m['serves']} (history through {m['seed_through']}), built {m['built_at']}")
    for key, fname in S.SEED_FILES.items():
        size = (out / fname).stat().st_size / 1e6
        print(f"  {fname:28s} {m['counts'][key]:>8,} rows  {size:6.2f} MB  {m['content_hashes'][key][:12]}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
