# Serving design (2026-10-09): the contract every serving module builds against

The gate passed (docs/model_report.md). This is how the gated models reach the
site. It pins module boundaries, file layouts and column names so modules built
in parallel agree. The spec (BUCKETVALUE_PROJECT_INSTRUCTIONS.md, sections 3.5,
4 and 5) still governs behaviour; this file only fixes the interfaces.

## 0. Principles (each one is a sibling bug)

- ONE feature path. Serving calls `features.build()` exactly as training did,
  with the history assembled from committed files. No second implementation of
  any feature, ever. Parity is proven by a test that runs the SERVE path on real
  data and compares to the training table bit for bit.
- The APP never runs `features.build()` (430k-row history is ~0.5 GB of
  columns; Streamlit Cloud has ~1 GB). The BOT runs it and commits per-date
  history-feature rows; the app adds only the volatile parts (injury context,
  lines, mark-outs) and runs the models.
- The bot writes ONLY under `state/`. The app writes nothing to disk.
- Freeze per game at tip: no process ever rewrites a projection row for a game
  whose tip has passed.
- Accuracy stays local (Barrett). The record is derived LOCALLY from the git
  history of the committed projection files; nothing record-like is committed.

## 1. Committed state: `state/` (bot-owned, one writer per path)

```
state/seed/events_v1.parquet          events through 2025-26 (EVENT_COLS + min + is_target),
                                      exactly the frame build_train_table.py passes to build():
                                      log-only warm-up 2014-15..2017-18 + participation
                                      events 2018-19..2025-26 (about 430k rows, ~5 MB)
state/seed/team_games_v1.parquet      team_games 2014-15..2025-26, RS + PI only (~0.7 MB)
state/seed/priors_v1.parquet          F.prior_seasons(logs 2011-12..2025-26), rows with
                                      season_end == 2027 only (the season this seed serves)
state/seed/schedule_dates_v1.parquet  T.schedule_dates(schedule 2014-15..2025-26)
state/seed/names_v1.parquet           (person_id, player_name): every spelling a person id has
                                      carried, participation + logs (injury name resolution)
state/seed/manifest_v1.json           seed_through "2025-26", serves "2026-27", row counts,
                                      content hashes, built_at
state/season/{season}/{pg,tg,part,games,odds}_{YYYY-MM-DD}.parquet
                                      bot partitions (moved here from cache/season/)
state/current/schedule_v1.parquet     cdn schedule (moved from cache/schedule_current_v1)
state/current/roster_v1.parquet       cdn playerIndex (moved from cache/roster_current_v1)
state/current/injury_players_v1.parquet  latest validated report, player rows
state/current/injury_teams_v1.parquet    latest report, per (game_id, team) state rows
state/serve/{YYYY-MM-DD}/features_v1.parquet     history features for that date's games
state/serve/{YYYY-MM-DD}/projections_v1.parquet  bot projections with the bot's context
```

`bucketlib/state.py` owns every path above (`STATE_DIR = cache.REPO_ROOT / "state"`)
and the readers. A seed for a season other than `manifest["serves"]` raises
`SeedMismatch` with "rebuild the seed" (priors for 2027-28 need the 2026-27 logs).

## 2. Modules and their public functions

### bucketlib/state.py
- paths: `SEED_DIR`, `season_dir(season)`, `current_path(name)`, `serve_dir(date)`.
- `load_seed() -> Seed` (dataclass: events, team_games, priors, schedule_dates, names, manifest).
- `season_partitions(season) -> dict[str, DataFrame]` with keys participation, games,
  player_games, team_games (concatenated `part_`, `games_`, `pg_`, `tg_` files).
- `history(season, before: date) -> History` (dataclass: events, team_games,
  schedule_dates, priors, names): seed + the season's partitions, every event
  STRICTLY before `before` (ET date). The partitions go through the SAME join and
  reconciliation as `tables.roster_games` (refactor that function to accept frames;
  never copy it).

### bucketlib/serve.py
- `targets_for(games, roster, season) -> DataFrame`: one row per (current roster
  player, game) for both teams; columns person_id, game_id, season, game_date,
  tip_utc, team, opp, home, player_name, two_way. Feeds `features.make_targets`.
- `history_features(date, games, roster, hist: History, schedule) -> DataFrame`:
  `features.build(hist.events, make_targets(targets), hist.team_games,
  schedule_dates incl. the target games, hist.priors, lines=None)`. Lines and
  context are added later.
- `completeness(date, schedule, season) -> list[str]`: game ids of FINAL state
  games before `date` involving a team that plays on `date` and not ingested.
  Non-empty means stale state: projections are provisional and never freezable.

### bucketlib/features.py (one addition)
- `apply_lines(out, lines)`: the existing lines block of `build()` moved into a
  function that `build()` calls; `lines` is (game_id, home_line, total). The
  app and bot call it on committed feature rows with live lines.

### bucketlib/context.py
- `report_states(games, schedule, report) -> (player_rows, team_rows)`: the
  per-game state logic of `horizons.snapshot_games` for ONE already-chosen
  report (refactor horizons so both use it; never copy it). Same states, same
  columns, `horizon="live"`, `cutoff_utc = min(now, tip - 5 min)`.
- `context_for(rows, games, schedule, report, now, names, marked_out=None) -> DataFrame`:
  resolve names (`livectx.resolve_names`, team-scoped, every spelling) and
  return `livectx.build_context(...)` output. `report=None` means no report:
  every team `no_report` (UNKNOWN, never healthy).
- `display_status(rows) -> DataFrame` (person_id, game_id, status, reason) for badges.

### bucketlib/models.py
- `HGB` configs (the gated hyperparameters, one definition; train_eval/fit_dist
  keep their own copies for reproducibility of the reports).
- `train(table) -> Models` and `save(models, dir)`, writing `models/{name}_v1.joblib`
  and `models/registry_v1.json` {model_version, trained_at, train_seasons,
  n_rows, features (ordered), stats, sklearn, joblib, numpy, python, files:
  {name: sha256}}.
- `load(dir=MODELS_DIR) -> Models`: raises `ModelMismatch` when the runtime
  scikit-learn differs from the registry, a sha256 differs, or a file is missing.
- `predict(models, rows) -> DataFrame` indexed like rows with p_play, mu_min
  (clip 0..53), rate_<s> for the 15 STATS, and E[stat | plays] as plain stat
  names: pts reb ast fg3m stl blk tov ftm fta fgm fga fg3a oreb dreb pf, plus
  pra pr pa ra stocks fantasy. Raises `MissingFeatures` naming every registry
  feature absent from rows (never fills).

### bucketlib/simprov.py
- `player_input(row) -> dist.PlayerInput` (moved from scripts/prob_gate.py, which
  then imports it).
- `SimProvider(projections, params, n=3000)`: `pmf(person_id, game_id, stat)`,
  `source(...) -> "model"`; per (person, game) one `dist.pmfs` call, cached; seed
  = zlib.crc32 of "person_id:game_id" so a rerun shows the same numbers.
- `medians(projections, params, stats) -> DataFrame` (person_id, game_id,
  med_<stat>) from the same cached pmfs (mid-p median).

### bucketlib/pipeline.py (phase 2)
- `project(feature_rows, games, schedule, report, lines, models, now,
  marked_out=None, provisional=False) -> DataFrame`: apply_lines -> context_for
  -> predict -> the projection frame (section 3).
- `live_projections(date, now, marked_out=None) -> (DataFrame, Info)` for the
  app: committed features + live report + live odds; tipped games come from the
  committed projections untouched; falls back to the committed projections
  (stale notice) when the live layer fails.

## 3. The projection frame (bot file and app frame, same columns)

identity: person_id, game_id, game_date, tip_utc, team, opp, home, player_name,
two_way, season
availability: p_play, inj_status (Out/Doubtful/Questionable/Probable/Available
or None), inj_reason, team_state
means (E[stat | plays]): min (= mu_min), pts reb ast fg3m stl blk tov ftm fta
fgm fga fg3a oreb dreb pf pra pr pa ra stocks fantasy
simulator inputs: mu_min, rate_<s> for the 15 STATS
props contract: games_prior (= career_games, played games before this one),
provisional (bool, never missing), model_version
provenance: report_ts_et, home_line, total, computed_at (UTC), frozen (bool)

## 4. Bot (scripts/daily_update.py) and Actions

Step 6 projections, for D in (today ET, tomorrow ET), state games only (RS, PI):
history(season, before=D) -> history_features -> write features; then project
with the bot's report and odds -> write projections, rows of games with
tip <= now carried over from the existing file unchanged. Tomorrow is
provisional. `if_changed=True` everywhere. `.github/workflows/daily.yml` ships
with `workflow_dispatch` only: the cron block stays commented until the deploy
STOP (a private repo spends Barrett's Actions minutes).

## 5. App

Game page tables from `live_projections`: MIN (mean), then mid-p medians "if he
plays"; P(play) only when listed or below 90%; status badges from the report
(Unknown is never rendered as healthy); Out rows greyed at the bottom with the
reason and no numbers; mark-out via `?out=` with the "your adjustments" banner;
pasted lines per player. Slate: the board (`props_ui.render_board`) fed by
`SimProvider`, card line-count pills, "Out: X, Y +2". About: the trust table
generated from data/stat_reliability.csv.

## 6. Amendments after build 1 (2026-10-09, verified and fixed)

- **models**: the registry nests versions under `versions` (also mirrored at the
  top level), records `trained_through` (props' training-window guard reads it)
  and `null_reference` (training null fractions per games-into-season bucket and
  team change). `models.check_nulls(models, rows)` raises `NullSkew` for a
  serve slate whose missingness training never saw (no injury report at all,
  missing lines). The pipeline calls it before `predict`.
- **state**: uningestable games (cdn box parse failures; the backfill used V3)
  are committed as overrides under `state/season/{season}/override/`
  (`scripts/build_serving_seed.py --override GAME_ID`); `state.ingested_ids` is
  the ONE definition of ingested and includes them. cdn box counts differ from
  the stats logs only by later stat corrections (148 of 26,757 player lines,
  at most 2 per count); the parity test proves the code path bit for bit and
  bounds the effect on predictions (p_play 0.014, minutes 0.14, PTS 0.24 at
  worst). Accepted: training keeps the corrected logs.
- **context**: the committed report is written by
  `context.write_committed(report, schedule)` (both files carry report_ts_et on
  every row; a QA or non-NBA report is refused: stale beats empty).
  `display_status` reads only inj_status, inj_reason, team_state. From tip minus
  5 minutes the live report is titled after the cutoff, so the app serves the
  COMMITTED projection for any game with now >= tip - 5 min.
- **simprov**: only the 13 gated stats come from the simulator (source
  "model"); the other ten come from `props.MeanDispersionProvider` (source
  "baseline", tier unmeasured) until the gate covers them (FTA needs and-ones in
  dist.py first). Build providers with `simprov.provider_for(...)` so the board
  and the Game tables share one simulation. A frame missing a required column
  raises `MissingColumns`.

## 7. Amendments after build 2 (2026-10-09)

- Projection frame: `adjusted` (row recomputed with the visitor's mark-outs)
  joins section 3; `frozen` is always False in the bot's file and set by the
  app. Files carry metadata `history` (the ingested games the features were
  built from, so the app can tell stale features) and a per-game input hash
  (`basis`) so a rerun with unchanged inputs leaves the file byte-identical.
- Freeze point is tip minus 5 minutes (`pipeline.FREEZE_LEAD`); a provisional
  row is never frozen, and a game that tips on provisional rows has no frozen
  projection (the local record skips it).
- No injury report at all: `pipeline.project` raises `NoReport`; nothing is
  published (bot) and the app serves the committed projections.
- Missing lines: WITHHELD ("Projections post when the line does"). Measured in
  `scripts/measure_missing_lines.py`: overall cost under 1% RMSE, but up to
  5.9% minutes bias for 10+ point favourites.
- A team with fewer than 12 projected rows (`pipeline.MIN_TEAM_ROWS`, the
  fewest in 19,112 training team-games) turns today's run RED.
- Board voids come from the served frame only (`props_ui.void_ids`), so a
  mark-out the pipeline did not apply (frozen game, stale fallback) never voids
  a line the page still shows.

## 8. Amendments after the pre-opening audit (2026-10-09)

Seven lenses (leakage, bot operations, app, props, sources, math, an opening-
night rehearsal on the 2025-26 opener); 62 findings confirmed by a skeptic and
fixed by area. What changed in the contract:

- **Completeness is per game and counts every earlier game** of either team that
  is not ingested, whatever its status (a late West Coast game still in progress
  when the bot runs is a gap), postponed games excluded
  (`serve.completeness_by_game`, `provisional_by_game`). A game before today not
  final 6 hours after its tip turns the run RED.
- **Postponed** only on positive evidence (cdn postponedStatus "Y" or a PPD
  status text; "N", "A" and blank are not); a postponed game is no state game, no
  rest day and no gap. The live code for a postponement is unverified until one
  happens (docs/known_issues.md).
- **Play-in is ingested, never projected**: `pipeline.PROJECTED_TYPES = {"RS"}`.
- **The report title is a stamp, not an input.** A same-content re-issue (every
  15 minutes) is not committed until the committed title is 3 hours behind
  (`context.COMMITTED_RESTAMP`); a report out of reach for every game is
  `NoReport`.
- **Null check**: `tm_out_max_min` is empty exactly where the player is listed
  Out (true on all 325,371 training rows); it is checked as that invariant
  (`models.STRUCTURAL_NAN`), not as a share, so a slate with many players Out
  is projected, not withheld.
- **Makes never exceed attempts**: `models.RATE_CEILINGS` caps FG3A and FGM at
  FGA, FG3M at FG3A and FGM, FTM at FTA; simprov refuses a degenerate draw.
- **Bot**: one bad box score never stops the others; a box not final yet is a
  WARN until 6 hours after tip; schedule, roster and odds fall back to the
  committed copy on ANY fetch or parse failure (RED still fires); odds files
  change only when a line does; a team with more than 18 roster rows on a game
  day (camp roster) is RED today; every commit goes through
  `scripts/commit_state.sh` (state/ only, rebase, abort on conflict, a keepalive
  commit after 45 idle days); a dead-man's switch pings `BV_HEALTHCHECK_URL`
  after every run.
- **App**: any exception from a feed or its parser is a feed failure; an empty
  schedule, a short roster or an ESPN body with no games never replaces good
  data; the newer of the memory copy and the committed copy is served; a line
  seen before tip stays after ESPN drops it, and a tipped game shows the line its
  projection used ("Pre-tip line"); actuals after tip come from `live.box` (cdn
  box score, 120 s); Share, `?player=` search, rest/B2B, the projected-starter dot
  (a DISPLAY rule: the five players at least 50% to play with the highest start
  share this season, minutes breaking ties), Upcoming / Postponed / Time TBD
  states. The card's "Out:" line names players at 15+ season minutes per game.
- **Props**: lines for games the site never projects are counted "not
  projected", another league's lines "other board"; a period board is withheld,
  and a re-paste can never replace a full-game line with a period line; adding
  lines mints a NEW token, so a shared link is read-only for everyone else; the
  record refuses mark-out-adjusted rows and rows without a model version.
