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
- The bot writes ONLY under `state/`. The app writes nothing to disk. (A
  visitor's pasted board is copied to THAT visitor's browser, localStorage, by
  props_ui.board_keeper, so a reload or a restart brings it back; the server
  still keeps it only in memory.)
- Licensed player lines (The Odds API) are bought by the APP's live layer only, for today's
  ET date, and kept in process memory only: the feed anchor, a sys.modules entry outside the
  bucketlib package, kept across a freshen purge and a Streamlit cache clear, lost on a
  restart. They never reach disk, git, a visitor's token board or a browser copy
  (section 10).
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
- `cohere(pred, rows) -> DataFrame` (added 2026-11-03): predict's output with
  each team-game's mu_min moved toward adding up (sum of p_play x mu_min toward
  241.5 team minutes, part of the way when who plays is uncertain, the
  heaviest players moved least, rows listed Out untouched) and the means
  recomposed from it. `rows` must hold every roster row of each team-game.
  docs/model_report.md, "Team minutes coherence".

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
  -> predict -> cohere -> the projection frame (section 3).
- `live_projections(date, now, marked_out=None) -> (DataFrame, Info)` for the
  app: committed features + live report + live odds; tipped games come from the
  committed projections untouched; falls back to the committed projections
  (stale notice) when the live layer fails.

The licensed odds feed's modules (2026-11-11; section 10 has the contract):

### bucketlib/live.py: the odds feed
- `feed_kick(work, *, mem=None, environ=None, pool=None, now=None)`,
  `feed_view(date, *, mem=None, wait_s=PAGE_WAIT_S, now=None)`,
  `prop_lines(work, ...)`, `prop_lines_memory(date, *, mem=None, now=None)`,
  `ensure_refresher(environ=None)`, `refresher_tick(mem, *, environ=None,
  pool=None, now=None)`, `feed_anchor()`, `new_feed_memory(now, had_ledger)`,
  `reset_feed_anchor_for_tests()`, and the key reader `_read_key(environ=None)`
  (BUCKETVALUE_ODDS_API_KEY, else the first line of the regular file that
  BUCKETVALUE_ODDS_API_KEY_FILE names, opened without blocking). `mem` None is
  the anchor's ledger. Events (free, 10 min), the per-game governor
  (bucketlib/oddsbudget.py), one task per game, no paid call within 5 minutes
  of tip, stale beats empty per game and per book, every possibly billed call
  counted. `FeedLive(state, value, asof, per_game, loading, generation, report,
  loading_games)`, state one of off, idle, ok, stale, paused, down.
### bucketlib/oddsbudget.py: the credit governor
- Pure: no I/O, no clock, no environment; every function takes `now` and plain
  values. `reset_window`, `in_reset_window`, `period_of`, `margin`,
  `default_reserve`, `busiest_month`, `p_max`, `plan`, `spending_ceiling`,
  `horizon_games`, `period_games`, `games_before`, `waiting_first`, `tip_for`,
  `before_tip`, `first_pull_time`, `due_kind`, `may_spend`,
  `first_pull_markets`, `first_kind`, `true_first_pull`, `brake_applies`,
  `backoff`, `breaker_on`, `breaker_trips`, `reset_seen`, `calls_in_window`.
- KINDS = ("first", "refresh", "pp") are due_kind's words, and REBUY = "rebuy"
  is first_kind's word for a first pull that may re-buy (B11) and due_kind's for
  a braked game's one re-buy (B21). SPEND_KINDS = KINDS + (REBUY,) are the kinds
  may_spend takes; its `kind` is required and anything else raises. REFUSALS,
  in may_spend's order: tip_guard, out_of_credits, bad_key, no_balance, paused,
  reserve, floor, ceiling, rate_cap.
- Game, Period, Plan and Slot are frozen dataclasses. Slot.has_lines (the game
  has a stored frame with lines, B12) is required and keyword only, with no
  default; Slot.braked (B21) is keyword only, default False.
### bucketlib/sources/oddsapi.py
- Pure: `read_settings`, `events_request`, `event_odds_request`, `parse_events`,
  `match_events`, `parse_event_odds`, `merge_refresh` (told `markets_requested`
  on every call, B3), `usage_from_headers`, `error_code`, `failure_kind`,
  `empty_feed`. FEED_COLUMNS = props.PARSED_COLUMNS + book, book_title, market,
  price_over, price_under, last_update, fetched_at, event_id, event_home,
  event_away, game_id, game_date, tip_utc. BOOKS are the five; BOOK_TITLE equals
  props.BOOK_TITLE; SEND_REGIONS = False.
### bucketlib/http.py: vendor_get
- One attempt, honest User-Agent, no impersonation, no redirects, its own pacing, a size cap
  during the download, a timeout of at least 0.5 s; returns status, the x-requests-*
  headers and the body; every failure is a fixed phrase or a documented code.
  VendorTransportError's kinds are timeout, connect, oversize and transport.
  `fetch` refuses the vendor's host before any request; `scrub` is the one
  function that turns vendor text into a string.
### bucketlib/props.py: the feed's lines and the board (layout C)
- `prepare_feed(feed, roster, projections)`, `resolve_event_names`,
  `compare_books(lines, dist_provider, projections, roster=None)` (compare()
  once per book and game) giving `FeedTables(tables, metas, report, failed)`,
  `feed_board(ft, paste_table, paste_meta, pasted_utc)` giving
  `FeedBoard(lines, meta, books_by_game, missing, paste_won, feed_won,
  paste_times)`, `grid_rows(shown, hidden=None)` (one row per player and stat
  with a shown line, ranked by its strongest shown line, each carrying its
  `cells`; frames or lists of row dicts) and `split_hidden(shown, hidden)`
  (the hidden lines the grid draws as the number alone, and those it never
  draws); `american_breakeven`, `side_words`. BOOK_ORDER = (draftkings,
  fanduel, betmgm, prizepicks, underdog) is the grid's column order;
  BOOK_TITLE and BOOK_CODE (DK, FD, MGM, PP, UD) are the only book names a
  page shows. FeedTables.failed marks a date whose feed holds lines that could
  not be compared: no table and no meta, so the board has no feed line to draw
  and the page explains the failure (B17: with a paste for the date, the board
  is the paste's own).
### bucketlib/props_ui.py: the feed on the page
- `date_board(token, date, projections, params, roster=None, feed=None,
  feed_info=None)`, `feed_notice_html`, `books_missing_html`,
  `books_missing_game_html`, `after_tip_note_html`, `lines_asof_text`,
  `loading_line_html`, `grid_legend_html` (GRID_LEGEND), `HIDDEN_DRAWN`,
  `merge_books` (Biggest discrepancies: one row per player, stat, line and
  side word, naming every book that posts it); ledger_html and
  render_player_lines draw the grid when handed the feed's facts. With
  feed=None every function returns the HTML it returned before the feed.
  RG_LINE = "".
### bucketlib/store.py: the page's side of the feed
- `feed_on`, `feed_start(view_date, sched) -> FeedTicket`, `feed_result(ticket)`,
  `feed_work(today, sched)`, `feed_tables(date, feed, served, params, roster)`,
  `feed_info(feed, today, environ=None)`, `feed_loading_watch(date_iso,
  generation, game_id=None)`.

## 3. The projection frame (bot file and app frame, same columns)

identity: person_id, game_id, game_date, tip_utc, team, opp, home, player_name,
two_way, season
availability: p_play, inj_status (Out/Doubtful/Questionable/Probable/Available
or None), inj_reason, team_state
means (E[stat | plays]): min (= mu_min), pts reb ast fg3m stl blk tov ftm fta
fgm fga fg3a oreb dreb pf pra pr pa ra stocks fantasy
simulator inputs: mu_min, rate_<s> for the 15 STATS
props contract: games_prior (= career_games, played games before this one),
provisional (bool, never missing), model_version (the registry's model_version
plus "+tm1", the team-minutes step, since 2026-11-03)
provenance: report_ts_et, home_line, total, computed_at (UTC), frozen (bool)

## 4. Bot (scripts/daily_update.py) and Actions

Step 6 projections, for D in (today ET, tomorrow ET), regular-season games only
(pipeline.PROJECTED_TYPES; play-in games are state games, never projected):
history(season, before=D) -> history_features -> write features; then project
with the bot's report and odds -> write projections, rows of games past tip
minus 5 minutes (pipeline.FREEZE_LEAD) carried over from the existing file
unchanged. Tomorrow is provisional. `if_changed=True` everywhere. `.github/workflows/daily.yml` shipped
with `workflow_dispatch` only (a private repo spends Barrett's Actions
minutes). From 2026-11-02 it ran on a private-repo schedule of four runs a
day, so a roster snapshot is committed every day (section 9); since
2026-11-04, when the repo went public, it runs the dense every-30-minutes
crons.

## 5. App

Game page tables from `live_projections`: MIN (mean), then mid-p medians "if he
plays"; P(play) only when listed or below 90%; status badges from the report
(Unknown is never rendered as healthy); Out rows greyed at the bottom with the
reason and no numbers; mark-out via `?out=` with the "your adjustments" banner;
pasted lines per player. Slate: the board (`props_ui.render_board`) fed by
`SimProvider`, card line-count pills, "Out: X, Y +2". About: the trust table
generated from data/stat_reliability.csv. Since 2026-11-05 (Barrett's call)
the Game table's FS column is PrizePicks' Fantasy Score formula on the medians
shown beside it, and a Fantasy Score line is judged against that same number
(props.fantasy_shown, props.fantasy_on_shown); docs/model_report.md has the
measured cost.

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
  lines mints a NEW token, so a shared link is read-only for everyone else, and a
  board restored from the visitor's browser gets a NEW token too; the record
  refuses mark-out-adjusted rows and rows without a model version.
  Feed lines are compared once per book and game, cached by the feed's
  generation and shared by every session; the board shows every book's line at
  once, one column per book, each shown line coloured by the model's lean
  against that book's own line, counts every book's lines, ranks a player and
  stat by its strongest shown line, and shows the sentinel where a book has not
  posted a game, never another book's line; and a visitor's pasted PrizePicks
  line replaces the feed's on that visitor's view only when it is newer, or a
  typed pick, a Demon or a Goblin.

## 9. Amendments after the 2026-11-02 rehearsal (the 2026-27 box lists actives only)

A live rehearsal of the bot (2026-11-02, 92 final games) found a source format
change no test could see (every fixture and the parity replay are 2025-26
payloads): the 2026-27 cdn liveData box lists only the players who dressed, so
the bot's participation had no inactive rows (played 2,079 / dnp 414 /
inactive 0), every injured player looked like a player with no 2026-27
history (Butler, Porzingis, Moody, Ingram, Beal ... at events_s 0 and p_play
0.55-0.74 on teams yet to file), and team expected minutes reached 301-303 of
240. Evidence and counts: docs/data_notes.md (cdn.nba.com). What changed in the
contract:

- **state**: `state/season/{season}/roster_{YYYY-MM-DD}.parquet`
  (`state.ROSTER_SNAPSHOT_COLS`: person_id, team, two_way, name), the roster the
  bot pulled, dated by the run's own ET date (never `--date`), written by
  step_roster with if_changed on every good pull and never from the committed
  copy. `state.roster_snapshot_path` is the one path; `roster_snapshot_dates`,
  `roster_for_game(season, game_date, first_day) -> RosterAt(roster, date,
  after)` (the latest snapshot on or before the game's ET date, else the
  earliest after it, `after=True`; snapshots dated before the season's first
  state game day are camp rosters and never used) and `known_names(season)`
  (every spelling the state holds) read them. They are no partition kind:
  `season_partitions` and `ingested_ids` ignore them.
- **participation partitions** may hold REBUILT inactive rows: source
  `box.ROSTER_MINUS_BOX` ("roster_minus_box"), status inactive, no start, 0
  seconds, jersey '', reason_code the cdn code of the player's report category
  (`box.REPORT_REASON_CODES`, every `injury.REASON_CATS`) or `box.UNLISTED`
  ("INACTIVE_UNLISTED", absence_kind unknown), reason_text the report's
  "Status: reason". `box.SOURCES` includes it; check_participation and
  check_table require such a row to be inactive and count it under its game's
  cdn source. No feature reads source or reasons: the rebuilt rows are the
  same events training has.
- **The rule** (`box.lists_inactives`, `box.rebuild_inactives`, one
  definition): a box listing no INACTIVE player on EITHER side is the new
  format (one inactive row anywhere is the old format and is parsed exactly as
  before); every player on a team's roster at game time not in the box (either
  side) gets a rebuilt row, his reason from the last real report before the t_15
  horizon (`daily_update.REBUILD_HORIZON`, through horizons.report_at /
  report_tables / game_states and livectx.resolve_names, the training
  snapshots' own functions). Only with a snapshot taken AFTER the game (the
  backlog) does the report add players: one the team's report listed Out who is
  on neither its roster nor the box left since and gets a row (resolved
  team-scoped, then by exact or alias name among every known person), and the
  report's team wins over the later roster's. A box outranks a snapshot: a
  roster player whom another team's box lists between the snapshot's date and
  the game's (both included; committed partitions plus every box of the run,
  `state.box_listings`, rebuilt rows excluded) is not rebuilt unless his own
  team's report lists him. Step 3 therefore parses every box of the run before
  it rebuilds any (`Rebuild.boxes`). In a play-in or playoff game
  (`box.TWO_WAY_INELIGIBLE`) the snapshot's two-way players get no row: they
  cannot play, and the training listing mostly left them out (2025-26's play-in:
  23 rows that would otherwise have been added, 5 missed in the one game that
  listed its two-way players; docs/known_issues.md). The snapshot's `two_way`
  flag is required there: a missing or null one makes the rebuild raise (the
  game is RED, not ingested, retried next run), never "not two-way". The
  info per team gains `ineligible`, and the log line says "N two-way left
  out".
- **Bot, step 3**: a new-format box with no usable snapshot is NOT ingested
  (RED, retried next run; a dry run warns): written without its inactive rows it
  would stay wrong for good, since a final game is never re-fetched. No report,
  a PDF that does not parse or a blocked report host leaves the reasons
  INACTIVE_UNLISTED and is never a failure (blocked: one WARN, no more fetches
  that run). Every report slot is asked once per run (`daily_update.OncePerSlot`)
  of one in-memory store shared with step 4 (`daily_update.report_store`): the
  store alone re-probes an absent slot under 48 hours old every time it is
  asked, so without the wrapper a morning run fetched the shared slots of last
  night's games once per game (240 fetches for 130 slots on two games of one
  night, tests/test_daily_update.py). One log line per rebuilt game names the
  snapshot (and whether it is the backlog case) and the report.
- **Roster-size guard**: a team over MAX_TEAM_ROWS (18) is RED today and a
  warning tomorrow only until the day after the regular season's first game
  day. In season it is checked against ESPN's roster (`espn.team_roster`,
  reachable from Actions): the same players is an info line, a difference a
  WARN naming the players only one source lists, ESPN unreadable a WARN; never
  RED and nothing dropped (LAC listed 19 on 2026-11-02, and so did ESPN).
- **bucketlib.http** loads CoreFoundation (darwin only) before curl_cffi: on
  macOS 27 a fresh interpreter could not import the client otherwise.
- **Parity proof extended** (tests/test_serve_parity.py): every 2025-26 box
  stripped of its INACTIVE players and rebuilt from a roster equal to its
  listing gives every player his original status (1,235 games), and the serve
  path on bot partitions rebuilt that way equals the training table bit for bit
  (2025-10-26, 2025-11-05; the same partitions without the rebuilt rows do
  not). The backlog approximation (a stale roster) is measured, not hidden:
  docs/known_issues.md.
- **Verified live** (a second rehearsal, 2026-11-02 07:19 UTC, the fixed bot
  on a fresh copy of the repo): green in 82 s; 724 inactive rows rebuilt over
  the 92 games, all from the 2026-11-02 snapshot (the backlog), every game with
  a report; team expected minutes (sum of p_play x minutes) median 242.1, max
  267.7, against 261.4 and 303.2 before (teams yet to file: mean 242.0, was
  284.6); Butler, Porzingis, Moody, Ingram, Beal, Strus, Jordan Miller and
  Konan Niederhauser at events_s 6-7 and p_play 0.09-0.15 (were 0 and
  0.42-0.74); LAC's 19 an info line, ESPN listing the same 19. That
  rehearsal wrote to a scratch copy: no snapshot is committed yet, so the
  backlog of the first committed run grows by a day each day until then
  (13 days stale if it is on 2026-11-02; docs/known_issues.md).
- **Not done, needs a decision**: (1) commit roster snapshots from now on,
  before the deploy (a dispatch run of the workflow), so the backlog stops
  growing. (2) The 2026-27 participation table the local backfill builds from
  the same cdn boxes (scripts/build_participation.py) has no inactive rows
  either, so the 2027-28 seed and any retrain on 2026-27 need an inactive
  source: the bot's rebuilt partitions, or V3 BoxScoreSummaryV3 `inactives` if
  it still lists them.

## 10. The licensed odds feed (2026-11-11)

Player lines for DraftKings, FanDuel, BetMGM, PrizePicks and Underdog, licensed
from The Odds API, display only (no model reads them). Built to the feed's
design (architecture_v2.md, whose section numbers "design 3.6" and the like
cite, and its budget replay RP; both are kept outside the repo) as amended:
layout C (Barrett, 2026-11-11: every book at once, one column per book in the
order DK, FD, MGM, PP, UD, each line with the model's lean arrow; no price is
shown for any book) and the build amendments B1 to B34, named below where they
decide a rule. The operator's side is docs/runbook.md section 8. With no key
configured none of it runs and every page is what it was.

### The anchor (live.feed_anchor)
- `live.ANCHOR_NAME = "_bucketvalue_feed_anchor"`: a `types.ModuleType` in
  `sys.modules`, outside the bucketlib package, made once per process (the
  first `setdefault` wins). Its attributes: `lock`, `state` (the ledger),
  `births`, `pool` (a ThreadPoolExecutor of ODDS_WORKERS + 1 = 5 threads, the
  walk taking one), `refresher` (the thread), `refresher_stop` (its Event) and
  `refresher_version`.
- freshen pops only `bucketlib` and `bucketlib.*` names, a Streamlit cache
  clear touches only Streamlit's caches and the sender's session, and the file
  watcher finds no file for a module with no `__file__` and a None `__spec__`:
  all three miss it. A restart loses it.
- Plain data and stdlib objects only (B4): no bucketlib instance or function is
  ever stored, so no code of a purged module runs from it, except the
  refresher loop, which looks up the current `bucketlib.live` on every tick.
- `feed_anchor()` replaces a ledger that is missing, of another
  FEED_MEMORY_VERSION (2) or of a bad shape, and logs `oddsapi: feed memory
  created (process start)` the first time in a process, `(replaced: version
  change)` after.

### The ledger (live.new_feed_memory)
```
{"v": FEED_MEMORY_VERSION,                  # 2
 "born_at": <UTC>, "had_ledger": <bool>,    # the brake and first_kind read both
 "lock": threading.Lock(),                  # guards every field below
 "event_locks": {},                         # event_id -> Lock: one call per game, never waited on
 "queued": set(), "futures": {},            # event_id -> Future, while queued or running
 "walk": None,                              # the running walk's Future: one walk at a time
 "events": None,                            # (fetched_at, today's matched events, report)
 "events_fail": None,                       # (until, kind, fails): the events call backing off
 "games": {},                               # event_id -> entry (below)
 "budget": {"remaining": None, "used": None, "last": None, "seen_at": None,
            "auth": "ok", "auth_until": None, "pending": 0, "mode": None},
 "calls": [],                               # UTC times of paid calls in the last 10 minutes
 "streak": 0, "breaker_day": None,          # billed answers in a row; the ET day stopped
 "generation": 0,                           # bumped on every change to a stored frame
 "work": None,                              # the latest work order
 "warned": {},                              # log-line code -> last UTC (once per period)
 "braked": False,                           # the memory-loss brake was decided (once)
 "brake_day": None,                         # the ET day it applies to, when it does
 "brake_used": None}                        # the `used` it was decided on (first_kind)
```
- `budget["auth"]` is "ok", "out_of_credits" or "bad_key"; `budget["mode"]` is
  the last plan's mode, whose change is logged. `pending` is the worst case of
  the calls in flight. Within one period `remaining` only falls and `used`
  only rises, unless the vendor's counters move by more than the calls that
  can be in flight (a reset, or new credits).
- `brake_used` is the account's `used` when the brake was decided: no paid
  call goes ahead before that, so it is what the period spent before this
  ledger, and first_kind's usage test reads it.

One entry per game (`live._new_entry`), keyed by the vendor's event id:
```
{"game_id", "game_date" (ISO), "tip_utc", "vendor_tip",
 "frame",                                   # the stored lines (FEED_COLUMNS), or None
 "pulls", "first_at", "last_at",            # possibly billed calls (3.5), the brake's included
 "empty_next",                              # an empty answer: not again before this
 "pp_missing", "pp_next",                   # the last pull lacked PrizePicks; the next free check
 "fail_until", "fails",                     # the backoff after a failure; failures in a row
 "books_at": {book: fetched_at},            # each book's own latest fetch
 "pulled": [...], "n_pulled",               # lines a book took down: 8 named, all counted
 "held",                                    # reserve, floor or ceiling refused the latest call
 "braked",                                  # the brake counted its first pull (B21)
 "free_failed"}                             # a free call failed with no answer (below)
```
- `held` is the budget guard that refused the game's latest paid call, or
  None once a call goes ahead or another guard decides it. The view reads it
  (a held game of today makes the day "paused"); the governor never does.
- `braked` marks a game the memory-loss brake counted as pulled at the
  ledger's birth: that pull was assumed, never bought by this ledger.
- `free_failed` (build 6's review): a call expected to be free failed with no
  answer, may have been billed and was not counted as a pull, and no clean
  answer has come for the game since (below).
- Validation (`_valid_ledger`, `_slot_of`): the version, the exact set of
  top-level and budget keys and each one's type; each entry when read, a
  malformed one counting as absent (logged). An entry from before `braked` or
  `free_failed` existed reads them as False, so a ledger kept across a code
  push stays valid and buys nothing more for it. Any change to the top-level
  or budget keys replaces the ledger, as a FEED_MEMORY_VERSION bump does.
- Pruning: each walk drops games dated before yesterday, their locks, old
  rate-cap times and old log marks.

### The work order (store.feed_work)
Written on every feed-on view of today (store.feed_start) and read by the walk
and the refresher. Plain data, cached 10 minutes on the schedule columns it
reads and the committed files' (path, mtime_ns, size):
```
{"date": "<today ET>", "written_at": "<UTC ISO>",
 "games": [{"game_id", "game_date", "tip_utc", "tbd", "away", "home"}],    # today's buyable games
 "horizon": [{"game_id", "game_date", "tip_utc", "tbd", "away", "home"}],  # B2 below
 "serve_files": {"<today>": [projections, features], "<tomorrow>": [...]},
 "busiest": <the season's busiest month in projected games>}
```
- Buyable: projected (pipeline.state_games) and in the date's committed
  projections file, or its features file when there is none (m26), so a bot
  outage day costs nothing.
- The horizon (B2) is every projected game from the earliest day a credit
  period current today could have opened (the last day of the month before)
  to the latest close of the next reset window, tipped games too, each with
  both teams (N4). It gives the governor the period's own games: the
  spending ceiling, the reset window's `used` test and the brake's usage test
  count them, and G counts their untipped part.

### The walk (live.feed_kick, _walk)
`feed_kick` never blocks and never raises. It reads the key and the settings
on the call (no key, or the switch off: nothing), stores the work order, and
submits a walk unless one is running. The walk:
1. reads the key and settings again, logs each settings problem (names, never
   values), prunes, and takes the order's buyable games of today (none: no
   call at all);
2. the free events call when its copy is older than 10 minutes and nothing
   holds it back (a failure backs off 90 s doubling to 60 min; a bad key waits
   30 min); events match schedule games on the ET date and the unordered
   team pair, and a game's tip is the earlier of the vendor's and the
   schedule's (a TBD schedule tip ignored);
3. `period_of` and `plan`, the brake (once per ledger), the mode-change log
   line and the reset watch;
4. each matched game of today, in tip order, with a due kind and not already
   queued, goes to the pool (`_pull_game`); the queue never holds more tasks
   than today's games. The walk's INFO line counts the billed calls and their
   cost once its last pull is done.

### One paid call (live._ticket, the call, live._settle_pull)
- Guards, in order (design 3.6): the game's own non-blocking lock; under the
  ledger lock a fresh `due_kind`; for a "first", `first_kind` (B11) and
  `true_first_pull` (B23, never while the entry's `free_failed` is set); then
  `may_spend`: tip_guard (now >= tip - 5 min), the auth state, an unknown
  balance, a paused plan, the reserve (remaining - pending - cost >= S), the
  morning-pull floor (remaining - pending - cost - S >= c_core x the
  horizon's other games still waiting for a first pull; never checked for a
  "first", B1), the spending ceiling (used + pending + cost <= ceiling(today);
  never checked for a true first pull, B23), the rate cap
  (BUCKETVALUE_ODDS_MAX_CALLS_10MIN in any 10 minutes); the key and the kill
  switch are read again before the call.
- Markets: a re-buy asks the core markets only (B20); a first pull asks the
  plan's markets only when the floor holds at their cost, else the core
  markets (B1, first_pull_markets); a refresh or a top-up asks the plan's.
  Books: the five, or PrizePicks alone for a "pp" top-up. With the shipped
  default (points only) every pull asks the same one market.
- A refusal by the reserve, the floor or the ceiling sets the entry's `held`
  and logs `oddsapi: budget held a pull today (<code>)` once per guard per ET
  day (B19).
- `pending += cost` before the call and back in a `finally`; a raise anywhere
  after the ticket settles the call once, as a failed one.

What a call's outcome does (design 3.5 as built: `_outcome`, `_answered`,
`_failed`):

| Outcome | Billed | A pull | The streak | Then |
|---|---|---|---|---|
| 200 with no bookmaker entry (nothing posted) | no | no | back to 0 | not again for 10 minutes (`empty_next`, or `pp_next` for a top-up); `free_failed` cleared |
| 200 with lines kept | yes | yes | back to 0 | `fails` 0, `free_failed` cleared, `pp_missing` when PrizePicks is bought and absent; the generation moves when the frame changed |
| 200 with entries but no line kept, not JSON, the wrong shape, or an unknown team | yes | yes | +1 | backoff |
| a body over VENDOR_MAX_BYTES, stopped mid-download ("oversize") | assumed | yes | +1 (B31) | backoff |
| a timeout, or a failure after the connection (kind "transport", a cut-off download) | possibly | yes, unless the call was expected free | never (B22, B31) | backoff; a free one sets `free_failed` |
| a connection failure | only when bytes had arrived | only then | never | backoff |
| 401, 404, 422, 429, 3xx, 5xx | only when x-requests-last > 0 | only then | +1 when billed | backoff; a 401 sets the auth state; a 404 ends the game's pulls for the ET day |

- Backoff after the n-th failure in a row on a game: 90 s, 3, 6, 12, 24 and
  48 minutes, then 60 minutes each time.
- The day breaker (B22 as B31 reads it): BREAKER_STREAK = 3 answers in a row,
  any games, that came back (a complete response, or an oversize one) and
  were billed and of no use stop paid calls until the next ET day, with one
  WARNING; the free events call goes on. A clean answer, an empty one
  included, resets the streak. A transport failure (a timeout, no
  connection, a download the network cut off) never grows it: it still
  takes its worst case off the balance, counts per the table and backs the
  game off.
- L2-1's free timeout, as it now stands: a call is expected to be free when
  the game's last answer had nothing (a PrizePicks-only top-up, or a pull
  with `empty_next` set) and the entry's `free_failed` is not set. A
  transport failure of such a call takes its worst case off the balance but
  is not one of the game's P pulls; it sets `free_failed`. Once per clean
  answer: while `free_failed` is set the game's next call is not taken as
  free (its failure counts as a pull) and its next first pull is no true
  first pull (the ceiling checks it). Only an empty answer or lines that land
  clear it; a failure, answered or not, never does.

### The kinds (oddsbudget.due_kind, first_kind)
Nothing is due while the plan does not spend, the day breaker is on, the tip
guard has passed, the game backs off (`fail_until`) or an empty answer is
under 10 minutes old. Then, with k the game's pulls and P the plan's:
- "rebuy" (B21): the brake counted the game's first pull (`braked`), that is
  still its only pull (k = 1), the ledger holds no lines for it, P >= 2, and
  now < t_last = tip - BUCKETVALUE_ODDS_LAST_PULL_LEAD_MIN: ONE re-buy, at
  once, guarded as B11's (the floor and the ceiling), core markets only. A
  re-buy that counts as a pull is the game's second, so one ledger never
  re-buys a game twice.
- nothing once k >= P.
- "first" (k = 0): at or after BUCKETVALUE_ODDS_FIRST_PULL_ET on the game's ET
  date. `first_kind` keeps it "first" (B1: the floor skipped) when the ledger
  was born at or before that time (`true_first_pull`: it can never re-buy what
  an earlier ledger bought, so it skips the ceiling too, B23, unless the
  entry's `free_failed` is set), or when the period was not spending before
  the ledger (no earlier ledger in this process, and 2 x brake_used < c_core x
  the period's earlier games); otherwise "rebuy" (B11): the pull may re-buy
  what the process before bought, and the floor and the ceiling guard it.
- "refresh", retried (B12): the game's pulls so far brought no stored lines
  after a failure (has_lines False, fails > 0): due as soon as its backoff
  allows, while it has pulls left. On the free plan (P = 2) that retry is the
  game's second pull, so the game keeps its morning lines but gets no pre-tip
  refresh.
- "refresh": pull k is due at t_first + (t_last - t_first) x k / (P - 1), no
  sooner than last_at + MIN_REFRESH, and only while that time is <= t_last. A
  late first pull spaces the rest from itself.
- "pp": a game whose pull lacked PrizePicks, every 10 minutes (`pp_next`),
  unless BUCKETVALUE_ODDS_PP_TOPUP is off; a due refresh comes first (B5). The
  check is free while PrizePicks has nothing (an empty answer is free by the
  vendor's docs, UNVERIFIED: docs/data_notes.md), and the first answer with
  lines counts as a pull, so on the free plan it takes the place of the
  evening refresh.

### The plan (oddsbudget.plan)
S = the reserve, A = used + remaining, B = remaining - S - pending, G = the
horizon's untipped games, made = their pulls in this ledger, c = the number
of markets:
- a bad key or out of credits: paused; the day breaker on: paused; the
  balance unknown: unknown (no paid call);
- P_full = floor((B / c_full + made) / G) >= 1: full, P = min(P_full,
  P_max(c_full)); else the same with the core markets: core; else B >= c_core:
  short (core markets, P = 1); else paused at the reserve.
- P_max(c) = min(48, 2 x max(1, floor((A - S) / (c x busiest)))): 4 on the free
  plan with points.
- ceiling(today) = min(A - S, ceil(1.5 x (A - S) x n_through(today) /
  n_period)), over the period's games. It reads the vendor's `used` and bounds
  every refresh, top-up and re-buy, and every first pull of a ledger born
  after the game's first-pull time, so no restart or crash loop pushes the
  period's spending more than half again above its pace. A true first pull
  skips it (B23), and one ledger never sees what another bought: a second
  process whose ledger was born at or before 08:50 (a local run with the key
  file started before 08:50, or left open overnight) buys that morning's
  games again, bounded only by the reserve.
- The credit period: credits reset on the 1st (the vendor's FAQ, for every
  plan); inside the reset window, from 05:00 ET on the last day of a month to
  08:00 ET on the 1st, `used` decides which period it is. There is no
  reset-day setting.

### The brake (live._brake, oddsbudget.brake_applies)
Decided once per ledger, at the first walk that knows `used` (then `braked`
is True, `brake_used` keeps that `used`, and `brake_day` is today when it
applies). It applies to a ledger born more than 60 minutes after the day's
first-pull time (after 09:50 ET by default) when this process had a ledger
before (a version change) or 2 x used >= c_core x the period's earlier games
(a feed was running). Each of that day's untipped games with no pull then
counts as pulled once, at the ledger's birth, and is marked `braked`; one
WARNING gives the count. The usage test never passes on a period's first day
(no earlier game) or with an unknown `used`.

### The view (live.feed_view)
- It waits up to `wait_s` (a page passes 3 s) for the walk and the date's
  calls, then reads memory: the current frames, and after tip the pre-tip
  copies, which nothing buys again.
- State: "paused" when today's paid calls are stopped (the auth state, the day
  breaker, a paused plan, or a game of today `held` by the reserve, the floor
  or the ceiling), with or without lines; else "stale" with lines, or "down"
  without, when the events call or a call of one of the date's games failed;
  else "ok" with lines, "idle" without. Another day's memory is never paused,
  stale or down.
- `per_game`: "asof" (the game's latest pull), "books" ({book: fetched_at}),
  "pretip", "pp_missing", and "reason" when it has no lines: "not_yet",
  "none_posted" (its last answer had no bookmaker entry), "paused",
  "not_published", or "lost_in_restart" (the ledger was born after the game's
  tip guard and holds no frame for it).

### The refresher (live.ensure_refresher, _refresher_loop, refresher_tick)
- Started by store.feed_start on a view of today with the feed and
  BUCKETVALUE_ODDS_REFRESHER on: a daemon thread, at most one per process. A
  dead one is restarted on the next such view, and a REFRESHER_VERSION change
  stops the old loop and starts a new one.
- Every REFRESHER_TICK_S (60 s) the loop calls the CURRENT
  `bucketlib.live.refresher_tick` on the anchor's ledger, so after a code push
  it runs the new code. The tick answers "off", "no_work", "idle" or "kicked".
- The tick: no work order, or one older than 36 hours: "no_work". Today's
  games come from the order when it is dated today and holds some; otherwise
  (an order from the evening before, or today's written before the bot
  published today's files, B15) from the horizon rows dated today whose game
  ids are in the first of today's committed serve files that exists, rows
  without both teams left out (logged once a day). It kicks the walk when a
  game of today is past its first-pull time, before its tip guard, not
  queued, and the balance is not known yet or due_kind says it is due. An idle
  tick makes no network call. An order's serve_files name only its own day and
  the next, so after a full day with no view of today every tick is "no_work"
  until a view writes a new order.
- It never calls `st.*`, builds the feed tables or reads Streamlit's caches.
  Nothing starts it before the first view of today after a process starts.

### The page side (store, props_ui)
- `feed_on()` is a key and the switch, read on the call: no network, no anchor.
  `feed_start` returns "off" before any other feed code runs (M9); for today it
  builds the work order, kicks the walk and starts the refresher inside one try
  ("kicked", even when that raised); another date is "memory". `feed_result`:
  "off" is FeedLive.off(), "kicked" waits up to 3 s (feed_view), "memory" reads
  memory without waiting; any exception is FeedLive.down().
- `feed_tables` is `st.cache_resource` with 4 entries, keyed by the date, the
  ledger's generation and strict digests of the feed's rows, the frame, the
  roster and the simulator parameters; a digest that cannot be made means not
  cached. The key also names the `props.FeedTables` class the call sees
  (B34): Streamlit's resource cache outlives freshen's purge, so each import
  of the code after a push rebuilds a date's tables once, never a credit,
  rather than handing tables of the old class to code that reads them as
  none. One push can mean several imports: up to one by freshen's purge, and
  one for each session open at the push, whose Streamlit file watcher drops
  the whole package at that session's next run. Tables that cannot be built
  while the feed holds lines are `FeedTables(failed=True)`; with a paste for
  the date the board is then the paste's own, under one note (B17).
- props_ui draws every book at once: grid_rows makes one row per player and
  stat with a shown line, ranked by its strongest shown line; each cell is its
  book's line with the model's lean at it (an arrow and a text equivalent on a
  shown line, the number alone on a hidden one, the whole-node sentinel for
  none); a feed line's title adds its own time when it is over a minute older
  than its game's latest pull, and a pasted PrizePicks line's title its B28
  word; the board's stamp is the oldest of the books' own fetch times; the
  notices of design 6 and the notes on the books that have not posted a game
  are built from the stamps and the `today` the page passes, never from a
  clock. No `?book=` is read or carried.
- `feed_loading_watch` (an `st.fragment` every 3 s) is drawn only while games
  load; from its own timed runs it reruns the app once the generation moves or
  nothing is loading, and it asks for that only in the first 20 runs of a
  loading episode (then it only shows the line).

### Restarts, replayed under B21 (tests/test_oddsbudget.py)
On the 2026-27 schedule (RP; the tests replay it one year earlier, so no date
in them is in the future), 2027-01-07, 12 games, free plan, points, a period
that was spending:

| Restarts on Jan 7 | Credits that day | How |
|---|---|---|
| none | 24 | an 08:50 pull and a pre-tip refresh per game |
| one at 15:00 | 24 | the 15:00 ledger re-buys the twelve at its birth, in place of their evening refreshes |
| three, at 13:00, 14:00 and 15:00 | 27 (24 before B21) | the 13:00 ledger re-buys the twelve; that spends the evening refreshes' credits, so the 14:00 and 15:00 ledgers plan P = 1 and buy nothing until 20:30, when the tips that passed lift P to 2 and the 15:00 ledger re-buys its three late games; 9 games end with 2 pulls, 3 with 3 |
| five, 09:00 to 09:40 (inside the brake's hour) | 72 | every pull of those ledgers is a re-buy that the floor checks and passes (B11): five re-buys of the twelve, less the twelve evening refreshes the plan then dropped, so 48 more than a normal day, all in the morning; the ceiling and the floor held, and every later game kept its morning pull |
| every 10 minutes, 10:00 to 18:00 (49) | 27 | 12 first pulls and 15 re-buys: the 10:00 ledger re-buys the twelve, later ones plan P = 1 until 20:30; no game bought more than three times; after every call the floor's promise and the ceiling held |

- Design 3.7's case D holds again under B11: in a morning crash loop the
  ceiling and the floor held.
- RP case C (470 used at 10:00 ET on Dec 31, the reset at 19:00): Dec 31 costs
  5 and Jan 1 12. The 10:00 ledger brakes Dec 31's games; with 5 credits above
  the reserve for 9 games it plans P = 1 until the 3 PM tips, then P = 2 for
  the five games still ahead, which are re-bought at 15:00, from the old
  balance, before the reset.
- RP case H (a key added at 14:00 on Jan 12 with 15 used): the month was not
  spending, so no brake: 24 calls for the day's 8 games, P = 3.

### Known limits, accepted as built
- L2-5: plan() counts the pulls already made at the plan's per-pull cost. It
  matters only with more than one market configured: a day's last first pull
  can then read the plan as full and ask every market, spending credits the
  core plan kept for that evening's refreshes. Fix before adding markets.
- L2-4 (not changed): a FEED_MEMORY_VERSION bump, or any change to the
  ledger's keys, replaces the ledger. Pushed between 08:50 and 09:50 ET it
  re-buys today's games (floor-checked, core markets); after 09:50 the brake
  applies (had_ledger) and B21 re-buys each braked game once while P >= 2.
  So such a change is pushed before 08:50 ET, never between 08:50 and 09:50.
- B32: a vendor that alternates "nothing posted" answers with timeouts it
  bills re-arms L2-1's free timeout at each empty answer. The backoff bounds
  it to about 13 credits per game per day, each taken off the local balance
  as its worst case, so the governor tightens. No cap: a billed timeout means
  the game had lines, so an empty answer right after one is unlikely.
- B33, restarts past 09:50 on the free plan: with P = 2, B21's re-buy is the
  game's last pull, so a book that posts after the restart, PrizePicks
  included, is not fetched that day; after a second restart the same day the
  next ledger can plan P = 1, and its braked games then get neither a re-buy
  nor an evening refresh (in the three-restart replay, 9 of 12 games held no
  lines from 14:00). Both leave the 9 AM view untouched. Kept as built
  (Barrett, 2026-11-11), to revisit after the test week or a move to a paid
  plan.
