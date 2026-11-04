# Known issues (open)

Findings from the adversarial verifiers of the four source modules
(workflow wf_99aca6ad-0e3, 2026-10-08) whose fixers were cut off by a session
limit after applying the high-severity fixes. Everything here was checked
against the code on 2026-10-09; items fixed since are removed, not struck.

## Fixed before the cutoff (for the record)
- box: 'finalbox' status; NWT_LEAGUE_SUSPENSION comments; V3 unknown ids are
  Absent; live-season game list from the cdn schedule (stats.nba.com hangs from
  Actions). Added 2026-10-09: a past-season CDN box frozen mid-game (17 games
  in 2019-20) falls back to the V3 final.
- statsnba: all-zero LGL-only rows excluded, not appearances (Markieff Morris
  0042100105 removed on rebuild, the only change across 12 seasons); empty
  pulls allowed only before a season's first scheduled game.
- espn: a cached scoreboard month is settled by its FETCH stamp, never the
  calendar; an empty month is never settled.
- injury: everything (six layouts, wrapped team names, UNK@UNK, report-level
  accept, title-time walk, truncated PDFs); 7,822/7,822 cached PDFs parse.

## Data conflicts resolved (2026-10-09)
- The cdn box score disagrees with the stats logs on WHO played in 4 of ~10,200
  games (0042100401 Kornet vs Nesmith 48s; 0022200012; 0022400831; 0022400920)
  and breaks the team player-seconds invariant in 2 (0022100124 BKN -301s,
  0022400061 BOS +260s). All six were re-parsed from the V3 box, which agrees
  with the stats logs the stat lines come from. In season the bot builds both
  stat lines and participation from the SAME cdn payload, so they cannot
  disagree with each other; a broken invariant there turns the run red.

## Open
- box (script layer): past-season empty game lists are cached; Absent/NotFinal games
  leave exit status 0 (the table's `complete` flag is correct); `--games`
  duplicates are not de-duplicated; checkpoint restamps on no-op runs.
  `dnp` mixes coach's decisions with dressed-but-injured (DND_INJURY) rows: the
  availability model must read reason_code, not status alone.
- statsnba: `neutral` misses 2023-24 Cup games in Las Vegas and pre-2024-25
  international games (derive from arena_city if neutral ever becomes a
  feature; today only `home` is used); HOW_ACQUIRED in past rosters carries
  later information (never use it as a feature); tidy tables stamped with
  write time; a cancelled game (BOS-IND 2012-13, 0021201214) fails its season
  (only a priors season, 2012-13, is affected).
- espn: 404s cached as absent forever; non-Blocked errors do not cancel queued
  requests; `et_date` on the 4 ±1-day matches is ESPN's corrupted date (the
  join is on game_id, so features are unaffected); providers 1 and 3 missing
  from the 2016-17 allow-list (lines still computed from the other books).
- names: the prefix fallback can bind 'Jay Williams' to Jaylin Williams when
  the index is built without teams; build per-team indexes (livectx does).

## Open: serving build 1 follow-ups (2026-10-09)
Found by the build 1 verifiers and the cross-module fix pass that followed.
- Simulator FTA is always EVEN. dist.simulate draws free-throw TRIPS and sets
  FTA = 2 x trips (no and-ones, no technicals, no three-shot fouls), so an odd
  FTA has zero probability and FTM inherits the gaps. Until and-ones are
  modelled AND scripts/prob_gate.py covers them, FTA, FTM, FGM, FGA, FG3A,
  FG2M, FG2A, OREB, DREB and PF are served from the LABELLED baseline
  (props.MeanDispersionProvider, source "baseline", tier unmeasured;
  simprov.BASELINE). The gated PTS (and every combo with PTS) is drawn from
  the same even FTA; the probabilistic gate measured it that way and it passed.
- FG3A share of 1. models.predict now caps rate_fg3a at rate_fga (the raw
  models disagreed on 21 rows of the training table, 14 of them played). At
  the cap the FG3A share is exactly 1, so those rows still draw NO two-point
  attempts, and the gated PTS distribution for such a player is built from
  threes and free throws only, then re-centred. The FG3A mean is no longer
  above the FGA mean, but the composed FG2A mean (FGA - FG3A) is now exactly
  0 where it used to be negative.
  MUST FIX BEFORE THE CAP SHIPS (simprov, not yet done on 2026-10-09):
  simprov._degenerate refuses FG2M, FG2A and FGM only inside price(), which
  runs on the GATED stats. All three are BASELINE stats: SimProvider.pmf
  sends them to props.MeanDispersionProvider and never through _degenerate.
  A negative FG2A mean was refused there (recentre cannot reach it), but a
  zero mean prices as the point mass at zero. On all 21 capped training rows
  FG2A went from refused to a point mass at 0: a certain Less (q_less = 1) on
  every 2-PT Attempted line and a Game-table median of 0. In 5 of the 14
  played rows the player did attempt a two. The fix is to refuse FG2A on the
  baseline path when the row's FG3A share clips (FG2M and FGM keep positive
  means and ordinary baseline pmfs, which tests/test_simprov.py already
  pins), with a test that a capped row's 2-PT Attempted line is unprojected.
  A share below 1 (from the player's own history, or a cap under 1) is a
  model decision for dist/simprov, not made here.
- docs/model_report_full_v1.csv was produced with DUPLICATED feature columns:
  scripts/validate_full.py appended livectx.CTX_COLS to a feature list that
  already held them, so its models saw 215 columns (193 unique; the 22 context
  columns twice). The script is fixed; the report was NOT rerun (hours). The
  ship gate (scripts/train_eval.py, docs/gate_*.csv) and production
  (models.train, models/registry_v1.json) used the correct 193. The
  published AVAILABILITY numbers come from that duplicated-column run:
  docs/model_report.md lines 57-60 (the log losses by horizon and the
  0.98-1.02 sum of P(play), section "v1, full features (2026-10-09): THE
  GATE") are read from docs/model_report_full_v1.csv. scripts/validate_full.py
  is the only writer of availability log loss; no gate artifact and neither
  scripts/train_eval.py nor scripts/gate.py measures it. Treat those numbers
  as unconfirmed until scripts/validate_full.py is rerun.
- iCloud eviction. The repo lives on the iCloud-synced Desktop, on a disk at
  97% (15 GiB free of 460 on 2026-10-09), and macOS evicted files mid-session:
  cache/train_table_full_v1.parquet and packages under .venv. An evicted file
  still exists() but reads as corrupt ("magic bytes", ArrowInvalid) or fails
  to import. Touch-read it (cat FILE > /dev/null) before concluding anything.
  tests/test_models.py SKIPS (does not fail) its table tests on an unreadable
  table, so a green run can hide an eviction: read the skip count. The fix is
  to free disk or move the repo out of the synced Desktop.
- The seed serves ONE season. state/seed was built for 2026-27; for 2027-28
  state.history raises SeedMismatch ("rebuild the seed") because its priors
  need the 2026-27 logs. Rebuilding (scripts/build_serving_seed.py --serves
  2027-28) runs LOCALLY on complete 2026-27 participation and log tables from
  the stats.nba.com backfill, which the bot cannot reach. Schedule it between
  the 2026-27 Finals and the 2027-28 opener, or the bot stops projecting.
- The props record: props.is_in_training_window now RAISES on a registry
  without a readable trained_through (it used to count every season as out of
  sample). season_record(graded) with NO registry still returns the
  unfiltered record by design; the local Accuracy page must always pass
  models.read_registry().
- scripts/build_train_table.py now takes its inputs from state.seed_inputs
  (one assembly for training and the seed; proven frame-identical to the old
  code on 2026-10-09). The per-season "phantoms / zero-second" diagnostic it
  printed is gone (pd.concat drops the per-season attrs inside seed_inputs);
  it now prints roster rows per season.

## Open after the pre-opening audit (2026-10-09)

- **Minutes draw (math-2).** Minutes are mu plus an additive residual from
  8-minute buckets, clipped at 0: the distribution jumps at bucket edges and
  piles mass at 0 for low-minute players (zero-mass calibration -9.5 points just
  below 8 projected minutes, +5.0 just above). A ratio draw blended between
  neighbouring buckets measured better on 2023-24/2024-25 (ECE 1.595 to 1.515,
  pick edge 3.70 to 3.78, RPS 2.0126 to 2.0115); proposal in the session
  scratchpad (auditfix_model/math2_proposal). Shipping it needs
  sim_params_v2, a fit_dist change, a fresh probabilistic gate and new tiers.
- **Overtime and the underdog tail (math-5).** Spec 3.5 asks for an explicit OT
  component and a spread-dependent low tail; the empirical residual carries OT
  implicitly, and spread-split minutes tables did NOT fix the starters-on-big-
  underdogs tail (PTS bottom-decile share 0.154 to 0.151), so that tail is on the
  rate side. Spec wording or a model change: Barrett's call.
- **FTM near the cap.** Capping FTM at FTA prices elite shooters as 100%; on the
  test seasons rows the model implied at 92-97% shot 85-87%, so FTM runs 4.5-10%
  high there (baseline-served stat, tier unmeasured).
- ~~**Team minutes in 2024-25** sum 6.0 short of actual on the full roster
  (docs/model_report.md, audit measurements); other seasons are +2.3 / +2.5.~~
  ADDRESSED 2026-11-03 (models.cohere, docs/model_report.md "Team minutes
  coherence"): the raw models still miss a team-game's sum (sd 12.2 minutes
  out of sample), and the served rows are moved part of the way back (sd 4.7).
  What remains: the availability side of a gap. A late scratch (Kelly Oubre
  Jr., p_play 0.99 at 6 PM on 2026-11-02, did not play) leaves his teammates'
  minutes short, and nothing before tip says so. The two scales were fitted on
  predictions of models trained through 2022-23. After a retrain, regenerate
  those with scripts/train_eval.py first: `scripts/fit_cohere.py --grid` reads
  train_eval's own walk-forward predictions and cannot see the production
  models (its docstring has the refit steps).
- **Postponement codes unverified live.** cdn marks a postponement with
  postponedStatus "Y" or a PPD status text as far as the archives show; if a
  2026-27 postponement uses another code, that game keeps both teams
  provisional and the run goes RED 6 hours after its tip. Fix: add the code to
  `cdn._postponed`.
- ~~playerIndex from Actions~~ RESOLVED 2026-10-11: the second Actions probe
  fetched it (OK, 0.4s). Every source the bot reads is now proven reachable from
  Actions; stats.nba.com still hangs there, as expected.
- **Dead-man's switch needs a secret.** daily.yml pings `BV_HEALTHCHECK_URL`
  after every run and only warns until the secret exists. A healthchecks.io
  check with the full cron and a 3-hour grace, on since 2026-11-04; the private
  cron and an 8-hour grace if the repo goes private again (docs/runbook.md
  sections 1 and 6).
- **Props thresholds are unmeasured guesses** until a real NBA paste exists:
  the period-board heuristics (0.6 ratio with a 2.0 floor, 0.55 median, 3
  players) and the 20-minute "bench role" void-risk note.
- **Committed-first rendering** (spec section 4, render committed projections
  first and upgrade): the page computes the live frame before drawing; the
  live layer's timeouts were cut to 4 s and fetches run in parallel instead.

## Open after the 2026-11-02 rehearsal: the 2026-27 box lists actives only

The live rehearsal (92 final games) found that a 2026-27 cdn box lists only the
players who dressed (docs/data_notes.md). The bot now rebuilds each such box's
inactive list from its roster snapshots minus the box (docs/serving_design.md
section 9). What remains approximate, measured, and what no test could see:

- **The backlog approximation.** The first run has no snapshot older than its
  own day, so it ingests every game since the opener with that day's roster
  (the earliest snapshot after each game): a first run on 2026-11-02 rebuilds
  the games of 10-20..11-01 from the 11-02 roster, 1 to 13 days younger than
  the game, and a first run a week later the games through 11-08, 1 to 20
  days younger. A player who joined a team after a game gets an
  inactive row for it; one who left before the snapshot without being on the
  report gets none. An outage does the mirror image (the latest snapshot
  before the game is days old). A day's roster cannot be pulled later, so
  every day before the first committed run adds a day to the backlog: the
  13-day row below is the bound only for a run committed on 2026-11-02, and
  a later first run drifts toward the season-end row (20 and 27 days: a
  first run on 11-09 or 11-16). Barrett's decision on 2026-11-02: the
  committed runs start before the deploy, on the private-repo schedule
  (four runs a day, .github/workflows/daily.yml, pushed before dawn on
  2026-11-02), so the first committed run is due at 10:17 AM ET that day
  (cron starts run hours late; one that slips past midnight ET makes it
  11-03 and 14 days). Check which date the first committed
  state/season/2026-27/roster_{date}.parquet carries. Measured on 2025-26 (scratchpad
  scripts stale_rows.py and stale_features.py of the 2026-11-02 session, and
  stale_rows_days.py for 20 and 27 days): every state box stripped of its
  INACTIVE players, rebuilt by the production code (box.rebuild_inactives
  through the bot's step_finals) from a stale roster, against the truth:

  | roster used for every game | wrongly added rows | missed rows | exact team-games |
  |---|---|---|---|
  | 13 days after the game (a first run committed on 2026-11-02) | 727 of 11,099 true (6.6%), 0.29 per team-game, max 6; 0.16-0.20 a team-game Oct-Dec, 0.85 in February (trade deadline) | 87 (0.8%), max 4 | 76.4% |
  | 20 days after (a first run on 2026-11-09) | 940 (8.5%), 0.38 per team-game, max 6; 0.20-0.26 Oct-Dec | 120 (1.1%), max 4 | 70.4% |
  | 27 days after (a first run on 2026-11-16) | 1,138 (10.3%), 0.46 per team-game, max 6; 0.27-0.32 Oct-Dec | 148 (1.3%), max 4 | 65.3% |
  | the season's last day (the worst case) | 2,317 (20.9%), 0.94 per team-game, max 7 | 459 (4.1%), max 4 | 37.9% |

  Before the box-evidence rule (a box dated between the snapshot and the game
  that lists a roster player for another team keeps him off) the wrongly added
  rows were 1,097 and 5,435 (13 days, season end). Effect through the serve
  path and the shipped models (history features of the stale rebuild in place
  of the training table's, context and lines unchanged; 2025-11-15, 12-18,
  01-14, 02-19, 04-10; measured for 13 days and the season end only): 13 days
  stale, the largest p_play shift is 0.08 on the three dates before the trade
  deadline (team expected minutes moved by at most 1.8 against the
  true-history projection) and 0.41 / 0.31 on 02-19 / 04-10 (a player signed
  days before, rebuilt absent for the games before he joined; team expected
  minutes moved by at most 9.0); season-end roster, 0.11-0.22 on the first
  four dates and 0.42 on 04-10 (team expected minutes moved by 0.8 to 9.6
  depending on the date). Without the box-evidence rule the season-end case
  reached 0.77 (MEM 1629111: 0.96 to 0.18, rebuilt absent for his future
  team on the nights he played for MEM) and team expected minutes moved by
  44.6. The features that move are the availability ones only: events_s,
  miss_streak, played_rate_l10, games_since_return, last_absence_len,
  events_with_team, team_changed. Nothing in the bot's sources says when a
  player joined a team, and a re-ingest later would read the same roster
  (snapshots exist only from the first committed run on), so the rest stays:
  the sooner the first committed run, the smaller the backlog. A final game's
  partitions are never rewritten.
- **Tests built on cached last-season payloads could not see a format change
  in this season's.** Every box fixture, the serve-parity replay and the
  2026-10-09 opening-night rehearsal ran on 2025-26 payloads; only a live run
  met a 2026-27 box. A real one is now a fixture
  (tests/fixtures/box/cdn_0022600005.json), and every rebuilt game is a log
  line naming its snapshot and report. The same blind spot holds for any
  source: refresh fixtures from the live season after its first week.
- **Next season's training needs inactive rows for 2026-27.** The local
  backfill (scripts/build_participation.py) parses the same cdn boxes and will
  build a 2026-27 participation table without inactive rows; the 2027-28 seed
  and any model retrained on 2026-27 would then learn from a history that
  hides every injured player. Use the bot's rebuilt partitions, or V3
  BoxScoreSummaryV3 `inactives` if it still lists them for 2026-27 (unchecked).
  Decision for Barrett, before the seed rebuild.
- **The training listing is not exactly the roster.** 2025-26 boxes left out
  a few rostered players (86 of 13,993 t_15 report rows resolve to nobody in
  their game's listing; 112 team-player-games fall inside a stint). A rebuild
  from the playerIndex gives those players rows training never had. Small, and
  in the direction of the truth.
- **Two-way players after the regular season.** A two-way contract cannot
  play in the play-in or the playoffs, and the training listing mostly leaves
  those players out of such boxes, but not always (2021-22..2025-26: 38 of 60
  play-in team-games and 824 of 844 playoff team-games listed 15 or fewer;
  the other play-in boxes listed theirs inactive, as
  INACTIVE_GLEAGUE_TWOWAY, INACTIVE_INELIGIBLE_TO_PLAY, INACTIVE_COACH or with
  no reason, or even active). The rebuild follows the majority
  (box.TWO_WAY_INELIGIBLE: in a play-in or playoff game the snapshot's
  two-way players get no row; a two-way the box lists keeps his box row).
  Measured on 2025-26's six play-in boxes (each team's last regular-season
  listing as the roster, two-way meaning listed INACTIVE_GLEAGUE_TWOWAY by his
  team at least once that season; scratchpad script playin_measure.py):
  without the rule 23 rows wrongly added (5 or 6 in each of four games, every
  one a two-way player), none missed; with it none added and 5 missed, all in
  0052500111 (MIA at CHA), whose box listed its five two-way players
  inactive. No projection reads a play-in game: the rows reach next season's
  events_with_team and any retrain on the bot's 2026-27 partitions. The
  measurements above give a play-in game its own listing as the roster, so
  they cannot see this.
- **Rebuilt reasons** are informational (no feature reads them): with a roster
  equal to the listing, 90.9% of rebuilt rows are named on the t_15 report and
  98.6% of those carry the box's own code (the rest mostly INACTIVE_COACH filed
  as an injury, and injuries filed under the concussion protocol).
- **Snapshots** are one file per ET date a run pulled a good roster (about
  14 KB each, every day of the year the bot runs), rewritten within the day
  when the roster changes, so a late-night run's snapshot can postdate a game of
  the same date. One dated before the season's first state game day (camp
  rosters) is never used.
