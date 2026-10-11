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
- **Team minutes in 2024-25** sum 6.0 short of actual on the full roster
  (docs/model_report.md, audit measurements); other seasons are +2.3 / +2.5.
- **Postponement codes unverified live.** cdn marks a postponement with
  postponedStatus "Y" or a PPD status text as far as the archives show; if a
  2026-27 postponement uses another code, that game keeps both teams
  provisional and the run goes RED 6 hours after its tip. Fix: add the code to
  `cdn._postponed`.
- **playerIndex from Actions** was not in the 2026-10-09 probe; `probe_hosts.py`
  now includes it. Run the probe workflow and read docs/probes before Oct 19.
- **Dead-man's switch needs a secret.** daily.yml pings `BV_HEALTHCHECK_URL`
  after every run and only warns until the secret exists (a healthchecks.io
  check with the cron schedule and a 3-hour grace).
- **Props thresholds are unmeasured guesses** until a real NBA paste exists:
  the period-board heuristics (0.6 ratio with a 2.0 floor, 0.55 median, 3
  players) and the 20-minute "bench role" void-risk note.
- **Committed-first rendering** (spec section 4, render committed projections
  first and upgrade): the page computes the live frame before drawing; the
  live layer's timeouts were cut to 4 s and fetches run in parallel instead.
