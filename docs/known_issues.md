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

## Open
- box (script layer): past-season empty game lists are cached; Cup final (006)
  games cannot be enumerated by build_participation (rest features take them
  from the schedule instead, so no feature is affected); Absent/NotFinal games
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
