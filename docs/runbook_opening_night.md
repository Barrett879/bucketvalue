# Opening-night runbook (2026-27 opener, Tuesday 2026-10-20)

The go-live sequence for the bot (`.github/workflows/daily.yml`,
`scripts/daily_update.py`), the checks after each manual run, and what to do
when the bot cannot run. Spec section 4 ("Opening-night runbook: manual runs
Oct 19 evening and Oct 20 morning").

Facts this is built on (checked 2026-10-09 against the live cdn schedule):

- The first game is **BOS at DET, 3:00 PM ET** on Oct 20 (gameDateTimeUTC
  19:00Z, status text "3:00 pm ET"), not 7:00 PM. NYK-PHI is 7:00 PM, SAS-OKC
  9:30 PM. A game freezes at tip minus 5 minutes, so the **first freeze is
  2:55 PM ET**. Re-check on Oct 19: the schedule can still move.
- The playerIndex holds camp rosters (18-21 a team, 615 players on Oct 9)
  until the cut-down the day before the opener. The bot turns a team with more
  than 18 roster rows RED today and a warning tomorrow (`MAX_TEAM_ROWS`).
- 36 runs a day of 6-10 minutes is about 300 runner-minutes a day: a private
  repo spends its free Actions minutes within days. Public first, then cron.

## 1. Before Oct 19 (once)

1. Commit the bot fixes. Pushing `scripts/probe_hosts.py` triggers
   `.github/workflows/probe.yml`: check the new `docs/probes/probe_github-actions_*.json`
   says `cdn_player_index` OK (the bot's roster source was not in the first
   probe). If it is not OK, stop: the roster step cannot run from Actions.
2. Flip the repo public (Barrett's call; Streamlit Cloud shares public repos only).
3. Deploy the app on Streamlit Cloud (clone-and-run first, spec lesson 19).
   Set app secrets only if the app reads any (none on 2026-10-09).
4. Dead-man's switch. Create a check on healthchecks.io (free tier is enough):
   - schedule type **Cron**, expression `17,47 13-23,0-6 * * *`, time zone
     **UTC** (the workflow's crons), grace time **3 hours**;
   - alerts by email, and phone/push if wanted;
   - copy its ping URL into the repo secret **`BV_HEALTHCHECK_URL`**
     (Settings, Secrets and variables, Actions).
   The workflow's last step pings it on every run that gets that far
   (`/fail` for a RED or failed run). No pings for 3 hours past an expected
   run means runs stopped starting (GitHub dropped or delayed the crons,
   the workflow got disabled, or the workflow file is broken). Until the
   secret is set, each run prints a warning instead.
5. Uncomment the `schedule:` block in `daily.yml` (only once the repo is
   public) and push it.

## 2. Oct 19, after the 5:00 PM ET cut-down

Dispatch **daily-update** by hand with the **date input EMPTY**. Never type a
date to "run tomorrow early": a future date is refused (exit 2), and a plain
run already projects tomorrow, as provisional.

Check in the run's Bot step log:

- `roster: N players`: about 30 x 15-18. Any line
  `WARN projections 2026-10-20: the roster pull lists more than 18 players for ...`
  means the playerIndex has not caught up with the cuts. Run again later; if it
  persists into Oct 20 it turns RED (waived players would be projected).
- `schedule: ... games` and that Oct 20 lists the opening games at the times
  above.
- `injuries: report ...`: the report time. On Oct 19 evening the slate may
  have no Oct 20 report yet; tomorrow's slate then logs a
  `WARN ... nothing new written: no injury report at all`, which is expected.
- `odds 2026-10-20: N priced games`: games without a line are WITHHELD
  ("Projections post when the line does"), never RED.
- The RED list at the end (`RUN IS RED`) and the job result.
- After the commit: `state/serve/2026-10-20/features_v1.parquet` exists, and
  `projections_v1.parquet` once a report and lines exist. Every Oct 20 row is
  `provisional` (tomorrow's slate).

## 3. Oct 20, morning (by about 11:00 AM ET, well before the 2:55 PM freeze)

Dispatch again, date input EMPTY. Check:

- no roster RED (`more than 18 players`);
- the injury report is today's (`injuries: report 2026-10-20 ...`);
- every opening game has a line (none withheld), or note which are withheld;
- `projections 2026-10-20: ... rows`, and no game listed
  `provisional on stale state` (on opening night there is no prior game, so
  completeness has nothing to miss);
- the site's Oct 20 slate shows the three games with numbers.

Then watch the first scheduled runs land in the Actions tab and the first
pings arrive on healthchecks.io.

## 3b. Oct 20, between about 1:15 and 2:30 PM ET (before the 2:55 PM freeze)

Dispatch once more, date input EMPTY. Teams file game-day statuses by 11 AM to
1 PM local, so this run is the one whose BOS at DET rows freeze: whatever the
last run before 2:55 PM ET committed is the frozen projection. If the scheduled
crons are already on, check that one landed in this window instead. After 2:55
PM nothing for BOS at DET is ever written again.

## 4. What RED means and what to do

| RED message (Bot step) | Meaning | Action |
|---|---|---|
| `... blocked` | A host refused Actions (Akamai) | The committed copy stands in. If it repeats for hours, use section 5 |
| `schedule/roster unusable (...)` | A bad payload or a parser refusal; the committed copy stands in | Look at the payload if it repeats |
| `box score of final game X is still not final ...` | A cdn box frozen mid-game (17 in 2019-20) | Backfill X locally, then `scripts/build_serving_seed.py --override X` and commit `state/season/{season}/override/` |
| `parse failed for X` / `box score absent / fetch failed for X` | That game only; every other game is still ingested | Same override route if it does not clear on the next runs |
| `N state game(s) before today are not ingested 6.0h after tip` | A stuck feed or box | The games named keep their teams' next games provisional until fixed |
| `projections D: ... more than 18 players` | Camp rosters in the playerIndex | Wait for the cdn index; nothing to commit |
| `projections D: nothing new written: no injury report at all` | No report fetched and none committed | The app serves the committed projections |
| `the bot changed files outside state/` | The allow-list (`scripts/commit_state.sh`) | A bug: nothing was committed |
| `state conflicts with a newer push` | This run computed on a stale base | Nothing; the next run redoes it on the new tip |

A late game still live after midnight is a WARN, not RED, for 6 hours after
its tip; its teams' next games stay provisional (never frozen) until it is
ingested.

## 5. Fallback: the Mac as the single writer

Spec 4's fallback order is cdn from Actions, then a job on Barrett's Mac as
the single writer, then ESPN summary box scores (NOT implemented: proposed to
drop from the spec), then the visible "data through" stamp. Use the Mac when
Actions cannot reach cdn for hours, or Actions is down.

1. ONE WRITER: disable the workflow first
   (`gh workflow disable daily-update`, or the Actions tab).
2. Use a dedicated clone, never the development tree:
   `git clone https://github.com/Barrett879/bucketvalue ~/bucketvalue-writer`,
   then create its `.venv` from `requirements.txt` (Python per `runtime.txt`).
3. Each run, from that clone:

   ```
   git pull --rebase origin main
   .venv/bin/python scripts/daily_update.py
   bash scripts/commit_state.sh --local main
   ```

   `commit_state.sh --local` stages and commits `state/` only and pushes with
   the same pull-rebase rules as the workflow.
4. To repeat it every 30 minutes, a launchd agent
   (`~/Library/LaunchAgents/com.bucketvalue.writer.plist`) can run
   `/bin/bash -lc 'cd ~/bucketvalue-writer && git pull -q --rebase origin main && .venv/bin/python scripts/daily_update.py; bash scripts/commit_state.sh --local main'`
   with `StartInterval` 1800. Installing it is Barrett's call.
5. To go back: unload the agent, then re-enable the workflow
   (`gh workflow enable daily-update`). Never both at once.

## 6. Offseason

`scripts/commit_state.sh` commits a dated `state/keepalive.txt` whenever the
newest commit is 45 days old, so GitHub's 60-day rule never disables the
schedule over the summer. The seed must be rebuilt for 2027-28 between the
Finals and the next opener (docs/known_issues.md, "The seed serves ONE season").
