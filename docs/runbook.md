# BucketValue runbook: launch and operations

The operator's runbook for the bot (`.github/workflows/daily.yml` ->
`scripts/daily_update.py` -> `scripts/commit_state.sh`) and the Streamlit app
(`app.py`, `pages/`). Rewritten 2026-11-03 for a mid-season launch, with every
claim re-checked against the code at that date. The opening-night version is in
git under its old name: `git show dad9bb9:docs/runbook_opening_night.md`, first
committed in fb9bd55. Spec section 4.

## 0. Where things stand (2026-11-03)

Done:
- **The Actions probes.** The probe of 2026-10-11 reached every source the bot
  read then, the playerIndex included (`docs/probes/`, `docs/known_issues.md`).
  ESPN's team-roster endpoint is read since 2026-11-02 by the in-season roster
  check. It sits on the probed host (site.api.espn.com) but was in no probe;
  a failure there is a WARN only.
- **The bot commits from Actions on a private-repo cron of four runs a day**
  (since 2026-11-02; bot commits 8ece28a, ebc144b, 8f8a126). Each ET day with
  at least one run that pulls a good roster and commits saves that day's roster
  snapshot, which a 2026-27 box needs because it lists active players only
  (`docs/known_issues.md`).
- **`tests.yml` is green on GitHub since dad9bb9.** That is from Barrett's
  screenshot of the Actions tab, and daily.yml's identical Tests step passed
  before bot commits 8ece28a and 8f8a126.
- **The team-minutes step** (models.cohere, a model_version ending in "+tm1")
  is in every projection computed since 055d752.

Not done, in the order of sections 1 and 3:
- **The healthcheck** (`BV_HEALTHCHECK_URL`) is not set.
- **The repo is private.**
- **No app is deployed.**
- **The full cron is still commented out in `daily.yml`.**

GitHub starts cron runs late and sometimes not at all. On 2026-11-02, 2 of the
4 slots committed. On 2026-11-03 no bot commit landed between 04:01 and 21:04
UTC; Barrett's 055d752 at 09:18 UTC was the only commit. So the bot's first
commit of that ET day came at 4:04 PM. `scripts/daily_update.py` cites 2 to 7
hours of lag measured on DiamondValue. The first line of the Bot step's log
(`BucketValue daily update: ET date D, run at ...`) gives each run's real start.

## 1. Now, while the repo is private: the dead-man's switch

A missed slot alerts nobody today, and a day without a good run loses that
day's roster snapshot for good. Set the switch before anything else:

1. On healthchecks.io (the free tier is enough), create a check with schedule
   type **Cron**, expression **`17 2,15,20,23 * * *`**, time zone **UTC**, and a
   grace time of **8 hours**. That is the private cron in `daily.yml`. Its runs
   are 3 to 13 hours apart (9:17 PM to 10:17 AM EST is 13), and starts run hours
   late.
2. Turn on alerts by email, and phone or push if wanted.
3. Copy the check's ping URL into the repo secret **`BV_HEALTHCHECK_URL`**
   (Settings, Secrets and variables, Actions).

The workflow's last step (`if: always()`) pings it on every run that starts:
success, or `/fail` for a RED or failed run. A run that never starts sends
nothing, and the missing ping is the alert. Until the secret exists, each run
prints a warning instead.

## 2. Before the flip: what becomes public

Making the repo public publishes its whole history. That is Barrett's call,
and these are the things to decide on first:

- **Every commit Barrett makes carries his personal address, as author and as
  committer.** That is 38 of the 43 commits before this file
  (`git log --all --format='%ae %ce' | sort | uniq -c`); the bot's carry
  bucketvalue-bot@users.noreply.github.com.
- **Real betting lines.** The spec says historical lines are never committed,
  but two committed things hold real lines:
  - The ESPN odds test fixture `tests/fixtures/espn/odds_events.json.gz`
    (bb70d2c) holds real spreads and totals of 21 games, 2015-16..2025-26.
  - The bot's 2026-27 captures, `state/season/2026-27/odds_*.parquet`, which
    the app reads.

  Decide on the fixture before the flip. The large line archives
  (`cache/game_lines_*`, `cache/local_only/`) were never committed.
- **The spec, every doc and the other test fixtures:**
  - injury-report PDFs;
  - cdn and ESPN payloads;
  - constructed PrizePicks fixtures, which are not real boards
    (`tests/fixtures/props/PROVENANCE.txt`).
- **Actions run logs, private-era runs included.** They carry the RED lists and
  the real start times. GitHub's dialog for changing visibility warns that
  Actions history becomes visible; that is platform behaviour, not checkable
  here. Delete old runs first if that matters.

What stays private:
- **cache/** (gitignored, and nothing under it except `cache/.gitkeep` was ever
  committed);
- **the accuracy record**: `pages/Accuracy.py`, `cache/accuracy_summary_*` and
  `cache/props_record_*` are gitignored and never in the nav.

## 3. Launch, in this order

The order is forced. The full cron spends a private repo's free minutes within
about a week. The healthcheck must match whichever cron is on. And the public
site should not show games frozen hours before tip.

1. **Pull** the development tree (`git pull --ff-only origin main`): the bot
   pushes state/ several times a day.
2. **Clone-and-run** (spec lesson 19), outside the iCloud Desktop:
   ```
   git clone https://github.com/Barrett879/bucketvalue ~/bucketvalue-check
   cd ~/bucketvalue-check
   /opt/anaconda3/bin/python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   .venv/bin/python -m pytest -q
   .venv/bin/streamlit run app.py
   ```
   - **The suite:** the same command the workflow runs. It must show 0 failed;
     the local-cache tests skip in a clone.
   - **The app:** check `/`, one `/Game?date=...&away=...&home=...` page of the
     newest committed date, and `/About`, against the list in step 7.
   - **What the render tests cover:** `tests/test_pages_smoke.py` renders the
     newest committed slates and every game page from a checkout; the workflow
     reruns it before each commit.
3. **Flip the repo public** (Barrett).
4. **Swap the cron.** In `daily.yml`, replace the line
   `- cron: "17 2,15,20,23 * * *"` with the two commented full-schedule lines
   (`17,47 13-23 * * *` and `17,47 0-6 * * *`), push, then **dispatch one run by
   hand** (Actions, daily-update, Run workflow, date input EMPTY).
   - **Why dispatch:** a push never triggers daily.yml (it has only `schedule`
     and `workflow_dispatch`), and the new crons take effect on their next slot.
   - **The date input:** never type a date to "run tomorrow early". A future
     date is refused (exit 2), and a plain run already projects tomorrow as
     provisional.
5. **Switch the healthcheck in the same sitting**: expression
   `17,47 13-23,0-6 * * *`, grace **3 hours**. The full expression on the
   private cron alarms within hours, and an 8-hour grace on a 30-minute cron
   hides an outage.
6. **Deploy on Streamlit Community Cloud** (Barrett's account):
   - **Settings:** repository `Barrett879/bucketvalue`, branch `main`, main file
     `app.py` (About and Game come from `pages/`).
   - **Python:** choose **3.12** under Advanced settings, the version the bot
     runs (`runtime.txt`, which the workflow reads). The registry pins
     scikit-learn, and `models.load()` refuses a different one rather than serve
     wrong numbers; `requirements.txt` pins it. No `packages.txt` is needed.
     Keep `.streamlit/config.toml`: its `enableStaticServing` serves the two
     webfonts.
   - **Secrets:** add the root-level secret **`BUCKETVALUE_READONLY = "1"`**. The
     app reads no `st.secrets`. It reads two environment variables:
     - `BUCKETVALUE_READONLY`, which declares the host shared. The backfill
       injury store then stays in memory, and the props record refuses to
       write.
     - `BUCKETVALUE_LOG`, optional, default WARNING.

   **Deploy no earlier than step 4.** On the private cron a game freezes on the
   last run before its tip minus 5 minutes. That is up to about 3 hours before
   an evening tip and up to about 5 before an afternoon one: a tip between
   about 10:30 AM and 3:30 PM EST freezes on the 10:17 AM run, such as the
   1:00 PM games of 11-15 and 11-21 and the 3:30 PM games of 11-08 and 11-22,
   unless the 3:17 PM run reaches its projections by 3:25. A late start freezes
   them earlier still, and the public site would show those numbers.
7. **Check the live app.** A healthy site shows these:
   - **`/` opens on the ET date of now minus 4 hours.** Its tiles read Games,
     First tip, Lines posted n/N and Injury reports filed n/N. The deck reads
     `Injury report: <time> (<date> report).`, and a note reads
     `Box scores through <Month> <day>.`, matching the bot's latest ingested
     finals.
   - **`Provisional.`** appears only on tomorrow's slate or on a game with a
     named gap. A game without both lines says
     `Projections post when the line does`.
   - **A game page** shows both teams' tables; after its tip, `Frozen at tip`
     and the box score.
   - **`/About`** shows the trust table.
   - **Untipped numbers are recomputed live,** so they can differ slightly from
     the committed file.

   **No notices**, and a notice that persists is a host problem to report, not
   an outage: every probe ran from GitHub's runners, never from Streamlit's
   hosts.
   - `Showing a saved schedule from N minutes ago.` is cdn. The page uses the
     newer of the process's last copy and the committed one, and projections
     and mark-outs still run on it.
   - `Game lines are unavailable right now.` is ESPN. The cards have no
     fallback, so untipped cards show no spread or total, and the next notice
     comes with it.
   - `Showing the published projections. The live update could not run (<why>)`
     means the live recompute failed: the committed rows are served, without
     mark-outs. `<why>` names the cause (`bucketlib/store.py`, `_STALE_WHY`):
     - `the lines feed did not answer` is ESPN;
     - `the injury report could not be fetched` or `no injury report could be
       read` means ak-static failed AND no committed report was usable;
     - `the schedule could not be read` is the schedule or the committed state;
     - `this date's history features are not published` means the bot has not
       committed that date's features yet;
     - `the models could not be loaded` is a deploy fault, a wrong
       scikit-learn on the host;
     - `the live inputs look unlike anything the models were trained on` is a
       null skew;
     - `the published features lack a column the models need` is a
       train/serve mismatch;
     - `the live update hit an error` is a bug.
   - An ak-static outage alone shows no notice while the bot's committed report
     is in reach.
   - The game page adds `Showing the box score from N minutes ago.` or
     `The box score is unavailable right now.` (cdn liveData, no committed
     copy).

   Not visible on the site: the `model_version` column of
   `state/serve/<date>/projections_v1.parquet` says which models made a slate
   (2026-11-04: `v1-2026-10-09+tm1`; 2026-11-02: `v1-2026-10-09`).

## 4. Daily operations: reading a run

The Bot step's log ends with `RUN IS RED (n):` and the list. A RED run still
commits what it wrote (stale beats empty), then fails the job.

**Normal lines, not faults**
- **`finals: ... lists no inactive player; N rebuilt from the D roster ...`** on
  every 2026-27 box. These are the inactive rows rebuilt from roster
  snapshots.
- **The ESPN roster cross-check line** for a team over 18 rows (LAC lists 19
  since 2026-11-02). It is an info line when ESPN agrees, and a WARN naming
  who only one source lists when it does not. It is never RED, and nothing is
  dropped.
- **`WARN projections D: <game> froze on rows computed with ..., but the <T> ET
  report was committed before its freeze point`.** The frozen rows missed a
  report the bot already held. On the private cron this is expected (section
  3, step 6); on the full cron it means a projections step failed or did not
  run in time.
- **`WARN finals: N state game(s) before today not ingested yet ...`** after
  late tips. Their teams' next games stay provisional until ingested.
- **`WARN finals: ... (the schedule says Final; retried next run)`.** A box that
  is not final yet, under 6 hours after tip. After that it becomes the
  `still not final ... after tip` RED.
- **`WARN schedule: unrecognised postponedStatus ...`.** A postponedStatus value
  cdn does not know (only "Y", or the status text PPD or Postponed, counts as
  postponed). Check the feed: if those games were postponed, add the code to
  the positive check in `cdn._postponed` (`docs/known_issues.md`); if not, add
  it to `cdn.NOT_POSTPONED`. Never add it as a postponement unchecked.

**RED messages (Bot step)**

| Message | Meaning | Action |
|---|---|---|
| `schedule blocked: ...`, `roster blocked: ...` | A hang, an Akamai 403/503, or repeated 429/5xx on that pull | The committed copy stands in. If it repeats for hours, section 5 |
| `schedule unusable (...)`, `roster unusable (...)`, `schedule parsed empty` | A bad payload or a parser refusal | The committed copy stands in. Look at the payload if it repeats |
| `roster has only N players (under 400)` | A truncated playerIndex | The committed roster stands in, and no snapshot is written for that pull |
| `box score blocked at X` | The box host refused | That run's box fetches stop: the games wait for a later run, and their teams' next games stay provisional |
| `box score absent for final game X`, `box score fetch failed for X`, `parse failed for X` | That game only | Retried every run. If it persists, see "A stuck game" below |
| `box score of final game X is still not final ... after tip` | A cdn box frozen mid-game | See "A stuck game" below |
| `box score of X lists no inactive player and there is no roster snapshot ...` | The season has no snapshot at all on or after its first game day: impossible while `roster_2026-11-02.parquet` is committed | Not ingested; retried next run. A missed day never causes it: the game is rebuilt from the latest snapshot before it, else the earliest after (`state.roster_for_game`) |
| `inactive rows of X could not be rebuilt (...)` | The rebuild failed for that game | Not ingested; retried next run. Read the exception |
| `N state game(s) before today are not ingested 6.0h after tip` | A stuck feed or box | The games named keep their teams' next games provisional (never frozen) until fixed |
| `injury reports blocked: ...`, `the newest injury report could not be fetched` or `does not parse` | The ak-static host, or a new PDF layout | The committed report stands in while in reach (30 hours). A parse failure needs a parser fix |
| `injuries: ..., and the committed report is unusable` | No usable report at all | Projections for the date are refused (no report is a slate training never saw), and the committed rows stand |
| `odds D: <Error>` | The ESPN odds pull failed for D | The committed odds snapshots of D still feed the projections. A game with no spread and total in them is withheld, and one projected earlier keeps its committed rows. The other date is still tried |
| `projections D: N untipped game(s) without a whole roster (under 12 rows a side)` | A broken roster pull for those teams | The game is still projected and committed with the short side. models.cohere leaves a side under 12 rows as predicted, so no minutes move onto the rows present. Fix the roster pull |
| `projections D: nothing new written: ...` (NoReport) or `projections D: nothing new written (a null skew): ...` | No report in reach, or serve rows unlike training's | The committed projections are served |
| `projections D: <Error>: ...` | An unexpected exception projecting D | The other date still ran. A bug: read the traceback |
| `projections: no roster (live or committed); nothing projected` | Neither roster exists | Nothing projected |
| `<step> step raised <Error>: ...` | An unexpected exception in a step | The other steps still ran. A bug: read the traceback |

The short-roster and nothing-new-written rows are RED for today's slate only.
For tomorrow's slate the same text prints as `WARN projections <tomorrow>: ...`,
and the run stays green: every later run retries it. The roster rule
`projections D: the roster pull lists more than 18 players` fired only through
the first game day (camp rosters); in season the ESPN cross-check replaces it.

**Workflow-step failures.** Each fails the job and pings `/fail` once the
secret is set.
- **A red Tests step:** the Bot never runs.
- **A red App smoke step:** the run's writes are discarded (the Commit step has
  no `if: always()`).
- **`the bot changed files outside state/`:** the allow-list in
  `scripts/commit_state.sh`. A bug, and nothing is committed.
- **`state conflicts with a newer push`:** this run computed on a stale base.
  Nothing to do: the next run redoes it.
- **`could not push state after 3 attempts`:** in Actions the run's state commit
  is lost with the runner, and the next run recomputes it. On the Mac
  (`--local`) it stays a local commit: push it, or `git reset --hard
  origin/main`, before the next run.
- **A job over 40 minutes** is cancelled.

**A lost roster day.** If no run of an ET day both pulls a good roster and
commits, that day's snapshot is gone:
- **A RED run that commits keeps it.**
- **These lose it if they hit every run of the day:** a red Tests or App smoke
  step, a conflict, a failed push, or a roster RED (`roster blocked`, `roster
  unusable`, `under 400`).

Fix forward, and dispatch so the roster step runs before midnight ET.

**A stuck game (2026-27).** Do not follow the old override route (backfill X,
then `scripts/build_serving_seed.py --override X`) for this season:
- the auto backfill refuses a live-season box that is not final;
- `state.override_from_backfill` copies the backfill's participation without
  rebuilding inactive rows, so the game would be committed for good without
  its injured players, which the bot itself refuses to do.

The candidate route runs on the Mac (stats.nba.com) and is unverified: the V3
inactives are unchecked (`docs/known_issues.md`).
1. `.venv/bin/python scripts/build_participation.py --games X --source v3`
2. Confirm the V3 rows list the inactive players. BoxScoreSummaryV3 inactives
   carry no reason.
3. `.venv/bin/python scripts/build_logs.py --seasons 2026-27 --types RS`. The
   override also needs player_games and team_games.
4. Only then, `.venv/bin/python scripts/build_serving_seed.py --override X`.
   Pull, then commit and push `state/season/2026-27/override/`.

Waiting has a cost:
- **Every later run is RED** (`N state game(s) before today are not ingested
  6.0h after tip`). So every run pings `/fail`, and the healthcheck stays down,
  which hides any new failure.
- **Every remaining game of both teams this season stays provisional,** never
  frozen.

Decide within days.

## 5. Fallback: the Mac as the single writer

Spec 4's fallback order: cdn from Actions; this Mac as the single writer; the
visible "data through" stamp. Use the Mac when Actions cannot reach cdn for
hours, or Actions is down.

1. **One writer.** Disable the workflow first: Actions, daily-update, the "..."
   menu, Disable workflow. `gh` is not installed on this Mac; installed, it
   would be `gh workflow disable daily-update -R Barrett879/bucketvalue`.
2. **A dedicated clone,** never the development tree:
   `git clone https://github.com/Barrett879/bucketvalue ~/bucketvalue-writer`,
   then a `.venv` from `requirements.txt`.
   - This Mac has no Python 3.12, which is what the bot runs. Either install
     CPython 3.12.8, or knowingly use `/opt/anaconda3/bin/python3 -m venv .venv`
     (3.13.9), as the development tree does.
3. **Each run,** from that clone:
   ```
   git pull --rebase origin main
   .venv/bin/python scripts/daily_update.py
   .venv/bin/python -m pytest -q tests/test_pages_smoke.py && bash scripts/commit_state.sh --local main
   ```
   - **The smoke test gates the commit,** as it does in the workflow. Without
     it, a broken render could be committed.
   - **`commit_state.sh --local`** commits `state/` only and pushes with the
     workflow's pull-rebase rules.
   - **A conflicting rebase** is aborted, but the local state commit stays.
     Reset with `git reset --hard origin/main` before the next run.
4. **Nothing pings the healthcheck from the Mac.** Pause the check while the
   workflow is disabled, or ping it with `curl -fsS <ping URL>` after each run.
5. **To repeat it every 30 minutes**, a launchd agent
   (`~/Library/LaunchAgents/com.bucketvalue.writer.plist`) can run
   `/bin/bash -lc 'cd ~/bucketvalue-writer && git pull -q --rebase origin main && { .venv/bin/python scripts/daily_update.py; .venv/bin/python -m pytest -q tests/test_pages_smoke.py && bash scripts/commit_state.sh --local main; }'`
   with `StartInterval` 1800.
   - Nothing commits after a failed `cd` or pull, or a failed smoke test.
   - Give it `StandardOutPath` and `StandardErrorPath`, or its RED lists are
     never seen.
   - It does not run while the Mac sleeps.
   - Installing it is Barrett's call.
6. **To go back:** unload the agent, re-enable the workflow (the same menu,
   Enable workflow), and resume the healthcheck. Never both writers at once.

## 6. Rollback: back to private

- **Swap the cron back first,** in the same change: the private line in
  `daily.yml`. The full schedule spends a private repo's 2,000 free minutes in
  about a week, and runs then stop starting: one lost roster snapshot a day.
- **Put the healthcheck back** on `17 2,15,20,23 * * *` with the 8-hour grace.
- **Take the app down, or make it private, before the repo.** Per the spec,
  Community Cloud shares apps from public repos only; that is platform
  behaviour, not checkable from here.
- **Taking only the site down** needs no change of visibility.

## 7. Once public, and the offseason

- **Free minutes.** Public runs cost no Actions minutes, which is what makes the
  36-run schedule affordable.
- **The 60-day rule.** GitHub disables a public repo's scheduled workflows after
  60 days without activity. `scripts/commit_state.sh` commits a dated
  `state/keepalive.txt` whenever the newest commit is 45 days old, but only
  inside a run that reaches the Commit step. A suite that stays red stops the
  keepalive too, and a disabled schedule needs a manual re-enable (Actions,
  daily-update, Enable workflow).
- **The seed.** It must be rebuilt for 2027-28 between the Finals and the next
  opener (`docs/known_issues.md`, "The seed serves ONE season").
