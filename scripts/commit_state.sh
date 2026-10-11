#!/usr/bin/env bash
# Commit and push the bot's state/ and NOTHING else. The one implementation
# of the bot's commit, used by .github/workflows/daily.yml (the "Commit state/
# only" step) and by the local single-writer fallback in
# .github/runbook_opening_night.md.
#
#   scripts/commit_state.sh [--local] [BRANCH]
#
# BRANCH defaults to $GITHUB_REF_NAME, else the current branch.
#
# 1. The allow-list. In CI (the default) anything changed outside state/,
#    tracked, untracked or ignored, fails the run: the bot writes ONLY state/.
#    Each porcelain line is "XY path" (the path quoted when it holds a space),
#    so every exclusion is anchored on the path right after the two status
#    columns: state/ itself, and the interpreter's and pytest's caches at ANY
#    depth (the Tests step leaves a root .pytest_cache/, the imports a
#    __pycache__/ in every package). --local (Barrett's Mac, a working tree
#    with cache/, .venv/ and his own edits) skips that check and pulls with
#    --autostash; either way only paths under state/ are staged and committed
#    (`git commit -- state/`).
# 2. The keepalive. GitHub disables a public repo's scheduled workflows after
#    60 days without repository activity. When the newest commit is older
#    than KEEPALIVE_DAYS, state/keepalive.txt gets today's ET date, so the
#    quiet offseason still commits once in a while.
# 3. Push with `pull --rebase` and retries. A rebase that CONFLICTS means this
#    run computed state on a base that a newer push already moved (a run that
#    queued behind another one): the rebase is aborted and the run fails with
#    that message, never retried (the same rebase conflicts again); the next
#    run recomputes on the new tip. A pull or push that fails for any other
#    reason (the network, a push that lost a race) is retried.
set -uo pipefail

KEEPALIVE_DAYS="${KEEPALIVE_DAYS:-45}"
ATTEMPTS="${ATTEMPTS:-3}"
SLEEP_UNIT="${SLEEP_UNIT:-15}"

local_mode=0
pull_opts=(--rebase)
if [ "${1:-}" = "--local" ]; then
  local_mode=1
  # Barrett's own uncommitted edits (outside state/) ride along untouched.
  pull_opts=(--rebase --autostash)
  shift
fi
branch="${1:-${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}}"

if [ "$local_mode" = 0 ]; then
  other=$(git status --porcelain --untracked-files=all --ignored \
          | grep -v -E '^.. "?(state/|(.*/)?(__pycache__|\.pytest_cache)/)' || true)
  if [ -n "$other" ]; then
    echo "::error::the bot changed files outside state/:"
    echo "$other"
    exit 1
  fi
fi

last=$(git log -1 --format=%ct 2>/dev/null || echo 0)
now=$(date +%s)
if [ $(( (now - last) / 86400 )) -ge "$KEEPALIVE_DAYS" ]; then
  mkdir -p state
  TZ=America/New_York date '+%Y-%m-%d' > state/keepalive.txt
  echo "keepalive: the newest commit is $(( (now - last) / 86400 )) days old; state/keepalive.txt stamped"
fi

git config user.name >/dev/null || git config user.name "BucketValue bot"
git config user.email >/dev/null || git config user.email "bucketvalue-bot@users.noreply.github.com"
git add -A state/ || exit 1
if git diff --cached --quiet -- state/; then
  echo "state unchanged: nothing to commit"
  exit 0
fi
git commit -q -m "Bot: state as of $(TZ=America/New_York date '+%Y-%m-%d %H:%M') ET" -- state/ || exit 1

rebasing() {
  local d
  d=$(git rev-parse --git-dir)
  [ -d "$d/rebase-merge" ] || [ -d "$d/rebase-apply" ]
}

for attempt in $(seq 1 "$ATTEMPTS"); do
  if git pull "${pull_opts[@]}" origin "$branch"; then
    if git push origin "HEAD:$branch"; then
      exit 0
    fi
    echo "push attempt $attempt failed; retrying"
  elif rebasing; then
    git rebase --abort || true
    echo "::error::state conflicts with a newer push: this run computed on a stale base and its" \
         "state is dropped (the next run recomputes it on the new tip)"
    exit 1
  else
    echo "pull attempt $attempt failed; retrying"
  fi
  [ "$attempt" -lt "$ATTEMPTS" ] && sleep $((attempt * SLEEP_UNIT))
done
echo "::error::could not push state after $ATTEMPTS attempts"
exit 1
