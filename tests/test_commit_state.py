"""scripts/commit_state.sh, the bot's commit (the daily workflow's "Commit
state/ only" step and the local single-writer fallback), run for real against
throwaway git repositories in a tmp dir: an origin, and clones that play the
bot's runs.

What it must do: commit state/ and nothing else; fail when anything outside
state/ changed (CI); stamp a keepalive once the newest commit is old; push
over a newer push that touched other files; and when a newer push touched
the SAME state files (a run that computed on a stale base), abort the rebase
and fail with that message instead of retrying a rebase that cannot succeed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "commit_state.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None,
                                reason="needs git and bash")


def _env(tmp: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=str(tmp / "gitconfig"), GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com",
               SLEEP_UNIT="0", GITHUB_REF_NAME="main", **extra)
    return env


def git(cwd: Path, *args, env: dict, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=check, capture_output=True,
                          text=True)


def commit_state(cwd: Path, env: dict, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True,
                          text=True)


@pytest.fixture
def repos(tmp_path):
    """(origin, clone a, clone b, env): origin holds one commit with a state
    file and a code file; a and b are clones of it, as two bot runs checked
    out from the same commit."""
    env = _env(tmp_path)
    (tmp_path / "gitconfig").write_text("[init]\n\tdefaultBranch = main\n")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main", env=env)
    (seed / "state" / "current").mkdir(parents=True)
    (seed / "state" / "current" / "injury_v1.bin").write_bytes(b"report 1\n")
    (seed / "state" / "current" / "odds_v1.bin").write_bytes(b"odds 1\n")
    (seed / "app.py").write_text("print('app')\n")
    git(seed, "add", "-A", env=env)
    git(seed, "commit", "-q", "-m", "seed", env=env)
    origin = tmp_path / "origin.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(origin), env=env)
    clones = []
    for name in ("a", "b"):
        git(tmp_path, "clone", "-q", str(origin), name, env=env)
        clones.append(tmp_path / name)
    return origin, clones[0], clones[1], env


def _log(origin: Path, env: dict) -> list[str]:
    return git(origin, "log", "--format=%s", "main", env=env).stdout.split("\n")[:-1]


def test_commits_state_only_and_nothing_when_unchanged(repos):
    origin, a, _, env = repos
    r = commit_state(a, env)
    assert r.returncode == 0 and "state unchanged" in r.stdout
    (a / "state" / "current" / "injury_v1.bin").write_bytes(b"report 2\n")
    r = commit_state(a, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _log(origin, env)[0].startswith("Bot: state as of")
    shown = git(origin, "show", "--name-only", "--format=", "main", env=env).stdout.split()
    assert shown == ["state/current/injury_v1.bin"]


def test_a_change_outside_state_fails_in_ci_and_is_never_committed_locally(repos):
    origin, a, _, env = repos
    (a / "app.py").write_text("print('edited')\n")
    (a / "state" / "current" / "odds_v1.bin").write_bytes(b"odds 2\n")
    r = commit_state(a, env)
    assert r.returncode == 1 and "outside state/" in r.stdout and "app.py" in r.stdout
    assert _log(origin, env) == ["seed"]
    # --local (a working tree with other edits): only state/ is committed.
    git(a, "add", "app.py", env=env)                        # even a staged edit stays out
    r = commit_state(a, env, "--local")
    assert r.returncode == 0, r.stdout + r.stderr
    shown = git(origin, "show", "--name-only", "--format=", "main", env=env).stdout.split()
    assert shown == ["state/current/odds_v1.bin"]


def test_a_newer_push_on_other_files_is_rebased_over(repos):
    origin, a, b, env = repos
    (a / "state" / "current" / "injury_v1.bin").write_bytes(b"report 2\n")
    assert commit_state(a, env).returncode == 0
    (b / "state" / "current" / "odds_v1.bin").write_bytes(b"odds 2\n")     # b never saw a's push
    r = commit_state(b, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(_log(origin, env)) == 3
    for f, want in (("injury_v1.bin", b"report 2\n"), ("odds_v1.bin", b"odds 2\n")):
        got = git(origin, "show", f"main:state/current/{f}", env=env).stdout.encode()
        assert got == want


def test_a_stale_run_that_conflicts_aborts_and_fails_without_retrying(repos):
    """Run b computed on the commit both runs were created on; run a pushed a
    different version of the same state file first. b's rebase conflicts:
    it is aborted (no rebase left in progress), b fails with the stale-base
    message after ONE attempt, and origin keeps a's state."""
    origin, a, b, env = repos
    (a / "state" / "current" / "injury_v1.bin").write_bytes(b"report 2 (run a)\n")
    assert commit_state(a, env).returncode == 0
    (b / "state" / "current" / "injury_v1.bin").write_bytes(b"report 2 (run b)\n")
    r = commit_state(b, env)
    assert r.returncode == 1
    assert "stale base" in r.stdout and "attempt 2" not in r.stdout + r.stderr
    gd = Path(git(b, "rev-parse", "--git-dir", env=env).stdout.strip())
    gd = gd if gd.is_absolute() else b / gd
    assert not (gd / "rebase-merge").exists() and not (gd / "rebase-apply").exists()
    assert git(b, "status", "--porcelain", env=env).stdout == ""
    got = git(origin, "show", "main:state/current/injury_v1.bin", env=env).stdout
    assert got == "report 2 (run a)\n"


def test_a_failed_pull_is_retried(repos, tmp_path):
    """A pull that fails for another reason (the remote unreachable) is
    retried ATTEMPTS times, then the run fails."""
    _, a, _, env = repos
    git(a, "remote", "set-url", "origin", str(tmp_path / "nowhere.git"), env=env)
    (a / "state" / "current" / "odds_v1.bin").write_bytes(b"odds 2\n")
    r = commit_state(a, env)
    assert r.returncode == 1 and "pull attempt 3 failed" in r.stdout
    assert "could not push state after 3 attempts" in r.stdout


def test_an_old_newest_commit_gets_a_keepalive(repos):
    """GitHub disables a public repo's schedules after 60 days without
    activity: once the newest commit is KEEPALIVE_DAYS old, a dated
    state/keepalive.txt is committed even with nothing else changed."""
    origin, a, _, env = repos
    old = dict(env, GIT_COMMITTER_DATE="2026-06-20T12:00:00", GIT_AUTHOR_DATE="2026-06-20T12:00:00")
    (a / "state" / "current" / "odds_v1.bin").write_bytes(b"odds 2\n")
    git(a, "add", "-A", env=old)
    git(a, "commit", "-q", "-m", "an old commit", env=old)
    git(a, "push", "-q", "origin", "HEAD:main", env=old)
    r = commit_state(a, env, "main")
    assert r.returncode == 0 and "keepalive" in r.stdout, r.stdout + r.stderr
    shown = git(origin, "show", "--name-only", "--format=", "main", env=env).stdout.split()
    assert shown == ["state/keepalive.txt"]
    # A fresh newest commit: no keepalive.
    r = commit_state(a, env, "main")
    assert r.returncode == 0 and "keepalive" not in r.stdout and "state unchanged" in r.stdout
