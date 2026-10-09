"""Only link pages that exist in git, and Accuracy is local only."""
import subprocess
from pathlib import Path

from bucketlib.theme import _NAV_PAGES

REPO = Path(__file__).resolve().parent.parent


def _tracked_or_staged_pages():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "pages"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    return {Path(p).stem for p in out if p.endswith(".py")}


def test_every_nav_page_exists_in_git():
    pages = _tracked_or_staged_pages()
    for label, url in _NAV_PAGES:
        assert url.strip("/") in pages, f"nav links {url} but pages/{url.strip('/')}.py is not in git"


def test_accuracy_is_never_linked_and_is_ignored():
    assert all(label != "Accuracy" for label, _ in _NAV_PAGES)
    r = subprocess.run(["git", "check-ignore", "-q", "pages/Accuracy.py"], cwd=REPO)
    assert r.returncode == 0, "pages/Accuracy.py must be gitignored (local only)"
