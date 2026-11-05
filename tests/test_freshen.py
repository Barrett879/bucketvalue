"""freshen.ensure(): every page loads the bucketlib on disk after a deploy.

The trap (2026-11-05, live): Streamlit drops a changed submodule from
sys.modules, but `from bucketlib import store` returns the package's old
attribute, so the new app.py called store.trust_line on an old store. These
tests rebuild that trap on a throwaway package and show ensure() springing it.
"""
from __future__ import annotations

import os
import sys

import pytest

import bucketlib
import freshen


def _bump(path) -> None:
    s = path.stat()
    os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns + 5_000_000_000))


@pytest.fixture
def fake_pkg(tmp_path, monkeypatch):
    root = tmp_path / "fakepkg"
    root.mkdir()
    (root / "__init__.py").write_text(
        "from freshen import code_digest, code_quick\n"
        "CODE_QUICK, CODE_DIGEST = code_quick(), code_digest()\n")
    (root / "mod.py").write_text("X = 1\n")
    monkeypatch.setattr(freshen, "PKG", "fakepkg")
    monkeypatch.setattr(freshen, "ROOT", root)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield root
    for name in [n for n in list(sys.modules) if n == "fakepkg" or n.startswith("fakepkg.")]:
        sys.modules.pop(name, None)


def test_the_package_records_the_code_it_was_imported_from():
    """And on unchanged code ensure() does nothing: the page tests' monkeypatches
    live on these very module objects."""
    assert bucketlib.CODE_QUICK == freshen.code_quick()
    assert bucketlib.CODE_DIGEST == freshen.code_digest()
    before = {n: m for n, m in sys.modules.items() if n.startswith("bucketlib")}
    try:
        assert freshen.ensure() is False
        assert all(sys.modules.get(n) is m for n, m in before.items())
    finally:
        sys.modules.update(before)


def test_a_changed_submodule_loads_where_a_from_import_stayed_stale(fake_pkg):
    from fakepkg import mod
    assert mod.X == 1 and freshen.ensure() is False
    (fake_pkg / "mod.py").write_text("X = 2\nNEW = True\n")
    _bump(fake_pkg / "mod.py")
    sys.modules.pop("fakepkg.mod")                   # what Streamlit's watcher does
    from fakepkg import mod as still                 # the trap
    assert still.X == 1 and not hasattr(still, "NEW")
    assert freshen.ensure() is True
    assert not any(n == "fakepkg" or n.startswith("fakepkg.") for n in sys.modules)
    from fakepkg import mod as now
    assert now.X == 2 and now.NEW
    assert freshen.ensure() is False                 # fresh again


def test_a_touched_file_with_the_same_code_is_not_a_deploy(fake_pkg):
    import fakepkg
    _bump(fake_pkg / "mod.py")                       # a checkout rewrote the mtime
    assert freshen.ensure() is False and sys.modules["fakepkg"] is fakepkg
    assert fakepkg.CODE_QUICK == freshen.code_quick()


def test_a_package_from_before_the_stamp_is_stale(fake_pkg):
    """The live process on 2026-11-05 imported bucketlib before it recorded a
    stamp: the first run after this deploy must purge it."""
    import fakepkg
    del fakepkg.CODE_QUICK, fakepkg.CODE_DIGEST
    assert freshen.ensure() is True and "fakepkg" not in sys.modules


def test_an_unreadable_tree_changes_nothing(fake_pkg, monkeypatch):
    import fakepkg

    def checkout_in_progress():
        raise OSError("a file vanished mid-checkout")
    monkeypatch.setattr(freshen, "code_quick", checkout_in_progress)
    assert freshen.ensure() is False and sys.modules["fakepkg"] is fakepkg


def test_no_package_loaded_is_nothing_to_do(fake_pkg):
    assert "fakepkg" not in sys.modules and freshen.ensure() is False
