"""Load the bucketlib on disk after a deploy, never a stale one.

Streamlit re-reads app.py and pages/*.py from disk on every run, but a module
the scripts imported stays in sys.modules for the life of the process. When a
push changes bucketlib, the file watcher drops only the changed submodule from
sys.modules, and `from bucketlib import store` then returns the PACKAGE's old
`store` attribute without importing anything. New scripts ran against old
modules: on 2026-11-05 the live slate called store.trust_line on a store that
predated it (an AttributeError under the cards for every visitor), and a
library-only change can sit unserved with no error at all.

ensure() runs at the top of every page, before any bucketlib import. The
package records CODE_QUICK (each file's mtime and size) and CODE_DIGEST (a hash
of every source file) when it is imported. When the code on disk differs from
that, ensure() drops the package and every submodule from sys.modules, so the
imports that follow load the code on disk. The common path is one stat() per
source file; the hash runs only when an mtime moved. Module state goes with a
purge (in-process pasted boards, live-feed caches), as it would on a reboot; a
visitor's board comes back from the copy in their browser
(props_ui.board_keeper).

The scripts drop THIS module from sys.modules before importing it, so it is
never stale itself.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PKG = "bucketlib"
ROOT = Path(__file__).resolve().parent / PKG


def _files() -> list[Path]:
    return sorted(ROOT.rglob("*.py"))


def code_quick() -> tuple:
    """((relative path, mtime_ns, size), ...) of every source file."""
    out = []
    for p in _files():
        s = p.stat()
        out.append((p.relative_to(ROOT).as_posix(), s.st_mtime_ns, s.st_size))
    return tuple(out)


def code_digest() -> str:
    """sha256 over every source file's path and bytes."""
    h = hashlib.sha256()
    for p in _files():
        h.update(p.relative_to(ROOT).as_posix().encode() + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()


def ensure() -> bool:
    """Drop a stale bucketlib from sys.modules; True when it did. A package
    imported before it recorded its stamp (older code) counts as stale. An
    unreadable tree (a checkout in progress) changes nothing: the next run
    looks again."""
    pkg = sys.modules.get(PKG)
    if pkg is None:
        return False
    try:
        quick = code_quick()
        if getattr(pkg, "CODE_QUICK", None) == quick:
            return False
        if getattr(pkg, "CODE_DIGEST", None) == code_digest():
            pkg.CODE_QUICK = quick          # touched by a checkout, same code
            return False
    except OSError:
        return False
    for name in [n for n in list(sys.modules) if n == PKG or n.startswith(PKG + ".")]:
        sys.modules.pop(name, None)
    return True
