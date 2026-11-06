"""Disk-cache toolkit for BucketValue.

Ported from GridironValue's gridlib/cache.py (itself from DiamondValue and
HoopsValue) with two deliberate changes, both lessons paid for elsewhere:

1. FRESHNESS NEVER READS FILE MTIME. Committed cache files are re-checked-out
   on every GitHub Actions run and every Streamlit Cloud deploy, which resets
   every mtime to "seconds ago", so an mtime TTL silently never fires.
   DiamondValue's evening lineup refresh did nothing for months because of it.
   Every cached payload here carries its own `fetched_at`, embedded in the JSON
   envelope or in the parquet schema metadata, and freshness is judged from
   that stamp alone.

2. WRITES RAISE. The sibling helpers caught every exception and logged a
   warning, so a failed write looked like success to the caller and a bot run
   could commit the OLD file as if refreshed. A write failure is now an error;
   callers that genuinely want best-effort wrap it themselves.

Unchanged invariants: CACHE_DIR uses a persistent /data disk when one is
mounted (Render), else the repo-local ./cache; every write is atomic (tmp
sibling + os.replace); READ_ONLY declares a shared deployment.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .util import now_utc

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("bucketvalue")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[bucketvalue] %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.environ.get("BUCKETVALUE_LOG", "WARNING").upper())

# ── CACHE_DIR ────────────────────────────────────────────────────────────────
_RENDER_DISK = Path("/data/cache")
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _RENDER_DISK if _RENDER_DISK.parent.exists() else REPO_ROOT / "cache"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

# ── Read-only deployments ────────────────────────────────────────────────────
# READ_ONLY means SHARED, not "no disk". On Streamlit Community Cloud the cache
# directory is the repo checkout, served to every visitor by one process. A
# visitor's paste lives in the in-process board store (props) and in that
# visitor's own browser (props_ui.board_keeper), never in this directory, and
# anything that would write the shared record (frozen projections, clears)
# must refuse.
# The record changes only through a commit made locally. Set
# BUCKETVALUE_READONLY=1 in the host's secrets; leave it unset locally.
READ_ONLY = os.environ.get("BUCKETVALUE_READONLY", "").strip().lower() in (
    "1", "true", "yes", "on")


def seed_disk_cache_from_repo() -> None:
    """Gap-fill a persistent disk from the committed repo cache, once.

    Never overwrites, so a CHANGED committed file never reaches an
    already-seeded disk: bump the filename version (_vN) to ship one.
    """
    repo_cache = REPO_ROOT / "cache"
    if CACHE_DIR == repo_cache or not repo_cache.is_dir():
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        have = set(os.listdir(CACHE_DIR))
        for src in repo_cache.iterdir():
            if src.is_file() and src.name not in have:
                shutil.copy2(src, CACHE_DIR / src.name)
    except Exception as e:  # noqa: BLE001 - seeding is best-effort
        logger.warning("disk-cache seed skipped: %s", e)


seed_disk_cache_from_repo()


def dc_path(name: str) -> Path:
    """Path inside CACHE_DIR; subdirectories in `name` are created."""
    p = CACHE_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Embedded fetch stamps ────────────────────────────────────────────────────
def stamp_now() -> float:
    return now_utc().timestamp()


def is_fresh(fetched_at: float | None, ttl_seconds: float, now: float | None = None) -> bool:
    """True when a payload stamped `fetched_at` (epoch seconds) is younger than
    `ttl_seconds`. A missing stamp is never fresh: a legacy file without one
    must be refetched, not trusted forever."""
    if fetched_at is None:
        return False
    t = time.time() if now is None else now
    return (t - float(fetched_at)) < ttl_seconds


# ── Atomic writers (raise on failure) ────────────────────────────────────────
def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp{os.getpid()}-{threading.get_ident()}")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


_PQ_META_KEY = b"bucketvalue"


def _jsonable(o):
    """json.dumps fallback for metadata: numpy scalars, Timestamps, dates."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except (TypeError, ValueError):
            pass
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a frame: values, column names and dtypes, not the index
    and not any metadata. Two writes of the same data hash the same even though
    their embedded fetch stamps differ."""
    h = hashlib.sha256()
    h.update(json.dumps([[str(c), str(t)] for c, t in df.dtypes.items()]).encode())
    if len(df):
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path, meta: dict | None = None,
                  if_changed: bool = False) -> bool:
    """Atomic parquet write. `meta` (JSON-able) rides in the schema metadata;
    `fetched_at` and `content_hash` are added automatically.

    if_changed=True skips the write when the existing file holds the same
    content, keeping its old stamp. The bot uses this for every COMMITTED file:
    the embedded stamp makes every write a new blob, so without it each cron run
    would commit, push and redeploy unchanged data (DiamondValue's bot wrote
    388 MB of blobs in 65 days that way). Returns True when it wrote.
    """
    digest = frame_hash(df)
    if if_changed and path.exists() and read_parquet_meta(path).get("content_hash") == digest:
        return False
    meta = dict(meta or {})
    meta.setdefault("fetched_at", stamp_now())
    meta["content_hash"] = digest
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata(
        {**existing, _PQ_META_KEY: json.dumps(meta, default=_jsonable).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(path)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return True


def read_parquet(path: Path) -> pd.DataFrame | None:
    """None when absent. A corrupt file raises: it is a bug to investigate,
    not a cache miss to paper over."""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def read_parquet_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    md = pq.read_schema(path).metadata or {}
    raw = md.get(_PQ_META_KEY)
    return json.loads(raw) if raw else {}


def json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def write_json(path: Path, obj, stamped: bool = False, gz: bool = False,
               if_changed: bool = False) -> bool:
    """Atomic JSON write. `stamped=True` wraps as
    {"fetched_at", "content_hash", "data"}. if_changed=True (stamped only)
    skips the write when the stored content hash matches; see write_parquet.
    Returns True when it wrote."""
    if stamped:
        digest = json_hash(obj)
        if if_changed and path.exists():
            old = read_json(path)
            if isinstance(old, dict) and old.get("content_hash") == digest:
                return False
        payload = {"fetched_at": stamp_now(), "content_hash": digest, "data": obj}
    else:
        payload = obj
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode()
    # mtime=0 keeps gzip output byte-stable for identical content.
    _atomic_write_bytes(path, gzip.compress(raw, mtime=0) if gz else raw)
    return True


def read_json(path: Path):
    """Parsed JSON, or None when absent. Transparently handles .gz files."""
    if not path.exists():
        return None
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def read_json_stamped(path: Path) -> tuple[object, float | None]:
    """(data, fetched_at) for a stamped envelope; (None, None) when absent.
    An unstamped legacy payload returns (payload, None), i.e. never fresh."""
    obj = read_json(path)
    if obj is None:
        return None, None
    if isinstance(obj, dict) and "fetched_at" in obj and "data" in obj:
        return obj["data"], obj["fetched_at"]
    return obj, None
