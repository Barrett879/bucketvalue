"""Embedded stamps, raising writes, and byte-stable skips for committed files."""
import pandas as pd

from bucketlib import cache


def test_parquet_roundtrip_with_stamp(tmp_path):
    p = tmp_path / "x.parquet"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert cache.write_parquet(df, p, meta={"source": "t"})
    meta = cache.read_parquet_meta(p)
    assert meta["source"] == "t" and meta["fetched_at"] > 0 and meta["content_hash"]
    assert cache.read_parquet(p).equals(df)


def test_if_changed_skips_identical_content_and_keeps_old_stamp(tmp_path):
    p = tmp_path / "x.parquet"
    df = pd.DataFrame({"a": [1, 2]})
    cache.write_parquet(df, p)
    stamp = cache.read_parquet_meta(p)["fetched_at"]
    raw = p.read_bytes()
    assert cache.write_parquet(df.copy(), p, if_changed=True) is False
    assert p.read_bytes() == raw and cache.read_parquet_meta(p)["fetched_at"] == stamp
    assert cache.write_parquet(pd.DataFrame({"a": [1, 3]}), p, if_changed=True) is True


def test_dtype_change_is_a_change(tmp_path):
    assert cache.frame_hash(pd.DataFrame({"a": [1, 2]})) != \
        cache.frame_hash(pd.DataFrame({"a": [1.0, 2.0]}))


def test_json_stamped_if_changed(tmp_path):
    p = tmp_path / "x.json.gz"
    assert cache.write_json(p, {"k": [1, 2]}, stamped=True, gz=True)
    raw = p.read_bytes()
    assert cache.write_json(p, {"k": [1, 2]}, stamped=True, gz=True, if_changed=True) is False
    assert p.read_bytes() == raw
    data, ts = cache.read_json_stamped(p)
    assert data == {"k": [1, 2]} and ts is not None


def test_unstamped_legacy_payload_is_never_fresh(tmp_path):
    p = tmp_path / "legacy.json"
    cache.write_json(p, [1, 2, 3])
    data, ts = cache.read_json_stamped(p)
    assert data == [1, 2, 3] and ts is None
    assert not cache.is_fresh(ts, ttl_seconds=10**9)


def test_write_failure_raises(tmp_path):
    import pytest
    bad = tmp_path / "file_not_dir"
    bad.write_text("x")
    with pytest.raises(Exception):
        cache.write_parquet(pd.DataFrame({"a": [1]}), bad / "child.parquet")
