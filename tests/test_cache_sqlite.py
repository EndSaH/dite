import sqlite3
import threading
from pathlib import Path

import numpy as np

from dite.cache import FileCache


def test_update_vlm_content_upsert_inserts_when_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.update_vlm_content(
        file_path=file_path,
        file_hash="hash-1",
        vlm_content="vlm result",
        vlm_version=2,
    )

    assert (
        cache.get_vlm_content(file_path, "hash-1", required_version=2) == "vlm result"
    )
    cache.close()


def test_get_vlm_content_reuses_same_hash_across_paths(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    source_path = tmp_path / "source.pdf"
    alias_path = tmp_path / "alias.pdf"
    source_path.write_text("same pdf bytes", encoding="utf-8")
    alias_path.write_text("same pdf bytes", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.update_vlm_content(
        file_path=source_path,
        file_hash="same-hash",
        vlm_content="vlm result",
        vlm_version=2,
    )

    assert cache.get_vlm_content(alias_path, "same-hash", required_version=2) == (
        "vlm result"
    )
    assert cache.get_vlm_content(alias_path, "same-hash", required_version=3) is None
    cache.close()


def test_update_vlm_content_upsert_updates_existing_row(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.save(
        file_path=file_path,
        file_hash="hash-2",
        file_mtime=file_path.stat().st_mtime,
        content_md="docling content",
        embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        model_version="embed-v1",
    )

    cache.update_vlm_content(
        file_path=file_path,
        file_hash="hash-2",
        vlm_content="vlm better content",
        vlm_version=2,
    )

    entry = cache.get_by_path(file_path)
    assert entry is not None
    assert entry.content_md == "docling content"
    assert entry.vlm_content == "vlm better content"
    assert entry.vlm_version == 2
    assert entry.embedding is None
    cache.close()


def test_save_embedding_does_not_drop_existing_vlm_cache(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.update_vlm_content(
        file_path=file_path,
        file_hash="hash-3",
        vlm_content="vlm kept",
        vlm_version=2,
    )

    cache.save(
        file_path=file_path,
        file_hash="hash-3",
        file_mtime=file_path.stat().st_mtime,
        content_md="docling content",
        embedding=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        model_version="embed-v1",
    )

    assert cache.get_vlm_content(file_path, "hash-3", required_version=2) == "vlm kept"
    cache.close()


def test_update_embedding_does_not_overwrite_content_layers(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.save(
        file_path=file_path,
        file_hash="hash-embedding",
        file_mtime=file_path.stat().st_mtime,
        content_md="full docling content",
        model_version="embed-v1",
    )
    cache.update_vlm_content(
        file_path=file_path,
        file_hash="hash-embedding",
        vlm_content="vlm fallback content",
        vlm_version=2,
    )

    cache.update_embedding(
        file_path=file_path,
        file_hash="hash-embedding",
        embedding=np.array([7.0, 8.0, 9.0], dtype=np.float32),
        model_version="embed-v2",
    )

    entry = cache.get_by_path(file_path)
    assert entry is not None
    assert entry.content_md == "full docling content"
    assert entry.vlm_content == "vlm fallback content"
    assert entry.vlm_version == 2
    assert (
        cache.get_embedding(
            file_path,
            "hash-embedding",
            required_model_version="embed-v2",
        )
        is not None
    )
    cache.close()


def test_cache_upsert_keeps_single_row_per_file_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    file_path = tmp_path / "doc.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.update_vlm_content(
        file_path=file_path,
        file_hash="hash-4",
        vlm_content="vlm-v1",
        vlm_version=1,
    )
    cache.update_vlm_content(
        file_path=file_path,
        file_hash="hash-4",
        vlm_content="vlm-v2",
        vlm_version=2,
    )
    cache.save(
        file_path=file_path,
        file_hash="hash-4",
        file_mtime=file_path.stat().st_mtime,
        content_md="docling content",
        model_version="embed-v1",
    )
    cache.close()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT
                COUNT(*),
                MIN(vlm_content),
                MAX(vlm_content),
                MIN(vlm_version),
                MAX(vlm_version)
            FROM file_cache
            WHERE file_path = ? AND file_hash = ?
            """,
            (str(file_path), "hash-4"),
        )
        row = cursor.fetchone()

    assert row is not None
    assert row[0] == 1
    assert row[1] == "vlm-v2"
    assert row[2] == "vlm-v2"
    assert row[3] == 2
    assert row[4] == 2


def test_cache_enforces_size_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    max_size_gb = 0.00005  # ~52KB
    cache = FileCache(db_path=db_path, max_size_gb=max_size_gb)

    for i in range(120):
        file_path = tmp_path / f"doc_{i}.txt"
        file_path.write_text("dummy", encoding="utf-8")
        cache.save(
            file_path=file_path,
            file_hash=f"hash-{i}",
            file_mtime=file_path.stat().st_mtime,
            content_md="x" * 2048,
            model_version="embed-v1",
        )

    stats = cache.get_stats()
    cache.close()

    assert stats["total_entries"] < 120
    assert db_path.stat().st_size <= int(max_size_gb * 1024 * 1024 * 1024 * 1.2)


def _fill_cache_rows(db_path: Path, tmp_path: Path, count: int) -> None:
    cache = FileCache(db_path=db_path)
    for i in range(count):
        cache.save(
            file_path=tmp_path / f"doc_{i:03d}.txt",
            file_hash=f"hash-{i:03d}",
            file_mtime=1.0,
            content_md="x" * 8192,
            embedding=np.zeros(256, dtype=np.float32),
            model_version="embed-v1",
            enforce_size_limit=False,
        )
    cache.close()


def _row_ids(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        return [row[0] for row in conn.execute("SELECT id FROM file_cache ORDER BY id")]


def test_size_limit_evicts_only_oldest_rows_until_under_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    _fill_cache_rows(db_path, tmp_path, 40)
    ids_before = _row_ids(db_path)
    max_size_bytes = 200 * 1024
    assert db_path.stat().st_size > 1.5 * max_size_bytes

    cache = FileCache(db_path=db_path, max_size_gb=max_size_bytes / 1024**3)
    deleted = cache.enforce_size_limit()
    cache.close()

    ids_after = _row_ids(db_path)
    # Only part of the cache is evicted, and it is the oldest part.
    assert 0 < deleted < len(ids_before)
    assert len(ids_after) == len(ids_before) - deleted
    assert ids_after == ids_before[deleted:]
    # Eviction stops near the 90% target instead of emptying the table:
    # each row takes roughly 9-10 KB, so well over 10 rows fit under 180 KB.
    assert len(ids_after) >= 10
    assert db_path.stat().st_size <= max_size_bytes


def test_size_limit_keeps_rows_when_only_free_pages_exceed_limit(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cache.db"
    _fill_cache_rows(db_path, tmp_path, 40)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM file_cache WHERE id > 5")
    ids_before = _row_ids(db_path)
    max_size_bytes = 200 * 1024
    # Deleted rows only move pages to the freelist; the file stays large.
    assert db_path.stat().st_size > 1.5 * max_size_bytes

    cache = FileCache(db_path=db_path, max_size_gb=max_size_bytes / 1024**3)
    deleted = cache.enforce_size_limit()
    cache.close()

    assert deleted == 0
    assert _row_ids(db_path) == ids_before
    assert db_path.stat().st_size <= max_size_bytes


def test_get_embedding_misses_when_model_version_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    file_path = tmp_path / "doc.txt"
    file_path.write_text("dummy", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.save(
        file_path=file_path,
        file_hash="hash-model",
        file_mtime=file_path.stat().st_mtime,
        content_md="docling content",
        embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        model_version="embed-v1",
    )

    assert (
        cache.get_embedding(
            file_path,
            "hash-model",
            required_model_version="embed-v1",
        )
        is not None
    )
    assert (
        cache.get_embedding(
            file_path,
            "hash-model",
            required_model_version="embed-v2",
        )
        is None
    )
    cache.close()


def test_get_stats_counts_current_and_stale_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.db"
    current = tmp_path / "current.txt"
    stale = tmp_path / "stale.txt"
    current.write_text("current", encoding="utf-8")
    stale.write_text("stale", encoding="utf-8")

    cache = FileCache(db_path=db_path)
    cache.save(
        file_path=current,
        file_hash="hash-current",
        file_mtime=current.stat().st_mtime,
        content_md="current content",
        embedding=np.array([1.0, 2.0], dtype=np.float32),
        model_version="embed-v1|input=current",
    )
    cache.save(
        file_path=stale,
        file_hash="hash-stale",
        file_mtime=stale.stat().st_mtime,
        content_md="stale content",
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        model_version="embed-v1",
    )

    stats = cache.get_stats(required_embedding_version="embed-v1|input=current")

    assert stats["with_embedding"] == 2
    assert stats["current_embeddings"] == 1
    assert stats["stale_embeddings"] == 1
    assert stats["current_embedding_version"] == "embed-v1|input=current"
    cache.close()


def test_file_cache_uses_distinct_connections_per_thread_and_closes_all(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cache.db"
    cache = FileCache(db_path=db_path)
    connections: list[sqlite3.Connection] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(3)

    def worker() -> None:
        try:
            conn = cache._get_conn()
            connections.append(conn)
            barrier.wait()
        except BaseException as exc:  # pragma: no cover - test helper
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()

    main_conn = cache._get_conn()
    connections.append(main_conn)
    barrier.wait()

    for thread in threads:
        thread.join()

    assert not errors
    assert len({id(conn) for conn in connections}) == 3

    cache.close()

    for conn in connections:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            continue
        raise AssertionError("connection should be closed by FileCache.close()")
