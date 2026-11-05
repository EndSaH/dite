import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from dite.app.i18n import t
from dite.cache import FileCache
from dite.util import log as log_module


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


def _trace_sqlite(monkeypatch: pytest.MonkeyPatch, *, fail_vacuum: bool) -> list[str]:
    """Record the SQL run on connections opened from now on.

    With fail_vacuum, VACUUM on those connections is interrupted by SQLite,
    which raises sqlite3.OperationalError like a full disk would.
    """
    statements: list[str] = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        last_sql = [""]

        def trace(sql: str) -> None:
            last_sql[0] = sql
            statements.append(sql)

        conn.set_trace_callback(trace)
        if fail_vacuum:
            conn.set_progress_handler(
                lambda: last_sql[0].strip().upper() == "VACUUM", 1
            )
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return statements


def _vacuum_count(statements: list[str]) -> int:
    return sum(1 for sql in statements if sql.strip().upper() == "VACUUM")


def _used_bytes(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return (page_count - freelist_count) * page_size


def test_size_limit_near_limit_does_not_vacuum_on_every_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "cache.db"
    row_bytes = 256 * 1024
    cache = FileCache(db_path=db_path)
    for i in range(20):
        cache.save(
            file_path=tmp_path / f"doc_{i:03d}.txt",
            file_hash=f"hash-{i:03d}",
            file_mtime=1.0,
            content_md="x" * row_bytes,
            model_version="",
            enforce_size_limit=False,
        )
    cache.close()
    # The data sits just under the limit. Rewriting a row with a bigger record
    # allocates the new record before freeing the old one, so each rewrite
    # leaves about one row of free pages and pushes the file over the limit.
    max_size_bytes = db_path.stat().st_size + row_bytes * 3 // 4
    cache = FileCache(db_path=db_path, max_size_gb=max_size_bytes / 1024**3)
    statements = _trace_sqlite(monkeypatch, fail_vacuum=False)

    # Like vectorize: one update_embedding per file, each enforcing the limit.
    for i in range(20):
        cache.update_embedding(
            file_path=tmp_path / f"doc_{i:03d}.txt",
            file_hash=f"hash-{i:03d}",
            embedding=np.ones(1024, dtype=np.float32),
            model_version="embed-v1",
        )
    cache.close()

    # One eviction down to the 90% target leaves room for the remaining
    # writes, instead of a full-database VACUUM after every write.
    assert _vacuum_count(statements) <= 1
    with sqlite3.connect(db_path) as conn:
        rows_with_content = conn.execute(
            "SELECT COUNT(*) FROM file_cache WHERE content_md IS NOT NULL"
        ).fetchone()[0]
    assert rows_with_content >= 15
    assert db_path.stat().st_size <= max_size_bytes


def test_size_limit_survives_failed_vacuum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "cache.db"
    _fill_cache_rows(db_path, tmp_path, 40)
    ids_before = _row_ids(db_path)
    max_size_bytes = 200 * 1024
    target_bytes = int(max_size_bytes * 0.9)
    cache = FileCache(db_path=db_path, max_size_gb=max_size_bytes / 1024**3)
    monkeypatch.setattr(log_module, "_logger", log_module.Logger())
    statements = _trace_sqlite(monkeypatch, fail_vacuum=True)

    # The eviction is committed although VACUUM fails; the failure is a warning.
    deleted = cache.enforce_size_limit()
    assert 0 < deleted < len(ids_before)
    assert _row_ids(db_path) == ids_before[deleted:]
    assert _vacuum_count(statements) == 1
    output = " ".join(capsys.readouterr().out.split())
    assert " ".join(t("cache_vacuum_failed", error="interrupted").split()) in output
    # The file could not shrink, so it stays over the limit from now on.
    assert db_path.stat().st_size > max_size_bytes

    def save(name: str) -> None:
        cache.save(
            file_path=tmp_path / f"{name}.txt",
            file_hash=f"hash-{name}",
            file_mtime=1.0,
            content_md="y" * 8192,
            model_version="embed-v1",
        )

    # Later writes reuse the free pages. While the data stays under the limit
    # they neither evict rows (although the file is over the limit) nor
    # retry VACUUM, and they do not raise.
    assert cache.enforce_size_limit() == 0
    added = 0
    for _ in range(20):
        if _used_bytes(db_path) > target_bytes:
            break
        save(f"new_{added}")
        added += 1
    assert target_bytes < _used_bytes(db_path) <= max_size_bytes
    assert len(_row_ids(db_path)) == len(ids_before) - deleted + added
    assert _vacuum_count(statements) == 1
    assert cache.get_content(tmp_path / "new_0.txt", "hash-new_0") == (
        "y" * 8192,
        None,
    )

    # Once the data goes over the limit again, rows are evicted and VACUUM is
    # retried; it fails again without raising.
    for _ in range(10):
        save(f"new_{added}")
        added += 1
        if _vacuum_count(statements) > 1:
            break
    assert _vacuum_count(statements) == 2
    assert len(_row_ids(db_path)) < len(ids_before) - deleted + added
    assert _used_bytes(db_path) <= target_bytes
    cache.close()


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


def test_clear_vlm_cache_only_invalidates_embeddings_built_on_vlm(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cache.db"
    docx = tmp_path / "plain.docx"
    pdf = tmp_path / "scanned.pdf"
    pdf_copy = tmp_path / "scanned copy.pdf"

    cache = FileCache(db_path=db_path)
    cache.save(
        file_path=docx,
        file_hash="hash-docx",
        file_mtime=1.0,
        content_md="docx content",
        embedding=np.array([1.0, 2.0], dtype=np.float32),
        model_version="embed-v1",
    )
    cache.save(
        file_path=pdf,
        file_hash="hash-pdf",
        file_mtime=1.0,
        content_md="pdf docling content",
        vlm_content="pdf vlm content",
        vlm_version=2,
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        model_version="embed-v1",
    )
    # A duplicate of the PDF reuses its VLM text through the hash lookup, so
    # its embedding is built on VLM content although its own row has none.
    cache.update_embedding(
        file_path=pdf_copy,
        file_hash="hash-pdf",
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        model_version="embed-v1",
    )

    result = cache.clear_vlm_cache()

    docx_embedding = cache.get_embedding(docx, "hash-docx", "embed-v1")
    assert docx_embedding is not None
    assert docx_embedding.tolist() == [1.0, 2.0]
    assert cache.get_embedding(pdf, "hash-pdf", "embed-v1") is None
    assert cache.get_embedding(pdf_copy, "hash-pdf", "embed-v1") is None
    assert result.vlm_entries == 1
    assert result.embeddings == 2
    assert cache.get_vlm_content(pdf, "hash-pdf", required_version=2) is None
    assert cache.get_content(pdf, "hash-pdf") == ("pdf docling content", None)
    assert cache.get_content(docx, "hash-docx") == ("docx content", None)

    stats = cache.get_stats()
    assert stats["with_embedding"] == 1
    assert stats["with_vlm"] == 0
    cache.close()
