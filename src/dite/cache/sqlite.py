"""SQLite 缓存实现"""

import contextlib
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 缓存版本号 - 当 VLM 策略变更时递增
VLM_CACHE_VERSION = 2  # v2: 多页提取（10页）


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """将 sqlite3.Row 转换为字典，方便使用 .get() 方法"""
    return dict(zip(row.keys(), row, strict=True))


@dataclass
class CacheEntry:
    """缓存条目"""

    file_path: str
    file_hash: str
    file_mtime: float
    content_md: str | None  # docling/markitdown 原始提取结果
    vlm_content: str | None  # VLM 回退结果（独立缓存）
    vlm_version: int | None  # VLM 策略版本
    embedding: bytes | None
    model_version: str
    created_at: str


@dataclass(frozen=True)
class VlmCacheClearResult:
    """clear_vlm_cache 的结果"""

    vlm_entries: int  # 清除了 VLM 内容的条目数
    embeddings: int  # 因此失效的 embedding 条目数


class FileCache:
    """
    文件缓存管理器

    使用 SQLite 存储文件的提取内容和 Embedding 向量，
    避免重复调用 API。支持基于文件哈希的去重。

    缓存分层：
    1. content_md: 文档转换结果（docling/markitdown）- 稳定，长期缓存
    2. vlm_content: VLM 回退结果 - 带版本号，策略变更时失效
    3. embedding: 向量 - 基于最终内容
    """

    def __init__(
        self,
        db_path: Path | None = None,
        cache_dir: Path | None = None,
        max_size_gb: float | None = None,
    ):
        """
        初始化缓存

        Args:
            db_path: 数据库文件路径，默认使用配置中的路径
            max_size_gb: 缓存上限（GB），None 或 <=0 表示不限制
        """
        if db_path is None:
            directory = cache_dir or (Path.home() / ".cache" / "dite")
            db_path = directory / "cache.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes: int | None = None
        if max_size_gb is not None and max_size_gb > 0:
            self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        self._conn_lock = threading.Lock()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._local = threading.local()
        # 上一次 VACUUM 是否失败（失败后文件缩不回去，会一直超限）
        self._vacuum_failed = False
        self._init_db()

    def _create_conn(self) -> sqlite3.Connection:
        # Connections are still used worker-locally, but disabling the thread check
        # lets the owner object close them centrally after worker threads finish.
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = self._create_conn()
        thread_id = threading.get_ident()
        with self._conn_lock:
            self._connections[thread_id] = conn
        self._local.conn = conn
        return conn

    def rollback(self) -> None:
        """回滚当前线程连接上未结束的事务，释放它持有的锁。

        COMMIT 因 database is locked 失败时 SQLite 不会结束事务，
        这个连接会一直锁着数据库，挡住其他连接的读写。
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.rollback()

    def _init_db(self) -> None:
        """初始化数据库表"""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS file_cache (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                file_mtime REAL NOT NULL,
                content_md TEXT,
                vlm_content TEXT,
                vlm_version INTEGER,
                embedding BLOB,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_path, file_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_file_path ON file_cache(file_path);
            CREATE INDEX IF NOT EXISTS idx_file_hash ON file_cache(file_hash);
        """)

        # 迁移：添加 vlm_content 和 vlm_version 列（如果不存在）
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE file_cache ADD COLUMN vlm_content TEXT")

        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE file_cache ADD COLUMN vlm_version INTEGER")

        conn.commit()

    @staticmethod
    def _used_bytes(conn: sqlite3.Connection) -> int:
        """数据库中实际在用的字节数（不含 freelist 中的空闲页）。

        DELETE 只会把页放进 freelist，文件本身不会变小，所以不能用文件大小
        判断删除是否已经足够。
        """
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
        return (page_count - freelist_count) * page_size

    @staticmethod
    def _oldest_ids_covering(
        conn: sqlite3.Connection, used: int, excess: int
    ) -> list[int]:
        """按淘汰顺序取最旧的若干行，使删除它们估计能释放 excess 字节。

        每行的占用按其数据长度乘以“在用字节数 / 全表数据长度”估算，
        把页内空隙、溢出页尾部等开销也摊进去，避免多删。
        """
        cursor = conn.execute(
            """
            SELECT id, row_bytes, SUM(row_bytes) OVER () AS total_bytes
            FROM (
                SELECT
                    id,
                    created_at,
                    COALESCE(length(CAST(content_md AS BLOB)), 0)
                        + COALESCE(length(CAST(vlm_content AS BLOB)), 0)
                        + COALESCE(length(embedding), 0)
                        + length(CAST(file_path AS BLOB))
                        + length(CAST(file_hash AS BLOB)) AS row_bytes
                FROM file_cache
            )
            ORDER BY created_at ASC, id ASC
            """
        )
        ids: list[int] = []
        covered = 0.0
        try:
            for row_id, row_bytes, total_bytes in cursor:
                scale = used / total_bytes if total_bytes else 1.0
                ids.append(row_id)
                covered += (row_bytes or 0) * scale
                if covered >= excess:
                    break
        finally:
            cursor.close()
        return ids

    def _enforce_size_limit(self) -> int:
        """执行缓存大小限制，返回删除的条目数。"""
        if self.max_size_bytes is None or not self.db_path.exists():
            return 0

        # 文件大小是在用数据量的上界：文件没超限时无需进一步检查
        if self.db_path.stat().st_size <= self.max_size_bytes:
            return 0

        conn = self._get_conn()
        deleted = 0
        target_size = int(self.max_size_bytes * 0.9)
        used = self._used_bytes(conn)

        # 文件超限后把实际数据删到上限的 90% 以下，再 VACUUM，留出余量。
        # 改写一行时 SQLite 先分配新记录、再释放旧的溢出页，文件会多出约一行空闲页；
        # 若只删到上限以下，数据贴着上限时每次写入都会让文件超限并触发一次全库 VACUUM。
        # VACUUM 失败过时文件缩不回去、会一直超限，这时只在实际数据超过上限时才删，
        # 否则每次写入都要淘汰一行。
        evict_above = self.max_size_bytes if self._vacuum_failed else target_size
        if used > evict_above:
            while used > target_size:
                ids = self._oldest_ids_covering(conn, used, used - target_size)
                if not ids:
                    break
                cursor = conn.executemany(
                    "DELETE FROM file_cache WHERE id = ?",
                    [(row_id,) for row_id in ids],
                )
                conn.commit()
                batch = cursor.rowcount if cursor.rowcount is not None else 0
                if batch <= 0:
                    break
                deleted += batch
                used = self._used_bytes(conn)

        conn.commit()
        # VACUUM 失败过时，只有这次删了行才重试，避免每次写入都去重写整个数据库
        if deleted > 0 or not self._vacuum_failed:
            self._vacuum()

        return deleted

    def _vacuum(self) -> None:
        """执行 VACUUM，把 freelist 中的页还给文件系统，文件才会真正变小。

        失败（例如磁盘剩余空间不够 VACUUM 复制数据库，或数据库被占用）时只警告：
        之前的删除已经提交，SQLite 会复用空闲页，缓存仍然可用。
        """
        try:
            # 使用独立连接执行 VACUUM，避免事务上下文冲突
            with sqlite3.connect(str(self.db_path)) as vacuum_conn:
                vacuum_conn.execute("VACUUM")
        except sqlite3.OperationalError as exc:
            self._vacuum_failed = True
            # 延迟导入：导入 dite.app 会加载 dite.cache，放在模块顶层会循环导入
            from dite.app.i18n import t
            from dite.util.log import get_logger

            get_logger().warning(t("cache_vacuum_failed", error=exc))
            return
        self._vacuum_failed = False

    def enforce_size_limit(self) -> int:
        """公开执行缓存大小治理。"""
        return self._enforce_size_limit()

    def get_by_path(self, file_path: Path) -> CacheEntry | None:
        """
        根据文件路径获取缓存

        Args:
            file_path: 文件路径

        Returns:
            缓存条目，如果不存在返回 None
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT *
            FROM file_cache
            WHERE file_path = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (str(file_path),),
        )
        row = cursor.fetchone()

        if row:
            data = _row_to_dict(row)
            return CacheEntry(
                file_path=data["file_path"],
                file_hash=data["file_hash"],
                file_mtime=data["file_mtime"],
                content_md=data["content_md"],
                vlm_content=data.get("vlm_content"),
                vlm_version=data.get("vlm_version"),
                embedding=data["embedding"],
                model_version=data["model_version"],
                created_at=data["created_at"],
            )
        return None

    def get_by_hash(self, file_hash: str) -> CacheEntry | None:
        """
        根据文件哈希获取缓存（用于去重）

        Args:
            file_hash: 文件的 SHA256 哈希

        Returns:
            缓存条目，如果不存在返回 None
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT *
            FROM file_cache
            WHERE file_hash = ?
              AND content_md IS NOT NULL
            LIMIT 1
            """,
            (file_hash,),
        )
        row = cursor.fetchone()

        if row:
            data = _row_to_dict(row)
            return CacheEntry(
                file_path=data["file_path"],
                file_hash=data["file_hash"],
                file_mtime=data["file_mtime"],
                content_md=data["content_md"],
                vlm_content=data.get("vlm_content"),
                vlm_version=data.get("vlm_version"),
                embedding=data["embedding"],
                model_version=data["model_version"],
                created_at=data["created_at"],
            )
        return None

    def update_embedding(
        self,
        file_path: Path,
        file_hash: str,
        embedding: np.ndarray,
        model_version: str,
        enforce_size_limit: bool = True,
    ) -> None:
        """更新向量缓存，不改写文档转换或 VLM 内容。"""
        conn = self._get_conn()
        file_mtime = file_path.stat().st_mtime if file_path.exists() else 0.0
        embedding_bytes = embedding.astype(np.float32).tobytes()

        conn.execute(
            """
            INSERT INTO file_cache (
                file_path, file_hash, file_mtime, content_md, vlm_content,
                vlm_version, embedding, model_version
            )
            VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?)
            ON CONFLICT(file_path, file_hash) DO UPDATE SET
                file_mtime = excluded.file_mtime,
                embedding = excluded.embedding,
                model_version = excluded.model_version
            """,
            (str(file_path), file_hash, file_mtime, embedding_bytes, model_version),
        )
        conn.commit()
        if enforce_size_limit:
            self._enforce_size_limit()

    def save(
        self,
        file_path: Path,
        file_hash: str,
        file_mtime: float,
        content_md: str,
        vlm_content: str | None = None,
        vlm_version: int | None = None,
        embedding: np.ndarray | None = None,
        model_version: str = "",
        enforce_size_limit: bool = True,
    ) -> None:
        """
        保存缓存条目

        Args:
            file_path: 文件路径
            file_hash: 文件哈希
            file_mtime: 文件修改时间
            content_md: 提取的 Markdown 内容（docling/markitdown 原始结果）
            vlm_content: VLM 回退内容
            vlm_version: VLM 策略版本
            embedding: Embedding 向量
            model_version: 使用的模型版本
        """
        conn = self._get_conn()

        embedding_bytes = None
        if embedding is not None:
            embedding_bytes = embedding.astype(np.float32).tobytes()

        conn.execute(
            """
            INSERT INTO file_cache
            (
                file_path, file_hash, file_mtime, content_md, vlm_content,
                vlm_version, embedding, model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path, file_hash) DO UPDATE SET
                file_mtime = excluded.file_mtime,
                content_md = excluded.content_md,
                vlm_content = COALESCE(excluded.vlm_content, file_cache.vlm_content),
                vlm_version = COALESCE(excluded.vlm_version, file_cache.vlm_version),
                embedding = COALESCE(excluded.embedding, file_cache.embedding),
                model_version = excluded.model_version
            """,
            (
                str(file_path),
                file_hash,
                file_mtime,
                content_md,
                vlm_content,
                vlm_version,
                embedding_bytes,
                model_version,
            ),
        )
        conn.commit()
        if enforce_size_limit:
            self._enforce_size_limit()

    def update_vlm_content(
        self,
        file_path: Path,
        file_hash: str,
        vlm_content: str,
        vlm_version: int,
        enforce_size_limit: bool = True,
    ) -> None:
        """
        更新 VLM 回退内容（不影响 docling 缓存）

        Args:
            file_path: 文件路径
            file_hash: 文件哈希
            vlm_content: VLM 回退内容
            vlm_version: VLM 策略版本
        """
        conn = self._get_conn()
        file_mtime = file_path.stat().st_mtime if file_path.exists() else 0.0
        conn.execute(
            """
            INSERT INTO file_cache (
                file_path, file_hash, file_mtime, content_md, vlm_content,
                vlm_version, embedding, model_version
            )
            VALUES (?, ?, ?, NULL, ?, ?, NULL, '')
            ON CONFLICT(file_path, file_hash) DO UPDATE SET
                vlm_content = excluded.vlm_content,
                vlm_version = excluded.vlm_version,
                embedding = NULL
            """,
            (str(file_path), file_hash, file_mtime, vlm_content, vlm_version),
        )
        conn.commit()
        if enforce_size_limit:
            self._enforce_size_limit()

    def get_embedding(
        self,
        file_path: Path,
        file_hash: str,
        required_model_version: str | None = None,
    ) -> np.ndarray | None:
        """
        获取缓存的 Embedding（支持哈希去重）。

        Args:
            file_path: 文件路径
            file_hash: 文件哈希
            required_model_version: 期望的 embedding 模型版本；不匹配则视为未命中

        Returns:
            Embedding 向量，如果缓存未命中返回 None
        """
        # 1. 先检查路径缓存
        entry = self.get_by_path(file_path)
        if (
            entry
            and entry.file_hash == file_hash
            and entry.embedding
            and (
                required_model_version is None
                or entry.model_version == required_model_version
            )
        ):
            return np.frombuffer(entry.embedding, dtype=np.float32)

        # 2. 哈希去重：检查是否有相同内容的其他文件
        entry = self.get_by_hash(file_hash)
        if (
            entry
            and entry.embedding
            and (
                required_model_version is None
                or entry.model_version == required_model_version
            )
        ):
            return np.frombuffer(entry.embedding, dtype=np.float32)

        return None

    def get_content(
        self, file_path: Path, file_hash: str
    ) -> tuple[str | None, str | None]:
        """
        获取缓存的原始文档转换内容（docling/markitdown）

        Args:
            file_path: 文件路径
            file_hash: 文件哈希

        Returns:
            (Markdown 内容, 来源文件路径) - 如果是去重复用，
            返回原始文件路径；否则返回 None
            如果缓存未命中，返回 (None, None)
        """
        # 1. 先检查路径缓存
        entry = self.get_by_path(file_path)
        if entry and entry.file_hash == file_hash and entry.content_md:
            return entry.content_md, None  # 自身缓存命中

        # 2. 哈希去重：检查是否有相同内容的其他文件
        entry = self.get_by_hash(file_hash)
        if entry and entry.content_md:
            return entry.content_md, entry.file_path  # 复用其他文件的缓存

        return None, None

    def get_vlm_content(
        self, file_path: Path, file_hash: str, required_version: int = VLM_CACHE_VERSION
    ) -> str | None:
        """
        获取缓存的 VLM 回退内容（版本敏感）

        Args:
            file_path: 文件路径
            file_hash: 文件哈希
            required_version: 要求的 VLM 版本

        Returns:
            VLM 内容，如果版本不匹配或未缓存返回 None
        """
        entry = self.get_by_path(file_path)
        if (
            entry
            and entry.file_hash == file_hash
            and entry.vlm_content
            and entry.vlm_version == required_version
        ):
            return entry.vlm_content

        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT vlm_content FROM file_cache
            WHERE file_hash = ?
              AND vlm_content IS NOT NULL
              AND vlm_version = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (file_hash, required_version),
        )
        row = cursor.fetchone()
        if row:
            return row["vlm_content"]

        return None

    def clear(self) -> int:
        """
        清空所有缓存

        Returns:
            删除的条目数
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM file_cache")
        count = cursor.fetchone()[0]

        conn.execute("DELETE FROM file_cache")
        conn.commit()

        return count

    def clear_vlm_cache(self) -> VlmCacheClearResult:
        """
        仅清除 VLM 缓存（保留 docling 结果）

        只有带 VLM 内容的文件，其 embedding 可能是基于 VLM 文本算出的，
        所以只让这些文件哈希下的 embedding 失效；按哈希而不是按行处理，
        是因为同哈希的其他路径会复用这份 VLM 内容和 embedding。
        其他文件的 embedding 保持不变。

        Returns:
            清除的 VLM 条目数与失效的 embedding 条目数
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            UPDATE file_cache
            SET embedding = NULL
            WHERE embedding IS NOT NULL
              AND file_hash IN (
                  SELECT file_hash FROM file_cache WHERE vlm_content IS NOT NULL
              )
            """
        )
        embeddings = cursor.rowcount
        cursor = conn.execute(
            """
            UPDATE file_cache
            SET vlm_content = NULL,
                vlm_version = NULL
            WHERE vlm_content IS NOT NULL
            """
        )
        vlm_entries = cursor.rowcount
        conn.commit()

        return VlmCacheClearResult(vlm_entries=vlm_entries, embeddings=embeddings)

    def get_stats(
        self, required_embedding_version: str | None = None
    ) -> dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含统计信息的字典
        """
        conn = self._get_conn()

        cursor = conn.execute("SELECT COUNT(*) FROM file_cache")
        total_entries = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM file_cache WHERE embedding IS NOT NULL"
        )
        with_embedding = cursor.fetchone()[0]

        if required_embedding_version is None:
            current_embeddings = with_embedding
            stale_embeddings = 0
        else:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM file_cache
                WHERE embedding IS NOT NULL AND model_version = ?
                """,
                (required_embedding_version,),
            )
            current_embeddings = cursor.fetchone()[0]
            stale_embeddings = with_embedding - current_embeddings

        cursor = conn.execute(
            "SELECT COUNT(*) FROM file_cache WHERE vlm_content IS NOT NULL"
        )
        with_vlm = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(DISTINCT file_hash) FROM file_cache")
        unique_hashes = cursor.fetchone()[0]

        # 数据库文件大小
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_entries": total_entries,
            "with_embedding": with_embedding,
            "current_embeddings": current_embeddings,
            "stale_embeddings": stale_embeddings,
            "current_embedding_version": required_embedding_version or "-",
            "with_vlm": with_vlm,
            "unique_hashes": unique_hashes,
            "db_size_mb": db_size / (1024 * 1024),
            "db_path": str(self.db_path),
            "vlm_cache_version": VLM_CACHE_VERSION,
        }

    def close(self) -> None:
        """关闭数据库连接"""
        with self._conn_lock:
            connections = list(self._connections.values())
            self._connections = {}

        for conn in connections:
            conn.close()

        self._local = threading.local()
