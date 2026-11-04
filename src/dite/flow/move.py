"""文件整理模块"""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dite.app.i18n import t

_ANSI_C_ESCAPES = {
    "\\": "\\\\",
    "'": "\\'",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def shell_quote(value: str | Path) -> str:
    """Quote one word for bash so it is never expanded or split.

    Paths and cluster names come from the scanned folder and from LLM output,
    so they must be treated as hostile: inside double quotes bash would still
    run ``$(...)`` and backticks. Plain values use POSIX single quotes. Values
    with control characters use bash ANSI-C quoting (``$'...'``) so a newline
    in a file name cannot end a line, including a comment line.
    """
    text = str(value)
    if not any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        return shlex.quote(text)
    escaped = []
    for ch in text:
        if ch in _ANSI_C_ESCAPES:
            escaped.append(_ANSI_C_ESCAPES[ch])
        elif ord(ch) < 32 or ord(ch) == 127:
            escaped.append(f"\\x{ord(ch):02x}")
        else:
            escaped.append(ch)
    return "$'" + "".join(escaped) + "'"


@dataclass
class FileOperation:
    """文件操作"""

    type: Literal["move", "copy"]
    source: Path
    destination: Path
    status: Literal["pending", "success", "failed"] = "pending"
    error: str | None = None


@dataclass
class OrganizePreview:
    """整理预览"""

    source_folder: Path
    target_folder: Path
    operations: list[FileOperation] = field(default_factory=list)
    cluster_map: dict[str, list[Path]] = field(
        default_factory=dict
    )  # 簇名称 -> 文件列表
    noise_files: list[Path] = field(default_factory=list)  # 未分类的文件
    journal_path: Path | None = None  # 最近一次 execute 写入的移动记录

    @staticmethod
    def _sanitize_cluster_name(cluster_name: str) -> str:
        """清洗簇名称，确保可安全用于目录名。"""
        name = cluster_name.strip()

        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "-")
        for i in range(32):
            name = name.replace(chr(i), "-")

        name = " ".join(name.split()).strip(" .")
        while "--" in name:
            name = name.replace("--", "-")

        if not name or name in {".", ".."}:
            name = "Cluster_Untitled"

        return name[:80]

    def _unique_destination(self, destination: Path) -> Path:
        """为目标路径分配唯一文件名，避免覆盖。"""
        existing_destinations = {op.destination for op in self.operations}
        candidate = destination
        counter = 1

        while candidate in existing_destinations or candidate.exists():
            candidate = destination.with_name(
                f"{destination.stem}_{counter}{destination.suffix}"
            )
            counter += 1

        return candidate

    def add_cluster(self, cluster_name: str, files: list[Path]) -> None:
        """添加一个簇的文件"""
        self._add_folder_operations(self._sanitize_cluster_name(cluster_name), files)

    def add_existing_folder(self, folder_name: str, files: list[Path]) -> None:
        """Move files into an existing subfolder of the target, keeping its name.

        Unlike ``add_cluster`` the name is not sanitized: it already exists on
        disk, and changing it would create a second folder next to it.
        """
        if not folder_name or Path(folder_name).name != folder_name:
            raise ValueError(f"not a direct subfolder name: {folder_name!r}")
        self._add_folder_operations(folder_name, files)

    def _add_folder_operations(self, folder_name: str, files: list[Path]) -> None:
        safe_cluster_name = folder_name
        self.cluster_map.setdefault(safe_cluster_name, [])
        self.cluster_map[safe_cluster_name].extend(files)

        # 生成移动操作
        cluster_folder = self.target_folder / safe_cluster_name
        for file in files:
            destination = self._unique_destination(cluster_folder / file.name)
            self.operations.append(
                FileOperation(
                    type="move",
                    source=file,
                    destination=destination,
                )
            )

    def add_noise(self, files: list[Path]) -> None:
        """添加未分类的文件"""
        self.noise_files = files

    def display(self, console: Console) -> None:
        """显示预览"""
        total_files = len(self.operations) + len(self.noise_files)
        # 头部信息
        header = Panel(
            f"[bold]{t('organize_source_folder')}[/bold] {self.source_folder}\n"
            f"[bold]{t('organize_target_folder')}[/bold] {self.target_folder}\n"
            f"[bold]{t('organize_files_found')}[/bold] {total_files}\n"
            f"[bold]{t('organize_clusters_suggested')}[/bold] {len(self.cluster_map)}",
            title=t("organize_preview_title"),
            border_style="blue",
        )
        console.print(header)
        console.print()

        # 显示每个簇
        for cluster_name, files in self.cluster_map.items():
            console.print(
                f"[bold cyan]【{cluster_name}】[/bold cyan] "
                f"({t('organize_files_count', count=len(files))})"
            )

            table = Table(show_header=False, box=None, padding=(0, 2))
            for f in files[:5]:  # 最多显示 5 个
                table.add_row("  ├──", f.name)
            if len(files) > 5:
                table.add_row("  └──", t("organize_more_files", count=len(files) - 5))
            console.print(table)

            console.print(f"  [dim]{t('organize_move_to', name=cluster_name)}[/dim]")
            console.print()

        # 显示未分类文件
        if self.noise_files:
            console.print(
                f"[bold yellow]【{t('organize_uncategorized')}】[/bold yellow] "
                f"({t('organize_files_count', count=len(self.noise_files))})"
            )
            for f in self.noise_files[:3]:
                console.print(f"  • {f.name}")
            if len(self.noise_files) > 3:
                console.print(
                    f"  {t('organize_more_files', count=len(self.noise_files) - 3)}"
                )
            console.print(f"  [dim]{t('organize_keep_in_place')}[/dim]")

    def generate_script(self, output_path: Path) -> None:
        """
        生成可执行的 shell 脚本

        Args:
            output_path: 脚本输出路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Every path and name below is quoted with shell_quote(): file names
        # and LLM-generated cluster names must never be parsed as shell code.
        lines = [
            "#!/bin/bash",
            f"# {t('organize_script_header')} - {timestamp}",
            f"# Source: {shell_quote(self.source_folder)}",
            f"# Target: {shell_quote(self.target_folder)}",
            "#",
            f"# {t('organize_script_usage')}",
            "#   1. Review this script",
            "#   2. chmod +x organize.sh",
            "#   3. ./organize.sh",
            "",
            "set -e  # Stop on error",
            "",
            f"# {t('organize_script_create_dirs')}",
        ]

        # 创建目录命令
        for cluster_name in self.cluster_map:
            dest_dir = self.target_folder / cluster_name
            lines.append(f"mkdir -p -- {shell_quote(dest_dir)}")

        lines.append("")
        lines.append(f"# {t('organize_script_copy_files')}")

        # 复制命令
        for op in self.operations:
            lines.append(
                f"cp -p -- {shell_quote(op.source)} {shell_quote(op.destination)}"
            )

        lines.append("")
        lines.append(f"# {t('organize_script_verify')}")
        lines.append("# (Uncomment the following lines to delete original files)")

        for op in self.operations:
            lines.append(f"# rm -- {shell_quote(op.source)}")

        lines.append("")
        done_message = t("organize_script_done", count=len(self.operations))
        lines.append(f"echo {shell_quote(done_message)}")
        lines.append(f"echo {shell_quote(t('organize_script_warning'))}")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def execute(
        self,
        dry_run: bool = True,
        *,
        journal_dir: Path | None = None,
    ) -> tuple[int, int]:
        """
        执行整理操作

        Args:
            dry_run: 如果为 True，只预览不执行
            journal_dir: 写入移动记录的目录（通常是 ``目标目录/.dite``）。
                每完成一次移动就追加一行，所以中途中断也能撤销已完成的部分。

        Returns:
            (成功数, 失败数)
        """
        if dry_run:
            return len(self.operations), 0

        success = 0
        failed = 0
        journal = (
            MoveJournal.create(
                journal_dir,
                source_folder=self.source_folder,
                target_folder=self.target_folder,
            )
            if journal_dir is not None
            else None
        )
        self.journal_path = journal.path if journal is not None else None

        try:
            # 创建目录
            for cluster_name in self.cluster_map:
                dest_dir = self.target_folder / cluster_name
                created = _missing_ancestors(dest_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)
                if journal is not None:
                    for directory in created:
                        journal.record_mkdir(directory)

            # 执行操作
            for op in self.operations:
                try:
                    created = _missing_ancestors(op.destination.parent)
                    op.destination.parent.mkdir(parents=True, exist_ok=True)
                    if journal is not None:
                        for directory in created:
                            journal.record_mkdir(directory)

                    # 先复制
                    shutil.copy2(op.source, op.destination)

                    # 验证复制成功
                    if (
                        op.destination.exists()
                        and op.destination.stat().st_size == op.source.stat().st_size
                    ):
                        op.source.unlink()  # 删除原文件
                        op.status = "success"
                        success += 1
                        if journal is not None:
                            journal.record_move(op.source, op.destination)
                    else:
                        op.destination.unlink()  # 删除不完整的副本
                        op.status = "failed"
                        op.error = t("error_copy_failed")
                        failed += 1

                except Exception as e:
                    op.status = "failed"
                    op.error = str(e)
                    failed += 1
        finally:
            if journal is not None:
                journal.close()

        return success, failed


JOURNAL_DIR_NAME = ".dite"
JOURNAL_VERSION = 1
_JOURNAL_PREFIX = "organize-"
_JOURNAL_SUFFIX = ".jsonl"


def _missing_ancestors(directory: Path) -> list[Path]:
    """Return the directories ``mkdir(parents=True)`` would create, top first."""
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    return list(reversed(missing))


def _open_journal(path: Path, mode: str):
    # surrogateescape round-trips file names that are not valid UTF-8.
    return open(path, mode, encoding="utf-8", errors="surrogateescape")


class MoveJournal:
    """Append-only JSONL record of one organize run, used by ``dite undo``."""

    def __init__(self, path: Path, handle) -> None:
        self.path = path
        self._handle = handle

    @classmethod
    def create(
        cls,
        journal_dir: Path,
        *,
        source_folder: Path,
        target_folder: Path,
    ) -> MoveJournal:
        journal_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = journal_dir / f"{_JOURNAL_PREFIX}{stamp}{_JOURNAL_SUFFIX}"
        journal = cls(path, _open_journal(path, "x"))
        journal._write(
            {
                "type": "header",
                "version": JOURNAL_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_folder": str(Path(source_folder).resolve()),
                "target_folder": str(Path(target_folder).resolve()),
            }
        )
        return journal

    def _write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def record_mkdir(self, directory: Path) -> None:
        self._write({"type": "mkdir", "path": str(Path(directory).resolve())})

    def record_move(self, source: Path, destination: Path) -> None:
        self._write(
            {
                "type": "move",
                "source": str(Path(source).absolute()),
                "destination": str(Path(destination).resolve()),
            }
        )

    def close(self) -> None:
        self._handle.close()


@dataclass
class JournalContents:
    """Parsed move journal."""

    path: Path
    header: dict[str, Any]
    moves: list[tuple[Path, Path]]  # (source, destination), in execution order
    created_dirs: list[Path]
    undone: bool


def read_journal(path: Path) -> JournalContents:
    """Load a move journal written by ``OrganizePreview.execute``."""
    header: dict[str, Any] = {}
    moves: list[tuple[Path, Path]] = []
    created_dirs: list[Path] = []
    undone = False
    with _open_journal(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "header":
                header = record
            elif kind == "move":
                moves.append((Path(record["source"]), Path(record["destination"])))
            elif kind == "mkdir":
                created_dirs.append(Path(record["path"]))
            elif kind == "undone":
                undone = True
    if header.get("version") != JOURNAL_VERSION:
        raise ValueError(f"not a DITE move journal: {path}")
    return JournalContents(
        path=path,
        header=header,
        moves=moves,
        created_dirs=created_dirs,
        undone=undone,
    )


def find_latest_journal(target_folder: Path) -> Path | None:
    """Return the newest journal under ``target/.dite`` that is not undone."""
    journal_dir = Path(target_folder) / JOURNAL_DIR_NAME
    if not journal_dir.is_dir():
        return None
    candidates = sorted(
        journal_dir.glob(f"{_JOURNAL_PREFIX}*{_JOURNAL_SUFFIX}"), reverse=True
    )
    for candidate in candidates:
        try:
            if not read_journal(candidate).undone:
                return candidate
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


@dataclass
class UndoResult:
    """Outcome (or plan, for a dry run) of undoing one journal."""

    journal_path: Path
    restored: list[tuple[Path, Path]] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)


def undo_journal(journal_path: Path, *, dry_run: bool = False) -> UndoResult:
    """Move every recorded file back, newest move first.

    A move is skipped when the organized file is gone (``missing``) or when
    something now occupies the original path (``conflict``); nothing is ever
    overwritten. Directories the run created are removed when empty, and the
    journal is marked undone so it cannot be replayed twice.
    """
    contents = read_journal(journal_path)
    if contents.undone:
        raise ValueError(f"journal already undone: {journal_path}")
    result = UndoResult(journal_path=journal_path)
    for source, destination in reversed(contents.moves):
        if not destination.exists():
            result.skipped.append((destination, "missing"))
            continue
        if source.exists():
            result.skipped.append((source, "conflict"))
            continue
        if not dry_run:
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
            except OSError as exc:
                result.skipped.append((destination, str(exc)))
                continue
        result.restored.append((destination, source))

    if dry_run:
        return result

    for directory in sorted(
        set(contents.created_dirs), key=lambda item: len(item.parts), reverse=True
    ):
        try:
            directory.rmdir()  # only succeeds when empty
        except OSError:
            continue
    with _open_journal(journal_path, "a") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "undone",
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "restored": len(result.restored),
                    "skipped": len(result.skipped),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return result
