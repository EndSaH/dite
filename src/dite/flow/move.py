"""文件整理模块"""

from __future__ import annotations

import errno
import filecmp
import json
import os
import re
import shlex
import shutil
import signal
import stat
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
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


def _name_key(name: str) -> str:
    """Comparison key for names, matching case-insensitive volumes.

    APFS and HFS+ (the macOS default) ignore case and treat the NFC and NFD
    spellings of a name as the same entry, so two names that are equal under
    this key are one file or folder there.
    """
    return unicodedata.normalize("NFC", name).casefold()


def _path_key(path: Path) -> str:
    return _name_key(os.fspath(path))


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
    # _name_key() of every folder name in cluster_map.
    _folder_keys: set[str] = field(default_factory=set, init=False, repr=False)
    # _path_key() of every planned destination.
    _planned_keys: set[str] = field(default_factory=set, init=False, repr=False)
    # directory -> _name_key() of the names already in it, listed once.
    _existing_names: dict[Path, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )

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
        """为目标路径分配唯一文件名，避免覆盖。

        Names are compared the way case-insensitive volumes (the macOS
        default) compare them, against the other planned destinations and
        against what is already in the folder: ``README.md`` and
        ``readme.md`` are one file there. ``execute`` checks again, because
        the folder can change between the preview and the move.
        """
        existing = self._names_in(destination.parent)
        candidate = destination
        counter = 1

        while (
            _path_key(candidate) in self._planned_keys
            or _name_key(candidate.name) in existing
        ):
            candidate = destination.with_name(
                f"{destination.stem}_{counter}{destination.suffix}"
            )
            counter += 1

        self._planned_keys.add(_path_key(candidate))
        return candidate

    def _names_in(self, directory: Path) -> set[str]:
        names = self._existing_names.get(directory)
        if names is None:
            try:
                names = {_name_key(name) for name in os.listdir(directory)}
            except OSError:
                names = set()  # not created yet, or unreadable
            self._existing_names[directory] = names
        return names

    def add_cluster(self, cluster_name: str, files: list[Path]) -> None:
        """添加一个簇的文件

        Every call is a separate cluster and gets a folder of its own. When an
        earlier cluster already uses the (sanitized) name, compared the way
        case-insensitive volumes compare names, this one becomes ``Name_2``,
        ``Name_3``, ...: clusters are never merged just because their names
        are equal.
        """
        name = self._sanitize_cluster_name(cluster_name)
        folder_name = name
        counter = 2
        while _name_key(folder_name) in self._folder_keys:
            folder_name = f"{name}_{counter}"
            counter += 1
        self._add_folder_operations(folder_name, files)

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
        self._folder_keys.add(_name_key(safe_cluster_name))
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
            # cp silently replaces a file that is already there (one that
            # appeared after the preview, say), so stop before each copy.
            "ensure_absent() {",
            '  if [ -e "$1" ] || [ -L "$1" ]; then',
            "    printf '%s %s\\n' "
            + shell_quote(t("organize_script_exists"))
            + ' "$1" >&2',
            "    exit 1",
            "  fi",
            "}",
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
            lines.append(f"ensure_absent {shell_quote(op.destination)}")
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
                每个文件先追加一行记录并落盘、再删除源文件，
                所以中途中断也能撤销已完成的部分。

        Returns:
            (成功数, 失败数)

        Nothing that exists is ever replaced: a destination that is taken
        by the time the file is moved (or that the file system treats as the
        same name) gets the next free ``_N`` name, and ``op.destination`` and
        the journal hold the name actually used. Each move is journaled
        before its source is deleted, and a failed move leaves no copy
        behind. A full or read-only target stops the remaining moves.
        SIGTERM and SIGHUP stop the run the way Ctrl-C does, cleanup
        included (see ``_interrupt_on_termination``).
        """
        if dry_run:
            return len(self.operations), 0
        with _interrupt_on_termination():
            return self._execute(journal_dir)

    def _execute(self, journal_dir: Path | None) -> tuple[int, int]:
        success = 0
        failed = 0
        # A run with nothing to move leaves no journal: an empty journal would
        # otherwise become "the latest" and hide the real one from undo.
        journal = (
            MoveJournal.create(
                journal_dir,
                source_folder=self.source_folder,
                target_folder=self.target_folder,
            )
            if journal_dir is not None and self.operations
            else None
        )
        self.journal_path = journal.path if journal is not None else None

        try:
            # 创建目录
            for cluster_name in self.cluster_map:
                dest_dir = self.target_folder / cluster_name
                _make_folder(dest_dir, journal)
                _remove_partials(dest_dir)

            # Every planned name, so a late rename cannot take another
            # operation's destination.
            taken = {_path_key(op.destination) for op in self.operations}
            abort_error: str | None = None

            # 执行操作
            for op in self.operations:
                if abort_error is not None:
                    op.status = "failed"
                    op.error = abort_error
                    failed += 1
                    continue
                try:
                    _make_folder(op.destination.parent, journal)
                    _move_file(op, journal, taken)
                    op.status = "success"
                    success += 1

                except Exception as e:
                    op.status = "failed"
                    op.error = str(e)
                    failed += 1
                    if isinstance(e, OSError) and e.errno in _ABORT_ERRNOS:
                        abort_error = str(e)
        finally:
            if journal is not None:
                journal.close()

        return success, failed


JOURNAL_DIR_NAME = ".dite"
JOURNAL_VERSION = 1
_JOURNAL_PREFIX = "organize-"
_JOURNAL_SUFFIX = ".jsonl"

# Errors after which every remaining move would fail the same way.
_ABORT_ERRNOS = frozenset(
    code
    for code in (errno.ENOSPC, errno.EROFS, getattr(errno, "EDQUOT", None))
    if code is not None
)
_MAX_RENAME_ATTEMPTS = 10_000

# Signals whose default action ends the process on the spot, running no
# cleanup at all: closing the terminal, shutdown, `timeout`, an IDE's stop.
_TERMINATION_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK")
    if hasattr(signal, name)
)


_held_back = 0  # depth of _Uninterrupted sections
_interrupt_pending = False


def _raise_interrupt(signum: int, frame: Any) -> None:
    """Signal handler: ``KeyboardInterrupt``, after any ``_Uninterrupted``."""
    global _interrupt_pending
    if _held_back:
        _interrupt_pending = True
        return
    raise KeyboardInterrupt


class _Uninterrupted:
    """Hold an interrupt back for a few statements that must run together.

    Only interrupts that come through ``_raise_interrupt`` wait.
    """

    def __enter__(self) -> None:
        global _held_back
        _held_back += 1

    def __exit__(self, *exc_info: object) -> None:
        global _held_back, _interrupt_pending
        _held_back -= 1
        if not _held_back and _interrupt_pending:
            _interrupt_pending = False
            raise KeyboardInterrupt


@contextmanager
def _interrupt_on_termination() -> Iterator[None]:
    """Make SIGTERM and SIGHUP raise ``KeyboardInterrupt`` while files move.

    Otherwise they end the process at once, and a copy cut short or a name
    not yet journaled stays behind; as a ``KeyboardInterrupt`` they go
    through the same cleanup as Ctrl-C. Ctrl-C goes through the same
    handler, so that ``_Uninterrupted`` can hold any of them back. Only
    signals still at their default action (Python's own handler, for
    SIGINT) are changed: an ignored one (``nohup``) or a handler the program
    installed is left alone, and so is everything outside the main thread,
    the only one that may set handlers.
    """
    global _interrupt_pending
    _interrupt_pending = False
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, *_TERMINATION_SIGNALS):
        default = (
            signal.default_int_handler if signum == signal.SIGINT else signal.SIG_DFL
        )
        try:
            if signal.getsignal(signum) == default:
                previous[signum] = signal.signal(signum, _raise_interrupt)
        except (OSError, ValueError):
            continue
    try:
        yield
    finally:
        for signum, handler in previous.items():
            with suppress(OSError, ValueError):
                signal.signal(signum, handler)


def _move_file(op: FileOperation, journal: MoveJournal | None, taken: set[str]) -> None:
    """Move ``op.source`` to ``op.destination`` (or the next free name).

    The new name is created first and journaled, and only then is the source
    deleted, so an error or an interrupt (Ctrl-C, SIGTERM, SIGHUP) leaves at
    worst a second name for a file the journal knows about, never a moved
    file that undo cannot find. The new name is removed again only while the
    source is known to be there: when creating or journaling it is
    interrupted or fails, or when deleting the source fails and the source
    still exists. A kill -9 or a power loss after the new name is made and
    before it is journaled leaves it behind unrecorded, a complete second
    name next to the untouched source; a copy that such a kill cuts short
    leaves only a hidden ``.dite-partial-*`` file (see ``_copy_new``).
    """
    new_name = _NewName()
    move_id: int | None = None
    try:
        op.destination = _create_unique(op.source, op.destination, taken, new_name)
        if journal is not None:
            move_id = journal.record_move(op.source, op.destination)
    except BaseException:
        # The source is untouched, so a name made for it is only a duplicate.
        new_name.discard(op.source)
        raise
    try:
        os.unlink(op.source)
    except OSError:
        # Only OSError is handled: a Ctrl-C can arrive once the unlink has
        # already removed the source, and then the new name is the only one
        # left. Even after an OSError the source may be gone, so look first.
        source_exists = _lexists_or_none(op.source)
        if source_exists is False:
            return  # deleted after all (a retried network unlink, say)
        if source_exists:
            try:
                os.unlink(op.destination)
            except OSError:
                pass  # both names stay; undo drops the recorded duplicate
            else:
                if journal is not None and move_id is not None:
                    with suppress(OSError):
                        journal.record_cancel(move_id)
        raise


def _lexists_or_none(path: Path) -> bool | None:
    """Whether ``path`` exists (even as a broken symlink); ``None`` if unknown."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return True


@dataclass
class _NewName:
    """The name ``_create_unique`` is making, so an interrupt can undo it."""

    path: Path | None = None
    created: bool = False
    # (st_dev, st_ino) of the files _copy_new made for it.
    copies: set[tuple[int, int]] = field(default_factory=set)

    def discard(self, source: Path) -> None:
        """Remove the name if it was made for ``source``.

        An interrupt raised just as the name is made leaves ``created``
        unset, but the name is then a hard link to ``source`` itself or one
        of the copies made for it.
        """
        if self.path is None:
            return
        if (
            self.created
            or _same_inode(source, self.path)
            or _identity(self.path) in self.copies
        ):
            with suppress(OSError):
                os.unlink(self.path)


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _same_inode(first: Path, second: Path) -> bool:
    try:
        a, b = os.lstat(first), os.lstat(second)
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _create_unique(
    source: Path, destination: Path, taken: set[str], new_name: _NewName
) -> Path:
    """Create ``destination`` (or ``stem_N.ext``) as a new name for ``source``."""
    candidate = destination
    counter = 1
    for _ in range(_MAX_RENAME_ATTEMPTS):
        new_name.path, new_name.created = candidate, False
        new_name.copies.clear()  # only files that exist for this attempt
        try:
            _create_new(source, candidate, new_name.copies)
        except FileExistsError:
            new_name.path = None  # someone else's file
        else:
            new_name.created = True
            return candidate
        while True:
            candidate = destination.with_name(
                f"{destination.stem}_{counter}{destination.suffix}"
            )
            counter += 1
            if _path_key(candidate) not in taken:
                taken.add(_path_key(candidate))
                break
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))


def _create_new(
    source: Path, destination: Path, copies: set[tuple[int, int]] | None = None
) -> None:
    """Give the content of ``source`` the new name ``destination``.

    Raises ``FileExistsError`` instead of replacing anything, judged by the
    file system itself (so case-insensitively on APFS). A hard link is used
    where possible: it is atomic and keeps the file as it is, inode and
    metadata included. Across file systems, on file systems without hard
    links (FAT, exFAT, some network shares) and for symlinks, whose target's
    content is moved as before, the file is copied instead; the identity of
    each file the copy makes is added to ``copies``.
    """
    if not source.is_symlink():
        try:
            os.link(source, destination)
            return
        except FileExistsError:
            raise
        except OSError:
            pass
    _copy_new(source, destination, set() if copies is None else copies)


_PARTIAL_PREFIX = ".dite-partial-"
_PARTIAL_NAME = re.compile(re.escape(_PARTIAL_PREFIX) + "[0-9a-f]{32}")


def _copy_new(source: Path, destination: Path, copies: set[tuple[int, int]]) -> None:
    """Copy ``source`` to the new name ``destination``.

    The data goes to a hidden ``.dite-partial-<random>`` file next to
    ``destination`` first, and only a complete copy that is on disk gets
    the real name, so a copy cut short (a full disk, an interrupt, even a
    kill -9 or a power loss) never leaves a truncated file under a real name,
    one that undo or the next organize would take for the user's. A partial
    file a kill leaves behind is removed by ``_remove_partials``.
    """
    if os.path.lexists(destination):  # spare a copy that could not be named
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))
    partial = destination.with_name(f"{_PARTIAL_PREFIX}{uuid.uuid4().hex}")
    try:
        # "x" never truncates an existing file; it raises FileExistsError.
        # A failed flush (a full disk) fails again in close(), which still
        # closes the file.
        with open(source, "rb") as fsrc, open(partial, "xb") as fdst:
            copies.add(_fd_identity(fdst.fileno()))
            shutil.copyfileobj(fsrc, fdst)
            fdst.flush()
            os.fsync(fdst.fileno())
            if os.fstat(fdst.fileno()).st_size != os.fstat(fsrc.fileno()).st_size:
                raise OSError(t("error_copy_failed"))
        shutil.copystat(source, partial)
        _name_copy(partial, destination, copies)
        # The new name must be on disk before the source can be deleted.
        _fsync_directory(destination.parent)
    except BaseException:
        if _identity(destination) in copies:  # named just before the error
            with suppress(OSError):
                os.unlink(destination)
        raise
    finally:
        with suppress(OSError):
            os.unlink(partial)  # the extra name; already gone after a rename


def _name_copy(partial: Path, destination: Path, copies: set[tuple[int, int]]) -> None:
    """Give the finished copy ``partial`` the name ``destination``.

    Never replaces a file: a hard link fails when the name is taken. Where
    there are no hard links, the name is claimed with an empty file created
    with O_EXCL, and the copy is renamed over that file of our own.
    """
    try:
        os.link(partial, destination)
        return
    except FileExistsError:
        raise
    except OSError:
        pass  # FAT, exFAT, some network shares
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    # Until its identity is known, an empty file left by an interrupt could
    # not be told from someone else's.
    with _Uninterrupted():
        fd = os.open(destination, flags, 0o666)
        try:
            copies.add(_fd_identity(fd))
        finally:
            os.close(fd)
    os.replace(partial, destination)


def _fd_identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def _fsync_directory(directory: Path) -> None:
    """Flush the entries of ``directory`` to disk, where that is possible."""
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return  # Windows cannot open a directory
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _remove_partials(directory: Path) -> None:
    """Delete the partial copies a killed run left in ``directory``.

    Only regular files with the names ``_copy_new`` makes are touched.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if _PARTIAL_NAME.fullmatch(name):
            path = directory / name
            with suppress(OSError):
                if stat.S_ISREG(os.lstat(path).st_mode):
                    os.unlink(path)


def _make_folder(directory: Path, journal: MoveJournal | None) -> None:
    """Create ``directory`` and its missing parents, journaling each first.

    As with a move, the record comes before the change: an interrupt or a
    full disk can leave a record of a folder that was never made (undo's
    rmdir then finds nothing), but never a folder undo does not know about.
    """
    if journal is not None:
        for missing in _missing_ancestors(directory):
            journal.record_mkdir(missing)
    directory.mkdir(parents=True, exist_ok=True)


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


_JOURNAL_OPEN_FLAGS = os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0)


class MoveJournal:
    """Append-only JSONL record of one organize run, used by ``dite undo``.

    Records are appended whole or not at all: each goes straight to the file
    (no buffer that a later flush could fail on) and is fsynced, and a write
    that fails part way (a full disk) is cut off again.
    Only a crash in the middle of a write can leave a partial last line,
    which ``read_journal`` ignores and ``reopen`` removes.
    """

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd = fd
        self._size = os.fstat(fd).st_size  # bytes of complete records
        self._next_move_id = 0

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
        flags = _JOURNAL_OPEN_FLAGS | os.O_CREAT | os.O_EXCL
        journal: MoveJournal | None = None
        try:
            with _Uninterrupted():  # an interrupt waits until the file is ours
                journal = cls(path, os.open(path, flags, 0o666))
            journal._write(
                {
                    "type": "header",
                    "version": JOURNAL_VERSION,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source_folder": str(Path(source_folder).resolve()),
                    "target_folder": str(Path(target_folder).resolve()),
                }
            )
        except BaseException:
            # Nothing has moved yet; an empty journal would only be clutter.
            if journal is not None:
                journal.close()
                with suppress(OSError):
                    os.unlink(path)
            raise
        return journal

    @classmethod
    def reopen(cls, path: Path) -> MoveJournal:
        """Open an existing journal to append undo records to it."""
        journal = cls(path, os.open(path, _JOURNAL_OPEN_FLAGS))
        try:
            journal._drop_partial_last_line()
        except BaseException:
            journal.close()
            raise
        return journal

    def _drop_partial_last_line(self) -> None:
        """Cut off a last record that a crash left without its newline.

        Otherwise the next record would be appended to that line and the
        journal could no longer be read.
        """
        with open(self.path, "rb") as handle:
            data = handle.read()
        complete = data.rfind(b"\n") + 1
        if complete < len(data):
            os.ftruncate(self._fd, complete)
        self._size = complete

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8", "surrogateescape")
        remaining = memoryview(encoded)
        try:
            while remaining:
                # A full disk shows as a short write, then a failing one.
                written = os.write(self._fd, remaining)
                if written <= 0:
                    raise OSError(errno.EIO, os.strerror(errno.EIO), str(self.path))
                remaining = remaining[written:]
            # On disk before the change it vouches for (deleting the source).
            os.fsync(self._fd)
        except BaseException:
            # Never leave half a record behind: the next one would be
            # appended to it, and the journal could no longer be read.
            with suppress(OSError):
                os.ftruncate(self._fd, self._size)
            raise
        self._size += len(encoded)

    def record_mkdir(self, directory: Path) -> None:
        self._write({"type": "mkdir", "path": str(Path(directory).resolve())})

    def record_move(self, source: Path, destination: Path) -> int:
        """Record a move before its source is deleted; returns its id.

        The size and mtime of the new file let undo check that what it
        moves back is still the file that was moved.
        """
        move_id = self._next_move_id
        self._next_move_id += 1
        info = os.lstat(destination)
        self._write(
            {
                "type": "move",
                "id": move_id,
                "source": str(Path(source).absolute()),
                "destination": str(Path(destination).resolve()),
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
        )
        return move_id

    def record_cancel(self, move_id: int) -> None:
        """The move was rolled back because its source could not be deleted."""
        self._write_done({"type": "cancel", "id": move_id})

    def record_restored(self, move_id: int) -> None:
        """Undo moved this file back; a retried undo leaves it alone."""
        self._write_done({"type": "restored", "id": move_id})

    def _write_done(self, record: dict[str, Any]) -> None:
        """Write a record of a change that is already made.

        An interrupt waits until the record is written: ``_write`` would
        otherwise cut it off, and a retried undo would look for the file
        where it no longer is.
        """
        with _Uninterrupted():
            self._write(record)

    def record_undone(self, *, restored: int, skipped: int) -> None:
        self._write(
            {
                "type": "undone",
                "at": datetime.now().isoformat(timespec="seconds"),
                "restored": restored,
                "skipped": skipped,
            }
        )

    def close(self) -> None:
        # Records are written and fsynced one by one: closing cannot lose any.
        fd, self._fd = self._fd, -1
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)


@dataclass(frozen=True)
class JournalMove:
    """One recorded move. Older journals carry no size or mtime."""

    id: int
    source: Path
    destination: Path
    size: int | None = None
    mtime_ns: int | None = None


@dataclass
class JournalContents:
    """Parsed move journal."""

    path: Path
    header: dict[str, Any]
    moves: list[tuple[Path, Path]]  # (source, destination), in execution order
    created_dirs: list[Path]
    undone: bool
    entries: list[JournalMove] = field(default_factory=list)  # same order
    restored_ids: set[int] = field(default_factory=set)  # moved back already

    @property
    def pending(self) -> list[JournalMove]:
        """Moves an undo still has to reverse."""
        return [entry for entry in self.entries if entry.id not in self.restored_ids]


def read_journal(path: Path) -> JournalContents:
    """Load a move journal written by ``OrganizePreview.execute``."""
    header: dict[str, Any] = {}
    moves: dict[int, JournalMove] = {}
    move_count = 0
    restored_ids: set[int] = set()
    created_dirs: list[Path] = []
    undone = False
    with _open_journal(path, "r") as handle:
        for line in handle:
            if not line.endswith("\n"):
                # A record counts once its newline is written. Only a crash
                # in the middle of a write leaves a last line without one,
                # and nothing was done on the strength of that record.
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "header":
                header = record
            elif kind == "move":
                # Older journals have no ids: a move's id is its position.
                move_id = record.get("id", move_count)
                move_count += 1
                moves[move_id] = JournalMove(
                    id=move_id,
                    source=Path(record["source"]),
                    destination=Path(record["destination"]),
                    size=record.get("size"),
                    mtime_ns=record.get("mtime_ns"),
                )
            elif kind == "cancel":
                moves.pop(record.get("id"), None)
            elif kind == "restored":
                restored_ids.add(record.get("id"))
            elif kind == "mkdir":
                created_dirs.append(Path(record["path"]))
            elif kind == "undone":
                undone = True
    if header.get("version") != JOURNAL_VERSION:
        raise ValueError(f"not a DITE move journal: {path}")
    return JournalContents(
        path=path,
        header=header,
        moves=[(entry.source, entry.destination) for entry in moves.values()],
        created_dirs=created_dirs,
        undone=undone,
        entries=list(moves.values()),
        restored_ids=restored_ids,
    )


def find_latest_journal(
    target_folder: Path, *, passed_over: list[Path] | None = None
) -> Path | None:
    """Return the newest journal under ``target/.dite`` with moves to undo.

    Journals that are already undone, or that recorded no move (every copy
    failed), are skipped. So is a journal none of whose remaining moves can
    be undone right now, because the organized files are missing or were
    changed; it is only read, never marked, so it is found again once its
    files are back (a renamed folder renamed back, a drive plugged in
    again). Such journals are appended to ``passed_over``, newest first. An
    undo that stopped after moving every file back is found too, so that a
    retry removes the run's folders and marks the journal undone.
    """
    journal_dir = Path(target_folder) / JOURNAL_DIR_NAME
    if not journal_dir.is_dir():
        return None
    candidates = sorted(
        journal_dir.glob(f"{_JOURNAL_PREFIX}*{_JOURNAL_SUFFIX}"), reverse=True
    )
    for candidate in candidates:
        try:
            contents = read_journal(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if contents.undone or not contents.entries:
            continue
        if _has_undo_work(contents):
            return candidate
        if passed_over is not None:
            passed_over.append(candidate)
    return None


def _has_undo_work(contents: JournalContents) -> bool:
    """Whether undo could move a file back now or has a skip to retry.

    With nothing pending, an undo stopped after moving every file back, and
    only its cleanup is left.
    """
    if not contents.pending:
        return True
    for entry in contents.pending:
        skip = _undo_move(entry, dry_run=True)
        if skip is None or skip[1] not in _ABSENT_SKIP_REASONS:
            return True
    return False


# Skips with nothing for the user to fix at the original location: the moved
# file is not there, or is not the file that was moved. Undo cannot do
# anything about them unless the file comes back.
_ABSENT_SKIP_REASONS = frozenset({"missing", "changed"})


@dataclass
class UndoResult:
    """Outcome (or plan, for a dry run) of undoing one journal."""

    journal_path: Path
    restored: list[tuple[Path, Path]] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def retryable(self) -> list[tuple[Path, str]]:
        """Skips that running undo again can still fix (conflicts, errors)."""
        return [item for item in self.skipped if item[1] not in _ABSENT_SKIP_REASONS]


def undo_journal(journal_path: Path, *, dry_run: bool = False) -> UndoResult:
    """Move every recorded file back, newest move first.

    Nothing is ever overwritten. A move is skipped when the organized file is
    gone (``missing``), when it is no longer the file that was moved there
    (``changed``: its size or mtime differ from the journal), or when
    something now occupies the original path (``conflict``). Directories the
    run created are removed when empty.

    Each file moved back is journaled right away, and the journal is marked
    undone only once every move is undone: after any skip it stays open, and
    undoing it again retries just the moves that are left. A file that an
    interrupted undo moved back without journaling it is recognized by its
    size and mtime (see ``_already_back``). Partial copies that a killed
    organize or undo left next to these files are removed. SIGTERM and
    SIGHUP stop the run the way Ctrl-C does.
    """
    if dry_run:
        return _undo_journal(journal_path, dry_run=True)
    with _interrupt_on_termination():
        return _undo_journal(journal_path, dry_run=False)


def _undo_journal(journal_path: Path, *, dry_run: bool) -> UndoResult:
    contents = read_journal(journal_path)
    if contents.undone:
        raise ValueError(f"journal already undone: {journal_path}")
    result = UndoResult(journal_path=journal_path)
    journal = None if dry_run else MoveJournal.reopen(journal_path)
    try:
        for entry in reversed(contents.pending):
            skip = _undo_move(entry, dry_run=dry_run)
            if skip is not None:
                result.skipped.append(skip)
                continue
            result.restored.append((entry.destination, entry.source))
            if journal is not None:
                journal.record_restored(entry.id)

        if journal is None:
            return result

        folders = set(contents.created_dirs)
        for entry in contents.pending:
            folders.update((entry.source.parent, entry.destination.parent))
        for directory in folders:
            _remove_partials(directory)
        for directory in sorted(
            set(contents.created_dirs), key=lambda item: len(item.parts), reverse=True
        ):
            try:
                directory.rmdir()  # only succeeds when empty
            except OSError:
                continue
        if not result.skipped:
            journal.record_undone(
                restored=len(result.restored), skipped=len(result.skipped)
            )
    finally:
        if journal is not None:
            journal.close()
    return result


def _undo_move(entry: JournalMove, *, dry_run: bool) -> tuple[Path, str] | None:
    """Move one file back; returns ``(path, reason)`` when it is skipped."""
    source, destination = entry.source, entry.destination
    try:
        info = os.lstat(destination)
    except FileNotFoundError:
        if _already_back(entry):
            return None  # only the record of the move back is missing
        return destination, "missing"
    except OSError as exc:
        return destination, str(exc)

    if os.path.lexists(source):
        if not _is_leftover_duplicate(source, destination):
            return source, "conflict"
        # The move (or an earlier undo) stopped between creating the new
        # name and deleting the old one: dropping the organized name is
        # all that is left to do.
        if not dry_run:
            try:
                os.unlink(destination)
            except OSError as exc:
                return destination, str(exc)
        return None

    if (entry.size is not None and info.st_size != entry.size) or (
        entry.mtime_ns is not None and info.st_mtime_ns != entry.mtime_ns
    ):
        return destination, "changed"
    if dry_run:
        return None

    try:
        source.parent.mkdir(parents=True, exist_ok=True)
        _create_new(destination, source)
    except FileExistsError:
        return source, "conflict"
    except OSError as exc:
        return destination, str(exc)
    try:
        os.unlink(destination)
    except OSError as exc:
        # As in _move_file: drop the name just made only while the organized
        # one is known to be there, or the file may be left under neither.
        destination_exists = _lexists_or_none(destination)
        if destination_exists is False:
            return None  # deleted after all: the file is back
        if destination_exists:
            with suppress(OSError):
                os.unlink(source)
        return destination, str(exc)
    return None


def _already_back(entry: JournalMove) -> bool:
    """Whether the moved file is at its original path, although not journaled.

    An undo stopped between moving a file back and journaling that (by an
    interrupt or a full disk), or a move rolled back without its ``cancel``
    record, leaves the organized name gone and the file at its source. It is
    told by the size and mtime the journal has for it; journals without
    them still report the file missing.
    """
    if entry.size is None or entry.mtime_ns is None:
        return False
    try:
        info = os.lstat(entry.source)
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_size == entry.size
        and info.st_mtime_ns == entry.mtime_ns
    )


def _is_leftover_duplicate(source: Path, destination: Path) -> bool:
    """Whether ``source`` is ``destination`` under a second name, or a copy.

    Both must be regular files (never a symlink, which would point at the
    file about to be removed), and either the same inode reached through two
    hard links or byte-identical copies.
    """
    try:
        src = os.lstat(source)
        dst = os.lstat(destination)
    except OSError:
        return False
    if not (stat.S_ISREG(src.st_mode) and stat.S_ISREG(dst.st_mode)):
        return False
    if (src.st_dev, src.st_ino) == (dst.st_dev, dst.st_ino):
        # Two names for one file; with a single link they are one name.
        return dst.st_nlink > 1
    if src.st_size != dst.st_size:
        return False
    try:
        return filecmp.cmp(source, destination, shallow=False)
    except OSError:
        return False
