"""Safety of organize/assign file moves and of ``dite undo``.

The tests run on case-sensitive Linux file systems, so case-insensitive
volumes (the macOS default) are simulated with names that differ only in
case, with NFC/NFD spellings, or with a symlink that makes two spellings of a
folder name point at the same directory.
"""

from __future__ import annotations

import errno
import io
import os
import shutil
import signal
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import dite.flow.move as move_module
from dite.app.cli import app
from dite.flow.move import (
    JOURNAL_DIR_NAME,
    FileOperation,
    OrganizePreview,
    find_latest_journal,
    read_journal,
    undo_journal,
)
from dite.report.move import apply_clustering_to_preview


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _execute(preview: OrganizePreview) -> tuple[int, int]:
    return preview.execute(
        dry_run=False, journal_dir=preview.target_folder / JOURNAL_DIR_NAME
    )


def _organized_files(target: Path) -> list[Path]:
    return sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and JOURNAL_DIR_NAME not in path.parts
    )


def _organize(tmp_path: Path, names: list[str]) -> tuple[OrganizePreview, list[Path]]:
    files = [
        _write(tmp_path / "src" / name, f"payload {index}")
        for index, name in enumerate(names)
    ]
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", files)
    assert _execute(preview) == (len(files), 0)
    return preview, files


# ASSIGN-1: never overwrite, compare names the way macOS does.


def test_plan_treats_names_differing_only_in_case_as_one_file(tmp_path: Path) -> None:
    first = _write(tmp_path / "src" / "a" / "README.md", "a")
    second = _write(tmp_path / "src" / "b" / "readme.md", "b")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )

    preview.add_cluster("Docs", [first, second])

    names = [op.destination.name for op in preview.operations]
    assert names == ["README.md", "readme_1.md"]


def test_plan_treats_nfc_and_nfd_spellings_as_one_file(tmp_path: Path) -> None:
    nfc = unicodedata.normalize("NFC", "café.pdf")
    nfd = unicodedata.normalize("NFD", "café.pdf")
    assert nfc != nfd
    first = _write(tmp_path / "src" / "a" / nfc, "a")
    second = _write(tmp_path / "src" / "b" / nfd, "b")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )

    preview.add_cluster("Docs", [first, second])

    first_name, second_name = (op.destination.name for op in preview.operations)
    assert first_name == nfc
    assert unicodedata.normalize("NFC", second_name) != nfc
    assert second_name.endswith("_1.pdf")


def test_plan_avoids_existing_files_that_differ_only_in_case(tmp_path: Path) -> None:
    target = tmp_path / "out"
    _write(target / "Docs" / "Report.PDF", "existing")
    _write(target / "Docs" / unicodedata.normalize("NFD", "café.txt"), "existing")
    report = _write(tmp_path / "src" / "report.pdf", "new")
    cafe = _write(tmp_path / "src" / unicodedata.normalize("NFC", "café.txt"), "new")
    preview = OrganizePreview(source_folder=tmp_path / "src", target_folder=target)

    preview.add_cluster("Docs", [report, cafe])

    names = [op.destination.name for op in preview.operations]
    assert names == ["report_1.pdf", unicodedata.normalize("NFC", "café_1.txt")]


def test_execute_never_overwrites_a_file_created_after_the_preview(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "original")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    planned = preview.operations[0].destination
    # Something (a sync client, another run) writes there before --execute.
    _write(planned, "someone else's file")

    assert _execute(preview) == (1, 0)

    assert planned.read_text(encoding="utf-8") == "someone else's file"
    actual = preview.operations[0].destination
    assert actual != planned
    assert actual.read_text(encoding="utf-8") == "original"
    contents = read_journal(preview.journal_path)
    assert [destination for _, destination in contents.moves] == [actual.resolve()]

    result = undo_journal(preview.journal_path)

    assert result.skipped == []
    assert source.read_text(encoding="utf-8") == "original"
    assert planned.read_text(encoding="utf-8") == "someone else's file"


def test_execute_never_overwrites_on_a_case_insensitive_volume(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path / "src" / "a" / "report.pdf", "AAAA")
    second = _write(tmp_path / "src" / "b" / "report.pdf", "BBBB")
    target = tmp_path / "out"
    (target / "Finance").mkdir(parents=True)
    # On APFS "finance" and "Finance" are the same folder.
    (target / "finance").symlink_to(target / "Finance", target_is_directory=True)
    preview = OrganizePreview(source_folder=tmp_path / "src", target_folder=target)
    preview.operations = [
        FileOperation("move", first, target / "Finance" / "report.pdf"),
        FileOperation("move", second, target / "finance" / "report.pdf"),
    ]

    assert _execute(preview) == (2, 0)

    stored = sorted(
        path.read_text(encoding="utf-8") for path in (target / "Finance").iterdir()
    )
    assert stored == ["AAAA", "BBBB"]

    result = undo_journal(preview.journal_path)

    assert result.skipped == []
    assert first.read_text(encoding="utf-8") == "AAAA"
    assert second.read_text(encoding="utf-8") == "BBBB"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_script_refuses_to_overwrite_existing_destination(tmp_path: Path) -> None:
    sources = [
        _write(tmp_path / "src" / "a.txt", "payload a"),
        _write(tmp_path / "src" / "b.txt", "payload b"),
    ]
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", sources)
    script_path = tmp_path / "organize.sh"
    preview.generate_script(script_path)
    # The user uncomments the delete section; then a file appears at one of
    # the destinations before the script runs.
    script = script_path.read_text(encoding="utf-8")
    script_path.write_text(script.replace("\n# rm -- ", "\nrm -- "), encoding="utf-8")
    occupied = preview.operations[1].destination
    _write(occupied, "someone else's file")

    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert occupied.read_text(encoding="utf-8") == "someone else's file"
    assert all(source.exists() for source in sources)


# ASSIGN-3: distinct clusters never share a folder.


def test_clusters_with_the_same_name_get_separate_folders(tmp_path: Path) -> None:
    files = [
        _write(tmp_path / "src" / f"f{index}.txt", str(index)) for index in range(8)
    ]
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )

    apply_clustering_to_preview(
        preview=preview,
        files=files,
        labels=[0, 0, 1, 1, 2, 2, 3, 3],
        cluster_names={
            0: "Financial Reports",
            1: "Financial Reports",
            2: "financial reports",
            3: "AI/ML",
        },
    )

    assert list(preview.cluster_map) == [
        "Financial Reports",
        "Financial Reports_2",
        "financial reports_3",
        "AI-ML",
    ]
    assert all(len(members) == 2 for members in preview.cluster_map.values())
    folders = [op.destination.parent.name for op in preview.operations]
    assert len(set(folders)) == 4

    console = Console(record=True, width=200)
    preview.display(console)
    shown = console.export_text()
    assert "Financial Reports_2" in shown
    assert "financial reports_3" in shown

    assert _execute(preview) == (8, 0)
    target_dirs = sorted(
        path.name for path in preview.target_folder.iterdir() if path.is_dir()
    )
    assert target_dirs == sorted([JOURNAL_DIR_NAME, *folders[::2]])


def test_sanitized_cluster_names_that_collide_get_separate_folders(
    tmp_path: Path,
) -> None:
    files = [
        _write(tmp_path / "src" / f"f{index}.txt", str(index)) for index in range(3)
    ]
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )

    preview.add_cluster("AI/ML", [files[0]])
    preview.add_cluster("AI-ML", [files[1]])
    preview.add_cluster("AI-ML_2", [files[2]])

    assert list(preview.cluster_map) == ["AI-ML", "AI-ML_2", "AI-ML_2_2"]


# ASSIGN-6: no unrecorded copies, journal before delete, rename in place.


def test_execute_moves_without_copying_on_the_same_file_system(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    inode = source.stat().st_ino
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])

    assert _execute(preview) == (1, 0)

    assert preview.operations[0].destination.stat().st_ino == inode
    assert not source.exists()


def _fail_unlink_of(monkeypatch, victim: Path) -> None:
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(victim):
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "remove", unlink)


def test_failed_source_delete_leaves_no_unrecorded_copy(
    tmp_path: Path, monkeypatch
) -> None:
    locked = _write(tmp_path / "src" / "locked" / "a.txt", "locked")
    free = _write(tmp_path / "src" / "b.txt", "free")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [locked, free])
    _fail_unlink_of(monkeypatch, locked)

    assert _execute(preview) == (1, 1)
    monkeypatch.undo()

    assert locked.read_text(encoding="utf-8") == "locked"
    assert _organized_files(preview.target_folder) == [
        preview.operations[1].destination
    ]
    contents = read_journal(preview.journal_path)
    assert [source for source, _ in contents.moves] == [free]

    result = undo_journal(preview.journal_path)

    assert result.skipped == []
    assert free.read_text(encoding="utf-8") == "free"
    assert _organized_files(preview.target_folder) == []


def test_copy_failure_leaves_no_partial_file_and_stops_when_disk_is_full(
    tmp_path: Path, monkeypatch
) -> None:
    sources = [
        _write(tmp_path / "src" / f"doc{index}.pdf", "x" * 4096) for index in range(4)
    ]
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", sources)
    attempts: list[str] = []

    def no_link(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def write_half_then_fail(fsrc, fdst, *_args, **_kwargs):
        attempts.append(os.fspath(fsrc.name))
        fdst.write(fsrc.read(2048))
        fdst.flush()
        raise OSError(errno.ENOSPC, "No space left on device")

    def copy2_half_then_fail(src, dst, *_args, **_kwargs):
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            write_half_then_fail(fsrc, fdst)

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(shutil, "copyfileobj", write_half_then_fail)
    monkeypatch.setattr(shutil, "copy2", copy2_half_then_fail)

    assert _execute(preview) == (0, 4)
    monkeypatch.undo()

    assert all(op.status == "failed" for op in preview.operations)
    assert _organized_files(preview.target_folder) == []
    assert all(source.stat().st_size == 4096 for source in sources)
    # A full disk stops the run instead of trying every remaining file.
    assert len(attempts) == 1


def test_journal_records_each_move_before_the_source_is_deleted(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    target = tmp_path / "out"
    preview = OrganizePreview(source_folder=tmp_path / "src", target_folder=target)
    preview.add_cluster("Topic", [source])
    seen_in_journal: list[bool] = []
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(source):
            journals = list((target / JOURNAL_DIR_NAME).glob("*.jsonl"))
            text = "".join(j.read_text(encoding="utf-8") for j in journals)
            seen_in_journal.append('"type": "move"' in text)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "remove", unlink)

    assert _execute(preview) == (1, 0)

    assert seen_in_journal == [True]


def _break_unlink_after_removing(monkeypatch, victim: Path, error: BaseException):
    """Make deleting ``victim`` raise ``error`` once the file is already gone."""
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        real_unlink(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(victim):
            raise error

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "remove", unlink)


def test_ctrl_c_right_after_the_source_delete_keeps_the_moved_file(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write(tmp_path / "src" / "thesis.pdf", "the only copy")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    # CPython raises a pending SIGINT when the unlink call returns, that is
    # after the source is already gone.
    _break_unlink_after_removing(monkeypatch, source, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _execute(preview)
    monkeypatch.undo()

    moved = preview.operations[0].destination
    assert not source.exists()
    assert moved.read_text(encoding="utf-8") == "the only copy"
    pending = read_journal(preview.journal_path).pending
    assert [(entry.source, entry.destination) for entry in pending] == [(source, moved)]

    undo_journal(preview.journal_path)

    assert source.read_text(encoding="utf-8") == "the only copy"


def test_source_delete_error_after_the_file_is_gone_keeps_the_moved_file(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    # A network file system can report an error for an unlink it carried out.
    _break_unlink_after_removing(
        monkeypatch, source, OSError(errno.EIO, "Input/output error")
    )

    assert _execute(preview) == (1, 0)
    monkeypatch.undo()

    moved = preview.operations[0].destination
    assert moved.read_text(encoding="utf-8") == "payload"
    assert [entry.source for entry in read_journal(preview.journal_path).pending] == [
        source
    ]


def test_ctrl_c_before_the_journal_write_leaves_no_unrecorded_name(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    real_link = os.link

    def link_then_interrupt(*args, **kwargs):
        real_link(*args, **kwargs)
        signal.raise_signal(signal.SIGINT)  # Ctrl-C as the link is made

    monkeypatch.setattr(os, "link", link_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        _execute(preview)
    monkeypatch.undo()

    assert source.read_text(encoding="utf-8") == "payload"
    assert _organized_files(preview.target_folder) == []
    assert read_journal(preview.journal_path).pending == []


def _no_link(*_args, **_kwargs):  # FAT, exFAT
    raise PermissionError(errno.EPERM, "Operation not permitted")


def _ctrl_c_on_journal_fsync(monkeypatch, journal_dir: Path, record: int) -> None:
    """Press Ctrl-C while the ``record``-th journal record from now is fsynced."""
    real_fsync = os.fsync
    count = 0

    def fsync(fd):
        nonlocal count
        real_fsync(fd)
        info = os.fstat(fd)
        if any(os.path.samestat(info, os.stat(j)) for j in journal_dir.glob("*.jsonl")):
            count += 1
            if count == record:
                signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(os, "fsync", fsync)


def test_ctrl_c_as_a_folder_is_journaled_leaves_no_unrecorded_folder(
    tmp_path: Path, monkeypatch
) -> None:
    first = _write(tmp_path / "src" / "a.txt", "a")
    second = _write(tmp_path / "src" / "b.txt", "b")
    target = tmp_path / "out"
    preview = OrganizePreview(source_folder=tmp_path / "src", target_folder=target)
    preview.add_cluster("One", [first])
    preview.add_cluster("Two", [second])
    # The records are the header, the folder One, then the folder Two.
    _ctrl_c_on_journal_fsync(monkeypatch, target / JOURNAL_DIR_NAME, 3)

    with pytest.raises(KeyboardInterrupt):
        _execute(preview)
    monkeypatch.undo()

    recorded = set(read_journal(preview.journal_path).created_dirs)
    folders = {path.resolve() for path in target.iterdir()}
    assert folders - {(target / JOURNAL_DIR_NAME).resolve()} <= recorded

    undo_journal(preview.journal_path)

    assert [path.name for path in target.iterdir()] == [JOURNAL_DIR_NAME]
    assert first.read_text(encoding="utf-8") == "a"
    assert second.read_text(encoding="utf-8") == "b"


@pytest.mark.parametrize("hard_links", [True, False], ids=["link", "copy"])
def test_ctrl_c_again_during_the_cleanup_still_removes_the_new_name(
    tmp_path: Path, monkeypatch, hard_links: bool
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    new_name = preview.operations[0].destination
    real_unlink = os.unlink

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt  # the first Ctrl-C, once the name is made

    def unlink(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(new_name):
            signal.raise_signal(signal.SIGINT)  # pressed again, and again
        return real_unlink(path, *args, **kwargs)

    if hard_links:
        monkeypatch.setattr(move_module.MoveJournal, "record_move", interrupted)
    else:
        monkeypatch.setattr(os, "link", _no_link)
        monkeypatch.setattr(move_module, "_fsync_directory", interrupted)
    monkeypatch.setattr(os, "unlink", unlink)

    with pytest.raises(KeyboardInterrupt):
        _execute(preview)
    monkeypatch.undo()

    assert source.read_text(encoding="utf-8") == "payload"
    assert _organized_files(preview.target_folder) == []


class _WithoutInodeNumber:
    """``os.lstat`` of a file system that reports st_ino 0 for every file."""

    def __init__(self, info: os.stat_result) -> None:
        self._info = info

    def __getattr__(self, name: str):
        return 0 if name == "st_ino" else getattr(self._info, name)


def _without_inode_numbers(monkeypatch) -> None:
    real_lstat = os.lstat

    def lstat(*args, **kwargs):
        return _WithoutInodeNumber(real_lstat(*args, **kwargs))

    monkeypatch.setattr(os, "lstat", lstat)


def test_ctrl_c_never_removes_a_file_it_did_not_make_without_inode_numbers(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    planned = preview.operations[0].destination
    real_link = os.link

    def someone_else_then_ctrl_c(src, dst, *args, **kwargs):
        _write(Path(dst), "someone else's file")  # appears just before the link
        signal.raise_signal(signal.SIGINT)
        return real_link(src, dst, *args, **kwargs)

    _without_inode_numbers(monkeypatch)
    monkeypatch.setattr(os, "link", someone_else_then_ctrl_c)

    with pytest.raises(KeyboardInterrupt):
        _execute(preview)
    monkeypatch.undo()

    assert planned.read_text(encoding="utf-8") == "someone else's file"
    assert source.read_text(encoding="utf-8") == "payload"


class _NearlyFullFile(io.FileIO):
    """A file on a disk with ``room`` bytes left; writing more raises ENOSPC."""

    def __init__(self, path, mode: str, room: int) -> None:
        super().__init__(path, mode)
        self.room = room

    def write(self, data) -> int:
        if self.room <= 0:
            raise OSError(errno.ENOSPC, "No space left on device")
        written = super().write(bytes(data)[: self.room])
        self.room -= written
        return written


def test_copy_that_fills_the_disk_while_flushing_leaves_no_partial_file(
    tmp_path: Path, monkeypatch
) -> None:
    sources = [
        _write(tmp_path / "src" / f"doc{index}.pdf", "x" * 4096) for index in range(3)
    ]
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", sources)
    attempts: list[str] = []
    real_open = open

    def no_link(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def open_on_full_disk(file, mode="r", *args, **kwargs):
        if mode != "xb":
            return real_open(file, mode, *args, **kwargs)
        attempts.append(os.fspath(file))
        # Buffered like a real file: the copy fits in the buffer, so the
        # error comes from flush() and again from close().
        return io.BufferedWriter(_NearlyFullFile(file, "xb", room=1000))

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(move_module, "open", open_on_full_disk, raising=False)

    assert _execute(preview) == (0, 3)
    monkeypatch.undo()

    assert _organized_files(preview.target_folder) == []
    assert all(source.stat().st_size == 4096 for source in sources)
    assert len(attempts) == 1


# Run in a child process, so that a signal can end it for real.
_KILLED_MID_COPY = """
import errno
import os
import shutil
import signal
import sys
from pathlib import Path

from dite.flow.move import (
    JOURNAL_DIR_NAME,
    OrganizePreview,
    find_latest_journal,
    undo_journal,
)

mode, signal_name = sys.argv[1], sys.argv[2]
src, out = Path(sys.argv[3]), Path(sys.argv[4])


def tree(path):
    path = Path(path)
    return next((r for r in (src, out) if r == path or r in path.parents), None)


def within_one_tree(real):
    def call(a, b, *args, **kwargs):
        if tree(a) != tree(b):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(a, b, *args, **kwargs)

    return call


# src and out stand for two file systems: no link or rename between them.
os.link = within_one_tree(os.link)
os.rename = within_one_tree(os.rename)
os.replace = within_one_tree(os.replace)
for fast_copy in ("_USE_CP_SENDFILE", "_USE_CP_COPY_FILE_RANGE", "_HAS_FCOPYFILE"):
    if hasattr(shutil, fast_copy):
        setattr(shutil, fast_copy, False)
real_copyfileobj = shutil.copyfileobj
copies = []


def copyfileobj(fsrc, fdst, *args, **kwargs):
    copies.append(fsrc.name)
    if len(copies) == 2:  # the signal comes with the second file half copied
        fdst.write(fsrc.read(1000))
        fdst.flush()
        os.kill(os.getpid(), getattr(signal, signal_name))
    return real_copyfileobj(fsrc, fdst, *args, **kwargs)


shutil.copyfileobj = copyfileobj
if mode == "execute":
    preview = OrganizePreview(source_folder=src, target_folder=out)
    preview.add_cluster("Topic", sorted(src.iterdir()))
    preview.execute(dry_run=False, journal_dir=out / JOURNAL_DIR_NAME)
else:
    undo_journal(find_latest_journal(out))
"""

_PAYLOADS = {f"doc{index}.pdf": bytes([65 + index]) * 5000 for index in range(3)}


def _write_payloads(folder: Path) -> None:
    folder.mkdir(parents=True)
    for name, data in _PAYLOADS.items():
        (folder / name).write_bytes(data)


def _assert_payloads_back(folder: Path) -> None:
    for name, data in _PAYLOADS.items():
        assert (folder / name).read_bytes() == data


def _kill_mid_copy(tmp_path: Path, mode: str, signal_name: str):
    """Organize (or undo) src <-> out as if on two file systems, in a child
    process that gets ``signal_name`` with the second file half copied."""
    child = tmp_path / "bin" / "killed_mid_copy.py"
    child.parent.mkdir()
    child.write_text(_KILLED_MID_COPY, encoding="utf-8")
    src_dir = Path(move_module.__file__).parents[2]
    return subprocess.run(
        [sys.executable, str(child), mode, signal_name]
        + [str(tmp_path / "src"), str(tmp_path / "out")],
        env=dict(os.environ, PYTHONPATH=str(src_dir)),
        capture_output=True,
        timeout=120,
    )


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX signals")
@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGHUP", "SIGKILL"])
def test_organize_killed_mid_copy_leaves_nothing_undo_cannot_clean_up(
    tmp_path: Path, signal_name: str
) -> None:
    _write_payloads(tmp_path / "src")

    run = _kill_mid_copy(tmp_path, "execute", signal_name)

    assert run.returncode != 0, run.stderr
    left = sorted(os.listdir(tmp_path / "out" / "Topic"))
    if signal_name == "SIGKILL":
        # Nothing runs on kill -9, but the half copy has no real name.
        assert [name for name in left if not name.startswith(".")] == ["doc0.pdf"]
    else:
        assert left == ["doc0.pdf"]
    assert (tmp_path / "src" / "doc1.pdf").read_bytes() == _PAYLOADS["doc1.pdf"]

    result = undo_journal(find_latest_journal(tmp_path / "out"))

    assert result.skipped == []
    _assert_payloads_back(tmp_path / "src")
    assert _organized_files(tmp_path / "out") == []


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX signals")
@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGKILL"])
def test_undo_killed_mid_copy_can_be_retried(tmp_path: Path, signal_name: str) -> None:
    _write_payloads(tmp_path / "src")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", sorted((tmp_path / "src").iterdir()))
    assert _execute(preview) == (3, 0)

    run = _kill_mid_copy(tmp_path, "undo", signal_name)

    assert run.returncode != 0, run.stderr
    # doc2 was copied back first; the half copy of doc1 has no real name, so
    # it is not taken for a file the user put there.
    left = sorted(os.listdir(tmp_path / "src"))
    assert [name for name in left if not name.startswith(".")] == ["doc2.pdf"]

    result = undo_journal(find_latest_journal(tmp_path / "out"))

    assert result.skipped == []
    _assert_payloads_back(tmp_path / "src")
    assert sorted(os.listdir(tmp_path / "src")) == sorted(_PAYLOADS)
    assert _organized_files(tmp_path / "out") == []


def test_without_hard_links_a_copy_never_replaces_a_file_that_appears_meanwhile(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    planned = preview.operations[0].destination
    real_copystat = shutil.copystat

    def no_link(*_args, **_kwargs):  # FAT, exFAT
        raise PermissionError(errno.EPERM, "Operation not permitted")

    def copystat_then_race(src, dst, *args, **kwargs):
        real_copystat(src, dst, *args, **kwargs)
        if not planned.exists():  # another program takes the name meanwhile
            planned.write_text("someone else's file", encoding="utf-8")

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(shutil, "copystat", copystat_then_race)

    assert _execute(preview) == (1, 0)

    moved = preview.operations[0].destination
    assert planned.read_text(encoding="utf-8") == "someone else's file"
    assert moved != planned
    assert moved.read_text(encoding="utf-8") == "payload"
    assert sorted(os.listdir(moved.parent)) == sorted([planned.name, moved.name])
    assert not source.exists()

    result = undo_journal(preview.journal_path)

    assert result.skipped == []
    assert source.read_text(encoding="utf-8") == "payload"
    assert os.listdir(source.parent) == [source.name]


@pytest.mark.parametrize("phase", ["execute", "undo"])
def test_without_hard_links_ctrl_c_as_the_name_is_claimed_leaves_no_empty_file(
    tmp_path: Path, monkeypatch, phase: str
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    if phase == "undo":
        assert _execute(preview) == (1, 0)
    real_open = os.open

    def no_link(*_args, **_kwargs):  # FAT, exFAT
        raise PermissionError(errno.EPERM, "Operation not permitted")

    def open_then_interrupt(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if flags & os.O_EXCL and Path(path).name == "a.txt":
            signal.raise_signal(signal.SIGINT)  # raised as open() returns
        return fd

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(os, "open", open_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        if phase == "execute":
            _execute(preview)
        else:
            undo_journal(preview.journal_path)
    monkeypatch.undo()

    if phase == "execute":
        assert source.read_text(encoding="utf-8") == "payload"
        assert _organized_files(preview.target_folder) == []
    else:
        # No empty file at the original location for a retry to trip over.
        assert not source.exists()
        assert undo_journal(preview.journal_path).skipped == []
        assert source.read_text(encoding="utf-8") == "payload"


def test_execute_removes_partial_copies_a_killed_run_left(tmp_path: Path) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    stale = _write(tmp_path / "out" / "Topic" / f".dite-partial-{'0' * 32}", "hal")
    unrelated = _write(tmp_path / "out" / "Topic" / ".dite-partial-notes", "mine")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])

    assert _execute(preview) == (1, 0)

    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "mine"


def _with_file_size_limit(limit: int, action):
    """Run ``action`` while files cannot grow past ``limit`` bytes.

    The kernel then cuts a write short and fails the next one, exactly as
    on a full disk, but only for the files written while the limit holds.
    """
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
    try:
        return action()
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous)


def test_a_journal_that_runs_out_of_space_stays_readable(tmp_path: Path) -> None:
    # Moves within one file system need no space, so the journal is the
    # first thing a full target refuses.
    files = [
        _write(tmp_path / "src" / f"report_{index:02d}_{'x' * 60}.txt", "y")
        for index in range(30)
    ]
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", files)

    success, failed = _with_file_size_limit(4096, lambda: _execute(preview))

    assert success > 0
    assert failed > 0
    assert success + failed == len(files)
    contents = read_journal(preview.journal_path)
    moved = [op for op in preview.operations if op.status == "success"]
    assert [entry.source for entry in contents.pending] == [op.source for op in moved]
    assert _organized_files(preview.target_folder) == sorted(
        op.destination for op in moved
    )
    assert find_latest_journal(preview.target_folder) == preview.journal_path

    result = undo_journal(preview.journal_path)

    assert result.skipped == []
    assert all(path.read_text(encoding="utf-8") == "y" for path in files)
    assert _organized_files(preview.target_folder) == []


def test_a_journal_that_cannot_be_started_moves_nothing(tmp_path: Path) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])

    with pytest.raises(OSError):
        _with_file_size_limit(64, lambda: _execute(preview))

    assert source.read_text(encoding="utf-8") == "payload"
    assert list((preview.target_folder / JOURNAL_DIR_NAME).iterdir()) == []
    assert _organized_files(preview.target_folder) == []


def test_ctrl_c_as_the_journal_is_created_leaves_no_empty_journal(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write(tmp_path / "src" / "a.txt", "payload")
    preview = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=tmp_path / "out"
    )
    preview.add_cluster("Topic", [source])
    real_open = os.open

    def open_then_interrupt(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path).suffix == ".jsonl" and flags & os.O_CREAT:
            signal.raise_signal(signal.SIGINT)  # raised as open() returns
        return fd

    monkeypatch.setattr(os, "open", open_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        _execute(preview)
    monkeypatch.undo()

    assert source.read_text(encoding="utf-8") == "payload"
    assert list((preview.target_folder / JOURNAL_DIR_NAME).iterdir()) == []


def test_undo_ignores_a_journal_line_cut_off_mid_write(tmp_path: Path) -> None:
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])
    with open(preview.journal_path, "ab") as handle:
        handle.write(b'{"type": "move", "id": 2, "source": "/somewh')

    assert find_latest_journal(preview.target_folder) == preview.journal_path
    result = undo_journal(preview.journal_path)

    assert result.skipped == []
    assert len(result.restored) == 2
    for index, path in enumerate(files):
        assert path.read_text(encoding="utf-8") == f"payload {index}"
    # Records appended by undo start on a line of their own.
    assert read_journal(preview.journal_path).undone is True


# ASSIGN-10: partial undo can be retried, moved files are verified.


def test_undo_cleans_up_after_a_move_interrupted_before_the_delete(
    tmp_path: Path,
) -> None:
    preview, files = _organize(tmp_path, ["a.txt"])
    moved = preview.operations[0].destination
    # The process died after the journal write but before the source delete.
    os.link(moved, files[0])

    result = undo_journal(preview.journal_path)

    assert result.skipped == []
    assert len(result.restored) == 1
    assert files[0].read_text(encoding="utf-8") == "payload 0"
    assert not moved.exists()
    assert read_journal(preview.journal_path).undone is True


def test_partial_undo_can_be_retried(tmp_path: Path) -> None:
    preview, files = _organize(tmp_path, ["a.txt", "sub/b.txt"])
    files[0].write_text("placeholder", encoding="utf-8")

    first = undo_journal(preview.journal_path)

    assert len(first.restored) == 1
    assert first.skipped == [(files[0], "conflict")]
    assert read_journal(preview.journal_path).undone is False
    assert find_latest_journal(preview.target_folder) == preview.journal_path

    files[0].unlink()
    second = undo_journal(preview.journal_path)

    # Only the file left behind is retried; b.txt is not reported missing.
    assert [source for _, source in second.restored] == [files[0]]
    assert second.skipped == []
    assert read_journal(preview.journal_path).undone is True
    for index, path in enumerate(files):
        assert path.read_text(encoding="utf-8") == f"payload {index}"
    assert _organized_files(preview.target_folder) == []


def test_undo_keeps_the_restored_file_when_a_delete_error_comes_late(
    tmp_path: Path, monkeypatch
) -> None:
    preview, files = _organize(tmp_path, ["a.txt"])
    moved = preview.operations[0].destination
    _break_unlink_after_removing(
        monkeypatch, moved, OSError(errno.EIO, "Input/output error")
    )

    result = undo_journal(preview.journal_path)
    monkeypatch.undo()

    assert files[0].read_text(encoding="utf-8") == "payload 0"
    assert not moved.exists()
    assert result.skipped == []


@pytest.mark.parametrize("record", [1, 3], ids=["first", "last"])
def test_ctrl_c_while_a_restore_is_journaled_can_be_retried(
    tmp_path: Path, monkeypatch, record: int
) -> None:
    preview, files = _organize(tmp_path, ["a.txt", "b.txt", "c.txt"])
    _ctrl_c_on_journal_fsync(monkeypatch, preview.journal_path.parent, record)

    with pytest.raises(KeyboardInterrupt):
        undo_journal(preview.journal_path)
    monkeypatch.undo()

    # The file is back already, so its record is kept.
    assert len(read_journal(preview.journal_path).restored_ids) == record
    assert find_latest_journal(preview.target_folder) == preview.journal_path

    retry = undo_journal(preview.journal_path)

    assert retry.skipped == []
    assert read_journal(preview.journal_path).undone is True
    assert not (preview.target_folder / "Topic").exists()
    for index, path in enumerate(files):
        assert path.read_text(encoding="utf-8") == f"payload {index}"


def test_undo_retry_finds_a_file_moved_back_before_its_record(
    tmp_path: Path, monkeypatch
) -> None:
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])

    def interrupted(self, move_id):
        raise KeyboardInterrupt  # Ctrl-C between the move back and its record

    monkeypatch.setattr(move_module.MoveJournal, "record_restored", interrupted)
    with pytest.raises(KeyboardInterrupt):
        undo_journal(preview.journal_path)
    monkeypatch.undo()
    assert files[1].read_text(encoding="utf-8") == "payload 1"
    assert read_journal(preview.journal_path).restored_ids == set()

    retry = undo_journal(find_latest_journal(preview.target_folder))

    assert retry.skipped == []
    assert sorted(source for _, source in retry.restored) == files
    assert read_journal(preview.journal_path).undone is True
    assert not (preview.target_folder / "Topic").exists()
    assert files[0].read_text(encoding="utf-8") == "payload 0"


def test_undo_compares_content_where_inode_numbers_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    preview, files = _organize(tmp_path, ["a.txt"])
    moved = preview.operations[0].destination
    os.link(moved, tmp_path / "backup.txt")  # a second name for the moved file
    files[0].write_text("another file", encoding="utf-8")
    _without_inode_numbers(monkeypatch)

    result = undo_journal(preview.journal_path)
    monkeypatch.undo()

    assert result.skipped == [(files[0], "conflict")]
    assert moved.read_text(encoding="utf-8") == "payload 0"
    assert files[0].read_text(encoding="utf-8") == "another file"


def test_undo_skips_files_replaced_after_the_move(tmp_path: Path) -> None:
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])
    replaced, touched = (op.destination for op in preview.operations)
    replaced.unlink()
    replaced.write_text("a different file", encoding="utf-8")
    stat = touched.stat()
    touched.write_text("payload X", encoding="utf-8")  # same size
    os.utime(touched, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

    plan = undo_journal(preview.journal_path, dry_run=True)
    result = undo_journal(preview.journal_path)

    for outcome in (plan, result):
        assert outcome.restored == []
        assert sorted(outcome.skipped) == sorted(
            [(replaced, "changed"), (touched, "changed")]
        )
    assert not any(path.exists() for path in files)
    assert replaced.read_text(encoding="utf-8") == "a different file"
    assert touched.read_text(encoding="utf-8") == "payload X"
    # Not marked undone, but undoing the target passes over it: nothing in
    # it can be moved back now.
    assert read_journal(preview.journal_path).undone is False
    passed_over: list[Path] = []
    assert find_latest_journal(preview.target_folder, passed_over=passed_over) is None
    assert passed_over == [preview.journal_path]


def _write_journal_without_file_identity(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A journal as older versions wrote it: no ids, sizes or mtimes."""
    source = tmp_path / "src" / "a.txt"
    moved = tmp_path / "out" / "Topic" / "a.txt"
    journal = tmp_path / "out" / JOURNAL_DIR_NAME / "organize-20250101-000000-0.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        "\n".join(
            [
                '{"type": "header", "version": 1, "created_at": "2025-01-01T00:00:00",'
                f' "source_folder": "{tmp_path / "src"}",'
                f' "target_folder": "{tmp_path / "out"}"}}',
                f'{{"type": "mkdir", "path": "{tmp_path / "out" / "Topic"}"}}',
                f'{{"type": "move", "source": "{source}", "destination": "{moved}"}}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return journal, source, moved


def test_undo_still_reads_journals_without_file_identity(tmp_path: Path) -> None:
    journal, source, moved = _write_journal_without_file_identity(tmp_path)
    _write(moved, "payload")

    assert find_latest_journal(tmp_path / "out") == journal
    result = undo_journal(journal)

    assert result.skipped == []
    assert source.read_text(encoding="utf-8") == "payload"
    assert not (tmp_path / "out" / "Topic").exists()
    assert read_journal(journal).undone is True


def test_undo_without_file_identity_cannot_tell_a_file_is_back(
    tmp_path: Path,
) -> None:
    journal, source, moved = _write_journal_without_file_identity(tmp_path)
    _write(source, "payload")  # back at its source, or a new file there

    result = undo_journal(journal)

    assert result.skipped == [(moved, "missing")]
    assert read_journal(journal).undone is False
    assert source.read_text(encoding="utf-8") == "payload"


def _write_cli_config(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_path = home / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "api:",
                "  base_url: https://api.example.com/v1",
                "  api_key: dummy-key",
                "cache:",
                f"  directory: {tmp_path / 'cache'}",
                "i18n:",
                "  locale: en",
            ]
        ),
        encoding="utf-8",
    )


def test_cli_undo_reports_partial_undo_and_retries(tmp_path: Path, monkeypatch) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])
    files[1].write_text("placeholder", encoding="utf-8")
    runner = CliRunner()

    first = runner.invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert first.exit_code == 0, first.output
    # The retry hint names the command (rich may wrap the line anywhere).
    assert "dite undo" in " ".join(first.output.split())
    assert files[0].read_text(encoding="utf-8") == "payload 0"

    files[1].unlink()
    second = runner.invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert second.exit_code == 0, second.output
    assert files[1].read_text(encoding="utf-8") == "payload 1"
    assert read_journal(preview.journal_path).undone is True


def test_cli_undo_writes_nothing_when_nothing_can_be_moved_back(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])
    files[1].write_text("placeholder", encoding="utf-8")
    assert len(undo_journal(preview.journal_path).restored) == 1
    # The user keeps the new b.txt and deletes the organized one instead.
    preview.operations[1].destination.unlink()
    before = preview.journal_path.read_bytes()

    result = CliRunner().invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert result.exit_code == 0, result.output
    assert "0 restored" in result.output
    assert preview.journal_path.read_bytes() == before
    assert find_latest_journal(preview.target_folder) is None
    assert files[1].read_text(encoding="utf-8") == "placeholder"


def test_cli_undo_moves_nothing_back_without_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["a.txt"])
    moved = preview.operations[0].destination
    elsewhere = tmp_path / "elsewhere.txt"
    os.link(moved, elsewhere)
    moved.unlink()
    real_undo = move_module.undo_journal

    def file_returns_after_the_plan(path, *, dry_run=False):
        result = real_undo(path, dry_run=dry_run)
        if dry_run:
            # The plan said nothing can be moved back; then the file is back.
            os.link(elsewhere, moved)
        return result

    monkeypatch.setattr(move_module, "undo_journal", file_returns_after_the_plan)

    # No --yes and no answer: nothing the user did not confirm may move.
    result = CliRunner().invoke(app, ["undo", str(preview.target_folder)])

    assert result.exit_code == 0, result.output
    assert moved.read_text(encoding="utf-8") == "payload 0"
    assert not files[0].exists()
    assert read_journal(preview.journal_path).undone is False


def test_cli_undo_finds_a_journal_again_once_its_files_are_back(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])
    folder = preview.target_folder / "Topic"
    renamed = folder.with_name("My topic")
    folder.rename(renamed)  # the user renames the folder dite made
    runner = CliRunner()

    # No --yes: nothing can be moved back, so nothing is asked or written.
    first = runner.invoke(app, ["undo", str(preview.target_folder)])

    assert first.exit_code == 0, first.output
    assert "0 restored" in first.output
    assert not any(path.exists() for path in files)
    assert read_journal(preview.journal_path).undone is False

    renamed.rename(folder)
    second = runner.invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert second.exit_code == 0, second.output
    for index, path in enumerate(files):
        assert path.read_text(encoding="utf-8") == f"payload {index}"
    assert read_journal(preview.journal_path).undone is True


def test_cli_undo_passes_over_a_run_whose_files_are_gone_for_now(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    target = tmp_path / "out"
    first_file = _write(tmp_path / "src" / "a.txt", "first run")
    second_file = _write(tmp_path / "src" / "b.txt", "second run")
    runs = []
    for folder, path in (("One", first_file), ("Two", second_file)):
        preview = OrganizePreview(source_folder=tmp_path / "src", target_folder=target)
        preview.add_cluster(folder, [path])
        assert _execute(preview) == (1, 0)
        runs.append(preview)
    (target / "Two").rename(target / "Elsewhere")
    runner = CliRunner()

    result = runner.invoke(app, ["undo", str(target), "--yes"])

    assert result.exit_code == 0, result.output
    assert first_file.read_text(encoding="utf-8") == "first run"
    assert not second_file.exists()
    # The user is told which run was passed over (rich may wrap anywhere).
    assert runs[1].journal_path.name in "".join(result.output.split())
    assert read_journal(runs[1].journal_path).undone is False

    (target / "Elsewhere").rename(target / "Two")
    again = runner.invoke(app, ["undo", str(target), "--yes"])

    assert again.exit_code == 0, again.output
    assert second_file.read_text(encoding="utf-8") == "second run"
    assert find_latest_journal(target) is None


def test_cli_undo_names_the_original_location_of_a_changed_file(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["deep/dir/thesis.txt"])
    moved = preview.operations[0].destination
    moved.write_text("edited", encoding="utf-8")

    result = CliRunner().invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert result.exit_code == 0, result.output
    # The edited file stays where it is, and the message says where it came
    # from (rich may wrap the line anywhere).
    assert moved.read_text(encoding="utf-8") == "edited"
    assert not files[0].exists()
    output = "".join(result.output.split())
    assert "changedorreplaced" in output
    assert f"itcamefrom{''.join(str(files[0]).split())}" in output


def test_cli_undo_stops_with_a_clear_message_when_the_journal_cannot_be_written(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])

    def disk_full(self, move_id):
        raise OSError(errno.ENOSPC, "No space left on device")

    runner = CliRunner()
    with monkeypatch.context() as patch:
        patch.setattr(move_module.MoveJournal, "record_restored", disk_full)

        result = runner.invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, OSError)
    assert "Undo stopped" in result.output
    # The one file moved back before the error stays back.
    assert files[1].read_text(encoding="utf-8") == "payload 1"

    # Once there is room again, running undo again finishes the job: the
    # file moved back without a record is not taken for a missing one.
    retry = runner.invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert retry.exit_code == 0, retry.output
    assert "Skipped" not in retry.output
    assert read_journal(preview.journal_path).undone is True
    assert not (preview.target_folder / "Topic").exists()
    for index, path in enumerate(files):
        assert path.read_text(encoding="utf-8") == f"payload {index}"


def test_cli_undo_finishes_an_undo_stopped_after_its_last_file(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])
    with monkeypatch.context() as patch:
        _ctrl_c_on_journal_fsync(patch, preview.journal_path.parent, 2)
        with pytest.raises(KeyboardInterrupt):
            undo_journal(preview.journal_path)
    assert read_journal(preview.journal_path).pending == []

    # No --yes and no answer: nothing is left to move, so nothing is asked.
    result = CliRunner().invoke(app, ["undo", str(preview.target_folder)])

    assert result.exit_code == 0, result.output
    assert read_journal(preview.journal_path).undone is True
    assert not (preview.target_folder / "Topic").exists()
    for index, path in enumerate(files):
        assert path.read_text(encoding="utf-8") == f"payload {index}"
