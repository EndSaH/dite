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
    # A retry cannot fix this, so the journal does not stay open for it.
    assert read_journal(preview.journal_path).undone is True


def test_undo_still_reads_journals_without_file_identity(tmp_path: Path) -> None:
    source = tmp_path / "src" / "a.txt"
    moved = _write(tmp_path / "out" / "Topic" / "a.txt", "payload")
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

    assert find_latest_journal(tmp_path / "out") == journal
    result = undo_journal(journal)

    assert result.skipped == []
    assert source.read_text(encoding="utf-8") == "payload"
    assert not (tmp_path / "out" / "Topic").exists()
    assert read_journal(journal).undone is True


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


def test_cli_undo_closes_a_journal_once_only_final_skips_remain(
    tmp_path: Path, monkeypatch
) -> None:
    _write_cli_config(tmp_path, monkeypatch)
    preview, files = _organize(tmp_path, ["a.txt", "b.txt"])
    files[1].write_text("placeholder", encoding="utf-8")
    assert len(undo_journal(preview.journal_path).restored) == 1
    # The user keeps the new b.txt and deletes the organized one instead.
    preview.operations[1].destination.unlink()

    result = CliRunner().invoke(app, ["undo", str(preview.target_folder), "--yes"])

    assert result.exit_code == 0, result.output
    assert read_journal(preview.journal_path).undone is True
    assert find_latest_journal(preview.target_folder) is None
    assert files[1].read_text(encoding="utf-8") == "placeholder"
