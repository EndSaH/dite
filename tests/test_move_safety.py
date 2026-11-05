"""Safety of organize/assign file moves and of ``dite undo``.

The tests run on case-sensitive Linux file systems, so case-insensitive
volumes (the macOS default) are simulated with names that differ only in
case, with NFC/NFD spellings, or with a symlink that makes two spellings of a
folder name point at the same directory.
"""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest
from rich.console import Console

from dite.flow.move import (
    JOURNAL_DIR_NAME,
    FileOperation,
    OrganizePreview,
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
