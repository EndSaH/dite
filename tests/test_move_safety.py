"""Safety of organize/assign file moves and of ``dite undo``.

The tests run on case-sensitive Linux file systems, so case-insensitive
volumes (the macOS default) are simulated with names that differ only in
case, with NFC/NFD spellings, or with a symlink that makes two spellings of a
folder name point at the same directory.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from dite.flow.move import (
    JOURNAL_DIR_NAME,
    OrganizePreview,
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
