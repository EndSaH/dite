"""Scan skip rules: hidden, dependency, package and system entries."""

import errno
import os
from pathlib import Path

import pytest

from dite.app.config import Config
from dite.app.i18n import get_locale, set_locale
from dite.flow.scan import scan_files
from dite.util.log import setup_logging

EXTENSIONS = Config().formats.all_extensions


@pytest.fixture(autouse=True)
def _restore_locale():
    previous = get_locale()
    yield
    set_locale(previous)


def _touch(path: Path, text: str = "payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _scan(root: Path, **kwargs) -> list[str]:
    files = scan_files(root, config=Config(), extensions=EXTENSIONS, **kwargs)
    return [path.relative_to(root).as_posix() for path in files]


def test_scan_skips_hidden_directories_and_files(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _touch(root / "keep.md")
    _touch(root / "sub" / "keep.txt")
    _touch(root / ".obsidian" / "notes.md")
    _touch(root / ".git" / "description.txt")
    _touch(root / ".hidden.txt")
    # AppleDouble metadata written next to real files by macOS on non-HFS volumes.
    _touch(root / "sub" / "._budget.pdf")
    _touch(root / "__MACOSX" / "sub" / "._thermo.pdf")

    assert _scan(root) == ["keep.md", "sub/keep.txt"]


def test_scan_skips_dependency_and_virtualenv_directories(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _touch(root / "proj" / "README.md")
    _touch(root / "proj" / "node_modules" / "lodash" / "README.md")
    _touch(root / "proj" / "__pycache__" / "notes.txt")
    _touch(root / "proj" / "lib" / "site-packages" / "pkg" / "LICENSE.md")
    # A virtualenv without a leading dot is recognised by its pyvenv.cfg marker.
    (root / "proj" / "venv").mkdir(parents=True)
    (root / "proj" / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    _touch(root / "proj" / "venv" / "share" / "doc.md")
    _touch(root / "proj" / "venv" / "lib" / "pkg-1.0.dist-info" / "entry_points.txt")
    # A plain folder that merely shares a common venv name is still scanned.
    _touch(root / "env" / "plan.md")

    assert _scan(root) == ["env/plan.md", "proj/README.md"]


def test_scan_skips_macos_package_directories(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _touch(root / "photo.jpg")
    _touch(root / "Photos Library.photoslibrary" / "originals" / "IMG_1.jpg")
    _touch(root / "Tool.app" / "Contents" / "Resources" / "readme.txt")
    _touch(root / "Report.rtfd" / "TXT.rtf")
    _touch(root / "Budget.numbers" / "preview.jpg")
    _touch(root / "Song.logicx" / "Resources" / "cover.png")
    _touch(root / "Backup.sparsebundle" / "notes.txt")
    _touch(root / "Upper.APP" / "Contents" / "readme.md")
    # Dotted folder names that are not package types are ordinary folders.
    _touch(root / "release-1.0" / "notes.md")

    assert _scan(root) == ["photo.jpg", "release-1.0/notes.md"]


def test_scan_skips_office_lock_and_system_files(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _touch(root / "budget.docx")
    _touch(root / "~$budget.docx")
    _touch(root / "sub" / "~$deck.pptx")
    _touch(root / "sub" / "slides.pptx")

    assert _scan(root) == ["budget.docx", "sub/slides.pptx"]


def test_scan_reads_hidden_root_folder_given_explicitly(tmp_path: Path) -> None:
    root = tmp_path / ".notes"
    _touch(root / "a.md")
    _touch(root / "sub" / "b.md")
    _touch(root / ".trash" / "c.md")

    assert _scan(root) == ["a.md", "sub/b.md"]


def test_scan_reads_package_or_virtualenv_root_given_explicitly(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "Report.rtfd"
    _touch(bundle / "TXT.rtf")
    assert _scan(bundle) == ["TXT.rtf"]

    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    _touch(venv / "share" / "doc.md")
    assert _scan(venv) == ["share/doc.md"]


def test_scan_non_recursive_skips_hidden_and_lock_files(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _touch(root / "keep.md")
    _touch(root / ".draft.md")
    _touch(root / "._keep.md")
    _touch(root / "~$report.docx")
    _touch(root / "sub" / "nested.md")

    assert _scan(root, recursive=False) == ["keep.md"]


def test_scan_prunes_skipped_directories_without_listing_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "docs"
    _touch(root / "keep.md")
    for index in range(3):
        _touch(root / "node_modules" / f"pkg{index}" / "deep" / "README.md")
    _touch(root / ".git" / "objects" / "info.txt")
    _touch(root / "App.app" / "Contents" / "readme.txt")

    listed: list[Path] = []
    real_scandir = os.scandir

    def recording_scandir(path="."):
        listed.append(Path(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", recording_scandir)

    assert _scan(root) == ["keep.md"]
    assert listed, "expected the scan to list at least the root folder"
    pruned = {"node_modules", ".git", "App.app"}
    assert not [path for path in listed if pruned & set(path.parts)]


@pytest.mark.parametrize("error", [errno.EACCES, errno.EPERM], ids=["EACCES", "EPERM"])
def test_scan_skips_subdirectory_it_may_not_enter(
    tmp_path: Path,
    monkeypatch,
    error: int,
) -> None:
    root = tmp_path / "docs"
    _touch(root / "keep.md")
    _touch(root / "sub" / "a.txt")
    locked = root / "locked"
    _touch(locked / "x.md")

    # A folder the user may not search or list, such as a root-owned
    # lost+found on a mounted drive (EACCES) or a macOS privacy-protected
    # folder (EPERM): the folder itself can be stat'ed, nothing inside it can,
    # and listing it fails. Faked because tests may run as root.
    def inside_locked(path) -> bool:
        return isinstance(path, (str, os.PathLike)) and Path(path).is_relative_to(
            locked
        )

    def refuse(path):
        raise PermissionError(error, os.strerror(error), os.fspath(path))

    real_stat = os.stat
    real_scandir = os.scandir

    def guarded_stat(path, *args, **kwargs):
        if inside_locked(path) and Path(path) != locked:
            refuse(path)
        return real_stat(path, *args, **kwargs)

    def guarded_scandir(path="."):
        if inside_locked(path):
            refuse(path)
        return real_scandir(path)

    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(os, "scandir", guarded_scandir)

    assert _scan(root) == ["keep.md", "sub/a.txt"]


def test_scan_keeps_exclude_paths_with_skip_rules(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _touch(root / "a.txt")
    _touch(root / "organized" / "b.txt")
    _touch(root / "organized" / ".dite" / "organize-1.jsonl")

    assert _scan(root, exclude_paths=[root / "organized"]) == ["a.txt"]


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("en", "Skipped 3 ignored directories and 4 ignored files"),
        ("zh-CN", "跳过 3 个忽略目录和 4 个忽略文件"),
    ],
)
def test_scan_logs_ignored_entry_counts(
    tmp_path: Path,
    capsys,
    locale: str,
    expected: str,
) -> None:
    root = tmp_path / "docs"
    _touch(root / "keep.md")
    _touch(root / ".git" / "HEAD")
    _touch(root / "node_modules" / "pkg" / "README.md")
    _touch(root / "Tool.app" / "Contents" / "readme.txt")
    _touch(root / ".DS_Store")
    _touch(root / "~$keep.docx")
    _touch(root / "Thumbs.db")
    _touch(root / "desktop.ini")

    setup_logging(verbose=True)
    set_locale(locale)
    try:
        files = _scan(root)
    finally:
        setup_logging()

    output = capsys.readouterr().out
    assert files == ["keep.md"]
    assert expected in output
    # System files are reported as ignored, not as unsupported formats.
    assert "unsupported files" not in output
    assert "个不支持的文件" not in output
