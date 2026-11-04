from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from dite.app.cli import app
from dite.flow.api import PipelineResult
from dite.flow.assign import (
    assign_to_folders,
    build_folder_profiles,
    list_category_folders,
)


def _direction(seed: int, dim: int = 32) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dim)
    return vector / np.linalg.norm(vector)


def _members(center: np.ndarray, count: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [center + spread * rng.normal(size=center.shape) for _ in range(count)]
    )


def test_folder_floor_reflects_how_tight_the_folder_is() -> None:
    center = _direction(0)
    profiles = build_folder_profiles(
        {
            "tight": _members(center, 6, 0.02, seed=1),
            "broad": _members(_direction(2), 6, 0.25, seed=3),
            "single": _direction(4)[None, :],
        }
    )
    by_name = {profile.name: profile for profile in profiles}

    assert by_name["tight"].floor > by_name["broad"].floor
    assert by_name["single"].floor is None
    assert by_name["tight"].member_count == 6


def test_new_files_join_close_folders_and_others_stay_unsorted() -> None:
    algebra, finance = _direction(10), _direction(11)
    profiles = build_folder_profiles(
        {
            "algebra": _members(algebra, 5, 0.05, seed=12),
            "finance": _members(finance, 5, 0.05, seed=13),
        }
    )
    files = [Path("new/eigen.md"), Path("new/budget.md"), Path("new/poem.md")]
    embeddings = np.stack(
        [
            algebra + 0.03 * _direction(20),
            finance + 0.03 * _direction(21),
            _direction(22),  # unrelated to both folders
        ]
    )

    decisions = assign_to_folders(files, embeddings, profiles)

    assert [decision.folder for decision in decisions] == ["algebra", "finance", None]
    assert decisions[2].closest in {"algebra", "finance"}
    assert decisions[2].similarity < decisions[2].floor


def test_single_member_folders_borrow_the_median_floor() -> None:
    algebra, finance = _direction(30), _direction(31)
    profiles = build_folder_profiles(
        {
            "algebra": _members(algebra, 4, 0.05, seed=32),
            "finance": finance[None, :],
        }
    )

    decisions = assign_to_folders(
        [Path("near-finance.md"), Path("far.md")],
        np.stack([finance + 0.02 * _direction(33), _direction(34)]),
        profiles,
    )

    assert decisions[0].folder == "finance"
    assert decisions[0].floor == profiles[0].floor
    assert decisions[1].folder is None


def test_min_similarity_overrides_folder_floors_and_zero_vectors_stay() -> None:
    algebra = _direction(40)
    profiles = build_folder_profiles({"algebra": _members(algebra, 4, 0.05, seed=41)})
    loose = 0.5 * algebra + 0.5 * _direction(42)

    default = assign_to_folders([Path("a.md")], loose[None, :], profiles)
    relaxed = assign_to_folders(
        [Path("a.md"), Path("empty.md")],
        np.stack([loose, np.zeros_like(loose)]),
        profiles,
        min_similarity=0.1,
    )

    assert default[0].folder is None
    assert relaxed[0].folder == "algebra"
    assert relaxed[1].folder is None and relaxed[1].similarity is None


def test_list_category_folders_skips_hidden_directories(tmp_path: Path) -> None:
    for name in ("b", "a", ".dite", ".git"):
        (tmp_path / name).mkdir()
    (tmp_path / "loose.txt").write_text("x", encoding="utf-8")

    assert [path.name for path in list_category_folders(tmp_path)] == ["a", "b"]


def _write_config(home: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(home))
    config_path = home / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "api:",
                "  base_url: https://api.example.com/v1",
                "  api_key: test-key",
                "cache:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )


def test_assign_cli_sorts_loose_files_into_existing_folders(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "papers"
    algebra_dir = root / "线性代数"
    finance_dir = root / "finance [2026]"
    algebra_dir.mkdir(parents=True)
    finance_dir.mkdir(parents=True)
    centers = {"algebra": _direction(50), "finance": _direction(51)}
    vectors: dict[str, np.ndarray] = {}
    for index in range(4):
        for name, folder in (("algebra", algebra_dir), ("finance", finance_dir)):
            path = folder / f"{name}-{index}.md"
            path.write_text(f"{name} {index}", encoding="utf-8")
            vectors[path.name] = centers[name] + 0.05 * _direction(60 + index)
    new_algebra = root / "eigenvalues.md"
    new_other = root / "poem.md"
    new_algebra.write_text("new algebra", encoding="utf-8")
    new_other.write_text("unrelated", encoding="utf-8")
    vectors[new_algebra.name] = centers["algebra"] + 0.02 * _direction(70)
    vectors[new_other.name] = _direction(71)

    def fake_embed_files(self, files, options):
        del self, options
        return PipelineResult(
            files=list(files),
            contents=["x"] * len(files),
            embeddings=np.stack([vectors[file.name] for file in files]),
            labels=np.full(len(files), -1, dtype=int),
            cluster_names={},
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.embed_files", fake_embed_files)
    _write_config(tmp_path, monkeypatch)
    runner = CliRunner()

    preview = runner.invoke(app, ["assign", str(root), "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert new_algebra.exists()

    executed = runner.invoke(app, ["assign", str(root), "--execute"], input="y\n")
    assert executed.exit_code == 0, executed.output
    assert (algebra_dir / "eigenvalues.md").read_text(encoding="utf-8") == (
        "new algebra"
    )
    assert not new_algebra.exists()
    assert new_other.exists()  # not close to any folder, left in place
    assert not (root / "linear algebra").exists()

    undone = runner.invoke(app, ["undo", str(root), "--yes"])
    assert undone.exit_code == 0, undone.output
    assert new_algebra.exists()
    # Existing folders were not created by the run and are never removed.
    assert algebra_dir.is_dir()


def test_assign_cli_requires_category_folders(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "flat"
    root.mkdir()
    (root / "a.md").write_text("a", encoding="utf-8")
    _write_config(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, ["assign", str(root), "--dry-run"])

    assert result.exit_code == 1
