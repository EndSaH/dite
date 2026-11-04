"""Sort new files into existing category folders.

Most real use is incremental: a folder is already organized and new files
arrive. Re-clustering everything would reshuffle folders that people have
already accepted, so instead each existing subfolder acts as a category and
each new file joins the folder whose content it is closest to.

How close is close enough is judged per folder, not with one global cosine
value: a folder's bar is set by how similar its own members are to the rest
of that folder. A tight folder (one course, one project) accepts only close
matches; a broad folder accepts looser ones. This keeps the rule meaningful
across embedding models, whose absolute similarity ranges differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dite.doc.embed import normalize_embeddings

# A new file must be at least as similar to a folder as the members in the
# folder's least-typical 10% are.
FOLDER_FLOOR_PERCENTILE = 10.0


@dataclass(frozen=True)
class FolderProfile:
    """A category folder summarized by its centroid and acceptance floor."""

    name: str
    member_count: int
    centroid: np.ndarray
    floor: float | None  # None when the folder has too few members to tell


@dataclass(frozen=True)
class AssignmentDecision:
    """Where one new file goes, and why."""

    file: Path
    folder: str | None  # accepted folder, or None to leave the file unsorted
    similarity: float | None  # similarity to the closest folder
    floor: float | None  # the bar the closest folder applied
    closest: str | None = None
    runner_up: str | None = None
    runner_up_similarity: float | None = None


def build_folder_profiles(
    members_by_folder: dict[str, np.ndarray],
) -> list[FolderProfile]:
    """Build centroid profiles from each folder's member embeddings.

    The floor uses leave-one-out similarity: each member is compared with the
    centroid of the *other* members, so a member does not vouch for itself.
    Folders with one member get no floor of their own; they borrow the median
    floor of the other folders when the decision is made.
    """
    profiles: list[FolderProfile] = []
    for name in sorted(members_by_folder):
        vectors = normalize_embeddings(np.asarray(members_by_folder[name]))
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            continue
        total = vectors.sum(axis=0)
        centroid = normalize_embeddings(total / vectors.shape[0])
        floor = None
        if vectors.shape[0] >= 2:
            leave_one_out = normalize_embeddings(
                (total[None, :] - vectors) / (vectors.shape[0] - 1)
            )
            similarities = np.sum(vectors * leave_one_out, axis=1)
            floor = float(np.percentile(similarities, FOLDER_FLOOR_PERCENTILE))
        profiles.append(
            FolderProfile(
                name=name,
                member_count=int(vectors.shape[0]),
                centroid=centroid,
                floor=floor,
            )
        )
    return profiles


def assign_to_folders(
    files: list[Path],
    embeddings: np.ndarray,
    profiles: list[FolderProfile],
    *,
    min_similarity: float | None = None,
) -> list[AssignmentDecision]:
    """Pick the most similar folder for each file, or leave it unsorted.

    ``min_similarity`` replaces every folder's floor with one absolute cosine
    value. Without it, folders without their own floor use the median floor
    of the others; if no folder has one, nothing is assigned.
    """
    if not profiles:
        return [
            AssignmentDecision(file=file, folder=None, similarity=None, floor=None)
            for file in files
        ]
    known_floors = [profile.floor for profile in profiles if profile.floor is not None]
    fallback_floor = float(np.median(known_floors)) if known_floors else None
    centroids = np.stack([profile.centroid for profile in profiles])
    vectors = normalize_embeddings(np.asarray(embeddings))
    decisions: list[AssignmentDecision] = []
    for file, vector in zip(files, vectors, strict=True):
        if not np.any(vector):
            # Failed embedding: no evidence either way.
            decisions.append(
                AssignmentDecision(file=file, folder=None, similarity=None, floor=None)
            )
            continue
        similarities = centroids @ vector
        order = np.argsort(-similarities, kind="stable")
        best = profiles[int(order[0])]
        best_similarity = float(similarities[order[0]])
        runner_up = profiles[int(order[1])] if len(order) > 1 else None
        runner_up_similarity = (
            float(similarities[order[1]]) if runner_up is not None else None
        )
        if min_similarity is not None:
            floor = float(min_similarity)
        else:
            floor = best.floor if best.floor is not None else fallback_floor
        accepted = floor is not None and best_similarity >= floor
        decisions.append(
            AssignmentDecision(
                file=file,
                folder=best.name if accepted else None,
                similarity=best_similarity,
                floor=floor,
                closest=best.name,
                runner_up=runner_up.name if runner_up is not None else None,
                runner_up_similarity=runner_up_similarity,
            )
        )
    return decisions


def list_category_folders(target: Path) -> list[Path]:
    """Immediate, visible subdirectories of ``target`` (sorted by name)."""
    return sorted(
        child
        for child in target.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )
