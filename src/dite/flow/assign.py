"""Sort new files into existing category folders.

Most real use is incremental: a folder is already organized and new files
arrive. Re-clustering everything would reshuffle folders that people have
already accepted, so instead each existing subfolder acts as a category and
each new file joins the folder whose content it is closest to.

How close is close enough is judged per folder, not with one global cosine
value. Each folder compares two distributions: how similar its own members
are to the rest of the folder (insiders) and how similar files from the
other folders are to it (outsiders). The bar sits halfway between the least
typical insiders and the most typical outsiders, like a one-dimensional
classifier boundary. This keeps the rule meaningful across embedding
models, whose absolute similarity ranges differ, and does not reject new
files just because a folder's members happen to be near-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dite.doc.embed import normalize_embeddings

# Insiders: the least typical 10% of a folder's members.
FOLDER_FLOOR_PERCENTILE = 10.0
# Outsiders: the 10% of other folders' files that look most like this folder.
OUTSIDER_CEILING_PERCENTILE = 90.0


@dataclass(frozen=True)
class FolderProfile:
    """A category folder summarized by its centroid and acceptance floor."""

    name: str
    member_count: int
    centroid: np.ndarray
    floor: float | None  # None when the folder has too few members to tell
    insider_floor: float | None = None
    outsider_ceiling: float | None = None


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
    """Build centroid profiles and acceptance bars from member embeddings.

    Insider similarity is leave-one-out: each member is compared with the
    centroid of the *other* members, so a member does not vouch for itself.
    The bar is the midpoint between the insiders' 10th percentile and the
    outsiders' 90th percentile. When the two overlap (the folder is not
    separable from the others) the stricter insider value is used; with no
    other folders, the insider value alone. Folders with one member get no
    bar of their own and borrow the median bar of the others later.
    """
    names = sorted(
        name
        for name, vectors in members_by_folder.items()
        if np.asarray(vectors).ndim == 2 and np.asarray(vectors).shape[0] > 0
    )
    normalized = {
        name: normalize_embeddings(np.asarray(members_by_folder[name]))
        for name in names
    }
    centroids = {
        name: normalize_embeddings(vectors.mean(axis=0))
        for name, vectors in normalized.items()
    }

    profiles: list[FolderProfile] = []
    for name in names:
        vectors = normalized[name]
        insider_floor = None
        if vectors.shape[0] >= 2:
            total = vectors.sum(axis=0)
            leave_one_out = normalize_embeddings(
                (total[None, :] - vectors) / (vectors.shape[0] - 1)
            )
            insiders = np.sum(vectors * leave_one_out, axis=1)
            insider_floor = float(np.percentile(insiders, FOLDER_FLOOR_PERCENTILE))
        others = [normalized[other] for other in names if other != name]
        outsider_ceiling = None
        if others:
            outsiders = np.concatenate(others) @ centroids[name]
            outsider_ceiling = float(
                np.percentile(outsiders, OUTSIDER_CEILING_PERCENTILE)
            )
        floor = insider_floor
        if (
            insider_floor is not None
            and outsider_ceiling is not None
            and insider_floor > outsider_ceiling
        ):
            floor = (insider_floor + outsider_ceiling) / 2
        profiles.append(
            FolderProfile(
                name=name,
                member_count=int(vectors.shape[0]),
                centroid=centroids[name],
                floor=floor,
                insider_floor=insider_floor,
                outsider_ceiling=outsider_ceiling,
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
