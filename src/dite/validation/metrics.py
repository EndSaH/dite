"""Validation and structure metrics for clustering experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from .manifest import ValidationCorpus

if TYPE_CHECKING:
    from dite.flow.api import PipelineResult

try:
    from hdbscan.validity import validity_index as _hdbscan_validity_index
except Exception:  # pragma: no cover - optional import in test env
    _hdbscan_validity_index = None


@dataclass(frozen=True)
class ConstraintMetrics:
    """External validation metrics derived from labeled corpus manifests."""

    must_link_total: int = 0
    must_link_recall: float | None = None
    must_not_link_total: int = 0
    must_not_link_violations: int = 0
    must_not_link_violation_rate: float | None = None
    cluster_id_fragmentation_total: int = 0
    cluster_id_fragmentation_by_id: dict[str, int] = field(default_factory=dict)
    cluster_id_purity_by_id: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StructureMetrics:
    """Internal structure metrics for clustering experiments."""

    density_validation_score: float | None = None
    filename_bias_rate: float | None = None
    similarity_profile: dict[str, float] | None = None


@dataclass(frozen=True)
class PartitionMetrics:
    """Standard external clustering metrics against manifest ``cluster_id``s.

    Noise is never a cluster: a file left as noise (-1), or labeled with a
    gold noise id such as ``noise_misc``, counts as its own singleton. That
    way leaving files unsorted costs recall but never precision, and gold
    noise files are not rewarded for being lumped together.

    BCubed precision drops when unrelated files are merged; BCubed recall
    drops when a gold group is split or left as noise.
    """

    num_labeled_files: int = 0
    num_evaluated_files: int = 0
    adjusted_rand_index: float | None = None
    adjusted_mutual_info: float | None = None
    bcubed_precision: float | None = None
    bcubed_recall: float | None = None
    bcubed_f1: float | None = None
    predicted_noise_rate: float | None = None


def is_gold_noise_cluster_id(cluster_id: str) -> bool:
    """Return whether a manifest ``cluster_id`` marks files with no group."""
    lowered = cluster_id.casefold()
    return lowered == "noise" or lowered.startswith("noise_")


def _assignments_by_relative_path(
    result: PipelineResult,
    corpus: ValidationCorpus,
) -> dict[str, int]:
    """Map manifest-style relative paths (POSIX separators) to labels."""
    assignments: dict[str, int] = {}
    for path, label in zip(result.files, result.labels, strict=True):
        resolved = path.resolve()
        if not resolved.is_relative_to(corpus.root):
            continue
        assignments[resolved.relative_to(corpus.root).as_posix()] = int(label)
    return assignments


def _bcubed(gold: list[str], predicted: list[str]) -> tuple[float, float]:
    """Return BCubed precision and recall for two flat labelings."""
    pair_counts = Counter(zip(predicted, gold, strict=True))
    predicted_sizes = Counter(predicted)
    gold_sizes = Counter(gold)
    total = len(gold)
    precision = (
        sum(
            count * count / predicted_sizes[pred]
            for (pred, _gold), count in pair_counts.items()
        )
        / total
    )
    recall = (
        sum(
            count * count / gold_sizes[gold_id]
            for (_pred, gold_id), count in pair_counts.items()
        )
        / total
    )
    return precision, recall


def build_partition_metrics(
    result: PipelineResult,
    corpus: ValidationCorpus,
) -> PartitionMetrics:
    """Compare predicted labels with manifest ``cluster_id`` groups.

    Labeled files missing from the result are evaluated as predicted noise.
    """
    assignments = _assignments_by_relative_path(result, corpus)
    gold: list[str] = []
    predicted: list[str] = []
    predicted_noise = 0
    for relative_path, cluster_id in sorted(corpus.cluster_ids_by_path.items()):
        label = assignments.get(relative_path, -1)
        if is_gold_noise_cluster_id(cluster_id):
            gold.append(f"gold-noise:{relative_path}")
        else:
            gold.append(f"gold:{cluster_id}")
        if label == -1:
            predicted_noise += 1
            predicted.append(f"pred-noise:{relative_path}")
        else:
            predicted.append(f"pred:{label}")

    total = len(gold)
    if total == 0:
        return PartitionMetrics()
    precision, recall = _bcubed(gold, predicted)
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    evaluated = sum(1 for path in corpus.cluster_ids_by_path if path in assignments)
    ari = ami = None
    if total >= 2:
        ari = float(adjusted_rand_score(gold, predicted))
        ami = float(adjusted_mutual_info_score(gold, predicted))
    return PartitionMetrics(
        num_labeled_files=total,
        num_evaluated_files=evaluated,
        adjusted_rand_index=ari,
        adjusted_mutual_info=ami,
        bcubed_precision=float(precision),
        bcubed_recall=float(recall),
        bcubed_f1=float(f1),
        predicted_noise_rate=predicted_noise / total,
    )


def partition_metrics_payload(
    metrics: PartitionMetrics,
) -> dict[str, float | int | None]:
    """Return a JSON-ready dict for reports and experiment output."""
    return {
        "num_labeled_files": metrics.num_labeled_files,
        "num_evaluated_files": metrics.num_evaluated_files,
        "adjusted_rand_index": metrics.adjusted_rand_index,
        "adjusted_mutual_info": metrics.adjusted_mutual_info,
        "bcubed_precision": metrics.bcubed_precision,
        "bcubed_recall": metrics.bcubed_recall,
        "bcubed_f1": metrics.bcubed_f1,
        "predicted_noise_rate": metrics.predicted_noise_rate,
    }


def build_constraint_metrics(
    result: PipelineResult,
    corpus: ValidationCorpus,
) -> ConstraintMetrics:
    """Build external validation metrics using the loaded manifest constraints."""
    assignments = _assignments_by_relative_path(result, corpus)

    must_link_hits = 0
    for left, right in corpus.constraints.must_link_pairs:
        left_label = assignments.get(left, -1)
        right_label = assignments.get(right, -1)
        if left_label != -1 and left_label == right_label:
            must_link_hits += 1

    must_not_link_violations = 0
    for left, right in corpus.constraints.must_not_link_pairs:
        left_label = assignments.get(left, -1)
        right_label = assignments.get(right, -1)
        if left_label != -1 and left_label == right_label:
            must_not_link_violations += 1

    members_by_cluster_id: dict[str, list[int]] = defaultdict(list)
    for relative_path, cluster_id in corpus.cluster_ids_by_path.items():
        members_by_cluster_id[cluster_id].append(assignments.get(relative_path, -1))

    fragmentation_by_id: dict[str, int] = {}
    purity_by_id: dict[str, float] = {}
    fragmentation_total = 0
    for cluster_id, labels in members_by_cluster_id.items():
        non_noise_labels = [label for label in labels if label != -1]
        unique_labels = sorted(set(non_noise_labels))
        fragmentation = max(0, len(unique_labels) - 1)
        fragmentation_by_id[cluster_id] = fragmentation
        fragmentation_total += fragmentation

        counts = Counter(labels)
        majority = counts.most_common(1)[0][1] if counts else 0
        purity_by_id[cluster_id] = majority / len(labels) if labels else 0.0

    must_link_total = len(corpus.constraints.must_link_pairs)
    must_not_link_total = len(corpus.constraints.must_not_link_pairs)
    return ConstraintMetrics(
        must_link_total=must_link_total,
        must_link_recall=(
            must_link_hits / must_link_total if must_link_total else None
        ),
        must_not_link_total=must_not_link_total,
        must_not_link_violations=must_not_link_violations,
        must_not_link_violation_rate=(
            must_not_link_violations / must_not_link_total
            if must_not_link_total
            else None
        ),
        cluster_id_fragmentation_total=fragmentation_total,
        cluster_id_fragmentation_by_id=fragmentation_by_id,
        cluster_id_purity_by_id=purity_by_id,
    )


def build_structure_metrics(
    result: PipelineResult,
    *,
    embeddings: np.ndarray | None = None,
) -> StructureMetrics:
    """Build internal structure metrics from a pipeline result."""
    labels = np.asarray(result.labels, dtype=int)
    vectors = np.asarray(embeddings if embeddings is not None else result.embeddings)

    density_validation_score = _compute_density_validation_score(vectors, labels)
    filename_bias_rate = _compute_filename_bias_rate(result)
    return StructureMetrics(
        density_validation_score=density_validation_score,
        filename_bias_rate=filename_bias_rate,
        similarity_profile=compute_similarity_profile(vectors),
    )


SIMILARITY_PROFILE_SAMPLE = 2000


def compute_similarity_profile(embeddings: np.ndarray) -> dict[str, float] | None:
    """Summarize the corpus cosine-similarity distribution.

    The clustering thresholds (content edges, small-cluster merge) are
    absolute cosine values, but each embedding model uses its own similarity
    range. These percentiles show where a threshold such as 0.92 falls for
    this corpus and model: if ``pairwise_p99`` is already above it, unrelated
    pairs will pass. Large corpora are sampled to keep this O(sample^2).
    """
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] == 0:
        return None
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 0]
    if vectors.shape[0] < 2:
        return None
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.shape[0] > SIMILARITY_PROFILE_SAMPLE:
        rng = np.random.default_rng(0)
        rows = rng.choice(vectors.shape[0], SIMILARITY_PROFILE_SAMPLE, replace=False)
        vectors = vectors[np.sort(rows)]
    similarities = vectors @ vectors.T
    upper = similarities[np.triu_indices(similarities.shape[0], k=1)]
    np.fill_diagonal(similarities, -np.inf)
    nearest = similarities.max(axis=1)
    return {
        "num_documents": int(vectors.shape[0]),
        "pairwise_p50": float(np.percentile(upper, 50)),
        "pairwise_p90": float(np.percentile(upper, 90)),
        "pairwise_p99": float(np.percentile(upper, 99)),
        "nearest_neighbor_p10": float(np.percentile(nearest, 10)),
        "nearest_neighbor_p50": float(np.percentile(nearest, 50)),
        "nearest_neighbor_p90": float(np.percentile(nearest, 90)),
    }


def _compute_density_validation_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    if _hdbscan_validity_index is None:
        return None
    if embeddings.size == 0 or labels.size == 0:
        return None
    non_noise = [label for label in set(labels.tolist()) if label != -1]
    if len(non_noise) < 2:
        return None
    try:
        return float(_hdbscan_validity_index(embeddings, labels))
    except Exception:
        return None


def _compute_filename_bias_rate(result: PipelineResult) -> float | None:
    # Imported here: dite.cluster pulls in dite.app, whose CLI imports this
    # module at load time.
    from dite.cluster.model import CONTENT_BACKED_EDGE_TYPES

    if not result.document_features or not result.candidate_edges:
        return None

    label_by_id = {
        feature.file_id: int(label)
        for feature, label in zip(result.document_features, result.labels, strict=True)
    }
    cluster_content_support: dict[int, bool] = defaultdict(bool)
    cluster_metadata_only_support: dict[int, bool] = defaultdict(bool)

    for edge in result.candidate_edges:
        left_label = label_by_id.get(edge.source_id, -1)
        right_label = label_by_id.get(edge.target_id, -1)
        if left_label == -1 or right_label == -1 or left_label != right_label:
            continue
        if edge.edge_type in CONTENT_BACKED_EDGE_TYPES:
            cluster_content_support[left_label] = True
        elif edge.edge_type in {"filename_similarity", "title_match"}:
            cluster_metadata_only_support[left_label] = True

    biased = 0
    eligible = 0
    for feature in result.document_features:
        label = label_by_id.get(feature.file_id, -1)
        if label == -1:
            continue
        eligible += 1
        if cluster_metadata_only_support.get(
            label, False
        ) and not cluster_content_support.get(label, False):
            biased += 1

    if eligible == 0:
        return None
    return biased / eligible
