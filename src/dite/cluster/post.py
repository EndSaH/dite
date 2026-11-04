"""Clustering post-processing helpers for clustering V2."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier

from dite.util.log import get_logger

KNN_DYNAMIC_THRESHOLD_MULTIPLIER = 2.0
ALL_NOISE_DISTANCE_THRESHOLD = 0.20


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Return cluster sizes excluding noise."""
    return {
        int(label): int(np.sum(labels == label))
        for label in sorted(int(label) for label in set(labels) if label != -1)
    }


def cluster_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    normalize_embeddings,
) -> dict[int, np.ndarray]:
    """Build normalized centroid embeddings for non-noise clusters."""
    centroids: dict[int, np.ndarray] = {}
    for label in sorted(int(label) for label in set(labels) if label != -1):
        members = embeddings[labels == label]
        centroid = normalize_embeddings(members.mean(axis=0))
        centroids[label] = centroid
    return centroids


def merge_small_clusters_by_similarity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    max_size: int,
    cosine_threshold: float,
    build_cluster_debug_labels,
    normalize_embeddings,
    SmallClusterMergeEvent,
    SmallClusterSkipEvent,
    t,
) -> tuple[
    np.ndarray,
    int,
    list,
    list,
    float | None,
]:
    """Merge tiny non-noise clusters into the most similar nearby cluster."""
    logger = get_logger()
    labels = labels.copy()
    if max_size < 1:
        return labels, 0, [], [], None

    initial_sizes = cluster_sizes(labels)
    candidates = sorted(
        (
            (size, label)
            for label, size in initial_sizes.items()
            if size <= max_size
        ),
        key=lambda item: (item[0], item[1]),
    )
    debug_labels = build_cluster_debug_labels(labels)

    merged_count = 0
    merge_events = []
    skip_events = []
    max_similarity_seen: float | None = None
    for _size, source_label in candidates:
        current_cluster_sizes = cluster_sizes(labels)
        if len(current_cluster_sizes) < 2:
            break
        if source_label not in current_cluster_sizes:
            continue
        source_size = current_cluster_sizes[source_label]
        if source_size > max_size:
            continue

        centroids = cluster_centroids(embeddings, labels, normalize_embeddings)
        source_centroid = centroids[source_label]
        best_target: int | None = None
        best_similarity = -1.0
        best_target_size = -1
        for target_label, target_size in current_cluster_sizes.items():
            if target_label == source_label:
                continue
            target_centroid = centroids[target_label]
            similarity = float(np.dot(source_centroid, target_centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best_target = target_label
                best_target_size = target_size
                continue
            if similarity == best_similarity and target_size > best_target_size:
                best_target = target_label
                best_target_size = target_size
                continue
            if (
                similarity == best_similarity
                and target_size == best_target_size
                and best_target is not None
                and target_label < best_target
            ):
                best_target = target_label
        if best_similarity >= 0 and (
            max_similarity_seen is None or best_similarity > max_similarity_seen
        ):
            max_similarity_seen = best_similarity

        if best_target is None:
            logger.debug(
                t(
                    "debug_cluster_small_merge_skipped",
                    source=debug_labels.get(source_label, str(source_label)),
                    source_size=source_size,
                    target="-",
                    target_size=0,
                    similarity=-1.0,
                    reason="no_target_cluster",
                )
            )
            skip_events.append(
                SmallClusterSkipEvent(
                    source_label=source_label,
                    source_size=source_size,
                    best_target_label=None,
                    best_target_size=None,
                    best_similarity=None,
                    reason="no_target_cluster",
                )
            )
            continue
        if best_similarity < cosine_threshold:
            logger.debug(
                t(
                    "debug_cluster_small_merge_skipped",
                    source=debug_labels.get(source_label, str(source_label)),
                    source_size=source_size,
                    target=debug_labels.get(best_target, str(best_target)),
                    target_size=best_target_size,
                    similarity=best_similarity,
                    reason="below_similarity_threshold",
                )
            )
            skip_events.append(
                SmallClusterSkipEvent(
                    source_label=source_label,
                    source_size=source_size,
                    best_target_label=best_target,
                    best_target_size=best_target_size,
                    best_similarity=best_similarity,
                    reason="below_similarity_threshold",
                )
            )
            continue

        target_size_before = current_cluster_sizes[best_target]
        logger.debug(
            t(
                "debug_cluster_small_merge_event",
                source=debug_labels.get(source_label, str(source_label)),
                source_size=source_size,
                target=debug_labels.get(best_target, str(best_target)),
                target_size=target_size_before,
                similarity=best_similarity,
            )
        )
        labels[labels == source_label] = best_target
        merged_count += 1
        merge_events.append(
            SmallClusterMergeEvent(
                source_label=source_label,
                source_size=source_size,
                target_label=best_target,
                target_size_before=target_size_before,
                similarity=best_similarity,
            )
        )

    logger.debug(
        t(
            "debug_cluster_small_merge_summary",
            candidates=len(candidates),
            merged=merged_count,
            skipped=len(skip_events),
        )
    )
    return labels, merged_count, merge_events, skip_events, max_similarity_seen


def repair_noise_with_knn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
    distance_threshold: float | None = None,
    item_names: list[str] | None = None,
    *,
    build_cluster_debug_labels,
    t,
) -> tuple[np.ndarray, int]:
    """Repair HDBSCAN noise points with conservative k-NN assignment."""
    logger = get_logger()
    debug_labels = build_cluster_debug_labels(labels)
    labels = labels.copy()
    core_mask = labels != -1

    if core_mask.all() or not core_mask.any():
        return labels, 0

    X_core, y_core = embeddings[core_mask], labels[core_mask]
    X_noise = embeddings[~core_mask]
    noise_indices = np.where(~core_mask)[0]

    actual_k = min(k, len(X_core))
    if actual_k < 1:
        return labels, 0

    knn = KNeighborsClassifier(n_neighbors=actual_k, metric="cosine")
    knn.fit(X_core, y_core)

    predictions = knn.predict(X_noise)
    neighbor_distances, neighbor_indices = knn.kneighbors(X_noise)
    # Gate on the nearest neighbour that actually belongs to the voted
    # cluster. The overall nearest neighbour may sit in a different cluster
    # than the k-vote picked, which let a point join cluster A because it was
    # close to a member of cluster B.
    predicted_distances = np.empty(len(predictions), dtype=float)
    for row, pred in enumerate(predictions):
        same_cluster = y_core[neighbor_indices[row]] == pred
        predicted_distances[row] = float(np.min(neighbor_distances[row][same_cluster]))

    if distance_threshold is None:
        core_dist_matrix = cosine_distances(X_core, X_core)
        np.fill_diagonal(core_dist_matrix, np.inf)
        nearest_core_distance = np.min(core_dist_matrix, axis=1)
        mean_core_distance = float(np.mean(nearest_core_distance))
        distance_threshold = mean_core_distance * KNN_DYNAMIC_THRESHOLD_MULTIPLIER
        logger.debug(
            t(
                "debug_cluster_knn_dynamic_threshold",
                threshold=distance_threshold,
                mean_core_distance=mean_core_distance,
            )
        )
    else:
        logger.debug(
            t("debug_cluster_knn_fixed_threshold", threshold=distance_threshold)
        )

    repaired_count = 0
    for pred, dist, idx in zip(
        predictions, predicted_distances, noise_indices, strict=False
    ):
        item_name = item_names[idx] if item_names is not None else f"#{idx}"
        debug_label = debug_labels.get(int(pred), str(pred))
        if dist <= distance_threshold:
            labels[idx] = pred
            repaired_count += 1
            logger.debug(
                t(
                    "debug_cluster_knn_assignment",
                    index=idx,
                    name=item_name,
                    label=debug_label,
                    distance=float(dist),
                    threshold=distance_threshold,
                )
            )
        else:
            logger.debug(
                t(
                    "debug_cluster_knn_kept",
                    index=idx,
                    name=item_name,
                    label=debug_label,
                    distance=float(dist),
                    threshold=distance_threshold,
                )
            )

    kept_noise = len(noise_indices) - repaired_count
    logger.debug(
        t("debug_cluster_knn_summary", repaired=repaired_count, kept=kept_noise)
    )

    return labels, repaired_count


def repair_all_noise_with_similarity(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray, int]:
    """Group an all-noise HDBSCAN result by conservative cosine connectivity."""
    labels = np.full(embeddings.shape[0], -1, dtype=int)
    if embeddings.shape[0] < min_cluster_size:
        return labels, 0

    threshold = (
        ALL_NOISE_DISTANCE_THRESHOLD
        if distance_threshold is None
        else distance_threshold
    )
    distances = cosine_distances(embeddings, embeddings)
    visited: set[int] = set()
    next_label = 0
    repaired_count = 0

    for start in range(embeddings.shape[0]):
        if start in visited:
            continue
        component: list[int] = []
        stack = [start]
        visited.add(start)

        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = np.where(distances[current] <= threshold)[0]
            for neighbor in neighbors:
                neighbor_index = int(neighbor)
                if neighbor_index in visited:
                    continue
                visited.add(neighbor_index)
                stack.append(neighbor_index)

        if len(component) < min_cluster_size:
            continue
        for index in component:
            labels[index] = next_label
        repaired_count += len(component)
        next_label += 1

    return labels, repaired_count
