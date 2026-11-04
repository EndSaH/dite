"""Core clustering algorithm implementation."""

from __future__ import annotations

import hdbscan
import numpy as np

from dite.app.config import ClusteringConfig, Config
from dite.app.i18n import t
from dite.cluster.name import _build_cluster_debug_labels
from dite.cluster.post import cluster_sizes
from dite.doc.embed import normalize_embeddings
from dite.util.log import get_logger

from .model import ClusterMetrics, ClusterResult


def cluster_documents(
    embeddings: np.ndarray,
    *,
    config: Config,
    repair_noise: bool = True,
    knn_k: int | None = None,
    knn_distance_threshold: float | None = None,
    clustering: ClusteringConfig | None = None,
    allow_single_cluster: bool = False,
    item_names: list[str] | None = None,
    cluster_inputs: np.ndarray | None = None,
    hdbscan_module=hdbscan,
    repair_noise_with_knn_fn=None,
    repair_all_noise_with_similarity_fn=None,
    merge_small_clusters_by_similarity_fn=None,
) -> ClusterResult:
    """Cluster documents with HDBSCAN plus conservative post-processing.

    ``cluster_inputs`` optionally replaces the space HDBSCAN runs in (for
    example PCA or UMAP coordinates). Noise repair and small-cluster merging
    always use the L2-normalized ``embeddings``, because their cosine
    thresholds are defined in the original embedding space.
    """
    from . import api as clusterer_module

    if repair_noise_with_knn_fn is None:
        repair_noise_with_knn_fn = clusterer_module._repair_noise_with_knn_for_algorithm
    if repair_all_noise_with_similarity_fn is None:
        repair_all_noise_with_similarity_fn = (
            clusterer_module._repair_all_noise_with_similarity_for_algorithm
        )
    if merge_small_clusters_by_similarity_fn is None:
        merge_small_clusters_by_similarity_fn = (
            clusterer_module._merge_small_clusters_by_similarity_for_algorithm
        )

    logger = get_logger()
    clustering_cfg = clustering or config.clustering
    embeddings = normalize_embeddings(embeddings)
    effective_knn_k = knn_k if knn_k is not None else clustering_cfg.knn_k
    effective_distance_threshold = (
        knn_distance_threshold
        if knn_distance_threshold is not None
        else clustering_cfg.knn_distance_threshold
    )

    logger.debug(t("debug_cluster_hdbscan_header"))
    logger.debug(
        t(
            "debug_cluster_hdbscan_min_cluster_size",
            value=clustering_cfg.min_cluster_size,
        )
    )
    logger.debug(
        t("debug_cluster_hdbscan_min_samples", value=clustering_cfg.min_samples)
    )
    logger.debug(
        t(
            "debug_cluster_hdbscan_epsilon",
            value=clustering_cfg.cluster_selection_epsilon,
        )
    )
    logger.debug(
        t(
            "debug_cluster_hdbscan_method",
            value=clustering_cfg.cluster_selection_method,
        )
    )
    logger.debug(
        t(
            "debug_cluster_input_vectors",
            count=embeddings.shape[0],
            dimension=embeddings.shape[1],
        )
    )

    if embeddings.shape[0] < clustering_cfg.min_cluster_size:
        labels = np.full(embeddings.shape[0], -1, dtype=int)
        return ClusterResult(
            labels=labels,
            cluster_names={},
            repaired_mask=np.zeros(labels.shape, dtype=bool),
            metrics=ClusterMetrics(initial_noise=int(labels.size)),
        )

    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=clustering_cfg.min_cluster_size,
        min_samples=clustering_cfg.min_samples,
        cluster_selection_epsilon=clustering_cfg.cluster_selection_epsilon,
        metric="euclidean",
        cluster_selection_method=clustering_cfg.cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    hdbscan_inputs = embeddings
    if cluster_inputs is not None:
        hdbscan_inputs = np.asarray(cluster_inputs, dtype=np.float64)
        if hdbscan_inputs.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "cluster_inputs must have one row per embedding: "
                f"{hdbscan_inputs.shape[0]} != {embeddings.shape[0]}"
            )
    labels = clusterer.fit_predict(hdbscan_inputs)

    unique_labels = set(labels)
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = int(np.sum(labels == -1))
    logger.debug(t("debug_cluster_initial_result", clusters=n_clusters, noise=n_noise))
    metrics = ClusterMetrics(initial_clusters=n_clusters, initial_noise=n_noise)

    debug_labels = _build_cluster_debug_labels(labels)
    if n_clusters > 0:
        cluster_size_labels = []
        for label in sorted(label for label in unique_labels if label != -1):
            size = int(np.sum(labels == label))
            cluster_size_labels.append(f"{debug_labels[int(label)]}:{size}")
        logger.debug(t("debug_cluster_sizes", sizes=", ".join(cluster_size_labels)))

    repaired_count = 0
    repaired_mask = np.zeros(labels.shape, dtype=bool)
    if repair_noise:
        before_repair = labels.copy()
        if n_clusters == 0 and n_noise > 0:
            labels, repaired_count = repair_all_noise_with_similarity_fn(
                embeddings,
                min_cluster_size=clustering_cfg.min_cluster_size,
                distance_threshold=effective_distance_threshold,
            )
        else:
            labels, repaired_count = repair_noise_with_knn_fn(
                embeddings,
                labels,
                k=effective_knn_k,
                distance_threshold=effective_distance_threshold,
                item_names=item_names,
            )
        repaired_mask = (before_repair == -1) & (labels != -1)
    metrics.noise_repaired = repaired_count

    if clustering_cfg.small_cluster_merge_enabled:
        labels_before_small_merge = labels.copy()
        (
            labels,
            metrics.small_clusters_merged,
            metrics.small_cluster_merge_events,
            metrics.small_cluster_skip_events,
            metrics.small_cluster_merge_max_similarity,
        ) = merge_small_clusters_by_similarity_fn(
            embeddings,
            labels,
            max_size=clustering_cfg.small_cluster_merge_max_size,
            cosine_threshold=clustering_cfg.small_cluster_merge_cosine_threshold,
        )
        metrics.small_cluster_merge_candidates = int(
            sum(
                1
                for size in cluster_sizes(labels_before_small_merge).values()
                if size <= clustering_cfg.small_cluster_merge_max_size
            )
        )
        metrics.small_cluster_merge_skipped = len(metrics.small_cluster_skip_events)

    return ClusterResult(
        labels=labels,
        cluster_names={},
        repaired_mask=repaired_mask,
        metrics=metrics,
    )


AGGLOMERATIVE_MAX_AUTO_CLUSTERS = 50
AGGLOMERATIVE_SILHOUETTE_SAMPLE = 2000


def _relabel_by_size_floor(raw_labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Map tree-cut ids to 0..k-1, turning groups below the floor into noise.

    Clusters are numbered in order of their first member, so labels are
    deterministic for a given input order.
    """
    sizes: dict[int, int] = {}
    for raw in raw_labels.tolist():
        sizes[raw] = sizes.get(raw, 0) + 1
    mapping: dict[int, int] = {}
    labels = np.full(raw_labels.shape[0], -1, dtype=int)
    for index, raw in enumerate(raw_labels.tolist()):
        if sizes[raw] < min_cluster_size:
            continue
        if raw not in mapping:
            mapping[raw] = len(mapping)
        labels[index] = mapping[raw]
    return labels


def _count_sized_groups(raw_labels: np.ndarray, min_cluster_size: int) -> int:
    sizes: dict[int, int] = {}
    for raw in raw_labels.tolist():
        sizes[raw] = sizes.get(raw, 0) + 1
    return sum(1 for size in sizes.values() if size >= min_cluster_size)


def _cut_for_target_clusters(
    tree: np.ndarray,
    count: int,
    target: int,
    min_cluster_size: int,
) -> tuple[np.ndarray, int]:
    """Cut the tree so that ``target`` groups reach ``min_cluster_size``.

    A plain ``maxclust`` cut counts isolated outliers as clusters, so a few
    noise documents would use up the folder budget and force real topics to
    merge. Finer cuts are tried until enough real groups appear; the cut
    closest to the target wins, preferring the coarser one.
    """
    from scipy.cluster.hierarchy import fcluster

    best_raw = fcluster(tree, t=target, criterion="maxclust")
    best_k = target
    best_gap = abs(_count_sized_groups(best_raw, min_cluster_size) - target)
    for k in range(target + 1, count + 1):
        if best_gap == 0:
            break
        raw = fcluster(tree, t=k, criterion="maxclust")
        groups = _count_sized_groups(raw, min_cluster_size)
        gap = abs(groups - target)
        if gap < best_gap:
            best_raw, best_k, best_gap = raw, k, gap
        if groups > target:
            break
    return best_raw, best_k


def _auto_cut_by_silhouette(
    tree: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, int | None, float | None]:
    """Pick the tree cut whose cosine silhouette is highest.

    Silhouette compares within-cluster and nearest-other-cluster distances of
    each document, so the choice does not depend on the absolute similarity
    scale of a particular embedding model. Ties keep the smaller count.
    """
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import silhouette_score

    count = vectors.shape[0]
    best_labels = np.ones(count, dtype=int)
    best_k: int | None = None
    best_score: float | None = None
    max_k = min(count - 1, AGGLOMERATIVE_MAX_AUTO_CLUSTERS)
    sample_size = (
        AGGLOMERATIVE_SILHOUETTE_SAMPLE
        if count > AGGLOMERATIVE_SILHOUETTE_SAMPLE
        else None
    )
    for k in range(2, max_k + 1):
        candidate = fcluster(tree, t=k, criterion="maxclust")
        if len(set(candidate.tolist())) < 2:
            continue
        score = float(
            silhouette_score(
                vectors,
                candidate,
                metric="cosine",
                sample_size=sample_size,
                random_state=0,
            )
        )
        if best_score is None or score > best_score:
            best_score = score
            best_k = k
            best_labels = candidate
    return best_labels, best_k, best_score


def cluster_documents_agglomerative(
    embeddings: np.ndarray,
    *,
    config: Config,
    clustering: ClusteringConfig | None = None,
    target_clusters: int | None = None,
    distance_threshold: float | None = None,
) -> ClusterResult:
    """Cluster documents by cutting a cosine average-linkage tree.

    Unlike HDBSCAN this needs no density estimate, so it behaves the same in
    high-dimensional embedding space and on small folders. The cut is chosen
    by ``target_clusters`` (a folder count), else ``distance_threshold``
    (cosine distance), else the best cosine silhouette. Groups smaller than
    ``clustering.min_cluster_size`` become noise and stay unsorted.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    logger = get_logger()
    clustering_cfg = clustering or config.clustering
    vectors = normalize_embeddings(embeddings).astype(np.float64)
    count = vectors.shape[0]
    min_size = max(1, clustering_cfg.min_cluster_size)
    if count < max(2, min_size):
        labels = np.full(count, -1, dtype=int)
        return ClusterResult(
            labels=labels,
            cluster_names={},
            repaired_mask=np.zeros(count, dtype=bool),
            metrics=ClusterMetrics(initial_noise=count),
        )

    # Zero vectors (failed embeddings) have undefined cosine distance; treat
    # them as unrelated to everything rather than letting NaN break linkage.
    distances = np.nan_to_num(pdist(vectors, metric="cosine"), nan=1.0)
    distances = np.clip(distances, 0.0, 2.0)
    tree = linkage(distances, method="average")

    if target_clusters is not None:
        raw, cut_k = _cut_for_target_clusters(
            tree, count, max(1, int(target_clusters)), min_size
        )
        cut_description = f"target_clusters={target_clusters} (tree cut k={cut_k})"
    elif distance_threshold is not None:
        raw = fcluster(tree, t=float(distance_threshold), criterion="distance")
        cut_description = f"distance_threshold={distance_threshold}"
    else:
        raw, best_k, best_score = _auto_cut_by_silhouette(tree, vectors)
        cut_description = (
            f"auto silhouette k={best_k} score={best_score:.3f}"
            if best_k is not None and best_score is not None
            else "auto silhouette (no valid cut, single group)"
        )

    labels = _relabel_by_size_floor(np.asarray(raw), min_size)
    n_clusters = len({int(label) for label in labels if label != -1})
    n_noise = int(np.sum(labels == -1))
    logger.debug(
        f"Agglomerative clustering: {cut_description}; "
        f"clusters={n_clusters}, noise={n_noise}"
    )
    return ClusterResult(
        labels=labels,
        cluster_names={},
        repaired_mask=np.zeros(count, dtype=bool),
        metrics=ClusterMetrics(initial_clusters=n_clusters, initial_noise=n_noise),
    )


def reduce_for_density_clustering(
    embeddings: np.ndarray,
    *,
    reducer: str,
    pca_components: int | None = None,
    umap_components: int = 10,
    umap_neighbors: int = 15,
) -> np.ndarray | None:
    """Project embeddings into a low-dimensional space for HDBSCAN.

    Returns ``None`` when no reduction applies (``reducer="none"`` or too few
    documents). Density estimates degrade in thousands of dimensions; BERTopic
    runs UMAP to a handful of dimensions before HDBSCAN for this reason.
    """
    count = embeddings.shape[0]
    if reducer == "pca":
        if pca_components is None:
            return None
        components = min(int(pca_components), count, embeddings.shape[1])
        if components < 1:
            return None
        from sklearn.decomposition import PCA

        return PCA(n_components=components, random_state=0).fit_transform(embeddings)

    if reducer == "umap":
        # UMAP's spectral layout needs a few more points than dimensions.
        components = min(int(umap_components), count - 2)
        neighbors = min(int(umap_neighbors), count - 1)
        if components < 2 or neighbors < 2:
            return None
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "topic_clustering.reducer = 'umap' needs the optional package "
                "umap-learn. Install it with `uv add umap-learn` (or "
                "`pip install umap-learn`), or set reducer to 'none' or 'pca'."
            ) from exc
        reducer_model = umap.UMAP(
            n_components=components,
            n_neighbors=neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        return reducer_model.fit_transform(embeddings)

    return None
