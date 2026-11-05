"""聚类模块"""

import concurrent.futures
import dataclasses
from pathlib import Path

import hdbscan
import numpy as np
from openai import OpenAI

from dite.app.config import Config
from dite.app.i18n import t
from dite.cluster.name import (
    CLUSTER_NAME_CONTENT_LIMIT as CLUSTER_NAME_CONTENT_LIMIT,
)
from dite.cluster.name import (
    CLUSTER_NAME_EXCERPT_LIMIT as CLUSTER_NAME_EXCERPT_LIMIT,
)
from dite.cluster.name import (
    CLUSTER_NAME_OUTPUT_LIMIT as CLUSTER_NAME_OUTPUT_LIMIT,
)
from dite.cluster.name import (
    _build_cluster_debug_labels,
    _cluster_name_from_async_response,
    _prepare_cluster_name_request_inputs,
    _unique_cluster_names,
    _warn_on_cluster_naming_fallbacks,
    merge_clusters_by_name,
)
from dite.cluster.name import (
    _build_cluster_naming_prompt as _build_cluster_naming_prompt,
)
from dite.cluster.name import (
    _cluster_debug_token as _cluster_debug_token,
)
from dite.cluster.name import (
    _compact_sample_for_naming as _compact_sample_for_naming,
)
from dite.cluster.name import (
    _extract_title_like_line as _extract_title_like_line,
)
from dite.cluster.name import (
    _fallback_name_from_file_name as _fallback_name_from_file_name,
)
from dite.cluster.name import (
    _heuristic_cluster_name as _heuristic_cluster_name,
)
from dite.cluster.name import (
    _is_invalid_cluster_name as _is_invalid_cluster_name,
)
from dite.cluster.name import (
    _looks_like_author_line as _looks_like_author_line,
)
from dite.cluster.name import (
    _normalize_cluster_name_text as _normalize_cluster_name_text,
)
from dite.cluster.post import (
    ALL_NOISE_DISTANCE_THRESHOLD as ALL_NOISE_DISTANCE_THRESHOLD,
)
from dite.cluster.post import (
    KNN_DYNAMIC_THRESHOLD_MULTIPLIER as KNN_DYNAMIC_THRESHOLD_MULTIPLIER,
)
from dite.cluster.post import (
    cluster_centroids as _cluster_centroids_impl,
)
from dite.cluster.post import (
    cluster_sizes as _cluster_sizes_impl,
)
from dite.cluster.post import (
    merge_small_clusters_by_similarity as _merge_small_clusters_by_similarity_impl,
)
from dite.cluster.post import (
    repair_all_noise_with_similarity as _repair_all_noise_with_similarity_impl,
)
from dite.cluster.post import (
    repair_noise_with_knn as _repair_noise_with_knn_impl,
)
from dite.doc.embed import normalize_embeddings
from dite.util.log import get_logger

from .algo import cluster_documents as _cluster_documents_impl
from .model import (
    ClusterMetrics as ClusterMetrics,
)
from .model import (
    ClusterResult,
    SmallClusterMergeEvent,
    SmallClusterSkipEvent,
)

__all__ = [
    "ALL_NOISE_DISTANCE_THRESHOLD",
    "CLUSTER_NAME_CONTENT_LIMIT",
    "CLUSTER_NAME_EXCERPT_LIMIT",
    "CLUSTER_NAME_OUTPUT_LIMIT",
    "KNN_DYNAMIC_THRESHOLD_MULTIPLIER",
    "ClusterMetrics",
    "ClusterResult",
    "_build_cluster_debug_labels",
    "_build_cluster_naming_prompt",
    "_cluster_debug_token",
    "_compact_sample_for_naming",
    "_extract_title_like_line",
    "_fallback_name_from_file_name",
    "_looks_like_author_line",
    "cluster_documents",
    "generate_all_cluster_names",
    "generate_cluster_name",
    "merge_clusters_by_name",
    "merge_small_clusters_by_similarity",
    "repair_noise_with_knn",
]


def _cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Return cluster sizes excluding noise."""
    return _cluster_sizes_impl(labels)


def _cluster_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Build normalized centroid embeddings for non-noise clusters."""
    return _cluster_centroids_impl(embeddings, labels, normalize_embeddings)


def merge_small_clusters_by_similarity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    max_size: int,
    cosine_threshold: float,
) -> tuple[
    np.ndarray,
    int,
    list[SmallClusterMergeEvent],
    list[SmallClusterSkipEvent],
    float | None,
]:
    """Merge tiny non-noise clusters into the most similar nearby cluster."""
    return _merge_small_clusters_by_similarity_impl(
        embeddings,
        labels,
        max_size=max_size,
        cosine_threshold=cosine_threshold,
        build_cluster_debug_labels=_build_cluster_debug_labels,
        normalize_embeddings=normalize_embeddings,
        SmallClusterMergeEvent=SmallClusterMergeEvent,
        SmallClusterSkipEvent=SmallClusterSkipEvent,
        t=t,
    )


def _merge_small_clusters_by_similarity_for_algorithm(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    max_size: int,
    cosine_threshold: float,
) -> tuple[
    np.ndarray,
    int,
    list[SmallClusterMergeEvent],
    list[SmallClusterSkipEvent],
    float | None,
]:
    """Internal bridge for the extracted algorithm module."""
    return merge_small_clusters_by_similarity(
        embeddings,
        labels,
        max_size=max_size,
        cosine_threshold=cosine_threshold,
    )


def repair_noise_with_knn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
    distance_threshold: float | None = None,
    item_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    使用 k-NN 修复 HDBSCAN 的噪音点（参考论文 §3.3.3）

    在标准 k-NN 修复基础上增加距离阈值保护：
    1. 在非噪音点上训练 k-NN
    2. 为噪音点预测标签并计算最近邻距离
    3. 仅当距离 <= 阈值时归入簇，否则保持噪音

    Args:
        embeddings: 向量矩阵 (N, D)
        labels: HDBSCAN 输出的标签数组
        k: k-NN 的邻居数（默认3，使用多数投票）
        distance_threshold: 余弦距离阈值。None 时使用动态阈值。

    Returns:
        修复后的标签数组和修复的噪音点数量
    """
    return _repair_noise_with_knn_impl(
        embeddings,
        labels,
        k=k,
        distance_threshold=distance_threshold,
        item_names=item_names,
        build_cluster_debug_labels=_build_cluster_debug_labels,
        t=t,
    )


def _repair_noise_with_knn_for_algorithm(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
    distance_threshold: float | None = None,
    item_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """Internal bridge for the extracted algorithm module."""
    return repair_noise_with_knn(
        embeddings,
        labels,
        k=k,
        distance_threshold=distance_threshold,
        item_names=item_names,
    )


def repair_all_noise_with_similarity(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray, int]:
    """Group an all-noise HDBSCAN result by conservative cosine connectivity."""
    return _repair_all_noise_with_similarity_impl(
        embeddings,
        min_cluster_size=min_cluster_size,
        distance_threshold=distance_threshold,
    )


def _repair_all_noise_with_similarity_for_algorithm(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray, int]:
    """Internal bridge for the extracted algorithm module."""
    return repair_all_noise_with_similarity(
        embeddings,
        min_cluster_size=min_cluster_size,
        distance_threshold=distance_threshold,
    )


def cluster_documents(
    embeddings: np.ndarray,
    *,
    config: Config,
    repair_noise: bool = True,
    knn_k: int | None = None,
    knn_distance_threshold: float | None = None,
    clustering=None,
    allow_single_cluster: bool = False,
    item_names: list[str] | None = None,
    cluster_inputs: np.ndarray | None = None,
) -> ClusterResult:
    """Compatibility facade around the extracted clustering algorithm module."""
    return _cluster_documents_impl(
        embeddings,
        config=config,
        repair_noise=repair_noise,
        knn_k=knn_k,
        knn_distance_threshold=knn_distance_threshold,
        clustering=clustering,
        allow_single_cluster=allow_single_cluster,
        item_names=item_names,
        cluster_inputs=cluster_inputs,
        hdbscan_module=hdbscan,
        repair_noise_with_knn_fn=repair_noise_with_knn,
        repair_all_noise_with_similarity_fn=repair_all_noise_with_similarity,
        merge_small_clusters_by_similarity_fn=merge_small_clusters_by_similarity,
    )


def generate_cluster_name(
    client: OpenAI,
    cluster_embeddings: np.ndarray | None,
    sample_contents: list[str],
    sample_names: list[str],
    *,
    config: Config,
    top_k: int = 5,
    llm_model: str | None = None,
) -> str:
    """Compatibility wrapper around the extracted naming implementation."""
    from dite.cluster.name import generate_cluster_name as _generate_cluster_name

    return _generate_cluster_name(
        client,
        cluster_embeddings,
        sample_contents,
        sample_names,
        config=config,
        top_k=top_k,
        llm_model=llm_model,
    )


def generate_all_cluster_names(
    client: OpenAI,
    result: ClusterResult,
    contents: list[str],
    files: list[Path],
    *,
    config: Config,
    embeddings: np.ndarray | None = None,
    merge_same_name: bool = True,
    llm_model: str | None = None,
    request_runtime=None,
) -> ClusterResult:
    """Compatibility wrapper that preserves monkeypatch points in tests."""
    from dite.cluster.name import (
        _display_cluster_name,
    )
    from dite.util.api import ChatCompletionRequest
    from dite.util.llm import build_chat_completion_kwargs

    labels = result.labels
    cluster_labels = sorted(int(label) for label in set(labels) if label != -1)
    cluster_names: dict[int, str] = {}
    if not cluster_labels:
        return ClusterResult(
            labels=labels.copy(),
            cluster_names={},
            repaired_mask=result.repaired_mask.copy(),
            metrics=dataclasses.replace(result.metrics),
        )

    debug_labels = _build_cluster_debug_labels(labels)
    cluster_indices_by_label = {label: [] for label in cluster_labels}
    for index, label in enumerate(labels):
        if label == -1:
            continue
        cluster_indices_by_label[int(label)].append(index)

    logger = get_logger()
    naming_outcomes: list[str] = []
    typed_runtime = request_runtime
    if typed_runtime is None:
        _ = client.chat.completions

        def name_cluster(label: int) -> tuple[int, str, int]:
            cluster_indices = cluster_indices_by_label[label]
            cluster_contents = [contents[i] for i in cluster_indices]
            cluster_file_names = [files[i].name for i in cluster_indices]
            cluster_embeddings = None
            if embeddings is not None:
                cluster_embeddings = embeddings[cluster_indices]

            name = generate_cluster_name(
                client,
                cluster_embeddings,
                cluster_contents,
                cluster_file_names,
                config=config,
                llm_model=llm_model,
            )
            naming_outcomes.append(name)
            return label, _display_cluster_name(label, name), len(cluster_indices)

        max_workers = max(
            1,
            min(config.processing.cluster_naming_workers, len(cluster_labels)),
        )
        if max_workers == 1:
            results = [name_cluster(label) for label in cluster_labels]
        else:
            results_by_label: dict[int, tuple[str, int]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_label = {
                    executor.submit(name_cluster, label): label
                    for label in cluster_labels
                }
                for future in concurrent.futures.as_completed(future_to_label):
                    label, display_name, count = future.result()
                    results_by_label[label] = (display_name, count)
            results = [(label, *results_by_label[label]) for label in cluster_labels]
    else:
        request_profile = config.request_profiles.cluster_naming
        requests: list[ChatCompletionRequest] = []
        request_meta: list[tuple[int, list[str], list[str], int]] = []
        model = llm_model or config.models.llm

        for label in cluster_labels:
            cluster_indices = cluster_indices_by_label[label]
            cluster_contents = [contents[i] for i in cluster_indices]
            cluster_file_names = [files[i].name for i in cluster_indices]
            cluster_embeddings = None
            if embeddings is not None:
                cluster_embeddings = embeddings[cluster_indices]

            prompt, selected_contents, selected_names = (
                _prepare_cluster_name_request_inputs(
                    cluster_embeddings,
                    cluster_contents,
                    cluster_file_names,
                    top_k=5,
                )
            )
            request_meta.append(
                (label, selected_contents, selected_names, len(cluster_indices))
            )
            requests.append(
                ChatCompletionRequest(
                    kwargs=build_chat_completion_kwargs(
                        client=client,
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        profile=request_profile,
                        reasoning_param_style=config.api.reasoning_param_style,
                    )
                )
            )

        response_results = typed_runtime.run_cluster_naming_batch(requests)
        results = []
        for (label, selected_contents, selected_names, count), response in zip(
            request_meta,
            response_results,
            strict=False,
        ):
            name = _cluster_name_from_async_response(
                response,
                selected_contents,
                selected_names,
                debug_labels.get(label, str(label)),
            )
            naming_outcomes.append(name)
            results.append((label, _display_cluster_name(label, name), count))

    _warn_on_cluster_naming_fallbacks(naming_outcomes)
    for label, display_name, count in results:
        cluster_names[label] = display_name
        logger.debug(
            t(
                "debug_cluster_name_result",
                label=debug_labels.get(label, str(label)),
                name=display_name,
                count=count,
            )
        )

    merged_count = 0
    updated_labels = labels.copy()
    updated_names = cluster_names
    if merge_same_name:
        updated_labels, updated_names, merged_count = merge_clusters_by_name(
            labels, cluster_names
        )
    updated_names = _unique_cluster_names(updated_names)
    updated_metrics = dataclasses.replace(
        result.metrics,
        name_clusters_merged=merged_count,
    )
    return ClusterResult(
        labels=updated_labels,
        cluster_names=updated_names,
        repaired_mask=result.repaired_mask.copy(),
        metrics=updated_metrics,
    )
