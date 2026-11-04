"""Canonical clustering stage helpers for the shared pipeline."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dite.app.config import Config
from dite.cluster import (
    apply_llm_adjudication,
    apply_rule_adjudication,
    build_adjudication_requests,
)
from dite.cluster.algo import (
    cluster_documents_agglomerative,
    reduce_for_density_clustering,
)
from dite.cluster.link import build_candidate_components, build_candidate_edges
from dite.cluster.model import (
    CONTENT_BACKED_EDGE_TYPES,
    AdjudicationDecision,
    AdjudicationRequest,
    CandidateComponent,
    CandidateEdge,
    ClusterDraft,
)
from dite.doc import DocumentFeatures, build_document_features
from dite.doc.embed import normalize_embeddings

from .model import ClusterMetrics, ClusterResult

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class CanonicalClusterStageResult:
    """Canonical clustering stage output before duplicate expansion."""

    canonical_embeddings: np.ndarray
    cluster_result: ClusterResult
    noise_repaired: int
    document_features: list[DocumentFeatures]
    candidate_edges: list[CandidateEdge]
    candidate_components: list[CandidateComponent]
    cluster_drafts: list[ClusterDraft]
    adjudication_requests: list[AdjudicationRequest]
    adjudication_decisions: list[AdjudicationDecision]


def build_cluster_drafts(
    labels: np.ndarray,
    *,
    canonical_document_features: list[DocumentFeatures],
    origin: str,
) -> list[ClusterDraft]:
    """Build explicit intermediate cluster drafts from current labels."""
    label_count = min(len(labels), len(canonical_document_features))
    labels = labels[:label_count]
    features = canonical_document_features[:label_count]
    file_ids_by_label: dict[int, list[str]] = {}
    noise_members: list[str] = []
    for label, feature in zip(labels, features, strict=True):
        if int(label) == -1:
            noise_members.append(feature.file_id)
            continue
        file_ids_by_label.setdefault(int(label), []).append(feature.file_id)

    drafts: list[ClusterDraft] = []
    for label in sorted(file_ids_by_label):
        drafts.append(
            ClusterDraft(
                draft_cluster_id=label,
                member_file_ids=sorted(file_ids_by_label[label]),
                origin=origin,
                noise_members=[],
                merge_candidates=[],
            )
        )
    if noise_members:
        drafts.append(
            ClusterDraft(
                draft_cluster_id=-1,
                member_file_ids=[],
                origin=origin,
                noise_members=sorted(noise_members),
                merge_candidates=[],
            )
        )
    return drafts


def apply_candidate_component_links(
    result: ClusterResult,
    canonical_document_features: list[DocumentFeatures],
    candidate_components: list[CandidateComponent],
) -> ClusterResult:
    """Apply conservative must-link merging from strong candidate components."""
    if not candidate_components:
        return result

    index_by_file_id = {
        feature.file_id: index
        for index, feature in enumerate(canonical_document_features)
    }
    labels = result.labels.copy()
    merged_any = False

    for component in candidate_components:
        if component.component_type != "near_duplicate_group":
            continue
        member_indices = [
            index_by_file_id[file_id]
            for file_id in component.member_file_ids
            if file_id in index_by_file_id
        ]
        if len(member_indices) < 2:
            continue
        component_labels = [int(labels[index]) for index in member_indices]
        non_noise_labels = sorted({label for label in component_labels if label != -1})
        if not non_noise_labels:
            continue
        target_label = non_noise_labels[0]
        for index in member_indices:
            if labels[index] != target_label:
                labels[index] = target_label
                merged_any = True

    if not merged_any:
        return result

    return ClusterResult(
        labels=labels,
        cluster_names=result.cluster_names.copy(),
        repaired_mask=result.repaired_mask.copy(),
        metrics=dataclasses.replace(result.metrics),
    )


ADJUDICATION_CLUSTER_MERGE_SUPPORT = 0.5


def apply_adjudication_links(
    result: ClusterResult,
    canonical_document_features: list[DocumentFeatures],
    adjudication_requests: list[AdjudicationRequest],
    adjudication_decisions: list[AdjudicationDecision],
    *,
    cluster_merge_support: float = ADJUDICATION_CLUSTER_MERGE_SUPPORT,
) -> ClusterResult:
    """Apply ``merge_edge`` decisions at cluster level.

    A merge edge is evidence about two documents, so it is applied without
    tearing either document out of the cluster it already belongs to:

    - noise linked to a clustered document joins that cluster (by majority
      when it links to several);
    - two different clusters merge only when merge edges reach at least
      ``cluster_merge_support`` of the smaller cluster's members, so one
      stray pair cannot fuse two topics or split a member off;
    - two noise documents stay noise (a pair is below any useful cluster
      size).
    """
    if not adjudication_decisions:
        return result

    request_by_id = {request.request_id: request for request in adjudication_requests}
    index_by_file_id = {
        feature.file_id: index
        for index, feature in enumerate(canonical_document_features)
    }
    initial = result.labels.copy()
    sizes: dict[int, int] = {}
    for label in initial.tolist():
        if label != -1:
            sizes[label] = sizes.get(label, 0) + 1

    # endpoints[(a, b)][a] = members of cluster a with a merge edge into b
    endpoints: dict[tuple[int, int], dict[int, set[int]]] = {}
    noise_votes: dict[int, list[int]] = {}
    for decision in adjudication_decisions:
        if decision.decision != "merge_edge":
            continue
        request = request_by_id.get(decision.request_id)
        if request is None or len(request.subjects) != 2:
            continue
        left_id, right_id = request.subjects
        if left_id not in index_by_file_id or right_id not in index_by_file_id:
            continue
        left_index = index_by_file_id[left_id]
        right_index = index_by_file_id[right_id]
        left_label = int(initial[left_index])
        right_label = int(initial[right_index])
        if left_label == -1 and right_label == -1:
            continue
        if left_label == -1:
            noise_votes.setdefault(left_index, []).append(right_label)
            continue
        if right_label == -1:
            noise_votes.setdefault(right_index, []).append(left_label)
            continue
        if left_label == right_label:
            continue
        key = (min(left_label, right_label), max(left_label, right_label))
        side = endpoints.setdefault(key, {key[0]: set(), key[1]: set()})
        side[left_label].add(left_index)
        side[right_label].add(right_index)

    parent: dict[int, int] = {label: label for label in sizes}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    clusters_merged = 0
    for (low, high), side in sorted(endpoints.items()):
        smaller = low if sizes[low] <= sizes[high] else high
        coverage = len(side[smaller]) / sizes[smaller]
        if coverage < cluster_merge_support:
            continue
        root_low, root_high = find(low), find(high)
        if root_low == root_high:
            continue
        parent[max(root_low, root_high)] = min(root_low, root_high)
        clusters_merged += 1

    labels = np.array(
        [find(label) if label != -1 else -1 for label in initial.tolist()],
        dtype=int,
    )
    noise_attached = 0
    for index, votes in sorted(noise_votes.items()):
        roots = [find(label) for label in votes]
        best = min(set(roots), key=lambda root: (-roots.count(root), root))
        labels[index] = best
        noise_attached += 1

    if clusters_merged == 0 and noise_attached == 0:
        return result

    return ClusterResult(
        labels=labels,
        cluster_names=result.cluster_names.copy(),
        repaired_mask=result.repaired_mask.copy(),
        metrics=dataclasses.replace(
            result.metrics,
            adjudication_clusters_merged=(
                result.metrics.adjudication_clusters_merged + clusters_merged
            ),
            adjudication_noise_attached=(
                result.metrics.adjudication_noise_attached + noise_attached
            ),
        ),
    )


def cluster_canonical_documents(
    embeddings: np.ndarray,
    *,
    config: Config,
    options: Any,
    item_names: list[str],
    canonical_document_features: list[DocumentFeatures],
    candidate_edges: list[CandidateEdge],
    candidate_components: list[CandidateComponent],
    cluster_documents_fn,
) -> ClusterResult:
    """Run clustering on canonical embeddings and apply conservative links."""
    cluster_mode = options.cluster_mode or config.topic_clustering.mode
    allow_single_cluster = (
        options.cluster_allow_single_cluster
        if options.cluster_allow_single_cluster is not None
        else config.topic_clustering.allow_single_cluster
    )
    pca_components_option = (
        options.cluster_pca_components
        if options.cluster_pca_components is not None
        else (
            config.topic_clustering.pca_components
            if config.topic_clustering.reducer == "pca"
            else None
        )
    )

    if cluster_mode == "graph":
        labels = np.full(len(canonical_document_features), -1, dtype=int)
        file_id_to_index = {
            feature.file_id: index
            for index, feature in enumerate(canonical_document_features)
        }
        adjacency: dict[int, set[int]] = {
            index: set() for index in range(len(canonical_document_features))
        }
        min_edge_score = config.candidate_generation.component_min_edge_score
        for edge in candidate_edges:
            if edge.score < min_edge_score:
                continue
            if edge.edge_type not in CONTENT_BACKED_EDGE_TYPES:
                # Title/filename matches alone must not form graph clusters.
                continue
            if (
                edge.source_id not in file_id_to_index
                or edge.target_id not in file_id_to_index
            ):
                continue
            left = file_id_to_index[edge.source_id]
            right = file_id_to_index[edge.target_id]
            adjacency[left].add(right)
            adjacency[right].add(left)

        visited: set[int] = set()
        next_label = 0
        for index in range(len(canonical_document_features)):
            if index in visited or not adjacency[index]:
                continue
            stack = [index]
            component_indices: list[int] = []
            visited.add(index)
            while stack:
                current = stack.pop()
                component_indices.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
            if len(component_indices) < 2:
                continue
            for component_index in component_indices:
                labels[component_index] = next_label
            next_label += 1
        unassigned_indices = [
            index for index, label in enumerate(labels) if label == -1
        ]
        if len(unassigned_indices) >= max(1, config.clustering.min_cluster_size):
            residual_result = cluster_documents_fn(
                embeddings[unassigned_indices],
                config=config,
                repair_noise=options.repair_noise,
                clustering=config.clustering,
                allow_single_cluster=allow_single_cluster,
                item_names=[item_names[index] for index in unassigned_indices],
            )
            residual_labels = residual_result.labels.copy()
            non_noise_labels = sorted(
                label for label in set(residual_labels) if int(label) != -1
            )
            label_map = {
                int(label): next_label + offset
                for offset, label in enumerate(non_noise_labels)
            }
            for offset, original_index in enumerate(unassigned_indices):
                residual_label = int(residual_labels[offset])
                if residual_label == -1:
                    continue
                labels[original_index] = label_map[residual_label]
            next_label += len(non_noise_labels)
            residual_noise_repaired = residual_result.metrics.noise_repaired
            residual_small_merged = residual_result.metrics.small_clusters_merged
            residual_name_merged = residual_result.metrics.name_clusters_merged
        else:
            residual_noise_repaired = 0
            residual_small_merged = 0
            residual_name_merged = 0

        return ClusterResult(
            labels=labels,
            cluster_names={},
            repaired_mask=np.zeros(labels.shape, dtype=bool),
            metrics=ClusterMetrics(
                initial_clusters=next_label,
                initial_noise=int(np.sum(labels == -1)),
                noise_repaired=residual_noise_repaired,
                small_clusters_merged=residual_small_merged,
                name_clusters_merged=residual_name_merged,
            ),
        )

    if cluster_mode == "agglomerative":
        topic = config.topic_clustering
        result = cluster_documents_agglomerative(
            embeddings,
            config=config,
            clustering=config.clustering,
            target_clusters=topic.agglomerative_target_clusters,
            distance_threshold=topic.agglomerative_distance_threshold,
        )
        return apply_candidate_component_links(
            result,
            canonical_document_features,
            candidate_components,
        )

    reducer = (
        "pca"
        if pca_components_option is not None
        else (config.topic_clustering.reducer)
    )
    cluster_inputs = reduce_for_density_clustering(
        embeddings,
        reducer=reducer,
        pca_components=pca_components_option,
        umap_components=config.topic_clustering.umap_components,
        umap_neighbors=config.topic_clustering.umap_neighbors,
    )
    cluster_kwargs: dict[str, Any] = {}
    if cluster_inputs is not None:
        cluster_kwargs["cluster_inputs"] = cluster_inputs

    result = cluster_documents_fn(
        embeddings,
        config=config,
        repair_noise=options.repair_noise,
        clustering=config.clustering,
        allow_single_cluster=allow_single_cluster,
        item_names=item_names,
        **cluster_kwargs,
    )
    return apply_candidate_component_links(
        result,
        canonical_document_features,
        candidate_components,
    )


def build_canonical_cluster_stage(
    *,
    files: list[Path],
    contents: list[str],
    file_hashes: list[str],
    file_reports: list[Any],
    canonical_indices: list[int],
    options: Any,
    config: Config,
    client,
    request_runtime,
    vectorize,
    expand_noise_repaired_count,
    cluster_documents_fn,
) -> CanonicalClusterStageResult:
    """Build all V2 stage objects and canonical clustering outputs."""
    document_features = [
        build_document_features(
            file,
            content,
            config=config,
            client=client,
            file_report=report,
        )
        for file, content, report in zip(files, contents, file_reports, strict=True)
    ]
    cluster_mode = options.cluster_mode or config.topic_clustering.mode
    canonical_files = [files[index] for index in canonical_indices]
    canonical_hashes = [file_hashes[index] for index in canonical_indices]
    canonical_contents = [contents[index] for index in canonical_indices]
    canonical_document_features = [
        document_features[index] for index in canonical_indices
    ]
    canonical_embeddings = vectorize(
        canonical_files,
        canonical_hashes,
        canonical_contents,
        options,
    )
    canonical_embeddings = normalize_embeddings(canonical_embeddings)
    canonical_count = len(canonical_document_features)
    embedding_count = int(canonical_embeddings.shape[0])
    if embedding_count < canonical_count:
        raise ValueError(
            "canonical embedding count is smaller than canonical document count: "
            f"{embedding_count} < {canonical_count}"
        )
    if embedding_count > canonical_count:
        canonical_embeddings = canonical_embeddings[:canonical_count]
    for canonical_feature, embedding in zip(
        canonical_document_features, canonical_embeddings, strict=True
    ):
        canonical_feature.content_embedding = embedding

    candidate_edges = build_candidate_edges(
        canonical_document_features,
        min_filename_token_overlap=(
            config.candidate_generation.filename_token_overlap_threshold
        ),
        min_content_similarity=config.candidate_generation.content_similarity_threshold,
    )
    candidate_components = build_candidate_components(
        canonical_document_features,
        candidate_edges,
        min_edge_score=config.candidate_generation.component_min_edge_score,
    )
    adjudication_requests = build_adjudication_requests(
        candidate_edges,
        candidate_components,
        score_threshold=config.cluster_adjudication.request_score_threshold,
    )
    adjudication_decisions = apply_rule_adjudication(
        adjudication_requests,
        edge_merge_threshold=config.cluster_adjudication.edge_merge_threshold,
    )
    if (
        config.cluster_adjudication.enable_llm_judging
        and client is not None
        and getattr(client, "chat", None) is not None
        and request_runtime is not None
    ):
        llm_decisions = apply_llm_adjudication(
            adjudication_requests,
            client=client,
            config=config,
            request_runtime=request_runtime,
            document_features=canonical_document_features,
        )
        decisions_by_request = {
            decision.request_id: decision for decision in adjudication_decisions
        }
        for decision in llm_decisions:
            decisions_by_request[decision.request_id] = decision
        adjudication_decisions = list(decisions_by_request.values())

    cluster_result = cluster_canonical_documents(
        canonical_embeddings,
        config=config,
        options=options,
        item_names=[file.name for file in canonical_files],
        canonical_document_features=canonical_document_features,
        candidate_edges=candidate_edges,
        candidate_components=candidate_components,
        cluster_documents_fn=cluster_documents_fn,
    )
    cluster_result = apply_adjudication_links(
        cluster_result,
        canonical_document_features,
        adjudication_requests,
        adjudication_decisions,
    )
    cluster_drafts = build_cluster_drafts(
        cluster_result.labels,
        canonical_document_features=canonical_document_features,
        origin=cluster_mode,
    )
    noise_repaired = expand_noise_repaired_count(
        canonical_indices,
        cluster_result.repaired_mask,
        file_hashes,
    )
    return CanonicalClusterStageResult(
        canonical_embeddings=canonical_embeddings,
        cluster_result=cluster_result,
        noise_repaired=noise_repaired,
        document_features=canonical_document_features,
        candidate_edges=candidate_edges,
        candidate_components=candidate_components,
        cluster_drafts=cluster_drafts,
        adjudication_requests=adjudication_requests,
        adjudication_decisions=adjudication_decisions,
    )
