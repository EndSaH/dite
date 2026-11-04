"""Cluster-level data models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Edge types whose evidence comes from document content. Title and filename
# edges are metadata hints: they are reported and can inform an LLM reviewer,
# but rules never merge or link documents on metadata alone.
CONTENT_BACKED_EDGE_TYPES = frozenset({"near_duplicate", "content_similarity"})


@dataclass
class CandidateEdge:
    """A candidate relation between two files or components."""

    source_id: str
    target_id: str
    edge_type: str
    score: float
    evidence: list[str] = field(default_factory=list)
    hard_constraint: str | None = None
    quality_guard: list[str] = field(default_factory=list)


@dataclass
class CandidateComponent:
    """A strongly-connected local group built before topic clustering."""

    component_id: str
    member_file_ids: list[str]
    component_type: str
    formation_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ClusterDraft:
    """Intermediate topic-clustering draft before final naming."""

    draft_cluster_id: int
    member_file_ids: list[str]
    origin: str
    noise_members: list[str] = field(default_factory=list)
    merge_candidates: list[int] = field(default_factory=list)


@dataclass
class AdjudicationRequest:
    """A structured request for boundary or merge adjudication."""

    request_id: str
    request_type: str
    subjects: list[str]
    candidate_targets: list[str] = field(default_factory=list)
    evidence_bundle: list[str] = field(default_factory=list)
    trigger_reason: str = ""
    score: float | None = None
    quality_guard: list[str] = field(default_factory=list)


@dataclass
class AdjudicationDecision:
    """A structured adjudication outcome."""

    request_id: str
    decision: str
    confidence: float
    reason: str
    supporting_evidence: list[str] = field(default_factory=list)
    model_used: str = "rules"
    fallback_used: bool = False


@dataclass
class ClusterResult:
    """Cluster labels plus derived metrics."""

    labels: np.ndarray
    cluster_names: dict[int, str]
    repaired_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    metrics: ClusterMetrics = field(default_factory=lambda: ClusterMetrics())

    @property
    def n_clusters(self) -> int:
        unique = set(self.labels)
        return len([label for label in unique if label != -1])

    @property
    def n_noise(self) -> int:
        return int(np.sum(self.labels == -1))

    @property
    def noise_repaired(self) -> int:
        return self.metrics.noise_repaired

    @property
    def small_clusters_merged(self) -> int:
        return self.metrics.small_clusters_merged

    @property
    def name_clusters_merged(self) -> int:
        return self.metrics.name_clusters_merged

    @property
    def total_clusters_merged(self) -> int:
        return self.small_clusters_merged + self.name_clusters_merged


@dataclass
class ClusterMetrics:
    """Structured clustering metrics used by pipeline and reporting layers."""

    initial_clusters: int = 0
    initial_noise: int = 0
    noise_repaired: int = 0
    small_clusters_merged: int = 0
    name_clusters_merged: int = 0
    small_cluster_merge_candidates: int = 0
    small_cluster_merge_skipped: int = 0
    adjudication_clusters_merged: int = 0
    adjudication_noise_attached: int = 0
    small_cluster_merge_max_similarity: float | None = None
    small_cluster_merge_events: list[SmallClusterMergeEvent] = field(
        default_factory=list
    )
    small_cluster_skip_events: list[SmallClusterSkipEvent] = field(
        default_factory=list
    )


@dataclass
class SmallClusterMergeEvent:
    """Detailed event for a successful small-cluster merge."""

    source_label: int
    source_size: int
    target_label: int
    target_size_before: int
    similarity: float


@dataclass
class SmallClusterSkipEvent:
    """Detailed event for a skipped small-cluster merge candidate."""

    source_label: int
    source_size: int
    best_target_label: int | None
    best_target_size: int | None
    best_similarity: float | None
    reason: str


@dataclass
class ClusterRepresentation:
    """User-facing structured representation for a finalized cluster."""

    cluster_id: int
    name: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    topic: str = ""
    domain: str = ""
    representative_file_ids: list[str] = field(default_factory=list)
    evidence_summary: str = ""
