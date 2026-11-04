from __future__ import annotations

from pathlib import Path

import numpy as np

from dite.app.config import Config
from dite.cluster.link import build_candidate_edges
from dite.cluster.model import ClusterMetrics, ClusterResult
from dite.cluster.stage import cluster_canonical_documents
from dite.doc import build_document_features
from dite.flow.model import PipelineOptions


def _all_noise_cluster_fn(embeddings, **kwargs):
    del kwargs
    count = embeddings.shape[0]
    return ClusterResult(
        labels=np.full(count, -1, dtype=int),
        cluster_names={},
        repaired_mask=np.zeros(count, dtype=bool),
        metrics=ClusterMetrics(initial_noise=count),
    )


def test_graph_mode_does_not_turn_filename_matches_into_clusters() -> None:
    paths = [
        Path("courseA/lecture1.pdf"),
        Path("courseB/lecture1.pdf"),
        Path("misc/budget.xlsx"),
    ]
    contents = [
        "Operating systems: processes and scheduling.",
        "Organic chemistry: alkanes and alkenes.",
        "Household budget for 2026.",
    ]
    embeddings = np.eye(3, dtype=np.float32)
    features = [
        build_document_features(path, content)
        for path, content in zip(paths, contents, strict=True)
    ]
    for feature, embedding in zip(features, embeddings, strict=True):
        feature.content_embedding = embedding
    edges = build_candidate_edges(features)
    assert [edge.edge_type for edge in edges] == ["filename_similarity"]

    result = cluster_canonical_documents(
        embeddings,
        config=Config(),
        options=PipelineOptions(cluster_mode="graph"),
        item_names=[path.name for path in paths],
        canonical_document_features=features,
        candidate_edges=edges,
        candidate_components=[],
        cluster_documents_fn=_all_noise_cluster_fn,
    )

    assert result.labels.tolist() == [-1, -1, -1]
