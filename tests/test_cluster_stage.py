from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dite.app.config import Config
from dite.cluster.algo import (
    cluster_documents_agglomerative,
    reduce_for_density_clustering,
)
from dite.cluster.api import cluster_documents, repair_noise_with_knn
from dite.cluster.link import build_candidate_edges
from dite.cluster.model import (
    AdjudicationDecision,
    AdjudicationRequest,
    ClusterMetrics,
    ClusterResult,
)
from dite.cluster.stage import (
    apply_adjudication_links,
    build_canonical_cluster_stage,
    cluster_canonical_documents,
)
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


def _separated_groups(
    sizes: list[int],
    *,
    outliers: int = 0,
    dim: int = 64,
    seed: int = 0,
    center_overlap: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=dim)
    shared /= np.linalg.norm(shared)
    vectors = []
    gold = []
    for group, size in enumerate(sizes):
        center = rng.normal(size=dim)
        center /= np.linalg.norm(center)
        center = (1 - center_overlap) * center + center_overlap * shared
        center /= np.linalg.norm(center)
        for _ in range(size):
            vectors.append(center + 0.05 * rng.normal(size=dim))
            gold.append(group)
    for _ in range(outliers):
        vectors.append(rng.normal(size=dim))
        gold.append(-1)
    return np.array(vectors, dtype=np.float32), gold


def _same_partition(labels: list[int], gold: list[int]) -> bool:
    pairs = set(zip(labels, gold, strict=True))
    clustered = {(label, group) for label, group in pairs if group != -1}
    return (
        len({label for label, _ in clustered}) == len(clustered)
        and len({group for _, group in clustered}) == len(clustered)
        and all(label == -1 for label, group in pairs if group == -1)
        and all(label != -1 for label, group in pairs if group != -1)
    )


def test_agglomerative_auto_cut_recovers_groups_and_keeps_outliers() -> None:
    embeddings, gold = _separated_groups([6, 5, 4], outliers=3)

    result = cluster_documents_agglomerative(embeddings, config=Config())

    assert _same_partition(result.labels.tolist(), gold)
    assert result.metrics.initial_clusters == 3
    assert result.metrics.initial_noise == 3


def test_agglomerative_target_counts_only_real_groups() -> None:
    # Groups sit closer to each other than to the outliers, so a plain
    # maxclust=3 cut would merge topics and keep outliers as "clusters".
    # Outliers must not use up the folder budget.
    embeddings, gold = _separated_groups(
        [6, 5, 4], outliers=4, seed=1, center_overlap=0.6
    )

    result = cluster_documents_agglomerative(
        embeddings, config=Config(), target_clusters=3
    )

    assert result.metrics.initial_clusters == 3
    assert _same_partition(result.labels.tolist(), gold)


def test_agglomerative_distance_threshold_and_size_floor() -> None:
    embeddings, _gold = _separated_groups([5, 2])
    config = Config()
    config.clustering.min_cluster_size = 3

    loose = cluster_documents_agglomerative(
        embeddings, config=config, distance_threshold=1.9
    )
    tight = cluster_documents_agglomerative(
        embeddings, config=config, distance_threshold=0.2
    )

    assert set(loose.labels.tolist()) == {0}
    # The 2-member group is below min_cluster_size and stays unsorted.
    assert tight.labels.tolist() == [0, 0, 0, 0, 0, -1, -1]


def test_agglomerative_handles_tiny_inputs_and_zero_vectors() -> None:
    tiny = cluster_documents_agglomerative(np.eye(2, dtype=np.float32), config=Config())
    assert tiny.labels.tolist() == [-1, -1]

    embeddings, _gold = _separated_groups([4, 4])
    embeddings = np.vstack([embeddings, np.zeros((1, embeddings.shape[1]))])
    result = cluster_documents_agglomerative(embeddings, config=Config())
    assert result.labels.shape == (9,)
    assert result.labels[-1] == -1


def _stage_features(count: int):
    return [
        build_document_features(Path(f"docs/file-{index}.md"), f"Body {index}")
        for index in range(count)
    ]


def test_stage_dispatches_agglomerative_mode_without_hdbscan() -> None:
    embeddings, gold = _separated_groups([4, 4])
    features = _stage_features(len(gold))

    def _unexpected(*args, **kwargs):
        raise AssertionError("density clustering must not run")

    result = cluster_canonical_documents(
        embeddings,
        config=Config(),
        options=PipelineOptions(cluster_mode="agglomerative"),
        item_names=[feature.name for feature in features],
        canonical_document_features=features,
        candidate_edges=[],
        candidate_components=[],
        cluster_documents_fn=_unexpected,
    )

    assert _same_partition(result.labels.tolist(), gold)


def test_pca_reducer_clusters_reduced_inputs_but_keeps_original_embeddings() -> None:
    embeddings, gold = _separated_groups([4, 4], dim=32)
    features = _stage_features(len(gold))
    seen = {}

    def _record(embeddings_arg, **kwargs):
        seen["embeddings"] = embeddings_arg
        seen["cluster_inputs"] = kwargs.get("cluster_inputs")
        return _all_noise_cluster_fn(embeddings_arg)

    config = Config()
    config.topic_clustering.reducer = "pca"
    config.topic_clustering.pca_components = 3
    cluster_canonical_documents(
        embeddings,
        config=config,
        options=PipelineOptions(),
        item_names=[feature.name for feature in features],
        canonical_document_features=features,
        candidate_edges=[],
        candidate_components=[],
        cluster_documents_fn=_record,
    )

    assert seen["embeddings"].shape == (8, 32)
    assert seen["cluster_inputs"].shape == (8, 3)


def test_umap_reducer_produces_low_dimensional_inputs() -> None:
    pytest.importorskip("umap")
    embeddings, _gold = _separated_groups([8, 8], dim=32)

    reduced = reduce_for_density_clustering(
        embeddings, reducer="umap", umap_components=4, umap_neighbors=5
    )

    assert reduced.shape == (16, 4)


def test_cluster_documents_runs_hdbscan_on_cluster_inputs() -> None:
    embeddings, gold = _separated_groups([5, 5, 5], dim=16)

    result = cluster_documents(
        np.zeros_like(embeddings) + 1.0,
        config=Config(),
        repair_noise=False,
        cluster_inputs=embeddings,
    )

    assert _same_partition(result.labels.tolist(), gold)
    with pytest.raises(ValueError, match="one row per embedding"):
        cluster_documents(embeddings, config=Config(), cluster_inputs=embeddings[:3])


def _unit(angle_degrees: float) -> list[float]:
    radians = np.deg2rad(angle_degrees)
    return [float(np.cos(radians)), float(np.sin(radians))]


def test_knn_repair_gates_on_distance_to_the_voted_cluster() -> None:
    # Noise at 0 deg. Its single nearest neighbour (2 deg) is in cluster 1,
    # but the 3-vote picks cluster 0, whose closest member is 30 deg away.
    embeddings = np.array(
        [_unit(30), _unit(32), _unit(34), _unit(2), _unit(0)], dtype=np.float32
    )
    labels = np.array([0, 0, 0, 1, -1], dtype=int)

    repaired, count = repair_noise_with_knn(
        embeddings, labels, k=3, distance_threshold=0.05
    )

    assert count == 0
    assert repaired.tolist() == [0, 0, 0, 1, -1]

    # Within the threshold of the voted cluster it is still repaired.
    repaired, count = repair_noise_with_knn(
        embeddings, labels, k=3, distance_threshold=0.2
    )
    assert count == 1
    assert repaired[-1] == 0


def _merge_request(index: int, left: str, right: str):
    request = AdjudicationRequest(
        request_id=f"edge:{index}",
        request_type="edge_review",
        subjects=[left, right],
        trigger_reason="content_similarity",
        score=0.95,
    )
    decision = AdjudicationDecision(
        request_id=request.request_id,
        decision="merge_edge",
        confidence=0.95,
        reason="test",
    )
    return request, decision


def _adjudicate(labels: list[int], pairs: list[tuple[int, int]]):
    features = [
        build_document_features(Path(f"docs/doc-{index}.md"), f"Body {index}")
        for index in range(len(labels))
    ]
    requests, decisions = [], []
    for number, (left, right) in enumerate(pairs):
        request, decision = _merge_request(
            number, features[left].file_id, features[right].file_id
        )
        requests.append(request)
        decisions.append(decision)
    result = ClusterResult(
        labels=np.array(labels, dtype=int),
        cluster_names={},
        repaired_mask=np.zeros(len(labels), dtype=bool),
        metrics=ClusterMetrics(),
    )
    return apply_adjudication_links(result, features, requests, decisions)


def test_single_merge_edge_does_not_tear_a_document_out_of_its_cluster() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    result = _adjudicate(labels, [(3, 4)])

    assert result.labels.tolist() == labels
    assert result.metrics.adjudication_clusters_merged == 0


def test_merge_edges_covering_half_the_smaller_cluster_merge_clusters() -> None:
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

    result = _adjudicate(labels, [(0, 6), (1, 7)])

    assert result.labels.tolist() == [0] * 10
    assert result.metrics.adjudication_clusters_merged == 1


def test_merge_edge_attaches_noise_and_leaves_noise_pairs_alone() -> None:
    labels = [0, 0, 0, 1, 1, 1, -1, -1, -1]

    result = _adjudicate(labels, [(6, 0), (6, 1), (6, 3), (7, 8)])

    # Doc 6 links to cluster 0 twice and cluster 1 once: majority wins.
    assert result.labels.tolist() == [0, 0, 0, 1, 1, 1, 0, -1, -1]
    assert result.metrics.adjudication_noise_attached == 1
    assert result.metrics.adjudication_clusters_merged == 0


def test_graph_mode_uses_configured_component_edge_threshold() -> None:
    paths = [Path("a/one.md"), Path("b/two.md")]
    features = [
        build_document_features(path, f"Distinct body for {path.stem}")
        for path in paths
    ]
    embeddings = np.array([[1.0, 0.0], [0.95, 0.312]], dtype=np.float32)
    for feature, embedding in zip(features, embeddings, strict=True):
        feature.content_embedding = embedding
    edges = build_candidate_edges(features, min_content_similarity=0.9)
    assert [edge.edge_type for edge in edges] == ["content_similarity"]
    assert 0.94 < edges[0].score < 0.96

    def _run(threshold: float) -> list[int]:
        config = Config()
        config.candidate_generation.component_min_edge_score = threshold
        result = cluster_canonical_documents(
            embeddings,
            config=config,
            options=PipelineOptions(cluster_mode="graph"),
            item_names=[path.name for path in paths],
            canonical_document_features=features,
            candidate_edges=edges,
            candidate_components=[],
            cluster_documents_fn=_all_noise_cluster_fn,
        )
        return result.labels.tolist()

    assert _run(0.9) == [0, 0]
    assert _run(0.97) == [-1, -1]


def _run_stage_capturing_embedding_texts(input_mode: str) -> list[str]:
    files = [Path("docs/eigen.md"), Path("docs/budget.md")]
    contents = [
        "<!-- image -->\n## Eigenvalues\nLet A be a square matrix. " * 3,
        "Household budget\nMonthly spending and savings plan.",
    ]
    captured: list[str] = []

    def _vectorize(files_arg, hashes_arg, texts, options_arg):
        del files_arg, hashes_arg, options_arg
        captured.extend(texts)
        return np.eye(len(texts), dtype=np.float32)

    build_canonical_cluster_stage(
        files=files,
        contents=contents,
        file_hashes=["h0", "h1"],
        file_reports=[None, None],
        canonical_indices=[0, 1],
        options=PipelineOptions(embedding_input_mode=input_mode),
        config=Config(),
        client=None,
        request_runtime=None,
        vectorize=_vectorize,
        expand_noise_repaired_count=lambda *args: 0,
        cluster_documents_fn=_all_noise_cluster_fn,
    )
    return captured


def test_summary_input_mode_embeds_feature_summaries() -> None:
    raw = _run_stage_capturing_embedding_texts("with_filename")
    summary = _run_stage_capturing_embedding_texts("summary")

    assert raw[0].startswith("<!-- image -->")
    assert summary[0].startswith("Title: Eigenvalues")
    assert "<!-- image -->" not in summary[0]
    assert summary[1].startswith("Title: Household budget")


def test_embedding_cache_key_changes_with_excerpt_settings(tmp_path: Path) -> None:
    from dite.flow.vector import vectorize_files

    class _Cache:
        def __init__(self) -> None:
            self.versions: list[str] = []

        def get_embedding(self, file, file_hash, *, required_model_version):
            del file, file_hash
            self.versions.append(required_model_version)
            return np.array([1.0, 0.0], dtype=np.float32)

    def _run(config: Config) -> str:
        cache = _Cache()
        vectorize_files(
            files=[tmp_path / "a.md"],
            file_hashes=["h"],
            contents=["text"],
            options=PipelineOptions(),
            config=config,
            cache=cache,
            client=None,
            logger=type("_Logger", (), {"debug": lambda self, message: None})(),
            get_embeddings_fn=None,
        )
        return cache.versions[0]

    default_version = _run(Config())
    changed = Config()
    changed.processing.text_truncate_limit = 8000

    assert "excerpt=" not in default_version
    assert _run(changed).endswith("|excerpt=8000:0.6:0.2")


def test_features_are_built_once_per_canonical_document(monkeypatch) -> None:
    calls: list[str] = []
    real_build = build_document_features

    def _counting_build(path, content, **kwargs):
        calls.append(path.name)
        return real_build(path, content, **kwargs)

    monkeypatch.setattr("dite.cluster.stage.build_document_features", _counting_build)
    files = [Path("a.md"), Path("copy-of-a.md"), Path("b.md")]

    build_canonical_cluster_stage(
        files=files,
        contents=["same", "same", "other"],
        file_hashes=["h-a", "h-a", "h-b"],
        file_reports=[None, None, None],
        canonical_indices=[0, 2],
        options=PipelineOptions(),
        config=Config(),
        client=None,
        request_runtime=None,
        vectorize=lambda files_arg, hashes, texts, options: np.eye(
            len(texts), dtype=np.float32
        ),
        expand_noise_repaired_count=lambda *args: 0,
        cluster_documents_fn=_all_noise_cluster_fn,
    )

    assert calls == ["a.md", "b.md"]


def test_analysis_features_run_concurrently_and_keep_order(monkeypatch) -> None:
    import threading
    import time

    from dite.cluster.stage import _build_features_parallel

    active = {"now": 0, "max": 0}
    lock = threading.Lock()
    real_build = build_document_features

    def _slow_build(path, content, **kwargs):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with lock:
            active["now"] -= 1
        return real_build(path, content)

    monkeypatch.setattr("dite.cluster.stage.build_document_features", _slow_build)
    config = Config()
    config.feature_extraction.analysis_enabled = True
    config.feature_extraction.analysis_workers = 4
    items = [(Path(f"doc-{index}.md"), f"Body {index}", None) for index in range(8)]

    features = _build_features_parallel(items, config=config, client=object())

    assert [feature.name for feature in features] == [
        f"doc-{index}.md" for index in range(8)
    ]
    assert active["max"] > 1


def test_common_filename_tokens_do_not_create_candidate_pairs() -> None:
    from dite.cluster.link import _filename_overlap_pairs

    # Every name shares "file"; none reaches Jaccard 0.8 with another.
    token_sets = [{"file", str(index)} for index in range(200)]
    token_sets.append({"file", "7"})  # a true near-duplicate of file 7

    pairs = _filename_overlap_pairs(token_sets, 0.8)

    assert pairs == {(7, 200)}
