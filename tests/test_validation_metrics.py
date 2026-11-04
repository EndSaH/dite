from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dite.flow.api import PipelineResult
from dite.validation import (
    build_constraint_metrics,
    build_partition_metrics,
    build_structure_metrics,
    compute_similarity_profile,
    load_validation_corpus,
    partition_metrics_payload,
)


def _write_manifest(root: Path, payload: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pipeline_result(root: Path, labels: list[int]) -> PipelineResult:
    files = [root / "a.txt", root / "b.txt", root / "c.txt"]
    return PipelineResult(
        files=files,
        contents=["alpha", "beta", "gamma"],
        document_features=[],
        embeddings=np.array(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]],
            dtype=np.float32,
        ),
        labels=np.array(labels, dtype=int),
        cluster_names={0: "A", 1: "B"},
    )


def test_load_validation_corpus_builds_constraints(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "representative",
            "description": "demo",
            "owner": "team",
            "files": [
                {
                    "path": "a.txt",
                    "cluster_id": "group_1",
                    "must_link": ["b.txt"],
                },
                {
                    "path": "b.txt",
                    "cluster_id": "group_1",
                },
                {
                    "path": "c.txt",
                    "cluster_id": "group_2",
                    "must_not_link": ["a.txt"],
                },
            ],
        },
    )

    corpus = load_validation_corpus(tmp_path)

    assert corpus.name == "sample"
    assert corpus.constraints.must_link_pairs == frozenset({("a.txt", "b.txt")})
    assert corpus.constraints.must_not_link_pairs == frozenset({("a.txt", "c.txt")})


def test_load_validation_corpus_rejects_missing_targets(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "regression",
            "description": "demo",
            "owner": "team",
            "files": [
                {
                    "path": "a.txt",
                    "cluster_id": "group_1",
                    "must_link": ["missing.txt"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="invalid must_link target"):
        load_validation_corpus(tmp_path)


def test_build_constraint_metrics_uses_manifest_pairs(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "representative",
            "description": "demo",
            "owner": "team",
            "files": [
                {
                    "path": "a.txt",
                    "cluster_id": "group_1",
                    "must_link": ["b.txt"],
                },
                {
                    "path": "b.txt",
                    "cluster_id": "group_1",
                },
                {
                    "path": "c.txt",
                    "cluster_id": "group_2",
                    "must_not_link": ["a.txt"],
                },
            ],
        },
    )
    corpus = load_validation_corpus(tmp_path)
    result = _pipeline_result(tmp_path, [0, 0, 0])

    metrics = build_constraint_metrics(result, corpus)

    assert metrics.must_link_total == 1
    assert metrics.must_link_recall == 1.0
    assert metrics.must_not_link_total == 1
    assert metrics.must_not_link_violations == 1
    assert metrics.must_not_link_violation_rate == 1.0
    assert metrics.cluster_id_fragmentation_total == 0
    assert metrics.cluster_id_purity_by_id["group_1"] == 1.0


def test_build_constraint_metrics_counts_fragmentation(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "representative",
            "description": "demo",
            "owner": "team",
            "files": [
                {"path": "a.txt", "cluster_id": "group_1"},
                {"path": "b.txt", "cluster_id": "group_1"},
                {"path": "c.txt", "cluster_id": "group_1"},
            ],
        },
    )
    corpus = load_validation_corpus(tmp_path)
    result = _pipeline_result(tmp_path, [0, 1, -1])

    metrics = build_constraint_metrics(result, corpus)

    assert metrics.cluster_id_fragmentation_total == 1
    assert metrics.cluster_id_fragmentation_by_id["group_1"] == 1
    assert metrics.cluster_id_purity_by_id["group_1"] == pytest.approx(1 / 3)


def test_build_structure_metrics_returns_none_without_feature_edges(
    tmp_path: Path,
) -> None:
    result = _pipeline_result(tmp_path, [0, 0, -1])

    metrics = build_structure_metrics(result)

    assert metrics.filename_bias_rate is None


def _labeled_corpus(root: Path, groups: dict[str, str]):
    for name in groups:
        (root / name).write_text(name, encoding="utf-8")
    _write_manifest(
        root,
        {
            "name": "sample",
            "tier": "representative",
            "description": "demo",
            "owner": "team",
            "files": [
                {"path": name, "cluster_id": cluster_id}
                for name, cluster_id in groups.items()
            ],
        },
    )
    return load_validation_corpus(root)


def _labeled_result(root: Path, labels: dict[str, int]) -> PipelineResult:
    names = sorted(labels)
    return PipelineResult(
        files=[root / name for name in names],
        contents=["x"] * len(names),
        embeddings=np.zeros((len(names), 2), dtype=np.float32),
        labels=np.array([labels[name] for name in names], dtype=int),
        cluster_names={},
    )


def test_partition_metrics_are_perfect_for_matching_partition(tmp_path: Path) -> None:
    corpus = _labeled_corpus(
        tmp_path,
        {"a.txt": "ml", "b.txt": "ml", "c.txt": "finance", "d.txt": "finance"},
    )
    result = _labeled_result(tmp_path, {"a.txt": 7, "b.txt": 7, "c.txt": 2, "d.txt": 2})

    metrics = build_partition_metrics(result, corpus)

    assert metrics.num_labeled_files == 4
    assert metrics.num_evaluated_files == 4
    assert metrics.adjusted_rand_index == pytest.approx(1.0)
    assert metrics.adjusted_mutual_info == pytest.approx(1.0)
    assert metrics.bcubed_precision == pytest.approx(1.0)
    assert metrics.bcubed_recall == pytest.approx(1.0)
    assert metrics.bcubed_f1 == pytest.approx(1.0)
    assert metrics.predicted_noise_rate == 0.0


def test_partition_metrics_separate_over_merge_from_fragmentation(
    tmp_path: Path,
) -> None:
    corpus = _labeled_corpus(
        tmp_path,
        {"a.txt": "ml", "b.txt": "ml", "c.txt": "finance", "d.txt": "finance"},
    )

    merged = build_partition_metrics(
        _labeled_result(tmp_path, {"a.txt": 0, "b.txt": 0, "c.txt": 0, "d.txt": 0}),
        corpus,
    )
    # Everything in one cluster: each file's cluster is half wrong.
    assert merged.bcubed_precision == pytest.approx(0.5)
    assert merged.bcubed_recall == pytest.approx(1.0)

    split = build_partition_metrics(
        _labeled_result(tmp_path, {"a.txt": 0, "b.txt": 1, "c.txt": 2, "d.txt": 3}),
        corpus,
    )
    # Every file alone: nothing wrong is merged, but each group is split.
    assert split.bcubed_precision == pytest.approx(1.0)
    assert split.bcubed_recall == pytest.approx(0.5)
    assert split.adjusted_rand_index == pytest.approx(0.0)


def test_partition_metrics_treat_noise_as_singletons(tmp_path: Path) -> None:
    corpus = _labeled_corpus(
        tmp_path,
        {
            "a.txt": "ml",
            "b.txt": "ml",
            "x.txt": "noise_misc",
            "y.txt": "noise_misc",
        },
    )

    # Predicted noise never counts as a shared cluster, and two gold-noise
    # files lumped together are a false merge, not a hit.
    metrics = build_partition_metrics(
        _labeled_result(tmp_path, {"a.txt": -1, "b.txt": -1, "x.txt": 5, "y.txt": 5}),
        corpus,
    )

    assert metrics.predicted_noise_rate == pytest.approx(0.5)
    # a, b: precision 1 (alone), recall 1/2. x, y: precision 1/2, recall 1.
    assert metrics.bcubed_precision == pytest.approx(0.75)
    assert metrics.bcubed_recall == pytest.approx(0.75)


def test_partition_metrics_count_missing_files_as_noise(tmp_path: Path) -> None:
    corpus = _labeled_corpus(tmp_path, {"a.txt": "ml", "b.txt": "ml"})

    metrics = build_partition_metrics(_labeled_result(tmp_path, {"a.txt": 0}), corpus)

    assert metrics.num_labeled_files == 2
    assert metrics.num_evaluated_files == 1
    assert metrics.predicted_noise_rate == pytest.approx(0.5)
    assert metrics.bcubed_recall == pytest.approx(0.5)


def test_partition_metrics_payload_is_json_ready(tmp_path: Path) -> None:
    corpus = _labeled_corpus(tmp_path, {"a.txt": "ml", "b.txt": "ml"})
    metrics = build_partition_metrics(
        _labeled_result(tmp_path, {"a.txt": 0, "b.txt": 0}), corpus
    )

    payload = partition_metrics_payload(metrics)

    assert json.loads(json.dumps(payload))["bcubed_f1"] == pytest.approx(1.0)


def test_similarity_profile_summarizes_cosine_distribution() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [0.99, 0.141], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32
    )

    profile = compute_similarity_profile(embeddings)

    # The zero vector is ignored; pairs: 0.99, 0.0, 0.141.
    assert profile["num_documents"] == 3
    assert profile["pairwise_p50"] == pytest.approx(0.141, abs=1e-3)
    pairs = [0.0, 0.141, 0.99]
    assert profile["pairwise_p99"] == pytest.approx(np.percentile(pairs, 99), abs=1e-3)
    assert profile["nearest_neighbor_p90"] == pytest.approx(0.99, abs=0.01)
    assert compute_similarity_profile(np.zeros((1, 2))) is None


def test_structure_metrics_include_similarity_profile(tmp_path: Path) -> None:
    result = _pipeline_result(tmp_path, [0, 0, 1])

    metrics = build_structure_metrics(result)

    assert metrics.similarity_profile is not None
    assert metrics.similarity_profile["num_documents"] == 3
