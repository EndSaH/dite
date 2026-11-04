"""Internal clustering experiment helper.

This script is intentionally not wired into the public CLI. It exists to compare
embedding input modes and sweep clustering parameters with file-level diffs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dite.app.config import Config, load_config
from dite.cache import FileCache
from dite.flow.api import PipelineOptions, PipelineResult, PipelineService
from dite.util.api import build_sync_openai_client
from dite.validation import (
    build_constraint_metrics,
    build_partition_metrics,
    build_structure_metrics,
    load_validation_corpus,
    partition_metrics_payload,
)


def _build_pipeline(
    config: Config, *, use_cache: bool
) -> tuple[PipelineService, FileCache | None]:
    cache = None
    if use_cache and config.cache.enabled:
        cache = FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
    client = build_sync_openai_client(config)
    return PipelineService(client=client, config=config, cache=cache), cache


def _cluster_summary(result: PipelineResult) -> dict[str, Any]:
    final_num_clusters = len(
        [label for label in set(result.labels) if label != -1]
    )
    return {
        "num_files": len(result.files),
        "initial_num_clusters": result.cluster_metrics.initial_clusters,
        "final_num_clusters": final_num_clusters,
        "num_noise": int((result.labels == -1).sum()),
        "noise_repaired": result.noise_repaired,
        "small_clusters_merged": result.cluster_metrics.small_clusters_merged,
        "name_clusters_merged": result.cluster_metrics.name_clusters_merged,
        "total_clusters_merged": result.clusters_merged,
        "small_cluster_merge_candidates": (
            result.cluster_metrics.small_cluster_merge_candidates
        ),
        "small_cluster_merge_skipped": (
            result.cluster_metrics.small_cluster_merge_skipped
        ),
    }


def _process_metrics(result: PipelineResult) -> dict[str, Any]:
    return {
        "num_files_total": len(result.files),
        "num_files_extraction_failed": sum(
            1
            for feature in result.document_features
            if feature.quality_flags.extraction_failed
        ),
        "num_files_short_text": sum(
            1
            for feature in result.document_features
            if feature.quality_flags.short_text
        ),
        "num_files_filename_dominant": sum(
            1
            for feature in result.document_features
            if feature.quality_flags.filename_dominant
        ),
        "candidate_edges_total": len(result.candidate_edges),
        "candidate_components_total": len(result.candidate_components),
        "cluster_drafts_total": len(
            [label for label in set(result.labels.tolist()) if label != -1]
        ),
        "adjudication_requests_total": len(result.adjudication_requests),
        "adjudication_requests_by_type": _count_by_key(
            [request.request_type for request in result.adjudication_requests]
        ),
        "adjudication_by_model": _count_by_key(
            [decision.model_used for decision in result.adjudication_decisions]
        ),
    }


def _cluster_diagnostics(result: PipelineResult) -> dict[str, Any]:
    return {
        "small_cluster_merge_max_similarity": (
            result.cluster_metrics.small_cluster_merge_max_similarity
        ),
        "small_cluster_merge_events": [
            asdict(event) for event in result.cluster_metrics.small_cluster_merge_events
        ],
        "small_cluster_skip_events": [
            asdict(event) for event in result.cluster_metrics.small_cluster_skip_events
        ],
    }


def _assignments(result: PipelineResult) -> dict[str, dict[str, Any]]:
    return {
        str(path): {
            "label": int(label),
            "cluster_name": result.cluster_names.get(int(label)),
            "is_noise": int(label) == -1,
        }
        for path, label in zip(result.files, result.labels, strict=True)
    }


def _fragmentation_score(result: PipelineResult) -> int:
    return (
        result.cluster_metrics.small_cluster_merge_candidates
        + result.cluster_metrics.small_cluster_merge_skipped
        + int((result.labels == -1).sum())
    )


def _count_by_key(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def _run_pipeline(
    folder: Path,
    *,
    config: Config,
    use_cache: bool,
    use_knn_repair: bool,
    options: PipelineOptions,
) -> PipelineResult:
    pipeline, cache = _build_pipeline(config, use_cache=use_cache)
    try:
        effective_options = replace(
            options,
            use_cache=cache is not None,
            use_embedding_cache=True,
            repair_noise=use_knn_repair,
            merge_same_name=False,
            allow_vlm_api=True,
            exclude_paths=[],
        )
        return pipeline.run(folder, effective_options)
    finally:
        if cache is not None:
            cache.close()


def _compare_inputs(
    folder: Path,
    *,
    config: Config,
    use_cache: bool,
    use_knn_repair: bool,
) -> dict[str, Any]:
    corpus = _maybe_load_validation_corpus(folder)
    with_filename = _run_pipeline(
        folder,
        config=config,
        use_cache=use_cache,
        use_knn_repair=use_knn_repair,
        options=PipelineOptions(embedding_input_mode="with_filename"),
    )
    content_only = _run_pipeline(
        folder,
        config=config,
        use_cache=use_cache,
        use_knn_repair=use_knn_repair,
        options=PipelineOptions(embedding_input_mode="content_only"),
    )
    with_assignments = _assignments(with_filename)
    content_assignments = _assignments(content_only)
    paths = sorted(set(with_assignments) | set(content_assignments))
    diff_entries = []
    for path in paths:
        default_assignment = {
            "label": -1,
            "cluster_name": None,
            "is_noise": True,
        }
        before = with_assignments.get(path, default_assignment)
        after = content_assignments.get(path, default_assignment)
        diff_entries.append(
            {
                "path": path,
                "from_label": before["label"],
                "to_label": after["label"],
                "from_name": before["cluster_name"],
                "to_name": after["cluster_name"],
                "from_noise": before["is_noise"],
                "to_noise": after["is_noise"],
                "changed": before != after,
            }
        )
    return {
        "folder": str(folder),
        "runs": {
            "with_filename": {
                "config": {
                    "input_mode": "with_filename",
                    "clustering": asdict(config.clustering),
                    "use_cache": use_cache and config.cache.enabled,
                    "repair_noise": use_knn_repair,
                },
                "summary": _cluster_summary(with_filename),
                "process_metrics": _process_metrics(with_filename),
                "structure_metrics": _structure_metrics_payload(with_filename),
                "constraint_metrics": _constraint_metrics_payload(
                    with_filename, corpus
                ),
                "partition_metrics": _partition_metrics_payload(with_filename, corpus),
                "diagnostics": _cluster_diagnostics(with_filename),
                "assignments": with_assignments,
            },
            "content_only": {
                "config": {
                    "input_mode": "content_only",
                    "clustering": asdict(config.clustering),
                    "use_cache": use_cache and config.cache.enabled,
                    "repair_noise": use_knn_repair,
                },
                "summary": _cluster_summary(content_only),
                "process_metrics": _process_metrics(content_only),
                "structure_metrics": _structure_metrics_payload(content_only),
                "constraint_metrics": _constraint_metrics_payload(
                    content_only, corpus
                ),
                "partition_metrics": _partition_metrics_payload(content_only, corpus),
                "diagnostics": _cluster_diagnostics(content_only),
                "assignments": content_assignments,
            },
        },
        "diff": {
            "summary": {
                "initial_num_clusters_delta": (
                    with_filename.cluster_metrics.initial_clusters
                    - content_only.cluster_metrics.initial_clusters
                ),
                "final_num_clusters_delta": (
                    len([label for label in set(with_filename.labels) if label != -1])
                    - len([label for label in set(content_only.labels) if label != -1])
                ),
                "num_noise_delta": int((with_filename.labels == -1).sum())
                - int((content_only.labels == -1).sum()),
                "small_clusters_merged_delta": (
                    with_filename.cluster_metrics.small_clusters_merged
                    - content_only.cluster_metrics.small_clusters_merged
                ),
                "name_clusters_merged_delta": (
                    with_filename.cluster_metrics.name_clusters_merged
                    - content_only.cluster_metrics.name_clusters_merged
                ),
                "total_clusters_merged_delta": (
                    with_filename.clusters_merged - content_only.clusters_merged
                ),
            },
            "entries": diff_entries,
        },
    }


def _sweep_configs(base: Config) -> list[tuple[str, Config, PipelineOptions]]:
    runs: list[tuple[str, Config, PipelineOptions]] = []
    run_id = 0
    for input_mode in ("with_filename", "content_only"):
        for min_cluster_size in (3, 4, 5):
            for min_samples in (1, 2, 3):
                for epsilon in (0.0, 0.1, 0.25, 0.4):
                    config = replace(
                        base,
                        clustering=replace(
                            base.clustering,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=epsilon,
                            cluster_selection_method="eom",
                            small_cluster_merge_enabled=True,
                            small_cluster_merge_max_size=4,
                            small_cluster_merge_cosine_threshold=0.92,
                        ),
                        topic_clustering=replace(
                            base.topic_clustering,
                            mode="density",
                            reducer="none",
                            pca_components=None,
                            allow_single_cluster=False,
                        ),
                    )
                    options = PipelineOptions(
                        embedding_input_mode=input_mode,
                    )
                    runs.append((f"baseline-{run_id:03d}", config, options))
                    run_id += 1
    return runs


def _extended_sweep_configs(base: Config) -> list[tuple[str, Config, PipelineOptions]]:
    runs: list[tuple[str, Config, PipelineOptions]] = []
    run_id = 0
    for input_mode in ("with_filename", "content_only"):
        for min_cluster_size in (3, 4, 5):
            for min_samples in (1, 2, 3):
                for epsilon in (0.0, 0.1, 0.25, 0.4):
                    for pca_components in (None, 50, 100):
                        for allow_single_cluster in (False, True):
                            reducer = "pca" if pca_components is not None else "none"
                            config = replace(
                                base,
                                clustering=replace(
                                    base.clustering,
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_epsilon=epsilon,
                                    cluster_selection_method="eom",
                                    small_cluster_merge_enabled=True,
                                    small_cluster_merge_max_size=4,
                                    small_cluster_merge_cosine_threshold=0.92,
                                ),
                                topic_clustering=replace(
                                    base.topic_clustering,
                                    mode="density",
                                    reducer=reducer,
                                    pca_components=pca_components,
                                    allow_single_cluster=allow_single_cluster,
                                ),
                            )
                            options = PipelineOptions(
                                embedding_input_mode=input_mode,
                            )
                            runs.append((f"extended-{run_id:03d}", config, options))
                            run_id += 1
    for input_mode in ("with_filename", "content_only"):
        for topic in _alternative_topic_clusterings(base):
            config = replace(base, topic_clustering=topic)
            options = PipelineOptions(embedding_input_mode=input_mode)
            runs.append((f"extended-{run_id:03d}", config, options))
            run_id += 1
    return runs


def _alternative_topic_clusterings(base: Config) -> list:
    """Agglomerative cuts and UMAP-reduced HDBSCAN, at default HDBSCAN params.

    UMAP runs are included only when the optional umap-learn package exists.
    """
    topics = [
        replace(
            base.topic_clustering,
            mode="agglomerative",
            reducer="none",
            pca_components=None,
            agglomerative_target_clusters=None,
            agglomerative_distance_threshold=threshold,
        )
        for threshold in (None, 0.3, 0.4, 0.5)
    ]
    try:
        import umap  # noqa: F401
    except ImportError:
        return topics
    topics.extend(
        replace(
            base.topic_clustering,
            mode="density",
            reducer="umap",
            pca_components=None,
            umap_components=components,
        )
        for components in (5, 10)
    )
    return topics


def _reducer_label(config: Config) -> str:
    topic = config.topic_clustering
    if topic.mode == "agglomerative":
        threshold = topic.agglomerative_distance_threshold
        if topic.agglomerative_target_clusters is not None:
            return f"agglomerative-k{topic.agglomerative_target_clusters}"
        if threshold is None:
            return "agglomerative-auto"
        return f"agglomerative-d{threshold}"
    if topic.reducer == "umap":
        return f"umap-{topic.umap_components}"
    if topic.pca_components is not None:
        return f"pca-{topic.pca_components}"
    return "none"


def _run_sweep(
    folder: Path,
    *,
    config: Config,
    use_cache: bool,
    use_knn_repair: bool,
    extended: bool,
) -> dict[str, Any]:
    corpus = _maybe_load_validation_corpus(folder)
    configs = _extended_sweep_configs(config) if extended else _sweep_configs(config)
    runs = []
    for run_id, run_config, options in configs:
        result = _run_pipeline(
            folder,
            config=run_config,
            use_cache=use_cache,
            use_knn_repair=use_knn_repair,
            options=options,
        )
        runs.append(
            {
                "run_id": run_id,
                "config": {
                    "input_mode": options.embedding_input_mode,
                    "clustering": asdict(run_config.clustering),
                    "topic_clustering": asdict(run_config.topic_clustering),
                    "allow_single_cluster": run_config.topic_clustering.allow_single_cluster,
                    "mode": run_config.topic_clustering.mode,
                    "reducer": _reducer_label(run_config),
                    "use_cache": use_cache and config.cache.enabled,
                    "repair_noise": use_knn_repair,
                },
                "summary": _cluster_summary(result),
                "process_metrics": _process_metrics(result),
                "structure_metrics": _structure_metrics_payload(result),
                "constraint_metrics": _constraint_metrics_payload(result, corpus),
                "partition_metrics": _partition_metrics_payload(result, corpus),
                "diagnostics": _cluster_diagnostics(result),
                "assignments": _assignments(result),
                "fragmentation_score": _fragmentation_score(result),
            }
        )
    # With a labeled manifest, rank by agreement with the gold groups first;
    # the unlabeled fragmentation heuristic only breaks ties.
    runs.sort(
        key=lambda item: (
            -((item.get("partition_metrics") or {}).get("bcubed_f1") or 0.0),
            item["fragmentation_score"],
            item["summary"]["num_noise"],
            item["summary"]["final_num_clusters"],
            item["summary"]["total_clusters_merged"],
            item["run_id"],
        )
    )
    return {
        "folder": str(folder),
        "baseline_config": asdict(config.clustering),
        "runs": runs,
    }


def _maybe_load_validation_corpus(folder: Path):
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return None
    return load_validation_corpus(folder, manifest_path=manifest_path)


def _constraint_metrics_payload(result: PipelineResult, corpus) -> dict[str, Any] | None:
    if corpus is None:
        return None
    metrics = build_constraint_metrics(result, corpus)
    return {
        "must_link_total": metrics.must_link_total,
        "must_link_recall": metrics.must_link_recall,
        "must_not_link_total": metrics.must_not_link_total,
        "must_not_link_violations": metrics.must_not_link_violations,
        "must_not_link_violation_rate": metrics.must_not_link_violation_rate,
        "cluster_id_fragmentation_total": metrics.cluster_id_fragmentation_total,
        "cluster_id_fragmentation_by_id": metrics.cluster_id_fragmentation_by_id,
        "cluster_id_purity_by_id": metrics.cluster_id_purity_by_id,
    }


def _partition_metrics_payload(result: PipelineResult, corpus) -> dict[str, Any] | None:
    if corpus is None:
        return None
    return partition_metrics_payload(build_partition_metrics(result, corpus))


def _structure_metrics_payload(result: PipelineResult) -> dict[str, Any]:
    metrics = build_structure_metrics(result)
    return {
        "density_validation_score": metrics.density_validation_score,
        "filename_bias_rate": metrics.filename_bias_rate,
        "similarity_profile": metrics.similarity_profile,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Internal clustering experiment helper"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_compare = {
        "folder": {
            "type": Path,
            "help": "Target directory",
        },
        "output": {
            "flags": ("--output",),
            "type": Path,
            "default": None,
            "help": "Write JSON to a file instead of stdout",
        },
        "no_cache": {
            "flags": ("--no-cache",),
            "action": "store_true",
            "help": "Disable file and embedding cache reuse",
        },
        "no_knn_repair": {
            "flags": ("--no-knn-repair",),
            "action": "store_true",
            "help": "Disable noise repair",
        },
    }

    compare_parser = subparsers.add_parser("compare-inputs")
    compare_parser.add_argument("folder", **common_compare["folder"])
    compare_parser.add_argument(
        *common_compare["output"]["flags"], type=Path, default=None
    )
    compare_parser.add_argument(
        *common_compare["no_cache"]["flags"], action="store_true"
    )
    compare_parser.add_argument(
        *common_compare["no_knn_repair"]["flags"], action="store_true"
    )

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("folder", **common_compare["folder"])
    sweep_parser.add_argument(
        *common_compare["output"]["flags"], type=Path, default=None
    )
    sweep_parser.add_argument(
        *common_compare["no_cache"]["flags"], action="store_true"
    )
    sweep_parser.add_argument(
        *common_compare["no_knn_repair"]["flags"], action="store_true"
    )
    sweep_parser.add_argument(
        "--extended",
        action="store_true",
        help="Enable PCA and allow_single_cluster experiment combinations",
    )
    return parser.parse_args()


def _emit(data: dict[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    config = load_config()
    if not args.folder.exists():
        raise SystemExit(f"Folder does not exist: {args.folder}")
    use_cache = not args.no_cache
    use_knn_repair = not args.no_knn_repair
    if args.command == "compare-inputs":
        data = _compare_inputs(
            args.folder,
            config=config,
            use_cache=use_cache,
            use_knn_repair=use_knn_repair,
        )
    else:
        data = _run_sweep(
            args.folder,
            config=config,
            use_cache=use_cache,
            use_knn_repair=use_knn_repair,
            extended=bool(args.extended),
        )
    _emit(data, args.output)


if __name__ == "__main__":
    main()
