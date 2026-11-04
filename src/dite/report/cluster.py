"""Cluster report builders and terminal rendering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.table import Table

from dite.app.i18n import t
from dite.util.log import get_logger

if TYPE_CHECKING:
    from dite.cluster.model import ClusterMetrics


def build_report(
    files: list[Path],
    contents: list[str],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    cluster_metrics: ClusterMetrics | None = None,
    *,
    document_features=None,
    candidate_edges=None,
    candidate_components=None,
    cluster_drafts=None,
    adjudication_requests=None,
    adjudication_decisions=None,
    cluster_representations=None,
    structure_metrics=None,
    constraint_metrics=None,
    partition_metrics=None,
) -> dict:
    """Build a structured clustering report for CLI and JSON output."""
    from dite.cluster.api import _build_cluster_debug_labels
    from dite.cluster.model import ClusterMetrics

    metrics = cluster_metrics or ClusterMetrics()
    clusters: dict[int, dict] = {}
    debug_labels = _build_cluster_debug_labels(labels)
    noise = []
    feature_by_path = {
        str(feature.path): feature for feature in (document_features or [])
    }
    failed_count = 0

    for file, content, label in zip(files, contents, labels, strict=True):
        feature = feature_by_path.get(str(file))
        is_failed = bool(
            feature is not None and feature.quality_flags.extraction_failed
        )
        if is_failed:
            failed_count += 1

        entry = {
            "path": str(file),
            "name": file.name,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "extraction_failed": is_failed,
        }

        if label == -1:
            noise.append(entry)
            continue

        cluster_label = int(label)
        if cluster_label not in clusters:
            clusters[cluster_label] = {
                "name": cluster_names.get(
                    cluster_label, t("cluster_default_name", label=cluster_label)
                ),
                "debug_label": debug_labels.get(cluster_label),
                "files": [],
            }
        clusters[cluster_label]["files"].append(entry)

    return {
        "summary": {
            "total_files": len(files),
            "num_clusters": len(clusters),
            "num_noise": len(noise),
            "num_extraction_failed": failed_count,
            "initial_num_clusters": metrics.initial_clusters,
            "initial_num_noise": metrics.initial_noise,
            "noise_repaired": metrics.noise_repaired,
            "small_clusters_merged": metrics.small_clusters_merged,
            "name_clusters_merged": metrics.name_clusters_merged,
            "total_clusters_merged": (
                metrics.small_clusters_merged + metrics.name_clusters_merged
            ),
            "small_cluster_merge_candidates": metrics.small_cluster_merge_candidates,
            "small_cluster_merge_skipped": metrics.small_cluster_merge_skipped,
            "num_document_features": len(document_features or []),
            "num_candidate_edges": len(candidate_edges or []),
            "num_candidate_components": len(candidate_components or []),
            "num_cluster_drafts": len(cluster_drafts or []),
            "num_adjudication_requests": len(adjudication_requests or []),
            "num_adjudication_decisions": len(adjudication_decisions or []),
            "num_cluster_representations": len(cluster_representations or {}),
        },
        "structure_metrics": {
            "density_validation_score": (
                None
                if structure_metrics is None
                else structure_metrics.get("density_validation_score")
            ),
            "filename_bias_rate": (
                None
                if structure_metrics is None
                else structure_metrics.get("filename_bias_rate")
            ),
        },
        "constraint_metrics": constraint_metrics or {},
        "partition_metrics": partition_metrics or {},
        "clusters": [clusters[label] for label in sorted(clusters)],
        "cluster_diagnostics": {
            "small_cluster_merge_max_similarity": (
                metrics.small_cluster_merge_max_similarity
            ),
            "small_cluster_merge_events": [
                event.__dict__.copy() for event in metrics.small_cluster_merge_events
            ],
            "small_cluster_skip_events": [
                event.__dict__.copy() for event in metrics.small_cluster_skip_events
            ],
        },
        "feature_diagnostics": {
            "filename_dominant_count": sum(
                1
                for feature in (document_features or [])
                if feature.quality_flags.filename_dominant
            ),
            "short_text_count": sum(
                1
                for feature in (document_features or [])
                if feature.quality_flags.short_text
            ),
            "candidate_edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type,
                    "score": edge.score,
                    "evidence": edge.evidence,
                    "quality_guard": edge.quality_guard,
                }
                for edge in (candidate_edges or [])
            ],
            "candidate_components": [
                {
                    "component_id": component.component_id,
                    "member_file_ids": component.member_file_ids,
                    "component_type": component.component_type,
                    "formation_evidence": component.formation_evidence,
                    "confidence": component.confidence,
                }
                for component in (candidate_components or [])
            ],
            "cluster_drafts": [
                {
                    "draft_cluster_id": draft.draft_cluster_id,
                    "member_file_ids": draft.member_file_ids,
                    "origin": draft.origin,
                    "noise_members": draft.noise_members,
                    "merge_candidates": draft.merge_candidates,
                }
                for draft in (cluster_drafts or [])
            ],
            "adjudication_requests": [
                {
                    "request_id": request.request_id,
                    "request_type": request.request_type,
                    "subjects": request.subjects,
                    "trigger_reason": request.trigger_reason,
                }
                for request in (adjudication_requests or [])
            ],
            "adjudication_decisions": [
                {
                    "request_id": decision.request_id,
                    "decision": decision.decision,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "model_used": decision.model_used,
                }
                for decision in (adjudication_decisions or [])
            ],
            "cluster_representations": [
                {
                    "cluster_id": representation.cluster_id,
                    "name": representation.name,
                    "summary": representation.summary,
                    "keywords": representation.keywords,
                    "topic": representation.topic,
                    "domain": representation.domain,
                    "representative_file_ids": representation.representative_file_ids,
                    "evidence_summary": representation.evidence_summary,
                }
                for representation in (cluster_representations or {}).values()
            ],
        },
        "noise": noise,
    }


def print_report(report: dict) -> None:
    """Render a clustering report to the terminal."""
    logger = get_logger()
    logger.print(f"\n[bold green]{t('cluster_report_title')}[/bold green]\n")

    summary = report["summary"]
    logger.print(f"[bold]{t('cluster_total_files')}[/bold] {summary['total_files']}")
    logger.print(f"[bold]{t('cluster_num_clusters')}[/bold] {summary['num_clusters']}")
    logger.print(f"[bold]{t('cluster_num_noise')}[/bold] {summary['num_noise']}")

    if summary.get("noise_repaired", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_knn_label')}[/bold] "
            f"{t('cluster_knn_repair', count=summary['noise_repaired'])}"
        )

    if summary.get("small_clusters_merged", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_merge_label')}[/bold] "
            f"{t('cluster_small_merged', count=summary['small_clusters_merged'])}"
        )
    if summary.get("name_clusters_merged", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_merge_label')}[/bold] "
            f"{t('cluster_name_merged', count=summary['name_clusters_merged'])}"
        )
    if summary.get("total_clusters_merged", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_merge_label')}[/bold] "
            f"{t('cluster_total_merged', count=summary['total_clusters_merged'])}"
        )

    if summary.get("num_extraction_failed", 0) > 0:
        logger.warning(
            t("cluster_extraction_failed", count=summary["num_extraction_failed"])
        )

    for cluster in report["clusters"]:
        cluster_title = cluster["name"]
        if logger.verbose and cluster.get("debug_label"):
            cluster_title = f"{cluster['debug_label']} | {cluster_title}"
        logger.print(
            f"\n[bold cyan]【{cluster_title}】[/bold cyan] "
            f"({t('organize_files_count', count=len(cluster['files']))})"
        )
        table = Table(show_header=False, box=None, padding=(0, 2))
        for file_entry in cluster["files"]:
            marker = (
                f"[dim]{t('cluster_report_extraction_failed_marker')}[/dim]"
                if file_entry.get("extraction_failed")
                else ""
            )
            table.add_row("  -", file_entry["name"], marker)
        logger.print_table(table)

    if report["noise"]:
        logger.print(
            f"\n[bold yellow]【{t('cluster_uncategorized')}】[/bold yellow] "
            f"({t('organize_files_count', count=len(report['noise']))})"
        )
        for file_entry in report["noise"]:
            marker = (
                f" [dim]{t('cluster_report_extraction_failed_marker')}[/dim]"
                if file_entry.get("extraction_failed")
                else ""
            )
            logger.print(f"  - {file_entry['name']}{marker}")
