"""CLI 命令定义"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import click
import numpy as np
import typer
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from rich.table import Table
from typer.core import TyperGroup

from dite import __version__
from dite.app.config import Config, load_config
from dite.app.i18n import set_locale, t
from dite.cache import FileCache
from dite.doc.embed import get_embedding_cache_version
from dite.report import build_report as _reporting_build_report
from dite.report import print_report as _reporting_print_report
from dite.report.move import (
    apply_clustering_to_preview as _apply_clustering_to_preview,
)
from dite.report.move import (
    finalize_organize_run as _finalize_organize_run,
)
from dite.report.pdf import (
    label_pdf_reason as _label_pdf_reason,
)
from dite.report.pdf import (
    print_extraction_detail_table as _print_extraction_detail_table,
)
from dite.report.pdf import (
    select_problem_file_reports as _select_problem_file_reports,
)
from dite.util.api import build_sync_openai_client
from dite.util.llm import format_api_error
from dite.util.log import get_logger, setup_logging
from dite.validation import (
    build_constraint_metrics,
    build_partition_metrics,
    build_structure_metrics,
    load_validation_corpus,
    partition_metrics_payload,
)

if TYPE_CHECKING:
    from dite.cluster.api import ClusterMetrics
    from dite.flow.api import (
        PipelineOptions,
        PipelineResult,
        PipelineService,
    )
    from dite.flow.move import OrganizePreview


_HELP_KEY_PREFIX = "__dite_help__:"
PipelineOptions = None
PipelineService = None
OrganizePreview = None
download_docling_pdf_models = None
get_docling_pdf_artifacts_path = None
has_docling_pdf_artifacts = None


def _help_key(key: str) -> str:
    return f"{_HELP_KEY_PREFIX}{key}"


def _translate_help_text(text: str | None) -> str | None:
    if text is None or not text.startswith(_HELP_KEY_PREFIX):
        return text
    return t(text.removeprefix(_HELP_KEY_PREFIX))


def _sync_command_help(command: click.Command) -> None:
    command.help = _translate_help_text(command.help)
    command.short_help = _translate_help_text(command.short_help)
    command.epilog = _translate_help_text(command.epilog)

    for param in command.params:
        if hasattr(param, "help"):
            param.help = _translate_help_text(param.help)

    if isinstance(command, click.Group):
        for subcommand in command.commands.values():
            _sync_command_help(subcommand)


class LocalizedTyperGroup(TyperGroup):
    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        cfg = load_config()
        set_locale(cfg.i18n.locale)
        _sync_command_help(self)
        return super().make_context(info_name, args, parent=parent, **extra)


def _install_cli_warning_filters() -> None:
    warnings.filterwarnings("ignore", message=".*cannot be loaded by Pillow.*")
    warnings.filterwarnings("ignore", message=".*cannot find loader.*")


def _get_pipeline_options_cls():
    global PipelineOptions
    if PipelineOptions is None:
        from dite.flow.api import PipelineOptions as _PipelineOptions

        PipelineOptions = _PipelineOptions
    return PipelineOptions


def _get_pipeline_service_cls():
    global PipelineService
    if PipelineService is None:
        from dite.flow.api import PipelineService as _PipelineService

        PipelineService = _PipelineService
    return PipelineService


def _get_organize_preview_cls():
    global OrganizePreview
    if OrganizePreview is None:
        from dite.flow.move import OrganizePreview as _OrganizePreview

        OrganizePreview = _OrganizePreview
    return OrganizePreview


def _get_docling_helpers():
    global download_docling_pdf_models
    global get_docling_pdf_artifacts_path
    global has_docling_pdf_artifacts
    if (
        download_docling_pdf_models is None
        or get_docling_pdf_artifacts_path is None
        or has_docling_pdf_artifacts is None
    ):
        from dite.io.docling import (
            download_docling_pdf_models as _download_docling_pdf_models,
        )
        from dite.io.docling import (
            get_docling_pdf_artifacts_path as _get_docling_pdf_artifacts_path,
        )
        from dite.io.docling import (
            has_docling_pdf_artifacts as _has_docling_pdf_artifacts,
        )

        if download_docling_pdf_models is None:
            download_docling_pdf_models = _download_docling_pdf_models
        if get_docling_pdf_artifacts_path is None:
            get_docling_pdf_artifacts_path = _get_docling_pdf_artifacts_path
        if has_docling_pdf_artifacts is None:
            has_docling_pdf_artifacts = _has_docling_pdf_artifacts
    return (
        download_docling_pdf_models,
        get_docling_pdf_artifacts_path,
        has_docling_pdf_artifacts,
    )


# 创建 CLI 应用
app = typer.Typer(
    name="dite",
    cls=LocalizedTyperGroup,
    help=_help_key("cli_description"),
    no_args_is_help=True,
)

# 缓存子命令
cache_app = typer.Typer(
    cls=LocalizedTyperGroup,
    help=_help_key("cli_help_cache_group"),
)
app.add_typer(cache_app, name="cache")

# 环境准备子命令
setup_app = typer.Typer(
    cls=LocalizedTyperGroup,
    help=_help_key("cli_help_setup_group"),
)
app.add_typer(setup_app, name="setup")


def version_callback(value: bool):
    """版本回调"""
    if value:
        logger = get_logger()
        logger.print(f"{t('version_prefix')} {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    color: Annotated[
        bool,
        typer.Option(
            "--color",
            help=_help_key("cli_help_color"),
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help=_help_key("cli_help_version"),
        ),
    ] = False,
):
    """DITE - Multimodal document intelligent clustering tool"""
    _install_cli_warning_filters()
    cfg = load_config()
    set_locale(cfg.i18n.locale)
    setup_logging(verbose=verbose, quiet=quiet, force_color=color)
    ctx.obj = {"config": cfg}


def _get_app_config(ctx: typer.Context | None) -> Config:
    if ctx is None or not ctx.obj or "config" not in ctx.obj:
        return load_config()
    return ctx.obj["config"]


def _get_client(config: Config) -> OpenAI:
    """获取 OpenAI 兼容客户端"""
    return build_sync_openai_client(config)


def _run_pipeline_or_exit(
    pipeline: PipelineService,
    folder: Path,
    options: PipelineOptions,
    cache: FileCache | None,
) -> PipelineResult:
    """Run pipeline and convert runtime/API failures to user-friendly CLI errors."""
    return _call_pipeline_or_exit(lambda: pipeline.run(folder, options), cache)


def _call_pipeline_or_exit(call, cache: FileCache | None):
    """Run a pipeline call; report runtime/API failures and exit with code 1."""
    logger = get_logger()
    try:
        return call()
    except (APIConnectionError, APITimeoutError):
        logger.error(t("error_api_connection_failed"))
    except APIError as exc:
        logger.error(t("error_api_request_failed", error=format_api_error(exc)))
    except Exception as exc:
        logger.error(t("error_processing_failed", error=exc))

    if cache:
        cache.close()
    raise typer.Exit(1)


def _build_report(
    files: list[Path],
    contents: list[str],
    embeddings: np.ndarray | None,
    labels: np.ndarray,
    cluster_names: dict[int, str],
    cluster_metrics: ClusterMetrics | None = None,
    *,
    validation_root: Path | None = None,
    document_features=None,
    candidate_edges=None,
    candidate_components=None,
    cluster_drafts=None,
    adjudication_requests=None,
    adjudication_decisions=None,
    cluster_representations=None,
) -> dict:
    """构建聚类报告"""
    structure_metrics_payload = None
    constraint_metrics_payload = None
    partition_metrics = None
    if embeddings is not None:
        structure_metrics = build_structure_metrics(
            type(
                "_Result",
                (),
                {
                    "labels": labels,
                    "embeddings": embeddings,
                    "document_features": document_features or [],
                    "candidate_edges": candidate_edges or [],
                },
            )()
        )
        structure_metrics_payload = {
            "density_validation_score": structure_metrics.density_validation_score,
            "filename_bias_rate": structure_metrics.filename_bias_rate,
            "similarity_profile": structure_metrics.similarity_profile,
        }
    if validation_root is not None:
        manifest_path = validation_root / "manifest.json"
        if manifest_path.exists():
            corpus = load_validation_corpus(validation_root, manifest_path=manifest_path)
            labeled_result = type(
                "_ConstraintResult",
                (),
                {
                    "files": files,
                    "labels": labels,
                },
            )()
            constraint_metrics = build_constraint_metrics(labeled_result, corpus)
            partition_metrics = partition_metrics_payload(
                build_partition_metrics(labeled_result, corpus)
            )
            constraint_metrics_payload = {
                "must_link_total": constraint_metrics.must_link_total,
                "must_link_recall": constraint_metrics.must_link_recall,
                "must_not_link_total": constraint_metrics.must_not_link_total,
                "must_not_link_violations": constraint_metrics.must_not_link_violations,
                "must_not_link_violation_rate": constraint_metrics.must_not_link_violation_rate,
                "cluster_id_fragmentation_total": constraint_metrics.cluster_id_fragmentation_total,
                "cluster_id_fragmentation_by_id": constraint_metrics.cluster_id_fragmentation_by_id,
                "cluster_id_purity_by_id": constraint_metrics.cluster_id_purity_by_id,
            }
    return _reporting_build_report(
        files,
        contents,
        labels,
        cluster_names,
        cluster_metrics,
        document_features=document_features,
        candidate_edges=candidate_edges,
        candidate_components=candidate_components,
        cluster_drafts=cluster_drafts,
        adjudication_requests=adjudication_requests,
        adjudication_decisions=adjudication_decisions,
        cluster_representations=cluster_representations,
        structure_metrics=structure_metrics_payload,
        constraint_metrics=constraint_metrics_payload,
        partition_metrics=partition_metrics,
    )


def _print_report(report: dict) -> None:
    """打印聚类报告"""
    _reporting_print_report(report)


@app.command(help=_help_key("scan_description"))
def scan(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_folder_scan")),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help=_help_key("cli_help_output_report")),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=_help_key("cli_help_disable_cache")),
    ] = False,
    no_knn_repair: Annotated[
        bool,
        typer.Option(
            "--no-knn-repair",
            help=_help_key("cli_help_disable_knn_repair"),
        ),
    ] = False,
):
    """
    Scan folder and perform cluster analysis.

    Scan documents in the specified folder, extract content, cluster using HDBSCAN,
    and generate cluster names using LLM.
    """
    # 如果子命令指定了日志级别，覆盖全局设置
    if verbose or quiet:
        setup_logging(verbose=verbose, quiet=quiet)

    logger = get_logger()
    config = _get_app_config(ctx)

    logger.print(f"[bold blue]{t('cli_title')}[/bold blue]\n")

    if not folder.exists():
        logger.error(t("scan_folder_not_found", folder=folder))
        raise typer.Exit(1)

    pipeline_options_cls = _get_pipeline_options_cls()
    pipeline_service_cls = _get_pipeline_service_cls()

    # 初始化
    client = _get_client(config)
    cache = (
        FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
        if not no_cache and config.cache.enabled
        else None
    )
    pipeline = pipeline_service_cls(client=client, config=config, cache=cache)
    result = _run_pipeline_or_exit(
        pipeline,
        folder,
        pipeline_options_cls(
            use_cache=cache is not None,
            use_embedding_cache=True,
            repair_noise=not no_knn_repair,
            merge_same_name=False,
            exclude_paths=[],
        ),
        cache,
    )

    if not result.files:
        if cache:
            cache.close()
        logger.error(t("scan_no_files"))
        raise typer.Exit(1)

    logger.status(t("scan_found_files", count=len(result.files)))

    extraction = result.extraction
    cache_msg_parts = []
    if extraction.doc_cache_hits:
        cache_msg_parts.append(t("cache_docling_hit", count=extraction.doc_cache_hits))
    if extraction.vlm_cache_hits:
        cache_msg_parts.append(t("cache_vlm_hit", count=extraction.vlm_cache_hits))
    if extraction.selected_vlm_files:
        cache_msg_parts.append(
            t("cache_vlm_fallback", count=extraction.selected_vlm_files)
        )
    if extraction.duplicate_count:
        cache_msg_parts.append(t("cache_duplicate", count=extraction.duplicate_count))
    cache_msg = f" ({', '.join(cache_msg_parts)})" if cache_msg_parts else ""
    logger.status(f"{t('scan_extraction_done')}{cache_msg}")

    if logger.verbose:
        logger.debug(
            t(
                "scan_extraction_verbose",
                primary_failures=extraction.primary_failures,
                source_fallback_needed=extraction.source_fallback_needed,
                vlm_api_page_calls=extraction.vlm_api_page_calls,
            )
        )

    if extraction.duplicate_groups and logger.verbose:
        logger.debug(t("debug_duplicate_groups"))
        for file_hash, file_list in extraction.duplicate_groups.items():
            logger.debug(t("debug_duplicate_group_hash", hash=file_hash[:12]))
            for f in file_list:
                logger.debug(t("debug_duplicate_group_file", name=Path(f).name))

    logger.status(t("scan_vectorizing_done", dim=result.embeddings.shape[1]))

    unique_labels = set(result.labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
    n_noise = int(np.sum(result.labels == -1))
    status_msg = t("scan_clustering_done", clusters=n_clusters, noise=n_noise)
    if result.noise_repaired > 0:
        status_msg += f" [{t('scan_status_knn_suffix', count=result.noise_repaired)}]"
    logger.status(status_msg)

    status_msg = t("scan_naming_done")
    if result.cluster_metrics.small_clusters_merged > 0:
        small_merged = result.cluster_metrics.small_clusters_merged
        status_msg += (
            f" [{t('scan_status_small_merged_suffix', count=small_merged)}]"
        )
    if result.cluster_metrics.name_clusters_merged > 0:
        name_merged = result.cluster_metrics.name_clusters_merged
        status_msg += (
            f" [{t('scan_status_name_merged_suffix', count=name_merged)}]"
        )
    logger.status(status_msg)

    # 关闭缓存
    if cache:
        cache.close()

    # 生成报告
    report = _build_report(
        result.files,
        result.contents,
        result.embeddings,
        result.labels,
        result.cluster_names,
        result.cluster_metrics,
        validation_root=folder,
        document_features=result.document_features,
        candidate_edges=result.candidate_edges,
        candidate_components=result.candidate_components,
        cluster_drafts=result.cluster_drafts,
        adjudication_requests=result.adjudication_requests,
        adjudication_decisions=result.adjudication_decisions,
        cluster_representations=result.cluster_representations,
    )

    # 保存 JSON
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.success(t("scan_report_saved", path=output))

    # 打印报告
    _print_report(report)


@app.command("pdf-check", help=_help_key("pdf_check_description"))
def pdf_check(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_folder_pdf_check")),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=_help_key("cli_help_disable_cache")),
    ] = False,
    cached_vlm_only: Annotated[
        bool,
        typer.Option(
            "--cached-vlm-only",
            help=_help_key("cli_help_cached_vlm_only"),
        ),
    ] = False,
):
    """Run PDF extraction checks without embedding, clustering, or naming."""
    if verbose or quiet:
        setup_logging(verbose=verbose, quiet=quiet)

    logger = get_logger()
    config = _get_app_config(ctx)

    logger.print(f"[bold blue]{t('cli_title')}[/bold blue]\n")

    if not folder.exists():
        logger.error(t("scan_folder_not_found", folder=folder))
        raise typer.Exit(1)

    from dite.flow.scan import scan_files

    pipeline_options_cls = _get_pipeline_options_cls()
    pipeline_service_cls = _get_pipeline_service_cls()

    files = scan_files(folder, config=config, extensions={".pdf"})
    if not files:
        logger.error(t("pdf_check_no_pdfs"))
        raise typer.Exit(1)

    client = _get_client(config)
    cache = (
        FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
        if not no_cache and config.cache.enabled
        else None
    )
    pipeline = pipeline_service_cls(client=client, config=config, cache=cache)

    try:
        result = pipeline.extract_files(
            files,
            pipeline_options_cls(
                use_cache=cache is not None,
                use_embedding_cache=False,
                repair_noise=False,
                merge_same_name=False,
                allow_vlm_api=not cached_vlm_only,
                exclude_paths=[],
            ),
        )
    except (APIConnectionError, APITimeoutError):
        logger.error(t("error_api_connection_failed"))
        if cache:
            cache.close()
        raise typer.Exit(1) from None
    except APIError as exc:
        logger.error(t("error_api_request_failed", error=format_api_error(exc)))
        if cache:
            cache.close()
        raise typer.Exit(1) from None
    except Exception as exc:
        logger.error(t("error_processing_failed", error=exc))
        if cache:
            cache.close()
        raise typer.Exit(1) from None

    if cache:
        cache.close()

    extraction = result.extraction
    threshold = config.processing.vlm_fallback_threshold
    weak_files = [
        (report.file, report.final_effective_length)
        for report in result.file_reports
        if report.final_effective_length < threshold
    ]
    empty_count = sum(1 for _file, length in weak_files if length == 0)

    logger.status(t("pdf_check_found_pdfs", count=len(result.files)))
    logger.status(
        t(
            "pdf_check_done",
            doc_cache_hits=extraction.doc_cache_hits,
            vlm_cache_hits=extraction.vlm_cache_hits,
            primary_failures=extraction.primary_failures,
            source_fallback_needed=extraction.source_fallback_needed,
            selected_vlm_files=extraction.selected_vlm_files,
            vlm_api_page_calls=extraction.vlm_api_page_calls,
            duplicates=extraction.duplicate_count,
            weak=len(weak_files),
            empty=empty_count,
        )
    )
    logger.print(t("pdf_check_note"))

    if logger.verbose and _select_problem_file_reports(result.file_reports):
        _print_extraction_detail_table(result.file_reports)

    if weak_files:
        table = Table(title=t("pdf_check_weak_table_title"))
        table.add_column(t("pdf_check_table_file"), style="path")
        table.add_column(t("pdf_check_table_reason"))
        table.add_column(t("pdf_check_table_final_effective_length"), justify="right")
        weak_report_by_file = {report.file: report for report in result.file_reports}
        for file, length in weak_files:
            report = weak_report_by_file[file]
            table.add_row(
                file.name,
                _label_pdf_reason(report.primary_error or report.source_reason),
                str(length),
            )
        logger.print_table(table)
        logger.error(t("pdf_check_failed", count=len(weak_files)))
        raise typer.Exit(1)

    logger.success(t("pdf_check_passed", count=len(result.files)))


@app.command(help=_help_key("organize_description"))
def organize(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_folder_organize")),
    ],
    target: Annotated[
        Path | None,
        typer.Option("--target", "-t", help=_help_key("cli_help_target_folder")),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help=_help_key("cli_help_preview_mode")),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help=_help_key("cli_help_execute_move")),
    ] = False,
    output_script: Annotated[
        Path | None,
        typer.Option("--output-script", help=_help_key("cli_help_output_script")),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=_help_key("cli_help_disable_cache")),
    ] = False,
    no_knn_repair: Annotated[
        bool,
        typer.Option(
            "--no-knn-repair",
            help=_help_key("cli_help_disable_knn_repair"),
        ),
    ] = False,
):
    """
    Organize documents in folder.

    Move files to subfolders based on clustering results.
    """
    # 如果子命令指定了日志级别，覆盖全局设置
    if verbose or quiet:
        setup_logging(verbose=verbose, quiet=quiet)

    logger = get_logger()
    config = _get_app_config(ctx)

    # 参数验证
    if not dry_run and not execute and not output_script:
        logger.warning(t("organize_specify_mode"))
        logger.print(t("organize_mode_help"))
        raise typer.Exit(1)

    if not folder.exists():
        logger.error(t("organize_folder_not_found", folder=folder))
        raise typer.Exit(1)

    # 设置目标文件夹
    if target is None:
        target = folder / "organized"

    logger.print(f"[bold blue]{t('cli_title')}[/bold blue]\n")

    organize_preview_cls = _get_organize_preview_cls()
    pipeline_options_cls = _get_pipeline_options_cls()
    pipeline_service_cls = _get_pipeline_service_cls()

    # 初始化
    client = _get_client(config)
    cache = (
        FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
        if not no_cache and config.cache.enabled
        else None
    )
    pipeline = pipeline_service_cls(client=client, config=config, cache=cache)
    result = _run_pipeline_or_exit(
        pipeline,
        folder,
        pipeline_options_cls(
            use_cache=cache is not None,
            use_embedding_cache=True,
            repair_noise=not no_knn_repair,
            merge_same_name=False,
            exclude_paths=[target.resolve()],
        ),
        cache,
    )

    if not result.files:
        if cache:
            cache.close()
        logger.error(t("scan_no_files"))
        raise typer.Exit(1)

    if result.noise_repaired > 0:
        logger.debug(t("scan_status_knn_suffix", count=result.noise_repaired))
    if result.cluster_metrics.small_clusters_merged > 0:
        logger.debug(
            t(
                "scan_status_small_merged_suffix",
                count=result.cluster_metrics.small_clusters_merged,
            )
        )
    if result.cluster_metrics.name_clusters_merged > 0:
        logger.debug(
            t(
                "scan_status_name_merged_suffix",
                count=result.cluster_metrics.name_clusters_merged,
            )
        )

    # 关闭缓存
    if cache:
        cache.close()

    # 构建预览
    preview = organize_preview_cls(
        source_folder=folder,
        target_folder=target,
    )

    _apply_clustering_to_preview(
        preview=preview,
        files=result.files,
        labels=result.labels,
        cluster_names=result.cluster_names,
    )
    _finalize_organize_run(
        preview=preview,
        logger=logger,
        output_script=output_script,
        execute=execute,
    )


@app.command(help=_help_key("assign_description"))
def assign(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_folder_assign")),
    ],
    target: Annotated[
        Path | None,
        typer.Option("--target", "-t", help=_help_key("cli_help_assign_target")),
    ] = None,
    min_similarity: Annotated[
        float | None,
        typer.Option(
            "--min-similarity", help=_help_key("cli_help_min_similarity")
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help=_help_key("cli_help_preview_mode")),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help=_help_key("cli_help_execute_move")),
    ] = False,
    output_script: Annotated[
        Path | None,
        typer.Option("--output-script", help=_help_key("cli_help_output_script")),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=_help_key("cli_help_disable_cache")),
    ] = False,
):
    """Sort new files into the existing subfolders of a target folder."""
    from rich.markup import escape

    from dite.flow.assign import (
        assign_to_folders,
        build_folder_profiles,
        list_category_folders,
    )
    from dite.flow.scan import scan_files

    if verbose or quiet:
        setup_logging(verbose=verbose, quiet=quiet)
    logger = get_logger()
    config = _get_app_config(ctx)

    if not dry_run and not execute and not output_script:
        logger.warning(t("organize_specify_mode"))
        logger.print(t("organize_mode_help"))
        raise typer.Exit(1)
    if not folder.is_dir():
        logger.error(t("organize_folder_not_found", folder=escape(str(folder))))
        raise typer.Exit(1)

    target_folder = target if target is not None else folder
    if not target_folder.is_dir():
        logger.error(
            t("organize_folder_not_found", folder=escape(str(target_folder)))
        )
        raise typer.Exit(1)

    extensions = config.formats.all_extensions
    members_by_folder: dict[str, list[Path]] = {}
    for category in list_category_folders(target_folder):
        members = scan_files(category, config=config, extensions=extensions)
        if members:
            members_by_folder[category.name] = members
    if not members_by_folder:
        logger.error(t("assign_no_categories", target=escape(str(target_folder))))
        raise typer.Exit(1)

    if target is None:
        # Loose files at the top level are new; subfolders are categories.
        new_files = scan_files(
            folder, config=config, extensions=extensions, recursive=False
        )
    else:
        new_files = scan_files(
            folder,
            config=config,
            extensions=extensions,
            exclude_paths=[target_folder.resolve()],
        )
    if not new_files:
        logger.print(t("assign_no_new_files"))
        return

    logger.print(f"[bold blue]{t('cli_title')}[/bold blue]\n")
    member_files = [file for files in members_by_folder.values() for file in files]
    cache = (
        FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
        if not no_cache and config.cache.enabled
        else None
    )
    pipeline = _get_pipeline_service_cls()(
        client=_get_client(config), config=config, cache=cache
    )
    options = _get_pipeline_options_cls()(
        use_cache=cache is not None, use_embedding_cache=True
    )
    embedded = _call_pipeline_or_exit(
        lambda: pipeline.embed_files([*member_files, *new_files], options), cache
    )
    if cache:
        cache.close()

    vectors_by_file = dict(zip(embedded.files, embedded.embeddings, strict=True))
    profiles = build_folder_profiles(
        {
            name: np.stack([vectors_by_file[file] for file in files])
            for name, files in members_by_folder.items()
        }
    )
    decisions = assign_to_folders(
        new_files,
        np.stack([vectors_by_file[file] for file in new_files]),
        profiles,
        min_similarity=min_similarity,
    )

    preview = _get_organize_preview_cls()(
        source_folder=folder, target_folder=target_folder
    )
    assigned_by_folder: dict[str, list[Path]] = {}
    unsorted: list[Path] = []
    for decision in decisions:
        if decision.folder is None:
            unsorted.append(decision.file)
            if decision.similarity is not None:
                logger.debug(
                    t(
                        "assign_decision_unsorted",
                        file=escape(decision.file.name),
                        folder=escape(decision.closest or "-"),
                        similarity=decision.similarity,
                        floor=(
                            "-" if decision.floor is None else f"{decision.floor:.3f}"
                        ),
                    )
                )
            continue
        assigned_by_folder.setdefault(decision.folder, []).append(decision.file)
        logger.debug(
            t(
                "assign_decision",
                file=escape(decision.file.name),
                folder=escape(decision.folder),
                similarity=decision.similarity,
                floor=decision.floor,
            )
        )
    for name in sorted(assigned_by_folder):
        preview.add_existing_folder(name, assigned_by_folder[name])
    preview.add_noise(unsorted)

    logger.print(
        t(
            "assign_summary",
            assigned=sum(len(files) for files in assigned_by_folder.values()),
            unsorted=len(unsorted),
        )
    )
    _finalize_organize_run(
        preview=preview,
        logger=logger,
        output_script=output_script,
        execute=execute,
    )


@app.command(help=_help_key("undo_description"))
def undo(
    path: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_undo_path")),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help=_help_key("cli_help_undo_dry_run")),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help=_help_key("cli_help_yes")),
    ] = False,
):
    """Undo the file moves recorded by organize/assign --execute."""
    from rich.markup import escape

    from dite.flow.move import find_latest_journal, read_journal, undo_journal

    logger = get_logger()
    journal_path = path if path.is_file() else find_latest_journal(path)
    if journal_path is None:
        logger.error(t("undo_no_journal", path=escape(str(path))))
        raise typer.Exit(1)
    try:
        contents = read_journal(journal_path)
    except (OSError, ValueError) as exc:
        logger.error(
            f"{t('undo_no_journal', path=escape(str(journal_path)))} "
            f"({escape(str(exc))})"
        )
        raise typer.Exit(1) from exc
    if contents.undone:
        logger.error(t("undo_already_done", path=escape(str(journal_path))))
        raise typer.Exit(1)

    plan = undo_journal(journal_path, dry_run=True)
    logger.print(
        t(
            "undo_preview",
            count=len(plan.restored),
            path=escape(str(journal_path)),
        )
    )
    for destination, source in plan.restored[:10]:
        logger.print(f"  {escape(str(destination))} -> {escape(str(source))}")
    if len(plan.restored) > 10:
        logger.print(t("organize_more_files", count=len(plan.restored) - 10))
    _print_undo_skips(logger, plan.skipped)
    if dry_run:
        return
    if not plan.restored:
        logger.success(t("undo_done", restored=0, skipped=len(plan.skipped)))
        return
    if not yes and not typer.confirm(t("undo_confirm")):
        logger.print(t("organize_cancelled"))
        return

    result = undo_journal(journal_path)
    _print_undo_skips(logger, result.skipped)
    logger.success(
        t("undo_done", restored=len(result.restored), skipped=len(result.skipped))
    )


def _print_undo_skips(logger, skipped: list[tuple[Path, str]]) -> None:
    from rich.markup import escape

    for skipped_path, reason in skipped:
        shown = escape(str(skipped_path))
        if reason == "missing":
            logger.warning(t("undo_skipped_missing", path=shown))
        elif reason == "conflict":
            logger.warning(t("undo_skipped_conflict", path=shown))
        else:
            logger.warning(f"{shown}: {escape(reason)}")


@cache_app.command("clear", help=_help_key("cli_help_cache_clear"))
def cache_clear(ctx: typer.Context):
    """Clear all cache."""
    logger = get_logger()
    config = _get_app_config(ctx)
    cache = FileCache(
        cache_dir=config.cache.directory,
        max_size_gb=config.cache.max_size_gb,
    )

    count = cache.clear()
    cache.close()

    logger.success(t("cache_cleared", count=count))


@cache_app.command("clear-vlm", help=_help_key("cli_help_cache_clear_vlm"))
def cache_clear_vlm(ctx: typer.Context):
    """Clear VLM cache only."""
    logger = get_logger()
    config = _get_app_config(ctx)
    cache = FileCache(
        cache_dir=config.cache.directory,
        max_size_gb=config.cache.max_size_gb,
    )

    result = cache.clear_vlm_cache()
    cache.close()

    logger.success(
        t(
            "cache_vlm_cleared",
            count=result.vlm_entries,
            embeddings=result.embeddings,
        )
    )


@cache_app.command("status", help=_help_key("cli_help_cache_status"))
def cache_status(ctx: typer.Context):
    """View cache status."""
    logger = get_logger()
    config = _get_app_config(ctx)
    cache = FileCache(
        cache_dir=config.cache.directory,
        max_size_gb=config.cache.max_size_gb,
    )

    embedding_cache_version = get_embedding_cache_version(config.models.embedding)
    stats = cache.get_stats(required_embedding_version=embedding_cache_version)
    cache.close()

    logger.print(f"[bold]{t('cache_status_title')}[/bold]\n")
    logger.print(f"[bold]{t('cache_db_path')}[/bold] {stats['db_path']}")
    logger.print(f"[bold]{t('cache_total_entries')}[/bold] {stats['total_entries']}")
    logger.print(f"[bold]{t('cache_with_embedding')}[/bold] {stats['with_embedding']}")
    logger.print(
        f"[bold]{t('cache_current_embedding')}[/bold] {stats['current_embeddings']}"
    )
    logger.print(
        f"[bold]{t('cache_stale_embedding')}[/bold] {stats['stale_embeddings']}"
    )
    logger.print(
        f"[bold]{t('cache_embedding_version')}[/bold] "
        f"{stats['current_embedding_version']}"
    )
    logger.print(f"[bold]{t('cache_with_vlm')}[/bold] {stats['with_vlm']}")
    logger.print(f"[bold]{t('cache_vlm_version')}[/bold] v{stats['vlm_cache_version']}")
    logger.print(f"[bold]{t('cache_unique_hashes')}[/bold] {stats['unique_hashes']}")
    logger.print(f"[bold]{t('cache_db_size')}[/bold] {stats['db_size_mb']:.2f} MB")


@setup_app.command("docling-pdf", help=_help_key("cli_help_setup_docling_pdf"))
def setup_docling_pdf(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=_help_key("cli_help_setup_docling_pdf_force"),
        ),
    ] = False,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress",
            help=_help_key("cli_help_setup_docling_pdf_progress"),
        ),
    ] = False,
):
    """Install local Docling PDF models required by DITE."""
    logger = get_logger()
    (
        download_docling_pdf_models_fn,
        get_docling_pdf_artifacts_path_fn,
        has_docling_pdf_artifacts_fn,
    ) = _get_docling_helpers()
    target_dir = get_docling_pdf_artifacts_path_fn()
    logger.print(t("setup_docling_pdf_start", path=target_dir))

    try:
        download_docling_pdf_models_fn(
            output_dir=target_dir,
            force=force,
            progress=progress,
        )
    except Exception as exc:
        logger.error(t("setup_docling_pdf_failed", error=exc))
        raise typer.Exit(1) from exc

    if not has_docling_pdf_artifacts_fn(target_dir):
        logger.error(t("setup_docling_pdf_incomplete", path=target_dir))
        raise typer.Exit(1)

    logger.success(t("setup_docling_pdf_done", path=target_dir))


if __name__ == "__main__":
    app()
