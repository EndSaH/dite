"""Canonical extraction stage helpers for the shared pipeline."""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import hashlib
import itertools
import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from dite.app.i18n import t
from dite.cache import VLM_CACHE_VERSION, FileCache
from dite.io import get_extractor
from dite.util.cancel import cancellation_scope, raise_if_cancelled
from dite.util.hash import compute_file_hash

from .model import (
    ExtractionFileReport,
    ExtractionSummary,
    ExtractionSummaryDelta,
    ExtractionWorkItem,
    ExtractionWorkResult,
)


@dataclass(frozen=True)
class ExtractionStageResult:
    """Extraction stage output before clustering."""

    contents: list[str]
    file_hashes: list[str]
    summary: ExtractionSummary
    file_reports: list[ExtractionFileReport]


def merge_summary_delta(
    summary: ExtractionSummary,
    delta: ExtractionSummaryDelta,
) -> None:
    """Merge per-file extraction deltas into the stage summary."""
    summary.doc_cache_hits += delta.doc_cache_hits
    summary.vlm_cache_hits += delta.vlm_cache_hits
    summary.primary_failures += delta.primary_failures
    summary.source_fallback_needed += delta.source_fallback_needed
    summary.selected_vlm_files += delta.selected_vlm_files
    summary.vlm_api_page_calls += delta.vlm_api_page_calls


# Longest exception text kept in a failed file's report; library errors can
# carry whole payloads.
_FAILURE_MESSAGE_LIMIT = 200


def _unreadable_file_hash(file: Path) -> str:
    """Stand-in hash for a file that could not be read.

    Derived from the path, so each unreadable file stays its own canonical
    sample and never matches a real content hash in the caches.
    """
    return "unreadable:" + hashlib.sha256(os.fsencode(file)).hexdigest()


def _describe_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    if len(message) > _FAILURE_MESSAGE_LIMIT:
        message = message[:_FAILURE_MESSAGE_LIMIT] + "..."
    return message


# A cache database that is locked, full, read-only or corrupt, or a file whose
# mtime can no longer be read for its cache row.
_CACHE_ERRORS = (sqlite3.Error, OSError)


class BestEffortCache:
    """Cache access for one file that never fails the file.

    The cache only saves work. When a lookup or a write raises, a warning is
    logged, the cache is left alone for the rest of this file, and the file is
    extracted and keeps its content as if it had no cache.
    """

    def __init__(self, cache, *, enabled: bool, file: Path, logger) -> None:
        self._cache = cache if enabled and cache else None
        self._file = file
        self._logger = logger

    @property
    def available(self) -> bool:
        return self._cache is not None

    def run[T](self, operation: Callable[[FileCache], T], default: T) -> T:
        """Return ``operation(cache)``, or ``default`` if the cache is unusable."""
        if self._cache is None:
            return default
        try:
            return operation(self._cache)
        except _CACHE_ERRORS as exc:
            # A COMMIT that failed with "database is locked" leaves its
            # transaction open, and the connection would keep the database
            # locked for every other thread and process.
            with contextlib.suppress(sqlite3.Error):
                self._cache.rollback()
            self._cache = None
            self._logger.warning(
                t(
                    "warning_extract_cache_failed",
                    path=escape(str(self._file)),
                    error=escape(_describe_error(exc)),
                )
            )
            return default


def _failed_work_result(
    item: ExtractionWorkItem,
    error: Exception,
    logger,
) -> ExtractionWorkResult:
    """Record a file whose extraction raised, so the rest of the run goes on."""
    message = _describe_error(error)
    logger.warning(
        t(
            "warning_extract_file_failed",
            path=escape(str(item.file)),
            error=escape(message),
        )
    )
    return ExtractionWorkResult(
        index=item.index,
        content="",
        file_hash=item.file_hash,
        report=ExtractionFileReport(
            file=item.file,
            primary_extractor="none",
            primary_success=False,
            primary_error=message,
            source_profile=None,
            source_effective_length=0,
            selected_source="primary",
            final_effective_length=0,
            excerpt_was_truncated=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
            file_hash=item.file_hash,
        ),
        summary_delta=ExtractionSummaryDelta(primary_failures=1),
    )


def extract_content_work_item(
    *,
    item: ExtractionWorkItem,
    options,
    truncate_limit: int,
    config,
    client,
    cache,
    logger,
    request_runtime,
    make_extractor_registry,
    extract_primary_result,
    build_cached_primary_result,
    supports_keyword_arg,
    resolve_document_extraction,
) -> ExtractionWorkResult:
    """Extract one canonical file and return its content, report, and summary delta."""
    file = item.file
    file_hash = item.file_hash
    registry = make_extractor_registry()
    logger.debug(t("debug_extract_processing_file", path=file))
    if file_hash:
        logger.debug(t("debug_extract_hash", hash=file_hash[:12]))

    doc_cache_hits = 0
    vlm_cache_hits = 0
    primary_result = None
    cached_vlm_content = None
    item_cache = BestEffortCache(
        cache, enabled=options.use_cache, file=file, logger=logger
    )

    if item_cache.available:
        cached_primary_content, source_file = item_cache.run(
            lambda cache: cache.get_content(file, file_hash), (None, None)
        )
        if cached_primary_content:
            extractor = get_extractor(
                file,
                client,
                config=config,
                registry=registry,
            )
            primary_result = build_cached_primary_result(
                file,
                cached_primary_content,
                extractor.name if extractor is not None else "cache",
            )
            doc_cache_hits = 1
            logger.debug(t("debug_extract_doc_cache_hit"))
            if source_file:
                logger.debug(
                    t(
                        "debug_extract_doc_cache_duplicate_source",
                        source=Path(source_file).name,
                    )
                )
        else:
            logger.debug(t("debug_extract_doc_cache_miss"))

    if file.suffix.lower() == ".pdf" and item_cache.available:
        cached_vlm_content = item_cache.run(
            lambda cache: cache.get_vlm_content(file, file_hash, VLM_CACHE_VERSION),
            None,
        )
        if cached_vlm_content:
            vlm_cache_hits = 1
            logger.debug(
                t(
                    "debug_extract_vlm_cache_hit",
                    length=len(cached_vlm_content),
                )
            )

    if primary_result is None:
        primary_result = extract_primary_result(file, registry)
        logger.debug(
            t(
                "debug_extract_doc_result",
                extractor=primary_result.extractor,
                success=primary_result.success,
                length=len(primary_result.content if primary_result.success else ""),
                error=primary_result.error or "-",
            )
        )

    resolve_kwargs = {
        "config": config,
        "enable_vlm_fallback": True,
        "allow_vlm_api": options.allow_vlm_api,
        "cached_vlm_content": cached_vlm_content,
        "primary_result": primary_result,
        "registry": registry,
    }
    if request_runtime is not None and supports_keyword_arg(
        resolve_document_extraction, "request_runtime"
    ):
        resolve_kwargs["request_runtime"] = request_runtime
    resolved = resolve_document_extraction(
        file,
        client,
        **resolve_kwargs,
    )

    if item_cache.available and resolved.primary_result.success and doc_cache_hits == 0:
        item_cache.run(
            lambda cache: cache.save(
                file_path=file,
                file_hash=file_hash,
                file_mtime=file.stat().st_mtime,
                content_md=resolved.primary_result.content,
                model_version=config.models.embedding,
                enforce_size_limit=False,
            ),
            None,
        )

    if (
        resolved.cache_write_intent.should_write
        and resolved.cache_write_intent.content
        and item_cache.available
    ):
        item_cache.run(
            lambda cache: cache.update_vlm_content(
                file,
                file_hash,
                resolved.cache_write_intent.content,
                VLM_CACHE_VERSION,
                enforce_size_limit=False,
            ),
            None,
        )

    from dite.doc.embed import ContentTruncator

    if len(resolved.final_content) > truncate_limit:
        logger.debug(
            t(
                "debug_extract_truncated",
                original=len(resolved.final_content),
                limit=truncate_limit,
            )
        )

    return ExtractionWorkResult(
        index=item.index,
        content=ContentTruncator.truncate_smart(
            resolved.final_content,
            max_chars=truncate_limit,
            head_ratio=config.processing.excerpt_head_ratio,
            middle_ratio=config.processing.excerpt_middle_ratio,
        ),
        file_hash=file_hash,
        report=ExtractionFileReport(
            file=file,
            primary_extractor=resolved.primary_result.extractor,
            primary_success=resolved.primary_result.success,
            primary_error=resolved.primary_result.error,
            source_profile=resolved.pdf_profile.kind if resolved.pdf_profile else None,
            source_effective_length=resolved.primary_effective_length,
            selected_source=resolved.selected_source,
            final_effective_length=resolved.final_effective_length,
            excerpt_was_truncated=len(resolved.final_content) > truncate_limit,
            vlm_api_page_calls=resolved.vlm_api_page_calls,
            sample_page_limit=resolved.sample_page_limit,
            file_hash=file_hash,
            source_reason=resolved.pdf_profile.reason if resolved.pdf_profile else None,
            fallback_needed=resolved.fallback_needed,
        ),
        summary_delta=ExtractionSummaryDelta(
            doc_cache_hits=doc_cache_hits,
            vlm_cache_hits=vlm_cache_hits,
            primary_failures=1 if resolved.primary_result.success is False else 0,
            source_fallback_needed=1 if resolved.fallback_needed else 0,
            selected_vlm_files=1 if resolved.selected_source != "primary" else 0,
            vlm_api_page_calls=resolved.vlm_api_page_calls,
        ),
    )


def extract_stage(
    *,
    files: list[Path],
    options,
    config,
    client,
    cache,
    logger,
    request_runtime,
    make_extractor_registry,
    extract_primary_result,
    build_cached_primary_result,
    supports_keyword_arg,
    resolve_document_extraction,
    work_item_runner=None,
) -> ExtractionStageResult:
    """Run the full extraction stage before clustering."""
    contents: list[str | None] = [None] * len(files)
    file_hashes: list[str | None] = [None] * len(files)
    file_reports: list[ExtractionFileReport | None] = [None] * len(files)
    work_results: list[ExtractionWorkResult | None] = [None] * len(files)
    summary = ExtractionSummary()
    truncate_limit = config.processing.text_truncate_limit
    hash_errors: dict[int, Exception] = {}
    for index, file in enumerate(files):
        try:
            file_hashes[index] = compute_file_hash(file)
        except Exception as exc:
            # Deleted after the scan, no read permission, a cloud placeholder:
            # only this file fails.
            file_hashes[index] = _unreadable_file_hash(file)
            hash_errors[index] = exc
    hash_to_indices: dict[str, list[int]] = {}
    for index, file_hash in enumerate(file_hashes):
        hash_to_indices.setdefault(file_hash, []).append(index)
    canonical_indices = [indices[0] for indices in hash_to_indices.values()]
    extract_workers = max(
        1,
        min(config.processing.extract_workers, len(canonical_indices)),
    )
    docling_pdf_semaphore = threading.BoundedSemaphore(
        max(1, config.processing.docling_pdf_workers)
    )
    work_items = [
        ExtractionWorkItem(index=i, file=files[i], file_hash=file_hashes[i])
        for i in canonical_indices
    ]

    cancel_event = threading.Event()

    def run_item(item: ExtractionWorkItem) -> ExtractionWorkResult:
        with cancellation_scope(cancel_event):
            raise_if_cancelled()
            if item.index in hash_errors:
                return _failed_work_result(item, hash_errors[item.index], logger)
            try:
                return extract_item(item)
            except Exception as exc:
                return _failed_work_result(item, exc, logger)

    def extract_item(item: ExtractionWorkItem) -> ExtractionWorkResult:
        if work_item_runner is not None:
            return work_item_runner(
                item,
                options,
                truncate_limit,
                docling_pdf_semaphore,
            )
        return extract_content_work_item(
            item=item,
            options=options,
            truncate_limit=truncate_limit,
            config=config,
            client=client,
            cache=cache,
            logger=logger,
            request_runtime=request_runtime,
            make_extractor_registry=make_extractor_registry,
            extract_primary_result=lambda file, registry: extract_primary_result(
                file, registry, docling_pdf_semaphore
            ),
            build_cached_primary_result=build_cached_primary_result,
            supports_keyword_arg=supports_keyword_arg,
            resolve_document_extraction=resolve_document_extraction,
        )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=extract_workers)
    pending_items = iter(work_items)
    try:
        # Keep only as many files in flight as there are workers, so an
        # interrupt has no queue of files left to drain.
        in_flight = {
            executor.submit(run_item, item)
            for item in itertools.islice(pending_items, extract_workers)
        }
        while in_flight:
            done, in_flight = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                result = future.result()
                contents[result.index] = result.content
                file_hashes[result.index] = result.file_hash
                file_reports[result.index] = result.report
                work_results[result.index] = result
                merge_summary_delta(summary, result.summary_delta)
            for item in itertools.islice(pending_items, len(done)):
                in_flight.add(executor.submit(run_item, item))
    except BaseException:
        # Ctrl+C (or a bug in this loop): tell the running files to stop at
        # their next check, which terminates Docling subprocesses and skips
        # VLM requests, and wait for them so that callers can close the
        # request runtime and the cache safely.
        cancel_event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)

    for indices in hash_to_indices.values():
        canonical_index = indices[0]
        canonical_result = work_results[canonical_index]
        if canonical_result is None:
            raise RuntimeError("extraction worker did not populate canonical result")
        for index in indices[1:]:
            contents[index] = canonical_result.content
            file_reports[index] = dataclasses.replace(
                canonical_result.report,
                file=files[index],
            )
            work_results[index] = dataclasses.replace(
                canonical_result,
                index=index,
                report=file_reports[index],
            )

    if options.use_cache and cache:
        cache.enforce_size_limit()

    hash_to_files: dict[str, list[str]] = {}
    for file, file_hash in zip(files, file_hashes, strict=False):
        if file_hash:
            hash_to_files.setdefault(file_hash, []).append(str(file))

    duplicate_groups = {
        file_hash: file_list
        for file_hash, file_list in hash_to_files.items()
        if file_hash and len(file_list) > 1
    }
    summary.duplicate_groups = duplicate_groups
    summary.duplicate_count = sum(
        len(file_list) - 1 for file_list in duplicate_groups.values()
    )
    summary.doc_cache_hits = 0
    for _file_hash, indices in hash_to_indices.items():
        actual_hits = sum(
            work_results[index].summary_delta.doc_cache_hits
            for index in indices[:1]
            if work_results[index] is not None
        )
        summary.doc_cache_hits += actual_hits + len(indices) - 1

    logger.debug(
        t(
            "debug_extract_summary",
            doc_cache_hits=summary.doc_cache_hits,
            vlm_cache_hits=summary.vlm_cache_hits,
            primary_failures=summary.primary_failures,
            source_fallback_needed=summary.source_fallback_needed,
            selected_vlm_files=summary.selected_vlm_files,
            vlm_api_page_calls=summary.vlm_api_page_calls,
            duplicates=summary.duplicate_count,
        )
    )

    if any(content is None for content in contents):
        raise RuntimeError("extraction worker did not populate all contents")
    if any(file_hash is None for file_hash in file_hashes):
        raise RuntimeError("extraction worker did not populate all file hashes")
    if any(report is None for report in file_reports):
        raise RuntimeError("extraction worker did not populate all file reports")
    if any(result is None for result in work_results):
        raise RuntimeError("extraction worker did not populate all work results")

    return ExtractionStageResult(
        contents=[content for content in contents if content is not None],
        file_hashes=file_hashes,
        summary=summary,
        file_reports=[report for report in file_reports if report is not None],
    )
