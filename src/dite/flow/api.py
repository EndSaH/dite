"""Shared document processing pipeline for scan and organize commands."""

import inspect
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from openai import OpenAI

from dite.app.config import Config
from dite.app.i18n import get_locale, t
from dite.cache import VLM_CACHE_VERSION, FileCache
from dite.cluster.api import (
    cluster_documents,
    generate_all_cluster_names,
)
from dite.cluster.stage import (
    build_canonical_cluster_stage,
)
from dite.cluster.view import build_cluster_representations
from dite.doc import DocumentFeatures
from dite.doc.embed import (
    ContentTruncator,
    get_embeddings,
    normalize_embeddings,
)
from dite.io import (
    ExtractorRegistry,
    get_extractor,
)
from dite.io.base import ExtractionResult
from dite.io.docling import DoclingExtractor, extract_docling_pdf_in_subprocess
from dite.io.route import (
    resolve_document_extraction,
)
from dite.util.api import AsyncRequestRuntime
from dite.util.log import get_logger

from .extract import BestEffortCache, extract_stage
from .model import (
    ExtractionFileReport,
    ExtractionSummary,
    ExtractionSummaryDelta,
    ExtractionWorkItem,
    ExtractionWorkResult,
    PipelineOptions,
    PipelineResult,
)
from .scan import scan_files
from .vector import (
    expand_by_file_hashes,
    expand_document_features_by_file_hashes,
    expand_noise_repaired_count,
    vectorize_files,
)


class PipelineService:
    """Unified scan/extract/embed/cluster/name pipeline."""

    def __init__(
        self,
        client: OpenAI,
        *,
        config: Config,
        cache: FileCache | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.cache = cache
        self.logger = get_logger()
        self._request_runtime: AsyncRequestRuntime | None = None

    def _make_extractor_registry(self) -> ExtractorRegistry:
        return ExtractorRegistry(config=self.config, client=self.client)

    @staticmethod
    def _supports_keyword_arg(func: object, keyword: str) -> bool:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return keyword in signature.parameters

    @contextmanager
    def _request_runtime_scope(self):
        runtime: AsyncRequestRuntime | None = None
        if (
            isinstance(self.client, OpenAI)
            and self.config.api.base_url
            and self.config.api.api_key
        ):
            runtime = AsyncRequestRuntime(self.config)
        previous_runtime = self._request_runtime
        if runtime is not None:
            runtime.start()
        self._request_runtime = runtime
        try:
            yield runtime
        finally:
            self._request_runtime = previous_runtime
            if runtime is not None:
                runtime.close()

    @staticmethod
    def _merge_summary_delta(
        summary: ExtractionSummary,
        delta: ExtractionSummaryDelta,
    ) -> None:
        summary.doc_cache_hits += delta.doc_cache_hits
        summary.vlm_cache_hits += delta.vlm_cache_hits
        summary.primary_failures += delta.primary_failures
        summary.source_fallback_needed += delta.source_fallback_needed
        summary.selected_vlm_files += delta.selected_vlm_files
        summary.vlm_api_page_calls += delta.vlm_api_page_calls

    def _extract_docling_pdf_primary_result(
        self,
        file_path: Path,
        extractor: DoclingExtractor,
        semaphore: threading.BoundedSemaphore,
    ) -> ExtractionResult:
        with semaphore:
            extra_kwargs = {}
            max_pages = getattr(extractor, "_pdf_max_pages", None)
            if max_pages is not None:
                extra_kwargs["max_pages"] = max_pages
            return extract_docling_pdf_in_subprocess(
                file_path,
                enable_ocr=extractor._enable_ocr,
                artifacts_path=extractor._artifacts_path,
                timeout_sec=extractor._pdf_timeout_sec,
                locale=get_locale(),
                device=extractor._device,
                **extra_kwargs,
            )

    def _extract_primary_result(
        self,
        file_path: Path,
        registry: ExtractorRegistry,
        docling_pdf_semaphore: threading.BoundedSemaphore,
    ) -> ExtractionResult:
        extractor = get_extractor(
            file_path,
            self.client,
            config=self.config,
            registry=registry,
        )
        if (
            isinstance(extractor, DoclingExtractor)
            and file_path.suffix.lower() == ".pdf"
        ):
            return self._extract_docling_pdf_primary_result(
                file_path,
                extractor,
                docling_pdf_semaphore,
            )

        from dite.io import route as router

        extract_kwargs = {
            "config": self.config,
            "registry": registry,
        }
        if self._request_runtime is not None and self._supports_keyword_arg(
            router.extract_document, "request_runtime"
        ):
            extract_kwargs["request_runtime"] = self._request_runtime
        return router.extract_document(
            file_path,
            self.client,
            **extract_kwargs,
        )

    def _extract_content_work_item(
        self,
        item: ExtractionWorkItem,
        options: PipelineOptions,
        truncate_limit: int,
        docling_pdf_semaphore: threading.BoundedSemaphore,
    ) -> ExtractionWorkResult:
        file = item.file
        file_hash = item.file_hash
        registry = self._make_extractor_registry()
        self.logger.debug(t("debug_extract_processing_file", path=file))
        if file_hash:
            self.logger.debug(t("debug_extract_hash", hash=file_hash[:12]))

        doc_cache_hits = 0
        vlm_cache_hits = 0
        primary_result = None
        cached_vlm_content = None
        item_cache = BestEffortCache(
            self.cache, enabled=options.use_cache, file=file, logger=self.logger
        )

        if item_cache.available:
            cached_primary_content, source_file = item_cache.read(
                lambda cache: cache.get_content(file, file_hash), (None, None)
            )
            if cached_primary_content:
                extractor = get_extractor(
                    file,
                    self.client,
                    config=self.config,
                    registry=registry,
                )
                primary_result = self._build_cached_primary_result(
                    file,
                    cached_primary_content,
                    extractor.name if extractor is not None else "cache",
                )
                doc_cache_hits = 1
                self.logger.debug(t("debug_extract_doc_cache_hit"))
                if source_file:
                    self.logger.debug(
                        t(
                            "debug_extract_doc_cache_duplicate_source",
                            source=Path(source_file).name,
                        )
                    )
            else:
                self.logger.debug(t("debug_extract_doc_cache_miss"))

        if file.suffix.lower() == ".pdf" and item_cache.available:
            cached_vlm_content = item_cache.read(
                lambda cache: cache.get_vlm_content(
                    file,
                    file_hash,
                    VLM_CACHE_VERSION,
                ),
                None,
            )
            if cached_vlm_content:
                vlm_cache_hits = 1
                self.logger.debug(
                    t(
                        "debug_extract_vlm_cache_hit",
                        length=len(cached_vlm_content),
                    )
                )

        if primary_result is None:
            primary_result = self._extract_primary_result(
                file,
                registry,
                docling_pdf_semaphore,
            )
            self.logger.debug(
                t(
                    "debug_extract_doc_result",
                    extractor=primary_result.extractor,
                    success=primary_result.success,
                    length=len(
                        primary_result.content if primary_result.success else ""
                    ),
                    error=primary_result.error or "-",
                )
            )

        resolve_kwargs = {
            "config": self.config,
            "enable_vlm_fallback": True,
            "allow_vlm_api": options.allow_vlm_api,
            "cached_vlm_content": cached_vlm_content,
            "primary_result": primary_result,
            "registry": registry,
        }
        if self._request_runtime is not None and self._supports_keyword_arg(
            resolve_document_extraction, "request_runtime"
        ):
            resolve_kwargs["request_runtime"] = self._request_runtime
        resolved = resolve_document_extraction(
            file,
            self.client,
            **resolve_kwargs,
        )

        if (
            item_cache.available
            and resolved.primary_result.success
            and doc_cache_hits == 0
        ):
            item_cache.write(
                lambda cache: cache.save(
                    file_path=file,
                    file_hash=file_hash,
                    file_mtime=file.stat().st_mtime,
                    content_md=resolved.primary_result.content,
                    model_version=self.config.models.embedding,
                    enforce_size_limit=False,
                )
            )

        if (
            resolved.cache_write_intent.should_write
            and resolved.cache_write_intent.content
            and item_cache.available
        ):
            item_cache.write(
                lambda cache: cache.update_vlm_content(
                    file,
                    file_hash,
                    resolved.cache_write_intent.content,
                    VLM_CACHE_VERSION,
                    enforce_size_limit=False,
                )
            )

        if len(resolved.final_content) > truncate_limit:
            self.logger.debug(
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
                head_ratio=self.config.processing.excerpt_head_ratio,
                middle_ratio=self.config.processing.excerpt_middle_ratio,
            ),
            file_hash=file_hash,
            report=ExtractionFileReport(
                file=file,
                primary_extractor=resolved.primary_result.extractor,
                primary_success=resolved.primary_result.success,
                primary_error=resolved.primary_result.error,
                source_profile=resolved.pdf_profile.kind
                if resolved.pdf_profile
                else None,
                source_effective_length=resolved.primary_effective_length,
                selected_source=resolved.selected_source,
                final_effective_length=resolved.final_effective_length,
                excerpt_was_truncated=len(resolved.final_content) > truncate_limit,
                vlm_api_page_calls=resolved.vlm_api_page_calls,
                sample_page_limit=resolved.sample_page_limit,
                file_hash=file_hash,
                source_reason=resolved.pdf_profile.reason
                if resolved.pdf_profile
                else None,
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

    def run(self, folder: Path, options: PipelineOptions) -> PipelineResult:
        self.logger = get_logger()
        files = scan_files(
            folder,
            config=self.config,
            extensions=self.config.formats.all_extensions,
            exclude_paths=options.exclude_paths,
        )
        if not files:
            return PipelineResult(
                files=[],
                contents=[],
                document_features=[],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
                cluster_drafts=[],
                extraction=ExtractionSummary(),
                file_reports=[],
            )

        with self._request_runtime_scope() as request_runtime:
            contents, file_hashes, extraction, file_reports = self._extract_contents(
                files, options
            )
            canonical_indices = self._canonical_indices(file_hashes)
            cluster_stage = build_canonical_cluster_stage(
                files=files,
                contents=contents,
                file_hashes=file_hashes,
                file_reports=file_reports,
                canonical_indices=canonical_indices,
                options=options,
                config=self.config,
                client=self.client,
                request_runtime=request_runtime,
                vectorize=self._vectorize,
                expand_noise_repaired_count=self._expand_noise_repaired_count,
                cluster_documents_fn=cluster_documents,
            )
            embeddings = self._expand_by_file_hashes(
                canonical_indices,
                cluster_stage.canonical_embeddings,
                file_hashes,
                len(files),
            )
            noise_repaired = self._expand_noise_repaired_count(
                canonical_indices,
                cluster_stage.cluster_result.repaired_mask,
                file_hashes,
            )
            naming_kwargs = {
                "config": self.config,
                "embeddings": cluster_stage.canonical_embeddings,
                "merge_same_name": options.merge_same_name,
                "llm_model": self.config.models.llm,
            }
            if request_runtime is not None and self._supports_keyword_arg(
                generate_all_cluster_names, "request_runtime"
            ):
                naming_kwargs["request_runtime"] = request_runtime
            canonical_files = [files[index] for index in canonical_indices]
            canonical_contents = [contents[index] for index in canonical_indices]
            cluster_result = generate_all_cluster_names(
                self.client,
                cluster_stage.cluster_result,
                canonical_contents,
                canonical_files,
                **naming_kwargs,
            )
            labels = self._expand_by_file_hashes(
                canonical_indices,
                cluster_result.labels,
                file_hashes,
                len(files),
            )
            expanded_document_features = self._expand_document_features_by_file_hashes(
                canonical_indices,
                cluster_stage.document_features,
                file_hashes,
            )

        return PipelineResult(
            files=files,
            contents=contents,
            document_features=expanded_document_features,
            candidate_edges=cluster_stage.candidate_edges,
            candidate_components=cluster_stage.candidate_components,
            cluster_drafts=cluster_stage.cluster_drafts,
            adjudication_requests=cluster_stage.adjudication_requests,
            adjudication_decisions=cluster_stage.adjudication_decisions,
            embeddings=embeddings,
            labels=labels,
            cluster_names=cluster_result.cluster_names,
            cluster_representations=build_cluster_representations(
                labels=labels,
                cluster_names=cluster_result.cluster_names,
                document_features=expanded_document_features,
                config=self.config,
                client=self.client,
                request_runtime=request_runtime,
            ),
            noise_repaired=noise_repaired,
            clusters_merged=cluster_result.total_clusters_merged,
            cluster_metrics=cluster_result.metrics,
            extraction=extraction,
            file_reports=file_reports,
        )

    def extract_files(
        self, files: list[Path], options: PipelineOptions
    ) -> PipelineResult:
        """Extract file contents without embedding, clustering, or naming."""
        self.logger = get_logger()
        if not files:
            return PipelineResult(
                files=[],
                contents=[],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
                extraction=ExtractionSummary(),
                file_reports=[],
            )

        with self._request_runtime_scope():
            extraction_stage = extract_stage(
                files=files,
                options=options,
                config=self.config,
                client=self.client,
                cache=self.cache,
                logger=self.logger,
                request_runtime=self._request_runtime,
                make_extractor_registry=self._make_extractor_registry,
                extract_primary_result=self._extract_primary_result,
                build_cached_primary_result=self._build_cached_primary_result,
                supports_keyword_arg=self._supports_keyword_arg,
                resolve_document_extraction=resolve_document_extraction,
            )
        return PipelineResult(
            files=files,
            contents=extraction_stage.contents,
            embeddings=np.array([]),
            labels=np.array([]),
            cluster_names={},
            cluster_representations={},
            cluster_drafts=[],
            extraction=extraction_stage.summary,
            file_reports=extraction_stage.file_reports,
        )

    def embed_files(
        self, files: list[Path], options: PipelineOptions
    ) -> PipelineResult:
        """Extract and embed files without clustering or naming.

        Uses the same content and embedding caches as ``run``; identical
        files are embedded once. Labels are all ``-1``.
        """
        self.logger = get_logger()
        if not files:
            return PipelineResult(
                files=[],
                contents=[],
                embeddings=np.array([]),
                labels=np.array([], dtype=int),
                cluster_names={},
                extraction=ExtractionSummary(),
                file_reports=[],
            )

        with self._request_runtime_scope():
            contents, file_hashes, extraction, file_reports = self._extract_contents(
                files, options
            )
            canonical_indices = self._canonical_indices(file_hashes)
            canonical_files = [files[index] for index in canonical_indices]
            canonical_hashes = [file_hashes[index] for index in canonical_indices]
            texts = [contents[index] for index in canonical_indices]
            if options.embedding_input_mode == "summary":
                from dite.doc import build_document_features
                from dite.doc.build import build_summary_embedding_text

                texts = [
                    build_summary_embedding_text(
                        build_document_features(
                            files[index],
                            contents[index],
                            config=self.config,
                            client=self.client,
                            file_report=file_reports[index],
                        )
                    )
                    for index in canonical_indices
                ]
            canonical_embeddings = normalize_embeddings(
                self._vectorize(canonical_files, canonical_hashes, texts, options)
            )
            embeddings = self._expand_by_file_hashes(
                canonical_indices,
                canonical_embeddings,
                file_hashes,
                len(files),
            )

        return PipelineResult(
            files=files,
            contents=contents,
            embeddings=embeddings,
            labels=np.full(len(files), -1, dtype=int),
            cluster_names={},
            extraction=extraction,
            file_reports=file_reports,
        )

    def _extract_contents(
        self,
        files: list[Path],
        options: PipelineOptions,
    ) -> tuple[list[str], list[str], ExtractionSummary, list[ExtractionFileReport]]:
        stage = extract_stage(
            files=files,
            options=options,
            config=self.config,
            client=self.client,
            cache=self.cache,
            logger=self.logger,
            request_runtime=self._request_runtime,
            make_extractor_registry=self._make_extractor_registry,
            extract_primary_result=self._extract_primary_result,
            build_cached_primary_result=self._build_cached_primary_result,
            supports_keyword_arg=self._supports_keyword_arg,
            resolve_document_extraction=resolve_document_extraction,
            work_item_runner=(
                lambda item, options, truncate_limit, docling_pdf_semaphore: (
                    self._extract_content_work_item(
                        item,
                        options,
                        truncate_limit,
                        docling_pdf_semaphore,
                    )
                )
            ),
        )
        return (
            stage.contents,
            stage.file_hashes,
            stage.summary,
            stage.file_reports,
        )

    @staticmethod
    def _hash_to_indices(file_hashes: list[str]) -> dict[str, list[int]]:
        hash_to_indices: dict[str, list[int]] = {}
        for index, file_hash in enumerate(file_hashes):
            hash_to_indices.setdefault(file_hash, []).append(index)
        return hash_to_indices

    @classmethod
    def _canonical_indices(cls, file_hashes: list[str]) -> list[int]:
        return [indices[0] for indices in cls._hash_to_indices(file_hashes).values()]

    @classmethod
    def _expand_by_file_hashes(
        cls,
        canonical_indices: list[int],
        canonical_values: np.ndarray,
        file_hashes: list[str],
        total_count: int,
    ) -> np.ndarray:
        return expand_by_file_hashes(
            canonical_indices,
            canonical_values,
            file_hashes,
            total_count,
        )

    @classmethod
    def _expand_noise_repaired_count(
        cls,
        canonical_indices: list[int],
        canonical_repaired_mask: np.ndarray,
        file_hashes: list[str],
    ) -> int:
        return expand_noise_repaired_count(
            canonical_indices,
            canonical_repaired_mask,
            file_hashes,
        )

    @classmethod
    def _expand_document_features_by_file_hashes(
        cls,
        canonical_indices: list[int],
        canonical_document_features: list[DocumentFeatures],
        file_hashes: list[str],
    ) -> list[DocumentFeatures]:
        return expand_document_features_by_file_hashes(
            canonical_indices,
            canonical_document_features,
            file_hashes,
        )

    @staticmethod
    def _build_cached_primary_result(
        _file_path: Path,
        content: str,
        extractor_name: str,
    ) -> ExtractionResult:
        return ExtractionResult(
            content=content,
            success=True,
            extractor=extractor_name,
        )

    def _vectorize(
        self,
        files: list[Path],
        file_hashes: list[str],
        contents: list[str],
        options: PipelineOptions,
    ) -> np.ndarray:
        return vectorize_files(
            files=files,
            file_hashes=file_hashes,
            contents=contents,
            options=options,
            config=self.config,
            cache=self.cache,
            client=self.client,
            logger=self.logger,
            get_embeddings_fn=get_embeddings,
        )
