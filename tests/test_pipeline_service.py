import concurrent.futures
import dataclasses
import shutil
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from openai import APIStatusError

from dite.app.config import (
    ChatCompletionProfileConfig,
    Config,
    RequestProfilesConfig,
)
from dite.app.i18n import set_locale
from dite.cache import FileCache
from dite.cluster.api import (
    ClusterMetrics,
    ClusterResult,
    cluster_documents,
    repair_noise_with_knn,
)
from dite.doc.embed import get_embedding_cache_version
from dite.flow.api import (
    ExtractionFileReport,
    ExtractionSummary,
    ExtractionSummaryDelta,
    ExtractionWorkResult,
    PipelineOptions,
    PipelineService,
)
from dite.flow.scan import scan_files
from dite.io.base import ExtractionResult
from dite.io.docling import DoclingExtractor
from dite.io.pdf.final import PDFCacheWriteIntent
from dite.io.route import (
    PDF_VLM_SAMPLE_PAGE_LIMIT,
    PDFProfile,
    ResolvedExtraction,
    VLMSamplingResult,
)
from dite.util.api import ChatCompletionResult
from dite.util.hash import compute_file_hash
from dite.util.log import setup_logging

TEST_CORPUS_DIR = Path(__file__).resolve().parents[1] / "docs" / "test" / "valid"


def make_cluster_result(
    labels: list[int] | np.ndarray | ClusterResult,
    *,
    cluster_names: dict[int, str] | None = None,
    repaired_mask: list[bool] | np.ndarray | None = None,
    metrics: ClusterMetrics | None = None,
) -> ClusterResult:
    if isinstance(labels, ClusterResult):
        base = labels
        return ClusterResult(
            labels=base.labels.copy(),
            cluster_names=cluster_names or base.cluster_names.copy(),
            repaired_mask=(
                np.array(repaired_mask, dtype=bool)
                if repaired_mask is not None
                else base.repaired_mask.copy()
            ),
            metrics=metrics or dataclasses.replace(base.metrics),
        )
    labels_array = np.array(labels, dtype=int)
    if repaired_mask is None:
        repaired_mask_array = np.zeros(labels_array.shape, dtype=bool)
    else:
        repaired_mask_array = np.array(repaired_mask, dtype=bool)
    return ClusterResult(
        labels=labels_array,
        cluster_names=cluster_names or {},
        repaired_mask=repaired_mask_array,
        metrics=metrics or ClusterMetrics(),
    )


def test_pipeline_reuses_vlm_cache_across_runs(tmp_path: Path, monkeypatch) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)

    call_count = {"doc": 0, "vlm": 0, "emb": 0}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, extractor, semaphore
        call_count["doc"] += 1
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_needs_vlm_fallback(
        content: str,
        file_path: Path,
        vlm_fallback_threshold=None,
        config=None,
    ) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        call_count["vlm"] += 1
        return VLMSamplingResult(
            result=ExtractionResult(
                content="this is vlm content",
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=3,
            sample_page_limit=max_pages,
        )

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del input_mode
        call_count["emb"] += 1
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Cluster_A"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.io.route.needs_vlm_fallback", fake_needs_vlm_fallback
    )
    monkeypatch.setattr(
        "dite.io.route._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    options = PipelineOptions(
        use_cache=True,
        use_embedding_cache=True,
        repair_noise=True,
        merge_same_name=True,
    )

    first = service.run(tmp_path, options)
    second = service.run(tmp_path, options)

    assert first.extraction.selected_vlm_files == 1
    assert first.extraction.vlm_cache_hits == 0
    assert first.extraction.vlm_api_page_calls == 3
    assert second.extraction.selected_vlm_files == 1
    assert second.extraction.vlm_cache_hits == 1
    assert call_count == {"doc": 2, "vlm": 1, "emb": 1}

    cache.close()


def test_extract_files_stops_before_embedding_and_clustering(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(
            content="usable extracted content" * 20,
            success=True,
            extractor="docling",
        )

    def fail_get_embeddings(*args, **kwargs) -> np.ndarray:
        raise AssertionError("embedding should not run in PDF extraction check")

    def fail_cluster_documents(*args, **kwargs):
        raise AssertionError("clustering should not run in PDF extraction check")

    def fail_generate_all_cluster_names(*args, **kwargs):
        raise AssertionError("naming should not run in PDF extraction check")

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fail_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fail_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        fail_generate_all_cluster_names,
    )

    result = service.extract_files(
        [doc],
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert result.files == [doc]
    assert result.contents == ["usable extracted content" * 20]
    assert result.embeddings.size == 0
    assert result.labels.size == 0
    assert result.cluster_names == {}


def test_extract_files_can_disable_vlm_api(tmp_path: Path, monkeypatch) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        raise AssertionError("VLM API should not run when disabled")

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.io.route.needs_vlm_fallback",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "dite.io.route._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )

    result = service.extract_files(
        [doc],
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            allow_vlm_api=False,
        ),
    )

    assert result.contents == [""]
    assert result.extraction.selected_vlm_files == 0


def test_pipeline_service_creates_fresh_registry_per_extraction_batch(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    created_registries: list[object] = []
    seen_registries: list[object] = []

    class _Registry:
        pass

    def fake_make_registry(self):
        registry = _Registry()
        created_registries.append(registry)
        return registry

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="prefetched", success=True, extractor="text")

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        config,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
    ) -> ResolvedExtraction:
        del (
            client,
            config,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
        )
        seen_registries.append(registry)
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=file_path.name,
                success=True,
                extractor="text",
            ),
            primary_effective_length=len(file_path.name),
            pdf_profile=None,
            fallback_needed=False,
            selected_source="primary",
            final_content=file_path.name,
            final_effective_length=len(file_path.name),
            vlm_content=None,
            vlm_source="none",
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    monkeypatch.setattr(
        PipelineService,
        "_make_extractor_registry",
        fake_make_registry,
    )
    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    service = PipelineService(client=object(), config=Config(), cache=None)
    options = PipelineOptions(use_cache=False, use_embedding_cache=False)

    service.extract_files([first], options)
    service.extract_files([second], options)

    assert len(created_registries) == 2
    assert created_registries[0] is not created_registries[1]
    assert seen_registries == created_registries


def test_extract_files_defers_cache_size_enforcement_until_batch_end(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    service = PipelineService(client=object(), config=Config(), cache=cache)
    save_flags: list[bool] = []
    enforce_calls = {"count": 0}

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
        del client, registry, config
        return ExtractionResult(
            content=f"{file_path.stem} content long enough",
            success=True,
            extractor="text",
        )

    def fake_save(
        self,
        file_path,
        file_hash,
        file_mtime,
        content_md,
        vlm_content=None,
        vlm_version=None,
        embedding=None,
        model_version="",
        enforce_size_limit=True,
    ) -> None:
        del (
            self,
            file_path,
            file_hash,
            file_mtime,
            content_md,
            vlm_content,
            vlm_version,
            embedding,
            model_version,
        )
        save_flags.append(enforce_size_limit)

    def fake_enforce(self) -> int:
        del self
        enforce_calls["count"] += 1
        return 0

    monkeypatch.setattr(
        "dite.io.route.extract_document", fake_extract_document
    )
    monkeypatch.setattr(FileCache, "save", fake_save)
    monkeypatch.setattr(FileCache, "enforce_size_limit", fake_enforce)

    result = service.extract_files(
        [first, second],
        PipelineOptions(use_cache=True, use_embedding_cache=False),
    )

    assert len(result.contents) == 2
    assert save_flags == [False, False]
    assert enforce_calls["count"] == 1


def test_extract_contents_preserves_input_order_when_workers_finish_out_of_order(
    tmp_path: Path, monkeypatch
) -> None:
    files = [tmp_path / "first.txt", tmp_path / "second.txt", tmp_path / "third.txt"]
    for file in files:
        file.write_text(file.stem, encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)
    delays = {"first.txt": 0.20, "second.txt": 0.05, "third.txt": 0.10}

    def fake_work_item(
        self,
        item,
        options,
        truncate_limit,
        docling_pdf_semaphore,
    ) -> object:
        del self, options, truncate_limit, docling_pdf_semaphore
        time.sleep(delays[item.file.name])
        return ExtractionWorkResult(
            index=item.index,
            content=f"content::{item.file.name}",
            file_hash=f"hash::{item.file.name}",
            report=ExtractionFileReport(
                file=item.file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(item.file.name),
                selected_source="primary",
                final_effective_length=len(item.file.name),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=f"hash::{item.file.name}",
            ),
            summary_delta=ExtractionSummaryDelta(
                primary_failures=0,
                source_fallback_needed=0,
                selected_vlm_files=0,
                vlm_api_page_calls=0,
            ),
        )

    monkeypatch.setattr(
        PipelineService,
        "_extract_content_work_item",
        fake_work_item,
    )

    contents, file_hashes, summary, reports = service._extract_contents(
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert contents == [
        "content::first.txt",
        "content::second.txt",
        "content::third.txt",
    ]
    assert file_hashes == [
        "hash::first.txt",
        "hash::second.txt",
        "hash::third.txt",
    ]
    assert [report.file.name for report in reports] == [
        "first.txt",
        "second.txt",
        "third.txt",
    ]
    assert summary.duplicate_count == 0


def test_docling_pdf_workers_limit_concurrent_docling_subprocesses(monkeypatch) -> None:
    service = PipelineService(client=object(), config=Config(), cache=None)
    service.config.processing.docling_pdf_workers = 1
    semaphore = threading.BoundedSemaphore(
        service.config.processing.docling_pdf_workers
    )
    active = {"count": 0, "max": 0}
    lock = threading.Lock()
    calls: list[dict[str, object]] = []

    class _Extractor:
        _enable_ocr = False
        _artifacts_path = None
        _pdf_timeout_sec = 0.5
        _device = "cpu"

    def fake_subprocess(
        file_path,
        *,
        enable_ocr,
        artifacts_path,
        timeout_sec,
        locale,
        device,
    ) -> ExtractionResult:
        calls.append(
            {
                "file_path": file_path,
                "enable_ocr": enable_ocr,
                "artifacts_path": artifacts_path,
                "timeout_sec": timeout_sec,
                "locale": locale,
                "device": device,
            }
        )
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return ExtractionResult(content="ok", success=True, extractor="docling")

    monkeypatch.setattr(
        "dite.flow.api.extract_docling_pdf_in_subprocess",
        fake_subprocess,
    )

    files = [Path("one.pdf"), Path("two.pdf"), Path("three.pdf")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                service._extract_docling_pdf_primary_result,
                file_path,
                _Extractor(),
                semaphore,
            )
            for file_path in files
        ]
        results = [future.result() for future in futures]

    assert all(result.success for result in results)
    assert active["max"] == 1
    assert len(calls) == 3
    assert {call["device"] for call in calls} == {"cpu"}


def test_extract_primary_result_routes_pdf_docling_to_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    service = PipelineService(client=object(), config=Config(), cache=None)
    calls: dict[str, object] = {}

    extractor = DoclingExtractor(pdf_timeout_sec=1.0)

    monkeypatch.setattr(
        "dite.flow.api.get_extractor",
        lambda *args, **kwargs: extractor,
    )

    def fake_subprocess(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        calls["file_path"] = file_path
        calls["extractor"] = extractor
        calls["semaphore"] = semaphore
        return ExtractionResult(content="pdf-result", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_subprocess,
    )

    result = service._extract_primary_result(
        pdf_path,
        registry=object(),
        docling_pdf_semaphore=threading.BoundedSemaphore(1),
    )

    assert result.content == "pdf-result"
    assert calls["file_path"] == pdf_path
    assert calls["extractor"] is extractor


def test_extract_primary_result_routes_non_pdf_through_router(
    tmp_path: Path, monkeypatch
) -> None:
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("payload", encoding="utf-8")
    service = PipelineService(client=object(), config=Config(), cache=None)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "dite.flow.api.get_extractor",
        lambda *args, **kwargs: object(),
    )

    def fake_extract_document(file_path: Path, client, *, config, registry=None):
        calls["file_path"] = file_path
        calls["client"] = client
        calls["config"] = config
        calls["registry"] = registry
        return ExtractionResult(content="text-result", success=True, extractor="text")

    monkeypatch.setattr(
        "dite.io.route.extract_document",
        fake_extract_document,
    )

    result = service._extract_primary_result(
        txt_path,
        registry="worker-registry",
        docling_pdf_semaphore=threading.BoundedSemaphore(1),
    )

    assert result.content == "text-result"
    assert calls == {
        "file_path": txt_path,
        "client": service.client,
        "config": service.config,
        "registry": "worker-registry",
    }


def test_pipeline_reuses_vlm_cache_by_hash_across_paths(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    cached_source = tmp_path / "cached-source.pdf"
    alias = docs / "alias.pdf"
    cached_source.write_text("same-pdf-bytes", encoding="utf-8")
    alias.write_text("same-pdf-bytes", encoding="utf-8")
    file_hash = compute_file_hash(alias)

    cache = FileCache(db_path=tmp_path / "cache.db")
    cache.update_vlm_content(
        file_path=cached_source,
        file_hash=file_hash,
        vlm_content="cached vlm content from another path",
        vlm_version=2,
    )
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)

    call_count = {"doc": 0, "vlm": 0, "emb": 0}
    captured: dict[str, list[str]] = {}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, extractor, semaphore
        call_count["doc"] += 1
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        call_count["vlm"] += 1
        raise AssertionError("VLM should not run when VLM hash cache exists")

    def fake_needs_vlm_fallback(
        content: str,
        file_path: Path,
        vlm_fallback_threshold=None,
        config=None,
    ) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del input_mode
        call_count["emb"] += 1
        captured["texts"] = texts
        captured["file_names"] = file_names
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Cluster_A"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.io.route._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )
    monkeypatch.setattr(
        "dite.io.route.needs_vlm_fallback", fake_needs_vlm_fallback
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        docs,
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=True,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert result.extraction.vlm_cache_hits == 1
    assert result.extraction.selected_vlm_files == 1
    assert call_count == {"doc": 1, "vlm": 0, "emb": 1}
    assert result.contents == ["cached vlm content from another path"]
    assert captured["texts"] == ["cached vlm content from another path"]
    assert captured["file_names"] == ["alias.pdf"]
    cache.close()


def test_pipeline_prefers_vlm_when_docling_content_is_glyph_noise(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    config.processing.text_truncate_limit = 5000
    service = PipelineService(client=object(), config=config, cache=cache)

    captured: dict[str, list[str]] = {}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(
            content=("/G25/G26/G27/G28 " * 40) + ("A" * 147),
            success=True,
            extractor="docling",
        )

    def fake_needs_vlm_fallback(
        content: str,
        file_path: Path,
        vlm_fallback_threshold=None,
        config=None,
    ) -> bool:
        return True

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        return VLMSamplingResult(
            result=ExtractionResult(
                content="这是 VLM 提取出的正常文本。" * 20,
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=4,
            sample_page_limit=max_pages,
        )

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del input_mode
        captured["texts"] = texts
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Cluster_A"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.io.route.needs_vlm_fallback", fake_needs_vlm_fallback
    )
    monkeypatch.setattr(
        "dite.io.route._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert len(result.files) == 1
    assert captured["texts"][0].startswith("这是 VLM 提取出的正常文本")
    cache.close()


def test_extract_files_does_not_cache_failed_vlm_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "failed.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    config.processing.text_truncate_limit = 100
    service = PipelineService(client=object(), config=config, cache=cache)
    observed = {"update_calls": 0}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        del file_path, client, config
        return VLMSamplingResult(
            result=ExtractionResult(
                content="",
                success=False,
                extractor="vlm_fallback",
                error="VLM fallback returned no usable content",
            ),
            api_page_calls=max_pages,
            sample_page_limit=max_pages,
        )

    def fake_update_vlm_content(*args, **kwargs) -> None:
        del args, kwargs
        observed["update_calls"] += 1

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.io.route._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )
    monkeypatch.setattr(cache, "update_vlm_content", fake_update_vlm_content)

    result = service.extract_files(
        [doc],
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=False,
            repair_noise=False,
            merge_same_name=False,
        ),
    )

    assert observed["update_calls"] == 0
    assert result.contents == [""]
    assert result.extraction.primary_failures == 1
    assert result.extraction.source_fallback_needed == 1
    assert result.extraction.selected_vlm_files == 0
    assert result.extraction.vlm_api_page_calls == PDF_VLM_SAMPLE_PAGE_LIMIT
    assert result.file_reports[0].selected_source == "primary"
    assert result.file_reports[0].vlm_api_page_calls == PDF_VLM_SAMPLE_PAGE_LIMIT
    assert result.file_reports[0].sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT
    cache.close()


def test_extract_files_uses_explicit_cache_write_intent_to_skip_vlm_cache_update(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "explicit-intent-skip.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)
    observed = {"update_calls": 0}

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="primary", success=True, extractor="docling")

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            file_path,
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content="primary",
                success=True,
                extractor="docling",
            ),
            primary_effective_length=7,
            pdf_profile=PDFProfile(
                kind="weak_text",
                effective_length=7,
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=True,
                success=True,
                reason="effective_text_below_threshold",
            ),
            fallback_needed=True,
            selected_source="primary",
            final_content="primary",
            final_effective_length=7,
            vlm_content="fresh vlm content",
            vlm_source="api",
            vlm_api_success=True,
            vlm_api_page_calls=2,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            cache_write_intent=PDFCacheWriteIntent(
                should_write=False,
                content=None,
            ),
        )

    def fake_update_vlm_content(*args, **kwargs) -> None:
        del args, kwargs
        observed["update_calls"] += 1

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )
    monkeypatch.setattr(cache, "update_vlm_content", fake_update_vlm_content)

    service.extract_files(
        [doc],
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=False,
            repair_noise=False,
            merge_same_name=False,
        ),
    )

    assert observed["update_calls"] == 0
    cache.close()


def test_extract_files_uses_explicit_cache_write_intent_to_update_vlm_cache(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "explicit-intent-write.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)
    observed: list[str] = []

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="primary", success=True, extractor="docling")

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            file_path,
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content="primary",
                success=True,
                extractor="docling",
            ),
            primary_effective_length=7,
            pdf_profile=PDFProfile(
                kind="weak_text",
                effective_length=7,
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=True,
                success=True,
                reason="effective_text_below_threshold",
            ),
            fallback_needed=True,
            selected_source="primary",
            final_content="primary",
            final_effective_length=7,
            vlm_content=None,
            vlm_source="none",
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            cache_write_intent=PDFCacheWriteIntent(
                should_write=True,
                content="explicit fresh vlm content",
            ),
        )

    def fake_update_vlm_content(
        file_path, file_hash, content, vlm_version, *, enforce_size_limit
    ) -> None:
        del file_path, file_hash, vlm_version, enforce_size_limit
        observed.append(content)

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )
    monkeypatch.setattr(cache, "update_vlm_content", fake_update_vlm_content)

    service.extract_files(
        [doc],
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=False,
            repair_noise=False,
            merge_same_name=False,
        ),
    )

    assert observed == ["explicit fresh vlm content"]
    cache.close()


def test_pipeline_real_scan_extract_and_cache_hits(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same content", encoding="utf-8")
    b.write_text("same content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del input_mode
        return np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0, 0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Cluster_A"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    options = PipelineOptions(
        use_cache=True,
        use_embedding_cache=True,
        repair_noise=True,
        merge_same_name=True,
    )

    first = service.run(tmp_path, options)
    second = service.run(tmp_path, options)

    assert len(first.files) == 2
    assert first.extraction.doc_cache_hits == 1
    assert first.extraction.duplicate_count == 1
    assert second.extraction.doc_cache_hits == 2
    assert second.extraction.duplicate_count == 1
    assert len(second.extraction.duplicate_groups) == 1
    assert first.extraction.selected_vlm_files == 0
    assert second.extraction.selected_vlm_files == 0

    cache.close()


def test_pipeline_deduplicates_same_hash_before_vlm_embedding_and_clustering(
    tmp_path: Path, monkeypatch
) -> None:
    docs = [tmp_path / name for name in ("paper.pdf", "paper (1).pdf", "paper (2).pdf")]
    for doc in docs:
        doc.write_bytes(b"%PDF-1.7 duplicate paper")

    service = PipelineService(client=object(), config=Config(), cache=None)
    calls = {"primary": 0, "resolve": 0, "embedding": 0, "cluster": 0}

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, registry, docling_pdf_semaphore
        calls["primary"] += 1
        return ExtractionResult(
            content=f"primary content from {file_path.name}",
            success=True,
            extractor="docling",
        )

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            registry,
            config,
        )
        calls["resolve"] += 1
        assert primary_result is not None
        return ResolvedExtraction(
            final_content=f"canonical summary for {file_path.name}",
            selected_source="vlm_api",
            primary_result=primary_result,
            primary_effective_length=len(primary_result.content),
            final_effective_length=120,
            pdf_profile=PDFProfile(
                kind="scanned_image",
                effective_length=0,
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=True,
                success=True,
                reason="test",
            ),
            fallback_needed=True,
            vlm_api_page_calls=3,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            cache_write_intent=PDFCacheWriteIntent(
                should_write=True,
                content="canonical vlm",
            ),
            vlm_content="canonical vlm",
            vlm_source="api",
            vlm_api_success=True,
        )

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del client, config, embedding_model, input_mode
        calls["embedding"] += 1
        assert texts == ["canonical summary for paper (1).pdf"]
        assert file_names == ["paper (1).pdf"]
        return np.array([[0.6, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        calls["cluster"] += 1
        assert embeddings.shape == (1, 2)
        assert item_names == ["paper (1).pdf"]
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        del client, embeddings, merge_same_name, llm_model, config
        assert result.labels.tolist() == [0]
        assert contents == ["canonical summary for paper (1).pdf"]
        assert files == [docs[1]]
        return make_cluster_result(result, cluster_names={0: "Duplicate Papers"})

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert calls == {"primary": 1, "resolve": 1, "embedding": 1, "cluster": 1}
    assert result.files == sorted(docs)
    assert result.contents == ["canonical summary for paper (1).pdf"] * 3
    np.testing.assert_allclose(
        result.embeddings,
        np.array([[0.6, 0.8], [0.6, 0.8], [0.6, 0.8]], dtype=np.float32),
    )
    assert result.labels.tolist() == [0, 0, 0]
    assert result.extraction.duplicate_count == 2
    assert result.extraction.selected_vlm_files == 1
    assert result.extraction.vlm_api_page_calls == 3
    assert result.extraction.doc_cache_hits == 2
    assert [report.file for report in result.file_reports] == sorted(docs)
    assert len({report.file_hash for report in result.file_reports}) == 1


def test_pipeline_reuses_hash_cached_embedding_for_duplicate_canonical(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "a.txt"
    cached_alias = tmp_path / "b.txt"
    content = "same duplicate content long enough for extraction"
    canonical.write_text(content, encoding="utf-8")
    cached_alias.write_text(content, encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    file_hash = compute_file_hash(cached_alias)
    cached_embedding = np.array([3.0, 4.0], dtype=np.float32)
    cache.save(
        file_path=cached_alias,
        file_hash=file_hash,
        file_mtime=cached_alias.stat().st_mtime,
        content_md=content,
        embedding=cached_embedding,
        model_version=get_embedding_cache_version(config.models.embedding),
    )
    service = PipelineService(client=object(), config=config, cache=cache)

    def fail_get_embeddings(*args, **kwargs) -> np.ndarray:
        raise AssertionError("embedding API should not run for hash-cached duplicate")

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        del repair_noise, config, clustering, allow_single_cluster
        np.testing.assert_allclose(
            embeddings,
            np.array([[0.6, 0.8]], dtype=np.float32),
        )
        assert item_names == ["a.txt"]
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        del client, embeddings, merge_same_name, llm_model, config
        assert result.labels.tolist() == [0]
        assert contents == [content]
        assert files == [canonical]
        return make_cluster_result(result, cluster_names={0: "Cached Duplicate"})

    monkeypatch.setattr("dite.flow.api.get_embeddings", fail_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=True,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert result.files == [canonical, cached_alias]
    assert result.contents == [content, content]
    np.testing.assert_allclose(
        result.embeddings,
        np.array([[0.6, 0.8], [0.6, 0.8]], dtype=np.float32),
    )
    assert result.labels.tolist() == [0, 0]
    assert result.extraction.doc_cache_hits == 2
    assert result.extraction.duplicate_count == 1
    cache.close()


def test_pipeline_expands_repaired_noise_count_to_duplicate_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    duplicate_a = tmp_path / "a.txt"
    duplicate_b = tmp_path / "b.txt"
    noise = tmp_path / "c.txt"
    duplicate_content = "same duplicate paper content long enough"
    duplicate_a.write_text(duplicate_content, encoding="utf-8")
    duplicate_b.write_text(duplicate_content, encoding="utf-8")
    noise.write_text("unrelated content long enough", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del client, config, embedding_model, input_mode
        assert texts == [duplicate_content, "unrelated content long enough"]
        assert file_names == ["a.txt", "c.txt"]
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        del embeddings, repair_noise, config, clustering, allow_single_cluster
        assert item_names == ["a.txt", "c.txt"]
        return make_cluster_result(
            [0, -1],
            repaired_mask=[True, False],
            metrics=ClusterMetrics(
                initial_clusters=1,
                initial_noise=1,
                noise_repaired=1,
            ),
        )

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        del client, embeddings, merge_same_name, llm_model, config
        assert result.labels.tolist() == [0, -1]
        assert contents == [duplicate_content, "unrelated content long enough"]
        assert files == [duplicate_a, noise]
        return make_cluster_result(result, cluster_names={0: "Repaired Duplicate"})

    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(use_cache=False, use_embedding_cache=False, repair_noise=True),
    )

    assert result.files == [duplicate_a, duplicate_b, noise]
    assert result.labels.tolist() == [0, 0, -1]
    assert result.noise_repaired == 2
    assert result.extraction.duplicate_count == 1


def test_pipeline_accumulates_small_and_name_cluster_merge_counts(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    third = tmp_path / "c.txt"
    first.write_text("alpha content long enough", encoding="utf-8")
    second.write_text("beta content long enough", encoding="utf-8")
    third.write_text("gamma content long enough", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del client, texts, config, file_names, embedding_model, input_mode
        return np.array(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
            dtype=np.float32,
        )

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        del (
            embeddings,
            repair_noise,
            config,
            clustering,
            allow_single_cluster,
            item_names,
        )
        return make_cluster_result(
            [0, 0, 1],
            repaired_mask=[False, False, False],
            metrics=ClusterMetrics(
                initial_clusters=2,
                initial_noise=0,
                noise_repaired=0,
                small_clusters_merged=1,
            ),
        )

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        del client, contents, files, embeddings, merge_same_name, llm_model, config
        return make_cluster_result(
            [0, 0, 0],
            cluster_names={0: "Merged Cluster"},
            repaired_mask=result.repaired_mask,
            metrics=ClusterMetrics(
                initial_clusters=result.metrics.initial_clusters,
                initial_noise=result.metrics.initial_noise,
                noise_repaired=result.metrics.noise_repaired,
                small_clusters_merged=result.metrics.small_clusters_merged,
                name_clusters_merged=1,
            ),
        )

    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(use_cache=False, use_embedding_cache=False, repair_noise=True),
    )

    assert result.cluster_metrics.initial_clusters == 2
    assert result.cluster_metrics.small_clusters_merged == 1
    assert result.cluster_metrics.name_clusters_merged == 1
    assert result.clusters_merged == 2
    assert result.cluster_names == {0: "Merged Cluster"}
    assert result.labels.tolist() == [0, 0, 0]


def test_extract_files_locks_down_failure_corpus_classification_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    files: list[Path] = []
    names = [
        *(f"failed-{index}.pdf" for index in range(1, 8)),
        "weak-1.pdf",
        "weak-2.pdf",
    ]
    for name in names:
        path = tmp_path / name
        path.write_bytes(f"%PDF-1.7 {name}".encode())
        files.append(path)

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        is_failed = file_path.name.startswith("failed-")
        primary_content = "" if is_failed else ("/G25/G26/G27/G28 " * 20 + "题目")
        primary_success = not is_failed
        primary_error = "parser failed" if is_failed else None
        reason = "parser_timeout_or_broken" if is_failed else "glyph_noise_dominates"
        profile_kind = "parser_timeout_or_broken" if is_failed else "weak_text"
        page_calls = 10 if is_failed else 6
        final_content = f"{file_path.stem} final content " * 20

        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=primary_content,
                success=primary_success,
                extractor="docling",
                error=primary_error,
            ),
            primary_effective_length=0 if is_failed else 2,
            pdf_profile=PDFProfile(
                kind=profile_kind,
                effective_length=0 if is_failed else 2,
                glyph_noise_tokens=0 if is_failed else 20,
                glyph_noise_ratio=0.0 if is_failed else 0.9,
                needs_vlm_fallback=True,
                success=primary_success,
                reason=reason,
            ),
            fallback_needed=True,
            selected_source="vlm_api",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=final_content,
            vlm_source="api",
            vlm_api_success=True,
            vlm_api_page_calls=page_calls,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        )

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = service.extract_files(
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert len(result.file_reports) == 9
    assert result.extraction.primary_failures == 7
    assert result.extraction.source_fallback_needed == 9
    assert result.extraction.selected_vlm_files == 9
    assert result.extraction.vlm_api_page_calls == 82
    assert result.extraction.duplicate_count == 0
    assert {report.source_profile for report in result.file_reports} == {
        "parser_timeout_or_broken",
        "weak_text",
    }
    assert {report.source_reason for report in result.file_reports} == {
        "parser_timeout_or_broken",
        "glyph_noise_dominates",
    }
    assert all(report.fallback_needed is True for report in result.file_reports)
    assert all(report.selected_source == "vlm_api" for report in result.file_reports)
    assert all(
        report.sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT
        for report in result.file_reports
    )
    assert all(report.final_effective_length >= 100 for report in result.file_reports)


def test_extract_files_detects_real_duplicate_group_without_cache(
    tmp_path: Path, monkeypatch
) -> None:
    fixture_dir = TEST_CORPUS_DIR
    duplicate_names = [
        "2506.12116v3.pdf",
        "2506.12116v3 (1).pdf",
        "2506.12116v3 (2).pdf",
    ]
    files: list[Path] = []
    for name in duplicate_names:
        copied = tmp_path / name
        shutil.copy2(fixture_dir / name, copied)
        files.append(copied)

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        final_content = f"{file_path.name} usable content " * 20
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=final_content,
                success=True,
                extractor="docling",
            ),
            primary_effective_length=len(final_content),
            pdf_profile=PDFProfile(
                kind="native_text",
                effective_length=len(final_content),
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=False,
                success=True,
                reason="usable_text_layer",
            ),
            fallback_needed=False,
            selected_source="primary",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=None,
            vlm_source=None,
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = service.extract_files(
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert result.extraction.duplicate_count == 2
    assert len(result.extraction.duplicate_groups) == 1
    duplicate_group = next(iter(result.extraction.duplicate_groups.values()))
    assert {Path(path).name for path in duplicate_group} == set(duplicate_names)
    assert len({report.file_hash for report in result.file_reports}) == 1


def test_pipeline_recomputes_embedding_when_model_changes(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("same content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    config.models.embedding = "embed-v2"
    cache.save(
        file_path=doc,
        file_hash="hash-1",
        file_mtime=doc.stat().st_mtime,
        content_md="same content",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        model_version="embed-v1",
    )
    service = PipelineService(client=object(), config=config, cache=cache)

    call_count = {"emb": 0}

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del input_mode
        call_count["emb"] += 1
        assert embedding_model == "embed-v2"
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Cluster_A"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=True,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert len(result.files) == 1
    assert call_count["emb"] == 1
    np.testing.assert_allclose(
        result.embeddings,
        np.array([[0.7474093, 0.6643638]], dtype=np.float32),
    )
    cache.close()


def test_pipeline_recomputes_embedding_when_input_version_changes(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("same content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    config.models.embedding = "embed-v1"
    file_hash = compute_file_hash(doc)
    cache.save(
        file_path=doc,
        file_hash=file_hash,
        file_mtime=doc.stat().st_mtime,
        content_md="same content",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        model_version="embed-v1",
    )
    service = PipelineService(client=object(), config=config, cache=cache)

    captured: dict[str, object] = {}

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        captured["texts"] = texts
        captured["file_names"] = file_names
        captured["embedding_model"] = embedding_model
        captured["input_mode"] = input_mode
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Cluster_A"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=True,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert captured == {
        "texts": ["same content"],
        "file_names": ["sample.txt"],
        "embedding_model": "embed-v1",
        "input_mode": "with_filename",
    }
    np.testing.assert_allclose(
        result.embeddings,
        np.array([[0.7474093, 0.6643638]], dtype=np.float32),
    )
    entry = cache.get_by_path(doc)
    assert entry is not None
    assert entry.model_version == get_embedding_cache_version("embed-v1")
    cache.close()


def test_pipeline_uses_smart_content_excerpt_for_embedding(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "book.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    config = Config()
    config.processing.text_truncate_limit = 160
    service = PipelineService(client=object(), config=config, cache=None)
    long_content = ("COVER " * 80) + ("RUST OWNERSHIP " * 20) + ("INDEX " * 80)
    captured: dict[str, object] = {}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(content=long_content, success=True, extractor="docling")

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del input_mode
        captured["texts"] = texts
        captured["file_names"] = file_names
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Cluster_A"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.io.route.needs_vlm_fallback", lambda *args, **kwargs: False
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    excerpt = captured["texts"][0]
    assert isinstance(excerpt, str)
    assert len(excerpt) <= config.processing.text_truncate_limit
    assert "COVER" in excerpt
    assert "RUST" in excerpt
    assert "OWNERSHIP" in excerpt
    assert "INDEX" in excerpt
    assert captured["file_names"] == ["book.pdf"]


def test_extract_files_preserves_final_length_separately_from_truncated_contents(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "long.pdf"
    doc.write_bytes(b"%PDF-1.7 long")

    config = Config()
    config.processing.text_truncate_limit = 80
    service = PipelineService(client=object(), config=config, cache=None)
    final_content = ("COVER " * 40) + ("RUST OWNERSHIP " * 20) + ("INDEX " * 40)

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            file_path,
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=final_content,
                success=True,
                extractor="docling",
            ),
            primary_effective_length=len(final_content),
            pdf_profile=PDFProfile(
                kind="native_text",
                effective_length=len(final_content),
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=False,
                success=True,
                reason="usable_text_layer",
            ),
            fallback_needed=False,
            selected_source="primary",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=None,
            vlm_source=None,
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = service.extract_files(
        [doc],
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert len(result.contents) == 1
    assert len(result.contents[0]) <= config.processing.text_truncate_limit
    assert result.file_reports[0].final_effective_length == len(final_content)
    assert result.file_reports[0].excerpt_was_truncated is True


def test_knn_repair_keeps_far_noise_with_threshold() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],  # near core cluster
            [-1.0, 0.0],  # far from core cluster
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, -1, -1], dtype=int)

    repaired_labels, repaired_count = repair_noise_with_knn(
        embeddings,
        labels,
        k=1,
        distance_threshold=0.2,
    )

    assert repaired_count == 1
    assert repaired_labels[3] == 0
    assert repaired_labels[4] == -1


def test_knn_repair_uses_dynamic_threshold() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],  # near core cluster
            [-1.0, 0.0],  # far from core cluster
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, -1, -1], dtype=int)

    repaired_labels, repaired_count = repair_noise_with_knn(
        embeddings,
        labels,
        k=1,
        distance_threshold=None,
    )

    assert repaired_count == 1
    assert repaired_labels[3] == 0
    assert repaired_labels[4] == -1


def test_repair_noise_with_knn_returns_unchanged_without_noise() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1], dtype=int)

    repaired_labels, repaired_count = repair_noise_with_knn(
        embeddings,
        labels,
        k=3,
        distance_threshold=0.2,
    )

    assert repaired_count == 0
    assert np.array_equal(repaired_labels, labels)


def test_repair_noise_with_knn_returns_unchanged_without_core_cluster() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ],
        dtype=np.float32,
    )
    labels = np.array([-1, -1, -1], dtype=int)

    repaired_labels, repaired_count = repair_noise_with_knn(
        embeddings,
        labels,
        k=3,
        distance_threshold=0.2,
    )

    assert repaired_count == 0
    assert np.array_equal(repaired_labels, labels)


def test_cluster_documents_repairs_all_noise_with_similarity_fallback(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer

    class _FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            del embeddings
            return np.array([-1, -1, -1, -1], dtype=int)

    monkeypatch.setattr(clusterer.hdbscan, "HDBSCAN", _FakeHDBSCAN)

    config = Config()
    config.clustering.min_cluster_size = 2
    result = cluster_documents(
        np.array(
            [
                [10.0, 0.0],
                [9.0, 0.1],
                [0.0, 10.0],
                [0.1, 9.0],
            ],
            dtype=np.float32,
        ),
        config=config,
        repair_noise=True,
        knn_distance_threshold=0.05,
    )

    assert result.noise_repaired == 4
    assert result.labels.tolist() == [0, 0, 1, 1]


def test_cluster_documents_keeps_all_noise_when_repair_disabled(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer

    class _FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            del embeddings
            return np.array([-1, -1], dtype=int)

    monkeypatch.setattr(clusterer.hdbscan, "HDBSCAN", _FakeHDBSCAN)

    config = Config()
    result = cluster_documents(
        np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32),
        config=config,
        repair_noise=False,
    )

    assert result.noise_repaired == 0
    assert result.labels.tolist() == [-1, -1]


def test_cluster_documents_uses_clustering_config_without_knn_when_disabled(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import cluster_documents

    captured: dict[str, object] = {}

    class _FakeHDBSCAN:
        def __init__(
            self,
            *,
            min_cluster_size,
            min_samples,
            cluster_selection_epsilon,
            metric,
            cluster_selection_method,
            allow_single_cluster,
        ) -> None:
            captured["init"] = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "metric": metric,
                "cluster_selection_method": cluster_selection_method,
                "allow_single_cluster": allow_single_cluster,
            }

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            captured["fit_predict_shape"] = embeddings.shape
            return np.array([0, -1], dtype=int)

    def fail_repair(*args, **kwargs):
        raise AssertionError("repair_noise_with_knn should not be called")

    monkeypatch.setattr(clusterer.hdbscan, "HDBSCAN", _FakeHDBSCAN)
    monkeypatch.setattr(clusterer, "repair_noise_with_knn", fail_repair)

    config = Config()
    config.clustering.min_cluster_size = 2
    config.clustering.min_samples = 2
    config.clustering.cluster_selection_epsilon = 0.5
    config.clustering.cluster_selection_method = "leaf"

    result = cluster_documents(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        config=config,
        repair_noise=False,
    )

    assert result.noise_repaired == 0
    assert np.array_equal(result.labels, np.array([0, -1], dtype=int))
    assert captured["init"] == {
        "min_cluster_size": 2,
        "min_samples": 2,
        "cluster_selection_epsilon": 0.5,
        "metric": "euclidean",
        "cluster_selection_method": "leaf",
        "allow_single_cluster": False,
    }
    assert captured["fit_predict_shape"] == (2, 2)


def test_cluster_documents_passes_knn_overrides_to_repair(monkeypatch) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import cluster_documents

    captured: dict[str, object] = {}

    class _FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            del embeddings
            return np.array([0, 0, -1], dtype=int)

    def fake_repair(
        embeddings: np.ndarray,
        labels: np.ndarray,
        k: int = 3,
        distance_threshold: float | None = None,
        item_names: list[str] | None = None,
    ) -> tuple[np.ndarray, int]:
        captured["shape"] = embeddings.shape
        captured["labels"] = labels.copy()
        captured["k"] = k
        captured["distance_threshold"] = distance_threshold
        captured["item_names"] = item_names
        return np.array([0, 0, 0], dtype=int), 1

    monkeypatch.setattr(clusterer.hdbscan, "HDBSCAN", _FakeHDBSCAN)
    monkeypatch.setattr(clusterer, "repair_noise_with_knn", fake_repair)

    config = Config()
    config.clustering.knn_k = 99
    config.clustering.knn_distance_threshold = 0.99
    config.clustering.small_cluster_merge_enabled = False

    result = cluster_documents(
        np.array(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
            dtype=np.float32,
        ),
        config=config,
        repair_noise=True,
        knn_k=7,
        knn_distance_threshold=0.3,
        item_names=["a.txt", "a-2.txt", "b.txt"],
    )

    assert result.noise_repaired == 1
    assert np.array_equal(result.labels, np.array([0, 0, 0], dtype=int))
    assert captured["shape"] == (3, 2)
    assert np.array_equal(captured["labels"], np.array([0, 0, -1], dtype=int))
    assert captured["k"] == 7
    assert captured["distance_threshold"] == 0.3
    assert captured["item_names"] == ["a.txt", "a-2.txt", "b.txt"]


def test_cluster_documents_normalizes_before_hdbscan(monkeypatch) -> None:
    from dite.cluster import api as clusterer

    captured: dict[str, np.ndarray] = {}

    class _FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            captured["embeddings"] = embeddings.copy()
            return np.array([0, 0], dtype=int)

    monkeypatch.setattr(clusterer.hdbscan, "HDBSCAN", _FakeHDBSCAN)

    config = Config()
    config.clustering.min_cluster_size = 2
    result = cluster_documents(
        np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        config=config,
        repair_noise=False,
    )

    assert result.noise_repaired == 0
    assert result.labels.tolist() == [0, 0]
    np.testing.assert_allclose(
        captured["embeddings"],
        np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
    )


def test_cluster_documents_merges_small_cluster_by_similarity(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import ClusterMetrics, cluster_documents

    class _FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            del embeddings
            return np.array([0, 0, 1], dtype=int)

    monkeypatch.setattr(clusterer.hdbscan, "HDBSCAN", _FakeHDBSCAN)

    config = Config()
    config.clustering.min_cluster_size = 2
    config.clustering.small_cluster_merge_enabled = True
    config.clustering.small_cluster_merge_max_size = 2
    config.clustering.small_cluster_merge_cosine_threshold = 0.9

    result = cluster_documents(
        np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, 0.02],
            ],
            dtype=np.float32,
        ),
        config=config,
        repair_noise=False,
    )

    assert np.array_equal(result.labels, np.array([0, 0, 0], dtype=int))
    assert np.array_equal(result.repaired_mask, np.array([False, False, False]))
    assert isinstance(result.metrics, ClusterMetrics)
    assert result.metrics.initial_clusters == 2
    assert result.metrics.initial_noise == 0
    assert result.metrics.noise_repaired == 0
    assert result.metrics.small_clusters_merged == 1
    assert result.metrics.small_cluster_merge_candidates == 2
    assert result.metrics.small_cluster_merge_skipped == 0
    assert result.metrics.small_cluster_merge_max_similarity is not None
    assert len(result.metrics.small_cluster_merge_events) == 1
    event = result.metrics.small_cluster_merge_events[0]
    assert event.source_label == 1
    assert event.target_label == 0
    assert event.source_size == 1
    assert event.target_size_before == 2


def test_cluster_documents_keeps_small_cluster_when_similarity_low(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import cluster_documents

    class _FakeHDBSCAN:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            del embeddings
            return np.array([0, 0, 1], dtype=int)

    monkeypatch.setattr(clusterer.hdbscan, "HDBSCAN", _FakeHDBSCAN)

    config = Config()
    config.clustering.min_cluster_size = 2
    config.clustering.small_cluster_merge_enabled = True
    config.clustering.small_cluster_merge_max_size = 2
    config.clustering.small_cluster_merge_cosine_threshold = 0.99

    result = cluster_documents(
        np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        config=config,
        repair_noise=False,
    )

    assert np.array_equal(result.labels, np.array([0, 0, 1], dtype=int))
    assert np.array_equal(result.repaired_mask, np.array([False, False, False]))
    assert result.metrics.small_clusters_merged == 0
    assert result.metrics.small_cluster_merge_candidates == 2
    assert result.metrics.small_cluster_merge_skipped == 2
    assert len(result.metrics.small_cluster_merge_events) == 0
    assert len(result.metrics.small_cluster_skip_events) == 2
    skip_event = result.metrics.small_cluster_skip_events[0]
    assert skip_event.source_label == 1
    assert skip_event.best_target_label == 0
    assert skip_event.reason == "below_similarity_threshold"


def test_merge_small_clusters_by_similarity_skips_large_source_clusters() -> None:
    from dite.cluster.api import merge_small_clusters_by_similarity

    labels, merged_count, _merge_events, _skip_events, _max_similarity = (
        merge_small_clusters_by_similarity(
        np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.array([0, 0, 0, 1], dtype=int),
        max_size=1,
        cosine_threshold=0.95,
    ))

    assert merged_count == 0
    assert np.array_equal(labels, np.array([0, 0, 0, 1], dtype=int))


def test_merge_small_clusters_by_similarity_prefers_larger_target_on_tie() -> None:
    from dite.cluster.api import merge_small_clusters_by_similarity

    labels, merged_count, _merge_events, _skip_events, _max_similarity = (
        merge_small_clusters_by_similarity(
        np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.8, -0.2],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array([0, 1, 1, 2], dtype=int),
        max_size=1,
        cosine_threshold=0.95,
    ))

    assert merged_count == 2
    assert labels[0] == 1


def test_cluster_documents_merges_small_cluster_with_real_hdbscan_defaults() -> None:
    from dite.cluster.api import cluster_documents

    config = Config()
    config.clustering.min_cluster_size = 2
    config.clustering.min_samples = 1
    config.clustering.cluster_selection_epsilon = 0.0
    config.clustering.cluster_selection_method = "eom"
    config.clustering.small_cluster_merge_enabled = True
    config.clustering.small_cluster_merge_max_size = 2
    config.clustering.small_cluster_merge_cosine_threshold = 0.9

    result = cluster_documents(
        np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        ),
        config=config,
        repair_noise=False,
    )

    assert np.array_equal(
        result.repaired_mask, np.array([False, False, False, False, False])
    )
    assert result.metrics.initial_clusters >= 2
    assert result.metrics.small_clusters_merged >= 0


# Identical content only counts as a near duplicate when there is enough of it
# (see ``NEAR_DUPLICATE_MIN_CHARS``); a shared one-line title is not enough.
_DUPLICATE_BODY = (
    "Vector spaces, linear maps and their matrices.\n"
    "A basis lets every vector be written uniquely as a linear combination.\n"
    "The rank-nullity theorem relates the dimension of the kernel and image.\n"
    "Eigenvalues are the roots of the characteristic polynomial det(A - tI).\n"
)


def test_pipeline_strong_semantic_components_do_not_prebind_canonical_labels(
    tmp_path: Path, monkeypatch
) -> None:
    doc_a = tmp_path / "linear-algebra-a.txt"
    doc_b = tmp_path / "linear-algebra-b.txt"
    doc_c = tmp_path / "minecraft-guide.txt"
    for path in (doc_a, doc_b, doc_c):
        path.write_text("payload", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_contents(files, options):
        del options
        contents = [
            "Linear Algebra Notes\n" + _DUPLICATE_BODY,
            "Linear Algebra Notes\n" + _DUPLICATE_BODY,
            "Minecraft Guide",
        ]
        file_hashes = [f"hash-{index}" for index, _ in enumerate(files)]
        reports = [
            ExtractionFileReport(
                file=file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(content),
                selected_source="primary",
                final_effective_length=len(content),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=file_hashes[index],
            )
            for index, (file, content) in enumerate(zip(files, contents, strict=True))
        ]
        return contents, file_hashes, ExtractionSummary(), reports

    monkeypatch.setattr(service, "_extract_contents", fake_extract_contents)
    monkeypatch.setattr(
        service,
        "_vectorize",
        lambda files, file_hashes, contents, options: np.array(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.cluster_documents",
        lambda embeddings, **kwargs: make_cluster_result(
            [0, 1, 2],
            metrics=ClusterMetrics(initial_clusters=3),
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        lambda client, result, contents, files, **kwargs: make_cluster_result(
            result,
            cluster_names={0: "Linear Algebra", 2: "Minecraft"},
        ),
    )

    result = service.run(
        tmp_path,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert len(result.candidate_components) == 1
    assert result.candidate_components[0].member_file_ids == [
        str(doc_a),
        str(doc_b),
    ]
    assert result.candidate_components[0].component_type == "near_duplicate_group"
    assert result.labels.tolist() == [0, 0, 2]
    assert len(result.document_features) == 3
    assert result.document_features[0].path == doc_a
    assert result.document_features[1].path == doc_b
    assert result.document_features[2].path == doc_c


def test_pipeline_near_duplicate_components_can_prebind_canonical_labels(
    tmp_path: Path, monkeypatch
) -> None:
    doc_a = tmp_path / "dup-a.txt"
    doc_b = tmp_path / "dup-b.txt"
    doc_c = tmp_path / "other.txt"
    for path in (doc_a, doc_b, doc_c):
        path.write_text("payload", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_contents(files, options):
        del options
        contents = [
            "Identical content block\n" + _DUPLICATE_BODY,
            "Identical content block\n" + _DUPLICATE_BODY,
            "Different topic",
        ]
        file_hashes = [f"hash-{index}" for index, _ in enumerate(files)]
        reports = [
            ExtractionFileReport(
                file=file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(content),
                selected_source="primary",
                final_effective_length=len(content),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=file_hashes[index],
            )
            for index, (file, content) in enumerate(zip(files, contents, strict=True))
        ]
        return contents, file_hashes, ExtractionSummary(), reports

    monkeypatch.setattr(service, "_extract_contents", fake_extract_contents)
    monkeypatch.setattr(
        service,
        "_vectorize",
        lambda files, file_hashes, contents, options: np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.cluster_documents",
        lambda embeddings, **kwargs: make_cluster_result(
            [0, 1, 2],
            metrics=ClusterMetrics(initial_clusters=3),
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        lambda client, result, contents, files, **kwargs: make_cluster_result(
            result,
            cluster_names={0: "Duplicates", 2: "Other"},
        ),
    )

    result = service.run(
        tmp_path,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert len(result.candidate_components) == 1
    assert result.candidate_components[0].component_type == "near_duplicate_group"
    assert result.labels.tolist() == [0, 0, 2]


def test_pipeline_graph_cluster_mode_uses_candidate_components_as_clusters(
    tmp_path: Path, monkeypatch
) -> None:
    doc_a = tmp_path / "linear-algebra-a.txt"
    doc_b = tmp_path / "linear-algebra-b.txt"
    doc_c = tmp_path / "minecraft-guide.txt"
    for path in (doc_a, doc_b, doc_c):
        path.write_text("payload", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_contents(files, options):
        del options
        contents = [
            "Linear Algebra Notes",
            "Linear Algebra Notes",
            "Minecraft Guide",
        ]
        file_hashes = [f"hash-{index}" for index, _ in enumerate(files)]
        reports = [
            ExtractionFileReport(
                file=file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(content),
                selected_source="primary",
                final_effective_length=len(content),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=file_hashes[index],
            )
            for index, (file, content) in enumerate(zip(files, contents, strict=True))
        ]
        return contents, file_hashes, ExtractionSummary(), reports

    monkeypatch.setattr(service, "_extract_contents", fake_extract_contents)
    monkeypatch.setattr(
        service,
        "_vectorize",
        lambda files, file_hashes, contents, options: np.array(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        lambda client, result, contents, files, **kwargs: make_cluster_result(
            result,
            cluster_names={0: "Linear Algebra"},
        ),
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            cluster_mode="graph",
        ),
    )

    assert result.labels.tolist() == [0, 0, -1]
    assert result.cluster_metrics.initial_clusters == 1
    assert result.cluster_metrics.initial_noise == 1


def test_pipeline_uses_topic_clustering_config_when_options_unset(
    tmp_path: Path, monkeypatch
) -> None:
    doc_a = tmp_path / "linear-algebra-a.txt"
    doc_b = tmp_path / "linear-algebra-b.txt"
    doc_c = tmp_path / "minecraft-guide.txt"
    for path in (doc_a, doc_b, doc_c):
        path.write_text("payload", encoding="utf-8")

    config = Config()
    config.topic_clustering.mode = "graph"
    service = PipelineService(client=object(), config=config, cache=None)

    def fake_extract_contents(files, options):
        del options
        contents = [
            "Linear Algebra Notes",
            "Linear Algebra Notes",
            "Minecraft Guide",
        ]
        file_hashes = [f"hash-{index}" for index, _ in enumerate(files)]
        reports = [
            ExtractionFileReport(
                file=file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(content),
                selected_source="primary",
                final_effective_length=len(content),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=file_hashes[index],
            )
            for index, (file, content) in enumerate(zip(files, contents, strict=True))
        ]
        return contents, file_hashes, ExtractionSummary(), reports

    monkeypatch.setattr(service, "_extract_contents", fake_extract_contents)
    monkeypatch.setattr(
        service,
        "_vectorize",
        lambda files, file_hashes, contents, options: np.array(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        lambda client, result, contents, files, **kwargs: make_cluster_result(
            result,
            cluster_names={0: "Linear Algebra"},
        ),
    )

    result = service.run(
        tmp_path,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert result.labels.tolist() == [0, 0, -1]
    assert result.cluster_drafts[0].origin == "graph"


def test_pipeline_graph_cluster_mode_falls_back_for_unconnected_tail(
    tmp_path: Path, monkeypatch
) -> None:
    doc_a = tmp_path / "linear-algebra-a.txt"
    doc_b = tmp_path / "linear-algebra-b.txt"
    doc_c = tmp_path / "numerical-analysis.txt"
    doc_d = tmp_path / "finance.txt"
    for path in (doc_a, doc_b, doc_c, doc_d):
        path.write_text("payload", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_contents(files, options):
        del options
        contents = [
            "Linear Algebra Notes",
            "Linear Algebra Notes",
            "Numerical analysis and interpolation",
            "Finance and valuation notes",
        ]
        file_hashes = [f"hash-{index}" for index, _ in enumerate(files)]
        reports = [
            ExtractionFileReport(
                file=file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(content),
                selected_source="primary",
                final_effective_length=len(content),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=file_hashes[index],
            )
            for index, (file, content) in enumerate(zip(files, contents, strict=True))
        ]
        return contents, file_hashes, ExtractionSummary(), reports

    monkeypatch.setattr(service, "_extract_contents", fake_extract_contents)
    monkeypatch.setattr(
        service,
        "_vectorize",
        lambda files, file_hashes, contents, options: np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
    )

    def fake_cluster_documents(embeddings, **kwargs):
        # Only the two tail docs should fall back into this call.
        assert embeddings.shape[0] == 2
        return make_cluster_result(
            [0, 0],
            metrics=ClusterMetrics(initial_clusters=1),
        )

    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        lambda client, result, contents, files, **kwargs: make_cluster_result(
            result,
            cluster_names={0: "Linear Algebra", 1: "Numerical Finance"},
        ),
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            cluster_mode="graph",
        ),
    )

    assert result.labels.tolist() == [0, 0, 1, 1]


def test_pipeline_adjudication_merge_edge_can_merge_labels(
    tmp_path: Path, monkeypatch
) -> None:
    doc_a = tmp_path / "alpha.txt"
    doc_b = tmp_path / "beta.txt"
    for path in (doc_a, doc_b):
        path.write_text("payload", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_contents(files, options):
        del options
        contents = [
            "Rust ownership and borrowing",
            "Rust ownership with lifetimes",
        ]
        file_hashes = [f"hash-{index}" for index, _ in enumerate(files)]
        reports = [
            ExtractionFileReport(
                file=file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(content),
                selected_source="primary",
                final_effective_length=len(content),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=file_hashes[index],
            )
            for index, (file, content) in enumerate(zip(files, contents, strict=True))
        ]
        return contents, file_hashes, ExtractionSummary(), reports

    monkeypatch.setattr(service, "_extract_contents", fake_extract_contents)
    monkeypatch.setattr(
        service,
        "_vectorize",
        lambda files, file_hashes, contents, options: np.array(
            [[1.0, 0.0], [0.93, 0.07]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.cluster_documents",
        lambda embeddings, **kwargs: make_cluster_result(
            [0, 1],
            metrics=ClusterMetrics(initial_clusters=2),
        ),
    )
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        lambda client, result, contents, files, **kwargs: make_cluster_result(
            result,
            cluster_names={0: "Rust", 1: "Rust"},
        ),
    )

    result = service.run(
        tmp_path,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert result.adjudication_decisions
    assert result.labels.tolist() == [0, 0]


def test_scan_files_excludes_target_directory(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")

    organized = root / "organized"
    organized.mkdir()
    (organized / "b.txt").write_text("beta", encoding="utf-8")

    files = scan_files(
        root,
        config=Config(),
        extensions={".txt"},
        exclude_paths=[organized],
    )

    assert [p.name for p in files] == ["a.txt"]


def test_scan_files_real_docs_fixture_uses_supported_extensions() -> None:
    fixture_dir = TEST_CORPUS_DIR
    assert fixture_dir.exists()

    extensions = Config().formats.all_extensions
    files = scan_files(fixture_dir, config=Config(), extensions=extensions)

    assert files
    assert all(path.suffix.lower() in extensions for path in files)


def test_scan_files_verbose_logs_follow_locale(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.bin").write_text("beta", encoding="utf-8")

    setup_logging(verbose=True)
    set_locale("en")

    files = scan_files(root, config=Config(), extensions={".txt"})

    output = capsys.readouterr().out
    assert [path.name for path in files] == ["a.txt"]
    assert "Scan folder:" in output
    assert "Scan completed:" in output
    assert "Skipped 1 unsupported files" in output
    assert "扫描目录" not in output


def test_pipeline_verbose_logs_include_extraction_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("payload", encoding="utf-8")

    calls: list[dict] = []

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)

            class _Item:
                embedding = [0.1, 0.2]

            class _Usage:
                total_tokens = 12

            class _Response:
                data = [_Item()]
                usage = _Usage()

            return _Response()

    class _Client:
        embeddings = _Embeddings()

    config = Config()
    service = PipelineService(client=_Client(), config=config, cache=None)

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
        return ExtractionResult(
            content="example content that is long enough",
            success=True,
            extractor="text",
        )

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        return make_cluster_result([0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return make_cluster_result(
            result,
            cluster_names={0: "Notes"},
            metrics=ClusterMetrics(),
        )

    monkeypatch.setattr(
        "dite.io.route.extract_document", fake_extract_document
    )
    monkeypatch.setattr(
        "dite.io.route.needs_vlm_fallback", lambda *args, **kwargs: False
    )
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    setup_logging(verbose=True)
    set_locale("en")

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    output = capsys.readouterr().out
    assert len(result.files) == 1
    assert "Processing file:" in output
    assert "Document extraction: extractor=text, success=True" in output
    assert "Extraction summary:" in output
    assert "Vectorizing 1 documents" in output
    assert calls[0]["input"] == [
        "File name: sample.txt\n\nContent:\nexample content that is long enough"
    ]


def test_generate_all_cluster_names_debug_uses_letter_labels(capsys) -> None:
    from dite.cluster.api import generate_all_cluster_names

    class _Completions:
        def create(self, **kwargs):
            name = "Alpha" if "a.txt" in kwargs["messages"][0]["content"] else "Beta"

            class _Message:
                def __init__(self, content: str) -> None:
                    self.content = content

            class _Choice:
                def __init__(self, content: str) -> None:
                    self.message = _Message(content)

            class _Response:
                choices = [_Choice(name)]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    setup_logging(verbose=True)
    set_locale("en")

    result = make_cluster_result([0, 2, 2])
    generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["doc a", "doc b", "doc c"],
        files=[Path("a.txt"), Path("b.txt"), Path("c.txt")],
        config=Config(),
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    output = capsys.readouterr().out
    assert "Cluster A named Alpha" in output
    assert "Cluster B named Beta" in output
    assert "Cluster 0 named" not in output
    assert "Cluster 2 named" not in output


def test_generate_all_cluster_names_async_debug_logs_follow_locale(capsys) -> None:
    from dite.cluster.api import generate_all_cluster_names
    from dite.util.api import ChatCompletionResult

    class _Client:
        chat = None

    class _Runtime:
        def run_cluster_naming_batch(self, requests):
            del requests
            return [
                ChatCompletionResult(
                    content=None,
                    error="boom",
                    queue_wait_sec=0.01,
                    request_elapsed_sec=0.02,
                )
            ]

    setup_logging(verbose=True)
    set_locale("en")

    result = make_cluster_result([0])
    generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["doc a"],
        files=[Path("a.txt")],
        config=Config(),
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
        request_runtime=_Runtime(),
    )

    output = capsys.readouterr().out

    assert "Cluster A async naming request failed:" in output
    assert "wait=0.010s" in output
    assert "request=0.020s" in output
    assert "fallback=" in output
    assert "异步命名请求失败" not in output


def test_generate_all_cluster_names_limits_cluster_naming_workers(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import generate_all_cluster_names

    current = 0
    max_concurrent = 0
    lock = threading.Lock()
    started_two = threading.Event()

    def fake_generate_cluster_name(
        client,
        cluster_embeddings,
        sample_contents,
        sample_names,
        *,
        config,
        top_k=5,
        llm_model=None,
    ) -> str:
        del client, cluster_embeddings, sample_contents, config, top_k, llm_model
        nonlocal current, max_concurrent
        with lock:
            current += 1
            max_concurrent = max(max_concurrent, current)
            if current == 2:
                started_two.set()
        started_two.wait(timeout=0.2)
        time.sleep(0.02)
        with lock:
            current -= 1
        return Path(sample_names[0]).stem.replace("-", " ").title()

    monkeypatch.setattr(clusterer, "generate_cluster_name", fake_generate_cluster_name)

    class _Chat:
        completions = object()

    class _Client:
        chat = _Chat()

    config = Config()
    config.processing.cluster_naming_workers = 2
    result = make_cluster_result([3, 1, 2, 0])

    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["a", "b", "c", "d"],
        files=[
            Path("cluster-3.txt"),
            Path("cluster-1.txt"),
            Path("cluster-2.txt"),
            Path("cluster-0.txt"),
        ],
        config=config,
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert named_result.name_clusters_merged == 0
    assert max_concurrent == 2
    assert named_result.cluster_names == {
        0: "Cluster 0",
        1: "Cluster 1",
        2: "Cluster 2",
        3: "Cluster 3",
    }


def test_generate_all_cluster_names_preserves_label_mapping_out_of_order_completion(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import generate_all_cluster_names

    delays = {
        "cluster-2-a.txt": 0.08,
        "cluster-5-a.txt": 0.03,
        "cluster-7-a.txt": 0.0,
    }

    def fake_generate_cluster_name(
        client,
        cluster_embeddings,
        sample_contents,
        sample_names,
        *,
        config,
        top_k=5,
        llm_model=None,
    ) -> str:
        del client, cluster_embeddings, sample_contents, config, top_k, llm_model
        first_name = sample_names[0]
        time.sleep(delays[first_name])
        return f"Name {Path(first_name).stem.split('-')[1]}"

    monkeypatch.setattr(clusterer, "generate_cluster_name", fake_generate_cluster_name)

    class _Chat:
        completions = object()

    class _Client:
        chat = _Chat()

    config = Config()
    config.processing.cluster_naming_workers = 3
    labels = np.array([5, 2, 5, 7, 2], dtype=int)
    result = make_cluster_result(labels)

    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["a", "b", "c", "d", "e"],
        files=[
            Path("cluster-5-a.txt"),
            Path("cluster-2-a.txt"),
            Path("cluster-5-b.txt"),
            Path("cluster-7-a.txt"),
            Path("cluster-2-b.txt"),
        ],
        config=config,
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert named_result.name_clusters_merged == 0
    assert np.array_equal(named_result.labels, labels)
    assert named_result.cluster_names == {
        2: "Name 2",
        5: "Name 5",
        7: "Name 7",
    }


def test_generate_all_cluster_names_merges_duplicate_names_after_concurrent_completion(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import generate_all_cluster_names

    delays = {
        "cluster-1-a.txt": 0.08,
        "cluster-3-a.txt": 0.0,
        "cluster-5-a.txt": 0.03,
    }
    names = {
        "cluster-1-a.txt": "Shared Topic",
        "cluster-3-a.txt": "Unique Topic",
        "cluster-5-a.txt": "Shared Topic",
    }

    def fake_generate_cluster_name(
        client,
        cluster_embeddings,
        sample_contents,
        sample_names,
        *,
        config,
        top_k=5,
        llm_model=None,
    ) -> str:
        del client, cluster_embeddings, sample_contents, config, top_k, llm_model
        first_name = sample_names[0]
        time.sleep(delays[first_name])
        return names[first_name]

    monkeypatch.setattr(clusterer, "generate_cluster_name", fake_generate_cluster_name)

    class _Chat:
        completions = object()

    class _Client:
        chat = _Chat()

    config = Config()
    config.processing.cluster_naming_workers = 3
    labels = np.array([5, 1, 3, 5, 1], dtype=int)
    result = make_cluster_result(labels)

    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["a", "b", "c", "d", "e"],
        files=[
            Path("cluster-5-a.txt"),
            Path("cluster-1-a.txt"),
            Path("cluster-3-a.txt"),
            Path("cluster-5-b.txt"),
            Path("cluster-1-b.txt"),
        ],
        config=config,
        embeddings=None,
        merge_same_name=True,
        llm_model="dummy-model",
    )

    assert named_result.name_clusters_merged == 1
    assert np.array_equal(named_result.labels, np.array([1, 1, 3, 1, 1], dtype=int))
    assert named_result.cluster_names == {
        1: "Shared Topic",
        3: "Unique Topic",
    }


def test_generate_all_cluster_names_prewarms_client_resources_before_worker_threads(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import generate_all_cluster_names

    tracker = {
        "chat_init_threads": [],
        "completions_init_threads": [],
    }
    main_thread_id = threading.get_ident()

    class _ChatResource:
        def __init__(self) -> None:
            self._completions = None

        @property
        def completions(self):
            if self._completions is None:
                tracker["completions_init_threads"].append(threading.get_ident())
                self._completions = object()
            return self._completions

    class _Client:
        def __init__(self) -> None:
            self._chat = None

        @property
        def chat(self):
            if self._chat is None:
                tracker["chat_init_threads"].append(threading.get_ident())
                self._chat = _ChatResource()
            return self._chat

    def fake_generate_cluster_name(
        client,
        cluster_embeddings,
        sample_contents,
        sample_names,
        *,
        config,
        top_k=5,
        llm_model=None,
    ) -> str:
        del cluster_embeddings, sample_contents, config, top_k, llm_model
        _ = client.chat.completions
        time.sleep(0.01)
        return Path(sample_names[0]).stem

    monkeypatch.setattr(clusterer, "generate_cluster_name", fake_generate_cluster_name)

    config = Config()
    config.processing.cluster_naming_workers = 3
    result = make_cluster_result([0, 1, 2])

    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["a", "b", "c"],
        files=[
            Path("cluster-0.txt"),
            Path("cluster-1.txt"),
            Path("cluster-2.txt"),
        ],
        config=config,
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert named_result.name_clusters_merged == 0
    assert named_result.cluster_names == {
        0: "cluster-0",
        1: "cluster-1",
        2: "cluster-2",
    }
    assert tracker["chat_init_threads"] == [main_thread_id]
    assert tracker["completions_init_threads"] == [main_thread_id]


def test_generate_all_cluster_names_integration_merges_same_names_deterministically():
    from dite.cluster.api import generate_all_cluster_names

    calls: list[str] = []

    class _Completions:
        def create(self, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            calls.append(prompt)
            if "Financial Statement" in prompt:
                time.sleep(0.01)
                name = "Financial Reports"
            else:
                time.sleep(0.04 if "cluster-8" in prompt else 0.0)
                name = "Linear Algebra Notes"

            class _Message:
                def __init__(self, content: str) -> None:
                    self.content = content

            class _Choice:
                def __init__(self, content: str) -> None:
                    self.message = _Message(content)

            class _Response:
                def __init__(self, content: str) -> None:
                    self.choices = [_Choice(content)]

            return _Response(name)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    config = Config()
    config.processing.cluster_naming_workers = 3
    labels = np.array([8, 8, 3, 3, 5, 5], dtype=int)
    result = make_cluster_result(labels)

    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=[
            "Linear Algebra Workbook\nMatrices and vectors.",
            "Linear Algebra Exercises\nEigenvalues and bases.",
            "Linear Algebra Lecture Notes\nVector spaces and rank.",
            "Linear Algebra Reference\nMatrix decompositions.",
            "Financial Statement Analysis\nCash flow and balance sheet.",
            "Financial Statement Basics\nIncome statement overview.",
        ],
        files=[
            Path("cluster-8-a.txt"),
            Path("cluster-8-b.txt"),
            Path("cluster-3-a.txt"),
            Path("cluster-3-b.txt"),
            Path("cluster-5-a.txt"),
            Path("cluster-5-b.txt"),
        ],
        config=config,
        embeddings=None,
        merge_same_name=True,
        llm_model="dummy-model",
    )

    assert len(calls) == 3
    assert named_result.name_clusters_merged == 1
    assert np.array_equal(named_result.labels, np.array([3, 3, 3, 3, 5, 5], dtype=int))
    assert named_result.cluster_names == {
        3: "Linear Algebra Notes",
        5: "Financial Reports",
    }


def test_generate_all_cluster_names_skips_client_access_for_noise_only_input() -> None:
    from dite.cluster.api import generate_all_cluster_names

    class _Client:
        @property
        def chat(self):
            raise AssertionError("noise-only input should not access client resources")

    result = make_cluster_result([-1, -1, -1])
    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["a", "b", "c"],
        files=[Path("a.txt"), Path("b.txt"), Path("c.txt")],
        config=Config(),
        embeddings=None,
        merge_same_name=True,
        llm_model="dummy-model",
    )

    assert np.array_equal(named_result.labels, result.labels)
    assert named_result.cluster_names == {}
    assert named_result.name_clusters_merged == 0


def test_generate_all_cluster_names_clamps_non_positive_worker_count_to_one() -> None:
    from dite.cluster.api import generate_all_cluster_names

    current = 0
    max_concurrent = 0
    lock = threading.Lock()

    class _Completions:
        def create(self, **kwargs):
            del kwargs
            nonlocal current, max_concurrent
            with lock:
                current += 1
                max_concurrent = max(max_concurrent, current)
            time.sleep(0.02)
            with lock:
                current -= 1

            class _Message:
                content = "Stable Topic"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    config = Config()
    config.processing.cluster_naming_workers = 0
    labels = np.array([0, 0, 1, 1], dtype=int)
    result = make_cluster_result(labels)

    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=[
            "Topic Zero\nAlpha",
            "Topic Zero\nBeta",
            "Topic One\nGamma",
            "Topic One\nDelta",
        ],
        files=[
            Path("zero-a.txt"),
            Path("zero-b.txt"),
            Path("one-a.txt"),
            Path("one-b.txt"),
        ],
        config=config,
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert max_concurrent == 1
    assert named_result.name_clusters_merged == 0
    assert np.array_equal(named_result.labels, labels)
    assert named_result.cluster_names == {
        0: "Stable Topic",
        1: "Stable Topic",
    }


def test_generate_all_cluster_names_keeps_duplicate_names_when_merge_disabled() -> None:
    from dite.cluster.api import generate_all_cluster_names

    class _Completions:
        def create(self, **kwargs):
            del kwargs

            class _Message:
                content = "Shared Topic"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    result = make_cluster_result([4, 4, 7, 7])
    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=[
            "Topic Four\nAlpha",
            "Topic Four\nBeta",
            "Topic Seven\nGamma",
            "Topic Seven\nDelta",
        ],
        files=[
            Path("four-a.txt"),
            Path("four-b.txt"),
            Path("seven-a.txt"),
            Path("seven-b.txt"),
        ],
        config=Config(),
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert named_result.name_clusters_merged == 0
    assert np.array_equal(named_result.labels, result.labels)
    assert named_result.cluster_names == {
        4: "Shared Topic",
        7: "Shared Topic",
    }


def test_generate_all_cluster_names_uses_async_request_runtime_when_available(
    monkeypatch,
) -> None:
    from dite.cluster import api as clusterer
    from dite.cluster.api import generate_all_cluster_names

    class _Runtime:
        def __init__(self) -> None:
            self.calls: list[list[object]] = []

        def run_cluster_naming_batch(self, requests):
            self.calls.append(requests)
            return [
                ChatCompletionResult(
                    content="Linear Algebra",
                    error=None,
                    queue_wait_sec=0.01,
                    request_elapsed_sec=0.02,
                ),
                ChatCompletionResult(
                    content="Financial Reports",
                    error=None,
                    queue_wait_sec=0.01,
                    request_elapsed_sec=0.03,
                ),
            ]

    def fail_generate_cluster_name(*args, **kwargs):
        raise AssertionError(
            "sync generate_cluster_name should not run when runtime exists"
        )

    monkeypatch.setattr(clusterer, "generate_cluster_name", fail_generate_cluster_name)

    class _Chat:
        completions = object()

    class _Client:
        base_url = "https://api.example.com/v1"
        chat = _Chat()

    runtime = _Runtime()
    result = make_cluster_result([0, 0, 1, 1])
    config = Config()
    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=[
            "Matrices and vectors",
            "Linear transformations",
            "Cash flow statement",
            "Balance sheet review",
        ],
        files=[
            Path("linear-a.txt"),
            Path("linear-b.txt"),
            Path("finance-a.txt"),
            Path("finance-b.txt"),
        ],
        config=config,
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
        request_runtime=runtime,
    )

    assert named_result.name_clusters_merged == 0
    assert np.array_equal(named_result.labels, result.labels)
    assert named_result.cluster_names == {
        0: "Linear Algebra",
        1: "Financial Reports",
    }
    assert len(runtime.calls) == 1
    assert len(runtime.calls[0]) == 2


def test_generate_all_cluster_names_labels_invalid_model_output_per_cluster() -> None:
    from dite.cluster.api import generate_all_cluster_names

    class _Completions:
        def create(self, **kwargs):
            del kwargs

            class _Message:
                content = "cover"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    set_locale("en")
    result = make_cluster_result([2, 2, 5, 5])
    named_result = generate_all_cluster_names(
        client=_Client(),
        result=result,
        contents=["   ", "\n\n", "   ", "\n\n"],
        files=[
            Path("cover.pdf"),
            Path("scan.png"),
            Path("cover(1).pdf"),
            Path("scan(1).png"),
        ],
        config=Config(),
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert named_result.name_clusters_merged == 0
    assert np.array_equal(named_result.labels, result.labels)
    assert named_result.cluster_names == {
        2: "Unnamed-2",
        5: "Unnamed-5",
    }


def test_merge_clusters_by_name_merges_duplicate_names() -> None:
    from dite.cluster.api import merge_clusters_by_name

    labels = np.array([0, 1, 1, 2, -1], dtype=int)
    cluster_names = {0: "机器学习", 1: "机器学习", 2: "财务"}

    merged_labels, merged_names, merged_count = merge_clusters_by_name(
        labels, cluster_names
    )

    assert merged_count == 1
    assert np.array_equal(merged_labels, np.array([0, 0, 0, 2, -1], dtype=int))
    assert merged_names == {0: "机器学习", 2: "财务"}


def test_merge_clusters_by_name_does_not_merge_unnamed_fallback() -> None:
    from dite.cluster.api import merge_clusters_by_name

    labels = np.array([0, 1, 2, 2], dtype=int)
    cluster_names = {0: "未命名", 1: "未命名", 2: "线性代数"}

    merged_labels, merged_names, merged_count = merge_clusters_by_name(
        labels, cluster_names
    )

    assert merged_count == 0
    assert np.array_equal(merged_labels, labels)
    assert merged_names == cluster_names


def test_generate_all_cluster_names_falls_back_to_unnamed_on_api_error() -> None:
    from dite.cluster.api import generate_all_cluster_names

    class _FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("api down")

    class _FailingChat:
        completions = _FailingCompletions()

    class _FailingClient:
        chat = _FailingChat()

    labels = np.array([0, 1], dtype=int)
    contents = ["机器学习导论\nA" * 80, "财务报表分析\nB" * 80]
    files = [Path("ml.txt"), Path("finance.txt")]
    embeddings = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)

    result = make_cluster_result(labels)
    named_result = generate_all_cluster_names(
        client=_FailingClient(),
        result=result,
        contents=contents,
        files=files,
        config=Config(),
        embeddings=embeddings,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert np.array_equal(named_result.labels, labels)
    assert named_result.cluster_names == {0: "机器学习导论", 1: "财务报表分析"}
    assert named_result.name_clusters_merged == 0


def test_generate_cluster_name_truncates_long_contents_before_api_call() -> None:
    from dite.cluster.api import (
        CLUSTER_NAME_CONTENT_LIMIT,
        CLUSTER_NAME_EXCERPT_LIMIT,
        generate_cluster_name,
    )

    captured: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            captured.append(kwargs)

            class _Message:
                content = "数学教材"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()
        base_url = "https://api.siliconflow.cn/v1"

    cfg = Config(
        request_profiles=RequestProfilesConfig(
            cluster_naming=ChatCompletionProfileConfig(
                max_tokens=64,
                reasoning_mode="off",
            )
        )
    )
    set_locale("en")
    long_text = "A" * (CLUSTER_NAME_CONTENT_LIMIT + 500)
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=[long_text],
        sample_names=["long.txt"],
        llm_model="Qwen/Qwen3.5-27B",
        config=cfg,
    )

    assert name == "数学教材"
    prompt = captured[0]["messages"][0]["content"]
    assert "File name: long.txt" in prompt
    assert "Title candidate: long" in prompt
    assert "Name this category in 2-4 English words." in prompt
    assert "A" * (CLUSTER_NAME_EXCERPT_LIMIT + 100) not in prompt
    assert captured[0]["max_tokens"] == 64
    assert captured[0]["extra_body"] == {"enable_thinking": False}


def test_generate_cluster_name_uses_representative_top_k_samples_from_embeddings() -> (
    None
):
    from dite.cluster.api import generate_cluster_name

    captured: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            captured.append(kwargs)

            class _Message:
                content = "Study Materials"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    set_locale("en")
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=np.array(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.97, 0.03],
                [0.96, 0.04],
                [0.95, 0.05],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        sample_contents=[
            "Doc 0\nTopic A summary.",
            "Doc 1\nTopic A summary.",
            "Doc 2\nTopic A summary.",
            "Doc 3\nTopic A summary.",
            "Doc 4\nTopic A summary.",
            "Outlier Doc\nTotally different topic.",
        ],
        sample_names=[
            "doc-0.txt",
            "doc-1.txt",
            "doc-2.txt",
            "doc-3.txt",
            "doc-4.txt",
            "outlier.txt",
        ],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "Study Materials"
    prompt = captured[0]["messages"][0]["content"]
    assert "outlier.txt" not in prompt
    assert "doc-0.txt" in prompt
    assert "doc-1.txt" in prompt
    assert "doc-2.txt" in prompt
    assert "doc-3.txt" in prompt
    assert "doc-4.txt" in prompt


def test_generate_cluster_name_uses_file_name_prompt_when_all_contents_blank() -> None:
    from dite.cluster.api import generate_cluster_name

    captured: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            captured.append(kwargs)

            class _Message:
                content = "Machine Learning Notes"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    set_locale("en")
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["   ", "\n\n"],
        sample_names=["machine-learning-notes.pdf", "neural-network-summary.md"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "Machine Learning Notes"
    prompt = captured[0]["messages"][0]["content"]
    assert "These files belong to the same category" in prompt
    assert "machine-learning-notes.pdf" in prompt
    assert "neural-network-summary.md" in prompt
    assert "Content excerpt:" not in prompt


def test_generate_cluster_name_falls_back_to_file_name_when_blank_and_api_fails():
    from dite.cluster.api import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("api down")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    set_locale("en")
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["   ", "\n\n"],
        sample_names=["machine-learning-notes.pdf", "neural-network-summary.md"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "machine learning notes"


def test_generate_cluster_name_returns_unnamed_when_no_signal_is_available() -> None:
    from dite.cluster.api import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("api down")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    set_locale("en")
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["   ", "\n\n"],
        sample_names=["scan.pdf", "cover.png"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "Unnamed"


def test_generate_cluster_name_builds_zh_prompt_from_content_samples() -> None:
    from dite.cluster.api import generate_cluster_name

    captured: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            captured.append(kwargs)

            class _Message:
                content = "线性代数"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    set_locale("zh-CN")
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["线性代数导论\n矩阵、向量与特征值。"],
        sample_names=["linear-algebra.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "线性代数"
    prompt = captured[0]["messages"][0]["content"]
    assert "以下是属于同一类别的代表文档信息：" in prompt
    assert "文件名: linear-algebra.pdf" in prompt
    assert "标题候选: 线性代数导论" in prompt
    assert "请用2-4个中文词为这个类别命名。" in prompt


def test_generate_cluster_name_truncates_long_valid_model_output() -> None:
    from dite.cluster.api import CLUSTER_NAME_OUTPUT_LIMIT, generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            del kwargs

            class _Message:
                content = "Advanced Linear Algebra Reference Materials"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    set_locale("en")
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["Linear Algebra Reference\nMatrices and decompositions."],
        sample_names=["linear-algebra-reference.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert (
        name
        == "Advanced Linear Algebra Reference Materials"[:CLUSTER_NAME_OUTPUT_LIMIT]
    )


def test_cluster_naming_helper_normalization_and_invalid_detection() -> None:
    from dite.cluster.api import (
        _is_invalid_cluster_name,
        _normalize_cluster_name_text,
    )

    assert _normalize_cluster_name_text(
        '  <!-- image -->  ## "Linear   Algebra"  '
    ) == ("Linear Algebra")
    assert _is_invalid_cluster_name("cover") is True
    assert _is_invalid_cluster_name("Page 3") is True
    assert _is_invalid_cluster_name("线性代数") is False


def test_cluster_naming_helper_author_and_title_detection() -> None:
    from dite.cluster.api import _extract_title_like_line, _looks_like_author_line

    assert _looks_like_author_line("Shuo Wang * Chunlong Xia") is True
    assert _looks_like_author_line("王硕 * 夏春龙") is False
    assert (
        _extract_title_like_line(
            "<!-- image -->\nShuo Wang * Chunlong Xia\nDocument Intelligence\n摘要"
        )
        == "Document Intelligence"
    )


def test_cluster_naming_helper_file_name_and_heuristic_fallbacks() -> None:
    from dite.cluster.api import (
        _fallback_name_from_file_name,
        _heuristic_cluster_name,
    )

    assert _fallback_name_from_file_name("linear-algebra (1).pdf") == "linear algebra"
    assert _fallback_name_from_file_name("scan.pdf") == ""
    assert (
        _heuristic_cluster_name(
            ["   ", "\n\n"],
            ["linear-algebra-notes.pdf", "scan.png"],
        )
        == "linear algebra notes"
    )


def test_cluster_naming_helper_debug_tokens_and_labels_roll_over() -> None:
    from dite.cluster.api import _build_cluster_debug_labels, _cluster_debug_token

    assert _cluster_debug_token(25) == "Z"
    assert _cluster_debug_token(26) == "AA"
    assert _cluster_debug_token(27) == "AB"
    assert _build_cluster_debug_labels(np.array([100, 2, 30], dtype=int)) == {
        2: "A",
        30: "B",
        100: "C",
    }


def test_cluster_naming_helper_builds_zh_file_name_only_prompt() -> None:
    from dite.cluster.api import _build_cluster_naming_prompt

    set_locale("zh-CN")
    prompt = _build_cluster_naming_prompt([], ["a.pdf", "b.pdf", "c.pdf"], 2)

    assert "以下文件属于同一类别，请根据文件名推测类别：" in prompt
    assert "a.pdf" in prompt
    assert "b.pdf" in prompt
    assert "c.pdf" not in prompt
    assert "请用2-4个中文词为这个类别命名。" in prompt


def test_cluster_naming_helper_compact_sample_uses_file_name_when_title_missing() -> (
    None
):
    from dite.cluster.api import _compact_sample_for_naming

    set_locale("en")
    sample = _compact_sample_for_naming(
        "linear-algebra-reference.pdf",
        "<!-- image -->\ncover\n",
    )

    assert "File name: linear-algebra-reference.pdf" in sample
    assert "Title candidate: linear algebra reference" in sample
    assert "Content excerpt: <!-- image --> cover" in sample


def test_generate_cluster_name_uses_heuristic_when_response_is_empty() -> None:
    from dite.cluster.api import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            class _Message:
                content = ""

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["线性代数导论\n这是一本教材的内容摘要。"],
        sample_names=["linear-algebra.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "线性代数导论"


def test_generate_cluster_name_skips_placeholder_and_author_lines() -> None:
    from dite.cluster.api import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            class _Message:
                content = ""

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=[
            "<!-- image -->\nShuo Wang * Chunlong Xia\nDocument Intelligence\n摘要",
        ],
        sample_names=["document-intelligence.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "Document Intelligence"


def test_generate_cluster_name_uses_heuristic_for_invalid_model_output() -> None:
    from dite.cluster.api import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            class _Message:
                content = "<!-- image -->"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["操作系统原理\n这是一本教材的内容摘要。"],
        sample_names=["os.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "操作系统原理"


def test_generate_cluster_name_retries_provider_server_error_then_succeeds() -> None:
    from dite.cluster.api import generate_cluster_name

    calls = {"count": 0}

    class _Completions:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                request = httpx.Request(
                    "POST",
                    "https://api.example.com/v1/chat/completions",
                )
                response = httpx.Response(
                    500,
                    request=request,
                    headers={"x-request-id": "rid-retry"},
                )
                raise APIStatusError(
                    "provider boom",
                    response=response,
                    body={
                        "code": 50507,
                        "message": "Request processing failed due to an unknown error.",
                        "data": None,
                    },
                )

            class _Message:
                content = "线性代数教材"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["线性代数导论\n教材内容摘要"],
        sample_names=["linear-algebra.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "线性代数教材"
    assert calls["count"] == 3


def test_generate_cluster_name_formats_final_api_error_in_debug(capsys) -> None:
    from dite.cluster.api import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            request = httpx.Request(
                "POST",
                "https://api.example.com/v1/chat/completions",
            )
            response = httpx.Response(
                500,
                request=request,
                headers={"x-request-id": "rid-final"},
            )
            raise APIStatusError(
                "provider boom",
                response=response,
                body={
                    "code": 50507,
                    "message": "Request processing failed due to an unknown error.",
                    "data": None,
                },
            )

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    setup_logging(verbose=True)
    set_locale("en")

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["操作系统原理\n这是一本教材的内容摘要。"],
        sample_names=["os.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    output = capsys.readouterr().out
    assert name == "操作系统原理"
    assert "Request processing failed" in output
    assert "unknown error." in output
    assert "status=500" in output
    assert "code=50507" in output
    assert "{'code': 50507" not in output


def test_extract_with_vlm_fallback_uses_passed_config(
    tmp_path: Path, monkeypatch
) -> None:
    from dite.app.config import Config
    from dite.io import router

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_text("fake", encoding="utf-8")
    cfg = Config()
    expected = ExtractionResult(
        content="vlm content",
        success=True,
        extractor="vlm_fallback",
    )
    captured: dict[str, object] = {}

    def fake_extract(
        file_path, client, config, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        captured["file_path"] = file_path
        captured["client"] = client
        captured["config"] = config
        captured["max_pages"] = max_pages
        return VLMSamplingResult(
            result=expected,
            api_page_calls=2,
            sample_page_limit=max_pages,
        )

    monkeypatch.setattr(router, "_extract_pdf_with_vlm_sampling", fake_extract)

    client = object()
    result = router.extract_with_vlm_fallback(pdf_path, client=client, config=cfg)

    assert result == expected
    assert captured["file_path"] == pdf_path
    assert captured["client"] is client
    assert captured["config"] is cfg
    assert captured["max_pages"] == PDF_VLM_SAMPLE_PAGE_LIMIT


def test_extract_with_vlm_fallback_requires_explicit_config(tmp_path: Path) -> None:
    from dite.io import router

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_text("fake", encoding="utf-8")

    with pytest.raises(TypeError):
        router.extract_with_vlm_fallback(
            pdf_path,
            client=object(),
        )
