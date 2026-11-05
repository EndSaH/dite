import json
import shlex
import shutil
from pathlib import Path

import httpx
import numpy as np
from openai import APIStatusError
from typer.testing import CliRunner

from dite.app.cli import app
from dite.cache import FileCache
from dite.cluster.api import ClusterMetrics, ClusterResult
from dite.flow.api import PipelineResult
from dite.io.base import ExtractionResult
from dite.io.route import (
    PDF_VLM_SAMPLE_PAGE_LIMIT,
    PDFProfile,
    ResolvedExtraction,
)
from dite.util.llm import format_api_error

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
            metrics=metrics or ClusterMetrics(),
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


def _write_test_config(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "api:",
                "  base_url: https://api.example.com/v1",
                "  api_key: dummy-key",
                "cache:",
                f"  directory: {tmp_path / 'cache'}",
                "i18n:",
                "  locale: en",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_scan_cli_end_to_end_with_cache(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha content", encoding="utf-8")
    (docs / "b.txt").write_text("beta content", encoding="utf-8")

    report_path = tmp_path / "report.json"
    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

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
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

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
        config=None,
    ):
        del client, contents, files, embeddings, merge_same_name, llm_model, config
        return make_cluster_result(result, cluster_names={0: "Cluster_A"})

    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    first = runner.invoke(
        app,
        [
            "scan",
            str(docs),
            "--output",
            str(report_path),
        ],
    )
    second = runner.invoke(
        app,
        [
            "scan",
            str(docs),
            "--output",
            str(report_path),
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_files"] == 2
    assert report["summary"]["num_clusters"] == 1
    assert report["summary"]["num_noise"] == 0

    cache = FileCache(db_path=tmp_path / "cache" / "cache.db")
    stats = cache.get_stats()
    assert stats["total_entries"] == 2
    assert stats["with_embedding"] == 2
    assert stats["with_vlm"] == 0
    cache.close()


def test_scan_cli_reports_real_duplicate_fixture_groups_only_in_verbose(
    tmp_path: Path, monkeypatch
) -> None:
    fixture_dir = TEST_CORPUS_DIR
    docs = tmp_path / "docs"
    docs.mkdir()
    duplicate_names = [
        "2506.12116v3.pdf",
        "2506.12116v3 (1).pdf",
        "2506.12116v3 (2).pdf",
    ]
    for name in duplicate_names:
        shutil.copy2(fixture_dir / name, docs / name)

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

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
        content = f"{file_path.name} usable text layer " * 10
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=content,
                success=True,
                extractor="docling",
            ),
            primary_effective_length=len(content),
            pdf_profile=PDFProfile(
                kind="native_text",
                effective_length=len(content),
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=False,
                success=True,
                reason="usable_text_layer",
            ),
            fallback_needed=False,
            selected_source="primary",
            final_content=content,
            final_effective_length=len(content),
            vlm_content=None,
            vlm_source=None,
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
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
        del client, texts, file_names, embedding_model, input_mode
        return np.array(
            [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
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
        del embeddings, repair_noise, clustering, allow_single_cluster, item_names
        return make_cluster_result([0, 0, 0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config=None,
    ):
        del client, contents, files, embeddings, merge_same_name, llm_model, config
        return make_cluster_result(result, cluster_names={0: "Duplicate Papers"})

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names",
        fake_generate_all_cluster_names,
    )

    result = runner.invoke(app, ["scan", str(docs), "--no-cache"])
    verbose_result = runner.invoke(app, ["scan", str(docs), "--no-cache", "--verbose"])

    assert result.exit_code == 0
    assert "Duplicates: 2" in result.output
    assert "Duplicate file groups detected:" not in result.output

    assert verbose_result.exit_code == 0
    assert "Duplicates: 2" in verbose_result.output
    assert "DEBUG: Duplicate file groups detected:" in verbose_result.output
    assert "2506.12116v3.pdf" in verbose_result.output
    assert "2506.12116v3 (1).pdf" in verbose_result.output
    assert "2506.12116v3 (2).pdf" in verbose_result.output


def test_scan_cli_handles_api_connection_error(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha content", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        import httpx
        from openai import APIConnectionError

        raise APIConnectionError(
            message="Connection error.",
            request=httpx.Request("POST", "https://api.example.com/v1/embeddings"),
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        [
            "scan",
            str(docs),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "API connection failed" in result.output
    assert "Traceback" not in result.output


def test_format_api_error_extracts_provider_fields() -> None:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(
        500,
        request=request,
        headers={"x-request-id": "rid-123"},
    )
    exc = APIStatusError(
        "provider boom",
        response=response,
        body={
            "code": 50507,
            "message": "Request processing failed due to an unknown error.",
            "data": None,
        },
    )

    formatted = format_api_error(exc)

    assert (
        formatted == "Request processing failed due to an unknown error. "
        "(status=500, code=50507, request_id=rid-123)"
    )


def test_organize_cli_execute_avoids_overwrite_and_moves_files(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    first = docs / "a" / "report.txt"
    second = docs / "b" / "report.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    target = tmp_path / "organized"
    existing_cluster = target / "finance-reports-"
    existing_cluster.mkdir(parents=True)
    (existing_cluster / "report.txt").write_text("existing", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        return PipelineResult(
            files=[first, second],
            contents=["first", "second"],
            embeddings=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            labels=np.array([0, 0], dtype=int),
            cluster_names={0: "finance/reports:*"},
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        [
            "organize",
            str(docs),
            "--target",
            str(target),
            "--execute",
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert (existing_cluster / "report.txt").read_text(encoding="utf-8") == "existing"
    assert (existing_cluster / "report_1.txt").read_text(encoding="utf-8") == "first"
    assert (existing_cluster / "report_2.txt").read_text(encoding="utf-8") == "second"
    assert not first.exists()
    assert not second.exists()


def test_organize_cli_passes_target_to_pipeline_exclude_paths(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run(self, folder: Path, options):
        captured["exclude_paths"] = options.exclude_paths
        captured["merge_same_name"] = options.merge_same_name
        return PipelineResult(
            files=[source],
            contents=["alpha"],
            embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
            labels=np.array([0], dtype=int),
            cluster_names={0: "Cluster_A"},
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    result = runner.invoke(app, ["organize", str(docs), "--dry-run"])

    assert result.exit_code == 0
    assert captured["exclude_paths"] == [(docs / "organized").resolve()]
    assert captured["merge_same_name"] is False


def test_scan_cli_disables_same_name_merge_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()
    captured: dict[str, bool] = {}

    def fake_run(self, folder: Path, options):
        captured["merge_same_name"] = options.merge_same_name
        return PipelineResult(
            files=[source],
            contents=["alpha"],
            embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
            labels=np.array([0], dtype=int),
            cluster_names={0: "Cluster_A"},
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    result = runner.invoke(app, ["scan", str(docs)])

    assert result.exit_code == 0
    assert captured["merge_same_name"] is False


def test_scan_cli_report_keeps_duplicate_aliases_in_same_cluster(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "paper.txt"
    second = docs / "paper copy.txt"
    first.write_text("same paper content", encoding="utf-8")
    second.write_text("same paper content", encoding="utf-8")
    report_path = tmp_path / "report.json"

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        del self, folder, options
        return PipelineResult(
            files=[first, second],
            contents=["same paper content", "same paper content"],
            embeddings=np.array([[0.6, 0.8], [0.6, 0.8]], dtype=np.float32),
            labels=np.array([0, 0], dtype=int),
            cluster_names={0: "Duplicate Papers"},
            noise_repaired=0,
            clusters_merged=0,
            cluster_metrics=ClusterMetrics(
                initial_clusters=1,
                small_cluster_merge_events=[],
                small_cluster_skip_events=[],
            ),
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        ["scan", str(docs), "--output", str(report_path)],
    )

    assert result.exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_files"] == 2
    assert report["summary"]["initial_num_clusters"] == 1
    assert report["summary"]["num_clusters"] == 1
    assert report["summary"]["num_noise"] == 0
    assert report["summary"]["small_clusters_merged"] == 0
    assert report["summary"]["name_clusters_merged"] == 0
    assert report["summary"]["total_clusters_merged"] == 0
    assert report["summary"]["initial_num_noise"] == 0
    assert report["summary"]["small_cluster_merge_candidates"] == 0
    assert report["summary"]["small_cluster_merge_skipped"] == 0
    assert report["cluster_diagnostics"]["small_cluster_merge_events"] == []
    assert report["cluster_diagnostics"]["small_cluster_skip_events"] == []
    assert report["cluster_diagnostics"]["small_cluster_merge_max_similarity"] is None
    assert report["clusters"][0]["name"] == "Duplicate Papers"
    assert {file["name"] for file in report["clusters"][0]["files"]} == {
        "paper.txt",
        "paper copy.txt",
    }


def test_organize_cli_script_moves_duplicate_aliases_to_same_cluster(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "paper.txt"
    second = docs / "paper copy.txt"
    first.write_text("same paper content", encoding="utf-8")
    second.write_text("same paper content", encoding="utf-8")
    script_path = tmp_path / "organize.sh"

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        del self, folder
        assert options.exclude_paths == [(docs / "organized").resolve()]
        return PipelineResult(
            files=[first, second],
            contents=["same paper content", "same paper content"],
            embeddings=np.array([[0.6, 0.8], [0.6, 0.8]], dtype=np.float32),
            labels=np.array([0, 0], dtype=int),
            cluster_names={0: "Duplicate Papers"},
            noise_repaired=0,
            clusters_merged=0,
            cluster_metrics=ClusterMetrics(initial_clusters=1),
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        ["organize", str(docs), "--output-script", str(script_path)],
    )

    assert result.exit_code == 0
    script = script_path.read_text(encoding="utf-8")
    cluster_dir = docs / "organized" / "Duplicate Papers"
    assert f"mkdir -p -- {shlex.quote(str(cluster_dir))}" in script
    assert (
        f"cp -p -- {shlex.quote(str(first))} "
        f"{shlex.quote(str(cluster_dir / first.name))}"
    ) in script
    assert (
        f"cp -p -- {shlex.quote(str(second))} "
        f"{shlex.quote(str(cluster_dir / second.name))}"
    ) in script


def test_organize_cli_output_script_generates_shell_script(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "paper.txt"
    source.write_text("paper", encoding="utf-8")
    script_path = tmp_path / "organize.sh"

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        return PipelineResult(
            files=[source],
            contents=["paper"],
            embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
            labels=np.array([0]),
            cluster_names={0: "Research Notes"},
            noise_repaired=0,
            clusters_merged=0,
            cluster_metrics=ClusterMetrics(initial_clusters=1),
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        ["organize", str(docs), "--output-script", str(script_path)],
    )

    assert result.exit_code == 0
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    assert "mkdir -p -- " in script
    assert "cp -p -- " in script
    assert str(source) in script
    assert source.exists()


def test_cache_clear_vlm_cli_clears_only_vlm_entries(
    tmp_path: Path, monkeypatch
) -> None:
    _write_test_config(tmp_path, monkeypatch)

    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"

    cache = FileCache(cache_dir=tmp_path / "cache")
    cache.save(
        file_path=file_a,
        file_hash="hash-a",
        file_mtime=1.0,
        content_md="doc-a",
        vlm_content="vlm-a",
        vlm_version=2,
        embedding=np.array([1.0, 2.0], dtype=np.float32),
        model_version="embed-v1",
    )
    cache.save(
        file_path=file_b,
        file_hash="hash-b",
        file_mtime=1.0,
        content_md="doc-b",
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        model_version="embed-v1",
    )
    cache.close()

    runner = CliRunner()
    result = runner.invoke(app, ["cache", "clear-vlm"])

    assert result.exit_code == 0
    assert "Cleared 1 VLM cache entries" in result.output
    assert "invalidated 1 embeddings" in result.output

    cache = FileCache(cache_dir=tmp_path / "cache")
    assert cache.get_vlm_content(file_a, "hash-a", required_version=2) is None

    entry_a = cache.get_by_path(file_a)
    entry_b = cache.get_by_path(file_b)
    assert entry_a is not None
    assert entry_b is not None
    assert entry_a.content_md == "doc-a"
    assert entry_b.content_md == "doc-b"
    assert entry_a.embedding is None
    assert entry_b.embedding is not None
    cache.close()


def test_pdf_check_cli_reports_real_fixture_failure_corpus_truthfully(
    tmp_path: Path, monkeypatch
) -> None:
    fixture_dir = TEST_CORPUS_DIR
    docs = tmp_path / "docs"
    docs.mkdir()
    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    failed_names = [
        "Linux-UNIX系统编程手册.pdf",
        "Rust 程序设计.pdf",
        "The-Art-of-Linear-Algebra-zh-CN.pdf",
        "信息安全数学基础.pdf",
        "抽象代数 - Algebra.pdf",
        "线性代数 中文第5版【Gilbert Strang】.pdf",
        "线性代数及其应用（第六版） - Linear Algebra and Its Applications.pdf",
    ]
    weak_names = [
        "物理考后再练.pdf",
        "线性代数及其应用 (David C. Lay Steven R. Lay Judi J. McDonald) "
        "(Z-Library).pdf",
    ]
    duplicate_names = [
        "2506.12116v3.pdf",
        "2506.12116v3 (1).pdf",
        "2506.12116v3 (2).pdf",
    ]

    for name in [*failed_names, *weak_names, *duplicate_names]:
        shutil.copy2(fixture_dir / name, docs / name)

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
        name = file_path.name
        if name in failed_names:
            final_content = f"{name} recovered by vlm " * 40
            return ResolvedExtraction(
                primary_result=ExtractionResult(
                    content="",
                    success=False,
                    extractor="docling",
                    error="parser failed",
                ),
                primary_effective_length=0,
                pdf_profile=PDFProfile(
                    kind="parser_timeout_or_broken",
                    effective_length=0,
                    glyph_noise_tokens=0,
                    glyph_noise_ratio=0.0,
                    needs_vlm_fallback=True,
                    success=False,
                    reason="parser_timeout_or_broken",
                ),
                fallback_needed=True,
                selected_source="vlm_api",
                final_content=final_content,
                final_effective_length=len(final_content),
                vlm_content=final_content,
                vlm_source="api",
                vlm_api_success=True,
                vlm_api_page_calls=10,
                sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            )

        if name in weak_names:
            primary_content = ("/G25/G26/G27/G28 " * 20) + "题目"
            final_content = f"{name} normalized by vlm " * 30
            return ResolvedExtraction(
                primary_result=ExtractionResult(
                    content=primary_content,
                    success=True,
                    extractor="docling",
                ),
                primary_effective_length=2,
                pdf_profile=PDFProfile(
                    kind="weak_text",
                    effective_length=2,
                    glyph_noise_tokens=20,
                    glyph_noise_ratio=0.9,
                    needs_vlm_fallback=True,
                    success=True,
                    reason="glyph_noise_dominates",
                ),
                fallback_needed=True,
                selected_source="vlm_api",
                final_content=final_content,
                final_effective_length=len(final_content),
                vlm_content=final_content,
                vlm_source="api",
                vlm_api_success=True,
                vlm_api_page_calls=6,
                sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            )

        final_content = f"{name} usable text layer " * 20
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

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = runner.invoke(app, ["pdf-check", str(docs), "--verbose", "--no-cache"])

    assert result.exit_code == 0
    assert "Found 12 PDF files" in result.output
    assert "PDF smoke check completed" in result.output
    assert "primary failures: 7" in result.output
    assert "fallback needed: 9" in result.output
    assert "selected VLM: 9" in result.output
    assert "VLM page calls: 82" in result.output
    assert "duplicates: 2" in result.output
    assert "weak: 0" in result.output
    assert "empty: 0" in result.output
    assert "VLM samples only the first 10 pages." in result.output
    assert "Extraction details" in result.output


def test_pdf_check_cli_uses_final_effective_length_instead_of_truncated_excerpt(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    config_path = _write_test_config(tmp_path, monkeypatch)
    config_path.write_text(
        "\n".join(
            [
                "api:",
                "  base_url: https://api.example.com/v1",
                "  api_key: dummy-key",
                "cache:",
                f"  directory: {tmp_path / 'cache'}",
                "processing:",
                "  text_truncate_limit: 20",
                "  vlm_fallback_threshold: 100",
                "i18n:",
                "  locale: en",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    final_content = "usable final pdf content " * 20

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

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = runner.invoke(app, ["pdf-check", str(docs), "--no-cache"])

    assert result.exit_code == 0
    assert "weak: 0" in result.output
    assert "passed the smoke check" in result.output


def test_pdf_check_verbose_lists_only_problem_files(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    good_pdf = docs / "ok.pdf"
    fallback_pdf = docs / "bad.pdf"
    good_pdf.write_bytes(b"%PDF-1.7 good")
    fallback_pdf.write_bytes(b"%PDF-1.7 fallback")

    _write_test_config(tmp_path, monkeypatch)
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()

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
        if file_path.name == "ok.pdf":
            final_content = "usable primary content " * 20
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

        final_content = "usable fallback content " * 20
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content="",
                success=False,
                extractor="docling",
                error="parser failed",
            ),
            primary_effective_length=0,
            pdf_profile=PDFProfile(
                kind="parser_timeout_or_broken",
                effective_length=0,
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=True,
                success=False,
                reason="parser_timeout_or_broken",
            ),
            fallback_needed=True,
            selected_source="vlm_api",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=final_content,
            vlm_source="api",
            vlm_api_success=True,
            vlm_api_page_calls=4,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        )

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = runner.invoke(app, ["pdf-check", str(docs), "--verbose", "--no-cache"])

    assert result.exit_code == 0
    assert "Extraction details" in result.output
    details_output = result.output.split("Extraction details", maxsplit=1)[1]
    assert "bad.pdf" in details_output
    assert "ok.pdf" not in details_output


def test_pdf_check_verbose_hides_problem_table_when_all_files_are_healthy(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    ok_pdf = docs / "ok.pdf"
    ok_pdf.write_bytes(b"%PDF-1.7 ok")

    _write_test_config(tmp_path, monkeypatch)
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()

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
        final_content = "usable primary content " * 20
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

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = runner.invoke(app, ["pdf-check", str(docs), "--verbose", "--no-cache"])

    assert result.exit_code == 0
    assert "passed the smoke check" in result.output
    assert "Extraction details" not in result.output


def test_pdf_check_cached_vlm_only_disables_vlm_api_in_real_pipeline_path(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "cached.pdf"
    pdf.write_bytes(b"%PDF-1.7 cached")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()
    observed: dict[str, bool] = {}

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
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        observed["allow_vlm_api"] = allow_vlm_api
        final_content = "cached vlm usable output " * 20
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content="",
                success=False,
                extractor="docling",
                error="parser failed",
            ),
            primary_effective_length=0,
            pdf_profile=PDFProfile(
                kind="parser_timeout_or_broken",
                effective_length=0,
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=True,
                success=False,
                reason="parser_timeout_or_broken",
            ),
            fallback_needed=True,
            selected_source="vlm_cache",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=final_content,
            vlm_source="cache",
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        )

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = runner.invoke(
        app, ["pdf-check", str(docs), "--cached-vlm-only", "--no-cache"]
    )

    assert result.exit_code == 0
    assert observed == {"allow_vlm_api": False}
    assert "selected VLM: 1" in result.output
    assert "VLM page calls: 0" in result.output
    assert "passed the smoke check" in result.output


def test_pdf_check_verbose_uses_human_labels_not_internal_codes(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "bad.pdf"
    pdf.write_bytes(b"%PDF-1.7 bad")

    _write_test_config(tmp_path, monkeypatch)
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()

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
        final_content = "usable fallback content " * 20
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content="",
                success=False,
                extractor="docling",
                error="parser failed",
            ),
            primary_effective_length=0,
            pdf_profile=PDFProfile(
                kind="parser_timeout_or_broken",
                effective_length=0,
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=True,
                success=False,
                reason="parser_timeout_or_broken",
            ),
            fallback_needed=True,
            selected_source="vlm_api",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=final_content,
            vlm_source="api",
            vlm_api_success=True,
            vlm_api_page_calls=4,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        )

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = runner.invoke(app, ["pdf-check", str(docs), "--verbose", "--no-cache"])

    assert result.exit_code == 0
    details_output = result.output.split("Extraction details", maxsplit=1)[1]
    assert "Primary parser failed" in details_output
    assert "VLM sampling" in details_output
    assert "parser_timeout_or_broken" not in details_output
    assert "vlm_api" not in details_output


def test_organize_execute_then_undo_cli_restores_files(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "first.txt"
    second = docs / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    target = tmp_path / "organized"

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        return PipelineResult(
            files=[first, second],
            contents=["first", "second"],
            embeddings=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            labels=np.array([0, 0], dtype=int),
            cluster_names={0: "Notes"},
        )

    monkeypatch.setattr("dite.flow.api.PipelineService.run", fake_run)

    organized = runner.invoke(
        app,
        ["organize", str(docs), "--target", str(target), "--execute"],
        input="y\n",
    )
    assert organized.exit_code == 0, organized.output
    assert not first.exists()
    assert "dite undo" in organized.output

    preview = runner.invoke(app, ["undo", str(target), "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert not first.exists()

    undone = runner.invoke(app, ["undo", str(target), "--yes"])
    assert undone.exit_code == 0, undone.output
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"
    assert not (target / "Notes").exists()

    again = runner.invoke(app, ["undo", str(target), "--yes"])
    assert again.exit_code == 1
