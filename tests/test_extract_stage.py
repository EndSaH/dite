"""Per-file failure isolation and Ctrl+C handling in the extraction stage."""

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dite.app.cli import app
from dite.app.config import Config
from dite.app.i18n import set_locale
from dite.cache import FileCache
from dite.flow.api import PipelineOptions, PipelineService
from dite.flow.extract import extract_stage
from dite.flow.model import (
    ExtractionFileReport,
    ExtractionSummaryDelta,
    ExtractionWorkResult,
)
from dite.io.base import ExtractionResult
from dite.io.pdf.final import PDFCacheWriteIntent
from dite.io.pdf.render import PDFRenderResult
from dite.io.pdf.vlm import extract_pdf_with_vlm_sampling
from dite.io.types import ResolvedExtraction
from dite.util.log import get_logger, setup_logging


def _write_files(tmp_path: Path, names: list[str]) -> list[Path]:
    files = []
    for name in names:
        path = tmp_path / name
        path.write_text(f"body of {name}", encoding="utf-8")
        files.append(path)
    return files


def _ok_result(item) -> ExtractionWorkResult:
    return ExtractionWorkResult(
        index=item.index,
        content=f"content::{item.file.name}",
        file_hash=item.file_hash,
        report=ExtractionFileReport(
            file=item.file,
            primary_extractor="text",
            primary_success=True,
            primary_error=None,
            source_profile=None,
            source_effective_length=10,
            selected_source="primary",
            final_effective_length=10,
            excerpt_was_truncated=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
            file_hash=item.file_hash,
        ),
        summary_delta=ExtractionSummaryDelta(),
    )


def _run_stage(files: list[Path], work_item_runner, *, workers: int = 2):
    config = Config()
    config.processing.extract_workers = workers
    return extract_stage(
        files=files,
        options=PipelineOptions(use_cache=False, use_embedding_cache=False),
        config=config,
        client=None,
        cache=None,
        logger=get_logger(),
        request_runtime=None,
        make_extractor_registry=None,
        extract_primary_result=None,
        build_cached_primary_result=None,
        supports_keyword_arg=None,
        resolve_document_extraction=None,
        work_item_runner=work_item_runner,
    )


def _extract_via_extract_files(service: PipelineService, files, options):
    result = service.extract_files(files, options)
    return result.contents, result.file_reports, result.extraction


def _extract_via_extract_contents(service: PipelineService, files, options):
    contents, _hashes, summary, reports = service._extract_contents(files, options)
    return contents, reports, summary


# ``extract_files`` (pdf-check) and ``_extract_contents`` (scan, organize,
# assign) run different work-item implementations through the same stage.
_ENTRY_POINTS = pytest.mark.parametrize(
    "extract",
    [_extract_via_extract_files, _extract_via_extract_contents],
    ids=["extract_files", "extract_contents"],
)


def test_extract_stage_records_a_raising_file_as_failed_and_keeps_going(
    tmp_path: Path, capsys
) -> None:
    set_locale("en")
    setup_logging()
    files = _write_files(tmp_path, ["a.txt", "b.txt", "c.txt"])

    def runner(item, options, truncate_limit, semaphore):
        del options, truncate_limit, semaphore
        if item.file.name == "b.txt":
            raise OSError("disk went away")
        return _ok_result(item)

    stage = _run_stage(files, runner)

    assert stage.contents == ["content::a.txt", "", "content::c.txt"]
    failed = stage.file_reports[1]
    assert failed.file == files[1]
    assert failed.primary_success is False
    assert failed.primary_error == "OSError: disk went away"
    assert failed.final_effective_length == 0
    assert [report.primary_success for report in stage.file_reports] == [
        True,
        False,
        True,
    ]
    assert stage.summary.primary_failures == 1
    output = capsys.readouterr().out
    assert "b.txt" in output
    assert "disk went away" in output


@_ENTRY_POINTS
def test_pipeline_keeps_going_when_an_extractor_raises(
    tmp_path: Path, monkeypatch, extract
) -> None:
    set_locale("en")
    setup_logging()
    files = _write_files(tmp_path, ["a.txt", "b.txt", "c.txt"])

    def fake_extract_document(file_path, client, registry=None, config=None):
        del client, registry, config
        if file_path.name == "b.txt":
            raise PermissionError(13, "Permission denied", str(file_path))
        return ExtractionResult(
            content=f"{file_path.stem} content long enough",
            success=True,
            extractor="text",
        )

    monkeypatch.setattr("dite.io.route.extract_document", fake_extract_document)
    service = PipelineService(client=object(), config=Config(), cache=None)

    contents, reports, summary = extract(
        service,
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert contents == ["a content long enough", "", "c content long enough"]
    assert reports[1].primary_success is False
    assert reports[1].primary_error.startswith("PermissionError: ")
    assert summary.primary_failures == 1


@_ENTRY_POINTS
def test_pipeline_records_unreadable_files_as_failed_without_extracting_them(
    tmp_path: Path, monkeypatch, extract
) -> None:
    set_locale("en")
    setup_logging()
    readable = _write_files(tmp_path, ["a.txt", "c.txt"])
    # Listed by the scan, then deleted before extraction reads them.
    gone = [tmp_path / "gone-1.txt", tmp_path / "gone-2.txt"]
    files = [readable[0], gone[0], gone[1], readable[1]]
    extracted: list[str] = []

    def fake_extract_document(file_path, client, registry=None, config=None):
        del client, registry, config
        extracted.append(file_path.name)
        return ExtractionResult(
            content=f"{file_path.stem} content long enough",
            success=True,
            extractor="text",
        )

    monkeypatch.setattr("dite.io.route.extract_document", fake_extract_document)
    service = PipelineService(client=object(), config=Config(), cache=None)

    contents, reports, summary = extract(
        service,
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert sorted(extracted) == ["a.txt", "c.txt"]
    assert contents == ["a content long enough", "", "", "c content long enough"]
    for report in reports[1:3]:
        assert report.primary_success is False
        assert report.primary_error.startswith("FileNotFoundError: ")
    # Two unreadable files are not duplicates of each other.
    assert reports[1].file_hash != reports[2].file_hash
    assert summary.duplicate_count == 0
    assert summary.primary_failures == 2


_CACHE_CALL_ORDER = ["get_content", "get_vlm_content", "save", "update_vlm_content"]


def _fake_pdf_primary_result(self, file_path, registry, docling_pdf_semaphore):
    del self, registry, docling_pdf_semaphore
    return ExtractionResult(
        content=f"{file_path.stem} primary text",
        success=True,
        extractor="docling",
    )


def _fake_resolve_with_vlm_cache_write(file_path, client, **kwargs):
    """A weak PDF whose fresh VLM text is selected and should be cached."""
    del client
    primary = kwargs["primary_result"]
    vlm_text = f"{file_path.stem} vlm text"
    return ResolvedExtraction(
        primary_result=primary,
        primary_effective_length=len(primary.content),
        pdf_profile=None,
        fallback_needed=True,
        selected_source="vlm_api",
        final_content=vlm_text,
        final_effective_length=len(vlm_text),
        vlm_content=vlm_text,
        vlm_source="api",
        vlm_api_success=True,
        vlm_api_page_calls=1,
        sample_page_limit=10,
        cache_write_intent=PDFCacheWriteIntent(should_write=True, content=vlm_text),
    )


@pytest.mark.parametrize("failing", _CACHE_CALL_ORDER)
@_ENTRY_POINTS
def test_pipeline_keeps_extracted_content_when_the_cache_fails(
    tmp_path: Path, monkeypatch, capsys, extract, failing: str
) -> None:
    set_locale("en")
    setup_logging()
    files = []
    for name in ["a.pdf", "b.pdf"]:
        path = tmp_path / name
        path.write_bytes(f"%PDF-1.7 {name}".encode())
        files.append(path)
    cache = FileCache(db_path=tmp_path / "cache" / "cache.db")
    calls: list[str] = []
    for method in _CACHE_CALL_ORDER:
        real = getattr(cache, method)

        def call(*args, _method=method, _real=real, **kwargs):
            calls.append(_method)
            if _method == failing:
                raise sqlite3.OperationalError("database is locked")
            return _real(*args, **kwargs)

        monkeypatch.setattr(cache, method, call)
    monkeypatch.setattr(
        PipelineService, "_extract_primary_result", _fake_pdf_primary_result
    )
    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        _fake_resolve_with_vlm_cache_write,
    )
    service = PipelineService(client=object(), config=Config(), cache=cache)

    try:
        contents, reports, summary = extract(
            service,
            files,
            PipelineOptions(use_cache=True, use_embedding_cache=False),
        )
    finally:
        cache.close()

    # A cache error costs only the speed-up: the extracted text is kept.
    assert contents == ["a vlm text", "b vlm text"]
    assert [report.primary_success for report in reports] == [True, True]
    assert [report.primary_error for report in reports] == [None, None]
    assert summary.primary_failures == 0
    # After its first cache error a file leaves the cache alone.
    skipped = _CACHE_CALL_ORDER[_CACHE_CALL_ORDER.index(failing) + 1 :]
    assert calls.count(failing) == 2
    assert not set(calls) & set(skipped), calls
    output = " ".join(capsys.readouterr().out.split())
    assert output.count("Cache read/write failed for") == 2
    assert output.count("OperationalError: database is locked") == 2


@_ENTRY_POINTS
def test_pipeline_keeps_the_content_of_a_file_deleted_after_extraction(
    tmp_path: Path, monkeypatch, capsys, extract
) -> None:
    set_locale("en")
    setup_logging()
    files = _write_files(tmp_path, ["a.txt", "b.txt"])

    def fake_extract_document(file_path, client, registry=None, config=None):
        del client, registry, config
        result = ExtractionResult(
            content=f"{file_path.stem} content long enough",
            success=True,
            extractor="text",
        )
        if file_path.name == "b.txt":
            # Gone before its cache row, which needs the mtime, is written.
            file_path.unlink()
        return result

    monkeypatch.setattr("dite.io.route.extract_document", fake_extract_document)
    cache = FileCache(db_path=tmp_path / "cache" / "cache.db")
    service = PipelineService(client=object(), config=Config(), cache=cache)

    try:
        contents, reports, summary = extract(
            service,
            files,
            PipelineOptions(use_cache=True, use_embedding_cache=False),
        )
        cached_a, _source = cache.get_content(files[0], reports[0].file_hash)
    finally:
        cache.close()

    assert contents == ["a content long enough", "b content long enough"]
    assert [report.primary_success for report in reports] == [True, True]
    assert summary.primary_failures == 0
    assert cached_a == "a content long enough"
    output = " ".join(capsys.readouterr().out.split())
    assert output.count("Cache read/write failed for") == 1
    assert "FileNotFoundError" in output


def test_extract_stage_stops_starting_files_after_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    files = _write_files(tmp_path, [f"{index:02d}.txt" for index in range(20)])
    processed: list[str] = []
    lock = threading.Lock()

    def runner(item, options, truncate_limit, semaphore):
        del options, truncate_limit, semaphore
        with lock:
            processed.append(item.file.name)
        if item.file.name == "03.txt":
            raise KeyboardInterrupt
        time.sleep(0.05)
        return _ok_result(item)

    with pytest.raises(KeyboardInterrupt):
        _run_stage(files, runner, workers=2)

    # At most the two files already running when 03.txt was interrupted, plus
    # the one that replaced a finished file; never the rest of the folder.
    assert len(processed) <= 5, processed
    assert "19.txt" not in processed


def test_extract_stage_tells_running_files_to_stop_after_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    from dite.util.cancel import cancellation_requested

    files = _write_files(tmp_path, ["slow.txt", "boom.txt"])
    slow_started = threading.Event()
    outcome: dict[str, bool] = {}

    def runner(item, options, truncate_limit, semaphore):
        del options, truncate_limit, semaphore
        if item.file.name == "boom.txt":
            slow_started.wait(5)
            raise KeyboardInterrupt
        slow_started.set()
        deadline = time.monotonic() + 5
        while not cancellation_requested() and time.monotonic() < deadline:
            time.sleep(0.01)
        outcome["saw_cancellation"] = cancellation_requested()
        return _ok_result(item)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run_stage(files, runner, workers=2)

    assert outcome == {"saw_cancellation": True}
    assert time.monotonic() - started < 3


class _Image:
    width = 800
    height = 600

    def save(self, buffer, _fmt):
        buffer.write(b"png")


class _RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def run_vlm_page_batch(self, requests, *, per_document_limit):
        self.calls.append((requests, per_document_limit))
        return []


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        client = self

        class _Completions:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                raise AssertionError("no VLM request after cancellation")

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


@pytest.mark.parametrize("use_runtime", [True, False], ids=["runtime", "sync"])
def test_pdf_vlm_sampling_sends_no_requests_once_cancelled(
    tmp_path: Path, monkeypatch, use_runtime: bool
) -> None:
    from dite.util.cancel import ExtractionCancelledError, cancellation_scope

    set_locale("en")
    setup_logging()
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(
        "dite.io.pdf.vlm.render_pdf_pages",
        lambda path, *, max_pages: PDFRenderResult(
            pages=[_Image(), _Image()],
            success=True,
            error=None,
            sample_page_limit=max_pages,
        ),
    )
    runtime = _RecordingRuntime()
    client = _RecordingClient()
    cancelled = threading.Event()
    cancelled.set()

    with cancellation_scope(cancelled), pytest.raises(ExtractionCancelledError):
        extract_pdf_with_vlm_sampling(
            file_path,
            client,
            config=Config(),
            request_runtime=runtime if use_runtime else None,
        )

    assert runtime.calls == []
    assert client.calls == []


def _write_test_config(tmp_path: Path, monkeypatch, locale: str) -> None:
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
                f"  locale: {locale}",
            ]
        ),
        encoding="utf-8",
    )


class _InterruptedPipelineService:
    def __init__(self, client, config=None, cache=None) -> None:
        del client, config, cache

    def run(self, folder, options):
        del folder, options
        raise KeyboardInterrupt

    def extract_files(self, files, options):
        del files, options
        raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("locale", "message"),
    [
        ("en", "Interrupted; the run was stopped."),
        ("zh-CN", "已中断，本次运行已停止。"),
    ],
)
@pytest.mark.parametrize("command", ["scan", "pdf-check"])
def test_cli_exits_130_with_a_message_on_ctrl_c(
    tmp_path: Path, monkeypatch, command: str, locale: str, message: str
) -> None:
    _write_test_config(tmp_path, monkeypatch, locale)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.pdf").write_bytes(b"%PDF-1.7")
    monkeypatch.setattr("dite.app.cli.PipelineService", _InterruptedPipelineService)

    result = CliRunner().invoke(app, [command, str(docs)])

    assert result.exit_code == 130, result.output
    assert message in result.output
