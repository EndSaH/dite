"""Per-file failure isolation in the extraction stage."""

from pathlib import Path

import pytest

from dite.app.config import Config
from dite.app.i18n import set_locale
from dite.flow.api import PipelineOptions, PipelineService
from dite.flow.extract import extract_stage
from dite.flow.model import (
    ExtractionFileReport,
    ExtractionSummaryDelta,
    ExtractionWorkResult,
)
from dite.io.base import ExtractionResult
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
