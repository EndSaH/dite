"""CLI smoke tests for Phase 0 baseline stability."""

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from dite.app.cli import app
from dite.cache import FileCache
from dite.doc.embed import get_embedding_cache_version
from dite.flow.api import ExtractionFileReport, ExtractionSummary, PipelineResult
from dite.flow.move import OrganizePreview
from dite.util.log import setup_logging


def _write_test_config(tmp_path: Path, monkeypatch, locale: str = "en") -> Path:
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
    return config_path


def test_dite_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_dite_scan_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0


def test_dite_scan_help_with_en_us_locale(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch, locale="en-US")
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0


def test_dite_help_uses_runtime_locale_after_import(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch, locale="en")

    english = runner.invoke(app, ["--help"])
    assert english.exit_code == 0
    assert "Multimodal document intelligent clustering tool" in english.output

    _write_test_config(tmp_path, monkeypatch, locale="zh-CN")
    chinese = runner.invoke(app, ["--help"])

    assert chinese.exit_code == 0
    assert "多模态文件智能聚类工具" in chinese.output


def test_dite_pdf_check_uses_pdf_only_and_stops_before_pipeline_scan(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "doc.pdf"
    txt = docs / "doc.txt"
    pdf.write_text("pdf", encoding="utf-8")
    txt.write_text("txt", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakePipelineService:
        def __init__(self, client, config=None, cache=None):
            calls["cache_enabled"] = cache is not None

        def extract_files(self, files, options):
            calls["files"] = [file.name for file in files]
            calls["use_embedding_cache"] = options.use_embedding_cache
            calls["repair_noise"] = options.repair_noise
            calls["allow_vlm_api"] = options.allow_vlm_api
            return PipelineResult(
                files=files,
                contents=["usable PDF content" * 20],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
            )

    monkeypatch.setattr("dite.app.cli.PipelineService", FakePipelineService)

    result = runner.invoke(app, ["pdf-check", str(docs)])

    assert result.exit_code == 0
    assert calls == {
        "cache_enabled": True,
        "files": ["doc.pdf"],
        "use_embedding_cache": False,
        "repair_noise": False,
        "allow_vlm_api": True,
    }
    assert "passed the smoke check" in result.output


def test_dite_pdf_check_reports_summary_note_and_verbose_details(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    monkeypatch.setenv("COLUMNS", "200")
    setup_logging()
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "doc.pdf"
    pdf.write_text("pdf", encoding="utf-8")

    class FakePipelineService:
        def __init__(self, client, config=None, cache=None):
            del client, config, cache

        def extract_files(self, files, options):
            del options
            return PipelineResult(
                files=files,
                contents=["usable PDF content" * 20],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
                extraction=ExtractionSummary(
                    doc_cache_hits=1,
                    vlm_cache_hits=2,
                    primary_failures=3,
                    source_fallback_needed=4,
                    selected_vlm_files=5,
                    vlm_api_page_calls=6,
                    duplicate_count=7,
                ),
                file_reports=[
                    ExtractionFileReport(
                        file=files[0],
                        primary_extractor="docling",
                        primary_success=False,
                        primary_error="broken",
                        source_profile="parser_timeout_or_broken",
                        source_effective_length=0,
                        selected_source="vlm_api",
                        final_effective_length=2500,
                        excerpt_was_truncated=False,
                        vlm_api_page_calls=6,
                        sample_page_limit=10,
                        file_hash="hash-1",
                        fallback_needed=True,
                    )
                ],
            )

    monkeypatch.setattr("dite.app.cli.PipelineService", FakePipelineService)

    result = runner.invoke(app, ["pdf-check", str(docs), "--verbose"])

    assert result.exit_code == 0
    assert "PDF smoke check completed" in result.output
    assert "doc cache: 1" in result.output
    assert "VLM cache: 2" in result.output
    assert "primary failures: 3" in result.output
    assert "fallback needed: 4" in result.output
    assert "selected VLM: 5" in result.output
    assert "VLM page calls: 6" in result.output
    assert "duplicates: 7" in result.output
    assert "weak: 0" in result.output
    assert "empty: 0" in result.output
    assert "VLM samples only the first 10 pages." in result.output
    assert "Extraction details" in result.output


def test_dite_pdf_check_verbose_suppresses_raw_docling_error_logs(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "doc.pdf"
    pdf.write_text("pdf", encoding="utf-8")

    class FakePipelineService:
        def __init__(self, client, config=None, cache=None):
            del client, config, cache

        def extract_files(self, files, options):
            del options
            logging.getLogger("docling.pipeline.standard_pdf_pipeline").error(
                "raw-docling-noise"
            )
            return PipelineResult(
                files=files,
                contents=["usable PDF content" * 20],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
            )

    monkeypatch.setattr("dite.app.cli.PipelineService", FakePipelineService)

    result = runner.invoke(app, ["pdf-check", str(docs), "--verbose"])

    assert result.exit_code == 0
    assert "raw-docling-noise" not in result.output
    assert "PDF smoke check completed" in result.output


def test_dite_pdf_check_fails_when_final_output_is_below_threshold(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    setup_logging()
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "doc.pdf"
    pdf.write_text("pdf", encoding="utf-8")

    class FakePipelineService:
        def __init__(self, client, config=None, cache=None):
            del client, config, cache

        def extract_files(self, files, options):
            del options
            return PipelineResult(
                files=files,
                contents=["usable PDF content" * 20],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
                extraction=ExtractionSummary(),
                file_reports=[
                    ExtractionFileReport(
                        file=files[0],
                        primary_extractor="docling",
                        primary_success=True,
                        primary_error=None,
                        source_profile="native_text",
                        source_effective_length=2500,
                        selected_source="primary",
                        final_effective_length=0,
                        excerpt_was_truncated=False,
                        vlm_api_page_calls=0,
                        sample_page_limit=10,
                        file_hash="hash-1",
                        source_reason="effective_text_below_threshold",
                    )
                ],
            )

    monkeypatch.setattr("dite.app.cli.PipelineService", FakePipelineService)

    result = runner.invoke(app, ["pdf-check", str(docs)])

    assert result.exit_code == 1
    assert "Weak final PDF outputs" in result.output
    assert "Effective text below threshold" in result.output
    assert "below threshold" in result.output


def test_dite_scan_reports_brief_summary_by_default_and_details_in_verbose(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    setup_logging()
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "doc.pdf"
    pdf.write_text("pdf", encoding="utf-8")

    fake_result = PipelineResult(
        files=[pdf],
        contents=["usable PDF content" * 20],
        embeddings=np.array([[0.1, 0.2, 0.3]]),
        labels=np.array([0]),
        cluster_names={0: "Topic"},
        extraction=ExtractionSummary(
            doc_cache_hits=1,
            vlm_cache_hits=2,
            primary_failures=3,
            source_fallback_needed=4,
            selected_vlm_files=5,
            vlm_api_page_calls=6,
            duplicate_count=7,
        ),
    )

    def fake_run_pipeline_or_exit(pipeline, folder, options, cache):
        del pipeline, folder, options, cache
        return fake_result

    monkeypatch.setattr("dite.app.cli._run_pipeline_or_exit", fake_run_pipeline_or_exit)

    result = runner.invoke(app, ["scan", str(docs)])

    assert result.exit_code == 0
    assert "Content extraction completed" in result.output
    assert "Doc cache: 1" in result.output
    assert "VLM cache: 2" in result.output
    assert "Selected VLM: 5" in result.output
    assert "Duplicates: 7" in result.output
    assert "Extraction details:" not in result.output

    setup_logging()
    verbose_result = runner.invoke(app, ["scan", str(docs), "--verbose"])

    assert verbose_result.exit_code == 0
    assert "Content extraction completed" in verbose_result.output
    assert "DEBUG: Extraction details:" in verbose_result.output
    assert "primary_failures=3" in verbose_result.output
    assert "fallback_needed=4" in verbose_result.output
    assert "vlm_page_calls=6" in verbose_result.output


def test_dite_cluster_ab_command_is_removed(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.txt").write_text("alpha topic notes", encoding="utf-8")

    result = runner.invoke(app, ["cluster-ab", str(docs)])

    assert result.exit_code != 0
    assert "No such command 'cluster-ab'" in result.output


def test_dite_scan_verbose_does_not_leak_internal_profile_codes(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    setup_logging()
    docs = tmp_path / "docs"
    docs.mkdir()
    pdf = docs / "doc.pdf"
    pdf.write_text("pdf", encoding="utf-8")

    fake_result = PipelineResult(
        files=[pdf],
        contents=["usable PDF content" * 20],
        embeddings=np.array([[0.1, 0.2, 0.3]]),
        labels=np.array([0]),
        cluster_names={0: "Topic"},
        extraction=ExtractionSummary(
            primary_failures=1,
            source_fallback_needed=1,
            selected_vlm_files=1,
            vlm_api_page_calls=4,
        ),
        file_reports=[
            ExtractionFileReport(
                file=pdf,
                primary_extractor="docling",
                primary_success=False,
                primary_error="broken",
                source_profile="parser_timeout_or_broken",
                source_effective_length=0,
                selected_source="vlm_api",
                final_effective_length=2500,
                excerpt_was_truncated=False,
                vlm_api_page_calls=4,
                sample_page_limit=10,
                file_hash="hash-1",
            )
        ],
    )

    def fake_run_pipeline_or_exit(pipeline, folder, options, cache):
        del pipeline, folder, options, cache
        return fake_result

    monkeypatch.setattr("dite.app.cli._run_pipeline_or_exit", fake_run_pipeline_or_exit)

    result = runner.invoke(app, ["scan", str(docs), "--verbose"])

    assert result.exit_code == 0
    assert "primary_failures=1" in result.output
    assert "fallback_needed=1" in result.output
    assert "vlm_page_calls=4" in result.output
    assert "parser_timeout_or_broken" not in result.output
    assert "vlm_api" not in result.output


def test_dite_organize_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["organize", "--help"])
    assert result.exit_code == 0


def test_dite_cache_status(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)

    result = runner.invoke(app, ["cache", "status"])
    assert result.exit_code == 0
    assert "Current embedding:" in result.output
    assert "Stale embedding:" in result.output


def test_dite_cache_status_reports_stale_embeddings(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    cache_path = tmp_path / "cache" / "cache.db"
    doc = tmp_path / "doc.txt"
    doc.write_text("payload", encoding="utf-8")
    cache = FileCache(db_path=cache_path)
    cache.save(
        file_path=doc,
        file_hash="hash-old",
        file_mtime=doc.stat().st_mtime,
        content_md="payload",
        embedding=np.array([1.0, 2.0], dtype=np.float32),
        model_version="old-embedding-format",
    )
    cache.close()

    result = runner.invoke(app, ["cache", "status"])

    assert result.exit_code == 0
    assert "With embedding: 1" in result.output
    assert "Current embedding: 0" in result.output
    assert "Stale embedding: 1" in result.output
    assert get_embedding_cache_version("Qwen/Qwen3-Embedding-8B") in result.output


def test_dite_setup_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)

    result = runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0


def test_dite_setup_docling_pdf_runs_downloader(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    target_dir = tmp_path / ".cache" / "dite" / "docling" / "models"

    calls: dict[str, object] = {}

    def fake_download(output_dir: Path, *, force: bool, progress: bool) -> Path:
        calls["output_dir"] = output_dir
        calls["force"] = force
        calls["progress"] = progress
        for name in (
            "docling-project--docling-layout-heron",
            "docling-project--docling-models",
        ):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        return output_dir

    monkeypatch.setattr("dite.app.cli.download_docling_pdf_models", fake_download)

    result = runner.invoke(app, ["setup", "docling-pdf"])

    assert result.exit_code == 0
    assert calls == {
        "output_dir": target_dir,
        "force": False,
        "progress": False,
    }
    assert "Docling PDF models are ready" in result.output


def test_dite_cache_status_creates_global_config_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    assert not config_path.exists()

    runner = CliRunner()
    result = runner.invoke(app, ["cache", "status"])

    assert result.exit_code == 0
    assert config_path.exists()


def test_dite_rejects_removed_config_option(tmp_path: Path, monkeypatch) -> None:
    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["--config", str(tmp_path / "x.yaml"), "--help"])

    assert result.exit_code != 0
    assert "No such option: --config" in result.output


def test_organize_preview_sanitizes_cluster_name(tmp_path: Path) -> None:
    preview = OrganizePreview(source_folder=tmp_path, target_folder=tmp_path / "out")
    preview.add_cluster("finance/reports:*", [tmp_path / "a.pdf"])

    assert "finance-reports-" in preview.cluster_map


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_organize_script_does_not_execute_hostile_names(tmp_path: Path) -> None:
    source_folder = tmp_path / "src"
    target_folder = tmp_path / "out"
    source_folder.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    # The script runs inside marker_dir, so any injected `touch` lands there.
    hostile_names = [
        "report$(touch subst).pdf",
        "notes`touch backtick`.md",
        'quote"; touch dquote; echo ".txt',
        "it's single.txt",
        "-leading-dash.txt",
        "line\ntouch newline\n.txt",
        "tab\there.txt",
    ]
    files = []
    for index, name in enumerate(hostile_names):
        path = source_folder / name
        path.write_text(f"payload {index}", encoding="utf-8")
        files.append(path)

    preview = OrganizePreview(source_folder=source_folder, target_folder=target_folder)
    # The cluster name comes from an LLM that read document content.
    preview.add_cluster("Finance `touch cluster` $(touch cluster2)", files)
    script_path = tmp_path / "organize.sh"
    preview.generate_script(script_path)

    # Uncomment the delete section too, as a user following the script would.
    script = script_path.read_text(encoding="utf-8")
    script = script.replace("\n# rm -- ", "\nrm -- ")
    script_path.write_text(script, encoding="utf-8")

    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=marker_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(marker_dir.iterdir()) == []
    for index, op in enumerate(preview.operations):
        assert op.source == files[index]
        assert op.destination.read_text(encoding="utf-8") == f"payload {index}"
        assert not op.source.exists()


def test_organize_preview_avoids_destination_overwrite(tmp_path: Path) -> None:
    source_folder = tmp_path / "src"
    target_folder = tmp_path / "out"
    source_folder.mkdir()
    target_folder.mkdir()

    first = source_folder / "a" / "report.pdf"
    second = source_folder / "b" / "report.pdf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("1", encoding="utf-8")
    second.write_text("2", encoding="utf-8")

    cluster_dir = target_folder / "cluster"
    cluster_dir.mkdir()
    (cluster_dir / "report.pdf").write_text("existing", encoding="utf-8")

    preview = OrganizePreview(source_folder=source_folder, target_folder=target_folder)
    preview.add_cluster("cluster", [first, second])

    assert preview.operations[0].destination.name == "report_1.pdf"
    assert preview.operations[1].destination.name == "report_2.pdf"


def test_organize_preview_execute_handles_partial_copy_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source_folder = tmp_path / "src"
    target_folder = tmp_path / "out"
    source_folder.mkdir(parents=True)

    ok_file = source_folder / "ok.txt"
    bad_file = source_folder / "bad.txt"
    ok_file.write_text("ok", encoding="utf-8")
    bad_file.write_text("bad", encoding="utf-8")

    preview = OrganizePreview(source_folder=source_folder, target_folder=target_folder)
    preview.add_cluster("reports", [ok_file, bad_file])

    import shutil

    real_copyfileobj = shutil.copyfileobj

    def cross_device_link(*_args, **_kwargs):
        raise OSError(18, "Invalid cross-device link")

    def flaky_copyfileobj(fsrc, fdst, *args, **kwargs):
        if Path(fsrc.name).name == "bad.txt":
            raise OSError("disk full")
        return real_copyfileobj(fsrc, fdst, *args, **kwargs)

    # Force the copy path a move across file systems takes.
    monkeypatch.setattr("dite.flow.move.os.link", cross_device_link)
    monkeypatch.setattr("dite.flow.move.shutil.copyfileobj", flaky_copyfileobj)

    success, failed = preview.execute(dry_run=False)

    assert success == 1
    assert failed == 1
    assert preview.operations[0].status == "success"
    assert preview.operations[1].status == "failed"
    assert preview.operations[0].destination.exists()
    assert not preview.operations[1].destination.exists()
    assert not ok_file.exists()
    assert bad_file.exists()


def _organize_with_journal(tmp_path: Path):
    from dite.flow.move import JOURNAL_DIR_NAME

    source_folder = tmp_path / "src"
    (source_folder / "sub").mkdir(parents=True)
    files = [source_folder / "a.txt", source_folder / "sub" / "b.txt"]
    for index, path in enumerate(files):
        path.write_text(f"payload {index}", encoding="utf-8")
    target_folder = tmp_path / "out"
    preview = OrganizePreview(source_folder=source_folder, target_folder=target_folder)
    preview.add_cluster("Reports", files)
    success, failed = preview.execute(
        dry_run=False, journal_dir=target_folder / JOURNAL_DIR_NAME
    )
    assert (success, failed) == (2, 0)
    return preview, files


def test_execute_writes_journal_and_undo_restores_files(tmp_path: Path) -> None:
    from dite.flow.move import find_latest_journal, read_journal, undo_journal

    preview, files = _organize_with_journal(tmp_path)
    assert preview.journal_path is not None
    assert find_latest_journal(preview.target_folder) == preview.journal_path
    contents = read_journal(preview.journal_path)
    assert [source for source, _ in contents.moves] == files
    assert not any(path.exists() for path in files)

    result = undo_journal(preview.journal_path)

    assert len(result.restored) == 2
    assert result.skipped == []
    for index, path in enumerate(files):
        assert path.read_text(encoding="utf-8") == f"payload {index}"
    # The cluster folder the run created is gone; the journal is kept.
    assert not (preview.target_folder / "Reports").exists()
    assert preview.journal_path.exists()
    assert read_journal(preview.journal_path).undone is True
    assert find_latest_journal(preview.target_folder) is None
    with pytest.raises(ValueError, match="already undone"):
        undo_journal(preview.journal_path)


def test_undo_never_overwrites_and_reports_missing_files(tmp_path: Path) -> None:
    from dite.flow.move import undo_journal

    preview, files = _organize_with_journal(tmp_path)
    moved_a, moved_b = (op.destination for op in preview.operations)
    files[0].write_text("new file at the old path", encoding="utf-8")
    moved_b.unlink()

    plan = undo_journal(preview.journal_path, dry_run=True)
    assert plan.restored == []
    assert sorted(reason for _, reason in plan.skipped) == ["conflict", "missing"]
    assert moved_a.exists()

    undo_journal(preview.journal_path)

    assert files[0].read_text(encoding="utf-8") == "new file at the old path"
    assert moved_a.read_text(encoding="utf-8") == "payload 0"
    # A non-empty cluster folder is left in place.
    assert (preview.target_folder / "Reports").exists()


def test_undo_round_trips_non_utf8_and_special_names(tmp_path: Path) -> None:
    import os

    from dite.flow.move import JOURNAL_DIR_NAME, undo_journal

    source_folder = tmp_path / "src"
    source_folder.mkdir()
    names = ["报告 [草稿] $(x).txt", "it's.txt"]
    if os.name == "posix":
        names.append(os.fsdecode(b"latin1-\xe9.txt"))
    files = []
    for name in names:
        path = source_folder / name
        path.write_text(name, encoding="utf-8", errors="surrogateescape")
        files.append(path)
    preview = OrganizePreview(source_folder=source_folder, target_folder=tmp_path / "o")
    preview.add_cluster("Mixed", files)
    preview.execute(dry_run=False, journal_dir=tmp_path / "o" / JOURNAL_DIR_NAME)

    result = undo_journal(preview.journal_path)

    assert len(result.restored) == len(files)
    assert all(path.exists() for path in files)


def test_empty_execute_writes_no_journal_and_does_not_hide_older_ones(
    tmp_path: Path,
) -> None:
    from dite.flow.move import JOURNAL_DIR_NAME, MoveJournal, find_latest_journal

    preview, _files = _organize_with_journal(tmp_path)
    real_journal = preview.journal_path

    # A later run with nothing to move (e.g. assign found no matches).
    empty_run = OrganizePreview(
        source_folder=tmp_path / "src", target_folder=preview.target_folder
    )
    assert empty_run.execute(
        dry_run=False, journal_dir=preview.target_folder / JOURNAL_DIR_NAME
    ) == (0, 0)
    assert empty_run.journal_path is None

    # Even a journal that recorded no successful move is skipped.
    MoveJournal.create(
        preview.target_folder / JOURNAL_DIR_NAME,
        source_folder=tmp_path / "src",
        target_folder=preview.target_folder,
    ).close()

    assert find_latest_journal(preview.target_folder) == real_journal
