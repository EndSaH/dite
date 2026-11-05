import multiprocessing
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import docling_child_helpers
import pytest

from dite.app.i18n import set_locale
from dite.io.base import ExtractionResult
from dite.io.docling import (
    DoclingExtractor,
    _docling_pdf_extract_child,
    _get_required_pdf_artifact_dirs,
    download_docling_pdf_models,
    extract_docling_pdf_in_subprocess,
    get_docling_pdf_artifacts_path,
    has_docling_pdf_artifacts,
)


def test_get_docling_pdf_artifacts_path_uses_dite_cache_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_docling_pdf_artifacts_path() == (
        tmp_path / ".cache" / "dite" / "docling" / "models"
    )


def test_has_docling_pdf_artifacts_requires_layout_and_tableformer(
    tmp_path: Path,
) -> None:
    required_dirs = _get_required_pdf_artifact_dirs()

    assert has_docling_pdf_artifacts(tmp_path) is False

    for name in required_dirs:
        (tmp_path / name).mkdir(parents=True)

    assert has_docling_pdf_artifacts(tmp_path) is True


def test_download_docling_pdf_models_downloads_minimal_set(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_download_models(output_dir: Path, **kwargs) -> None:
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "docling.utils.model_downloader.download_models",
        fake_download_models,
    )

    result = download_docling_pdf_models(tmp_path, force=True, progress=True)

    assert result == tmp_path
    assert captured["output_dir"] == tmp_path
    assert captured["kwargs"] == {
        "force": True,
        "progress": True,
        "with_layout": True,
        "with_tableformer": True,
        "with_code_formula": False,
        "with_picture_classifier": False,
        "with_smolvlm": False,
        "with_granitedocling": False,
        "with_granitedocling_mlx": False,
        "with_smoldocling": False,
        "with_smoldocling_mlx": False,
        "with_granite_vision": False,
        "with_rapidocr": False,
        "with_easyocr": False,
    }


def test_docling_pdf_returns_clear_error_when_models_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    result = DoclingExtractor().extract(pdf_path)

    assert result.success is False
    assert result.extractor == "docling"
    assert (
        result.error == "Docling PDF models are not installed. Run "
        "`uv run dite setup docling-pdf` first."
    )


def test_docling_pdf_times_out(tmp_path: Path, monkeypatch) -> None:
    set_locale("en")
    pdf_path = tmp_path / "slow.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    class _SlowConverter:
        def convert(self, _file_path: str):
            time.sleep(1)
            raise AssertionError("timeout did not interrupt conversion")

    monkeypatch.setattr(
        "dite.io.docling.has_docling_pdf_artifacts",
        lambda _artifacts_path: True,
    )
    monkeypatch.setattr(
        DoclingExtractor,
        "_get_converter",
        lambda _self: _SlowConverter(),
    )

    result = DoclingExtractor(pdf_timeout_sec=0.01).extract(pdf_path)

    assert result.success is False
    assert result.extractor == "docling"
    assert result.error == "Docling PDF extraction timed out after 0.01s"


def test_docling_extractor_passes_configured_device_to_accelerator_options(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeAcceleratorOptions:
        def __init__(self, *, device: str) -> None:
            captured["device"] = device

    class _FakePdfPipelineOptions:
        def __init__(self) -> None:
            self.do_ocr = True
            self.artifacts_path = None
            self.accelerator_options = None

    class _FakePdfFormatOption:
        def __init__(self, *, pipeline_options) -> None:
            captured["pipeline_options"] = pipeline_options

    class _FakeDocumentConverter:
        def __init__(self, *, format_options) -> None:
            captured["format_options"] = format_options

    monkeypatch.setattr(
        "docling.datamodel.accelerator_options.AcceleratorOptions",
        _FakeAcceleratorOptions,
    )
    monkeypatch.setattr(
        "docling.datamodel.pipeline_options.PdfPipelineOptions",
        _FakePdfPipelineOptions,
    )
    monkeypatch.setattr(
        "docling.document_converter.PdfFormatOption",
        _FakePdfFormatOption,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _FakeDocumentConverter,
    )

    extractor = DoclingExtractor(
        artifacts_path=tmp_path,
        pdf_timeout_sec=60.0,
        device="cpu",
    )
    converter = extractor._get_converter()

    assert isinstance(converter, _FakeDocumentConverter)
    assert captured["device"] == "cpu"
    pipeline_options = captured["pipeline_options"]
    assert pipeline_options.do_ocr is False
    assert pipeline_options.artifacts_path == tmp_path
    assert pipeline_options.accelerator_options is not None


def test_extract_docling_pdf_in_subprocess_returns_child_result(
    tmp_path: Path, monkeypatch
) -> None:
    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    expected = ExtractionResult(
        content="docling child result",
        success=True,
        extractor="docling",
    )
    captured: dict[str, object] = {}

    class _Conn:
        def __init__(self, result: ExtractionResult | None) -> None:
            self._result = result
            self.closed = False

        def poll(self, timeout=None) -> bool:
            del timeout
            return self._result is not None

        def recv(self) -> ExtractionResult:
            assert self._result is not None
            return self._result

        def close(self) -> None:
            self.closed = True

    class _Process:
        exitcode = 0

        def __init__(self, target, args) -> None:
            captured["target"] = target
            captured["args"] = args
            self._alive = False

        def start(self) -> None:
            captured["started"] = True

        def join(self, timeout=None) -> None:
            captured.setdefault("join_calls", []).append(timeout)

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            raise AssertionError("terminate should not run in success path")

        def kill(self) -> None:
            raise AssertionError("kill should not run in success path")

    class _Context:
        def Pipe(self, duplex=False):
            assert duplex is False
            return _Conn(expected), _Conn(None)

        def Process(self, target, args):
            return _Process(target, args)

    monkeypatch.setattr(
        "dite.io.docling.multiprocessing.get_context",
        lambda method: _Context(),
    )

    result = extract_docling_pdf_in_subprocess(
        pdf_path,
        timeout_sec=0.5,
        locale="en",
    )

    assert result == expected
    assert captured["started"] is True
    # The child is reaped after its result has been read.
    assert captured["join_calls"]


def test_docling_pdf_extract_child_silences_docling_logging_before_extract(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    class _Conn:
        def __init__(self) -> None:
            self.payload = None
            self.closed = False

        def send(self, payload) -> None:
            self.payload = payload

        def close(self) -> None:
            self.closed = True

    def fake_silence_docling_logging() -> None:
        events.append("silence")

    class _FakeDoclingExtractor:
        def __init__(self, **kwargs) -> None:
            del kwargs
            events.append("init")

        def extract(self, file_path: Path) -> ExtractionResult:
            events.append(f"extract:{file_path.name}")
            return ExtractionResult(
                content="child result",
                success=True,
                extractor="docling",
            )

    monkeypatch.setattr(
        "dite.io.docling.silence_docling_logging",
        fake_silence_docling_logging,
    )
    monkeypatch.setattr(
        "dite.io.docling.DoclingExtractor",
        _FakeDoclingExtractor,
    )

    conn = _Conn()
    _docling_pdf_extract_child(
        str(pdf_path),
        False,
        None,
        "en",
        "auto",
        conn,
    )

    assert events == ["silence", "init", f"extract:{pdf_path.name}"]
    assert conn.closed is True
    assert conn.payload == ExtractionResult(
        content="child result",
        success=True,
        extractor="docling",
    )


def test_extract_docling_pdf_in_subprocess_times_out_and_terminates(
    tmp_path: Path, monkeypatch
) -> None:
    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    lifecycle: list[str] = []

    class _Conn:
        def poll(self, timeout=None) -> bool:
            del timeout
            return False

        def recv(self) -> ExtractionResult:
            raise AssertionError("recv should not run on timeout")

        def close(self) -> None:
            lifecycle.append("close_conn")

    class _Process:
        exitcode = None

        def __init__(self, target, args) -> None:
            del target, args
            self._alive = True

        def start(self) -> None:
            lifecycle.append("start")

        def join(self, timeout=None) -> None:
            lifecycle.append(f"join:{timeout}")

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            lifecycle.append("terminate")
            self._alive = False

        def kill(self) -> None:
            lifecycle.append("kill")

    class _Context:
        def Pipe(self, duplex=False):
            assert duplex is False
            return _Conn(), _Conn()

        def Process(self, target, args):
            return _Process(target, args)

    monkeypatch.setattr(
        "dite.io.docling.multiprocessing.get_context",
        lambda method: _Context(),
    )

    result = extract_docling_pdf_in_subprocess(
        pdf_path,
        timeout_sec=0.25,
        locale="en",
    )

    assert result.success is False
    assert result.extractor == "docling"
    assert result.error == "Docling PDF extraction timed out after 0.25s"
    assert "terminate" in lifecycle


class _RecordingConverter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def convert(self, source, **kwargs):
        self.calls.append((source, kwargs))

        class _Document:
            def export_to_markdown(self) -> str:
                return "# Title\nBody"

        return type("_Result", (), {"document": _Document()})()


def _extract_with_recording_converter(
    tmp_path: Path, monkeypatch, file_name: str, **extractor_kwargs
):
    path = tmp_path / file_name
    path.write_bytes(b"%PDF-1.7" if file_name.endswith(".pdf") else b"PK")
    monkeypatch.setattr(
        "dite.io.docling.has_docling_pdf_artifacts", lambda *args, **kwargs: True
    )
    converter = _RecordingConverter()
    extractor = DoclingExtractor(pdf_timeout_sec=None, **extractor_kwargs)
    monkeypatch.setattr(extractor, "_get_converter", lambda: converter)
    result = extractor.extract(path)
    return result, converter.calls


def test_docling_pdf_conversion_is_limited_to_the_first_pages(
    tmp_path: Path, monkeypatch
) -> None:
    result, calls = _extract_with_recording_converter(
        tmp_path, monkeypatch, "book.pdf", pdf_max_pages=12
    )

    assert result.success is True
    # page_range limits conversion; max_num_pages would reject long PDFs.
    assert calls[0][1] == {"page_range": (1, 12)}


def test_docling_page_limit_can_be_disabled_and_skips_non_pdf(
    tmp_path: Path, monkeypatch
) -> None:
    _result, unlimited = _extract_with_recording_converter(
        tmp_path, monkeypatch, "book.pdf", pdf_max_pages=None
    )
    _result, slides = _extract_with_recording_converter(
        tmp_path, monkeypatch, "deck.pptx", pdf_max_pages=12
    )

    assert unlimited[0][1] == {}
    assert slides[0][1] == {}


def test_docling_child_process_receives_page_limit(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Conn:
        def send(self, payload) -> None:
            captured["payload"] = payload

        def close(self) -> None:
            pass

    class _FakeDoclingExtractor:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def extract(self, file_path: Path) -> ExtractionResult:
            return ExtractionResult(content="ok", success=True, extractor="docling")

    monkeypatch.setattr("dite.io.docling.silence_docling_logging", lambda: None)
    monkeypatch.setattr("dite.io.docling.DoclingExtractor", _FakeDoclingExtractor)

    _docling_pdf_extract_child(
        str(tmp_path / "a.pdf"), False, None, "en", "auto", _Conn(), 25
    )

    assert captured["kwargs"]["pdf_max_pages"] == 25


def test_pipeline_passes_page_limit_to_docling_subprocess(monkeypatch) -> None:
    import threading

    from dite.app.config import Config
    from dite.flow.api import PipelineService

    captured: dict[str, object] = {}

    def _fake_subprocess(file_path, **kwargs):
        captured.update(kwargs)
        return ExtractionResult(content="ok", success=True, extractor="docling")

    monkeypatch.setattr(
        "dite.flow.api.extract_docling_pdf_in_subprocess", _fake_subprocess
    )
    config = Config()
    service = PipelineService(client=object(), config=config, cache=None)
    extractor = service._make_extractor_registry().get_docling()

    service._extract_docling_pdf_primary_result(
        Path("book.pdf"), extractor, threading.BoundedSemaphore(1)
    )

    assert captured["max_pages"] == config.processing.docling_pdf_max_pages == 50


def _extract_with_real_child(
    tmp_path: Path, monkeypatch, child, *, timeout_sec: float
) -> tuple[ExtractionResult, float]:
    """Run the real spawn-based wrapper with ``child`` as the child body."""
    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr("dite.io.docling._docling_pdf_extract_child", child)
    started = time.monotonic()
    result = extract_docling_pdf_in_subprocess(
        pdf_path,
        timeout_sec=timeout_sec,
        locale="en",
    )
    return result, time.monotonic() - started


def test_extract_docling_pdf_in_subprocess_reads_results_larger_than_a_pipe(
    tmp_path: Path, monkeypatch
) -> None:
    children_before = set(multiprocessing.active_children())

    result, elapsed = _extract_with_real_child(
        tmp_path,
        monkeypatch,
        docling_child_helpers.send_large_result,
        timeout_sec=20,
    )

    assert result.success is True, result.error
    assert len(result.content) == docling_child_helpers.LARGE_CONTENT_SIZE
    # The child sends at once; running into the timeout means the parent
    # waited for an exit while the child was blocked on a full pipe.
    assert elapsed < 10
    assert set(multiprocessing.active_children()) <= children_before


def test_extract_docling_pdf_in_subprocess_reports_a_killed_child_as_failure(
    tmp_path: Path, monkeypatch
) -> None:
    children_before = set(multiprocessing.active_children())

    result, elapsed = _extract_with_real_child(
        tmp_path,
        monkeypatch,
        docling_child_helpers.die_without_result,
        timeout_sec=20,
    )

    assert result.success is False
    assert result.content == ""
    assert result.extractor == "docling"
    assert result.error == (
        "Docling PDF subprocess exited without a result (exit code -9, SIGKILL)"
    )
    assert elapsed < 10
    assert set(multiprocessing.active_children()) <= children_before


def test_extract_docling_pdf_in_subprocess_terminates_a_child_that_overruns(
    tmp_path: Path, monkeypatch
) -> None:
    children_before = set(multiprocessing.active_children())

    result, elapsed = _extract_with_real_child(
        tmp_path,
        monkeypatch,
        docling_child_helpers.sleep_forever,
        timeout_sec=1,
    )

    assert result.success is False
    assert result.error == "Docling PDF extraction timed out after 1s"
    assert elapsed < 10
    assert set(multiprocessing.active_children()) <= children_before


def test_extract_docling_pdf_in_subprocess_stops_the_child_when_cancelled(
    tmp_path: Path, monkeypatch
) -> None:
    from dite.util.cancel import ExtractionCancelledError, cancellation_scope

    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(
        "dite.io.docling._docling_pdf_extract_child",
        docling_child_helpers.sleep_forever,
    )
    children_before = set(multiprocessing.active_children())
    cancelled = threading.Event()
    timer = threading.Timer(0.5, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with cancellation_scope(cancelled), pytest.raises(ExtractionCancelledError):
            extract_docling_pdf_in_subprocess(pdf_path, timeout_sec=30, locale="en")
    finally:
        timer.cancel()

    assert time.monotonic() - started < 10
    assert set(multiprocessing.active_children()) <= children_before


def test_extract_docling_pdf_in_subprocess_starts_no_child_once_cancelled(
    tmp_path: Path, monkeypatch
) -> None:
    from dite.util.cancel import ExtractionCancelledError, cancellation_scope

    def _no_context(method):
        raise AssertionError(f"no {method} subprocess after cancellation")

    # Replace the module docling.py sees, not the stdlib's own get_context.
    monkeypatch.setattr(
        "dite.io.docling.multiprocessing", SimpleNamespace(get_context=_no_context)
    )
    cancelled = threading.Event()
    cancelled.set()

    with cancellation_scope(cancelled), pytest.raises(ExtractionCancelledError):
        extract_docling_pdf_in_subprocess(
            tmp_path / "paper.pdf", timeout_sec=30, locale="en"
        )


def test_extract_docling_pdf_in_subprocess_closes_the_pipe_if_the_child_fails_to_start(
    tmp_path: Path, monkeypatch
) -> None:
    class _Conn:
        closed = False

        def close(self) -> None:
            self.closed = True

    class _Process:
        def __init__(self, target, args) -> None:
            del target, args

        def start(self) -> None:
            raise OSError(24, "Too many open files")

        def is_alive(self) -> bool:
            return False

    pipe = (_Conn(), _Conn())
    context = SimpleNamespace(Pipe=lambda duplex: pipe, Process=_Process)
    monkeypatch.setattr(
        "dite.io.docling.multiprocessing",
        SimpleNamespace(get_context=lambda method: context),
    )

    with pytest.raises(OSError, match="Too many open files"):
        extract_docling_pdf_in_subprocess(
            tmp_path / "paper.pdf", timeout_sec=30, locale="en"
        )

    assert [conn.closed for conn in pipe] == [True, True]


def test_docling_pdf_child_leaves_ctrl_c_to_the_parent(monkeypatch) -> None:
    from dite.io.docling import _run_docling_pdf_child

    events: list[object] = []
    # Replace the module docling.py sees, not the stdlib's own signal.signal.
    monkeypatch.setattr(
        "dite.io.docling.signal",
        SimpleNamespace(
            signal=lambda signum, handler: events.append((signum, handler)),
            SIGINT=signal.SIGINT,
            SIG_IGN=signal.SIG_IGN,
        ),
    )

    _run_docling_pdf_child(lambda *args: events.append(args), "a.pdf", False)

    assert events == [(signal.SIGINT, signal.SIG_IGN), ("a.pdf", False)]
