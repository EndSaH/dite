"""Docling 提取器 - 用于现代文档格式"""

import multiprocessing
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path

from dite.app.i18n import set_locale, t
from dite.util.cancel import raise_if_cancelled
from dite.util.log import silence_docling_logging

from .base import BaseExtractor, ExtractionResult

_DOCLING_PDF_SETUP_COMMAND = "uv run dite setup docling-pdf"
# How often the parent, while waiting on a Docling child, checks whether the
# run has been cancelled.
_DOCLING_PDF_CANCEL_CHECK_SEC = 0.2
# How long a child that has already sent its result gets to exit by itself.
_DOCLING_PDF_CHILD_EXIT_GRACE_SEC = 5.0


class DoclingPDFTimeoutError(TimeoutError):
    """Raised when Docling spends too long on a PDF."""


@contextmanager
def _docling_pdf_timeout(seconds: float | None) -> Iterator[None]:
    """Bound Docling PDF conversion time on the main thread."""
    if (
        seconds is None
        or seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise DoclingPDFTimeoutError

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def get_docling_pdf_artifacts_path() -> Path:
    """返回 DITE 约定的 Docling PDF 模型目录。"""
    return Path.home() / ".cache" / "dite" / "docling" / "models"


def _get_required_pdf_artifact_dirs() -> tuple[str, str]:
    """返回 Docling PDF 最小必需模型目录。"""
    from docling.datamodel.pipeline_options import LayoutOptions
    from docling.models.table_structure_model import TableStructureModel

    return (
        LayoutOptions().model_spec.model_repo_folder,
        TableStructureModel._model_repo_folder,
    )


def has_docling_pdf_artifacts(artifacts_path: Path | None = None) -> bool:
    """检查 Docling PDF 所需模型是否已准备好。"""
    base_path = artifacts_path or get_docling_pdf_artifacts_path()
    required_dirs = _get_required_pdf_artifact_dirs()
    return all((base_path / name).exists() for name in required_dirs)


def download_docling_pdf_models(
    output_dir: Path | None = None,
    *,
    force: bool = False,
    progress: bool = False,
) -> Path:
    """下载 DITE 所需的最小 Docling PDF 模型集合。"""
    from docling.utils.model_downloader import download_models

    target_dir = output_dir or get_docling_pdf_artifacts_path()
    download_models(
        output_dir=target_dir,
        force=force,
        progress=progress,
        with_layout=True,
        with_tableformer=True,
        with_code_formula=False,
        with_picture_classifier=False,
        with_smolvlm=False,
        with_granitedocling=False,
        with_granitedocling_mlx=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
        with_rapidocr=False,
        with_easyocr=False,
    )
    return target_dir


def _suppress_docling_warnings() -> None:
    """抑制 docling 内部的警告日志（如 WMF 图片加载警告）"""
    silence_docling_logging()


def _docling_pdf_extract_child(
    file_path: str,
    enable_ocr: bool,
    artifacts_path: str | None,
    locale: str,
    device: str,
    conn: Connection,
    max_pages: int | None = None,
) -> None:
    try:
        set_locale(locale)
        silence_docling_logging()
        extractor = DoclingExtractor(
            enable_ocr=enable_ocr,
            artifacts_path=Path(artifacts_path) if artifacts_path is not None else None,
            pdf_timeout_sec=None,
            device=device,
            pdf_max_pages=max_pages,
        )
        conn.send(extractor.extract(Path(file_path)))
    finally:
        conn.close()


def _run_docling_pdf_child(target: Callable[..., None], *args: object) -> None:
    """Child entry point: leave Ctrl+C to the parent, then run ``target``."""
    # Ctrl+C sends SIGINT to the whole process group. The parent decides what
    # an interrupt means and terminates this process itself; ignoring the
    # signal here avoids racing that decision and a second traceback. It only
    # takes effect once the child runs this function: a Ctrl+C while the
    # spawned interpreter is still importing dite can still print one.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    target(*args)


def _stop_process(process: multiprocessing.process.BaseProcess) -> None:
    """Terminate a running child and reap it so it does not linger."""
    process.terminate()
    process.join(1)
    if process.is_alive():
        process.kill()
        process.join()


def _describe_exit_code(exitcode: int | None) -> str:
    if exitcode is not None and exitcode < 0:
        try:
            return f"{exitcode}, {signal.Signals(-exitcode).name}"
        except ValueError:
            pass
    return str(exitcode)


def extract_docling_pdf_in_subprocess(
    file_path: Path,
    *,
    enable_ocr: bool = False,
    artifacts_path: Path | None = None,
    timeout_sec: float | None = 60.0,
    locale: str,
    device: str = "auto",
    max_pages: int | None = None,
) -> ExtractionResult:
    raise_if_cancelled()
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_run_docling_pdf_child,
        args=(
            _docling_pdf_extract_child,
            str(file_path),
            enable_ocr,
            str(artifacts_path) if artifacts_path is not None else None,
            locale,
            device,
            child_conn,
            max_pages,
        ),
    )
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec

    started = False
    try:
        process.start()
        started = True
        child_conn.close()
        # Read the result while the child is still running. A result larger
        # than the pipe buffer (64 KiB on Linux) keeps the child blocked in
        # send() until it is read, so waiting for the child to exit first
        # would only end at the timeout.
        while True:
            raise_if_cancelled()
            wait_sec = _DOCLING_PDF_CANCEL_CHECK_SEC
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ExtractionResult(
                        content="",
                        success=False,
                        extractor=DoclingExtractor.name,
                        error=t(
                            "error_docling_pdf_timeout",
                            seconds=timeout_sec or 0,
                        ),
                    )
                wait_sec = min(wait_sec, remaining)
            if parent_conn.poll(wait_sec):
                break

        try:
            result = parent_conn.recv()
        except (EOFError, OSError):
            # poll() also returns on EOF: the child died without sending a
            # result, e.g. killed by the OOM killer or a crash in native code.
            process.join(_DOCLING_PDF_CHILD_EXIT_GRACE_SEC)
            if process.is_alive():
                _stop_process(process)
            return ExtractionResult(
                content="",
                success=False,
                extractor=DoclingExtractor.name,
                error=t(
                    "error_docling_pdf_subprocess_failed",
                    exitcode=_describe_exit_code(process.exitcode),
                ),
            )
        process.join(_DOCLING_PDF_CHILD_EXIT_GRACE_SEC)
        return result
    finally:
        # Timeout, cancellation, or an interrupt while waiting: do not leave
        # the child running or unreaped.
        if started and process.is_alive():
            _stop_process(process)
        parent_conn.close()


class DoclingExtractor(BaseExtractor):
    """使用 Docling 提取 PDF/DOCX/PPTX 等现代文档格式"""

    name = "docling"

    @property
    def supported_extensions(self) -> set[str]:
        # 包含可能被错误命名的 OOXML 文件
        return {".pdf", ".docx", ".pptx", ".xlsx", ".ppt", ".doc", ".xls"}

    def __init__(
        self,
        enable_ocr: bool = False,
        artifacts_path: Path | None = None,
        pdf_timeout_sec: float | None = 60.0,
        device: str = "auto",
        pdf_max_pages: int | None = None,
    ):
        """
        Args:
            enable_ocr: 是否启用 OCR（默认禁用，因为非常慢）
            device: Docling 推理设备（auto/cpu/cuda/cuda:N/mps）
            pdf_max_pages: 只转换 PDF 的前 N 页（None 表示全部）。只有一段
                摘录会进入后续流程，转换整本书既慢又容易超时。
        """
        self._converter = None
        self._enable_ocr = enable_ocr
        self._artifacts_path = artifacts_path or get_docling_pdf_artifacts_path()
        self._pdf_timeout_sec = pdf_timeout_sec
        self._device = device
        self._pdf_max_pages = (
            pdf_max_pages if pdf_max_pages is not None and pdf_max_pages > 0 else None
        )

    def _get_converter(self):
        """延迟加载 Docling（避免启动时的开销）"""
        if self._converter is None:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            # 抑制 docling 内部警告（导入后才能配置）
            _suppress_docling_warnings()

            # 配置 PDF 处理选项
            pdf_opts = PdfPipelineOptions()
            pdf_opts.do_ocr = self._enable_ocr  # 禁用 OCR 大幅加速
            pdf_opts.artifacts_path = self._artifacts_path
            pdf_opts.accelerator_options = AcceleratorOptions(device=self._device)

            self._converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pdf_opts),
                }
            )
        return self._converter

    def extract(self, file_path: Path) -> ExtractionResult:
        """提取文档内容为 Markdown"""
        if file_path.suffix.lower() == ".pdf" and not has_docling_pdf_artifacts(
            self._artifacts_path
        ):
            return ExtractionResult(
                content="",
                success=False,
                extractor=self.name,
                error=t(
                    "error_docling_pdf_models_missing",
                    command=_DOCLING_PDF_SETUP_COMMAND,
                ),
            )
        try:
            converter = self._get_converter()
            if file_path.suffix.lower() == ".pdf":
                convert_kwargs = {}
                if self._pdf_max_pages is not None:
                    # page_range limits the pages converted; max_num_pages
                    # would instead reject longer documents outright.
                    convert_kwargs["page_range"] = (1, self._pdf_max_pages)
                with _docling_pdf_timeout(self._pdf_timeout_sec):
                    result = converter.convert(str(file_path), **convert_kwargs)
            else:
                result = converter.convert(str(file_path))
            content = result.document.export_to_markdown()

            return ExtractionResult(
                content=content,
                success=True,
                extractor=self.name,
            )
        except DoclingPDFTimeoutError:
            return ExtractionResult(
                content="",
                success=False,
                extractor=self.name,
                error=t(
                    "error_docling_pdf_timeout",
                    seconds=self._pdf_timeout_sec or 0,
                ),
            )
        except Exception as e:
            # 只保留错误类型和简短消息，避免 JSON 泄漏
            error_type = type(e).__name__
            error_msg = str(e)
            # 如果错误消息过长（通常是 JSON），只取前 100 字符
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            return ExtractionResult(
                content="",
                success=False,
                extractor=self.name,
                error=f"{error_type}: {error_msg}",
            )
