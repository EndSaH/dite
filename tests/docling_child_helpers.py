"""Stand-ins for the Docling PDF child process used by the subprocess tests.

``extract_docling_pdf_in_subprocess`` starts its child with the ``spawn``
method, so a replacement target has to be a module-level function that the
child interpreter can import. Each function takes the same arguments as
``dite.io.docling._docling_pdf_extract_child``.
"""

import os
import signal
import time

from dite.io.base import ExtractionResult

# Far more than a pipe buffer (64 KiB on Linux) holds.
LARGE_CONTENT_SIZE = 2 * 1024 * 1024


def send_large_result(
    file_path, enable_ocr, artifacts_path, locale, device, conn, max_pages=None
) -> None:
    del file_path, enable_ocr, artifacts_path, locale, device, max_pages
    try:
        conn.send(
            ExtractionResult(
                content="x" * LARGE_CONTENT_SIZE,
                success=True,
                extractor="docling",
            )
        )
    finally:
        conn.close()


def die_without_result(
    file_path, enable_ocr, artifacts_path, locale, device, conn, max_pages=None
) -> None:
    """Die the way the OOM killer ends a child: SIGKILL, nothing sent."""
    del file_path, enable_ocr, artifacts_path, locale, device, conn, max_pages
    os.kill(os.getpid(), signal.SIGKILL)


def sleep_forever(
    file_path, enable_ocr, artifacts_path, locale, device, conn, max_pages=None
) -> None:
    del file_path, enable_ocr, artifacts_path, locale, device, conn, max_pages
    time.sleep(3600)
