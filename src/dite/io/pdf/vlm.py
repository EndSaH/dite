"""PDF VLM fallback implementation."""

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from openai import OpenAI

from dite.app.config import Config
from dite.app.i18n import get_locale, t
from dite.util.api import AsyncRequestRuntime, ChatCompletionRequest
from dite.util.cancel import raise_if_cancelled
from dite.util.log import get_logger

from ..base import ExtractionResult
from .policy import PDF_VLM_SAMPLE_PAGE_LIMIT
from .render import render_pdf_pages


@dataclass(frozen=True)
class VLMSamplingResult:
    """VLM PDF sampling result plus runtime-only metrics."""

    result: ExtractionResult
    api_page_calls: int
    sample_page_limit: int


def _build_vlm_fallback_prompt(page_num: int) -> str:
    """Build a locale-aware fallback prompt for scanned PDF pages."""
    if get_locale() == "zh-CN":
        if page_num == 1:
            return (
                "这是一份扫描文档的第一页。请详细描述这份文档的内容，"
                "包括：文档类型、标题、核心主题、关键信息。"
                "如果有表格或图表，请描述其内容。"
                "直接输出描述，不要啰嗦。"
            )
        return (
            f"这是文档的第 {page_num} 页。请描述此页的主要内容，"
            "包括重要段落、公式、图表等信息。"
            "直接输出描述，不要啰嗦。"
        )

    if page_num == 1:
        return (
            "This is the first page of a scanned document. "
            "Describe the document in detail, including its type, title, core topic, "
            "and key information. If there are tables or charts, describe them too. "
            "Return only the description."
        )
    return (
        f"This is page {page_num} of the document. "
        "Describe the main content of this page, including important paragraphs, "
        "formulas, charts, and other key information. Return only the description."
    )


def _format_vlm_page_content(page_num: int, content: str) -> str:
    """Format page content markers in the active locale."""
    if get_locale() == "zh-CN":
        return f"【第{page_num}页】\n{content}"
    return f"[Page {page_num}]\n{content}"


def _build_vlm_chat_request(
    *,
    model: str,
    image_url: str,
    prompt_text: str,
    timeout: float | None = None,
) -> ChatCompletionRequest:
    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return ChatCompletionRequest(kwargs=kwargs)


def _encode_image_data_url(image, *, mime_type: str = "image/png") -> tuple[str, float]:
    """Serialize a PIL image to an in-memory data URL."""
    buffer = BytesIO()
    image.save(buffer, "PNG")
    data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    size_kb = len(buffer.getvalue()) / 1024
    return f"data:{mime_type};base64,{data}", size_kb


def _resize_page_if_needed(image):
    """Resize a page image to the shared PDF/image VLM limits."""
    max_long = 1920
    max_short = 1080
    width, height = image.width, image.height

    if width > height:
        if width <= max_long and height <= max_short:
            return image, None
        ratio = min(max_long / width, max_short / height)
    else:
        if height <= max_long and width <= max_short:
            return image, None
        ratio = min(max_short / width, max_long / height)

    new_size = (int(width * ratio), int(height * ratio))
    return image.resize(new_size), (width, height, new_size[0], new_size[1])


def extract_pdf_with_vlm_sampling(
    file_path: Path,
    client: OpenAI,
    *,
    config: Config,
    max_pages: int = PDF_VLM_SAMPLE_PAGE_LIMIT,
    request_runtime: AsyncRequestRuntime | None = None,
) -> VLMSamplingResult:
    """Use VLM to sample PDF pages for fallback extraction."""
    logger = get_logger()
    raise_if_cancelled()
    try:
        rendered = render_pdf_pages(file_path, max_pages=max_pages)
        if not rendered.success:
            return VLMSamplingResult(
                result=ExtractionResult(
                    content="",
                    success=False,
                    extractor="vlm_fallback",
                    error=rendered.error or t("error_pdf_render_failed"),
                ),
                api_page_calls=0,
                sample_page_limit=rendered.sample_page_limit,
            )

        requests: list[ChatCompletionRequest] = []
        page_numbers: list[int] = []
        for page_num, image in enumerate(rendered.pages, start=1):
            logger.debug(
                t(
                    "debug_vlm_page_processing",
                    page=page_num,
                    total=len(rendered.pages),
                    width=image.width,
                    height=image.height,
                )
            )
            image, resized = _resize_page_if_needed(image)
            if resized is not None:
                old_width, old_height, new_width, new_height = resized
                logger.debug(
                    t(
                        "debug_vlm_page_resized",
                        old_width=old_width,
                        old_height=old_height,
                        new_width=new_width,
                        new_height=new_height,
                    )
                )

            image_url, file_size_kb = _encode_image_data_url(image)
            logger.debug(t("debug_vlm_image_size", size_kb=file_size_kb))
            requests.append(
                _build_vlm_chat_request(
                    model=config.models.vlm,
                    image_url=image_url,
                    prompt_text=_build_vlm_fallback_prompt(page_num),
                    timeout=60.0,
                )
            )
            page_numbers.append(page_num)

        api_page_calls = len(requests)
        all_contents: list[str] = []

        # The run may have been interrupted while the pages were rendering;
        # cancellation is a BaseException and passes the handlers below.
        raise_if_cancelled()
        if request_runtime is not None:
            results = request_runtime.run_vlm_page_batch(
                requests,
                per_document_limit=config.processing.vlm_pages_per_document,
            )
            success_count = 0
            failed_count = 0
            max_wait_sec = 0.0
            max_request_sec = 0.0
            for page_num, result in zip(page_numbers, results, strict=False):
                max_wait_sec = max(max_wait_sec, result.queue_wait_sec)
                max_request_sec = max(max_request_sec, result.request_elapsed_sec)
                if result.error is not None:
                    failed_count += 1
                    logger.debug(
                        t(
                            "debug_vlm_page_result_failed",
                            page=page_num,
                            error=result.error,
                            wait_sec=result.queue_wait_sec,
                            request_sec=result.request_elapsed_sec,
                        )
                    )
                    continue
                page_content = (result.content or "").strip()
                if not page_content:
                    failed_count += 1
                    logger.debug(
                        t(
                            "debug_vlm_page_result_empty",
                            page=page_num,
                            wait_sec=result.queue_wait_sec,
                            request_sec=result.request_elapsed_sec,
                        )
                    )
                    continue
                success_count += 1
                all_contents.append(_format_vlm_page_content(page_num, page_content))
                logger.debug(
                    t(
                        "debug_vlm_page_result_done",
                        page=page_num,
                        length=len(page_content),
                        wait_sec=result.queue_wait_sec,
                        request_sec=result.request_elapsed_sec,
                    )
                )
            if results:
                logger.debug(
                    t(
                        "debug_vlm_batch_summary",
                        success=success_count,
                        failed=failed_count,
                        max_wait_sec=max_wait_sec,
                        max_request_sec=max_request_sec,
                    )
                )
        else:
            for page_num, request in zip(page_numbers, requests, strict=False):
                raise_if_cancelled()
                try:
                    logger.debug(t("debug_vlm_api_call"))
                    response = client.chat.completions.create(**request.kwargs)
                    page_content = (response.choices[0].message.content or "").strip()
                    if not page_content:
                        continue
                    all_contents.append(
                        _format_vlm_page_content(page_num, page_content)
                    )
                    logger.debug(
                        t(
                            "debug_vlm_page_done",
                            page=page_num,
                            length=len(page_content),
                        )
                    )
                except Exception as api_error:
                    logger.debug(
                        t(
                            "debug_vlm_page_failed",
                            page=page_num,
                            error=api_error,
                        )
                    )

        if not all_contents:
            return VLMSamplingResult(
                result=ExtractionResult(
                    content="",
                    success=False,
                    extractor="vlm_fallback",
                    error=t("error_pdf_vlm_no_usable_content"),
                ),
                api_page_calls=api_page_calls,
                sample_page_limit=rendered.sample_page_limit,
            )

        return VLMSamplingResult(
            result=ExtractionResult(
                content="\n\n".join(all_contents),
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=api_page_calls,
            sample_page_limit=rendered.sample_page_limit,
        )
    except Exception as exc:
        error_msg = str(exc)
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        logger.warning(t("warning_vlm_fallback_failed", error=error_msg))
        return VLMSamplingResult(
            result=ExtractionResult(
                content="",
                success=False,
                extractor="vlm_fallback",
                error=error_msg,
            ),
            api_page_calls=0,
            sample_page_limit=max_pages,
        )
