"""Shared router-layer types."""

from __future__ import annotations

from dataclasses import dataclass, field

from dite.app.config import Config

from .base import ExtractionResult
from .docling import DoclingExtractor
from .markitdown import MarkItDownExtractor
from .pdf.final import PDFCacheWriteIntent, ResolvedSource, VLMSource
from .pdf.policy import PDFProfile
from .text import TextExtractor
from .vlm import VLMExtractor


@dataclass(frozen=True)
class ResolvedExtraction:
    """Final extraction decision after optional PDF VLM fallback resolution."""

    primary_result: ExtractionResult
    primary_effective_length: int
    pdf_profile: PDFProfile | None
    fallback_needed: bool
    selected_source: ResolvedSource
    final_content: str
    final_effective_length: int
    vlm_content: str | None
    vlm_source: VLMSource
    vlm_api_success: bool
    vlm_api_page_calls: int
    sample_page_limit: int | None
    cache_write_intent: PDFCacheWriteIntent = field(
        default_factory=lambda: PDFCacheWriteIntent(
            should_write=False,
            content=None,
        )
    )


class ExtractorRegistry:
    """Replaceable extractor registry with instance-bound state."""

    def __init__(self, config: Config, client=None) -> None:
        self.config = config
        self._client = client
        self._docling: DoclingExtractor | None = None
        self._markitdown: MarkItDownExtractor | None = None
        self._text: TextExtractor | None = None
        self._vlm: VLMExtractor | None = None

    def get_docling(self) -> DoclingExtractor:
        if self._docling is None:
            self._docling = DoclingExtractor(
                pdf_timeout_sec=self.config.processing.docling_pdf_timeout_sec,
                device=self.config.processing.docling_device,
                pdf_max_pages=self.config.processing.docling_pdf_max_pages,
            )
        return self._docling

    def get_markitdown(self) -> MarkItDownExtractor:
        if self._markitdown is None:
            self._markitdown = MarkItDownExtractor()
        return self._markitdown

    def get_text(self) -> TextExtractor:
        if self._text is None:
            self._text = TextExtractor()
        return self._text

    def get_vlm(self) -> VLMExtractor:
        if self._vlm is None:
            self._vlm = VLMExtractor(config=self.config, client=self._client)
        return self._vlm
