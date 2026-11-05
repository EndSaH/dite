"""Feature builders for clustering V2."""

from __future__ import annotations

import re
from pathlib import Path

from openai import OpenAI

from dite.app.config import Config
from dite.doc.analyze import analyze_document
from dite.doc.embed import FILENAME_FALLBACK_MAX_CHARS
from dite.doc.model import (
    DocumentFeatures,
    EntityFeatures,
    LayoutHints,
    MetadataFeatures,
    QualityFlags,
)

_EXCERPT_LIMIT = 200
_SHORT_TEXT_THRESHOLD = 100
_FAILED_TEXT_THRESHOLD = 10
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "notes",
    "guide",
    "report",
    "intro",
    "introduction",
    "study",
    "document",
    "content",
}


_TITLE_SCAN_LINE_LIMIT = 15
_TITLE_MAX_LENGTH = 120
_CJK_PATTERN = re.compile(r"[一-鿿]")
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_GLYPH_CODE_PATTERN = re.compile(r"(?:/G[0-9A-F]{2})+")
_LETTER_OR_DIGIT_PATTERN = re.compile(r"[^\W_]")
# A line holding nothing but a page number: "3", "- 3 -", "Page 3 of 10",
# "p. 3", "3 / 10", "第 3 页", "第3页 共10页".
_PAGE_NUMBER_LINE_PATTERN = re.compile(
    r"[-–—\s]*(?:(?:page|pg|p)\.?\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?[-–—\s]*"
    r"|第\s*\d{1,4}\s*页(?:\s*[/,，]?\s*共\s*\d{1,4}\s*页)?",
    re.IGNORECASE,
)
_LEADING_NUMBERING_PATTERN = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)*[.、)]?|[ivxIVX]+[.、)]|[一二三四五六七八九十]+[、.])\s+"
)
# Lines that look like a title but carry no topic: document furniture,
# extractor placeholders and bare numbering. Matching one of these across two
# files says nothing about whether the files belong together.
_GENERIC_TITLES = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "agenda",
    "appendix",
    "bibliography",
    "chapter",
    "content",
    "contents",
    "cover",
    "draft",
    "figure",
    "foreword",
    "image",
    "index",
    "introduction",
    "notes",
    "outline",
    "overview",
    "preface",
    "readme",
    "references",
    "slide",
    "summary",
    "table of contents",
    "title",
    "toc",
    "unknown",
    "untitled",
    "目录",
    "摘要",
    "前言",
    "序言",
    "引言",
    "简介",
    "概述",
    "附录",
    "参考文献",
    "致谢",
    "封面",
    "大纲",
    "笔记",
    "无标题",
}
_GENERIC_TITLE_PATTERNS = (
    re.compile(
        r"(?:page|slide|chapter|lecture|lesson|part|section|unit|week|lab|"
        r"homework|hw|exercise|figure|fig|table)\s*[0-9ivx]+"
    ),
    re.compile(r"第\s*[0-9一二三四五六七八九十百零]+\s*[页章节讲课部分周篇]"),
    re.compile(r"[0-9\W_]+"),
)


_TERM_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
# Frequent function-word bigrams that would otherwise dominate CJK keywords.
_CJK_STOP_BIGRAMS = {
    "一个",
    "以及",
    "所以",
    "因为",
    "如果",
    "可以",
    "我们",
    "他们",
    "你们",
    "这个",
    "那个",
    "这些",
    "那些",
    "进行",
    "通过",
    "没有",
    "就是",
    "不是",
    "什么",
    "已经",
    "其中",
    "之间",
    "以下",
    "以上",
    "对于",
    "由于",
    "但是",
    "并且",
    "或者",
}


# Bigrams containing one of these usually straddle a word boundary
# ("与特" in "特征值与特征向量"), so they are skipped as keywords.
_CJK_FUNCTION_CHARS = set(
    "的了是在和与及或等为以对于从将把被也都而之其个这那我你他它们中上下"
)


def _is_keyword_token(token: str) -> bool:
    if token in _STOPWORDS or token in _CJK_STOP_BIGRAMS:
        return False
    return not (_is_cjk(token) and any(ch in _CJK_FUNCTION_CHARS for ch in token))


def _is_cjk(term: str) -> bool:
    return bool(_CJK_PATTERN.match(term))


def _tokenize_text(text: str) -> list[str]:
    """Return normalized tokens from text, a file name or a path fragment.

    Latin letters and digits form word tokens. Chinese has no spaces, so a run
    of CJK characters is split into overlapping character bigrams (a single
    character stays as is). Treating a whole run as one token made every
    Chinese sentence a unique "keyword" and made Chinese file names match
    only when identical.
    """
    tokens: list[str] = []
    for term in _TERM_PATTERN.findall(text):
        if not _is_cjk(term):
            tokens.append(term.casefold())
        elif len(term) == 1:
            tokens.append(term)
        else:
            tokens.extend(term[index : index + 2] for index in range(len(term) - 1))
    return tokens


def _display_terms(text: str) -> list[str]:
    """Return word-like terms for display, keeping CJK runs whole."""
    return [term.casefold() for term in _TERM_PATTERN.findall(text)]


def strip_markup(text: str) -> str:
    """Remove extractor placeholders and markup that carry no document topic.

    Docling emits ``<!-- image -->`` for every picture; leaving it in place lets
    unrelated image-led files share a "title" and a top keyword.
    """
    cleaned = _HTML_COMMENT_PATTERN.sub(" ", text)
    cleaned = _MARKDOWN_IMAGE_PATTERN.sub(" ", cleaned)
    return _HTML_TAG_PATTERN.sub(" ", cleaned)


def meaningful_lines(text: str) -> list[str]:
    """Return the lines of ``text`` that can carry document content.

    Markup and extractor placeholders (``<!-- image -->``, page-break and
    formula comments, Markdown images), PDF glyph codes, lines without a
    letter or digit (rules, empty table rows) and bare page-number lines are
    dropped; whitespace inside a line is collapsed. A scanned PDF that Docling
    turned into ``<!-- image -->`` plus page numbers has no lines left.
    """
    cleaned = _GLYPH_CODE_PATTERN.sub(" ", strip_markup(text))
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = " ".join(raw_line.split())
        if not _LETTER_OR_DIGIT_PATTERN.search(line):
            continue
        if _PAGE_NUMBER_LINE_PATTERN.fullmatch(line):
            continue
        lines.append(line)
    return lines


def meaningful_text(text: str) -> str:
    """Return the content of ``text`` without markup, placeholders or layout.

    This is what quality flags and identical-content checks compare: two
    copies of a document that differ only in placeholders, page numbers or
    line wrapping give the same string.
    """
    return " ".join(meaningful_lines(text))


def filename_title(stem: str) -> str:
    """Return the title-like form of a file stem (``a_b-c`` -> ``a b c``)."""
    return re.sub(r"[_-]+", " ", stem).strip()


def normalize_title_candidate(line: str) -> str:
    """Normalize one extracted line into a comparable title string."""
    cleaned = strip_markup(line).strip()
    cleaned = cleaned.lstrip("#*>-+| \t").rstrip("|* \t")
    cleaned = cleaned.strip("\"'`")
    return re.sub(r"\s+", " ", cleaned).strip()


def is_generic_title(title: str) -> bool:
    """Return whether a title is document furniture rather than a topic."""
    lowered = title.casefold().strip(" .:：-—_")
    lowered = _LEADING_NUMBERING_PATTERN.sub("", lowered).strip(" .:：-—_")
    if not lowered or lowered in _GENERIC_TITLES:
        return True
    return any(pattern.fullmatch(lowered) for pattern in _GENERIC_TITLE_PATTERNS)


def _is_usable_title(title: str) -> bool:
    min_length = 3 if _CJK_PATTERN.search(title) else 4
    if not min_length <= len(title) <= _TITLE_MAX_LENGTH:
        return False
    if title.startswith(("![", "<", "|")):
        return False
    return not is_generic_title(title)


def _title_candidates(file_path: Path, content: str) -> list[str]:
    """Build conservative title candidates from extracted content and file name.

    Content-derived titles come first. The file-stem title is appended last as a
    display fallback; candidate-edge building does not treat it as a title.
    """
    titles: list[str] = []
    scanned = 0
    for line in content.splitlines():
        if not line.strip():
            continue
        scanned += 1
        if scanned > _TITLE_SCAN_LINE_LIMIT:
            break
        candidate = normalize_title_candidate(line)
        if _is_usable_title(candidate):
            titles.append(candidate)
            break
    stem_title = filename_title(file_path.stem)
    if stem_title and stem_title not in titles:
        titles.append(stem_title)
    return titles


def _infer_keywords_and_topic(
    content: str,
    title_candidates: list[str],
) -> tuple[list[str], str]:
    """Infer lightweight keywords/topic without invoking a model."""
    tokens = [
        token
        for token in _tokenize_text(strip_markup(content))
        if _is_keyword_token(token)
    ]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    keywords = [token for token, _count in ranked[:5]]
    if title_candidates:
        title_tokens = [
            token
            for token in _display_terms(title_candidates[0])
            if token not in _STOPWORDS
        ]
        if title_tokens:
            return keywords, " ".join(title_tokens[:3])
    if keywords:
        return keywords, " ".join(keywords[:3])
    return [], ""


def _infer_domain(keywords: list[str], path_tokens: list[str]) -> str:
    joined = set(keywords) | set(path_tokens)
    if {"matrix", "vector", "algebra", "math"} & joined:
        return "education"
    if {"minecraft", "server", "mod", "fabric"} & joined:
        return "gaming"
    if {"invoice", "bank", "finance", "revenue"} & joined:
        return "finance"
    if {"python", "rust", "api", "code"} & joined:
        return "tech"
    return ""


def build_document_features(
    file_path: Path,
    content: str,
    *,
    config: Config | None = None,
    client: OpenAI | None = None,
    file_report=None,
) -> DocumentFeatures:
    """Build a conservative V2 feature object from current pipeline artifacts."""
    # Quality is judged on what is left after placeholders and page numbers:
    # ``<!-- image -->`` from a scan without OCR is not extracted text.
    meaningful_length = len(meaningful_text(content))
    parent_tokens = []
    for part in file_path.parent.parts:
        parent_tokens.extend(_tokenize_text(part))

    primary_success = getattr(file_report, "primary_success", True)
    selected_source = getattr(file_report, "selected_source", "")
    primary_extractor = getattr(file_report, "primary_extractor", "")
    extraction_trace = ""
    if primary_extractor or selected_source:
        extraction_trace = f"{primary_extractor}->{selected_source}".strip("->")

    quality_flags = QualityFlags(
        extraction_failed=(
            (not primary_success) or meaningful_length < _FAILED_TEXT_THRESHOLD
        ),
        short_text=meaningful_length < _SHORT_TEXT_THRESHOLD,
        # Judged on the final text, not the primary extractor: a PDF rescued by
        # VLM fallback has real content even though the primary parse failed.
        # The cut-off mirrors the embedding input rule: the vector was built
        # from the file name alone, or from the file name plus placeholders and
        # page numbers. Either way it carries file-name evidence only.
        filename_dominant=(
            meaningful_length <= FILENAME_FALLBACK_MAX_CHARS and bool(file_path.stem)
        ),
    )
    title_candidates = _title_candidates(file_path, content)
    keywords, topic = _infer_keywords_and_topic(content, title_candidates)
    domain = _infer_domain(keywords, parent_tokens)
    language = ""
    summary = title_candidates[0] if title_candidates else ""
    layout = LayoutHints()
    entities = EntityFeatures(
        keywords=keywords,
        topic=topic,
        domain=domain,
    )

    analysis_enabled = bool(
        config is not None
        and config.feature_extraction.analysis_enabled
        and client is not None
        and content.strip()
    )
    if analysis_enabled:
        analysis = analyze_document(
            client,
            content,
            config=config,
            max_content_length=config.feature_extraction.analysis_max_content_length,
            max_retries=config.feature_extraction.analysis_max_retries,
        )
        if analysis.summary.strip():
            summary = analysis.summary.strip()
        if analysis.content.keywords:
            entities.keywords = analysis.content.keywords
        if analysis.content.topic.strip():
            entities.topic = analysis.content.topic.strip()
        if analysis.content.domain.strip():
            entities.domain = analysis.content.domain.strip()
        language = analysis.content.language.strip()
        layout = LayoutHints(
            document_type=analysis.layout.type,
            columns=analysis.layout.columns,
            has_table=analysis.layout.has_table,
            has_image_heavy_layout=("image" in analysis.layout.visual_elements),
            template_signals=(
                [analysis.template_hints] if analysis.template_hints.strip() else []
            ),
        )
        quality_flags.low_confidence_analysis = analysis.confidence < 0.5
        quality_flags.language_uncertain = language in {"", "mixed"}

    return DocumentFeatures(
        file_id=str(file_path),
        path=file_path,
        name=file_path.name,
        stem=file_path.stem,
        extension=file_path.suffix.lower(),
        content_text=content,
        content_excerpt=(
            content[:_EXCERPT_LIMIT] + "..."
            if len(content) > _EXCERPT_LIMIT
            else content
        ),
        language=language,
        token_count_estimate=len(_tokenize_text(content)),
        summary=summary,
        metadata=MetadataFeatures(
            file_name_tokens=_tokenize_text(file_path.stem),
            parent_path_tokens=parent_tokens,
            title_candidates=title_candidates,
        ),
        entities=entities,
        layout=layout,
        quality_flags=quality_flags,
        selected_source=str(selected_source),
        extraction_trace=extraction_trace,
    )


SUMMARY_EMBEDDING_EXCERPT_CHARS = 1000


def build_summary_embedding_text(
    feature: DocumentFeatures,
    *,
    excerpt_chars: int = SUMMARY_EMBEDDING_EXCERPT_CHARS,
) -> str:
    """Build the ``summary`` embedding input from structured features.

    Instead of a head/middle/tail cut of raw text, this embeds what the
    document is about: its content title, summary, topic and keywords, plus a
    short cleaned opening excerpt. With ``feature_extraction.analysis_enabled``
    the summary, topic and keywords come from the LLM analyzer; otherwise they
    are the deterministic title and keyword features. Formatting noise such
    as tables, image placeholders and reference lists mostly drops out.
    """
    if not feature.content_text.strip():
        # Nothing was extracted; any title or topic would come from the file
        # name, so say so plainly instead of dressing it up as content.
        return f"File name: {feature.name}"
    stem_title = filename_title(feature.stem)
    content_titles = [
        title for title in feature.title_candidates if title != stem_title
    ]
    title = content_titles[0] if content_titles else ""
    lines: list[str] = []
    if title:
        lines.append(f"Title: {title}")
    if feature.summary and feature.summary not in {title, stem_title}:
        lines.append(f"Summary: {feature.summary}")
    if feature.topic:
        lines.append(f"Topic: {feature.topic}")
    if feature.keywords:
        lines.append(f"Keywords: {', '.join(feature.keywords[:10])}")
    excerpt = re.sub(r"\s+", " ", strip_markup(feature.content_text)).strip()
    if excerpt:
        lines.append(f"Excerpt: {excerpt[:excerpt_chars]}")
    return "\n".join(lines) or f"File name: {feature.name}"
