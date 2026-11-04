"""向量化模块"""

from typing import Literal

import numpy as np
from openai import APIStatusError, OpenAI

from dite.app.config import Config
from dite.app.i18n import t
from dite.util.log import get_logger

EmbeddingInputMode = Literal["with_filename", "content_only", "summary"]
# Texts at or below this length are embedded from the file name alone in
# ``with_filename`` mode, so their vectors carry filename evidence only.
FILENAME_FALLBACK_MAX_CHARS = 10
_EMBEDDING_INPUT_VERSIONS: dict[EmbeddingInputMode, str] = {
    "with_filename": "filename-smart-content-normalized-v2",
    "content_only": "content-only-normalized-v1",
    "summary": "feature-summary-v1",
}


def get_embedding_cache_version(
    embedding_model: str,
    input_mode: EmbeddingInputMode = "with_filename",
    *,
    variant: str | None = None,
) -> str:
    """Return the cache key for the current embedding input format.

    ``variant`` records settings that change the embedded text without
    changing the input mode (excerpt size and split, the analysis model for
    summary inputs). It is omitted for default settings, so existing caches
    stay valid.
    """
    version = f"{embedding_model}|input={_EMBEDDING_INPUT_VERSIONS[input_mode]}"
    return f"{version}|{variant}" if variant else version


def get_embedding_cache_variant(config: Config, input_mode: str) -> str | None:
    """Describe non-default settings that change the embedded text."""
    defaults = type(config.processing)()
    processing = config.processing
    parts: list[str] = []
    excerpt = (
        processing.text_truncate_limit,
        processing.excerpt_head_ratio,
        processing.excerpt_middle_ratio,
    )
    default_excerpt = (
        defaults.text_truncate_limit,
        defaults.excerpt_head_ratio,
        defaults.excerpt_middle_ratio,
    )
    if excerpt != default_excerpt:
        parts.append(f"excerpt={excerpt[0]}:{excerpt[1]:g}:{excerpt[2]:g}")
    if input_mode == "summary":
        analysis = (
            config.models.llm if config.feature_extraction.analysis_enabled else "off"
        )
        parts.append(f"analysis={analysis}")
    return ";".join(parts) or None


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return L2-normalized embeddings without changing zero vectors."""
    result = np.asarray(embeddings, dtype=np.float32)
    if result.size == 0:
        return result
    if result.ndim == 1:
        norm = float(np.linalg.norm(result))
        if norm == 0.0:
            return result
        return result / norm

    norms = np.linalg.norm(result, axis=1, keepdims=True)
    normalized = result.copy()
    nonzero = norms[:, 0] > 0
    normalized[nonzero] = normalized[nonzero] / norms[nonzero]
    return normalized


def _get_file_name(file_names: list[str] | None, index: int) -> str | None:
    if file_names is None or index >= len(file_names):
        return None
    return file_names[index]


def _build_embedding_input(
    text: str,
    file_name: str | None,
    index: int,
    *,
    input_mode: EmbeddingInputMode,
) -> tuple[str, str | None]:
    """Build the actual embedding input and optional filename-fallback marker."""
    stripped = text.strip() if text else ""
    if input_mode == "with_filename":
        if len(stripped) > FILENAME_FALLBACK_MAX_CHARS:
            if file_name:
                return f"File name: {file_name}\n\nContent:\n{stripped}", None
            return stripped, None

        fallback_name = file_name or f"file_{index}"
        return f"File name: {fallback_name}", fallback_name

    # content_only, and summary (whose text is prepared from document
    # features before this point): embed the text exactly as given.
    if stripped:
        return stripped, None
    return f"file_{index}", f"file_{index}"


def get_embeddings(
    client: OpenAI,
    texts: list[str],
    *,
    config: Config,
    file_names: list[str] | None = None,
    embedding_model: str | None = None,
    input_mode: EmbeddingInputMode = "with_filename",
) -> np.ndarray:
    """
    批量获取文本的 Embedding 向量

    Args:
        client: OpenAI 兼容客户端
        texts: 文本列表
        file_names: 文件名列表（作为分类信号，并用于空文本回退）

    Returns:
        Embedding 向量矩阵 (N, D)
    """
    logger = get_logger()
    model = embedding_model or config.models.embedding
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    logger.debug(t("debug_vectorizing_documents", count=len(texts)))
    logger.debug(t("debug_vectorizing_model", model=model))

    valid_texts = []
    fallback_count = 0
    fallback_names: list[str] = []
    for i, text in enumerate(texts):
        built_input, fallback_name = _build_embedding_input(
            text,
            _get_file_name(file_names, i),
            i,
            input_mode=input_mode,
        )
        valid_texts.append(built_input)
        if fallback_name is not None:
            fallback_count += 1
            fallback_names.append(fallback_name)

    if fallback_count > 0:
        logger.debug(
            t(
                "debug_vector_fallback_names",
                count=fallback_count,
                names=", ".join(fallback_names[:5]),
            )
        )

    # 统计文本长度
    text_lengths = [len(t) for t in valid_texts]
    logger.debug(
        t(
            "debug_vector_text_stats",
            min_length=min(text_lengths),
            max_length=max(text_lengths),
            avg_length=sum(text_lengths) // len(text_lengths),
        )
    )

    batch_size = max(1, int(config.processing.embedding_batch_size))
    embeddings: list[list[float]] = []
    total_tokens = 0
    for start in range(0, len(valid_texts), batch_size):
        batch_embeddings, batch_tokens = _embed_batch(
            client,
            model,
            valid_texts[start : start + batch_size],
        )
        embeddings.extend(batch_embeddings)
        total_tokens += batch_tokens

    result = normalize_embeddings(np.array(embeddings))

    logger.debug(t("debug_vector_dimension", dimension=result.shape[1]))
    logger.debug(t("debug_vector_api_usage", tokens=total_tokens))

    return result


def _is_request_too_large(exc: APIStatusError) -> bool:
    """Return whether a provider rejection may be about the batch size.

    Providers cap inputs per request differently (some at 10, OpenAI at
    2048) and report it as 400 or 413. Transient errors (429, 5xx) are
    already retried by the SDK and are not split.
    """
    return getattr(exc, "status_code", None) in {400, 413}


def _embed_batch(
    client: OpenAI,
    model: str,
    inputs: list[str],
) -> tuple[list[list[float]], int]:
    """Embed one batch, halving it when the provider rejects its size.

    When a batch is rejected, its first half is retried first. If even a
    single input is rejected, the error is not about size and is raised, so a
    bad model name costs about log2(batch) requests, not one per input.
    """
    try:
        response = client.embeddings.create(model=model, input=inputs)
    except APIStatusError as exc:
        if len(inputs) <= 1 or not _is_request_too_large(exc):
            raise
        middle = len(inputs) // 2
        get_logger().debug(
            t(
                "debug_vector_batch_split",
                size=len(inputs),
                status=getattr(exc, "status_code", "-"),
            )
        )
        left, left_tokens = _embed_batch(client, model, inputs[:middle])
        right, right_tokens = _embed_batch(client, model, inputs[middle:])
        return left + right, left_tokens + right_tokens

    items = list(response.data)
    if len(items) != len(inputs):
        raise ValueError(
            "embedding response size mismatch: "
            f"sent {len(inputs)} inputs, received {len(items)} vectors"
        )
    # The API returns an `index` per item; order by it so vectors can never
    # be attached to the wrong file. Stable sort keeps order if it is absent.
    items.sort(key=lambda item: getattr(item, "index", 0) or 0)
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return [item.embedding for item in items], tokens


class ContentTruncator:
    """
    内容截断器：保留首尾，中间采样

    用于将长文档截断到 Embedding 模型的 token 限制内。
    """

    EMBEDDING_MAX_TOKENS = 8192
    CONTENT_BUDGET = 7500  # tokens

    DEFAULT_HEAD_RATIO = 0.6
    DEFAULT_MIDDLE_RATIO = 0.2
    _MIDDLE_MARKER = "\n\n[... middle omitted ...]\n\n"
    _TAIL_MARKER = "\n\n[... omitted ...]\n\n"

    @staticmethod
    def truncate_smart(
        content: str,
        max_chars: int = 15000,
        *,
        head_ratio: float = DEFAULT_HEAD_RATIO,
        middle_ratio: float = DEFAULT_MIDDLE_RATIO,
    ) -> str:
        """
        智能截断：保留开头，按比例采样中段和结尾

        默认策略：
        - 首部 60%：标题、摘要、引言通常在开头
        - 中部 20%：采样中间内容
        - 尾部 20%：结论通常在结尾，但参考文献、索引、版权页也常在结尾

        ``head_ratio`` and ``middle_ratio`` split the budget; the tail gets
        the rest. A part with a zero share is dropped together with its
        marker, so ``head_ratio=1`` keeps only the opening of the document.

        Args:
            content: 原始内容
            max_chars: 最大字符数

        Returns:
            截断后的内容
        """
        if len(content) <= max_chars:
            return content
        head_ratio = min(max(float(head_ratio), 0.0), 1.0)
        middle_ratio = min(max(float(middle_ratio), 0.0), 1.0 - head_ratio)
        tail_ratio = 1.0 - head_ratio - middle_ratio
        has_middle = middle_ratio > 0
        has_tail = tail_ratio > 1e-9

        marker_len = (len(ContentTruncator._MIDDLE_MARKER) if has_middle else 0) + (
            len(ContentTruncator._TAIL_MARKER) if has_tail else 0
        )
        if max_chars <= marker_len + 3:
            return content[:max_chars]

        content_budget = max_chars - marker_len
        head_len = int(content_budget * head_ratio)
        mid_len = int(content_budget * middle_ratio) if has_middle else 0
        tail_len = content_budget - head_len - mid_len if has_tail else 0

        parts = [content[:head_len]]
        if has_middle:
            mid_start = max((len(content) - mid_len) // 2, 0)
            parts.append(ContentTruncator._MIDDLE_MARKER)
            parts.append(content[mid_start : mid_start + mid_len])
        if has_tail:
            parts.append(ContentTruncator._TAIL_MARKER)
            parts.append(content[-tail_len:] if tail_len > 0 else "")
        return "".join(parts)
