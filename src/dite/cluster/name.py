"""Cluster naming and representation helpers."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from openai import APIError, OpenAI
from rich.markup import escape
from sklearn.metrics.pairwise import cosine_distances

from dite.app.config import Config
from dite.app.i18n import get_locale, t
from dite.util.api import (
    AsyncRequestRuntime,
    ChatCompletionRequest,
    ChatCompletionResult,
)
from dite.util.llm import (
    build_chat_completion_kwargs,
    format_api_error,
    should_retry_api_error,
    strip_reasoning_text,
)
from dite.util.log import get_logger

if TYPE_CHECKING:
    from dite.cluster.model import ClusterResult

CLUSTER_NAME_CONTENT_LIMIT = 600
CLUSTER_NAME_EXCERPT_LIMIT = 180
CLUSTER_NAME_OUTPUT_LIMIT = 24
CLUSTER_NAME_MAX_RETRIES = 3
# Warn once at least this share of clusters kept a heuristic name.
CLUSTER_NAME_FALLBACK_WARN_RATIO = 0.5
_CLUSTER_NAME_ERROR_PREVIEW_LIMIT = 300
NON_MERGEABLE_CLUSTER_NAMES = {"", "未命名", "Unnamed"}
PLACEHOLDER_CLUSTER_NAMES = {
    "image",
    "cover",
    "scan",
    "scanned",
    "untitled",
    "unknown",
}
CLUSTER_NAME_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}
# "Name: ...", "**Category:** ...", "名称：..." labels in front of a reply.
_CLUSTER_NAME_LABEL_RE = re.compile(
    r"^(?:(?:category|cluster|folder)\s+)?(?:name|category|topic)\s*[:：]\s*"
    r"|^(?:类别|簇|分类)?(?:名称|名字|名|类别|主题)\s*[:：]\s*",
    re.IGNORECASE,
)
_REASONING_TAG_RE = re.compile(r"</?think(?:ing)?\b", re.IGNORECASE)

# Why a cluster kept a heuristic name instead of the model's answer.
_FALLBACK_FAILED = "failed"
_FALLBACK_EMPTY = "empty"
_FALLBACK_INVALID = "invalid"


class _ClusterName(str):
    """A cluster name that records why the model's answer was not used."""

    fallback_reason: str | None
    error: str

    def __new__(
        cls, value: str, fallback_reason: str | None = None, error: str = ""
    ) -> _ClusterName:
        name = super().__new__(cls, value)
        name.fallback_reason = fallback_reason
        name.error = error
        return name


def _prepare_cluster_name_request_inputs(
    cluster_embeddings: np.ndarray | None,
    sample_contents: list[str],
    sample_names: list[str],
    *,
    top_k: int,
) -> tuple[str, list[str], list[str]]:
    """Select representative samples and build the naming prompt."""
    if cluster_embeddings is not None and len(cluster_embeddings) > 0:
        centroid = cluster_embeddings.mean(axis=0)
        distances = cosine_distances(
            cluster_embeddings, centroid.reshape(1, -1)
        ).flatten()
        top_indices = np.argsort(distances)[:top_k]
        sample_contents = [
            sample_contents[i] for i in top_indices if i < len(sample_contents)
        ]
        sample_names = [sample_names[i] for i in top_indices if i < len(sample_names)]

    compact_samples = []
    for content, file_name in zip(sample_contents, sample_names, strict=False):
        stripped = content.strip()
        if not stripped:
            continue
        compact_samples.append(
            _compact_sample_for_naming(
                file_name,
                stripped[:CLUSTER_NAME_CONTENT_LIMIT],
            )
        )

    prompt = _build_cluster_naming_prompt(compact_samples, sample_names, top_k)
    return prompt, sample_contents, sample_names


def _normalize_cluster_name_text(text: str) -> str:
    """Normalize a raw cluster-name candidate into a single readable line."""
    cleaned = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    cleaned = cleaned.strip()
    cleaned = cleaned.lstrip("#*- ").strip()
    cleaned = cleaned.strip("\"'`")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _looks_like_author_line(text: str) -> bool:
    """Detect typical academic author lists so they are not used as titles."""
    if re.search(r"[\u4e00-\u9fff]", text):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", text)
    if not 4 <= len(words) <= 8:
        return False
    if any(word.casefold() in CLUSTER_NAME_STOPWORDS for word in words):
        return False

    has_separator = any(ch in text for ch in "*,&;")
    if not has_separator and len(words) < 6:
        return False

    return all(
        word[:1].isupper() and any(ch.islower() for ch in word[1:]) for word in words
    )


def _is_invalid_cluster_name(text: str) -> bool:
    """Return whether a cluster-name candidate is empty or low-information."""
    cleaned = _normalize_cluster_name_text(text)
    if not cleaned or cleaned in NON_MERGEABLE_CLUSTER_NAMES:
        return True
    if cleaned.startswith("第") and "页" in cleaned[:6]:
        return True
    if re.fullmatch(r"[\W_]+", cleaned):
        return True
    # Reasoning tags and other markup such as "<think>" or "<|im_end|>".
    if _REASONING_TAG_RE.search(cleaned) or re.fullmatch(r"<[^<>]*>", cleaned):
        return True

    lowered = cleaned.casefold()
    if lowered in PLACEHOLDER_CLUSTER_NAMES:
        return True
    if re.fullmatch(r"(page|slide)\s*\d+", lowered):
        return True
    return bool(_looks_like_author_line(cleaned))


def _cluster_name_from_reply(content: str) -> str:
    """Extract the name candidate from a model reply.

    Reasoning blocks, Markdown emphasis and "Name:" style labels are removed;
    the result may still be invalid and must go through the usual checks.
    """
    answer = strip_reasoning_text(content)
    first_line = next((line for line in answer.splitlines() if line.strip()), "")
    name = _normalize_cluster_name_text(first_line)
    name = _CLUSTER_NAME_LABEL_RE.sub("", name, count=1)
    return _normalize_cluster_name_text(name.rstrip("*"))


def _fallback_name_from_file_name(file_name: str) -> str:
    """Return a readable fallback name from the source file name."""
    stem = Path(file_name).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    cleaned = _normalize_cluster_name_text(stem)
    if _is_invalid_cluster_name(cleaned):
        return ""
    return cleaned[:CLUSTER_NAME_OUTPUT_LIMIT]


def _display_cluster_name(label: int, name: str) -> str:
    """Return a user-facing cluster name without changing cluster structure."""
    cleaned = _normalize_cluster_name_text(name)
    if _is_invalid_cluster_name(cleaned):
        return f"{t('cluster_unnamed_label')}-{label}"
    return cleaned[:CLUSTER_NAME_OUTPUT_LIMIT]


def _cluster_debug_token(index: int) -> str:
    """Return a stable A/B/C-style debug token for cluster ordering."""
    if index < 26:
        return chr(ord("A") + index)
    return _cluster_debug_token(index // 26 - 1) + chr(ord("A") + index % 26)


def _build_cluster_debug_labels(labels: np.ndarray) -> dict[int, str]:
    """Map internal numeric labels to stable debug-facing tokens."""
    cluster_labels = sorted(label for label in set(labels) if label != -1)
    return {
        label: _cluster_debug_token(index) for index, label in enumerate(cluster_labels)
    }


def _extract_title_like_line(content: str) -> str:
    """Pick a short, title-like line from extracted content."""
    for raw_line in content.splitlines():
        line = _normalize_cluster_name_text(raw_line)
        if len(line) < 4 or len(line) > 120:
            continue
        if _is_invalid_cluster_name(line):
            continue
        return line
    return ""


def _compact_sample_for_naming(file_name: str, content: str) -> str:
    """Build a compact naming summary for one representative sample."""
    title = _extract_title_like_line(content) or _fallback_name_from_file_name(
        file_name
    )
    if not title:
        title = Path(file_name).stem
    excerpt = re.sub(r"\s+", " ", content.strip())[:CLUSTER_NAME_EXCERPT_LIMIT]
    if get_locale() == "zh-CN":
        return f"文件名: {file_name}\n标题候选: {title}\n内容片段: {excerpt}"
    return (
        f"File name: {file_name}\nTitle candidate: {title}\nContent excerpt: {excerpt}"
    )


def _heuristic_cluster_name(sample_contents: list[str], sample_names: list[str]) -> str:
    """Fallback to a conservative display label when LLM naming is unavailable."""
    for content in sample_contents:
        title = _extract_title_like_line(content)
        if title:
            return title[:CLUSTER_NAME_OUTPUT_LIMIT]

    for file_name in sample_names:
        fallback_name = _fallback_name_from_file_name(file_name)
        if fallback_name:
            return fallback_name

    return t("cluster_unnamed_label")


def _build_cluster_naming_prompt(
    compact_samples: list[str], sample_names: list[str], top_k: int
) -> str:
    """Build a locale-aware prompt for cluster naming."""
    if get_locale() == "zh-CN":
        forbidden_terms = (
            "「聚类」「分类」「类别」「类型」「封面」「扫描」「文档」"
            "「资料」「合集」「汇编」"
        )
        if compact_samples:
            combined = "\n---\n".join(compact_samples[:top_k])
            return (
                f"以下是属于同一类别的代表文档信息：\n\n{combined}\n\n"
                f"请用2-4个中文词为这个类别命名。\n"
                f"要求：\n"
                f"1. 名称应描述文档的【主题内容】，而非文档的形式或载体\n"
                f"2. 禁止使用以下词语：{forbidden_terms}\n"
                f"3. 优先参考标题候选和文件名中的主题词\n"
                f"4. 直接输出名称，不要解释"
            )

        names = "\n".join(sample_names[:top_k])
        return (
            f"以下文件属于同一类别，请根据文件名推测类别：\n\n{names}\n\n"
            f"请用2-4个中文词为这个类别命名。\n"
            f"要求：\n"
            f"1. 名称应描述文档的【主题内容】，而非文档的形式或载体\n"
            f"2. 禁止使用以下词语：{forbidden_terms}\n"
            f"3. 直接输出名称，不要解释"
        )

    forbidden_terms = (
        '"cluster", "category", "type", "cover", "scan", "document", '
        '"file", "collection", "archive"'
    )
    if compact_samples:
        combined = "\n---\n".join(compact_samples[:top_k])
        return (
            "Here are representative documents from the same category:\n\n"
            f"{combined}\n\n"
            f"Name this category in 2-4 English words.\n"
            f"Requirements:\n"
            "1. The name must describe the topical content, not the document "
            "form or medium\n"
            f"2. Do not use these terms: {forbidden_terms}\n"
            f"3. Prefer topic words from the title candidate and file name\n"
            f"4. Output only the name"
        )

    names = "\n".join(sample_names[:top_k])
    return (
        "These files belong to the same category. Infer the category from "
        f"their file names:\n\n{names}\n\n"
        f"Name this category in 2-4 English words.\n"
        f"Requirements:\n"
        "1. The name must describe the topical content, not the document "
        "form or medium\n"
        f"2. Do not use these terms: {forbidden_terms}\n"
        f"3. Output only the name"
    )


def generate_cluster_name(
    client: OpenAI,
    cluster_embeddings: np.ndarray | None,
    sample_contents: list[str],
    sample_names: list[str],
    *,
    config: Config,
    top_k: int = 5,
    llm_model: str | None = None,
) -> str:
    """Generate a cluster name via LLM with conservative fallbacks."""
    logger = get_logger()
    model = llm_model or config.models.llm
    request_profile = config.request_profiles.cluster_naming
    prompt, sample_contents, sample_names = _prepare_cluster_name_request_inputs(
        cluster_embeddings,
        sample_contents,
        sample_names,
        top_k=top_k,
    )

    last_error = ""
    for attempt in range(CLUSTER_NAME_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                **build_chat_completion_kwargs(
                    client=client,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    profile=request_profile,
                    reasoning_param_style=config.api.reasoning_param_style,
                )
            )
            choice = response.choices[0]
            message = choice.message
            raw_name = message.content or ""
            reasoning_content = getattr(message, "reasoning_content", None)
            answer = strip_reasoning_text(raw_name)
            if not answer:
                logger.debug(
                    t(
                        "debug_cluster_name_empty_response",
                        model=response.model,
                        finish_reason=choice.finish_reason,
                        reasoning_chars=len(reasoning_content or raw_name.strip()),
                    )
                )
                fallback = _heuristic_cluster_name(sample_contents, sample_names)
                logger.debug(t("debug_cluster_name_empty_fallback", fallback=fallback))
                return _ClusterName(fallback, _FALLBACK_EMPTY)

            name = _cluster_name_from_reply(answer)
            if _is_invalid_cluster_name(name):
                fallback = _heuristic_cluster_name(sample_contents, sample_names)
                logger.debug(
                    t("debug_cluster_name_invalid_fallback", fallback=fallback)
                )
                return _ClusterName(fallback, _FALLBACK_INVALID)
            return name[:CLUSTER_NAME_OUTPUT_LIMIT]
        except APIError as exc:
            last_error = format_api_error(exc)
            is_last_attempt = attempt == CLUSTER_NAME_MAX_RETRIES - 1
            if not is_last_attempt and should_retry_api_error(exc):
                logger.debug(
                    t(
                        "debug_cluster_name_retry",
                        attempt=attempt + 1,
                        max_retries=CLUSTER_NAME_MAX_RETRIES,
                        error=last_error,
                    )
                )
                continue
        except Exception as exc:
            last_error = str(exc)
        break

    fallback = _heuristic_cluster_name(sample_contents, sample_names)
    logger.debug(
        t(
            "debug_cluster_name_failed_fallback",
            fallback=fallback,
            error=last_error,
        )
    )
    return _ClusterName(fallback, _FALLBACK_FAILED, last_error)


def _cluster_name_from_async_response(
    response: ChatCompletionResult,
    selected_contents: list[str],
    selected_names: list[str],
    debug_label: str,
) -> str:
    """Return the cluster name for one batched naming response."""
    logger = get_logger()
    timing = {
        "label": debug_label,
        "wait_sec": response.queue_wait_sec,
        "request_sec": response.request_elapsed_sec,
    }
    if response.error is not None:
        fallback = _heuristic_cluster_name(selected_contents, selected_names)
        logger.debug(
            t(
                "debug_cluster_name_async_request_failed",
                error=response.error,
                fallback=fallback,
                **timing,
            )
        )
        error = response.error
        error_text = (
            format_api_error(error) if isinstance(error, APIError) else str(error)
        )
        return _ClusterName(fallback, _FALLBACK_FAILED, error_text)

    answer = strip_reasoning_text(response.content)
    if not answer:
        fallback = _heuristic_cluster_name(selected_contents, selected_names)
        logger.debug(
            t("debug_cluster_name_async_response_empty", fallback=fallback, **timing)
        )
        return _ClusterName(fallback, _FALLBACK_EMPTY)

    name = _cluster_name_from_reply(answer)
    if _is_invalid_cluster_name(name):
        fallback = _heuristic_cluster_name(selected_contents, selected_names)
        logger.debug(
            t("debug_cluster_name_async_response_invalid", fallback=fallback, **timing)
        )
        return _ClusterName(fallback, _FALLBACK_INVALID)
    return name


def _warn_on_cluster_naming_fallbacks(names: list[str]) -> None:
    """Tell the user when many clusters kept heuristic names, and why."""
    reasons = [getattr(name, "fallback_reason", None) for name in names]
    fallback_count = sum(reason is not None for reason in reasons)
    if fallback_count == 0 or fallback_count < (
        len(names) * CLUSTER_NAME_FALLBACK_WARN_RATIO
    ):
        return

    logger = get_logger()
    failed_count = reasons.count(_FALLBACK_FAILED)
    empty_count = reasons.count(_FALLBACK_EMPTY)
    logger.warning(
        t(
            "warning_cluster_naming_fallback",
            fallback=fallback_count,
            total=len(names),
            failed=failed_count,
            empty=empty_count,
            invalid=reasons.count(_FALLBACK_INVALID),
        )
    )
    if failed_count:
        first_error = next(
            getattr(name, "error", "")
            for name, reason in zip(names, reasons, strict=True)
            if reason == _FALLBACK_FAILED
        )
        logger.warning(
            t(
                "warning_cluster_naming_first_error",
                error=escape(first_error[:_CLUSTER_NAME_ERROR_PREVIEW_LIMIT]),
            )
        )
    if empty_count:
        logger.warning(t("warning_cluster_naming_reasoning_hint"))


def merge_clusters_by_name(
    labels: np.ndarray,
    cluster_names: dict[int, str],
) -> tuple[np.ndarray, dict[int, str], int]:
    """Merge clusters that received the same display name."""
    logger = get_logger()
    debug_labels = _build_cluster_debug_labels(labels)

    name_to_labels: dict[str, list[int]] = {}
    for label, name in cluster_names.items():
        if name not in name_to_labels:
            name_to_labels[name] = []
        name_to_labels[name].append(label)

    merged_count = 0
    labels = labels.copy()
    new_cluster_names = {}

    for name, label_list in name_to_labels.items():
        if name.strip() in NON_MERGEABLE_CLUSTER_NAMES:
            for label in label_list:
                new_cluster_names[label] = name
            continue

        if len(label_list) > 1:
            target_label = label_list[0]
            for other_label in label_list[1:]:
                labels[labels == other_label] = target_label
                merged_count += 1
                logger.debug(
                    t(
                        "debug_cluster_merge",
                        source=debug_labels.get(other_label, str(other_label)),
                        target=debug_labels.get(target_label, str(target_label)),
                        name=name,
                    )
                )
            new_cluster_names[target_label] = name
        else:
            new_cluster_names[label_list[0]] = name

    return labels, new_cluster_names, merged_count


def generate_all_cluster_names(
    client: OpenAI,
    result: ClusterResult,
    contents: list[str],
    files: list[Path],
    *,
    config: Config,
    embeddings: np.ndarray | None = None,
    merge_same_name: bool = True,
    llm_model: str | None = None,
    request_runtime: AsyncRequestRuntime | None = None,
) -> ClusterResult:
    """Generate names for all clusters and optionally merge duplicate names."""
    from dite.cluster.model import ClusterResult

    labels = result.labels
    cluster_labels = sorted(int(label) for label in set(labels) if label != -1)
    cluster_names: dict[int, str] = {}
    if not cluster_labels:
        return ClusterResult(
            labels=labels.copy(),
            cluster_names={},
            repaired_mask=result.repaired_mask.copy(),
            metrics=dataclasses.replace(result.metrics),
        )

    debug_labels = _build_cluster_debug_labels(labels)
    cluster_indices_by_label = {label: [] for label in cluster_labels}
    for index, label in enumerate(labels):
        if label == -1:
            continue
        cluster_indices_by_label[int(label)].append(index)

    logger = get_logger()
    naming_outcomes: list[str] = []

    if request_runtime is None:
        _ = client.chat.completions

        def name_cluster(label: int) -> tuple[int, str, int]:
            cluster_indices = cluster_indices_by_label[label]
            cluster_contents = [contents[i] for i in cluster_indices]
            cluster_file_names = [files[i].name for i in cluster_indices]
            cluster_embeddings = None
            if embeddings is not None:
                cluster_embeddings = embeddings[cluster_indices]

            name = generate_cluster_name(
                client,
                cluster_embeddings,
                cluster_contents,
                cluster_file_names,
                config=config,
                llm_model=llm_model,
            )
            naming_outcomes.append(name)
            return label, _display_cluster_name(label, name), len(cluster_indices)

        max_workers = max(
            1,
            min(config.processing.cluster_naming_workers, len(cluster_labels)),
        )
        if max_workers == 1:
            results = [name_cluster(label) for label in cluster_labels]
        else:
            results_by_label: dict[int, tuple[str, int]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_label = {
                    executor.submit(name_cluster, label): label
                    for label in cluster_labels
                }
                for future in concurrent.futures.as_completed(future_to_label):
                    label, display_name, count = future.result()
                    results_by_label[label] = (display_name, count)
            results = [(label, *results_by_label[label]) for label in cluster_labels]
    else:
        request_profile = config.request_profiles.cluster_naming
        requests: list[ChatCompletionRequest] = []
        request_meta: list[tuple[int, list[str], list[str], int]] = []
        model = llm_model or config.models.llm

        for label in cluster_labels:
            cluster_indices = cluster_indices_by_label[label]
            cluster_contents = [contents[i] for i in cluster_indices]
            cluster_file_names = [files[i].name for i in cluster_indices]
            cluster_embeddings = None
            if embeddings is not None:
                cluster_embeddings = embeddings[cluster_indices]

            prompt, selected_contents, selected_names = (
                _prepare_cluster_name_request_inputs(
                    cluster_embeddings,
                    cluster_contents,
                    cluster_file_names,
                    top_k=5,
                )
            )
            request_meta.append(
                (label, selected_contents, selected_names, len(cluster_indices))
            )
            requests.append(
                ChatCompletionRequest(
                    kwargs=build_chat_completion_kwargs(
                        client=client,
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        profile=request_profile,
                        reasoning_param_style=config.api.reasoning_param_style,
                    )
                )
            )

        response_results = request_runtime.run_cluster_naming_batch(requests)
        results = []
        for (label, selected_contents, selected_names, count), response in zip(
            request_meta,
            response_results,
            strict=False,
        ):
            name = _cluster_name_from_async_response(
                response,
                selected_contents,
                selected_names,
                debug_labels.get(label, str(label)),
            )
            naming_outcomes.append(name)
            results.append((label, _display_cluster_name(label, name), count))

    _warn_on_cluster_naming_fallbacks(naming_outcomes)
    for label, display_name, count in results:
        cluster_names[label] = display_name
        logger.debug(
            t(
                "debug_cluster_name_result",
                label=debug_labels.get(label, str(label)),
                name=display_name,
                count=count,
            )
        )

    merged_count = 0
    updated_labels = labels.copy()
    updated_names = cluster_names
    if merge_same_name:
        updated_labels, updated_names, merged_count = merge_clusters_by_name(
            labels, cluster_names
        )
    updated_metrics = dataclasses.replace(
        result.metrics,
        name_clusters_merged=merged_count,
    )
    return ClusterResult(
        labels=updated_labels,
        cluster_names=updated_names,
        repaired_mask=result.repaired_mask.copy(),
        metrics=updated_metrics,
    )
