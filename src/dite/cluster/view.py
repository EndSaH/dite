"""Cluster representation builders for clustering V2."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
from openai import OpenAI

from dite.app.config import Config
from dite.app.i18n import get_locale
from dite.doc.build import strip_markup
from dite.util.api import ChatCompletionRequest
from dite.util.llm import build_chat_completion_kwargs, parse_json_object

from .model import ClusterRepresentation

_REPRESENTATIVE_LIMIT = 3
_REPRESENTATION_EXCERPT_CHARS = 300


def _representative_order(features) -> list[int]:
    """Order cluster members from most to least central.

    Members are ranked by cosine similarity to the cluster centroid when every
    member has a content embedding; otherwise the original order is kept.
    """
    embeddings = [feature.content_embedding for feature in features]
    if not features or any(embedding is None for embedding in embeddings):
        return list(range(len(features)))
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(features):
        return list(range(len(features)))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    centroid = normalized.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0.0:
        return list(range(len(features)))
    similarity = normalized @ (centroid / centroid_norm)
    # Stable sort keeps the original order for ties.
    return sorted(range(len(features)), key=lambda index: -float(similarity[index]))


def build_cluster_representations(
    *,
    labels,
    cluster_names,
    document_features,
    config: Config | None = None,
    client: OpenAI | None = None,
    request_runtime=None,
) -> dict[int, ClusterRepresentation]:
    """Build conservative cluster representations from current document features."""
    cluster_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if int(label) == -1:
            continue
        cluster_indices.setdefault(int(label), []).append(index)

    representations: dict[int, ClusterRepresentation] = {}
    central_features_by_label = {}
    for label, indices in cluster_indices.items():
        features = [document_features[index] for index in indices]
        order = _representative_order(features)
        central_features = [features[index] for index in order]
        central_features_by_label[label] = central_features
        summaries = [feature.summary for feature in central_features if feature.summary]
        topics = [feature.topic for feature in features if feature.topic]
        domains = [feature.domain for feature in features if feature.domain]
        keywords = [
            keyword for feature in features for keyword in feature.keywords if keyword
        ]
        top_keywords = [keyword for keyword, _count in Counter(keywords).most_common(5)]
        topic = Counter(topics).most_common(1)[0][0] if topics else ""
        domain = Counter(domains).most_common(1)[0][0] if domains else ""
        summary = summaries[0] if summaries else ""
        evidence_parts = []
        if topic:
            evidence_parts.append(f"topic={topic}")
        if domain:
            evidence_parts.append(f"domain={domain}")
        if top_keywords:
            evidence_parts.append(f"keywords={', '.join(top_keywords[:3])}")

        representations[label] = ClusterRepresentation(
            cluster_id=label,
            name=cluster_names.get(label, ""),
            summary=summary,
            keywords=top_keywords,
            topic=topic,
            domain=domain,
            representative_file_ids=(
                [central_features[0].file_id] if central_features else []
            ),
            evidence_summary="; ".join(evidence_parts),
        )

    if (
        config is not None
        and client is not None
        and request_runtime is not None
        and config.cluster_representation.mode == "llm_enhanced"
        and representations
    ):
        _apply_llm_representations(
            representations,
            central_features_by_label=central_features_by_label,
            config=config,
            client=client,
            request_runtime=request_runtime,
        )

    return representations


def _apply_llm_representations(
    representations: dict[int, ClusterRepresentation],
    *,
    central_features_by_label,
    config: Config,
    client: OpenAI,
    request_runtime,
) -> None:
    """Enrich deterministic representations with one batched LLM pass."""
    labels = sorted(representations)
    requests = []
    for label in labels:
        representation = representations[label]
        central_features = central_features_by_label[label]
        prompt = _cluster_representation_prompt(
            cluster_name=representation.name,
            summaries=[
                feature.summary for feature in central_features if feature.summary
            ],
            topics=[feature.topic for feature in central_features if feature.topic],
            domains=[feature.domain for feature in central_features if feature.domain],
            keywords=representation.keywords,
            file_names=[Path(feature.file_id).name for feature in central_features],
            excerpts=[
                _excerpt(feature.content_text)
                for feature in central_features[:_REPRESENTATIVE_LIMIT]
            ],
            member_count=len(central_features),
        )
        requests.append(
            ChatCompletionRequest(
                kwargs=build_chat_completion_kwargs(
                    client=client,
                    model=config.models.llm,
                    messages=[{"role": "user", "content": prompt}],
                    profile=config.request_profiles.cluster_representation,
                    response_format={"type": "json_object"},
                )
            )
        )

    responses = request_runtime.run_cluster_naming_batch(requests)
    for label, response in zip(labels, responses, strict=False):
        if response.error is not None or not response.content:
            continue
        llm_representation = _parse_llm_cluster_representation(response.content)
        if llm_representation is None:
            continue
        representation = representations[label]
        representation.summary = llm_representation["summary"] or representation.summary
        representation.topic = llm_representation["topic"] or representation.topic
        representation.domain = llm_representation["domain"] or representation.domain
        if llm_representation["keywords"]:
            representation.keywords = llm_representation["keywords"]
        if llm_representation["evidence_summary"]:
            representation.evidence_summary = llm_representation["evidence_summary"]


def _excerpt(content: str) -> str:
    text = re.sub(r"\s+", " ", strip_markup(content or "")).strip()
    return text[:_REPRESENTATION_EXCERPT_CHARS]


def _cluster_representation_prompt(
    *,
    cluster_name: str,
    summaries: list[str],
    topics: list[str],
    domains: list[str],
    keywords: list[str],
    file_names: list[str],
    excerpts: list[str] | None = None,
    member_count: int | None = None,
) -> str:
    joined_summaries = "\n".join(f"- {summary}" for summary in summaries[:3]) or "-"
    joined_topics = ", ".join(topics[:5]) or "-"
    joined_domains = ", ".join(domains[:5]) or "-"
    joined_keywords = ", ".join(keywords[:8]) or "-"
    joined_files = ", ".join(file_names[:5]) or "-"
    joined_excerpts = (
        "\n".join(f"- {excerpt}" for excerpt in (excerpts or []) if excerpt) or "-"
    )
    language = "Simplified Chinese" if get_locale() == "zh-CN" else "English"
    return (
        "Describe this group of documents for someone browsing an organized "
        "folder.\n"
        "Return strict JSON with keys summary (one sentence), topic (a few "
        "words), domain (one word), keywords (list of up to 5 strings) and "
        "evidence_summary (one sentence on what the documents share). "
        f"Write the values in {language}.\n"
        f"Cluster name: {cluster_name or '-'}\n"
        f"Member count: {member_count if member_count is not None else '-'}\n"
        f"Representative files: {joined_files}\n"
        f"Summaries:\n{joined_summaries}\n"
        f"Topics: {joined_topics}\n"
        f"Domains: {joined_domains}\n"
        f"Keywords: {joined_keywords}\n"
        f"Excerpts from the most central documents:\n{joined_excerpts}\n"
    )


def _parse_llm_cluster_representation(content: str) -> dict[str, object] | None:
    payload = parse_json_object(content)
    if payload is None:
        return None
    raw_keywords = payload.get("keywords") or []
    if isinstance(raw_keywords, str):
        raw_keywords = re.split(r"[,，;；]", raw_keywords)
    if not isinstance(raw_keywords, list):
        raw_keywords = []
    return {
        "summary": str(payload.get("summary", "")).strip(),
        "topic": str(payload.get("topic", "")).strip(),
        "domain": str(payload.get("domain", "")).strip(),
        "keywords": [str(item).strip() for item in raw_keywords if str(item).strip()],
        "evidence_summary": str(payload.get("evidence_summary", "")).strip(),
    }
