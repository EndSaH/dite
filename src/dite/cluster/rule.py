"""Conservative rule-based adjudication for clustering V2."""

from __future__ import annotations

import re

from openai import OpenAI

from dite.app.config import Config
from dite.app.i18n import get_locale
from dite.cluster.model import (
    CONTENT_BACKED_EDGE_TYPES,
    AdjudicationDecision,
    AdjudicationRequest,
    CandidateComponent,
    CandidateEdge,
)
from dite.doc import DocumentFeatures
from dite.doc.build import filename_title, strip_markup
from dite.util.api import ChatCompletionRequest
from dite.util.llm import (
    build_chat_completion_kwargs,
    parse_json_bool,
    parse_json_object,
)

DEFAULT_REQUEST_SCORE_THRESHOLD = 0.85
DEFAULT_EDGE_MERGE_THRESHOLD = 0.92
_ADJUDICATION_EXCERPT_CHARS = 600


def build_adjudication_requests(
    candidate_edges: list[CandidateEdge],
    candidate_components: list[CandidateComponent],
    *,
    score_threshold: float = DEFAULT_REQUEST_SCORE_THRESHOLD,
) -> list[AdjudicationRequest]:
    """Build minimal adjudication requests from currently ambiguous edge cases.

    Only edges whose score reaches ``score_threshold`` become edge reviews.
    """
    requests: list[AdjudicationRequest] = []
    for component in candidate_components:
        if len(component.member_file_ids) < 2:
            continue
        requests.append(
            AdjudicationRequest(
                request_id=f"component:{component.component_id}",
                request_type="component_review",
                subjects=component.member_file_ids,
                evidence_bundle=component.formation_evidence,
                trigger_reason="strong_component_candidate",
            )
        )
    for edge in candidate_edges:
        if edge.score < score_threshold:
            continue
        requests.append(
            AdjudicationRequest(
                request_id=f"edge:{edge.source_id}->{edge.target_id}",
                request_type="edge_review",
                subjects=[edge.source_id, edge.target_id],
                evidence_bundle=edge.evidence,
                trigger_reason=edge.edge_type,
                score=edge.score,
                quality_guard=edge.quality_guard,
            )
        )
    return requests


def apply_rule_adjudication(
    requests: list[AdjudicationRequest],
    *,
    edge_merge_threshold: float = DEFAULT_EDGE_MERGE_THRESHOLD,
) -> list[AdjudicationDecision]:
    """Apply conservative rule-only adjudication to current requests.

    Content-backed edge reviews scoring at least ``edge_merge_threshold`` are
    promoted to ``merge_edge``. Title and filename edges always stay
    review-only: a shared first line or file name is a hint, not proof.
    """
    decisions: list[AdjudicationDecision] = []
    for request in requests:
        if request.request_type == "component_review":
            decisions.append(
                AdjudicationDecision(
                    request_id=request.request_id,
                    decision="keep_component",
                    confidence=1.0,
                    reason=(
                        "Strong candidate component preserved for downstream "
                        "clustering."
                    ),
                    supporting_evidence=request.evidence_bundle,
                )
            )
            continue
        if (
            request.request_type == "edge_review"
            and request.trigger_reason in CONTENT_BACKED_EDGE_TYPES
            and request.score is not None
            and request.score >= edge_merge_threshold
            and "filename_dominant" not in request.quality_guard
        ):
            decisions.append(
                AdjudicationDecision(
                    request_id=request.request_id,
                    decision="merge_edge",
                    confidence=min(request.score, 1.0),
                    reason="High-confidence semantic edge promoted to merge decision.",
                    supporting_evidence=request.evidence_bundle,
                )
            )
            continue
        decisions.append(
            AdjudicationDecision(
                request_id=request.request_id,
                decision="review_edge",
                confidence=0.5,
                reason=(
                    "Strong edge retained as evidence for later merge or "
                    "assignment review."
                ),
                supporting_evidence=request.evidence_bundle,
            )
        )
    return decisions


def apply_llm_adjudication(
    requests: list[AdjudicationRequest],
    *,
    client: OpenAI,
    config: Config,
    request_runtime,
    document_features: list[DocumentFeatures] | None = None,
) -> list[AdjudicationDecision]:
    """Apply optional LLM adjudication for edge review requests.

    The prompt carries each document's title, summary, keywords and a content
    excerpt from ``document_features``; without content the model could only
    guess from file paths.
    """
    feature_by_id = {feature.file_id: feature for feature in (document_features or [])}
    llm_requests: list[ChatCompletionRequest] = []
    request_index: list[AdjudicationRequest] = []
    for request in requests:
        if request.request_type != "edge_review":
            continue
        llm_requests.append(
            ChatCompletionRequest(
                kwargs=build_chat_completion_kwargs(
                    client=client,
                    model=config.models.llm,
                    messages=[
                        {
                            "role": "user",
                            "content": _adjudication_prompt(request, feature_by_id),
                        }
                    ],
                    profile=config.request_profiles.cluster_adjudication,
                    response_format={"type": "json_object"},
                )
            )
        )
        request_index.append(request)

    if not llm_requests:
        return []

    responses = request_runtime.run_cluster_naming_batch(llm_requests)
    decisions: list[AdjudicationDecision] = []
    for request, response in zip(request_index, responses, strict=True):
        if response.error is not None or not response.content:
            decisions.append(
                AdjudicationDecision(
                    request_id=request.request_id,
                    decision="review_edge",
                    confidence=0.5,
                    reason="LLM adjudication failed; fell back to review state.",
                    supporting_evidence=request.evidence_bundle,
                    model_used="llm",
                    fallback_used=True,
                )
            )
            continue
        parsed = _parse_llm_adjudication(response.content)
        if parsed is None:
            decisions.append(
                AdjudicationDecision(
                    request_id=request.request_id,
                    decision="review_edge",
                    confidence=0.5,
                    reason="LLM adjudication returned invalid JSON; fell back.",
                    supporting_evidence=request.evidence_bundle,
                    model_used="llm",
                    fallback_used=True,
                )
            )
            continue
        decisions.append(
            AdjudicationDecision(
                request_id=request.request_id,
                decision="merge_edge" if parsed["should_merge"] else "review_edge",
                confidence=parsed["confidence"],
                reason=parsed["reason"],
                supporting_evidence=request.evidence_bundle,
                model_used="llm",
                fallback_used=False,
            )
        )
    return decisions


def _describe_subject(
    label: str,
    subject_id: str,
    feature: DocumentFeatures | None,
) -> str:
    if feature is None:
        return f"Document {label}: {subject_id}\n(no extracted content available)"
    stem_title = filename_title(feature.stem)
    content_titles = [
        title for title in feature.title_candidates if title != stem_title
    ]
    excerpt = re.sub(r"\s+", " ", strip_markup(feature.content_text)).strip()
    lines = [
        f"Document {label}",
        f"File: {feature.name}",
        f"Folder: {feature.path.parent.as_posix() or '.'}",
        f"Title: {content_titles[0] if content_titles else '-'}",
    ]
    if feature.summary and feature.summary not in content_titles:
        lines.append(f"Summary: {feature.summary}")
    if feature.keywords:
        lines.append(f"Keywords: {', '.join(feature.keywords[:8])}")
    if feature.quality_flags.filename_dominant:
        lines.append("Note: text extraction failed; only the file name is known.")
    lines.append(f"Excerpt: {excerpt[:_ADJUDICATION_EXCERPT_CHARS] or '(empty)'}")
    return "\n".join(lines)


def _adjudication_prompt(
    request: AdjudicationRequest,
    feature_by_id: dict[str, DocumentFeatures] | None = None,
) -> str:
    feature_by_id = feature_by_id or {}
    evidence = "\n".join(f"- {item}" for item in request.evidence_bundle) or "-"
    quality_guard = ", ".join(request.quality_guard) or "-"
    score = "-" if request.score is None else f"{request.score:.3f}"
    subjects = "\n\n".join(
        _describe_subject(label, subject_id, feature_by_id.get(subject_id))
        for label, subject_id in zip("ABCDEFGH", request.subjects, strict=False)
    )
    reason_language = (
        "Write the reason in Simplified Chinese."
        if get_locale() == "zh-CN"
        else "Write the reason in English."
    )
    return (
        "You are helping organize a folder of files into topical subfolders.\n"
        "Decide whether the documents below belong in the same subfolder.\n"
        "Answer should_merge=true only when they share a specific subject: the "
        "same project, course, topic, or document family. A shared file name, "
        "a generic heading, the same file format, or only a broad domain is "
        "not enough.\n"
        "Return strict JSON with keys should_merge (boolean), confidence "
        "(number from 0 to 1) and reason (one short sentence). "
        f"{reason_language}\n\n"
        f"Candidate signal: {request.trigger_reason} (score {score})\n"
        f"Quality guard: {quality_guard}\n"
        f"Evidence:\n{evidence}\n\n"
        f"{subjects}\n"
    )


def _parse_llm_adjudication(content: str) -> dict[str, object] | None:
    payload = parse_json_object(content)
    if payload is None or "should_merge" not in payload:
        return None
    should_merge = parse_json_bool(payload.get("should_merge"))
    confidence_raw = payload.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    reason = str(payload.get("reason", "")).strip() or "LLM adjudication result"
    return {
        "should_merge": should_merge,
        "confidence": max(0.0, min(confidence, 1.0)),
        "reason": reason,
    }
