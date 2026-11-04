"""Conservative rule-based adjudication for clustering V2."""

from __future__ import annotations

import json

from openai import OpenAI

from dite.app.config import Config
from dite.cluster.model import (
    AdjudicationDecision,
    AdjudicationRequest,
    CandidateComponent,
    CandidateEdge,
)
from dite.util.api import ChatCompletionRequest
from dite.util.llm import build_chat_completion_kwargs

DEFAULT_REQUEST_SCORE_THRESHOLD = 0.85
DEFAULT_EDGE_MERGE_THRESHOLD = 0.92


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

    Edge reviews scoring at least ``edge_merge_threshold`` are promoted to
    ``merge_edge``; everything else stays a review-only decision.
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
) -> list[AdjudicationDecision]:
    """Apply optional LLM adjudication for review-only edge requests."""
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
                            "content": _adjudication_prompt(request),
                        }
                    ],
                    profile=config.request_profiles.cluster_naming,
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


def _adjudication_prompt(request: AdjudicationRequest) -> str:
    evidence = "\n".join(f"- {item}" for item in request.evidence_bundle) or "-"
    quality_guard = ", ".join(request.quality_guard) or "-"
    score = "-" if request.score is None else f"{request.score:.3f}"
    return (
        "Return strict JSON with keys should_merge, confidence, reason.\n"
        f"Request type: {request.request_type}\n"
        f"Trigger: {request.trigger_reason}\n"
        f"Score: {score}\n"
        f"Quality guard: {quality_guard}\n"
        f"Subjects: {', '.join(request.subjects)}\n"
        f"Evidence:\n{evidence}\n"
    )


def _parse_llm_adjudication(content: str) -> dict[str, object] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    should_merge = bool(payload.get("should_merge"))
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
