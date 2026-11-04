"""Candidate edge and component builders for clustering V2."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dite.cluster.model import CandidateComponent, CandidateEdge
from dite.doc import DocumentFeatures
from dite.doc.build import filename_title, is_generic_title


def _content_title_keys(feature: DocumentFeatures) -> set[str]:
    """Return comparable content-derived titles for one document.

    The file-stem fallback title is excluded: two ``lecture1.pdf`` files from
    different courses share a stem, not a title. Filename overlap is scored
    separately as ``filename_similarity``.
    """
    stem_key = filename_title(feature.stem).casefold()
    keys: set[str] = set()
    for title in feature.title_candidates:
        key = title.casefold().strip()
        if not key or key == stem_key or is_generic_title(key):
            continue
        keys.add(key)
    return keys


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    union = left | right
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.size == 0 or right_array.size == 0:
        return 0.0
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def build_candidate_edges(
    document_features: list[DocumentFeatures],
    *,
    min_filename_token_overlap: float = 0.8,
    min_content_similarity: float = 0.92,
) -> list[CandidateEdge]:
    """Build a conservative first-pass candidate edge set.

    Edge types are checked in order of evidence strength: identical content,
    content-embedding similarity, shared content title, filename overlap. Only
    the first two are content-backed (see ``CONTENT_BACKED_EDGE_TYPES``); title
    and filename edges are kept as evidence but never force a merge on their
    own.
    """
    edges: list[CandidateEdge] = []
    token_sets = [set(feature.file_name_tokens) for feature in document_features]
    title_sets = [_content_title_keys(feature) for feature in document_features]
    stripped_contents = [feature.content_text.strip() for feature in document_features]

    for index, left in enumerate(document_features):
        left_tokens = token_sets[index]
        left_titles = title_sets[index]
        left_content = stripped_contents[index]

        for right_index in range(index + 1, len(document_features)):
            right = document_features[right_index]
            right_tokens = token_sets[right_index]
            shared_titles = left_titles & title_sets[right_index]
            similarity = _jaccard(left_tokens, right_tokens)
            # A filename-dominant document was embedded from its file name, so
            # its cosine similarity is filename evidence, not content evidence.
            content_comparable = not (
                left.quality_flags.filename_dominant
                or right.quality_flags.filename_dominant
            )
            content_similarity = (
                _cosine_similarity(left.content_embedding, right.content_embedding)
                if content_comparable
                else 0.0
            )

            evidence: list[str] = []
            edge_type = None
            score = 0.0

            if left_content and left_content == stripped_contents[right_index]:
                edge_type = "near_duplicate"
                score = max(content_similarity, 1.0)
                evidence.append("identical_content")
            elif content_similarity >= min_content_similarity:
                edge_type = "content_similarity"
                score = content_similarity
                evidence.append(f"content_similarity:{content_similarity:.3f}")
                if shared_titles:
                    evidence.append(f"title_match:{sorted(shared_titles)[0]}")
            elif shared_titles:
                edge_type = "title_match"
                score = 1.0
                evidence.append(f"title_match:{sorted(shared_titles)[0]}")
            elif (
                similarity >= min_filename_token_overlap
                and left_tokens
                and right_tokens
            ):
                edge_type = "filename_similarity"
                score = similarity
                evidence.append(
                    f"filename_token_overlap:{sorted(left_tokens & right_tokens)}"
                )

            if edge_type is None:
                continue

            quality_guard = []
            if left.quality_flags.short_text or right.quality_flags.short_text:
                quality_guard.append("short_text")
            if (
                left.quality_flags.filename_dominant
                or right.quality_flags.filename_dominant
            ):
                quality_guard.append("filename_dominant")

            hard_constraint = None
            if edge_type == "near_duplicate":
                hard_constraint = "must_link"

            edges.append(
                CandidateEdge(
                    source_id=left.file_id,
                    target_id=right.file_id,
                    edge_type=edge_type,
                    score=score,
                    evidence=evidence,
                    hard_constraint=hard_constraint,
                    quality_guard=quality_guard,
                )
            )

    return edges


def build_candidate_components(
    document_features: list[DocumentFeatures],
    edges: list[CandidateEdge],
    *,
    min_edge_score: float = 0.9,
) -> list[CandidateComponent]:
    """Build conservative connected components from strong edges."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    component_type_by_file_id: dict[str, str] = {}
    for edge in edges:
        if edge.score < min_edge_score:
            continue
        if edge.hard_constraint == "must_link" or edge.edge_type == "near_duplicate":
            component_type_by_file_id.setdefault(edge.source_id, "near_duplicate_group")
            component_type_by_file_id.setdefault(edge.target_id, "near_duplicate_group")
        else:
            component_type_by_file_id.setdefault(
                edge.source_id, "strong_semantic_group"
            )
            component_type_by_file_id.setdefault(
                edge.target_id, "strong_semantic_group"
            )
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)

    visited: set[str] = set()
    file_ids = [feature.file_id for feature in document_features]
    components: list[CandidateComponent] = []
    component_index = 0

    for file_id in file_ids:
        if file_id in visited or file_id not in adjacency:
            continue
        stack = [file_id]
        members: list[str] = []
        visited.add(file_id)
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        if len(members) < 2:
            continue
        component_index += 1
        member_types = {
            component_type_by_file_id.get(member, "strong_semantic_group")
            for member in members
        }
        component_type = (
            "near_duplicate_group"
            if member_types == {"near_duplicate_group"}
            else "strong_semantic_group"
        )
        components.append(
            CandidateComponent(
                component_id=f"component-{component_index}",
                member_file_ids=sorted(members),
                component_type=component_type,
                formation_evidence=[component_type],
                confidence=1.0,
            )
        )

    return components
