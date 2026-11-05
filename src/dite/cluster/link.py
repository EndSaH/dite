"""Candidate edge and component builders for clustering V2."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from dite.cluster.model import CandidateComponent, CandidateEdge
from dite.doc import DocumentFeatures
from dite.doc.build import filename_title, is_generic_title, meaningful_lines

_SIMILARITY_BLOCK_ROWS = 1024
# Identical content forces a ``must_link``, so it must be more than a shared
# scanner footer or cover line. Counted over distinct cleaned lines, so a
# footer repeated on every page counts once.
NEAR_DUPLICATE_MIN_CHARS = 200


def _duplicate_key(feature: DocumentFeatures) -> str:
    """Return the cleaned content compared for identical-content edges.

    Placeholders, page numbers and layout are ignored, so two copies of a
    document still match when one has an extra ``<!-- image -->`` or different
    line wrapping. Returns ``""`` (never a duplicate) when too little content
    is left: two scans that Docling turned into ``<!-- image -->`` share that
    text whatever they show.
    """
    if feature.quality_flags.filename_dominant:
        return ""
    lines = meaningful_lines(feature.content_text)
    if sum(len(line) for line in dict.fromkeys(lines)) < NEAR_DUPLICATE_MIN_CHARS:
        return ""
    return " ".join(lines)


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


def _content_similarity_pairs(
    document_features: list[DocumentFeatures],
    threshold: float,
) -> set[tuple[int, int]] | None:
    """Pairs whose embedding cosine may reach ``threshold``, via one matmul.

    Returns ``None`` when embeddings cannot be stacked (missing or mixed
    dimensions), in which case callers fall back to checking every pair.
    Filename-dominant documents are excluded: their vectors encode the file
    name, not content.
    """
    vectors = [feature.content_embedding for feature in document_features]
    present = [
        index
        for index, (feature, vector) in enumerate(
            zip(document_features, vectors, strict=True)
        )
        if vector is not None
        and np.asarray(vector).size > 0
        and not feature.quality_flags.filename_dominant
    ]
    if len(present) < 2:
        return set()
    shapes = {np.asarray(vectors[index]).shape for index in present}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        return None
    matrix = np.stack(
        [np.asarray(vectors[index], dtype=np.float32) for index in present]
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    matrix[nonzero] = matrix[nonzero] / norms[nonzero]
    pairs: set[tuple[int, int]] = set()
    # Row blocks keep memory at block x n instead of n x n. A small tolerance
    # keeps borderline pairs; the exact pairwise value decides afterwards, so
    # results match the pairwise computation.
    for start in range(0, matrix.shape[0], _SIMILARITY_BLOCK_ROWS):
        block = matrix[start : start + _SIMILARITY_BLOCK_ROWS] @ matrix.T
        rows, cols = np.nonzero(block >= threshold - 1e-6)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            left = start + row
            if col > left:
                pairs.add((present[left], present[col]))
    return pairs


def _index_pairs(groups: dict[object, list[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(set(members))
        for position, left in enumerate(ordered):
            for right in ordered[position + 1 :]:
                pairs.add((left, right))
    return pairs


def _candidate_pairs(
    document_features: list[DocumentFeatures],
    *,
    token_sets: list[set[str]],
    title_sets: list[set[str]],
    duplicate_keys: list[str],
    min_filename_token_overlap: float,
    min_content_similarity: float,
) -> list[tuple[int, int]]:
    """Every pair that could produce an edge, found without an O(n^2) loop.

    Identical contents and shared titles come from hash groups, filename
    overlap from a token index (Jaccard > 0 needs a shared token), content
    similarity from a matrix product. The per-pair rules in
    ``build_candidate_edges`` still make the final decision.
    """
    count = len(document_features)
    content_pairs = _content_similarity_pairs(document_features, min_content_similarity)
    # Non-positive thresholds accept pairs with no shared token or a zero
    # similarity (missing embeddings), so every pair is a candidate.
    if (
        content_pairs is None
        or min_filename_token_overlap <= 0
        or min_content_similarity <= 0
    ):
        return [
            (left, right) for left in range(count) for right in range(left + 1, count)
        ]

    by_content: dict[object, list[int]] = defaultdict(list)
    by_title: dict[object, list[int]] = defaultdict(list)
    for index in range(count):
        if duplicate_keys[index]:
            by_content[duplicate_keys[index]].append(index)
        for title in title_sets[index]:
            by_title[title].append(index)

    pairs = content_pairs | _index_pairs(by_content) | _index_pairs(by_title)
    pairs |= _filename_overlap_pairs(token_sets, min_filename_token_overlap)
    return sorted(pairs)


def _filename_overlap_pairs(
    token_sets: list[set[str]],
    threshold: float,
) -> set[tuple[int, int]]:
    """Pairs whose token Jaccard may reach ``threshold`` (prefix filtering).

    Tokens are ordered from rarest to most common. Two sets with Jaccard
    >= t must share a token among the first ``|A| - ceil(t*|A|) + 1`` tokens
    of each, so only those prefixes are indexed. Common tokens such as
    "notes" or "2024" then never pair up every file in the folder.
    """
    frequency: Counter[str] = Counter(
        token for tokens in token_sets for token in tokens
    )
    by_prefix_token: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(token_sets):
        if not tokens:
            continue
        ordered = sorted(tokens, key=lambda token: (frequency[token], token))
        prefix_length = len(ordered) - math.ceil(threshold * len(ordered)) + 1
        for token in ordered[: max(1, prefix_length)]:
            by_prefix_token[token].append(index)

    pairs: set[tuple[int, int]] = set()
    for members in by_prefix_token.values():
        if len(members) < 2:
            continue
        for position, left in enumerate(members):
            left_size = len(token_sets[left])
            for right in members[position + 1 :]:
                right_size = len(token_sets[right])
                # Jaccard <= min/max of the set sizes; skip hopeless pairs.
                if min(left_size, right_size) < threshold * max(left_size, right_size):
                    continue
                pairs.add((min(left, right), max(left, right)))
    return pairs


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
    own. Edges are returned in (source index, target index) order.
    """
    edges: list[CandidateEdge] = []
    token_sets = [set(feature.file_name_tokens) for feature in document_features]
    title_sets = [_content_title_keys(feature) for feature in document_features]
    duplicate_keys = [_duplicate_key(feature) for feature in document_features]

    for index, right_index in _candidate_pairs(
        document_features,
        token_sets=token_sets,
        title_sets=title_sets,
        duplicate_keys=duplicate_keys,
        min_filename_token_overlap=min_filename_token_overlap,
        min_content_similarity=min_content_similarity,
    ):
        left = document_features[index]
        right = document_features[right_index]
        left_tokens = token_sets[index]
        right_tokens = token_sets[right_index]
        left_key = duplicate_keys[index]
        shared_titles = title_sets[index] & title_sets[right_index]
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

        if left_key and left_key == duplicate_keys[right_index]:
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
        elif similarity >= min_filename_token_overlap and left_tokens and right_tokens:
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

        hard_constraint = "must_link" if edge_type == "near_duplicate" else None

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
