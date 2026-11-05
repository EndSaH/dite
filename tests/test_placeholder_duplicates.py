"""Placeholder-only text must not count as document content.

Docling emits ``<!-- image -->`` for every picture and nothing else for a
scanned PDF without OCR or an image-only DOCX/PPTX. Two such files share the
same "text" whatever they show, so neither identical text nor the cosine of
their (file name + placeholder) vectors says anything about their topic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dite.app.config import Config
from dite.cluster.link import build_candidate_components, build_candidate_edges
from dite.cluster.model import CONTENT_BACKED_EDGE_TYPES, ClusterMetrics, ClusterResult
from dite.cluster.stage import (
    build_canonical_cluster_stage,
    cluster_canonical_documents,
)
from dite.doc import build_document_features
from dite.flow.model import PipelineOptions

_BODY = (
    "Quarterly maintenance report for the north warehouse.\n"
    "The loading dock lift was serviced on 3 March and passed inspection.\n"
    "Two forklifts need new batteries before the summer peak; the order is "
    "placed and delivery is expected within six weeks.\n"
    "Roof leak above aisle 14 was patched; monitor after heavy rain.\n"
)
_SCAN = "<!-- image -->\n\n1\n\n<!-- image -->\n\n2\n\n<!-- image -->\n\n3\n"


class _FailedReport:
    primary_success = False
    selected_source = "primary"
    primary_extractor = "docling"


def _fixed_labels(labels: list[int]):
    def _cluster(embeddings, **kwargs):
        del kwargs
        assert embeddings.shape[0] == len(labels)
        return ClusterResult(
            labels=np.array(labels, dtype=int),
            cluster_names={},
            repaired_mask=np.zeros(len(labels), dtype=bool),
            metrics=ClusterMetrics(initial_clusters=len(set(labels) - {-1})),
        )

    return _cluster


def _run_stage(
    files: list[Path],
    contents: list[str],
    labels: list[int],
    *,
    embeddings: np.ndarray | None = None,
    reports: list | None = None,
    cluster_mode: str | None = None,
):
    vectors = np.eye(len(files), dtype=np.float32) if embeddings is None else embeddings
    return build_canonical_cluster_stage(
        files=files,
        contents=contents,
        file_hashes=[f"hash-{index}" for index in range(len(files))],
        file_reports=reports or [None] * len(files),
        canonical_indices=list(range(len(files))),
        options=PipelineOptions(cluster_mode=cluster_mode),
        config=Config(),
        client=None,
        request_runtime=None,
        vectorize=lambda files_arg, hashes, texts, options: vectors,
        expand_noise_repaired_count=lambda *args: 0,
        cluster_documents_fn=_fixed_labels(labels),
    )


def _features(contents: dict[str, str], *, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = []
    for path, content in contents.items():
        feature = build_document_features(Path(path), content)
        feature.content_embedding = rng.normal(size=16).astype(np.float32)
        features.append(feature)
    return features


def test_placeholder_only_text_is_flagged_as_filename_evidence() -> None:
    for content in (
        "<!-- image -->",
        "<!-- image -->\n\n<!-- image -->",
        _SCAN,
        "<!-- image -->\n\nPage 1 of 2\n\f<!-- image -->\n\n- 2 -\n",
        "<!-- image -->\n\n第 1 页\n\n<!-- formula-not-decoded -->\n\n| --- |\n",
    ):
        feature = build_document_features(Path("scans/receipt.pdf"), content)
        flags = feature.quality_flags
        assert flags.filename_dominant is True, content
        assert flags.extraction_failed is True, content
        assert flags.short_text is True, content

    real = build_document_features(Path("ops/report.pdf"), "<!-- image -->\n" + _BODY)
    assert real.quality_flags.filename_dominant is False
    assert real.quality_flags.extraction_failed is False
    assert real.quality_flags.short_text is False


def test_placeholder_vectors_are_not_content_similarity_evidence() -> None:
    # Different placeholder text, so only the cosine could link these two.
    features = _features(
        {
            "travel/beach_photos.docx": "<!-- image -->",
            "finance/scanned_invoice_2024.docx": "<!-- image -->\n\n<!-- image -->",
        }
    )
    features[1].content_embedding = features[0].content_embedding.copy()

    assert build_candidate_edges(features) == []


def test_identical_placeholder_scans_get_no_content_backed_edge() -> None:
    features = _features(
        {
            "travel/passport_scan.pdf": "<!-- image -->\n\n<!-- image -->",
            "medical/xray_report.pdf": "<!-- image -->\n\n<!-- image -->",
            "travel/visa_scan.pdf": _SCAN,
            "finance/receipt_scan.pdf": _SCAN,
        }
    )
    # Same vector too: a placeholder-only vector is file name evidence.
    features[1].content_embedding = features[0].content_embedding.copy()

    edges = build_candidate_edges(features)
    components = build_candidate_components(features, edges)

    assert not [edge for edge in edges if edge.edge_type in CONTENT_BACKED_EDGE_TYPES]
    assert not [edge for edge in edges if edge.hard_constraint == "must_link"]
    assert not [
        component
        for component in components
        if component.component_type == "near_duplicate_group"
    ]


def test_short_or_repeated_boilerplate_is_not_a_near_duplicate() -> None:
    footer = "Scanned with CamScanner"
    pages = "".join(
        f"<!-- image -->\n\n{footer}\n\n{page}\n\n" for page in range(1, 31)
    )
    features = _features(
        {
            "a/scan.pdf": footer,
            "b/notes.pdf": footer,
            "c/contract.pdf": pages,
            "d/lease.pdf": pages,
            "e/failed.pdf": "1",
            "f/broken.pdf": "1",
        }
    )

    edges = build_candidate_edges(features)

    assert [edge for edge in edges if edge.edge_type == "near_duplicate"] == []
    assert [edge for edge in edges if edge.hard_constraint == "must_link"] == []


def test_placeholder_docs_are_not_pulled_into_another_cluster() -> None:
    files = [
        Path("travel/itinerary.md"),
        Path("finance/budget.md"),
        Path("travel/beach_photos.docx"),
        Path("finance/scanned_invoice_2024.docx"),
    ]
    contents = [
        "Trip itinerary: flights to Lisbon, hotel near the old town.",
        "Household budget: monthly spending and savings plan.",
        "<!-- image -->",
        "<!-- image -->",
    ]

    result = _run_stage(files, contents, [0, 1, 0, 1])

    assert result.cluster_result.labels.tolist() == [0, 1, 0, 1]
    assert not [
        edge for edge in result.candidate_edges if edge.edge_type == "near_duplicate"
    ]


def test_extraction_failed_scans_are_not_pulled_out_of_noise() -> None:
    files = [
        Path("invoices/invoice-01.pdf"),
        Path("invoices/invoice-02.pdf"),
        Path("travel/passport_scan.pdf"),
        Path("medical/xray_report.pdf"),
    ]
    contents = [
        "Invoice 01: consulting services, total due 1,200 EUR.",
        "Invoice 02: consulting services, total due 1,450 EUR.",
        "<!-- image -->\n\n<!-- image -->",
        "<!-- image -->\n\n<!-- image -->",
    ]
    reports = [None, None, _FailedReport(), _FailedReport()]

    result = _run_stage(files, contents, [0, 0, 0, -1], reports=reports)

    assert result.cluster_result.labels.tolist() == [0, 0, 0, -1]


def test_graph_mode_does_not_cluster_placeholder_scans_together() -> None:
    files = [Path("travel/passport_scan.pdf"), Path("medical/xray_report.pdf")]
    contents = [_SCAN, _SCAN]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    result = _run_stage(
        files, contents, [-1, -1], embeddings=embeddings, cluster_mode="graph"
    )

    assert result.cluster_result.labels.tolist() == [-1, -1]


def test_real_duplicates_are_still_near_duplicates() -> None:
    features = _features(
        {
            "a/maintenance.pdf": _BODY,
            "b/maintenance-copy.pdf": _BODY,
            "c/other.md": "Household budget: monthly spending and savings plan.",
        }
    )

    edges = build_candidate_edges(features)
    components = build_candidate_components(features, edges)

    duplicates = [edge for edge in edges if edge.edge_type == "near_duplicate"]
    assert len(duplicates) == 1
    assert duplicates[0].hard_constraint == "must_link"
    assert "identical_content" in duplicates[0].evidence
    assert [component.component_type for component in components] == [
        "near_duplicate_group"
    ]


def test_duplicates_differing_only_in_placeholders_and_layout_still_match() -> None:
    files = [
        Path("a/maintenance.pdf"),
        Path("b/maintenance.docx"),
        Path("c/other.md"),
    ]
    reflowed = "<!-- image -->\n\n" + _BODY.replace("\n", "\n\n  ") + "\n\n2\n"
    contents = [
        _BODY,
        reflowed,
        "Household budget: monthly spending and savings plan.",
    ]

    result = _run_stage(files, contents, [0, 1, 1])

    duplicates = [
        edge for edge in result.candidate_edges if edge.edge_type == "near_duplicate"
    ]
    assert len(duplicates) == 1
    assert duplicates[0].hard_constraint == "must_link"
    assert result.cluster_result.labels.tolist() == [0, 0, 1]


def test_graph_mode_still_links_real_duplicates() -> None:
    features = _features({"a/maintenance.pdf": _BODY, "b/maintenance-copy.pdf": _BODY})
    embeddings = np.stack([feature.content_embedding for feature in features])
    edges = build_candidate_edges(features)

    result = cluster_canonical_documents(
        embeddings,
        config=Config(),
        options=PipelineOptions(cluster_mode="graph"),
        item_names=[feature.name for feature in features],
        canonical_document_features=features,
        candidate_edges=edges,
        candidate_components=build_candidate_components(features, edges),
        cluster_documents_fn=_fixed_labels([-1, -1]),
    )

    assert result.labels.tolist() == [0, 0]
