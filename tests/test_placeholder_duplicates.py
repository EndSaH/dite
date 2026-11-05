"""Placeholder-only text must not count as document content.

Docling emits ``<!-- image -->`` for every picture and nothing else for a
scanned PDF without OCR or an image-only DOCX/PPTX. Two such files share the
same "text" whatever they show, so neither identical text nor the cosine of
their (file name + placeholder) vectors says anything about their topic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dite.cluster.link import build_candidate_edges
from dite.doc import build_document_features

_BODY = (
    "Quarterly maintenance report for the north warehouse.\n"
    "The loading dock lift was serviced on 3 March and passed inspection.\n"
    "Two forklifts need new batteries before the summer peak; the order is "
    "placed and delivery is expected within six weeks.\n"
    "Roof leak above aisle 14 was patched; monitor after heavy rain.\n"
)
_SCAN = "<!-- image -->\n\n1\n\n<!-- image -->\n\n2\n\n<!-- image -->\n\n3\n"


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
