from __future__ import annotations

from pathlib import Path

import numpy as np

from dite.app.config import Config
from dite.cluster import (
    AdjudicationDecision,
    AdjudicationRequest,
    apply_llm_adjudication,
    apply_rule_adjudication,
    build_adjudication_requests,
)
from dite.cluster.link import (
    build_candidate_components,
    build_candidate_edges,
)
from dite.cluster.model import CandidateComponent, CandidateEdge, ClusterRepresentation
from dite.cluster.view import build_cluster_representations
from dite.doc import (
    DocumentFeatures,
    EntityFeatures,
    LayoutHints,
    MetadataFeatures,
    QualityFlags,
    build_document_features,
)


def test_document_features_defaults_and_top_level_proxies() -> None:
    features = DocumentFeatures(
        file_id="doc-1",
        path=Path("docs/example.pdf"),
        name="example.pdf",
        stem="example",
        extension=".pdf",
    )

    assert features.content_text == ""
    assert features.summary == ""
    assert features.title_candidates == []
    assert features.file_name_tokens == []
    assert features.parent_path_tokens == []
    assert features.keywords == []
    assert features.topic == ""
    assert features.domain == ""
    assert features.quality_flags.extraction_failed is False


def test_document_features_preserve_nested_feature_models() -> None:
    features = DocumentFeatures(
        file_id="doc-2",
        path=Path("docs/report.md"),
        name="report.md",
        stem="report",
        extension=".md",
        metadata=MetadataFeatures(
            file_name_tokens=["report", "q3"],
            parent_path_tokens=["finance", "2025"],
            title_candidates=["Quarterly Report"],
        ),
        entities=EntityFeatures(
            entities=["OpenAI"],
            keywords=["revenue", "forecast"],
            topic="finance",
            domain="finance",
        ),
        layout=LayoutHints(
            document_type="report",
            columns="double",
            has_table=True,
            template_signals=["corporate"],
            page_count=12,
        ),
        quality_flags=QualityFlags(
            extraction_failed=False,
            short_text=False,
            filename_dominant=False,
        ),
        content_embedding=np.array([0.1, 0.2], dtype=np.float32),
    )

    assert features.title_candidates == ["Quarterly Report"]
    assert features.file_name_tokens == ["report", "q3"]
    assert features.parent_path_tokens == ["finance", "2025"]
    assert features.keywords == ["revenue", "forecast"]
    assert features.topic == "finance"
    assert features.domain == "finance"
    assert features.layout.document_type == "report"
    assert features.layout.has_table is True
    assert np.allclose(features.content_embedding, np.array([0.1, 0.2]))


def test_pipeline_result_exposes_document_features_slot() -> None:
    from dite.flow.api import PipelineResult

    result = PipelineResult(
        files=[],
        contents=[],
        document_features=[],
        embeddings=np.array([]),
        labels=np.array([], dtype=int),
        cluster_names={},
    )

    assert result.document_features == []
    assert result.candidate_edges == []
    assert result.candidate_components == []
    assert result.adjudication_requests == []
    assert result.adjudication_decisions == []
    assert result.cluster_representations == {}


def test_build_document_features_uses_file_report_and_generates_tokens() -> None:
    class _Report:
        primary_success = False
        selected_source = "vlm_api"
        primary_extractor = "docling"

    features = build_document_features(
        Path("finance/2025/Q3-report.pdf"),
        "Quarterly Report\nRevenue grew significantly this quarter.",
        file_report=_Report(),
    )

    assert features.file_id.endswith("Q3-report.pdf")
    assert features.file_name_tokens == ["q3", "report"]
    assert features.parent_path_tokens == ["finance", "2025"]
    assert features.title_candidates[0] == "Quarterly Report"
    assert features.selected_source == "vlm_api"
    assert features.extraction_trace == "docling->vlm_api"
    assert features.quality_flags.extraction_failed is True
    assert features.quality_flags.short_text is True
    # The primary parse failed, but VLM fallback produced real text, so the
    # document's embedding is content-based rather than filename-based.
    assert features.quality_flags.filename_dominant is False
    assert features.summary == "Quarterly Report"
    assert "revenue" in features.keywords
    assert "quarterly" in features.topic


def test_filename_only_text_is_flagged_filename_dominant() -> None:
    features = build_document_features(Path("scans/scan.pdf"), "  p.1  ")

    assert features.quality_flags.filename_dominant is True
    assert features.quality_flags.extraction_failed is True


def test_title_candidates_skip_placeholders_and_generic_headings() -> None:
    features = build_document_features(
        Path("course/linear_algebra_ch2.pptx"),
        "<!-- image -->\n\n目录\n\n## 1. Introduction\n\n"
        "## Eigenvalues and eigenvectors\nLet A be a square matrix.\n"
        "<!-- image -->\n<!-- image -->\n",
    )

    assert features.title_candidates == [
        "Eigenvalues and eigenvectors",
        "linear algebra ch2",
    ]
    assert "image" not in features.keywords


def test_title_candidates_fall_back_to_file_stem_when_content_has_no_title() -> None:
    features = build_document_features(
        Path("notes/lecture-01.md"),
        "<!-- image -->\n\nChapter 1\n\n---\n",
    )

    assert features.title_candidates == ["lecture 01"]


def test_shared_placeholder_titles_and_file_stems_do_not_force_merges() -> None:
    rng = np.random.default_rng(0)
    contents = {
        "finance/q3_invoice.pdf": (
            "<!-- image -->\n\nInvoice No. 2026-031, total due 4,200 USD."
        ),
        "course/linear_algebra_ch2.pptx": (
            "<!-- image -->\n\n## Eigenvalues and eigenvectors\nLet A be a matrix."
        ),
        "courseA/lecture1.pdf": "Operating systems: processes and scheduling.",
        "courseB/lecture1.pdf": "Organic chemistry: alkanes and alkenes.",
    }
    features = []
    for path, content in contents.items():
        feature = build_document_features(Path(path), content)
        feature.content_embedding = rng.normal(size=16).astype(np.float32)
        features.append(feature)

    edges = build_candidate_edges(features)
    components = build_candidate_components(features, edges)
    requests = build_adjudication_requests(edges, components)
    decisions = apply_rule_adjudication(requests)

    assert [edge.edge_type for edge in edges] == ["filename_similarity"]
    assert {edges[0].source_id, edges[0].target_id} == {
        "courseA/lecture1.pdf",
        "courseB/lecture1.pdf",
    }
    assert all(decision.decision != "merge_edge" for decision in decisions)


def test_title_match_edges_are_review_only_without_content_similarity() -> None:
    requests = build_adjudication_requests(
        [
            CandidateEdge(
                source_id="doc-a",
                target_id="doc-b",
                edge_type="title_match",
                score=1.0,
                evidence=["title_match:quarterly report"],
            ),
            CandidateEdge(
                source_id="doc-c",
                target_id="doc-d",
                edge_type="filename_similarity",
                score=1.0,
                evidence=["filename_token_overlap:['lecture1']"],
            ),
        ],
        [],
    )

    decisions = apply_rule_adjudication(requests)

    assert [decision.decision for decision in decisions] == [
        "review_edge",
        "review_edge",
    ]


def test_content_similarity_edge_keeps_title_evidence() -> None:
    left = DocumentFeatures(
        file_id="doc-a",
        path=Path("a/report.md"),
        name="report.md",
        stem="report",
        extension=".md",
        content_text="Quarterly Report\nRevenue grew.",
        metadata=MetadataFeatures(title_candidates=["Quarterly Report"]),
        content_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    right = DocumentFeatures(
        file_id="doc-b",
        path=Path("b/summary.md"),
        name="summary.md",
        stem="summary",
        extension=".md",
        content_text="Quarterly Report\nRevenue fell.",
        metadata=MetadataFeatures(title_candidates=["Quarterly Report"]),
        content_embedding=np.array([0.99, 0.05], dtype=np.float32),
    )

    edges = build_candidate_edges([left, right])

    assert len(edges) == 1
    assert edges[0].edge_type == "content_similarity"
    assert "title_match:quarterly report" in edges[0].evidence


def test_filename_only_embeddings_do_not_create_content_edges() -> None:
    left = DocumentFeatures(
        file_id="a/scan.pdf",
        path=Path("a/scan.pdf"),
        name="scan.pdf",
        stem="scan",
        extension=".pdf",
        quality_flags=QualityFlags(filename_dominant=True),
        content_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    right = DocumentFeatures(
        file_id="b/scan.pdf",
        path=Path("b/scan.pdf"),
        name="scan.pdf",
        stem="scan",
        extension=".pdf",
        quality_flags=QualityFlags(filename_dominant=True),
        content_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    edges = build_candidate_edges([left, right])

    assert edges == []


def test_candidate_edge_and_component_models_keep_evidence() -> None:
    edge = CandidateEdge(
        source_id="doc-a",
        target_id="doc-b",
        edge_type="content_similarity",
        score=0.93,
        evidence=["same top keywords", "high cosine similarity"],
        hard_constraint="must_link",
        quality_guard=["short_text"],
    )
    component = CandidateComponent(
        component_id="component-1",
        member_file_ids=["doc-a", "doc-b"],
        component_type="strong_semantic_group",
        formation_evidence=["content_similarity>=0.9", "title overlap"],
        confidence=0.88,
    )

    assert edge.evidence == ["same top keywords", "high cosine similarity"]
    assert edge.hard_constraint == "must_link"
    assert edge.quality_guard == ["short_text"]
    assert component.member_file_ids == ["doc-a", "doc-b"]
    assert component.confidence == 0.88


def test_build_candidate_edges_and_components_from_titles_and_filename_tokens() -> None:
    left = DocumentFeatures(
        file_id="doc-a",
        path=Path("study/linear-algebra-a.pdf"),
        name="linear-algebra-a.pdf",
        stem="linear-algebra-a",
        extension=".pdf",
        metadata=MetadataFeatures(
            file_name_tokens=["linear", "algebra"],
            parent_path_tokens=["study"],
            title_candidates=["Linear Algebra Notes"],
        ),
    )
    right = DocumentFeatures(
        file_id="doc-b",
        path=Path("study/linear-algebra-b.pdf"),
        name="linear-algebra-b.pdf",
        stem="linear-algebra-b",
        extension=".pdf",
        metadata=MetadataFeatures(
            file_name_tokens=["linear", "algebra"],
            parent_path_tokens=["study"],
            title_candidates=["Linear Algebra Notes"],
        ),
    )
    unrelated = DocumentFeatures(
        file_id="doc-c",
        path=Path("games/minecraft-guide.md"),
        name="minecraft-guide.md",
        stem="minecraft-guide",
        extension=".md",
        metadata=MetadataFeatures(
            file_name_tokens=["minecraft", "guide"],
            parent_path_tokens=["games"],
            title_candidates=["Minecraft Guide"],
        ),
    )

    edges = build_candidate_edges([left, right, unrelated])
    components = build_candidate_components([left, right, unrelated], edges)

    assert len(edges) == 1
    assert edges[0].source_id == "doc-a"
    assert edges[0].target_id == "doc-b"
    assert edges[0].edge_type == "title_match"
    assert len(components) == 1
    assert components[0].member_file_ids == ["doc-a", "doc-b"]
    assert components[0].component_type == "strong_semantic_group"


def test_build_candidate_edges_uses_content_embedding_similarity() -> None:
    left = DocumentFeatures(
        file_id="doc-a",
        path=Path("docs/a.txt"),
        name="a.txt",
        stem="a",
        extension=".txt",
        content_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    right = DocumentFeatures(
        file_id="doc-b",
        path=Path("docs/b.txt"),
        name="b.txt",
        stem="b",
        extension=".txt",
        content_embedding=np.array([0.95, 0.05], dtype=np.float32),
    )
    unrelated = DocumentFeatures(
        file_id="doc-c",
        path=Path("docs/c.txt"),
        name="c.txt",
        stem="c",
        extension=".txt",
        content_embedding=np.array([0.0, 1.0], dtype=np.float32),
    )

    edges = build_candidate_edges(
        [left, right, unrelated],
        min_content_similarity=0.9,
    )

    assert len(edges) == 1
    assert edges[0].edge_type == "content_similarity"
    assert edges[0].hard_constraint is None
    assert edges[0].source_id == "doc-a"
    assert edges[0].target_id == "doc-b"


def test_build_adjudication_requests_and_rule_decisions() -> None:
    edges = [
        CandidateEdge(
            source_id="doc-a",
            target_id="doc-b",
            edge_type="content_similarity",
            score=0.93,
            evidence=["high cosine similarity"],
        )
    ]
    components = [
        CandidateComponent(
            component_id="component-1",
            member_file_ids=["doc-a", "doc-b"],
            component_type="strong_semantic_group",
            formation_evidence=["strong_candidate_edges"],
            confidence=1.0,
        )
    ]

    requests = build_adjudication_requests(edges, components)
    decisions = apply_rule_adjudication(requests)

    assert len(requests) == 2
    assert requests[0].request_type == "component_review"
    assert requests[1].request_type == "edge_review"
    assert requests[1].score == 0.93
    assert len(decisions) == 2
    assert decisions[0].decision == "keep_component"
    assert decisions[1].decision == "merge_edge"
    assert all(isinstance(item, AdjudicationRequest) for item in requests)
    assert all(isinstance(item, AdjudicationDecision) for item in decisions)


def test_adjudication_thresholds_are_configurable() -> None:
    edges = [
        CandidateEdge(
            source_id="doc-a",
            target_id="doc-b",
            edge_type="content_similarity",
            score=0.93,
            evidence=["content_similarity:0.930"],
        ),
        CandidateEdge(
            source_id="doc-c",
            target_id="doc-d",
            edge_type="content_similarity",
            score=0.88,
            evidence=["content_similarity:0.880"],
        ),
    ]

    strict_requests = build_adjudication_requests(edges, [], score_threshold=0.9)
    assert [request.subjects for request in strict_requests] == [["doc-a", "doc-b"]]

    default_requests = build_adjudication_requests(edges, [])
    assert len(default_requests) == 2

    strict_decisions = apply_rule_adjudication(
        default_requests, edge_merge_threshold=0.95
    )
    assert [decision.decision for decision in strict_decisions] == [
        "review_edge",
        "review_edge",
    ]
    default_decisions = apply_rule_adjudication(default_requests)
    assert [decision.decision for decision in default_decisions] == [
        "merge_edge",
        "review_edge",
    ]


def test_apply_llm_adjudication_can_override_review_edges() -> None:
    requests = [
        AdjudicationRequest(
            request_id="edge:a->b",
            request_type="edge_review",
            subjects=["a", "b"],
            evidence_bundle=["high cosine similarity"],
            trigger_reason="content_similarity",
            score=0.9,
        )
    ]

    class _Runtime:
        def run_cluster_naming_batch(self, requests):
            del requests
            return [
                type(
                    "_Result",
                    (),
                    {
                        "content": (
                            '{"should_merge": true, "confidence": 0.88, '
                            '"reason": "shared topic"}'
                        ),
                        "error": None,
                    },
                )()
            ]

    class _Client:
        base_url = "https://api.example.com/v1"

    decisions = apply_llm_adjudication(
        requests,
        client=_Client(),
        config=Config(),
        request_runtime=_Runtime(),
    )

    assert len(decisions) == 1
    assert decisions[0].decision == "merge_edge"
    assert decisions[0].model_used == "llm"
    assert decisions[0].fallback_used is False


class _RecordingRuntime:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.batches: list[list] = []

    def run_cluster_naming_batch(self, requests):
        self.batches.append(list(requests))
        return [
            type("_Result", (), {"content": content, "error": None})()
            for content in self.contents[: len(requests)]
        ]


class _FakeClient:
    base_url = "https://api.example.com/v1"


def test_llm_adjudication_prompt_includes_document_content() -> None:
    features = [
        build_document_features(
            Path("courseA/lecture1.pdf"),
            "Operating Systems\nProcesses, threads and CPU scheduling policies.",
        ),
        build_document_features(
            Path("courseB/lecture1.pdf"),
            "Organic Chemistry\nAlkanes, alkenes and reaction mechanisms.",
        ),
    ]
    request = AdjudicationRequest(
        request_id="edge:a->b",
        request_type="edge_review",
        subjects=[features[0].file_id, features[1].file_id],
        evidence_bundle=["filename_token_overlap:['lecture1']"],
        trigger_reason="filename_similarity",
        score=1.0,
    )
    runtime = _RecordingRuntime(
        ['{"should_merge": false, "confidence": 0.9, "reason": "different"}']
    )

    decisions = apply_llm_adjudication(
        [request],
        client=_FakeClient(),
        config=Config(),
        request_runtime=runtime,
        document_features=features,
    )

    kwargs = runtime.batches[0][0].kwargs
    prompt = kwargs["messages"][0]["content"]
    assert "Operating Systems" in prompt
    assert "CPU scheduling" in prompt
    assert "Organic Chemistry" in prompt
    assert "Folder: courseA" in prompt
    profiles = Config().request_profiles
    assert kwargs["max_tokens"] == profiles.cluster_adjudication.max_tokens
    assert kwargs["max_tokens"] > profiles.cluster_naming.max_tokens
    assert decisions[0].decision == "review_edge"


def test_llm_adjudication_reads_string_false_and_fenced_json() -> None:
    request = AdjudicationRequest(
        request_id="edge:a->b",
        request_type="edge_review",
        subjects=["a", "b"],
        trigger_reason="content_similarity",
        score=0.9,
    )
    for content, expected in [
        ('{"should_merge": "false", "confidence": 0.8, "reason": "no"}', "review_edge"),
        (
            '```json\n{"should_merge": true, "confidence": 0.8, "reason": "yes"}\n```',
            "merge_edge",
        ),
        ('Sure! {"should_merge": "yes", "confidence": "0.7"}', "merge_edge"),
    ]:
        decisions = apply_llm_adjudication(
            [request],
            client=_FakeClient(),
            config=Config(),
            request_runtime=_RecordingRuntime([content]),
        )
        assert decisions[0].decision == expected
        assert decisions[0].fallback_used is False


def test_llm_cluster_representations_are_batched_and_use_central_members() -> None:
    def _feature(file_id: str, embedding: list[float], content: str):
        return DocumentFeatures(
            file_id=file_id,
            path=Path(file_id),
            name=Path(file_id).name,
            stem=Path(file_id).stem,
            extension=".txt",
            content_text=content,
            summary=content.split("\n")[0],
            content_embedding=np.array(embedding, dtype=np.float32),
        )

    features = [
        _feature("a/outlier.txt", [0.0, 1.0], "Outlier note"),
        _feature("a/core-1.txt", [1.0, 0.1], "Matrix rank\nRank-nullity theorem."),
        _feature("a/core-2.txt", [1.0, 0.0], "Eigenvalues\nCharacteristic polynomial."),
        _feature("b/budget.txt", [0.0, 1.0], "Budget 2026\nMonthly spending."),
    ]
    runtime = _RecordingRuntime(
        [
            '{"summary": "Linear algebra notes", "topic": "linear algebra", '
            '"domain": "education", "keywords": ["matrix"], '
            '"evidence_summary": "shared algebra topics"}',
            '{"summary": "Household budget", "topic": "budget", '
            '"domain": "finance", "keywords": "budget, spending", '
            '"evidence_summary": "money"}',
        ]
    )
    config = Config()
    config.cluster_representation.mode = "llm_enhanced"

    representations = build_cluster_representations(
        labels=np.array([0, 0, 0, 1], dtype=int),
        cluster_names={0: "Linear Algebra", 1: "Budget"},
        document_features=features,
        config=config,
        client=_FakeClient(),
        request_runtime=runtime,
    )

    assert len(runtime.batches) == 1
    assert len(runtime.batches[0]) == 2
    first_kwargs = runtime.batches[0][0].kwargs
    assert first_kwargs["max_tokens"] == (
        config.request_profiles.cluster_representation.max_tokens
    )
    assert "Rank-nullity theorem" in first_kwargs["messages"][0]["content"]
    assert representations[0].representative_file_ids == ["a/core-1.txt"]
    assert representations[0].summary == "Linear algebra notes"
    assert representations[1].keywords == ["budget", "spending"]


def test_cluster_representation_defaults() -> None:
    representation = ClusterRepresentation(cluster_id=3, name="Linear Algebra")

    assert representation.cluster_id == 3
    assert representation.name == "Linear Algebra"
    assert representation.summary == ""
    assert representation.keywords == []


def test_build_cluster_representations_uses_feature_signals() -> None:
    features = [
        DocumentFeatures(
            file_id="doc-a",
            path=Path("docs/a.txt"),
            name="a.txt",
            stem="a",
            extension=".txt",
            summary="Intro to linear algebra",
            entities=EntityFeatures(
                keywords=["matrix", "vector", "matrix"],
                topic="linear algebra",
                domain="education",
            ),
        ),
        DocumentFeatures(
            file_id="doc-b",
            path=Path("docs/b.txt"),
            name="b.txt",
            stem="b",
            extension=".txt",
            summary="Second note",
            entities=EntityFeatures(
                keywords=["matrix", "basis"],
                topic="linear algebra",
                domain="education",
            ),
        ),
    ]

    representations = build_cluster_representations(
        labels=np.array([0, 0], dtype=int),
        cluster_names={0: "Linear Algebra"},
        document_features=features,
    )

    representation = representations[0]
    assert representation.name == "Linear Algebra"
    assert representation.summary == "Intro to linear algebra"
    assert representation.topic == "linear algebra"
    assert representation.domain == "education"
    assert representation.keywords[0] == "matrix"
    assert representation.representative_file_ids == ["doc-a"]


def test_build_cluster_representations_can_use_llm_enhanced_mode() -> None:
    features = [
        DocumentFeatures(
            file_id="doc-a",
            path=Path("docs/a.txt"),
            name="a.txt",
            stem="a",
            extension=".txt",
            summary="Intro to linear algebra",
            entities=EntityFeatures(
                keywords=["matrix", "vector"],
                topic="linear algebra",
                domain="education",
            ),
        )
    ]

    class _Runtime:
        def run_cluster_naming_batch(self, requests):
            del requests
            return [
                type(
                    "_Result",
                    (),
                    {
                        "content": (
                            '{"summary":"LLM summary","topic":"algebra",'
                            '"domain":"education","keywords":["basis"],'
                            '"evidence_summary":"llm evidence"}'
                        ),
                        "error": None,
                    },
                )()
            ]

    class _Client:
        base_url = "https://api.example.com/v1"

    config = Config()
    config.cluster_representation.mode = "llm_enhanced"

    representations = build_cluster_representations(
        labels=np.array([0], dtype=int),
        cluster_names={0: "Linear Algebra"},
        document_features=features,
        config=config,
        client=_Client(),
        request_runtime=_Runtime(),
    )

    representation = representations[0]
    assert representation.summary == "LLM summary"
    assert representation.topic == "algebra"
    assert representation.keywords == ["basis"]
    assert representation.evidence_summary == "llm evidence"


def test_build_document_features_infers_domain_from_keywords_and_path() -> None:
    features = build_document_features(
        Path("games/minecraft/server-guide.md"),
        "Minecraft server guide for Fabric mod setup.",
    )

    assert features.domain == "gaming"
    assert "minecraft" in features.topic
    assert "server" in features.topic


def test_build_document_features_can_use_optional_analysis() -> None:
    class _Completions:
        def create(self, **kwargs):
            del kwargs

            class _Message:
                content = (
                    '{"layout":{"type":"technical report","columns":"double",'
                    '"has_table":true,"visual_elements":["image"]},'
                    '"content":{"topic":"clustering","keywords":["hdbscan","draft"],'
                    '"language":"en","domain":"tech"},'
                    '"summary":"Cluster pipeline analysis",'
                    '"template_hints":"arXiv paper","confidence":0.9}'
                )

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    config = Config()
    config.feature_extraction.analysis_enabled = True

    features = build_document_features(
        Path("research/cluster-paper.md"),
        "HDBSCAN clustering pipeline draft analysis",
        config=config,
        client=_Client(),
    )

    assert features.summary == "Cluster pipeline analysis"
    assert features.keywords == ["hdbscan", "draft"]
    assert features.topic == "clustering"
    assert features.domain == "tech"
    assert features.language == "en"
    assert features.layout.document_type == "technical report"
    assert features.layout.has_table is True
    assert features.layout.has_image_heavy_layout is True
