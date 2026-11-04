from pathlib import Path

import pytest
import yaml

from dite.app.config import load_config


def test_load_config_creates_global_config_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"

    assert not config_path.exists()

    cfg = load_config()

    assert config_path.exists()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["models"]["embedding"] == cfg.models.embedding
    assert data["cache"]["directory"] == str(tmp_path / ".cache" / "dite")
    assert ".markdown" in data["formats"]["documents"]


def test_load_config_supports_markdown_extension_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = load_config()

    assert ".markdown" in cfg.formats.documents


def test_load_config_uses_updated_clustering_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = load_config()

    assert cfg.clustering.min_cluster_size == 3
    assert cfg.clustering.min_samples == 2
    assert cfg.clustering.cluster_selection_epsilon == 0.25
    assert cfg.clustering.cluster_selection_method == "eom"
    assert cfg.clustering.small_cluster_merge_enabled is True
    assert cfg.clustering.small_cluster_merge_max_size == 4
    assert cfg.clustering.small_cluster_merge_cosine_threshold == 0.92


def test_load_config_backfills_markdown_alias_for_existing_md_configs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "formats:",
                "  documents: [.md, .txt]",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert ".md" in cfg.formats.documents
    assert ".markdown" in cfg.formats.documents
    assert ".txt" in cfg.formats.documents


def test_load_config_ignores_workspace_dite_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "i18n:",
                "  locale: en-US",
            ]
        ),
        encoding="utf-8",
    )

    (tmp_path / "dite.yaml").write_text(
        "\n".join(
            [
                "i18n:",
                "  locale: zh-CN",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()
    assert cfg.i18n.locale == "en-US"


def test_load_config_reads_cluster_naming_request_profile(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "request_profiles:",
                "  cluster_naming:",
                "    max_tokens: 64",
                "    reasoning_mode: off",
                "    thinking_budget: 256",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.request_profiles.cluster_naming.max_tokens == 64
    assert cfg.request_profiles.cluster_naming.reasoning_mode == "off"
    assert cfg.request_profiles.cluster_naming.thinking_budget == 256
    # Profiles not mentioned in the file keep their own JSON-sized defaults.
    assert cfg.request_profiles.cluster_adjudication.max_tokens == 300
    assert cfg.request_profiles.cluster_representation.max_tokens == 300


def test_load_config_partial_json_profile_keeps_task_default_budget(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "request_profiles:",
                "  cluster_adjudication:",
                "    reasoning_mode: off",
                "  cluster_representation:",
                "    max_tokens: 512",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.request_profiles.cluster_adjudication.max_tokens == 300
    assert cfg.request_profiles.cluster_adjudication.reasoning_mode == "off"
    assert cfg.request_profiles.cluster_representation.max_tokens == 512
    assert cfg.request_profiles.cluster_naming.max_tokens == 50


def test_load_config_reads_cluster_naming_workers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "processing:",
                "  extract_workers: 4",
                "  docling_pdf_workers: 1",
                "  cluster_naming_workers: 3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.processing.extract_workers == 4
    assert cfg.processing.docling_pdf_workers == 1
    assert cfg.processing.cluster_naming_workers == 3


def test_load_config_reads_docling_device(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "processing:",
                "  docling_device: cpu",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.processing.docling_device == "cpu"


def test_load_config_reads_feature_extraction_settings(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "feature_extraction:",
                "  analysis_enabled: true",
                "  analysis_max_content_length: 1234",
                "  analysis_max_retries: 5",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.feature_extraction.analysis_enabled is True
    assert cfg.feature_extraction.analysis_max_content_length == 1234
    assert cfg.feature_extraction.analysis_max_retries == 5


def test_load_config_reads_v2_clustering_domains(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "candidate_generation:",
                "  filename_token_overlap_threshold: 0.7",
                "  content_similarity_threshold: 0.88",
                "  component_min_edge_score: 0.91",
                "topic_clustering:",
                "  mode: graph",
                "  reducer: pca",
                "  pca_components: 64",
                "  allow_single_cluster: true",
                "cluster_adjudication:",
                "  enable_llm_judging: true",
                "  edge_merge_threshold: 0.95",
                "  request_score_threshold: 0.9",
                "cluster_representation:",
                "  mode: llm_enhanced",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.candidate_generation.filename_token_overlap_threshold == 0.7
    assert cfg.candidate_generation.content_similarity_threshold == 0.88
    assert cfg.candidate_generation.component_min_edge_score == 0.91
    assert cfg.topic_clustering.mode == "graph"
    assert cfg.topic_clustering.reducer == "pca"
    assert cfg.topic_clustering.pca_components == 64
    assert cfg.topic_clustering.allow_single_cluster is True
    assert cfg.cluster_adjudication.enable_llm_judging is True
    assert cfg.cluster_adjudication.edge_merge_threshold == 0.95
    assert cfg.cluster_adjudication.request_score_threshold == 0.9
    assert cfg.cluster_representation.mode == "llm_enhanced"


def test_load_config_reads_agglomerative_and_umap_settings(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "topic_clustering:",
                "  mode: agglomerative",
                "  reducer: umap",
                "  umap_components: 8",
                "  umap_neighbors: 12",
                "  agglomerative_target_clusters: 12",
                "  agglomerative_distance_threshold: 0.4",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.topic_clustering.mode == "agglomerative"
    assert cfg.topic_clustering.reducer == "umap"
    assert cfg.topic_clustering.umap_components == 8
    assert cfg.topic_clustering.umap_neighbors == 12
    assert cfg.topic_clustering.agglomerative_target_clusters == 12
    assert cfg.topic_clustering.agglomerative_distance_threshold == 0.4


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("  mode: agglomarative", "topic_clustering.mode"),
        ("  reducer: tsne", "topic_clustering.reducer"),
    ],
)
def test_load_config_rejects_unknown_topic_clustering_values(
    tmp_path: Path, monkeypatch, line: str, message: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(["topic_clustering:", line]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_config()


def test_load_config_reads_api_runtime_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "api:",
                "  connect_timeout_sec: 3.0",
                "  read_timeout_sec: 45.0",
                "  write_timeout_sec: 12.0",
                "  pool_timeout_sec: 4.0",
                "  max_retries: 5",
                "  max_connections: 20",
                "  max_keepalive_connections: 10",
                "  keepalive_expiry_sec: 9.5",
                "processing:",
                "  vlm_api_workers: 6",
                "  vlm_pages_per_document: 3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.api.connect_timeout_sec == 3.0
    assert cfg.api.read_timeout_sec == 45.0
    assert cfg.api.write_timeout_sec == 12.0
    assert cfg.api.pool_timeout_sec == 4.0
    assert cfg.api.max_retries == 5
    assert cfg.api.max_connections == 20
    assert cfg.api.max_keepalive_connections == 10
    assert cfg.api.keepalive_expiry_sec == 9.5
    assert cfg.processing.vlm_api_workers == 6
    assert cfg.processing.vlm_pages_per_document == 3


def test_load_config_rejects_keepalive_larger_than_max_connections(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "api:",
                "  max_connections: 8",
                "  max_keepalive_connections: 16",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_config()
    except ValueError as exc:
        assert "max_keepalive_connections" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("load_config should reject invalid API pool settings")
