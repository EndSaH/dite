"""配置管理模块"""

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class APIConfig:
    """API 配置"""

    base_url: str = ""
    api_key: str = ""
    connect_timeout_sec: float = 5.0
    read_timeout_sec: float = 60.0
    write_timeout_sec: float = 60.0
    pool_timeout_sec: float = 5.0
    max_retries: int = 2
    max_connections: int = 32
    max_keepalive_connections: int = 16
    keepalive_expiry_sec: float = 5.0


@dataclass
class ModelsConfig:
    """模型配置"""

    embedding: str = "Qwen/Qwen3-Embedding-8B"
    vlm: str = "Qwen/Qwen3-VL-32B-Instruct"
    llm: str = "Qwen/Qwen3-32B"


@dataclass
class ChatCompletionProfileConfig:
    """Chat completion request profile."""

    max_tokens: int = 50
    reasoning_mode: Literal["default", "off", "on"] = "off"
    thinking_budget: int | None = None


def _json_task_profile() -> ChatCompletionProfileConfig:
    # JSON answers with a free-text reason do not fit in the 50-token naming
    # budget; a truncated object fails to parse and silently falls back.
    return ChatCompletionProfileConfig(max_tokens=300)


@dataclass
class RequestProfilesConfig:
    """Task-specific request profiles."""

    cluster_naming: ChatCompletionProfileConfig = field(
        default_factory=ChatCompletionProfileConfig
    )
    cluster_adjudication: ChatCompletionProfileConfig = field(
        default_factory=_json_task_profile
    )
    cluster_representation: ChatCompletionProfileConfig = field(
        default_factory=_json_task_profile
    )


@dataclass
class ClusteringConfig:
    """聚类参数配置"""

    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.25
    cluster_selection_method: str = "eom"  # "eom" 产生更大的簇，"leaf" 更细粒度
    knn_k: int = 3  # k-NN 噪音修复邻居数
    knn_distance_threshold: float | None = None
    small_cluster_merge_enabled: bool = True
    small_cluster_merge_max_size: int = 4
    small_cluster_merge_cosine_threshold: float = 0.92


@dataclass
class ProcessingConfig:
    """处理参数配置"""

    text_truncate_limit: int = 4000
    # Share of the excerpt budget taken from the start and the middle of a
    # long document; the end gets the rest. Tails are often references,
    # indexes or copyright pages, so head-heavy splits are worth testing.
    excerpt_head_ratio: float = 0.6
    excerpt_middle_ratio: float = 0.2
    vlm_fallback_threshold: int = 100  # 有效内容少于此阈值时触发 VLM 回退
    docling_pdf_timeout_sec: float = 60.0
    docling_device: str = "auto"
    extract_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    docling_pdf_workers: int = 1
    cluster_naming_workers: int = 2
    vlm_api_workers: int = 8
    vlm_pages_per_document: int = 4
    # Inputs per embeddings request. Rejected batches are halved automatically,
    # so this is an upper bound, not something every provider must accept.
    embedding_batch_size: int = 32


@dataclass
class FeatureExtractionConfig:
    """Optional feature enrichment settings for clustering V2."""

    analysis_enabled: bool = False
    analysis_max_content_length: int = 4000
    analysis_max_retries: int = 2


@dataclass
class CandidateGenerationConfig:
    """Candidate edge and component generation controls."""

    filename_token_overlap_threshold: float = 0.8
    content_similarity_threshold: float = 0.92
    component_min_edge_score: float = 0.9


@dataclass
class TopicClusteringConfig:
    """Topic-clustering controls beyond legacy HDBSCAN defaults."""

    # density: HDBSCAN (+ noise repair / small-cluster merge)
    # graph: strong content edges first, HDBSCAN on the rest
    # agglomerative: cosine average-linkage tree, cut by count or distance
    mode: str = "density"
    # Applies to density mode only: none / pca / umap (umap needs umap-learn).
    reducer: str = "none"
    pca_components: int | None = None
    umap_components: int = 10
    umap_neighbors: int = 15
    allow_single_cluster: bool = False
    # Agglomerative cut: a target cluster count wins over a distance threshold;
    # with neither set, the cut with the best cosine silhouette is chosen.
    agglomerative_target_clusters: int | None = None
    agglomerative_distance_threshold: float | None = None


TOPIC_CLUSTERING_MODES = frozenset({"density", "graph", "agglomerative"})
TOPIC_CLUSTERING_REDUCERS = frozenset({"none", "pca", "umap"})


@dataclass
class ClusterAdjudicationConfig:
    """Boundary adjudication settings for V2."""

    enable_llm_judging: bool = False
    edge_merge_threshold: float = 0.92
    request_score_threshold: float = 0.85


@dataclass
class ClusterRepresentationConfig:
    """Cluster representation settings for V2."""

    mode: str = "deterministic"


@dataclass
class CacheConfig:
    """缓存配置"""

    enabled: bool = True
    directory: Path = field(default_factory=lambda: Path.home() / ".cache" / "dite")
    max_size_gb: float = 5.0


@dataclass
class FormatsConfig:
    """支持的文件格式"""

    documents: list[str] = field(
        default_factory=lambda: [
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".md",
            ".markdown",
            ".txt",
            ".rtf",
        ]
    )
    images: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".gif"]
    )

    @property
    def all_extensions(self) -> set[str]:
        """获取所有支持的扩展名"""
        return set(self.documents + self.images)


@dataclass
class I18nConfig:
    """Internationalization configuration"""

    locale: Literal["zh-CN", "en", "en-US"] = "en"


@dataclass
class Config:
    """DITE 全局配置"""

    api: APIConfig = field(default_factory=APIConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    request_profiles: RequestProfilesConfig = field(
        default_factory=RequestProfilesConfig
    )
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    feature_extraction: FeatureExtractionConfig = field(
        default_factory=FeatureExtractionConfig
    )
    candidate_generation: CandidateGenerationConfig = field(
        default_factory=CandidateGenerationConfig
    )
    topic_clustering: TopicClusteringConfig = field(
        default_factory=TopicClusteringConfig
    )
    cluster_adjudication: ClusterAdjudicationConfig = field(
        default_factory=ClusterAdjudicationConfig
    )
    cluster_representation: ClusterRepresentationConfig = field(
        default_factory=ClusterRepresentationConfig
    )
    cache: CacheConfig = field(default_factory=CacheConfig)
    formats: FormatsConfig = field(default_factory=FormatsConfig)
    i18n: I18nConfig = field(default_factory=I18nConfig)


def _expand_env_vars(value: str) -> str:
    """展开字符串中的环境变量（支持 ${VAR} 和 $VAR 格式）"""
    # 匹配 ${VAR} 格式
    pattern = r"\$\{([^}]+)\}"

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(pattern, replacer, value)


def _process_dict(d: dict[str, Any]) -> dict[str, Any]:
    """递归处理字典，展开环境变量"""
    result = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _process_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _expand_env_vars(v) if isinstance(v, str) else v for v in value
            ]
        else:
            result[key] = value
    return result


def _normalize_reasoning_mode(value: Any) -> Any:
    """Normalize YAML-loaded reasoning mode values."""
    if value is False:
        return "off"
    if value is True:
        return "on"
    return value


def _normalize_document_extensions(extensions: list[str]) -> list[str]:
    """Keep Markdown aliases in sync for backward-compatible format support."""
    normalized: list[str] = []
    for extension in extensions:
        lowered = extension.lower()
        if lowered not in normalized:
            normalized.append(lowered)

    if ".md" in normalized and ".markdown" not in normalized:
        normalized.append(".markdown")
    elif ".markdown" in normalized and ".md" not in normalized:
        normalized.append(".md")

    return normalized


def _dict_to_config(data: dict[str, Any]) -> Config:
    """将字典转换为 Config 对象"""
    config = Config()

    if "api" in data:
        config.api = APIConfig(**data["api"])

    if "models" in data:
        config.models = ModelsConfig(**data["models"])

    if "request_profiles" in data:
        request_profiles_data = data["request_profiles"].copy()
        defaults = RequestProfilesConfig()
        for profile_name, profile_data in list(request_profiles_data.items()):
            if not isinstance(profile_data, dict):
                continue
            profile_values = profile_data.copy()
            if "reasoning_mode" in profile_values:
                profile_values["reasoning_mode"] = _normalize_reasoning_mode(
                    profile_values["reasoning_mode"]
                )
            default_profile = getattr(defaults, profile_name, None)
            if isinstance(default_profile, ChatCompletionProfileConfig):
                # Unspecified keys keep the task's own default, not the
                # naming profile's 50-token budget.
                profile_values = {
                    "max_tokens": default_profile.max_tokens,
                    "reasoning_mode": default_profile.reasoning_mode,
                    "thinking_budget": default_profile.thinking_budget,
                    **profile_values,
                }
            request_profiles_data[profile_name] = ChatCompletionProfileConfig(
                **profile_values
            )
        config.request_profiles = RequestProfilesConfig(**request_profiles_data)

    if "clustering" in data:
        config.clustering = ClusteringConfig(**data["clustering"])

    if "processing" in data:
        config.processing = ProcessingConfig(**data["processing"])

    if "feature_extraction" in data:
        config.feature_extraction = FeatureExtractionConfig(
            **data["feature_extraction"]
        )

    if "candidate_generation" in data:
        config.candidate_generation = CandidateGenerationConfig(
            **data["candidate_generation"]
        )

    if "topic_clustering" in data:
        config.topic_clustering = TopicClusteringConfig(**data["topic_clustering"])

    if "cluster_adjudication" in data:
        config.cluster_adjudication = ClusterAdjudicationConfig(
            **data["cluster_adjudication"]
        )

    if "cluster_representation" in data:
        config.cluster_representation = ClusterRepresentationConfig(
            **data["cluster_representation"]
        )

    if "cache" in data:
        cache_data = data["cache"].copy()
        if "directory" in cache_data:
            cache_data["directory"] = Path(cache_data["directory"]).expanduser()
        config.cache = CacheConfig(**cache_data)

    if "formats" in data:
        formats_data = data["formats"].copy()
        if "documents" in formats_data:
            formats_data["documents"] = _normalize_document_extensions(
                formats_data["documents"]
            )
        config.formats = FormatsConfig(**formats_data)

    if "i18n" in data:
        config.i18n = I18nConfig(**data["i18n"])

    processing = config.processing
    if not (
        0.0 <= processing.excerpt_head_ratio <= 1.0
        and 0.0 <= processing.excerpt_middle_ratio <= 1.0
        and processing.excerpt_head_ratio + processing.excerpt_middle_ratio <= 1.0
    ):
        raise ValueError(
            "processing.excerpt_head_ratio and excerpt_middle_ratio must be "
            "between 0 and 1 and sum to at most 1"
        )

    topic = config.topic_clustering
    if topic.mode not in TOPIC_CLUSTERING_MODES:
        raise ValueError(
            f"topic_clustering.mode must be one of "
            f"{sorted(TOPIC_CLUSTERING_MODES)}, got {topic.mode!r}"
        )
    if topic.reducer not in TOPIC_CLUSTERING_REDUCERS:
        raise ValueError(
            f"topic_clustering.reducer must be one of "
            f"{sorted(TOPIC_CLUSTERING_REDUCERS)}, got {topic.reducer!r}"
        )

    if config.api.max_keepalive_connections > config.api.max_connections:
        raise ValueError(
            "api.max_keepalive_connections must be less than or equal to "
            "api.max_connections"
        )

    return config


def _global_config_path() -> Path:
    """返回全局配置文件路径"""
    return Path.home() / ".config" / "dite" / "config.yaml"


def _yaml_compatible(value: Any) -> Any:
    """将配置值转换为 YAML 可序列化对象"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_compatible(item) for item in value]
    return value


def _default_config_data() -> dict[str, Any]:
    """生成默认配置字典"""
    return _yaml_compatible(asdict(Config()))


def _ensure_global_config_file(path: Path) -> None:
    """确保全局配置文件存在，不存在则写入默认配置"""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            _default_config_data(),
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def load_config() -> Config:
    """
    加载全局配置文件。

    配置来源固定为 ~/.config/dite/config.yaml。
    如果目录或文件不存在，会自动创建并写入默认配置。

    Returns:
        Config 对象
    """
    config_file = _global_config_path()
    _ensure_global_config_file(config_file)

    with config_file.open(encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    data = _process_dict(raw_data)
    return _dict_to_config(data)
