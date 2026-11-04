# 配置

本文档描述当前代码里实际使用的配置模型。

## 配置来源

DITE 始终从下面这个路径加载配置：

`~/.config/dite/config.yaml`

如果文件不存在，DITE 会自动创建默认配置文件。

当前不支持这些配置方式：

- 工作区本地配置文件
- 按命令临时覆盖配置
- CLI `--config` 选项

## 环境变量展开

YAML 中的字符串值支持环境变量展开。

当前实际可靠支持的形式是：

- `${VAR}`

代码里对 `$VAR` 也有过意图上的兼容说明，但当前真正会被展开的是 `${VAR}`。

## 顶层配置段

当前顶层配置包含这些部分：

- `api`
- `models`
- `request_profiles`
- `clustering`
- `processing`
- `feature_extraction`
- `candidate_generation`
- `topic_clustering`
- `cluster_adjudication`
- `cluster_representation`
- `cache`
- `formats`
- `i18n`

## `api`

字段包括：

- `base_url`
- `api_key`
- `connect_timeout_sec`
- `read_timeout_sec`
- `write_timeout_sec`
- `pool_timeout_sec`
- `max_retries`
- `max_connections`
- `max_keepalive_connections`
- `keepalive_expiry_sec`

当前校验规则：

- `api.max_keepalive_connections` 必须小于等于 `api.max_connections`

## `models`

字段包括：

- `embedding`
- `vlm`
- `llm`

当前默认值指向 Qwen3 系列模型，但代码本质上把它们当作可配置的 OpenAI-compatible 模型标识符处理。

## `request_profiles`

当前实际实现的请求配置有：

- `cluster_naming`：簇命名，默认 `max_tokens: 50`
- `cluster_adjudication`：可选的 LLM 边裁决（`cluster_adjudication.enable_llm_judging`），默认 `max_tokens: 300`
- `cluster_representation`：可选的 LLM 簇表示（`cluster_representation.mode: llm_enhanced`），默认 `max_tokens: 300`

后两个任务要求模型返回带理由的 JSON，50 个 token 装不下，截断的 JSON 会解析失败并静默回退，所以它们有独立的预算。配置文件里只写了部分字段时，没写的字段沿用该任务自己的默认值。

每个请求配置下的字段都是：

- `max_tokens`
- `reasoning_mode`
- `thinking_budget`

`reasoning_mode` 当前可接受的值：

- `default`
- `off`
- `on`

当前代码还会把 YAML 中的布尔值做归一化：

- `false` 变成 `off`
- `true` 变成 `on`

## `clustering`

字段包括：

- `min_cluster_size`
- `min_samples`
- `cluster_selection_epsilon`
- `cluster_selection_method`
- `knn_k`
- `knn_distance_threshold`
- `small_cluster_merge_enabled`
- `small_cluster_merge_max_size`
- `small_cluster_merge_cosine_threshold`

这些参数控制 HDBSCAN 聚类，以及可选的 k-NN 噪音修复。

需要特别区分的一点：

- `allow_single_cluster` 当前不是全局配置项。
- `cluster_pca_components` 当前也不是全局配置项。
- 它们只存在于实验路径的 `PipelineOptions` 和内部工具里。

当前默认值是：

- `min_cluster_size=3`
- `min_samples=2`
- `cluster_selection_epsilon=0.25`
- `cluster_selection_method="eom"`
- `knn_k=3`
- `knn_distance_threshold=None`
- `small_cluster_merge_enabled=true`
- `small_cluster_merge_max_size=4`
- `small_cluster_merge_cosine_threshold=0.92`

这些默认值已经比早期版本更保守，并且允许对小簇做一轮受限的簇间再判定。参数含义、已知问题和计划中的后续实验方向，见 `docs/clustering.md`。

需要特别区分的一点：

- `clustering` 仍然是 legacy HDBSCAN 参数域。
- V2 的候选生成、主题聚类、边界裁决和簇表示，不再继续塞进这里。

## `feature_extraction`

字段包括：

- `analysis_enabled`
- `analysis_max_content_length`
- `analysis_max_retries`

当前行为要点：

- 默认关闭。
- 开启后，`DocumentFeatures` 会尝试使用 analyzer 增强摘要、关键词、主题、领域和布局提示。
- analyzer 失败不会中止主流程，而是回退到现有轻规则特征。

## `candidate_generation`

字段包括：

- `filename_token_overlap_threshold`
- `content_similarity_threshold`
- `component_min_edge_score`

当前行为要点：

- 控制 `CandidateEdge` 和 `CandidateComponent` 的保守生成阈值。
- 正文完全一致的文件会优先形成 `near_duplicate` 关系。
- 普通强语义关系不会再自动抢主题聚类主决策。

## `topic_clustering`

字段包括：

- `mode`：`density`（默认，HDBSCAN）/ `graph` / `agglomerative`
- `reducer`：`none`（默认）/ `pca` / `umap`，只作用于 `density` 模式
- `pca_components`
- `umap_components`（默认 10）
- `umap_neighbors`（默认 15）
- `allow_single_cluster`
- `agglomerative_target_clusters`（默认空）
- `agglomerative_distance_threshold`（默认空）

当前行为要点：

- 这是 V2 主题聚类层的正式默认配置来源。
- `PipelineOptions.cluster_mode`、`cluster_pca_components` 和 `cluster_allow_single_cluster` 仍存在，但只作为内部实验覆盖层，不是公开 CLI 配置接口。
- `mode` 和 `reducer` 写错时，加载配置会直接报错，不再悄悄退回 density。
- `reducer="pca"` 且 `pca_components` 非空时，HDBSCAN 在 PCA 坐标上运行。
- `reducer="umap"` 时，HDBSCAN 在 UMAP 坐标上运行（余弦度量、`min_dist=0`、固定随机种子）。需要可选依赖 `umap-learn`：`uv add umap-learn`。没装会报错并给出安装提示。
- 用了降维时，只有 HDBSCAN 跑在降维坐标上；k-NN 噪音修复和小簇合并仍然用原始归一化 embedding，因为它们的余弦阈值是在原始空间定义的。`cluster_selection_epsilon` 则是降维空间里的距离。
- `mode="agglomerative"` 用余弦距离 + average linkage 建层次树再切分，不做密度估计，所以在高维和小目录上都比较稳定。切分方式按优先级：
  - `agglomerative_target_clusters`：想要几个文件夹。只统计成员数达到 `clustering.min_cluster_size` 的簇，零散的离群文件不占名额。
  - `agglomerative_distance_threshold`：余弦距离阈值，越小分得越细。
  - 两个都不设：自动选轮廓系数（余弦）最高的切分，最多考虑 50 个簇。这个准则只比较簇内和簇间距离的相对大小，不依赖某个 embedding 模型的绝对相似度区间。
- agglomerative 模式下，成员数小于 `clustering.min_cluster_size` 的组记为噪音、留在原处；不做 k-NN 噪音修复和小簇合并。近重复预绑定照常生效。它需要 O(n²) 的距离内存，几千个文件以内没问题。
- 这些新选项都不是默认值。按 `docs/validation.md` 的晋升门槛，需要先在代表性集和对抗集上用 `partition_metrics` 对比，再决定要不要改默认。

## `cluster_adjudication`

字段包括：

- `enable_llm_judging`
- `edge_merge_threshold`
- `request_score_threshold`

当前行为要点：

- 默认关闭 LLM 裁决。
- 开启后，只会对 `edge_review` 类型请求尝试做保守 LLM 判定。
- LLM 失败会回退到规则裁决结果。

## `cluster_representation`

字段包括：

- `mode`

当前行为要点：

- `deterministic` 是默认模式。
- `llm_enhanced` 会在有 `client` 和异步 runtime 时尝试增强簇摘要、主题、领域和证据说明。
- 失败时会回退到 deterministic 聚合表示。

## `processing`

字段包括：

- `text_truncate_limit`
- `vlm_fallback_threshold`
- `docling_pdf_timeout_sec`
- `docling_device`
- `extract_workers`
- `docling_pdf_workers`
- `cluster_naming_workers`
- `vlm_api_workers`
- `vlm_pages_per_document`
- `embedding_batch_size`

当前行为要点：

- `text_truncate_limit` 是最终内容截断上限，供 `extract_content` 使用。
- `vlm_fallback_threshold` 是判断 PDF 提取结果是否过弱的有效长度阈值。
- `extract_workers` 限制规范样本文件的并行提取数。
- `docling_pdf_workers` 限制 Docling PDF 子进程的并发量。
- `cluster_naming_workers` 限制簇命名请求并发量。
- `vlm_api_workers` 限制全局 VLM 请求并发量。
- `vlm_pages_per_document` 限制单个 PDF 内部的页级并发量，而不是总采样页数。
- `embedding_batch_size`（默认 32）是每个 embeddings 请求最多携带的文本数。服务商如果因为批次太大返回 400/413，这一批会自动对半拆开重试，所以它是上限，不要求服务商一定接受这么多条。429 和 5xx 由 SDK 按 `api.max_retries` 重试。

需要特别区分的一点：

- PDF 的 VLM 回退最多只采样前 10 页。
- `vlm_pages_per_document` 只控制这批采样页的并发度，不控制采样深度。

## `cache`

字段包括：

- `enabled`
- `directory`
- `max_size_gb`

当前行为要点：

- 缓存存放在配置目录下的 SQLite 数据库中。
- 缓存大小限制已经实现。
- 当数据库文件超过配置上限时，会删除较旧条目并执行 `VACUUM`。

## `formats`

字段包括：

- `documents`
- `images`

当前规范化规则：

- 文档扩展名会被统一转成小写。
- `.md` 和 `.markdown` 会保持同步，保证兼容性。

## `i18n`

字段包括：

- `locale`

当前支持的值：

- `zh-CN`
- `en`
- `en-US`
