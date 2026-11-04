# 聚类策略

本文档描述 DITE 当前仓库中的聚类策略现状、已知问题、外部最佳实践，以及下一步改动计划。

如果你要看下一阶段的大重构方案，而不是当前实现说明，请直接看 `docs/clustering-v2.md`。

它不是 HDBSCAN 教程。它的目标，是把当前实现讲清楚，然后给出一条能落地的收紧路线。

如果本文档与代码冲突，以代码为准。

## 当前实现锚点

聚类相关事实，当前主要来自这些实现位置：

- `src/dite/app/config.py`
  - `ClusteringConfig`
- `src/dite/cluster/api.py`
  - `cluster_documents()`
  - `repair_noise_with_knn()`
  - `repair_all_noise_with_similarity()`
  - `merge_clusters_by_name()`
  - `generate_all_cluster_names()`
- `src/dite/cluster/post.py`
  - `merge_small_clusters_by_similarity()`
- `src/dite/flow/model.py`
  - `PipelineOptions`
- `src/dite/app/cli.py`
  - `scan()`
  - `organize()`

## 当前行为

当前主流水线中的聚类阶段大致是：

1. 对 embedding 做 L2 归一化。
2. 构建 `DocumentFeatures`、`CandidateEdge`、`CandidateComponent` 和 `ClusterDraft`。
3. 使用 density 或 graph 主题聚类路径生成中间草案。
4. 如果启用噪音修复，只对 `-1` 噪音点做后处理。
5. 如果启用小簇再合并，对小簇做一轮受限的簇间再判定。
6. 对非噪音簇生成名称与簇表示。
7. 可选地按“完全相同的簇名”做后置合并。

这里有两个现实约束必须讲清楚：

- k-NN 噪音修复不参与 HDBSCAN 本身，也不会把两个已经形成的非噪音簇重新并回去。
- 同名簇合并发生在命名之后，只是症状性补丁，不是主路径聚类策略。

### 当前默认参数

当前默认参数定义在 `ClusteringConfig`：

- `min_cluster_size=3`
- `min_samples=2`
- `cluster_selection_epsilon=0.25`
- `cluster_selection_method="eom"`
- `knn_k=3`
- `knn_distance_threshold=None`
- `small_cluster_merge_enabled=true`
- `small_cluster_merge_max_size=4`
- `small_cluster_merge_cosine_threshold=0.92`

它们带来的直接结果是：

- HDBSCAN 默认值已经比早期版本更保守，不再刻意放大小簇。
- 已经形成的小簇会进入一轮受限的相似度再合并，而不是完全没人管。
- `eom` 仍然比 `leaf` 更保守，当前方向是先减少碎簇，再用小簇再判定补洞。

### 当前后处理边界

当前后处理分成两类：

- 噪音修复
  - `repair_noise_with_knn()` 只给噪音点分配已有簇标签。
  - 先由 k 个近邻投票选簇，再检查噪音点到"这个簇里最近成员"的距离是否在阈值内。以前检查的是到全体最近邻的距离，而最近邻可能属于另一个簇，导致点被分进一个其实离它很远的簇。
  - 如果 HDBSCAN 结果全是噪音，会走 `repair_all_noise_with_similarity()` 这条保守回退路径。
- 裁决结果（`merge_edge`）按簇生效
  - 噪音文件和已成簇文件之间有 `merge_edge`：噪音文件并入那个簇；连到多个簇时按边数多数决定。
  - 两个已成形的簇之间：只有跨簇 `merge_edge` 覆盖了较小簇至少一半的成员时才整簇合并。单独一条边不会再把某个文件从原簇里拽出来。
  - 两个噪音文件之间：保持噪音。
  - 报告 summary 里的 `adjudication_clusters_merged`、`adjudication_noise_attached` 记录了这一步的效果。
- graph 模式
  - 成簇用的边阈值读 `candidate_generation.component_min_edge_score`，不再写死 0.9。

### 阈值和 embedding 模型的关系

内容边（`content_similarity_threshold`）、小簇合并（`small_cluster_merge_cosine_threshold`）、规则裁决（`edge_merge_threshold`）用的都是绝对余弦值，默认 0.92。不同 embedding 模型的相似度区间差别很大：有的模型里无关文档也有 0.8 以上的相似度。换模型后，这些阈值的含义就变了。

`scan` 报告和实验工具的 `structure_metrics.similarity_profile` 会给出当前语料的相似度分布：

- `pairwise_p50` / `p90` / `p99`：所有文档两两之间余弦相似度的分位数
- `nearest_neighbor_p10` / `p50` / `p90`：每篇文档与最相近文档的相似度分位数

判读方法：如果 `pairwise_p99` 已经高于 0.92，说明随机两篇文档里就有 1% 会被当成内容强边，阈值偏松；如果 `nearest_neighbor_p90` 都低于 0.92，说明内容强边几乎不会出现，阈值偏紧。阈值的调整应该参考这些分位数，并在 `docs/validation.md` 的验证集上确认 `partition_metrics` 没有变差。
- 小簇级再合并
  - `merge_small_clusters_by_similarity()` 只处理 source 小簇。
  - target 只要求是另一个非噪音簇，不要求必须是大簇。
  - 合并依据是簇质心余弦相似度。
  - 当前实现是单轮确定性遍历，不会无限迭代到收敛。
- 命名后处理
  - `merge_clusters_by_name()` 可以把同名簇合并。
  - 但 `PipelineOptions.merge_same_name` 当前默认是 `False`。
  - CLI 当前也明确把 `merge_same_name=False` 传入流水线。

这意味着当前用户可见行为是：

- “同名簇”允许存在。
- 命名层不会默认替聚类层收拾残局。

## 当前已知问题

### 1. 同名簇只是症状，不是根因

如果两个本该同类的小簇被拆开，命名阶段给出相同或相近的名字，其实只是把拆簇问题暴露出来了。

把名字当作主合并依据，不够稳，也不够可解释。

### 2. 当前小簇再合并还是受限补丁

当前已经有二阶段簇间再判定，但它只覆盖 source 小簇，只看质心，不处理中型簇和大簇。

这说明当前系统已经承认“一阶段 HDBSCAN 结果不总是够用”，但补救范围仍然偏窄。

### 3. 当前实现更偏“直接高维 embedding 聚类”

当前默认实现仍以归一化后的 embedding 为主题聚类主输入，但现在已经存在显式的 `topic_clustering` 配置层，并支持 graph / PCA 等可选路径。

这条路径不是错，但它需要更认真地做参数校准和样本验证。HDBSCAN 官方文档也明确提醒过，高维数据上的密度聚类更容易退化。

### 4. embedding 混入文件名仍然可能污染结果

当前默认 embedding 输入会在正文足够长时混入文件名，在正文太短时几乎退回成“只看文件名”。

这很实用，但它也可能把本来应该更近的内容拉开，或者把本来不该靠近的文件拉近。

实验工具里还有第三种输入模式 `summary`：不嵌入原文的首/中/尾截取，而是嵌入"标题、摘要、主题、关键词 + 一段 1000 字以内的开头摘录"（去掉 Docling 占位符等标记）。开启 `feature_extraction.analysis_enabled` 时，摘要、主题、关键词来自 LLM 分析；否则用确定性的标题和关键词特征。没有提取到正文的文件只嵌入 `File name: ...`。它的缓存键会带上分析用的模型（或 `analysis=off`）。注意 LLM 分析结果目前没有缓存，每次运行都会重新调用。

中文分词：特征层（文件名词元、关键词）把连续的中文字符切成字符二元组。以前一整段中文被当成一个词，中文文件名只有完全相同时才算重合，关键词也全是整句。主题（`topic`）仍然保留完整的中文标题，方便阅读。

当前 `content_only` 实验模式也不是绝对“零文件名影响”：

- 当正文完全没有可用内容时，它会回退到稳定占位输入。
- 这个回退不会混入真实文件名，但也不是纯正文信号。

当前新增加的几个现实约束：

- `CandidateComponent` 已经区分 `near_duplicate_group` 和 `strong_semantic_group`。
- 只有 near-duplicate 族群会在主题聚类前保守预绑定。
- `cluster_representation.mode` 和 `cluster_adjudication.enable_llm_judging` 已有真实可选路径，但默认仍保持保守关闭或 deterministic。
- 候选边里只有 `near_duplicate` 和 `content_similarity` 算内容证据。`title_match` 和 `filename_similarity` 只是线索：会写进报告、交给 LLM 裁决参考，但规则裁决不会因为它们产出 `merge_edge`，graph 模式也不会用它们成簇。
- 标题候选会跳过 Docling 的 `<!-- image -->` 占位符、Markdown 图片和泛化标题（目录、摘要、Abstract、Chapter 1 之类），只看正文前 15 个非空行。文件名主干只作为展示兜底，不参与标题匹配。
- `filename_dominant` 按最终正文判断：正文不超过 10 个字符时，`with_filename` 模式的 embedding 实际只来自文件名，这类文档之间的向量相似度不计入内容证据。主提取失败但 VLM 兜底拿到正文的文档不再算 `filename_dominant`。

## 外部最佳实践对照

这里不重复整段资料，只保留对当前仓库最相关的结论。

### HDBSCAN 参数层面的结论

- `min_cluster_size` 应该表达“一个最小有意义簇至少多大”，而不是“尽量别让东西掉进噪音”。
- `min_samples` 应该单独调，不要长期默认成最宽松的 `1`。
- 如果必须保留较小的 `min_cluster_size`，应该优先认真评估 `cluster_selection_epsilon`，而不是一直钉死在 `0.0`。
- `cluster_selection_method="eom"` 仍然是更稳妥的默认选择；`leaf` 会进一步提高细粒度，不适合当前“拆簇过碎”的主问题。

### 文本聚类流程层面的结论

- “噪音回填”和“簇间合并”应该视为两类不同后处理。
- 如果结果里已经存在很多语义相近的小簇，应该使用簇级表示做保守合并，而不是只靠文档级 k-NN。
- 同名合并可以保留，但更适合作为显式可选补丁，而不是默认主路径。

### 可观测性层面的结论

不要只看最终簇名。更可靠的诊断对象是：

- 初始簇数量
- 簇大小分布
- 噪音率
- 小簇占比
- 同名簇数量
- 调参前后的稳定性

对当前仓库来说，先把这些指标暴露出来，比立刻上复杂新算法更重要。

## 当前仓库的建议方向

外部最佳实践翻译到 DITE 当前代码库后，最合理的方向是：

1. 先观察当前小簇再合并到底在合并什么。
2. 再验证文件名混入 embedding 是否在拉偏结果。
3. 不把命名后同名合并当成主解法。
4. 在有足够样本证据之前，不要贸然把整个系统切到更重的降维或主题建模框架。

## 下一步改动计划

下面的计划按“先收集证据，再做最小结构修正”的顺序推进。

### 阶段 1：聚类观测收口

目标：

- 让小簇再合并的真实行为可见，而不是只看总合并数。

建议改动：

- 在 debug 日志中记录：
  - source 小簇大小
  - target 簇大小
  - 最佳相似度
  - 合并/跳过原因
- 在 JSON 报告中补充：
  - `initial_num_clusters`
  - `initial_num_noise`
  - `small_cluster_merge_candidates`
  - `small_cluster_merge_skipped`
  - 小簇合并明细事件

完成标准：

- 普通 `scan` JSON 足够支撑后续自动分析。
- 小簇再合并不再只能用“总合并数”猜行为。

当前状态：

- 这一阶段已经基本完成。
- `scan` JSON 已经暴露初始簇数、初始噪音数、小簇候选数、小簇跳过数和小簇事件明细。

### 阶段 2：embedding 文件名影响 A/B

目标：

- 验证“文件名混入 embedding”是否在实际目录上拉偏聚类结果。

优先试验方向：

- 使用内部实验工具，对同一目录跑两次：
  - `with_filename`
  - `content_only`
- 输出并排统计、文件级 diff 和差异 JSON

当前不建议先做的事：

- 把实验能力直接升级成正式配置面
- 同时引入分块、降维或更重的 topic reduction

完成标准：

- 能在真实目录上稳定比较两种输入模式。
- embedding cache key 会因输入模式而隔离。

当前状态：

- 这一阶段也已经基本完成。
- 当前内部实验入口是 `tools/cluster_experiments.py compare-inputs`。

### 阶段 3：决定是否继续改默认聚类策略

目标：

- 在有观测和 A/B 结果之后，再决定是否继续改默认参数或扩大二阶段簇间再判定范围。

建议实现方向：

- 决定是否：
  - 继续收紧 HDBSCAN 参数
  - 扩大二阶段簇间再判定的覆盖面
  - 研究更重的降维或分块路径

当前不建议的主解法：

- 在缺乏新证据前继续拍脑袋改默认值
- 重新把主讨论拉回 same-name merge

完成标准：

- 下一轮改默认值时，能明确知道自己在修什么，而不是盲改。

当前状态：

- 这一步还没有完成。
- 当前本地数据规模已经足够启动一轮像样的实验，但还需要继续清理代表性集合的偏向性。

### 阶段 4：重新评估更重的流程改动

目标：

- 只有在前三阶段仍然不足时，再考虑更重的结构变更。

候选方向：

- 在聚类前引入可选降维实验
- 增加 `allow_single_cluster` 之类更明确的结构开关
- 研究更正式的 topic reduction 路径

这一步当前不应当作为默认主线，因为它会同时扩大：

- 配置复杂度
- 测试矩阵
- 用户可见行为变化范围

## 当前推荐顺序

如果只选一条最务实的推进路线，当前推荐顺序是：

1. 先补回归测试和聚类统计。
2. 再校准 `min_cluster_size`、`min_samples`、`cluster_selection_epsilon`。
3. 然后补一个可选的小簇再合并阶段。
4. 最后才决定是否需要更重的降维或主题压缩。

## 参考资料

下面这些资料直接影响了本文档中的判断：

- HDBSCAN parameter selection
  - https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
- HDBSCAN FAQ
  - https://hdbscan.readthedocs.io/en/latest/faq.html
- HDBSCAN API
  - https://hdbscan.readthedocs.io/en/latest/api.html
- hdbscan: Hierarchical density based clustering
  - https://joss.theoj.org/papers/10.21105/joss.00205
- Using HDBSCAN's epsilon threshold
  - https://hdbscan.readthedocs.io/en/latest/how_to_use_epsilon.html
- How to Use t-SNE Effectively / hybrid cluster selection context in practice is discussed by:
  - https://arxiv.org/abs/1911.02282
- BERTopic parameter tuning
  - https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
- BERTopic outlier reduction
  - https://maartengr.github.io/BERTopic/getting_started/outlier_reduction/outlier_reduction.html
- BERTopic topic reduction
  - https://maartengr.github.io/BERTopic/getting_started/topicreduction/topicreduction.html
- Sentence-Transformers clustering examples
  - https://sbert.net/examples/sentence_transformer/applications/clustering/README.html
