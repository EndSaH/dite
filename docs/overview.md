# DITE 概览

本文档描述当前仓库中的 DITE 实现现状。它是事实说明，不是设计提案。

## 今天的 DITE 能做什么

DITE 是一个面向本地文件夹的 CLI 文档聚类工具。它会扫描目标目录，从支持的文档和图片中提取可用文本，构建 embedding，按语义相似度聚类文件，生成簇名称，并可选地把文件整理到子目录中。

当前用户可见命令包括：

- `dite scan`
- `dite pdf-check`
- `dite organize`
- `dite assign`
- `dite undo`
- `dite cache status`
- `dite cache clear`
- `dite cache clear-vlm`
- `dite setup docling-pdf`

## 当前文档入口

当前实现对应的文档位于：

- `docs/overview.md`
- `docs/cli.md`
- `docs/configuration.md`
- `docs/architecture.md`
- `docs/clustering.md`
- `docs/clustering-v2.md`
- `docs/related-work.md`
- `docs/literature-insights.md`
- `docs/design-data-model.md`
- `docs/design-pipeline-contract.md`
- `docs/design-evaluation-protocol.md`
- `docs/refactor-plan-v2.md`
- `docs/extraction.md`
- `docs/cache.md`
- `docs/roadmap.md`
- `docs/archive-review.md`
- `docs/validation.md`
- `docs/experiments.md`

## 今天的 DITE 还不能做什么

当前代码库没有实现这些能力：

- Watch 模式
- 用户反馈闭环
- HTML 报告生成
- 图片重命名
- 独立的“增量扫描模式”
- Prompt 管理命令
- 类似 `--config` 这样的工作区级配置覆盖

当前也没有公开的聚类实验 CLI；实验能力只存在于仓库内的内部工具脚本中，例如 `tools/cluster_experiments.py`。

`docs/archive/` 下保留的是历史设计与规划材料，不是当前实现说明。

## 支持的文件类型

支持的文档格式：

- `.pdf`
- `.docx`
- `.doc`
- `.pptx`
- `.ppt`
- `.xlsx`
- `.xls`
- `.md`
- `.markdown`
- `.txt`
- `.rtf`

支持的图片格式：

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.gif`

## 高层流程

当前端到端流程是：

1. 扫描目标目录中的文件。
2. 按唯一文件哈希抽取规范样本。
3. 尽量复用缓存和重复文件结果。
4. 基于最终提取内容生成 embedding。
5. 使用 HDBSCAN 做聚类。
6. 视情况对部分噪音点做 k-NN 修复。
7. 用 LLM 生成簇名称。
8. 输出报告，或者生成整理预览。

## 当前几个关键约束

- 唯一配置来源是 `~/.config/dite/config.yaml`。
- PDF 的 VLM 回退是“提取质量回退”，不是全文完整性审计。
- PDF 的 VLM 采样只检查前 10 页。
- 当前聚类默认值已经比早期版本更保守，并且主流水线里已经有一轮受限的小簇再合并；详见 `docs/clustering.md`。
- 核心代码里保留了同名簇合并，但它发生在簇命名之后，更像对拆簇问题的症状性补丁；当前 CLI 明确传入 `merge_same_name=False`。
- `doc/analyze.py` 模块存在，但当前主流水线并不把它作为生产路径的一部分。

## 事实来源

如果本文档与代码冲突，以代码为准。
