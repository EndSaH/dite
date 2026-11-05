# 架构

本文档描述 DITE 当前运行时架构的真实形态。

## 主要模块

顶层包结构包括：

- `src/dite/app/`
- `src/dite/doc/`
- `src/dite/io/`
- `src/dite/flow/`
- `src/dite/cluster/`
- `src/dite/report/`
- `src/dite/cache/sqlite.py`
- `src/dite/util/`

## 当前运行层次

当前实现大致可以分成这些层：

1. CLI 编排层
2. 文件扫描层
3. 提取决策层
4. 缓存复用与重复文件折叠层
5. embedding 生成层
6. 聚类与噪音修复层
7. 簇命名层
8. 报告或整理输出层

## CLI 层

`src/dite/app/cli.py` 当前承担 CLI 入口与少量展示编排。

它主要负责：

- 加载全局配置
- 创建共享的 OpenAI-compatible 客户端
- 在启用时创建缓存句柄
- 组装 `PipelineOptions`
- 调用 `PipelineService`
- 渲染用户可见输出

大部分业务逻辑都放在 CLI 之下，而不是直接写在命令函数里。

## 流水线层

`src/dite/flow/api.py` 是当前的核心编排中心。

关键数据结构包括：

- `PipelineOptions`
- `PipelineResult`
- `ExtractionSummary`
- `ExtractionFileReport`

`PipelineService.run()` 当前执行完整流水线：

1. 扫描文件。
2. 提取内容。
3. 按唯一哈希折叠规范样本。
4. 对规范样本内容做向量化。
5. 对规范样本 embedding 做聚类。
6. 把标签和 embedding 展开回重复文件。
7. 生成簇名称。
8. 返回完整结果和提取阶段遥测信息。

`PipelineService.extract_files()` 是 `pdf-check` 使用的轻量路径，只做内容提取和文件级报告，不做 embedding、聚类和命名。

## 扫描层

`src/dite/flow/scan.py` 默认递归扫描目录。

当前几个重要行为：

- 只包含支持的扩展名。
- 返回结果会排序。
- 排除路径是通过 resolved path 做相对关系判断的。

## 提取层

提取层是当前复杂度最高的热点区域。

关键模块包括：

- `io/route.py`
- `io/pdf/policy.py`
- `io/pdf/final.py`
- `io/pdf/vlm.py`
- `io/docling.py`
- `io/markitdown.py`
- `io/text.py`
- `io/vlm.py`

当前最关键的分层是：

- `extract_document()` 只负责主提取
- `resolve_document_extraction()` 负责 PDF 回退与最终内容选择

这样做的原因是：主提取内容和 PDF VLM 回退内容拥有不同的缓存生命周期和决策路径，不能混成同一层。

## 缓存层

`src/dite/cache/sqlite.py` 实现了基于 SQLite 的缓存层。

当前缓存保存三类核心东西：

- 主提取内容
- 带版本号的 VLM 回退内容
- 带输入版本号的 embedding

流水线会在提取和 embedding 之前先根据文件哈希折叠重复文件，尽量避免昂贵工作重复发生。

## Embedding 层

`src/dite/doc/embed.py` 负责从文本构建 embedding。

当前行为要点：

- 内容长度超过 10 个字符时，会把文件名作为前缀一起送入 embedding 输入。
- 接近空文本时，会退化为只使用文件名的 embedding 输入。
- 结果会被归一化。
- embedding 缓存版本由模型名和输入格式版本共同决定。

## 聚类层

`src/dite/cluster/api.py` 当前负责：

- HDBSCAN 聚类
- 可选的 k-NN 噪音修复
- 小簇级再合并
- 簇命名
- 命名后的可选同名簇合并

当前小簇级再合并发生在 HDBSCAN 和噪音修复之后，基于簇表示对 source 小簇做一轮受限的相似度重审。

当前 CLI 层明确把同名簇合并保持为关闭状态。

当前实现仍然是受限版本：

- 只处理 source 小簇
- 只做单轮确定性遍历
- 不处理所有中型和大型簇

聚类参数、已知问题和下一步计划，见 `docs/clustering.md`。

## 整理层

`src/dite/flow/move.py` 负责根据聚类结果构建整理预览。

当前实际支持：

- 预览渲染
- shell 脚本生成
- 直接执行：先给文件建立新名字，再删除源文件。同一文件系统内用硬链接，不复制数据；跨文件系统或不支持硬链接时，先复制到同一文件夹里的隐藏临时文件，落盘并校验大小后才换成正式文件名。任何情况下都不覆盖已有文件
- 移动记录（`<target>/.dite/organize-*.jsonl`）：每一步先写记录并落盘，再删除源文件；`dite undo` 按记录把文件移回，部分失败或被中断时可以重试

当前尚不支持：

- 事务式回滚
- 用户反馈记录

## Analyzer 当前状态

`src/dite/doc/analyze.py` 模块存在，也有测试，但当前主流水线并不把它作为生产路径使用。今天的主路径仍然是“直接提取内容，再直接做 embedding”。
