# 缓存

本文档描述 DITE 当前的 SQLite 缓存实现。

## 目的

缓存存在的意义，是避免重复做昂贵工作：

- 主文档转换
- PDF 的 VLM 回退
- embedding 生成

它同时也支持基于文件内容哈希的重复文件复用。

## 存储位置

缓存实现位于 `src/dite/cache/sqlite.py`。

默认数据库位置是：

- `~/.cache/dite/cache.db`

实际目录来自全局配置里的 `cache.directory`。

## 当前表结构

当前 `file_cache` 表包含这些字段：

- `id`
- `file_path`
- `file_hash`
- `file_mtime`
- `content_md`
- `vlm_content`
- `vlm_version`
- `embedding`
- `model_version`
- `created_at`

当前约束与索引包括：

- `UNIQUE(file_path, file_hash)`
- `file_path` 索引
- `file_hash` 索引

## 当前缓存分层

当前缓存逻辑上分成三层：

1. 主提取内容
2. VLM 回退内容
3. embedding

### 第一层：主提取内容

对应字段：

- `content_md`

它保存的是主提取路径输出，例如：

- Docling
- MarkItDown
- 纯文本读取器

这一层被视为相对稳定、适合长期缓存的内容。

### 第二层：VLM 回退内容

对应字段：

- `vlm_content`
- `vlm_version`

这一层专门保存 PDF 的 VLM 回退结果，并且故意和主提取内容分开。

当前 VLM 缓存版本常量是：

- `VLM_CACHE_VERSION = 2`

它的作用是：当 VLM 回退策略变化时，可以显式让旧缓存失效。

### 第三层：Embedding

对应字段：

- `embedding`
- `model_version`

embedding 版本不是单个固定常量，而是由两部分拼起来的：

- 当前 embedding 模型名
- 当前 embedding 输入格式版本

当前输入版本常量是：

- `filename-smart-content-normalized-v2`
- `content-only-normalized-v1`

所以一个实际的 embedding 缓存版本长得像：

- `<embedding-model>|input=filename-smart-content-normalized-v2`
- `<embedding-model>|input=content-only-normalized-v1`

这意味着当前 embedding 缓存不仅区分模型名，还区分两种输入模式：

- `with_filename`
- `content_only`

## 读取路径

当前缓存读取分成两种方式：

- 路径优先查找
- 哈希复用查找

### 路径优先查找

`get_by_path()` 会返回某个文件路径最新的一条缓存记录。

之所以先走路径，是因为当前文件路径是最直接的命中键。

### 哈希复用查找

如果路径查找没命中，缓存会尝试复用其他具有相同 `file_hash` 的文件结果。

这就是不同路径下重复文件能够复用结果的原因。

按层来看，当前行为是：

- 主提取内容可以按哈希复用
- VLM 内容可以按哈希和 VLM 版本复用
- embedding 可以按哈希和 embedding 版本复用

## 主提取内容缓存流

当前读取路径：

1. 先查当前文件路径。
2. 如果没命中，再查相同文件哈希的其他记录。

当前写入路径：

- `PipelineService` 只有在“主提取成功”且“当前文件没有主内容缓存命中”时才会写入主提取内容。

一个重要细节：

- 缓存保存的是主提取输出，不一定是最终 PDF 选中的内容。

原因是 PDF 的最终内容有可能来自 VLM 回退，而不是主提取。

## VLM 缓存流

当前读取路径：

1. 先查当前文件路径，要求 `file_hash` 和 `vlm_version` 一起匹配。
2. 如果没命中，再查任意拥有相同 `file_hash` 和 `vlm_version` 的缓存行。

当前写入路径：

- 流水线只有在最终 PDF 内容选择阶段返回了显式“应该写入缓存”的意图时，才会写入 VLM 内容。
- 实际上，这通常意味着“来自 API 的 VLM 内容，而且它真的被选为了最终内容”。

一个重要细节：

- 某份 VLM 缓存可以存在，但以后仍然可能因为主提取内容更好而被放弃。

## Embedding 缓存流

`get_embedding()` 当前读取路径是：

1. 先查当前文件路径。
2. 如果没命中，再查相同文件哈希的其他记录。
3. 如果指定了 `model_version`，必须版本匹配才算命中。

当前流水线写入方式：

- 只对唯一规范样本计算 embedding。
- 新生成的 embedding 会通过 `update_embedding()` 写回缓存。

一个重要细节：

- embedding 缓存绑定的是“流水线最终送去做 embedding 的内容”。
- 如果 embedding 模型名或输入格式版本变化，旧 embedding 不会继续被静默复用。

## 重复文件复用

当前流水线会在正式提取前先计算文件哈希。

当前行为：

- 每个唯一哈希只让第一个文件进入正式提取和 embedding 流程。
- 其余重复文件复用规范样本的内容、文件报告、标签和 embedding。
- 重复文件数量会进入提取摘要。

这不是一个独立的“增量扫描系统”，而是缓存查找和规范样本折叠带来的真实复用。

## 失效规则

### 写入 VLM 内容时会使 embedding 失效

当前 `update_vlm_content()` 在写入 VLM 回退内容时，会顺带清空 `embedding`。

原因很直接：

- 最终内容可能已经变了，之前的 embedding 不再可靠。

### 清空 VLM 缓存时也会让依赖它的 embedding 失效

当前 `clear_vlm_cache()` 只处理带 VLM 内容的条目：

- 这些条目的 `vlm_content` 和 `vlm_version` 被置空。
- 与这些条目文件哈希相同的所有行，`embedding` 被置空。
- 其他文件的 `embedding` 和所有 `content_md` 保持不变。

原因是：

- 一旦 VLM 最终内容被清掉，基于它构建的 embedding 也就不再可信。
- 按哈希失效，是因为同哈希的其他路径会复用这份 VLM 内容，它们的 embedding 同样基于 VLM 文本。

`dite cache clear-vlm` 会分别报告清除的 VLM 条目数和失效的 embedding 数。

### Embedding 版本不匹配时会被当成 miss

旧 embedding 不会因为版本变了就被物理删除，但读取时会被视为 stale，从而跳过不复用。

## Upsert 语义

当前缓存写入采用 upsert 风格。

### `save()`

`save()` 会针对 `(file_path, file_hash)` 这对键插入或更新一条记录。

当前更新行为：

- `content_md` 会被新值覆盖。
- 如果传入的 `vlm_content` 和 `vlm_version` 是空值，旧值会保留。
- 如果传入的 `embedding` 是空值，旧 embedding 会保留。
- `model_version` 会被新值覆盖。

### `update_vlm_content()`

`update_vlm_content()` 会插入或更新一条记录，并执行这些事：

- 写入 `vlm_content`
- 写入 `vlm_version`
- 清空 `embedding`

它在更新时不会覆盖已有的主提取内容。

### `update_embedding()`

`update_embedding()` 会插入或更新一条记录，并执行这些事：

- 写入 `embedding`
- 写入 `model_version`
- 更新时保留已有的主提取内容和 VLM 内容

## 大小限制执行方式

当前缓存可以通过 `cache.max_size_gb` 设置数据库文件上限。

当前淘汰行为是：

1. 数据库文件没超过配置大小时，什么都不做。
2. 文件超限时，按 `(page_count - freelist_count) * page_size` 计算实际在用的数据量。删行只会把页放进 freelist，文件不会变小，所以不看文件大小。
3. 实际数据量超过上限时，按 `created_at ASC, id ASC` 删除最旧的条目，删到实际数据量降到上限的 90% 以下为止。每批删除的条数按还需释放的字节数估算，删够就停，不会把整个缓存清空。
4. 如果实际数据没超限，只是文件被空闲页撑大，不删任何条目。
5. 最后执行 `VACUUM`，把空闲页还给文件系统。

这是一种按创建时间淘汰的简单策略，不是真正按访问频率跟踪的 LRU。

## 缓存状态命令会显示什么

`dite cache status` 当前会显示：

- 数据库路径
- 总条目数
- 含 embedding 的条目数
- 当前 embedding 版本命中数
- 旧 embedding 版本数
- 含 VLM 内容的条目数
- 唯一文件哈希数
- 数据库大小（MB）
- 当前 VLM 缓存版本

## 当前边界

- 缓存保存的是提取文本和 embedding，不保存聚类标签或整理计划。
- 主提取内容和 VLM 回退内容是故意分开的。
- PDF 的最终内容选择发生在缓存层之外。
- 当前缓存不提供文件整理的事务撤销能力。
