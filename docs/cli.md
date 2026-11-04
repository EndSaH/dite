# DITE 命令行

本文档描述当前 CLI 中已经实现的命令和行为。

## 全局选项

当前可用的全局选项有：

- `--verbose`, `-v`
- `--quiet`, `-q`
- `--color`
- `--version`
- `--install-completion`
- `--show-completion`
- `--help`

当前没有全局 `--config` 选项。配置始终从 `~/.config/dite/config.yaml` 读取。

## `dite scan`

用途：

- 扫描目录
- 提取内容
- 生成 embedding
- 聚类文件
- 生成簇名称
- 可选输出 JSON 报告

参数与选项：

- 位置参数 `folder`
- `--output`, `-o`
- `--no-cache`
- `--no-knn-repair`
- `--verbose`, `-v`
- `--quiet`, `-q`

当前行为要点：

- 如果目录不存在，命令会以状态码 `1` 退出。
- 如果没有找到支持的文件，命令会以状态码 `1` 退出。
- 当前 CLI 默认关闭同名簇合并。
- 这一步发生在簇命名之后，只能补救一部分“本该同簇却被拆开、而且恰好被命成同名”的情况，不被当作主路径聚簇策略。
- 只要缓存启用，主内容缓存和 embedding 缓存都会启用。

当前 JSON 报告结构包括：

- `summary.total_files`
- `summary.num_clusters`
- `summary.num_noise`
- `summary.num_extraction_failed`
- `summary.initial_num_clusters`
- `summary.initial_num_noise`
- `summary.noise_repaired`
- `summary.small_clusters_merged`
- `summary.name_clusters_merged`
- `summary.total_clusters_merged`
- `summary.small_cluster_merge_candidates`
- `summary.small_cluster_merge_skipped`
- `cluster_diagnostics.small_cluster_merge_max_similarity`
- `cluster_diagnostics.small_cluster_merge_events[]`
- `cluster_diagnostics.small_cluster_skip_events[]`
- `clusters[]`
- `noise[]`

每个聚类文件条目包含：

- `path`
- `name`
- `content_preview`
- `extraction_failed`

## `dite pdf-check`

用途：

- 只针对 PDF 运行提取检查
- 跳过 embedding、聚类和命名
- 判断最终提取结果是否达到配置中的可用性阈值

参数与选项：

- 位置参数 `folder`
- `--no-cache`
- `--cached-vlm-only`
- `--verbose`, `-v`
- `--quiet`, `-q`

当前行为要点：

- 只扫描 `.pdf` 文件。
- 这是“最终提取结果可用性”的烟雾测试，不是全文完整性审计。
- 如果启用 `--cached-vlm-only`，就不会再调用 VLM API，只允许使用已有的 VLM 缓存。
- 如果某个文件最终有效内容长度低于 `processing.vlm_fallback_threshold`，它会被判定为 weak。
- 如果存在 weak 文件，命令会以状态码 `1` 退出。

当前 PDF 回退限制：

- VLM 只采样前 10 页。

## `dite organize`

用途：

- 运行与 `scan` 相同的聚类流水线
- 根据聚类结果生成整理预览
- 支持预览、生成脚本或直接执行整理

参数与选项：

- 位置参数 `folder`
- `--target`, `-t`
- `--dry-run`
- `--execute`
- `--output-script`
- `--no-cache`
- `--no-knn-repair`
- `--verbose`, `-v`
- `--quiet`, `-q`

当前行为要点：

- `--dry-run`、`--execute`、`--output-script` 三者必须至少指定一个。
- 默认目标目录是 `<folder>/organized`。
- 目标目录会被排除在扫描之外，避免把新生成的输出再次读回去。
- 当前 CLI 默认关闭同名簇合并。
- 这一步发生在簇命名之后，只能补救一部分拆簇症状，不解决前面聚类本身把同类文件切开的根因。
- 噪音文件会保留在原地，不参与移动。
- `--execute` 会把每次成功的移动追加写入 `<target>/.dite/organize-<时间>.jsonl`（每移动一个文件写一行并立即落盘，中途中断也能撤销已完成的部分），结束时提示撤销命令。
- `--output-script` 生成的脚本里，所有路径和簇名都经过 shell 转义，文件名里的 `$(...)`、反引号、引号、换行都不会被执行。

## `dite undo`

用途：

- 撤销一次 `organize --execute`（或 `assign --execute`）的文件移动

参数与选项：

- 位置参数 `path`：移动记录文件，或目标目录（使用其 `.dite/` 下最新一份未撤销的记录）
- `--dry-run`：只列出会移回哪些文件
- `--yes`, `-y`：不再询问确认

当前行为要点：

- 按移动的逆序把文件移回原位置。
- 整理后的文件已经不存在时跳过（`missing`）；原位置已经有同名文件时跳过（`conflict`），绝不覆盖。
- 删除这次运行新建、现在已为空的目录；非空目录保留。
- 记录会被标记为已撤销，不能重复撤销；记录文件本身保留，便于追溯。

## 内部实验

- 当前没有公开的聚类实验命令。
- 旧的 `cluster-ab` 已被移除，避免把半成品实验接口继续暴露给普通用户。
- 输入模式对比、参数 sweep 和文件级 diff 只通过仓库内的内部实验工具执行。

当前内部实验入口是：

- `python tools/cluster_experiments.py compare-inputs <folder>`
- `python tools/cluster_experiments.py sweep <folder>`

当前实验工具支持：

- `--output`
- `--no-cache`
- `--no-knn-repair`
- `sweep --extended`

更完整的真实测试与实验顺序，见 `docs/experiments.md`。

执行方式：

- 预览模式只显示计划。
- 脚本模式会写出一个 shell 脚本。
- 执行模式会先询问确认，再进行文件整理。

当前实际移动策略：

1. 创建目标目录。
2. 把文件复制到目标位置，并尽量保留元数据。
3. 校验目标文件存在且大小匹配。
4. 只有校验通过后才删除源文件。

当前脚本导出策略偏保守：

- 会创建目录。
- 会使用 `cp -p` 复制文件。
- `rm` 删除源文件的行默认是注释状态，而不是自动启用。

## `dite cache status`

显示当前缓存数据库统计信息，包括：

- 数据库路径
- 总条目数
- 含 embedding 的条目数
- 当前 embedding 版本命中数
- 过期 embedding 数
- 含 VLM 内容的条目数
- VLM 缓存版本
- 唯一文件哈希数
- 数据库大小（MB）

## `dite cache clear`

清空整个缓存数据库中的所有缓存条目。

## `dite cache clear-vlm`

只清除 VLM 回退缓存，同时保留主提取内容和其他缓存行。

## `dite setup docling-pdf`

安装 DITE 所需的本地 Docling PDF 模型文件。

选项：

- `--force`
- `--progress`

当前行为：

- 下载结束后会检查目标模型文件是否完整存在。
- 如果下载失败，或下载后校验仍不完整，命令会以状态码 `1` 退出。
