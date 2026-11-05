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

## 扫描范围

`scan`、`pdf-check`、`organize`、`assign` 用同一套规则查找文件：只收受支持扩展名的文件，并跳过下面这些条目。被跳过的目录整个不进入，里面的文件既不提取，也不会被整理移动。

- 名字以 `.` 开头的目录和文件，例如 `.git`、`.venv`、`.obsidian`、`.DS_Store`、macOS 的 AppleDouble 元数据文件 `._*`。
- 依赖和系统目录：`node_modules`、`__pycache__`、`site-packages`、`__MACOSX`、`$RECYCLE.BIN`、`System Volume Information`，以及包含 `pyvenv.cfg` 的 Python 虚拟环境（目录名不必带点，例如 `venv`、`env`）。
- Finder 当成单个文件显示的 macOS 包目录，例如 `.app`、`.bundle`、`.framework`、`.plugin`、`.kext`、`.pkg`、`.photoslibrary`、`.musiclibrary`、`.tvlibrary`、`.imovielibrary`、`.fcpbundle`、`.logicx`、`.xcodeproj`、`.xcworkspace`、`.sparsebundle`、`.rtfd`、`.pages`。完整列表见 `src/dite/flow/scan.py` 中的 `PACKAGE_DIR_SUFFIXES`。包目录只按名字后缀识别，所以名字恰好以这些后缀结尾的普通文件夹（例如 `Client.key`、`Lecture.pages`）也会被跳过；需要扫描它时，把它改名，或直接把它作为参数传入。
- 临时和系统文件：Office 锁文件 `~$*`、`Thumbs.db`、`desktop.ini`。

这些规则只检查命令参数所指目录之下的条目：直接把一个隐藏目录（或包目录、虚拟环境）作为参数传入时，它的内容照常扫描。名称比较不区分大小写。规则目前不可配置；`--verbose` 会打印跳过的目录和文件数量。

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
- 每个簇的显示名互不相同（不区分大小写和 Unicode NFC/NFD 写法）：各簇是分别命名的，名字撞上时按簇的顺序，先出现的用原名，后面的依次加 `_2`、`_3` 后缀。
- 簇名最长 24 个字符。超出时在词与词之间截断，不会切在英文单词中间（例如 `Machine Learning Research Papers` 显示为 `Machine Learning`）；中文没有空格，按字截断；单个词本身超过 24 个字符时只能硬截断。
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
- 不同的簇总是放进不同的文件夹。簇名在命名阶段已经互不相同；清洗非法字符后仍然撞名的（例如 `AI/ML` 和 `AI-ML`）同样处理：先出现的用原名，后面的依次加 `_2`、`_3` 后缀，预览里分开列出。
- `--execute` 会把每次移动追加写入 `<target>/.dite/organize-<时间>.jsonl`（每个文件先写一行记录并落盘、再删除源文件，中途中断也能撤销已完成的部分），结束时提示撤销命令。
- 执行时绝不覆盖已有文件。文件名按大小写不敏感、不区分 Unicode NFC/NFD 写法的方式判重（与 macOS 默认的 APFS 一致），同名时加 `_1`、`_2` 后缀；预览之后才出现在目标位置的文件也不会被覆盖。
- 同一文件系统内用硬链接加删除原名的方式移动，不复制数据，文件元数据保持不变；跨文件系统或文件系统不支持硬链接时才复制：先写到同一文件夹里的隐藏临时文件 `.dite-partial-<随机串>`，写完并落盘后才换成正式文件名，所以复制被打断时正式文件名下不会出现不完整的文件。源文件删不掉时撤回这次移动；只有无法确认源文件是否还在时才保留两个名字，撤销时会清理多出来的那个。目标磁盘写满或只读时停止剩余的移动。
- 执行中收到 SIGTERM 或 SIGHUP（关闭终端、关机、`timeout`、IDE 停止进程）时和 Ctrl-C 一样停下，并清理正在进行的那一步；已完成的移动都在记录里，可以撤销。`kill -9`、断电这类无法处理的终止可能留下一个隐藏的 `.dite-partial-*` 临时文件（不会被当成文档扫描；下次整理到这个文件夹或撤销这次运行时自动删除，也可以手动删除），或者留下一个内容完整、还没记入记录的副本，源文件原样还在。
- `--output-script` 生成的脚本里，所有路径和簇名都经过 shell 转义，文件名里的 `$(...)`、反引号、引号、换行都不会被执行；目标位置已有文件时脚本会停止，不覆盖。

## `dite assign`

用途：

- 把新文件归入已经整理好的子文件夹，不重新聚类、不改动已有文件夹

参数与选项：

- 位置参数 `folder`：新文件所在的文件夹
- `--target`, `-t`：已有分类子文件夹所在的目录。不指定时就是 `folder` 本身，此时只处理 `folder` 根目录下的散落文件；指定时处理 `folder` 下的全部文件（排除 `target`）
- `--min-similarity`：用一个固定的余弦相似度门槛代替自动门槛
- `--dry-run`、`--execute`、`--output-script`：与 `organize` 相同，三者至少指定一个
- `--no-cache`、`--verbose`、`--quiet`

当前行为要点：

- `target` 下的每个子文件夹是一个分类，里面（递归）受支持的文件是它的成员；没有成员的子文件夹忽略。按“扫描范围”会被跳过的子文件夹（以 `.` 开头的、`node_modules`、虚拟环境、macOS 包目录等）不是分类，新文件不会被移进去。
- 每个新文件去和各分类质心最相似的那个比较。门槛由该分类自己定，同时参考两个分布：本分类成员与"其余成员质心"的相似度（取第 10 百分位，即最不典型的成员），以及其他分类的文件与本分类质心的相似度（取第 90 百分位，即最像本分类的外人）。门槛取两者中点，相当于在"自己人"和"外人"之间划一条线；两个分布重叠时用更严格的成员值。只有一个成员的分类借用其他分类门槛的中位数。这样不依赖某个 embedding 模型的绝对相似度区间，也不会因为某个分类成员彼此几乎一样就把门槛抬到够不着。
- 不够接近任何分类的文件保留原位。`--verbose` 会逐个打印相似度和门槛。
- 文件移进已有文件夹时保留文件夹原名，不做名称清洗；同名文件（同样按大小写不敏感的方式判断）自动加 `_1` 后缀，不覆盖。
- `--execute` 同样写移动记录，可以用 `dite undo` 撤销；已有的分类文件夹不会因为撤销被删除。没有任何文件要移动时不写记录。

## `dite undo`

用途：

- 撤销一次 `organize --execute`（或 `assign --execute`）的文件移动

参数与选项：

- 位置参数 `path`：移动记录文件，或目标目录（使用其 `.dite/` 下最新一份未撤销的记录）
- `--dry-run`：只列出会移回哪些文件
- `--yes`, `-y`：不再询问确认

当前行为要点：

- 按移动的逆序把文件移回原位置。
- 整理后的文件已经不存在时跳过（`missing`）；整理后的文件与记录里的大小或修改时间对不上（被修改过或换成了别的文件）时跳过（`changed`），文件留在整理后的位置，提示里给出它的原位置，需要的话手动移回；原位置已经有同名文件时跳过（`conflict`），绝不覆盖。
- 如果 `--execute` 在写完记录、删除源文件之前被中断，原位置和整理后的位置会是同一个文件（或内容完全相同的副本），撤销时只删掉整理后的那个名字。
- 磁盘写满时记录不会留下写了一半的行；进程崩溃留下的半行会被忽略，不会让这份记录读不出来、进而被跳过去撤销更早的一次运行。
- 删除这次运行新建、现在已为空的目录；非空目录保留。
- 撤销同样会在 SIGTERM/SIGHUP 时像 Ctrl-C 一样停下并清理；跨文件系统移回时也先写隐藏临时文件，所以被打断后原位置不会出现不完整的文件，重试时不会因此报 `conflict`。撤销会删除这次运行涉及的文件夹里残留的 `.dite-partial-*` 临时文件。
- 每移回一个文件都立即记入记录。只有全部文件都移回之后，记录才会被标记为已撤销，不能重复撤销；有任何跳过时记录都保持未撤销，再次执行 `dite undo` 只处理剩下的条目。一个文件都移不回时不询问、也不写任何东西。记录文件本身保留，便于追溯。
- 记录写不进去（例如磁盘写满）时撤销会停下并给出提示，已经移回的文件保持不动，解决之后再次执行即可继续。
- 撤销被中断（Ctrl-C、SIGTERM、SIGHUP）或因为记录写不进去而停下之后，再次执行 `dite undo` 会接着做完：已经移回原位置、但还没来得及记入记录的文件，按记录里的大小和修改时间认出来，算作已移回，不会被误报为 `missing`（旧版本写的、没有这两项信息的记录仍然只能报 `missing`）。所有文件都已移回、只差收尾时，不再询问确认，直接删除这次运行新建的空目录并把记录标记为已撤销。
- 传目标目录时，取最新一份未撤销、并且现在有文件可以移回（或有可以重试的 `conflict`、读写错误）的记录，所以可以连续执行 `dite undo` 逐次往回撤。某次运行剩下的文件现在都不在（`missing`）或都被改过（`changed`）时，会提示并跳过这份记录，但不会标记它：文件回来后（例如把改名的文件夹改回原名）再执行 `dite undo` 又会找到它。所有记录都没有能移回的文件时，显示最新一份记录里各文件的跳过原因。想撤某一份特定的记录，直接传它的文件路径。
- 已知限制：整理后的文件如果被永久删除了，它所在的那份记录永远不会被标记为已撤销（其余文件都已移回也一样），以后每次 `dite undo <目标目录>` 都会提示跳过这份记录。这不妨碍撤销更早的运行；确定不再需要这份记录时，可以手动删除 `.dite/` 下对应的记录文件。

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

带 VLM 内容的文件（以及与它哈希相同的文件）的 embedding 是基于 VLM 文本算出的，会一并失效；其他文件的 embedding 保持不变。命令会分别报告清除的 VLM 条目数和失效的 embedding 数。

## `dite setup docling-pdf`

安装 DITE 所需的本地 Docling PDF 模型文件。

选项：

- `--force`
- `--progress`

当前行为：

- 下载结束后会检查目标模型文件是否完整存在。
- 如果下载失败，或下载后校验仍不完整，命令会以状态码 `1` 退出。
