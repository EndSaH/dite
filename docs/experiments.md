# 实验流程

本文档描述当前 DITE 在沙箱外执行真实测试与聚类实验的推荐顺序。

它不是参数调优理论说明，而是一份可直接执行的操作手册。

如果本文档与代码冲突，以代码为准；如果本文档与 `tools/cluster_experiments.py` 的参数不一致，以脚本实现为准。

## 适用范围

这份流程用于：

- 本地 E2E smoke
- 输入模式 A/B
- 基础参数 sweep
- 必要时的扩展实验

它假设你已经具备：

- 可工作的 Python/uv 环境
- 当前仓库代码
- 本地验证集：
  - `docs/test/valid`
  - `docs/test/rep`
  - `docs/test/adv`

## 日志原则

每次真实实验都应当同时保存：

- 结构化 JSON 输出
- CLI 标准输出日志

所有正式测试一律使用最详细的可用日志级别。

推荐做法：

- JSON 用 `--output`
- 终端日志用 `tee`
- 对正式 CLI 命令显式加 `--verbose`

不要只看终端输出，也不要只保留 JSON。

注意：

- `dite scan`、`dite organize`、`dite pdf-check` 支持 `--verbose`
- `tools/cluster_experiments.py` 当前不支持 `--verbose`
- 对内部实验脚本，现阶段只能通过 `tee` 保留完整 stdout/stderr

## 结果目录

每轮实验先创建一个独立输出目录：

```bash
OUTDIR="/tmp/dite-exp-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
printf '%s\n' "$OUTDIR"
```

后续所有输出都写到这个目录。

## 阶段 1：基线完整性检查

目的：

- 确认仓库当前没有明显破损

建议先跑：

```bash
uv run ruff check src tests tools | tee "$OUTDIR/ruff.log"
uv run pytest | tee "$OUTDIR/pytest.log"
```

通过标准：

- `ruff` 全绿
- 全量测试无失败
- 如果 real API 未配置，允许相关测试被 `skip`

## 阶段 2：正式 CLI 的 E2E smoke

目的：

- 确认正式用户路径没坏

### 2.1 回归集

```bash
uv run dite scan docs/test/valid --verbose --output "$OUTDIR/scan-valid.json" --no-cache 2>&1 | tee "$OUTDIR/scan-valid.log"
uv run dite organize docs/test/valid --verbose --dry-run --no-cache 2>&1 | tee "$OUTDIR/organize-valid.log"
uv run dite pdf-check docs/test/valid --verbose --no-cache 2>&1 | tee "$OUTDIR/pdf-check-valid.log"
```

### 2.2 代表性集

```bash
uv run dite scan docs/test/rep --verbose --output "$OUTDIR/scan-rep.json" --no-cache 2>&1 | tee "$OUTDIR/scan-rep.log"
uv run dite organize docs/test/rep --verbose --dry-run --no-cache 2>&1 | tee "$OUTDIR/organize-rep.log"
```

### 2.3 对抗集

```bash
uv run dite scan docs/test/adv --verbose --output "$OUTDIR/scan-adv.json" --no-cache 2>&1 | tee "$OUTDIR/scan-adv.log"
uv run dite organize docs/test/adv --verbose --dry-run --no-cache 2>&1 | tee "$OUTDIR/organize-adv.log"
```

通过标准：

- 命令成功退出
- JSON 报告写出成功
- `organize --dry-run` 无异常
- `pdf-check` 如果失败，必须是样本内容问题，而不是程序崩溃

## 阶段 3：输入模式 A/B

目的：

- 回答“文件名混入 embedding 是否在拉偏结果”

优先顺序：

1. `rep`
2. `adv`
3. `valid`

### 3.1 代表性集

```bash
uv run python tools/cluster_experiments.py compare-inputs docs/test/rep --output "$OUTDIR/compare-rep.json" --no-cache 2>&1 | tee "$OUTDIR/compare-rep.log"
```

### 3.2 对抗集

```bash
uv run python tools/cluster_experiments.py compare-inputs docs/test/adv --output "$OUTDIR/compare-adv.json" --no-cache 2>&1 | tee "$OUTDIR/compare-adv.log"
```

### 3.3 回归集

```bash
uv run python tools/cluster_experiments.py compare-inputs docs/test/valid --output "$OUTDIR/compare-valid.json" --no-cache 2>&1 | tee "$OUTDIR/compare-valid.log"
```

重点看：

- `diff.entries`
- `initial_num_clusters_delta`
- `final_num_clusters_delta`
- `num_noise_delta`
- `small_clusters_merged_delta`
- `name_clusters_merged_delta`
- `total_clusters_merged_delta`

## 阶段 4：基础参数 sweep

目的：

- 在不引入更重流程的前提下比较默认参数邻域

### 4.1 代表性集

```bash
uv run python tools/cluster_experiments.py sweep docs/test/rep --output "$OUTDIR/sweep-rep.json" --no-cache 2>&1 | tee "$OUTDIR/sweep-rep.log"
```

### 4.2 对抗集

```bash
uv run python tools/cluster_experiments.py sweep docs/test/adv --output "$OUTDIR/sweep-adv.json" --no-cache 2>&1 | tee "$OUTDIR/sweep-adv.log"
```

### 4.3 回归集

```bash
uv run python tools/cluster_experiments.py sweep docs/test/valid --output "$OUTDIR/sweep-valid.json" --no-cache 2>&1 | tee "$OUTDIR/sweep-valid.log"
```

重点看：

- `fragmentation_score`
- `num_noise`
- `final_num_clusters`
- `total_clusters_merged`

## 阶段 5：扩展实验

只有当基础 sweep 没给出清晰信号时，才跑这一阶段。

当前扩展实验包括：

- `PCA`
- `allow_single_cluster`
- `topic_clustering.mode: agglomerative`（自动轮廓系数切分，以及余弦距离阈值 0.3 / 0.4 / 0.5）
- `reducer: umap`（5 维和 10 维；只有装了 `umap-learn` 才会加入）
- `embedding_input_mode: summary`（默认 HDBSCAN 参数和上面几种替代聚类各跑一次）

有 `manifest.json` 时，结果按 `partition_metrics.bcubed_f1` 排序。判读时同时看 `bcubed_precision` 有没有下降：只涨召回、精确率掉了的方案，等于用误并换来的，不能晋升默认值。

命令：

```bash
uv run python tools/cluster_experiments.py sweep docs/test/rep --output "$OUTDIR/sweep-rep-extended.json" --no-cache --extended 2>&1 | tee "$OUTDIR/sweep-rep-extended.log"
```

不要一开始就在 `valid` 和 `adv` 上跑 `--extended`。

## 阶段 6：缓存稳定性检查

目的：

- 确认缓存启用后结果结构不漂移

建议至少对 `rep` 跑一轮：

```bash
uv run dite scan docs/test/rep --verbose --output "$OUTDIR/scan-rep-cached-1.json" 2>&1 | tee "$OUTDIR/scan-rep-cached-1.log"
uv run dite scan docs/test/rep --verbose --output "$OUTDIR/scan-rep-cached-2.json" 2>&1 | tee "$OUTDIR/scan-rep-cached-2.log"
uv run python tools/cluster_experiments.py compare-inputs docs/test/rep --output "$OUTDIR/compare-rep-cached.json" 2>&1 | tee "$OUTDIR/compare-rep-cached.log"
```

重点看：

- 第二次 `scan` 的缓存命中是否明显增加
- 聚类结果有没有异常漂移
- 内部实验在缓存开启时是否仍然稳定

## 快速查看结果

列出本轮产物：

```bash
find "$OUTDIR" -maxdepth 1 -type f | sort
```

查看 A/B 差值摘要：

```bash
python - <<'PY' "$OUTDIR/compare-rep.json" "$OUTDIR/compare-adv.json" "$OUTDIR/compare-valid.json"
import json, sys
for path in sys.argv[1:]:
    data = json.load(open(path, encoding='utf-8'))
    print("\\n==", path)
    print(data["diff"]["summary"])
PY
```

查看 sweep 前 10 个组合：

```bash
python - <<'PY' "$OUTDIR/sweep-rep.json" "$OUTDIR/sweep-adv.json" "$OUTDIR/sweep-valid.json"
import json, sys
for path in sys.argv[1:]:
    data = json.load(open(path, encoding='utf-8'))
    print("\\n==", path)
    for row in data["runs"][:10]:
        print(
            row["run_id"],
            row["fragmentation_score"],
            row["summary"]["num_noise"],
            row["summary"]["final_num_clusters"],
            row["config"]["input_mode"],
            row["config"]["clustering"]["min_cluster_size"],
            row["config"]["clustering"]["min_samples"],
            row["config"]["clustering"]["cluster_selection_epsilon"],
            row["config"]["reducer"],
            row["config"]["allow_single_cluster"],
        )
PY
```

查看 `rep` 里哪些文件发生变化：

```bash
python - <<'PY' "$OUTDIR/compare-rep.json"
import json, sys
data = json.load(open(sys.argv[1], encoding='utf-8'))
changed = [x for x in data["diff"]["entries"] if x["changed"]]
print("changed:", len(changed))
for row in changed[:80]:
    print(row["path"])
    print("  from:", row["from_label"], row["from_name"], row["from_noise"])
    print("  to:  ", row["to_label"], row["to_name"], row["to_noise"])
PY
```

## 人工判读顺序

### `valid`

- 主要看回归稳定性
- 重复样本是否仍然稳定
- 已知回归样本是否被打坏

### `rep`

- 主要看趋势
- 哪些组合减少碎簇
- 哪些组合减少噪音
- 同时不能靠粗暴大并取得好看数字

### `adv`

- 主要看关系
- `must_link` 是否被拆
- `must_not_link` 是否被误并

## 停止条件

下面任一情况出现时，应该先停下来，不要继续盲跑更重实验：

- `compare-rep` 显示 `with_filename` 明显拉偏结果
- `sweep-rep` 的前几名组合都在 `adv` 上犯明显错误
- `valid` 的回归样本开始被打坏

## 当前推荐顺序

如果只跑一轮最务实的实验，当前推荐顺序是：

1. 基线测试
2. `valid/rep/adv` 的 CLI smoke
3. `compare-inputs rep`
4. `compare-inputs adv`
5. `compare-inputs valid`
6. `sweep rep`
7. 必要时再做 `sweep adv`、`sweep valid`
8. 只有在前面还没答案时，才做 `sweep rep --extended`
