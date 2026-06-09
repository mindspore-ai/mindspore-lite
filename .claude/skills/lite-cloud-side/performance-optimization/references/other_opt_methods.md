---
name: "mslite-ascend-perf-opt"
description: "MindSpore Lite/Ascend 性能优化攻略与命令模板。做基线/benchmark、msprof profiling、混合精度、融合/解融合、算子替换、精度对齐与归档时调用。"
---

# MindSpore Lite（Ascend）性能优化 Playbook

## 目标与输出物

- 目标：把端到端 **Model execute time**（开启 `GLOG_v=1` 后从日志抓取，单位 `us`）从基线优化到目标值，同时保持精度达标。
- 必须产出：
  - 基线与每次尝试的 `benchmark*.log`（记录 `PredictFromHost] Model execute in ... us`，并建议同时记录 `AclrtMemcpy` 的 `H2D/D2H`）
  - 每次尝试的 profiling：`op_statistic_*.csv` + `op_summary_*.csv`
  - 每次尝试的精度对齐结果（vs non-fuse ONNX/其他基线）
  - 一张“尝试表格”（阶段/变更点/execute(us)/H2D(us)/D2H(us)/相对基线/精度/产物路径）

## 工作流（Agent 必须遵守）

一句话原则：一次只改一个点；只要“精度达标 + 性能收益”，就把该版本定为新基线，再尝试下一项手段。

### A. 固定口径（不统一口径就不要开始优化）

- 固定输入 shape（例如 `(1,1280)`），固定 `$Benchmark --warmUpLoopCount/--loopCount`。
- 每次尝试必须保留 4 份产物：
  - 导出日志（export）
  - 转换日志（convert）
  - 性能日志（benchmark）
  - 精度对齐日志（accuracy）
- profiling 只用于解释“为什么”，不能替代 benchmark 的口径。

### B. “一次一改”闭环

对每个优化点，按下面顺序跑完再进入下一项：

1) 导出 fused ONNX（用于 MSLite/Ascend） + non-fuse ONNX（用于精度基线）
2) `$Convert` → fused MindIR
3) 开启 `GLOG_v=1` 跑 `$Benchmark`，从日志抓取 `Model execute in ... us`（并可同步抓取 `H2D/D2H` copy）
4) 精度对齐（推荐 non-fuse ONNXRuntime CPU 作为基线）
5)（可选但建议）`msprof` profiling 导出 `op_statistic/op_summary`，定位瓶颈与副作用
6) 更新尝试表（包含 MindIR 路径、日志路径、profiling 包路径）

## 固定口径（先统一，再开始优化）

- 性能口径：开启 `GLOG_v=1` 后，从日志抓取：
  - `PredictFromHost] Model execute in ... us`
  - `AclrtMemcpy] [H2D] Host to Device copy in ... us`
  - `AclrtMemcpy] [D2H] Device to Host copy in ... us`
  固定输入 shape（例如 `(1,1280)`），固定 `LoopCount/WarmUpLoopCount`。
- profiling 口径：用 `msprof` 包裹 `$Benchmark`，并固定 `--warmUpLoopCount=0 --loopCount=1`，避免 count 倍增干扰。
- 精度口径：同一组 query/docs，比较两端输出 score 向量的 `max_abs_diff/mean_abs_diff/rmse/cosine`，同时确认 top-k 排序一致。

## 0. 环境与目录规范

```bash
cd <repo_or_example_dir>
source /path-to/env.sh
```
注意：/path-to/env.sh包含CANN包以及Mindspore Lite Convert以及Benchmark工具

- 强制每个尝试用独立目录：`OUT=./<date>_<trial_name>_s<seq>`
- 强制用相对路径写输出，避免权限/路径问题
- 只用 `$Convert`/`$Benchmark`（来自 env.sh）以保证工具链一致

## 1. 基线跑通（Benchmark）

准备统一输入：

```bash
SHAPE="input_ids:1,1280;attention_mask:1,1280;doc_token_indices:1,64;query_token_index:1,1"
```

跑基线：

```bash
export GLOG_v=1
export GLOG_logtostderr=1

$Benchmark \
  --modelFile=<path/to/model_graph.mindir> \
  --device=Ascend \
  --inputShape="$SHAPE" \
  2>&1 | tee <OUT>/benchmark_s1280.log
```

记录（从 `benchmark_s1280.log` 抓取）：

- `PredictFromHost] Model execute in ... us`
- `AclrtMemcpy] [H2D] Host to Device copy in ... us`
- `AclrtMemcpy] [D2H] Device to Host copy in ... us`

建议抓取命令：

```bash
grep -E "PredictFromHost\\] Model execute in|AclrtMemcpy\\] \\[(H2D|D2H)\\] " <OUT>/benchmark_s1280.log
```

并记录当前使用的 ini / onnx / mindir 路径。

## 2. Profiling（msprof → export summary）

```bash
msprof \
  --output=<OUT>/prof/profiling \
  --sys-hardware-mem=on --sys-hardware-mem-freq=100 \
  --ai-core=on --aic-freq=100 \
  $Benchmark \
    --modelFile=<path/to/model_graph.mindir> \
    --device=Ascend \
    --inputShape="$SHAPE" \
    --warmUpLoopCount=0 --loopCount=1 \
  2>&1 | tee <OUT>/prof/benchmark_msprof.log
```

导出 summary（model-id 常见为 1 或 2147483648，以 query 输出为准）：

```bash
PROF_DIR=$(ls -1d <OUT>/prof/profiling/PROF_* | tail -n 1)
msprof export summary -dir "$PROF_DIR" --model-id=1 --iteration-id=10 \
  2>&1 | tee <OUT>/prof/msprof_export.log
```

重点看：
- `op_statistic_*.csv`：热点算子（Total Time/Ratio/Count）
- `op_summary_*.csv`：热点算子对应的节点名、输入 dtype/layout、是否被融合为 Custom

### 2.1 Profiling 归档与打包（只保留必要 CSV）

推荐归档结构（便于后续复验、对比、压缩包体积）：

```
<OUT>/prof_pack/<case_name>/
  op_statistic.csv
  op_summary.csv
```

把多个 case 打包：

```bash
tar -czf <DATE>_profiling_csvs_<cases>.tar.gz -C <OUT>/prof_pack .
```

## 3. 精度对齐（推荐：non-fuse ONNX 作为基线）

### 3.1 生成 non-fuse ONNX（只要一次）

```bash
python3 export_xxx_onnx.py --disable-fusion-opt ...
```

### 3.2 对齐命令模板

```bash
TOK=<tokenizer_or_weight_path>
NONFUSE_ONNX=<path/to/non_fuse.onnx>
MINDIR=<path/to/model_graph.mindir>

python3 infer_xxx_onnx.py --model-path "$NONFUSE_ONNX" --tokenizer "$TOK" --device CPU --mode listwise --max-length 1280 > <OUT>/infer_onnx.log
python3 infer_xxx_mslite.py --model-path "$MINDIR" --tokenizer "$TOK" --device ascend --device-id 0 --mode listwise --max-length 1280 > <OUT>/infer_mslite.log
```

解析并输出指标（示例：从日志中抓 `Score:`）：

```bash
python3 - <<'PY'
import re, numpy as np
def parse_scores(p):
    t=open(p,'r',encoding='utf-8',errors='ignore').read()
    return np.array([float(m.group(1)) for m in re.finditer(r"Score:\\s*([-+]?\\d+\\.\\d+(?:[eE][-+]?\\d+)?)", t)], dtype=np.float64)
def metrics(a,b):
    d=np.abs(a-b)
    return dict(
        max_abs_diff=float(d.max()),
        mean_abs_diff=float(d.mean()),
        rmse=float(np.sqrt(((a-b)**2).mean())),
        cosine=float(a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)),
    )
onnx=parse_scores("<OUT>/infer_onnx.log")
ms=parse_scores("<OUT>/infer_mslite.log")
assert onnx.shape==ms.shape, (onnx.shape, ms.shape)
print(metrics(onnx, ms))
PY
```

## 4. 优化手段清单（可扩展，带“验证模板”）

本章是给 Agent 的“优化手段目录”。每一项都按同一结构组织，方便以后持续加新手段：

- 目的（Hypothesis）
- 改动方式（Change）
- 验证方式（Validate：benchmark + accuracy + profiling）
- 常见副作用（Pitfalls）
- 回退方式（Rollback）
- 经验备注（Case notes，可选）

### 4.1 基线构建与对照（先把“地基”打稳）

- 目的：明确优化的对照组，避免“换了口径但以为变快”。
- 改动方式：
  - 生成 fused（Ascend）与 non-fuse（ORT）两套导出产物，并强制隔离用途。
  - fixed shape / dynamic bucket 必须在同一口径下比较。
- 验证方式：
  - 同一组输入，固定 `$Benchmark` 参数与 seq_len 档位。
  - 精度对齐以 non-fuse ORT 为基线。
- 常见副作用：
  - non-fuse ONNX 不能跑 fused Custom。
  - listwise 推理如果分块策略错误，会导致多次 predict，端到端时间虚高。

### 4.2 Attention / PFA（PromptFlashAttention）路径优化

#### 4.2.1 GQA 适配（num_key_value_heads）

- 目的：让后端按 GQA 走更优实现，避免图内显式 KV repeat/tile。
- 改动方式：
  - `Custom(PromptFlashAttention)` 增加 `num_key_value_heads_i` 属性。
  - tracing forward 如遇 head 维不匹配，只在 forward 做 repeat 以通过导出，但保持最终图仍为 Custom 节点。
- 验证方式：
  - profiling 看 `TileD`/repeat 类算子是否消失。
  - 精度必须对齐（关注 softmax/归一化链路）。
- 常见副作用：
  - tracing repeat 如果落图，会引入大量算子，反而变慢。

#### 4.2.2 PFA layout（减少 Transpose/TransData）

- 目的：减少 layout 转换与 transpose。
- 改动方式：
  - `Custom(PromptFlashAttention)` 设置 `input_layout_s`，并同步调整 PFA 输出后处理（slice/reshape 顺序）。
- 验证方式：
  - profiling 对比 `Transpose/TransData` 的 Count 与 TotalTime。
- 常见副作用：
  - layout 变化会影响后续算子选型；必须逐步验证，必要时回退。

#### 4.2.3 PFA num_heads 拆分（经验：收益不稳定/通常无收益）

- 目的：假设拆分 heads 可能提高并行/调度效率。
- 改动方式：
  - 将 `query` 的 head 维拆成两半，串行执行两个 PFA，最后 concat。
  - 必须保证输入 shape/属性与对照完全一致再比较。
- 验证方式：
  - isolated PFA：单 PFA vs split2 PFA 做 benchmark + profiling。
  - 若 isolated 都无收益，不建议上整网。
- 常见副作用：
  - PFA 变成两次调用，框架侧开销（concat/cast）可能抵消一切收益。

### 4.3 QKV/QK 合并类（减少 GEMM 数，但容易引入 Split/TransData）

#### 4.3.1 QKV 合并（q_proj/k_proj/v_proj → qkv_proj）

- 目的：减少 Q/K/V 三次投影 GEMM。
- 改动方式：
  - fused 导出路径注入 `qkv_proj`，forward 里 split 回 q/k/v。
- 验证方式：
  - profiling 必看：`StridedSliceD/Split*/Concat*` 是否暴涨；`TransData` 总耗时是否升高。
- 常见副作用：
  - “省掉 GEMM”但“新增大量切片/搬运”，整体更慢。
  - 图规模可能膨胀，转换日志出现 protobuf 过大警告。

#### 4.3.2 QK 合并（仅合并 q_proj + k_proj）

- 目的：比 QKV 合并更保守，尝试减少一部分 GEMM。
- 改动方式：
  - 只合并 Q/K，V 保持不变；split 回 Q/K。
- 验证方式：
  - 重点看 concat/split、cast、transpose 是否带来额外开销。
- 常见副作用：
  - 仍然可能变慢（concat/split/格式搬运抵消收益）。

### 4.4 GEMM/BMM/MatMulV2 替换类（影响后端内核选择）

#### 4.4.1 BatchMatMul → MatMul / MatMulV2（Custom）

- 目的：让后端走更优 MatMul 内核或避免低效的 BatchMatMul。
- 改动方式：
  - 在导出侧用 Custom(MatMulV2) 替换目标 MatMul 子图（通常在 projection 或其他线性层上）。
- 验证方式：
  - profiling 看 `BatchMatMul` 是否消失，是否出现 `MatMul/MatMulV2`，并对比 `TransData/Cast/Concat` 的副作用。
- 常见副作用：
  - 替换后可能引入 concat/add/cast 等额外算子，导致整网反而变慢。

### 4.5 精度策略类（混合精度 / fp16）

#### 4.5.1 allow_mix_precision + mixlist（推荐的“可控”路径）

- 目的：只让少数热点下沉 fp16，同时保护归一化/softmax 等敏感链路为 fp32。
- 改动方式：
  - converter 配置：`ge.exec.precision_mode=allow_mix_precision`
  - 指定 mixlist：`ge.exec.modify_mixlist=...json`
- 验证方式：
  - 精度必须严查：score 是否越界（如 cosine similarity > 1/< -1）、排序是否改变。
  - profiling 必看：`Cast/TransData` 是否爆炸性增长。
- 常见副作用：
  - 归一化链路（ReduceMean/Sqrt/Rsqrt/Reciprocal/RealDiv/Square 等）若下沉 fp16，可能导致数值崩坏。

#### 4.5.2 force_fp16（不推荐作为默认基线）

- 目的：追求极限性能。
- 改动方式：`ge.exec.precision_mode=force_fp16`
- 验证方式：精度对齐通常更难通过，且可能出现整体偏移/越界。
- 常见副作用：精度不通过时不应继续叠加其他优化点，必须先收敛精度策略。

### 4.6 融合/解融合类（用于“极限探测”与“意外收益”）

#### 4.6.1 禁用某类融合（例：RmsNorm/AddRmsNorm）

- 目的：验证 fused kernel 是否低效；拆开后是否能触发更优后端实现。
- 改动方式：导出脚本提供开关（例如 `--disable_rmsnorm_fusion`）。
- 验证方式：
  - benchmark 直接对比；profiling 看 RmsNorm/归一化链路总耗时变化。
- 常见副作用：
  - 图变大、算子变多；必须精度对齐（norm 对精度敏感）。

### 4.7 免拷贝
待补充...

### 4.8 量化
待补充...

### 4.9 KV Cache
待补充...

### 4.10 AOE
待补充...

## 5. 转换失败/运行失败快速定位（Ascend 常用）

- 转换失败优先看 plog：
  - 从 converter 输出中提取进程号（例如 `LITE(<pid>,...)`）
  - 打开：`${HOME}/ascend/log/debug/plog/plog-<pid>_*.log`
- 线上编译/算子选择失败时，重点关注：
  - “No supported Ops kernel and engine are found”
  - fault op_name / task_id（常与 dtype/layout 不匹配有关）

## 6. 归档模板（建议粘贴到项目的 xxx_PERF_OPT_TRIALS.md）

```
| 阶段 | 变更点 | execute(us) | H2D(us) | D2H(us) | 相对基线 | 精度 | 产物 |
|------|--------|-------------|---------|---------|----------|------|------|
| 基线 | ...    | ...         | ...     | ...     | 0        | 通过 | ...  |
| 尝试X| ...    | ...         | ...     | ...     | ...      | ...  | ...  |
```

归档时强制附上：
- 导出/转换/benchmark/profiling/精度对齐的命令与日志路径
- `op_statistic/op_summary` 的路径或 tar.gz 包路径


## 8. 项目内参考资料（建议 Agent 先读）

- [`get_profiling_data.md`](../../open-source-model-migration/references/get_profiling_data.md)：msprof 采集、export summary、打包归档的标准流程
