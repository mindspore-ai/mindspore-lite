# Profiling 数据获取与解析（Benchmark + msprof）

本文档沉淀使用 MindSpore Lite Benchmark 在 Ascend 上采集 profiling 数据，并用 msprof 解析导出 summary 的完整流程；同时记录在实际执行中遇到的问题与解决办法。

## 1. 前置条件

- 已完成 ONNX→MindIR 转换，拿到可跑通的 MindIR（含 fp32 / fp16 两个版本均可）。
- 已配置好 Ascend + MindSpore Lite 运行环境，并确保当前 shell 使用的 CANN 与推理环境一致。建议统一：

```bash
source /path_to/env.sh
```

确保 `$Benchmark`、`$Convert`、`msprof` 等命令可直接使用。

## 2. Profiling 配置文件准备

建议为每个场景（例如 fp32_s1280 / fp16_s1280）建立独立目录，避免相互覆盖：

```bash
mkdir -p 0529_prof_fp32_s1280
mkdir -p 0529_prof_fp16_s1280
```

推荐方式是直接用 `msprof` 作为外层采集工具；不再依赖 `Benchmark --configFile=prof.ini` 这种方式（一些场景下存在环境/路径不一致导致的问题）。

## 3. 使用 msprof 采集 profiling 原始数据（推荐）

核心点：使用 CANN 原生 `msprof` 命令包裹要运行的二进制（可以是推理脚本，也可以是 `$Benchmark`）。

### 3.1 确认使用的 msprof 来自当前环境

不要写死 `/usr/local/...` 的 Toolkit 路径。很多机器上同时存在“系统 CANN”（如 `/usr/local/Ascend/...`）与“个人 CANN”（如 home 目录下的安装/解压版本）。应以当前 shell 的 `PATH/LD_LIBRARY_PATH` 为准：

```bash
which msprof
msprof --help | head -n 20
echo "$LD_LIBRARY_PATH" | tr ":" "\n" | grep -E "ascend-toolkit|Ascend" | head -n 20
```

如果 `which msprof` 指向的路径与当前 `LD_LIBRARY_PATH` 中实际使用的 CANN 不一致，优先重新 `source env.sh`，保证 profiling 与推理使用同一套环境。

### 3.2 msprof 命令模板

原始命令用法（示例参数可按需要调整）：

```bash
msprof \
  --output=./profiling \
  --sys-hardware-mem=on \
  --sys-hardware-mem-freq=100 \
  --ai-core=on \
  --aic-freq=100 \
  <需要执行的脚本或者二进制文件> [args...]
```

以 `$Benchmark` 作为被采集程序（seq_len=1280，4 输入）示例：

```bash
export GLOG_v=1
export GLOG_logtostderr=1

msprof \
  --output=./profiling \
  --sys-hardware-mem=on \
  --sys-hardware-mem-freq=100 \
  --ai-core=on \
  --aic-freq=100 \
  $Benchmark \
    --modelFile=/path/to/xxx_graph.mindir \
    --device=Ascend \
    --inputShape="input_ids:1,1280;attention_mask:1,1280;doc_token_indices:1,64;query_token_index:1,1" \
  2>&1 | tee benchmark_msprof.log
```

执行完成后，检查 `./profiling/` 下是否生成 `PROF_*` 目录：

```bash
ls -la profiling
```

## 4. 使用 msprof 解析并导出 summary

### 4.1 执行 export summary

在 profiling 目录中选择一个 PROF 目录（例如 `profiling/PROF_xxx`），执行：

```bash
msprof export summary \
  -dir profiling/PROF_xxx \
  --model-id=<MODEL_ID> \
  --iteration-id=10 \
  2>&1 | tee msprof_export.log
```

成功后，会在该 PROF 目录下生成：

- `mindstudio_profiler_output/`：导出的 summary（csv、README 等）
- （有时）`device_0/sqlite/`、`host/sqlite/`：解析中间库文件

常见文件示例：

- `mindstudio_profiler_output/op_statistic_*.csv`
- `mindstudio_profiler_output/op_summary_*.csv`
- `mindstudio_profiler_output/task_time_*.csv`
- `mindstudio_profiler_output/step_trace_*.csv`
- `mindstudio_profiler_output/aicpu_*.csv`

## 5. PROF 目录重命名与打包

建议把 `profiling/PROF_xxx` 改名为可识别的目录名，例如：

- `PROF_jina_fp32_bucket_s1280`
- `PROF_jina_fp16_bucket_s1280`

示例：

```bash
mv profiling/PROF_xxx profiling/PROF_jina_fp32_bucket_s1280
```

然后打包：

```bash
tar -czf /dest-path/PROF_jina_fp32_bucket_s1280.tar.gz \
  -C profiling PROF_jina_fp32_bucket_s1280
```

同理为 fp16 场景生成另一份 tar.gz。

## 6. 常见问题与解决方案

### 6.1 msprof 提示 model id 无效

现象：

```text
The model id 1 is invalid. Must select from {2147483648}.
```

原因：

- `msprof export summary` 的 `--model-id` 必须与该次采集数据中记录的 model id 一致。
- model id 可能为 1，也可能为较大的固定值（例如 `2147483648`），取决于运行时与模型加载方式。

解决方案：

1. 直接从 msprof 报错信息中获取合法的 model id（它会打印候选集合）。
2. 或从 Benchmark 日志中找到实际 modelId（示例）：

```text
modelId[2147483648]
```

3. 用该值重新执行 msprof：

```bash
msprof export summary -dir profiling/PROF_xxx --model-id=2147483648 --iteration-id=10
```

### 6.2 msprof 提示路径不存在

现象：

```text
The path "profiling/PROF_xxx" does not exist.
```

原因：

- PROF 目录已被重命名（例如从 `PROF_000001_...` 改成了 `PROF_jina_fp32_bucket_s1280`），但命令仍用旧路径。

解决方案：

- 使用当前实际目录名执行：

```bash
msprof export summary -dir profiling/PROF_jina_fp32_bucket_s1280 --model-id=... --iteration-id=10
```

### 6.3 msprof 提示 host 没有 summary 数据

现象：

```text
There is no summary data to export for ".../host"
```

说明：

- 这通常不影响 device 侧 summary 导出；最终仍会在 `mindstudio_profiler_output/` 产出 csv。

处理建议：

- 以 `mindstudio_profiler_output/` 是否生成为准；只要 csv 生成即可继续下一步打包。

### 6.4 重复执行 export summary 速度很快

现象：

```text
The data ... has been analyzed. Parsing phase will be skipped.
```

说明：

- msprof 会检测到 sqlite/complete 标记，从而跳过重复解析，属于正常行为。

### 6.5 多个 PROF 目录如何选

现象：

- `profiling/` 下出现多个 `PROF_*` 目录（多次运行 Benchmark）。

处理建议：

- 选最新生成的（按时间戳/目录名判断），或每次运行前先清理 `profiling/`，避免混淆：

```bash
rm -rf profiling
```

### 6.6 profiling 采集可能出现二次失败

现象：

- 一次 profiling 导出完成后，后续又出现 `Running profiling failed`。

处理建议：

- 以生成的 `profiling/PROF_*` 与 `mindstudio_profiler_output/*.csv` 为准，存在即可重命名并打包；必要时单独开目录只跑一次采集。
