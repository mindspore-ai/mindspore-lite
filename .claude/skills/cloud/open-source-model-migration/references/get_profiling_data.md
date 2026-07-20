# Profiling 数据获取与解析（Benchmark + msprof + torch_npu）

本文档沉淀在 Ascend 上采集 profiling 数据的完整流程，覆盖两条互补的路径：

- **路径 A：MindSpore Lite Benchmark + msprof CLI**（§2–§5）— 用于已经迁移到 MindIR、需要拿到 op_summary / op_statistic 来做算子级分析的场景。
- **路径 B：torch_npu `profiler.profile` API**（§6）— 用于 PyTorch 原生模型（迁移前 / 对照基线），或需要对推理脚本中某一段 Python 代码做精细 trace 的场景。

两条路径产出的 CSV 列含义一致（详见 §7），可以直接互相 diff；典型用法是同时跑两路、对比同一算子在两边是否落到相同 NPU kernel（详见 §8）。

## 1. 前置条件

- 已配置好 Ascend + 推理环境（MindSpore Lite 或 torch_npu，至少其一），并确保当前 shell 使用的 CANN 与推理环境一致。建议统一：

```bash
source /path_to/env.sh
```

- 路径 A 还要求 `$Benchmark`、`$Convert`、`msprof` 等命令可直接使用。
- 路径 B 还要求 `torch_npu` 已安装并能 `import torch_npu`。

确认 msprof / torch_npu 来源与当前 CANN 一致（很多机器上同时存在系统 CANN 与个人 CANN，混用会采集到错误的 aic 算子名）：

```bash
which msprof
echo "$LD_LIBRARY_PATH" | tr ":" "\n" | grep -E "ascend-toolkit|Ascend" | head -n 5
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
```

## 2. 路径 A — MindSpore Lite Benchmark + msprof：配置准备

建议为每个场景（例如 fp32_s1280 / fp16_s1280）建立独立目录，避免相互覆盖：

```bash
mkdir -p 0529_prof_fp32_s1280
mkdir -p 0529_prof_fp16_s1280
```

推荐方式是直接用 `msprof` 作为外层采集工具；不再依赖 `Benchmark --configFile=prof.ini` 这种方式（一些场景下存在环境/路径不一致导致的问题）。

## 3. 路径 A — 使用 msprof 采集 profiling 原始数据（推荐）

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

## 4. 路径 A — 使用 msprof 解析并导出 summary

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

## 5. 路径 A — PROF 目录重命名与打包

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

## 6. 路径 B — torch_npu `profiler.profile` API

适用于：

- PyTorch 原生模型（迁移前的对照基线）；
- 已经迁到 MindIR 但需要把"同一份输入"在 torch_npu 上跑一遍、看落到了哪些 NPU kernel；
- 想对推理脚本里某一段 Python 代码做 trace（不需要看整个进程的 acl 调用）。

与路径 A 的关键区别：

| | 路径 A (Benchmark + msprof CLI) | 路径 B (torch_npu profiler API) |
|---|---|---|
| 调用方式 | `msprof` 包住 Benchmark 二进制 | Python 里 `with prof:` 或 `prof.start()/stop()` |
| 采集范围 | 整个被采集进程的 ACL 调用 | profiler 上下文内的 ACL 调用 |
| 解析步骤 | 需要再跑一次 `msprof export summary` | **不需要**，profiler 内置解析器会自动产出 CSV |
| 输出目录 | `PROF_xxx/mindstudio_profiler_output/op_summary_*.csv` | `ascend{pid}_{ts}_ascend_pt/ASCEND_PROFILER_OUTPUT/{op_statistic,operator_details,kernel_details}.csv` |
| CSV 列 | 与路径 B 一致 | 与路径 A 一致 |

### 6.1 前置条件

```bash
source /path_to/env.sh   # 保证 torch_npu 与 msprof 走同一套 CANN
python -c "import torch_npu; print(torch_npu.__version__)"
```

### 6.2 最小可运行模板

```python
import torch
import torch_npu
import torch_npu.profiler

PROF_DIR = "./profiling_torch_npu"

# 推理模型/输入准备就绪后（model.eval()、half()、samples/mask/caption 等已搬到 npu）
# ——先做 1~2 次 warmup，避免 JIT/lazy 编译算子耗时混入 profile 区间
with torch.no_grad():
    _ = model(samples, mask, caption)
torch.npu.synchronize()

experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
)
prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        PROF_DIR, analyse_flag=True
    ),
    record_shapes=True,
    profile_memory=False,    # True 会显著拖慢，按需开
    with_stack=False,
    with_flops=False,
    with_modules=True,
    experimental_config=experimental_config,
)

prof.start()
with torch.no_grad():
    _ = model(samples, mask, caption)
torch.npu.synchronize()
prof.stop()
```

`on_trace_ready` 里的 `analyse_flag=True` 让 profiler 在 stop 后**自动解析**原始数据并生成 CSV，**无需再跑 `msprof export summary`**。

### 6.3 推荐：放在独立进程里跑

torch_npu 加载一个大模型后，ACL context 会占住约 8 GB HBM 且**不随 `del + gc.collect + empty_cache` 释放**，要等进程退出才归还。如果同一进程里再起 MindSpore Lite / msprof 子进程，常常 OOM（典型报错：`Memory_Allocation_Failure ... rtMalloc ... size:17179934720`）。

解决方法：

1. **torch_npu profiling 独占一个 Python 进程**：只加载 torch_npu 模型、跑 warmup + 1 次 profiled 推理、立即退出。
2. **MSLite profiling 独占另一个 Python 进程**：用 `msprof` 包一个只跑 MindIR 的小脚本（如 `_MSLITE_RUNNER_TEMPLATE`），不要在同一进程里先跑 torch_npu。
3. 如果只有一张卡、被其他进程占用部分 HBM，先用 `npu-smi info` 找一张空卡（例如 device 1），通过 `ASCEND_RT_VISIBLE_DEVICES` 或 `torch.npu.set_device(id)` 切过去。

独立脚本骨架（torch_npu 侧）：

```python
# _prof_torch_npu.py
import os, sys, torch, torch_npu
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")

# 1) 加载模型 + 输入（按业务代码准备）
model = ...           # .npu().half().eval()
samples, mask, cap = ...   # 已搬到 npu

# 2) warmup（不要 profile）
with torch.no_grad():
    _ = model(samples, mask, cap)
torch.npu.synchronize()

# 3) 仅 profile 一次推理（套用 §6.2 的 prof.start/stop 模板）
# 4) 进程退出，HBM 归还
```

### 6.4 输出目录与 CSV 文件名

torch_npu profiler 自动产出的目录结构（与路径 A **不同**，但列含义一致）：

```text
profiling_torch_npu/
└── ascend{pid}_{ts}_ascend_pt/
    └── ASCEND_PROFILER_OUTPUT/
        ├── op_statistic.csv         # 算子类型聚合：OP Type / Count / Total Time(us) / Ratio(%)
        ├── operator_details.csv     # host 侧算子：Name / Input Shapes / Host/Device Duration
        ├── kernel_details.csv       # NPU kernel：Name / Type / Input Shapes / Input Data Types /
                                     #                   Output Shapes / Output Data Types / Duration(us) /
                                     #                   Block Dim / aicore_time / vec_time / ...
        ├── api_statistic.csv
        ├── memory_record.csv        # profile_memory=True 时才有
        ├── step_trace_time.csv
        └── npu_module_mem.csv
```

与路径 A 的 CSV 名称对应关系：

| 路径 A (`mindstudio_profiler_output/`) | 路径 B (`ASCEND_PROFILER_OUTPUT/`) | 含义 |
|---|---|---|
| `op_statistic_*.csv` | `op_statistic.csv` | 按 OP Type 聚合 |
| `op_summary_*.csv` | `kernel_details.csv` | 逐个 NPU kernel（Op Name + Type + I/O shapes/dtypes + 耗时） |
| — | `operator_details.csv` | host 侧算子调用栈 + 耗时（路径 A 没有对应） |

> 在路径 B 里查"某个算子落到了哪个 NPU kernel" 用 `kernel_details.csv`；查"该算子类型的总耗时占比" 用 `op_statistic.csv`。

### 6.5 采集后补做解析（可选）

如果 profiler 退出时 `analyse_flag=True` 没跑完（例如进程被 kill），可以手动补一次解析：

```python
from torch_npu.profiler.profiler import analyse
analyse(profiler_path="/path/to/profiling_torch_npu/ascend{pid}_{ts}_ascend_pt/")
```

## 7. CSV 列含义速查（两路通用）

以下表格用于跨路径 diff 时快速对照列名。两路 CSV 的**列名略有不同**，但表达的字段是一致的。

### 7.1 op_statistic.csv / op_statistic_*.csv

| 列名 | 含义 |
|---|---|
| `Device_id` | NPU device id |
| `OP Type` | NPU kernel 类型名（如 `TopKD`、`BatchMatMul`、`Cast`） |
| `Core Type` | `AI_CORE` / `AICPU` / ... |
| `Count` | 该 OP Type 在采集区间内被调用的次数 |
| `Total Time(us)` / `Min` / `Avg` / `Max` | 耗时统计（微秒） |
| `Ratio(%)` | 占整个采集区间总耗时的比例 |

### 7.2 kernel_details.csv / op_summary_*.csv

最常用的几列：

| 列名 | 含义 |
|---|---|
| `Name` / `Op Name` | kernel 实例名（路径 A 通常带拓扑路径，如 `/transformer/TopK`；路径 B 通常是 `TopKV2437` 这类自动编号） |
| `Type` / `OP Type` | kernel 类型名（与 op_statistic 的 `OP Type` 对齐） |
| `Input Shapes` | 形如 `"1,87040;2048"`，分号分隔多输入 |
| `Input Data Types` | 形如 `FLOAT;FLOAT16` |
| `Output Shapes` / `Output Data Types` | 同上 |
| `Duration(us)` / `Task Duration(us)` | 该 kernel 实际执行耗时 |
| `Block Dim` | 并行规模（典型值 8） |

剩余 `aicore_time` / `vec_time` / `mac_time` / `mte1_time` / `mte2_time` 等是分项耗时，用于判断 kernel 是 cube-bound 还是 vector-bound，做性能调优时再用。

## 8. 对比两路 profiling 数据（找算子差异 / kernel 差异）

迁移精度问题排查时的典型流程：**对同一份输入，分别在 torch_npu 和 MindIR (MSLite) 上跑一遍推理并各自采集 profiling，然后看每个关键算子在两边是否落到同一个 NPU kernel**。

### 8.1 操作步骤

1. 准备一份固定的输入（推荐保存为 `.npz`，两边都从这个 npz 读，保证输入字节一致）：

   ```python
   import numpy as np
   np.savez("inputs.npz", samples=samples_np, mask=mask_np, input_embeddings=emb_np)
   ```

2. 用 §6.3 的独立进程模板在 torch_npu 上跑一次 profiling，输出到 `profiling_torch_npu/`。

3. 用 §3 的 msprof + 独立 runner 在 MSLite 上跑一次 profiling，输出到 `profiling_mslite/`：

   ```bash
   msprof --output=./profiling_mslite --ai-core=on --aic-freq=100 \
          --aic-metrics=PipeUtilization --model-execution=on \
          python _mslite_runner.py   # 内部仅 build_from_file + predict(inputs.npz)
   ```

4. 对比 op_statistic 里的 OP Type 集合，确认两边用到的算子类型一致；再对关键算子（如 `TopKD`、`BatchMatMul`）查 kernel_details / op_summary 的 Input/Output Shapes + Data Types 是否一致。

### 8.2 一段最小对比脚本

```python
import csv, glob

def load_op_stat(path):
    with open(path) as f:
        return {r["OP Type"]: r for r in csv.DictReader(f)}

torch_stat = load_op_stat("profiling_torch_npu/ascend*/ASCEND_PROFILER_OUTPUT/op_statistic.csv")
msl_stat   = load_op_stat(glob.glob("profiling_mslite/PROF_*/mindstudio_profiler_output/op_statistic_*.csv")[-1])

# 算子类型集合差异
only_torch = set(torch_stat) - set(msl_stat)
only_msl   = set(msl_stat)   - set(torch_stat)
print("only in torch_npu:", only_torch)
print("only in mslite   :", only_msl)

# 共有算子的耗时对比
for op in sorted(set(torch_stat) & set(msl_stat)):
    t, m = torch_stat[op], msl_stat[op]
    print(f"{op:20s} torch={t['Count']}x{float(t['Avg Time(us)']):.1f}us  "
          f"mslite={m['Count']}x{float(m['Avg Time(us)']):.1f}us")
```

## 9. 常见问题与解决方案

### 9.1 msprof 提示 model id 无效

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

### 9.2 msprof 提示路径不存在

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

### 9.3 msprof 提示 host 没有 summary 数据

现象：

```text
There is no summary data to export for ".../host"
```

说明：

- 这通常不影响 device 侧 summary 导出；最终仍会在 `mindstudio_profiler_output/` 产出 csv。

处理建议：

- 以 `mindstudio_profiler_output/` 是否生成为准；只要 csv 生成即可继续下一步打包。

### 9.4 重复执行 export summary 速度很快

现象：

```text
The data ... has been analyzed. Parsing phase will be skipped.
```

说明：

- msprof 会检测到 sqlite/complete 标记，从而跳过重复解析，属于正常行为。

### 9.5 多个 PROF 目录如何选

现象：

- `profiling/` 下出现多个 `PROF_*` 目录（多次运行 Benchmark）。

处理建议：

- 选最新生成的（按时间戳/目录名判断），或每次运行前先清理 `profiling/`，避免混淆：

```bash
rm -rf profiling
```

### 9.6 profiling 采集可能出现二次失败

现象：

- 一次 profiling 导出完成后，后续又出现 `Running profiling failed`。

处理建议：

- 以生成的 `profiling/PROF_*` 与 `mindstudio_profiler_output/*.csv` 为准，存在即可重命名并打包；必要时单独开目录只跑一次采集。

### 9.7 torch_npu + MSLite 同进程 OOM（路径 B 特有）

现象：

```
Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by GE module.
Call rtMalloc fail, purpose: page caching, type = 2, size:17179934720, device_id:0
```

原因：

- torch_npu 加载大模型后，ACL context 会长期占住约 8 GB HBM，`del + gc.collect + empty_cache` 都不释放，必须进程退出才归还。同一进程里再让 MSLite / msprof 子进程 `build_from_file` 一个 MindIR（同样要 rtMalloc 一大块），常常 OOM。

处理建议：

- 严格按 §6.3：torch_npu profiling 与 MSLite profiling 分别在两个独立 Python 进程里跑。
- 如只有一张卡，先用 `npu-smi info` 找空卡，`ASCEND_RT_VISIBLE_DEVICES` 切过去。

### 9.8 torch_npu profiler 没产出 CSV

现象：

- `prof.stop()` 后输出目录里只有原始 `host/` + `device_x/`，没有 `ASCEND_PROFILER_OUTPUT/*.csv`。

原因：

- `on_trace_ready` 里 `analyse_flag=False`（或不传），profiler 不会自动解析。

处理建议：

- 在 `tensorboard_trace_handler(PROF_DIR, analyse_flag=True)` 显式传 `True`；或事后手动补一次解析（§6.5）：
  ```python
  from torch_npu.profiler.profiler import analyse
  analyse(profiler_path=".../ascend{pid}_{ts}_ascend_pt/")
  ```
