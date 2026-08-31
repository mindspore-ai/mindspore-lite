---
name: "cann-dump-locating"
description: "跨 CANN 版本精度差异的算子级定位流程。覆盖 Profiling 对比定位可疑算子、CANN Dump 配置、msaccucmp 转换、逐算子输入输出对比判断根因。"
---

# 跨 CANN 版本精度差异的算子级 Dump 定位

> 本文档是 `precision_troubleshooting` skill 的细化文档之一，对应场景 B（跨 CANN 版本精度定位）。
> 场景 A（三阶段对齐）见 [three_stage_alignment.md](three_stage_alignment.md)；通用原则与场景匹配见 [SKILL.md](../SKILL.md)。

本流程聚焦"不同 CANN 版本下、同一 MindSpore Lite 版本用 `convert` 转换出的不同 MindIR 模型推理精度差异定位"场景：通过 Profiling 对比和 CANN 算子 Dump 功能，从算子级别定位导致精度差异的根本原因。

## 术语与前提

- 离线转换工具：`converter_lite`（或环境里别名 `Convert`），将 ONNX 模型转换为 `.mindir` 模型文件
- Profiling 数据：`mindstudio_profiler_output/op_summary_*.csv`，包含算子维度 Block Num、Mix Block Num、Format 等信息
- CANN Dump：CANN 框架提供的算子级数据导出功能，可 dump 指定算子的输入输出 tensor 数据
- 对比基线：OK 版本（精度正确） vs NOT_OK 版本（精度异常）——均指 **CANN 版本**。每个 CANN 版本各自 `convert` 出一份 MindIR，并在各自 CANN 环境下推理；精度正确的那份 MindIR 作为基线，另一份为待定位对象。两份必须在相同输入数据下推理

## 精度定位流程

### Step 1：Profiling 对比分析（定位可疑算子）

从两个 CANN 版本（OK 版本和 NOT_OK 版本）的 `mindstudio_profiler_output/op_summary_*.csv` 文件中筛选存在差异的算子。

```bash
# 定位 Block Num 差异算子（按算子名对齐后对比）
diff <(grep "Block" ok/mindstudio_profiler_output/op_summary_*.csv | sort) \
     <(grep "Block" not_ok/mindstudio_profiler_output/op_summary_*.csv | sort)
```

> **前提**：该 diff 命令假设两个版本算子集合一致，仅个别算子 Block/Mix Block/Format 有差异。若两个版本算子集合本身不同（算子名都对不上），diff 会产生大量噪声，无法定位——此时应先核对算子集合是否一致（如按算子名 join 对比），确认一致后再用上述 diff。

关键对比指标：
- **Block Num**：算子 tiling 分块数量，可用于评估算子运行拆分的子任务数据量
- **Mix Block Num**：混合分块数量（与精度直接相关）
- **Format**：数据排布格式（NCHW、NHWC、NCHWc 等），数据排布差异可能引入数值精度变化

通过交叉对比定位可疑算子后，下一步通过 Dump 验证。

### Step 2：配置 CANN Dump 功能

#### dump_config.ini

```ini
[ascend_context]
dump_config_file=./dump.json
```

#### dump.json

```json
{
    "dump":{
        "dump_list":[
            {
                "model_name":"模型名",
                "layer":[
                    "ScatterElements_2740",
                    "Pow_2744/SquareReduceMean_2745"
                ]
            }
        ],
        "dump_path":"/path/to/dump",
        "dump_mode":"all",
        "dump_op_switch":"off"
    }
}
```

字段说明：
- **model_name**：MindIR 模型名称（不含 .mindir 后缀），必须与模型文件名一致
- **layer**：要 dump 的算子名称列表，格式为 `node_name` 或 `node_name/sub_op_name`
- **dump_path**：dump 文件输出目录
- **dump_mode**：`all` 表示 dump 算子的输入和输出
- **dump_op_switch**：设置为 `"off"` 表示仅 dump `dump_list` 中指定的算子

### Step 3：触发 Dump 并转换数据

#### 推理触发 Dump

```python
import mindspore_lite as mslite

net = mslite.Model()
context = mslite.Context()
context.target = ["ascend"]
context.ascend.device_id = 0

config_path = "dump_config.ini"
net.build_from_file("model.mindir", mslite.ModelType.MINDIR, context, config_path=config_path)

inputs_np = [input0, input1]  # 实际输入数据
res = net.predict(inputs_np)
```

运行推理后，会在 `dump_path` 下生成算子的输入输出 bin 文件。

#### bin 转 npy

使用 CANN 提供的 msaccucmp 工具将 dump 出的 bin 文件转换为 npy 格式：

```bash
DUMP_DIR="/path/to/dump/20260701113705/0/model_name/1/0"
OUTPUT_DIR="/path/to/dump/20260701113705"

python ${ASCEND_HOME_PATH}/tools/operator_cmp/compare/msaccucmp.py convert \
    -d ${DUMP_DIR}/Square.Pow_2744_SquareReduceMean_2745.726.44.1782877026275963 \
    -out ${OUTPUT_DIR} \
    -t npy
```

参数说明：
- `-d`：原始 dump 目录（包含 .input.0.bin、.output.0.bin 等文件）
- `-out`：转换后 npy 文件输出目录
- `-t`：输出格式（npy 或 numpy）

### Step 4：对比 Dump 数据（确定根因算子）

分别在 OK 和 NOT_OK CANN 版本下触发 Dump，对比同一算子的输入输出数据：

```python
import numpy as np

ok_input = np.load("dump/ok_9.0/OperatorName.input.0.npy")
ok_output = np.load("dump/ok_9.0/OperatorName.output.0.npy")

not_ok_input = np.load("dump/not_ok_9.1/OperatorName.input.0.npy")
not_ok_output = np.load("dump/not_ok_9.1/OperatorName.output.0.npy")

# 先校验两版本 dtype 一致：dump 精度格式不同（如一侧 fp16 一侧 fp32）会导致
# shape/数值都对不上，必须统一到同一精度格式再比较
for name, a, b in [("input", ok_input, not_ok_input), ("output", ok_output, not_ok_output)]:
    if a.dtype != b.dtype:
        # 统一到 float32 再比较，避免 fp16 vs fp32 的表示差异被误判为精度问题
        a = a.astype(np.float32)
        b = b.astype(np.float32)
    print(f"{name} dtype: ok={ok_input.dtype}, not_ok={not_ok_input.dtype}")

# 对比输入
print(f"Input max diff: {np.abs(ok_input - not_ok_input).max()}")
print(f"Input mean diff: {np.abs(ok_input - not_ok_input).mean()}")

# 对比输出
print(f"Output max diff: {np.abs(ok_output - not_ok_output).max()}")
print(f"Output mean diff: {np.abs(ok_output - not_ok_output).mean()}")
```

**判断依据**：
- 若输入一致、输出不一致 → 该算子为精度问题根因
- 若输入就不一致 → 精度问题源自上游算子，需继续向上追溯

## 执行与验证

### 端到端检查清单

- Profiling 对比已完成，已定位至少一个可疑算子（Block Num、Mix Block Num 或 Format 存在差异）
- `dump.json` 中 `model_name` 与 MindIR 文件名（不含后缀）完全一致
- `dump_config.ini` 路径正确，推理时成功加载
- Dump 输出目录生成了对应算子的 `.bin` 文件
- bin 文件已通过 `msaccucmp.py convert` 成功转换为 `.npy` 文件
- OK 与 NOT_OK 两个版本下同一算子的 npy 数据已加载并完成对比
- 根据输入输出 diff 判断结果已明确精度问题根因算子，或已确定需继续向上追溯

## 标准与约束

- Profiling 对比必须在相同输入条件下进行，确保两个版本的 Profiling 数据具有可比性
- Dump 输入数据应使用真实推理数据，避免使用随机数据导致结果不具代表性
- Dump 目录名包含算子名称、Block 号、Device ID、时间戳等信息，如 `Square.Pow_2744_SquareReduceMean_2745.726.44.1782877026275963`，定位 dump 文件时需注意目录命名规则
- `dump_path` 下会按时间戳和设备 ID 创建子目录，实际路径需根据实际 dump 输出结构调整
- Dump 对比时需使用相同精度格式（如均为 float16 或 float32）进行数值比较
