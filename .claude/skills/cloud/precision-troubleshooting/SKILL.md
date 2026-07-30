---
name: precision-troubleshooting
description: MindSpore Lite MindIR模型精度问题定位技能。覆盖从CANN Profiling对比分析定位可疑算子、配置算子Dump、到最终对比Dump数据确定精度差异根因算子的全流程。
---

# MindSpore Lite 云侧推理模型精度问题定位技能

本技能聚焦"MindSpore Lite ONNX→MindIR 模型转换后精度差异定位"场景：当不同 CANN 版本下推理结果不一致时，通过 Profiling 对比和 CANN 算子 Dump 功能，从算子级别定位导致精度差异的根本原因。

## 何时调用

- 用户在不同 CANN 版本下使用同一 MindIR 模型推理，发现输出结果存在精度差异
- 用户需要从算子级别定位精度差异根因，而非仅判断整体输出是否一致
- 用户已获取两个 CANN 版本的 `mindstudio_profiler_output` 数据，需要对比分析定位可疑算子
- 用户需要配置 CANN 算子 Dump 功能来验证特定算子的输入输出精度

## 术语与前提

- 离线转换工具：`converter_lite`（或环境里别名 `Convert`），将 ONNX 模型转换为 `.mindir` 模型文件
- Profiling 数据：`mindstudio_profiler_output/op_summary_*.csv`，包含算子维度 Block Num、Mix Block Num、Format 等信息
- CANN Dump：CANN 框架提供的算子级数据导出功能，可 dump 指定算子的输入输出 tensor 数据
- 对比基线：OK 版本（精度正确） vs NOT_OK 版本（精度异常），需在相同输入数据下分别推理并对比

## 精度定位流程

### Step 1：Profiling 对比分析（定位可疑算子）

从两个 CANN 版本（OK 版本和 NOT_OK 版本）的 `mindstudio_profiler_output/op_summary_*.csv` 文件中筛选存在差异的算子。

```bash
# 定位 Block Num 差异算子
diff <(grep "Block" ok/mindstudio_profiler_output/op_summary_*.csv | sort) \
     <(grep "Block" not_ok/mindstudio_profiler_output/op_summary_*.csv | sort)
```

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
