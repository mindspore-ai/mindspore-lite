---
name: onnx-model-conversion-and-deployment
description: MindSpore Lite云侧推理 Ascend 后端离线转换（ONNX → MindIR）与推理部署全流程。覆盖固定 shape、动态分档、纯动态 shape 的转换策略，以及 MindIR 推理验证与部署注意事项。
---

# MindSpore Lite云侧推理 Ascend 后端离线模型转换 Playbook（converter_lite，optimize=ascend_oriented）

本技能聚焦“MindSpore Lite云侧推理 Ascend 后端离线转换”场景：输入 ONNX 模型，通过 MindSpore Lite `converter_lite`转换工具生成可在 Ascend 执行的 MindIR模型文件（并在离线阶段完成图编译优化）。

## 何时调用

- 用户在拥有onnx模型，需要将onnx模型转换成MindIR模型文件，并确保能在 Ascend 离线转换链路跑通
- 用户执行 `converter_lite --optimize=ascend_oriented` 转换失败，需要定位失败节点与修复策略
- 用户需要将onnx模型通过`converter_lite`转换工具，转换成固定shape，动态分档，纯动态shape模型转换失败，需要定位失败节点与修复策略

## 术语与前提

- 离线转换工具：`converter_lite`（或环境里别名 `Convert`）
- 云侧推理： 通过MindSpore Lite提供的转换工具将onnx模型转换成`.mindir`模型文件，支持 Ascend ACL离线推理
- Ascend 离线优化：`--optimize=ascend_oriented`

## 模型转换流程

### Step 0：构建与环境（必须确认）

检查当前环境是否存在MindSpore Lite二进制转换工具`converter_lite`(可以通过环境变量中LD_LIBRARY_PATH确认，MindSpore Lite转换工具安装路径)，若不存在，需要基于昇腾环境编译MindSpore Lite云侧推理转换工具二进制包，编译流程如下：

```bash
export MSLITE_ENABLE_ACL=on
export MSLITE_ENABLE_CLOUD_INFERENCE=on

bash build.sh -I arm64 -j16
```

编译产物通常在 `output/`目录下的tar包中

### Step 1：离线转换命令模板(ONNX → MindIR)

#### 固定shape场景

```bash
# 固定shape场景(必须配置`--optimize=ascend_oriented`参数)
./converter_lite \
  --fmk=ONNX \
  --modelFile=prefill.onnx \
  --outputFile=prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --inputShape="input_ids:1,128;attention_mask:1,128;position_ids:1,128"
```

#### 动态分档场景

```bash
# 动态分档场景(必须配置`--optimize=ascend_oriented`参数)
./converter_lite \
  --fmk=ONNX \
  --modelFile=prefill.onnx \
  --outputFile=prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=config.ini
```

```txt
# config.ini
# 动态分档场景配置文件
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1"
ge.dynamicDims="128,128,128;256,256,256;512,512,512"
```

#### 纯动态shape场景

当模型包含动态维度（batch/seq_len/image_size），离线构图往往需要 `--inputShape` 约束到“可编译形态”：

```bash
# 纯动态shape场景(必须配置`--optimize=ascend_oriented`参数)
./converter_lite \
  --fmk=ONNX \
  --modelFile=model.onnx \
  --outputFile=out/model \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=config.ini
```

```txt
# config.ini
# 纯动态shape场景配置文件
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1"
```

### 转换验证与约束

- 按照模型转换流程，执行模型转换命令，验证转换结果，保证转换命令可以转换成功，能生成对应的mindir文件(日志打印显示转换转换成功，则转换过程中的ERROR、WARNING日志可以忽略)
- 在使用动态分档时候，档位上为100，当档位过多时候，建议直接使用动态shape场景，避免离线构图失败
- 在模型转换过程中，可以考虑在配置文件中新增plugin_custom_ops=All，以支持通过MindSpore Lite融合pass使能内置融合算子，使能方式如下：
  ```text
  ## 开启plugin_custom_ops
  [ascend_context]
  plugin_custom_ops=All
  ```

## 模型推理部署流程

本节给出“converter_lite 离线转换成功后”的推理部署闭环：如何基于现有环境的推理包、加载 MindIR 并完成一次推理。

### Step 1：准备推理运行环境（必须确认）

1. Ascend 驱动与 CANN

- 确认已安装 Ascend Driver/Firmware 与 Ascend Toolkit（CANN），且 `npu-smi info` 可正常显示设备信息
- 推理进程需要能找到 CANN 运行时库与 OPP：

1. MindSpore Lite 推理包

- Python wheel（适用于 Python 推理验证/集成）

```bash
python -c "import mindspore_lite; print(mindspore_lite.__file__)"
```

### Step 2：准备模型文件结构（MindIR + variables）

离线转换成功后，输出目录通常包含：

- `${model}_graph.mindir`
- `${model}_variables/data_0`

部署时必须保证 `*_graph.mindir` 与同级的 `*_variables/` 目录同时存在，且 `data_0` 完整。

### Step 3：推理加载与执行（Python 示例）

#### 3.1 基础单模型推理（模板）

```python
import numpy as np
import mindspore_lite as mslite

context = mslite.Context()
context.target = ["ascend"]
context.ascend.device_id = 0

model = mslite.Model()
model.build_from_file("model_graph.mindir", mslite.ModelType.MINDIR, context)

inputs = model.get_inputs()
print([(t.name, t.shape, t.dtype) for t in inputs])

# 按模型输入名/shape 构造 numpy 数据（示例）
x0 = np.zeros(inputs[0].shape, dtype=np.int32)
outputs = model.predict([mslite.Tensor(x0)])
print([o.get_data_to_numpy().shape for o in outputs])
```

部署时关注点：

- MindSpore Lite当前不支持int64的输入数据，所以原始onnx模型中的int64的数据类型，在转换mindir会被转换成int32，在推理部署的时候，需要使用int32类型的数据

#### 3.2 LLM prefill + decode 拆分推理（固定 shape 推荐）

适用前提：

- prefill 输入固定 `seq_len`（例如 256）
- decode 固定 step=1，且 KV cache 总长固定（例如 512）
- decode 侧 KV 更新逻辑需要固定 shape（常见做法是 scatter 写入当前位置，避免 cat 导致维度增长）

推理流程（高层）：

1. prefill：输入 `input_ids/attention_mask/position_ids`（固定 256） → 输出 `logits_last` 与 `present_kv`（KV 总长建议直接固定到 512）
2. decode loop：每步输入 `input_ids[1,1] + attention_mask[1,512] + position_id[1,1] + past_kv` → 输出下一 token 的 logits 与新的 `present_kv`

部署时关注点：

- 固定 shape：避免 `-1` 维度进入离线构图/推理
- attention_mask 的长度与 past_kv 的长度必须匹配 `max_total_len`

### Step 4：常见部署问题定位

1. `WARNING: Ascend custom operator path not found`

- 通常是 Ascend 环境变量未配置完整（`LD_LIBRARY_PATH/ASCEND_OPP_PATH/PYTHONPATH`）
- 如果模型包含 `Custom` 节点（CANN 融合算子），必须确保相应 OPP/自定义算子已在当前环境可见

1. build_from_file 很慢/首次编译耗时长

- Ascend 后端可能在首次加载时进行图编译与缓存，属于正常现象；建议在部署启动阶段预热一次

## 模型精度验证

建议以“同一输入的 logits/next_token 一致性”为主线做最小闭环验证，避免在 LLM 场景引入过长序列导致对齐复杂。

### Step A：准备对齐基线

- PyTorch 原模型（或 ONNXRuntime）作为基线
- 使用相同 tokenizer 与相同 prompt
- 固定随机种子（若涉及采样则禁用采样，使用 greedy）

### Step B：验证项（推荐顺序）

1. prefill last token 的 next_token 是否一致（argmax）
2. decode 前 N 步 next_token 是否一致（例如 N=8）
3. 若存在微小差异：记录每步 logits topK，定位差异从第几步开始出现（通常与精度/算子融合有关）

## 执行与验证

### 端到端检查清单

- converter_lite 日志出现 `CONVERT RESULT SUCCESS:0`
- 产物包含 `*_graph.mindir` 与同级 `*_variables/data_0`
- 推理加载 build_from_file 成功（无 RuntimeError）
- 以固定 shape 输入执行一次 predict 成功（无 shape mismatch）
- LLM 场景：prefill + decode 至少生成若干 token（例如 4/8），过程无维度变化与 cache 越界

## 标准与约束

- Ascend 离线转换必须带 `--optimize=ascend_oriented`
- 在onnx模型离线转换过程中，优先使用纯动态；
- 部署脚本/服务禁止打印或泄露用户数据与模型权重路径中的敏感信息

