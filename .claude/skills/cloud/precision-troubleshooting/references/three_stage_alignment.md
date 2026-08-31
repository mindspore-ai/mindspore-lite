---
name: "three-stage-alignment"
description: "Torch CPU → ONNX → MindIR 三阶段精度对齐完整流程。覆盖统一输入构造、逐层定位、fp32/混合精度配置、Custom 融合算子对接排查、ONNX 添加中间输出定位、大模型降层数加速。"
---

# Torch CPU → ONNX → MindIR 精度对齐

> 本文档是 `precision_troubleshooting` skill 的细化文档之一，对应场景 A（三阶段对齐）。
> 场景 B（跨 CANN 版本精度定位）见 [cann_dump_locating.md](cann_dump_locating.md)；通用原则与场景匹配见 [SKILL.md](../SKILL.md)。

本流程用于确保模型从 PyTorch 导出 ONNX、再经 MindSpore Lite 转换为 MindIR 后，三者推理结果精度一致。

## 前置条件

- 已有 Torch 导出脚本，可导出 ONNX 模型
- 已安装 MindSpore Lite 转换工具 `converter_lite`
- 已安装 onnxruntime
- 已有可用的 Ascend 推理环境

---

## 阶段 1：Torch CPU vs ONNX 精度验证

### Step 1.1：构造统一测试输入

生成一份固定的 numpy 输入数据，三个阶段共用：

```python
import numpy as np

np.random.seed(42)
# 按模型输入构造，此处以示例形状为例
input_data = {
    "input1": np.random.randn(*shape1).astype(np.float32),
    "input2": np.random.randn(*shape2).astype(np.float32),
}
np.savez("test_input.npz", **input_data)
```

### Step 1.2：运行 Torch CPU 推理

```python
import torch
import numpy as np

input_data = dict(np.load("test_input.npz"))
model = torch_model.eval()

with torch.no_grad():
    torch_inputs = {k: torch.from_numpy(v) for k, v in input_data.items()}
    torch_output = model(**torch_inputs)

# 统一转 numpy
if isinstance(torch_output, torch.Tensor):
    torch_output = torch_output.cpu().numpy()
elif isinstance(torch_output, (tuple, list)):
    torch_output = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in torch_output]
```

> **大模型标杆选择**：若模型相对较大，Torch CPU 执行很慢，精度定位时间成本太高，可用 **Torch NPU** 结果作为标杆（与 Torch CPU 等价但快得多）：
> - **Ascend 800I A2**：建议直接用 Torch NPU 作为标杆
> - **Ascend 300I Duo**：优先考虑 Torch CPU；若模型实在太大，300I Duo 也建议用 Torch NPU
>
> 标杆从 Torch CPU 切到 Torch NPU 后，阶段 1 的对比口径同步调整：Torch NPU vs ONNX。

### Step 1.3：运行 ONNX 推理

```python
import onnxruntime as ort
import numpy as np

input_data = dict(np.load("test_input.npz"))
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
onnx_output = session.run(None, input_data)
```

### Step 1.4：精度对比

```python
import numpy as np

def compare_outputs(ref, test, name="output"):
    if isinstance(ref, (list, tuple)):
        for i, (r, t) in enumerate(zip(ref, test)):
            compare_outputs(r, t, f"{name}[{i}]")
        return
    max_abs = np.max(np.abs(ref - test))
    mean_abs = np.mean(np.abs(ref - test))
    cosine = np.dot(ref.flatten(), test.flatten()) / (
        np.linalg.norm(ref.flatten()) * np.linalg.norm(test.flatten()) + 1e-12
    )
    print(f"[{name}] max_abs={max_abs:.2e}, mean_abs={mean_abs:.2e}, cosine={cosine:.6f}")

compare_outputs(torch_output, onnx_output, "torch vs onnx")
```

**判断标准**：
- 通过：max_abs < 1e-3 且 cosine > 0.999 → 进入阶段 2
- 不通过：执行 Step 1.5 逐层定位差异来源（阶段 1 是 Torch vs ONNX，两者都属"标杆"侧，数值超阈值即说明导出/权重/前后处理有问题，无需走端到端兜底）

### Step 1.5：Torch vs ONNX 精度问题定位

当整体输出差异超出阈值时，需逐层定位差异来源：

#### 方法一：PyTorch 逐层 hook 抓取中间输出

```python
import torch

layer_outputs = {}
hooks = []

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            layer_outputs[name] = output.detach().cpu().numpy()
        elif isinstance(output, (tuple, list)):
            layer_outputs[name] = [
                o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o
                for o in output
            ]
    return hook

# 注册 hook（按模型结构选择关键层）
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.GELU)):
        hooks.append(module.register_forward_hook(make_hook(name)))

with torch.no_grad():
    model(**torch_inputs)

for h in hooks:
    h.remove()
```

#### 方法二：导出中间层 ONNX 做分段对比

将模型按模块拆分导出多个 ONNX，逐段对比（下方为示意，`...` 处需填入实际输入与 export 参数，非可直接运行的代码）：

```python
# 示意：导出前半部分和后半部分
class Part1(torch.nn.Module):
    def forward(self, x):
        return model.part1(x)

class Part2(torch.nn.Module):
    def forward(self, x):
        return model.part2(x)

torch.onnx.export(Part1(), dummy_input, "model_part1.onnx", input_names=["x"], output_names=["y"])
torch.onnx.export(Part2(), dummy_input, "model_part2.onnx", input_names=["x"], output_names=["y"])
```

对每个分段分别在 Torch 和 ORT 上推理，缩小差异范围。

#### 常见 Torch → ONNX 精度问题

| 问题 | 原因 | 处理方式 |
|------|------|---------|
| 模型部分权重为随机数 | 模型未完整加载权重（如缺少 checkpoint、key 不匹配被跳过），导致部分层使用随机初始化值，Torch 与 ONNX 推理时随机值不一致 | 导出前检查 `model.load_state_dict()` 是否有 `missing_keys` 或 `unexpected_keys`；用 `torch.load()` 后逐层比对权重与 checkpoint 是否一致 |
| Torch 与 ONNX Runtime 输入前处理/输出后处理逻辑不一致 | 两套推理脚本分别实现了 tokenize、padding、attention_mask 构造、logits→scores 等逻辑，实现细节差异（如 padding 侧、mask 取反、归一化顺序、是否去最后一个 token）导致送入模型的输入本身就不同，或对输出的解释方式不同 | 两套脚本共用同一份前处理/后处理实现；或用同一份 numpy 输入（`test_input.npz`）喂给两套模型，并在 raw logits 层面对比，绕开前后处理差异 |

---

## 阶段 2：ONNX vs MindIR 精度验证

### Step 2.0：确认 ONNX 模型不含 Custom 融合算子（前置约束）

进入本阶段前，必须确认用于 ONNX vs MindIR 对比的 ONNX 模型**不含 Custom 自定义融合算子**：

- 带 Custom 算子的 ONNX **无法用 ONNX Runtime 推理**（ONNX Runtime 不认识 Custom 节点的语义），阶段 1 的 ONNX 基线对比就失效了
- 通常为优化性能会修改 torch 导出脚本，使能导出带融合算子（如 PromptFlashAttention、RmsNorm、SwiGlu 等）的 ONNX，再转换成带对应 CANN 融合算子的 MindIR。这类带 Custom 的 ONNX 不能直接用作精度基线

**两套 ONNX 的分工**：

| ONNX 类型 | 用途 | 能否跑 ONNX Runtime |
|-----------|------|---------------------|
| non-fuse ONNX（未适配融合算子） | 精度基线，阶段 1/2 的 ONNX 侧对比 | 能 |
| fused ONNX（带 Custom 融合算子） | 转换成带 CANN 融合算子的 MindIR，用于性能/部署 | 不能 |

> 精度对齐流程中，**ONNX 侧基线始终用 non-fuse ONNX**；fused ONNX 仅用于转换出性能版 MindIR。若只有 fused ONNX，需先导出一份 non-fuse 版本作为基线。

#### 带 Custom 融合算子的 MindIR 精度问题定位

**前提：必须先确认 non-fuse MindIR 精度 OK，才能把 fused MindIR 的精度问题归因到 Custom 算子。**

定位顺序：

1. **先转 non-fuse MindIR**：用 non-fuse ONNX 转换出不含 Custom 的 MindIR，完成阶段 2 的精度验证，确认其与 ONNX 基线对齐
2. **再转 fused MindIR**：用 fused ONNX（带 Custom）转换出含 CANN 融合算子的 MindIR，与 ONNX 基线对比
3. **归因判断**：
   - non-fuse MindIR 精度 OK，fused MindIR 精度不 OK → 问题由 Custom 融合算子对接引入，进入下方排查
   - non-fuse MindIR 精度就不 OK → 问题在算子本身或转换链路，按 Step 2.4 走常规定位流程，不应归因到 Custom

只有当前提成立时，才优先分析 Custom 融合算子对接是否正确。排查方向：

- **算子属性**：Custom 节点的属性（如 `num_heads`、`input_layout`、`scale` 等）是否与实际计算语义一致
- **输入数据**：传给 Custom 算子的输入是否正确（如 Q/K/V 的 head 维顺序、是否多了/少了 repeat/tile）
- **输入输出 dtype**：Custom 算子的输入输出 dtype 是否与预期一致（如 fp16/fp32 选择、是否有隐式 Cast）

验证方法——**构造融合算子单模型**：

1. 把要验证的 Custom 融合算子单独导出成一个只含该算子的小 ONNX（或直接构造单算子输入）
2. 转换成单算子 MindIR 并推理
3. 标杆用**融合算子适配前的等价小算子部分**（即 non-fuse 形态下完成同样计算的那个子图）在 ONNX Runtime 上的输出
4. 对比两者，定位该融合算子对接是否存在精度问题

> 这种隔离验证能快速区分"是 Custom 对接错了"还是"其他算子本身的精度问题"，避免在整网 diff 里反复兜圈子。

### Step 2.1：转换 ONNX → MindIR

先以默认配置转换：

```bash
./converter_lite \
  --fmk=ONNX \
  --modelFile=model.onnx \
  --outputFile=model \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

### Step 2.2：运行 MindIR 推理

```python
import numpy as np
import mindspore_lite as mslite

input_data = dict(np.load("test_input.npz"))

context = mslite.Context()
context.target = ["ascend"]
context.ascend.device_id = 0

model = mslite.Model()
model.build_from_file("model_graph.mindir", mslite.ModelType.MINDIR, context)

inputs = model.get_inputs()
ms_inputs = []
for t in inputs:
    data = input_data.get(t.name.rstrip("\x00"))
    if data is None:
        raise KeyError(f"找不到输入 {t.name!r} 对应的测试数据，请确认 test_input.npz 的 key 与模型输入名一致；避免喂零导致精度对比失真")
    ms_inputs.append(mslite.Tensor(data))

outputs = model.predict(ms_inputs)
mslite_output = [o.get_data_to_numpy() for o in outputs]
```

### Step 2.3：精度对比

```python
compare_outputs(onnx_output, mslite_output, "onnx vs mindir")
```

**判断标准**：
- 通过：max_abs < 1e-3 且 cosine > 0.999 → 三者精度对齐，流程结束
- 不通过：数值超阈值时，若场景为图/视频生成、LLM 等，优先将 MindIR 与 Torch CPU 的端到端结果（生成的图片/视频/文本输出）提供给用户判定；确有效果问题才执行 Step 2.4

### Step 2.4：ONNX vs MindIR 精度问题修复

#### 2.4.1 尝试 fp32 强制模式

在转换配置中强制 fp32 精度：

```ini
# config.ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

```bash
./converter_lite \
  --fmk=ONNX \
  --modelFile=model.onnx \
  --outputFile=model_fp32 \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=config.ini
```

用 fp32 MindIR 重新推理并对比：

```python
compare_outputs(onnx_output, mslite_fp32_output, "onnx vs mindir_fp32")
```

**判断**：
- 通过：fp32 下精度满足 → 进入 2.4.3 尝试混合精度优化性能
- 不通过：fp32 下仍有精度问题 → Ascend 800I A2 上先试 2.4.2（饱和模式 / bf16 混合精度），仍不通过再进 2.4.4 算子级定位；非 800I A2 直接进 2.4.4

#### 2.4.2 Ascend 800I A2 上的额外精度手段（force_fp32 不通过时）

在 Ascend 800I A2 硬件上，若 `force_fp32` 仍存在精度问题，可尝试以下两手段：

**手段一：使能饱和模式**

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE=SATURATION_MODE
```

设置后重新转换/推理，看精度是否改善。饱和模式针对溢出类精度问题（如累加/指数运算上溢）。

**手段二：bfloat16 混合精度（模型含 bfloat16 计算时）**

若模型本身用 bfloat16 计算，用 bf16 与 float32 混用的混合精度配置：

```ini
# config.ini
[acl_init_options]
ge.exec.precision_mode=allow_mix_precision_bf16
```

> bf16 与 fp16 是不同的低比特路径：fp16 精度损失可能来自表示范围/累加误差，bf16 牺牲尾数换更大表示范围。若模型原生 bf16，优先用此配置而非 fp16 黑名单路径。

#### 2.4.3 混合精度配置（fp32 通过后的性能优化）

fp32 模式虽然精度正确，但性能较差。通过混合精度配置，仅让关键算子走 fp32，其余走 fp16，兼顾精度与性能。

##### (1) 采集 Profiling 获取模型算子列表

Profiling 采集与解析方法见 [get_profiling_data](../../open-source-model-migration/references/get_profiling_data.md)。

采集后，profiling 结果目录结构通常为：

```
./profiling/PROF_000001_20260606152013489_CBQGBMIOHQJAMIFC/mindstudio_profiler_output/
├── op_statistic_0.json
├── op_summary_0.csv
└── ...
```

从 profiling 结果中获取模型执行的算子：

```bash
# 提取算子类型列表
python3 -c "
import json, glob
files = glob.glob('./profiling/PROF_*/mindstudio_profiler_output/op_statistic_*.json')
if not files:
    raise FileNotFoundError('未找到 op_statistic_*.json，请确认 profiling 目录结构')
with open(files[0]) as f:
    data = json.load(f)
op_types = set()
for op in data:
    op_types.add(op.get('Op Type', op.get('op_type', '')))
for t in sorted(op_types):
    print(t)
"
```

> 注意：profiling 输出文件名可能是 `op_statistic_*.json` 或 `op_summary_*.csv`，视 CANN 版本而定。

##### (2) 优先将计算密集型算子加入 fp32 黑名单

将计算相关的算子优先加入 fp32 黑名单（即强制走 fp32），非计算类算子（如 Reshape、Transpose、Concat 等）无需加入：

```json
{
  "black-list": {
    "to-add": ["Square", "Add", "Div", "Exp", "Erf", "Softmax", "ReduceMean"]
  }
}
```

> 优先级：涉及累加/归约/指数/幂运算的算子最容易出现 fp16 精度损失，应优先加入黑名单。

##### (3) 生成混合精度配置并转换

```ini
# config.ini
[acl_build_options]
ge.exec.precision_mode=allow_mix_precision_fp16
ge.exec.modify_mixlist=./op_fp32.json
```

```json
// op_fp32.json —— 建议从少量最可疑算子开始（如 Square、Add、Div），验证后再逐步添加
{
  "black-list": {
    "to-add": ["Square", "Add", "Div"]
  }
}
```

```bash
./converter_lite \
  --fmk=ONNX \
  --modelFile=model.onnx \
  --outputFile=model_mix \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=config.ini
```

##### (4) 验证混合精度推理结果

```python
compare_outputs(onnx_output, mslite_mix_output, "onnx vs mindir_mix")
```

若精度不满足，继续向黑名单中添加可疑算子，重复 (3)(4)。

##### (5) 注意事项

- **昇腾硬件限制**：部分算子不支持 fp32 实现，若转换或推理报错，需将该算子从黑名单移除，让其走 fp16，保证功能正常
- 逐步添加算子到黑名单，避免一次性添加过多导致难以定位哪个算子是关键影响项
- 混合精度配置与 fp32 配置的误差可能略有差异，需重新验证

#### 2.4.4 fp32 仍有精度问题时的定位方法

当 force_fp32 模式下 ONNX 与 MindIR 仍存在精度差异时，通过 ONNX 模型添加中间输出的方式逐算子定位根因算子。

##### (1) 采集 Profiling 获取算子列表

Profiling 采集与解析方法见 [get_profiling_data](../../open-source-model-migration/references/get_profiling_data.md)。

##### (2) ONNX 模型添加中间输出对比

在 ONNX 模型的关键模块位置将中间结果作为模型输出，然后对该 ONNX 执行正常的模型转换和推理，直接对比 MindIR 推理得到的对应输出与 ONNX Runtime 的输出。

```python
import onnx

model = onnx.load("model.onnx")
# 将目标中间节点添加为模型输出
for node in model.graph.node:
    if node.name == "TargetNodeName":
        for output in node.output:
            value_info = onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(value_info)
onnx.save(model, "model_with_mid_outputs.onnx")
```

转换后的 MindIR 推理输出会自动转为 ND 格式，与 ONNX Runtime 输出格式一致，无需处理格式转换。

##### (3) 大模型转换耗时过长的处理

若模型很大或 Convert 转换一次耗时很长（半小时以上），每次添加中间输出后重新转换会严重影响定位效率。此时可**降低模型层数**来加速定位：

- 在导出脚本中增加层数配置参数，导出只含少量层（如 2-4 层 decoder layer）的精简模型
- 对精简模型反复执行"添加中间输出 → 转换 → 推理对比"的循环，快速锁定精度问题出现的层
- 定位到问题层后再在完整模型上验证

##### 常见 ONNX → MindIR 精度问题

| 问题 | 原因 | 处理方式 |
|------|------|---------|
| fp16 累加误差 | Square、Add等在 fp16 下精度不足 | 将相关算子加入 fp32 黑名单 |

---

## 完整精度对齐流程图

```
Torch CPU 推理 ──┐
                 ├─→ 对比 ──通过──→ 进入阶段2
ONNX 推理 ──────┘            │
                             └─不通过→ 逐层定位 Torch vs ONNX 差异

[阶段2前置] 确认 ONNX 基线为 non-fuse 版本（不含 Custom）
                              │
              ┌─── non-fuse MindIR vs ONNX ──不通过→ 按 Step 2.4 常规定位（fp32/混合精度/中间输出）
              │                                          │
              │                                    （问题不在 Custom，勿误归因）
              │
              └─── fused MindIR vs ONNX ──────不通过→ 前提：non-fuse MindIR 已 OK
                                                         │
                                                         ▼
                                              归因到 Custom 对接 → 验证属性/输入/dtype
                                              构造单算子模型 vs 等价小算子子图隔离验证
```

## 执行检查清单

1. 统一测试输入已保存（`test_input.npz`），三阶段共用
2. Torch CPU vs ONNX 精度验证完成
3. 若 Torch vs ONNX 有差异，已逐层定位并修复
4. 阶段 2 前已确认用于基线的 ONNX 不含 Custom 融合算子（non-fuse 版本）
5. 已先验证 non-fuse MindIR 精度 OK（作为归因 Custom 的前提）
6. 若 fused MindIR 有精度问题，已优先排查 Custom 融合算子对接（属性/输入/dtype），必要时构造单算子模型验证
7. ONNX vs MindIR 精度验证完成
8. 若有差异，已尝试 fp32 强制模式
9. 若 fp32 通过，已配置混合精度（`op_fp32.json`）并验证
10. 若 fp32 仍有差异，已通过 ONNX 添加中间输出定位根因算子（大模型可用降层数加速）
11. 最终三者精度满足阈值（max_abs < 1e-3 / cosine > 0.999），或数值略超阈值但用户已确认 MindIR vs Torch CPU 端到端效果可接受
