# Torch PTQ Int8 → 量化 ONNX 导出模式

> 本文档是 `performance-optimization` skill 的细化策略文档之一。
> 适用场景：对 PyTorch 模型做静态 PTQ int8 量化后，导出为含量化自定义算子的 ONNX 图，用于 Ascend 等硬件后端的 int8 低比特推理（减少显存/带宽）。

本模式描述了如何对 PyTorch 模型做静态 PTQ int8 量化后，导出为含量化自定义算子的 ONNX 图。适用于需要在 Ascend 等硬件后端上以 int8 低比特推理的场景。

## 核心思路

```
正常 FP32 导出 + 量化层替代 + 基于 torch.autograd.Function 的 symbolic 图构建
```

关键洞察：**在 forward 中保持 FP32 计算以保证 tracing 正确；在 symbolic 中用 Custom 算子描述量化计算图。** 这样既不影响 PyTorch tracing 的数值对标，又能让导出的 ONNX 图中出现硬件后端所需的量化算子节点。

## 适用场景判断

当用户满足以下条件时，应使用此模式：

| 条件 | 说明 |
|---|---|
| 模型包含 Linear / MatMul 算子 | 量化针对 `nn.Linear` 及 `F.linear` 调用（CNN/MLP/Transformer/RNN 等任何含线性层的架构均可） |
| 目标后端支持 int8 Custom 算子 | 如 Ascend 的 `AscendQuant` / `QuantBatchMatmul` |
| 需要低比特推理减少显存/带宽 | int8 相比 fp16 可减半带宽 |
| 有校准数据或可合成 | 静态量化需要收集激活分布 |

## 模式结构总览

```
Step 1: 识别可量化的线性层 ── 找出需要替换的 Linear 及其融合模式
Step 2: 挂载 Observer ── 注册激活值收集器
Step 3: 校准 ── 用代表性数据前向传播收集激活范围
Step 4: 计算量化参数 ── 激活 scale + weight int8 量化（可选 SmoothQuant）
Step 5: 构造量化替代层 ── torch.autograd.Function（forward=FP32, symbolic=Custom算子）
Step 6: 条件替换导出 ── 只在量化分支用替代层，FP32 分支保持原样
```

---

## Step 1：识别可量化的线性层及其融合模式

检查模型中所有 `nn.Linear`，按以下模式分类：

**模式 A：独立 Linear**（最常见）
```
hidden → Linear(d_in, d_out) → output
```
每个 Linear 独立校准、独立量化。

**模式 B：Weight 拼接的融合 Linear**
```
QKV 模式：   weight = cat([q_proj, k_proj, v_proj], dim=0)
Gate/Up 模式：weight = cat([gate_proj, up_proj], dim=0)
```
多个子 Linear 的 weight 在输出维度拼接为一个大 weight，一次计算后按 chunk 拆分。优势是减少矩阵乘调用次数，量化时也作为一个整体处理。

**模式 C：输入拼接的融合 Linear**
```
weight = cat([expert1, expert2, ...], dim=1)
```
多个子 Linear 的 weight 在输入维度拼接，一次计算后按 chunk 拆分。

识别后，为每种模式设计一套 Observer 挂载和量化参数计算方案。

## Step 2：挂载 Observer

### 标准做法

对每个待量化的 Linear 挂载 PyTorch Observer：

```python
from torch.ao.quantization.observer import MinMaxObserver

obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
module._ptq_act_obs = obs
handle = module.register_forward_pre_hook(
    lambda mod, inputs: mod._ptq_act_obs(inputs[0].detach())
)
```

### 融合 Linear 的特殊处理

对于 Step 1 中识别出的融合 Linear（如 QKV、Gate/Up），它们不在 `nn.Module` 列表中，需要在父模块（如 attention、mlp）上挂载 Observer：

```python
# 父模块上挂载
parent_module._ptq_merged_act_obs = obs
```

### 同时收集 per-channel 激活最大值（用于 SmoothQuant）

在每个 pre-hook 中额外记录 per-channel 最大值：

```python
flat_x = x.detach().reshape(-1, x.shape[-1])
per_ch = flat_x.abs().max(dim=0).values
# 累积取 max
if not hasattr(mod, "_ptq_act_per_ch_max") or mod._ptq_act_per_ch_max is None:
    mod._ptq_act_per_ch_max = per_ch
else:
    mod._ptq_act_per_ch_max = torch.maximum(mod._ptq_act_per_ch_max, per_ch)
```

## Step 3：校准（Calibration）

用代表性数据运行模型前向传播，收集激活分布。

### 自回归模型的校准模式

对于 LLM 等自回归模型，校准需模拟推理过程：

```
for each sample:
    1. Prefill: 将 prompt 一次性输入，收集 prefill 阶段激活
    2. Decode: 逐 token 自回归生成，收集 decode 阶段激活
       循环 max_decode_steps 步
```

### 非自回归模型

直接运行前向传播即可，无需分段。

### 校准数据要求

- 覆盖模型在部署时可能遇到的输入范围
- 至少几十到几百条样本
- 可从实际数据采集，也可合成（确保数值范围合理）

### 校准数据收集

校准数据需在目标推理环境中（MindSpore Lite / TensorRT / OpenVINO / ONNX Runtime 等）提前采集，导出为标记好的**输入-输出对**，供 PTQ 量化脚本在 PyTorch 中模拟运行。核心原则：**"在哪推理，就在哪采集"**——校准数据应反映真实部署场景的输入分布。

#### 通用采集模式

根据模型类型选择相应模式：

**模式 A：自回归模型（LLM）**

```text
对于每条样本：
  1. 准备输入文本并 tokenize
  2. Prefill: 将完整 token 序列输入模型，得到 logits 和 KV cache
  3. Decode: 逐 token 自回归生成（argmax / 采样），串行更新 KV cache
  4. 导出：prefill 输入数据 + decode 生成的 token 序列
```

**模式 B：非自回归模型（分类/视觉/非流式模型）**

```text
对于每条样本：
  1. 准备输入数据（文本 / 图像 / 特征）
  2. 单次前向传播，得到输出
  3. 导出：输入数据 + 输出结果（或 ground-truth 标签）
```

**模式 C：有状态模型（RNN/LSTM/流式模型）**

```text
对于每条样本：
  1. 准备输入序列
  2. 按时间步展开前向传播，维护隐状态
  3. 导出：输入的完整序列 + 每步的输出 / 隐状态
```

#### 推荐的通用 JSONL 格式

PTQ 量化脚本需要知道每条样本的**输入是什么**以及**模型应该如何响应**，建议采用以下通用格式：

| 字段 | 类型 | 必要性 | 说明 |
|---|---|---|---|
| `input.{name}` | 任意 | 必填 | 模型输入张量，按名称组织（如 `input.input_ids`, `input.attention_mask`） |
| `output.{name}` | 任意 | 可选 | 期望输出或用于对齐的参考值 |
| `meta.prompt` | str | 可选 | 原始输入文本（便于溯源） |
| `meta.sequence` | list | 自回归必填 | 模型自回归生成的 token ID 列表（不含 prefill 输入） |
| `meta.kv_cache_len` | int | 自回归可选 | 固定 KV cache 长度 |
| `meta.{custom}` | 任意 | 可选 | 自定义元信息（timestamp、标签、id 等） |

**自回归模型推荐格式**：

```json
{"meta": {"prompt": "输入文本", "sequence": [101, 202, 303], "kv_cache_len": 512},
 "input": {
   "input_ids": [[1, 123, 456, 0, 0]],
   "attention_mask": [[1, 1, 1, 0, 0]],
   "position_ids": [[0, 1, 2, 0, 0]]
 }}
```

**非自回归模型推荐格式**：

```json
{"meta": {"label": 3, "source": "val_set"},
 "input": {
   "pixel_values": [[[...]]],
   "attention_mask": [[1, 1, 1, 0, 0]]
 },
 "output": {
   "logits": [[0.1, 0.8, 0.05, 0.05]]
 }}
```

#### 数据采集的代码集成模式

在推理脚本中添加 `--dump-calib` 选项的通用实现模板：

```python
def dump_calib_record(args, input_data, output_data, generated_ids=None):
    """将一条样本的校准数据追加写入 JSONL。"""
    record = {"input": {}, "meta": {}}

    # 收集输入张量（转为 Python 原生类型以便 JSON 序列化）
    for name, tensor in input_data.items():
        record["input"][name] = tensor.astype(np.int64).tolist()

    # 收集自回归序列（如有）
    if generated_ids is not None:
        record["meta"]["sequence"] = [int(x) for x in generated_ids]

    # 收集元信息
    record["meta"]["timestamp"] = int(time.time())

    # 追加写入
    with open(args.dump_calib, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

CLI 参数定义：

```python
parser.add_argument(
    "--dump-calib", type=str, default="",
    help="Path to append one JSONL calibration record "
         "(input tensors + generated sequence).",
)
```

#### 典型使用方式

```bash
# 逐条采集多条样本，追加到同一 JSONL 文件
python run_inference.py \
    --model ./model.mindir \
    --prompt "sample prompt 1" \
    --max-new-tokens 64 \
    --dump-calib ./calib.jsonl

python run_inference.py \
    --model ./model.mindir \
    --prompt "sample prompt 2" \
    --max-new-tokens 64 \
    --dump-calib ./calib.jsonl

# 或通过脚本批量采集
python batch_collect_calib.py \
    --model ./model.mindir \
    --input-file ./prompts.txt \
    --dump-calib ./calib.jsonl
```

#### 采集完成后：校准数据的使用

PTQ 量化脚本通过读取 JSONL，在 PyTorch 中**回放**推理过程来收集激活分布：

```
JSONL 记录
  │
  ├─ input.input_ids ──────────→ PyTorch 模型 forward（prefill）
  ├─ input.attention_mask ─────→ 得到 KV cache + logits
  ├─ meta.sequence (token IDs) ─→ 逐 token decode（max_decode_steps 步）
  │                               每步运行 decode forward，收集激活
  │
  └─ 所有激活分布 → 计算量化参数
```

注意：量化脚本**不使用** JSONL 中的 logits 或推理引擎的输出，而是用 PyTorch 重新跑模型。所以 JSONL 只需保存**输入张量**和**生成的 token 序列**，不需要保存 logits 或中间激活。

#### 关键设计要点

1. **输入分布覆盖**：校准数据集应覆盖部署场景的各种输入类型、长度、内容。自回归模型需尤其关注 prompt 长度的多样性（短 prompt 和长 prompt 的激活分布可能不同）
2. **生成步数控制**：`--max-new-tokens` 控制每条样本的 decode 步数，步数越多校准越充分，但耗时线性增长。推荐 32~128 步
3. **输入-量化参数一致性**：采集时使用的 KV cache 长度、padding 策略等需与量化导出脚本的参数一致
4. **数据流解耦**：数据采集（目标推理引擎）与量化导出（PyTorch）是**独立的两个阶段**。这使得你可以在实际部署硬件上采集数据，在开发机上做量化，无需在部署环境安装 PyTorch
5. **仅需输入和 token 序列**：量化回放时只依赖 `input.*` 张量和生成的 token IDs，不需要推理引擎的输出 logits，降低采集端的复杂度

## Step 4：计算量化参数

校准完成后，对每个量化目标计算：

### 4a：激活 Scale 计算

```python
maxabs = torch.maximum(obs.max_val.abs(), obs.min_val.abs()).clamp_min(1e-8)
x_scale = maxabs / 127.0    # 对称量化，int8 范围 [-127, 127]
# 保存为浮点数
module._ptq_x_scale = float(x_scale.cpu().item())
```

### 4b：可选 SmoothQuant（仅当有 per-channel 激活最大值时）

原理：将激活的量化难度"转移"一部分到 weight 侧，降低激活离群值的影响。

```python
# 1. 计算平滑因子 per input channel
max_act_per_ch = module._ptq_act_per_ch_max    # [in_features]
max_w_per_col = weight.abs().max(dim=0).values  # [in_features]
s = max_act_per_ch ** alpha / max_w_per_col ** (1.0 - alpha)
s = s.clamp(1e-4, 1e4)

# 2. 对 weight 应用平滑
weight_smoothed = weight * s.unsqueeze(0)

# 3. 重新计算激活 scale（平滑后的激活最大值变小了）
max_act_smoothed = (max_act_per_ch / s).max().clamp_min(1e-8)
x_scale = max_act_smoothed / 127.0

# 4. 保存平滑因子供导出时使用
module._ptq_smooth_scale = s  # ONNX 中作为 Div 的除数
```

**alpha 选择指南**：
- alpha=0.5：均匀分担量化难度
- alpha>0.5：更多分担到 weight 侧（weight 精度损失容忍度更高时）
- alpha<0.5：更多分担到激活侧

### 4c：Weight 量化（Per-channel 对称 int8）

```python
def quantize_weight_symmetric_int8(weight, clip_ratio=0.0):
    w = weight.detach()
    
    # 可选：裁剪离群值
    if clip_ratio > 0 and w.numel() > 100:
        flat = w.abs().view(-1)
        k = max(1, int(flat.numel() * clip_ratio))
        threshold, _ = flat.topk(k)
        w = w.clamp(-threshold[-1], threshold[-1])
    
    # Per-channel int8：每行一个 scale
    maxabs = w.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
    w_scale = maxabs / 127.0                    # [out_features, 1]
    w_q = torch.clamp(torch.round(w / w_scale), -127, 127).to(torch.int8)
    return w_q, w_scale.view(-1)                # w_q: int8, w_scale: [out_features]
```

### 4d：存储量化参数

将所有量化参数以模块属性保存（名称可自定义，但模式统一）：

| 属性 | 用途 | 适用 |
|---|---|---|
| `_ptq_w_q` | int8 量化后的 weight | 所有量化层 |
| `_ptq_w_scale` | weight 的 scale 因子（per-channel） | 所有量化层 |
| `_ptq_x_scale` | 激活的 scale 因子（标量） | 所有量化层 |
| `_ptq_smooth_scale` | SmoothQuant 平滑因子（可选） | 启用 SmoothQuant 时 |
| `_ptq_act_per_ch_max` | per-channel 激活最大值（校准用，用完清除） | 校准阶段 |
| `_ptq_act_obs` | Observer 实例（用完清除） | 校准阶段 |

#### 融合 Linear 的参数存储

融合 Linear 的量化参数存储在父模块上，命名加前缀区分：

```python
# 例如 attention 模块上的 QKV 参数
attention._ptq_qkv_w_q = ...
attention._ptq_qkv_w_scale = ...
attention._ptq_qkv_x_scale = ...

# 例如 mlp 模块上的 Gate/Up 参数
mlp._ptq_gate_up_w_q = ...
mlp._ptq_gate_up_w_scale = ...
```

## Step 5：构造量化替代层

核心思想：用 `torch.autograd.Function` 实现"计算-导出二象性"——
- **forward**：跑 FP32 `F.linear`，保证 PyTorch tracing 正确、数值可对标
- **symbolic**：构建 ONNX 图，将 FP32 linear 替换为 int8 量化算子子图

### 辅助函数：ONNX Cast 类型编码

```python
def _onnx_cast_to_i_from_dtype(dtype: torch.dtype) -> int:
    """将 torch dtype 映射为 ONNX Cast 的 to_i 属性值。"""
    if dtype == torch.float16:
        return 10    # ONNX enum: FLOAT16
    if dtype == torch.float32:
        return 1     # ONNX enum: FLOAT
    if dtype == torch.bfloat16:
        return 16    # ONNX enum: BFLOAT16
    raise RuntimeError(f"Unsupported dtype for ONNX Cast: {dtype}")
```

### 核心类：`_QuantLinearSymInt8`

这是整个量化导出的核心。forward 保持 FP32，symbolic 构建以下 ONNX 子图：

```
输入 x (fp32)
  │
  ├─ [可选] Div(x, smooth_scale)     ← SmoothQuant：输入侧除以平滑因子
  │
  ├─ Custom("AscendQuant")           ← 激活从 fp32 量化为 int8
  │     src_t_i=1 (float32)
  │     dst_t_i=3 (int8)
  │     scale_f = 1 / x_scale
  │     offset_f = 0.0
  │
  ├─ Custom("QuantBatchMatmul")      ← int8 × int8 矩阵乘
  │     输入: x_i8 (int8), w_q (int8), combined_scale (uint64)
  │     可选输入: offset, bias, pertoken_scale (均不传入)
  │     属性:
  │       transpose_x1_s="false"
  │       transpose_x2_s="true"          ← weight 已在导出侧转置
  │       dtype_i=1 (float32 中间精度)    ← 用于 per-channel Mul 前的反量化
  │
  ├─ [可选] Mul(correction)          ← Per-channel 反量化补偿
  │     correction = w_scale / w_scale_mean  (float32 常量)
  │
  ├─ Add(bias)                       ← 加偏置
  │
  └─ Cast(to_i=target_dtype)         ← 转回输入 dtype (fp16/bf16/fp32)
```

完整实现：

```python
class _QuantLinearSymInt8(torch.autograd.Function):
    """量化线性层：forward 跑 FP32，symbolic 构建 Ascend 量化 ONNX 子图。"""

    @staticmethod
    def forward(ctx, x, weight_fp, bias_fp,
                x_scale_f: float, w_q, correction,
                w_scale_mean: float, smooth_scale, out_to_i: int):
        # forward 始终保持 FP32 计算，保证 tracing 正确性
        return F.linear(x, weight_fp, bias_fp)

    @staticmethod
    def symbolic(g, x, weight_fp, bias_fp,
                 x_scale_f: float, w_q, correction,
                 w_scale_mean: float, smooth_scale, out_to_i: int):
        import struct

        x_scale_f = float(x_scale_f)
        w_scale_mean = float(w_scale_mean)
        per_channel = correction is not None

        # ── SmoothQuant：输入侧除以平滑因子 ──
        if smooth_scale is not None:
            x = g.op("Div", x, smooth_scale)

        # ── AscendQuant：激活 int8 量化 ──
        ascend_scale = 1.0 / max(x_scale_f, 1e-8)
        x_i8 = g.op(
            "Custom", x,
            type_s="AscendQuant",
            input_names_s=["x"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0],
            src_t_i=1,      # kNumberTypeFloat32
            dst_t_i=3,      # kNumberTypeInt8
            scale_f=float(ascend_scale),
            offset_f=0.0,
        )

        # ── 打包 combined_scale = x_scale × w_scale_mean ──
        combined_scale = x_scale_f * w_scale_mean
        scale_bits = struct.unpack("<I", struct.pack("<f", combined_scale))[0]
        packed_scale = scale_bits
        scale_tensor = torch.tensor([packed_scale], dtype=torch.int64)
        scale_const = g.op("Constant", value_t=scale_tensor)

        # ── QuantBatchMatmul：int8 × int8 ──
        y = g.op(
            "Custom", x_i8, w_q, scale_const,
            type_s="QuantBatchMatmul",
            input_names_s=["x1", "x2", "scale", "offset", "bias", "pertoken_scale"],
            optional_input_names_s=["offset", "bias", "pertoken_scale"],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            transpose_x1_s="false",
            transpose_x2_s="true",
            dtype_i=1,  # float32 中间精度，为 per-channel Mul 做准备
        )

        # ── Per-channel 反量化补偿 ──
        if per_channel:
            y = g.op("Mul", y, correction)

        # ── Bias ──
        y = g.op("Add", y, bias_fp)

        # ── Cast 回目标 dtype ──
        y = g.op("Cast", y, to_i=int(out_to_i))
        y.setType(x.type())
        return y
```

### 包装函数：`quant_linear_symmetric_int8`

该函数负责将量化参数预处理（计算 per-channel correction、打包 scale），然后调用 `_QuantLinearSymInt8.apply`：

```python
def quant_linear_symmetric_int8(x, weight_fp, bias_fp,
                                 x_scale, w_q, w_scale,
                                 smooth_scale=None):
    """
    量化线性层包装函数。

    Args:
        x: 输入张量 (fp32/fp16/bf16)
        weight_fp: 原始 FP weight（forward 阶段使用）
        bias_fp: 偏置（None 时需传零张量）
        x_scale: 激活 scale（标量）
        w_q: int8 量化后的 weight
        w_scale: weight scale（[out_features] per-channel 或标量）
        smooth_scale: 可选 SmoothQuant 平滑因子 [in_features]

    Returns:
        输出张量（FP32 计算，ONNX 中为量化路径）
    """
    out_to_i = int(_onnx_cast_to_i_from_dtype(x.dtype))
    x_scale_f = float(x_scale)

    if smooth_scale is not None:
        smooth_scale = smooth_scale.to(x.dtype)  # 与 export dtype 一致

    # 预处理 per-channel correction
    if isinstance(w_scale, torch.Tensor) and w_scale.dim() == 1:
        w_scale_np = w_scale.detach().cpu().numpy()
        w_scale_mean = float(w_scale_np.mean())
        correction = (w_scale_np / w_scale_mean).astype(np.float32)
        correction = torch.from_numpy(correction)
    else:
        w_scale_mean = float(w_scale) if isinstance(w_scale, torch.Tensor) else float(w_scale)
        correction = None

    return _QuantLinearSymInt8.apply(
        x, weight_fp, bias_fp, x_scale_f, w_q,
        correction, w_scale_mean, smooth_scale, out_to_i,
    )
```

**scale 传递技巧说明**：
- `QuantBatchMatmul` 只接受一个标量 scale 参数，因此需要将 `x_scale`（激活 scale）和 `w_scale_mean`（weight scale 均值）**合并打包**：`combined_scale = x_scale × w_scale_mean`，以 uint64 形式通过 ONNX Constant 节点传入
- per-channel 的 **scale 差异**（各 output channel 的 scale 与均值之间的偏差）通过 `correction = w_scale / w_scale_mean` 在 `QuantBatchMatmul` 输出后做 **Mul 补偿**——这样既满足硬件接口约束，又保留了 per-channel 量化的精度优势

### 融合线性层的包装器

融合 Linear（QKV、Gate/Up）的量化调用模式：

```python
# 融合 Linear 的 forward 模式
def fused_linear_forward(hidden_states, weights, biases, quant_params):
    # 输出维度拼接 weight
    w = torch.cat(weights, dim=0)
    # 偏置拼接（可能为 None）
    b = torch.cat(biases, dim=0) if all(b is not None for b in biases) else None

    if TORCH_PTQ_INT8 and quant_params_available:
        y = quant_linear_symmetric_int8(hidden_states, w, b, ...)
    else:
        y = F.linear(hidden_states, w, b)

    # 按各子 weight 的输出维度拆分
    split_sizes = [w.shape[0] for w in weights]
    return y.split(split_sizes, dim=-1)
```

## Step 6：条件替换导出

### 核心原则

- **FP32 分支**：保持原始代码不动（使用 `F.linear` 或 `nn.Linear`）
- **量化分支**：在 forward 中判断量化参数是否存在，替换为 Step 5 的包装函数
- 两个分支共享同一份模型定义，通过条件切换

### 导出流程模板

```python
def export_model_ptq(model, output_dir, device, calib_data=None):
    # 1. 准备模型（训练态 → 推理态）
    model.eval()
    prefill_model = ModelWrapper(model, mode="prefill")
    decode_model = ModelWrapper(model, mode="decode")
    
    # 2. 导出 FP32 ONNX
    export_onnx(prefill_model, prefill_path, dummy_inputs_fp32)
    
    # 3. 量化校准 + 参数计算
    if ENABLE_PTQ:
        # 挂载 Observer
        attach_observers(decode_model)
        # 运行校准
        run_calibration(prefill_model, decode_model, calib_data)
        # 计算量化参数
        compute_quant_params(decode_model)
        # 清除 Observer
        cleanup_observers(decode_model)
    
    # 4. 导出量化 ONNX
    export_onnx(decode_model, decode_path, dummy_inputs_decode)
```

### 条件替换的判断模式

```python
# 在模型 forward 中的典型判断模式
if TORCH_PTQ_INT8 and hasattr(linear_module, "_ptq_w_q"):
    # 使用量化线性层
    output = quant_linear_forward(x, ...)
else:
    # 使用原始 FP32 线性层
    output = linear_module(x)
```

对于融合 Linear，判断参数存储在父模块上：

```python
if TORCH_PTQ_INT8 and hasattr(parent_module, "_ptq_merged_w_q"):
    gate, up = quant_fused_linear_forward(x, ...)
else:
    gate, up = original_fused_linear_forward(x, ...)
```

## 命令行接口通用模板

当将此模式适配到新项目时，推荐暴露以下参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ptq-int8` | 启用 | 启用/禁用 PTQ int8 量化 |
| `--ptq-calib-data` | 空 | 校准数据路径（JSONL 或其他格式） |
| `--ptq-max-samples` | 32 | 最大校准样本数 |
| `--ptq-max-decode-steps` | 32 | 自回归模型每样本最大 decode 步数 |
| `--smooth-alpha` | 0.5~0.65 | SmoothQuant alpha（0=纯 weight 平滑，1=纯激活平滑） |
| `--weight-clip-ratio` | 0.0 | Weight 离群值裁剪比例 |

## 常见模型适配示例（非穷举）

以下示例展示不同模型架构如何应用此模式。核心原则是**任何含 `nn.Linear` / `F.linear` / MatMul 的模型均可适配**，下表仅为常见参考：

| 模型类型 | 可融合线性层 | Observer 挂载位置 | 特殊处理 |
|---|---|---|---|
| LLaMA / Qwen | QKV, Gate/Up | attention / mlp 模块 | Q/KV norm 不量化，仅投影层量化 |
| GPT / BLOOM | QKV | attention 模块 | 偏置处理（可能为 None） |
| LLaVA / VLM | Vision encoder 中的线性层 | vision encoder 子模块 | 仅文本 Decode 侧量化，视觉编码保持 FP32 |
| BERT | QKV, FFN (dense) | attention / intermediate/output | 双向注意力，校准时无需分 Prefill/Decode |
| MOE | Expert 合并 Linear | expert 模块 | 路由和门控不量化 |
| MLP / MLP-Mixer | 全连接层 | fc 模块 | 无融合 Linear，每层独立量化 |
| CNN (含 Linear 分类头) | 分类头 Linear | classifier 模块 | 卷积层不量化，仅量化全连接头；校准时单次前向即可 |
| RNN / LSTM | 输入-隐层 / 隐层-输出 Linear | rnn / lstm 模块 | 循环步间共享量化参数；校准时需展开时序收集激活 |

## 通用注意事项

1. **量化 vs 不量化的边界**：仅对带宽敏感的大线性层量化（embedding、layernorm、小投影层通常不量化）
2. **Prefill vs Decode 差异化**：自回归模型可在 Prefill（计算密集型）保持 FP32，Decode（访存密集型）使用 int8
3. **校准数据代表性**：校准数据分布需与部署数据分布一致，否则量化误差会放大
4. **Custom 算子平台依赖**：`AscendQuant`/`QuantBatchMatmul` 等算子需目标后端的 CANN/MindSpore Lite 版本支持
5. **数值验证**：量化前后输出的 cosine similarity / max abs error 需在可接受范围内
6. **融合 Linear 校准的陷阱**：QKV 合并后一起量化，等效于三个线性层共享同一个激活 scale，可能导致某个子投影的精度损失比独立量化更大
