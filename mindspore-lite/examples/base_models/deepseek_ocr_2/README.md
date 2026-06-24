# DeepSeek-OCR-2 ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `deepseek-ai/DeepSeek-OCR-2`（DeepSeek-VL-V2 架构，~3.4B MoE 多模态 OCR 模型）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上推理。

模型拆分为三个子图：视觉编码器（SAM-ViT-B + Qwen2 decoder + 投影）、LLM Prefill、LLM Decode。其中 **MoE（64 路由专家，top-6，+2 共享）被展平为可导出的纯张量计算**。

---

## 1. 环境准备

### 系统要求

- Python 3.11、Linux、昇腾环境（MindSpore Lite + Ascend 驱动）
- `source /home/yf/env.sh`（CANN 8.5.0，torch 2.9.0+cpu，transformers 5.9.0，mindspore-lite 2.9.0）

### 依赖

| 软件包 | 版本 |
|---|---|
| torch | 2.9.0（CPU，仅用于导出） |
| transformers | 5.9.0 |
| torchvision | 0.24.0（DeepSeek 建模代码 `import torchvision`） |
| onnx / onnxruntime | 1.19 / 1.24 |
| mindspore-lite | 2.9.0 |
| einops / addict / easydict | DeepSeek custom code 依赖 |

```bash
pip install torch==2.9.0 transformers==5.9.0 torchvision==0.24.0 onnx==1.19.1 onnxruntime==1.24.2 einops addict easydict
```

### transformers 5.x 兼容（关键）

DeepSeek-OCR-2 的 `trust_remote_code` 建模代码面向 transformers 4.46。在 5.9.0 下需 2 个 shim + 配置默认值补丁才能加载（已封装在导出脚本 `_apply_transformers_shims` / `_patch_config`）：

- shim `LlamaFlashAttention2`（5.x 已移除）= `LlamaAttention` 子类；
- shim `is_torch_fx_available`（5.x 已移除）= 返回 False；
- 补齐 `attention_dropout`、`rms_norm_eps`、`rope_theta`、`pad_token_id` 等 DeepSeek-V2 配置默认值（5.x config loader 会跳过），强制 `_attn_implementation="eager"`。

> 已验证：transformers 5.9.0 下可成功 `from_pretrained` 加载真实权重（3.389B 参数），**无需单独 venv**。

---

## 2. 模型下载

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('deepseek-ai/DeepSeek-OCR-2'))"
# 默认缓存到 ~/.cache/modelscope/，软链/拷贝到 ./DeepSeek-OCR-2
```

---

## 3. 开源算法调研与导出规划

### 架构

| 组件 | 说明 |
|------|------|
| 视觉 | `ImageEncoderViT`（SAM-ViT-B，1024²，窗口注意力+相对位置）→ `Qwen2Decoder2Encoder`（24 层 Qwen2，non-causal/causal 混合 mask）→ `MlpProjector`（Linear 896→1280） |
| 语言 | DeepSeek-V2 MoE：12 层，hidden 1280，**layer0 dense，layer1-11 MoE（64 路由专家 top-6 + 2 共享）**；`use_mla=False` → 标准 MHA（10 头，head_dim 128，rotate-half RoPE）；vocab 129280 |
| 动态分辨率 | 原图按宽高比切 0–6 个 768² crops + 1 个 1024² 全局视图 |

### 导出规划

| 子图 | 处理 |
|------|------|
| Vision | 固定 `n_crops` 个 crops + 1 全局视图；Qwen2Decoder 的自定义 4D mask（`.nonzero()` 不可 trace）在固定 grid 下预计算烘焙进图（`_vision_decoder_mask`） |
| LLM Prefill | 标准 MHA + rotate-half RoPE（`RotaryMul` 自定义算子）；动态 seq |
| LLM Decode | 固定 KV cache + `Scatter` 更新 + `IncreFlashAttention` |
| **MoE 展平** | 原 `moe_infer` 用 `argsort()`+`.cpu().item()` 循环（数据依赖控制流，**不可 trace**）→ 展平：`topk(softmax(gate))` 门控 + `einsum` 计算全部 64 专家 + `gather` 按 top-6 聚合 + 加 2 共享专家 |

---

## 4. ONNX 模型导出

```bash
cd ./mindspore-lite/examples/base_models/deepseek_ocr_2

python export_deepseek_ocr_2_onnx.py \
  --model-id ./DeepSeek-OCR-2 \
  --output-dir ./deepseek_ocr_2_onnx \
  --device cpu \
  --n-crops 2 \
  --kv-cache-len 2048 \
  --dtype fp16
```

### 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 本地权重目录 | 必填 |
| `--output-dir` | 输出目录 | `./deepseek_ocr_2_onnx` |
| `--device` | 导出设备 | `cpu` |
| `--n-crops` | 固定局部 crop 数（控制图像 token 数） | `2` |
| `--kv-cache-len` | decode 固定 KV cache 长度 | `2048` |
| `--dtype` | 导出精度（fp16/fp32） | `fp16` |
| `--skip-vision` | 仅导出 LLM | `False` |

### 导出产物

```text
deepseek_ocr_2_onnx/
├── deepseek_ocr_2_vision.onnx          # 视觉编码器
├── deepseek_ocr_2_llm_prefill.onnx     # LLM prefill（+ external data，~GB 级，64 专家权重）
├── deepseek_ocr_2_llm_decode.onnx      # LLM decode
└── onnx__Einsum_* / onnx__MatMul_*     # 大权重外部化数据
```

> LLM 权重 >2GB，ONNX 自动外部化；转换时同目录加载。

---

## 5. MindSpore Lite 转换

```bash
cd ./mindspore-lite/examples/base_models/deepseek_ocr_2
CONV=./converter_lite

# Vision（force_fp32，避免 ViT 注意力溢出）
$CONV --fmk=ONNX --modelFile=./deepseek_ocr_2_onnx/deepseek_ocr_2_vision.onnx \
  --outputFile=./deepseek_ocr_2_onnx/deepseek_ocr_2_vision \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/deepseek_ocr_2_vision.config

# Prefill（默认 fp16，不用 force_fp32）
$CONV --fmk=ONNX --modelFile=./deepseek_ocr_2_onnx/deepseek_ocr_2_llm_prefill.onnx \
  --outputFile=./deepseek_ocr_2_onnx/deepseek_ocr_2_llm_prefill \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/deepseek_ocr_2_llm_prefill.config

# Decode（默认 fp16）
$CONV --fmk=ONNX --modelFile=./deepseek_ocr_2_onnx/deepseek_ocr_2_llm_decode.onnx \
  --outputFile=./deepseek_ocr_2_onnx/deepseek_ocr_2_llm_decode \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/deepseek_ocr_2_llm_decode.config
```

> 转换日志中的 warning（protobuf size、SetupParamInitSubGraph 等）可忽略。大模型产物为 `*_graph.mindir` + `*_variables/`。

---

## 6. MindSpore Lite 推理

```bash
python infer_deepseek_ocr_2_mslite.py \
  --vision-model ./deepseek_ocr_2_onnx/deepseek_ocr_2_vision_graph.mindir \
  --prefill-model ./deepseek_ocr_2_onnx/deepseek_ocr_2_llm_prefill_graph.mindir \
  --decode-model ./deepseek_ocr_2_onnx/deepseek_ocr_2_llm_decode_graph.mindir \
  --tokenizer ./DeepSeek-OCR-2 \
  --image ./your_doc.png \
  --prompt "<image>\nFree OCR. " \
  --n-crops 2 --kv-cache-len 2048 --prefill-seq 640 \
  --device ascend --device-id 0
```

推理脚本纯 numpy/PIL（无 torch），复刻 `model.infer()` 的图像预处理（全局 1024² + N 个 768² crop，normalize 0.5/0.5）与 `<image>` token 布局。

---

## 7. 性能数据

> 待端到端联调通过后在 Atlas 300I Duo（310P3）实测填入。

| 指标 | 300I Duo |
|---|---|
| Vision (ms) | 待测 |
| Prefill (ms) | 待测 |
| Avg Decode Step (ms) | 待测 |
| Throughput (tok/s) | 待测 |

---

## 8. 精度对齐

用同一图像/prompt 对比 HuggingFace 原始 `model.infer()` 与 MindSpore Lite 输出（OCR 文本）。如偏差：

- 确保 `--n-crops` 与导出一致（图像 token 数 = n_crops×144 + 257）。
- LLM 用默认 fp16 转换（**不要 force_fp32**，见 FAQ）。
- MoE 门控：确认 `topk_method=greedy`、`n_group=1`（本配置）。

---

## 9. 常见问题与已知限制

### 1) `force_fp32` 破坏 LLM 输出

与 GLM-OCR 一致，`converter_lite` 对该 DeepSeek MoE 图，`force_fp32`/`allow_fp32_to_fp16` 会破坏输出（空串/EOS）。LLM 必须用默认 fp16 转换。vision 用 `force_fp32`（避免 ViT 溢出）。

### 2) MoE 不可直接 trace

原 `DeepseekV2MoE.moe_infer` 用 `argsort()` 派发 + `.cpu().item()` 循环（数据依赖控制流）。导出脚本 `_moe_forward` 已展平：`einsum` 全专家 + `gather` top-6 + 共享专家。已验证 prefill 可成功导出（64 专家权重外部化）。

### 3) Qwen2 vision decoder 的自定义 mask

`CustomQwen2ModelInner._create_custom_4d_mask` 用 `.nonzero()` + Python 循环，不可 trace。导出脚本在固定 grid 下预计算 mask（`_vision_decoder_mask`）并直接驱动 Qwen2 层。

### 4) 动态分块 → 固定 crops

原始 `dynamic_preprocess` 按宽高比切 0–6 个 crop；导出固定 `--n-crops`（默认 2），推理须用相同值。

### 5) LlamaAttention 5.x 无 `num_heads` 属性

transformers 5.x `LlamaAttention` 只有 `head_dim`/`scaling`，无 `num_heads`。导出脚本 `_attn_dims` 由投影权重推导 `(num_heads, num_kv_heads, head_dim)`。

### 6) 进行中的工作

LLM decode 导出、vision 端到端、整体精度对齐仍在联调；MoE 展平 prefill 导出与 transformers 5.x 加载方案已验证可用。

---

## 10. 参考资源

- [DeepSeek-OCR-2（ModelScope）](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite)

---

## 11. 许可证

DeepSeek-OCR-2 模型遵循其原始许可证。本教程遵循相应依赖的许可证要求。
