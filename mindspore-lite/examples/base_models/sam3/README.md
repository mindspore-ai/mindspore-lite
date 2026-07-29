# SAM3 — MindSpore Lite 云侧推理部署

## 模型概述

SAM3 (Segment Anything Model 3) 是 Meta 发布的统一分割基础模型，支持通过文本或视觉提示（点、框、掩码）进行图像和视频中的目标检测、分割与跟踪。本方案适配 SAM3.1 图像模型（`sam3.1_multiplex.pt`），将其拆分为三个子模型并导出为 ONNX，再转换为 MindIR，在 Ascend 300I DUO硬件上通过 MindSpore Lite 进行推理。

| 属性 | 值 |
|------|-----|
| 模型名称 | SAM3.1 Image Model |
| 原始框架 | PyTorch 2.3+ |
| 权重文件 | `sam3.1_multiplex.pt` (3.3 GB) |
| 输入分辨率 | 1008 × 1008 |
| 文本上下文长度 | 32 tokens (CLIP BPE) |
| 检测查询数 | 200 |
| 掩码分辨率 | 288 × 288 |
| 精度 | FP32 |

## 模块拆分策略

SAM3 图像模型由 ViT 视觉骨干、CLIP 文本编码器和 DETR 检测器三部分组成。由于模型参数量大（848M）且包含多种复杂算子，采用三模块独立导出策略：

| 模块 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `sam3_image_encoder` | ViT-32L + FPN 骨干网络 | image [1,3,1008,1008] | 3级FPN特征 + 3级位置编码 |
| `sam3_language_encoder` | CLIP 文本编码器 (24L) | text_tokens [1,32] | language_features [32,1,256], mask [1,32] |
| `sam3_decoder` | DETR编码器+解码器+分割头 | FPN特征 + 位置编码 + 文本特征 | pred_logits, pred_boxes, pred_masks, presence_logit |

## 环境准备

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

## 文件结构

```text
sam3/
├── export_sam3_onnx.py          # ONNX 导出脚本（三模块）
├── infer_sam3_onnx.py           # ONNX Runtime 推理脚本
├── infer_sam3_mslite.py         # MindSpore Lite 推理脚本
├── config/
│   ├── config_image_encoder.ini     # 图像编码器转换配置
│   ├── config_language_encoder.ini  # 文本编码器转换配置
│   └── config_decoder.ini           # 解码器转换配置
├── onnx/                        # 导出的 ONNX 模型
│   ├── sam3_image_encoder.onnx
│   ├── sam3_language_encoder.onnx
│   └── sam3_decoder.onnx
└── mindir/                      # 转换后的 MindIR 模型
    ├── sam3_image_encoder.mindir
    ├── sam3_language_encoder.mindir
    └── sam3_decoder.mindir
```

## 使用方法

### 1. ONNX 导出

```bash
python export_sam3_onnx.py \
    --checkpoint /path/to/sam3.1_multiplex.pt \
    --output-dir ./onnx
```

导出后需将 ONNX 外部数据合并为单文件：

```bash
python -c "
import onnx, os
for name in ['sam3_image_encoder', 'sam3_language_encoder', 'sam3_decoder']:
    m = onnx.load(f'./onnx/{name}.onnx', load_external_data=True)
    onnx.save_model(m, f'./onnx_consolidated/{name}.onnx',
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=f'{name}.onnx.data', size_threshold=1024, convert_attribute=True)
"
```

### 2. ONNX → MindIR 转换

```bash
mkdir -p mindir

# 图像编码器
converter_lite --fmk=ONNX \
    --modelFile=./onnx_consolidated/sam3_image_encoder.onnx \
    --outputFile=./mindir/sam3_image_encoder \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=config_image_encoder.ini

# 文本编码器
converter_lite --fmk=ONNX \
    --modelFile=./onnx_consolidated/sam3_language_encoder.onnx \
    --outputFile=./mindir/sam3_language_encoder \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=config_language_encoder.ini

# 解码器
converter_lite --fmk=ONNX \
    --modelFile=./onnx_consolidated/sam3_decoder.onnx \
    --outputFile=./mindir/sam3_decoder \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=config_decoder.ini
```

### 3. ONNX 推理验证

```bash
# 精度对齐验证（torch vs ONNX）
python infer_sam3_onnx.py --onnx-dir ./onnx --align-check \
    --checkpoint /path/to/sam3.1_multiplex.pt

# 图像推理
python infer_sam3_onnx.py --onnx-dir ./onnx \
    --image /path/to/image.jpg --prompt "a dog"
```

### 4. MindSpore Lite 推理

```bash
# 精度对齐验证（MindIR vs ONNX）
python infer_sam3_mslite.py --mindir-dir ./mindir --onnx-dir ./onnx --align-check

# 图像推理
python infer_sam3_mslite.py --mindir-dir ./mindir \
    --image /path/to/image.jpg --prompt "a dog"
```

## 精度对齐结果

### PyTorch vs ONNX (cosine similarity)

| 模块 | 输出 | Cosine Sim | 状态 |
|------|------|-----------|------|
| Image Encoder | backbone_fpn_0 | 1.000000 | PASS |
| Image Encoder | backbone_fpn_1 | 1.000000 | PASS |
| Image Encoder | backbone_fpn_2 | 1.000000 | PASS |
| Language Encoder | language_features | 1.000000 | PASS |
| Decoder | pred_logits | 1.000000 | PASS |
| Decoder | pred_boxes | 1.000000 | PASS |
| Decoder | pred_masks | 1.000000 | PASS |
| Decoder | presence_logit | 1.000000 | PASS |

### MindIR vs ONNX (cosine similarity)

| 模块 | 输出 | Cosine Sim | 状态 |
|------|------|-----------|------|
| Image Encoder | backbone_fpn_0 | 1.0000 | PASS |
| Image Encoder | backbone_fpn_1 | 0.9999 | PASS |
| Image Encoder | backbone_fpn_2 | 0.9999 | PASS |
| Language Encoder | language_features | 0.9999 | PASS |
| Decoder | pred_logits | 0.9999 | PASS |
| Decoder | pred_boxes | 0.9999 | PASS |
| Decoder | pred_masks | 0.9998 | PASS |
| Decoder | presence_logit | 1.0000 | PASS |

## 性能数据

### 模型文件大小

| 模块 | ONNX 大小 | MindIR 大小 |
|------|----------|------------|
| sam3_image_encoder | 1749 MB | 991 MB |
| sam3_language_encoder | 1347 MB | 677 MB |
| sam3_decoder | 92 MB | 73 MB |
| **合计** | **3188 MB** | **1741 MB** |

### MindIR 推理性能 (Ascend Atlas 300I Duo, FP32, batch=1)

| 模块 | 平均推理时间 |
|------|------------|
| Image Encoder (ViT-32L + FPN) | 0.786 s |
| Language Encoder (CLIP-24L) | 0.017 s |
| Decoder (DETR + SegHead) | 0.287 s |
| **端到端总计** | **1.090 s** |

## 算子适配说明

在 ONNX 导出过程中，对以下 PyTorch 算子进行了替换以兼容 ONNX/MindIR：

| 问题 | 原始实现 | 替换方案 |
|------|---------|---------|
| RoPE 复数运算 | `torch.view_as_complex` / `view_as_real` | 手动实部/虚部算术运算 |
| SDPA 注意力 | `F.scaled_dot_product_attention` | 手动 softmax + matmul 实现 |
| 位置编码缓存 | `if cache_key in self.cache` 条件分支 | 移除缓存检查，直接计算 |
| 窗口分区 | `if pad_h > 0` 条件分支 | 始终执行 F.pad（0 padding 为 no-op） |
| 绝对位置编码 | `if size != h` 条件分支 + `tile` | 预计算为 buffer，避免运行时条件 |
| MLP 融合算子 | `torch.ops.aten._addmm_activation` | 标准 `nn.Linear` + `nn.functional.gelu/relu` |
| 掩码预测 einsum | `torch.einsum("bqc,chw->bqhw")` | `matmul` + `reshape` |
| BFloat16 权重 | 检查点 BFloat16 参数 | 加载后 `.float()` 转换 |

## 常见问题

### Q: ONNX Runtime 推理结果与 PyTorch 不一致？

ONNX Runtime 的图优化（`ORT_ENABLE_ALL`）可能改变计算顺序导致数值差异。使用 `ORT_DISABLE_ALL` 优化级别可避免此问题。

### Q: 图像编码器 ONNX 转 MindIR 失败（If 节点）？

ViT 骨干中的 `get_abs_pos` 函数和 `PositionEmbeddingSine` 包含数据依赖的条件分支，会生成 ONNX `If` 节点。需通过预计算位置编码为 buffer 来消除条件分支。

### Q: MindIR 加载报 "Offline converted MindIR is not supported"？

使用 `--optimize=ascend_oriented` 转换时需要配合 `--configFile` 指定 `plugin_custom_ops=All`，否则生成的 MindIR 无法被 MSLite 运行时加载。
