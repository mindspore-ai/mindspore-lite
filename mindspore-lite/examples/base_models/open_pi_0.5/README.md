# PI0.5 Base ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 PI0.5 Base 视觉-语言-动作（VLA）模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

PI0.5 Base 采用双 Transformer 架构，模型被拆分为 2 个 ONNX 文件：

1. **Prefix Encoder**（`prefix_encoder.onnx`）：SigLIP ViT + PaliGemma LLM（Gemma 2B），对 3 张图像和文本 prompt 进行编码，输出 KV cache（只运行一次）
2. **Denoise Step**（`denoise_step.onnx`）：Action Expert（Gemma 300M），基于 KV cache 进行单步去噪，输出速度场（循环调用 10 次，Euler 积分）

## 模型架构

```log
┌─────────────────────────────────────────────────────────┐
│                    PI0.5 Base Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Prefix Encoder (运行 1 次)                                │
│  ┌──────────┐   ┌─────────────────────────────────┐      │
│  │  SigLIP  │──▶│  PaliGemma LLM (Gemma 2B, 18层) │──┐   │
│  │  ViT     │   │  输出: KV cache (36 tensors)     │  │   │
│  └──────────┘   └─────────────────────────────────┘  │   │
│        ▲                                             │   │
│   3 × 224×224 图像 + 文本 tokens                       │   │
│                                                       │   │
│  Denoise Loop (运行 10 次, Euler 积分)                  │   │
│  ┌─────────────────────────────────────────────┐      │   │
│  │  Action Expert (Gemma 300M, 18层)             │      │   │
│  │  输入: x_t + timestep + KV cache ─────────────┘   │   │
│  │  输出: v_t (速度场)                                │   │
│  └─────────────────────────────────────────────┘      │   │
│        ▲                                              ▼   │
│     随机噪声 ──10步 Euler 积分──▶ 机器人动作 (50×32)      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_llm_layers` | 18 | PaliGemma LLM 和 Action Expert 的层数 |
| `head_dim` | 256 | 注意力头维度 |
| `num_kv_heads` | 1 | GQA 的 KV 头数 |
| `patches_per_image` | 256 | 每张图像的 patch 数（224/14）² |
| `action_dim` | 32 | 动作维度 |
| `action_horizon` | 50 | 动作时序长度 |
| `max_token_len` | 200 | 最大文本 token 长度 |
| `prefix_seq_len` | 968 | 3×256 + 200，前缀序列长度 |

---

## 1. 环境准备

### 1.1 依赖版本

| 软件包            | 版本       |
| -------------- | -------- |
| Python         |  3.11     |
| torch          | >= 2.0   |
| torchvision    | 0.15.0   |
| transformers    | 4.53.2   |
| safetensors    | >= 0.4   |
| sentencepiece    | >= 0.1   |
| opencv-python    | >= 4.5   |
| numpy          | 2.2.6     |
| onnx           | >= 1.15   |
| onnxruntime    | 1.23.2   |
| CANN           | 8.5.0    |
| mindspore-lite | 2.8.0    |

### 1.2 环境安装

```bash
pip install torch transformers safetensors sentencepiece onnx opencv-python
# 安装[open pi](https://github.com/Physical-Intelligence/openpi)项目依赖（用于加载模型结构）
pip install -e .
```

### 1.3 使能昇腾CANN环境

配置 Ascend 推理环境：

```bash
source /path/to/Ascend/set_env.sh
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
python export_pi0.5_onnx.py \
  --checkpoint_dir ./pi05_base \
  --output_dir ./onnx_output_fp16
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--checkpoint_dir` | 模型权重目录（包含 model.safetensors） | `/patch/to/pi05_base` |
| `--output_dir` | ONNX 输出目录 | `./onnx_output_fp16` |

### 产出

由于 PaliGemma LLM（Gemma 2B）权重超过 2GB，ONNX 导出会将权重存为外部数据文件：

```text
onnx_output_fp16/
├── prefix_encoder.onnx                              # 图定义 (~1.6 MB)
├── model.paligemma_with_expert.paligemma.*.weight   # 外部权重文件 (~13 GB)
└── denoise_step.onnx                                # 完整模型 (~1.7 GB)
```

> **注意**：不要删除 `model.*` 开头的外部数据文件，它们是 prefix_encoder.onnx 的权重。

### ONNX 模型输入输出

**Prefix Encoder** — `prefix_encoder.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `image_0` | `(1, 3, 224, 224)` | float32 | 第 1 张图像 |
| 输入 | `image_1` | `(1, 3, 224, 224)` | float32 | 第 2 张图像 |
| 输入 | `image_2` | `(1, 3, 224, 224)` | float32 | 第 3 张图像 |
| 输入 | `img_mask_0` | `(1,)` | bool | 第 1 张图像是否有效 |
| 输入 | `img_mask_1` | `(1,)` | bool | 第 2 张图像是否有效 |
| 输入 | `img_mask_2` | `(1,)` | bool | 第 3 张图像是否有效 |
| 输入 | `lang_tokens` | `(1, 200)` | int64 | 文本 token IDs |
| 输入 | `lang_masks` | `(1, 200)` | bool | 文本 mask |
| 输出 | `prefix_pad_masks` | `(1, 968)` | bool | 前缀 padding mask |
| 输出 | `kv_key_0` ~ `kv_key_17` | `(1, 1, 968, 256)` | float32 | 18 层 KV cache key |
| 输出 | `kv_val_0` ~ `kv_val_17` | `(1, 1, 968, 256)` | float32 | 18 层 KV cache value |

> prefix_seq_len = 3 × 256 (patches) + 200 (tokens) = 968。KV cache 按 key_0, val_0, key_1, val_1, ... 交错排列，共 37 个输出。

**Denoise Step** — `denoise_step.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `state` | `(1, 32)` | float32 | 机器人状态 |
| 输入 | `x_t` | `(1, 50, 32)` | float32 | 当前噪声动作 |
| 输入 | `timestep` | `(1,)` | float32 | 扩散时间步 |
| 输入 | `prefix_pad_masks` | `(1, 968)` | bool | 前缀 padding mask |
| 输入 | `kv_key_0` ~ `kv_key_17` | `(1, 1, 968, 256)` | float32 | 18 层 KV cache key |
| 输入 | `kv_val_0` ~ `kv_val_17` | `(1, 1, 968, 256)` | float32 | 18 层 KV cache value |
| 输出 | `v_t` | `(1, 50, 32)` | float32 | 速度场 |

> ONNX 模型包含 state 输入（PI0 的 state_proj），但是由于 PI0.5 中 state 未使用，因此导出的 ONNX 模型中不存在 state 输入。

---

## 3. ONNX 转 MindIR

### 配置文件

创建 `./configs/config.ini`：

```ini
# config.ini
[acl_build_options]
ge.exec.precision_mode=allow_mix_precision_fp16
ge.exec.modify_mixlist="./op_fp32.json"
```

`op_fp32.json` 文件内容：

```json
{
  "black-list": {
    "to-add": ["Square"]
  }
}
```

### 转换命令

使用 `converter_lite` 命令：

```bash
# 创建mindir_output用于保存转换后的mindir模型
mkdir mindir_output
Convert=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# Prefix Encoder 转换（使用混合精度）
$Convert --fmk=ONNX \
  --modelFile=./onnx_output_fp16/prefix_encoder.onnx \
  --outputFile=./mindir_output/prefix_encoder \
  --saveType=MINDIR \
  --optimize=ascend_oriented \
  --configFile=./configs/config.ini

# Denoise Step 转换
$Convert --fmk=ONNX \
  --modelFile=./onnx_output_fp16/denoise_step.onnx \
  --outputFile=./mindir_output/denoise_step \
  --saveType=MINDIR \
  --optimize=ascend_oriented
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--fmk` | 输入模型格式（ONNX） |
| `--modelFile` | 输入 ONNX 模型路径 |
| `--outputFile` | 输出路径（不带扩展名） |
| `--saveType` | 输出格式（MINDIR） |
| `--device` | 目标设备（Ascend） |
| `--optimize` | 优化模式（`ascend_oriented`） |
| `--configFile` | 配置文件路径             |

### 产出

模型超过 2GB 时，输出分为 `*_graph.mindir`（图定义）和 `*_variables/`（权重数据）：

```text
./mindir_output/
├── prefix_encoder_graph.mindir              # 图定义 (~17 KB)
├── prefix_encoder_variables/
│   └── data_0                               # 权重数据 (~5.3 GB)
└── denoise_step.mindir                      # 完整模型 (~827 MB)
```

### MindIR 模型输入输出

**Prefix Encoder** — `prefix_encoder_graph.mindir`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `image_0` | `(1, 3, 224, 224)` | float32 | 第 1 张图像 |
| 输入 | `image_1` | `(1, 3, 224, 224)` | float32 | 第 2 张图像 |
| 输入 | `image_2` | `(1, 3, 224, 224)` | float32 | 第 3 张图像 |
| 输入 | `img_mask_0` | `(1,)` | bool | 第 1 张图像是否有效 |
| 输入 | `img_mask_1` | `(1,)` | bool | 第 2 张图像是否有效 |
| 输入 | `img_mask_2` | `(1,)` | bool | 第 3 张图像是否有效 |
| 输入 | `lang_tokens` | `(1, 200)` | int32 | 文本 token IDs |
| 输入 | `lang_masks` | `(1, 200)` | bool | 文本 mask |
| 输出 | `prefix_pad_masks` | `(1, 968)` | bool | 前缀 padding mask |
| 输出 | `kv_key_0` ~ `kv_key_17` | `(1, 1, 968, 256)` | float16 | 18 层 KV cache key（交错排列） |
| 输出 | `kv_val_0` ~ `kv_val_17` | `(1, 1, 968, 256)` | float16 | 18 层 KV cache value（交错排列） |

**Denoise Step** — `denoise_step.mindir`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `x_t` | `(1, 50, 32)` | float32 | 当前噪声动作 |
| 输入 | `timestep` | `(1,)` | float32 | 扩散时间步 |
| 输入 | `prefix_pad_masks` | `(1, 968)` | bool | 前缀 padding mask |
| 输入 | `kv_key_0` ~ `kv_key_17` | `(1, 1, 968, 256)` | float16 | 18 层 KV cache key（交错排列） |
| 输入 | `kv_val_0` ~ `kv_val_17` | `(1, 1, 968, 256)` | float16 | 18 层 KV cache value（交错排列） |
| 输出 | `v_t` | `(1, 50, 32)` | float32 | 速度场 |

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_pi0.5_mindir.py \
  --prefix_model ./mindir_output/prefix_encoder_graph.mindir \
  --denoise_model ./mindir_output/denoise_step.mindir \
  --device Ascend \
  --prompt "pick up the cup" \
  --tokenizer_path ./paligemma_tokenizer.model \
  --output ./mindir_inference_result.npy \
  --num_steps 10 \
  --seed 42
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--prefix_model` | Prefix Encoder MindIR 路径 | `./mindir_output/prefix_encoder_graph.mindir` |
| `--denoise_model` | Denoise Step MindIR 路径 | `./mindir_output/denoise_step.mindir` |
| `--device` | 推理设备（Ascend/GPU/CPU） | `Ascend` |
| `--prompt` | 任务描述文本 | `"pick up the cup"` |
| `--num_steps` | 去噪步数 | `10` |
| `--seed` | 随机种子 | `42` |
| `--tokenizer_path` | PaliGemma tokenizer 路径 | 自动搜索 |
| `--output` | 输出文件路径 | `mindir_inference_result.npy` |

### 推理示例输出

```text
Loaded MINDIR model: ./prefix_encoder_mix_fp16_gelu_graph.mindir
Loaded MINDIR model: ./denoise_step_fp16_gelu.mindir
Loaded tokenizer from ./paligemma_tokenizer.model
Starting MindIR inference with zero-copy KV cache (float16)...
Prefix encoder: KV cache on device directly (36 tensors, ~17.0 MB, dtype=DataType.FLOAT16)
  Prefix encoder: 162.6 ms
  Step 0/10: 11.6 ms, v_t range=[-4.6055, 5.2227]
  Step 1/10: 10.1 ms, v_t range=[-4.2109, 4.6523]
  Step 2/10: 9.8 ms, v_t range=[-3.9277, 4.3594]
  Step 3/10: 9.9 ms, v_t range=[-3.7305, 4.1953]
  Step 4/10: 10.1 ms, v_t range=[-3.6094, 4.1406]
  Step 5/10: 9.9 ms, v_t range=[-3.5605, 4.1367]
  Step 6/10: 9.9 ms, v_t range=[-3.5508, 4.1719]
  Step 7/10: 9.8 ms, v_t range=[-3.5469, 4.2109]
  Step 8/10: 9.7 ms, v_t range=[-3.5098, 4.2266]
  Step 9/10: 9.7 ms, v_t range=[-3.3887, 3.9863]
============================================================
TIMING SUMMARY
============================================================
  Preprocess:                       7.1 ms
  Prefix Encoder:                 162.6 ms
  Denoise Loop (total):           100.5 ms  (10 steps x 10.0 ms/step)
  Postprocess:                      0.0 ms
------------------------------------------------------------
  Model inference total:          263.1 ms
  End-to-end total:               270.2 ms
============================================================
Actions shape: (1, 50, 32)
Actions sample (first 3 steps, first 8 dims):
[[ 0.46174708  0.41178453  0.27532262 -0.14957762  0.70670354 -0.4259094
   0.42204726  0.4033562 ]
 [ 0.4657508   0.42654204  0.277089   -0.10103166  0.7008375  -0.40461636
   0.47210693  0.390789  ]
 [ 0.44961077  0.3882835   0.28475562 -0.13684797  0.68957794 -0.41835123
   0.44557405  0.36658883]]
Actions range: [-0.7562, 0.8884]
Results saved to ./mindir_inference_result.npy
```

## 5. 性能数据

### 测试环境

| 项目 | 配置 |
|------|------|
| 硬件 | Atlas 300I Duo（Ascend NPU, aarch64） |
| 模型 | PI0.5 Base (Gemma 2B + Gemma 300M) |
| 精度 | float16 |
| 去噪步数 | 10 |
| 动作维度 | 50 × 32 |

### 各阶段推理性能

**Prefix Encoder**

| 指标 | 值 |
|------|-----|
| 输入 | 3 × (1, 3, 224, 224) 图像 + (1, 200) 文本 tokens |
| 输出 | 1 × prefix_pad_masks + 36 × KV cache tensors |
| 耗时 | **162.6 ms** |

**Denoise Step（单步）**

| 指标 | 值 |
|------|-----|
| 输入 | x_t (1, 50, 32) + timestep (1,) + prefix_pad_masks (1, 968) + 36 × KV tensors |
| 输出 | v_t (1, 50, 32) |
| 平均单步耗时 | **10.0 ms** |

### 端到端推理性能（10 步去噪）

| 指标 | 耗时 (ms) |
|------|----------|
| Preprocess | 7.1 |
| Prefix Encoder | 162.6 |
| Denoise Loop（10 steps） | 100.5 |
| Postprocess | 0.0 |
| **模型推理总耗时** | **263.1** |
| **端到端总耗时** | **270.2** |

## 6. 常见问题 FAQ

### 6.1 执行导出脚本export_pi0.5_onnx.py报错

**问题1：**

```log
No module named 'openpi.models_pytorch'
```

**原因：**\
没有将openpi代码仓加入到python环境中。

**解决方案：**

```bash
export PYTHONPATH=/path/to/openpi/src:.:$PYTHONPATH #替换成实际的路径
```

**问题2：**

```log
ValueError: transformers_replace is not installed correctly.
```

**原因：**\
openpi对transformer库做了适配，但是适配的代码没有拷贝到transformer库中相应的位置。

**解决方案：**\
将openpi代码仓中./src/models_pytorch/transformers_replace/目录下的所有东西拷贝到python中transformers库下。

---

## 7. 参考资源

- [OpenPI 项目](https://github.com/Physical-Intelligence/openpi)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [CANN 开发文档](https://www.hiascend.com/document)
- [PaliGemma 论文](https://arxiv.org/abs/2407.07726)

---

## 8. 许可证

本教程遵循 OpenPI 项目的许可证。
