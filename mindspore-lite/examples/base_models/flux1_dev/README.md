# FLUX.1-dev ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `black-forest-labs/FLUX.1-dev`（~12B Rectified-Flow MMDiT 文生图扩散模型）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上完成端到端文生图推理与精度对齐。

FLUX.1-dev 由四部分组成：CLIP-L 文本编码器、T5-XXL 文本编码器、MMDiT Transformer（去噪器，19 double + 38 single blocks，inner_dim 3072）、以及 16 通道 VAE 解码器。本教程将四部分**全部导出为 MindIR** 并在昇腾上推理（transformer+VAE 在 dev0，T5+CLIP 在 dev1）。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux（昇腾环境，MindSpore Lite 2.9.0 + CANN 8.5.0 + Ascend 驱动）
- Atlas 300I Duo（310P3，单卡 ~44GB），transformer(24GB) 单卡可运行

### 依赖版本（建议）

| 软件包            | 版本   |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.9.0（CPU 即可，仅用于导出/对齐） |
| diffusers      | 0.38.0（原生支持 `FluxTransformer2DModel`） |
| transformers   | 5.9.0 |
| onnx           | 1.19.1 |
| mindspore-lite | 2.9.0 |
| CANN           | 8.5.0 |

### 安装命令

```bash
pip install torch==2.9.0 diffusers==0.38.0 transformers==5.9.0 onnx==1.19.1 onnxruntime
```

### 初始化环境

```bash
source /home/yf/env.sh   # CANN / mindspore-lite / converter_lite
```

---

## 2. 模型下载

从 ModelScope 下载 diffusers 格式权重（transformer 23.8GB + T5 9.5GB + CLIP 0.25GB + VAE 0.17GB，共 ~33.7GB）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('black-forest-labs/FLUX.1-dev', cache_dir='/home/yf/modelscope_cache'))"
```

将其软链到 `./FLUX.1-dev`：

```bash
ln -sfn /home/yf/modelscope_cache/models/black-forest-labs/FLUX.1-dev ./FLUX.1-dev
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本将 FLUX.1-dev 拆为四个 ONNX 子图：

1. **transformer**（`flux1_transformer.onnx`）：packed latents(4096×64) + T5 embeds(256×4096) + CLIP pooled(768) + timestep/guidance + img_ids/txt_ids → noise_pred。注意力替换为 CANN `PromptFlashAttention` Custom 算子。
2. **vae_decoder**（`flux1_vae_decoder.onnx`）：16 通道 latent [1,16,128,128] → RGB [1,3,1024,1024]。
3. **t5_encoder**（`flux1_t5_encoder.onnx`）：input_ids [1,256] → last_hidden_state [1,256,4096]。
4. **clip_encoder**（`flux1_clip_encoder.onnx`）：input_ids [1,77] → pooled embed [1,768]。

### 自定义算子策略

FLUX 注意力为全双向（无 causal / 无 padding mask）。脚本 monkeypatch diffusers 的注意力派发函数，将 `scaled_dot_product_attention` 替换为 CANN `PromptFlashAttention` Custom 节点（BNSD 布局、`sparse_mode=0`、不传 atten_mask）；q/k-norm（PyTorch `nn.RMSNorm`，legacy 导出器无 `aten::rms_norm` 符号）替换为 CANN `RmsNorm` Custom 节点（fp32 累加、数学等价）。其余（q/k/v 投影、RoPE、AdaLN、投影层）均走标准 diffusers 算子 trace。这样只需改注意力与归一化两处，避免手写 57 层，并显著缩小 ONNX 图规模、加速导出。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/flux1_dev

python export_flux1_dev_onnx.py \
  --model-id ./FLUX.1-dev \
  --output-dir ./flux1_dev_onnx \
  --resolution 1024 1024 \
  --t5-seq-len 256 \
  --dtype fp16
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 本地 diffusers 权重目录 | `./FLUX.1-dev` |
| `--output-dir` | ONNX 输出目录 | `./flux1_dev_onnx` |
| `--resolution` | 输出图像分辨率 H W | `1024 1024` |
| `--t5-seq-len` | 固定 T5 序列长度（256 标准；512 支持更长 prompt） | `256` |
| `--dtype` | 导出精度（fp16/fp32） | `fp16` |
| `--no-custom-op` | 不把注意力替换为 Custom 算子 | `False` |
| `--components` | 导出子集（transformer,vae,t5,clip） | 全部 |

### 模型架构参数

| 参数 | 值 |
|------|------|
| inner_dim | 3072（24 heads × 128） |
| num_layers (double) | 19 |
| num_single_layers | 38 |
| joint_attention_dim (T5) | 4096 |
| pooled_projection_dim (CLIP) | 768 |
| in_channels (packed latent) | 64（16 × 2×2 pack） |
| guidance_embeds | True（dev，guidance=3.5） |
| VAE latent channels | 16，scale 0.3611，shift 0.1159 |

---

## 4. ONNX 模型结构说明

transformer ONNX 使用 MindSpore Lite 自定义算子（`PromptFlashAttention`），**不支持直接用 ONNX Runtime 推理**，需通过 `converter_lite` 转 MindIR 后运行。VAE/T5/CLIP 为标准算子图。

### 模型中包含的自定义算子（transformer）

| 算子 | 数量 | 说明 |
|------|------|------|
| PromptFlashAttention | 57（19 double + 38 single） | 全双向注意力，无 mask |
| RmsNorm | 76（double 的 q/k-norm 38 + single 的 norm 38） | q/k-norm，fp32 累加 |

---

## 5. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/flux1_dev

CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# transformer
$CONV --fmk=ONNX --modelFile=./flux1_dev_onnx/flux1_transformer.onnx \
  --outputFile=./flux1_dev_onnx/flux1_transformer \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/flux1_transformer.config

# vae
$CONV --fmk=ONNX --modelFile=./flux1_dev_onnx/flux1_vae_decoder.onnx \
  --outputFile=./flux1_dev_onnx/flux1_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/flux1_vae_decoder.config

# t5
$CONV --fmk=ONNX --modelFile=./flux1_dev_onnx/flux1_t5_encoder.onnx \
  --outputFile=./flux1_dev_onnx/flux1_t5_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/flux1_t5_encoder.config

# clip
$CONV --fmk=ONNX --modelFile=./flux1_dev_onnx/flux1_clip_encoder.onnx \
  --outputFile=./flux1_dev_onnx/flux1_clip_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/flux1_clip_encoder.config
```

### config 说明

- `flux1_transformer.config`：固定 4096 latent token + 256 T5 token，`force_fp16`，`plugin_custom_ops=All`。
- `flux1_vae_decoder.config` / `flux1_t5_encoder.config` / `flux1_clip_encoder.config`：固定 shape，`force_fp16`。

> 转换日志中可能出现 `protobuf size` / `ge.proto.ModelDef exceeded maximum protobuf size` 等 warning（transformer 权重 ~24GB 外置化），**不影响最终产物**（产出 `*_graph.mindir` + `*_variables/`），可忽略。

---

## 6. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/flux1_dev

python infer_flux1_dev_mslite.py \
  --transformer-model ./flux1_dev_onnx/flux1_transformer_graph.mindir \
  --vae-model         ./flux1_dev_onnx/flux1_vae_decoder_graph.mindir \
  --t5-model          ./flux1_dev_onnx/flux1_t5_encoder_graph.mindir \
  --clip-model        ./flux1_dev_onnx/flux1_clip_encoder_graph.mindir \
  --model-dir ./FLUX.1-dev \
  --prompt "A cat holding a sign that says hello world" \
  --seed 0 --steps 28 --guidance 3.5 \
  --transformer-device 0 --text-device 1 \
  --output ./flux1_output.png
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--transformer-model` | transformer MindIR | 必填 |
| `--vae-model` | VAE MindIR | 必填 |
| `--t5-model` | T5 MindIR | 必填 |
| `--clip-model` | CLIP MindIR | 必填 |
| `--model-dir` | diffusers 权重目录（取 tokenizer + scheduler） | `./FLUX.1-dev` |
| `--prompt` | 文本提示词 | 必填 |
| `--seed` | 随机种子（初始噪声） | `0` |
| `--steps` | 去噪步数 | `28` |
| `--guidance` | guidance scale（dev=3.5） | `3.5` |
| `--t5-seq-len` | T5 序列长度（须与导出一致） | `256` |
| `--height/--width` | 图像尺寸（须与导出一致） | `1024` |
| `--transformer-device` | transformer/VAE 所在昇腾卡 | `0` |
| `--text-device` | T5/CLIP 所在昇腾卡 | `1` |
| `--output` | 输出 PNG 路径 | `./flux1_output.png` |

### 流程

1. CLIP+T5（dev1）编码 prompt → `encoder_hidden_states` / `pooled`。
2. numpy 固定 seed 生成噪声 → pack → 构造 `img_ids`/`txt_ids`。
3. 按 `FlowMatchEulerDiscreteScheduler`（CPU，与原算法一致）逐 `timestep`：transformer(dev0) → noise_pred → `scheduler.step` 更新 latents（28 步）。
4. unpack latents → `/scaling + shift` → VAE(dev0) → RGB 图像 → 保存 PNG。

---

## 7. 性能数据

> 以下为 Atlas 300I Duo（310P3）实测数据（1024×1024，28 步，fp16，单图）。数值以推理脚本端到端打印为准。

| 指标 | 300I Duo 耗时 |
|---|---|
| 文本编码 (CLIP+T5, dev1) | _（待运行后填入）_ |
| Transformer 总计 (28 步) | _（待运行后填入）_ |
| Transformer 单步平均 | _（待运行后填入）_ |
| VAE 解码 (dev0) | _（待运行后填入）_ |
| 端到端 | _（待运行后填入）_ |

---

## 8. 精度对齐

提供 `align_flux1_dev.py`，在固定 prompt/seed/latent 下对每个组件做 HF(CPU fp16) vs MindIR(Ascend) 数值比对（每组件 1 次 forward，快速且严谨）：

```bash
python align_flux1_dev.py \
  --model-dir ./FLUX.1-dev \
  --mindir-dir ./flux1_dev_onnx \
  --prompt "A cat holding a sign that says hello world" --seed 0
```

输出 CLIP pooled / T5 last_hidden_state / transformer noise_pred / VAE image 的 `max_abs`、`mean_abs`、`max_rel`。fp16 下 noise_pred/image 的 `max_abs` 通常 < ~1e-2。

如需整图端到端比对，加 `--full-baseline`（在 CPU 上跑一次完整 HF `FluxPipeline`，较慢），生成 `flux1_baseline.png`，与 `flux1_output.png` 目视/PSNR 对比。

常见误差源与对策：

- **fp16 溢出**：transformer/VAE 已默认 `force_fp16`；若 noise_pred 偏差大，可对 transformer config 改 `force_fp32` 重新转换。
- **T5 序列长度不一致**：推理 `--t5-seq-len` 必须与导出一致（256）。
- **图像尺寸不一致**：`--height/--width` 必须与导出 `--resolution` 一致。
- **初始噪声不一致**：精度比对时两端 `--seed` 必须相同。

---

## 9. 常见问题

### 1) 转换时报 `ge.proto.ModelDef exceeded maximum protobuf size`

transformer 权重 ~24GB，`ascend_oriented` 转换会把权重外置到 `*_variables/`，日志打印此信息**不影响**产物。

### 2) transformer 图编译耗时长

12B 模型 57 层在 310P3 上的图编译可能需要数十分钟，属正常现象；固定 shape 已是最快路径。

### 3) `Only support CustomAscend, but got ...`

MindIR 必须用 `--optimize=ascend_oriented` 转换（保留 Custom 算子映射）。请勿用 `--optimize=general`。

### 4) 单卡 OOM

transformer(24GB) + 激活在单卡 ~44GB 内可运行；若同时加载 T5 导致 OOM，确保 T5/CLIP 在 `--text-device` 另一张卡（默认 dev1）。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [FLUX.1-dev 模型页（ModelScope）](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)
- [Black Forest Labs](https://blackforestlabs.ai/)
- [Diffusers FLUX 文档](https://huggingface.co/docs/diffusers/api/pipelines/flux)

---

## 11. 许可证

FLUX.1-dev 遵循 [HuggingFace Community License（非商用）](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)。本教程遵循相应依赖的许可证要求。
