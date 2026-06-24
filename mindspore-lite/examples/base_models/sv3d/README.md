> ⚠️ **待验证（SV3D 非标准 diffusers 模型）**：stabilityai/sv3d 不在 diffusers 的 `stable_video_diffusion` 管线中，其加载与条件化（SV3D 的视角/多视角/4D 条件）需使用上游 `stabilityai/generative-models` 仓库。本目录代码以 SVD-XT 示例为结构模板（UNetSpatioTemporalConditionModel + AutoencoderKLTemporalDecoder + CLIP 图像编码器 + EulerDiscreteScheduler）克隆适配，**导出/推理/对齐脚本中的视角与运动条件（azimuth/elevation/motion）相关逻辑需在 Phase B 验证时依据上游仓库源码确认与修正**。导出切分、MSLite 推理骨架、配置与计时约定可直接复用。

# SV3D ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `stabilityai/sv3d`（SV3D，图生视频扩散模型）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上完成端到端图生视频推理与精度对齐。

SV3D 由三部分组成：CLIP-ViT-H 图像编码器（`image_encoder`，projection_dim=1024，输出 `image_embeds` 作为 UNet 交叉注意力条件）、时空 UNet（`UNetSpatioTemporalConditionModel`，去噪器，`in_channels=8`、`cross_attention_dim=1024`）、4 通道时序 VAE 解码器（`AutoencoderKLTemporalDecoder`，含沿时间轴的 Conv3d）。本教程将这三部分**全部导出为 MindIR** 并在昇腾上推理（CLIP 在 dev1，UNet+VAE 在 dev0）。VAE **编码器**仅在 CPU torch 上运行一次（对条件帧编码得到 `image_latents`），不导出；`EulerDiscreteScheduler` 也在 CPU 上运行。

固定配置：576x1024 分辨率（16:9），25 帧（SV3D 默认 `unet.config.num_frames`），latent 尺寸 72x128。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux（昇腾环境，MindSpore Lite 2.9.0 + CANN 8.5.0 + Ascend 驱动）
- Atlas 300I Duo（310P3，单卡 ~44GB）

### 依赖版本（建议）

| 软件包            | 版本   |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.9.0（CPU 即可，仅用于导出/VAE 编码/对齐/scheduler） |
| diffusers      | 0.38.0（原生支持 SVD pipeline 与 UNetSpatioTemporalConditionModel） |
| transformers   | 5.9.0 |
| onnx           | 1.19.1 |
| imageio        | 2.31（mp4 导出，可选；失败自动回退 PNG） |
| imageio-ffmpeg | 0.4.9（imageio mp4 后端） |
| mindspore-lite | 2.9.0 |
| CANN           | 8.5.0 |

### 安装命令

```bash
pip install torch==2.9.0 diffusers==0.38.0 transformers==5.9.0 onnx==1.19.1 \
  imageio imageio-ffmpeg onnxruntime
```

### 初始化环境

```bash
source /home/yf/env.sh   # CANN / mindspore-lite / converter_lite
```

---

## 2. 模型下载

从 ModelScope / HuggingFace 下载 diffusers 格式权重（image_encoder ~3.5GB + unet ~6GB + vae ~0.3GB，共 ~10GB）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('AI-ModelScope/sv3d', \
    cache_dir='/home/yf/modelscope_cache'))"
```

将其软链到 `./sv3d`：

```bash
ln -sfn /home/yf/modelscope_cache/AI-ModelScope/sv3d \
  ./sv3d
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本将 SV3D 拆为三个 ONNX 子图：

1. **image_encoder**（`svd_image_encoder.onnx`）：CLIP-ViT-H（带 projection），pixel_values [1,3,224,224] → image_embeds [1,1024]。该 embed 经 pipeline `unsqueeze(1)` 成为 UNet 交叉注意力的 `encoder_hidden_states` [B,1,1024]（seq_len=1）。
2. **unet**（`svd_unet.onnx`）：sample [2,25,8,72,128]（8 = 4 噪声 + 4 VAE image_latents 通道拼接，2 = CFG 的 uncond+cond） + timestep [2] + encoder_hidden_states [2,1,1024]（CLIP image_embeds） + added_time_ids [2,3]（fps-1, motion_bucket_id, noise_aug_strength） → noise_pred [2,25,4,72,128]。自+交叉注意力替换为 CANN `PromptFlashAttention` Custom 算子。
3. **vae_decoder**（`svd_vae_decoder.onnx`）：时序 VAE 解码器（单帧 chunk），4 通道 latent [1,4,72,128] → RGB 帧 [1,3,576,1024]。

### 条件机制与 fps 处理

- **条件图像编码**：输入图像先经 `video_processor.preprocess`（resize 到 576x1024、归一化 [-1,1]）+ noise_aug_strength 噪声，再在 CPU 上用 `AutoencoderKLTemporalDecoder.encode` 编码得到 `image_latents` [1,4,72,128]（取 `latent_dist.mode()`）。该 latent 在 UNet 前向时沿帧维 repeat 成 [1,25,4,72,128]，并在 dim=2（通道）与噪声 latent 拼接成 8 通道输入。
- **CLIP 条件**：同一输入图像经 antialias-resize 到 224x224 + CLIP normalize，送 `image_encoder` 得 `image_embeds` [1,1024] → unsqueeze → [1,1,1024]，作为 UNet 交叉注意力的 K/V 源（seq_len=1）。
- **fps 处理**：SVD 训练时按 `fps-1` 微调（参见 diffusers 源码注释），故 pipeline 在构造 `added_time_ids` 前先 `fps = fps - 1`（默认 `fps=7` → 用 6）。`added_time_ids = [fps-1, motion_bucket_id, noise_aug_strength]`，经 `add_time_proj`（256 维正弦编码 ×3 = 768）+ `add_embedding` 线性投影，与时间嵌入相加。
- **per-frame CFG**：SVD 的 classifier-free guidance 在帧维线性变化 `linspace(min_gs, max_gs, num_frames)`（默认 1.0→3.0），UNet 单次前向同时算 uncond+cond（batch=2），guidance_scale 按 [1,F,1,1,1] 广播。

### 自定义算子策略

SVD UNet 的 `AttnProcessor2_0` 将 q/k/v 投影后 reshape 为 `(batch, num_heads, seq, head_dim)`（BNSD 布局），再调用 `torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)`。脚本 monkeypatch 该函数，将 SDPA 调用替换为 CANN `PromptFlashAttention` Custom 节点（q/k/v **已是 BNSD，无需转置**、`sparse_mode=0`、不传 atten_mask，对时空自注意力和单 token 交叉注意力均成立——两者都是全双向无 mask）。其余（q/k/v 投影、group-norm、resnet、时空 transformer block 的 time embedding 注入、残差、输出投影）均走标准 diffusers 算子 trace。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/sv3d

python export_sv3d_onnx.py \
  --model-id ./sv3d \
  --output-dir ./sv3d_onnx \
  --resolution 576 1024 \
  --num-frames 25 \
  --dtype fp32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 本地 diffusers 权重目录 | `./sv3d` |
| `--output-dir` | ONNX 输出目录 | `./sv3d_onnx` |
| `--resolution` | 输出帧分辨率 H W | `576 1024` |
| `--num-frames` | 视频帧数（须与导出/推理一致） | `25` |
| `--dtype` | 导出精度（fp32 推荐，converter 转 fp16） | `fp32` |
| `--no-custom-op` | 不把注意力替换为 Custom 算子 | `False` |
| `--components` | 导出子集（image_encoder,unet,vae_decoder） | 全部 |

### 模型架构参数

| 参数 | 值 |
|------|------|
| in_channels (UNet sample) | 8（4 噪声 + 4 VAE image_latent 通道拼接） |
| out_channels (latent) | 4 |
| num_frames | 25（SV3D 默认） |
| cross_attention_dim | 1024（CLIP image_embeds） |
| block_out_channels | (320, 640, 1280, 1280) |
| num_attention_heads | (5, 10, 20, 20)（head_dim=64） |
| addition_time_embed_dim | 256 |
| projection_class_embeddings_input_dim | 768（= 3 × 256，3 个 added_time_ids） |
| image_encoder projection_dim | 1024（CLIP-ViT-H） |
| VAE latent channels | 4，scaling_factor 0.18215 |
| added_time_ids | 3：(fps-1, motion_bucket_id, noise_aug_strength) |
| scheduler | EulerDiscreteScheduler（CPU），prediction_type=epsilon |

---

## 4. ONNX 模型结构说明

unet ONNX 使用 MindSpore Lite 自定义算子（`PromptFlashAttention`），**不支持直接用 ONNX Runtime 推理**，需通过 `converter_lite` 转 MindIR 后运行。image_encoder / vae_decoder 为标准算子图。

### 模型中包含的自定义算子（unet）

| 算子 | 数量 | 说明 |
|------|------|------|
| PromptFlashAttention | ~140（含 down/mid/up 所有时空 transformer block 的自注意力 + 交叉注意力） | 全双向注意力，无 mask |

> 每个 transformer block 含 1 个自注意力（沿帧 token）+ 1 个交叉注意力（对 1 个 CLIP image token），数量取决于 `transformer_layers_per_block`。

---

## 5. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/sv3d

CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# image_encoder (CLIP-ViT-H)
$CONV --fmk=ONNX --modelFile=./sv3d_onnx/svd_image_encoder.onnx \
  --outputFile=./sv3d_onnx/svd_image_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/svd_image_encoder.config

# unet
$CONV --fmk=ONNX --modelFile=./sv3d_onnx/svd_unet.onnx \
  --outputFile=./sv3d_onnx/svd_unet \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/svd_unet.config

# vae_decoder
$CONV --fmk=ONNX --modelFile=./sv3d_onnx/svd_vae_decoder.onnx \
  --outputFile=./sv3d_onnx/svd_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/svd_vae_decoder.config
```

### config 说明

- `svd_unet.config`：固定 [2,25,8,72,128] latent + CLIP seq=1 + added_time_ids[2,3]，`force_fp16`，`plugin_custom_ops=All`。batch=2 对应 CFG 单次前向。
- `svd_image_encoder.config`：固定 pixel_values [1,3,224,224]，`force_fp16`。
- `svd_vae_decoder.config`：固定单帧 latent [1,4,72,128]，`force_fp16`（时序 Conv3d 以 num_frames=1 导出，host 逐帧解码）。

> 转换日志中可能出现 `protobuf size` / `ge.proto.ModelDef exceeded maximum protobuf size` 等 warning（unet 权重外置化），**不影响最终产物**（产出 `*_graph.mindir` + `*_variables/`），可忽略。

---

## 6. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/sv3d

python infer_sv3d_mslite.py \
  --mindir-dir   ./sv3d_onnx \
  --model-dir    ./sv3d \
  --image        ./conditioning.jpg \
  --seed 0 --steps 25 --fps 7 \
  --motion-bucket-id 127 --noise-aug-strength 0.02 \
  --min-guidance 1.0 --max-guidance 3.0 \
  --image-device 1 --unet-device 0 --vae-device 0 \
  --output ./svd_output.mp4
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mindir-dir` | 含 3 个 `*_graph.mindir` 的目录 | 必填 |
| `--model-dir` | diffusers 权重目录（取 feature_extractor/scheduler/vae config + CPU VAE） | 必填 |
| `--image` | 条件图像路径 | 必填 |
| `--seed` | 随机种子（初始噪声 + noise_aug） | `0` |
| `--steps` | 去噪步数 | `25` |
| `--fps` | 帧率（训练时按 fps-1 条件，内部自动减 1） | `7` |
| `--motion-bucket-id` | 运动量条件 | `127` |
| `--noise-aug-strength` | 条件图噪声增强 | `0.02` |
| `--min-guidance` / `--max-guidance` | per-frame CFG 范围 | `1.0` / `3.0` |
| `--height/--width` | 帧尺寸（须与导出一致） | `576` / `1024` |
| `--num-frames` | 帧数（须与导出一致） | `25` |
| `--image-device` | CLIP 图像编码器所在昇腾卡 | `1` |
| `--unet-device` | UNet 所在昇腾卡 | `0` |
| `--vae-device` | VAE 所在昇腾卡 | `0` |
| `--output` | 输出 mp4（失败回退 PNG 序列） | `./svd_output.mp4` |

### 流程

1. **CLIP 编码（dev1）**：条件图像 resize 到 576x1024 → antialias-resize 到 224x224 → CLIP normalize → `image_encoder` → `image_embeds` [1,1024] → CFG 扩展 [2,1,1024]。
2. **VAE 编码（CPU torch，一次）**：同一图像 resize 到 576x1024、归一化 [-1,1]、加 noise_aug 噪声 → `AutoencoderKLTemporalDecoder.encode` → `image_latents` [1,4,72,128] → repeat 成 [1,25,4,72,128]（条件帧，与噪声在通道维拼接）。
3. **噪声初始化**：numpy 固定 seed 生成 [1,25,4,72,128]，乘 `scheduler.init_noise_sigma`。
4. **去噪循环（25 步，CPU scheduler + Ascend UNet）**：每步 `scale_model_input` → sample 与 image_latents 通道拼接成 8 通道 [2,25,8,72,128]（CFG batch=2）→ UNet 单次前向 → `noise = uncond + gs*(cond-uncond)`（gs 为 per-frame `linspace(min_gs,max_gs,25)`）→ `scheduler.step` 更新 latents。
5. **解码（dev0，逐帧）**：latents `/scaling_factor` → 每帧单独送 VAE 单帧图（num_frames=1）→ 拼接 25 帧 → `(x/2+0.5).clip*255` → uint8 → imageio 导出 mp4（失败回退 PNG）。

---

## 7. 性能数据

> 以下为 Atlas 300I Duo（310P3）实测数据（576×1024，25 帧，25 步，fp16，单视频，CFG x2 + per-frame guidance）。数值以推理脚本端到端打印为准。

| 指标 | 300I Duo 耗时 |
|---|---|
| 图像编码 (CLIP-ViT-H, dev1) | _（待运行后填入）_ |
| UNet 总计 (25 步, CFG x2) | _（待运行后填入）_ |
| UNet 单步平均 | _（待运行后填入）_ |
| VAE 解码 (25 帧, dev0) | _（待运行后填入）_ |
| 端到端 | _（待运行后填入）_ |

---

## 8. 精度对齐

提供 `align_sv3d.py`，在固定条件图像/seed/latent 下对 HF(CPU fp32) 与 MindIR(Ascend fp16) 做逐帧端到端比对：用 seeded torch generator 生成共享初始 latents [1,25,4,72,128]，分别喂给 HF `StableVideoDiffusionPipeline`（CPU）与 MSLite 推理，比较输出视频的逐帧与聚合 `max_abs`、`mean_abs`、`PSNR`。

```bash
python align_sv3d.py \
  --mindir-dir ./sv3d_onnx \
  --model-dir  ./sv3d \
  --image      ./conditioning.jpg \
  --seed 0 --steps 25 --fps 7 \
  --motion-bucket-id 127 --noise-aug-strength 0.02 \
  --min-guidance 1.0 --max-guidance 3.0
```

> HF CPU baseline 对 576×1024 / 25 帧 / 25 步非常慢（UNet 需反复前向 25 帧 × 25 步 × 2 CFG）；如需快速验证可减小 `--steps`（两端始终使用相同步数与设置）。fp16 下视频帧 `PSNR` 通常 ≥ 28dB，`max_abs` 通常 < ~0.15（[0,1] 归一化范围）。

常见误差源与对策：

- **fps 减 1**：SVD 训练时按 `fps-1` 条件，构造 `added_time_ids` 前必须 `fps = fps - 1`（默认 7→6）；推理脚本已自动处理，对齐脚本两端一致。
- **noise_aug 噪声不一致**：条件帧 VAE 编码前会加 `noise_aug_strength * randn`；两端须用相同 seed 才能严格对齐，否则条件 latent 有微小差异（影响首帧）。
- **fp16 溢出**：UNet/VAE 已默认 `force_fp16`；若偏差大，可对 unet config 改 `force_fp32` 重新转换。
- **per-frame guidance 不一致**：SVD 的 guidance 在帧维线性插值（`linspace(min_gs, max_gs, num_frames)`），而非全局标量；推理脚本已实现该广播，须确保对齐两端 min/max 一致。
- **初始噪声不一致**：精度比对时两端 `--seed` 必须相同；align 脚本已用共享 latents npy 保证一致。

---

## 9. 常见问题

### 1) 转换时报 `ge.proto.ModelDef exceeded maximum protobuf size`

UNet 权重较大，`ascend_oriented` 转换会把权重外置到 `*_variables/`，日志打印此信息**不影响**产物。

### 2) UNet 图编译耗时长

SVD UNet 含 4 个 down/mid/up 时空 transformer block，固定 [2,25,8,72,128] shape 在 310P3 上的图编译可能需要数十分钟，属正常现象；固定 shape 已是最快路径。

### 3) `Only support CustomAscend, but got ...`

MindIR 必须用 `--optimize=ascend_oriented` 转换（保留 Custom 算子映射）。请勿用 `--optimize=general`。

### 4) 单卡 OOM

SVD UNet（batch=2 × 25 帧 × 8 通道）激活较大；若同时加载 CLIP 导致 OOM，确保 CLIP 在 `--image-device` 另一张卡（默认 dev1）。VAE 逐帧解码（num_frames=1）已最小化显存。

### 5) sample 通道数不对（非 8）

SVD UNet 的 `in_channels=8` = 4 噪声 + 4 VAE image_latent。pipeline 在送入 UNet 前将 `image_latents`（repeat 到 25 帧）与噪声沿**通道维（dim=2）**拼接。若未拼接或通道数错，UNet 报维度错误。推理脚本已在 `_denoise` 内完成拼接。

### 6) encoder_hidden_states seq_len 不是 1

SVD 的交叉注意力 K/V 来自 CLIP `image_embeds`（单 pooled token，[B,1,1024]），不是 patch token 序列。若误传 `last_hidden_state`（如 [B,257,1024]）会导致 cross_attention_dim 不匹配。

### 7) mp4 导出失败（缺 ffmpeg）

`imageio.mimsave` 依赖 `imageio-ffmpeg`；若未安装或无 `libx264`，脚本自动回退为按帧保存 PNG（`<output>_frameNNN.png`）。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [SV3D 模型页（HuggingFace）](https://huggingface.co/stabilityai/sv3d)
- [SVD 论文（Stable Video Diffusion）](https://arxiv.org/abs/2311.15127)
- [Diffusers SVD 文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_video_diffusion)

---

## 11. 许可证

SV3D 遵循 [Stability AI Community License](https://huggingface.co/stabilityai/sv3d/blob/main/LICENSE.md)。本教程遵循相应依赖的许可证要求。
