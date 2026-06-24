# Wan2.1-I2V-14B-480P ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) 图像生成视频（Image-to-Video）模型按网络结构拆分导出为 ONNX，使用 ONNX Runtime 验证子模型推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 **Ascend Atlas 300I Duo** 上完成端到端推理与测速。

Wan2.1-I2V-14B 在 Wan2.1-T2V 的纯文本条件之外，额外引入 **CLIP 图像条件**（image embedding），并沿用 Wan2.1 的**标量时间步**调度（注意：与 Wan2.2-TI2V 的 `expand_timesteps` 每 token 时间步不同）。首帧图像条件通过**通道拼接**注入 DiT：推理时 transformer 输入为 36 通道张量 `cat([latents(16), mask_lat_size(4), latent_condition(16)], dim=1)`，其中 `latent_condition` 是条件图经 VAE 编码（CPU，argmax 模式）并按 `latents_mean/latents_std` 反归一化后的潜变量，`mask_lat_size` 是首帧为 1、其余帧为 0、按 `vae_scale_factor_temporal=4` 广播为 4 通道的逐潜帧掩码。本目录基于已验证的 `wan2_2_ti2v_5b`（同为 I2V + CLIP 图像条件）与 `wan2_1_t2v_1_3b`（标量时间步）模板改写适配 14B 大模型。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0+cpu（导出与 VAE 编码仅在 CPU） |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 1.26.4 |
| transformers | 5.9.0 |
| diffusers | 0.38.0 |
| mindspore-lite | 2.10.0 |
| CANN | 8.5.0 |

```bash
source /home/yf/env.sh   # CANN + mindspore-lite runtime + converter_lite
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 transformers diffusers mindspore-lite imageio pillow
```

### 获取模型权重

```bash
# 从 ModelScope 下载，缓存到本地后做符号链接
python -c "from modelscope import snapshot_download as s; print(s('Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', cache_dir='/home/yf/modelscope_cache'))"
ln -sfn /home/yf/modelscope_cache/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers Wan2.1-I2V-14B-480P-Diffusers
```

`MODEL_DIR`（权重目录）需包含 `text_encoder/`、`image_encoder/`、`transformer/`、`vae/`、`tokenizer/`、`feature_extractor/`、`scheduler/`。

### 大模型显存说明（14B）

14B 模型（fp16 约 28GB / fp32 约 56GB）单卡放不下全量 fp32，部署策略：

- **导出在 CPU 上以 float32 进行**（主机 ~190GB 空闲内存足够），转换后 MindIR 以 `force_fp16` 推理。
- **推理采用组件级分芯**（非张量并行）：文本编码器 + CLIP 图像编码器 → dev1，transformer（14B，单独占一芯）→ dev0，VAE 解码器 → dev2。详见第 5 节。
- 所有 config 强制 `force_fp16`（**不要**用 `force_fp32`——8B + fp32 在 44GB 的 310P3 上 OOM）。

---

## 2. 模型导出 ONNX

按结构拆分为四个固定 shape 子模型：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `wan_text_encoder` (UMT5-XXL) | input_ids[1,512], attention_mask[1,512] | last_hidden_state[1,512,4096] |
| `wan_clip_image_encoder` (CLIP ViT-H/14) | pixel_values[1,3,224,224] | image_embeds[1,257,1280] |
| `wan_transformer` (DiT 14B, in_channels=36) | hidden_states[1,**36**,21,60,104], timestep[**1**], encoder_hidden_states[1,512,4096], encoder_hidden_states_image[1,257,1280] | noise_pred[1,16,21,60,104] |
| `wan_vae_decoder` (3D VAE) | latents[1,16,21,60,104] | video[1,3,81,480,832] |

固定配置：480×832、81 帧（21 个 latent 帧）。`hidden_states` 为 **36 通道**（`latents(16) + mask_lat_size(4) + latent_condition(16)`，通道拼接，由推理脚本在 numpy 侧组装），`timestep` 为标量 `[1]`（Wan2.1 调度；注意与 Wan2.2-TI2V 的 per-token `[1,32760]` 不同）。若需其它分辨率/帧数，需重新导出并重新转换。

### 导出命令

```bash
cd mindspore-lite/examples/base_models/wan2_1_i2v_14b_480p

python export_wan2_1_i2v_14b_480p_onnx.py \
  --model-dir ./Wan2.1-I2V-14B-480P-Diffusers \
  --output-dir ./wan2_1_i2v_14b_480p_onnx \
  --height 480 --width 832 --num-frames 81 --max-seq-len 512 \
  --dtype float32
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-dir` | 权重目录 | 必填 |
| `--output-dir` | ONNX 输出目录 | `./wan2_1_i2v_14b_480p_onnx` |
| `--parts` | 导出哪些子模型 | `text,clip,transformer,vae` |
| `--height/--width` | 视频分辨率（16 的倍数） | `480/832` |
| `--num-frames` | 视频帧数（4k+1） | `81` |
| `--max-seq-len` | UMT5 最大序列长度 | `512` |
| `--dtype` | 导出精度（建议 float32，便于转换） | `float32` |
| `--no-custom-op` | 不把注意力替换为 CANN Custom 算子 | `False` |

### 产出文件

```text
./wan2_1_i2v_14b_480p_onnx/
├── wan_text_encoder.onnx (+ 外部权重 .data)
├── wan_clip_image_encoder.onnx (+ 外部权重 .data)
├── wan_transformer.onnx (+ 外部权重 .data)
└── wan_vae_decoder.onnx (+ 外部权重 .data)
```

### 导出注意事项

- **注意力替换为 CANN `PromptFlashAttention` Custom 算子**：Wan 的时空自注意力、文本交叉注意力、图像交叉注意力均为全双向（无 mask），脚本 monkeypatch diffusers 的注意力派发（`transformer_wan.dispatch_attention_fn`），将所有注意力导出为 `PromptFlashAttention` Custom 节点（BNSD、`sparse_mode=0`），从而避免在 ~33k token 全注意力上实体化 O(seq²) 得分矩阵。其余（q/k/v 投影、added_kv 投影、RMSNorm、RoPE、3D patchify）走标准算子。
- **Wan2.1 标量时间步**：transformer 导出时 `timestep` 固定为标量 `[1]`（Wan2.1 调度）；推理脚本每步传入当前标量时间步 `np.array([float(t)])`，与 diffusers `pipeline_wan_i2v.py` 的非 `expand_timesteps` 分支一致。
- **36 通道条件拼接**：transformer 导出 `hidden_states` 固定为 `[1, 36, 21, 60, 104]`，对应 `in_channels=36`（`16 latents + 4 mask + 16 condition`）；推理脚本每步用 `np.concatenate([latents, mask_lat_size, latent_condition], axis=1)` 在 numpy 侧重建该 36 通道输入。
- **CLIP 图像编码器**：导出 `CLIPVisionModel`（ViT-H/14，`image_encoder/` 子目录），取倒数第二层 hidden state（`hidden_states[-2]`），形状 `[1, 257, 1280]`（256 patch + 1 CLS）。
- 导出走 **legacy 导出器**（`torch.onnx.utils.export`），`do_constant_folding=False`（长序列图常量折叠会在 CPU 上 OOM）。
- 模型以 **float32** 加载导出，避免 ONNX 全图 FLOAT16 导致转换器报错。

---

## 3. ONNX 推理

> 说明：`wan_transformer.onnx` 含 `Custom` 节点（PromptFlashAttention），**ONNX Runtime 无法直接执行**；文本编码器、CLIP 图像编码器与 VAE 为标准算子图，可用 ONNX Runtime 验证。transformer 的精度基准以 HF diffusers pipeline 为准（见第 7 节）。

```bash
# （可选）用 ONNX Runtime 验证文本编码器 / CLIP / VAE
python - <<'PY'
import numpy as np, onnxruntime as ort
m = ort.InferenceSession("wan2_1_i2v_14b_480p_onnx/wan_clip_image_encoder.onnx",
                         providers=["CPUExecutionProvider"])
px = np.zeros((1, 3, 224, 224), np.float32)
out = m.run(None, {"pixel_values": px})
print("clip_image_encoder out shape:", out[0].shape)  # (1, 257, 1280)
PY
```

执行日志（待运行后填入）：

```log
（待运行后填入实际输出）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

`converter_lite` 为 MindSpore Lite 提供的离线转换工具。

```bash
CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# 文本编码器
$CONV --fmk=ONNX --modelFile=./wan2_1_i2v_14b_480p_onnx/wan_text_encoder.onnx \
  --outputFile=./wan2_1_i2v_14b_480p_onnx/wan_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_text_encoder.config

# CLIP 图像编码器
$CONV --fmk=ONNX --modelFile=./wan2_1_i2v_14b_480p_onnx/wan_clip_image_encoder.onnx \
  --outputFile=./wan2_1_i2v_14b_480p_onnx/wan_clip_image_encoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_clip_image_encoder.config

# transformer
$CONV --fmk=ONNX --modelFile=./wan2_1_i2v_14b_480p_onnx/wan_transformer.onnx \
  --outputFile=./wan2_1_i2v_14b_480p_onnx/wan_transformer \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_transformer.config

# VAE 解码器
$CONV --fmk=ONNX --modelFile=./wan2_1_i2v_14b_480p_onnx/wan_vae_decoder.onnx \
  --outputFile=./wan2_1_i2v_14b_480p_onnx/wan_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_vae_decoder.config
```

### 配置文件

`configs/wan_transformer.config`（注意 Wan2.1 I2V 的 36 通道 `hidden_states` 与标量 `timestep`）：

```ini
[acl_build_options]
input_format="ND"
input_shape="hidden_states:1,36,21,60,104;timestep:1;encoder_hidden_states:1,512,4096;encoder_hidden_states_image:1,257,1280"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

### 产出说明

```text
./wan2_1_i2v_14b_480p_onnx/
├── wan_text_encoder_graph.mindir        + wan_text_encoder_variables/
├── wan_clip_image_encoder_graph.mindir  + wan_clip_image_encoder_variables/
├── wan_transformer_graph.mindir         + wan_transformer_variables/
└── wan_vae_decoder_graph.mindir         + wan_vae_decoder_variables/
```

执行日志（待运行后填入）：

```log
CONVERT RESULT SUCCESS:0   （待运行后填入完整日志）
```

---

## 5. MindSpore Lite 推理

```bash
python infer_wan2_1_i2v_14b_480p_mslite.py \
  --mindir-dir ./wan2_1_i2v_14b_480p_onnx \
  --model-dir ./Wan2.1-I2V-14B-480P-Diffusers \
  --image ./condition.jpg \
  --prompt "A cat walking on a beach, cinematic, 4k." \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 \
  --output wan_output.mp4
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir-dir` | MindIR 目录（含 4 个 `*_graph.mindir`） | 必填 |
| `--model-dir` | 权重目录（tokenizer + feature_extractor + scheduler + vae config） | 必填 |
| `--image` | 条件图（首帧，PIL 可读格式） | 必填 |
| `--prompt/--negative-prompt` | 文本提示 | 见默认 |
| `--height/--width/--num-frames` | 必须与导出/转换一致 | `480/832/81` |
| `--num-inference-steps` | 去噪步数 | `50` |
| `--guidance-scale` | CFG 强度 | `5.0` |
| `--text-device/--clip-device/--transformer-device/--vae-device` | 组件分芯 | `1/1/0/2` |
| `--latents-npy` | 预生成噪声（精度对齐用） | 无 |

说明（固定 shape 约束）：`ascend_oriented` 转换按固定 shape 编译，推理侧 `--height/--width/--num-frames` 必须与导出一致；变更需重新导出+转换。

### 组件级分芯（14B 专用）

14B transformer（fp16 约 28GB）需要单独占用一颗 44GB 的 310P3 芯片，因此采用**三芯组件级分芯**（非张量并行）：

| 组件 | 设备 | 说明 |
| --- | --- | --- |
| 文本编码器（UMT5）+ CLIP 图像编码器 | dev1 | 两个小编码器共用一芯 |
| transformer（14B DiT） | dev0 | 单独占一芯（~28GB fp16） |
| VAE 解码器 | dev2 | 单独占一芯 |

VAE 编码器（仅用于编码条件图，运行一次）在 CPU 上用 torch 执行，不进入 Ascend MindIR 集合。

执行日志（待运行后填入，含性能数据）：

```log
（待运行后填入实际输出：生成视频路径 + 各阶段耗时）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3，每芯 ~44GB），CANN 8.5.0，MindSpore Lite 2.10.0。

> 性能数据以推理脚本端到端打印为准；下表为**待实测填入**（运行 `infer_wan2_1_i2v_14b_480p_mslite.py` 后回填真实数值，不使用估算或占位假数据）。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 文本编码 (UMT5, dev1) | _待运行填入_ |
| CLIP 图像编码 (dev1) | _待运行填入_ |
| VAE 编码条件图 (CPU, 一次) | _待运行填入_ |
| Transformer 总计 (50 步 × CFG×2) | _待运行填入_ |
| Transformer 单步平均 | _待运行填入_ |
| VAE 解码 (dev2) | _待运行填入_ |
| **端到端** | **_待运行填入_** |

---

## 7. 精度对齐

端到端对比 HF diffusers `WanImageToVideoPipeline`（CPU float32 基线，`expand_timesteps=False` 即 Wan2.1 标量时间步调度）与 MSLite（Ascend）的生成视频：使用相同 prompt、相同条件图、相同初始噪声（seed 固定的 torch 生成器，存 npy 后两路共用）、相同调度器参数，逐帧比较 max/mean abs 误差与 PSNR。

```bash
# 注意：HF CPU 基线较慢，建议先用较少帧/步数跑通端到端对齐
python align_wan2_1_i2v_14b_480p.py \
  --mindir-dir ./wan2_1_i2v_14b_480p_onnx \
  --model-dir ./Wan2.1-I2V-14B-480P-Diffusers \
  --image ./condition.jpg \
  --num-frames 21 --num-inference-steps 10
```

执行日志（待运行后填入）：

```log
（待运行后填入：max_abs / mean_abs / PSNR）
```

---

## 8. 常见问题

1. 现象：导出 transformer 时 CPU 内存暴涨/被 OOM 杀死。
   - 原因：legacy 导出器默认常量折叠，长序列（~33k token）图折叠常量占满内存。
   - 解决方案：`do_constant_folding=False`（本脚本已设置）；注意力已替换为 Custom 算子，fallback 仅为保形 stub。
2. 现象：ONNX Runtime 无法运行 `wan_transformer.onnx`。
   - 原因：该 ONNX 含 `PromptFlashAttention` Custom 节点。
   - 解决方案：transformer 经 converter 转 MindIR 后在 Ascend 运行；精度基准用 HF pipeline。
3. 现象：converter 报 `do not support data_type: 10`。
   - 原因：模型以 fp16 加载导出导致全图 FLOAT16。
   - 解决方案：以 `--dtype float32` 导出。
4. 现象：推理 shape 不匹配。
   - 原因：`--height/--width/--num-frames` 与导出/转换不一致。
   - 解决方案：三者必须与导出一致；变更需重新导出+转换。
5. 现象：transformer 输入 `hidden_states` 通道数对不上。
   - 原因：Wan2.1 I2V 的 transformer 输入是 36 通道（`latents(16)+mask(4)+condition(16)` 通道拼接），不是纯 16 通道潜变量。
   - 解决方案：确认导出/转换/推理三处的 `hidden_states` 形状一致（`[1,36,21,60,104]`）；推理脚本每步用 `np.concatenate([latents, mask_lat_size, latent_condition], axis=1)` 重建。
6. 现象：transformer 输入 `timestep` 维度对不上。
   - 原因：Wan2.1 I2V 使用标量时间步 `[1]`，不是 Wan2.2 TI2V 的 per-token `[1,32760]`。
   - 解决方案：确认导出/转换/推理三处的 `timestep` 形状一致（`[1]` 标量）。
7. 现象：推理 OOM（14B 在单芯上显存不足）。
   - 原因：14B fp16 约 28GB，与其它组件挤在同一 44GB 芯片上可能 OOM。
   - 解决方案：使用组件级分芯（默认 `--transformer-device 0 --vae-device 2 --text-device 1 --clip-device 1`），让 14B transformer 独占 dev0；切勿用 `force_fp32`。
8. 现象：缺少 `feature_extractor/` 子目录。
   - 原因：CLIP 图像预处理依赖 `CLIPImageProcessor` 配置。
   - 解决方案：确认 `MODEL_DIR` 含 `feature_extractor/`（与 Wan2.2-TI2V 同构）；缺则从 `openai/clip-vit-large-patch14` 复用。

---

## 9. 参考资源与许可证

- 上游模型：<https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- 同构参考（本目录改写模板）：`../wan2_2_ti2v_5b/`（Wan2.2-TI2V-5B，同为 I2V + CLIP 图像条件，per-token 时间步）、`../wan2_1_t2v_1_3b/`（Wan2.1-T2V-1.3B，标量时间步）
- 本目录脚本遵循 MindSpore Lite 仓库许可证；上游模型权重许可证以其仓库为准（Apache-2.0）。
