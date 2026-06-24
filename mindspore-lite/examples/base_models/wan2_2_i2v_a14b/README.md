# Wan2.2-I2V-A14B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B-Diffusers) 图像生成视频（Image-to-Video）**MoE** 模型按网络结构拆分导出为 ONNX，使用 ONNX Runtime 验证子模型推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 **Ascend Atlas 300I Duo** 上完成端到端推理与测速。

**关键：Wan2.2-A14B 的 MoE 并非常见的 per-token top-k 路由 MoE（如 DeepSeek/HiDream），而是面向扩散去噪的「双专家稠密条件」设计**——一个高噪声专家（`transformer/`）处理早期高噪声步、一个低噪声专家（`transformer_2/`）处理晚期低噪声步，在固定时间步阈值 `boundary_ratio × num_train_timesteps`（即 Wan2.2 论文的 `t_moe` SNR 阈值）处切换。两个专家均为标准的稠密 `WanTransformer3DModel`，**切换发生在调度器循环（Python 层），不在被追踪的图内部**，因此导出完全 JIT-trace 安全——无需对路由做任何 monkeypatch（详见第 8 节 FAQ）。

Wan2.2-I2V-A14B 同时使用 Wan2.2 的 `expand_timesteps` 调度（per-token 时间步）与 **mask-mix 16 通道** 图像条件（区别于 Wan2.1-I2V 的 36 通道拼接方案）。本目录基于已验证的 `wan2_2_ti2v_5b`（同 `expand_timesteps` + mask-mix 路径）模板，叠加双专家 MoE 切换适配。

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
python -c "from modelscope import snapshot_download as s; print(s('Wan-AI/Wan2.2-I2V-A14B-Diffusers', cache_dir='/home/yf/modelscope_cache'))"
ln -sfn /home/yf/modelscope_cache/Wan-AI/Wan2.2-I2V-A14B-Diffusers Wan2.2-I2V-A14B-Diffusers
```

`MODEL_DIR`（权重目录）需包含 `text_encoder/`、`image_encoder/`、**`transformer/`（高噪声专家）**、**`transformer_2/`（低噪声专家）**、`vae/`、`tokenizer/`、`feature_extractor/`、`scheduler/`。

---

## 2. 模型导出 ONNX

按结构拆分为五个固定 shape 子模型（比单专家多一个低噪声专家）：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `wan_text_encoder` (UMT5-XXL) | input_ids[1,512], attention_mask[1,512] | last_hidden_state[1,512,4096] |
| `wan_clip_image_encoder` (CLIP ViT-H/14) | pixel_values[1,3,224,224] | image_embeds[1,257,1280] |
| `wan_transformer_high_noise` (高噪声专家, DiT 14B) | hidden_states[1,16,21,60,104], timestep[1,32760], encoder_hidden_states[1,512,4096], encoder_hidden_states_image[1,257,1280] | noise_pred[1,16,21,60,104] |
| `wan_transformer_low_noise` (低噪声专家, DiT 14B) | 同上（与高噪声专家 I/O 完全一致） | noise_pred[1,16,21,60,104] |
| `wan_vae_decoder` (3D VAE) | latents[1,16,21,60,104] | video[1,3,81,480,832] |

固定配置：480×832、81 帧（21 个 latent 帧）。transformer 输入 16 通道（mask-mix），`timestep` 形状 `[1, 32760]`（Wan2.2 `expand_timesteps` 下每个时空 token 一个时间步，21 × 30 × 52 = 32760，对应 patch_size (1,2,2) 的 token 数）。两个专家的图形状完全一致，仅权重不同。若需其它分辨率/帧数，需重新导出并重新转换。

### 导出命令

```bash
cd mindspore-lite/examples/base_models/wan2_2_i2v_a14b

python export_wan2_2_i2v_a14b_onnx.py \
  --model-dir ./Wan2.2-I2V-A14B-Diffusers \
  --output-dir ./wan2_2_i2v_a14b_onnx \
  --height 480 --width 832 --num-frames 81 --max-seq-len 512 \
  --dtype float32
```

> 说明：两个 ~14B 专家分别在 CPU float32 下顺序导出，脚本在两次导出之间释放内存（`gc.collect` + `del`），单台 ~190GB 可用内存的主机即可顺序完成。若内存紧张，可用 `--parts text,clip,transformer,vae` 分批，并单独 `--parts transformer` 两次（手动改子目录）。

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-dir` | 权重目录 | 必填 |
| `--output-dir` | ONNX 输出目录 | `./wan2_2_i2v_a14b_onnx` |
| `--parts` | 导出哪些子模型 | `text,clip,transformer,vae` |
| `--height/--width` | 视频分辨率（16 的倍数） | `480/832` |
| `--num-frames` | 视频帧数（4k+1） | `81` |
| `--max-seq-len` | UMT5 最大序列长度 | `512` |
| `--dtype` | 导出精度（建议 float32，便于转换） | `float32` |
| `--no-custom-op` | 不把注意力替换为 CANN Custom 算子 | `False` |

### 产出文件

```text
./wan2_2_i2v_a14b_onnx/
├── wan_text_encoder.onnx (+ 外部权重 .data)
├── wan_clip_image_encoder.onnx (+ 外部权重 .data)
├── wan_transformer_high_noise.onnx (+ 外部权重 .data)   # 高噪声专家
├── wan_transformer_low_noise.onnx  (+ 外部权重 .data)   # 低噪声专家
└── wan_vae_decoder.onnx (+ 外部权重 .data)
```

### 导出注意事项

- **MoE 双专家稠密条件**：Wan2.2-A14B 的「混合专家」实为两个稠密 `WanTransformer3DModel`，在 `boundary_ratio` 处由调度器循环切换（高噪声步用 `transformer/`，低噪声步用 `transformer_2/`）。两者各自独立导出为稠密 ONNX，**不存在 per-token 路由**，因此无需任何路由 monkeypatch / gather 替换布尔索引等技巧，导出完全 trace 安全（见第 8 节 FAQ）。
- **注意力替换为 CANN `PromptFlashAttention` Custom 算子**：两个专家的自注意力、文本交叉注意力、图像交叉注意力均为全双向（无 mask），脚本 monkeypatch diffusers 的注意力派发（`transformer_wan.dispatch_attention_fn`），将所有注意力导出为 `PromptFlashAttention` Custom 节点（BNSD、`sparse_mode=0`），从而避免在 ~33k token 全注意力上实体化 O(seq²) 得分矩阵。两次导出共享同一 patched dispatch。其余（q/k/v 投影、added_kv 投影、RMSNorm、RoPE、3D patchify）走标准算子。
- **Wan2.2 `expand_timesteps` 时间步**：transformer 导出时 `timestep` 固定为 `[1, 32760]`（每 token 一个时间步）；推理脚本用 `first_frame_mask[:, ::2, ::2] * t` 在 numpy 侧重建该 per-token 时间步，与 diffusers `pipeline_wan_i2v.py` 的 `expand_timesteps` 分支一致。
- **mask-mix 16 通道条件**：transformer 输入 16 通道（不是 Wan2.1-I2V 的 36 通道拼接）；推理侧每步按 `(1-mask)*condition + mask*latents` 在 numpy 重建。
- **CLIP 图像编码器**：导出 `CLIPVisionModel`（ViT-H/14，`image_encoder/` 子目录），取倒数第二层 hidden state（`hidden_states[-2]`），形状 `[1, 257, 1280]`（256 patch + 1 CLS）。
- 导出走 **legacy 导出器**（`torch.onnx.utils.export`），`do_constant_folding=False`（长序列图常量折叠会在 CPU 上 OOM）。
- 模型以 **float32** 加载导出，避免 ONNX 全图 FLOAT16 导致转换器报错。

---

## 3. ONNX 推理

> 说明：`wan_transformer_*.onnx` 含 `Custom` 节点（PromptFlashAttention），**ONNX Runtime 无法直接执行**；文本编码器、CLIP 图像编码器与 VAE 为标准算子图，可用 ONNX Runtime 验证。transformer 的精度基准以 HF diffusers pipeline 为准（见第 7 节）。

```bash
# （可选）用 ONNX Runtime 验证文本编码器 / CLIP / VAE
python - <<'PY'
import numpy as np, onnxruntime as ort
m = ort.InferenceSession("wan2_2_i2v_a14b_onnx/wan_clip_image_encoder.onnx",
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
$CONV --fmk=ONNX --modelFile=./wan2_2_i2v_a14b_onnx/wan_text_encoder.onnx \
  --outputFile=./wan2_2_i2v_a14b_onnx/wan_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_text_encoder.config

# CLIP 图像编码器
$CONV --fmk=ONNX --modelFile=./wan2_2_i2v_a14b_onnx/wan_clip_image_encoder.onnx \
  --outputFile=./wan2_2_i2v_a14b_onnx/wan_clip_image_encoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_clip_image_encoder.config

# 高噪声专家 transformer
$CONV --fmk=ONNX --modelFile=./wan2_2_i2v_a14b_onnx/wan_transformer_high_noise.onnx \
  --outputFile=./wan2_2_i2v_a14b_onnx/wan_transformer_high_noise \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_transformer_high_noise.config

# 低噪声专家 transformer
$CONV --fmk=ONNX --modelFile=./wan2_2_i2v_a14b_onnx/wan_transformer_low_noise.onnx \
  --outputFile=./wan2_2_i2v_a14b_onnx/wan_transformer_low_noise \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_transformer_low_noise.config

# VAE 解码器
$CONV --fmk=ONNX --modelFile=./wan2_2_i2v_a14b_onnx/wan_vae_decoder.onnx \
  --outputFile=./wan2_2_i2v_a14b_onnx/wan_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_vae_decoder.config
```

### 配置文件

`configs/wan_transformer_high_noise.config`（低噪声专家配置与之完全一致；注意 Wan2.2 `expand_timesteps` 的 per-token timestep 与 16 通道 mask-mix 输入）：

```ini
[acl_build_options]
input_format="ND"
input_shape="hidden_states:1,16,21,60,104;timestep:1,32760;encoder_hidden_states:1,512,4096;encoder_hidden_states_image:1,257,1280"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

### 产出说明

```text
./wan2_2_i2v_a14b_onnx/
├── wan_text_encoder_graph.mindir              + wan_text_encoder_variables/
├── wan_clip_image_encoder_graph.mindir        + wan_clip_image_encoder_variables/
├── wan_transformer_high_noise_graph.mindir    + wan_transformer_high_noise_variables/
├── wan_transformer_low_noise_graph.mindir     + wan_transformer_low_noise_variables/
└── wan_vae_decoder_graph.mindir               + wan_vae_decoder_variables/
```

执行日志（待运行后填入）：

```log
CONVERT RESULT SUCCESS:0   （待运行后填入完整日志）
```

---

## 5. MindSpore Lite 推理

```bash
python infer_wan2_2_i2v_a14b_mslite.py \
  --mindir-dir ./wan2_2_i2v_a14b_onnx \
  --model-dir ./Wan2.2-I2V-A14B-Diffusers \
  --image ./condition.jpg \
  --prompt "A cat walking on a beach, cinematic, 4k." \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 \
  --boundary-ratio 0.25 \
  --output wan_output.mp4
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir-dir` | MindIR 目录（含 5 个 `*_graph.mindir`） | 必填 |
| `--model-dir` | 权重目录（tokenizer + feature_extractor + scheduler + vae config） | 必填 |
| `--image` | 条件图（首帧，PIL 可读格式） | 必填 |
| `--prompt/--negative-prompt` | 文本提示 | 见默认 |
| `--height/--width/--num-frames` | 必须与导出/转换一致 | `480/832/81` |
| `--num-inference-steps` | 去噪步数 | `50` |
| `--guidance-scale` | CFG 强度 | `5.0` |
| `--boundary-ratio` | MoE 双专家切换阈值（占 num_train_timesteps 比例） | `0.25` |
| `--text-device/--clip-device/--transformer-device/--vae-device` | 组件分芯 | `1/1/0/0` |
| `--latents-npy` | 预生成噪声（精度对齐用） | 无 |

说明（固定 shape 约束）：`ascend_oriented` 转换按固定 shape 编译，推理侧 `--height/--width/--num-frames` 必须与导出一致；变更需重新导出+转换。文本编码器与 CLIP 图像编码器在 dev1；**两个 transformer 专家与 VAE 解码器共享 dev0**——两个专家运行在互斥的时间步区间（高/低噪声），从不同时占用，故可共置一芯（组件级分芯，非张量并行）。VAE 编码器（仅用于编码条件图，运行一次）在 CPU 上用 torch 执行，不进入 Ascend MindIR 集合。

执行日志（待运行后填入，含性能数据）：

```log
（待运行后填入实际输出：生成视频路径 + 各阶段耗时）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3，每芯 ~44GB），CANN 8.5.0，MindSpore Lite 2.10.0。

> 性能数据以推理脚本端到端打印为准；下表为**待实测填入**（运行 `infer_wan2_2_i2v_a14b_mslite.py` 后回填真实数值，不使用估算或占位假数据）。每步只跑 1 个专家（按 `boundary_ratio` 切换），故单步耗时与单专家 14B 相当；总步数中高/低噪声步占比取决于 `boundary_ratio`。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 文本编码 (UMT5, dev1) | _待运行填入_ |
| CLIP 图像编码 (dev1) | _待运行填入_ |
| VAE 编码条件图 (CPU, 一次) | _待运行填入_ |
| Transformer 总计 (50 步 × CFG×2，双专家切换) | _待运行填入_ |
| Transformer 单步平均 | _待运行填入_ |
| VAE 解码 (dev0) | _待运行填入_ |
| **端到端** | **_待运行填入_** |

---

## 7. 精度对齐

端到端对比 HF diffusers `WanImageToVideoPipeline`（CPU float32 基线，`expand_timesteps=True` + `boundary_ratio` 双专家 MoE）与 MSLite（Ascend）的生成视频：使用相同 prompt、相同条件图、相同初始噪声（seed 固定的 torch 生成器，存 npy 后两路共用）、相同调度器参数与相同 `boundary_ratio`，逐帧比较 max/mean abs 误差与 PSNR。

```bash
# 注意：HF CPU 基线较慢，建议先用较少帧/步数跑通端到端对齐
python align_wan2_2_i2v_a14b.py \
  --mindir-dir ./wan2_2_i2v_a14b_onnx \
  --model-dir ./Wan2.2-I2V-A14B-Diffusers \
  --image ./condition.jpg \
  --num-frames 21 --num-inference-steps 10 \
  --boundary-ratio 0.25
```

执行日志（待运行后填入）：

```log
（待运行后填入：max_abs / mean_abs / PSNR）
```

---

## 8. 常见问题

1. **Wan2.2-A14B 的 MoE 是 per-token 路由吗？导出时需要 monkeypatch 路由吗？**
   - **不是，也无需 monkeypatch。** Wan2.2-A14B 的 MoE 是面向扩散去噪的「双专家稠密条件」设计：高噪声专家（`transformer/`）与低噪声专家（`transformer_2/`）均为标准稠密 `WanTransformer3DModel`，在固定时间步阈值 `boundary_ratio × num_train_timesteps`（Wan2.2 论文的 `t_moe`，由 SNR 单调递减到 `SNR_min` 的一半确定）处由调度器循环切换。**切换是 Python 层的时间步分支，不在被追踪的图内部**，因此不存在 data-dependent 控制流（`.tolist()` / 布尔索引 / top-k 路由），导出完全 JIT-trace 安全。无需 skill 经验 #23 的「gather 替换布尔索引」或展平专家等任何缓解措施。
2. **为什么导出两个 transformer ONNX，而不是一个？**
   - 两个专家权重独立、各自是完整 14B 稠密 DiT；导出为两个独立 ONNX（`wan_transformer_high_noise` / `wan_transformer_low_noise`）后各自转 MindIR，推理侧按 `boundary_ratio` 在两者间切换。这完全对齐 diffusers `WanImageToVideoPipeline` 的 `transformer` + `transformer_2` + `boundary_ratio` 设计。
3. 现象：导出 transformer 时 CPU 内存暴涨/被 OOM 杀死。
   - 原因：legacy 导出器默认常量折叠，长序列（~33k token）图折叠常量占满内存。
   - 解决方案：`do_constant_folding=False`（本脚本已设置）；注意力已替换为 Custom 算子，fallback 仅为保形 stub；两个专家顺序导出并在之间 `del` + `gc.collect` 释放。
4. 现象：ONNX Runtime 无法运行 `wan_transformer_*.onnx`。
   - 原因：该 ONNX 含 `PromptFlashAttention` Custom 节点。
   - 解决方案：transformer 经 converter 转 MindIR 后在 Ascend 运行；精度基准用 HF pipeline。
5. 现象：converter 报 `do not support data_type: 10`。
   - 原因：模型以 fp16 加载导出导致全图 FLOAT16。
   - 解决方案：以 `--dtype float32` 导出。
6. 现象：推理 shape 不匹配。
   - 原因：`--height/--width/--num-frames` 与导出/转换不一致。
   - 解决方案：三者必须与导出一致；变更需重新导出+转换。
7. 现象：transformer 输入 `timestep` 维度对不上 / 输入通道数对不上。
   - 原因：Wan2.2-I2V-A14B 使用 `expand_timesteps`，时间步为 per-token（`[1, 32760]`）；并采用 mask-mix 16 通道条件（不是 Wan2.1-I2V 的 36 通道拼接）。
   - 解决方案：确认导出/转换/推理三处的 `timestep` 形状（`[1, 32760]` 对应 480×832×81）与 `hidden_states` 通道（16）一致；推理脚本已在 numpy 侧用 `first_frame_mask[:, ::2, ::2] * t` 重建时间步、用 `(1-mask)*condition + mask*latents` 重建输入。
8. 现象：缺少 `feature_extractor/` 或 `transformer_2/` 子目录。
   - 原因：CLIP 图像预处理依赖 `CLIPImageProcessor` 配置；`transformer_2/` 是 Wan2.2-A14B 低噪声专家。
   - 解决方案：确认 `MODEL_DIR` 含 `feature_extractor/` 与 `transformer_2/`（Wan2.2-A14B-Diffusers 仓库自带）；缺 `feature_extractor/` 则从 `openai/clip-vit-large-patch14` 复用。

---

## 9. 参考资源与许可证

- 上游模型：<https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B-Diffusers>
- Wan2.2 MoE 架构说明（双专家稠密条件设计）：Wan2.2 仓库 README「Mixture-of-Experts (MoE) Architecture」章节
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- 同构参考（本目录改写模板）：`../wan2_2_ti2v_5b/`（Wan2.2-TI2V-5B，同 `expand_timesteps` + mask-mix 16 通道 + CLIP 条件）
- 条件路径对照：`../wan2_1_i2v_14b_480p/`（Wan2.1-I2V-14B，36 通道拼接 + 标量时间步，单专家）
- 本目录脚本遵循 MindSpore Lite 仓库许可证；上游模型权重许可证以其仓库为准（Apache-2.0）。
