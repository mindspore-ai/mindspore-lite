# Stable Audio 3 Small (music-base) ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `stabilityai/stable-audio-3-small-music-base`（Stability AI 的文生音频潜在扩散 Transformer）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上完成端到端文生音频推理与精度对齐。

Stable Audio 3 Small 基于 **stable-audio-tools**（Stability AI 自有仓库，非 diffusers）实现，由三部分组成：

1. **T5 文本编码器**：将 prompt 编码为文本嵌入 `[1, 256, 768]`。
2. **DiT 去噪器（潜在扩散 Transformer）**：在 64 通道音频潜空间上做去噪，cross-attention 注入文本嵌入，global_cond 注入 `(sigma, seconds)` 条件。
3. **音频自编码器解码器**：将去噪后的 latent `[1, 64, 313]` 解码为立体声音频波形 `[1, 2, 320000]`（10s @ 32kHz）。

> ⚠️ **架构假设**：Stable Audio 3 的官方实现为 stable-audio-tools，其 `model_config.json` 不在 diffusers 内。本教程基于 stable-audio-tools 的公开架构（T5 文本编码器 + 潜在 DiT + 音频自编码器）编写；具体子模块名、潜空间维度（64 通道、downsampling=1024）、DiT 隐藏维度（1536=24×64，深度 24）、global_cond 拼接布局等若与实际 checkpoint 的 `model_config.json` 不一致，需通过 `--latent-channels / --dit-hidden / --global-cond-dim` 等 CLI 参数覆盖（详见第 9 节 FAQ）。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux（昇腾环境，MindSpore Lite 2.9.0 + CANN 8.5.0 + Ascend 驱动）
- Atlas 300I Duo（310P3，单卡 ~44GB），DiT+音频解码器 单卡可运行

### 依赖版本（建议）

| 软件包              | 版本   |
|--------------------|------|
| Python             | 3.11 |
| torch              | 2.9.0（CPU 即可，仅用于导出/调度器/对齐） |
| stable-audio-tools | 最新（从 GitHub 安装） |
| transformers       | 5.9.0（含 T5 tokenizer） |
| soundfile          | 0.13.x（写 WAV） |
| numpy              | 2.x |
| mindspore-lite     | 2.9.0 |
| CANN               | 8.5.0 |

### 安装命令

```bash
# stable-audio-tools（Stability AI 官方仓库，Stable Audio 3 的加载入口）
pip install "stable-audio-tools @ git+https://github.com/Stability-AI/stable-audio-tools"

pip install torch==2.9.0 transformers==5.9.0 soundfile onnx onnxruntime numpy
```

### 初始化环境

```bash
source /home/yf/env.sh   # CANN / mindspore-lite / converter_lite
```

---

## 2. 模型下载

从 HuggingFace 下载 `stabilityai/stable-audio-3-small-music-base`（含 `model_config.json` + 权重 `.safetensors`/`.pt`）：

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('stabilityai/stable-audio-3-small-music-base', \
  local_dir='./stable-audio-3-small-music-base'))"
```

下载后目录应包含：

```
stable-audio-3-small-music-base/
├── model_config.json        # stable-audio-tools 配置（DiT / 自编码器 / 文本编码器）
├── model.safetensors        # 权重
└── (可选) tokenizer/
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本将 stable-audio-3-small 拆为三个 ONNX 子图（10s 固定时长）：

1. **text_encoder**（`stable_audio_text_encoder.onnx`）：`input_ids[1,256]` + `attention_mask[1,256]` → `last_hidden_state[1,256,768]`。
2. **dit**（`stable_audio_dit.onnx`）：`x[1,64,313]` + `t[1]`（sigma）+ `cross_attn_cond[1,256,768]` + `global_cond[1,1536]` → `velocity_pred[1,64,313]`。注意力替换为 CANN `PromptFlashAttention` Custom 算子。
3. **audio_decoder**（`stable_audio_audio_decoder.onnx`）：`latents[1,64,313]` → `audio[1,2,~320000]`。

### 自定义算子策略

stable-audio-tools 的 DiT 注意力为全双向（无 causal / 无 padding mask）。脚本 monkeypatch `F.scaled_dot_product_attention`，将其替换为 CANN `PromptFlashAttention` Custom 节点（BNSD 布局、`sparse_mode=0`、不传 atten_mask），其余（q/k/v 投影、AdaLN、时间嵌入、global_cond MLP）均走标准 stable-audio-tools 算子 trace。这样只需改注意力一处，避免手写 DiT 全部层。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/stable_audio_3_small_music_base

python export_stable_audio_3_small_music_base_onnx.py \
  --model-dir ./stable-audio-3-small-music-base \
  --output-dir ./stable_audio_onnx \
  --seconds 10 \
  --sample-rate 32000 \
  --latent-channels 64 \
  --latent-downsampling 1024 \
  --text-seq-len 256 \
  --text-dim 768 \
  --dit-hidden 1536 \
  --global-cond-dim 1536 \
  --dtype float32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-dir` | stable-audio-3-small checkpoint 目录（含 `model_config.json`） | 必填 |
| `--output-dir` | ONNX 输出目录 | `./stable_audio_onnx` |
| `--parts` | 导出子集（text,dit,decoder） | 全部 |
| `--seconds` | 生成音频时长（秒，固定） | `10` |
| `--sample-rate` | 采样率 | `32000` |
| `--latent-channels` | 潜空间通道数 | `64` |
| `--latent-downsampling` | 每个 latent frame 对应的音频样本数 | `1024` |
| `--text-seq-len` | T5 文本序列长度 | `256` |
| `--text-dim` | T5 输出维度 | `768` |
| `--dit-hidden` | DiT 隐藏维度 | `1536` |
| `--dit-heads` | DiT 注意力头数 | `24` |
| `--global-cond-dim` | global_cond 向量维度 | `1536` |
| `--dtype` | 导出精度（float32 推荐，converter 转 fp16） | `float32` |
| `--no-custom-op` | 不把注意力替换为 Custom 算子 | `False` |

### 模型架构参数（10s 配置）

| 参数 | 值 |
|------|------|
| 潜空间通道数 | 64 |
| latent downsampling | 1024（1 latent frame ≈ 1024 音频样本） |
| latent_frames（10s） | `ceil(10×32000/1024) = 313` |
| 音频输出 | 立体声 `[1, 2, 320000]`（10s @ 32kHz） |
| T5 文本维度 | 768 |
| DiT 隐藏维度 | 1536（24 heads × 64） |
| DiT 深度 | 24 |
| global_cond 维度 | 1536（sigma + seconds 拼接） |

---

## 4. ONNX 模型结构说明

DiT ONNX 使用 MindSpore Lite 自定义算子（`PromptFlashAttention`），**不支持直接用 ONNX Runtime 推理**，需通过 `converter_lite` 转 MindIR 后运行。文本编码器 / 音频解码器为标准算子图（可单独用 ORT 验证）。

### 模型中包含的自定义算子（DiT）

| 算子 | 数量 | 说明 |
|------|------|------|
| PromptFlashAttention | 24（DiT 每层自注意力 + 交叉注意力） | 全双向注意力，无 mask |

> 实际算子数量取决于 checkpoint 的 DiT 层数与是否含交叉注意力；以导出后 ONNX 图为准。

---

## 5. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/stable_audio_3_small_music_base

CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# text encoder
$CONV --fmk=ONNX --modelFile=./stable_audio_onnx/stable_audio_text_encoder.onnx \
  --outputFile=./stable_audio_onnx/stable_audio_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/stable_audio_text_encoder.config

# dit
$CONV --fmk=ONNX --modelFile=./stable_audio_onnx/stable_audio_dit.onnx \
  --outputFile=./stable_audio_onnx/stable_audio_dit \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/stable_audio_dit.config

# audio decoder
$CONV --fmk=ONNX --modelFile=./stable_audio_onnx/stable_audio_audio_decoder.onnx \
  --outputFile=./stable_audio_onnx/stable_audio_audio_decoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/stable_audio_audio_decoder.config
```

### config 说明

- `stable_audio_text_encoder.config`：固定 `input_ids[1,256]` + `attention_mask[1,256]`，`force_fp16`。
- `stable_audio_dit.config`：固定 `x[1,64,313]` + `t[1]` + `cross_attn_cond[1,256,768]` + `global_cond[1,1536]`，`force_fp16`，`plugin_custom_ops=All`。
- `stable_audio_audio_decoder.config`：固定 `latents[1,64,313]`，`force_fp16`。

> 转换日志中可能出现 `protobuf size` 等 warning（DiT 权重外置化），**不影响最终产物**（产出 `*_graph.mindir` + `*_variables/`），可忽略。

---

## 6. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/stable_audio_3_small_music_base

python infer_stable_audio_3_small_music_base_mslite.py \
  --mindir-dir ./stable_audio_onnx \
  --model-dir ./stable-audio-3-small-music-base \
  --prompt "128 BPM tech house drum loop, punchy kick, deep bass, 909 hi-hats" \
  --negative-prompt "" \
  --seconds 10 --num-inference-steps 100 --guidance-scale 4.0 \
  --text-device 1 --dit-device 0 --decoder-device 0 \
  --seed 42 \
  --output ./stable_audio_output.wav
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mindir-dir` | 含三个 `*_graph.mindir` 的目录 | 必填 |
| `--model-dir` | stable-audio-3-small checkpoint 目录（取 tokenizer） | 必填 |
| `--prompt` | 文本提示词 | 见上 |
| `--negative-prompt` | 负向提示词（CFG） | `""` |
| `--output` | 输出 WAV 路径 | `./stable_audio_output.wav` |
| `--seconds` | 生成时长（须与导出一致） | `10` |
| `--num-inference-steps` | 去噪步数 | `100` |
| `--guidance-scale` | CFG scale | `4.0` |
| `--sigma-min / --sigma-max` | sigma 调度范围 | `0.0001 / 1000.0` |
| `--seed` | 初始噪声种子 | `42` |
| `--latents-npy` | 预生成 latent（用于对齐） | `None` |
| `--text-device` | 文本编码器所在昇腾卡 | `1` |
| `--dit-device` | DiT 所在昇腾卡 | `0` |
| `--decoder-device` | 音频解码器所在昇腾卡 | `0` |

### 流程

1. T5（dev1）编码 prompt → `cross_attn_cond[1,256,768]`（cond + uncond 各一次）。
2. numpy 固定 seed 生成噪声 × sigma_max → 初始 latent `[1,64,313]`。
3. 按 `FlowMatchingEulerScheduler`（CPU，sigma_max → sigma_min 几何调度）逐 step：CPU 构造 `global_cond` → DiT(dev0) 跑 cond/uncond → CFG 合并 → Euler 更新 latent（100 步）。
4. 音频解码器(dev0) → 波形 → clip[-1,1] → 写 32kHz WAV。

---

## 7. 性能数据

> 以下为 Atlas 300I Duo（310P3）实测数据（10s 立体声，100 步，fp16）。数值以推理脚本端到端打印为准。

| 指标 | 300I Duo 耗时 |
|---|---|
| 文本编码 (T5, dev1) | _（待运行后填入）_ |
| DiT 总计 (100 步) | _（待运行后填入）_ |
| DiT 单步平均 | _（待运行后填入）_ |
| 音频解码 (dev0) | _（待运行后填入）_ |
| 端到端 | _（待运行后填入）_ |

---

## 8. 精度对齐

提供 `align_stable_audio_3_small_music_base.py`，在固定 prompt/seed/latent 下对每个组件做 stable-audio-tools(CPU fp32) vs MindIR(Ascend fp16) 数值比对（每组件 1 次 forward，快速且严谨）：

```bash
python align_stable_audio_3_small_music_base.py \
  --model-dir ./stable-audio-3-small-music-base \
  --mindir-dir ./stable_audio_onnx \
  --prompt "128 BPM tech house drum loop" --seed 42
```

输出 T5 last_hidden_state / DiT velocity_pred / 音频波形 的 `max_abs`、`mean_abs`、`max_rel`。fp16 下 `max_abs` 通常在 `1e-2` ~ `1e-1` 量级（长序列注意力 + fp16 累积误差）。

常见误差源与对策：

- **fp16 溢出**：DiT 已默认 `force_fp16`；若 velocity_pred 偏差大，可对 DiT config 改 `force_fp32` 重新转换。
- **global_cond 布局不一致**：Stable Audio 3 的 `(sigma, seconds)` 拼接布局可能因 stable-audio-tools 版本而异（见第 9 节 FAQ），需对照 `model_config.json` 的 `NumberConditioner` 配置。
- **初始噪声不一致**：精度比对时两端 `--seed` 必须相同。
- **T5 序列长度不一致**：推理 `--text-seq-len` 必须与导出一致（256）。

---

## 9. 常见问题

### 1) `model_config.json not found`

stable-audio-3-small 必须用 stable-audio-tools 加载，其入口是 checkpoint 目录下的 `model_config.json`。请确认 `--model-dir` 指向解压后的 checkpoint（含 `model_config.json` + `model.safetensors`/`.pt`），而非 HuggingFace diffusers 子目录。

### 2) 架构参数与实际 checkpoint 不符（关键假设）

本教程基于 stable-audio-tools 的公开 Stable Audio 3 Small 架构编写，以下假设**需要用实际 `model_config.json` 验证**，若不符需通过 CLI 参数覆盖：

| 假设项 | 默认值 | 覆盖参数 | 备注 |
|---|---|---|---|
| latent 通道数 | 64 | `--latent-channels` | stable-audio-tools `model.model.io_channels` |
| latent downsampling | 1024 | `--latent-downsampling` | 自编码器总下采样比，决定 `latent_frames` |
| T5 文本维度 | 768 | `--text-dim` | 可能是 t5-base(768) 或更大 |
| DiT 隐藏维度 | 1536 | `--dit-hidden` | `model.model.dim` |
| DiT 深度 | 24 | `--dit-depth` | `len(model.model.layers)` |
| global_cond 维度 | 1536 | `--global-cond-dim` | 通常 = DiT hidden |
| 文本编码器子模块名 | `text_t5` | （在导出脚本 `_find_t5` 中调整） | `conditioner.conditioners` 的 key |

> 若上述与实际不符，导出的 ONNX shape 会与权重不匹配，导出会报 shape 错误。请先用 stable-audio-tools 在 CPU 上加载模型并打印 `dit.config` / `autoencoder.config` 核对，再调整 CLI 参数。

### 3) `global_cond` 拼接布局（待验证）

`infer_*.py` 中的 `_build_global_cond` 按 `[sinusoidal(seconds_total), seconds_start_ratio, seconds_total, sigma]` 拼接并 pad/truncate 到 `global_cond_dim`。stable-audio-tools 实际可能用 `NumberConditioner` 先嵌入再投影，布局略有差异。**若对齐脚本显示 DiT 误差大，最先排查此处**——可改为直接传 stable-audio-tools `conditioner` 在 CPU 上算出的 `global_cond`（在推理脚本里加一步 CPU 预处理）。

### 4) `Only support CustomAscend, but got ...`

MindIR 必须用 `--optimize=ascend_oriented` 转换（保留 Custom 算子映射）。请勿用 `--optimize=general`。

### 5) DiT 图编译耗时长

10s（313 latent frame）的 DiT 在 310P3 上的图编译可能需要数十分钟，属正常现象；固定 shape 已是最快路径。

### 6) 写 WAV 失败：`Neither soundfile nor scipy is installed`

推理脚本优先用 `soundfile` 写 WAV（float32），失败则回退 `scipy.io.wavfile`（int16）。请确保至少安装其一：`pip install soundfile`。

### 7) stable-audio-tools API 变更

stable-audio-tools 仍在迭代，`create_model_from_config` / `create_autoencoder_from_config` / `create_conditioner_from_config` 的签名可能变化。若导入失败，请对照安装的 stable-audio-tools 版本调整 `_load_submodels`。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Stable Audio 3 Small 模型页（HuggingFace）](https://huggingface.co/stabilityai/stable-audio-3-small-music-base)
- [stable-audio-tools 仓库（GitHub）](https://github.com/Stability-AI/stable-audio-tools)
- [Stability AI](https://stability.ai/)

---

## 11. 许可证

Stable Audio 3 Small 遵循 [Stability AI Community License（非商用）](https://huggingface.co/stabilityai/stable-audio-3-small-music-base/blob/main/LICENSE.md)。本教程遵循相应依赖的许可证要求。
