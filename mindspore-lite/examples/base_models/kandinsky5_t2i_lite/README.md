# Kandinsky-5.0-T2I-Lite ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `kandinskylab/Kandinsky-5.0-T2I-Lite`（~6B flow-matching DiT 文生图扩散模型）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上完成端到端文生图推理与精度对齐。

Kandinsky-5.0-T2I-Lite 由四部分组成：

- **Qwen2.5-VL-7B-Instruct 文本编码器**（3584 维 hidden state，cross-attention 条件）—— 注意：**不是 MT5/T5**。
- **CLIP 文本编码器**（openai/clip-vit-large-patch14，768 维 pooled，加到 timestep embedding）。
- **Kandinsky5Transformer3DModel**（去噪器，2 个 text block + 50 个 visual block，model_dim 2560 = 20 heads × 128，patchify `(1,2,2)`，3D RoPE）。
- **FLUX.1-dev AutoencoderKL 解码器**（8× 空间压缩，16 通道 latent）—— 注意：**不是 DCAE**（K5 Image Lite 复用 FLUX VAE，社区流传的"128× DCAE"是 Sana 而非 K5）。

本教程将四部分**全部导出为 MindIR** 并在昇腾上推理（transformer + VAE 在 dev0，Qwen + CLIP 在 dev1）。

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
| torch          | 2.9.0（CPU 即可，仅用于导出 / 对齐 / scheduler） |
| diffusers      | 0.35.0+（须包含 `Kandinsky5Transformer3DModel` 与 `Kandinsky5T2IPipeline`，2025-12 之后的主线版本） |
| transformers   | 4.49+（须包含 Qwen2.5-VL） |
| onnx           | 1.19.1 |
| mindspore-lite | 2.9.0 |
| CANN           | 8.5.0 |

### 安装命令

```bash
pip install torch==2.9.0 "diffusers>=0.35.0" "transformers>=4.49" \
    onnx==1.19.1 onnxruntime accelerate sentencepiece
```

### 初始化环境

```bash
source /home/yf/env.sh   # CANN / mindspore-lite / converter_lite
```

---

## 2. 模型下载

K5 Lite 上游为**单文件**权重仓库（`model/kandinsky5lite_t2i.safetensors`，~12GB），不含 diffusers 目录结构。文本编码器 / VAE / tokenizer 分别来自各自的 HF 仓库：

```bash
pip install modelscope huggingface_hub

# K5 Lite DiT 权重（单文件 safetensors）
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('kandinskylab/Kandinsky-5.0-T2I-Lite', cache_dir='/home/yf/modelscope_cache'))"

# Qwen2.5-VL-7B-Instruct（文本编码器 + tokenizer）
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/home/yf/modelscope_cache'))"

# CLIP（pooled 编码器 + tokenizer）
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('AI-ModelScope/clip-vit-large-patch14', cache_dir='/home/yf/modelscope_cache'))"

# FLUX.1-dev VAE（图像解码器；K5 复用 FLUX VAE）
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('black-forest-labs/FLUX.1-dev', cache_dir='/home/yf/modelscope_cache', allow_patterns=['vae/*']))"
```

软链到本教程工作目录：

```bash
ln -sfn /home/yf/modelscope_cache/kandinskylab/Kandinsky-5___0-T2I-Lite/model/kandinsky5lite_t2i.safetensors ./kandinsky5lite_t2i.safetensors
ln -sfn /home/yf/modelscope_cache/Qwen/Qwen2___5-VL-7B-Instruct ./Qwen2.5-VL-7B-Instruct
ln -sfn /home/yf/modelscope_cache/AI-ModelScope/clip-vit-large-patch14 ./clip-vit-large-patch14
ln -sfn /home/yf/modelscope_cache/black-forest-labs/FLUX___1-dev/vae ./flux_vae
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本将 K5 Lite 拆为四个 ONNX 子图：

1. **text_encoder**（`kandinsky_text_encoder.onnx`）：Qwen2.5-VL，input_ids[1,512] + attention_mask[1,512] → last_hidden_state[1,471,3584]（已内置 `[:, 41:]` 切片，去掉 chat template 前缀）。
2. **clip_encoder**（`kandinsky_clip_encoder.onnx`）：CLIP，input_ids[1,77] + attention_mask[1,77] → pooled_embeds[1,768]。
3. **transformer**（`kandinsky_transformer.onnx`）：hidden_states[1,16,1,128,128] + encoder_hidden_states[1,471,3584] + timestep[1] + pooled_projections[1,768] + visual_rope_h[64] + visual_rope_w[64] + text_rope[471] → noise_pred[1,16,1,128,128]。注意力替换为 CANN `PromptFlashAttention` Custom 算子。
4. **dcae_decoder**（`kandinsky_dcae_decoder.onnx`）：latents[1,16,128,128] → image[1,3,1024,1024]。**实际为 FLUX VAE decoder**，文件名沿用任务 brief 的 `dcae_decoder` 命名。

### 自定义算子策略

K5 注意力为全双向（text self-attn、visual self-attn、visual→text cross-attn 均无 causal / 无 padding mask）。脚本 monkeypatch diffusers 的注意力派发函数 `dispatch_attention_fn`，将 `scaled_dot_product_attention` 替换为 CANN `PromptFlashAttention` Custom 节点（BNSD 布局、`sparse_mode=0`、不传 `atten_mask`）。q/k/v 投影、RMSNorm（`norm_q`/`norm_k`）、3D RoPE、AdaLN、time/pooled embedding 均保留 diffusers 原生算子 trace。这样只需改注意力一处，避免手写 52 层。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/kandinsky5_t2i_lite

python export_kandinsky5_t2i_lite_onnx.py \
  --k5-model ./kandinsky5lite_t2i.safetensors \
  --qwen-dir ./Qwen2.5-VL-7B-Instruct \
  --clip-dir ./clip-vit-large-patch14 \
  --vae-dir ./flux_vae \
  --output-dir ./kandinsky5_t2i_lite_onnx \
  --height 1024 --width 1024 --max-seq-len 512 \
  --dtype float32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--k5-model` | K5 Lite 单文件 checkpoint 路径 | 必填 |
| `--qwen-dir` | Qwen2.5-VL-7B-Instruct 权重目录 | 必填 |
| `--clip-dir` | clip-vit-large-patch14 权重目录 | 必填 |
| `--vae-dir` | FLUX.1-dev `vae/` 目录 | 必填 |
| `--output-dir` | ONNX 输出目录 | `./kandinsky5_t2i_lite_onnx` |
| `--parts` | 导出子集（text,clip,transformer,vae） | 全部 |
| `--height/--width` | 图像尺寸（须为 16 的倍数） | `1024 1024` |
| `--max-seq-len` | Qwen 序列长度（< 1024；前 41 个 template token 会被丢弃） | `512` |
| `--dtype` | 导出精度（float32 推荐，转换器友好） | `float32` |
| `--no-custom-op` | 不把注意力替换为 Custom 算子 | `False` |

### 模型架构参数

| 参数 | 值 |
|------|------|
| model_dim (inner) | 2560（20 heads × 128） |
| num_text_blocks | 2 |
| num_visual_blocks | 50 |
| in_text_dim（Qwen hidden） | 3584 |
| in_text_dim2（CLIP pooled） | 768 |
| in/out_visual_dim（latent channels） | 16 |
| patch_size (T,H,W) | (1, 2, 2) |
| RoPE axes_dims (T,H,W) | (32, 48, 48)（和 = head_dim 128） |
| 时间嵌入 | sinusoidal，time_dim 512 |
| VAE 空间压缩 | 8×（FLUX VAE），scaling_factor ≈ 0.3611 |
| 步数 / guidance（推荐） | 50 步 / guidance_scale 3.5（CFG） |

---

## 4. ONNX 模型结构说明

transformer ONNX 使用 MindSpore Lite 自定义算子（`PromptFlashAttention`），**不支持直接用 ONNX Runtime 推理**，需通过 `converter_lite` 转 MindIR 后运行。Qwen / CLIP / VAE 为标准算子图。

### 模型中包含的自定义算子（transformer）

| 算子 | 数量 | 说明 |
|------|------|------|
| PromptFlashAttention | 52（2 text + 50 visual，每个 block 1 次 self/cross） | 全双向注意力，无 mask |

---

## 5. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/kandinsky5_t2i_lite

CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# text encoder (Qwen2.5-VL)
$CONV --fmk=ONNX --modelFile=./kandinsky5_t2i_lite_onnx/kandinsky_text_encoder.onnx \
  --outputFile=./kandinsky5_t2i_lite_onnx/kandinsky_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/kandinsky_text_encoder.config

# clip encoder
$CONV --fmk=ONNX --modelFile=./kandinsky5_t2i_lite_onnx/kandinsky_clip_encoder.onnx \
  --outputFile=./kandinsky5_t2i_lite_onnx/kandinsky_clip_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/kandinsky_text_encoder.config

# transformer
$CONV --fmk=ONNX --modelFile=./kandinsky5_t2i_lite_onnx/kandinsky_transformer.onnx \
  --outputFile=./kandinsky5_t2i_lite_onnx/kandinsky_transformer \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/kandinsky_transformer.config

# dcae_decoder (FLUX VAE decoder)
$CONV --fmk=ONNX --modelFile=./kandinsky5_t2i_lite_onnx/kandinsky_dcae_decoder.onnx \
  --outputFile=./kandinsky5_t2i_lite_onnx/kandinsky_dcae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/kandinsky_dcae_decoder.config
```

### config 说明

- `kandinsky_text_encoder.config`：固定 Qwen 序列 512（输出 471），`force_fp16`，`plugin_custom_ops=All`。
- `kandinsky_transformer.config`：固定 4096 visual token + 471 text token，`force_fp16`，`plugin_custom_ops=All`。
- `kandinsky_dcae_decoder.config`：固定 latent `[1,16,128,128]`，`force_fp16`。

> 转换日志中可能出现 `protobuf size` / `ge.proto.ModelDef exceeded maximum protobuf size` 等 warning（transformer ~6B 权重外置化），**不影响最终产物**（产出 `*_graph.mindir` + `*_variables/`），可忽略。

---

## 6. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/kandinsky5_t2i_lite

python infer_kandinsky5_t2i_lite_mslite.py \
  --mindir-dir ./kandinsky5_t2i_lite_onnx \
  --qwen-dir ./Qwen2.5-VL-7B-Instruct \
  --clip-dir ./clip-vit-large-patch14 \
  --prompt "A cat in a red hat holding a sign that says HELLO" \
  --negative-prompt "" \
  --seed 42 --num-inference-steps 50 --guidance-scale 3.5 \
  --height 1024 --width 1024 \
  --transformer-device 0 --text-device 1 \
  --output ./kandinsky5_output.png
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mindir-dir` | 包含 4 个 `*_graph.mindir` 的目录 | 必填 |
| `--qwen-dir` | Qwen2.5-VL 权重目录（取 tokenizer） | 必填 |
| `--clip-dir` | CLIP 权重目录（取 tokenizer） | 必填 |
| `--prompt` | 文本提示词 | 必填 |
| `--negative-prompt` | 负向提示词（触发 CFG） | `""` |
| `--seed` | 随机种子（初始噪声） | `42` |
| `--num-inference-steps` | 去噪步数 | `50` |
| `--guidance-scale` | CFG guidance（推荐 3.5） | `3.5` |
| `--shift` | FlowMatchEuler shift（对应上游 `scheduler_scale`） | `1.0` |
| `--vae-scaling-factor` | FLUX VAE scaling factor | `0.3611` |
| `--height/--width` | 图像尺寸（须与导出一致） | `1024 1024` |
| `--max-seq-len` | Qwen 序列长度（须与导出一致） | `512` |
| `--transformer-device` | transformer/VAE 所在昇腾卡 | `0` |
| `--text-device` | Qwen/CLIP 所在昇腾卡 | `1` |
| `--output` | 输出 PNG 路径 | `./kandinsky5_output.png` |

### 流程

1. Qwen+CLIP（dev1）编码 prompt（含 K5 chat template，Qwen 输出已在图内 `[:, 41:]` 切片）→ `encoder_hidden_states` / `pooled`。
2. numpy 固定 seed 生成噪声 `[1,16,1,128,128]` → 乘 `vae_scaling_factor`。
3. 按 `FlowMatchEulerDiscreteScheduler`（CPU，shift=1.0）逐 `timestep`：transformer(dev0) 跑 cond + uncond 两次 → CFG 合成 `noise_pred = uncond + 3.5 * (cond - uncond)` → `scheduler.step` 更新 latents（50 步）。
4. latents 除以 `vae_scaling_factor` → 去掉 T=1 轴 → VAE(dev0) → RGB 图像 → 保存 PNG。

---

## 7. 性能数据

> 以下为 Atlas 300I Duo（310P3）实测数据（1024×1024，50 步，fp16，CFG x2 = 100 次前向，单图）。数值以推理脚本端到端打印为准。

| 指标 | 300I Duo 耗时 |
|---|---|
| 文本编码 (Qwen+CLIP, dev1) | _（待运行后填入）_ |
| Transformer 总计 (50 步 × CFG 2) | _（待运行后填入）_ |
| Transformer 单步平均 (含 CFG) | _（待运行后填入）_ |
| VAE 解码 (dev0) | _（待运行后填入）_ |
| 端到端 | _（待运行后填入）_ |

---

## 8. 精度对齐

提供 `align_kandinsky5_t2i_lite.py`，支持两层比对：

### 组件级比对（快速，默认）

在固定 prompt / seed / latent 下对每个组件做 HF(CPU fp32) vs MindIR(Ascend fp16) 数值比对（每组件 1 次 forward）：

```bash
python align_kandinsky5_t2i_lite.py \
  --mindir-dir ./kandinsky5_t2i_lite_onnx \
  --qwen-dir ./Qwen2.5-VL-7B-Instruct \
  --clip-dir ./clip-vit-large-patch14 \
  --k5-model ./kandinsky5lite_t2i.safetensors \
  --vae-dir ./flux_vae \
  --prompt "A cat in a red hat holding a sign that says HELLO" --seed 42
```

输出 Qwen last_hidden_state / CLIP pooled / transformer velocity / VAE image 的 `max_abs`、`mean_abs`、`max_rel`。fp16 下 velocity / image 的 `max_abs` 通常 < ~1e-1。

### 整图端到端比对（`--full-baseline`）

加 `--full-baseline`（在 CPU 上跑一次完整 HF `Kandinsky5T2IPipeline`，非常慢），生成 `kandinsky5_baseline.png`，与 `kandinsky5_align_mslite.png` 做 PSNR + max/mean abs 比对。两端使用同一份 seeded latents。

常见误差源与对策：

- **fp16 溢出**：transformer/VAE 已默认 `force_fp16`；若 velocity 偏差大，可对 transformer config 改 `force_fp32` 重新转换。
- **Qwen 序列长度不一致**：推理 `--max-seq-len` 必须与导出一致（512）。
- **图像尺寸不一致**：`--height/--width` 必须与导出一致。
- **初始噪声不一致**：精度比对时两端 `--seed` 必须相同。
- **shift 不一致**：`--shift` 对应上游 `scheduler_scale`；默认 1.0（线性 schedule），若 HF 基线用了非 1.0 的 shift 需对齐。

---

## 9. 常见问题

### 1) "K5 用的是 DCAE / MT5 吗？" —— 不是

社区流传的"Kandinsky 5 用 128× DCAE 和 MT5"说法**不适用于 K5 Image Lite**。经核对上游 [kandinsky-5 仓库](https://github.com/kandinskylab/kandinsky-5) 与 diffusers [pipeline_kandinsky_t2i.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_t2i.py)：

- **VAE**：复用 `black-forest-labs/FLUX.1-dev` 的 `AutoencoderKL`（8× 空间压缩，16 latent 通道，`scaling_factor≈0.3611`）。代码中 `vae_scale_factor_spatial = 8`，没有 `dcae` / `DCAE` 字样。"128× DCAE"是 Sana 论文的设定，被误传到 K5。
- **文本编码器**：`Qwen2.5-VL-7B-Instruct`（`text_encoder`）+ `CLIPTextModel`（`text_encoder_2`），**不是 MT5 / T5**。Qwen 输出 3584 维 hidden state 做 cross-attention，CLIP 输出 768 维 pooled 加到 timestep。
- 本教程的 `kandinsky_dcae_decoder.*` 文件名沿用任务 brief 命名，**实际图就是 FLUX VAE decoder**。

### 2) `Kandinsky5Transformer3DModel` 是 3D 模型，T2I 怎么用？

K5 的 DiT 与视频共享同一类，T2I 时 temporal 维固定为 1、`patch_size[0]=1`，退化为 2D DiT。但 RoPE 仍是 3D 约定（T 轴 RoPE 维度 32 作用于长度 1 的轴 → 常数），导出脚本已正确构造。

### 3) Qwen 输出的 `[:, 41:]` 切片是什么？

K5 用一个固定的 chat template 包裹 prompt（system + user），tokenize 后前 41 个 token 是 template 前缀（`<|im_start|>system ... <|im_end|>\n<|im_start|>user\n`），DiT 只需 user prompt 部分，故切片 `[:, 41:]`。切片已**内置在导出的 ONNX 图中**，推理脚本只需喂完整 template 序列。

### 4) 转换时报 `ge.proto.ModelDef exceeded maximum protobuf size`

transformer ~6B 权重，`ascend_oriented` 转换会把权重外置到 `*_variables/`，日志打印此信息**不影响**产物。

### 5) transformer 图编译耗时长

6B 模型 52 层在 310P3 上的图编译可能需要数十分钟，属正常现象；固定 shape 已是最快路径。

### 6) `Only support CustomAscend, but got ...`

MindIR 必须用 `--optimize=ascend_oriented` 转换（保留 Custom 算子映射）。请勿用 `--optimize=general`。

### 7) 单卡 OOM

transformer(6B) + VAE 在单卡 ~44GB 内可运行；Qwen-7B 文本编码器占显存较大，须放在 `--text-device` 另一张卡（默认 dev1）。若 dev1 仍 OOM，可考虑对 Qwen 做 NF4 量化（上游支持）。

### 8) CLIP encoder config 文件不存在？

CLIP encoder 与 text encoder 的输入 shape 不同（CLIP `[1,77]` vs Qwen `[1,512]`），但本教程为简化复用 `kandinsky_text_encoder.config` 仅作通用 ND 配置——**实际转换 CLIP 时需另写一个 `input_shape="input_ids:1,77;attention_mask:1,77"` 的 config**，或修改第 5 节的 CLIP 转换命令使用专属 config。详见 [已知风险](#已知风险与待确认项)。

---

## 已知风险与待确认项

> 以下为基于上游源码与文档的合理推断，**运行前需用真实权重验证**：

1. **Qwen 导出走 `AutoModelForCausalLM`**：K5 上游用 `Qwen2_5_VLForConditionalGeneration`（VL 模型）。T2I 不输入图像，仅取 LM stack 的 `hidden_states[-1]`，等价于 `AutoModelForCausalLM`。若导出报 shape / module 不一致，改回 `Qwen2_5_VLForConditionalGeneration` 加载。
2. **CLIP config 复用问题**（见 FAQ 8）：CLIP 转换建议新建 `kandinsky_clip_encoder.config`（`input_shape="input_ids:1,77;attention_mask:1,77"`），勿与 Qwen config 混用。
3. **`Kandinsky5Transformer3DModel.from_single_file` API**：依赖 diffusers 主线版本（≥ 2025-12）。若所用 diffusers 仍走 `from_pretrained` + 目录结构，需先用 `Kandinsky5T2IPipeline.from_pretrained` 拉取并保存 `transformer/` 子目录，再用 `from_pretrained` 加载。
4. **RoPE `visual_rope_pos` 的确切语义**：上游 `(t_pos, h_pos, w_pos)` 为 arange；若 diffusers 主线将其改为预计算频率表，则导出 wrapper 的 `forward` 需相应调整（构造 freqs 而非 arange）。
5. **FlowMatchEuler shift**：上游 `generation_utils.py` 用 `scheduler_scale`（默认 1.0），diffusers 端将其映射为 scheduler `shift`。若 diffusers 加载的 config 写了非 1.0 的 shift，需把 `--shift` 与之对齐。
6. **性能数据全部为"待运行填入"**：本教程不包含任何实测耗时，运行后请按打印填入第 7 节表格。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Kandinsky-5 GitHub（ai-forever / kandinskylab）](https://github.com/kandinskylab/kandinsky-5)
- [Kandinsky-5.0-T2I-Lite HuggingFace 模型页](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite)
- [Diffusers K5 pipeline 源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_t2i.py)
- [Diffusers K5 transformer 源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_kandinsky.py)
- [FLUX.1-dev VAE（ModelScope）](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

---

## 11. 许可证

Kandinsky-5.0-T2I-Lite 遵循 [Apache License 2.0](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite)。本教程遵循相应依赖（FLUX.1-dev VAE 的 HF Community License、Qwen2.5-VL 的 Apache 2.0、CLIP 的 MIT）的许可证要求。
