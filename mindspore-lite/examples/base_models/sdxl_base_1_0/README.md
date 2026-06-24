# Stable-Diffusion-XL-Base-1.0 ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `stabilityai/stable-diffusion-xl-base-1.0`（SDXL base，~6.6B 参数的 UNet 文生图扩散模型）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上完成端到端文生图推理与精度对齐。

SDXL base 由四部分组成：CLIP-L 文本编码器（768 维）、CLIP-G 文本编码器（带 projection，1280 维，含 pooled 输出）、UNet2DConditionModel（去噪器，cross_attention_dim=2048）、4 通道 VAE 解码器。本教程将四部分**全部导出为 MindIR** 并在昇腾上推理（UNet+VAE 在 dev0，两个文本编码器在 dev1）。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux（昇腾环境，MindSpore Lite 2.9.0 + CANN 8.5.0 + Ascend 驱动）
- Atlas 300I Duo（310P3，单卡 ~44GB），UNet(~10GB) 单卡可运行

### 依赖版本（建议）

| 软件包            | 版本   |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.9.0（CPU 即可，仅用于导出/对齐） |
| diffusers      | 0.38.0（原生支持 SDXL UNet2DConditionModel） |
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

从 ModelScope / HuggingFace 下载 diffusers 格式权重（UNet ~10.5GB + text_encoder_2 ~6.9GB + text_encoder ~0.25GB + VAE ~0.34GB，共 ~18GB）：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('AI-ModelScope/stable-diffusion-xl-base-1.0', \
    cache_dir='/home/yf/modelscope_cache'))"
```

将其软链到 `./stable-diffusion-xl-base-1.0`：

```bash
ln -sfn /home/yf/modelscope_cache/AI-ModelScope/stable-diffusion-xl-base-1.0 \
  ./stable-diffusion-xl-base-1.0
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本将 SDXL base 拆为四个 ONNX 子图：

1. **text_encoder**（`sdxl_text_encoder.onnx`）：CLIP-L，input_ids [1,77] → last_hidden_state [1,77,768]（**倒数第二层** hidden_states[-2]，遵循 SDXL `encode_prompt` 约定）。
2. **text_encoder_2**（`sdxl_text_encoder_2.onnx`）：CLIP-G（带 projection），input_ids [1,77] → last_hidden_state [1,77,1280] + text_embeds [1,1280]（pooled）。
3. **unet**（`sdxl_unet.onnx`）：sample [1,4,128,128] + timestep [1] + encoder_hidden_states [1,77,2048]（concat 768+1280） + added_cond_kwargs{text_embeds [1,1280], time_ids [1,6]} → noise_pred [1,4,128,128]。自+交叉注意力替换为 CANN `PromptFlashAttention` Custom 算子。
4. **vae_decoder**（`sdxl_vae_decoder.onnx`）：4 通道 latent [1,4,128,128] → RGB [1,3,1024,1024]。

### 自定义算子策略

SDXL UNet 的 `AttnProcessor2_0` 将 q/k/v 投影后 reshape 为 `(batch, num_heads, seq, head_dim)`（BNSD 布局），再调用 `torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)`。脚本 monkeypatch 该函数，将 SDPA 调用替换为 CANN `PromptFlashAttention` Custom 节点（q/k/v **已是 BNSD，无需转置**、`sparse_mode=0`、不传 atten_mask，对 16384 个 latent token 的自注意力和 77 个 CLIP token 的交叉注意力均成立——两者都是全双向无 mask）。其余（q/k/v 投影、group-norm、norm_q/norm_k 的 LayerNorm、残差、输出投影、time/additive embedding）均走标准 diffusers 算子 trace。这样只需改 SDPA 一处，避免手写 UNet 全部注意力层。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/sdxl_base_1_0

python export_sdxl_base_1_0_onnx.py \
  --model-id ./stable-diffusion-xl-base-1.0 \
  --output-dir ./sdxl_base_1_0_onnx \
  --resolution 1024 1024 \
  --dtype fp32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 本地 diffusers 权重目录 | `./stable-diffusion-xl-base-1.0` |
| `--output-dir` | ONNX 输出目录 | `./sdxl_base_1_0_onnx` |
| `--resolution` | 输出图像分辨率 H W | `1024 1024` |
| `--dtype` | 导出精度（fp32 推荐，converter 会转 fp16） | `fp32` |
| `--no-custom-op` | 不把注意力替换为 Custom 算子 | `False` |
| `--components` | 导出子集（text_encoder_1,text_encoder_2,unet,vae） | 全部 |

### 模型架构参数

| 参数 | 值 |
|------|------|
| cross_attention_dim | 2048（768 CLIP-L + 1280 CLIP-G） |
| block_out_channels | (320, 640, 1280, 1280) |
| attention head_dim | 64（对应 heads 5/10/20/20） |
| addition_time_embed_dim | 256 |
| projection_class_embeddings_input_dim | 2816（1280 + 6×256） |
| in/out_channels (latent) | 4 |
| text_embeds dim (pooled CLIP-G) | 1280 |
| time_ids dim | 6（orig_h, orig_w, crop_top, crop_left, target_h, target_w） |
| VAE latent channels | 4，scaling_factor 0.13025 |
| scheduler | EulerDiscreteScheduler（CPU），prediction_type=epsilon |

---

## 4. ONNX 模型结构说明

unet ONNX 使用 MindSpore Lite 自定义算子（`PromptFlashAttention`），**不支持直接用 ONNX Runtime 推理**，需通过 `converter_lite` 转 MindIR 后运行。VAE / text_encoder / text_encoder_2 为标准算子图。

### 模型中包含的自定义算子（unet）

| 算子 | 数量 | 说明 |
|------|------|------|
| PromptFlashAttention | ~160（含 down/mid/up 所有 transformer block 的自注意力 + 交叉注意力） | 全双向注意力，无 mask |

> 精确数量取决于 SDXL UNet 的 transformer_layers_per_block 配置（base 为 (0,2,10)/down +(10,)/mid +(0,2,10)/up），每个 transformer block 含 1 个自注意力 + 1 个交叉注意力。

---

## 5. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/sdxl_base_1_0

CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# text_encoder 1 (CLIP-L)
$CONV --fmk=ONNX --modelFile=./sdxl_base_1_0_onnx/sdxl_text_encoder.onnx \
  --outputFile=./sdxl_base_1_0_onnx/sdxl_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/sdxl_text_encoder.config

# text_encoder 2 (CLIP-G)
$CONV --fmk=ONNX --modelFile=./sdxl_base_1_0_onnx/sdxl_text_encoder_2.onnx \
  --outputFile=./sdxl_base_1_0_onnx/sdxl_text_encoder_2 \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/sdxl_text_encoder_2.config

# unet
$CONV --fmk=ONNX --modelFile=./sdxl_base_1_0_onnx/sdxl_unet.onnx \
  --outputFile=./sdxl_base_1_0_onnx/sdxl_unet \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/sdxl_unet.config

# vae
$CONV --fmk=ONNX --modelFile=./sdxl_base_1_0_onnx/sdxl_vae_decoder.onnx \
  --outputFile=./sdxl_base_1_0_onnx/sdxl_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/sdxl_vae_decoder.config
```

### config 说明

- `sdxl_unet.config`：固定 128×128 latent + 77 CLIP token + time_ids[1,6]，`force_fp16`，`plugin_custom_ops=All`。
- `sdxl_text_encoder.config` / `sdxl_text_encoder_2.config` / `sdxl_vae_decoder.config`：固定 shape，`force_fp16`。

> 转换日志中可能出现 `protobuf size` / `ge.proto.ModelDef exceeded maximum protobuf size` 等 warning（unet 权重 ~10GB 外置化），**不影响最终产物**（产出 `*_graph.mindir` + `*_variables/`），可忽略。

---

## 6. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/sdxl_base_1_0

python infer_sdxl_base_1_0_mslite.py \
  --mindir-dir   ./sdxl_base_1_0_onnx \
  --model-dir    ./stable-diffusion-xl-base-1.0 \
  --prompt "A cat holding a sign that says hello world, highly detailed, 4k" \
  --negative-prompt "lowres, blurry, worst quality, low quality" \
  --seed 0 --steps 30 --guidance 5.0 \
  --unet-device 0 --vae-device 0 --text-device 1 \
  --output ./sdxl_output.png
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mindir-dir` | 含 4 个 `*_graph.mindir` 的目录 | 必填 |
| `--model-dir` | diffusers 权重目录（取 tokenizer + scheduler + vae config） | 必填 |
| `--prompt` | 文本提示词 | 必填 |
| `--negative-prompt` | 负向提示词（CFG） | `lowres, blurry, ...` |
| `--seed` | 随机种子（初始噪声） | `0` |
| `--steps` | 去噪步数 | `30` |
| `--guidance` | classifier-free guidance scale | `5.0` |
| `--height/--width` | 图像尺寸（须与导出一致） | `1024` |
| `--unet-device` | UNet 所在昇腾卡 | `0` |
| `--vae-device` | VAE 所在昇腾卡 | `0` |
| `--text-device` | 两个文本编码器所在昇腾卡 | `1` |
| `--output` | 输出 PNG 路径 | `./sdxl_output.png` |

### 流程

1. CLIP-L + CLIP-G（dev1）编码 prompt：取两编码器 hidden_states[-2]，concat → `encoder_hidden_states` [1,77,2048]；CLIP-G 的 pooled → `text_embeds` [1,1280]。负向 prompt 同理。
2. numpy 固定 seed 生成噪声 [1,4,128,128]，乘 `scheduler.init_noise_sigma`。
3. 按 `EulerDiscreteScheduler`（CPU，与原算法一致）逐 `timestep`：`scale_model_input`（除 `(sigma²+1)^0.5`）→ UNet 输入堆叠 [uncond, cond] 一次前向（CFG x2）→ `noise = uncond + guidance*(cond-uncond)` → `scheduler.step` 更新 latents（30 步）。
4. latents `/scaling_factor` → VAE(dev0) → RGB → `(x/2+0.5).clip*255` → 保存 PNG。

---

## 7. 性能数据

> 以下为 Atlas 300I Duo（310P3）实测数据（1024×1024，30 步，fp16，单图，CFG x2）。数值以推理脚本端到端打印为准。

| 指标 | 300I Duo 耗时 |
|---|---|
| 文本编码 (CLIP-L+CLIP-G, dev1) | _（待运行后填入）_ |
| UNet 总计 (30 步, CFG x2) | _（待运行后填入）_ |
| UNet 单步平均 | _（待运行后填入）_ |
| VAE 解码 (dev0) | _（待运行后填入）_ |
| 端到端 | _（待运行后填入）_ |

---

## 8. 精度对齐

提供 `align_sdxl_base_1_0.py`，在固定 prompt/seed/latent 下对 HF(CPU fp32) 与 MindIR(Ascend fp16) 做整图端到端比对：用 seeded torch generator 生成共享初始 latents，分别喂给 HF `StableDiffusionXLPipeline`（CPU）与 MSLite 推理，比较输出图像的 `max_abs`、`mean_abs`、`PSNR`。

```bash
python align_sdxl_base_1_0.py \
  --mindir-dir ./sdxl_base_1_0_onnx \
  --model-dir  ./stable-diffusion-xl-base-1.0 \
  --prompt "A cat holding a sign that says hello world, highly detailed, 4k" \
  --seed 0 --steps 30 --guidance 5.0
```

> HF CPU baseline 对 1024×1024 / 30 步较慢；如需快速验证可减小 `--steps`（两端始终使用相同步数与设置）。fp16 下图像 `PSNR` 通常 ≥ 30dB，`max_abs` 通常 < ~0.1（[0,1] 归一化范围）。

常见误差源与对策：

- **fp16 溢出**：UNet/VAE 已默认 `force_fp16`；若偏差大，可对 unet config 改 `force_fp32` 重新转换。
- **text_encoder 取错层**：SDXL 使用 `hidden_states[-2]`（倒数第二层），不是最后一层；导出 wrapper 已遵循此约定。
- **time_ids 不一致**：默认 `[1024,1024,0,0,1024,1024]`（原图/目标尺寸均等于生成尺寸，无裁剪）；若改 `--height/--width` 须同步重导出。
- **初始噪声不一致**：精度比对时两端 `--seed` 必须相同；align 脚本已用共享 latents npy 保证一致。

---

## 9. 常见问题

### 1) 转换时报 `ge.proto.ModelDef exceeded maximum protobuf size`

UNet 权重 ~10GB，`ascend_oriented` 转换会把权重外置到 `*_variables/`，日志打印此信息**不影响**产物。

### 2) UNet 图编译耗时长

SDXL UNet 数百层（含 transformer_layers_per_block 高达 10）在 310P3 上的图编译可能需要数十分钟，属正常现象；固定 shape 已是最快路径。

### 3) `Only support CustomAscend, but got ...`

MindIR 必须用 `--optimize=ascend_oriented` 转换（保留 Custom 算子映射）。请勿用 `--optimize=general`。

### 4) 单卡 OOM

UNet(~10GB) + 激活在单卡 ~44GB 内可运行；若同时加载两个文本编码器导致 OOM，确保文本编码器在 `--text-device` 另一张卡（默认 dev1）。

### 5) encoder_hidden_states 维度不对（非 2048）

SDXL 的 `encoder_hidden_states` 必须是 CLIP-L(768) 与 CLIP-G(1280) 的 concat = 2048。若只用了单个编码器或取了最后一层（非倒数第二层），维度会错。导出 wrapper 已严格按 `hidden_states[-2]` concat。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [SDXL base 模型页（HuggingFace）](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL 论文（Scaling up Diffusion Models）](https://arxiv.org/abs/2307.01952)
- [Diffusers SDXL 文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)

---

## 11. 许可证

SDXL base 1.0 遵循 [Creative Commons Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)。本教程遵循相应依赖的许可证要求。
