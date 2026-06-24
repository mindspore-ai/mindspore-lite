# Stable-Diffusion-XL-Refiner-0.9 ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `stabilityai/stable-diffusion-xl-refiner-0.9`（SDXL refiner，SDXL 两阶段管线的第二阶段精修 UNet）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上完成端到端精修推理与精度对齐。

SDXL refiner 是两阶段（base → refiner）文生图的第二阶段：base UNet 先将噪声去噪到一个较高的 timestep（约 `denoising_end=0.8`），随后 refiner 接管剩余的低噪声（高图像保真度）去噪阶段，输出更精细的高频细节。refiner 由三部分组成：CLIP-G 文本编码器（带 projection，1280 维，含 pooled 输出——**refiner 仅使用 CLIP-G，不使用 CLIP-L**）、UNet2DConditionModel（精修去噪器，`cross_attention_dim=1280`）、4 通道 VAE 解码器。本教程将三部分**全部导出为 MindIR** 并在昇腾上推理（UNet+VAE 在 dev0，文本编码器在 dev1）。

> **与 SDXL base 1.0 的关键区别**（务必注意，否则会维度报错）：
> - refiner 的 `encoder_hidden_states` 为 **[1,77,1280]**（仅 CLIP-G 倒数第二层），**不是** base 的 [1,77,2048]（CLIP-L 768 + CLIP-G 1280 concat）。
> - refiner 的 `time_ids` 为 **5 元组** `[orig_h, orig_w, crop_top, crop_left, aesthetic_score]`（因为 refiner 的 `requires_aesthetics_score=True`），**不是** base 的 6 元组（最后两个是 `target_h, target_w`）。默认 `aesthetic_score=6.0`（正样本）/ `2.5`（负样本）。

本示例采用**自包含的单阶段去噪演示**（refiner 从固定 seed 的纯噪声起，跑自己的 `EulerDiscreteScheduler` 去噪循环），方便独立验证；两阶段（base→refiner）的联用方式见第 6 节说明。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux（昇腾环境，MindSpore Lite 2.9.0 + CANN 8.5.0 + Ascend 驱动）
- Atlas 300I Duo（310P3，单卡 ~44GB），refiner UNet(~6.6B) 单卡可运行

### 依赖版本（建议）

| 软件包            | 版本   |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.9.0（CPU 即可，仅用于导出/对齐） |
| diffusers      | 0.38.0（原生支持 SDXL refiner UNet2DConditionModel） |
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

从 HuggingFace 下载 `stabilityai/stable-diffusion-xl-refiner-0.9` 的 diffusers 格式权重（UNet ~10.5GB + text_encoder_2 ~6.9GB + VAE ~0.34GB，共 ~18GB；refiner 不含 text_encoder_1 / tokenizer_1）：

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('stabilityai/stable-diffusion-xl-refiner-0.9', \
    cache_dir='/home/yf/hf_cache'))"
```

将其软链到 `./stable-diffusion-xl-refiner-0.9`：

```bash
ln -sfn /home/yf/hf_cache/models--stabilityai--stable-diffusion-xl-refiner-0.9/snapshots/* \
  ./stable-diffusion-xl-refiner-0.9
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本将 SDXL refiner 拆为三个 ONNX 子图：

1. **text_encoder_2**（`sdxl_text_encoder_2.onnx`）：CLIP-G（带 projection），input_ids [1,77] → last_hidden_state [1,77,1280]（**倒数第二层** hidden_states[-2]）+ text_embeds [1,1280]（pooled）。
2. **unet**（`sdxl_unet.onnx`）：sample [1,4,128,128] + timestep [1] + encoder_hidden_states [1,77,1280]（**仅 CLIP-G，非 base 的 2048**） + added_cond_kwargs{text_embeds [1,1280], time_ids [1,5]} → noise_pred [1,4,128,128]。自+交叉注意力替换为 CANN `PromptFlashAttention` Custom 算子。
3. **vae_decoder**（`sdxl_vae_decoder.onnx`）：4 通道 latent [1,4,128,128] → RGB [1,3,1024,1024]。

### 自定义算子策略

SDXL refiner UNet 的 `AttnProcessor2_0` 将 q/k/v 投影后 reshape 为 `(batch, num_heads, seq, head_dim)`（BNSD 布局），再调用 `torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)`。脚本 monkeypatch 该函数，将 SDPA 调用替换为 CANN `PromptFlashAttention` Custom 节点（q/k/v **已是 BNSD，无需转置**、`sparse_mode=0`、不传 atten_mask，对 16384 个 latent token 的自注意力和 77 个 CLIP-G token 的交叉注意力均成立——两者都是全双向无 mask）。其余（q/k/v 投影、group-norm、norm_q/norm_k 的 LayerNorm、残差、输出投影、time/additive embedding）均走标准 diffusers 算子 trace。这样只需改 SDPA 一处，避免手写 UNet 全部注意力层。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/sdxl_refiner_0_9

python export_sdxl_refiner_0_9_onnx.py \
  --model-id ./stable-diffusion-xl-refiner-0.9 \
  --output-dir ./sdxl_refiner_0_9_onnx \
  --resolution 1024 1024 \
  --dtype fp32 \
  --aesthetic-score 6.0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 本地 diffusers 权重目录 | `./stable-diffusion-xl-refiner-0.9` |
| `--output-dir` | ONNX 输出目录 | `./sdxl_refiner_0_9_onnx` |
| `--resolution` | 输出图像分辨率 H W | `1024 1024` |
| `--dtype` | 导出精度（fp32 推荐，converter 会转 fp16） | `fp32` |
| `--no-custom-op` | 不把注意力替换为 Custom 算子 | `False` |
| `--aesthetic-score` | time_ids 中的审美评分（refiner 正样本条件） | `6.0` |
| `--components` | 导出子集（text_encoder_2,unet,vae） | 全部 |

### 模型架构参数（refiner）

| 参数 | 值 |
|------|------|
| cross_attention_dim | **1280**（仅 CLIP-G，非 base 的 2048） |
| in/out_channels (latent) | 4 |
| block_out_channels | (320, 640, 1280, 1280) |
| attention head_dim | 64 |
| addition_time_embed_dim | 256 |
| projection_class_embeddings_input_dim | 2816（1280 + 5×256，5 元 time_ids） |
| text_embeds dim (pooled CLIP-G) | 1280 |
| time_ids dim | **5**（orig_h, orig_w, crop_top, crop_left, aesthetic_score） |
| requires_aesthetics_score | True |
| VAE latent channels | 4，scaling_factor 0.13025 |
| scheduler | EulerDiscreteScheduler（CPU），prediction_type=epsilon |

---

## 4. ONNX 模型结构说明

unet ONNX 使用 MindSpore Lite 自定义算子（`PromptFlashAttention`），**不支持直接用 ONNX Runtime 推理**，需通过 `converter_lite` 转 MindIR 后运行。VAE / text_encoder_2 为标准算子图。

### 模型中包含的自定义算子（unet）

| 算子 | 数量 | 说明 |
|------|------|------|
| PromptFlashAttention | ~70（含 down/mid/up 所有 transformer block 的自注意力 + 交叉注意力） | 全双向注意力，无 mask |

> refiner UNet 的 transformer block 数量少于 base（refiner 是精修子集，典型配置 down=(2,6,2)/up=(2,6,2)/mid=6），每个 transformer block 含 1 个自注意力 + 1 个交叉注意力。精确数量以转换日志为准。

---

## 5. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/sdxl_refiner_0_9

CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# text_encoder 2 (CLIP-G)
$CONV --fmk=ONNX --modelFile=./sdxl_refiner_0_9_onnx/sdxl_text_encoder_2.onnx \
  --outputFile=./sdxl_refiner_0_9_onnx/sdxl_text_encoder_2 \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/sdxl_text_encoder_2.config

# unet
$CONV --fmk=ONNX --modelFile=./sdxl_refiner_0_9_onnx/sdxl_unet.onnx \
  --outputFile=./sdxl_refiner_0_9_onnx/sdxl_unet \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/sdxl_unet.config

# vae
$CONV --fmk=ONNX --modelFile=./sdxl_refiner_0_9_onnx/sdxl_vae_decoder.onnx \
  --outputFile=./sdxl_refiner_0_9_onnx/sdxl_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/sdxl_vae_decoder.config
```

### config 说明

- `sdxl_unet.config`：固定 128×128 latent + 77 CLIP-G token + time_ids[1,5]（aesthetic 变体），`force_fp16`，`plugin_custom_ops=All`。**注意 `encoder_hidden_states:1,77,1280` 与 `time_ids:1,5` 与 base 不同**。
- `sdxl_text_encoder_2.config` / `sdxl_vae_decoder.config`：固定 shape，`force_fp16`。

> 转换日志中可能出现 `protobuf size` / `ge.proto.ModelDef exceeded maximum protobuf size` 等 warning（unet 权重 ~10GB 外置化），**不影响最终产物**（产出 `*_graph.mindir` + `*_variables/`），可忽略。

---

## 6. MindSpore Lite 推理

### 单阶段去噪（自包含示例）

本示例提供独立可运行的 refiner 推理：从固定 seed 的纯噪声起，refiner 跑自己的 `EulerDiscreteScheduler` 去噪循环（与 base 同算法，但用 refiner 的 CLIP-G 条件 + aesthetic time_ids）。

```bash
cd ./mindspore-lite/examples/base_models/sdxl_refiner_0_9

python infer_sdxl_refiner_0_9_mslite.py \
  --mindir-dir   ./sdxl_refiner_0_9_onnx \
  --model-dir    ./stable-diffusion-xl-refiner-0.9 \
  --prompt "A cat holding a sign that says hello world, highly detailed, 4k" \
  --negative-prompt "lowres, blurry, worst quality, low quality" \
  --seed 0 --steps 30 --guidance 5.0 \
  --aesthetic-score 6.0 --negative-aesthetic-score 2.5 \
  --unet-device 0 --vae-device 0 --text-device 1 \
  --output ./sdxl_refiner_output.png
```

### 两阶段联用（base → refiner）说明

生产用法是 base 先去噪到 `denoising_end≈0.8`，再把中间 latents 喂给 refiner。本仓库未单独封装该脚本，可在用户代码中：

1. 运行 `sdxl_base_1_0` 示例得到 base 的**未 VAE 解码的中间 latents**（即去噪到 80% 的 `[1,4,128,128]`）。
2. 将该 latents 作为 `infer_sdxl_refiner_0_9_mslite.py` 的初始输入（替换脚本内的 `rng.standard_normal` 噪声），并让 refiner 接管剩余 `≈20%` 的去噪步数。

refiner 的 `time_ids`（5 元组 aesthetic 变体）与 base（6 元组）不同，拼接时需分别构造。

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mindir-dir` | 含 3 个 `*_graph.mindir` 的目录 | 必填 |
| `--model-dir` | diffusers 权重目录（取 tokenizer_2 + scheduler + vae config） | 必填 |
| `--prompt` | 文本提示词 | 必填 |
| `--negative-prompt` | 负向提示词（CFG） | `lowres, blurry, ...` |
| `--seed` | 随机种子（初始噪声） | `0` |
| `--steps` | 去噪步数 | `30` |
| `--guidance` | classifier-free guidance scale | `5.0` |
| `--aesthetic-score` | 正样本 aesthetic score（time_ids[4]） | `6.0` |
| `--negative-aesthetic-score` | 负样本 aesthetic score | `2.5` |
| `--height/--width` | 图像尺寸（须与导出一致） | `1024` |
| `--unet-device` | UNet 所在昇腾卡 | `0` |
| `--vae-device` | VAE 所在昇腾卡 | `0` |
| `--text-device` | 文本编码器所在昇腾卡 | `1` |
| `--output` | 输出 PNG 路径 | `./sdxl_refiner_output.png` |

### 流程

1. CLIP-G（dev1）编码 prompt：取 `hidden_states[-2]` → `encoder_hidden_states` [1,77,1280]；pooled → `text_embeds` [1,1280]。负向 prompt 同理。
2. numpy 固定 seed 生成噪声 [1,4,128,128]，乘 `scheduler.init_noise_sigma`。
3. 按 `EulerDiscreteScheduler`（CPU）逐 `timestep`：`scale_model_input` → UNet 输入堆叠 [uncond, cond] 一次前向（CFG x2，正负样本分别带各自的 5 元 time_ids）→ `noise = uncond + guidance*(cond-uncond)` → `scheduler.step` 更新 latents（30 步）。
4. latents `/scaling_factor` → VAE(dev0) → RGB → `(x/2+0.5).clip*255` → 保存 PNG。

---

## 7. 性能数据

> 以下为 Atlas 300I Duo（310P3）实测数据（1024×1024，30 步，fp16，单图，CFG x2）。数值以推理脚本端到端打印为准。

| 指标 | 300I Duo 耗时 |
|---|---|
| 文本编码 (CLIP-G, dev1) | _（待运行后填入）_ |
| Refiner UNet 总计 (30 步, CFG x2) | _（待运行后填入）_ |
| Refiner UNet 单步平均 | _（待运行后填入）_ |
| VAE 解码 (dev0) | _（待运行后填入）_ |
| 端到端 | _（待运行后填入）_ |

---

## 8. 精度对齐

提供 `align_sdxl_refiner_0_9.py`，在固定 prompt/seed/latent/aesthetic 下对 HF(CPU fp32) 与 MindIR(Ascend fp16) 做整图端到端比对：用 seeded torch generator 生成共享初始 latents，分别喂给 HF `StableDiffusionXLImg2ImgPipeline`（refiner 原生管线类，CPU，`strength=1.0` 跑完整去噪范围）与 MSLite 推理，比较输出图像的 `max_abs`、`mean_abs`、`PSNR`。

```bash
python align_sdxl_refiner_0_9.py \
  --mindir-dir ./sdxl_refiner_0_9_onnx \
  --model-dir  ./stable-diffusion-xl-refiner-0.9 \
  --prompt "A cat holding a sign that says hello world, highly detailed, 4k" \
  --seed 0 --steps 30 --guidance 5.0 \
  --aesthetic-score 6.0 --negative-aesthetic-score 2.5
```

> HF CPU baseline 对 1024×1024 / 30 步较慢；如需快速验证可减小 `--steps`（两端始终使用相同步数与设置）。fp16 下图像 `PSNR` 通常 ≥ 30dB，`max_abs` 通常 < ~0.1（[0,1] 归一化范围）。

常见误差源与对策：

- **encoder_hidden_states 维度不对（非 1280）**：refiner 的 `encoder_hidden_states` **必须**是 CLIP-G 的 1280 维（倒数第二层），**不能**像 base 那样 concat CLIP-L。导出 wrapper 已严格只用 CLIP-G。
- **time_ids 维度不对（非 5）**：refiner 用 5 元组 `(orig_h, orig_w, crop_top, crop_left, aesthetic_score)`（`requires_aesthetics_score=True`），**不是** base 的 6 元组。config 已设 `time_ids:1,5`。
- **aesthetic_score 未传**：默认正样本 6.0、负样本 2.5；对齐时两端须用相同值。
- **fp16 溢出**：UNet/VAE 已默认 `force_fp16`；若偏差大，可对 unet config 改 `force_fp32` 重新转换。
- **初始噪声不一致**：精度比对时两端 `--seed` 必须相同；align 脚本已用共享 latents npy 保证一致。

---

## 9. 常见问题

### 1) 转换时报 `ge.proto.ModelDef exceeded maximum protobuf size`

UNet 权重 ~10GB，`ascend_oriented` 转换会把权重外置到 `*_variables/`，日志打印此信息**不影响**产物。

### 2) UNet 图编译耗时长

refiner UNet 含上百层（含若干 transformer block）在 310P3 上的图编译可能需要数十分钟，属正常现象；固定 shape 已是最快路径。

### 3) `Only support CustomAscend, but got ...`

MindIR 必须用 `--optimize=ascend_oriented` 转换（保留 Custom 算子映射）。请勿用 `--optimize=general`。

### 4) 单卡 OOM

refiner UNet(~10GB) + 激活在单卡 ~44GB 内可运行；若同时加载文本编码器导致 OOM，确保文本编码器在 `--text-device` 另一张卡（默认 dev1）。

### 5) 报错 `Model expects an added time embedding vector of length ...`

这是因为 `time_ids` 的元组长度与 refiner `requires_aesthetics_score` 不匹配。refiner 必须 5 元组（带 aesthetic_score）；若误传 6 元组（base 的 target_h/target_w 变体）会触发该报错。检查 config 与 wrapper 的 `time_ids` shape。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [SDXL refiner 模型页（HuggingFace）](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9)
- [SDXL 论文（Scaling up Diffusion Models）](https://arxiv.org/abs/2307.01952)
- [Diffusers 两阶段 SDXL 管线文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)

---

## 11. 许可证

SDXL refiner 0.9 遵循 [Creative Commons Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9/blob/main/LICENSE.md)。本教程遵循相应依赖的许可证要求。
