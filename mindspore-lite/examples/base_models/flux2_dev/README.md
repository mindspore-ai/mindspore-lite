# FLUX.2-dev ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `black-forest-labs/FLUX.2-dev`（~32B MMDiT 文生图扩散模型）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo / 310P3）上完成端到端文生图推理与精度对齐。

FLUX.2-dev 由三部分组成：**Mistral3** 文本编码器（~24B，48GB）、**MMDiT Transformer**（去噪器，8 double + 48 single blocks，inner_dim 6144，64GB）、以及 32 通道 **VAE** 解码器。

### 关键约束与策略

FLUX.2-dev 体量远超单张 300I Duo（44GB）：

| 组件 | 体积(bf16) | 单卡(44GB) | 部署策略 |
|---|---|---|---|
| transformer | 64GB（~32B） | ❌ 超卡 | **流水线并行 2 卡**：导出期拆成 part0/part1 两个 ~32GB MindIR，分别跑 dev0/dev1 |
| Mistral3 文本编码器 | 48GB（~24B） | ❌ 超卡 | **CPU 推理**（每 prompt 一次），不导 MindIR |
| VAE | 0.34GB | ✅ | 单卡 MindIR |

> Mistral3（48GB）同样超单卡；因其每条 prompt 仅运行一次，工程上拆 2 卡收益不匹配，故在 CPU 上用 transformers 直接运行，与精度基线一致。如需全 MindIR，可按 transformer 同样方式拆分。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- 昇腾环境：MindSpore Lite 2.9.0 + CANN 8.5.0 + Ascend 驱动
- **Atlas 300I Duo（310P3）至少 3 张卡**：transformer part0（dev0）+ part1（dev1）+ VAE（dev0 或 dev2）
- 主机内存 ≥ 128GB（Mistral3 CPU 推理 + transformer 转换）

### 依赖版本（建议）

| 软件包 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.9.0（CPU 即可，用于导出/对齐/Mistral3） |
| diffusers | 0.38.0（原生支持 `Flux2Transformer2DModel`、`AutoencoderKLFlux2`） |
| transformers | 5.9.0（原生支持 `Mistral3ForConditionalGeneration`） |
| onnx | 1.19.1 |
| mindspore-lite | 2.9.0 |
| CANN | 8.5.0 |

```bash
pip install torch==2.9.0 diffusers==0.38.0 transformers==5.9.0 onnx==1.19.1 onnxruntime
source /home/yf/env.sh
```

---

## 2. 模型下载

从 ModelScope 下载 diffusers 格式权重（transformer 64.5GB + Mistral3 48GB + VAE 0.34GB，共 ~113GB）：

```bash
python -c "from modelscope import snapshot_download; \
  print(snapshot_download('black-forest-labs/FLUX.2-dev', cache_dir='/home/yf/modelscope_cache'))"
ln -sfn /home/yf/modelscope_cache/black-forest-labs/FLUX___2-dev ./FLUX.2-dev
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本产出 3 个 ONNX：

1. **transformer_part0**（`flux2_transformer_part0.onnx`）：embedders + temb/mods + RoPE + 8 double + 前 16 single blocks → 中间 hidden_states。约 32GB。
2. **transformer_part1**（`flux2_transformer_part1.onnx`）：重算 temb/mods/RoPE + 后 32 single blocks + norm_out + proj_out → noise_pred。约 32GB。
3. **vae_decoder**（`flux2_vae_decoder.onnx`）：32 通道 latent → RGB 图像。

### Transformer 拆分原理

- 拆分点选在 single blocks 中间（`--split-single 16`）：part0 = 8 double + 16 single，part1 = 32 single，两者参数量大致均衡（各 ~16B / ~32GB fp16），单卡可容纳。
- **导出期即拆**（而非转换期），每个 ONNX ~32GB，使主机 ONNX→MindIR 转换内存可控（~40GB）。
- part1 **重算** temb/modulation/RoPE（从 timestep/guidance/ids），数值与原整图完全一致（已验证：tiny 模型上 part0+part1 与完整模型 max diff = 0.0），跨卡每步仅传递一个 `hidden_states` 张量。
- Mistral3 不导出（CPU 运行）。

### 自定义算子策略

与 FLUX.1 一致：注意力替换为 CANN `PromptFlashAttention` Custom 节点（BNSD、`sparse_mode=0`、无 mask），q/k-norm（`nn.RMSNorm`）替换为 CANN `RmsNorm` Custom 节点。monkeypatch diffusers FLUX.2 的 `dispatch_attention_fn` 与 `torch.nn.RMSNorm.forward`。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/flux2_dev

python export_flux2_dev_onnx.py \
  --model-id ./FLUX.2-dev \
  --output-dir ./flux2_dev_onnx \
  --resolution 1024 1024 \
  --seq-len 512 \
  --split-single 16 \
  --dtype fp16
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 本地 diffusers 权重目录 | `./FLUX.2-dev` |
| `--output-dir` | ONNX 输出目录 | `./flux2_dev_onnx` |
| `--resolution` | 输出图像分辨率 H W | `1024 1024` |
| `--seq-len` | Mistral3 token 序列长度 | `512` |
| `--split-single` | part0 包含的 single block 数（其余归 part1） | `16`（均衡 ~32GB/卡） |
| `--dtype` | 导出精度 | `fp16` |
| `--components` | 导出子集（transformer,vae） | 全部 |

### 模型架构参数

| 参数 | 值 |
|------|------|
| inner_dim | 6144（48 heads × 128） |
| num_layers (double) | 8 |
| num_single_layers | 48 |
| joint_attention_dim (Mistral3) | 15360（3 层 × 5120 堆叠） |
| in_channels (packed latent) | 128（32 × 2×2 patchify） |
| guidance_embeds | True |
| VAE latent channels | 32（VAE 内含 BatchNorm） |
| RoPE ids 维度 | 4（axes_dims [32,32,32,32]） |

---

## 4. ONNX 模型结构说明

transformer 两个子图使用自定义算子（**不支持 ONNX Runtime 直接推理**，需 `converter_lite` 转 MindIR）。VAE 为标准算子图。

### 自定义算子（transformer part0 + part1）

| 算子 | 说明 |
|------|------|
| PromptFlashAttention | 全双向注意力（double 块 8 + single 块 48），无 mask |
| RmsNorm | q/k-norm，fp32 累加 |

---

## 5. MindSpore Lite 转换

```bash
cd ./mindspore-lite/examples/base_models/flux2_dev
CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# transformer part0 (-> dev0)
$CONV --fmk=ONNX --modelFile=./flux2_dev_onnx/flux2_transformer_part0.onnx \
  --outputFile=./flux2_dev_onnx/flux2_transformer_part0 \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/flux2_transformer_part0.config

# transformer part1 (-> dev1)
$CONV --fmk=ONNX --modelFile=./flux2_dev_onnx/flux2_transformer_part1.onnx \
  --outputFile=./flux2_dev_onnx/flux2_transformer_part1 \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/flux2_transformer_part1.config

# vae
$CONV --fmk=ONNX --modelFile=./flux2_dev_onnx/flux2_vae_decoder.onnx \
  --outputFile=./flux2_dev_onnx/flux2_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/flux2_vae_decoder.config
```

### config 说明

- `flux2_transformer_part0.config`：固定 4096 latent token(128ch) + 512 Mistral3 token(15360) + 4D ids，`force_fp16`。
- `flux2_transformer_part1.config`：固定 `hidden_mid:1,4608,6144`（4096 img + 512 txt），`force_fp16`。
- `flux2_vae_decoder.config`：固定 `latents:1,32,128,128`，`force_fp16`。

> 转换日志中 `protobuf size` / `ge.proto.ModelDef exceeded maximum protobuf size` 等 warning（权重外置化），**不影响产物**（`*_graph.mindir` + `*_variables/`）。part0/part1 各 ~32GB，图编译可能耗时数十分钟。

---

## 6. MindSpore Lite 推理（流水线并行）

```bash
cd ./mindspore-lite/examples/base_models/flux2_dev

python infer_flux2_dev_mslite.py \
  --transformer-part0 ./flux2_dev_onnx/flux2_transformer_part0_graph.mindir \
  --transformer-part1 ./flux2_dev_onnx/flux2_transformer_part1_graph.mindir \
  --vae-model         ./flux2_dev_onnx/flux2_vae_decoder_graph.mindir \
  --model-dir ./FLUX.2-dev \
  --prompt "A cat holding a sign that says hello world" \
  --seed 0 --steps 28 --guidance 3.5 \
  --part0-device 0 --part1-device 1 --vae-device 0 \
  --output ./flux2_output.png
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--transformer-part0/part1` | transformer 两半 MindIR | 必填 |
| `--vae-model` | VAE MindIR | 必填 |
| `--model-dir` | diffusers 权重目录（Mistral3 + tokenizer + scheduler + VAE bn 统计） | `./FLUX.2-dev` |
| `--prompt` | 文本提示词 | 必填 |
| `--seed` | 随机种子 | `0` |
| `--steps` | 去噪步数 | `28` |
| `--guidance` | guidance scale | `3.5` |
| `--seq-len` | Mistral3 序列长度（须与导出一致） | `512` |
| `--part0-device` / `--part1-device` | transformer 两半所在卡 | `0` / `1` |
| `--vae-device` | VAE 所在卡 | `0` |
| `--output` | 输出 PNG | `./flux2_output.png` |

### 流程

1. **Mistral3（CPU）** 编码 prompt（取第 10/20/30 层 hidden states 堆叠 → 15360 维）→ `encoder_hidden_states` + `text_ids`。
2. numpy 固定 seed 生成噪声 [1,128,64,64]（patchified 空间）→ pack → 4D `img_ids`。
3. 逐 timestep（FlowMatchEuler 调度，CPU）：**part0(dev0) → hidden_mid → 拷贝到主机 → part1(dev1) → noise_pred** → `scheduler.step`（28 步）。
4. unpack → BN 反归一化（`vae.bn` 统计）→ unpatchify → **VAE(dev) decode** → RGB 图像 → 保存。

> 每步 dev0→主机→dev1 一次张量中转（`hidden_mid` ~56MB），延迟可接受。

---

## 7. 性能数据

> 以下为 Atlas 300I Duo（310P3）实测数据（1024×1024，28 步，fp16，单图，流水线并行 2 卡）。数值以推理脚本端到端打印为准。

| 指标 | 300I Duo 耗时 |
|---|---|
| 文本编码 (Mistral3, CPU) | _（待运行后填入）_ |
| Transformer 总计 (part0+part1, 28 步) | _（待运行后填入）_ |
| Transformer 单步平均 | _（待运行后填入）_ |
| VAE 解码 | _（待运行后填入）_ |
| 端到端 | _（待运行后填入）_ |

---

## 8. 精度对齐

`align_flux2_dev.py` 对 transformer 拆分与 VAE 做数值比对（每组件 1 次 forward）：

```bash
python align_flux2_dev.py \
  --model-dir ./FLUX.2-dev \
  --mindir-dir ./flux2_dev_onnx \
  --prompt "A cat holding a sign that says hello world" --seed 0
```

输出 transformer `noise_pred`（part0+part1 MindIR vs HF 完整模型）与 VAE image 的 `max_abs`/`mean_abs`/`max_rel`。Mistral3 两端均在 CPU 运行，无需比对。fp16 下 `max_abs` 通常 < ~1e-2。

常见误差源与对策：

- **fp16 溢出**：默认 `force_fp16`；若偏差大，可对 transformer config 改 `force_fp32` 重转。
- **序列长度/分辨率不一致**：`--seq-len` 与 `--resolution` 须与导出一致。
- **初始噪声不一致**：比对时两端 `--seed` 相同。
- **拆分点不一致**：推理 `--split-single` 无需指定（拆分在导出期固化），但须使用匹配的 part0/part1 MindIR。

---

## 9. 常见问题

### 1) transformer 单卡 OOM

FLUX.2 transformer 64GB 不可单卡。必须使用 part0/part1 两半（`--part0-device`/`--part1-device` 不同卡）。若某半仍 OOM，减小 `--split-single`（更多 single 归 part1）重新导出。

### 2) Mistral3 CPU 推理慢 / 内存高

Mistral3 ~24B，CPU 单条 prompt 编码需数分钟、~48GB 内存。属预期（每 prompt 一次）。如需加速/上卡，可按 transformer 同样方式拆 2 卡导出 MindIR。

### 3) 主机转换 64GB OOM

导出期已拆成 2 个 ~32GB ONNX，分别转换（各 ~40GB 内存）。若仍紧张，先释放其他进程内存。

### 4) `ge.proto.ModelDef exceeded maximum protobuf size`

权重外置化信息，**不影响产物**。

### 5) 范围限定

本教程仅覆盖 **T2I 生成**路径（无 reference-token KV cache，即 FLUX.2 的图像理解/编辑输入路径不在范围）。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [FLUX.2-dev 模型页（ModelScope）](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev)
- [Black Forest Labs](https://blackforestlabs.ai/)
- [Diffusers FLUX.2 文档](https://huggingface.co/docs/diffusers/api/pipelines/flux2)

---

## 11. 许可证

FLUX.2-dev 遵循相应许可证（详见模型页 LICENSE.md）。本教程遵循相应依赖的许可证要求。
