# Wan2.1-T2V-1.3B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) 文本生成视频模型按网络结构拆分导出为 ONNX，转换为 MindSpore Lite MindIR，并在 **Huawei Ascend Atlas 300I Duo** 上完成端到端推理部署。

默认按官方生成规格（**480×832、81 帧（4k+1）、5 秒 @16fps、UniPC 调度器（权重自带 flow_shift=3.0）、50 步、CFG 5.0**）导出与推理；帧数/分辨率可按需指定（约束见第 2 节），变更后需重新导出并转换对应子模型。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.8.0 |
| onnx | 1.19.1 |
| onnxsim | 0.6.5 |
| numpy | 1.26.4 |
| transformers | 4.57.0 |
| diffusers | 0.38.0 |
| mindspore-lite | 2.11.0 |
| CANN | 9.0.1 |
| imageio + imageio-ffmpeg | 任意（保存 mp4） |

---

## 2. 模型导出 ONNX

按结构拆分为三个固定 shape 子模型（下表为 81 帧官方规格）：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `wan_text_encoder` (UMT5-XXL) | input_ids[1,512], attention_mask[1,512] | last_hidden_state[1,512,4096] |
| `wan_transformer` (DiT 1.3B) | hidden_states[1,16,21,60,104], timestep[1], encoder_hidden_states[1,512,4096] | noise_pred[1,16,21,60,104] |
| `wan_vae_decoder` (3D VAE) | latents[1,16,13,60,104] | video[1,3,49,480,832] |

视频超过 49 帧时，VAE 导出为 **13-latent 段解码器**（`wan_vae_decoder_13lat.onnx`），推理脚本自动按重叠分段解码并拼接；不超过时导出为单趟全量解码（`wan_vae_decoder.onnx`），输入 shape 跟随帧数。49 帧的由来：VAE 时间上采样 4 倍，13 个 latent 帧单趟最多解码 4×13−3 = 49 个视频帧（−3 为因果卷积的时间重叠）；官方规格 81 帧对应 21 个 latent 帧（(81−1)/4+1），故需分段。

### 导出命令

```bash
cd mindspore-lite/examples/base_models/wan2_1_t2v_1_3b

python export_wan2_1_t2v_1_3b_onnx.py \
  --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --output-dir ./wan2_1_t2v_1_3b_onnx \
  --height 480 --width 832 --num-frames 81 --max-seq-len 512 \
  --dtype float32
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-dir` | 权重目录 | 必填 |
| `--output-dir` | ONNX 输出目录 | `./wan2_1_t2v_1_3b_onnx` |
| `--parts` | 导出哪些子模型 | `text,transformer,vae` |
| `--height/--width` | 视频分辨率（16 的倍数） | `480/832` |
| `--num-frames` | 视频帧数（4k+1） | `81` |
| `--max-seq-len` | UMT5 最大序列长度 | `512` |
| `--dtype` | 导出精度（float32，便于转换） | `float32` |
| `--pfa-inner-precise` | PFA 计算模式（0=高精度；1/4 更快但精度不达标，不建议） | `0` |
| `--pfa-layout` | PFA 布局 | `BNSD` |

### 产出文件

```text
./wan2_1_t2v_1_3b_onnx/
├── wan_text_encoder.onnx        (+ 外置权重)
├── wan_transformer_pfa.onnx     (PFA 注入；+ 外置权重 ~5.7G)
└── wan_vae_decoder_13lat.onnx   (+ 外置权重)
```

（`--num-frames` 不超过 49 时，VAE 产出为 `wan_vae_decoder.onnx`。）

### 导出实测日志（dynamo）

```log
[export] UMT5 text encoder ...
[export] saved ./wan2_1_t2v_1_3b_onnx/wan_text_encoder.onnx
[export] Wan transformer (DiT) ...
[export] PFA injection (BNSD): 60 chains -> Custom=60 Softmax=0 MatMul=303 Transpose=423
[export] saved ./wan2_1_t2v_1_3b_onnx/wan_transformer_pfa.onnx
[export] Wan VAE decoder ...
[export] saved ./wan2_1_t2v_1_3b_onnx/wan_vae_decoder_13lat.onnx
[export] done -> ./wan2_1_t2v_1_3b_onnx
# transformer/VAE 均为 0 符号维全静态图（dynamo 导出）
```

---

## 3. MindSpore Lite 转换（ONNX → MindIR）

`converter_lite` 为 MindSpore Lite 提供的离线转换工具。

```bash
Convert=./mindspore-lite-2.11.0-linux-aarch64/tools/converter/converter/converter_lite

# 文本编码器
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_text_encoder.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_text_encoder.config

# transformer（PFA 注入图）
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_transformer_pfa.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_transformer_pfa \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_transformer.config

# VAE 解码器（13-latent 段解码器）
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_vae_decoder_13lat.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_vae_decoder_13lat \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_vae_decoder.config
```

### 配置文件

`configs/wan_text_encoder.config`：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,512;attention_mask:1,512"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

`configs/wan_transformer.config`：

```ini
[acl_build_options]
input_format="ND"
input_shape="hidden_states:1,16,21,60,104;timestep:1;encoder_hidden_states:1,512,4096"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

`configs/wan_vae_decoder.config`：

```ini
[acl_build_options]
input_format="ND"
input_shape="latents:1,16,13,60,104"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

`configs/wan_transformer_tp.config`（TP rank 图转换用，见 4.1 节）：

```ini
[ascend_context]
plugin_custom_ops=All

[ge_session_options]
ge.constLifecycle=session
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.precision_mode=force_fp16
ge.externalWeight=2
```

说明：

- `force_fp16`：300 IDUO Cube 算子仅支持 fp16 矩阵乘，fp16 转换性能显著更好（见第 5 节）；文本编码器保持 `force_fp32`（UMT5 数值范围大，fp16 下易溢出，且其耗时占比极小）。
- `plugin_custom_ops=All`：将 PFA Custom 节点映射到 CANN 内置算子，必须保留。
- `wan_transformer_tp.config` 无 `acl_build_options`：TP rank 图以 `--optimize=none` 转为纯 MindIR（无离线 OM），其 `Custom(AllReduce)` 节点只能在推理期建立 HCCL world 后由 GE 在线编译，会话选项（force_fp16 等）由推理脚本自动生成携带。
- **修改分辨率/帧数后**，`input_shape` 的 latent 维需同步更新：
  - transformer：`1,16,(帧数-1)/4+1,高/8,宽/8`（比如81 帧，转换后第三维度为21）
  - VAE：超过 49 帧时，shape为 `1,16,13,高/8,宽/8`；不超过则为 `1,16,(帧数-1)/4+1,高/8,宽/8`

### 产出说明

权重外置与否取决于体积：超过阈值的产出 `*_graph.mindir` + `*_variables/`（transformer、text encoder），较小的产出单文件 `*.mindir`（VAE）。推理脚本对两种命名都兼容（见第 4 节）。

```text
./wan2_1_t2v_1_3b_onnx/
├── wan_text_encoder_graph.mindir     + wan_text_encoder_variables/
├── wan_transformer_pfa_graph.mindir  + wan_transformer_pfa_variables/
└── wan_vae_decoder_13lat.mindir      (单文件)
```

### 转换实测日志

```log
# 文本编码器 / transformer（PFA 注入图）/ VAE 解码器
CONVERT RESULT SUCCESS:0
# 期间出现的 Do constant fold failed / InferShapeByNNACL node_view 告警为良性，可忽略
```

---

## 4. MindSpore Lite 推理

```bash
python infer_wan2_1_t2v_1_3b_mslite.py \
  --mindir-dir ./wan2_1_t2v_1_3b_onnx \
  --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A cat walking on a beach, cinematic, 4k." \
  --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 \
  --device-a 0 --device-b 1 \
  --output wan_output.mp4
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir-dir` | MindIR 目录（含 3 个子模型 mindir） | 必填 |
| `--model-dir` | 权重目录（tokenizer + scheduler + vae config） | 必填 |
| `--prompt/--negative-prompt` | 文本提示 | 见默认 |
| `--height/--width/--num-frames` | 必须与导出/转换一致 | `480/832/81` |
| `--num-inference-steps` | 去噪步数 | `50` |
| `--guidance-scale` | CFG 强度 | `5.0` |
| `--seed` | 噪声种子 | `42` |
| `--device-a/--device-b` | 双芯 CFG 并行（选两颗空闲芯片） | `0/1` |
| `--tp-size` | `2` = TP2×CFG 双卡四芯路径（见 4.1 节） | `1` |
| `--tp-devices` | 四芯列表：cond_r0,cond_r1,uncond_r0,uncond_r1（每对须同卡） | `0,1,2,3` |
| `--output` | 输出视频 | `wan_output.mp4` |

说明：

- 固定 shape 约束：推理侧 `--height/--width/--num-frames` 必须与导出一致；变更需重新导出+转换（`input_shape` 公式见第 3 节）。
- **双芯 CFG 并行**：每步去噪的 cond/uncond 两个分支分别跑在两颗芯片上，较单芯串行提速约 2×。只有一颗空闲芯片时可将 `--device-a` 与 `--device-b` 设为同一颗。

三个子模型的芯片分配（默认单卡双芯）：

| 子模型 | 芯片 |
| --- | --- |
| 文本编码器 (UMT5) | `--device-a`（单芯） |
| Transformer (DiT) | cond 分支 → `--device-a`，uncond 分支 → `--device-b`，两芯并行 |
| VAE 解码器 | ≤49 帧单趟解码 → 仅 `--device-a`；>49 帧分段解码 → 两芯并行 |

### 推理实测日志（81 帧 / 50 步，官方规格）

```log
[load] models built in 26s
[text] 1.14s
[denoise] 10/50 (436s elapsed)
[denoise] 50/50 (2151s elapsed)
[denoise] total 2151s
[vae] 2-seg decode (dual-chip) 20.0s -> 81 frames
[infer] saved 81 frames @16fps -> wan_output.mp4 (5.06s, 480x832)

--- Performance ---
  Model load:                      25.6 s
  Text encode (UMT5):              1.14 s
  Denoise (50 steps):    2150.9 s
  VAE decode:                       20.0 s
  End-to-end:                    2197.5 s (36.6 min)
# 输出 wan_output.mp4：81 帧 @ 480x832，像素范围正常
```

### 4.1 进阶：TP2×CFG 双卡四芯推理

上文默认模式（单卡双芯）是**精度严格验收的版本**（输出与参考实现逐位一致）。如需进一步压缩单步时延，可启用 TP2×CFG：4 颗芯片组成 2 个独立 HCCL world（cond/uncond 各一个），world 内 transformer 按 TP=2 切分（行并行权重 shard + PFA head 切片 + `Custom(AllReduce)` 归约），CFG 合成与 UniPC 更新仍在主进程 CPU 侧完成。与默认模式的差异（三步都要带 TP 参数）：

```bash
# ① 导出：每个 rank 一次（行并行 shard + AllReduce 注入）
python export_wan2_1_t2v_1_3b_onnx.py --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --output-dir ./wan2_1_t2v_1_3b_onnx --tp-size 2 --rank 0
python export_wan2_1_t2v_1_3b_onnx.py --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --output-dir ./wan2_1_t2v_1_3b_onnx --tp-size 2 --rank 1

# ② 转换：rank 图为纯 MindIR（optimize=none），由推理期在线 GE 编译
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_transformer_pfa_rank0.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_transformer_pfa_rank0 \
  --optimize=none --saveType=MINDIR --configFile=./configs/wan_transformer_tp.config
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_transformer_pfa_rank1.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_transformer_pfa_rank1 \
  --optimize=none --saveType=MINDIR --configFile=./configs/wan_transformer_tp.config

# ③ 推理：--tp-devices 为 cond_r0,cond_r1,uncond_r0,uncond_r1
python infer_wan2_1_t2v_1_3b_mslite.py \
  --mindir-dir ./wan2_1_t2v_1_3b_onnx --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --tp-size 2 --tp-devices 0,1,2,3 --output output_tp.mp4
```

（text encoder 与 VAE 不做 TP 切分，沿用第 2/3 节转换的产物；`--num-frames/--height/--width` 等其余参数用法不变。）

三个子模型的芯片分配（TP2×CFG 双卡四芯）：

| 子模型 | 芯片 |
| --- | --- |
| 文本编码器 (UMT5) | `--tp-devices` 第 1 颗（单芯） |
| Transformer (DiT) | 4 芯全部：cond world = 前 2 颗，uncond world = 后 2 颗，world 内按 TP=2 切分 |
| VAE 解码器 | ≤49 帧单趟解码 → 第 1 颗；>49 帧分段解码 → 前 2 颗并行 |

说明与约束：

- **每个 world 的 2 芯必须在同一张卡上**（如 `0,1` 与 `2,3`），HCCL 走卡内 HCCS 直连。
- **精度口径**：TP 模式输出与默认双芯模式**未达逐位一致**（`AllReduce` 归约顺序引入微小数值差异）；81 帧实测 corr 0.9913 / PSNR 29.85 dB，目检无可感知差异。默认双芯模式保持逐位一致口径。

## 5. 性能数据

**实测（81 帧官方规格）**：

| 指标 | 单卡双芯 | TP2×CFG 双卡四芯 |
| --- | ---: | ---: |
| 模型加载 | 25.6 s | 186.8 s |
| 文本编码 (UMT5) | 1.1 s | 1.2 s |
| Transformer 单步（稳态） | 43.0 s | 25.1 s |
| 去噪总计（50 步） | 2150.9 s | 1386.7 s |
| VAE 解码 | 20.0 s | 19.2 s |
| **端到端** | **2197.5 s（36.6 min）** | **1593.9 s（26.6 min）** |

> 四芯列含一次性开销：模型加载含首次运行 rank 图在线 GE 编译（约 187 s，产生编译缓存后明显缩短）；去噪首步约 155 s 为 HCCL world 冷启动。

---

## 6. 常见问题

1. **转换报 `context is a null pointer` / `InferShapeAndType for node ... failed`。**
   - 原因：ONNX 图含符号维（`unk__*`），GE 无法推断 shape。legacy 导出器对含 `.size()` 驱动 reshape 的图会留符号维。
   - 解决：transformer/VAE 用 `dynamo=True` 导出（脚本默认），产出 0 符号维全静态图。小模型可用 `onnxsim.simplify()` 修复。
2. **转换报 `i: 3 out of range ... ValueNode<If>`。**
   - 原因：图中有控制流 `If` 节点。来源：`F.scaled_dot_product_attention`（后端选择 If）、`torch.squeeze(size=1 维)`（条件 squeeze If）。
   - 解决：脚本已将 SDPA 替换为显式 BatchMatMul+Softmax+BatchMatMul、squeeze 替换为 `x[:, 0]`（均无 If）。
3. **导出报 `UnsupportedOperatorError: aten::_upsample_nearest_exact2d`。**
   - 原因：Wan VAE 上采样用的内部 op 无 legacy 符号。
   - 解决：脚本 `_patch_wan_upsample` 将 2× nearest-exact 替换为 `repeat_interleave(2)`（整数因子下数学等价）。
4. **VAE 解码输出帧数远少于预期（时间维未上采样）。**
   - 原因：单 pass 未执行时间上采样（流式 `WanResample.upsample3d` 仅在第二次调用才跑 `time_conv`）。
   - 解决：脚本 `_patch_wan_resample_singlepass` 让 `upsample3d` 始终执行 `time_conv`+交错，并丢弃时间 index-1 匹配 `2t-1` 装配。
5. **导出 transformer 时 CPU 内存暴涨/OOM。**
   - dynamo 导出 + `external_data=True` 已外置权重；若仍紧张，确认 `do_constant_folding` 关闭（legacy 路径）或减少并发进程。
6. **修改分辨率/帧数后转换或推理报 shape 不匹配。**
   - 固定 shape 部署：需重新导出 ONNX、同步修改 `configs/*.config` 的 `input_shape`（公式见第 3 节）、重新转换。
7. **加载 MindIR 报 `External data path escapes model directory`。**
   - mslite 的外置权重安全检查拒绝软链：`*_variables/` 目录必须是真实目录（目录内的数据文件可以硬链接）。
8. **transformer 转换报内存需求超百 GB / OOM。**
   - 原因：使用了未注入 PFA 的 ONNX（如旧版脚本导出的 `wan_transformer.onnx`）。
   - 解决：使用本目录导出脚本重新导出（自动产出 `wan_transformer_pfa.onnx`）。
9. **推理偶发 `AICore kernel execute failed`（如 `te_cast` 算子）。**
   - 个别芯片的瞬时故障，与模型无关；换一颗芯片重跑即可。

---

## 7. 参考资源与许可证

- 上游模型：<https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- 本目录脚本遵循 MindSpore Lite 仓库许可证；上游模型权重许可证以其仓库为准（Apache-2.0）。
