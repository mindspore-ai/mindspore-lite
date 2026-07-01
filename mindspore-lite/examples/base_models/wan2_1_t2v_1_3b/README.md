# Wan2.1-T2V-1.3B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) 文本生成视频模型按网络结构拆分导出为 ONNX，转换为 MindSpore Lite MindIR，并在 **Ascend Atlas 300I Duo** 上完成端到端推理部署。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0+cpu |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| onnxsim | 0.6.5（小模型常量折叠，可选回退） |
| numpy | 1.26.4 |
| transformers | 5.9.0 |
| diffusers | 0.38.0 |
| mindspore-lite | 2.9.0 |
| CANN | 8.5.0 |
| imageio | 任意（保存 mp4） |

---

## 2. 模型导出 ONNX

按结构拆分为三个固定 shape 子模型：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `wan_text_encoder` (UMT5-XXL) | input_ids[1,512], attention_mask[1,512] | last_hidden_state[1,512,4096] |
| `wan_transformer` (DiT 1.3B) | hidden_states[1,16,4,60,104], timestep[1], encoder_hidden_states[1,512,4096] | noise_pred[1,16,4,60,104] |
| `wan_vae_decoder` (3D VAE) | latents[1,16,4,60,104] | video[1,3,13,480,832] |

固定配置：480×832、13 帧（4 个 latent 帧）。`timestep` 形状 `[1]`。若需其它分辨率/帧数，需重新导出并重新转换。

### 导出命令

```bash
cd mindspore-lite/examples/base_models/wan2_1_t2v_1_3b

python export_wan2_1_t2v_1_3b_onnx.py \
  --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --output-dir ./wan2_1_t2v_1_3b_onnx \
  --height 480 --width 832 --num-frames 13 --max-seq-len 512 \
  --dtype float32
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-dir` | 权重目录 | 必填 |
| `--output-dir` | ONNX 输出目录 | `./wan2_1_t2v_1_3b_onnx` |
| `--parts` | 导出哪些子模型 | `text,transformer,vae` |
| `--height/--width` | 视频分辨率（16 的倍数） | `480/832` |
| `--num-frames` | 视频帧数（4k+1） | `81`（命令行覆盖为 13） |
| `--max-seq-len` | UMT5 最大序列长度 | `512` |
| `--dtype` | 导出精度（float32，便于转换） | `float32` |
| `--no-custom-op` | （遗留）兼容保留；注意力始终以标准算子导出 | `False` |

### 产出文件

```text
./wan2_1_t2v_1_3b_onnx/
├── wan_text_encoder.onnx (+ wan_text_encoder_variables 外置权重)
├── wan_transformer.onnx  (+ wan_transformer.onnx.data 外置权重，dynamo 导出)
└── wan_vae_decoder.onnx  (+ wan_vae_decoder.onnx.data 外置权重，dynamo 导出)
```

### 导出实测日志（dynamo）

```log
[export] UMT5 text encoder ...
[export] saved ./wan2_1_t2v_1_3b_onnx/wan_text_encoder.onnx          (958K + 外置权重)
[export] Wan transformer (DiT) ...
[torch.onnx] Obtain model graph ... ✅   Translate the graph into ONNX ... ✅
[export] saved ./wan2_1_t2v_1_3b_onnx/wan_transformer.onnx           (7.8M + onnx.data 5.3G)
[export] Wan VAE decoder ...
[export] saved ./wan2_1_t2v_1_3b_onnx/wan_vae_decoder.onnx           (830K + onnx.data 280M)
[export] done -> ./wan2_1_t2v_1_3b_onnx
# 三个 onnx 均为 0 符号维全静态图；transformer 含 423 MatMul / 60 Softmax / 0 Custom / 0 If
```

---

## 3. MindSpore Lite 转换（ONNX → MindIR）

`converter_lite` 为 MindSpore Lite 提供的离线转换工具。

```bash
Convert=./mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# 文本编码器
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_text_encoder.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_text_encoder.config

# transformer
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_transformer.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_transformer \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_transformer.config

# VAE 解码器
$Convert --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_vae_decoder.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_vae_decoder.config
```

### 配置文件

`configs/wan_transformer.config`（其余两个结构相同，仅 `input_shape` 不同）：

```ini
[acl_build_options]
input_format="ND"
input_shape="hidden_states:1,16,4,60,104;timestep:1;encoder_hidden_states:1,512,4096"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

> 说明：`plugin_custom_ops=All` 在本教程无 Custom 节点时为空操作（注意力已走标准算子）；保留不影响。

### 产出说明

权重外置与否取决于体积：超过阈值的产出 `*_graph.mindir` + `*_variables/`（transformer、text encoder），较小的产出单文件 `*.mindir`（VAE）。推理脚本对两种命名都兼容（见第 5 节）。

```text
./wan2_1_t2v_1_3b_onnx/
├── wan_text_encoder_graph.mindir  + wan_text_encoder_variables/
├── wan_transformer_graph.mindir   + wan_transformer_variables/
└── wan_vae_decoder.mindir         (143M，单文件)
```

### 转换实测日志

```log
# 文本编码器
CONVERT RESULT SUCCESS:0
# transformer（dynamo 导出的全静态图）
CONVERT RESULT SUCCESS:0
# VAE 解码器
CONVERT RESULT SUCCESS:0
```

---

## 4. MindSpore Lite 推理

```bash
python infer_wan2_1_t2v_1_3b_mslite.py \
  --mindir-dir ./wan2_1_t2v_1_3b_onnx \
  --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A cat walking on a beach, cinematic, 4k." \
  --height 480 --width 832 --num-frames 13 \
  --num-inference-steps 50 --guidance-scale 5.0 \
  --output wan_output.mp4
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir-dir` | MindIR 目录（含 3 个子模型 mindir） | 必填 |
| `--model-dir` | 权重目录（tokenizer + scheduler + vae config） | 必填 |
| `--prompt/--negative-prompt` | 文本提示 | 见默认 |
| `--height/--width/--num-frames` | 必须与导出/转换一致 | `480/832/81`（命令行覆盖为 13） |
| `--num-inference-steps` | 去噪步数 | `50` |
| `--guidance-scale` | CFG 强度 | `5.0` |
| `--text-device/--transformer-device/--vae-device` | 组件分芯 | `1/0/0` |
| `--latents-npy` | 预生成噪声（精度对齐用） | 无 |

说明（固定 shape 约束）：`ascend_oriented` 转换按固定 shape 编译，推理侧 `--height/--width/--num-frames` 必须与导出一致；变更需重新导出+转换。文本编码器在 dev1，transformer/VAE 在 dev0（组件级分芯）。

### 推理实测日志（13 帧 / 50 步）

```log
[infer] saved video -> wan_output.mp4

--- Performance ---
  Text encode (UMT5):      668.11 ms
  Transformer total:       244410.69 ms
  Transformer avg/step:    4888.21 ms (50 steps, CFG x2)
  VAE decode:              4022.13 ms
  End-to-end:              249100.93 ms
# 输出 wan_output.mp4：13 帧 @ 480x832，像素范围正常
```

---

## 5. 性能数据

测试环境：Ascend Atlas 300I Duo，CANN 8.5.0，MindSpore Lite 2.9.0 runtime / 2.10.0 Python，480×832、13 帧、50 步、fp16、CFG=5。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 文本编码 (UMT5, dev1) | 668 |
| Transformer 总计 (50 步 × CFG×2 = 100 次 forward) | 244,411 |
| Transformer 单步平均（含 CFG 双路） | 4,888 |
| Transformer 单次 forward 平均 | ~2,444 |
| VAE 解码 (dev0, 13 帧) | 4,022 |
| **端到端** | **249,101 (~4.2 分钟)** |

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
4. **VAE 解码只出 4 帧（应 13 帧）。**
   - 原因：单 pass 未执行时间上采样（流式 `WanResample.upsample3d` 仅在第二次调用才跑 `time_conv`）。
   - 解决：脚本 `_patch_wan_resample_singlepass` 让 `upsample3d` 始终执行 `time_conv`+交错，并丢弃时间 index-1 匹配 `2t-1` 装配（4→7→13）。
5. **导出 transformer 时 CPU 内存暴涨/OOM。**
   - dynamo 导出 + `external_data=True` 已外置权重；若仍紧张，确认 `do_constant_folding` 关闭（legacy 路径）或减少并发进程。

---

## 7. 参考资源与许可证

- 上游模型：<https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- 本目录脚本遵循 MindSpore Lite 仓库许可证；上游模型权重许可证以其仓库为准（Apache-2.0）。
