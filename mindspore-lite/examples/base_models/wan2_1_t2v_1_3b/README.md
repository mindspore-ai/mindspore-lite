# Wan2.1-T2V-1.3B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) 文本生成视频模型按网络结构拆分导出为 ONNX，使用 ONNX Runtime 验证子模型推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 **Ascend Atlas 300I Duo** 上完成端到端推理与测速。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0+cpu（导出仅在 CPU） |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 1.26.4 |
| transformers | 5.9.0 |
| diffusers | 0.38.0 |
| mindspore-lite | 2.10.0 |
| CANN | 8.5.0 |

```bash
source /home/yf/env.sh   # CANN + mindspore-lite runtime + converter_lite
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 transformers diffusers mindspore-lite imageio
```

### 获取模型权重

```bash
# 从 ModelScope 下载，缓存到本地后做符号链接
python -c "from modelscope import snapshot_download as s; print(s('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', cache_dir='/home/yf/modelscope_cache'))"
ln -sfn /home/yf/modelscope_cache/Wan-AI/Wan2___1-T2V-1___3B-Diffusers Wan2.1-T2V-1.3B-Diffusers
```

`MODEL_DIR`（权重目录）需包含 `text_encoder/`、`transformer/`、`vae/`、`tokenizer/`、`scheduler/`。

---

## 2. 模型导出 ONNX

按结构拆分为三个固定 shape 子模型：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `wan_text_encoder` (UMT5-XXL) | input_ids[1,512], attention_mask[1,512] | last_hidden_state[1,512,4096] |
| `wan_transformer` (DiT 1.3B) | hidden_states[1,16,4,60,104], timestep[1], encoder_hidden_states[1,512,4096] | noise_pred[1,16,4,60,104] |
| `wan_vae_decoder` (3D VAE) | latents[1,16,4,60,104] | video[1,3,81,480,832] |

固定配置：480×832、13 帧（4 个 latent 帧）。`timestep` 形状 `[1]`，对应 13 帧视频（短片段 demo）。若需其它分辨率/帧数，需重新导出并重新转换。

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
| `--num-frames` | 视频帧数（4k+1） | `81` |
| `--max-seq-len` | UMT5 最大序列长度 | `512` |
| `--dtype` | 导出精度（建议 float32，便于转换） | `float32` |
| `--no-custom-op` | 不把注意力替换为 CANN Custom 算子 | `False` |

### 产出文件

```text
./wan2_1_t2v_1_3b_onnx/
├── wan_text_encoder.onnx (+ 外部权重 .data)
├── wan_transformer.onnx (+ 外部权重 .data)
└── wan_vae_decoder.onnx (+ 外部权重 .data)
```

### 导出注意事项

- **注意力替换为 CANN `PromptFlashAttention` Custom 算子**：Wan 的时空自注意力和文本交叉注意力均为全双向（无 mask），脚本 monkeypatch diffusers 的注意力派发（`transformer_wan.dispatch_attention_fn`），将注意力导出为 `PromptFlashAttention` Custom 节点（BNSD、`sparse_mode=0`），从而避免在 32k token 全注意力上实体化 O(seq²) 得分矩阵。其余（q/k/v 投影、RMSNorm、RoPE、3D patchify）走标准算子。
- 导出走 **legacy 导出器**（`torch.onnx.utils.export`），`do_constant_folding=False`（长序列图常量折叠会在 CPU 上 OOM）。
- 模型以 **float32** 加载导出，避免 ONNX 全图 FLOAT16 导致转换器报错。

---

## 3. ONNX 推理

> 说明：`wan_transformer.onnx` 含 `Custom` 节点（PromptFlashAttention），**ONNX Runtime 无法直接执行**；文本编码器与 VAE 为标准算子图，可用 ONNX Runtime 验证。transformer 的精度基准以 HF diffusers pipeline 为准（见第 7 节）。

```bash
# （可选）用 ONNX Runtime 验证文本编码器 / VAE
python - <<'PY'
import numpy as np, onnxruntime as ort
m = ort.InferenceSession("wan2_1_t2v_1_3b_onnx/wan_text_encoder.onnx", providers=["CPUExecutionProvider"])
ids = np.zeros((1,512), np.int64); mask = np.ones((1,512), np.int64)
out = m.run(None, {"input_ids": ids, "attention_mask": mask})
print("text_encoder out shape:", out[0].shape)
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
$CONV --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_text_encoder.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_text_encoder \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_text_encoder.config

# transformer
$CONV --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_transformer.onnx \
  --outputFile=./wan2_1_t2v_1_3b_onnx/wan_transformer \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/wan_transformer.config

# VAE 解码器
$CONV --fmk=ONNX --modelFile=./wan2_1_t2v_1_3b_onnx/wan_vae_decoder.onnx \
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

### 产出说明

```text
./wan2_1_t2v_1_3b_onnx/
├── wan_text_encoder_graph.mindir  + wan_text_encoder_variables/
├── wan_transformer_graph.mindir   + wan_transformer_variables/
└── wan_vae_decoder_graph.mindir   + wan_vae_decoder_variables/
```

执行日志（待运行后填入）：

```log
CONVERT RESULT SUCCESS:0   （待运行后填入完整日志）
```

---

## 5. MindSpore Lite 推理

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
| `--mindir-dir` | MindIR 目录（含 3 个 `*_graph.mindir`） | 必填 |
| `--model-dir` | 权重目录（tokenizer + scheduler + vae config） | 必填 |
| `--prompt/--negative-prompt` | 文本提示 | 见默认 |
| `--height/--width/--num-frames` | 必须与导出/转换一致 | `480/832/81` |
| `--num-inference-steps` | 去噪步数 | `50` |
| `--guidance-scale` | CFG 强度 | `5.0` |
| `--text-device/--transformer-device/--vae-device` | 组件分芯 | `1/0/0` |
| `--latents-npy` | 预生成噪声（精度对齐用） | 无 |

说明（固定 shape 约束）：`ascend_oriented` 转换按固定 shape 编译，推理侧 `--height/--width/--num-frames` 必须与导出一致；变更需重新导出+转换。文本编码器在 dev1，transformer/VAE 在 dev0（组件级分芯）。

执行日志（待运行后填入，含性能数据）：

```log
（待运行后填入实际输出：生成视频路径 + 各阶段耗时）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3，每芯 ~44GB），CANN 8.5.0，MindSpore Lite 2.10.0。

> 性能数据以推理脚本端到端打印为准；下表为**待实测填入**（运行 `infer_wan2_1_t2v_1_3b_mslite.py` 后回填真实数值，不使用估算或占位假数据）。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 文本编码 (UMT5, dev1) | _待运行填入_ |
| Transformer 总计 (50 步 × CFG×2) | _待运行填入_ |
| Transformer 单步平均 | _待运行填入_ |
| VAE 解码 (dev0) | _待运行填入_ |
| **端到端** | **_待运行填入_** |

---

## 7. 精度对齐

端到端对比 HF diffusers `WanPipeline`（CPU float32 基线）与 MSLite（Ascend）的生成视频：使用相同 prompt、相同初始噪声（seed 固定的 torch 生成器，存 npy 后两路共用）、相同调度器参数，逐帧比较 max/mean abs 误差与 PSNR。

```bash
# 注意：HF CPU 基线较慢，建议先用较少帧/步数跑通端到端对齐
python align_wan2_1_t2v_1_3b.py \
  --mindir-dir ./wan2_1_t2v_1_3b_onnx \
  --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
  --num-frames 21 --num-inference-steps 10
```

执行日志（待运行后填入）：

```log
（待运行后填入：max_abs / mean_abs / PSNR）
```

---

## 8. 常见问题

1. 现象：导出 transformer 时 CPU 内存暴涨/被 OOM 杀死。
   - 原因：legacy 导出器默认常量折叠，长序列（32k token）图折叠常量占满内存。
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

---

## 9. 参考资源与许可证

- 上游模型：<https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- 本目录脚本遵循 MindSpore Lite 仓库许可证；上游模型权重许可证以其仓库为准（Apache-2.0）。
