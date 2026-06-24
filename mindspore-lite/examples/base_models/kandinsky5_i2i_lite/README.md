# Kandinsky-5.0-I2I-Lite ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [kandinskylab/Kandinsky-5.0-I2I-Lite](https://www.modelscope.cn/models/kandinskylab/Kandinsky-5.0-I2I-Lite) 图生图（image-to-image）模型按网络结构拆分导出为 ONNX，将 ONNX 转换为 MindSpore Lite MindIR，并在 **Ascend Atlas 300I Duo** 上完成端到端图生图推理与精度对齐。

- **Qwen2.5-VL-7B-Instruct 文本编码器**（3584 维 hidden state）。I2I 把 prompt + 缩小后的源图包进 chat template（55 个模板 token，T2I 为 41），切片 `[:, 55:]`。
- **CLIP 文本编码器**（clip-vit-large-patch14，768 维 pooled）。I2I 的 CLIP 只看纯文本 prompt（源图不经过 CLIP）。
- **Kandinsky5Transformer3DModel**（`visual_cond=True`）：patchify 输入为 `cat([noise(16), image_latents(16), mask(1)]) = 33` 通道（channels-last）。
- **FLUX.1-dev AutoencoderKL 解码器**（8× 空间，16 通道 latent）。源图在 CPU 上用 VAE **编码器**编码（不导出），仅 VAE **解码器**导出到 Ascend。

> 注意：K5 Image Lite **不是 DCAE / MT5**（社区"128× DCAE"是 Sana，非 K5）；K5 复用 FLUX VAE，文本编码器是 Qwen2.5-VL。

---

## 1. 环境准备

### 依赖版本（建议）

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0+cpu |
| onnx / onnxruntime | 1.19.1 / 1.24.2 |
| numpy | 1.26.4 |
| transformers | 4.49+（含 Qwen2.5-VL） |
| diffusers | 0.38.0（含 Kandinsky5I2IPipeline） |
| mindspore-lite | 2.10.0 |
| CANN | 8.5.0 |

```bash
source /home/yf/env.sh
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 transformers diffusers mindspore-lite modelscope
```

---

## 2. 模型下载

```bash
# K5 I2I Lite DiT 权重
python -c "from modelscope import snapshot_download as s; print(s('kandinskylab/Kandinsky-5.0-I2I-Lite', cache_dir='/home/yf/modelscope_cache'))"
# Qwen2.5-VL-7B-Instruct（文本编码器 + tokenizer）
python -c "from modelscope import snapshot_download as s; print(s('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/home/yf/modelscope_cache'))"
# CLIP（pooled 编码器）
python -c "from modelscope import snapshot_download as s; print(s('AI-ModelScope/clip-vit-large-patch14', cache_dir='/home/yf/modelscope_cache'))"
# FLUX.1-dev VAE（K5 复用）
python -c "from modelscope import snapshot_download as s; print(s('black-forest-labs/FLUX.1-dev', cache_dir='/home/yf/modelscope_cache', allow_patterns=['vae/*']))"

ln -sfn /home/yf/modelscope_cache/kandinskylab/Kandinsky-5___0-I2I-Lite ./kandinsky5lite_i2i
ln -sfn /home/yf/modelscope_cache/Qwen/Qwen2___5-VL-7B-Instruct ./Qwen2.5-VL-7B-Instruct
ln -sfn /home/yf/modelscope_cache/AI-ModelScope/clip-vit-large-patch14 ./clip-vit-large-patch14
ln -sfn /home/yf/modelscope_cache/black-forest-labs/FLUX___1-dev/vae ./flux_vae
```

---

## 3. 模型导出 ONNX

四个子模型（1024×1024 固定 shape）：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `kandinsky_text_encoder`（Qwen2.5-VL） | input_ids[1,512], attention_mask[1,512] | last_hidden_state[1,512,3584] |
| `kandinsky_clip_encoder`（CLIP） | input_ids[1,77], attention_mask[1,77] | pooled_embeds[1,768] |
| `kandinsky_transformer`（I2I DiT, visual_cond） | noise[1,1,128,128,16], image_latents[1,1,128,128,16], mask[1,1,128,128,1], encoder_hidden_states[1,457,3584], timestep[1], pooled_projections[1,768], visual_rope_h[64], visual_rope_w[64], text_rope[457] | noise_pred[1,1,128,128,16] |
| `kandinsky_dcae_decoder`（FLUX VAE） | latents[1,16,128,128] | image[1,3,1024,1024] |

### 导出命令

```bash
cd mindspore-lite/examples/base_models/kandinsky5_i2i_lite

python export_kandinsky5_i2i_lite_onnx.py \
  --k5-model ./kandinsky5lite_i2i \
  --qwen-dir ./Qwen2.5-VL-7B-Instruct \
  --clip-dir ./clip-vit-large-patch14 \
  --vae-dir ./flux_vae \
  --output-dir ./kandinsky5_i2i_lite_onnx \
  --height 1024 --width 1024 --dtype float32
```

### 自定义算子策略

transformer 注意力为全双向（无 mask），脚本 monkeypatch 注意力派发，导出为 CANN `PromptFlashAttention` Custom 节点（BNSD、`sparse_mode=0`）；fallback 仅为保形 stub，避免长序列 trace 实体化 O(seq²) 得分。q/k-norm 用 RMSNorm（导出侧分解为标准算子）。导出走 legacy exporter、opset 17、float32、`do_constant_folding=False`。

### 产出文件

```text
./kandinsky5_i2i_lite_onnx/
├── kandinsky_text_encoder.onnx (+ .data)
├── kandinsky_clip_encoder.onnx (+ .data)
├── kandinsky_transformer.onnx (+ .data)
└── kandinsky_dcae_decoder.onnx (+ .data)
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
for c in text_encoder clip_encoder transformer dcae_decoder; do
  $CONV --fmk=ONNX \
    --modelFile=./kandinsky5_i2i_lite_onnx/kandinsky_${c}.onnx \
    --outputFile=./kandinsky5_i2i_lite_onnx/kandinsky_${c} \
    --optimize=ascend_oriented --saveType=MINDIR \
    --configFile=./configs/kandinsky_${c}.config
done
```

执行日志（待运行后填入）：

```log
CONVERT RESULT SUCCESS:0   （待运行后填入完整日志）
```

---

## 5. MindSpore Lite 推理

```bash
python infer_kandinsky5_i2i_lite_mslite.py \
  --mindir-dir ./kandinsky5_i2i_lite_onnx \
  --qwen-dir ./Qwen2.5-VL-7B-Instruct \
  --clip-dir ./clip-vit-large-patch14 \
  --vae-dir ./flux_vae \
  --prompt "make it a watercolor painting" \
  --source-image ./input.png \
  --height 1024 --width 1024 \
  --num-inference-steps 50 --guidance-scale 3.5 \
  --output ./kandinsky5_i2i_output.png
```

`--height/--width` 必须与导出/转换一致；变更需重新导出+转换。组件分芯：text_encoder+clip → dev1，transformer+vae → dev0。源图经 CPU 端 FLUX VAE 编码器编码为 image_latents（不导出），与 noise、mask 拼成 33 通道送 DiT。

执行日志（待运行后填入，含性能数据）：

```log
（待运行后填入：各阶段耗时 + 输出 PNG 路径）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

> 性能数据以推理脚本端到端打印为准；下表为**待实测填入**（运行 `infer_kandinsky5_i2i_lite_mslite.py` 后回填真实数值，不使用估算或占位假数据）。

| 指标 | 耗时 (ms) |
| --- | ---: |
| VAE 编码源图 (CPU) | _待运行填入_ |
| 文本编码 (Qwen+CLIP, dev1) | _待运行填入_ |
| Transformer 总计 (50 步 × CFG×2) | _待运行填入_ |
| Transformer 单步平均 | _待运行填入_ |
| VAE 解码 (dev0) | _待运行填入_ |
| **端到端** | **_待运行填入_** |

---

## 7. 精度对齐

端到端对比 HF diffusers `Kandinsky5I2IPipeline`（CPU float32）与 MSLite（Ascend）：相同 prompt、相同源图、相同 seed 初始噪声，比较输出图像 max/mean abs 误差与 PSNR。

```bash
python align_kandinsky5_i2i_lite.py \
  --mindir-dir ./kandinsky5_i2i_lite_onnx \
  --qwen-dir ./Qwen2.5-VL-7B-Instruct --clip-dir ./clip-vit-large-patch14 \
  --vae-dir ./flux_vae --k5-model ./kandinsky5lite_i2i \
  --prompt "make it a watercolor painting" --source-image ./input.png \
  --height 1024 --width 1024 --num-inference-steps 50
```

执行日志（待运行后填入）：

```log
（待运行后填入：max_abs / mean_abs / PSNR）
```

---

## 8. 常见问题

1. **K5 用的是 DCAE / MT5 吗？** —— 不是。K5 Image Lite 复用 FLUX.1-dev AutoencoderKL（8×、16 通道），文本编码器是 Qwen2.5-VL。
2. **transformer 输入为什么是 33 通道？** —— I2I 把 `noise(16) + image_latents(16) + mask(1)` 沿通道拼接（`visual_cond=True`），T2I 是 16 通道。
3. **`[:, 55:]` 切片是什么？** —— I2I 的 chat template 前 55 个 token 是模板/系统提示，送给 DiT 前去掉（T2I 为 41）。
4. **源图怎么处理？** —— 源图在 CPU 端经 FLUX VAE 编码器编码为 image_latents（编码器不导出），缩小一半后塞进 Qwen 序列作为视觉 token。
5. **转换 protobuf 超大 / transformer 编译慢** —— ascend_oriented 固定 shape 编译重，确认 `CONVERT RESULT SUCCESS:0`，必要时降低并发。

---

## 9. 参考资源与许可证

- 上游：<https://github.com/kandinskylab/kandinsky-5>、ModelScope `kandinskylab/Kandinsky-5.0-I2I-Lite`
- MindSpore Lite：<https://www.mindspore.cn/lite>
- 本目录脚本遵循 MindSpore Lite 仓库许可证；上游模型/代码许可证以其仓库为准。
