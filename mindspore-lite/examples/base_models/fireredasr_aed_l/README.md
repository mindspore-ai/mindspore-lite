# FireRedASR-AED-L ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 FireRedASR-AED-L 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

FireRedASR-AED-L 是面向中文语音识别的 AED（Attention-Encoder-Decoder）模型，模型被拆分为 2 个 ONNX 文件：

1. **Encoder**（`fireredasr_aed_encoder.onnx`）：Conformer 编码器，并对所有 decoder 层预计算 cross-attention 的 K/V 投影
2. **DecoderStep**（`fireredasr_aed_decoder_step.onnx`）：单步 Transformer decoder（含 self-attention KV cache 与 PromptFlashAttention 融合）

## 模型架构

FireRedASR-AED-L 采用经典 AED 结构：

- **Encoder**：16 层 Conformer（d_model=1280，n_head=20，d_k=64），输出 encoder hidden 与 cross-attention K/V
- **Decoder**：16 层 Transformer decoder（d_model=1280，n_head=20，d_k=64），每层包含 self-attention（带 KV cache）+ cross-attention + MLP
- **特征提取**：80 维 kaldi_native_fbank，CMVN 归一化

自回归解码时，encoder 只跑一次，decoder 按 token 循环执行（每步 KV cache +1）。

---

## 1. 环境准备

### 依赖版本

| 软件包              | 版本     |
|------------------|--------|
| Python           | 3.11   |
| torch            | 2.10.0 |
| onnx             | 1.19.1 |
| onnxruntime      | 1.24.2 |
| kaldi_native_fbank | 1.22.3 |
| kaldiio          | 2.18.1 |
| CANN             | 8.5    |
| mindspore-lite   | 2.9.0  |

```bash
pip install torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 kaldi_native_fbank==1.22.3 kaldiio==2.18.1
```

说明：

- FireRedASR 源码需克隆并加入 `PYTHONPATH`：<https://github.com/FireRedTeam/FireRedASR>
- 权重目录示例路径：`/path/to/FireRedASR-AED-L`（下文用 `MODEL_DIR` 表示），需包含 `model.pth.tar`、`cmvn.ark`、`dict.txt` 等

---

## 2. 模型导出 ONNX

### 导出命令

```bash
python export_fireredasr_aed_l_onnx.py \
  --fireredasr-repo /path/to/FireRedASR \
  --model-dir /path/to/FireRedASR-AED-L \
  --output-dir ./outputs
```

默认导出融合版 decoder（self-attn 使用 PromptFlashAttention，适配 Ascend）。如需 ONNXRuntime 推理或精度对照，加 `--disable-attn-fusion` 关闭融合。

### 参数说明

| 参数                       | 说明                                              | 默认值    |
|--------------------------|-------------------------------------------------|--------|
| `--fireredasr-repo`      | FireRedASR 源码路径（加入 `PYTHONPATH`）                | —      |
| `--model-dir`            | 权重目录（含 `model.pth.tar`、`cmvn.ark`、`dict.txt`）  | 必填     |
| `--output-dir`           | ONNX 输出目录                                       | 必填     |
| `--device`               | 导出设备（cpu/cuda）                                  | `cpu`  |
| `--disable-attn-fusion`  | 关闭 decoder self-attn PromptFlashAttention 融合   | 默认开启融合 |

### 产出

```text
outputs/
├── onnx_encoder/
│   └── fireredasr_aed_encoder.onnx        # Encoder + cross-attn K/V 投影
└── onnx_decoder/
    └── fireredasr_aed_decoder_step.onnx   # 单步 Decoder（融合版）
```

### ONNX 模型输入输出 Shape

**Encoder** — `fireredasr_aed_encoder.onnx`

| 方向  | 名称                | Shape                                | Dtype  | 说明                          |
|-----|-------------------|--------------------------------------|--------|-----------------------------|
| 输入 | `padded_input`    | `(batch, time, 80)`                  | float32 | 80 维 fbank 特征              |
| 输入 | `input_lengths`   | `(batch,)`                           | int64   | 每条音频实际帧数                 |
| 输出 | `encoder_outputs` | `(batch, time', 1280)`               | float32 | Conformer 编码输出            |
| 输出 | `enc_mask`        | `(batch, 1, time')`                  | uint8   | encoder padding mask         |
| 输出 | `cross_k`         | `(batch, 16, 20, time', 64)`         | float32 | 16 层 decoder 的 cross-attn K |
| 输出 | `cross_v`         | `(batch, 16, 20, time', 64)`         | float32 | 16 层 decoder 的 cross-attn V |

> 16 = decoder 层数，20 = attention head 数，64 = d_k；`time'` 由 encoder 下采样率决定。

**DecoderStep** — `fireredasr_aed_decoder_step.onnx`

| 方向  | 名称                | Shape                                  | Dtype  | 说明                          |
|-----|-------------------|----------------------------------------|--------|-----------------------------|
| 输入 | `ys`              | `(batch, tgt_len)`                     | int64   | 已生成的 token 序列（含 SoS）   |
| 输入 | `src_mask`        | `(batch, 1, src_len)`                  | uint8   | cross-attn mask              |
| 输入 | `cache_k_self`    | `(batch, 16, 20, cached_len, 64)`      | float32 | self-attn KV cache（K）       |
| 输入 | `cache_v_self`    | `(batch, 16, 20, cached_len, 64)`      | float32 | self-attn KV cache（V）       |
| 输入 | `cross_k`         | `(batch, 16, 20, src_len, 64)`         | float32 | cross-attn K（来自 encoder）   |
| 输入 | `cross_v`         | `(batch, 16, 20, src_len, 64)`         | float32 | cross-attn V（来自 encoder）   |
| 输出 | `log_probs`       | `(batch, 1, vocab_size)`               | float32 | 下一步 token 的 log softmax   |
| 输出 | `new_cache_k_self` | `(batch, 16, 20, new_cached_len, 64)` | float32 | 更新后的 self-attn K cache    |
| 输出 | `new_cache_v_self` | `(batch, 16, 20, new_cached_len, 64)` | float32 | 更新后的 self-attn V cache    |

> 每步 `cached_len += 1`，`src_len` 固定（由 encoder 输出决定）；`vocab_size` 取决于词表。

---

## 3. ONNX 转 MindIR

### 转换命令

```bash
Convert=mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# Encoder：保持 fp32 动态（与 encoder 内部 F.pad + 动态 input_lengths 兼容）
$Convert --fmk=ONNX \
  --modelFile=./outputs/onnx_encoder/fireredasr_aed_encoder.onnx \
  --outputFile=./outputs/mindir_encoder \
  --optimize=ascend_oriented \
  --saveType=MINDIR

# DecoderStep：force_fp16 + 6 个 cached_len bucket（src_len=50）
$Convert --fmk=ONNX \
  --modelFile=./outputs/onnx_decoder/fireredasr_aed_decoder_step.onnx \
  --outputFile=./outputs/mindir_decoder_step_fp16 \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./fireredasr_aed_decoder_step.ini
```

### 参数说明

| 参数             | 说明                                       |
|----------------|------------------------------------------|
| `--fmk`        | 输入模型格式（ONNX）                            |
| `--modelFile`  | 输入 ONNX 模型路径                            |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）                     |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented`             |
| `--saveType`   | 输出格式（MINDIR）                            |
| `--configFile` | decoder 转换必须指定 `fireredasr_aed_decoder_step.ini` |

### 配置文件

`fireredasr_aed_decoder_step.ini`（decoder 转换时必须）：

```ini
[acl_build_options]
input_format="ND"
input_shape="ys:1,1;src_mask:1,1,-1;cache_k_self:1,16,20,-1,64;cache_v_self:1,16,20,-1,64;cross_k:1,16,20,-1,64;cross_v:1,16,20,-1,64"
ge.dynamicDims="50,1,1,50,50;50,2,2,50,50;50,3,3,50,50;50,4,4,50,50;50,5,5,50,50;50,6,6,50,50"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=BatchMatmulToMatmul
```

说明：

- 6 个 bucket 对应 `cached_len ∈ {1,2,3,4,5,6,200}`，`src_len` 固定为 50（对应测试音频 `feat_frames=197`）
- 5 个 `-1` 维度依次绑定 `src_mask.src_len`、`cache_k_self.cached_len`、`cache_v_self.cached_len`、`cross_k.src_len`、`cross_v.src_len`
- encoder **不做混精**：encoder 内部 `F.pad` + 动态 `input_lengths` 与 GE bucket 路径存在已知兼容性问题（predict 阶段失败），保留 fp32 动态即可

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
outputs/
├── mindir_encoder_graph.mindir
├── mindir_encoder_variables/
├── mindir_decoder_step_fp16_graph.mindir
└── mindir_decoder_step_fp16_variables/
```

转换成功后日志末尾出现 `CONVERT RESULT SUCCESS:0`。

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_fireredasr_aed_l_mslite.py \
  --fireredasr-repo /path/to/FireRedASR \
  --mindir-dir ./outputs \
  --encoder-mindir mindir_encoder_graph.mindir \
  --decoder-mindir mindir_decoder_step_fp16.mindir \
  --model-dir /path/to/FireRedASR-AED-L \
  --wav-path /path/to/test.wav \
  --device npu \
  --max-len 200 --sos-id 3 --eos-id 4
```

### 参数说明

| 参数                  | 说明                                          | 默认值                                |
|---------------------|---------------------------------------------|------------------------------------|
| `--fireredasr-repo` | FireRedASR 源码路径                             | —                                  |
| `--mindir-dir`      | MindIR 输出目录（含 `mindir_*_graph.mindir`）     | —                                  |
| `--encoder-mindir`  | encoder MindIR 文件名                          | `mindir_encoder_graph.mindir`      |
| `--decoder-mindir`  | decoder_step MindIR 文件名                     | `mindir_decoder_step_graph.mindir` |
| `--model-dir`       | 权重目录（含 `cmvn.ark`、`dict.txt`）              | 必填                                 |
| `--wav-path`        | 16kHz/16bit PCM WAV 路径                       | 必填                                 |
| `--device`          | 推理设备（`npu` / `cpu`）                        | `npu`                              |
| `--max-len`         | 最大解码步数（greedy）                             | 200                                |
| `--sos-id` / `--eos-id` | 起止 token id                              | 3 / 4                              |

### 推理示例输出

测试音频：<https://github.com/FireRedTeam/FireRedASR/raw/main/examples/wav/IT0011W0001.wav>

```text
Model build time: 28746.11 ms
Feature time: 2217.89 ms, feat_frames: 197, feat_len: 197
Encoder time: 31.90 ms, src_len: 50, feat_len: 197
Total decode time: 46.26 ms, avg decode step: 9.25 ms, steps: 5
Total time: 78.16 ms, throughput: 51.18 tok/s, num_tokens: 4
Recognition result: 换一首歌
```

说明：

- 脚本对 encoder 与 decoder 默认预热 3 轮后再计时；若不预热，首轮耗时可能显著偏大
- `Total time` = `Encoder time` + `Total decode time` + `Feature time`
- `steps = num_tokens + 1`（最后一步用于预测到 `eos`）

---

## 5. 性能数据

### 测试环境

| 项目   | 配置                                  |
|------|-------------------------------------|
| 硬件   | Atlas 300I Duo（Ascend NPU）          |
| 模型   | FireRedASR-AED-L                    |
| 测试音频 | [IT0011W0001.wav](https://github.com/FireRedTeam/FireRedASR/raw/main/examples/wav/IT0011W0001.wav)（中文，feat_frames=197，src_len=50，5 步解码） |
| 精度   | encoder fp32 + decoder force_fp16 + 6 buckets |

### 端到端推理性能

| 阶段              | 耗时 (ms) | 说明                          |
|-----------------|---------|-----------------------------|
| Feature 提取      | 2217.89 | 80 维 fbank + CMVN（仅在启动时跑一次） |
| Encoder         | 31.90   | Conformer 16 层，src_len=50   |
| Decoder（总 5 步）  | 46.26   | 含 5 次 decoder_step 调用        |
| 总耗时  | 2296.05   |         |
| 吞吐量               | **51.18 tok/s** | |
| 生成 token 数        | 4            ||

---

## 6. 常见问题

1. **转换耗时很久且输出大量 Warning**
   - 现象：`converter_lite` 执行时间较长，并输出大量 Warning
   - 原因：decoder_step 体积大（约 1.5GB），`ascend_oriented` 会做较重的图优化与编译准备
   - 解决方案：等待转换结束，关注是否出现 `CONVERT RESULT SUCCESS:0`

2. **转换日志出现 `ge.proto.ModelDef exceeded maximum protobuf size of 2GB`**
   - 现象：转换日志中出现 protobuf size 超限相关提示
   - 原因：GE 内部中间表示可能超过 2GB，属于大模型常见现象
   - 解决方案：以最终转换结果为准，确认末尾 `CONVERT RESULT SUCCESS:0`

3. **Ascend 推理报错 `input data size is wrong` / `Acl memcpy ... size: 0`**
   - 现象：MSLite 推理阶段报输入 size 为 0 的错误
   - 原因：Ascend 侧不接受 0-size Tensor，本模型 decoder_step 的 KV cache 若传入 `cached_len=0` 会触发该问题
   - 解决方案：推理侧避免 0-size 输入（例如 cache 初始化为 `cached_len=1` 的零张量）

4. **混精版 MindIR 跑其他音频 predict 失败**
   - 现象：用本节配置转换的 MindIR 跑非 `IT0011W0001` 的音频（如 `src_len=105/309`）时 `model.predict()` 报错
   - 原因：当前 6 个 cached_len bucket 与 `src_len=50` 强绑定（`ge.dynamicDims` 第一列为 `src_mask.src_len`，全部写死 50）
   - 解决方案：按音频长度档（短 / 中 / 长）拆分多个 MindIR，每个 MindIR 内部只对 cached_len bucket；推理侧实现轻量 router 根据 `feat_len` 选 MindIR

5. **为什么只融合 decoder_step 的 self-attn，没有融合 cross-attn / encoder？**
   - decoder_step 当前仅 self-attn 使用 PromptFlashAttention；cross-attn 保持原始 attention
   - cross-attn 场景 `Qs=1, Kvs=src_len` 且需要 mask，会触发 CANN PromptFlashAttention 的 tiling 约束（`Qs!=Kvs 且带 mask 时要求 mask=NULL`）
   - encoder 为 Conformer（包含相对位置注意力 / 卷积分支），attention 子图更复杂，融合风险高且收益不如 decoder_step（decoder_step 按 token 循环多次执行）

---

## 7. 参考资源

- [FireRedASR 官方仓库](https://github.com/FireRedTeam/FireRedASR)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库的许可证要求
- FireRedASR 模型与代码的许可证请以其上游仓库为准
