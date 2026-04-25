# CosyVoice2-0.5B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 CosyVoice2-0.5B 拆分导出为 ONNX，使用 ONNX Runtime 端到端验证（输出音频），并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend 上推理（输出 mel 特征与性能数据）。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0 |
| transformers   | 5.6.2  |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 9.0    |
| mindspore-lite | 2.8.0  |
| matcha-tts | 0.0.7.2  |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite qwen-asr==0.0.6
```

### 获取模型权重与源码

> **注意**：本示例依赖 CosyVoice 源码（HiFT 声码器），需要单独获取。

```bash
# CosyVoice 源码（HiFT 声码器 PyTorch 推理需要从中导入）
git clone https://github.com/FunAudioLLM/CosyVoice.git

# CosyVoice2-0.5B 权重（参考 CosyVoice 项目文档下载）
```

> 说明：`--model-dir` 为权重目录（含 `llm.pt`、`flow.pt`、`hift.pt`、可选 `campplus.onnx`、`speech_tokenizer_v1.onnx` 等），`--model-code-dir` 为 CosyVoice 源码目录。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/cosyvoice2_0.5b

python export_cosyvoice2_onnx.py \
  --model-dir /path/to/CosyVoice2-0.5B \
  --model-code-dir /path/to/CosyVoice \
  --output-dir ./cosyvoice2_onnx \
  --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-dir` | CosyVoice2-0.5B 权重目录 | 见脚本默认值 |
| `--model-code-dir` | CosyVoice 源码目录 | 见脚本默认值 |
| `--output-dir` | ONNX 输出目录 | `./cosyvoice2_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--skip-llm` | 跳过 LLM 导出 | `False` |
| `--skip-flow` | 跳过 Flow 导出 | `False` |

### 产出文件

```log
cosyvoice2_onnx/
├── cosyvoice2_llm_prefill.onnx
├── cosyvoice2_llm_decode.onnx
├── cosyvoice2_flow_encoder.onnx
└── cosyvoice2_flow_estimator.onnx
```

### 导出注意事项

- **LLM Prefill 的 dummy `speech_len` 必须设为 `0`**：因为实际推理时经常没有 prompt 音频（`speech_ids` 为空），如果导出时用 `speech_len > 0` 的 dummy，MSLite Ascend runtime 无法处理 size=0 的 tensor。详见常见问题第 6 条。

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_cosyvoice2_onnx.py \
  --onnx-dir ./cosyvoice2_onnx \
  --model-dir /path/to/CosyVoice2-0.5B \
  --model-code-dir /path/to/CosyVoice \
  --text "你好，很高兴认识你" \
  --output output.wav \
  --seed 0
```

**执行日志：**

```log
Input text: 你好， 很高兴认识你

[1/4] Running LLM (Prefill + Decode)...
  LLM generated 50 tokens...
  LLM finished: 83 speech tokens

[2/4] Running Flow Encoder...

[3/4] Running Flow Estimator...

[4/4] Running HiFT vocoder...

Saved to test.wav (3.32s)
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--onnx-dir` | ONNX 模型目录 | `./cosyvoice2_onnx` |
| `--model-dir` | 权重目录 | 见脚本默认值 |
| `--model-code-dir` | CosyVoice 源码目录 | 见脚本默认值 |
| `--text` | 待合成文本 | `你好，很高兴认识你。` |
| `--prompt-wav` | 可选提示音频（语音克隆） | `None` |
| `--output` | 输出 wav 路径 | `output.wav` |
| `--seed` | 随机种子（影响采样与 flow 初始噪声） | `0` |
| `--flow-cfg` | Flow CFG 系数 | `0.7` |
| `--flow-steps` | Flow Euler 步数 | `10` |
| `--decode-mode` | LLM 解码模式（greedy/ras） | `ras` |

> **注意**：`greedy` 模式用于调试时与 ONNX 对比中间输出（deterministic），`ras` 模式遵循 CosyVoice2 原始采样行为（更自然）。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

> LLM 子模型建议配置 `force_fp32`，避免 FP16 下 attention mask 极小值带来的数值问题。

```bash
Converter=/path/to/mindspore-lite/tools/converter/converter_lite

# LLM Prefill
$Converter --fmk=ONNX \
  --modelFile=cosyvoice2_onnx/cosyvoice2_llm_prefill.onnx \
  --outputFile=cosyvoice2_mindir/cosyvoice2_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=config.ini

# LLM Decode
$Converter --fmk=ONNX \
  --modelFile=cosyvoice2_onnx/cosyvoice2_llm_decode.onnx \
  --outputFile=cosyvoice2_mindir/cosyvoice2_llm_decode \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=config.ini

# Flow Encoder
$Converter --fmk=ONNX \
  --modelFile=cosyvoice2_onnx/cosyvoice2_flow_encoder.onnx \
  --outputFile=cosyvoice2_mindir/cosyvoice2_flow_encoder \
  --optimize=ascend_oriented \
  --saveType=MINDIR

# Flow Estimator
$Converter --fmk=ONNX \
  --modelFile=cosyvoice2_onnx/cosyvoice2_flow_estimator.onnx \
  --outputFile=cosyvoice2_mindir/cosyvoice2_flow_estimator \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

> **Flow Encoder**：请用本目录最新 `export_cosyvoice2_onnx.py` 重新导出。旧 ONNX 里 `Upsample1D` 会在 `/encoder/up_layer/Resize` 产生 **3 维** 输入，Ascend GE 要求 **4 维**，`converter_lite` 会报 `The dim of input x is not 4`（见 `convert.log` 与常见问题第 5 条）。

### 配置文件

`config.ini`：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出说明

模型超过 2GB 时，MindIR 会拆分为 `*_graph.mindir` + `*_variables/` 目录，推理时使用 `*_graph.mindir` 加载。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_cosyvoice2_mslite.py \
  --mindir-dir ./cosyvoice2_mindir \
  --model-dir /path/to/CosyVoice2-0.5B \
  --model-code-dir /path/to/CosyVoice \
  --device ascend \
  --device-id 0 \
  --text "你好，很高兴认识你" \
  --output output.wav \
  --seed 0
```

**执行日志：**

```log
Input text: 你好，很高兴认识你

[1/4] Running LLM (Prefill + Decode)...
  LLM generated 50 tokens...
  LLM finished: 60 speech tokens
  LLM prefill: 601.15 ms
  LLM decode : 1223.60 ms (steps=60, avg_step=20.39 ms)
  LLM total  : 1824.75 ms

[2/4] Running Flow Encoder...
  Flow Encoder: 30.40 ms

[3/4] Running Flow Estimator (CFM)...
  Flow Estimator (10 steps, cfg=0.7): 350.97 ms

[4/4] Running HiFT vocoder...
  HiFT vocoder: 1218.97 ms

Saved to output.wav (2.40s)
Total time: 3521.43 ms (RTF: 1.467)

[Performance Markdown]
| 指标 | 耗时 (ms) |
|---|---:|
| LLM Prefill | 601.15 |
| LLM Total Decode | 1223.60 |
| Avg Decode Step | 20.39 |
| Flow Encoder | 30.40 |
| Flow Estimator | 350.97 |
| HiFT Vocoder | 1218.97 |
| **Total** | **3521.43** |
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mindir-dir` | MindIR 模型目录 | `./cosyvoice2_mindir` |
| `--model-dir` | 权重目录 | 见脚本默认值 |
| `--model-code-dir` | CosyVoice 源码目录（HiFT 需要） | 见脚本默认值 |
| `--device` | 推理设备（cpu/ascend） | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--text` | 待合成文本 | `你好，很高兴认识你。` |
| `--prompt-wav` | 可选提示音频（语音克隆） | `None` |
| `--output` | 输出 wav 路径 | `output.wav` |
| `--seed` | 随机种子 | `0` |
| `--flow-cfg` | Flow CFG 系数 | `0.7` |
| `--flow-steps` | Flow Euler 步数 | `10` |
| `--decode-mode` | LLM 解码模式（greedy/ras） | `ras` |

### 说明

- `infer_cosyvoice2_mslite.py` 支持 **端到端 wav 输出**。
- LLM + Flow 使用 MindSpore Lite 推理，HiFT vocoder 使用 PyTorch CPU（cosyvoice 源码）。
- 如需提示音频（语音克隆），直接通过 `--prompt-wav` 传入 wav 文件。

---

## 6. 性能数据

### 性能测试结果（Atlas 800I A2）

测试模型：CosyVoice2-0.5B
测试条件：输入文本"你好，很高兴认识你"，Ascend 推理

| 指标 | 耗时 (ms) |
|---|---:|
| LLM Prefill | 601.15 |
| LLM Decode | 1223.60(Total), 20.39(Avg Step) |
| Flow Encoder | 30.40 |
| Flow Estimator | 350.97 |
| HiFT Vocoder | 1218.97 |
| **Total** | **3521.43** |

| 指标 | 值 |
|---|---:|
| Audio Duration | 2.40 s |
| RTF | 1.467 |

> **注意**：本性能测试依赖 CosyVoice 源码（HiFT 声码器部分）。RTF (Real-Time Factor) 为总推理时间与音频时长之比，值越小表示实时性越好。RTF < 1 表示快于实时。HiFT Vocoder模型使用torch cpu计算，暂未导出为onnx。

---

## 7. 常见问题

**1. 导出时报 `GuardOnDataDependentSymNode`**

- **现象**：`torch.onnx.export` 导出失败。
- **原因**：新导出器内部走 `torch.export`，不支持数据依赖控制流。
- **解决方案**：导出时使用 legacy 导出器（脚本已通过 `dynamo=False` 固定）。

**2. 导出 Flow Encoder 后，ONNX 推理”首句清晰，后续全是噪声”**

- **现象**：生成音频只有最前面（例如”你好”）清晰，后面变成噪声。
- **根因**：Flow Encoder 导出时 `mask` 被错误地固定为 dummy 导出长度，导致真实推理时后半段 mel 帧被 mask 掉，Flow Estimator 不更新这些帧，最终保留随机噪声。
- **如何确认**：在 ONNX 推理日志中打印 `flow_mask.mean()`，会出现明显小于 1 的值（例如约 0.37，代表只有 ~40/108 帧有效）。
- **解决方案**：
    - 在 `export_cosyvoice2_onnx.py` 的 Flow Encoder wrapper 中，**不要**用 `torch.tensor([mel_len])` 构造 mask（legacy exporter 可能把它 trace 成常量）。
    - 直接用动态 shape 构造全 1 mask：`attn_mask = h.new_ones((h.shape[0], 1, mel_len))`。
    - 重新导出 ONNX，并重新转换为 MindIR。

**3. MSLite 推理输出恒为常数**

- **现象**：LLM logits 全相同/输出异常。
- **原因**：FP16 下 attention mask 极小值参与计算导致数值问题。
- **解决方案**：对 LLM 子模型转换时使用 `config.ini` 配置 `force_fp32`。

**4. MSLite 输入 dtype 不匹配**

- **现象**：报错 `required xx, given yy`。
- **原因**：MindIR 期望 `int32`，传入了 `int64`。
- **解决方案**：推理脚本中显式 `.astype(np.int32)`（本目录脚本已处理）。

**5. converter_lite 转换 `cosyvoice2_flow_encoder.onnx` 失败：`Resize` / `ResizeNearestNeighborV2` 维度错误**

- **现象**：日志类似 `OpName:[/encoder/up_layer/Resize] "The dim of input x is not 4”`（见本目录 `convert.log`）。
- **原因**：CosyVoice `UpsampleConformerEncoder` 中 `Upsample1D` 对 `(B, C, T)` 使用 `F.interpolate(..., scale_factor=stride)`，ONNX 导出为 **3 维** `Resize`；Ascend GE 上该算子要求 **4 维** 输入。
- **解决方案**：使用本仓库更新后的 `export_cosyvoice2_onnx.py` 重新导出 Flow Encoder（脚本在构建 Flow 前对 `Upsample1D.forward` 做了等价改写：`unsqueeze(-1)` → `(B, C, T, 1)` 上 `scale_factor=(stride, 1)` 的 `interpolate` → `squeeze(-1)`，使图中 `Resize` 为 4 维），再执行 `converter_lite`。

**6. MSLite 推理时报错：`Acl memcpy input X data to device failed, src input size: 0`**

- **现象**：MSLite 推理时报错，日志显示 `src input size: 0, dst device buffer size: 0`，通常发生在 LLM Prefill 阶段。
- **原因**：导出 ONNX 时使用的 `speech_len > 0`（例如 dummy 值为 6），但实际推理时 `speech_len = 0`（无 prompt 音频）。MSLite Ascend runtime 无法处理 size=0 的空 tensor，而 ONNX Runtime 可以。
- **解决方案**：在 `export_cosyvoice2_onnx.py` 中，将 LLM Prefill 导出的 dummy `speech_len` 改为 `0`（或确保 dummy 输入覆盖实际推理的最小值，包括空数组情况），重新导出 ONNX 并转换 MindIR。
    - 修改位置：`export_cosyvoice2_onnx.py` 中 `_export_llm()` 函数
    - 修改前：`speech_len = 6`
    - 修改后：`speech_len = 0`
- **通用经验**：导出 ONNX 时，dummy 输入的各维度应覆盖实际推理时的所有可能范围，特别是最小值（包括 0）。对于可选输入（如 prompt speech tokens），必须用空 dummy（长度 0）验证。

**7. CosyVoice2 MSLite 推理音频内容错误且时长异常**

- **现象**：ONNX 推理音频正常（时长约 2.1s），MSLite 推理生成音频”瞎说”（能听出是汉语但内容错误）且时长异常（约 6.72s）。
- **根因**：
    - 当 `speech_len=0`（无 prompt 音频）时，MSLite Ascend runtime 无法处理空 tensor。
    - ONNX 导出时部分维度 `dim_value=0`（如空 speech 输入），MindIR 转换后这些零维可能被错误处理。
    - 导致 prefill 阶段 KV cache 维度错误，后续 decode 生成的 token 序列异常。
- **解决方案**：

  **推理侧**：当 `speech_len == 0` 时，padding 为 `[0]` token（`speech_ids = [[0]]`），推理后从输出中移除 padding：

  ```python
  pad_empty_speech = (speech_len == 0)
  if pad_empty_speech:
      speech_ids_np = np.array([[0]], dtype=np.int64)
      speech_len = 1
  # ... run inference ...
  if pad_empty_speech:
      logits = logits[:, :-1, :]
      past_kv = past_kv[:, :, :, :-1, :]
      total_len = total_len - 1
  ```

  **导出侧**：添加 `_sanitize_onnx_zero_dims()` 函数，将 `dim_value=0` 转为 `dim_param`（动态维度）：

  ```python
  def _sanitize_onnx_zero_dims(onnx_path: Path) -> None:
      model = onnx.load(str(onnx_path))
      def sanitize_value_info(vi):
          if not vi.type.HasField("tensor_type"):
              return
          tt = vi.type.tensor_type
          if not tt.HasField("shape"):
              return
          for i, d in enumerate(tt.shape.dim):
              if d.HasField("dim_value") and int(d.dim_value) == 0 and not d.HasField("dim_param"):
                  d.dim_param = f"{vi.name}_dim{i}"
                  d.ClearField("dim_value")
      for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
          sanitize_value_info(vi)
      onnx.save(model, str(onnx_path))
  ```

- **验证方法**：添加 `--decode-mode greedy` 选项，使 ONNX/MSLite 均使用确定性采样，便于对比中间输出（logits/KV cache）定位精度差异。

---

## 8. 参考资源

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html)

---

## 9. 许可证

CosyVoice2-0.5B 模型遵循 Apache License 2.0。

