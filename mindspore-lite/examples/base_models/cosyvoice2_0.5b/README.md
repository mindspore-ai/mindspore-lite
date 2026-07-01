# CosyVoice2-0.5B ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 CosyVoice2-0.5B 拆分为 5 个子模型导出为 ONNX，转换为 MindSpore Lite MindIR，
并在 Ascend NPU 上完成端到端文本→语音合成推理。

CosyVoice2-0.5B 是阿里通义实验室的 0.5B 参数语音合成模型，支持零样本语音克隆、跨语言合成、
自然语言控制等功能。本目录把模型按推理流程拆分为 5 个 ONNX：

1. **LLM Prefill**（`cosyvoice2_llm_prefill.onnx`）：一次性处理文本 token + 可选 prompt speech token，
   输出首个 speech token 的 logits 与初始 KV cache
2. **LLM Decode**（`cosyvoice2_llm_decode.onnx`）：基于 past KV cache 自回归生成后续 speech token
3. **Flow Encoder**（`cosyvoice2_flow_encoder.onnx`）：把 speech token 解码为 Flow 匹配的初始条件
   （`mu` / `spks` / `cond` / `mask`）
4. **Flow Estimator**（`cosyvoice2_flow_estimator.onnx`）：CFM 10 步欧拉采样，把随机噪声变成 mel
5. **HiFT Vocoder**（`cosyvoice2_hift.onnx`）：mel → 波形

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
| matcha-tts     | 0.0.7.2 |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite qwen-asr==0.0.6
```

### 获取模型权重与源码

> **注意**：CosyVoice 源码仅在 **ONNX 导出** 阶段被 import（用于加载 `Qwen2LM` / `HiFTGenerator` / Flow 等模型定义）；推理阶段完全基于 MindIR，不需要源码。

```bash
# CosyVoice 源码（仅导出 ONNX 时需要）
git clone https://github.com/FunAudioLLM/CosyVoice.git

# CosyVoice2-0.5B 权重（参考 CosyVoice 项目文档下载）
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/cosyvoice2_0.5b

python export_cosyvoice2_onnx.py \
  --model-dir /path/to/CosyVoice2-0.5B \
  --model-code-dir /path/to/CosyVoice \
  --output-dir ./pfa_fused/cosyvoice2_onnx
```

可用 `--skip-llm` / `--skip-flow` / `--skip-hift` 单独跳过某个子模型。

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-dir` | CosyVoice2-0.5B 权重目录 | 见脚本默认值 |
| `--model-code-dir` | CosyVoice 源码目录 | 见脚本默认值 |
| `--output-dir` | ONNX 输出目录 | `./pfa_fused/cosyvoice2_onnx` |
| `--device` | 导出设备（cpu） | `cpu` |
| `--skip-llm` | 跳过 LLM 导出 | `False` |
| `--skip-flow` | 跳过 Flow 导出 | `False` |
| `--skip-hift` | 跳过 HiFT 声码器导出 | `False` |
| `--disable-fusion` | 关闭 LLM Decode attention 的 `Custom(PromptFlashAttention)` 融合 | `False` |
| `--disable-fusion-estimator` | 关闭 Flow Estimator attention 融合（monkey-patch diffusers） | `False` |

### 产出

```text
pfa_fused/cosyvoice2_onnx/
├── cosyvoice2_llm_prefill.onnx       # LLM Prefill 模型 (~1.9 GB)
├── cosyvoice2_llm_decode.onnx        # LLM Decode 模型  (~1.4 GB)
├── cosyvoice2_flow_encoder.onnx      # Flow Encoder    (~178 MB)
├── cosyvoice2_flow_estimator.onnx    # Flow Estimator  (~274 MB)
└── cosyvoice2_hift.onnx              # HiFT Vocoder    (~80 MB, T_mel 纯动态)
```

### ONNX 模型输入输出 Shape

**LLM Prefill** — `cosyvoice2_llm_prefill.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| 输入 | `text_ids` | `(batch, text_len)` | int32 | 文本 token |
| 输入 | `speech_ids` | `(batch, speech_len)` | int32 | prompt speech token（可为 0） |
| 输入 | `attention_mask` | `(batch, total_len)` | int32 | 注意力掩码 |
| 输入 | `position_ids` | `(batch, total_len)` | int32 | 位置 ID |
| 输出 | `logits` | `(batch, total_len, 6564)` | float32 | 下一个 speech token 预测 |
| 输出 | `present_key_values` | `(48, batch, 2, total_len, 64)` | float32 | 初始 KV cache |

> `48 = 24 层 × 2 (Q/KV)`，`num_kv_heads=2`，`head_dim=64`。Qwen2-0.5B 配置：
> `num_attention_heads=14`、`num_key_value_heads=2`、`hidden_size=896`。

**LLM Decode** — `cosyvoice2_llm_decode.onnx`（动态分档 256/512）

| 方向 | 名称 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| 输入 | `speech_id` | `(1, 1)` | int32 | 单步 speech token |
| 输入 | `attention_mask` | `(1, total_seq+1)` | int32 | 累积掩码（动态维） |
| 输入 | `position_ids` | `(1, 1)` | int32 | 单步位置 |
| 输入 | `past_key_values` | `(48, 1, 2, past_seq, 64)` | float32 | KV cache（动态维） |
| 输出 | `logits` | `(1, 1, 6564)` | float32 | 单步 logits |
| 输出 | `present_key_values` | `(48, 1, 2, total_seq+1, 64)` | float32 | 更新后 KV cache |

> 档位定义：`ge.dynamicDims="257,256;513,512"`。第一列是 `attention_mask` 长度（past_seq+1），
> 第二列是 `past_key_values` 的 seq_len。档位选择逻辑见 `infer_cosyvoice2_mslite.py::_pick_llm_decode_gear`。

**Flow Encoder** — `cosyvoice2_flow_encoder.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| 输入 | `token` | `(batch, token_len)` | int32 | speech token 序列 |
| 输入 | `token_len` | `(batch,)` | int32 | 实际长度 |
| 输入 | `embedding` | `(batch, 192)` | float32 | prompt 音色 embedding |
| 输入 | `prompt_feat` | `(batch, prompt_len, 80)` | float32 | prompt mel 特征 |
| 输出 | `mu` / `spks` / `cond` / `mask` | — | float32 | Flow Estimator 初始条件 |

**Flow Estimator** — `cosyvoice2_flow_estimator.onnx`（动态分档 128/256/512/1024）

| 方向 | 名称 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| 输入 | `x` | `(2, 80, mel_len)` | float32 | 噪声（batch=2 因 CFG） |
| 输入 | `mask` | `(2, 1, mel_len)` | float32 | mel mask |
| 输入 | `mu` | `(2, 80, mel_len)` | float32 | Flow 目标 |
| 输入 | `t` | `(2,)` | float32 | CFM 时间步 |
| 输入 | `spks` | `(2, 80)` | float32 | 音色 |
| 输入 | `cond` | `(2, 80, mel_len)` | float32 | 条件 |
| 输出 | `estimator_out` | `(2, 80, mel_len)` | float32 | 速度场估计 |

> batch=2 是 CFG（Classifier Free Guidance）始终开启导致。档位定义：
> `ge.dynamicDims="128,128,128,128;256,256,256,256;512,512,512,512;1024,1024,1024,1024"`，
> 对应 4 个 `-1` 维（`x`/`mask`/`mu`/`cond` 的 mel_len）。

**HiFT Vocoder** — `cosyvoice2_hift.onnx`（纯动态）

| 方向 | 名称 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| 输入 | `mel` | `(1, T_mel, 80)` | float32 | mel 特征（T_mel 动态） |
| 输出 | `wav` | `(1, 1, T_wav)` | float32 | 波形（T_wav = T_mel × 480） |

> HiFT 纯动态通过 `patch_conv_transpose1d_dynamic` 把 `F.conv_transpose1d` 替换为等价的
> `Conv1d-on-dilated-input` 实现，绕过 Ascend `te_conv2dtranspose` 动态 shape 崩溃 bug。

### 导出注意事项

- **LLM Prefill 的 dummy `speech_len` 必须设为 `0`**：实际推理经常没有 prompt 音频（`speech_ids` 为空），
  导出时若用 `speech_len > 0` 的 dummy，MSLite Ascend runtime 无法处理 size=0 的 tensor。

---

## 3. ONNX 转 MindIR

### 转换命令

> **重要**：所有子模型转换时都必须指定对应的 `--configFile`，配置 `force_fp32` 避免 FP16 下
> attention mask 极小值带来的数值问题。

```bash
Convert=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# LLM Prefill（纯动态，config.ini 仅设 force_fp32）
$Convert --fmk=ONNX \
  --modelFile=pfa_fused/cosyvoice2_onnx/cosyvoice2_llm_prefill.onnx \
  --outputFile=pfa_fused/cosyvoice2_mindir/cosyvoice2_llm_prefill \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=config.ini

# LLM Decode（动态分档 256/512）
$Convert --fmk=ONNX \
  --modelFile=pfa_fused/cosyvoice2_onnx/cosyvoice2_llm_decode.onnx \
  --outputFile=pfa_fused/cosyvoice2_mindir/cosyvoice2_llm_decode \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=llm_decode.ini

# Flow Encoder（纯动态）
$Convert --fmk=ONNX \
  --modelFile=pfa_fused/cosyvoice2_onnx/cosyvoice2_flow_encoder.onnx \
  --outputFile=pfa_fused/cosyvoice2_mindir/cosyvoice2_flow_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=config.ini

# Flow Estimator（动态分档 128/256/512/1024）
$Convert --fmk=ONNX \
  --modelFile=pfa_fused/cosyvoice2_onnx/cosyvoice2_flow_estimator.onnx \
  --outputFile=pfa_fused/cosyvoice2_mindir/cosyvoice2_flow_estimator \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=flow_estimator.ini

# HiFT Vocoder（纯动态）
$Convert --fmk=ONNX \
  --modelFile=pfa_fused/cosyvoice2_onnx/cosyvoice2_hift.onnx \
  --outputFile=pfa_fused/cosyvoice2_mindir/cosyvoice2_hift \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=hift.ini
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--fmk` | 输入模型格式（ONNX） |
| `--modelFile` | 输入 ONNX 模型路径 |
| `--outputFile` | 输出 MindIR 路径（不带扩展名） |
| `--optimize` | 优化模式，必须指定 `ascend_oriented` |
| `--saveType` | 输出格式（MINDIR） |
| `--configFile` | 配置文件路径（指定 input_shape / ge.dynamicDims / force_fp32） |

### 配置文件

**`config.ini`** — 纯动态子模型共用（LLM Prefill、Flow Encoder）：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

**`llm_decode.ini`** — LLM Decode 动态分档（256/512）：

```ini
[acl_build_options]
input_format="ND"
input_shape="speech_id:1,1;attention_mask:1,-1;position_ids:1,1;past_key_values:48,1,2,-1,64"
ge.dynamicDims="257,256;513,512"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

**`flow_estimator.ini`** — Flow Estimator 动态分档（128/256/512/1024，CFG batch=2）：

```ini
[acl_build_options]
input_format="ND"
input_shape="x:2,80,-1;mask:2,1,-1;mu:2,80,-1;t:2;spks:2,80;cond:2,80,-1"
ge.dynamicDims="128,128,128,128;256,256,256,256;512,512,512,512;1024,1024,1024,1024"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

**`hift.ini`** — HiFT 纯动态：

```ini
[acl_build_options]
input_format="ND"
input_shape="mel:1,-1,80"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出

模型超过 2GB 时，MindIR 会拆分为 `*_graph.mindir` + `*_variables/` 目录：

```text
pfa_fused/cosyvoice2_mindir/
├── cosyvoice2_llm_prefill_graph.mindir        # Prefill 图定义 (~1.9 KB)
├── cosyvoice2_llm_prefill_variables/          # Prefill 权重数据 (~2.7 GB)
├── cosyvoice2_llm_decode.mindir               # Decode 单文件 (~735 MB)
├── cosyvoice2_flow_encoder.mindir             # Flow Encoder (~204 MB)
├── cosyvoice2_flow_estimator.mindir           # Flow Estimator (~210 MB)
└── cosyvoice2_hift.mindir                     # HiFT Vocoder (~120 MB)
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_cosyvoice2_mslite.py \
  --mindir-dir ./pfa_fused/cosyvoice2_mindir \
  --model-dir /path/to/CosyVoice2-0.5B \
  --device ascend --device-id 0 \
  --text "你好，很高兴认识你。" \
  --output output.wav \
  --seed 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--mindir-dir` | MindIR 模型目录 | `./cosyvoice2_mindir` |
| `--model-dir` | CosyVoice2-0.5B 权重目录（用于加载 tokenizer、speech_tokenizer 等附属资源） | 见脚本默认值 |
| `--device` | 推理设备（cpu/ascend） | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--text` | 待合成文本 | `你好，很高兴认识你。` |
| `--prompt-wav` | 可选提示音频（语音克隆） | `None` |
| `--output` | 输出 wav 路径 | `output.wav` |
| `--seed` | 随机种子 | `0` |
| `--flow-cfg` | Flow CFG 系数 | `0.7` |
| `--flow-steps` | Flow Euler 步数 | `10` |
| `--decode-mode` | LLM 解码模式（greedy/ras） | `ras` |

### 动态 shape 调度逻辑

`infer_cosyvoice2_mslite.py` 内部按子模型分别处理 resize / pad：

| 子模型 | 函数 | 调度逻辑 |
|---|---|---|
| LLM Prefill | `_run_llm_prefill` | 每次按实际 shape `resize` |
| LLM Decode | `_run_llm_decode_step` + `_pick_llm_decode_gear` | 按 `past_seq_len` 选 256/512 档，pad `attention_mask` 和 `past_key_values` 到档位 |
| Flow Encoder | `_run_flow_encoder` | 每次按实际 shape `resize` |
| Flow Estimator | `_run_flow_estimator` + `_pick_flow_est_gear` | 按 `mel_len` 选 128/256/512/1024 档，pad `x`/`mask`/`mu`/`cond` 到档位 |
| HiFT Vocoder | `_run_hift_mslite` | 每次按实际 `T_mel` `resize`，无 padding |

档位常量定义在 `infer_cosyvoice2_mslite.py:175,178`：

```python
LLM_DECODE_GEARS = (256, 512)
FLOW_EST_GEARS = (128, 256, 512, 1024)
```

### 推理示例输出

输入文本 `你好，很高兴认识你。`（seed=0）：

```text
Initializing MindSpore Lite context for Ascend...
Loading LLM Prefill from pfa_fused/cosyvoice2_mindir/cosyvoice2_llm_prefill_graph.mindir...
Loading LLM Decode from pfa_fused/cosyvoice2_mindir/cosyvoice2_llm_decode.mindir...
Loading Flow Encoder from pfa_fused/cosyvoice2_mindir/cosyvoice2_flow_encoder.mindir...
Loading Flow Estimator from pfa_fused/cosyvoice2_mindir/cosyvoice2_flow_estimator.mindir...
Loading HiFT (pure dynamic) from pfa_fused/cosyvoice2_mindir/cosyvoice2_hift.mindir...
Input text: 你好，很高兴认识你。

[1/4] Running LLM (Prefill + Decode)...
  LLM generated 77 tokens...
  LLM finished: 78 speech tokens
  LLM prefill: 95.34 ms
  LLM decode : 878.10 ms (steps=78, avg_step=11.26 ms)
  LLM total  : 973.44 ms

[2/4] Running Flow Encoder...
  Flow Encoder: 39.88 ms

[3/4] Running Flow Estimator (CFM)...
  Flow Estimator (10 steps, cfg=0.7): 189.03 ms

[4/4] Running HiFT vocoder...
  HiFT vocoder: 36.00 ms

Saved to output.wav (2.68s)
Total time: 1220.38 ms (RTF: 0.455)
```

如需语音克隆，加上 `--prompt-wav /path/to/reference.wav`。

---

## 5. 性能数据

### 测试环境

| 项目 | 配置 |
|---|---|
| 硬件 | Atlas 300I Duo |
| 模型 | CosyVoice2-0.5B |
| 精度 | force_fp32（所有子模型） |
| CANN | 8.5.0 |
| MindSpore Lite | 2.9.0 |

### 各子模型性能（输入 `你好，很高兴认识你。`，输出 ~2.7s 音频）

| 阶段 | Shape 策略 | 实际档位 | 耗时 (ms) |
|---|---|---|---:|
| LLM Prefill | 纯动态 | — | 95.34 |
| LLM Decode (78 步) | 动态分档 256/512 | 256 (past_seq ≤ 256) | 878.10 (avg 11.26/step) |
| Flow Encoder | 纯动态 | — | 39.88 |
| Flow Estimator (10 步) | 动态分档 128/256/512/1024 | 256 (mel_len=134) | 189.03 |
| HiFT Vocoder | 纯动态 | — | 36.00 |
| **Total** | — | — | **1220.38** |
| **RTF** | — | — | **0.455** |

> RTF (Real-Time Factor) = 总推理时间 / 音频时长。RTF < 1 表示快于实时。

---

## 6. 常见问题

**1. 导出时报 `GuardOnDataDependentSymNode`**

- **现象**：`torch.onnx.export` 导出失败
- **原因**：新导出器内部走 `torch.export`，不支持数据依赖控制流
- **解决**：导出时使用 legacy 导出器（脚本已通过 `dynamo=False` 固定）

**2. 导出 Flow Encoder 后，ONNX 推理"首句清晰，后续全是噪声"**

- **现象**：生成音频只有最前面（例如"你好"）清晰，后面变成噪声
- **根因**：Flow Encoder 导出时 `mask` 被错误地固定为 dummy 导出长度，导致真实推理时后半段 mel 帧被 mask 掉
- **解决**：在 `export_cosyvoice2_onnx.py` 的 Flow Encoder wrapper 中，用动态 shape 构造全 1 mask：
  `attn_mask = h.new_ones((h.shape[0], 1, mel_len))`

**3. MSLite 推理输出恒为常数**

- **现象**：LLM logits 全相同/输出异常
- **原因**：FP16 下 attention mask 极小值参与计算导致数值问题
- **解决**：所有子模型转换时使用 `config*.ini` 配置 `force_fp32`

**4. HiFT 转换时报 `input x shape should be 4D` 或 `If node`**

- **现象**：`converter_lite` 报 `i: 3 out of range: 3, cnode: If`
- **根因**：`ManualISTFT.forward` 用了 `y.squeeze(1)`，ONNX tracer 会发射控制流 `If` 节点
- **解决**：用 `y[:, 0]` 替代 `squeeze(1)`（脚本已修复）

**5. HiFT 推理报 `op[/ups.0/Mul] shape cannot broadcast`**

- **现象**：纯动态 HiFT 推理时 aicore 异常
- **根因**：`F.conv_transpose1d` 替换实现里 `(input.unsqueeze(-1) * mask)` 的 broadcast 在 Ascend 上被融合后无法推导动态 shape
- **解决**：用 `repeat_interleave(S) + arange/mod mask` 构造同形 mul（见 `patch_conv_transpose1d_dynamic`）

**6. LLM Prefill 推理报 size=0 tensor 错误**

- **现象**：无 prompt 音频时崩溃
- **解决**：导出 LLM Prefill 时 dummy `speech_len` 必须设为 `0`

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice2 技术报告](https://funaudiollm.github.io/cosyvoice2/)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
