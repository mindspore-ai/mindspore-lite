# Qwen3-TTS ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-TTS 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

## 目录内容

| 文件 / 目录 | 说明 |
|---|---|
| `export_qwen3_tts_1_7b.py` | 一键导出 Talker（Prefill/Step）、Speech Decoder、Code Predictor 的 ONNX 模型（含 PTQ int8 量化） |
| `infer_qwen3_tts_1_7b_mindir.py` | MindIR 端到端推理（默认直接推理并输出 wav） |
| `configs/` | `converter_lite` 转换配置文件（默认 fp32 流程，动态维度档位等） |
| `configs_bf16/` | `converter_lite` 转换配置文件（Atlas 800I A2 bf16 流程，`precision_mode=allow_mix_precision_bf16`） |
| `quant_calib.jsonl` | PTQ int8 量化校准数据（导出 Talker 时默认使用） |

## 模型架构

Qwen3-TTS 是一个端到端的语音生成大模型。在本工程端到端推理链路中，模型被拆分为 4 个子模型：

1. **Talker Prefill**（`prefill/talker_prefill.onnx`）：提示词 Prefill，一次性处理文本与条件的输入，输出首步 logits、hidden 与 KV cache。
2. **Talker Step**（`step/talker_step.onnx`）：单步 Decode，基于前一步的隐藏状态与 KV cache 进行自回归增量生成。
3. **Code Predictor**（`generate_process.onnx`）：根据 Talker 的输出，预测具体的语音离散编码（`codec_ids`）。
4. **Speech Decoder**（`speech_decoder.onnx`）：将生成的离散编码解码为最终的音频波形数据。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.12.0 |
| torch          | 2.12.0+cu130 |
| transformers   | 4.57.3 |
| onnx           | 1.21.0 |
| onnxruntime    | 1.26.0 |
| soundfile      | 0.13.1 |
| qwen3-tts      | 0.1.1 |
| mindspore-lite | 2.8.0 |

> **注意**：Ascend 推理需要正确配置 CANN 环境变量，并安装 mindspore-lite 对应版本。

---

## 2. 模型导出 ONNX

### 导出命令

使用 `export_qwen3_tts_1_7b.py` 脚本一键导出全部子模型（Talker Prefill/Step、Speech Decoder、Code Predictor）：

```bash
python export_qwen3_tts_1_7b.py \
  --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --output_root ./onnx_models \
  --dtype float32 \
  --talker_export_seq_len 32 \
  --speech_example_seq_len 16
```

> **Atlas 800I A2（bf16）流程**：在 A2 上部署时，导出需将 `--dtype` 设置为 `bfloat16`、`--code_predictor_ifa_layout` 设置为 `BSND`：
>
> ```bash
> python export_qwen3_tts_1_7b.py \
>   --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
>   --output_root ./onnx_models_bf16 \
>   --dtype bfloat16 \
>   --code_predictor_ifa_layout BSND \
>   --talker_export_seq_len 32 \
>   --speech_example_seq_len 16
> ```
>
> **说明**：
> - 导出的 ONNX 包含 Ascend Custom 算子（如 `RotaryMul`、`Scatter`、`IncreFlashAttention`、`AscendQuant`、`QuantBatchMatmul` 等），**无法使用 CPU/CUDA 上的 ONNX Runtime 直接推理**，仅用于通过 `converter_lite` 转换为 MindIR 后在 Ascend 部署。
> - 当 `--ptq-calib-data` 指向的文件存在时（默认 `./quant_calib.jsonl`），Talker 导出会默认启用 **PTQ int8 MatMul 量化**（SmoothQuant + int8 权重）；如需关闭可追加 `--disable-ptq`。

### 参数说明

| 参数                         | 说明                                 | 默认值                                    |
|----------------------------|------------------------------------|----------------------------------------|
| `--model_path`             | 模型路径（HuggingFace 格式或本地目录）      | `../Qwen3-TTS-12Hz-1.7B-CustomVoice`   |
| `--output_root`            | 输出根目录                           | `./onnx_batch_fp32`                    |
| `--opset`                  | ONNX opset 版本                     | `17`                                   |
| `--dtype`                  | 导出精度（float32/float16/bfloat16）  | `float32`                              |
| `--talker_export_seq_len`  | Talker Prefill 导出序列长度           | `32`                                   |
| `--speech_example_seq_len` | Speech Decoder 示例序列长度          | `16`                                   |
| `--code_predictor_ifa_layout` | Code Predictor IFA 布局          | `BNSD`（可选 `BSND`）                   |
| `--ptq-calib-data`         | PTQ 校准数据路径                      | `./quant_calib.jsonl`                  |
| `--ptq-smooth-alpha`       | SmoothQuant alpha（0=纯 weight, 1=纯激活） | `0.65`                             |
| `--ptq-weight-clip-ratio`  | Weight 离群值裁剪比例                 | `0.01`                                 |
| `--ptq-skip-layers`        | 跳过量化的层                         | 默认跳过 `layer.0-11` 与 `layer.16-27`  |
| `--disable-ptq`            | 禁用 PTQ int8 量化                  | 关闭                                    |

### 产出

默认将输出以下目录与模型文件（均在 `--output_root` 下）：

```log
./onnx_models/
├── prefill/
│   └── talker_prefill.onnx
├── step/
│   └── talker_step.onnx
├── generate_process.onnx
└── speech_decoder.onnx
```

各子模型输入/输出：

| 模型 | 输入 | 输出 |
|---|---|---|
| `prefill/talker_prefill.onnx` | `inputs_embeds`、`attention_mask` | `logits_last`、`hidden_last`、`past_k`、`past_v`、`prompt_len` |
| `step/talker_step.onnx` | `step_embed`、`past_k`、`past_v`、`position_ids_step`、`cache_len` | `logits_last`、`hidden_last`、`past_k_out`、`past_v_out` |
| `generate_process.onnx` | `inputs_embeds`、`next_id`、`last_id_hidden`、`trailing_step` | `codec_ids`、`step_embed` |
| `speech_decoder.onnx` | `codes` | `wav` |

---

## 3. MindSpore Lite 转换

### 转换命令

```bash
# 1. 激活 CANN 包环境
source /path/to/cann/set_env.sh

# 2. 设置 mindspore-lite 工具的环境变量与动态库路径
export MSLITE_HOME=/path/to/mindspore-lite-2.8.0-linux-aarch64
export LD_LIBRARY_PATH=${MSLITE_HOME}/runtime/lib:${MSLITE_HOME}/tools/converter/lib:${LD_LIBRARY_PATH}

# 3. 声明 converter_lite 的实际路径
export Convert=${MSLITE_HOME}/tools/converter/converter/converter_lite

# 4. 执行转换（每个子模型对应一份配置文件）
$Convert --modelFile=./onnx_models/prefill/talker_prefill.onnx --fmk=ONNX --outputFile=./mindir/talker_prefill --optimize=ascend_oriented --configFile=./configs/config_talker_prefill.ini
$Convert --modelFile=./onnx_models/step/talker_step.onnx --fmk=ONNX --outputFile=./mindir/talker_step --optimize=ascend_oriented --configFile=./configs/config_talker_step.ini
$Convert --modelFile=./onnx_models/speech_decoder.onnx --fmk=ONNX --outputFile=./mindir/speech_decoder --optimize=ascend_oriented --configFile=./configs/config_tokenizer_decoder.ini
$Convert --modelFile=./onnx_models/generate_process.onnx --fmk=ONNX --outputFile=./mindir/generate_process --optimize=ascend_oriented --configFile=./configs/config_code_predictor.ini
```

> **说明**：`infer_qwen3_tts_1_7b_mindir.py` 期望的 MindIR 文件名为 `talker_prefill_graph.mindir`、`talker_step_graph.mindir`、`generate_process.mindir`、`speech_decoder.mindir`，请保持上述 `--outputFile` 命名。如果只需要转换部分子模型，只执行对应的 `$Convert ...` 行即可。
>
> **Atlas 800I A2（bf16）流程**：A2 上 bf16 模型转换需使用 `configs_bf16/` 下的配置文件（`ge.exec.precision_mode=allow_mix_precision_bf16`）：
>
> ```bash
> $Convert --modelFile=./onnx_models_bf16/prefill/talker_prefill.onnx --fmk=ONNX --outputFile=./mindir_bf16/talker_prefill --optimize=ascend_oriented --configFile=./configs_bf16/config_talker_prefill.ini
> $Convert --modelFile=./onnx_models_bf16/step/talker_step.onnx --fmk=ONNX --outputFile=./mindir_bf16/talker_step --optimize=ascend_oriented --configFile=./configs_bf16/config_talker_step.ini
> $Convert --modelFile=./onnx_models_bf16/speech_decoder.onnx --fmk=ONNX --outputFile=./mindir_bf16/speech_decoder --optimize=ascend_oriented --configFile=./configs_bf16/config_tokenizer_decoder.ini
> $Convert --modelFile=./onnx_models_bf16/generate_process.onnx --fmk=ONNX --outputFile=./mindir_bf16/generate_process --optimize=ascend_oriented --configFile=./configs_bf16/config_code_predictor.ini
> ```

### 参数说明

`converter_lite` 的关键参数说明：

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径（指定动态维度等）           |

### 配置文件

转换配置位于 `configs/` 目录下，各子模型对应一份配置，例如 `configs/config_tokenizer_decoder.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="codes:-1,16,60"
ge.dynamicDims="1;2;5"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

其余配置内容摘要：

| 配置文件 | input_shape | ge.dynamicDims |
|---|---|---|
| `config_talker_prefill.ini` | `inputs_embeds:-1,32,2048;attention_mask:-1,32` | `1,1;2,2;5,5` |
| `config_talker_step.ini` | `step_embed:-1,1,2048;past_k:28,-1,8,512,128;past_v:28,-1,8,512,128;position_ids_step:3,-1,1;cache_len:-1,1` | `1,1,1,1,1;2,2,2,2,2;5,5,5,5,5` |
| `config_code_predictor.ini` | `inputs_embeds:-1,2,2048;next_id:-1,1;last_id_hidden:-1,1,2048;trailing_step:-1,1,2048` | `1,1,1,1;2,2,2,2;5,5,5,5` |
| `config_tokenizer_decoder.ini` | `codes:-1,16,60` | `1;2;5` |

> **bf16 配置（`configs_bf16/`）**：Atlas 800I A2 bf16 流程使用 `configs_bf16/` 下的同名配置文件，动态维度档位与 `configs/` 完全一致，仅将 `ge.exec.precision_mode` 由 `force_fp32` 改为 `allow_mix_precision_bf16`。

### 产出

成功转换后，默认在 `./mindir/` 目录下生成 `.mindir` 模型文件：

```log
mindir/
├── talker_prefill_graph.mindir
├── talker_step_graph.mindir
├── generate_process.mindir
└── speech_decoder.mindir
```

---

## 4. MindSpore Lite 推理

### 推理命令

使用 `infer_qwen3_tts_1_7b_mindir.py` 在 Ascend 设备上进行端到端推理：

```bash
python infer_qwen3_tts_1_7b_mindir.py \
  --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --mindir_dir ./mindir \
  --input_dtype float32 \
  --device_id 0 \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --max_new_tokens 60 \
  --output output_custom_voice.mindir.wav
```

> **Atlas 800I A2（bf16）流程**：A2 上推理 bf16 MindIR 时，需将 `--dtype` 与 `--input_dtype` 均指定为 `bfloat16`，并将 `--mindir_dir` 指向 bf16 转换产出的目录：
>
> ```bash
> python infer_qwen3_tts_1_7b_mindir.py \
>   --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
>   --mindir_dir ./mindir_bf16 \
>   --dtype bfloat16 \
>   --input_dtype bfloat16 \
>   --device_id 0 \
>   --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
>   --language Chinese \
>   --speaker Vivian \
>   --max_new_tokens 60 \
>   --output output_custom_voice.mindir_bf16.wav
> ```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|-------------------------|
| `--model_path`     | 原模型目录（加载 tokenizer 等）              | `../Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `--device_map`     | 原模型加载设备                             | `cpu`                   |
| `--dtype`          | 原模型加载精度                             | `float32`               |
| `--input_dtype`    | MindIR 输入精度（float32/bfloat16）       | `float32`               |
| `--mindir_dir`     | MindIR 模型所在目录                      | `./mindir`              |
| `--device_id`      | Ascend 设备 ID                          | `0`                     |
| `--seed`           | 随机种子                                 | `0`                     |
| `--output`         | 输出 wav 路径                            | `output_custom_voice.mindir.wav` |
| `--dump-calib`     | 追加一条 PTQ 校准记录到 JSONL 文件           | 空                       |
| `--text`           | 输入文本                                  | `"其实我真的有发现..."`   |
| `--language`       | 语言选项                                  | `Chinese`               |
| `--speaker`        | 发音人                                    | `Vivian`                |
| `--max_new_tokens` | 最大新 token 数                          | `256`                   |

---

## 5. 性能数据

### 性能测试结果

*(实际性能数据请以具体环境的 Benchmark 测试结果为准)*

| 指标                       | Atlas 300I Duo Time     | Atlas 800I A2 Time     |
|--------------------------|-------------------------|------------------------|
| speech_decoder (ms)              | 157     | 52     |
| Prefill (ms)             | 132     | 11     |
| Total Decode (ms)        | 1659     | 362     |
| **Avg decode step (ms)** | **27.65**| **6.04**|
| Total generate_process (ms)        | 1377  | 534     |
| **Avg generate_process step (ms)** | **22.9**| **8.9**|
| Total (ms)               | 3548     | 1192     |
| **Throughput (tok/s)**   | **17.1**| **50.3**|

| 端到端指标 | Atlas 300I Duo | Atlas 800I A2 |
|---|---:|---:|
| RTF | 0.875     | 0.3     |

> **RTF 含义**：Real-Time Factor，端到端生成 1 秒音频所消耗的推理时间与 1 秒的比值。RTF < 1 表示实时或快于实时，RTF = 0.875 表示生成 1 秒音频耗时约 0.875 秒。
> **提示**：可使用 mindspore-lite 提供的 `benchmark` 工具直接评测 `.mindir` 的吞吐与耗时。

---

## 6. 常见问题

### Q1: 转换或推理时遇到 `aclmdlLoadFromMem failed` 或 `Load om data failed`

优先排查 CANN 环境是否正确初始化（`source set_env.sh`），配置文件是否正确（建议强制使用 `ge.exec.precision_mode=force_fp32` 规避溢出或精度错误），以及是否包含了未支持的算子或融合策略。

### Q2: 语音解码输出被截断

检查 `configs/config_tokenizer_decoder.ini` 中的动态维度配置。推理脚本按 60 帧切块解码（`codes` 形状 `[batch, 16, 60]`），`ge.dynamicDims="1;2;5"` 为 batch 档位；如果实际 batch 或单次解码帧数超出配置范围，MindSpore Lite 可能会执行失败或截断。需要同步调整 `input_shape` 与动态档位配置。

### Q3: ONNXRuntime 与 MindSpore Lite 生成的音频不一致

因为生成过程存在自回归与采样（如 `repetition_penalty`、温度等参数），即使底层算子有极小的精度差异（FP32 vs Ascend 计算单元底层的浮点实现），也会导致生成出的 Token 或 Codec ID 产生分岔。属于自回归采样的正常现象，可先确认 `--input_dtype` 与导出时的 `--dtype` 是否一致，或使用 `ge.exec.precision_mode=force_fp32` 强制浮点精度后再对比。

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-TTS 官方开源仓库](https://github.com/QwenLM/Qwen3-TTS)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本项目及模型遵循 `Apache-2.0` 许可证（以项目实际声明为准）。
