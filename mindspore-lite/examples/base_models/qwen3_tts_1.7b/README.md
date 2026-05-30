# Qwen3-TTS ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-TTS 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

## 模型架构

Qwen3-TTS 是一个端到端的语音生成大模型。在本工程端到端推理链路中，模型被拆分为 4 个子模型：

1. **Talker Prefill**（`talker_prefill.onnx`）：提示词 Prefill，一次性处理文本与条件的输入，输出首步 logits、hidden 与 KV cache。
2. **Talker Step**（`talker_step.onnx`）：单步 Decode，基于前一步的隐藏状态与 KV cache 进行自回归增量生成。
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

使用 `export_qwen3_ttts_1.7b.py` 脚本导出所有子模型。通过 `--export_custom_ops` 控制是否导出带 Ascend Custom 算子的 ONNX：

```bash
# 导出纯净 ONNX（不含 Custom），可用于 CPU/CUDA 上 ONNX Runtime 验证
python export_qwen3_ttts_1.7b.py \
  --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --output_root . \
  --talker_dtype float32 \
  --speech_dtype float32

# 导出带 Ascend Custom 算子的 ONNX（用于转换 MindIR 在 Ascend 部署）
python export_qwen3_ttts_1.7b.py \
  --export_custom_ops \
  --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --output_root . \
  --talker_dtype float32 \
  --speech_dtype float32
```

> **注意**：开启 `--export_custom_ops` 后导出的 `talker_step.onnx` 及 `generate_process.onnx` 包含 Ascend Custom 算子，这类模型**无法使用 CPU/CUDA 上的 ONNX Runtime 进行推理验证**（即无法运行 `infer_qwen3_tts_1.7b_onnx.py`）。它们仅用于通过 `converter_lite` 转换为 MindIR 后在 Ascend 部署。
> 如需单独导出某一类模型，可追加 `--export_talker`、`--export_speech` 或 `--export_code_predictor`。

### 参数说明

| 参数                         | 说明                                 | 默认值                                    |
|----------------------------|------------------------------------|----------------------------------------|
| `--model_path`             | 模型路径（HuggingFace 格式或本地目录）      | `../Qwen3-TTS-12Hz-1.7B-CustomVoice`   |
| `--output_root`            | 输出根目录                           | `.`                                    |
| `--export_custom_ops`      | 是否导出带 Ascend Custom 算子的 ONNX      | 关闭                                     |
| `--talker_dtype`           | Talker 模型导出精度（float32/float16）  | `float32`                              |
| `--talker_export_seq_len`  | Prefill 导出序列长度示例               | `512`                                  |
| `--speech_dtype`           | Speech Decoder 导出精度              | `float32`                              |

### 产出

默认将输出以下目录与模型文件：

```log
./
├── onnx_models_talker_core_kv_transpose/
│   ├── talker_prefill.onnx
│   ├── talker_step.onnx
│   └── meta.json
├── onnx_models_talker_core/
│   └── generate_process.onnx
└── onnx_models_speech_tokenizer/
    └── speech_decoder.onnx
```

---

## 3. ONNX 推理

### ONNX Runtime 推理

使用 `infer_qwen3_tts_1.7b_onnx.py` 脚本在 CPU 或 CUDA 上进行 ONNX 端到端推理：

```bash
python infer_qwen3_tts_1.7b_onnx.py \
  --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --onnx_dir ./onnx_models_talker_core_kv_transpose \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --output_wav output_custom_voice.onnx.wav
```

### 参数说明

| 参数                 | 说明                       | 默认值                                        |
|--------------------|--------------------------|--------------------------------------------|
| `--model_path`     | 原模型目录（加载 tokenizer 等）| `../Qwen3-TTS-12Hz-1.7B-CustomVoice`       |
| `--onnx_dir`       | Talker 等 ONNX 模型所在目录   | `onnx_models_talker_core_fp32_no_custom`   |
| `--text`           | 输入文本                    | `"其实我真的有发现..."`                       |
| `--language`       | 语言选项                    | `Chinese`                                  |
| `--speaker`        | 发音人                      | `Vivian`                                   |
| `--output_wav`     | 输出音频路径                 | `output_custom_voice.onnx.wav`             |

---

## 4. MindSpore Lite 转换

### 转换命令

```bash
# 1. 激活 CANN 包环境
source /path/to/cann/set_env.sh

# 2. 设置 mindspore-lite 工具的环境变量与动态库路径
export MSLITE_HOME=/path/to/mindspore-lite-2.8.0-linux-aarch64
export LD_LIBRARY_PATH=${MSLITE_HOME}/runtime/lib:${MSLITE_HOME}/tools/converter/lib:${LD_LIBRARY_PATH}

# 3. 声明 converter_lite 的实际路径
export Convert=${MSLITE_HOME}/tools/converter/converter/converter_lite

# 4. 执行转换（等价于 convert.sh 中的实际调用内容）
$Convert --modelFile=./onnx_models_talker_core_kv_transpose/talker_prefill.onnx --fmk=ONNX --outputFile=./mindir/talker_prefill --optimize=ascend_oriented --configFile=./configs/config_talker_prefill.ini
$Convert --modelFile=./onnx_models_talker_core_kv_transpose/talker_step.onnx --fmk=ONNX --outputFile=./mindir/talker_step --optimize=ascend_oriented --configFile=./configs/config_talker_step.ini
$Convert --modelFile=./onnx_models_speech_tokenizer/speech_decoder.onnx --fmk=ONNX --outputFile=./mindir/speech_decoder --optimize=ascend_oriented --configFile=./configs/config_tokenizer_decoder.ini
$Convert --modelFile=./onnx_models_talker_core/generate_process.onnx --fmk=ONNX --outputFile=./mindir/generate_process --optimize=ascend_oriented --configFile=./configs/config_code_predictor.ini
```

> **说明**：如果只需要转换部分子模型，只执行对应的 `$Convert ...` 行即可。

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

转换配置位于 `configs/` 目录下，例如 `configs/config_tokenizer_decoder.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="codes:1,16,-1"
ge.dynamicDims="2;3;4;6;8;10;12;14;16;18;20;22;24;26;28;30;32;34;36;38;40;42;44;46;48;50;52;54;56;58;60"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出

成功转换后，默认在 `./mindir/` 目录下生成 `.mindir` 模型文件：

```log
mindir/
├── talker_prefill.mindir
├── talker_step.mindir
├── generate_process.mindir
└── speech_decoder.mindir
```

---

## 5. MindSpore Lite 推理

### 推理命令

使用 `infer_qwen3_tts_1_7b_mindir.py` 在 Ascend 设备上进行端到端推理：

```bash
python infer_qwen3_tts_1_7b_mindir.py \
  --mode infer \
  --model_path ../Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --mindir_dir ./mindir \
  --device_id 0 \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --language Chinese \
  --speaker Vivian \
  --output output_custom_voice.mindir.wav
```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|-------------------------|
| `--mode`           | 运行模式（`infer` 正常推理 / `compare` 精度对齐）| `infer`                 |
| `--mindir_dir`     | MindIR 模型所在目录                      | `./mindir`              |
| `--device_id`      | Ascend 设备 ID                          | `0`                     |
| `--text`           | 输入文本                                  | `"其实我真的有发现..."`   |

---

## 6. 性能数据

### 性能测试结果

*(实际性能数据请以具体环境的 Benchmark 测试结果为准)*

| 指标                       | Atlas 300I Duo Time     | Atlas 800I A2 Time     |
|--------------------------|-------------------------|------------------------|
| speech_decoder (ms)              | 157     | 8     |
| Prefill (ms)             | 132     | 15     |
| Total Decode (ms)        | 1659     | 480     |
| **Avg decode step (ms)** | **27.65**| **8**|
| Total generate_process (ms)        | 1377  | 780     |
| **Avg generate_process step (ms)** | **22.9**| **13**|
| Total (ms)               | 3548     | 1283     |
| **Throughput (tok/s)**   | **17.1**| **46.8**|

| 端到端指标 | Atlas 300I Duo | Atlas 800I A2 |
|---|---:|---:|
| RTF | 0.875     | 0.34     |

> **RTF 含义**：Real-Time Factor，端到端生成 1 秒音频所消耗的推理时间与 1 秒的比值。RTF < 1 表示实时或快于实时，RTF = 0.875 表示生成 1 秒音频耗时约 0.875 秒。
> **提示**：可使用 mindspore-lite 提供的 `benchmark` 工具直接评测 `.mindir` 的吞吐与耗时。

---

## 7. 常见问题

### Q1: 转换或推理时遇到 `aclmdlLoadFromMem failed` 或 `Load om data failed`

优先排查 CANN 环境是否正确初始化（`source set_env.sh`），配置文件是否正确（建议强制使用 `ge.exec.precision_mode=force_fp32` 规避溢出或精度错误），以及是否包含了未支持的算子或融合策略。

### Q2: 语音解码输出被截断

检查 `configs/config_tokenizer_decoder.ini` 中的 `ge.dynamicDims` 设置。如果生成的语音片段 `codes` 的序列长度超过了配置文件中枚举的最大长度，MindSpore Lite 可能会执行失败或截断。需要同步扩大动态档位配置。

### Q3: ONNXRuntime 与 MindSpore Lite 生成的音频不一致

因为生成过程存在自回归与采样（如 `repetition_penalty`、温度等参数），即使底层算子有极小的精度差异（FP32 vs Ascend 计算单元底层的浮点实现），也会导致生成出的 Token 或 Codec ID 产生分岔。可使用 `infer_qwen3_tts_1_7b_mindir.py --mode compare` 对单步推理模块进行精度比对。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-TTS 官方开源仓库](https://github.com/QwenLM/Qwen3-TTS)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本项目及模型遵循 `Apache-2.0` 许可证（以项目实际声明为准）。
