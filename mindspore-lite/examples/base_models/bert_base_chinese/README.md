# BERT-base-chinese ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 BERT-base-chinese (BertForMaskedLM) 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。BERT-base-chinese模型是针对中文数据集进行预训练而成的模型，并且对词片段应用了训练和随机输入掩码，模型相关参数：num_hidden_layers=12、vocab_size=21128、type_vocab_size=2。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.10   |
| torch          | 2.11.0 |
| transformers   | 4.40.0 |
| onnx           | 1.20.1 |
| onnxruntime    | 1.23.2 |
| CANN           | 8.5.0  |
| mindspore-lite | 2.8.0  |

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/base_models/bert-base-chinese

python export_bert_base_chinese_onnx.py \
  --model-path ./bert-base-chinese \
  --output-dir ./bert_base_chinese_onnx
```

### 参数说明

| 参数             | 说明         | 默认值      |
|----------------|------------|-----------|
| `--model-path` | 模型路径       | `./bert-base-chinese` |
| `--output-dir` | 输出目录       | `./bert_base_chinese_onnx` |
| `--device`     | 设备类型       | `cpu`     |
| `--opset`      | ONNX opset 版本 | `14`      |

### 产出

```log
bert-base-chinese/
├── bert_base_chinese_onnx/
│   └── bert_base_chinese.onnx     # ONNX 模型 (~390MB)
└── bert-base-chinese/              # 原始模型权重
```

---

## 3. ONNX 推理

### ONNX Runtime 推理

```bash
python infer_bert_base_chinese_onnx.py \
  --model ./bert_base_chinese_onnx/bert_base_chinese.onnx \
  --tokenizer ./bert-base-chinese \
  --text "今天天气很好，我[MASK]外面去玩。" \
  --top-k 5
```

**执行日志：**

```log
Loading ONNX model from: ./bert_base_chinese_onnx/bert_base_chinese.onnx
Loading tokenizer from: ./bert-base-chinese

Input text: 今天天气很好，我[MASK]外面去玩。

==================================================
Predictions:

[MASK] at position 9:
  到: 17.0824
  去: 14.5169
  在: 12.9820
  们: 11.9745
  往: 11.3062
==================================================

Average latency: 47.15 ms
```

---

## 4. MindSpore Lite 转换

### 配置文件

创建 `config.ini`：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 转换命令

```bash
cd bert_base_chinese_onnx

Converter=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

$Converter --fmk=ONNX \
  --modelFile=./bert_base_chinese.onnx \
  --outputFile=./bert_base_chinese_ascend \
  --optimize=ascend_oriented \
  --configFile=./config.ini
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径                      |

### 产出

```log
bert_base_chinese_onnx/
├── bert_base_chinese.onnx       # ONNX 模型 (~390MB)
└── bert_base_chinese_ascend.mindir     # Ascend 优化版 MindIR (~447MB)
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_bert_base_chinese_mslite.py \
  --model ./bert_base_chinese_onnx/bert_base_chinese_ascend.mindir \
  --tokenizer ./bert-base-chinese \
  --text "今天天气很好，我[MASK]外面去玩。" \
  --top-k 5 \
  --device ascend
```

**执行日志：**

```log
Loading MindIR model from: ./bert_base_chinese_onnx/bert_base_chinese_ascend.mindir
Loading tokenizer from: ./bert-base-chinese
Using device: ascend

Input text: 今天天气很好，我[MASK]外面去玩。

==================================================
Predictions:

[MASK] at position 9:
  到: 17.0850
  去: 14.5203
  在: 12.9850
  们: 11.9785
  往: 11.3137
==================================================

Average latency: 5.85 ms
```

---

## 6. 性能数据

### 性能测试结果（Atlas 300I Duo）

测试模型：BERT-base-chinese (BertForMaskedLM)
测试条件：输入序列长度 32，Atlas 300I Duo，CANN 8.5.0，MindSpore Lite 2.8.0

| 指标            | Mean (ms) |
|---------------|-----------|
| 延迟            | 5.85      |

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本教程遵循 BERT 模型的许可证。
