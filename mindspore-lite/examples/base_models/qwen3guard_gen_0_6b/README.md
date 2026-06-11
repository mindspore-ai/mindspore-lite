# Qwen3Guard-Gen-0.6B ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen3Guard-Gen-0.6B 的 ONNX 导出脚本及基于 MindSpore Lite 的推理脚本。

## 概览

Qwen3Guard-Gen-0.6B 是通义千问团队推出的**内容安全审核生成模型**，基于 Qwen3 架构，将安全分类任务转化为指令跟随任务。支持：

- 三级严重性分类：Safe（安全）、Controversial（争议）、Unsafe（不安全）
- 9 类安全类别识别（暴力、违法、色情、PII、自杀自伤、不道德、政治敏感、版权、越狱）
- 支持 prompt 审核（输入）和 response 审核（输出）
- 支持 119 种语言和方言

本实现采用 Prefill + Decode 分离的推理架构。

## 模型架构参数

| 参数 | 值 |
|---|---|
| `architectures` | Qwen3ForCausalLM |
| `hidden_size` | 1024 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (GQA) |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | true |

## 环境依赖

| 软件包 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.7.1 |
| transformers | 5.9.0 |
| onnx | 1.19+ |
| mindspore-lite | 2.9.0 |
| CANN | 8.5.0 |

## 快速开始

### 1. 导出 ONNX

```bash
python export_qwen3guard_gen_0_6b_onnx.py \
    --model-id ./Qwen/Qwen3Guard-Gen-0.6B \
    --output-dir ./qwen3guard_gen_0_6b_onnx \
    --device cpu
```

### 导出产物

- `qwen3guard_gen_llm_prefill.onnx`：Prefill 模型
- `qwen3guard_gen_llm_decode.onnx`：Decode 模型

## MindSpore Lite 模型转换

```bash
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3guard_gen_0_6b_onnx/qwen3guard_gen_llm_prefill.onnx \
    --outputFile=./qwen3guard_gen_0_6b_onnx/qwen3guard_gen_llm_prefill \
    --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/config.ini

converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3guard_gen_0_6b_onnx/qwen3guard_gen_llm_decode.onnx \
    --outputFile=./qwen3guard_gen_0_6b_onnx/qwen3guard_gen_llm_decode \
    --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/config.ini
```

### `./configs/config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode = allow_fp32_to_fp16
```

## MindSpore Lite 推理

```bash
python infer_qwen3guard_gen_0_6b_mslite.py \
    --prefill-model ./qwen3guard_gen_0_6b_onnx/qwen3guard_gen_llm_prefill_graph.mindir \
    --decode-model ./qwen3guard_gen_0_6b_onnx/qwen3guard_gen_llm_decode_graph.mindir \
    --tokenizer ./Qwen/Qwen3Guard-Gen-0.6B \
    --device ascend
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--tokenizer` | Tokenizer 路径 | `Qwen/Qwen3Guard-Gen-0.6B` |
| `--device` | 推理设备 | `ascend` |

## 性能数据

### 推理性能（Atlas 300I Duo）

| 指标 | 值 |
|---|---|
| Prefill 延迟 | ~100 ms |
| Decode 平均延迟（稳态） | ~35 ms/step |
| 吞吐量 | ~22 tok/s |

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3Guard-Gen-0.6B ModelScope](https://modelscope.cn/models/Qwen/Qwen3Guard-Gen-0.6B)
- [Qwen3Guard 技术报告](https://arxiv.org/abs/2510.14276)

## 许可证

本工具遵循 Qwen3 模型的许可证要求。
