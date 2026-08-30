# MindSpore Lite LLM

端侧大语言模型推理模块（HarmonyOS / Kirin NPU）：**C++ 推理引擎** + **Python 转换工具链**，
面向 `.msl` 单文件模型包。

- **推理引擎**（C++17，`include/llm/` + `src/`）：模型加载、tokenize、prefill/decode 生成循环、
  采样，通过纯 C API（`MSLLM*`）暴露，无 C++ 符号外泄。
- **转换工具链**（Python，`export/`）：HF/GGUF → ONNX 骨架 → omg 编译 → 单文件 `.msl`，
  入口 `export/mslite_llm_export.py`。

> 本文档是快速上手入口。架构概览见 [docs/DESIGN.md](docs/DESIGN.md)，API 规范见 [docs/LLM-API.md](docs/LLM-API.md)，
> 交付格式见 [docs/PROTOCOL.md](docs/PROTOCOL.md)，设计决策见 [docs/DESIGN.md](docs/DESIGN.md) §3。

## 快速上手

### 1. 导出模型（Python，需要 CANN DDK + Kirin NPU 目标）

```bash

pip install -r requirements.txt   # transformers 必须为 4.57.x
python export/mslite_llm_export.py \
    --target kirin9020 \
    --model ./Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
    --output ./Qwen2.5-0.5B-Instruct-Q4-0.msl

```text

### 2. 构建 C++ 引擎并运行

```bash

# host x86（默认 release + 单测 + 产出发布归档）
bash build.sh -j8
# debug
bash build.sh -d -j8
# OHOS 交叉编译（Kirin NPU）：交叉编译 .so 后组装发布归档
bash build.sh -b nnrt -j8

# 产物：output/mindspore-lite-llm-linux-x64-{version}.tar.gz + output/tool/mslite-llm-{version}.whl

# 单元测试（无需 NPU 硬件）
ctest --test-dir build --output-on-failure

# 端到端示例（需要 .msl 模型包）
./build/examples/mslite-chat ./Qwen2.5-0.5B-Instruct-Q4-0.msl "你好" 5 2 4

```text

lite_llm 是独立构建模块（不依赖主仓 `MSLITE_ENABLE_LLM`）。

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/DESIGN.md](docs/DESIGN.md) | 设计总览：定位、架构分层、关键决策 |
| [docs/LLM-API.md](docs/LLM-API.md) | 公共 C API 规范（`MSLLM*`） |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | `.msl` v1 单文件格式规范 |
| [export/README.md](export/README.md) | 导出工具链使用说明 |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | 领域术语表 |

## 目录结构

```

lite_llm/
├── include/llm/       # 公共 C API 头（llm.h, llm_types.h）
├── src/               # C++ 引擎（backend / manifest / pipeline / sampler / tokenizer）
├── export/            # Python 转换工具链（mslite_llm_export.py + models/ + utils/）
├── docs/               # 设计文档（DESIGN / LLM-API / PROTOCOL / GLOSSARY）
├── tests/             # ut（C++ gtest）/ py（pytest）/ data（golden）
└── examples/          # mslite-chat 示例

```

## 支持范围

- **平台**：HarmonyOS（Kirin NPU，NNRT 后端）。CPU / Hetero 后端已下线（产品线单轨 NPU）；
  后端按 API 分目录 `src/backend/<api>/`，未来将扩展高通 NPU / GPU / iOS Metal。
- **模型**：Qwen2.5-0.5B（HF / GGUF Q4_0）、MiniMind-3（Qwen3 dense），
  新增模型在 `export/models/<model>/` 下扩展。
- **能力**：同步/流式生成 + Abort、BPE / SentencePiece tokenizer（增量 UTF-8 流式解码）、
  greedy / top-k / top-p / temperature / repetition penalty 采样、导出时固化的 chat template。
  VLM / LoRA / 推测解码 / Chunk Prefill / PD 分离为预留能力，公共路径返回 `NOT_SUPPORTED`。
