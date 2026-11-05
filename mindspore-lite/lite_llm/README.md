# MindSpore Lite LLM

MindSpore Lite LLM作为端侧大语言模型推理引擎，支持HarmonyOS（Kirin NPU）设备上的LLM轻量化、高性能推理。

- **推理引擎**（C++17，`include/llm/` + `src/`）：模型加载、tokenize、prefill/decode 生成循环、
  采样，通过纯 C API（`MSLLM*`）暴露，无 C++ 符号外泄。
- **转换工具链**（Python，`export/`）：HF/GGUF → ONNX 骨架 → omg 编译 → 单文件 `.msl`，
  入口 `export/mslite_llm_export.py`。

> 注意：LLM模块接纳AI生成代码，但需要提交者充分理解代码内容！
>
> 本文档是快速上手入口。架构概览见 [docs/DESIGN.md](docs/DESIGN.md)，API 规范见 [docs/LLM-API.md](docs/LLM-API.md)，
> 交付格式见 [docs/PROTOCOL.md](docs/PROTOCOL.md)，设计决策见 [docs/DESIGN.md](docs/DESIGN.md) §3。

## 快速上手

### 1. 导出模型（Python，需要 CANN DDK + Kirin NPU 目标）

```bash
cd mindspore-lite/lite_llm/
pip install -r requirements.txt
python export/mslite_llm_export.py \
    --target kirin9020 \
    --model ./Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
    --output ./Qwen2.5-0.5B-Instruct-Q4-0.msl
```

### 2. 构建 C++ 引擎并运行

#### 2.1 本地编译（用于调测引擎代码）

```bash
# host（默认 release + 产出发布归档）
bash build.sh -j8

# host (debug版本)
bash build.sh -d -j8

# 单元测试（不依赖 Kirin NPU 硬件）
bash tests/run_llm_ut.sh --output-dir ./ut_results --jobs 8
```

#### 2.2 鸿蒙交叉编译（用于鸿蒙设备运行）

下载鸿蒙交叉编译工具链：前往[HarmonyOS官网](https://developer.huawei.com/consumer/cn/download/command-line-tools-for-hmos)，下载Command Line Tools 6.1.1 Release，解压出的 `command-line-tools/sdk/default/openharmony/native/` 子目录即为OpenHarmony NDK。

```bash
# 配置OHOS_NDK环境变量，指向下载的ndk native目录
export OHOS_NDK=/path/to/your/native/

bash build.sh -b nnrt -j8
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/DESIGN.md](docs/DESIGN.md) | 设计总览：定位、架构分层、关键决策 |
| [docs/LLM-API.md](docs/LLM-API.md) | 公共 C API 规范（`MSLLM*`） |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | `.msl` v1 单文件格式规范 |
| [export/README.md](export/README.md) | 导出工具链使用说明 |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | 领域术语表 |

## 目录结构

```text
lite_llm/
├── include/llm/       # 公共 C API 头（llm.h, llm_types.h）
├── src/               # C++ 引擎（backend / manifest / pipeline / sampler / tokenizer）
├── export/            # Python 转换工具链（mslite_llm_export.py + models/ + utils/）
├── docs/               # 设计文档（DESIGN / LLM-API / PROTOCOL / GLOSSARY）
├── tests/             # ut（C++ gtest）/ py（pytest）/ data（golden）
└── examples/          # mslite-chat 示例
```

## 支持范围

- **平台**：HarmonyOS（Kirin NPU，NNRT 后端）；
  后端按 API 分目录 `src/backend/`，未来将扩展CPU / 高通 NPU / GPU / iOS Metal。
- **模型**：Qwen2.5-0.5B（HF / GGUF Q4_0）、MiniMind-3（Qwen3 dense），
  新增模型在 `export/models/<model>/` 下扩展。
- **能力**：同步/流式生成 + Abort、BPE / SentencePiece tokenizer（增量 UTF-8 流式解码）、
  greedy / top-k / top-p / temperature / repetition penalty 采样、导出时固化的 chat template。
 公共路径返回 `NOT_SUPPORTED`。
