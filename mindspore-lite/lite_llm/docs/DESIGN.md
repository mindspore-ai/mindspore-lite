# lite_llm 设计总览

> 本文档是 lite_llm 的**设计入口**：描述模块定位、架构分层与关键设计决策。
> 细节请查阅权威文档，避免重复维护导致漂移：
>
> | 主题 | 权威文档 |
> |------|---------|
> | 公共 C API（MSLLM\*） | [LLM-API.md](LLM-API.md) |
> | .msl v1 交付格式 | [PROTOCOL.md](PROTOCOL.md) |
> | 关键架构决策 | 见本文档 §3 |
> | 导出工具使用 | [../export/README.md](../export/README.md) |

## 1. 定位

MindSpore Lite LLM（`mslite-llm`）是 MindSpore Lite 框架中的**端侧大语言模型推理模块**（HarmonyOS / Kirin NPU），由两部分组成：

| 部分 | 语言 | 目录 | 职责 |
|------|------|------|------|
| **推理引擎** | C++17 | `mindspore-lite/lite_llm/` | 模型加载、tokenize、prefill/decode 生成循环、采样，通过纯 C API 对外暴露 |
| **转换工具链** | Python | `mindspore-lite/lite_llm/export/` | HF/GGUF → ONNX 骨架 → omg 编译 → 单文件 `.msl`，入口 `mslite_llm_export.py` |

两者通过 `.msl` 单文件模型包衔接：Python 工具链生成，C++ 引擎加载运行。

### 适用平台

| 平台 | 后端 | 模型格式 |
|------|------|---------|
| HarmonyOS (Kirin NPU) | `NNRTBackend`（NNRT API） | `.msl`（内含 `.omc` 图） |
| CPU / Hetero | 已下线（产品线单轨 NPU） | — |

### 能力矩阵

| 能力 | 状态 |
|------|------|
| 文本生成（同步/流式 + Abort 中断） | 已接通 |
| Tokenize / Detokenize（BPE + SentencePiece，增量 UTF-8 流式解码） | 已接通 |
| 采样（greedy / top-k / top-p / temperature / repetition penalty） | 已接通 |
| Chat Template（导出时固化） | 已接通 |
| KV Cache | NNRT 设备侧管理 |
| W4A16 量化推理 | 已接通（`src/backend/nnrt/nnrt_embedding_dequant.*`） |
| VLM / LoRA / 推测解码 / Chunk Prefill / PD 分离 | 预留，公共路径返回 `NOT_SUPPORTED` |

## 2. 架构分层

```mermaid

flowchart TB
    L1[Layer 1: Public C API] --> L2[Layer 2: InternalEngine]
    L2 --> L3[Layer 3: Pipeline]
    L3 --> L4A[Tokenizer]
    L3 --> L4B[Sampler]
    L3 --> L4C[ModelInstance]
    L4C --> L5[Layer 4: Backend]
    L5 --> L5A[NNRTBackend / NNRTExecutor]

```text

| Layer | 文件 | 核心职责 |
|-------|------|---------|
| L1 Public C API | `include/llm/llm.h` + `llm_types.h` | 纯 C ABI，opaque handle，extern "C" |
| L2 InternalEngine | `src/pipeline/llm.cpp` | 状态机（Created/Ready/Generating）、并发控制、生成循环编排 |
| L3 Pipeline | `src/pipeline/model_instance.cpp` | 单模型实例：manifest 解析、资源加载、backend 工厂注入 |
| L4 Subsystems | `src/tokenizer/` `src/sampler/` `src/manifest/` | 编解码、采样策略、.msl 包读取 |
| L5 Backend | `src/backend/` `src/backend/nnrt/` | 抽象 `Backend` 接口 + NNRT 实现（图编译/执行/KV 管理、W4A16 反量化） |

## 3. 关键设计决策

各决策的推理过程（为何如此选、否决了什么）记录如下；行为契约的权威定义在 LLM-API.md / PROTOCOL.md。

### 3.1 纯 C API（`MSLLM*`）

对外只有 `MSLLM*` C ABI（opaque handle + `MSLLMStatus`），无 C++ 符号暴露；ABI 面为
CreateModel / BuildModel / Generate / StreamGenerate / ApplyChatTemplate / Abort /
Set-GetGenerationConfig / GetUsage / Destroy。理由：

- **端侧绑定语言多**（C/C++/Java/ArkTS），纯 C ABI 是唯一稳定的跨语言边界；C++ 符号在编译期就泄露实现细节，破坏 ABI 稳定性。
- **tokenize/detokenize 不开放**：责任边界（多字节截断归属调用方）与内部契约（词表/模板）不暴露；代价是调用方无法预估 token 数，库必须显式报上下文超限。
- **buffer 契约**：调用方预分配缓冲，不足返回 `BUFFER_TOO_SMALL` + `required_size` 回填，库不分配、无配对 free。
- 错误码 11 值、`MSLlmFinishReason` 6 值；旧 `MSLlmInit` 形态降为内部或 deprecated 薄封装。

### 3.2 单文件 .msl v1 交付格式

`.MSL` header + KV 元数据 + 资源表 + 数据区，无外部 manifest，元数据内嵌。理由：

- **消灭格式漂移**：旧 KCAP v0 的打包端存 basename、运行时按完整相对路径查找，带子目录模型必挂；v1 统一完整相对路径，两侧单一事实源。
- **单文件自包含**：部署物只有一个 `.msl`，无外部 `manifest.json`。
- **消除外部工具依赖**：导出链路不再依赖预编译 C++ `mspacker_tool`（需 subprocess + staging 目录），改由 Python `packager.py` 直接写包。
- **前向兼容**：KV 加键不升版本，新增元数据字段无需改布局。

### 3.3 NPU 单轨（CPU 产品线下线）

产品聚焦 Kirin NPU 端侧推理；CPU 后端与旧导出工具链已删除，不留「本机 e2e 调试」后门。理由：

- 旧 `tools/mslite_llm/` 导出链（pip 包）是 `export/` 的过期拷贝且引用不存在的模块；运行时 `src/backend/cpu/` 长期失活、日常 OHOS 构建已是 NPU 单后端。
- 双轨维护成本大于收益：两套工具链两套后端，无真实用户。
- 验证策略改为：导出侧用 transformers 做精度对齐标杆；runtime 侧真机（HarmonyOS + Kirin NPU）验证（`examples/mslite-chat` 加载 .msl 跑公共 API 推理，ST 以模型为维度看护，见 tests/st/）。

### 3.4 多后端目录范式

后端按**运行时 API** 分目录：`src/backend/nnrt/`（现存），未来 `qnn/` `metal/` `gpu/`；共享核心（`backend.h`、`backend_factory.*`）留在 `src/backend/` 根部。编译期每后端一个开关（当前 `MSLITE_LLM_ENABLE_NNRT`）。

### 3.5 采样在主机侧

生成循环中采样在 CPU 完成，NPU 只做前向：`NnrtExecutor::Execute` 把 logits 拷回 host，交给 `Pipeline` 的 `Sampler`（greedy/top_k/top_p/temperature/repetition_penalty），与 CPU 路径共用同一采样链。每次 decode 多拷 ~600KB fp32 logits 的代价在 kirin9020 真机上接受，待真机 profiling 后再评估设备端采样（需改 NNRT 契约 + NPU kernel）。

### 3.6 Chat template 导出时固化

Jinja 求值留在 Python 导出侧（用与 HF 相同的真 Jinja2 引擎），运行时不做 Jinja 求值（否决 minja 路线）：`.msl` 是离线预导出的不可变产物，模板在导出时已知固定（llama.cpp 必须运行时求值是因其是通用 GGUF 加载器）；导出时把模板编译为「受限 IR」（常量段 + 占位符 + 控制标志）序列化进包，运行时只解释 IR，不新增 C++ 三方依赖。

### 3.7 生命周期与并发

EngineState 原子状态机（Created/Ready/Generating）；生成中 Destroy/Build 返回 `BUSY`（禁止排队与阻塞等待）；Abort 仅置标志由生成循环响应。

### 3.8 Backend 工厂注入

`src/backend/backend_factory.*` 允许测试注入 fake backend，引擎级 UT 不依赖 NPU 硬件。

## 4. 目录导览

```

lite_llm/
├── include/llm/       # 公共 C API 头（llm.h, llm_types.h）
├── src/               # C++ 引擎（backend/manifest/pipeline/sampler/tokenizer）
├── export/            # Python 转换工具链（mslite_llm_export.py + utils/ + models/）
├── docs/              # 设计文档
│   ├── DESIGN.md      # 本文档（含关键设计决策 §3）
│   ├── LLM-API.md     # 公共 API 规范
│   ├── PROTOCOL.md    # .msl v1 格式规范
│   └── GLOSSARY.md    # 领域术语表
├── tests/             # ut（C++ gtest）/ py（pytest）/ st（模型级真机验证）/ data（golden）
├── examples/          # mslite-chat 示例（公共 API 真机验证入口）

```text

## 5. 开发指南入口

- 构建与测试：`lite_llm/build.sh`（见 §6）及 `CMakeLists.txt`（`MSLITE_LLM_ENABLE_NNRT` / `MSLITE_LLM_BUILD_TESTS`）。
- 新增模型：在 `export/models/<model>/` 建目录（文件以 `<model>_` 前缀命名），复用 `utils/` 通用件。
- 变更原则：行为契约变更需同步 `LLM-API.md` / `PROTOCOL.md` / 能力矩阵；设计决策变更更新 §3。

## 6. 构建与打包

lite_llm 独立构建（不集成主仓 `MSLITE_ENABLE_LLM`）：

```bash

bash mindspore-lite/lite_llm/build.sh -j8          # host x86, release + 单测 + 发布归档
bash mindspore-lite/lite_llm/build.sh -d -j8       # debug
bash mindspore-lite/lite_llm/build.sh -b nnrt -j8       # Kirin NPU 后端（OHOS 交叉编译，.so 面向 Kirin NPU）

```text

打包产物（`lite_llm/output/`，host 与 OHOS 均产出）：

- `tool/mslite-llm-{version}.whl`（导出工具链，含 torch_custom 算子接口）
- `mindspore-lite-llm-linux-x64-{version}.tar.gz`（发布归档，含运行时）

```

└── mindspore-lite-llm-linux-x64-{version}.tar.gz
    ├── tool/mslite-llm-{version}.whl     # mslite-llm-export 入口
    ├── include/llm/*.h                   # 公共 C API 头
    ├── lib/libmindspore-lite-llm.so      # 引擎动态库
    └── ascendc_ops/*.run                 # CI 产出的算子包（存在时收集）

```

版本号单一来源：`lite_llm/version.txt`（CMake 的 .so VERSION、wheel、归档共用）。
