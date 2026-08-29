# Lite LLM 领域词汇表

> 本词汇表是 grill-with-docs / domain-modeling 会话的产物，术语定义以
> [LLM-API.md](LLM-API.md)（已评审）为准，[DESIGN.md](DESIGN.md) 中的
> 旧术语（如 `MSLiteLlmInit`、`MSLG` 主推）视为过期，不参与定义。
> 判定规则：**doc 与 runtime 冲突时，以本表 + LLM-API.md + 实际代码为准。**

## 概念层（domain concepts）

| 术语 | 定义 | 来源 |
|------|------|------|
| **Model** | 模型推理的资源能力集合及内部对象（Transformer 模型 + Tokenizer + Sampler + LoRA/投机 等）。接口对外无状态依赖，不处理多轮对话与多 Session 并发。 | LLM-API.md |
| **Session** | 推理会话，持有调用者的对话上下文，调用 Model 资源（如 Agent 场景多个 SubAgent 复用同一 Model，KV 上下文不同）。**Session 是资源容器，不是对话历史。** | LLM-API.md |
| **PrefixCache** | KV cache 缓存池，复用历史 prompt 前缀的 KV，减少重复 prefill。纯优化：命中即快、可逐出、语义永远正确。 | LLM-API.md |
| **LLM API 三层级** | L1 服务化（Request/Response，OpenAI json API 形态）；L2 文本生成（Generate pipeline，单文本进单文本出）；L3 细粒度（Prefill/Append/Decode + Model/Tokenizer/Sampler 对象）。**首轮只开放 L2。** | LLM-API.md |
| **接口边界（不开放）** | Tokenize/Detokenize、第 3 层级要素对象、解码循环原料不进入公开 API。理由：责任边界（多字节截断归属）、不暴露内部契约（词表/template）。代价：调用方无法预估 token 数 → 库必须显式报上下文超限。 | LLM-API.md |
| **流式 / 非流式** | 流式=每 token 一次回调上报，终止原因经 reason 上报；非流式=一次性返回完整文本。 | LLM-API.md |
| **方案 A / B / C** | A=库拥有循环与线程（fire-and-forget，回调跨线程）；B=库拥有循环、调用方拥有线程（阻塞式回调，返回即完成）；C=调用方拥有循环与线程（需开放第 3 层级）。**决策：方案 B。** | LLM-API.md |
| **buffer 契约** | 调用方预分配缓冲；不足返回 `BUFFER_TOO_SMALL` + `required_size` 回填，不写部分内容。库不分配、无配对的 free。 | LLM-API.md |

## 包与运行时（package & runtime）

| 术语 | 定义 | 来源 |
|------|------|------|
| **.msl 包** | 自包含模型包目录：manifest.json（唯一真相来源）+ 图 + vocab + sampler + checksums + 可选组件。 | PROTOCOL.md |
| **manifest 三层配置** | Tier1 manifest（导出时固定不可变）→ Tier2 init config（会话级）→ Tier3 request config（请求级）。覆盖规则：manifest < config < request。 | PROTOCOL.md / LLM-API.md |
| **split-graph（~~CPU~~ 已下线，产品线单轨 NPU）** | prefill 图（无 past）+ decode 图（带 past KV），运行时以 `decode_states_` 在请求内推进 KV。`present_kv` 的 `state_id` 必须与 decode `past_kv` 匹配。**这是 CPU 主链路格式，不是 MSLG。** | cpu_litert_integration.md / cpu_backend.cpp |
| **graph_io schema** | 每个 graph tensor 的 name/role/dtype/shape/past_len_axis/token_axis/vocab_axis/fill/state_id 声明；运行时按名称匹配，不按数组顺序。 | cpu_litert_integration.md |
| **KV state 对齐** | 跨图（prefill→decode→decode…）KV 靠 `state_id` 标识身份，不靠 tensor 名称。 | PROTOCOL.md |
| **decode 能力（capabilities）** | `decode_max_past_len` + `dynamic_past_len` 或静态 `decode_past_len` / `decode_p<N>` 变体；past_len 选路径，不是第二缓存身份。 | cpu_backend.cpp |
| **generation policy** | manifest.generation 声明 stop/suppress token id；存在即权威（空数组=无策略），缺失才回退 vocab.bin legacy。 | PROTOCOL.md |
| **MSLG / MSLT** | 旧二进制格式。MSLG（graph.bin，"MSLG" magic）为**旧图格式，不推荐**；MSLT（vocab.bin，"MSLT" magic）为词表/编码资产格式，继续使用。 | PROTOCOL.md |
| **MSLiteLlm\*** | DESIGN.md 中引用的旧 C API 命名（`MSLiteLlmInit/Chat`），与当前 runtime（`MSLlm*`）及 LLM-API.md（`MSLLM*`）均不一致，视为过期符号。 | DESIGN.md（过期） |

## 推理语义（generation semantics）

| 术语 | 定义 | 来源 |
|------|------|------|
| **无状态 Generate** | 每次 `Generate` 独立完成一次生成，不保留跨调用状态；多轮对话由调用方拼完整历史。KV 每次独立分配/复位。 | LLM-API.md |
| **CONTEXT_OVERFLOW** | prompt 本身超出上下文窗口 → 同步返回错误码，**禁止静默截断**。与 `FINISHED_BY_MAX_CONTEXT_LENGTH`（生成中触及上限、有部分输出）区分。 | LLM-API.md |
| **finish reason** | 流式终结回调的 `reason`：EOS / MAX_CONTEXT_LENGTH / MAX_OUTPUT_LENGTH / STOPPED_BY_USER / ERROR。与返回值互不替代。 | LLM-API.md |
| **BUSY** | 已有在途生成时，再次发起生成或配置类接口 → 显式返回 BUSY，禁止排队与阻塞等待。 | LLM-API.md |
| **chat template** | 库内渲染的纯文本拼接（不 token 化）；模板配置随 tokenizer 从模型包加载。`MSLLMApplyChatTemplate` 独立暴露。 | LLM-API.md |

## 导出工具链（toolchain）

| 术语 | 定义 | 来源 |
|------|------|------|
| **NNRT** | 华为端侧神经网络运行时（Neural Network Runtime），HarmonyOS Kirin NPU 的推理 API。lite_llm 当前唯一后端，目录 `src/backend/nnrt/`，开关 `MSLITE_LLM_ENABLE_NNRT`，公共枚举 `MSLLM_BACKEND_NNRT`。 | DESIGN.md §3.4 |
| **QNN** | Qualcomm 神经网络运行时（Qualcomm Neural Network），骁龙平台 NPU 推理 API。规划中的第二个后端（`src/backend/qnn/`，`MSLLM_BACKEND_QNN`）。 | DESIGN.md §3.4 |
| **HF → .msl 主链路** | download → (quantize) → export(split prefill/decode ONNX) → LiteRT convert → package。**当前 split 导出与自动 graph_io/architecture 生成缺失。** | 本会话分析 |
| **architecture.json** | 描述转换后图真实结构的架构参数（层数/隐藏/头/KV 头/head_dim/vocab/rope…），packager 写入 manifest。 | cpu_litert_integration.md |
| **graph_io.json** | prefill/decode 两图的完整 IO 声明（含每层 KV state）。Qwen2.5-0.5B 需 24 层 × 2 (K/V) × 2 图 的 state 声明。 | cpu_litert_integration.md |
| **generation_policy.json** | 从 `generation_config.json` 提取 stop/suppress token id；源模型未声明但协议要求屏蔽的 token 显式传入（如 `<|im_start|>`）。 | cpu_litert_integration.md |

## 验收与质量（acceptance）

| 术语 | 定义 | 来源 |
|------|------|------|
| **token 级 parity** | C++ 运行时编解码结果与 HF tokenizer 逐 token 对齐（中文/英文/代码/数字/emoji 语料）。**当前 BPECodec 预切分与 Qwen tiktoken 正则不一致，是 Qwen2.5 正确性头号风险。** | 本会话分析 |
| **E2E golden** | 真实 Qwen2.5-0.5B .msl 包 + 固定输出 golden，经公共 API 重复生成验证（ST 以模型为维度看护：`MSLITE_LLM_ST_DEVICE=1 pytest tests/st --gguf=... --msl=...`，见 tests/st/conftest.py）。 | tests/st/ |
| **层 1 / 层 2 精度归因** | 对齐「量化模型精度」时把误差分成两层：**层 1 = NPU 算子实现误差**（NPU vs 同一量化图的 CPU 参考，实测 cosine 0.99995+）；**层 2 = 权重量化固有损失**（量化图 vs fp16 golden，实测 cosine 0.958）。层 1 可修算子，层 2 只能换量化方案（W8A16/AWQ/GPTQ）。 | DESIGN.md §3.3 / 实测 |
| **量化基准（fp16）** | 精度对齐的锚点取 Q4_0 的量化源 **fp16 GGUF 权重**（llama.cpp 链 bf16→fp16→Q4_0），而非 bf16 原始——避免把 bf16→fp16 舍入计入量化误差。bf16→fp16 对权重无损，可用 HF 模型以 `torch_dtype=fp16` 等价复现。 | DESIGN.md §3.3 |
