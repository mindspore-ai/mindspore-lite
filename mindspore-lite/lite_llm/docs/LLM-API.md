
本文介绍MindSpore Lite LLM的API设计 ，基于使用场景、业界实践提炼出LLM核心API，提供文本生成的基础能力。

LLM接口从高阶到低阶可分为3个层级：
1）**基于Request/Response 请求/响应的服务化接口**：用户把LLM当做一个服务，比如vllm服务，用OpenAI json API形式发送异步的用户文本并等待返回响应。
2）**文本生成类接口**：例如Transformers库，提供基础的Generate() pipeline接口，输入单条文本，输出单轮结果。
3）**细粒度推理接口**：提供细粒度的Prefill/Append/Decode接口，以及Model/Tokenizer/Sampler等要素的细粒度对象，由开发者自行组合推理流程。

层级1服务化组件依赖网络传输，涉及REST等重量级接口，在端侧使用负担大；层级3需要开放大量细粒度API，开放&维护更新成本高。作为首轮开放的LLM API，本文档聚焦于提供2）文本生成类接口，作为腰部API ，兼顾轻量化和易用性。

### 概念空间建模

首先明确本LLM API需要解决的问题，基本使用场景是Chat聊天：和AI多轮对话，并返回结果。例如：

---
**Round 1**
🧑 **用户**：今天天气怎么样？
🤖 **AI**：今天北京晴，25°C，很适合出门。

**Round 2**
🧑 **用户**：那适合户外运动吗？
🤖 **AI**：是的，温度适宜，推荐去公园慢跑。
****

大尺寸LLM模型通常在云侧模型，云侧推理如何支持Chat？以云上vLLM为例，vLLM 支持多轮对话推理，但不提供原生的会话（Session）管理能力，客户端每次请求必须携带**完整的对话历史**，vLLM 只负责根据传入的 `messages` 进行推理，请求结束后就"忘记"了这次对话。虽然 vLLM 不管理会话，但它有 Prefix Caching 机制：当多轮对话中历史消息作为前缀重复出现时，vLLM 会自动复用这些前缀的 KV Cache，避免重复计算。

```Python

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# 多轮对话：客户端负责拼接完整历史
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个 helpful 的助手。"},
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "今天北京晴，25°C，很适合出门。"},
        {"role": "user", "content": "那适合户外运动吗？"},  # 依赖上文历史
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)
# 预期输出: "是的，温度适宜，推荐去公园慢跑。"

```text

在端侧可以提取3种Chat场景：
**场景1**：单用户多轮对话。由Model内部KVCache基于线性规则匹配命中，实现历史KVCache的复用，同时Model不用维护推理轮次的状态机。
**场景2**：单用户多Session，但是Session间串行执行，典型例子是用垂域LLM依次处理不同文本Prompt。Model内部KVCache线性匹配命中，对于新Session命中失败的情况，Model自动清空KVCache用于新一轮对话，这属于Cache的更新策略，而不需要记录对话轮次状态。
**场景3**：多用户多Session并发（目前没有看到），可以由每个Session注册1个KVCache对象，此场景作为扩展项，与上述场景1、2 Cache策略不冲突。

基于以上场景绘制用例图如下：

![image.png](https://raw.gitcode.com/user-images/assets/7404303/9ff6413c-1557-455c-bbc6-bdd808e83ce7/image.png 'image.png')

在这张图中，我们定义3个概念，把场景与我们的API术语建立联系。

#### Model

模型推理的资源能力集合及内部对象，包含Transformers里面的Model、Tokenizer、Sampler，以及Lora、投机Pipeline等，特点是调用接口对外无状态依赖，不需要处理多轮对话和多Session并发。

#### Session

推理会话，包含调用者的对话上下文，它调用Model资源，但不仅仅是Model，比如：推理过程中在数据库记录下历史消息。区分Session和Model的作用是：方便描述对Model的复用，；例如：Agent场景创建多个Session作为SubAgent，但是复用同一Model，只是上下文KVCache有区别。

#### PrefixCache

KVCache缓存池，用于复用历史Prompt的KVCache，减少Prefill计算。

通过Model、Session、PrefixCache 3个对象，可以支持上述3个场景的程序化描述，这是接口设计的起点，用概念描述问题，而不是解决问题。

---

### 接口设计

从上节来看，`Session`定位L1 请求&响应接口，`PrefixCache`作为组件可以被`Model`集成，因为定位于L2接口设计，本节提供`Model`接口的详细设计，关于`Session`和`PrefixCache`仅纳入未来设计考虑，要求是扩展不冲突。

#### **接口定义**

在LLM使用中，为了用户响应的及时性，并不是一轮Prompt完整推理结束才返回完整结果，而是需要逐Token返回，这样获取首Token后即可发给用户，缩短TTFT响应时延。处于这个考虑，我们定义两个术语。

##### 术语

- **流式**：每次只生成1个Token，通过异步回调逐 token 上报；终止原因（EOS/超限/用户中止/出错）由LLM运行时判定并经回调的 `reason` 上报。适用于用户交互场景。
- **非流式**：一次性生成全部文本，直到EOS或者超过生成长度限制而终止，适用于后台非交互的生成场景。

##### 接口使用场景分析

| 组合分析 | 非流式                        | 流式                                                      |
| ---- | -------------------------- | ------------------------------------------------------- |
| 同步   | 同步 + 非流式：常用，易用性好，返回完整文本    | 同步 + 流式：交互场景的默认形态，阻塞调用 + 逐 token 回调，生成结束原因经 `reason` 上报 |

##### 公共类型定义

```C

/** LLM model handle. */
typedef struct MSLLMModel *MSLLMModelHandle;

/** Message roles for chat template rendering. */
typedef enum {
    MSLLM_ROLE_SYSTEM = 0,
    MSLLM_ROLE_USER = 1,
    MSLLM_ROLE_ASSISTANT = 2,
} MSLLMRole;

/** A single message, consumed by MSLLMApplyChatTemplate. */
typedef struct {
    MSLLMRole role; // 消息角色：系统提示词、用户输入、AI响应
    const char *content; // 消息内容
} MSLLMChatMessage;

```text

消息类型的形态：

`MSLLMChatMessage` 采用结构化形态（枚举 `role` + 字符串指针 `content`），而非 JSON 字符串。与 transformers / vLLM 的 messages 协议对齐：两者的进程内 API 侧都是结构化对象（transformers `apply_chat_template` 接收`List[Dict]`，vLLM / OpenAI 的 SDK 是字典或结构体），JSON 只出现在 HTTP 传输层——本接口是进程内 C ABI，无此需求。

这与 「配置走 JSON」是同一个「扩展性放对位置」原则的分工：

|      | `MSLLMSetGenerationConfig` | `MSLLMApplyChatTemplate`                   |
| ---- | -------------------------- | ------------------------------------------ |
| 调用频率 | 会话初始化时一次性，不在热路径            | 每轮对话 / 批量渲染，热路径                            |
| 扩展频率 | 配置项频繁增删                    | 消息协议极稳定（聊天消息结构多年不变）                        |
| 形态收益 | 可选 key 语义天然支持字段增删，不破坏 ABI  | 热路径零解析开销；角色序列不校验（#8/#9），仅 content == NULL → INVALID_ARGS（#10） |

扩展路径（首轮只开放 role/content，未来按业界惯例演进，均 ABI 兼容）：

- **角色扩展**：枚举尾部追加（如将来工具调用新增 `MSLLM_ROLE_TOOL`），不破坏已有值。
- **多模态**：content 未来结构化为 parts 数组（业界路径：OpenAI / vLLM 将图片作为 content 的结构化值，而非新增同级字段）。

##### 创建LLM模型

```C

/**
 * @brief Create LLM model object.
 *
 * @return LLM model object handle.
 */
MSLLMModelHandle MSLLMCreateModel(void);

```text

##### 销毁LLM模型

```C

/**
 * @brief Destroy LLM model object.
 *
 * 生成进行中（GENERATING）拒绝销毁并返回 kMSLLM_ERROR_BUSY（#16）；
 * 正确序列 = Abort → 等生成返回 → Destroy。
 *
 * @param llm_model LLM model object handle.
 */
MSLLMStatus MSLLMDestroyModel(MSLLMModelHandle llm_model);

```text

##### 加载模型到LLM模型

```C

/**
 * @brief Build the LLM model from .msl file path.
 *
 * 仅 CREATED 态有效，exactly-once（#20）：READY 态重复编译返回
 * kMSLLM_ERROR_NOT_SUPPORTED，生成中返回 kMSLLM_ERROR_BUSY。
 * 换模型需 Destroy → Create → Build 重走。
 *
 * @param llm_model LLM model object handle.
 * @param model_path Define the LLM model(.msl) file path.
 * @return MSLLMStatus.
 */
MSLLMStatus MSLLMBuildModel(MSLLMModelHandle llm_model, const char *model_path);

```text

##### LLM文本生成配置

可选。未显式设置的字段使用运行时默认值：`MSLLMCreateModel` 初始化为 `max_new_tokens=256`（0 = 不设输出上限）、`do_sample=false`（greedy）、`temperature=1.0`、`top_k=1`、`top_p=1.0`、`repetition_penalty=1.0`。

```C

typedef struct {
    // output length option：0 = 不设输出上限（生成到 EOS 或撞上下文窗口）；负值非法
    int max_new_tokens;

    // sampling parameters：do_sample=false 时以下字段全部忽略（greedy）
    bool do_sample;
    float temperature;       // 合法区间 [0, 2]，越界 → INVALID_ARGS
    int top_k;               // 合法区间 >= 0（0 = 禁用），越界 → INVALID_ARGS
    float top_p;             // 合法区间 [0, 1]（0 或 1 = 禁用），越界 → INVALID_ARGS
    float repetition_penalty;
} MSLLMGenerationConfig;

```text

设置选项：

```C

/**
 * @brief Set the generation configuration for LLM model.
 *
 * 结构体（非 JSON）。边界校验（#3/#4/#6）：max_new_tokens < 0、temperature
 * 不在 [0,2]、top_k < 0、top_p 不在 [0,1] → kMSLLM_ERROR_INVALID_ARGS 且保留
 * 旧配置不变。生成进行中 → kMSLLM_ERROR_BUSY。
 *
 * @param llm_model LLM model object handle.
 * @param config Generation configuration (struct, not JSON).
 * @return MSLLMStatus. kMSLLM_ERROR_BUSY while a generation is in flight on
 *         this model.
 */
MSLLMStatus MSLLMSetGenerationConfig(MSLLMModelHandle llm_model, const MSLLMGenerationConfig config);

```text

查询配置（#7）：

```C

/**
 * @brief Get the current generation configuration.
 *
 * @param llm_model LLM model handle.
 * @param config [out] Receives the current generation config (a copy).
 * @return kMSLLM_SUCCESS on success, kMSLLM_ERROR_INVALID_ARGS if
 *         llm_model or config is NULL.
 */
MSLLMStatus MSLLMGetGenerationConfig(MSLLMModelHandle llm_model,
                                  MSLLMGenerationConfig *config);

```text

##### 应用 Chat template（渲染完整 prompt）

可选。需要LLM Tokenizer模板（模型包随附）时使用；调用方也可自拼字符串直接调用生成接口。渲染是纯文本拼接，不涉及 token 化。

```C

/**
 * @brief Render a full prompt from role/content messages using the model's
 *        chat template (loaded from the model package).
 *
 * Pure text rendering: no tokenization is involved. Template config is a
 * model resource loaded from the model package, hence the model argument.
 *
 * @param llm_model LLM model object handle.
 * @param messages Array of role/content messages (e.g. full multi-turn history).
 * @param num_messages Number of messages.
 * @param add_generation_prompt 非零则追加生成提示（如尾部 assistant 起始符）；
 *        零则仅渲染既有会话。
 * @param generated_prompt Caller-provided output buffer for the rendered text.
 * @param prompt_size Size of generated_prompt in bytes.
 * @return MSLLMStatus. kMSLLM_ERROR_BUFFER_TOO_SMALL if prompt_size is insufficient
 *         (nothing written; retry with a larger buffer). 消息的 content == NULL
 *         返回 kMSLLM_ERROR_INVALID_ARGS（#10），空串 "" 合法。角色序列不校验：
 *         system 可选（#8）、末条为 assistant 是合法 prefill/continue 用法（#9）。
 *         kMSLLM_ERROR_BUSY while a generation is in flight on this model (D9).
 */
MSLLMStatus MSLLMApplyChatTemplate(MSLLMModelHandle llm_model,
    const MSLLMChatMessage *messages, int num_messages, int add_generation_prompt,
    char *generated_prompt, int prompt_size);

```text

##### 使用LLM模型生成文本

###### 文本生成接口

```C

/**
 * @brief Finish reason of LLM generation.
 */
enum MSLLMFinishReason {
    /** the LLM is running, not finished. */
    kMSLLM_RUNNING = 0,
    /** the LLM is finished by EOS token. */
    kMSLLM_FINISHED_BY_EOS = 1,
    /** the LLM is finished by Exceeding max_context_length limit. */
    kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH = 2,
    /** the LLM is finished by Exceeding max_output_length limit. */
    kMSLLM_FINISHED_BY_MAX_OUTPUT_LENGTH = 3,
    /** the LLM is stopped by user. */
    kMSLLM_STOPPED_BY_USER = 4,
    /** the LLM is finished due to an inference error (INFERENCE / OOM);
     *  the exact error code is returned by the generate function. */
    kMSLLM_FINISHED_BY_ERROR = 5
};

/**
 * @brief Generate text with specific user prompt, blocking until finished.
 *
 * max_new_tokens 语义（#3）：0 = 不设输出上限，生成到 EOS 或撞上下文窗口；
 * >0 = 显式输出上限（先到先止）。非流式不可中止（D8）。
 *
 * @param generated_text Caller-provided output buffer.
 * @param text_size Size of generated_text in bytes.
 * @return MSLLMStatus. kMSLLM_ERROR_BUFFER_TOO_SMALL if text_size is insufficient
 *         (nothing written). See「buffer 契约」(D4).
 */
MSLLMStatus MSLLMGenerate(MSLLMModelHandle llm_model,
    const char *prompt,
    char *generated_text, int text_size);

```text

> **说明**
> 涉及到采样：采样模块入NPU图时，无法动态改，运行时会报错提示。导出时manifest.json需要说明TopK不能修改。

###### 流式生成接口

解码循环运行在**发起调用的线程**上，函数阻塞，每生成一个 token 调用一次回调，全部生成完毕后返回。

```C

/**
 * @brief Streaming callback, invoked once per generated token.
 *
 * Runs on the caller's thread, inside the MSLLMStreamGenerate call.
 * Contract:
 *   - MUST NOT throw: exceptions crossing the extern "C" boundary terminate.
 *   - @param token is only valid for the duration of the call; copy to retain.
 *   - May call MSLLMAbort from within (it is non-blocking); calling config
 *     interfaces (MSLLMSetGenerationConfig / MSLLMApplyChatTemplate) from within
 *     returns kMSLLM_ERROR_BUSY (the model is GENERATING on this thread).
 *
 * @param token Incremental text, NULL on the final invocation. 中文/emoji 等多字节
 *        UTF-8 字符由库按完整字符交付（#17：tokenizer 增量解码，跨 token 的字节
 *        在库内缓冲，凑齐才回调；单 token 未凑齐时回调空串）。
 * @param reason kMSLLM_RUNNING while generating, terminal value on the last
 *        call (EOS / length limits / user abort / error).
 * @param callback_data User pointer passed through from MSLLMStreamGenerate.
 */
typedef void (*MSLLMStreamCallback)(const char *token,
    MSLLMFinishReason reason, void *callback_data);

/**
 * @brief Generate text streamly: one callback invocation per token.
 *
 * BLOCKS until generation completes. The decode loop runs on the calling
 * thread; the callback is invoked on that same thread. The function returns
 * once the last token is produced — return means completion, no wait primitive
 * is needed.
 *
 * Call this from a Task/Worker on UI platforms, so the UI thread is not
 * blocked during prefill (first-token latency can reach hundreds of ms).
 *
 * @return Status of the call itself. Generation end reasons (EOS / length
 *         limits / user abort) are reported via the callback's reason.
 *         On a mid-generation failure the final callback is still invoked
 *         (token=NULL, reason=kMSLLM_FINISHED_BY_ERROR) before the error
 *         code (INFERENCE / OOM) is returned. Returns kMSLLM_ERROR_BUSY if
 *         another thread has an in-flight generation on this model.
 */
MSLLMStatus MSLLMStreamGenerate(MSLLMModelHandle llm_model,
    const char *prompt,
    MSLLMStreamCallback callback, void *callback_data);

```text

`MSLLMStreamGenerate`参考实现：

```C

class LLMModel {
    // 用户结束标识
    atomic_bool is_aborted = false;
}

void MSLLMStreamGenerate(llm_model) {
    bool finished = false;
    while(!finished) {
        // LLMModel内部生成单个Token，包含Prefill/Decode；
        token = Forward();

        // 检查是否用户中止，如果用户中止，完成本次推理后，结束自回归循环
        if (llm_model->is_aborted_) {
            reason = STOPPED_BY_USER;
            finished = true;
            llm_model->is_aborted_ = false; // 复位标识
        }

        OnToken(token, reason); // 执行用户回调返回结果
    }
    return;
}

```text

###### **中止流式生成接口**

```C

/**
 * @brief Request early termination of an in-progress streaming generation.
 *
 * Only affects MSLLMStreamGenerate: 仅在 GENERATING 态才置中止标志（#13），
 * 其它状态调用是 no-op 但仍返回 kMSLLM_SUCCESS（#15：幂等）。
 * the decode loop observes the flag,
 * the blocked call returns kMSLLM_SUCCESS, and the final callback's
 * reason is kMSLLM_STOPPED_BY_USER.
 *
 * Has no effect during MSLLMGenerate (non-streaming): non-streaming
 * generations cannot be interrupted (D8).
 *
 * Safe to call from any thread, including from within the stream callback.
 * Non-blocking: only sets a flag.
 */
MSLLMStatus MSLLMAbort(MSLLMModelHandle llm_model);

```text

#### **错误码设计**

```C

enum MSLLMStatus {
    kMSLLM_SUCCESS = 0,
    /** invalid handle, null pointer, or out-of-range parameter. */
    kMSLLM_ERROR_INVALID_ARGS = 1,
    /** model package missing, corrupt, or incompatible. */
    kMSLLM_ERROR_MODEL_LOAD = 2,
    /** failure during prefill or decode execution. */
    kMSLLM_ERROR_INFERENCE = 3,
    /** memory or KV cache allocation failure. */
    kMSLLM_ERROR_OOM = 4,
    /** feature reserved in the ABI but not implemented by this runtime. */
    kMSLLM_ERROR_NOT_SUPPORTED = 5,
    /** file or IO error. */
    kMSLLM_ERROR_IO = 6,
    /** model already has an in-flight generation. */
    kMSLLM_ERROR_BUSY = 7,
    /** prompt alone exceeds max_context_len; nothing was generated. */
    kMSLLM_ERROR_CONTEXT_OVERFLOW = 8,
    /** unexpected internal error. */
    kMSLLM_ERROR_INTERNAL = 9,
    /** caller-provided buffer is too small; nothing was written. */
    kMSLLM_ERROR_BUFFER_TOO_SMALL = 10,
};

```text

`kMSLLM_ERROR_CONTEXT_OVERFLOW` 是「接口边界」决策的直接产物：由于 tokenize 不开放，调用方无法在请求前自行判断 prompt 长度，因此库**必须**把"prompt 本身已超出上下文窗口"作为一个同步返回的独立错误码明确报出，不能静默截断 prompt。

它与 `kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH` 的区别：前者表示 prompt 未能进入推理、一个 token 都没生成；后者表示生成过程中触及上下文上限而正常终止，已有部分输出。

错误上报：

| 错误类别          | 上报方式                                                 | 示例                                                               |
| ------------- | ---------------------------------------------------- | ---------------------------------------------------------------- |
| 调用本身失败（生成未开始） | `MSLLMStreamGenerate` / `MSLLMGenerate` 返回错误码；无回调    | `INVALID_ARGS`、`BUSY`、`NOT_SUPPORTED`、`CONTEXT_OVERFLOW`         |
| 生成中途失败        | 流式：终结回调（token=NULL，reason=`ERROR`）后返回错误码；非流式：直接返回错误码 | `INFERENCE`、`OOM`                                                |
| 生成终止原因（非错误）   | 回调的 `reason` 参数（仅流式）                                 | `EOS`、`MAX_CONTEXT_LENGTH`、`MAX_OUTPUT_LENGTH`、`STOPPED_BY_USER` |

返回值与 `reason` 互不替代：返回值表示"调用是否成功执行"，`reason` 表示"生成因何结束"。即使正常结束（EOS），`reason` 也是 `kMSLLM_FINISHED_BY_EOS` 而非错误码。中途出错时两者同时出现：终结回调的 `reason` 说明"生成因出错终止"（`kMSLLM_FINISHED_BY_ERROR`），返回值给出具体错误码（`INFERENCE` / `OOM`）。

#### **接口状态图**

会话状态机。阻塞式形态下，同一线程上的状态迁移是确定性的；`GENERATING` 态仅对**其他线程**可观测（此时它们调用生成接口会得到 `BUSY`）。

```text

                MSLLMCreateModel
                        │
                        ▼
                   ┌─────────┐
                   │ CREATED │
                   └─────────┘
                        │ MSLLMBuildModel
                        ▼
                   ┌─────────┐  MSLLMSetGenerationConfig
                   │  READY  │◀─────────────┐
                   └─────────┘              │
                        │                   │ 生成函数返回
        MSLLMGenerate ──┤                   │ (阻塞式：返回即完成)
        MSLLMStreamGenerate                 │
                        ▼                   │
                 ┌────────────┐             │
                 │ GENERATING │─────────────┘
                 └────────────┘
                    │      ▲
                    │      └─ 仅其他线程可见；此时再次发起生成、或调用配置接口
                    │         （MSLLMSetGenerationConfig / MSLLMApplyChatTemplate）
                    │         → kMSLLM_ERROR_BUSY
                    │ 三种出口：
                    │   ① 正常终止（EOS / 长度上限）
                    │   ② MSLLMAbort —— 仅流式响应（非流式期间调用被忽略）
                    │   ③ 中途出错（INFERENCE / OOM）
                    ▼
                函数返回，回到 READY：
                - 流式：终结回调（token=NULL）的 reason = EOS /
                  MAX_CONTEXT_LENGTH / MAX_OUTPUT_LENGTH / STOPPED_BY_USER /
                  ERROR；① ② 返回 SUCCESS，③ 返回对应错误码
                - 非流式：① 返回 SUCCESS（完整文本）；③ 返回错误码；② 不适用

```

销毁会话在生成返回之后进行即可 —— 阻塞式形态下发起调用的线程天然满足这一点。`MSLLMDestroyModel` 在 GENERATING 态拒绝销毁并返回 `BUSY`（#16），正确序列 = Abort → 等生成返回 → Destroy。

#### **示例：文本生成**

```C

int main() {
    // 创建一个LLM会话
    MSLLMModelHandle llm_model = MSLLMCreateModel();
    if (llm_model == nullptr) {
        LOGE("create model error");
        return -1;
    }

    // 加载编译会话所需的模型
    const char *model_path = "/path/to/model";
    MSLLMStatus ret = MSLLMBuildModel(llm_model, model_path);
    if (ret != kMSLLM_SUCCESS) {
        MSLLMDestroyModel(llm_model);
        return -1;
    }

    // 一次生成所有Token
    int prompt_size = 1024;
    char[1024] prompt;
    char result[1024];

    // 调用模板拼接Prompt
    MSLLMChatMessage messages[] = {
        {
            .role = MSLLM_ROLE_SYSTEM,
            .content = "You are a very helpful assistant.",
        },
        {
            .role = MSLLM_ROLE_USER,
            .content = "Write a quick sort code in python.",
        },
    }
    MSLLMApplyChatTemplate(llm_model,
        messages, 2, 1, prompt, prompt_size);

    // 调用一次接口，生成所有文本。同步阻塞，无需回调，也不涉及线程管理。
    // 调用方预分配缓冲（见「buffer 契约」，D4）：缓冲不足返回
    // kMSLLM_ERROR_BUFFER_TOO_SMALL 且不写入，调用方按场景上限一次给足。
    MSLLMStatus status = MSLLMGenerate(llm_model, prompt, result, sizeof(result));
    if (status == kMSLLM_ERROR_BUFFER_TOO_SMALL) {
        MS_LOG(ERROR) << "buffer too small";
        return -1;
    }
    if (status != kMSLLM_SUCCESS) {
        MS_LOG(ERROR) << "MSLLMGenerate error: " << status;
        return -1;
    }
    LOGI("model response: %s", result);

    // 复用相同的会话执行新的推理
    MSLLMChatMessage messages2[] = {
        {
            .role = MSLLM_ROLE_SYSTEM,
            .content = "You are a very helpful assistant.",
        },
        {
            .role = MSLLM_ROLE_USER,
            .content = "Write atravel plan at ShangHai for me.",
        },
    }
    MSLLMApplyChatTemplate(llm_model,
        messages2, 2, 1, prompt, prompt_size);

    MSLLMGenerate(llm_model, prompt, result, sizeof(result));
    LOGI("model response: %s", result);

    MSLLMDestroyModel(llm_model);
    return 0;
}

```text

#### **示例：流式推理**

阻塞式流式：LLM库仅执行解码循环，调用方负责创建&维护后台解码线程。

```C

// 调用方的实现
// 回调在发起调用的线程上执行（UI 场景应置于 Task/Worker 中，见下文）。
// 契约：不得抛异常；token 指针仅回调期间有效；可调用非阻塞的 MSLLMAbort。
void OnToken(const char *token, MSLLMFinishReason reason, void *user_data) {
    auto *ctx = static_cast<MyUiContext *>(user_data);

    if (token != nullptr) {
        // 建议异步队列，尽快执行完，不要阻塞，做好对消费线程的通知
        // 必须拷贝：token 指向库内部临时缓冲，回调返回后失效。
        lock();
        ctx->AppendText(std::string(token));
        unlock();
        ctx->notify(); // 通知业务线程消费
    }

    if (reason != kMSLLM_RUNNING) {
        MS_LOG(INFO) << "generation finished, reason: " << reason;
        ctx->NotifyFinished(reason);
    }
}

int main() {
    MSLLMModelHandle llm_model = MSLLMCreateModel();
    if (llm_model == nullptr) {
        LOGE("create model error");
        return -1;
    }

    MSLLMStatus ret = MSLLMBuildModel(llm_model, "/path/to/model");
    if (ret != kMSLLM_SUCCESS) {
        MSLLMDestroyModel(llm_model);
        return -1;
    }

    // 完整 prompt 由调用方组装，同上（多轮历史由调用方拼入）。
    const char *prompt =
        "system: You are a very helpful assistant. \n user: write a quick sort code in python";

    MyUiContext ctx;

    // 调用者负责创建后台线程，执行流式推理
    infer_task = new thread([&]() {
        // 阻塞式流式：函数阻塞直到生成结束，回调在调用线程上逐 token 触发。
        // 生成结束原因（EOS/超限/用户中止）经回调的 reason 上报；函数返回值
        // 只表示调用本身是否成功。
        MSLLMStatus status = MSLLMStreamGenerate(llm_model, prompt, OnToken, &ctx);
        if (status == kMSLLM_ERROR_BUSY) {
            MS_LOG(ERROR) << "model already has an in-flight generation";
            MSLLMDestroyModel(llm_model);
            return -1;
        }
        if (status != kMSLLM_SUCCESS) {
            MS_LOG(ERROR) << "MSLLMStreamGenerate error: " << status;
            MSLLMDestroyModel(llm_model);
            return -1;
        }
    }

    // 等待后台推理任务结束
    infer_task.join();

    // 函数返回即生成完成，无需任何等待原语；直接销毁即可。
    MSLLMDestroyModel(llm_model);
    return 0;
}

```text

UI 场景下，阻塞式调用必须放在 Task/Worker 中发起，避免冻结 UI 线程：

```C

// ArkTS 侧示意：在 Worker 中发起阻塞调用，回调在 Worker 线程上触发。
worker.postMessage({ prompt: prompt_text });
// worker.onexit 或回调内再通知主线程刷新 UI。

```text

流式场景提前终止：从任意线程（含回调内部）调用 `MSLLMAbort`，解码循环在下一步检查到中止标志，函数随即返回（`kMSLLM_SUCCESS`），最终回调的 `reason` 为 `kMSLLM_STOPPED_BY_USER`。`MSLLMAbort` 仅对 `MSLLMStreamGenerate` 生效；`MSLLMGenerate`（非流式）期间调用被忽略——非流式没有回调通道上报中止原因，因此不可中止；需要"停止按钮"打断长生成时请使用流式接口。

```C

MSLLMAbort(llm_model);  // 非阻塞接口，通知流式推理结束，执行完立即返回；被阻塞的 MSLLMStreamGenerate 随后自行返回

```text

由于并发发起第二个生成会返回 `kMSLLM_ERROR_BUSY`，"打断当前生成并立即重新提问"的正确序列是：

```C

MSLLMAbort(llm_model);  // 请求中止，非阻塞
// 等待进行中的 MSLLMStreamGenerate 返回（它在 Abort 后自行收敛），
// 回到 READY 后再发起新请求，此时不会返回 BUSY。

```cpp

---

### 设计分析

#### **接口边界：不对外开放的能力**

**决策：第 3层级（Model/Tokenizer/Sampler 三要素）不对外开放，Tokenize/Detokenize 不进入公开 API。**

理由与影响：

- **责任边界完整。** 文本 ↔ token 的转换完全由库内部持有，调用方只见文本。这是方案 B 承担 token 边界拼接责任的前提——若调用方能拿到 token 序列，多字节字符截断的责任归属立刻变得模糊。
- **不暴露内部契约。** tokenizer 词表、特殊 token、chat template 的编码细节属于模型包内部实现，开放后即成为需长期兼容的 ABI。
- **代价：调用方无法预估 token 数。** 无法在发起请求前判断 prompt 是否超出 `max_context_len`，也无法自行做 prompt 截断。因此**上下文超限必须由库给出明确错误码与终止原因**，不能静默截断（见错误码设计与 `kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH`）。
- **代价：调用方无法自建解码循环**，因此方案 C 形态（调用方自建循环）的流式接口在本 API 下不可实现（见流式取舍章节）。
- **代价：token 计量依赖库上报。** 若调用方需要按 token 计费或统计，必须由库经独立查询接口 `MSLLMGetUsage` 上报（见「token 计量上报」，D5 已定稿），不能自行计算。

Chat template 的应用位置是此边界的直接推论：由于 tokenizer 不开放，模板套用也应在库内完成，经 `MSLLMApplyChatTemplate` 独立暴露。「会话与多轮对话」决策（调用方维护对话历史、把完整上下文作为单次 prompt 传入）与之衔接：调用方把完整消息数组传给 `MSLLMApplyChatTemplate` 渲染，再调用生成接口。当前示例中直接使用预拼好的 `system:` / `user:` 前缀字符串只是为了展示生成接口本身，不代表推荐用法。

#### **设计取舍：流式推理的线程归属**

流式推理涉及两个正交的问题：

1. **谁拥有解码循环**（逐 token 自回归推进的逻辑）？
2. **谁拥有线程**（循环跑在哪个线程上）？

由此分出三种形态：

- **方案 A（库拥有循环与线程）**：`MSLLMStreamGenerate` 立即返回，库自建工作线程跑解码循环，每吐一个 token 回调一次（fire-and-forget 异步）。
- **方案 B（库拥有循环，调用方拥有线程）**：`MSLLMStreamGenerate` **阻塞**，解码循环直接跑在发起调用的线程上，每吐一个 token 回调一次，全部生成完毕后才返回。返回即完成，无需任何等待原语。
- **方案 C（调用方拥有循环与线程）**：库只提供单步推进接口（形如 `MSLLMGenerateNext`），调用方自建循环与线程。

**决策：采用方案 B —— 阻塞式回调。**

- 方案 C 不可实现：第 3 层级不对外开放（见「接口边界」），调用方没有自行组装解码循环的原料（tokenize 不在公开 API 中）。
- 方案 A 的"库自建线程"并非方案 A 的优点，而是其风险的来源：回调跨线程、参数生命周期悬垂、完成屏障缺失、绑定层线程附着、测试非确定 —— 这些风险全部集中在"库拥有线程"这一属性上。**把线程还给调用方后，这些风险系统性消失**（见下两表）。

##### 为什么不是方案 A（worker-thread 异步）

| 风险维度           | 方案 A（库拥有线程）                                         | 方案 B（调用方拥有线程）                                     |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 回调线程           | 库的工作线程 → 回调内拿到的 token 在库线程上，要更新 UI 必须跨线程投递回 UI 线程（见下方「跨线程投递」说明） | 回调在发起调用的线程上，token 直接落在调用方线程，无需任何投递 |
| 完成信号           | 需 `MSLLMWait` 完成屏障，否则调用方自建条件变量              | 函数返回即完成，无需任何原语                                 |
| 请求参数生命周期   | 调用方返回后指针悬垂 → 必须深拷贝                            | 函数阻塞期间参数必然存活 → 无此约束                          |
| 回调上下文生命周期 | 需绑定会话兜底回收，否则泄漏                                 | 生命周期 = 函数调用，天然无泄漏                              |
| 绑定层             | 需 `napi_threadsafe_function` / `AttachCurrentThread` / GIL（均为跨线程投递与线程附着的机制） | 回调跑在调用方线程，绑定层原样直通，无需上述机制             |
| 测试               | 回调时序非确定，需额外同步                                   | 纯同步、确定性                                               |
| 线程占用           | 每请求一个库线程（端侧内存敏感）                             | 零库线程，复用调用方线程                                     |

**关于"跨线程投递（marshal）"**：回调在哪个线程执行，决定了数据要不要"搬运"。UI 框架通常只允许在 UI 线程更新界面，因此若回调跑在库的工作线程上，调用方必须把每个 token 从库线程投递回 UI 线程，再由 UI 线程执行界面更新 —— 这个"打包数据 + 投递 + 目标线程执行"的过程就是跨线程投递。ArkTS 里对应 `EventHub` / `emitter` / Worker 的 `postMessage`，Android 里是 `runOnUiThread` / `Handler.post`，iOS 是 `DispatchQueue.main.async`；在方案 A 下它对每个 token 发生一次，且绑定层还需处理线程附着（如 `napi_threadsafe_function`）。方案 B 下回调在调用方自己的线程上触发，数据不跨线程，因此没有这个过程，只剩结束时一次性的线程间通知。

方案 A 唯一不可替代之处是"零线程代码的 fire-and-forget 异步"。但端侧 UI 场景无论如何都要把推理移出 UI 线程（prefill 首 token 可达秒级），线程迁移责任本就在调用方；C/C++ 服务端调用方自带线程池，包一层阻塞调用是常规操作。因此方案 A 的收益在本文档的部署场景（HarmonyOS 端侧）中不成立。

##### 采用方案 B 的理由

| #    | 优点                   | 说明                                                         |
| ---- | ---------------------- | ------------------------------------------------------------ |
| 1    | 调用侧代码量小         | 调用方无需编写解码循环，也无需 mutex/condition_variable/queue 等线程间同步原语 —— 回调在调用线程上直接递送 token，无跨线程数据流动。批处理场景零线程；UI 场景仅需一个一次性 Task/Worker 包住阻塞调用（见「方案 B 的代价」） |
| 2    | 回调零跨线程投递       | 回调在发起调用的线程上执行，token 直接落在这个线程，无需跨线程投递回 UI 线程（见「跨线程投递」说明）；UI 场景在 Task/Worker 中发起调用即可 |
| 3    | 参数生命周期安全       | 函数阻塞期间调用方传入的指针必然存活，深拷贝约束从契约中消失 |
| 4    | Token 边界责任留在库内 | 逐 token 解码对 CJK/emoji 会产生不完整 UTF-8 字节序列。拼接缓冲若交给调用方，等于要求每个调用方都自己处理一遍多字节截断 |
| 5    | 生成状态生命周期闭环   | 解码循环退出即复位生成状态；调用方无法中途脱离循环，状态必然收敛 |
| 6    | 取消语义单点           | 中止只需翻转一个标志位，由库在解码循环内检查；`MSLLMAbort` 由其他线程调用 |
| 7    | 库零线程               | 无线程池、无每请求建线程的开销、无 worker 生命周期管理，端侧内存敏感场景友好 |

##### 方案 B 的代价

仅一条，需要正视：**库不再提供 fire-and-forget 异步**。调用方想要"发起后立即返回"，必须自己在 Task/Worker 中发起阻塞调用。这是把"谁开线程"的决定权交给调用方的必然结果，也是方案 B 换走方案 A 全部风险所付的代价。

##### 规范要求

以下约束**必须**在接口契约与实现中同时满足，否则会成为线上问题。编号（R1…R7）用于实现对齐与评审检查。

| #   | 风险                            | 规范要求                                                                                                                                                                                                                                                                                  |
| --- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | 回调运行在发起调用的线程上，阻塞期间该线程被占用      | 契约显式声明；UI 场景调用方须在 Task/Worker 中发起，避免冻结 UI 线程。回调**不得抛异常**（异常穿过 `extern "C"` 边界将导致 `std::terminate`）                                                                                                                                                                                    |
| R2  | 回调内再次调用生成类接口（重入）              | 生成接口先检查会话状态、再上锁；生成进行中调用返回 `kMSLLM_ERROR_BUSY`。同一线程上真实重入不可能（循环尚未结束），此约束主要保护实现不因此死锁                                                                                                                                                                                                     |
| R3  | 回调携带的 token 文本指针指向库内部临时缓冲     | 契约声明"指针仅回调期间有效，需要留存必须自行拷贝"                                                                                                                                                                                                                                                            |
| R4  | 推理过程错误的上报                     | **全部错误同步返回**（函数返回错误码）。流式中途出错（`INFERENCE` / `OOM`）先发终结回调（token=NULL，reason=`kMSLLM_FINISHED_BY_ERROR`）再返回错误码——"最后一次调用 token=NULL"的契约在所有路径成立。生成终止原因经回调 `reason` 上报（`EOS` / `MAX_CONTEXT_LENGTH` / `MAX_OUTPUT_LENGTH` / `STOPPED_BY_USER` / `ERROR`）。两者互不替代：返回值表示调用是否成功，reason 表示生成因何结束 |
| R5  | 跨线程并发发起第二次生成（另一线程在生成期间调用生成接口） | **显式返回 `kMSLLM_ERROR_BUSY`**，禁止排队与内部阻塞等待                                                                                                                                                                                                                                              |
| R6  | 绑定层实现差异                       | 回调跑在调用方线程，绑定层无需线程附着，直接透传回调即可                                                                                                                                                                                                                                                          |
| R7  | 测试确定性                         | 天然同步；无需额外等待原语构造断言点                                                                                                                                                                                                                                                                    |

##### 线程安全规约

- 单个会话同一时刻仅一个生成在执行；并发发起返回 `BUSY`（R5）。
- 不同会话之间线程安全。
- 生成进行中（`GENERATING` 态），配置类接口同样返回 `kMSLLM_ERROR_BUSY`：`MSLLMSetGenerationConfig`（生成期间变更配置会竞态）与 `MSLLMApplyChatTemplate`（统一规则，避免调用方逐接口记忆并发语义，D9）；`MSLLMGetUsage` 同理（见 D5）。
- `MSLLMAbort` 可从任意线程调用，包括回调所在线程与另一线程；非阻塞，仅置标志。**仅对 `MSLLMStreamGenerate` 生效**；`MSLLMGenerate`（非流式）期间调用被忽略（D8）。
- 销毁会话必须在生成返回之后进行 —— 阻塞式形态下调用方天然满足（发起调用的线程阻塞在函数内），无需额外规约。

#### **设计取舍：会话与多轮对话**

**决策：Generate 系列接口本身不直接支持多轮对话，其定位对齐 Transformers 的 `generate()` 接口 —— 每次调用独立完成一次生成。**

具体语义：

- **每次 `MSLLMGenerate` / `MSLLMStreamGenerate` 调用都是独立的一次生成**：内部以本次调用的 prompt 为完整输入，不保留、不延续任何跨调用状态。多次调用互不影响。
- **多轮对话由调用方组织**：调用方自行维护对话历史（拼上历史轮次的 assistant 回答），把完整上下文作为单次调用的 prompt 传入。库不感知也不存储对话历史。
- **KV cache 语义**：每次生成独立分配/复位 KV cache，调用返回后即释放，不存在"跨调用续用 KV cache"的场景。历史 KV 的**缓存**（而非续用）是另一回事，见「多轮对话的高效性」—— 缓存是纯优化，可逐出，不改变"每次调用独立"的语义。
- **会话（Session）是资源容器，不是对话上下文**：`MSLLMModelHandle` 封装的是模型加载、后端、配置等运行时资源；它不代表一段对话，也不持有对话历史。

**由此直接得出的结论：**

| 曾待定项                     | 结论                                                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 是否需要 `MSLLMResetModel    | **不需要**。没有"清空历史"的操作，因为从不保留历史。会话是资源容器，其状态只由 `MSLLMSModelonConfig` 等配置接口改变                                                                             |
| `max_context_len` 耗尽时的行为 | 单次生成内触及上限即正常终止：`kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH`（生成中）/ `kMSLLM_ERROR_CONTEXT_OVERFLOW`（prompt 本身超限）。无滑窗淘汰，每次调用都是新窗口（层 2 显式保活开启时除外，见「多轮对话的高效性」） |
| 第二次调用是否延续上一轮             | **否**。独立生成，调用方需将历史拼入 prompt。历史前缀的 KV 由缓存加速（见「多轮对话的高效性」）；显式保活（层 2）属预留配置，默认关闭                                                                          |

调用方无需在两次调用间做任何"复位"动作 —— 每次调用的起始状态本就一致。多轮对话的正确用法（模板渲染 + 生成两步）见下节「Chat template 的应用位置」。

#### **设计取舍：多轮对话的高效性**

「会话与多轮对话」决策（Generate 无状态）成立的前提，是多轮对话不能因为"每次重拼完整历史"而逐轮变慢。效率不来自跨调用状态，而来自**可复用的中间结果**：

| | 状态（session 内多轮） | 缓存（无状态 + 可复用中间结果） |
|---|---|---|
| 接口语义 | 调用带历史、库记住上下文 | 每次独立，但历史 prompt 的 KV 被缓存 |
| 失效性 | 硬绑定上下文 | 纯优化，可任意清空、逐出 |
| 正确性 | 用错状态就答错 | 命中即用，不命中重算，永远正确 |
| 实现复杂度 | KV 复用 + 位置编码续算 + 跨调用所有权 | 前缀匹配 + 命中时续算，其余照旧 |

第 N 轮的完整 prompt（历史 + 本轮输入）与第 N-1 轮共享前 ~N-1 轮的全部 token，因此**前缀缓存天然就是多轮对话的加速器** —— Transformers/vLLM 生态均以此实现无状态 `generate()` 的多轮高效。

**层 1：前缀缓存（Prefix Caching）—— 库内部透明，接口不变**

- 库内部缓存历史前缀的 KV cache，命中时从命中点续算，只 prefill 新增部分（历史越长的轮次，加速越明显）。
- 接口完全不变，调用方无需任何动作；多轮对话自动获得加速。
- 缓存是纯优化：命中则快，不命中（如前缀改变、缓存被逐出）则完整重算，语义始终正确。
- 缓存生命周期与逐出由库管理（容量配额 / LRU），调用方无感知也无控制。
- **层 1 与层 2 在热多轮场景的常驻内存相当**：命中历史前缀的前提就是历史 KV 常驻（在缓存池或会话中，占用相近）。层 1 的优势不在内存占用更低，而在可控性——配额/LRU 逐出（内存压力下自动腾挪，代价是重算）、跨会话前缀去重共享（如多个会话共用同一系统提示时只存一份）、切换对话后自然老化。内存压力下层 1 会主动让位，层 2 的常驻 KV 则不会自动释放。

**层 2：session 级 KV 复用（显式保活）—— 预留，走路径 a**

若要求"多轮之间 KV 必须常驻、从第 2 轮起近乎零重算"（性能目标明确时），以配置项方式提供，**不新增接口**：

- 复用 `MSLLMSetGenerationConfig`，新增配置项（如 `preserve_kv_between_calls`）：置真后，生成返回不释放 KV cache，下一轮生成复用；位置编码继续累计。
- 这是对「KV cache 语义」的显式开关：默认关闭（每次独立、返回即释放，保持无状态语义），开启后由调用方负责上下文生命周期的边界（何时不再需要历史，应显式关闭或调用方自行重建会话）。
- 实现复杂度与风险高于层 1（跨调用 KV 所有权、位置编码续算、与 `max_context_len` 交互），故标记为**预留**：除非出现明确性能目标，默认只实现层 1。

#### **设计取舍：Chat template 的应用位置**

**决策：模板渲染在库内完成，经独立的 `MSLLMApplyChatTemplate` 接口暴露；`MSLLMGenerate` / `MSLLMStreamGenerate` 保持接收裸 prompt 字符串。**

**模板渲染不依赖 tokenizer，但模板配置由 tokenizer 从模型包加载。** 这是对 HF 生态惯例的对齐，而非本仓库独有：

- **渲染是纯文本拼接**。`Apply()` 只做 `{{role}}` / `{{content}}` 的字符串替换与拼接，不涉及任何 token id、词表或特殊 token。
- **模板配置随 tokenizer 走**。HF 将 chat template 存在 `tokenizer_config.json`（挂在 tokenizer 名下，可被 `apply_chat_template(messages, chat_template=...)` 覆盖）；vLLM、TGI、llama.cpp（GGUF 元数据）均遵循此约定。本仓库从模型包的 tokenizer 资源读取模板类型与自定义模板字符串，与之一致。

**由此得出：**

- `MSLLMGenerate` 接收裸字符串 —— 对齐 HF 的 `generate()` 定位（已定决策），模板能力与生成能力分离。
- **新增 `MSLLMApplyChatTemplate(model, messages, num_messages, add_generation_prompt, buffer, buffer_size)`**：把消息数组渲染为完整 prompt 文本。它不属于「接口边界」禁止的 Tokenize —— 输入输出都是文本，不产生 token id、不暴露词表；开放的是模板渲染能力。角色序列不校验（#8/#9）；`content==NULL` → INVALID_ARGS（#10）。
- 该接口需要 **session** 参数：模板配置从模型包加载，属会话资源（与"会话是资源容器"一致），不是纯函数。
- 调用方三选一：自拼字符串直接 Generate（不用模板）；`ApplyChatTemplate` 渲染后 Generate（用库内模板）；自持自定义模板自拼（此时模板细节是调用方责任，需自行与模型配套）。
- 与「会话与多轮对话」决策衔接：多轮历史由调用方维护，以完整 messages 数组传给 `ApplyChatTemplate`，渲染出完整 prompt 后再 Generate。

**对话式场景的正确用法**（多轮对话是调用方责任）：

```C

// 第 N 轮：调用方维护完整历史，交给库内模板渲染
MSLLMChatMessage msgs[] = {
    {.role = MSLLM_ROLE_SYSTEM, .content = system_prompt},
    {.role = MSLLM_ROLE_USER, .content = history_user_1},
    {.role = MSLLM_ROLE_ASSISTANT, .content = history_assistant_1},
    {.role = MSLLM_ROLE_USER, .content = "本轮用户输入"},
};
char prompt[4096];
MSLLMApplyChatTemplate(llm_model, msgs, 4, 1, prompt, sizeof(prompt));
MSLLMGenerate(llm_model, prompt, result, sizeof(result));

// 把本次回答追加进 history，供下一轮使用
history.emplace_back(MSLLM_ROLE_ASSISTANT, result);

```

#### **设计取舍：配置接口形态**

**决策：`MSLLMSetGenerationConfig` 采用结构体，而非 JSON 字符串。**

理由：

- **配置项变动不频繁**，仅涉及新增，不影响兼容性。
- **边界显式校验（#3/#4/#6）**：`max_new_tokens < 0`、`temperature ∉ [0,2]`、`top_k < 0`、`top_p ∉ [0,1]` → `INVALID_ARGS` 且保留旧配置；校验在 Set 时做，生成接口不再重复查。`do_sample=false` 时采样参数静默忽略（#5）。

#### **设计取舍：buffer 契约**

**决策（#1）：调用方一次性预分配缓冲；缓冲过小时返回 `kMSLLM_ERROR_BUFFER_TOO_SMALL` 且不写入任何内容，不回填所需尺寸。库不分配内存，不引入配对的释放接口。`MSLLMGenerate` 与 `MSLLMApplyChatTemplate` 共用同一契约。**

端侧输出长度可预期（最大上下文 32k），调用方自行分配足量缓冲即可，无需 `required_size` 回填。

为什么不是库分配 + `MSLLMFreeOutput`：

- **多一个接口，多一份生命周期规则**。库分配意味着必须有配对的释放接口，调用方要记住"这个指针是库分配的，必须调库的 free"——这是一条额外的、容易被忘记或用错分配器的规则。调用方预分配是调用方已经在管理的内存，生命周期跟局部变量或调用方自己的堆内存完全一致，不新增规则。
- **文本长度从应用角度可预期**。调用方通常知道自己场景下的典型输出长度（聊天回复、代码生成等都有合理上界），预分配一个够用的缓冲是常规做法。

语义：

- 调用方传入 `buffer` 和 `buffer_size`。
- 缓冲足够：写入完整文本（含 `\0` 结尾），返回 `kMSLLM_SUCCESS`。
- 缓冲不足：不写入（不做部分截断，避免调用方误用不完整文本），返回 `kMSLLM_ERROR_BUFFER_TOO_SMALL`。
- 重试是完整重新调用（`MSLLMGenerate` 重新生成 / `MSLLMApplyChatTemplate` 重新渲染），不是续写——两个接口都是无状态的单次调用。
