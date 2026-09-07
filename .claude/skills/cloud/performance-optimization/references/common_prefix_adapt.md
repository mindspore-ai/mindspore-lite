# 公共前缀（Common Prefix）适配

> 本文档是 `performance-optimization` skill 的细化策略文档之一。
> 适用场景：LLM 推理中存在固定的公共前缀（如 system prompt、文档片段），多次请求只有用户输入（suffix）在变化。**首次请求算一次前缀 KV cache，后续请求直接复用前缀 KV，只跑 suffix 模型**，把"每次都要重算 system prompt"的浪费省掉。
> 通用测量与验证流程见 SKILL.md；profiling/命令模板见 [other_opt_methods.md](other_opt_methods.md)。

## 适用条件

- 输入能切成两段：**固定 prefix + 动态 suffix**（典型：system prompt + 用户问题；文档 RAG + 问题）。
- 只需要一个 token 输出（如选择题判定、单 token 分类）或允许「前缀独立先跑一次」（首次有 prefix 耗时）。
- 后端支持 GQA / 多 KV 头的 LLM（Qwen3 / Qwen2 / LLaMA3 等 decoder-only Transformer）。

不适用的场景：

- prefix 经常变、或者根本没有固定公共前缀 —— 退化回普通 prefill/decode。
- 必须自回归生成多 token 且 suffix 与 prefix 强耦合 —— 仍走 prefill+decode 主链路。

## 核心思路：两个模型分工

```
┌─────────────────────────────┐
│ Prefix 模型                 │  ← 跑一次：prefix tokens → KV cache
│ (input: prefix tokens)      │     输出 [2*L, B, H_kv, P, D]
│ (output: present_kv only)   │     存在 device 侧复用
└──────────────┬──────────────┘
               │ KV cache (复用)
               ▼
┌─────────────────────────────┐
│ Suffix 模型                 │  ← 每次请求都跑
│ (input: suffix tokens +     │     输入：suffix tokens + past_kv + attention_mask
│  past_key_values)           │     输出：最后一个 token 的 logits [B, 1, vocab]
│ (output: last-token logits) │
└─────────────────────────────┘
```

**关键收益**：后续请求直接复用 prefix KV cache，**只跑 suffix 模型**——suffix 模型 seq_len 远小于 prefix+suffix 之和，省掉 prefix 重算开销。

## 三步落地

### 第 1 步：导出 Prefix / Suffix 两个 ONNX 模型

Prefix 模型只跑 `embed → N 层 transformer → 输出 KV cache`（不接 `lm_head`），Suffix 模型接 `lm_head` 并只输出最后一个 token 的 logits（slice_last）。

**核心要点**（以 Qwen3 为例，对应 [`export_qwen3_onnx.py`](../../../../../mindspore-lite/examples/base_models/qwen3_0.6b/export_qwen3_onnx.py)）：

- `--enable-common-prefix` 触发 `export_prefix_suffix()`：分别导出 `qwen3_prefix.onnx` 与 `qwen3_suffix.onnx`。
- Prefix 模型 I/O：
  - 输入：`input_ids (B, P)`、`attention_mask (B, P)`、`position_ids (B, P)`，int32
  - 输出：`present_key_values (2*L, B, H_kv, P, D)`，float16
  - L = num_layers，H_kv = num_kv_heads（GQA），D = head_dim
- Suffix 模型 I/O：
  - 输入：`input_ids (B, S)`、`attention_mask (B, P+S)`、`position_ids (B, S)`、`past_key_values (2*L, B, H_kv, P, D)`
  - 输出：`logits (B, 1, vocab)`（**只取最后一个真实 token**，靠 `attention_mask.sum → index_select` 实现 slice_last，最小化 D2H）
  - P = prefix 档位长度（固定），S = suffix 档位长度
- Prefix 与 suffix 都复用 `_cann_attn_forward` / `_cann_mlp_forward`，与普通 prefill 共享同一套融合开关（`enable_rotarymul / enable_pfa / enable_bmm2mm`）。
- Suffix 模型 attention 内部用 `key = cat([past_kv_k, k_new], dim=2)` 拼成完整 KV，再走 attention。

**Prefix 与 Suffix 的最小 forward 骨架（PyTorch）：**

```python
class Qwen3PrefixModel(nn.Module):
    """跑一遍 transformer 层，只把每层的 K/V 收集起来输出。"""
    def forward(self, input_ids, attention_mask, position_ids):
        h = self.embed_tokens(input_ids)
        pos = self.rotary_emb(h, position_ids)
        bool_mask = make_bool_causal_mask(attention_mask, q_len=q_len, k_len=q_len, past_len=0)
        residual = h
        h = self.layers[0].input_layernorm(h)
        present = []
        for layer in self.layers:
            attn_out, pk, pv = cann_attn_forward(layer.self_attn, h, pos, bool_mask,
                                                  None, None, enable_pfa=True)
            present.append(pk); present.append(pv)
            h, residual = add_rms_norm(residual, attn_out, layer.post_attention_layernorm)
            mlp_out = cann_mlp_forward(layer.mlp, h, enable_swiglu=False)
            h, residual = add_rms_norm(residual, mlp_out, next_norm_or_final)
        return torch.stack(present, dim=0)  # [2*L, B, H_kv, P, D]


class Qwen3SuffixModel(nn.Module):
    """接 prefix KV + suffix tokens，只输出最后一 token 的 logits。"""
    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        h = self.embed_tokens(input_ids)
        pos = self.rotary_emb(h, position_ids)
        past_len = past_key_values[0].shape[2]   # = P
        q_len = input_ids.shape[1]                # = S
        k_len = past_len + q_len
        bool_mask = make_bool_causal_mask(attention_mask, q_len, k_len, past_len)
        residual = h
        h = self.layers[0].input_layernorm(h)
        for i, layer in enumerate(self.layers):
            pk_in = past_key_values[2 * i]
            pv_in = past_key_values[2 * i + 1]
            # 关键：suffix 必须用 InnerPFA（支持 q_len != k_len）
            attn_out, _, _ = cann_attn_forward(layer.self_attn, h, pos, bool_mask,
                                                pk_in, pv_in, enable_pfa=True,
                                                use_inner_pfa=True)
            h, residual = add_rms_norm(residual, attn_out, layer.post_attention_layernorm)
            mlp_out = cann_mlp_forward(layer.mlp, h, enable_swiglu=False)
            h, residual = add_rms_norm(residual, mlp_out, next_norm_or_final)
        logits = self.lm_head(h)
        # slice_last：用 attention_mask.sum 定位最后一个真实 token，index_select 提 logits
        seq_lens = attention_mask.sum(dim=1)                       # [B]
        last_idx = (seq_lens - 1).to(torch.int32)
        sliced = logits.index_select(1, last_idx.long())[:, :1, :]  # [B, 1, vocab]
        return sliced
```

**Attention 分支里 use_inner_pfa 的关键差异**（见 `_cann_attn_forward`）：

```python
if use_inner_pfa:
    attn_out = _CannInnerPromptFlashAttention.apply(
        q, k, v, bool_mask, num_heads, num_kv_heads, scale)
else:
    attn_out = _CannPromptFlashAttention.apply(
        q, k, v, bool_mask, num_heads, num_kv_heads, scale)
```

`InnerPromptFlashAttention` 与 `PromptFlashAttention` 在 ONNX 侧都走 `Custom` 节点，区别只在 `type_s` 与输入签名（InnerPFA 多保留 `pse_shift` 槽位）：

```python
# InnerPFA：5 个输入槽位，pse_shift 不传，atten_mask 可选
g.op("Custom", q, k, v, atten_mask,
     type_s="InnerPromptFlashAttention",
     input_names_s=["query", "key", "value", "pse_shift", "atten_mask"],
     optional_input_names_s=["atten_mask", "pse_shift"],
     output_names_s=["attention_out"],
     output_num_i=1, input_index_i=[0, 1, 2, 4],
     num_heads_i=..., num_key_value_heads_i=..., scale_value_f=...,
     input_layout_s="BNSD", inner_precise_i=0)
```

### 第 2 步：ONNX → MindIR 转换（按档位配置 dynamicDims）

Prefix 与 Suffix 各自一个 ini，**`ge.dynamicDims` 必须覆盖推理侧会用到档位**：

- Prefix：固定一个或多个 prefix 长度档位（典型：`P=480, 768`）。
- Suffix：每个档 8 个值，对应 4 个动态维 `(B, S)`、`(B, P+S)`、`(B, S)`、`(2*L, B, H_kv, P, D)`，P 在所有档中固定。
  - 例（Qwen3-0.6B，P=768）：`S ∈ {32, 64, 96, 128, 256, 384, 512, 640}` → `P+S ∈ {800, 832, 864, 896, 1024, 1152, 1280, 1408}`。

转换命令（与场景 A/B 同模板，只换 model/config 文件）：

```bash
$Convert --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR \
  --modelFile=.../qwen3_prefix.onnx \
  --outputFile=.../qwen3_prefix \
  --configFile=.../qwen3_llm_prefill_prefix.ini

$Convert --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR \
  --modelFile=.../qwen3_suffix.onnx \
  --outputFile=.../qwen3_suffix \
  --configFile=.../qwen3_llm_prefill_suffix.ini
```

两个 ini 均使用 `allow_mix_precision` + `op_fp32.json` 黑名单（与场景 A/B 相同），强制 RmsNorm 等敏感算子保 FP32。

### 第 3 步：推理 —— prefix 跑一次、suffix 复用 KV

按 inference `ge.dynamicDims` 设定 prefix / suffix 档位，输入做 padding 到最近 bucket（参考 [`infer_qwen3_0.6b_mindir.py`](../../../../../mindspore-lite/examples/base_models/qwen3_0.6b/infer_qwen3_0.6b_mindir.py) 中 `Qwen3CommonPrefixInferencer`）：

```python
class Qwen3CommonPrefixInferencer:
    def __init__(self, prefix_model_path, suffix_model_path, tokenizer_id,
                 prefix_seq_len=768, suffix_buckets=[32, 64, 96, 128, 256, 384, 512, 640],
                 device="ascend", device_id=0):
        self.prefix_model = mslite.Model(); self.prefix_model.build_from_file(prefix_model_path, ...)
        self.suffix_model = mslite.Model(); self.suffix_model.build_from_file(suffix_model_path, ...)
        self.prefix_seq_len = prefix_seq_len
        self._prefix_kv_tensor = mslite.Tensor(
            shape=[2*L, 1, H_kv, prefix_seq_len, D], dtype=FLOAT16, device="ascend:0")

    def compute_prefix_cache(self, prefix_text):
        # 1) tokenize + pad 到 prefix_seq_len
        # 2) 跑 prefix 模型 → present_kv（首屏耗时）
        # 3) 把输出 buffer 缓存为 _prefix_kv_tensor，后续 suffix 直接复用
        ...

    def infer_suffix(self, suffix_text):
        # 1) tokenize suffix + pad 到最近 suffix bucket
        # 2) 拼 attention_mask = [prefix_mask; suffix_mask]（prefix 真实长度 + suffix bucket 长度）
        # 3) 构造 inputs：suffix_tokens + attention_mask + positions + past_key_values
        # 4) predict → argmax → decode token
        ...
```

**关键点**：

- `attention_mask` 拼接：prefix 真实长度段全 1、padding 段全 0；suffix bucket 长度段全 1（prefix padding 部分被 mask 掉，suffix 不会 attend 到）。
- `position_ids` 从 `prefix_seq_len` 开始递增（不是从 0）；prefix padding 位置设为 0。
- prefix KV 用 device Tensor 预分配，suffix 直接把它作为 `past_key_values` 输入，**整套链路在 Ascend 侧完成，避免 D2H/H2D 反复拷贝**（与 [zero_copy_inference.md](zero_copy_inference.md) §"Qwen3-VL 三阶段流水线" 同思路）。
- 首次请求 = `prefix_time + suffix_time`；后续请求（prefix KV 复用）= **只跑 suffix**，吞吐与单步 latency 都接近单独跑一个小 prefill。

## 与 InnerPromptFlashAttention（InnerPFA）搭配

公共前缀场景下 **suffix 模型 `q_len != k_len`**（`q = suffix_len`, `k = prefix_len + suffix_len`），而 CANN 300I Duo 上的标准 `PromptFlashAttention` 算子**强制 `q_len == k_len`**（否则转换或推理期报 `attention mask must be NULL, when Qs is not equal to Kvs`）。要让 suffix 模型走 PFA 融合，**必须改用 `InnerPromptFlashAttention`**。

### InnerPFA 的 ONNX 侧改造

`_CannInnerPromptFlashAttention` 与 `_CannPromptFlashAttention` 共用同一份 PyTorch reference 数值语义，但 `symbolic` 输出的 `Custom` 节点签名不同：

| 字段 | PromptFlashAttention | InnerPromptFlashAttention |
|------|---------------------|---------------------------|
| `type_s` | `PromptFlashAttention` | `InnerPromptFlashAttention` |
| `input_names_s` | `[query, key, value, atten_mask]` | `[query, key, value, pse_shift, atten_mask]`（多 1 个 `pse_shift` 槽位） |
| `optional_input_names_s` | `[atten_mask]` | `[atten_mask, pse_shift]` |
| `input_index_i` | `[0, 1, 2, 3]` | `[0, 1, 2, 4]`（pse_shift 槽位不传，指向 atten_mask 的真实索引 4） |
| `q_len vs k_len` | 要求 `==` | 支持 `!=` |

PyTorch `forward` 中数值参考实现相同（`matmul → mask add → softmax → matmul`），**仅 ONNX `symbolic` 出口不同**。这意味着 PyTorch 端调试不影响；转换期根据 `type_s` 自动路由到不同 CANN 算子。

### InnerPFA 的运行时使能（必装）

`InnerPromptFlashAttention` 属于 MSLite **自定义算子包**，**不是 CANN 内建算子**。在 ONNX→MindIR 转换时，`converter_lite` 默认不会找到 InnerPFA 的解析器，必须先安装：

```bash
# 1. 安装 MSLite 自定义算子包（仅一次，MSLite ≥ 2.11 提供 install.sh）
bash <mslite-tar-path>/tools/custom_kernels/install.sh

# 2. 设置环境变量（每个新 shell 都需执行，或写入 ~/.bashrc）
source <cann-path>/ascend-toolkit/latest/opp/vendors/mslite_custom_ops/bin/set_env.bash
```

安装 + 设好环境变量后，再跑 `$Convert` 即可让 suffix 模型转换时自动命中 InnerPFA 算子。如果不装，会看到转换报错或推理期 fallback 到非融合路径，性能明显下降。

> Prefix 模型 `q_len == k_len`，仍用标准 `PromptFlashAttention`，**无需 InnerPFA 包**。只需 suffix 模型走 InnerPFA。

### 什么时候不开 PFA

- 若不想装自定义算子包：**跳过 `--enable-pfa`**，suffix attention 走原生 PyTorch matmul + softmax + matmul 路径（功能等价）。
- 若 prefix 极短 / 整段请求 prefix+suffix 总长都很小，开 PFA 的 128 对齐 padding 反而可能拉低收益——按 §SKILL.md 通用原则 7 做一次 fused vs unfused benchmark 对比。

## 落地检查清单

1. **ONNX 模型拆分**：prefix 只输出 KV（无 lm_head、无 logits），suffix 接 lm_head 并 slice_last 输出 `[B, 1, vocab]`。
2. **PyTorch forward 数值对齐**：prefix KV 直接喂 suffix，对齐端到端 `argmax(logits)` 与 baseline prefill 一致。
3. **Suffix 走 InnerPFA**（如开启 PFA）：`symbolic` 用 `type_s="InnerPromptFlashAttention"` + 5 槽位 `input_names_s` + `input_index_i=[0,1,2,4]`。
4. **运行时使能**：MSLite ≥ 2.11 + 装好 `custom_kernels` + `source set_env.bash`，否则 suffix 转换/推理会失败或回退。
5. **转换 ini**：`ge.dynamicDims` 把 prefix 档位（如 480/768）和所有 suffix bucket（S 与 P+S）都覆盖。
6. **推理 padding**：prefix tokenize 后右 padding 到 `prefix_seq_len`，suffix 上 pad 到最近的 suffix bucket；`position_ids` 从 `prefix_seq_len` 起算。
7. **复用 device KV**：prefix 输出的 KV buffer 用 `mslite.Tensor(..., device="ascend:0")` 预分配并复用为 suffix 的 `past_key_values`（见 [zero_copy_inference.md](zero_copy_inference.md)）。
8. **混合精度黑名单**：prefix 与 suffix 都保留 `op_fp32.json` 黑名单（RmsNorm 敏感算子保 FP32）。
9. **精度对齐**：与普通 prefill+decode 基线比 prefix KV 与 suffix logits 的 max_abs / cosine；top-1 token 必须一致。

## 性能参考（Qwen3-0.6B，Atlas 300I Duo，P=768）

| 阶段 | 耗时 |
|------|------|
| Prefix（P=768） | ~70–87 ms |
| Suffix（S=64）首次 | ~18–24 ms |
| Suffix（S=64）后续复用 prefix KV | **~18 ms**（纯 suffix，无 prefix 重算） |
| Suffix（S=512） | ~70–72 ms |

> 后续请求省掉 prefix 重算 ≈ 70–87 ms；同样请求若走完整 prefill，prefix 长度越长收益越大。

## 参考实现

- 完整 ONNX 导出：[`export_qwen3_onnx.py`](../../../../../mindspore-lite/examples/base_models/qwen3_0.6b/export_qwen3_onnx.py)（`Qwen3PrefixModel` / `Qwen3SuffixModel` / `_CannInnerPromptFlashAttention`）
- 转换 ini：[`qwen3_llm_prefill_prefix.ini`](../../../../../mindspore-lite/examples/base_models/qwen3_0.6b/configs/qwen3_llm_prefill_prefix.ini) 与 [`qwen3_llm_prefill_suffix.ini`](../../../../../mindspore-lite/examples/base_models/qwen3_0.6b/configs/qwen3_llm_prefill_suffix.ini)
- 推理脚本：[`infer_qwen3_0.6b_mindir.py`](../../../../../mindspore-lite/examples/base_models/qwen3_0.6b/infer_qwen3_0.6b_mindir.py) 中 `Qwen3CommonPrefixInferencer`
- 端到端 README 与场景 C 性能数据：[`README.md §4 场景 C`](../../../../../mindspore-lite/examples/base_models/qwen3_0.6b/README.md)
- 通用免拷贝：见 [zero_copy_inference.md](zero_copy_inference.md)
- Custom 改写规范：见 [custom_operator_fusion.md](custom_operator_fusion.md)