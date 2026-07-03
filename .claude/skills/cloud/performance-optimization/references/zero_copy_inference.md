# MindSpore Lite 推理免拷贝（Ascend）

> 本文档是 `performance-optimization` skill 的细化策略文档之一。
> 适用场景：降低 Host↔Device 拷贝与重复分配开销，适用于 Decoder/KV cache、多模型流水线、自回归迭代、多分支融合等“模型输出会被下一阶段继续使用”的场景。

目标：降低 Host↔Device 拷贝与重复分配开销。免拷贝不仅适用于 Decoder/KV cache，自任何“模型输出会被下一阶段继续使用”的场景都适用，包括：

- 多模型流水线：Model A 的输出作为 Model B 的输入（例如 vision → prefill → decode）
- 自回归/迭代：同一模型的输出在下一步作为输入的一部分（例如 KV cache、状态量、循环 buffer）
- 多分支融合：同一中间张量被多处复用（避免反复 `get_data_to_numpy()`）

## 核心做法

1. 全链路尽量用 Ascend 侧 Tensor 传递中间结果

- 输入与输出尽量使用 `mslite.Tensor(shape=[...], dtype=..., device="ascend:<id>")` 创建并复用。
- `Model.predict(inputs, outputs=...)` 使用预分配的输出 buffer，减少输出分配与拷贝。
- 大张量（如特征、KV cache）尽量保持在 device，避免落到 numpy 后再回传。

2. 复用 buffer / ping-pong

- 当某个输出会成为下一步输入时，优先直接复用返回的 device Tensor。
- 若希望配合 `outputs=` 固定输出 buffer，可用双 buffer 交替复用（ping-pong）避免覆盖：
  - 第 N 次：`in=A`，`out=B`
  - 第 N+1 次：`in=B`，`out=A`

3. 仅对必须在 CPU 侧处理的数据回拷

- 小张量（例如单步 logits、少量标量）可以 `get_data_to_numpy()` 之后在 CPU 做 `argmax/后处理`。
- 大张量不回拷：将后处理尽量下沉到下一模型或后端算子。

## 实施步骤（建议模板）

1. 找出关键“中间大张量”（从 A 输出到 B 输入 / 下一步循环输入），把它们改成 device Tensor 传递。
2. 预创建并缓存输入/输出 Tensor（按固定 shape/dtype），循环内只做 `set_data_from_numpy()` 更新小输入。
3. 对照优化前后统计：
   - 单阶段耗时（尤其是循环阶段的 avg step）
   - 端到端耗时
   - Host↔Device 拷贝次数（可通过日志/Profiling 侧面验证）

## 具体示例：Qwen3-VL 三阶段流水线免拷贝

以下示例基于 Qwen3-VL 的 Vision → Prefill → Decode 三阶段推理流程，展示免拷贝模式的完整落地方式。

### 1. 预分配 Device Tensor（输入侧）

在 Decode 循环开始前，一次性创建所有输入/输出 Tensor（固定 shape/dtype），并指定 `device="ascend:<id>"`：

```python
def prepare_decode_io(max_seq_len: int, past_kv_fixed: np.ndarray, device_id: int):
    """预分配 decode 循环所需的 device Tensor（输入 + 输出 buffer）。"""
    device_str = f"ascend:{int(device_id)}"

    # ── 输入 Tensor（小张量，每步 set_data_from_numpy 更新）──
    t_input_ids = mslite.Tensor(
        shape=[1, 1], dtype=mslite.DataType.INT32, device=device_str
    )
    t_attention_mask = mslite.Tensor(
        shape=[1, int(max_seq_len)], dtype=mslite.DataType.INT32, device=device_str
    )
    t_position_ids = mslite.Tensor(
        shape=[4, 1, 1], dtype=mslite.DataType.INT32, device=device_str
    )
    t_cache_pos = mslite.Tensor(
        shape=[1], dtype=mslite.DataType.INT32, device=device_str
    )

    # ── 大张量：KV cache 输入（从 prefill 输出拷贝到固定 buffer）──
    t_past_in = mslite.Tensor(
        shape=list(past_kv_fixed.shape),
        dtype=_np_dtype_to_mslite(past_kv_fixed.dtype),
        device=device_str,
    )
    # 首次用 numpy 数据填充，后续复用此 device Tensor
    t_past_in.set_data_from_numpy(past_kv_fixed)

    # ── 输出预分配 buffer──
    outs = decode_model.get_outputs()
    logits_shape = tuple(int(x) for x in outs[0].shape)
    past_shape = tuple(int(x) for x in outs[1].shape)
    t_logits_out = mslite.Tensor(
        shape=list(logits_shape), dtype=mslite.DataType.FLOAT16, device=device_str,
    )
    t_past_out = mslite.Tensor(
        shape=list(past_shape), dtype=mslite.DataType.FLOAT16, device=device_str,
    )

    return {
        "t_input_ids": t_input_ids,
        "t_attention_mask": t_attention_mask,
        "t_position_ids": t_position_ids,
        "t_cache_pos": t_cache_pos,
        "t_past_in": t_past_in,
        "t_past_out": t_past_out,
        "out_bufs": [t_logits_out, t_past_out],
    }
```

### 2. Decode 循环：set_data_from_numpy 更新小输入 + 复用 device Tensor

每步仅用 `set_data_from_numpy()` 更新**小张量**（当前 token ID、position_ids、cache_pos），大张量（attention_mask 固定 buffer、KV cache）始终保持在 device 侧：

```python
def decode_step(io, cache_pos, step_id, attn_mask_fixed, position_ids_step):
    """单步 decode：零拷贝推理。"""
    # 小输入用 set_data_from_numpy 写入预分配的 device Tensor
    io["t_input_ids"].set_data_from_numpy(step_id)              # [1, 1]
    io["t_attention_mask"].set_data_from_numpy(attn_mask_fixed)  # 复用已分配 buffer
    io["t_position_ids"].set_data_from_numpy(position_ids_step)  # [4, 1, 1]
    io["t_cache_pos"].set_data_from_numpy(
        np.array([cache_pos], dtype=np.int32)                    # 标量
    )

    inputs = [
        io["t_input_ids"],
        io["t_attention_mask"],
        io["t_position_ids"],
        io["t_past_in"],       # KV cache 输入（device Tensor，免拷贝）
        io["t_cache_pos"],
    ]
    # outputs= 使用预分配的 device buffer，避免 predict 内部重新分配
    return decode_model.predict(inputs, outputs=io["out_bufs"])
```

### 3. Ping-pong Buffer 交换（KV cache 输出复用）

KV cache 的**输出会作为下一步的输入**，因此使用双 buffer 交替复用，避免一次额外的拷贝：

```python
def decode_loop(io, past_kv_fixed, max_new_tokens, prompt_len, max_seq_len,
                rope_deltas_np, eos_token_id):
    """自回归 decode 循环：ping-pong 交换 KV cache buffer。"""
    generated = []
    cache_pos = int(prompt_len)
    attn_mask_fixed = np.zeros((1, max_seq_len), dtype=np.int32)
    attn_mask_fixed[0, :prompt_len] = 1

    # 首次将 prefill 输出的 KV cache 填入 device Tensor
    io["t_past_in"].set_data_from_numpy(past_kv_fixed)

    for step in range(max_new_tokens - 1):
        if eos_token_id is not None and generated and generated[-1] == eos_token_id:
            break
        if cache_pos >= max_seq_len:
            break

        step_id = np.array([[generated[-1]]], dtype=np.int32)
        attn_mask_fixed[0, :cache_pos + 1] = 1
        text_pos_step = np.array([[[cache_pos]]], dtype=np.int32)
        mm_pos_step = (text_pos_step + rope_deltas_np.reshape(1, 1, 1)).repeat(3, axis=0)
        position_ids_step = np.concatenate([text_pos_step, mm_pos_step], axis=0)

        decode_out = decode_step(io, cache_pos, step_id, attn_mask_fixed, position_ids_step)

        # ── Ping-pong：交换 in/out buffer ──
        t_prev_in = io["t_past_in"]
        io["t_past_in"] = decode_out[1]      # 本次输出的 KV cache 作为下次输入
        io["t_past_out"] = t_prev_in          # 原输入 buffer 回收为下次输出
        io["out_bufs"][1] = io["t_past_out"]

        logits = decode_out[0].get_data_to_numpy()  # 小张量回拷到 CPU 做 argmax
        generated.append(int(np.argmax(logits[0, -1])))
        cache_pos += 1

    return generated
```

### 4. 多模型流水线：Vision → Prefill 的中间结果传递

Vision 模型的输出（image_embeds、deepstack_embeds）通过 numpy 作为 Prefill 模型的输入（当前示例）。**若需进一步优化**，可将 Vision 输出直接构造为 device Tensor，避免 Host↔Device 来回拷贝：

```python
# 当前方案：通过 numpy 中转（仍有 Host↔Device 拷贝）
vision_out = vision_model.predict(vision_inputs)
image_embeds = vision_out[0].get_data_to_numpy()       # Device → Host
deepstack_embeds = vision_out[1].get_data_to_numpy()   # Device → Host

# ...

# prefill 输入构造时，numpy 数据由 predict 内部传回 device
prefill_out = prefill_model.predict(prefill_inputs)    # Host → Device

# ── 优化方案：Vision 输出直接作为 device Tensor ──
vision_out = vision_model.predict(vision_inputs)
# 直接从 predict 返回结果中取出 device Tensor（不调 get_data_to_numpy）
image_embeds_tensor = vision_out[0]   # 保持 device 侧
deepstack_embeds_tensor = vision_out[1]
# 将 device Tensor 直接传入 prefill 的 feed dict
prefill_feed = {
    "input_ids": input_ids_np,
    "attention_mask": attention_mask_np,
    "position_ids": position_ids_np,
    "image_embeds": image_embeds_tensor,     # device Tensor
    "deepstack_embeds": deepstack_embeds_tensor,  # device Tensor
}
# prefill 输入构造时，对 numpy 数据用 _mslite_tensor，对 device Tensor 直接引用
prefill_inputs = build_mslite_inputs(prefill_model, prefill_feed)
prefill_out = prefill_model.predict(prefill_inputs)
```

### 5. Prefill 阶段：Past KV 输出直接写入 Decode 的预分配 Device Tensor

Prefill 输出的 past_kv 是 numpy（`get_data_to_numpy()`），然后通过 `set_data_from_numpy()` 填入 Decode 的 device Tensor（一次性拷贝，后续 decode 零拷贝）：

```python
# prefill 阶段
prefill_out = prefill_model.predict(prefill_inputs)
past_kv = prefill_out[1].get_data_to_numpy()  # Host 侧

# 在固定长度 buffer 中放置 past_kv
past_kv_fixed = np.zeros(
    (past_kv.shape[0], past_kv.shape[1], past_kv.shape[2],
     max_seq_len, past_kv.shape[4]),
    dtype=past_kv.dtype,
)
past_kv_fixed[:, :, :, :prompt_len, :] = past_kv

# 一次性拷贝到 device Tensor（此后 decode 循环中不再拷贝）
io["t_past_in"].set_data_from_numpy(past_kv_fixed)
```

### 6. 总结：各阶段的免拷贝策略

| 阶段 | 免拷贝策略 | 关键代码 |
|------|-----------|---------|
| Vision → Prefill | 优化方案：Vision 输出 device Tensor 直接喂给 Prefill | `vision_out[0]` 直接传入 feed dict |
| Prefill → Decode | 一次 `set_data_from_numpy()` 将 past_kv 写入 device Tensor | `io["t_past_in"].set_data_from_numpy(past_kv_fixed)` |
| Decode 循环输入 | 预分配 device Tensor，每步 `set_data_from_numpy()` 更新小输入 | 固定 shape 的 `mslite.Tensor(..., device=device_str)` |
| Decode 循环输出 | `outputs=` 预分配 buffer + ping-pong 交换 | `predict(inputs, outputs=io["out_bufs"])` |
| 采样后处理 | 仅小张量（logits）回拷到 CPU | `decode_out[0].get_data_to_numpy()` |

## 注意事项

- 运行前必须正确加载 Ascend/CANN 环境（例如 `source set_env.sh`，确保 `libgraph.so` 等可用），否则 Ascend device Tensor 分配会失败。
- `Tensor(shape=..., dtype=..., device=...)` 是推荐构造方式；`Tensor(numpy_obj, device=...)` 依赖 Ascend 插件完成 device 内存分配，环境不完整时更容易失败。
- 跨模型传递 Tensor 时需确保 dtype/shape/布局与下游模型输入严格匹配；不匹配时宁可显式做一次转换，也不要隐式回拷导致性能退化。
