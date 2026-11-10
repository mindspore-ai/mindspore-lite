---
name: parallel-deployment
description: 开源 LLM 多卡/多芯并行部署（模型并行两条路线）：TP（Megatron 权重分片 + Custom(AllReduce)，1p/2p/4p，300I Duo 与 800I A2 差异）与 CP（Ulysses 序列并行 + AllGather，单卡双芯、公共前缀、零拷贝 KV）。覆盖导出/转换/多进程 HCCL 推理、通信算子、编译缓存与已知限制。用户要把模型部署到多卡/多芯 Ascend、或做 TP/CP 切分时调用。
---

# 开源模型多卡并行部署（TP / CP）

本技能覆盖将开源模型部署到 Ascend 多芯/多卡推理的两条路线：**TP（张量并行，切权重）** 与 **CP（上下文/序列并行，切序列）**，共同流程为 **导出分片 ONNX → 转换 MindIR → 多进程 HCCL 推理**。TP 以 Qwen2.5-7B / Qwen3-8B 为例，CP 以 Qwen3-Reranker-8B 为例。

## 何时调用

- 用户想把模型部署到多芯/多卡 Ascend 推理（"多卡推理"、"张量并行"、"TP=2/4"）
- 用户想做**序列并行/上下文并行**（"CP"、"Ulysses"、"单卡双芯"、"长序列分摊"、"公共前缀 + KV 复用"）
- 用户需要实现 AllReduce / AllGather 通信算子、Megatron 风格权重分片，或序列切分导出
- 用户的多卡推理遇到精度问题或 HCCL 通信问题
- 用户需要在 300I Duo 或 800I A2 上做 TP 部署

## 与其他 Skill 的关系

| 阶段 | 关联 Skill | 说明 |
|------|-----------|------|
| ① 单卡导出/验证 | [open-source-model-migration](../../open-source-model-migration/SKILL.md) | 先跑通单卡（1p），再做多卡 |
| ② Custom 算子融合 | [performance-optimization → custom_operator_fusion.md](../../performance-optimization/references/custom_operator_fusion.md) | RMSNorm/Attention/Scatter 等 Custom 算子的改写方法 |
| ③ ONNX→MindIR 转换 | [onnx-model-conversion-and-deployment](../../onnx-model-conversion-and-deployment/SKILL.md) | 离线 acl 路径（1p）和 GE online 路径（2p/4p） |
| ④ 代码质量 | [clean-code-check](../../common/clean-code-check/SKILL.md) | C++/Python/Shell 代码规范 |

> **调用顺序**：先完成 [open-source-model-migration](../../open-source-model-migration/SKILL.md) 的单卡导出 + [onnx-model-conversion-and-deployment](../../onnx-model-conversion-and-deployment/SKILL.md) 的单卡转换与推理验证，确认 1p 精度正确后，再调用本 Skill 做多卡适配。CP 路线的 prefix 图无集合通信，也可先在单芯验证 K/V 形状与数值，再上双芯跑 suffix。

---

## 并行方式选型：TP 还是 CP

| 维度 | TP（张量并行） | CP（上下文/序列并行） |
|------|---------------|----------------------|
| 切分对象 | 权重（列并行 / 行并行） | **输入序列**（token 维切段） |
| 每 rank 权重 | 1/N | **全量复制** |
| 显存收益 | 有（≈1/N） | 无（权重不降；KV/激活按段降） |
| 通信 | AllReduce（每层 1 次，量 = 全量激活） | AllGather ×2/层（seq 方向 + head 方向） |
| 单请求延迟 | 分摊计算，但有 AllReduce 同步开销 | 分摊计算；通信量随序列长度线性增长 |
| 典型适用 | 单芯放不下权重、要扩到更多卡 | 权重放得下（单卡 44 GB）、要多芯分摊长序列、卡内 HCCS |
| 300I Duo 推荐 | TP=2（同卡双芯），TP=4 需跨卡 PCIe、性能受限 | **CP=2（同卡双芯，HCCS ~60 GB/s）** |

> 一句话：**权重放得下、想降长序列单请求延迟 → CP；权重放不下或要扩到更多卡 → TP**。两者也可组合（先 TP 切权重、再 CP 切序列），但需自行确认 HCCL 通信域与 rank_table 的对应关系。

---

## 1. 张量并行（TP）机制

### 1.1 核心原理

MindSpore Lite 2.10 的 TP 是**导出时分片**（非运行时拆分）：

```
HF 模型 → export.py --tp-size N
  → 每个 rank 导出一份分片 ONNX（权重按 Megatron 切分）
  → converter_lite --optimize=none 转为 MindIR（GE online 路径）
  → 推理时每 rank 加载自己的 MindIR，图内 Custom(AllReduce) 经 GE lower 为 HcomAllReduce
```

### 1.2 Megatron 分片策略

| 组件 | 分片策略 | 说明 |
|------|---------|------|
| QKV / gate_up | **列并行**（output dim 切分） | 每 rank 持有 output_size/N 的权重 |
| o_proj / down_proj / lm_head | **行并行**（input dim 切分）+ **AllReduce** | 每 rank 计算部分输出，AllReduce 求和 |
| RMSNorm / rotary / embed_tokens | **复制**（不切分） | 所有 rank 持有完整副本 |
| KV cache | **按 KV head 切分** | num_kv_heads/N 个 head/rank |

### 1.3 Qwen2.5-7B 分片参数（示例）

| 参数 | 全量 | TP=2 (per rank) | TP=4 (per rank) |
|------|------|-----------------|-----------------|
| Q heads | 28 | 14 | 7 |
| KV heads | 4 | 2 | 1 |
| Intermediate | 18944 | 9472 | 4736 |
| KV cache shape | (28,1,4,512,128) | (28,1,2,512,128) | (28,1,1,512,128) |
| AllReduce / step | — | 57 | 57 |

---

## 2. 导出：权重分片 + Custom 算子

### 2.1 导出脚本关键逻辑

```python
# TP 全局变量
TP_SIZE = 1  # 由 --tp-size 设置
TP_RANK = 0  # 由 --rank 设置

# QKV 列并行
q_per = q_w.shape[0] // TP_SIZE
qs, qe = TP_RANK * q_per, (TP_RANK + 1) * q_per
w = torch.cat([q_w[qs:qe], k_w[ks:ke], v_w[ks:ke]], dim=0)

# o_proj 行并行 + AllReduce
o_w_local = o_w[:, TP_RANK * q_dim_local : (TP_RANK+1) * q_dim_local]
out_proj = allreduce_sum(F.linear(out, o_w_local))
```

### 2.2 Custom 算子清单（TP 推理必需）

TP 推理依赖以下 Custom 算子（通过 `torch.autograd.Function` 的 `symbolic` 方法发射到 ONNX）：

#### 2.2.1 Custom(AllReduce) — 通信算子（核心）

```python
class _AllReduceCustom(torch.autograd.Function):
    """发射 Custom(type=AllReduce) → GE 运行时 lower 为 HcomAllReduce"""
    @staticmethod
    def symbolic(g, x):
        return g.op("Custom", x,
                    type_s="AllReduce", op_s="sum",
                    group_s="hccl_world_group",
                    rank_size_i=int(TP_SIZE), fusion_i=0)
```

- **eager fallback**：identity（仅用于 trace，真实通信在 GE 运行时）
- **HcomAllReduce 输出 fp32**（即使输入是 fp16），下游需注意 dtype 匹配

#### 2.2.2 Custom(Scatter) — KV cache 更新

```python
# 原生 torch.scatter 实现（导出为 ONNX ScatterElements）
def scatter(var, indices, updates, ...):
    out = var.clone()
    out = torch.scatter(out, dim=2, index=idx, src=src)
    return out
```

---

## 3. 转换：ONNX → MindIR

### 3.1 双路径

| 模式 | 转换方式 | 权重处理 | 运行时 |
|------|---------|---------|--------|
| **1p（单卡）** | `--optimize=ascend_oriented` + configFile | 外部化到 `_variables/` | 离线编译，无需 config_file |
| **2p/4p（多卡）** | `--optimize=none` | 内嵌 MindIR | online GE，需 config_file（rank_table + provider=ge）|

详见 [onnx-model-conversion-and-deployment](../../onnx-model-conversion-and-deployment/SKILL.md)。

### 3.2 2p/4p 特殊要求

1. **`plugin_custom_ops=All`**：config_file 中必须配置，让 Custom 算子走 Ascend 插件
2. **静态 shape**：TP 导出使用 `static=True`（无 dynamic_axes），因为 GE online 路径不解析动态维度

## 4. 推理：多进程 HCCL

### 4.1 推理架构

```
Driver process (编排)
  ├── Worker rank0 (device D0): build prefill+decode → prefill → decode loop
  ├── Worker rank1 (device D1): build prefill+decode → prefill → decode loop
  └── (4p: rank2, rank3 同理)
      图内 Custom(AllReduce) 经 HcomAllReduce 做跨 rank 通信
```

### 4.2 关键实现点

1. **Context 配置**：`ctx.ascend.provider = "ge"` + `rank_id` + `device_id` + config_file（含 rank_table）
2. **rank_table.json**：由 `infer.sh` 根据设备列表自动生成
3. **预热（warmup）**：所有 rank 必须同步完成预热后才能开始计时推理
4. **退出**：`os._exit(0)`（HCCL communicator destroy 是集合操作，正常 return 会死锁）

## 5. 硬件差异：300I Duo vs 800I A2

### 5.1 通用部分（两者一致）

- TP 导出逻辑（Megatron 分片 + Custom AllReduce 发射）
- 多进程 HCCL 推理架构（driver + workers）
- rank_table.json 格式
- Custom 算子 ONNX symbolic 发射方法

### 5.2 差异点

| 维度 | Atlas 300I Duo | Atlas 800I A2 |
|------|------------------------|----------------------|
| **内存** | LPDDR4X 204GB/s/chip（NOT HBM） | HBM2e ~1TB/s |
| **Cube 精度** | fp16 only（无 f32f32f32 mmad） | fp16 + fp32 Cube |
| **卡内互联** | HCCS ~60GB/s（2 芯） | HCCS ~300GB/s（8 芯） |
| **卡间互联** | PCIe ~16GB/s（NO HCCS 跨卡） | HCCS 跨卡 ~100GB/s |
| **每卡芯数** | 2 芯/卡 | 8 芯/卡 |
| **最大 TP（同卡）** | TP=2 | TP=8 |
| **最大 TP（跨卡）** | TP=4+（跨卡 PCIe，性能受限） | TP=16+（跨卡 HCCS） |
| **decode 带宽瓶颈** | 极严重（14GB / 204GB/s ≈ 69ms 下限） | 较轻（14GB / 1TB/s ≈ 14ms 下限） |

### 5.3 300I Duo 特有限制

1. **decode 带宽瓶颈**：LPDDR4X 带宽远低于 HBM，单芯 decode ~86ms，必须靠 TP 分摊带宽
2. **28 层 TP=4 decode GE miscompile**：GE 图编译器在 >4 层 4-rank decode 图上产生确定性精度错误（详见第 6 节）
3. **DataCopyPad/ReduceSum/Broadcast**：300I Duo 的 AscendC 不支持这些算子，自定义算子开发需规避

### 5.4 800I A2 优势

1. **HBM 带宽**：~1TB/s vs 204GB/s，decode 性能远优
2. **fp32 Cube**：支持 f32 矩阵乘（300I Duo 只能 fp16），算法选择更多
3. **8 芯/卡**：单卡可做 TP=8，无需跨卡 PCIe
4. **HCCS 跨卡**：跨卡带宽 ~100GB/s（vs 300I Duo PCIe 16GB/s），TP>8 时性能损耗小

---

## 6. 完整工作流模板（TP）

### Step 0：确认单卡（1p）正常

先完成 [open-source-model-migration](../../open-source-model-migration/SKILL.md) + [onnx-model-conversion-and-deployment](../../onnx-model-conversion-and-deployment/SKILL.md)，确认 1p 精度正确。

### Step 1：导出分片 ONNX

```python
# export.py 中设置 TP 分片
TP_SIZE = <N>  # 2 or 4
for rank in range(TP_SIZE):
    export_tp_prefill(model, out_dir, rank, TP_SIZE, ...)
    export_tp_decode(model, out_dir, rank, TP_SIZE, ...)
```

### Step 2：Custom 算子构建

```bash
bash build_all_ops.sh <out>  # 构建 mslite_custom_ops
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/mslite_custom_ops
```

### Step 3：转换 MindIR

```bash
# converter_lite --optimize=none --saveType=MINDIR（每 rank 每子图）
```

### Step 4：多进程推理验证

```bash
# 生成 rank_table + config
# python3 infer.py --device-ids D0,D1,...,DN-1 --tp-size N
```

### Step 5：精度验证

- 对比 1p/2p 输出的 first token（prefill argmax 应完全一致）
- 对比多步 decode 生成的 token 序列
- 如有差异，用小模型（2~4 层）做 4p vs 2p token 对比（排除通信问题 vs 图编译问题）

### Step 6：性能测试

- 3 轮 warmup → 计时 prefill + N 步 decode → 计算 avg decode step + 吞吐

---

## 7. 参考实现（TP）

| 模型 | 目录 | TP 支持 | 特点 |
|------|------|---------|------|
| Qwen2.5-7B | `examples/base_models/qwen2.5_7b/` | 1p ✅ / 2p ✅（与 1p 逐 token 一致） | 4 KV heads；2p decode 67ms(15 tok/s) |
| Qwen3-8B | `examples/base_models/qwen3_8b/` | 1p ✅ / 2p ✅（与 1p 逐 token 一致） | **8 KV heads**（TP=4 每 rank 2 KV head，原生 GQA decode 无需 MHA workaround） |

---

# 第二部分：上下文并行（CP）

> 以下 CP 内容基于 **Qwen3-Reranker-8B（CP=2，单卡双芯）+ mslite 2.11.0（`8d42c12a`）** 实测；
> 与 TP 的差异先看开头「并行方式选型」。CP 的关键点有三个：**序列切段 + AllGather**、**prefix/suffix 双图**、**零拷贝 KV**。

## 8. 上下文并行（CP）机制

### 8.1 核心原理

CP（Ulysses）切的是**序列维**，不是权重：每 rank 持完整权重，只算自己那段 token，attention 前把序列 gather 齐、算完再切回本地段。

```
HF 模型 → export.py --cp-size 2
  → 每个 rank 导出一份 ONNX（权重全量复制，输入序列按 rank 切段）
  → converter_lite --optimize=none → MindIR（GE online 路径）
  → 双进程（同卡两芯）各加载自己的 MindIR；图内 Custom(AllGather) lower 为 HcomAllGather
```

### 8.2 Ulysses 切分策略

| 组件 | 切分方式 | 说明 |
|------|---------|------|
| 权重（QKV / o_proj / MLP / lm_head） | **复制**（不切分） | 每 rank 完整权重；CP 不省权重显存，只分摊计算 |
| 输入序列 | **按 token 切段** | rank r 持 `[r·S/N, (r+1)·S/N)`；`position_ids` 用**全局绝对位置** |
| attention | 先 `AllGather(seq)` 拼成全序列再算（本 rank 负责 `32/N` 个 q 头、`8/N` 个 kv 头） | 每层第 1 次通信 |
| attention 输出 | `AllGather(heads)` 拼回全部头，再切回本段 | 每层第 2 次通信 |
| KV cache | 每 rank 只存本段（prefix 图则存整个 prefix 桶） | CP 省 KV 显存 |
| lm_head | 全量权重；host 取**拥有全局最后一个 token** 的 rank 的 logits | 左 padding 下恒为 rank N-1 |

### 8.3 Qwen3-Reranker-8B CP2 参数（示例）

| 参数 | 全量 | CP=2（per rank） |
|------|------|-----------------|
| layers | 36 | 36（完整） |
| hidden | 4096 | 4096 |
| q heads | 32 | 16（计算），AllGather 后拼回 32 |
| kv heads | 8（GQA） | 4（计算） |
| head_dim | 128 | 128 |
| intermediate | 12288 | 12288（完整） |
| 序列 | S | S/2；`attention_mask` 仍是**全序列** S |
| 权重显存 | ~15 GB（fp16） | **~15 GB/rank（不降）** |
| 每层通信 | — | 2 × AllGather |

---

## 9. CP 导出：序列切分 + prefix/suffix 双图

### 9.1 导出脚本关键逻辑

```python
CP_SIZE = 2      # --cp-size
CP_RANK = 0      # 循环写入

seg = seq_len // CP_SIZE
ids_local = ids[CP_RANK*seg:(CP_RANK+1)*seg]                 # 只留本段
pos_local = list(range(CP_RANK*seg, (CP_RANK+1)*seg))        # 全局绝对位置
attn      = [1] * seq_len                                    # mask 仍是全序列

dynamic = {"input_ids":      {1: "seq_local"},
           "position_ids":   {1: "seq_local"},
           "attention_mask": {1: "seq_full"},                # 注意：全序列动态
           "logits":         {1: "last"}}
```

与 TP 的导出差异：**TP 切权重**（每 rank 权重量不同、图不同）；**CP 切输入**（权重逐 rank 完全相同，只有序列段不同）。

### 9.2 Custom 算子：只有 AllGather

```python
class _AllGatherCustom(torch.autograd.Function):
    """发射 Custom(type=AllGather) → GE 运行时 lower 为 HcomAllGather"""
    @staticmethod
    def symbolic(g, x, dim):
        return g.op("Custom", x, type_s="AllGather", op_s=dim,
                    group_s="hccl_world_group", rank_size_i=int(CP_SIZE))
```

- **方向 1（序列）**：`[1, S/N, H]` → `[1, S, H]`，用于全序列 attention；
- **方向 2（head）**：`[1, S, (32/N)·128]` → `[1, S, 32·128]`，用于 o_proj 前拼齐全头；
- 与 TP 的 AllReduce 不同：AllGather 是**拼接**语义、不做归约，所以不像 AllReduce 那样引入求和误差；代价是通信量随序列线性增长。

### 9.3 prefix / suffix 双图契约（Reranker 场景）

公共前缀（system + `<Instruct>` 模板）固定，拆成两张图：

| 图 | shape | 产出 / 输入 | rank 相关性 |
|----|-------|------------|------------|
| **prefix** | 固定 512 桶（128 对齐） | 产出 36 层 K/V `[36, 1, 512, 8, 128]` fp16 | **逐 rank 完全相同**（全头、无集合通信，单芯可跑） |
| **suffix** | 动态 4 档（512/1024/1536/2048） | 输入 = 本 rank 的 suffix 段 + 全序列 mask + prefix 的 K/V；输出全局最后 token 的 logits | 逐 rank 不同（段不同） |

suffix 的关键 config（`[ascend_context]` / `[ge_session_options]` / `[ge_graph_options]`）：

```ini
[ascend_context]
plugin_custom_ops=All
model_cache_mode=mem_opt
model_cache_dir=<dir>/rank{r}          # 必须按 rank 分开：GE 的 key 不含 rank

[ge_session_options]
ge.constLifecycle=session
ge.exec.staticMemoryPolicy=2
ge.exec.precision_mode=force_fp16
ge.externalWeight=2

[ge_graph_options]
ge.inputShape=input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1;past_key_cache:36,1,-1,8,128;past_value_cache:36,1,-1,8,128
ge.dynamicDims=512,1024,512,512,512;1024,1536,1024,512,512;1536,2048,1536,512,512;2048,2560,2048,512,512
ge.dynamicNodeType=1
```

> 五元组顺序 = `input_ids, attention_mask, position_ids, past_key_cache, past_value_cache`；`attention_mask` 长度 = 512（prefix 桶）+ suffix 桶。

---

## 10. CP 转换：静态 prefix + 动态 suffix

| 图 | 转换要点 | 运行时 |
|----|---------|--------|
| prefix | `converter_lite --optimize=none --saveType=MINDIR --inputShape="input_ids:1,512;attention_mask:1,512;position_ids:1,512"` | 静态图，**绝不调 `resize()`**（resize 会把图拉回 GE 动态路径，KV 从 layer1 起错） |
| suffix | `converter_lite --optimize=none --saveType=MINDIR`（不带 `--inputShape`） | 动态图，按桶 `resize`；档位由 `ge.dynamicDims` 提供 |

与 TP 的差异：TP 全静态（GE online 不解析动态维）；CP 的 suffix 必须动态（业务序列长度变化），prefix 用静态换确定性。

---

## 11. CP 推理：零拷贝 KV + 双进程 HCCL

### 11.1 推理架构

```
Driver process（编排）
  ├── Worker rank0 (device D0): build prefix+suffix → prefix 算 KV → 服务 suffix
  └── Worker rank1 (device D1): build prefix+suffix → prefix 算 KV → 服务 suffix
        prefix 输出的 K/V 以 device 张量直接交给 suffix —— 零拷贝，不进 host、不落 npy
        图内 Custom(AllGather) → HcomAllGather（每层 2 次）
```

### 11.2 关键实现点

1. **HCCL 对齐**：`rank_table.json` + `ctx.ascend.rank_id` + `provider="ge"`；两个 rank 必须同时到场，单 rank 预热会死锁；
2. **零拷贝 KV**：预分配 `mslite.Tensor(shape=[36,1,512,8,128], dtype=fp16, device="ascend:<id>")` 两枚 → `prefix.predict(feeds, outputs=[t_k, t_v])` → **同一对张量**直接作为 suffix 的输入（全程无 H2D/D2H）；
3. **suffix 必须显式传 outputs**：vocab 151669 被 GE 按 32 对齐到 151680，走默认输出路径会在 `ConvertGeTensorNoCopy` 报 dtype/size mismatch；
4. **打分取 owner rank**：`score = softmax([logit(no), logit(yes)])[1]`，`yes=9693 / no=2152`，取 `CP_SIZE-1` 的 logits（左 padding 下全局最后一个 token 恒在 rank N-1）；
5. **prefix 图不要 resize**（静态构建后 mslite 仍报 `[-1]`，resize 会切回动态路径并损坏 KV）；
6. **缓存按 rank 分目录**：`model_cache_dir=<dir>/rank{r}`（GE 的 key 不含 rank，共用会静默错值）；
7. **退出用 `os._exit(0)`**：HCCL communicator destroy 是集合操作，正常 return 会死锁。

---

## 12. CP 完整工作流模板

### Step 0：单芯先验 prefix

prefix 图无集合通信，可先单芯 `build + predict` 验证 K/V 形状与数值，排除导出问题后再上双芯。

### Step 1：导出（每 rank 一份 prefix + suffix）

```bash
python3 export_qwen3_reranker_8b_onnx_cp.py --model-id <hf_dir> --output-dir <out> \
        --dtype fp16 --cp-size 2 --prefix-len 512 --suffix-len 384 --mode both
```

### Step 2：转换（prefix 静态 / suffix 动态）

```bash
# prefix：--inputShape 固定 512
# suffix：不带 --inputShape，靠 ge.dynamicDims 分档
```

### Step 3：双进程推理（零拷贝 KV）

```bash
python3 infer_qwen3_reranker_8b_mslite_cp.py --device-ids 0,1 --model-id <hf_dir> \
        --mindir-root <out>/mindir --cfg-dir <out>/configs \
        --prefix-bucket 512 --prefix-text-file prefix_official.txt \
        --suffix-buckets 512,1024,1536,2048 --queries t2rerank_20_official.tsv \
        --warmup 1 --json-out scores.json
```

### Step 4：精度验证

- 与 CPU eager（fp32）比：`score`、`max|logit−eager|`、AUC/MRR/nDCG；
- 参考值（T2Reranking 40 对）：首对 score **0.1322**（eager 0.1306）、`max|logit−eager|` **0.48**、AUC **0.7675**（eager 0.7650）、16/20 成对正确。

### Step 5：性能测试

- suffix 按桶扫描：`warmup ≥ 1` + `repeats ≥ 10` 取 median；
- 记录冷启动（含 GE 特化编译）与命中缓存两套数值。

---

## 13. CP 实测数据与参考实现

实测环境：Atlas 300I Duo（单卡双芯，device 0/1）+ CANN 9.1.0 + mslite 2.11.0（`8d42c12a`），Qwen3-Reranker-8B，fp16。

| 项 | 数值 |
|----|------|
| 整图 CP2 端到端（256/512/1024/2048/4096） | 133.0 / 182.4 / 372.4 / 798.5 / 1870.2 ms |
| prefix（512 桶）单算 | 265.0 ms（两芯各算一份） |
| suffix（256/512/1024/1536/2048） | 141.5 / 203.9 / 432.5 / 686.4 / 943.6 ms |
| 冷启动（含 GE 特化编译） / 命中缓存 | 687 s / 330 s |
| 编译缓存占用 | 15 GB/rank（`weight/` 14.8 GB） |

参考实现：`examples/base_models/qwen3_reranker_8b/`（CP2 + 公共前缀 + 零拷贝 KV，含导出/转换/推理脚本与 README）。

---

## 14. CP 已知限制与排查

| 现象 | 原因 | 处理 |
|------|------|------|
| 首请求不比整图 prefill 快 | prefix 在两芯上各算一份（无集合通信），且「prefix + suffix」两段串行 | 预期行为，收益从第二个请求开始；想再降可把 prefix 改成「算一份 + KV 广播」 |
| 两 rank 共用一个缓存目录 → 数值全错 | GE 缓存 key 不含 rank，两个 rank 互相复用编译产物 | 必须 `model_cache_dir=<dir>/rank{r}`；实测 score 0.1322 → 0.0168、40 对相关 0.17 |
| `model_cache_dir` 未预建 | 新版降级为「不启用缓存」，不报错但每次全量重编 | 先 `mkdir -p`（空间按 rank 算 ~15 GB） |
| `ge.graph_compiler_cache_dir` 目录不存在 | mslite 不校验该入口，GE 在**首次 predict** 报 E13026（build 仍成功） | 预建目录，或改用 `model_cache_dir` |
| 只读 cwd 冷启动失败 | TBE 要写 `cwd/kernel_meta` | 预置缓存（热缓存可正常跑），或重定向 kernel_meta |
| `ConvertGeTensorNoCopy` dtype/size mismatch | vocab 151669 被按 32 对齐到 151680 | suffix 显式传 `outputs=[logits]` |
| KV 从 layer1 起全错 | 对静态 prefix 图调了 `resize()` | 不要 resize |
| HCCL EI0005 / EI0006 | 两 rank 预热或请求顺序不一致 | barrier 对齐，预热顺序（图序、档位序）两 rank 一致 |
| `suffix N tokens > max bucket` | 真实序列超过 `ge.dynamicDims` 最大档 | 补档（改 config 的 dynamicDims）或做长度截断策略 |
| 共享前缀少 1 token、suffix 位置错位 | `--prefix-text "$(cat f)"` 被 shell 吃掉尾随换行 | 用 `--prefix-text-file`（逐字节读入） |
