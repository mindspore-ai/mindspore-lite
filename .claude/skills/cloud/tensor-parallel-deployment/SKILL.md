---
name: tensor-parallel-deployment
description: 开源 LLM 模型的多卡张量并行部署：Megatron 风格权重分片导出 → ONNX→MindIR 转换（离线 acl / GE online 双路径）→ 多进程 HCCL 推理。覆盖 1p/2p/4p 三种模式，含 Custom 算子（AllReduce/Scatter/RMSNorm 等）开发与通信机制、300I Duo 与 800I A2 差异、已知限制与修复方案。用户想把模型部署到多卡/多芯 Ascend 推理时调用。
---

# 开源模型多卡张量并行部署

本技能覆盖将开源 LLM 模型（如 Qwen2.5-7B）部署到 Ascend 多芯/多卡推理的全流程：**导出分片 ONNX → 转换 MindIR → 多进程 HCCL 推理**。

## 何时调用

- 用户想把模型部署到多芯/多卡 Ascend 推理（"多卡推理"、"张量并行"、"TP=2/4"）
- 用户需要实现 AllReduce 通信算子或 Megatron 风格权重分片
- 用户的多卡推理遇到精度问题或 HCCL 通信问题
- 用户需要在 300I Duo 或 800I A2 上做 TP 部署

## 与其他 Skill 的关系

| 阶段 | 关联 Skill | 说明 |
|------|-----------|------|
| ① 单卡导出/验证 | [open-source-model-migration](../../open-source-model-migration/SKILL.md) | 先跑通单卡（1p），再做多卡 |
| ② Custom 算子融合 | [performance-optimization → custom_operator_fusion.md](../../performance-optimization/references/custom_operator_fusion.md) | RMSNorm/Attention/Scatter 等 Custom 算子的改写方法 |
| ③ ONNX→MindIR 转换 | [onnx-model-conversion-and-deployment](../../onnx-model-conversion-and-deployment/SKILL.md) | 离线 acl 路径（1p）和 GE online 路径（2p/4p） |
| ④ 代码质量 | [clean-code-check](../../common/clean-code-check/SKILL.md) | C++/Python/Shell 代码规范 |

> **调用顺序**：先完成 [open-source-model-migration](../../open-source-model-migration/SKILL.md) 的单卡导出 + [onnx-model-conversion-and-deployment](../../onnx-model-conversion-and-deployment/SKILL.md) 的单卡转换与推理验证，确认 1p 精度正确后，再调用本 Skill 做多卡适配。

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

## 6. 完整工作流模板

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

## 7. 参考实现

| 模型 | 目录 | TP 支持 | 特点 |
|------|------|---------|------|
| Qwen2.5-7B | `examples/base_models/qwen2.5_7b/` | 1p ✅ / 2p ✅（与 1p 逐 token 一致） | 4 KV heads；2p decode 67ms(15 tok/s) |
| Qwen3-8B | `examples/base_models/qwen3_8b/` | 1p ✅ / 2p ✅（与 1p 逐 token 一致） | **8 KV heads**（TP=4 每 rank 2 KV head，原生 GQA decode 无需 MHA workaround） |

---
