# Qwen2.5-7B-Instruct ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `Qwen2.5-7B-Instruct` 导出为 ONNX、转换为 MindIR，并在 Ascend 上完成端到端推理。支持**单卡（1p）、双芯张量并行（2p）、四芯张量并行（4p）**三种部署模式。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- 昇腾环境（MindSpore Lite 2.10 + CANN 9.0.1）

### 依赖

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.8.0（仅导出时需要） |
| transformers   | 4.51.0 |
| accelerate     | ≥ 0.26 |
| onnx           | 1.21.0 |
| onnxruntime    | 1.24.0 |
| mindspore-lite | 2.10.0 |
| CANN           | 9.0.1 |

```bash
pip install torch==2.8.0 transformers==4.51.0 accelerate onnx==1.21.0 onnxruntime==1.24.0
```

---

## 2. 部署模式总览

| 模式 | 命令 | 设备 | 适用场景 |
|------|------|------|----------|
| **1p 单卡** | `infer.sh 0` | 1 芯 | 基础部署、功能验证 |
| **2p 双芯** | `infer.sh 0,1` | 2 芯（同卡 HCCS） | 性能优化 |
| **4p 四芯** | `infer.sh 0,1,2,3` | 4 芯（跨 2 卡 PCIe） | 极致性能（带宽分摊最多） |

---

## 3. 代码结构

```text
qwen2.5_7b/
├── Qwen2.5-7B-Instruct/          # HF 权重
├── export_qwen2_5_7b_onnx.py     # 统一导出脚本（--tp-size 1/2/4，--kv-cache-len）
├── infer_qwen2_5_7b_mslite.py    # 统一推理脚本（1p 零拷贝 + 2p/4p 多进程 TP，含 bench 模式）
├── export_and_convert.sh         # 导出+转换入口（1p/2p/4p）
├── infer.sh                      # 推理入口（infer.sh <device_ids>）
├── configs/                      # 1p 离线转换配置（dynamicDims 32/64/128，KV=256）
└── README.md                     # 本文件
```

### 快速使用

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_7b

# ① 导出 + 转换
bash export_and_convert.sh 1p    # 单卡（离线 acl，ascend_oriented）
bash export_and_convert.sh 2p    # 双芯（GE online，optimize=none）
bash export_and_convert.sh 4p    # 四芯（GE online，optimize=none）

# ② 推理
bash infer.sh 0                  # 1p 单卡
bash infer.sh 0,1                # 2p 双芯（同卡 HCCS）
bash infer.sh 2,3,4,5            # 4p 四芯（跨 2 卡）
```

---

## 4. 模型导出

### 4.1 导出原理

导出脚本将 Qwen2.5-7B 拆分为 prefill + decode 两个 ONNX 子图：

- **Prefill**：处理完整 prompt，输出 logits + KV cache
- **Decode**：逐 token 生成，输入 past KV cache，输出更新后的 cache + logits

**张量并行（TP）分片**

| 组件 | 分片策略 |
|------|---------|
| QKV / gate_up | 列并行（output dim 切分） |
| o_proj / down_proj / lm_head | 行并行 + AllReduce |
| RMSNorm / rotary / embed | 复制（不切分） |
| KV cache | 按 KV head 切分（4 heads → TP=2 每 rank 2 head, TP=4 每 rank 1 head） |

### 4.2 导出命令

由 `export_and_convert.sh` 统一处理：

```bash
# 脚本内部调用（用户只需 bash export_and_convert.sh 1p/2p/4p）
python3 export_qwen2_5_7b_onnx.py \
  --model-id ./Qwen2.5-7B-Instruct \
  --output-dir <OUT_DIR> \
  --device cpu --dtype fp16 \
  --tp-size <1|2|4>
```

### 4.3 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HF 模型路径 | `./Qwen2.5-7B-Instruct` |
| `--output-dir` | 输出目录 | `./qwen2_5_7b_onnx` |
| `--dtype` | fp16 / bf16 / fp32（推荐 fp16） | `fp16` |
| `--tp-size` | 张量并行度（1=单卡, 2=双芯, 4=四芯） | `1` |
| `--num-layers` | 调试：截断为 N 层（0=全部 28 层） | `0` |

---

## 5. ONNX → MindIR 转换

### 5.1 转换路径

| 模式 | 转换方式 | 说明 |
|------|---------|------|
| **1p** | `--optimize=ascend_oriented` + configFile | 离线编译，权重外部化到 `_variables/` |
| **2p/4p** | `--optimize=none` | online GE 路径，运行时编译 |

### 5.2 产物路径

```text
# 1p
qwen2_5_7b_onnx/{prefill,decode}/*_rank0_graph.mindir + _variables/

# 2p
qwen2_5_7b_tp_onnx/{prefill,decode}/*_rank{0,1}_graph.mindir

# 4p
qwen2_5_7b_tp4_onnx/{prefill,decode}/*_rank{0..3}_graph.mindir
```

---

## 6. 推理

### 6.1 推理架构

#### 1p 单卡（零拷贝 decode）

```text
[Prefill MindIR] → first_token + KV device tensors
                          ↓ (同芯片零拷贝 swap)
[Decode MindIR] ← KV buffers（全程不过 host）
```

- KV cache 常驻 device，每步只搬 logits（~0.6MB）
- prefill→decode KV 交接：同芯片直接 swap（零拷贝）

#### 2p/4p 张量并行（多进程 + HCCL）

```text
Driver process
  ├── Worker rank0: [Prefill rank0] → KV → [Decode rank0] ←→ HcomAllReduce
  ├── Worker rank1: [Prefill rank1] → KV → [Decode rank1] ←→ HcomAllReduce
  └── (4p: rank2, rank3 同理)
```

- 每 rank 一个进程，同一 HCCL group
- 图内 Custom(AllReduce) → GE HcomAllReduce 做跨 rank 求和

### 6.2 推理命令

```bash
# 1p 单卡
bash infer.sh 0

# 2p 双芯（同卡）
bash infer.sh 0,1

# 4p 四芯（跨 2 卡）
bash infer.sh 2,3,4,5
```

`infer.sh` 自动根据设备数量选择模式，并生成 rank_table.json + config_file.ini。

---

## 7. 性能数据（KV=256，prefill 档 32/64/128）

| 配置 | 设备数 | 互联拓扑 | Prefill (ms) | Decode (ms/step) | 吞吐 (tok/s) | 精度 |
|------|--------|---------|:---:|:---:|:---:|:---:|
| **1p 单卡** | 1 芯 | — | 97 | **92** | 10.9 |
| **2p 双芯** | 2 芯 | 卡内 HCCS 60GB/s | 67 | **65** | 15.4 |
| **4p 四芯** | 4 芯 | 卡间 PCIe 16GB/s | 46 | **38** | 26.1 |

### 示例输出（2p 双芯）

```text
Input Prompt: 你好，请用一句话介绍一下你自己
Generated Response: 你好，我叫Qwen，是来自阿里云的大规模语言模型，
                  可以帮你回答问题、创作文字，还能完成各种文本生成任务。
```

---

## 8. 多卡张量并行说明

### 硬件拓扑（Atlas 300I Duo）

| 拓扑 | 带宽 | 适用 |
|------|------|------|
| 卡内 HCCS（2 芯） | ~60 GB/s | TP=2 |
| 卡间 PCIe（跨卡） | ~16 GB/s | TP=4 |
| LPDDR4X 内存 | 204 GB/s/chip | decode BW-bound |

### TP 机制

MindSpore Lite 2.10 的 TP 是**导出时分片**（非运行时拆分）：每 rank 导出独立的 MindIR，推理时图内 Custom(AllReduce) 经 GE lower 为 HcomAllReduce 做跨 rank 通信。

---

## 9. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

---

## 10. 许可证

本教程遵循 Qwen2.5-7B-Instruct 模型及相关依赖的许可证要求。
