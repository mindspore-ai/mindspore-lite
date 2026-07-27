# Qwen3-8B ONNX 模型导出与 MindSpore Lite 多卡推理部署教程

本教程介绍如何将 `Qwen3-8B` 模型导出为 ONNX、转换为 MindSpore Lite MindIR，并完成**单卡（1p）/ 双卡张量并行（TP=2）/ 四卡张量并行（TP=4）**端到端推理。prefill 与 decode 两个子图均转换为 MindIR 在 Ascend 上推理。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- 昇腾环境（需安装 MindSpore Lite 与 Ascend 驱动 / CANN）

### 依赖版本（建议）

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.8.0（仅导出 ONNX 时需要，推理不再使用） |
| transformers   | 4.51.0 |
| accelerate     | ≥ 0.26 |
| onnx           | 1.21.0 |
| onnxruntime    | 1.24.0 |
| mindspore-lite | 2.10.0 |
| modelscope     | ≥ 1.20（自动下载权重） |
| CANN           | 9.0.1 |

### 安装命令

```bash
pip install torch==2.8.0 transformers==4.51.0 accelerate onnx==1.21.0 onnxruntime==1.24.0
```

---

## 2. 模型架构参数

| 参数 | 值 |
|------|------|
| hidden_size | 4096 |
| num_attention_heads | 32 |
| num_hidden_layers | 36 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 12288 |
| max_position_embeddings | 40960 |
| vocab_size | 151936 |
| rms_norm_eps | 1e-6 |
| 架构 | Qwen3ForCausalLM |

### 各并行度每卡 KV 头数

| 并行度 | 每卡 attn 头 | 每卡 KV 头 | KV cache shape（每卡） |
|--------|-------------|-----------|----------------------|
| 1p     | 32 | 8 | `36,1,8,256,128` |
| 2p     | 16 | 4 | `36,1,4,256,128` |
| 4p     | 8  | 2 | `36,1,2,256,128` |

> 每卡 KV 头数 ≥ 2，原生 GQA 全程可用。

---

## 3. 一键导出 + 转换（推荐）

`export_and_convert.sh` 会导出 ONNX、转换为 MindIR：

```bash
cd ./mindspore-lite/examples/base_models/qwen3_8b

bash export_and_convert.sh 1p    # 单卡（dynamic dims, ascend_oriented）
bash export_and_convert.sh 2p    # TP=2（static, optimize=none, HCCL）
bash export_and_convert.sh 4p    # TP=4（static, optimize=none, HCCL）
```

### 产物路径

| 并行度 | 产物目录 | MindIR 文件 |
|--------|---------|------------|
| 1p | `qwen3_8b_onnx/` | `prefill/qwen3_8b_llm_prefill_rank0_graph.mindir`、`decode/qwen3_8b_llm_decode_rank0_graph.mindir` |
| 2p | `qwen3_8b_tp_onnx/` | 各 `*_rank{0,1}_graph.mindir` |
| 4p | `qwen3_8b_tp4_onnx/` | 各 `*_rank{0..3}_graph.mindir` |

---

## 4. 模型导出 ONNX（脚本说明）

导出脚本 `export_qwen3_8b_onnx.py` 是统一脚本，按 `--tp-size` 切分：

1. **LLM Prefill**：处理输入 prompt，输出 `logits` + decode 兼容的 KV cache。
2. **LLM Decode**：单 token 递归生成，输入 `past_key_cache`/`past_value_cache`，输出更新后的 cache。

### 张量并行（TP）切分策略

- **QKV投影**：列并行，每卡持 `num_heads/TP` 个 q 头、`num_kv_heads/TP` 个 kv 头。
- **o_proj / down_proj / lm_head**：行并行，每卡计算局部结果后插入 `Custom(AllReduce)` 做跨卡求和。
- **q_norm / k_norm**（按头 RMSNorm）：按头轴切分到每卡。
- **prefill 输出的 KV cache** 直接是 decode 期望的布局（heads-first、按卡已切分、zero-pad 到 `KV_CACHE_LEN=256`），无需主机侧重构。

### 导出命令（手动）

```bash
# 1p：单卡，dynamic axes（配合 ascend_oriented + dynamic dims 转换）
python export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B --output-dir ./qwen3_8b_onnx \
  --device cpu --dtype fp16 --tp-size 1

# 2p / 4p：每卡导出一个 shard（static，配合 optimize=none 在线转换）
python export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B --output-dir ./qwen3_8b_tp4_onnx \
  --device cpu --dtype fp16 --tp-size 4
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或本地目录 | `./Qwen3-8B` |
| `--output-dir` | 导出输出目录 | `./qwen3_8b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dtype` | 导出精度（fp16/fp32/bf16），**推荐 fp16** | `fp16` |
| `--tp-size` | 张量并行度（1/2/4） | `1` |
| `--num-layers` | >0 时切片到 N 层（调试，0=全部 36 层） | `0` |
| `--use-dynamo` | 启用新 ONNX dynamo 导出路径 | `False` |

---

## 5. MindSpore Lite 转换

### 单卡（1p）：`--optimize=ascend_oriented` + 动态分档

```bash
# Prefill（动态 seq 32/64/128；seq>128 需 --kv-cache-len 重导出，见 REPORT.md §3.3）
converter_lite --fmk=ONNX \
  --modelFile=./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_rank0.onnx \
  --outputFile=./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_rank0 \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/qwen3_8b_llm_prefill.config

# Decode（固定 shape）
converter_lite --fmk=ONNX \
  --modelFile=./qwen3_8b_onnx/decode/qwen3_8b_llm_decode_rank0.onnx \
  --outputFile=./qwen3_8b_onnx/decode/qwen3_8b_llm_decode_rank0 \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/qwen3_8b_llm_decode.config
```

### 多卡（2p/4p）：`--optimize=none`（在线 GE，provider=ge + HCCL）

```bash
for SUB in prefill decode; do
  for R in 0 1 2 3; do   # TP=4；TP=2 则 0 1
    converter_lite --fmk=ONNX \
      --modelFile=./qwen3_8b_tp4_onnx/$SUB/qwen3_8b_llm_${SUB}_rank${R}.onnx \
      --outputFile=./qwen3_8b_tp4_onnx/$SUB/qwen3_8b_llm_${SUB}_rank${R} \
      --optimize=none --saveType=MINDIR
  done
done
```

### config 文件示例

`./configs/qwen3_8b_llm_prefill.config`：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1"
ge.dynamicDims="32,32,32;64,64,64;128,128,128"

[acl_init_options]
ge.exec.precision_mode=must_keep_origin_dtype

[ascend_context]
plugin_custom_ops=All
```

`./configs/qwen3_8b_llm_decode.config`：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,256;position_ids:1,1;past_key_cache:36,1,8,256,128;past_value_cache:36,1,8,256,128"

[acl_init_options]
ge.exec.precision_mode=must_keep_origin_dtype

[ascend_context]
plugin_custom_ops=All
```

---

## 6. MindSpore Lite 推理

### 一键启动（推荐）

`infer.sh` 按设备 ID 数量自动选择路径（内部调用统一的 `infer_qwen3_8b_mslite.py`，由脚本按设备数派发 1p/2p/4p 并自动解析模型路径、生成 HCCL rank-table）：

```bash
cd ./mindspore-lite/examples/base_models/qwen3_8b

bash infer.sh 2            # 单卡（device 2，zero-copy decode）
bash infer.sh 2,3          # TP=2（devices 2,3，单卡双芯 HCCS）
bash infer.sh 2,3,4,5      # TP=4（devices 2,3,4,5，两卡四芯）
```

### 统一推理脚本 `infer_qwen3_8b_mslite.py`

一个脚本同时覆盖 1p / 2p / 4p，按 `--device-ids` 的数量派发：

- **1 个设备**：单卡 zero-copy decode（KV cache 常驻 Ascend、免拷贝，仅 logits 每步 D2H；prefill 输出的 KV 直接 swap 进 decode 输入 buffer）。
- **2 / 4 个设备**：张量并行多进程（每卡一个 worker，统一加载 prefill-rank + decode-rank MindIR，同一 HCCL group 贯穿始终；driver 进程编排）。

模型路径按设备数自动从 `qwen3_8b_onnx` / `qwen3_8b_tp_onnx` / `qwen3_8b_tp4_onnx` 解析，HCCL `rank_table.json` + `config_file.ini` 自动生成到 `./tp_run/`，无需手工准备。

```bash
# 单卡（1p）
python infer_qwen3_8b_mslite.py \
  --device-ids 0 \
  --model-id ./Qwen3-8B \
  --prompt "你好，请用一句话介绍一下你自己" \
  --max-new-tokens 128

# 四卡（TP=4）
python infer_qwen3_8b_mslite.py \
  --device-ids 2,3,4,5 \
  --model-id ./Qwen3-8B \
  --prompt "你好，请用一句话介绍一下你自己" \
  --max-new-tokens 64 \
  --warmup 3
```

---

## 7. 性能数据与优化

| 拓扑 | 精度模式 | Prefill | Decode(ms/step) | 吞吐 |
|------|---------|---------|--------|------|
| **1p** | `must_keep_origin_dtype` | **132.6 ms** | **93.5 ms** | **10.7 tok/s** |
| **2p** 单卡(dev0,1) | `enforce_fp32`（Context） | 127 ms | 64 ms | 15.6 tok/s |
| **4p** 跨卡(dev0-3) | `enforce_fp32`（Context） | 80 ms | 46 ms | 21 tok/s |

> 完整性能/精度分析（含 prefill 计算受限缩放规律、decode 带宽效率、4p 精度 8 步排查链、优化空间）见 **[REPORT.md](REPORT.md)**。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-8B（ModelScope）](https://www.modelscope.cn/models/Qwen/Qwen3-8B)
- [Qwen3-8B（HuggingFace）](https://huggingface.co/Qwen/Qwen3-8B)

---

## 9. 许可证

本教程遵循 Qwen3-8B 模型及相关依赖的许可证要求。
