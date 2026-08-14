# Jina Reranker V3 Listwise ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Jina Reranker V3 模型以原生 Listwise 架构导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。同时支持 Listwise 和 Pointwise 两种打分模式。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
|---|---|
| Python | 3.11.15 |
| torch | 2.10.0 |
| transformers | 4.57.0 |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 2.4.4 |
| mindspore-lite | 2.9.0 |
| CANN | 8.5 |

```bash
pip install transformers==4.57.0 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==2.4.4
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/base_models/jina_reranker_v3

python export_jina_reranker_v3_onnx.py \
  --model-id ./weights/jina-reranker-v3 \
  --output-dir ./onnx/fuse \
  --device cpu
```

说明：

- 默认导出融合 ONNX（包含 CANN Custom 融合算子），用于 MindSpore Lite/Ascend 侧的融合性能验证与部署。
- 如果要跑 ONNX Runtime 推理，必须加 `--disable-fusion-opt` 导出 non-fuse ONNX。

### 参数说明

| 参数             | 说明                        | 默认值                    |
|----------------|---------------------------|------------------------|
| `--model-id`   | HuggingFace 模型路径或本地目录     | `jinaai/jina-reranker-v3` |
| `--output-dir` | 输出目录                      | `./onnx`               |
| `--device`     | 导出设备                      | `cpu`                  |
| `--disable-fusion-opt` | 禁用融合导出，生成 non-fuse ONNX（用于 ONNX Runtime 推理） | 默认关闭（未使能） |
| `--enable_bmm2mm_fusion` | 使能 BMM->MMv2 优化：Linear 走 2D MatMulV2 Custom（仅融合导出有效） | 默认关闭（未使能） |
| `--enable_rmsnorm_fusion` | 使能 RmsNorm/AddRmsNorm 融合 Custom 算子 | 默认关闭（未使能） |
| `--enable_qk_merge` | 使能 QK merge：合并 q_proj+k_proj 为一个 Linear（仅融合导出有效） | 默认关闭（未使能） |
| `--enable_swiglu_fusion` | 使能 SwiGlu 融合 Custom 算子（cat+silu*up）。 | 默认关闭（未使能） |

### 产出

```log
onnx/fuse/
├── jina_reranker_v3.onnx
└── jina_reranker_v3.onnx.data
```

### 模型架构说明

导出的 ONNX 模型包含以下核心组件：

- **Qwen3-0.6B Backbone**：28 层 Transformer，因果自注意力机制
- **MLP Projector**：1024→512→512 维度映射（带 ReLU 激活）
- **Cosine Similarity**：计算 query 与各 document 的相关性分数

**ONNX 模型输入：**

| 输入名                  | 形状                  | 类型     | 说明                          |
|----------------------|---------------------|--------|-----------------------------|
| `input_ids`          | (batch, seq_len)    | int64  | 完整 listwise prompt 的 token IDs |
| `attention_mask`     | (batch, seq_len)    | int64  | 注意力掩码                       |
| `doc_token_indices`  | (batch, num_docs)   | int64  | `<\|embed_token\|>` 的位置索引    |
| `query_token_index`  | (batch, 1)          | int64  | `<\|rerank_token\|>` 的位置索引   |

**ONNX 模型输出：**

| 输出名      | 形状                  | 类型    | 说明              |
|----------|---------------------|-------|-----------------|
| `scores` | (batch, num_docs)   | float | 每个 document 的相关性分数 |

### Listwise vs Pointwise 模式

两种模式使用同一个 ONNX 模型，区别在于输入格式：

| 模式          | 输入格式                       | 特点               |
|-------------|----------------------------|------------------|
| **Listwise**  | 1 个 query + N 个 doc 在同一上下文窗口 | 跨文档交互，精度更高       |
| **Pointwise** | 1 个 query + 1 个 doc 逐对推理     | 无跨文档交互，内存占用更低    |

---

## 3. ONNX 推理

说明：ONNX Runtime 只能运行 non-fuse ONNX。融合 ONNX 含 `Custom` 节点，ONNX Runtime 通常无法执行。

### Listwise 模式推理

```bash
python infer_jina_reranker_v3_onnx.py \
  --model-path onnx/non_fuse/jina_reranker_v3.onnx \
  --tokenizer ./weights/jina-reranker-v3 \
  --max-length 1280 \
  --mode listwise \
  --device CPU
```

**执行日志：**

```log
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 15
Loading tokenizer from ./weights/jina-reranker-v3
Loading ONNX model from onnx/non_fuse/jina_reranker_v3.onnx

Running inference in listwise mode...

Reranking results (listwise mode):

[1] Score: 0.2986
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.2247
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.1901
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.1519
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1604
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1702
Document: Basketball is one of the most popular sports in the United States....

============================================================
Higher scores indicate better relevance to the query.
============================================================
```

### Pointwise 模式推理

```bash
python infer_jina_reranker_v3_onnx.py \
  --model-path onnx/non_fuse/jina_reranker_v3.onnx \
  --tokenizer ./weights/jina-reranker-v3 \
  --max-length 1280 \
  --mode pointwise \
  --device CPU
```

**执行日志：**

```log
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 15
Loading tokenizer from ./weights/jina-reranker-v3
Loading ONNX model from onnx/non_fuse/jina_reranker_v3.onnx

Running inference in pointwise mode...

Reranking results (pointwise mode):

[1] Score: 0.3454
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.3388
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.3152
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.2499
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1041
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1387
Document: Basketball is one of the most popular sports in the United States....

============================================================
Higher scores indicate better relevance to the query.
============================================================
```

### 参数说明

| 参数             | 说明                       | 默认值                              |
|----------------|--------------------------|----------------------------------|
| `--model-path` | ONNX 模型路径               | `./onnx/non_fuse/jina_reranker_v3.onnx` |
| `--tokenizer`  | HuggingFace tokenizer 路径 | `./weights/jina-reranker-v3`        |
| `--max-length` | 最大序列长度                  | `4096`                           |
| `--mode`       | 打分模式：listwise / pointwise | `listwise`                       |
| `--device`     | 推理设备（CPU）          | `CPU`                            |

---

## 4. MindSpore Lite 转换

### 转换命令

```bash
# 动态分档：batch=1，seq_len=256~4096（stride=128，共 31 档）
$Convert --fmk=ONNX \
  --modelFile=./onnx/fuse/jina_reranker_v3.onnx \
  --outputFile=./onnx/fuse/jina_reranker_v3 \
  --saveType=MINDIR \
  --optimize=ascend_oriented \
  --configFile=./config.ini \
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径                      |

### 配置文件

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;doc_token_indices:1,64;query_token_index:1,1"
ge.dynamicDims="256,256;384,384;512,512;640,640;768,768;896,896;1024,1024;1152,1152;1280,1280;1408,1408;1536,1536;1664,1664;1792,1792;1920,1920;2048,2048;2176,2176;2304,2304;2432,2432;2560,2560;2688,2688;2816,2816;2944,2944;3072,3072;3200,3200;3328,3328;3456,3456;3584,3584;3712,3712;3840,3840;3968,3968;4096,4096"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="./op_fp32.json"

[ascend_context]
plugin_custom_ops=BatchMatmulToMatmul
```

```json
{
    "black-list":{
        "to-add":[
            "RealDiv",
            "SquareSumV1"
        ]
    }
}
```

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
onnx/fuse
├── jina_reranker_v3_graph.mindir      # MindIR 图定义
├── jina_reranker_v3_variables/data_0   # 权重数据
└── jina_reranker_v3.onnx               # 原始 ONNX 模型
```

---

## 5. MindSpore Lite 推理

说明：

- 固定 shape MindIR：推理侧必须使用 `padding="max_length"` 并保持固定 `--max-length`，否则可能出现 shape broadcast 失败或推理结果不稳定。
- 动态分档 MindIR：推理侧必须实现“选档 + pad”，并在每次推理前执行 `model.resize(inputs, dims)`（本脚本已内置 resize）。

### Listwise 模式推理

```bash
python infer_jina_reranker_v3_mslite.py \
  --model-path onnx/fuse/jina_reranker_v3_graph.mindir \
  --tokenizer ./weights/jina-reranker-v3 \
  --max-length 1280 \
  --device ascend \
  --device-id 0 \
  --mode listwise
```

**执行日志：**

```log
WARNING:root:Ascend custom operator path not found
Loading tokenizer from ./weights/jina-reranker-v3
Initializing MindSpore Lite context for ascend...
Loading model from onnx/fuse/jina_reranker_v3_graph.mindir...

Running inference in listwise mode...

Reranking results (listwise mode):

[1] Score: 0.2984
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.2250
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.1889
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.1512
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1604
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1700
Document: Basketball is one of the most popular sports in the United States....

### MSLite 推理输入输出（本次运行）

| 项目 | 值 |
|---|---|
| mode | `listwise` |
| max_length(bucket) | `1280` |
| input_ids Shape | `(1, 1280)` |
| attention_mask Shape | `(1, 1280)` |
| doc_token_indices Shape | `(1, 64)` |
| query_token_index Shape | `(1, 1)` |
| scores Shape | `(1, 6)` |

### 端到端推理性能（本次运行）

| 指标 | 耗时 (ms) |
|---|---:|
| Tokenize + pad | 2.50 |
| Model resize | 0.37 |
| Model predict | 92.90 |
| Postprocess | 0.13 |
| **总耗时** | **95.90** |
============================================================
Higher scores indicate better relevance to the query.
============================================================
```

#### 性能数据(300I DUO)

端到端推理性能（listwise mode）

| 指标 | 耗时 (ms) |
|---|---:|
| Tokenize + pad | 2.50 |
| Model resize | 0.37 |
| Model predict | 92.90 |
| Postprocess | 0.13 |
| **总耗时** | **95.90** |

### Pointwise 模式推理

```bash
python infer_jina_reranker_v3_mslite.py \
  --model-path onnx/fuse/jina_reranker_v3_graph.mindir \
  --tokenizer ./weights/jina-reranker-v3 \
  --max-length 1280 \
  --device ascend \
  --device-id 0 \
  --mode pointwise
```

**执行日志：**

```log
WARNING:root:Ascend custom operator path not found
Loading tokenizer from ./weights/jina-reranker-v3
Initializing MindSpore Lite context for ascend...
Loading model from onnx/fuse/jina_reranker_v3_graph.mindir...

Running inference in pointwise mode...

Reranking results (pointwise mode):

[1] Score: 0.3456
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.3388
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.3151
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.2502
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1040
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1388
Document: Basketball is one of the most popular sports in the United States....

### MSLite 推理输入输出（本次运行）

| 项目 | 值 |
|---|---|
| mode | `pointwise` |
| max_length(bucket) | `1280` |
| input_ids Shape | `(1, 1280)` |
| attention_mask Shape | `(1, 1280)` |
| doc_token_indices Shape | `(1, 64)` |
| query_token_index Shape | `(1, 1)` |
| scores Shape | `(1, 1)` |

### 端到端推理性能（本次运行）

| 指标 | 耗时 (ms) |
|---|---:|
| Tokenize + pad | 9.44 |
| Model resize | 0.71 |
| Model predict | 559.39 |
| Postprocess | 0.09 |
| **总耗时** | **569.63** |
============================================================
Higher scores indicate better relevance to the query.
============================================================
```

#### 性能数据(300I DUO)

端到端推理性能（pointwise mode）

| 指标 | 耗时 (ms) |
|---|---:|
| Tokenize + pad | 9.44 |
| Model resize | 0.71 |
| Model predict | 559.39 |
| Postprocess | 0.09 |
| **总耗时** | **569.63** |

### 性能数据说明

> 注：不同设备、不同 CANN 版本、不同 doc 长度分布都会影响性能。本表用于提供可复现实验入口与对比口径。
>
> 注：Tokenize + pad 在 CPU 上执行，受 CPU 负载影响波动较大；Pointwise 模式需对 N 个 prompt 一次性 batch tokenize + pad，数据量是 Listwise 的 N 倍，因此该项耗时显著高于 Listwise。
>
> Listwise 与 pointwise 的差异：
> - Listwise：把 query + N 篇 doc 按顺序写进同一条输入序列里（`batch=1, seq=max_length`），一次前向输出 N 个分数（shape 为 `(1, N)`）。实现上：每篇 doc 末尾插入 `<|embed_token|>`，并用 `doc_token_indices` 记录这些 token 在序列中的位置；推理时从 `hidden_states` 里取出每篇 doc 的 embedding，再与 query embedding 计算 cosine 得分。
> - Pointwise：每次只输入 query + 1 篇 doc（输出 shape 为 `(1, 1)`），要对 N 篇 doc 打分就需要跑 N 次前向，因此总耗时通常近似 N 倍（再叠加少量调度/框架开销）。
>
> `max_length` 说明：
> - 本 README 的对比用例为了口径一致，listwise/pointwise 都使用 `--max-length=1280`（1280 只是一个示例 bucket）。
> - pointwise 通常可以选比listwise更短的 `max_length` 来提速：保证能容纳 query+doc，并把输入向上 pad 到 128 对齐的 bucket（例如 512/768/1024/1280）即可。

### 参数说明

| 参数             | 说明                                    | 默认值                     |
|----------------|---------------------------------------|-------------------------|
| `--model-path` | MindIR 模型路径（`*_graph.mindir`）         | 必填                      |
| `--tokenizer`  | HuggingFace tokenizer 路径              | `./weights/jina-reranker-v3` |
| `--max-length` | 最大序列长度                                | `4096`                  |
| `--device`     | 推理设备（ascend/cpu）                      | `cpu`                   |
| `--device-id`  | Ascend 设备 ID                          | `0`                     |
| `--mode`       | 打分模式：listwise / pointwise             | `listwise`              |

### 分块策略说明

当文档数量较多或总长度超过上下文窗口时，Listwise 模式自动执行分块处理：

1. 将文档按 token 长度分块，每块不超过 `max_length - 2 * query_length`
2. 每块独立推理，获取该块内各文档的分数
3. 以各块最高分数的归一化值作为权重，用于跨块分数归一化
4. 合并所有块的分数，按降序排列返回结果

---

## 6. 常见问题

### Q1: ONNX 导出时报 `IndexError: tuple index out of range`

**现象：** 导出时 `masking_utils.sdpa_mask` 报错

**原因：** Transformers v4.50+ 的 Qwen3 模型在 JIT tracing 时，`_preprocess_mask_arguments` 返回的 `q_length` 变为 0 维 Tensor，导致向后兼容检查失败

**解决方案：** 导出脚本已内置 monkey-patch，自动将 0 维 Tensor 转为 int。如仍有问题，优先对齐本文档的 transformers/torch 版本，并用 plog 获取更完整报错（见 Q4）。

### Q2: MSLite 推理结果恒为常数或全零

**现象：** ONNX 推理正常，但 MSLite 输出恒为常数（如 0.5）或全零

**原因：** FP16 精度下 attention mask 的极小值溢出

**解决方案：** 当前 `config.ini` 采用 `allow_mix_precision` + `op_fp32.json` mixlist 方案，把 `RealDiv`、`SquareSumV1` 等 RmsNorm 链路上的关键算子锁定 FP32，在保持精度的同时获得更优性能（详见第 8 节优化记录）。

### Q3: 动态分档不会自动 pad 命中档位

**现象：** 输入 `seq_len=1201`，期望自动 pad 到 1280 并命中 1280 档位

**原因：** `ge.dynamicDims` 只是“允许的 shape 列表”，不是自动 shape 变换器。是否 pad、如何 pad 属于业务语义（pad_token、left/right pad、attention_mask 语义），必须由推理侧实现。

**解决方案：** 业务侧或推理脚本实现“选档 + pad”，并保证传入的 shape 精确等于某个档位（例如 1280/2048/…/4096）。动态分档 MindIR 每次推理前需 `model.resize(inputs, dims)`（本脚本已内置）。

### Q4: 转换/推理失败但终端报错不全

**现象：** 终端只看到类似 `"[ERROR] FE(1087713,converter_lite):..."` 或 `"[ERROR] LITE(1087713,...,converter_lite):..."`，缺少更详细的根因信息

**解决方案：** 从错误行中提取进程号（如 `1087713`），查看 `~/ascend/log/debug/plog/plog-1087713` 获取更完整的 plog 以定位根因。

### Q5: 输入 dtype 不匹配

**现象：** 报错 `required 34, given 35`

**原因：** MindIR 期望 int32，传入了 int64

**解决方案：** MSLite 推理脚本中已自动转换 dtype 为 int32

### Q6: Listwise 和 Pointwise 分数差异

**现象：** 两种模式的分数数值不同

**原因：** 这是正常行为。Listwise 模式中文档之间通过 causal attention 交互，分数反映了文档间的相对关系；Pointwise 模式各文档独立评分，无跨文档交互。推荐使用 Listwise 模式以获得更准确的排序

### Q7: 文档数量超过 64 个

**现象：** 超过 MAX_DOCS 限制

**解决方案：** 模型自动执行分块处理，每块最多 64 个文档，通过加权融合得到最终排序

### Q8: MindSpore Lite 转换或 ONNX Runtime 推理报 FLOAT16 类型错误

**现象：** converter_lite 报错 `do not support data_type: 10`，或 ONNX Runtime 报错 `Type Error: Type (tensor(float16)) of output arg does not match expected type (tensor(float))`

**原因：** 模型以 float16 加载导出时，ONNX 图的类型声明和初始值均为 FLOAT16，MindSpore Lite 的 Clip 等解析器不支持 FLOAT16；仅转换初始值 dtype 不够，图中节点的输入/输出类型声明也需同步修改

**解决方案：** 导出脚本已改为以 `torch_dtype=torch.float32` 加载模型，从源头确保 ONNX 全图为 float32。请确保使用最新版本的导出脚本重新导出 ONNX 模型

### Q9: ONNX 推理时报 `INVALID_PROTOBUF: Protobuf parsing failed`

**现象：** ONNX 模型导出成功，但 ONNX Runtime 加载时报错 `Protobuf parsing failed`

**原因：** 大模型 ONNX 使用外部数据文件（`.onnx.data`），优化后 `onnx.save` 未正确更新外部数据文件，导致 `.onnx` 与 `.onnx.data` 不匹配

**解决方案：** 导出脚本已修复此问题：`onnx.load` 时加 `load_external_data=True` 加载完整数据，保存时删除旧 `.onnx.data` 并使用 `onnx.save_model` 正确写入外部数据文件。请确保使用最新版本的导出脚本重新导出

### Q10: MSLite Pointwise 模式推理结果与 ONNX 不一致

**现象：** MSLite Listwise 模式结果与 ONNX 一致，但 Pointwise 模式部分文档分数严重偏移（如无关文档得分异常高）

**原因：** MSLite 使用 `--optimize=ascend_oriented` 转换时，GE 会针对固定输入 shape 编译优化图。Pointwise 模式下每个文档单独 tokenize 导致序列长度不同，GE 无法正确适配变长输入，部分推理结果错误

**解决方案：** 推理脚本已修改为将所有文档一起 tokenize 并 padding 至等长，保证每次推理的输入 shape 一致。与原始 Pointwise 版本（`jina_reranker_v3`）的处理方式保持一致

---

## 7. PFA 融合问题定位与修复总结（已验证跑通）

本节总结PromptFlashAttention融合过程中遇到的关键问题、触发条件与最终修复方案。

### 7.1 现象

融合导出的 listwise 模型（`seq_len=1201`）可导出、可转换，但 Benchmark 阶段会进入 `aclnnPromptFlashAttentionV3` 并报错，典型报错包括：

- `attention mask must be NULL, when Qs,Kvs is unAlign ... Qs = 1201, Kvs = 1201`
- 或在错误映射场景下：`attenMask should not be null when sparse_mode is 4`

### 7.2 触发条件（归因结论）

结论：是否进入 `aclnnPromptFlashAttentionV3` 主要由 **Custom(PromptFlashAttention) 的输入形态 + 序列长度对齐约束** 共同决定，而不是 fp16/fp32。

- **3-input（query/key/value）形态**：通常走 aclop，Benchmark 可跑通。
- **4-input（query/key/value + atten_mask）形态**：更容易进入 V3 分支；当 `seq_len` 非 128 对齐（如 1201）时，后端对 `atten_mask` 有约束，会触发 “mask 必须为 NULL” 类报错。
- **4-input 也可跑通**：当 `seq_len` 满足对齐约束（例如 128）时，带 `atten_mask` 的 PFA 可 Convert + Benchmark 成功。
- **额外坑点（融合导出实现相关）**：原有融合实现里 `input_index=[0,1,2,4]` 会导致 `atten_mask` 映射丢失，并与 `sparse_mode=4` 的 mask 必选约束发生冲突，从而触发 `attenMask should not be null when sparse_mode is 4`。

约束依据：

- PFA 约束说明（摘录）：Atlas 推理系列加速卡上，`Q_S` 或 `KV_S` 非 128 对齐时不支持配置 `atten_mask`。

### 7.3 最终修复方案（仅重写 PFA 融合）

目标：保持模型外部 `seq_len=1201` 不变，在 PFA 融合内部做 128 对齐 padding，使得可以合法传入 `atten_mask` 并跑通 Benchmark。

做法：

- 在 PFA 融合前，对 `query/key/value` 的 `S` 维进行 padding：`padded_len = ceil(seq_len/128)*128`（1201→1280）。
- 基于 `attention_mask` 构造 padded 的 bool causal+padding mask（shape `B,1,padded_len,padded_len`），保证 padding token 与 padding query 行都被 mask 掉。
- PFA 输出后 slice 回原始 `seq_len`，再继续后续算子。
- PFA Custom 的导出属性采用最小集合（对齐单算子可跑通的形式），避免 `input_index/sparse_mode/optional_input_names/num_key_value_heads` 等配置导致映射/约束冲突。

### 7.5 动态分档验证（batch=1，seq_len=256~4096，stride=128，共 31 档）

关键点：

- 动态分档不自动 pad：业务/推理侧必须实现“选档 + pad”。
- 动态分档 MindIR 每次推理前应 `model.resize(inputs, dims)`（本仓脚本已内置）。
- 精度建议对齐 ORT(non-fuse ONNX)，并确保同一 `max_length`（例如 1280 或 2048）。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Jina Reranker V3 官方文档](https://huggingface.co/jinaai/jina-reranker-v3)
- [Jina Reranker V3 技术报告](https://arxiv.org/abs/2509.25085)

---

## 9. 许可证

本教程遵循 Apache 2.0 许可证。模型权重遵循 CC BY-NC 4.0 许可证。
