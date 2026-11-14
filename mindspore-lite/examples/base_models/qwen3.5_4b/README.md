# Qwen3.5-4B ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 Qwen3.5-4B 导出为 ONNX，转换为 MindSpore Lite MindIR，并在
Atlas 300I Duo 上完成 1p 与 TP=2 推理部署。

## 模型架构

Qwen3.5-4B 是图文多模态模型，语言部分采用 GatedDeltaNet 线性注意力 + 全注意力混合架构。

| 参数 | 值 |
|------|-----|
| hidden_size | 2560 |
| num_hidden_layers | 32（24 linear_attention + 8 full_attention） |
| num_attention_heads / num_key_value_heads | 16 / 4 |
| head_dim | 256 |
| linear_num_key_heads / linear_num_value_heads | 16 / 32 |
| linear_key_head_dim / linear_value_head_dim | 128 / 128 |
| linear_conv_kernel_dim | 4 |
| intermediate_size | 9216 |
| vocab_size | 248320 |
| image_token_id | 248056 |
| vision patch_size / spatial_merge_size | 16 / 2 |

模型拆分为三个 ONNX：

1. **Vision Tower**（`qwen3_5_vision.onnx`）：输出 `image_embeds`
2. **LLM Prefill**（`qwen3_5_llm_prefill_rank*.onnx`）：输出 logits、conv/recurrent state 与 KV cache
3. **LLM Decode**（`qwen3_5_llm_decode_rank*.onnx`）：自回归单步生成

Vision 为独立模型，不参与 TP 切分；无图像输入时不导出、不加载 Vision。

### TP 切分策略

- 全注意力 q/k/v：列并行，每 rank 8 q head、2 kv head；o_proj 行并行 + AllReduce
- GDN in_proj_qkv/z/b/a：按 head 列并行；conv 按通道切分；out_proj 行并行 + AllReduce
- MLP gate/up：列并行；down：行并行 + AllReduce
- Embedding / norms / lm_head：复制

> `q_proj` 的行布局是每 head 的 `[q, gate]` 交错，按连续 8 个 head（4096 行）切分。

## 示例文件

| 文件 | 作用 |
|------|------|
| `export_qwen3_5_4b_onnx.py` | 1p 导出（Vision / Prefill / Decode，支持 `--no-vision`） |
| `export_qwen3_5_4b_tp_onnx.py` | TP=2 导出（Prefill + Decode，双 rank） |
| `infer_qwen3_5_4b_mslite.py` | 1p 推理（支持 `--no-image`） |
| `infer_qwen3_5_4b_tp_mslite.py` | TP=2 推理（双进程 HCCL） |
| `configs/config.ini` | 1p 转换配置（Vision / 静态 Prefill / Decode） |
| `configs/config_prefill_img16.ini` | 无图 / 128px，seq 256~4096 动态档 |
| `configs/config_prefill_img256.ini` | 512px（256 image token） |

常用开关：

| 开关 | 默认 | 说明 |
|------|------|------|
| `--enable-cgdr-custom` | off | Prefill 使用 `ChunkGatedDeltaRule` 自定义算子 |
| `--enable-rgdr-custom` | off | Decode 使用 `RecurrentGatedDeltaRule` + 固定容量 KV |
| `--no-vision` / `--no-image` | off | 无图部署 |
| `--static-prefill` | off | 固定 seq Prefill 图（图像场景使用） |

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
|--------|------|
| Python | 3.11 |
| torch | 2.9 / 2.10（CPU 导出） |
| transformers | ≥5.6（脚本兼容 5.14 的 `block_type`） |
| onnx / onnxruntime | 1.19+ / 1.24+ |
| Pillow / numpy | 12.x / 1.26 |
| CANN | 9.0.1 |
| MindSpore Lite | 2.11.0（源码构建） |
| Ascend 硬件 | Atlas 300I Duo（单卡双芯） |

```bash
source /path/to/CANN/set_env.sh
export MSLITE_HOME_PATH=/path/to/mindspore-lite-2.11.0-linux-aarch64
export PATH="$MSLITE_HOME_PATH/tools/converter/converter:$PATH"
export LD_LIBRARY_PATH="$MSLITE_HOME_PATH/tools/converter/lib:$MSLITE_HOME_PATH/runtime/lib:$LD_LIBRARY_PATH"
```

### Custom 算子

```bash
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_OPP_PATH/vendors/mslite_custom_ops"
export MSLITE_OP_TILING="$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$MSLITE_OP_TILING/lib/linux/aarch64:$MSLITE_OP_TILING:$LD_LIBRARY_PATH"
```

自定义算子源码位于 `tools/custom_kernels/ascend_ops/`，可用
`build_all_ops.sh <output_dir>` 构建。修改 kernel 后需清理
`~/atc_data/kernel_cache/` 下与算子相关的 `te_*gateddeltarule*` 缓存，并重新执行转换。

### 权重获取

```bash
git clone https://www.modelscope.cn/Qwen/Qwen3.5-4B.git ./Qwen3.5-4B
```

## 2. 模型导出 ONNX

### 2.1 1p

无图部署：

```bash
python export_qwen3_5_4b_onnx.py \
  --model-id ./Qwen3.5-4B --output-dir ./qwen3_5_4b_onnx \
  --device cpu --dummy-seq-len 32 --max-seq-len 4608 \
  --enable-cgdr-custom --enable-rgdr-custom --no-vision
```

有图部署（固定 seq Prefill）：

```bash
python export_qwen3_5_4b_onnx.py \
  --model-id ./Qwen3.5-4B --output-dir ./qwen3_5_4b_img512_onnx \
  --device cpu --vision-image-size 512 --dummy-seq-len 512 \
  --max-seq-len 4608 --static-prefill \
  --enable-cgdr-custom --enable-rgdr-custom
```

- 256px 图像使用 `--vision-image-size 256 --dummy-seq-len 256`。
- `--vision-image-size` 支持 128/256/512，image token 数为 `(size/32)^2`。
- `--component all|vision|prefill|decode` 可单独导出。

### 2.2 TP=2

```bash
# 无图
python export_qwen3_5_4b_tp_onnx.py \
  --model-id ./Qwen3.5-4B --output-dir ./qwen3_5_4b_tp_onnx \
  --component all --no-vision --image-tokens 16 --dynamic-seq --max-seq-len 4608

# 有图（image_tokens：256px→64，512px→256）
python export_qwen3_5_4b_tp_onnx.py \
  --model-id ./Qwen3.5-4B --output-dir ./qwen3_5_4b_tp512_onnx \
  --component all --vision-image-size 512 --image-tokens 256 \
  --dynamic-seq --max-seq-len 4608
```

每个 rank 输出独立目录：

```text
qwen3_5_4b_tp_onnx/
├── rank0/{prefill,decode}/qwen3_5_llm_*_rank0.onnx
└── rank1/{prefill,decode}/qwen3_5_llm_*_rank1.onnx
```

## 3. ONNX 转 MindIR

### 3.1 1p

```bash
Convert="$MSLITE_HOME_PATH/tools/converter/converter/converter_lite"

$Convert --fmk=ONNX --modelFile=qwen3_5_4b_onnx/vision/qwen3_5_vision.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_vision \
  --optimize=ascend_oriented --configFile=configs/config.ini --saveType=MINDIR

$Convert --fmk=ONNX --modelFile=qwen3_5_4b_onnx/prefill/qwen3_5_llm_prefill.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_llm_prefill \
  --optimize=ascend_oriented --configFile=configs/config_prefill_img16.ini --saveType=MINDIR

# Prefill（图像 / 固定 seq）
$Convert --fmk=ONNX --modelFile=qwen3_5_4b_img512_onnx/prefill/qwen3_5_llm_prefill.onnx \
  --outputFile=qwen3_5_4b_img512_mindir/qwen3_5_llm_prefill \
  --optimize=ascend_oriented --configFile=configs/config.ini --saveType=MINDIR

$Convert --fmk=ONNX --modelFile=qwen3_5_4b_onnx/decode/qwen3_5_llm_decode.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_llm_decode \
  --optimize=ascend_oriented --device=Ascend --configFile=configs/config.ini --saveType=MINDIR
```

- 无图 Prefill 使用 `config_prefill_img16.ini`（seq 256/512/1024/2048/4096 动态档）。
- 图像静态 Prefill 使用 `configs/config.ini`（无 `input_shape` / `dynamicDims`），seq 与导出时
  `--dummy-seq-len` 一致。
- `config_prefill_img256.ini` 为 512px 动态档位配置，供需要动态 seq 的图像导出使用。
- 图像场景的导出/转换目录为 `qwen3_5_4b_img512_onnx` / `qwen3_5_4b_img512_mindir`；
  256px 场景改用对应的 `--vision-image-size 256 --dummy-seq-len 256` 与自定义目录。

### 3.2 TP=2

```bash
for r in 0 1; do
  $Convert --fmk=ONNX \
    --modelFile=qwen3_5_4b_tp_onnx/rank$r/prefill/qwen3_5_llm_prefill_rank$r.onnx \
    --outputFile=qwen3_5_4b_tp_mindir/rank$r/qwen3_5_llm_prefill_rank$r \
    --optimize=none --saveType=MINDIR
  $Convert --fmk=ONNX \
    --modelFile=qwen3_5_4b_tp_onnx/rank$r/decode/qwen3_5_llm_decode_rank$r.onnx \
    --outputFile=qwen3_5_4b_tp_mindir/rank$r/qwen3_5_llm_decode_rank$r \
    --optimize=none --saveType=MINDIR
done
```

TP 使用 `optimize=none`，运行时由 `provider=ge` + HCCL 编译，`rank_table.json` 与
`ge.dynamicDims` 由推理脚本生成。

## 4. 推理

### 4.1 1p

有图：

```bash
python infer_qwen3_5_4b_mslite.py \
  --vision-model qwen3_5_4b_img512_mindir/qwen3_5_vision.mindir \
  --prefill-model qwen3_5_4b_img512_mindir/qwen3_5_llm_prefill_graph.mindir \
  --decode-model qwen3_5_4b_img512_mindir/qwen3_5_llm_decode_graph.mindir \
  --processor ./Qwen3.5-4B --image ./demo.jpeg --image-size 512 \
  --prompt "Describe this image." --max-new-tokens 128 --device ascend --device-id 0
```

> 该示例使用 seq=512 的静态图像 Prefill；若“文本 + 图像 token”的实际总长度不是 512，需按
> 6.3 节 pad 到 512，或导出对应 seq 的静态 Prefill 模型。

无图：

```bash
python infer_qwen3_5_4b_mslite.py \
  --prefill-model qwen3_5_4b_mindir/qwen3_5_llm_prefill_graph.mindir \
  --decode-model qwen3_5_4b_mindir/qwen3_5_llm_decode_graph.mindir \
  --processor ./Qwen3.5-4B --prompt "你好，介绍一下你自己。" \
  --max-new-tokens 128 --no-image --device ascend --device-id 0
```

### 4.2 TP=2

```bash
python infer_qwen3_5_4b_tp_mslite.py \
  --prefill-dir qwen3_5_4b_tp_mindir --decode-dir qwen3_5_4b_tp_mindir \
  --processor ./Qwen3.5-4B --no-image --prompt "你好，介绍一下你自己。" \
  --max-new-tokens 32 --device-ids 0,1
```

有图时增加 `--vision-model`、`--image`、`--image-size`，并使用对应 image token 的 TP 模型。
Vision 仅在 rank0 执行，image embeds 通过 Host 分发给两个 rank。两个 rank 必须同步执行
`predict`（AllReduce 在图中）。

## 5. 性能数据

测试环境：Atlas 300I Duo，CANN 9.0.1，MindSpore Lite 2.11.0；1p 单芯，TP=2 使用
同卡 device 0/1；每次运行不复用模型缓存。

### 5.1 Prefill（无图，动态档位）

| seq | 1p | TP=2 |
|---:|---:|---:|
| 256 | 198.1 ms | 112.6 ms |
| 512 | 361.8 ms | 222.8 ms |
| 1024 | 713.6 ms | 442.0 ms |
| 2048 | 1484.6 ms | 943.4 ms |
| 4096 | 3442.1 ms | 2156.0 ms |

### 5.2 Decode（cap=4608，单步）

| 上下文 | 1p | TP=2 |
|---:|---:|---:|
| 256 | 115.8 ms | 56.8 ms |
| 512 | 115.8 ms | 56.9 ms |
| 1024 | 116.5 ms | 57.2 ms |
| 2048 | 127.5 ms | 57.7 ms |
| 4096 | 115.1 ms | 58.3 ms |

### 5.3 图像场景

| 图像尺寸 | image token | Vision | Prefill 1p | Prefill TP=2 | Decode 上下文 | Decode 1p | Decode TP=2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 64 | 26.0 ms | 201.0 ms | 117.7 ms | 80 | 118.1 ms | 56.8 ms |
| 512×512 | 256 | 170.9 ms | 363.7 ms | 223.3 ms | 272 | 118.1 ms | 56.8 ms |

## 6. 限制与注意事项

### 6.1 固定 / 动态 shape 总览

| 部分 | Shape 行为 | 说明 |
|---|---|---|
| Vision | **固定** | 按 `--vision-image-size` 固定分辨率和 grid；输出 image token 数固定（128→16、256→64、512→256）；换图像尺寸需重新导出 Vision |
| 1p 无图 Prefill | **动态 seq** | `configs/config_prefill_img16.ini`，seq 256/512/1024/2048/4096 分档 |
| 1p 图像 Prefill | **固定 seq** | `--static-prefill`，seq 由 `--dummy-seq-len` 固定；image token 数由 Vision 尺寸固定 |
| TP Prefill | **动态 seq + 固定 image token 数** | seq 为 5 档；image token 数由 `--image-tokens` 固定 |
| Decode | 容量固定 | step=1；KV 容量由 `--max-seq-len` 固定（默认 4608） |

### 6.2 文本 + 图像混合输入

- 文本 token 与图像占位 token 在**同一条序列**中，Prefill 一次前向处理整条序列；总长度 =
  文本 token 数 + 图像 token 数（含模板 token）。
- 图像占位 token 数必须与 `image_embeds` 行数一致（16/64/256 分别对应 128/256/512 图像尺寸）。
- 位置编码为 4D mRoPE（`position_ids` 形状 `[4, 1, seq_len]`），图像 token 使用对应 grid 的
  t/h/w 位置。
- 仅支持单图；多图需要 Vision、processor 与导出脚本额外适配，当前未验证。

### 6.3 1p 图像 Prefill 的固定 seq

- 300I Duo GE 对“动态 seq + 动态 `image_embeds` + 图内 image mask/merge”的 1p Prefill 图存在
  `Batch_4 topological sort failed` 缺陷，因此 1p 图像 Prefill 导出为固定 seq 静态图
  （`--static-prefill`）。
- 文本长度可以在不超过固定 seq 的范围内变化，但“文本 + 图像 token”的总长度必须 pad 到该
  固定 seq；否则运行时 `Model.resize` 会失败。
- 当前 1p 推理脚本不会自动把输入 padding 到静态 Prefill 的 seq：需要按模型固定 seq 补齐
  （例如在最后一个 token 前插入填充 token），或按实际 prompt 总长度导出对应 seq 的静态图。
- TP 的 Prefill 为动态 seq 图，按 256/512/1024/2048/4096 选档后同样需要把总长度 pad 到档位。
- 要彻底消除该限制，可把图像 embedding 合并移到 Host（`inputs_embeds` 输入），Prefill 只保留
  按 seq 的动态分档。

### 6.4 无图部署

- 使用 `--no-vision`（导出）/ `--no-image`（推理）时不导出、不加载 Vision。
- Prefill 图仍保留 `image_embeds` 输入，推理时喂 16 行哑数据；`input_ids` 中不含 image token
  时该输入不参与计算。
- 不支持传入空 `(0, 2560)` 的 `image_embeds`，会在 gear/resize 校验时报错。

### 6.5 其他

1. Vision 不参与 TP 切分，TP 时仅在 rank0 执行一次，image embeds 通过 Host 分发给两个 rank。
2. TP 固定 batch=1；Decode 为单步静态图，prompt 长度必须小于 `--max-seq-len`（默认 4608）。
3. 图像尺寸支持 128/256/512；1024 分辨率未验证，对应配置已移除。
4. 调试期间不使用模型缓存：删除 `model_build_cache_*` / `kernel_meta*`，每次新建
   `ge.graph_compiler_cache_dir`；首次运行包含 GE 在线编译时间。
5. 修改自定义算子 kernel 后需按第 1 节说明清理 kernel cache 并重新执行 converter。

## 7. 参考

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3.5-4B 官方文档](https://huggingface.co/Qwen/Qwen3.5-4B)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
