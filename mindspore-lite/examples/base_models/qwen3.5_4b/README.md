# Qwen3.5-4B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3.5-4B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Qwen3.5-4B 是一个同时处理图像与文本的多模态大模型，采用混合线性注意力（GatedDeltaNet）与全注意力架构。模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_5_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`qwen3_5_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 logits、conv_state、recurrent_state 与 KV cache
3. **LLM Decode**（`qwen3_5_llm_decode.onnx`）：基于 conv_state + recurrent_state + KV cache 做自回归增量生成

## 模型架构

Qwen3.5-4B 采用混合注意力架构：

- **线性注意力层**（GatedDeltaNet）：使用 conv_state + recurrent_state 进行状态传递，无需 KV cache
- **全注意力层**（Full Attention）：使用标准 KV cache 进行状态传递

| 参数 | 值 |
|------|-----|
| hidden_size | 2560 |
| num_hidden_layers | 32 |
| num_attention_heads | 16 |
| num_key_value_heads | 4 |
| head_dim | 256 |
| vocab_size | 248320 |
| linear_attention_layers | 24 |
| full_attention_layers | 8 |
| image_token_id | 248056 |
| patch_size | 16 |

各层类型由 `config.json` 中的 `layer_types` 字段定义，每 4 层中有 3 个 linear_attention 和 1 个 full_attention。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
|--------|------|
| Python | 3.11 |
| torch | 2.10.0 |
| transformers | 5.6.2 |
| accelerate | 1.14.0 |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 1.26.4 |
| Pillow | 12.3.0 |
| CANN | 8.5.0 |
| mindspore-lite | 2.9.0 |
| Qwen3.5-4B权重 | ModelScope revision `ed182e32090db791077e12e0f58d22f3daafa173` |

```bash
pip install transformers==5.6.2 torch==2.10.0 accelerate==1.14.0 \
  onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 Pillow==12.3.0
pip install /path/to/mindspore_lite-2.9.0-*.whl
```

### CGDR/RGDR Custom算子

启用CGDR/RGDR优化路径前，转换和推理进程必须能够同时发现Atlas 300I Duo对应的
`ChunkGatedDeltaRule`和`RecurrentGatedDeltaRule`。算子源码已经包含在MindSpore Lite
仓库中，不需要额外下载算子仓库。

若使用上表中的MindSpore Lite 2.9.0运行时，可从当前MindSpore Lite源码构建Custom
算子，并通过环境变量进行进程内加载。该方式不会写入CANN安装目录：

```bash
source /path/to/CANN/set_env.sh
export MSLITE_HOME_PATH=/path/to/mindspore-lite-2.9.0-linux-aarch64
export MSLITE_SOURCE=/path/to/mindspore-lite-source
export CUSTOM_OPS_OUTPUT="$MSLITE_SOURCE/output/custom_ops"

export PATH="$MSLITE_HOME_PATH/tools/converter/converter:$PATH"
export LD_LIBRARY_PATH="$MSLITE_HOME_PATH/tools/converter/lib:$MSLITE_HOME_PATH/runtime/lib:$LD_LIBRARY_PATH"

mkdir -p "$(dirname "$CUSTOM_OPS_OUTPUT")"
cd "$MSLITE_SOURCE"
bash mindspore-lite/tools/custom_kernels/ascend_ops/build_all_ops.sh "$CUSTOM_OPS_OUTPUT"

export MSLITE_CUSTOM_OPP="$(find "$CUSTOM_OPS_OUTPUT" -mindepth 2 -maxdepth 2 \
  -type d -name mslite_custom_ops -print -quit)"
test -n "$MSLITE_CUSTOM_OPP"
export ASCEND_CUSTOM_OPP_PATH="$MSLITE_CUSTOM_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export MSLITE_OP_TILING="$MSLITE_CUSTOM_OPP/op_impl/ai_core/tbe/op_tiling"
export LD_LIBRARY_PATH="$MSLITE_CUSTOM_OPP/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$MSLITE_OP_TILING/lib/linux/aarch64:$MSLITE_OP_TILING:$LD_LIBRARY_PATH"
```

若使用由当前源码构建的MindSpore Lite aarch64发布包，包内已经包含对应的Custom
算子vendor，可使用随包提供的脚本安装并加载：

```bash
source /path/to/CANN/set_env.sh
export MSLITE_HOME_PATH=/path/to/mindspore-lite-<version>-linux-aarch64

bash "$MSLITE_HOME_PATH/tools/custom_kernels/install.sh"
source "$ASCEND_OPP_PATH/vendors/mslite_custom_ops/bin/set_env.bash"

export PATH="$MSLITE_HOME_PATH/tools/converter/converter:$PATH"
export LD_LIBRARY_PATH="$MSLITE_HOME_PATH/tools/converter/lib:$MSLITE_HOME_PATH/runtime/lib:$LD_LIBRARY_PATH"
```

执行`converter_lite`和推理脚本的每个新终端均需加载上述对应环境。

### 模型权重

性能表使用ModelScope revision `ed182e32090db791077e12e0f58d22f3daafa173`，下载前需安装
Git LFS。
将该版本的Qwen3.5-4B模型权重下载到当前目录下的`Qwen3.5-4B/`文件夹：

```bash
git clone https://www.modelscope.cn/Qwen/Qwen3.5-4B.git
git -C Qwen3.5-4B checkout ed182e32090db791077e12e0f58d22f3daafa173
git -C Qwen3.5-4B lfs pull
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/base_models/qwen3.5_4b

python export_qwen3_5_4b_onnx.py \
  --model-id ./Qwen3.5-4B \
  --output-dir ./qwen3_5_4b_onnx \
  --device cpu \
  --vision-image-size 128 \
  --dummy-seq-len 32 \
  --enable-cgdr-custom \
  --enable-rgdr-custom
```

`--enable-cgdr-custom`用于将Prefill阶段的线性注意力子图替换为
`ChunkGatedDeltaRule`，`--enable-rgdr-custom`用于将Decode阶段的线性注意力
子图替换为`RecurrentGatedDeltaRule`。不指定这两个参数时仍导出原始展开子图。
脚本会将Vision、Prefill和Decode分别写入`--output-dir`下的独立子目录；使用
`--component`单独导出时也保持相同目录结构。这样可避免大模型ONNX的外置
权重文件重名覆盖；不要将这些子目录中的外置权重混放。

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen3.5-4B` |
| `--output-dir` | 输出目录 | `./qwen3_5_4b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--vision-image-size` | Vision 模型输入图像尺寸（正方形边长） | `128` |
| `--dummy-seq-len` | 导出时 dummy 序列长度 | `8` |
| `--dtype` | 导出精度（fp16/fp32） | `fp16` |
| `--component` | 导出全部模型或仅导出Prefill/Decode（all/prefill/decode） | `all` |
| `--enable-cgdr-custom` | Prefill接入CGDR Custom算子 | `False` |
| `--enable-rgdr-custom` | Decode接入RGDR Custom算子 | `False` |

### 导出产出与模型 Shape

```log
qwen3_5_4b_onnx/
├── vision/
│   └── qwen3_5_vision.onnx          # Vision Tower 模型 (~637MB)
├── prefill/
│   └── qwen3_5_llm_prefill.onnx     # Prefill 图 (~0.95MB, 外置权重 ~9.68GB)
└── decode/
    └── qwen3_5_llm_decode.onnx      # Decode 图 (~0.88MB, 外置权重 ~9.68GB)
```

#### Vision Tower Shape

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | pixel_values | `[64, 1536]` | float16 | 64=8x8 patches, 1536=3x2x16x16 |
| Output | image_embeds | `[16, 2560]` | float16 | 16 个 image token, hidden_size=2560 |

#### LLM Prefill Shape（未融合路径Batch和序列长度动态；CGDR路径Batch=1）

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | input_ids | `[batch, seq_len]` | int64 | 输入 token IDs |
| Input | attention_mask | `[batch, seq_len]` | int64 | 注意力掩码 |
| Input | position_ids | `[4, batch, seq_len]` | int64 | 4D mRoPE 位置编码 |
| Input | image_embeds | `[num_image_tokens, 2560]` | float16 | 图像特征 |
| Output | logits | `[batch, seq_len, 248320]` | float16 | 预测 logits |
| Output | present_conv_states | `[24, batch, 8192, 3]` | float16 | 卷积状态（24 层线性注意力） |
| Output | present_recurrent_states | `[24, batch, 32, 128, 128]` | float32 | 循环状态（24 层线性注意力） |
| Output | present_kv_cache | `[16, batch, 4, seq_len, 256]` | float16 | KV cache（8 层全注意力 x 2） |

#### LLM Decode Shape（未融合路径Batch动态；RGDR路径Batch=1、Step=1，KV长度动态）

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | input_ids | `[batch, step]` | int64 | 单步 token ID (step=1) |
| Input | attention_mask | `[batch, total_seq_len]` | int64 | 累积注意力掩码 |
| Input | position_ids | `[4, batch, step]` | int64 | 4D mRoPE 位置编码 |
| Input | past_conv_states | `[24, batch, 8192, 3]` | float16 | 上一步卷积状态 |
| Input | past_recurrent_states | `[24, batch, 32, 128, 128]` | float32 | 上一步循环状态 |
| Input | past_kv_cache | `[16, batch, 4, past_seq_len, 256]` | float16 | 上一步 KV cache |
| Output | logits | `[batch, step, 248320]` | float16 | 预测 logits |
| Output | present_conv_states | `[24, batch, 8192, 3]` | float16 | 更新后卷积状态 |
| Output | present_recurrent_states | `[24, batch, 32, 128, 128]` | float32 | 更新后循环状态 |
| Output | present_kv_cache | `[16, batch, 4, total_seq_len, 256]` | float16 | 更新后 KV cache |

---

## 3. ONNX 转 MindSpore Lite MindIR

### 转换命令

使用 `converter_lite` 工具将 ONNX 模型转换为 MindIR 格式：

启用CGDR/RGDR后，转换和推理终端必须已经按“CGDR/RGDR Custom算子”小节完成
算子构建或安装，并加载对应环境变量。
当前CGDR/RGDR优化导出和推理路径仅验证并固定支持`batch=1`；Prefill序列长度和
Decode KV长度仍可变。

```bash
# 设置 converter_lite 路径
Convert="$MSLITE_HOME_PATH/tools/converter/converter/converter_lite"

# Vision 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_4b_onnx/vision/qwen3_5_vision.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_vision \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR

# Prefill 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_4b_onnx/prefill/qwen3_5_llm_prefill.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_llm_prefill \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR

# Decode 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_4b_onnx/decode/qwen3_5_llm_decode.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_llm_decode \
  --optimize=ascend_oriented \
  --device=Ascend \
  --configFile=config.ini \
  --saveType=MINDIR
```

### 转换产出

模型超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
qwen3_5_4b_mindir/
├── qwen3_5_vision.mindir                          # Vision MindIR (~683MB)
├── qwen3_5_llm_prefill_graph.mindir               # Prefill 图定义 (~2.6KB)
├── qwen3_5_llm_prefill_variables/data_0           # Prefill 权重 (~21GB)
├── qwen3_5_llm_decode_graph.mindir                # Decode 图定义 (~2.7KB)
└── qwen3_5_llm_decode_variables/data_0            # Decode 权重 (~11GB)
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen3_5_4b_mslite.py \
  --vision-model qwen3_5_4b_mindir/qwen3_5_vision.mindir \
  --prefill-model qwen3_5_4b_mindir/qwen3_5_llm_prefill_graph.mindir \
  --decode-model qwen3_5_4b_mindir/qwen3_5_llm_decode_graph.mindir \
  --processor ./Qwen3.5-4B \
  --image "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
  --prompt "Describe this image." \
  --max-new-tokens 128 \
  --image-size 128 \
  --device ascend \
  --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--vision-model` | Vision MindIR 模型路径 | 必填 |
| `--prefill-model` | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填 |
| `--decode-model` | Decode MindIR 模型路径（`*_graph.mindir`） | 必填 |
| `--processor` | HuggingFace processor 路径 | `./Qwen3.5-4B` |
| `--image` | 输入图像路径或 URL | `./demo.jpeg` |
| `--prompt` | 输入文本 | `"Describe this image."` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--image-size` | 图像尺寸（必须与导出 `--vision-image-size` 一致） | `128` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--host-state-roundtrip` | Decode每步将State复制到Host再回传NPU，用于回退和A/B验证 | `False` |

在Ascend设备上，推理脚本默认将Conv State、Recurrent State和KV Cache保留在
NPU侧，并对固定形状的Logits、Conv State及Recurrent State输出Tensor进行复用。
指定`--host-state-roundtrip`后恢复为原始Host中转路径。

Vision、Prefill和Decode模型按执行阶段依次加载，当前阶段结束后释放对应模型，
以降低Host侧峰值内存占用。

### 外部资源说明

- README 示例中使用 Qwen 官方 demo 图片：`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`。
- 该 URL 仅用于示例推理和性能测试，运行时会从网络读取图片；离线环境或网络受限环境请显式传入本地图片路径。
- 权重和 processor 路径通过 `--model-id` / `--processor` 参数传入，推理/导出代码未硬编码权重或图片下载 URL。

---

## 5. 性能数据

### 测试环境

- **硬件**: Atlas 300I Duo
- **CANN**: 8.5.0
- **MindSpore Lite**: 2.9.0
- **测试图片**: Qwen3.5 官方 demo 图 (`demo.jpeg`)
- **Prompt**: "Describe this image."
- **image_size**: 128 (pixel_values: [64, 1536])
- **max_new_tokens**: 128

### 推理性能（Ascend NPU）

Prefill使用固定输入，每轮重新构造Host Tensor，预热2轮后测试5轮并取中位数；
Decode使用固定Past State，预热3轮后测试10轮并取中位数。表内均统计从按模型
输入顺序整理Tensor到`Model.predict`返回的时间，不包含输出`get_data_to_numpy()`。
Prefill在计时前创建并跨预热/测试轮复用四个Ascend输出Tensor，其Shape依次为
`[1, S, 248320]`、`[24, 1, 8192, 3]`、`[24, 1, 32, 128, 128]`和
`[16, 1, 4, S, 256]`。
Decode将Conv/Recurrent/KV State及输出Buffer常驻NPU，每轮只重建小输入并
复用Buffer。
Prefill的序列长度即`input_ids`和`attention_mask`长度；Decode的序列长度指
`past_kv_cache`长度，对应`attention_mask`长度为序列长度加1。Decode固定Past
State由对应长度的Prefill输出生成，计时前不包含checkpoint生成与模型加载。
固定输入由上述官方demo图和prompt生成长度32的基础输入；长序列在基础输入
最后一个token（id 198）之前插入`seq_len - 32`个token id 220，
`attention_mask`保持全1，并使用推理脚本相同的mRoPE逻辑重建`position_ids`。
各长度的Decode checkpoint由对应Prefill执行一次后输出的
Conv/Recurrent/KV State构造，单步Decode输入token id同为220。
测试结果如下：

| 序列长度 | Prefill Predict（ms） | Decode Predict（ms/token） |
| ---: | ---: | ---: |
| 32 | 159.296 | 81.543 |
| 512 | 506.303 | 84.688 |
| 1024 | 1009.435 | 90.531 |
| 2048 | 2081.616 | 101.986 |

序列长度32、512和1024下，当前路径与未融合参考路径生成的前两个Token一致。
中间Tensor的通过标准为余弦相似度不低于0.999、
NRMSE不高于0.02且最大绝对误差不高于1.0；实测Prefill最差值分别为
0.9999855、0.005379和0.449219，Decode最差值分别为0.9999951、0.003441和
0.699219，均满足标准。

> 注意：首次运行时Ascend GE图编译耗时较长。上述结果不包含模型加载和首次编译时间。
> CGDR的L0C同步修复已经随MindSpore Lite PR #1174合入；请使用与当前模型源码
> 同一MindSpore Lite checkout构建的Custom算子vendor。

### Prefill 输入 Shape 详细说明

以 `image_size=128`、prompt="Describe this image." 为例：

- `input_ids: [1, 32]` - 32 个 token (system + image placeholder + user prompt + generation prefix)，包含 16 个 image token（由 Vision Tower 产生 16 个 patch）
- `attention_mask: [1, 32]` - 全 1
- `position_ids: [4, 1, 32]` - 4D mRoPE 位置编码 (text_pos, temporal, height, width)
- `image_embeds: [16, 2560]` - Vision Tower 输出

### Decode 输入 Shape 详细说明

每步 decode 的输入：

- `input_ids: [1, 1]` - 单个 token
- `attention_mask: [1, past_seq_len + 1]` - 随步数递增 (33, 34, 35, ...)
- `position_ids: [4, 1, 1]` - 单步位置编码
- `past_conv_states: [24, 1, 8192, 3]` - 24 层线性注意力的卷积状态
- `past_recurrent_states: [24, 1, 32, 128, 128]` - 24 层线性注意力的循环状态
- `past_kv_cache: [16, 1, 4, past_seq_len, 256]` - 8 层全注意力的 KV cache (16=8x2)

---

## 6. 推理结果示例

使用 Qwen3.5 官方 demo 图片和 prompt "Describe this image." 的推理结果：

```text

==================================================
Input Prompt: Describe this image.
Generated Response: The user wants a description of the image.

1.  **Identify the main subjects:** There are two figures in the water. One is a person, and the other is a dog.
2.  **Describe the person:**
    *   Gender: Female (long blonde hair).
    *   Clothing: Wearing a plaid shirt (blue and white/grey pattern) and dark pants (possibly jeans or leggings).
    *   Action: Sitting in the shallow water, facing the dog.
3.  **Describe the dog:**
    *   Breed: Looks like a Golden Retriever
==================================================

```

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3.5-4B 官方文档](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本教程遵循 Qwen3.5-4B 模型的许可证。
