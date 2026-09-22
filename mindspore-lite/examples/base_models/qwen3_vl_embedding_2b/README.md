# Qwen3-VL-Embedding-2B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 模型导出为 ONNX 格式，并转换为 MindSpore Lite 的 MINDIR 模型，在昇腾（Ascend）上完成文本、图像以及图文混合输入的 Embedding 推理。模型输出为 2048 维向量，推理侧对最后一个有效 token 做池化并 L2 归一化。

为了让视觉部分能够以固定 Shape 编译到昇腾上，示例把模型拆分为两个 ONNX 分块：

- **Vision 分块**（`qwen3_vl_embedding_2b_vision`）：输入为切 patch 后的像素 `pixel_values [num_patches, 1536]`，输出图像特征 `image_embeds` 以及 3 个 Deepstack 特征 `ds0/ds1/ds2`（均为 `[num_merged, 2048]`）。图像的位置编码与 Vision RoPE 表在导出时按 `--vision-image-size` 固化进计算图。
- **Text-Image 分块**（`qwen3_vl_embedding_2b_text_image`）：输入为 `input_ids`、`attention_mask`、3 行 M-RoPE 的 `position_ids` 以及 Vision 分块的 4 个输出，输出最后一层隐状态 `last_hidden_state`。图像特征在计算图内部散列到 `<|image_pad|>` 位置，Deepstack 特征注入也在图内完成。

按默认的 `--vision-image-size 1024` 导出时：1024x1024 图像 -> 64x64 的 patch 网格 -> 4096 个 patch -> 空间合并（merge 2x2）后得到 1024 个图像 token。

> 注意：ONNX 导出必须使用 float32。MindSpore Lite 转换工具不支持部分算子的 FLOAT16 类型声明（例如 `Clip` 会报 `do not support data_type: 10`）。

---

## 1. 环境准备

### 1.1 系统要求

- Linux（推荐 Ubuntu 20.04 及以上）
- Python 3.9 及以上
- 已安装 CANN，并配置好 MindSpore Lite 的运行环境

### 1.2 依赖安装

本教程验证时使用的版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.0 |
| torch | 2.12.0 |
| transformers | 5.4.0 |
| onnx | 1.23.0 |
| numpy | 2.5.1 |
| pillow | 12.3.0 |
| mindspore-lite | 2.9.0 |
| CANN | 8.5.0 |

### 1.3 环境校验

```bash
python -c "import torch, transformers, onnx, numpy, PIL; print(torch.__version__, transformers.__version__, onnx.__version__)"
python -c "import mindspore_lite as mslite; print(mslite.__version__)"
```

---

## 2. 模型导出 ONNX

### 2.1 导出命令

```bash
cd examples/base_models/qwen3_vl_embedding_2b
python export_qwen3_vl_embedding_image_onnx.py \
  --model-id ./Qwen3-VL-Embedding-2B \
  --output-dir ./qwen3_vl_embedding_onnx \
  --device cpu \
  --vision-image-size 1024
```

### 2.2 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| --model-id | 本地模型目录或 HuggingFace 模型名 | ./Qwen3-VL-Embedding-2B |
| --output-dir | ONNX 输出目录 | ./qwen3_vl_embedding_onnx |
| --device | 导出使用的设备 | cpu |
| --module | 导出 all / vision / text | all |
| --vision-image-size | 固化 Vision 位置表与 RoPE 表的图像边长 | 1024 |
| --vision-name | Vision ONNX 文件名 | qwen3_vl_embedding_2b_vision.onnx |
| --text-name | Text-Image ONNX 文件名 | qwen3_vl_embedding_2b_text_image.onnx |
| --use-fused-gelu-tanh-nz | 使用 CANN 融合算子 FusedGeluTanhNZ | 关闭 |
| --use-fused-rms-norm-nz | 使用 CANN 融合算子 FusedRmsNormNZ | 关闭 |
| --use-fused-qk-norm-rope-bsh | 使用 CANN 融合算子 FusedQKNormRopeBSH | 关闭 |

### 2.3 导出产物

```text
qwen3_vl_embedding_onnx/
├── qwen3_vl_embedding_2b_vision.onnx
├── qwen3_vl_embedding_2b_vision.onnx.data
├── qwen3_vl_embedding_2b_text_image.onnx
└── qwen3_vl_embedding_2b_text_image.onnx.data
```

权重以外部数据文件（`.onnx.data`）形式保存，转换时 `.onnx` 与 `.onnx.data` 必须放在同一目录。

### 2.4 注意事项

- `--vision-image-size` 决定 Vision 分块的输入 Shape：默认 1024 对应 4096 个 patch（`pixel_values [4096, 1536]`）与 1024 个图像 token，推理脚本的 `--image-size` 必须与此一致。
- 导出精度固定为 float32，请勿手动改成 fp16。
- 三个 `--use-fused-*` 开关默认关闭；只有确认 CANN 版本提供对应融合算子时才建议打开。

---

## 3. MindSpore Lite 转换

### 3.1 转换命令

```bash
converter_lite --fmk=ONNX \
  --modelFile=./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_vision.onnx \
  --outputFile=./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_vision \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_vl_embedding_vision.ini

converter_lite --fmk=ONNX \
  --modelFile=./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_text_image.onnx \
  --outputFile=./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_text_image \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_vl_embedding_text_image.ini
```

### 3.2 转换配置说明

`configs/qwen3_vl_embedding_vision.ini`：Vision 分块为固定 Shape，`input_shape` 直接声明为 `pixel_values:4096,1536`。

```ini
[acl_build_options]
input_format="ND"
input_shape="pixel_values:4096,1536"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision

[ascend_context]
plugin_custom_ops=All
```

`configs/qwen3_vl_embedding_text_image.ini`：Text-Image 分块只保留序列长度动态，并做动态分档（128/512/1024/2048/4096）。

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:3,1,-1;image_embeds:1024,2048;ds0:1024,2048;ds1:1024,2048;ds2:1024,2048"
ge.dynamicDims="128,128,128;512,512,512;1024,1024,1024;2048,2048,2048;4096,4096,4096"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="configs/op_fp32.json"

[ascend_context]
plugin_custom_ops=All
```

`configs/op_fp32.json` 把 RMSNorm 相关的算子（`RealDiv`、`SquareSumV1`、`Square`、`Sqrt`、`ReduceMean`）固定在 fp32 上计算，避免混合精度下精度下降过多。

### 3.3 转换产物

```text
qwen3_vl_embedding_onnx/
├── qwen3_vl_embedding_2b_vision.mindir
├── qwen3_vl_embedding_2b_text_image_graph.mindir
└── qwen3_vl_embedding_2b_text_image_variables/
    └── data_0
```

Text-Image 分块权重超过 2GB，转换后拆成 `*_graph.mindir` 与 `*_variables/` 两部分，推理时两者必须放在同一目录，并加载 `*_graph.mindir`。

### 3.4 注意事项

- 修改 `--vision-image-size` 后，Vision 分块的 patch 数与合并后 token 数会变化，需要同步更新两个 ini 中的 `input_shape` 后重新导出、转换。
- 转换失败报 `Check shape failed` 时，先检查 ini 里的 `input_shape` 是否与 ONNX 实际输入一致。

---

## 4. MindSpore Lite 推理

### 4.1 纯文本 Embedding

```bash
python infer_qwen3_vl_embedding_mslite.py \
  --vision-model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_vision.mindir \
  --text-model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_text_image_graph.mindir \
  --processor ./Qwen3-VL-Embedding-2B \
  --texts "The cat is sleeping" "A feline is resting" "The car is driving" \
  --image "" \
  --device ascend --device-id 0 \
  --compute-similarity
```

`--image` 默认使用示例图片（远程 URL），纯文本请求需显式传入 `--image ""` 关闭图像输入；此时不会触发 Vision 分块，脚本会为图像特征输入填充全 0 占位数据（昇腾不支持 0 尺寸张量，这些占位行不会被实际读取）。

### 4.2 图文 Embedding

```bash
python infer_qwen3_vl_embedding_mslite.py \
  --vision-model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_vision.mindir \
  --text-model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_text_image_graph.mindir \
  --processor ./Qwen3-VL-Embedding-2B \
  --texts "a photo of a cat" "hello world" \
  --image-size 1024 \
  --device ascend --device-id 0 \
  --compute-similarity
```

图像会与每个文本一起编码，同时脚本会单独输出图像自身的 Embedding，用于计算图文相似度。不传 `--image` 时使用脚本内置的默认示例图片（远程 URL）；离线环境请传入本地图片路径。

### 4.3 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| --vision-model | Vision MINDIR 路径 | 必填 |
| --text-model | Text-Image MINDIR 路径（`*_graph.mindir`） | 必填 |
| --processor | 处理器目录或 HuggingFace 模型名 | Qwen/Qwen3-VL-Embedding-2B |
| --texts | 待编码文本，可多个 | Hello world / Hi there / Good morning |
| --image | 图片路径或 URL；不传时使用默认示例图片，传入空字符串 `--image ""` 关闭图像输入 | https://hbr.org/resources/images/article_assets/2018/03/mar18_9_824179306.jpg |
| --instruction | 指令前缀（官方默认指令） | Represent the user's input. |
| --image-size | 图像边长，必须与导出 `--vision-image-size` 一致 | 1024 |
| --seq-len-buckets | 文本序列长度分档，需与 `ge.dynamicDims` 一致 | 128,512,1024,2048,4096 |
| --device | 推理设备（转换配置为 `ascend_oriented`，请在昇腾上运行） | cpu |
| --device-id | 昇腾设备 ID | 0 |
| --compute-similarity | 打印相似度矩阵 | 关闭 |

### 4.4 预期输出

> 说明：日志中可能出现 `use_fast` 弃用警告与 Ascend 自定义算子路径提示，均为无害告警。

```text
Loading processor from ./Qwen3-VL-Embedding-2B...
Loading model from ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_vision.mindir...
Vision model inputs:
  - pixel_values  shape=[4096, 1536]  dtype=float32
Loading model from ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_text_image_graph.mindir...
Text model inputs:
  - input_ids   shape=[1, -1]   dtype=int32
  - attention_mask      shape=[1, -1]   dtype=int32
  - position_ids        shape=[3, 1, -1]        dtype=int32
  - image_embeds        shape=[1024, 2048]      dtype=float32
  - ds0 shape=[1024, 2048]      dtype=float32
  - ds1 shape=[1024, 2048]      dtype=float32
  - ds2 shape=[1024, 2048]      dtype=float32
Image grid_thw=[1, 64, 64], image tokens=1024, vision inference=419.76 ms
============================================================
embeddings shape: (2, 2048)
image embedding shape: (2048,)
============================================================

Text-to-text similarity (cosine):
  text[0] vs text[1]: 0.6662

Image-to-text similarity (cosine):
  image vs text[0] ('a photo of a cat'): 0.6542
  image vs text[1] ('hello world'): 0.8165

--- Performance ---
  Vision inference:  419.76 ms
  Text inference:    332.29 ms (per call)
  Total:             2074.81 ms
```

具体数值会随输入变化，上表来自一次 1024x1024 随机图片 + 两条文本的实际运行。其中 `Text inference` 为单次文本模型调用的平均耗时（本次运行共调用 3 次：两条图文请求各一次、纯图像请求一次），`Total` 为端到端墙钟时间。

---

## 5. 性能数据

Atlas 300I Duo实测，1024x1024 图片：

| 阶段 | 耗时 |
| --- | --- |
| Vision 推理 | 422.47 ms |
| Text-Image 推理（图文） | 327.90 ms（单次调用平均，本次共 3 次调用） |
| 纯文本（128 分档） | 31.31 ms |
| 端到端（图文两条文本） | 1607.23 ms |

模型加载耗时约 30 秒/个，未计入上表。

---

## 6. 常见问题

### 6.1 转换报错 `Clip ... do not support data_type: 10`

MindSpore Lite 转换工具不支持部分算子的 FLOAT16 类型声明，请确保使用脚本默认的 float32 导出，不要手动改成 fp16。

### 6.2 转换报错 `Check shape failed`

`configs/*.ini` 中的 `input_shape` 与 ONNX 实际输入不一致，请检查：

- Vision 分块：`pixel_values` 的行数应为 `(image_size / 16) ** 2`，默认 1024 对应 4096；
- Text-Image 分块：`image_embeds/ds0/ds1/ds2` 的行数应为 `(image_size / 16 / 2) ** 2`，默认 1024 对应 1024。

修改 `--vision-image-size` 后需同步更新并重新导出、转换。

### 6.3 推理报错 `input data type not match, required 34, given 35`

输入张量类型与模型声明不符（34 为 int32，35 为 int64）。推理脚本会按模型声明的类型自动转换，如自行调用请把 `input_ids/attention_mask/position_ids` 转成 int32。

### 6.4 推理报错 `aclmdlSetInputDynamicDims failed`

序列长度没有命中转换时 `ge.dynamicDims` 编译出的分档。推理脚本会自动把输入左填充到最近的分档；如修改了分档列表，请同时更新 `configs/qwen3_vl_embedding_text_image.ini` 与推理脚本的 `--seq-len-buckets`。

### 6.5 图文精度异常或提示图像 token 数量不匹配

推理脚本的 `--image-size` 必须与导出时的 `--vision-image-size` 一致（默认均为 1024）。不一致时图像 token 数与 Vision 输出的行数无法对齐。

### 6.6 模型下载失败

> **外部资源说明**：`HF_ENDPOINT=https://hf-mirror.com` 和 `git clone https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B` 仅作为模型下载失败时的手动下载示例；导出脚本本身未硬编码权重下载 URL，生产或离线环境可直接传入本地权重目录。

### 6.7 纯文本请求为什么要填充图像特征

昇腾不支持 0 尺寸张量，纯文本请求没有图像特征，脚本会填充全 0 占位数据；由于提示词中没有 `<|image_pad|>` 位置，这些占位数据不会被读取。

---

## 7. 文件结构

```text
qwen3_vl_embedding_2b/
├── export_qwen3_vl_embedding_image_onnx.py   # 双分块 ONNX 导出脚本
├── infer_qwen3_vl_embedding_mslite.py        # MindSpore Lite 推理脚本（双模型）
├── convert.sh                                # ONNX -> MINDIR 转换脚本
├── configs/                                  # 转换配置
│   ├── qwen3_vl_embedding_vision.ini
│   ├── qwen3_vl_embedding_text_image.ini
│   └── op_fp32.json
└── qwen3_vl_embedding_onnx/                  # 导出与转换产物
```

---

## 8. 参考资源

- [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html)
- [MindSpore Lite 模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)

---

## 9. 许可证

Apache License 2.0
