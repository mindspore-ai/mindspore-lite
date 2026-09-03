# FLUX.1-dev ONNX 导出与 MindSpore Lite 推理

本文介绍如何将 [FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)
拆分导出为固定 shape 的标准算子 ONNX，转换为 MindSpore Lite MindIR，并在 Atlas 300I Duo 上完成文生图推理。

FLUX.1-dev 由 CLIP-L、T5-XXL、12B Transformer 去噪器和 VAE 解码器组成。本目录当前验证规格为
512×512、T5 序列长度 256。分辨率或序列长度变化后，需要重新导出并转换对应子模型。

本实现不依赖 Custom 算子：Transformer attention、RMSNorm、LayerNorm、RoPE 和 VAE attention
均展开为 ONNX 标准算子。运行脚本仅依赖 MindSpore Lite、NumPy、Pillow 和 Transformers tokenizer，
不依赖 PyTorch 或 Diffusers runtime。

## 1. 环境准备

### 1.1 验证环境

以下版本已在 Atlas 300I Duo 测试环境验证：

| 软件 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.10.0+cpu |
| ONNX | 1.19.1 |
| ONNX Runtime | 1.24.2 |
| ONNX Script | 0.5.6 |
| Transformers | 5.6.2 |
| Diffusers | 0.38.0 |
| ModelScope | 1.33.0 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| attrs | 26.1.0 |
| Pillow | 12.3.0 |
| MindSpore Lite | 2.9.0 |
| CANN | 8.5.0 |

测试机单个逻辑 Atlas 300I Duo 设备约有 44 GB 显存。本实现默认把 Transformer 和 VAE 放在 device 0，
把 T5 和 CLIP 放在 device 1。

安装导出依赖：

```bash
python -m pip install \
  torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 \
  onnxscript==0.5.6 transformers==5.6.2 diffusers==0.38.0 \
  modelscope==1.33.0 numpy==1.26.4 pillow==12.3.0 \
  scipy==1.13.1 attrs==26.1.0 \
  accelerate sentencepiece

# 推理需要与 runtime 同版本、同架构的 MindSpore Lite Python wheel。
python -m pip install /path/to/mindspore_lite-2.9.0-*.whl

python -c "import mindspore_lite as mslite; print(mslite.__version__)"
```

### 1.2 下载模型

```python
from modelscope import snapshot_download

snapshot_download(
    "black-forest-labs/FLUX.1-dev",
    local_dir="./FLUX.1-dev",
)
```

下载后的目录至少需要包含：

```text
FLUX.1-dev/
├── scheduler/
├── text_encoder/
├── text_encoder_2/
├── tokenizer/
├── tokenizer_2/
├── transformer/
└── vae/
```

建议预留至少 200 GB 磁盘空间。模型权重约 32 GB，四个 FP32 ONNX 及外置数据约 64 GB；
Transformer 的在线 GE MindIR 仍保存 FP32 外置权重，四个 MindIR 还需要约 56 GB。首次在线编译还会生成缓存。

## 2. 导出 ONNX

### 2.1 子模型接口

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `flux1_clip_encoder` | `input_ids [1,77]` | `pooled_projections [1,768]` |
| `flux1_t5_encoder` | `input_ids [1,256]` | `last_hidden_state [1,256,4096]` |
| `flux1_transformer` | `hidden_states [1,1024,64]`、`encoder_hidden_states [1,256,4096]`、`pooled_projections [1,768]`、`timestep [1]`、`img_ids [1024,3]`、`txt_ids [256,3]`、`guidance [1]` | `noise_pred [1,1024,64]` |
| `flux1_vae_decoder` | `latents [1,16,64,64]` | `image [1,3,512,512]` |

512×512 图像对应 64×64 VAE latent。FLUX 将相邻 2×2 latent 打包为一个 token，
所以 Transformer 输入为 1024 个、每个 64 通道的 image token。

导出侧做了以下兼容处理：

- Transformer attention 显式展开为 `MatMul -> Softmax -> MatMul`。
- RMSNorm 和 LayerNorm 展开为基础算子，归一化统计保持 FP32。
- RoPE 使用 FP32 频率计算，并用 reshape/transpose 表达实部交错，避免动态 shape 和标量 Gather。
- VAE 使用 legacy `AttnProcessor`，把 SDPA 展开为标准 BMM/Softmax/BMM。
- 所有权重以 FP32 ONNX 导出，避免 FLOAT16 类型声明导致 parser 兼容问题。

### 2.2 导出命令

```bash
cd mindspore-lite/examples/base_models/flux1_dev

python export_flux1_dev_onnx.py \
  --model-dir ./FLUX.1-dev \
  --output-dir ./flux1_dev_onnx \
  --height 512 \
  --width 512 \
  --t5-seq-len 256 \
  --device cpu
```

可使用 `--parts` 单独导出组件。例如只导出 Transformer：

```bash
python export_flux1_dev_onnx.py \
  --model-dir ./FLUX.1-dev \
  --output-dir ./flux1_dev_onnx \
  --parts transformer \
  --height 512 --width 512 --t5-seq-len 256 \
  --device cpu
```

`--parts` 支持 `transformer,vae,t5,clip` 的任意逗号分隔子集。12B Transformer 的导出会占用较多
CPU 内存和磁盘，建议单独执行，并确保系统可用内存不少于 100 GB。

### 2.3 实测 ONNX 产物

| 子模型 | ONNX 图 | 外置权重 |
| --- | ---: | ---: |
| CLIP-L | 1.3 MB | 470 MB |
| T5-XXL | 2.6 MB | 18 GB |
| Transformer | 12 MB | 45 GB |
| VAE decoder | 640 KB | 254 MB |

四个图均已检查为固定 shape，且不包含 `If`、`Loop`、`Range`、`LayerNormalization` 或 Double 张量。
Transformer 实测为 6916 个节点，其中 `MatMul` 535 个、`Softmax` 57 个、`Gather` 0 个。

## 3. ONNX 转 MindIR

### 3.1 初始化 converter 环境

```bash
export MSLITE_HOME=/path/to/mindspore-lite-2.9.0-linux-aarch64
export CONVERTER=${MSLITE_HOME}/tools/converter/converter/converter_lite

export PATH=/usr/bin:${PATH}
source /path/to/Ascend/cann-8.5.0/set_env.sh
unset ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=${MSLITE_HOME}/tools/converter/lib:${MSLITE_HOME}/runtime/lib:${LD_LIBRARY_PATH}
```

如果 converter 报 TBE/TEFUSION dynamic handle 初始化失败，需要在当前 shell 中确认 `/usr/bin` 位于
虚拟环境之前，然后重新 source CANN 的 `set_env.sh`。

### 3.2 精度配置

CLIP 和 T5 在离线转换时使用：

```ini
[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="./configs/op_fp32.json"
```

Transformer 约有 12B 参数，使用 `--optimize=ascend_oriented` 离线编译时实测 host 峰值超过
190 GiB。为降低 converter 内存，Transformer 改用 `--optimize=none`，在 MindSpore Lite 构建模型时
通过 `flux1_transformer_runtime.config` 在线编译：

```ini
[ascend_context]
model_cache_mode=mem_opt
mixprecision_list_path=op_fp32.json

[ge_session_options]
ge.constLifecycle=session
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.externalWeight=1
```

推理脚本同时设置 `provider=ge` 和 `precision_mode=preferred_optimal`；后者是 MindSpore Lite Python API
中 `allow_mix_precision` 的名称。脚本会相对 runtime config 所在目录解析 `mixprecision_list_path`，
因此可从任意工作目录调用；在线编译缓存仍生成在当前工作目录。`op_fp32.json` 将
`RealDiv`、`SquareSumV1`、`Square`、`Sqrt` 和
`ReduceMean` 保持为 FP32，主体 MatMul 使用 FP16。不要直接改成 `force_fp16`：实测 CLIP 虽然可以
运行，但真实 prompt 对 ORT 的余弦从 0.9999996 降到约 0.15。Transformer 的 FP32 initializer 约
44.35 GiB，也无法在单张 44 GB Atlas 300I Duo 上留出运行空间，因此混合精度是当前部署必需项。

VAE 已验证可直接使用 `force_fp16`。

### 3.3 转换命令

以下命令需要从本目录执行，以便 converter 解析 `./configs/op_fp32.json`：

```bash
mkdir -p ./flux1_dev_mindir

${CONVERTER} --fmk=ONNX \
  --modelFile=./flux1_dev_onnx/flux1_clip_encoder.onnx \
  --outputFile=./flux1_dev_mindir/flux1_clip_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/flux1_clip_encoder.config

${CONVERTER} --fmk=ONNX \
  --modelFile=./flux1_dev_onnx/flux1_t5_encoder.onnx \
  --outputFile=./flux1_dev_mindir/flux1_t5_encoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/flux1_t5_encoder.config

${CONVERTER} --fmk=ONNX \
  --modelFile=./flux1_dev_onnx/flux1_transformer.onnx \
  --outputFile=./flux1_dev_mindir/flux1_transformer \
  --optimize=none --saveType=MINDIR

${CONVERTER} --fmk=ONNX \
  --modelFile=./flux1_dev_onnx/flux1_vae_decoder.onnx \
  --outputFile=./flux1_dev_mindir/flux1_vae_decoder \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/flux1_vae_decoder.config
```

CLIP、T5 和 VAE 使用 `ascend_oriented`，产物已经包含离线编译结果；Transformer 使用 `none`，
在 `Model.build_from_file` 时在线 GE 编译。converter 会把较大的模型保存为 `*_graph.mindir` 与同目录下的
`*_variables/`，小模型可能保存为单个 `*.mindir`。不要移动或遗漏 variables 目录；推理脚本兼容两种命名形式。

当前实测产物：

| 子模型 | MindIR 形式 | 大小 |
| --- | --- | ---: |
| CLIP-L | 单文件 | 310 MB |
| T5-XXL | graph + variables | 9.2 GB |
| Transformer | graph + variables | 44.4 GB |
| VAE decoder | 单文件 | 129 MB |

## 4. MindSpore Lite 推理

```bash
python infer_flux1_dev_mslite.py \
  --mindir-dir ./flux1_dev_mindir \
  --model-dir ./FLUX.1-dev \
  --transformer-config ./configs/flux1_transformer_runtime.config \
  --prompt "A cat holding a sign that says MindSpore Lite" \
  --height 512 --width 512 --t5-seq-len 256 \
  --num-inference-steps 28 \
  --guidance-scale 3.5 \
  --seed 42 \
  --transformer-device 0 \
  --text-device 1 \
  --vae-device 0 \
  --output ./flux1_output.png
```

推理流程如下：

1. CLIP-L 和 T5-XXL 编码 prompt。
2. NumPy 生成固定 seed 的 Gaussian latent，并完成 FLUX latent pack 和位置 ID 构造。
3. 根据 `scheduler/scheduler_config.json` 生成与 Diffusers 一致的 FlowMatch Euler schedule。
4. Transformer 逐步预测 noise，CPU 侧使用 NumPy Euler 更新 latent。
5. latent unpack 后，按 VAE config 的 scaling factor 和 shift factor 还原 VAE 输入。
6. VAE 解码并保存 512×512 PNG。

可通过 `--latents-npy` 传入 `[1,16,64,64]` 或 `[1,1024,64]` 的预生成 latent，以复现精度对比。
NumPy 和 PyTorch 使用不同的随机数生成器，因此相同 seed 不保证与 Diffusers 的初始 latent 逐元素一致；
需要严格对比时应传入同一份 `--latents-npy`。

Transformer 第一次运行会在当前工作目录生成 `model_build_cache_0/`。实测首次在线 GE build 约 116 秒，
首个 `predict`（包含 TBE 编译）约 222 秒，缓存约 23 GB；后续运行保留并复用该缓存。运行目录必须可写，
且不要在模型或配置不变时删除缓存。

## 5. 验证结果

测试环境：Atlas 300I Duo、CANN 8.5.0、MindSpore Lite 2.9.0、512×512、T5 长度 256。

### 5.1 组件精度

| 子模型 | ONNX Runtime 对 MindSpore Lite | 结果 |
| --- | --- | ---: |
| CLIP-L | 真实 prompt pooled embedding | cosine = 0.9999996 |
| T5-XXL | 18 个有效 prompt token | cosine = 0.9999778 |
| T5-XXL | 全部 256 token | cosine = 0.9998246 |
| VAE decoder | 同一随机 latent | cosine = 0.9999377 |
| Transformer | 同一固定输入 | cosine = 0.9986137 |

CLIP 单次前向约 4.4 ms，T5 单次前向约 149 ms。T5 的较大最大误差集中在 padding token，
有效 token 的最大绝对误差为 0.01845。Transformer 对同一固定输入的平均绝对误差为 0.04269，
最大绝对误差为 0.35972；输出全部为有限值。

### 5.2 端到端性能

| 阶段 | 耗时 |
| --- | ---: |
| 文本编码 | 165.60 ms |
| Transformer 单步平均 | 1165.15 ms |
| Transformer 28 步 | 32624.16 ms |
| VAE 解码 | 109.45 ms |
| 端到端 | 33022.15 ms |

以上数据复用了已生成的在线 GE 缓存。Transformer 第 1 步包含权重载入，耗时 6485.39 ms；
第 2～28 步稳定在约 966～973 ms。输出为 512×512 RGB PNG，28 步均成功完成。

## 6. 常见问题

### 6.1 Transformer 导出耗时长或磁盘快速增长

Transformer 约有 12B 参数，FP32 ONNX 外置权重约 45 GB。建议用 `--parts transformer` 单独在 CPU
导出，确保可用内存不少于 100 GB、剩余磁盘不少于 60 GB。

Transformer 不要改为 `--optimize=ascend_oriented`：该模式在 12B 图上会把 ONNX 权重、FuncGraph 和
离线 GE 编译数据同时驻留在 host 内存。使用本文的 `--optimize=none`，并在推理时传入在线 GE 配置。

### 6.2 VAE 转换在 Attention Reshape 失败

PyTorch SDPA 可能生成 GE 无法正确推导的四维 Reshape。当前脚本已切换到 legacy `AttnProcessor`，
用标准 BMM/Softmax/BMM 表达相同计算；不要移除该设置。

### 6.3 模型可以运行但输出语义明显不对

先对同一输入比较 ONNX Runtime 与 MindSpore Lite 中间输出。不要只检查“能否前向”。CLIP 的实测表明
`force_fp16` 会让归一化敏感链精度严重下降；使用本目录的 `allow_mix_precision + op_fp32.json` 配置。

### 6.4 找不到大模型权重

大模型转换结果由 `*_graph.mindir` 和 `*_variables/` 组成，两者必须位于同一目录。
推理脚本会先查找 `_graph.mindir`，再回退到单文件 `.mindir`。

### 6.5 输入 dtype 不匹配

ONNX token ID 是 int64，converter 后 MindIR 通常要求 int32。推理脚本会根据 `model.get_inputs()`
返回的 dtype 自动转换；自定义推理代码也需要做相同处理。

### 6.6 为什么没有 negative prompt 和双路 CFG

FLUX.1-dev 使用模型自身的 guidance embedding。`guidance-scale` 作为 Transformer 输入，不执行传统
Stable Diffusion 的正负 prompt 双路 classifier-free guidance。

### 6.7 修改分辨率后 shape 不匹配

当前 ONNX 与 MindIR 固定为 512×512、T5 长度 256。修改 `--height`、`--width` 或 `--t5-seq-len` 后，
需要重新导出、转换所有受影响组件。

## 7. 参考资源与许可证

- [FLUX.1-dev ModelScope 模型页](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)
- [FLUX.1-dev Hugging Face 模型页](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Diffusers FLUX 文档](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite)

本目录代码遵循 MindSpore Lite 仓库许可证。FLUX.1-dev 权重遵循
**FLUX.1 [dev] Non-Commercial License v1.1.1**；使用、修改或分发权重前请阅读上游 `LICENSE.md`，
并遵守其非商业用途限制。
