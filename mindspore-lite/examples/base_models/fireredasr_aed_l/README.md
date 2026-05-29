# FireRedASR-AED-L ONNX 模型导出与 MindSpore Lite 推理部署教程

本目录提供 FireRedASR-AED-L 的 ONNX 导出、ONNXRuntime 推理基线，以及 MindSpore Lite(MindIR) 推理示例。

## 1. 环境准备

依赖版本（参考）：

| 组件 | 版本/说明 |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.10.0 |
| ONNX | 1.19.1 |
| onnxruntime | 1.24.2 |
| kaldi_native_fbank | 1.22.3 |
| kaldiio | 2.18.1 |
| MindSpore Lite | 2.9.0 |
| CANN | 8.5 |

安装依赖（合并一行）：

```bash
python -m pip install torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 kaldi_native_fbank==1.22.3 kaldiio==2.18.1
```

说明：

- FireRedASR 代码需要克隆并加入 `PYTHONPATH`。源码链接：https://github.com/FireRedTeam/FireRedASR#
- 权重目录使用本机路径（示例：`/path/to/FireRedASR-AED-L`），本文档中用 `MODEL_DIR` 表示。
- 若在特定环境中需要初始化工具链，可先执行（按实际环境修改）：`source /path/to/env.sh`。

## 2. 模型导出 ONNX

建议将所有产物与日志放到本目录的 `./tmp_test/<experiment_name>/` 下，便于复现实验与记录性能数据。

导出命令（默认：encoder + decoder_step self-attn 融合版 ONNX，包含 PromptFlashAttention 节点，用于 Ascend 性能优化）：

```bash
python export_fireredasr_aed_l_onnx.py \
  --fireredasr-repo /path/to/FireRedASR \
  --model-dir /path/to/FireRedASR-AED-L \
  --output-dir ./outputs \
  --device cpu
```

说明：默认导出的 decoder_step ONNX 依赖 PromptFlashAttention，ONNXRuntime 默认不识别该算子；如果需要跑 ONNX 推理，请使用 `--disable-attn-fusion` 导出非融合 decoder_step。

导出命令（ONNX 推理用：encoder + decoder_step 非融合版 ONNX）：

```bash
python export_fireredasr_aed_l_onnx.py \
  --fireredasr-repo /path/to/FireRedASR \
  --model-dir /path/to/FireRedASR-AED-L \
  --output-dir ./outputs \
  --device cpu \
  --disable-attn-fusion
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--fireredasr-repo` | FireRedASR 代码仓路径（加入 `PYTHONPATH`） |
| `--model-dir` | 权重目录（包含 `model.pth.tar`、`cmvn.ark`、`dict.txt` 等） |
| `--output-dir` | 输出目录 |
| `--device` | `cpu` / `cuda`（仅影响导出时权重加载与 dummy 推理设备） |
| `--disable-attn-fusion` | 禁用 decoder_step self-attn 融合（默认开启融合；仅当需要跑 ONNXRuntime 推理或做精度对照时关闭） |

产出文件树（示例）：

```text
./outputs/
├── onnx_encoder/
│   └── fireredasr_aed_encoder.onnx
└── onnx_decoder/
    └── fireredasr_aed_decoder_step.onnx
```

## 3. ONNX 推理

推理命令（ONNXRuntime）：

```bash
python infer_fireredasr_aed_l_onnx.py \
  --fireredasr-repo /path/to/FireRedASR \
  --onnx-dir ./outputs \
  --model-dir /path/to/FireRedASR-AED-L \
  --wav-path /path/to/test.wav \
  --provider CPUExecutionProvider \
  --max-len 200 --sos-id 3 --eos-id 4
```

说明：ONNXRuntime 推理依赖 `onnx_decoder/fireredasr_aed_decoder_step.onnx`（非融合），请先用 `--disable-attn-fusion` 进行导出。

示例音频：

- 原始下载链接：<https://github.com/FireRedTeam/FireRedASR/raw/main/examples/wav/IT0011W0001.wav>

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--onnx-dir` | ONNX 输出目录（包含 `onnx_encoder/` 与 `onnx_decoder/`） |
| `--wav-path` | 16kHz/16bit PCM WAV |
| `--provider` | ONNXRuntime provider，例如 `CPUExecutionProvider` |
| `--max-len` | 最大解码步数（greedy） |
| `--sos-id` / `--eos-id` | 起止 token id |

执行日志（示例，与实际输出一致）：

```text
{'text': '换一首歌', 'elapsed_sec': 5.0558202266693115, 'num_tokens': 4}
```

## 4. MindSpore Lite 转换

转换命令（ONNX → MindIR）：

```bash
converter_lite --fmk=ONNX \
  --modelFile=./outputs/onnx_encoder/fireredasr_aed_encoder.onnx \
  --outputFile=./outputs/mindir_encoder \
  --optimize=ascend_oriented --saveType=MINDIR

# decoder_step（默认导出为融合版；如需 ONNX 推理或做精度对照，请先用 --disable-attn-fusion 导出非融合版）
converter_lite --fmk=ONNX \
  --modelFile=./outputs/onnx_decoder/fireredasr_aed_decoder_step.onnx \
  --outputFile=./outputs/mindir_decoder_step \
  --optimize=ascend_oriented --saveType=MINDIR
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--modelFile` | 输入 ONNX |
| `--outputFile` | 输出前缀（生成 `*_graph.mindir` + `*_variables/`） |
| `--optimize=ascend_oriented` | Ascend 定向优化 |
| `--saveType=MINDIR` | 输出 MindIR |
| `--configFile` | 可选配置（例如 `./config.ini` 中的 `ge.exec.precision_mode=force_fp32`；通常用于稳定性/精度验证，可能降低性能） |

产出文件树（示例）：

```text
./outputs/
├── mindir_encoder_graph.mindir
├── mindir_encoder_variables/
├── mindir_decoder_step_graph.mindir
└── mindir_decoder_step_variables/
```

执行日志（示例，与实际输出一致）：

```text
CONVERT RESULT SUCCESS:0
```

## 5. MindSpore Lite 推理

推理命令（MindSpore Lite Python，Ascend/NPU）：

```bash
python infer_fireredasr_aed_l_mslite.py \
  --fireredasr-repo /path/to/FireRedASR \
  --mindir-dir ./outputs \
  --encoder-mindir mindir_encoder_graph.mindir \
  --decoder-mindir mindir_decoder_step_graph.mindir \
  --model-dir /path/to/FireRedASR-AED-L \
  --wav-path /path/to/test.wav \
  --device npu \
  --max-len 200 --sos-id 3 --eos-id 4
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--mindir-dir` | MindIR 输出目录（包含 `mindir_*_graph.mindir`） |
| `--encoder-mindir` | encoder mindir 文件名（默认 `mindir_encoder_graph.mindir`） |
| `--decoder-mindir` | decoder_step mindir 文件名（默认 `mindir_decoder_step_graph.mindir`，可切换为融合版/对照版） |
| `--device` | `npu`（Ascend/NPU；CPU 未验证） |
| `--max-len` | 最大解码步数（greedy） |

执行日志（示例，与实际输出一致）：

```text
Model build time: 24809.47 ms
Feature time: 2178.50 ms, feat_frames: 197, feat_len: 197
Encoder time: 31.86 ms, src_len: 50, feat_len: 197
Total decode time: 68.46 ms, avg decode step: 13.69 ms, steps: 5
Total time: 100.31 ms, throughput: 39.87 tok/s, num_tokens: 4
Recognition result: 换一首歌

```

说明（ascend_oriented 固定 shape 约束）：

- 使用 `ascend_oriented` 转换后，建议在验证与业务侧固定输入 shape（或采用分档路由）。
- 推理侧建议固定输入 shape；脚本会打印 `src_len/feat_len/decode_steps` 等信息，便于确认输入与期望 shape 兼容。
- 脚本对 encoder 与 decoder_step 默认预热 3 轮后再计时；若不预热，首轮耗时可能显著偏大。
- `Total time` 为 `Encoder time + Total decode time`（不含 `Feature time` 与 `Model build time`）。

## 6. 性能数据

测试环境：昇腾 Atlas 300I Duo（Ascend NPU）

| 指标 | 耗时 (ms) |
| --- | --- |
| Feature | 2178.50 |
| Encoder | 31.86 |
| DecoderStep（5 steps） | 68.46 |
| **总耗时** | **100.31** |
| **Avg decode step** | **13.69** |
| **吞吐量** | **39.87 tok/s** |
| **生成 token 数** | **4** |

说明：

- `steps` 是 decoder_step 实际执行次数（包含最后一次用于预测到 `eos` 的 step）。
- `生成 token 数` 不包含 `eos`（脚本在预测到 `eos` 时直接停止，不把该 token 计入输出 token_ids），因此常见 `steps = 生成 token 数 + 1`。

## 7. 常见问题

1. 转换耗时很久且输出大量 Warning
  - 现象：`converter_lite` 执行时间较长，并输出大量 Warning。
  - 原因：decoder_step 体积大（约 1.5GB），`ascend_oriented` 会做较重的图优化与编译准备。
  - 解决方案：等待转换结束，关注是否出现 `CONVERT RESULT SUCCESS:0`；必要时减少并发、确保机器内存充足。

2. 转换日志出现 `ge.proto.ModelDef exceeded maximum protobuf size of 2GB`
  - 现象：转换日志中出现 protobuf size 超限相关提示。
  - 原因：GE 内部中间表示可能超过 2GB，属于大模型常见现象。
  - 解决方案：以最终转换结果为准，确认末尾 `CONVERT RESULT SUCCESS:0`。

3. 为什么只融合 decoder_step 的 self-attn，没有融合 cross-attn/encoder？
  - decoder_step 当前仅 self-attn 使用 PromptFlashAttention；cross-attn 保持原始 attention。
  - 原因：在 cross-attn 场景中 `Qs=1, Kvs=src_len` 且需要 mask，容易触发 CANN PromptFlashAttention 的 tiling 约束（典型表为 “Qs!=Kvs 且带 mask 时要求 mask=NULL”），因此先保证“能转+能跑+有收益”的 self-attn 融合闭环。
  - encoder 为 Conformer（包含相对位置注意力/卷积分支），attention 子图更复杂，融合风险更高且收益不如 decoder_ste（decoder_step 会按 token 循环多次执行），因此暂不在 encoder 上做 attention 融合。

4. Ascend 推理报错 `input data size is wrong` / `Acl memcpy ... size: 0`
  - 现象：MSLite 推理阶段报输入 size 为 0 的错误。
  - 原因：Ascend 侧不接受 0-size Tensor，本模型 decoder_step 的 KV cache 若传入 `cached_len=0` 会触发该问题。
  - 解决方案：推理侧避免 0-size 输入（例如 cache 初始化为 `cached_len=1` 的零张量）。

## 8. 参考资源

- FireRedASR：<https://github.com/FireRedTeam/FireRedASR>
- MindSpore Lite：<https://www.mindspore.cn/lite>
- ONNXRuntime：<https://onnxruntime.ai/>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库的许可证要求。
- FireRedASR 模型与代码的许可证请以其上游仓库为准。
