# 阶段 7：README 文档

## 7.1 文档结构（9 个板块，模板必须可直接落地）

目标：Agent 仅靠 README 模板与实际运行输出，就能写出最终可复现的 README（格式参考 [`../../../../../mindspore-lite/examples/base_models/cosyvoice2_0.5b/README.md`](../../../../../mindspore-lite/examples/base_models/cosyvoice2_0.5b/README.md)）。

README 必须包含以下 9 个板块，且每个板块都要给出“命令 + 参数表 + 产出/日志”中至少两项（第 6/7/8/9 按要求即可）：

```markdown
# {MODEL_DISPLAY_NAME} ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 {MODEL_DISPLAY_NAME} 导出为 ONNX（优先单 ONNX；非必要不拆分；仅在为解决 KV cache 的 prefill/decode 需求时才拆分，且通过输入/输出实现），使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 {TARGET_DEVICE} 上推理与测速（含性能数据记录位置）。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | x.y.z |
| torch | x.y.z |
| onnx | x.y.z |
| onnxruntime | x.y.z |
| numpy | x.y.z |
| CANN | x.y |
| mindspore-lite | x.y.z |
| 其他（如 tokenizer/音频/图像依赖） | x.y.z |

```bash
pip install torch==... onnx==... onnxruntime==... numpy==... mindspore-lite ...
```

### 获取模型权重与源码

```bash
# 模型源码（如需）
git clone {UPSTREAM_REPO_URL}

# 模型权重（按项目说明下载）
```

说明：
- `MODEL_DIR` 为权重目录（需写清楚包含哪些关键文件，例如 `model.pth.tar`/`config.json`/`tokenizer.model`/`cmvn.ark` 等）。
- `MODEL_CODE_DIR` / `--{repo}` 为上游源码目录（如果推理或导出需要 import 上游代码）。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/{MODEL_NAME}

python export_{MODEL_NAME}_onnx.py \
  --model-dir {MODEL_DIR} \
  --output-dir ./outputs \
  --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-dir` | 权重目录 | 见脚本默认值 |
| `--output-dir` | ONNX 输出目录 | `./outputs` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| 其他 | 按脚本实际参数补齐 | 见脚本默认值 |

### 产出文件（文件树必须与实际一致）

```text
./outputs/
├── onnx_*.onnx
└── ...
```

### 导出注意事项（必须写实际踩坑点）

- 默认导出是否启用融合/自定义算子（如有）。
- 若存在融合开关，必须提供 disable 参数导出 non-fuse ONNX（例如 `--disable-attn-fusion`，供 ONNX Runtime 使用）。

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_{MODEL_NAME}_onnx.py \
  --onnx-dir ./outputs \
  --model-dir {MODEL_DIR} \
  --input ... \
  --provider CPUExecutionProvider
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--onnx-dir` | ONNX 模型目录 | `./outputs` |
| `--model-dir` | 权重目录 | 见脚本默认值 |
| `--provider` | ORT provider | `CPUExecutionProvider` |
| 其他 | 按脚本实际参数补齐 | 见脚本默认值 |

### 执行日志（必须粘贴真实输出）

```log
{粘贴实际输出：例如 text、score、耗时、输出文件路径等}
```

说明（必须写清楚）：
- ONNX Runtime 通常只能用于 non-fuse 导出；若融合 ONNX 包含 `Custom` 节点，ORT 可能无法执行（需明确写出本模型的实际情况）。
- 若输入是音频/图片等外部资源，尽可能提供原始下载链接（例如上游仓库 sample 文件、数据集链接），避免只给本地路径。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

说明：`converter_lite` 为 MindSpore Lite 版本包中提供的离线转换工具。

```bash
converter_lite --fmk=ONNX \
  --modelFile=./outputs/{xxx}.onnx \
  --outputFile=./outputs/{xxx}_mindir \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--modelFile` | 输入 ONNX |
| `--outputFile` | 输出前缀 |
| `--optimize=ascend_oriented` | Ascend 定向优化 |
| `--saveType=MINDIR` | 输出 MindIR |
| `--configFile` | 可选配置（指定数据类型，动静态模式以及其他编译优化等功能） |

### 配置文件（如需）

`config.ini`（纯动态示例）：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp16

[acl_build_options]
input_shape="input1:1,-1;input2:1,-1;input3:1,64;input4:1,1"

[ascend_context]
plugin_custom_ops=BatchMatmulToMatmul
```

### 产出说明（必须写实际产物形态）

```text
./outputs/
├── {name}_graph.mindir
└── {name}_variables/
```

执行日志（必须粘贴真实输出）：

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_{MODEL_NAME}_mslite.py \
  --mindir-dir ./outputs \
  --model-dir {MODEL_DIR} \
  --input ... \
  --device ascend \
  --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir-dir` | MindIR 模型目录 | `./outputs` |
| `--model-dir` | 权重目录 | 见脚本默认值 |
| `--device` | 推理设备（cpu/ascend） | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| 其他 | 按脚本实际参数补齐 | 见脚本默认值 |

### 执行日志（必须粘贴真实输出，含性能数据）

```log
{粘贴实际输出：结果 + 耗时/吞吐/avg_step 等}

[Performance Markdown]
| 指标 | 耗时 (ms) |
|---|---:|
| ... | ... |
```

说明（ascend_oriented 固定 shape 约束必须写清楚）：
- 如果转换使用 `ascend_oriented`，README 必须明确“推理侧如何保证固定 shape”（例如固定 `--max-length`、固定输入尺寸、或分档路由）。
- 如果模型采用 bucket 分档，README 必须列出所有 bucket 及路由规则。

---

## 6. 性能数据

测试环境：{硬件型号}，CANN {x.y.z}，MindSpore Lite {x.y.z}

性能数据建议以推理脚本端到端打印为准（可覆盖后续“免拷贝”等整体链路优化收益；`benchmark` 的单图 avg time 往往无法体现 e2e 性能收益）。

| 指标 | 耗时 (ms) |
| --- | ---: |
| Preprocess / Feature | ... |
| Encoder | ... |
| Decoder（{N} steps） | ... |
| **总耗时** | **...** |
| **Avg decode step** | **...** |
| **吞吐量** | **... tok/s** |
| **生成 token 数** | **...** |

说明：
- 若非自回归模型，可移除 Decoder/Avg decode step/吞吐量/生成 token 数等行，只保留与模型结构对应的阶段耗时与总耗时。
- 若需要补充算子级/子图级对比，可额外给出 `benchmark --timeProfiling=true` 的结果作为辅助信息。

---

## 7. 常见问题

必须使用编号列表；每条包含：现象、原因、解决方案。例如：

1. 现象：converter 很慢且有大量 warning
   - 原因：大模型 + ascend_oriented 编译优化重
   - 解决方案：确认最终 `CONVERT RESULT SUCCESS:0`；确保内存充足；必要时降低并发

---

## 8. 参考资源

- 上游模型仓库：{UPSTREAM_REPO_URL}
- MindSpore Lite 文档：{URL}
- ONNXRuntime 文档：{URL}

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游模型与代码许可证以其仓库为准。
```

## 7.2 文档要求

- 使用中文编写
- 路径使用相对路径
- 执行日志应与实际运行输出一致
- 多模态模型需分别给出文本-only 和图文场景的命令
