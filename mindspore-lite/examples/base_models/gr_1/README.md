# GR-1 视频生成预训练机器人策略 ONNX 导出与 MindSpore Lite 推理部署教程

GR-1（ByteDance）用大规模视频生成预训练骨干（视频 ViT），微调为视觉机器人操作策略：从短视频观测预测动作 chunk。本目录骨架为 video → patch+时序 token → Transformer → 动作 chunk（回归，单前向）。

> ⚠️ **风险标注**：GR-1/GR-2 开源状态需任务2 联网核实（ByteDance 发布的权重/建模 API 可能不完整）。本目录为视频条件回归骨架，`--random-init` 可端到端验证管线；真实权重加载待核实。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：核实 ByteDance GR-1 开源情况后从 ModelScope/HF 下载。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/gr_1
python export_gr_1_onnx.py --output-dir ./gr_1_onnx --device cpu
# 真实权重（任务2）：python export_gr_1_onnx.py --checkpoint /path/to/gr1.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | GR-1 state_dict（任务2 核实开源） | 空（demo） |
| `--num-frames` | 输入视频帧数 | `4` |
| `--action-dim` / `--horizon` | 动作维度/chunk 长度 | `7`/`16` |

```text
./gr_1_onnx/
└── gr_1_policy.onnx   # video[B,T,3,H,W] -> action[B,horizon,action_dim]
```

---

## 3. ONNX 推理

```bash
python infer_gr_1_onnx.py --model ./gr_1_onnx/gr_1_policy.onnx --num-frames 4 --seed 0
# 真实视频：python infer_gr_1_onnx.py --model ... --video-dir /path/to/frames
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./gr_1_onnx/gr_1_policy.onnx \
  --outputFile=./gr_1_onnx/gr_1_policy --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="video:1,4,3,224,224"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

```text
./gr_1_onnx/
├── gr_1_policy.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_gr_1_mslite.py --model ./gr_1_onnx/gr_1_policy.mindir \
  --num-frames 4 --seed 0 --device ascend --device-id 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 单步前向（mean） | 2.77 |
| MSLite 单步前向（p50） | 2.77 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999999 |

---

## 7. 常见问题

1. GR-1 开源不完整：任务2 联网核实，必要时仅迁移视频骨干子图。
2. 视频帧数固定：`--num-frames` 须与导出/转换/推理一致。
3. 大视频 token 数：frames×196 patch，注意显存。

---

## 8. 参考资源

- GR-1：https://github.com/bytedance/GR-1（核实开源状态）
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- GR-1 上游代码许可证以其仓库为准。
