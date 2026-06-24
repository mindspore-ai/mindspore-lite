# PETR ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 PETR（Position Embedding Transformation，多视图相机 3D 检测）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。PETR 用 3D 位置嵌入把多视图特征对齐到 3D 空间，经 Transformer 解码出 3D 框，算子以标准 Transformer + CNN 为主。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmcv-full      | 1.7.0     |
| mmdet          | 2.28.2    |
| mmdetection3d  | 1.0.0rc4  |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2 onnx==1.14.0 onnxruntime==1.16.0
# 从 mmdetection3d 源码安装 1.0.0rc4
```

权重与 config 来源：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 的 `configs/petr/`。

---

## 2. 模型说明

```log
多视图图像 (1,6,3,320,800) → backbone(ResNet) → neck → 3D pos embed + Transformer → head
                                                                              ↓
                                                          cls_scores (1,Q,cls) + bbox_preds (1,Q,code)
```

| 类型 | 名称          | Shape                     | 说明                 |
| ---- | ------------- | ------------------------- | -------------------- |
| 输入 | imgs          | \[1, 6, 3, 320, 800]      | 6 路相机图像         |
| 输出 | cls_scores    | \[1, Q, num_cls]          | query 级类别分数     |
| 输出 | bbox_preds    | \[1, Q, code]             | query 级 3D 框回归   |

> Q 为 query 数（PETR≈434，PETRv2≈814）。导出为 head 的 query 级原始输出；3D 框解码（含 cam2img 投影）留后处理，阶段 2 在 numpy 实现。

---

## 3. ONNX 导出

```bash
cd examples/base_models/petr
python export_petr_onnx.py \
  --config /path/to/mmdetection3d/configs/petr/petr_r50_704x256_24ep.py \
  --checkpoint /path/to/petr_r50_..._.pth \
  --output petr_onnx/petr.onnx --ncams 6 --img-h 320 --img-w 800 --opset 17
```

产出：`petr_onnx/petr.onnx`

---

## 4. ONNX 推理

```bash
python infer_petr_onnx.py --model ./petr_onnx/petr.onnx --ncams 6 --img-h 320 --img-w 800
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input, shape=(1, 6, 3, 320, 800), seed=1024
  cls_scores: (1, Q, num_cls)
  bbox_preds: (1, Q, code)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./petr_onnx/petr.onnx \
  --outputFile=./petr_onnx/petr_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_petr_mslite.py \
  --model ./petr_onnx/petr_ascend.mindir \
  --device ascend --device-id 0 --ncams 6 --img-h 320 --img-w 800
```

执行日志（占位）：

```log
latency_ms_mean: TBD
VmRSS: TBD KB
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：输入 `(1, 6, 3, 320, 800)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 延迟 p99 (ms) | 备注       |
| -------------- | ----- | --------------------- | ------------- | ---------- |
| ONNX Runtime   | CPU   | TBD                   | TBD           | 精度基准   |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | TBD           | force_fp16 |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：导出时报 `head.forward` 需要 img_metas
   - **原因**：PETR head 依赖 img_metas（cam2img 等）生成 3D pos embed
   - **解决方案**：阶段 2 把 cam2img 作为张量输入注入，或用真实 img_metas；本 scaffold 用空 img_metas 走通主干

2. **现象**：fp16 Transformer 输出异常
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [mmdetection3d / PETR](https://github.com/open-mmlab/mmdetection3d)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 mmdetection3d 上游为准（Apache-2.0）。
