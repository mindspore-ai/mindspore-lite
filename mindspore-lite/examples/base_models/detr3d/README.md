# DETR3D ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 DETR3D（基于 deformable 采样的多视图相机 3D 检测）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。DETR3D 将 3D 参考点投影到多视图后采样特征，再经 Transformer 解码 3D 框。

> **阶段 2 风险点**：geometry decoder 中的 deformable/采样算子可能不被 Ascend 直接支持，遇阻塞按 plan「记录暂搁」处理。

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
```

权重与 config 来源：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 的 `configs/detr3d/`。

---

## 2. 模型说明

```log
多视图图像 (1,6,3,320,800) → ResNet → neck → 3D 参考点投影 + deformable 采样 + Transformer → head
                                                                                          ↓
                                                              cls_scores (1,Q,cls) + bbox_preds (1,Q,code)
```

| 类型 | 名称          | Shape                | 说明               |
| ---- | ------------- | -------------------- | ------------------ |
| 输入 | imgs          | \[1, 6, 3, 320, 800] | 6 路相机图像       |
| 输出 | cls_scores    | \[1, Q, num_cls]     | query 级类别分数   |
| 输出 | bbox_preds    | \[1, Q, code]        | query 级 3D 框回归 |

---

## 3. ONNX 导出

```bash
cd examples/base_models/detr3d
python export_detr3d_onnx.py \
  --config /path/to/mmdetection3d/configs/detr3d/detr3d_...py \
  --checkpoint /path/to/detr3d_...pth \
  --output detr3d_onnx/detr3d.onnx --ncams 6 --img-h 320 --img-w 800 --opset 17
```

产出：`detr3d_onnx/detr3d.onnx`

---

## 4. ONNX 推理

```bash
python infer_detr3d_onnx.py --model ./detr3d_onnx/detr3d.onnx --ncams 6 --img-h 320 --img-w 800
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
$Converter --fmk=ONNX --modelFile=./detr3d_onnx/detr3d.onnx \
  --outputFile=./detr3d_onnx/detr3d_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_detr3d_mslite.py \
  --model ./detr3d_onnx/detr3d_ascend.mindir \
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

1. **现象**：转换报 deformable / 采样算子不支持
   - **原因**：geometry decoder 的可变形采样算子 Ascend 暂不支持
   - **解决方案**：阶段 2 按记录暂搁，或用标准 attention 替换采样算子重导出

2. **现象**：head.forward 需要 img_metas
   - **解决方案**：阶段 2 注入真实 img_metas；本 scaffold 用空 img_metas 走通主干

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [mmdetection3d / DETR3D](https://github.com/open-mmlab/mmdetection3d)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 mmdetection3d 上游为准（Apache-2.0）。
