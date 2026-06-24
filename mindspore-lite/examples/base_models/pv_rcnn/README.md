# PV-RCNN ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 PV-RCNN（LiDAR 点云 3D 检测）的检测头导出为 ONNX，转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理测速。PV-RCNN 用 spconv 体素主干 + RoI-grid PointNet pooling + Anchor3DHead。

> **导出范围**：scaffold 导出 **dense 2D 路径**（backbone+neck+AnchorHead，输入为 BEV 特征图）。spconv + RoI-grid pooling 是阶段 2 阻塞点（同 SECOND，额外含 RoI grid），按 plan「记录暂搁」。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmcv-full / mmdet / mmdetection3d | 1.7.0 / 2.28.2 / 1.0.0rc4 |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2 onnx==1.14.0 onnxruntime==1.16.0
```

权重与 config 来源：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 的 `configs/pv_rcnn/`。

---

## 2. 模型说明

```log
点云 → voxelization+spconv+RoI-grid pooling（阶段2阻塞，本scaffold跳过）→ dense BEV (1,256,200,176) → 2D backbone → neck → AnchorHead
                                                                                                                  ↓
                                                                                          cls_scores + bbox_preds + dir_cls
```

| 类型 | 名称          | Shape               | 说明                |
| ---- | ------------- | ------------------- | ------------------- |
| 输入 | bev_feat      | \[1, 256, 200, 176] | spconv 后的稠密 BEV |
| 输出 | cls_scores    | \[1, A, h, w]       | anchor 级类别分数   |
| 输出 | bbox_preds    | \[1, A*code, h, w]  | anchor 级 3D 框回归 |
| 输出 | dir_cls       | \[1, A*2, h, w]     | 方向分类            |

---

## 3. ONNX 导出

```bash
cd examples/base_models/pv_rcnn
python export_pv_rcnn_onnx.py \
  --config /path/to/mmdetection3d/configs/pv_rcnn/..._nus.py \
  --checkpoint /path/to/pv_rcnn_...pth \
  --output pv_rcnn_onnx/pv_rcnn.onnx --opset 17
```

产出：`pv_rcnn_onnx/pv_rcnn.onnx`

---

## 4. ONNX 推理

```bash
python infer_pv_rcnn_onnx.py --model ./pv_rcnn_onnx/pv_rcnn.onnx
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random dense BEV, shape=(1, 256, 200, 176), seed=1024
  cls_scores / bbox_preds / dir_cls: ...
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./pv_rcnn_onnx/pv_rcnn.onnx \
  --outputFile=./pv_rcnn_onnx/pv_rcnn_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_pv_rcnn_mslite.py \
  --model ./pv_rcnn_onnx/pv_rcnn_ascend.mindir \
  --device ascend --device-id 0
```

执行日志（占位）：

```log
latency_ms_mean: TBD
VmRSS: TBD KB
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：dense BEV `(1,256,200,176)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 备注               |
| -------------- | ----- | --------------------- | ------------------ |
| ONNX Runtime   | CPU   | TBD                   | 精度基准（dense）  |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | force_fp16（dense）|

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **阶段 2 阻塞（已记录暂搁）**：实测 **mmdetection3d v1.0.0rc4 的 configs 不含 PV-RCNN / VoxelRCNN**（该版本仅含 centerpoint/pointpillars/second/point_rcnn 等；PV-RCNN 配置在更老的 mmdet3d 0.x）。加之 PV-RCNN 含 point 分支(PointNet++ SA) + voxel 分支 + RoI 特征融合，结构复杂。故 PV-RCNN 在当前 mmdet3d 栈下无法验证。解决路径：① 用更老 mmdet3d（0.x）的 PV-RCNN config+checkpoint（需另建环境）；② 或仅验证其 voxel-backbone→head 子路径（近似）。暂按记录暂搁。

2. **现象**：需要端到端（点云→检测）
   - **原因**：spconv + RoI-grid pooling Ascend 无原生支持
   - **解决方案**：阶段 2 用 AscendC 实现 spconv/roi-grid，或 dense 近似；暂无则记录暂搁

2. **现象**：fp16 检测精度下降
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [mmdetection3d / PV-RCNN](https://github.com/open-mmlab/mmdetection3d)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 mmdetection3d 上游为准（Apache-2.0）。
