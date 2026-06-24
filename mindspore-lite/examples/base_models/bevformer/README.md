# BEVFormer ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 BEVFormer（时序跨视图 BEV 感知）导出为 ONNX，并转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理部署。

> **阶段 2 核心风险点**：BEVFormer 的 Temporal Cross-View Deformable Attention（MSDeformableAttention）依赖 mmcv CUDA op，ONNX 导出会注册为 Custom 节点。阶段 2 验证在此处大概率阻塞，按 plan「记录暂搁」，需 AscendC 可变形注意力算子才能跑通。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmcv-full / mmdet / mmdetection3d | 1.7.0 / 2.28.2 / 1.0.0rc4 |
| onnx           | 1.14.0    |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2 onnx==1.14.0
# 从 mmdetection3d 安装
```

权重与 config 来源：[fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)。

---

## 2. 模型说明

```log
多视图图像 (1,6,3,320,800) → ResNet → neck → Temporal Deformable Attention → BEV → Decoder → head
                                                                                          ↓
                                                              cls_scores (1,Q,cls) + bbox_preds (1,Q,code)
```

| 类型 | 名称          | Shape                | 说明               |
| ---- | ------------- | -------------------- | ------------------ |
| 输入 | imgs          | \[1, 6, 3, 320, 800] | 6 路相机图像       |
| 输出 | cls_scores    | \[1, Q, num_cls]     | query 级类别分数   |
| 输出 | bbox_preds    | \[1, Q, code]        | query 级 3D 框回归 |

> 时序扩展（阶段 2）：引入 `prev_bev`，多帧对齐。

---

## 3. ONNX 导出

```bash
cd examples/base_models/bevformer
python export_bevformer_onnx.py \
  --config /path/to/BEVFormer/configs/...py \
  --checkpoint /path/to/bevformer.pth \
  --output bevformer_onnx/bevformer.onnx --opset 17
```

产出：`bevformer_onnx/bevformer.onnx`（含 Custom deformable 节点）

---

## 4. ONNX 推理

> Custom deformable 算子 ORT 无法执行，`infer_bevformer_onnx.py` 仅用于结构检查；推理走 MindIR。

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./bevformer_onnx/bevformer.onnx \
  --outputFile=./bevformer_onnx/bevformer_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp32`（BEV/Transformer 模型 fp16 易溢出）。

> 阶段 2 转换大概率在 Custom deformable 算子处失败 → 记录暂搁。

---

## 6. MindSpore Lite 推理

```bash
python infer_bevformer_mslite.py \
  --model ./bevformer_onnx/bevformer_ascend.mindir \
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
测试条件：输入 `(1,6,3,320,800)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 备注                       |
| -------------- | ----- | --------------------- | -------------------------- |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | 需 deformable 自定义算子  |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：转换报 deformable attention 算子不支持
   - **原因**：MSDeformableAttention 依赖 CUDA op
   - **解决方案**：阶段 2 用 AscendC 实现可变形注意力；或用标准 attention 替换（精度损失）；暂无则记录暂搁

2. **现象**：fp16 Transformer 溢出
   - **解决方案**：保持 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 BEVFormer 上游为准（Apache-2.0）。
