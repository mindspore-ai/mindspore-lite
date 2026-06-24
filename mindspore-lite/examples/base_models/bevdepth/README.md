# BEVDepth ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 BEVDepth（含显式深度监督的 BEVDet 变体）导出为 ONNX，并转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理部署。BEVDepth 复用 BEVDet 的 `Custom::BEVPoolV3` 算子模式。

> **注意**：导出的 ONNX 含自定义算子 `Custom::BEVPoolV3`，**不支持 ONNX Runtime 推理**，需转 MindIR 在 Ascend 运行。

> **阶段 2 风险点**：BEVPoolV3 需 AscendC 自定义实现；若暂无可用 kernel，按 plan「记录暂搁」。

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
# 从 BEVDepth 仓库安装 mmdet3d（editable）
```

权重与 config 来源：[Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)。

---

## 2. 模型说明

```log
原始输入(6相机) → Image Backbone → Image Neck → LSS(depth+BEVPool) → BEV特征 → BEV Encoder → CenterHead
  (1,6,3,256,704)   (ResNet)        (FPN)        (CustomBEVPoolV3)  (1,64,128,128)              ↓
                                                                                              reg/height/dim/rot/vel/heatmap
```

| 类型 | 名称      | Shape                    | 说明              |
| ---- | --------- | ------------------------ | ----------------- |
| 输入 | img       | \[1, 6, 3, 256, 704]     | 6 路相机图像      |
| 输出 | reg/height/dim/rot/vel/heatmap | 各 \[1, C, 128, 64] | CenterHead 检测 |

> 与 BEVDet 区别：BEVDepth 加深度监督网络，提升深度估计精度（pipeline 其余一致）。

---

## 3. 自定义 BEVPoolV3 算子

复用 `bevdet/` 示例的 `CustomBEVPoolV3`：`torch.autograd.Function` + `symbolic(g, ...)` 注册 `Custom::BEVPoolV3`，导出用 `OperatorExportTypes.ONNX_FALLTHROUGH`。详见 [bevdet README](../bevdet/README.md) 第 3 节。

---

## 4. ONNX 导出

```bash
cd examples/base_models/bevdepth
python export_bevdepth_onnx.py \
  --config /path/to/BEVDepth/configs/...py \
  --checkpoint /path/to/bevdepth.pth \
  --output bevdepth_onnx/bevdepth.onnx --opset 17
```

产出：`bevdepth_onnx/bevdepth.onnx`

---

## 5. ONNX 推理

> BEVPoolV3 不支持 ONNX Runtime，`infer_bevdepth_onnx.py` 仅用于结构检查；推理请走 MindIR。

---

## 6. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./bevdepth_onnx/bevdepth.onnx \
  --outputFile=./bevdepth_onnx/bevdepth_ascend \
  --optimize=ascend_oriented --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp32`（与 BEVDet 一致，BEV 模型 fp16 易精度损失）。

```log
CONVERT RESULT SUCCESS:0
```

---

## 7. MindSpore Lite 推理

```bash
python infer_bevdepth_mslite.py \
  --model ./bevdepth_onnx/bevdepth_ascend.mindir \
  --device ascend --device-id 0
```

执行日志（占位）：

```log
latency_ms_mean: TBD
VmRSS: TBD KB
```

---

## 8. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：输入 `(1,6,3,256,704)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 备注            |
| -------------- | ----- | --------------------- | --------------- |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | force_fp32, BEVPool |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡；脚本已内置 VmRSS/VmHWM 监控。

---

## 9. 常见问题

1. **现象**：转换/推理报 BEVPoolV3 不支持
   - **原因**：缺 AscendC BEVPoolV3 kernel
   - **解决方案**：阶段 2 复用 `bevdet` 的 BEVPoolV3 自定义算子；暂无则记录暂搁

2. **现象**：fp16 检测精度异常
   - **解决方案**：保持 `config.ini` 的 `force_fp32`

---

## 10. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVDet 参考实现](../bevdet/README.md)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 BEVDepth 上游为准（Apache-2.0）。
