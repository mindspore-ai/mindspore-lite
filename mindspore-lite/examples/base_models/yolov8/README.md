# YOLOv8 ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 YOLOv8（2D 目标检测，COCO 80 类）导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend NPU 上推理与测速。

YOLOv8 是自动驾驶感知链路中最基础的 2D 检测模型之一，常用于相机目标检测。本目录以 `yolov8n`（nano）为默认变体，脚本支持 `n/s/m/l/x` 全系列。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本      |
|----------------|---------|
| Python         | 3.11    |
| torch          | 2.9.0   |
| ultralytics    | 8.4.71  |
| onnx           | 1.19.1  |
| onnxruntime    | 1.24.2  |
| opencv-python  | 4.11.0  |
| numpy          | 1.26.4  |
| CANN           | 8.5.0   |
| mindspore-lite | 2.10.0  |

```bash
pip install torch==2.9.0+cpu torchvision==0.24.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.4.71 onnx==1.19.1 onnxruntime==1.24.2 opencv-python "numpy<2.0" mindspore-lite
```

### 获取模型权重

```bash
# ultralytics 会在首次导出时自动下载 yolov8n.pt（COOC 预训练权重）
# 也可手动下载：https://github.com/ultralytics/assets/releases（yolov8n.pt）
```

---

## 2. 模型说明

### 架构

```log
输入图像 (1,3,640,640) → CSPDarknet backbone → FPN+PAN neck → Detect head → 输出 [1,84,8400]
                          (84 = 4 box + 80 类)             (anchor-free, 解码 + NMS 在后处理)
```

### 输入输出

| 类型 | 名称       | Shape            | 说明                          |
| ---- | ---------- | ---------------- | ----------------------------- |
| 输入 | images     | [1, 3, 640, 640] | RGB，letterbox，归一化到 0~1  |
| 输出 | output0    | [1, 84, 8400]   | 4 维 box(cx,cy,w,h) + 80 类分数 |

> 说明：YOLOv8 导出的 ONNX **不含 NMS**，需在后处理中完成置信度过滤与 NMS（见 `infer_yolov8_onnx.py` / `infer_yolov8_mslite.py` 中的 `_postprocess`，纯 numpy 实现）。

---

## 3. ONNX 导出

### 导出命令

```bash
cd examples/base_models/yolov8

python export_yolov8_onnx.py \
  --model-variant yolov8n \
  --output-dir . \
  --img-size 640 \
  --opset 17
```

### 参数说明

| 参数               | 说明                       | 默认值     |
| ------------------ | -------------------------- | ---------- |
| `--model-variant`  | YOLOv8 变体 n/s/m/l/x      | `yolov8n`  |
| `--output-dir`     | 输出目录                   | `.`        |
| `--img-size`       | 输入图像大小               | `640`      |
| `--opset`          | ONNX opset 版本            | `17`       |
| `--dynamic`        | 启用动态 batch             | `False`    |

### 产出

```text
yolov8/
└── yolov8n.onnx      # ONNX 模型
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

$Converter --fmk=ONNX \
  --modelFile=./yolov8n.onnx \
  --outputFile=./yolov8n_ascend \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

### 参数说明

| 参数             | 说明                          |
| ---------------- | ----------------------------- |
| `--fmk`          | 输入模型格式（ONNX）          |
| `--modelFile`    | 输入 ONNX 模型路径            |
| `--outputFile`   | 输出 MindIR 路径（不带扩展名）|
| `--optimize`     | 优化模式，必须 `ascend_oriented` |
| `--saveType`     | 输出格式 `MINDIR`             |
| `--configFile`   | 配置文件路径                  |

### 配置文件 `config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp16
```

> 默认 `force_fp16`（性能优先）。本教程实测 cos=1.000000，fp16 精度已达标。

### 产出

```text
yolov8/
└── yolov8n_ascend.mindir
```

执行日志：

```log
CONVERT RESULT SUCCESS:0
```

> 转换过程中可能出现 `Cannot find input: of node: /model.X/Resize` 警告，属 ultralytics 导出 Resize 算子的已知良性提示，不影响转换结果（`SUCCESS:0`）。

---

## 5. MindSpore Lite 推理

### 推理命令（Ascend NPU）

```bash
python infer_yolov8_mslite.py \
  --model ./yolov8n_ascend.mindir \
  --device ascend --device-id 0 \
  --warmup 5 --runs 20
```

### 执行日志

```log
Using random input, shape=(1, 3, 640, 640), seed=1024
Raw output shape: (1, 84, 8400)

Performance:
  batch_size: 1, warmup: 5, runs: 50
  latency_ms_mean: 5.861
Memory:
  VmRSS: 1128720 KB
```

> 说明：`ascend_oriented` 转换针对固定 shape 编译，推理侧需固定 `--img-size 640` 与导出一致。

---

## 7. 性能数据

测试环境：Atlas 300I Duo，CANN 8.5.0，MindSpore Lite 2.10.0（converter_lite 取自 2.9.0 工具包）
测试条件：输入 `(1, 3, 640, 640)`，固定随机种子 1024，warmup=5；MSLite runs=50、ORT runs=20 取均值

| 后端               | 设备        | 延迟 mean (ms) | 备注                          |
| ------------------ | ----------- | --------------  | ----------------------------- |
| MindSpore Lite     | Atlas 300I Duo   | 5.86           | force_fp16，Ascend后端推理 |

**精度对齐**（seed=1024 随机输入；ORT fp32 CPU vs MSLite fp16 Ascend，raw logits `[1,84,8400]`）：cos=**1.000000**，mean_abs=6.14e-3，max_abs=2.15（来自个别极值点 —— logit 最大值约 638，fp16 取整所致），逐 anchor 类别 argmax 一致率 **99.82%**。

**真实图片端到端验证**（ultralytics 内置 `bus.jpg`）：两后端检出**完全相同的 5 个目标**（4 person + 1 bus），框坐标偏差 <0.5px、置信度偏差 <0.005 —— fp16 转换无损检测语义。

进程内存 VmRSS≈1.10GB（远低于 44GB 显存的 80% 阈值）。

---

## 8. 常见问题

1. **现象**：转换或推理报 `Unsupported operator`
   - **原因**：ultralytics 高版本导出可能含较新算子
   - **解决方案**：确认 opset=17；必要时降级 opset 或升级 mindspore-lite

2. **现象**：MSLite 输出全零或检测数为 0，但 ONNX 正常
   - **原因**：fp16 精度溢出
   - **解决方案**：将 `config.ini` 改为 `force_fp32` 重新转换

3. **现象**：`--image` 检测框位置偏移
   - **原因**：letterbox 的 ratio/pad 还原错误
   - **解决方案**：确认 ONNX 与 MSLite 使用同一套 `_postprocess`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [YOLOv8 / ultralytics](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

本目录脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 ultralytics 上游为准（AGPL-3.0）。
