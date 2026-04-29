# YOLOv10-X ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 YOLOv10-X 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.10   |
| torch          | 2.11.0 |
| ultralytics    | 8.4.43 |
| onnx           | 1.20.1 |
| onnxruntime    | 1.23.2 |
| CANN           | 8.5.0  |
| mindspore-lite | 2.8.0  |

```bash
pip install torch==2.11.0+cpu torchvision==0.26.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.4.43 onnx==1.20.1 onnxruntime==1.23.2 mindspore-lite
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/base_models/yolo_v10_X

python export_yolov10x_onnx.py \
  --model-variant yolov10x \
  --output-dir . \
  --img-size 640
```

### 参数说明

| 参数             | 说明         | 默认值      |
|----------------|------------|-----------|
| `--model-variant` | YOLOv10 变体 | `yolov10x` |
| `--output-dir` | 输出目录       | `.`       |
| `--img-size`   | 输入图像大小     | `640`     |
| `--opset`      | ONNX opset 版本 | `12`      |
| `--dynamic`     | 启用动态 batch  | `False`   |

### 产出

```log
yolo_v10_X/
├── yolov10x.onnx     # ONNX 模型 (~112.8MB)
└── yolov10x.pt       # 原始权重备份
```

---

## 3. ONNX 推理

### ONNX Runtime 推理

```bash
python infer_yolov10x_onnx.py \
  --model ./yolov10x.onnx \
  --image ./city-streets.png \
  --device cpu \
  --warmup 3 \
  --runs 5
```

**执行日志：**

```log
ONNX: loading model from ./yolov10x.onnx...
ONNX: inference session created with providers: ['CPUExecutionProvider']
Detection Results:
Image: ./city-streets.png
  Det 0: cls=2 (car), conf=0.9581, bbox=[559.1, 473.2, 799.4, 597.6]
  Det 1: cls=2 (car), conf=0.9472, bbox=[221.6, 420.6, 498.7, 522.2]
  Det 2: cls=0 (person), conf=0.9153, bbox=[590.5, 538.6, 666.8, 667.4]
  Det 3: cls=0 (person), conf=0.9111, bbox=[687.5, 325.8, 739.3, 415.4]
  Det 4: cls=1 (bicycle), conf=0.9103, bbox=[1.5, 646.5, 137.6, 730.5]
  Det 5: cls=1 (bicycle), conf=0.9067, bbox=[561.1, 597.9, 694.2, 672.9]
  Det 6: cls=1 (bicycle), conf=0.9053, bbox=[440.1, 658.9, 579.3, 735.5]
  Det 7: cls=0 (person), conf=0.8811, bbox=[42.3, 587.6, 98.8, 711.5]
  Det 8: cls=9 (traffic light), conf=0.8664, bbox=[722.3, 164.5, 751.6, 235.7]
  Det 9: cls=9 (traffic light), conf=0.8425, bbox=[260.6, 69.9, 291.6, 126.9]
  Det 10: cls=0 (person), conf=0.8421, bbox=[476.5, 601.0, 553.3, 733.6]
  Det 11: cls=9 (traffic light), conf=0.8388, bbox=[470.5, 82.2, 501.7, 138.8]
  Det 12: cls=2 (car), conf=0.7987, bbox=[324.1, 36.9, 362.8, 67.9]
  Det 13: cls=0 (person), conf=0.7086, bbox=[335.1, 422.6, 391.1, 460.6]
  Det 14: cls=0 (person), conf=0.6417, bbox=[400.1, 339.6, 422.7, 381.9]
  Det 15: cls=24 (backpack), conf=0.6123, bbox=[632.2, 549.6, 668.1, 582.9]
  Det 16: cls=2 (car), conf=0.5191, bbox=[526.3, 131.8, 585.1, 179.0]
  Det 17: cls=2 (car), conf=0.4941, bbox=[335.9, 20.1, 367.3, 44.7]
  Det 18: cls=7 (truck), conf=0.4201, bbox=[543.9, 179.2, 632.9, 252.2]
  Det 19: cls=24 (backpack), conf=0.3578, bbox=[481.8, 592.2, 534.2, 648.7]
```

---

## 4. MindSpore Lite 转换

### 转换命令

```bash
Converter=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

$Converter --fmk=ONNX \
  --modelFile=./yolov10x.onnx \
  --outputFile=./yolov10x_ascend \
  --optimize=ascend_oriented \
  --configFile=config.ini
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径                      |

### 配置文件

`config.ini`:

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出

```log
yolo_v10_X/
├── yolov10x_ascend.mindir     # Ascend 优化版 MindIR (~72MB)
└── yolov10x.onnx              # ONNX 模型 (~112.8MB)
```

---

## 5. MindSpore Lite 推理

### 推理命令（Ascend NPU）

```bash
python infer_yolov10x_mslite.py \
  --model ./yolov10x_ascend.mindir \
  --image ./city-streets.png \
  --device ascend \
  --warmup 3 \
  --runs 5
```

**执行日志：**

```log
Loading model from ./yolov10x_ascend.mindir...
MindSpore Lite inference start.
Detection Results:
Image: ./city-streets.png
  Det 0: cls=2 (car), conf=0.9582, bbox=[559.1, 473.2, 799.4, 597.6]
  Det 1: cls=2 (car), conf=0.9473, bbox=[221.5, 420.6, 498.7, 522.2]
  Det 2: cls=0 (person), conf=0.9152, bbox=[590.5, 538.6, 666.8, 667.4]
  Det 3: cls=0 (person), conf=0.9111, bbox=[687.5, 325.8, 739.3, 415.4]
  Det 4: cls=1 (bicycle), conf=0.9103, bbox=[1.5, 646.5, 137.6, 730.5]
  Det 5: cls=1 (bicycle), conf=0.9067, bbox=[561.1, 597.9, 694.2, 672.9]
  Det 6: cls=1 (bicycle), conf=0.9054, bbox=[440.2, 658.9, 579.3, 735.5]
  Det 7: cls=0 (person), conf=0.8814, bbox=[42.3, 587.6, 98.8, 711.5]
  Det 8: cls=9 (traffic light), conf=0.8664, bbox=[722.3, 164.5, 751.6, 235.7]
  Det 9: cls=0 (person), conf=0.8426, bbox=[476.5, 601.0, 553.3, 733.7]
  Det 10: cls=9 (traffic light), conf=0.8424, bbox=[260.6, 69.9, 291.6, 126.9]
  Det 11: cls=9 (traffic light), conf=0.8389, bbox=[470.5, 82.2, 501.7, 138.8]
  Det 12: cls=2 (car), conf=0.7986, bbox=[324.1, 36.9, 362.8, 67.9]
  Det 13: cls=0 (person), conf=0.7092, bbox=[335.1, 422.6, 391.1, 460.6]
  Det 14: cls=0 (person), conf=0.6414, bbox=[400.1, 339.6, 422.7, 381.9]
  Det 15: cls=24 (backpack), conf=0.6119, bbox=[632.2, 549.6, 668.1, 582.9]
  Det 16: cls=2 (car), conf=0.5186, bbox=[526.3, 131.8, 585.1, 179.0]
  Det 17: cls=2 (car), conf=0.4938, bbox=[335.9, 20.1, 367.3, 44.7]
  Det 18: cls=7 (truck), conf=0.4201, bbox=[543.9, 179.2, 632.9, 252.2]
  Det 19: cls=24 (backpack), conf=0.3580, bbox=[481.8, 592.2, 534.2, 648.7]

Performance:
  batch_size: 1
  warmup: 3, runs: 5
  latency_ms_mean: 28.333
```

---

## 6. 性能数据

### 性能测试结果（300IDuo）

测试模型：YOLOv10-X
测试条件：输入 640x640，昇腾 NPU，CANN 8.5.0，MindSpore Lite 2.8.0

| 指标            | Mean (ms) |
|---------------|-----------|
| 延迟            | 28.33     |

> 注：测试图片为 city-streets.png，检测到 20 个目标。

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [YOLOv10 官方文档](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本教程遵循 YOLOv10 模型的许可证。
