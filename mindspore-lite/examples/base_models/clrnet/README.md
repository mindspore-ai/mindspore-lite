# CLRNet ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 CLRNet（车道线检测）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。CLRNet 基于 prior 的车道线检测，主干为 ResNet/HRNet，纯卷积结构，算子友好。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmcv-full      | 1.7.0     |
| mmdet          | 2.28.2    |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2 onnx==1.14.0 onnxruntime==1.16.0
# 从 Turoad/CLRNet 源码安装
```

权重与 config 来源：[Turoad/CLRNet](https://github.com/Turoad/CLRNet)。

---

## 2. 模型说明

```log
输入图像 (1,3,288,800) → backbone(ResNet) → neck → CLRHead(ROIGather) → 车道线 logits + coords
```

| 类型 | 名称          | 说明                       |
| ---- | ------------- | -------------------------- |
| 输入 | img           | \[1, 3, 288, 800] 前视图像 |
| 输出 | lane_logits   | 车道存在性（按 prior）     |
| 输出 | lane_coords   | 车道线坐标回归             |

> 导出为 head 原始输出；车道线 NMS/拟合留后处理，阶段 2 在 numpy 中实现并用于 ONNX↔MSLite 精度对齐。

---

## 3. ONNX 导出

```bash
cd examples/base_models/clrnet
python export_clrnet_onnx.py \
  --config /path/to/clrnet/config.py \
  --checkpoint /path/to/clrnet.pth \
  --output clrnet_onnx/clrnet.onnx --img-h 288 --img-w 800 --opset 17
```

产出：`clrnet_onnx/clrnet.onnx`

---

## 4. ONNX 推理

```bash
python infer_clrnet_onnx.py --model ./clrnet_onnx/clrnet.onnx --img-h 288 --img-w 800
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input, shape=(1, 3, 288, 800), seed=1024
  lane_logits: (1, N_priors)
  lane_coords: (1, N_priors, ...)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./clrnet_onnx/clrnet.onnx \
  --outputFile=./clrnet_onnx/clrnet_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。产出 `clrnet_ascend.mindir`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_clrnet_mslite.py \
  --model ./clrnet_onnx/clrnet_ascend.mindir \
  --device ascend --device-id 0 --img-h 288 --img-w 800
```

执行日志（warmup=5, runs=50, Ascend）：

```log
latency_ms_mean: 12.73
latency_ms_p99:  13.09
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：输入 `(1, 3, 320, 800)`，固定随机种子 1024，warmup=5；MSLite runs=50

| 后端           | 设备  | 延迟 mean (ms) | 延迟 p99 (ms) | 备注               |
| -------------- | ----- | -------------- | ------------- | ------------------ |
| ONNX Runtime   | CPU   | 795.47         | —             | 精度基准(fp32,CPU) |
| MindSpore Lite | 310P3 | 12.73          | 13.09         | force_fp16, Ascend |

**精度对齐**（seed=1024；ORT fp32 CPU vs MSLite fp16 Ascend，输出 `[1,192,78]`）：cos=0.999999 / mean_abs 5.5e-4。

> **权重/ONNX 来源说明**：阶段 2 实测使用 ModelScope `CVHub520/clrnet_tusimple_r18` 提供的**预导出 ONNX**（X-AnyLabeling 版，opset11，输入 `input [1,3,320,800]`，输出 `[1,192,78]`，n_offsets=72/max_lanes=5）。该包未提供 raw checkpoint，故跳过自导出，直接 ONNX→MindIR→Ascend 验证。如需从 Turoad/CLRNet 源码自导出，按第 3 节命令（需 CLRNet 仓库与对应 .pth）。

> **资源约束（阶段 2 验证必须遵守）**：内存/显存占用不得超过总量的 80%（310P3 每卡 44GB → 显存阈值 ~35.4GB；系统 RAM 同理）。每次验证前执行 `npu-smi info` + `free -h`，选最空闲卡 `--device-id`，单进程单卡，fp16 优先；推理脚本已内置 VmRSS/VmHWM 监控。若逼近阈值立即停止并降配置（降 batch / 拆子图）。

---

## 8. 常见问题

1. **现象**：导出报 head 输出字段名不符
   - **原因**：CLRNet/CLRHead 不同版本字段名（logits/lane_coords）不同
   - **解决方案**：按实际版本调整 `CLRNetWrapper.forward` 解包

2. **现象**：fp16 下车道坐标精度下降
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Turoad/CLRNet](https://github.com/Turoad/CLRNet)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 CLRNet 上游为准。
