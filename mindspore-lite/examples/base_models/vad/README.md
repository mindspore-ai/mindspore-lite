# VAD ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 VAD（矢量化端到端自动驾驶）导出为 ONNX，并转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理部署。VAD 用 BEVFormer 风格编码器 + 矢量化地图/agent query + planning head 直接输出自车规划轨迹。

> **阶段 2 风险点**：BEV 编码器含 Temporal Deformable Attention（同 BEVFormer），阶段 2 大概率阻塞，需 AscendC 可变形注意力算子。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmdet / mmcv-full | 2.28.2 / 1.7.0 |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2 onnx==1.14.0 onnxruntime==1.16.0
```

权重来源：[hustvl/VAD](https://github.com/hustvl/VAD)（提供 `model.VAD`）。

---

## 2. 模型说明

```log
多视图图像 (1,6,3,320,800) → BEV encoder(deformable) → 矢量化 query → planning head → ego_traj (1,12,2) + scores
```

| 类型 | 名称      | Shape                | 说明                |
| ---- | --------- | -------------------- | ------------------- |
| 输入 | imgs      | \[1, 6, 3, 320, 800] | 6 路相机图像        |
| 输出 | ego_traj  | \[1, 12, 2]          | 自车未来 12 步轨迹  |
| 输出 | scores    | \[1, K]              | 多模态规划概率      |

---

## 3. ONNX 导出

```bash
cd examples/base_models/vad
python export_vad_onnx.py \
  --model-module model --model-class VAD \
  --checkpoint /path/to/vad.pth \
  --output vad_onnx/vad.onnx --opset 17
```

产出：`vad_onnx/vad.onnx`（含 Custom BEV/deformable 节点）

---

## 4. ONNX 推理

> Custom 算子 ORT 无法执行，`infer_vad_onnx.py` 仅用于结构检查；推理走 MindIR。

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./vad_onnx/vad.onnx \
  --outputFile=./vad_onnx/vad_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp32`。

> 阶段 2 在 deformable 算子处大概率失败 → 记录暂搁。

---

## 6. MindSpore Lite 推理

```bash
python infer_vad_mslite.py \
  --model ./vad_onnx/vad_ascend.mindir \
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

1. **现象**：转换报 deformable 算子不支持
   - **解决方案**：同 BEVFormer，需 AscendC 可变形注意力；暂无则记录暂搁

2. **现象**：fp16 规划轨迹异常
   - **解决方案**：保持 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [hustvl/VAD](https://github.com/hustvl/VAD)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 VAD 上游为准。
