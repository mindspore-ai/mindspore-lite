# DriveVLM ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 DriveVLM（面向自动驾驶的视觉语言大模型，7B+ LLM）的 **视觉编码器** 导出为 ONNX，转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理。

> **架构**：DriveVLM = Vision encoder + LLM。本 scaffold 导出 vision encoder（图像 → image_embeds）；**LLM 部分（prefill + decode）复用 `qwen2_7b` 的导出与推理链路**（见阶段 2 计划）。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| transformers   | >=4.40    |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 transformers onnx==1.14.0 onnxruntime==1.16.0 --index-url https://download.pytorch.org/whl/cpu
```

权重来源：[tsinghua-mars-lab/DriveVLM](https://github.com/tsinghua-mars-lab/DriveVLM)（HF 模型 ID 或本地路径）。

---

## 2. 模型说明

```log
驾驶场景图像 (1,3,336,336) → Vision encoder(ViT/SigLIP) → image_embeds (1,N,D)
                                                         ↓
                                              投影后拼接进 LLM（复用 qwen2_7b 链路）
```

| 类型 | 名称           | Shape              | 说明                |
| ---- | -------------- | ------------------ | ------------------- |
| 输入 | pixel_values   | \[1, 3, 336, 336]  | 单帧驾驶场景图像    |
| 输出 | image_embeds   | \[1, N, D]         | 视觉 token 嵌入     |

> N 为 patch token 数，D 为隐藏维度。LLM 部分输入 = text token embeds + image_embeds，复用 [qwen2_7b](../qwen2_7b/README.md) 的 prefill/decode 拆分与 KV cache 链路。

---

## 3. ONNX 导出

```bash
cd examples/base_models/drivlvlm
python export_drivlvlm_onnx.py \
  --model-id /path/to/DriveVLM \
  --output drivlvlm_onnx/drivlvlm_vision.onnx --opset 17
```

产出：`drivlvlm_onnx/drivlvlm_vision.onnx`

---

## 4. ONNX 推理

```bash
python infer_drivlvlm_onnx.py --model ./drivlvlm_onnx/drivlvlm_vision.onnx
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input, shape=(1, 3, 336, 336), seed=1024
  image_embeds: (1, N, D)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./drivlvlm_onnx/drivlvlm_vision.onnx \
  --outputFile=./drivlvlm_onnx/drivlvlm_vision_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

> LLM 部分转换参考 `qwen2_7b`（force_fp32，prefill/decode 两个 config）。

---

## 6. MindSpore Lite 推理

```bash
python infer_drivlvlm_mslite.py \
  --model ./drivlvlm_onnx/drivlvlm_vision_ascend.mindir \
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
测试条件：输入 `(1,3,336,336)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 备注                  |
| -------------- | ----- | --------------------- | --------------------- |
| ONNX Runtime   | CPU   | TBD                   | 精度基准（vision）    |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | force_fp16（vision）  |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。DriveVLM 7B LLM 部分体量大，阶段 2 须严格监控；验证前 `npu-smi info` + `free -h`，单进程单卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：LLM 部分如何部署
   - **解决方案**：复用 [qwen2_7b](../qwen2_7b/README.md) 的 prefill/decode 拆分 + KV cache 链路，阶段 2 按 qwen 流程转换与推理

2. **现象**：trust_remote_code 模型加载失败
   - **解决方案**：确保 transformers 版本满足；使用本地路径

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [tsinghua-mars-lab/DriveVLM](https://github.com/tsinghua-mars-lab/DriveVLM)
- [qwen2_7b 参考链路](../qwen2_7b/README.md)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 DriveVLM 上游为准。
