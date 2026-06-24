# Deep-Video-Deblurring ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **Deep-Video-Deblurring**(视频去模糊)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。输入 7 帧模糊序列,输出清晰中心帧。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/shuochsu/Deep-Video-Deblurring.git ./deep_video_deblurring_src
# 按上游下载 deep_video_deblurring.pth
```

> 注意:上游含训练用损失,推理仅取前向网络;forward 签名需 Phase 2 核对。

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/deep_video_deblurring
python export_deep_video_deblurring_onnx.py \
  --repo-dir ./deep_video_deblurring_src \
  --model-file model.DeepVideoDeblurring --ckpt ./deep_video_deblurring.pth \
  --output-dir ./deep_video_deblurring_onnx --device cpu \
  --num-frames 7 --height 256 --width 256
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码目录 | `./deep_video_deblurring_src` |
| `--model-file` | 模型模块路径 | `model.DeepVideoDeblurring` |
| `--ckpt` | 权重 | `./deep_video_deblurring.pth` |
| `--num-frames` | 输入帧数 | `7` |
| `--height/--width` | 固定尺寸(须被 16 整除) | `256` / `256` |

```text
./deep_video_deblurring_onnx/
└── deep_video_deblurring.onnx   # 输入 seq [1,7,3,256,256],输出 sharp_frame [1,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_deep_video_deblurring_onnx.py \
  --onnx ./deep_video_deblurring_onnx/deep_video_deblurring.onnx \
  --input ./blurry.png --output ./sharp_onnx.png --device cpu
```

```log
（待 Phase 2 验证后填入真实输出：路径 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX \
  --modelFile=./deep_video_deblurring_onnx/deep_video_deblurring.onnx \
  --outputFile=./deep_video_deblurring_onnx/deep_video_deblurring \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

```ini
[acl_build_options]
input_format="ND"
input_shape="seq:1,7,3,256,256"
[acl_init_options]
ge.exec.precision_mode=force_fp16
[ascend_context]
plugin_custom_ops=All
```

```log
（待 Phase 2 验证后填入:CONVERT RESULT SUCCESS:0）
```

---

## 5. MindSpore Lite 推理

```bash
python infer_deep_video_deblurring_mslite.py \
  --mindir ./deep_video_deblurring_onnx/deep_video_deblurring.mindir \
  --input ./blurry.png --output ./sharp_mslite.png --device ascend --device-id 0
```

- 内存守护:建模型前检查 RAM/HBM,>80% 告警退出。
- 固定 shape:`--height/--width` 须与导出/转换一致。

```log
（待 Phase 2 验证后填入真实输出：内存预算报告 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 6. 性能数据

测试环境:Atlas 300I Duo(310P3),CANN 8.5.0,MindSpore Lite 2.9.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 推理(7帧,256×256,mean) | （待填） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. 无法导入模型 → 确认 `--repo-dir`/`--model-file` 正确;推理仅取前向网络。
2. shape 不匹配 → 统一 `--height/--width` 与 `config.ini`。
3. 内存守护告警 → 释放内存或降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/shuochsu/Deep-Video-Deblurring>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
