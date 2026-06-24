# TOFlow ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **TOFlow**(Task-Oriented Flow,视频超分)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。TOFlow 用 SpyNet 光流对齐多帧低清,输出 4x 超分中心帧。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/anchen1011/toflow.git ./toflow_src
# 按上游/BasicSR 下载 toflow_x4.pth
```

> 注意:TOFlow 也可从 BasicSR/mmagic 导入,`--model-file` 指定,Phase 2 用实际源码核对。

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/toflow
python export_toflow_onnx.py \
  --repo-dir ./toflow_src --model-file model.TOFlow --ckpt ./toflow_x4.pth \
  --output-dir ./toflow_onnx --device cpu \
  --num-frames 7 --lr-height 64 --lr-width 64
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码目录 | `./toflow_src` |
| `--model-file` | 模型模块路径(dotted) | `model.TOFlow` |
| `--ckpt` | 权重 | `./toflow_x4.pth` |
| `--num-frames` | 输入低清帧数 | `7` |
| `--lr-height/--lr-width` | 低清尺寸(须被 32 整除) | `64` / `64` |

```text
./toflow_onnx/
└── toflow.onnx   # 输入 lr_seq [1,7,3,64,64],输出 sr_frame [1,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_toflow_onnx.py --onnx ./toflow_onnx/toflow.onnx \
  --input ./lr.png --output ./sr_onnx.png --device cpu
```

```log
（待 Phase 2 验证后填入真实输出：路径 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./toflow_onnx/toflow.onnx \
  --outputFile=./toflow_onnx/toflow --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

```ini
[acl_build_options]
input_format="ND"
input_shape="lr_seq:1,7,3,64,64"
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
python infer_toflow_mslite.py --mindir ./toflow_onnx/toflow.mindir \
  --input ./lr.png --output ./sr_mslite.png --device ascend --device-id 0
```

- 内存守护:建模型前检查 RAM/HBM,>80% 告警退出。
- 固定 shape:`--lr-height/--lr-width` 须与导出/转换一致。

```log
（待 Phase 2 验证后填入真实输出：内存预算报告 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 6. 性能数据

测试环境:Atlas 300I Duo(310P3),CANN 8.5.0,MindSpore Lite 2.9.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 推理(7帧,64×64→256×256,mean) | （待填） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. 无法导入 TOFlow → 确认 `--repo-dir` 与 `--model-file` 路径正确(或改用 BasicSR/mmagic 导入)。
2. shape 不匹配 → 统一 `--lr-height/--lr-width`(须被 32 整除)与 `config.ini`。
3. 内存守护告警 → 释放内存或降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/anchen1011/toflow>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
