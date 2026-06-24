# RIFE_LITE ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **RIFE_LITE**(Real-Time Intermediate Flow Estimation,视频插帧)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。RIFE_LITE 用中间光流估计做插帧,核心算子为光流估计 + `flow_warp`(基于 `grid_sample` 的双线性采样)。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/hzwer/RIFE.git ./RIFE_LITE
# 按上游 README 下载 RIFE_Lite.pth(放 train_log 或当前目录)
```

> 注意:RIFE_LITE 版本模块路径不同(v2/v3/v4/lite),用 `--model-file` 指定,如 `model.RIFE_HDv3`。Phase 2 用实际源码核对。

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/rife_lite
python export_rife_lite_onnx.py \
  --repo-dir ./RIFE_LITE --model-file model.RIFE_HDv3 \
  --ckpt ./RIFE_Lite.pth \
  --output-dir ./rife_lite_onnx --device cpu --height 256 --width 256
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码根目录 | `./RIFE_LITE` |
| `--model-file` | 模型模块路径(dotted) | `model.RIFE_HDv3` |
| `--ckpt` | 权重 | `./RIFE_Lite.pth` |
| `--height/--width` | 固定尺寸(须被 32 整除) | `256` / `256` |

```text
./rife_lite_onnx/
└── rife_lite.onnx   # 输入 img0/img1 [1,3,256,256],输出 mid_frame [1,3,256,256]
```

固定 shape 约束:改分辨率须同步 `config.ini` 重新导出+转换。timestep 固定 0.5(中点)。

---

## 3. ONNX 推理

```bash
python infer_rife_lite_onnx.py --onnx ./rife_lite_onnx/rife_lite.onnx \
  --img0 ./frame0.png --img1 ./frame1.png --output ./rife_lite_mid_onnx.png \
  --height 256 --width 256 --device cpu
```

```log
（待 Phase 2 验证后填入真实输出：路径 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./rife_lite_onnx/rife_lite.onnx \
  --outputFile=./rife_lite_onnx/rife_lite --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

```ini
[acl_build_options]
input_format="ND"
input_shape="img0:1,3,256,256;img1:1,3,256,256"
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
python infer_rife_lite_mslite.py --mindir ./rife_lite_onnx/rife_lite.mindir \
  --img0 ./frame0.png --img1 ./frame1.png --output ./rife_lite_mid_mslite.png \
  --height 256 --width 256 --device ascend --device-id 0
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
| 推理(256×256,mean) | （待填） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. 无法导入 RIFE_LITE → 确认 `--repo-dir` 为 RIFE_LITE 根目录,`--model-file` 与权重版本匹配。
2. converter 报 GridSample 相关 → 确认 mindspore-lite 版本支持 grid_sample;必要时降 opset。
3. shape 不匹配 → 统一 `--height/--width`(须被 32 整除)与 `config.ini`。
4. 内存守护告警 → 释放内存或降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/hzwer/RIFE>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
