# AMT_S ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **AMT_S**(Real-Time Intermediate Flow Estimation,视频插帧)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。AMT_S 用中间光流估计做插帧,核心算子为光流估计 + `flow_warp`(基于 `grid_sample` 的双线性采样)。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/MCG-NKU/AMT.git ./AMT_S
# 按上游 README 下载 amt-s.pth(放 train_log 或当前目录)
```

> 注意:AMT_S 版本模块路径不同(v2/v3/v4/lite),用 `--model-file` 指定,如 `model.AMT_S`。Phase 2 用实际源码核对。

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/amt_s
python export_amt_s_onnx.py \
  --repo-dir ./AMT_S --model-file model.AMT_S \
  --ckpt ./amt-s.pth \
  --output-dir ./amt_s_onnx --device cpu --height 256 --width 256
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码根目录 | `./AMT_S` |
| `--model-file` | 模型模块路径(dotted) | `model.AMT_S` |
| `--ckpt` | 权重 | `./amt-s.pth` |
| `--height/--width` | 固定尺寸(须被 32 整除) | `256` / `256` |

```text
./amt_s_onnx/
└── amt_s.onnx   # 输入 img0/img1 [1,3,256,256],输出 mid_frame [1,3,256,256]
```

固定 shape 约束:改分辨率须同步 `config.ini` 重新导出+转换。timestep 固定 0.5(中点)。

---

## 3. ONNX 推理

```bash
python infer_amt_s_onnx.py --onnx ./amt_s_onnx/amt_s.onnx \
  --img0 ./frame0.png --img1 ./frame1.png --output ./amt_s_mid_onnx.png \
  --height 256 --width 256 --device cpu
```

```log
（待 Phase 2 验证后填入真实输出：路径 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./amt_s_onnx/amt_s.onnx \
  --outputFile=./amt_s_onnx/amt_s --optimize=ascend_oriented \
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
CONVERT RESULT SUCCESS:0   # amt_s.mindir fp16
```

---

## 5. MindSpore Lite 推理

```bash
python infer_amt_s_mslite.py --mindir ./amt_s_onnx/amt_s.mindir \
  --img0 ./frame0.png --img1 ./frame1.png --output ./amt_s_mid_mslite.png \
  --height 256 --width 256 --device ascend --device-id 0
```

- 内存守护:建模型前检查 RAM/HBM,>80% 告警退出。
- 固定 shape:`--height/--width` 须与导出/转换一致。

```log
[memory-budget] RAM 27.1% / NPU0 HBM 3.1%
  latency_ms_mean: 95.820  p50: 95.545
  proc_rss_mb: 1099 (hwm=1118)
```

---

## 6. 性能数据

测试环境:Atlas 300I Duo(310P3),CANN 8.5.0,MindSpore Lite 2.9.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 推理(256×256,mean) | 95.820 |
| 推理(256×256,p50) | 95.545 |
| 进程 RSS 峰值 (MB) | 1118 |
| NPU0 HBM 峰值 (MB) | ~1375 (3.1%/44280) |

## 7. 常见问题

1. 无法导入 AMT_S → 确认 `--repo-dir` 为 AMT_S 根目录,`--model-file` 与权重版本匹配。
2. converter 报 GridSample 相关 → 确认 mindspore-lite 版本支持 grid_sample;必要时降 opset。
3. shape 不匹配 → 统一 `--height/--width`(须被 32 整除)与 `config.ini`。
4. 内存守护告警 → 释放内存或降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/MCG-NKU/AMT>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
