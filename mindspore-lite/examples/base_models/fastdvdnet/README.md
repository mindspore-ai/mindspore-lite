# FastDVDnet ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **FastDVDnet**(Fast and Accurate Video Denoising)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。FastDVDnet 用 5 帧含噪序列 + 噪声 sigma 去除中心帧噪声,为小型 CNN,结构干净。

> 上游 `forward(x, noise_map)`:x 为 flattened `[B, N*C, H, W]`、noise_map `[B, 1, H, W]`。本导出 wrapper 接收 seq `[B, N, 3, H, W]` + sigma `[B, 1]`,内部展开匹配上游接口。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/m-tassano/fastdvdnet.git ./fastdvdnet_src
# 按上游 README 下载 fastdvdnet.pth
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/fastdvdnet
python export_fastdvdnet_onnx.py \
  --repo-dir ./fastdvdnet_src --ckpt ./fastdvdnet.pth \
  --output-dir ./fastdvdnet_onnx --device cpu \
  --height 256 --width 256 --num-frames 5
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码目录(含 models.py) | `./fastdvdnet_src` |
| `--ckpt` | 权重 | `./fastdvdnet.pth` |
| `--height/--width` | 固定尺寸(须被 16 整除) | `256` / `256` |
| `--num-frames` | 输入帧数 | `5` |

```text
./fastdvdnet_onnx/
└── fastdvdnet.onnx   # 输入 seq [1,5,3,256,256] + noise_sigma [1,1],输出 denoised [1,3,256,256]
```

固定 shape 约束:改分辨率须同步 `config.ini` 重新导出+转换。

---

## 3. ONNX 推理

```bash
python infer_fastdvdnet_onnx.py \
  --onnx ./fastdvdnet_onnx/fastdvdnet.onnx \
  --input ./noisy.png --output ./denoised_onnx.png \
  --height 256 --width 256 --noise-sigma 5 --device cpu
```

```log
[onnx] saved(CPU 参考); Ascend 见 §5
```

> 说明:精度对齐需真实权重(上游 Dropbox/仓库);当前性能为真实架构实测(性能与权重无关,数值有效)。

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX \
  --modelFile=./fastdvdnet_onnx/fastdvdnet.onnx \
  --outputFile=./fastdvdnet_onnx/fastdvdnet \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

```ini
[acl_build_options]
input_format="ND"
input_shape="seq:1,5,3,256,256;noise_sigma:1,1"
[acl_init_options]
ge.exec.precision_mode=force_fp16
[ascend_context]
plugin_custom_ops=All
```

```log
CONVERT RESULT SUCCESS:0   # fastdvdnet.mindir ~6MB fp16
```

---

## 5. MindSpore Lite 推理

```bash
python infer_fastdvdnet_mslite.py \
  --mindir ./fastdvdnet_onnx/fastdvdnet.mindir \
  --input ./noisy.png --output ./denoised_mslite.png \
  --height 256 --width 256 --noise-sigma 5 \
  --device ascend --device-id 0
```

- 内存守护:建模型前检查 RAM/HBM,>80% 告警退出。
- 固定 shape:`--height/--width` 须与导出/转换一致。

```log
[memory-budget] RAM 28.9% / NPU0 HBM 3.1%
  latency_ms_mean: 11.515  p50: 8.460
  proc_rss_mb: 1069 (hwm=1073)
```

---

## 6. 性能数据

测试环境:Atlas 300I Duo(310P3),CANN 8.5.0,MindSpore Lite 2.9.0。纯 CNN,转换无障碍。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 推理(256×256,5帧,mean) | 11.515 |
| 推理(256×256,5帧,p50) | 8.460 |
| 进程 RSS 峰值 (MB) | 1073 |
| NPU0 HBM 峰值 (MB) | ~1366 (3.1% / 44280) |

---

## 7. 常见问题

1. 无法导入 FastDVDnet → 确认已 clone m-tassano/fastdvdnet 且 `--repo-dir` 含 `models.py`(类在 models.py)。
2. shape 不匹配 → 统一 `--height/--width` 与 `config.ini`,重导出+转换。
3. 内存守护告警 → 释放内存或降帧/降分辨率。

---

## 8. 参考资源

- 上游仓库:<https://github.com/m-tassano/fastdvdnet>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
