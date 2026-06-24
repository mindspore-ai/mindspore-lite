# STDF ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **STDF**(Spatio-Temporal Deblocking Filter,视频压缩伪影去除)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/RyanXingHL/STDF.git ./stdf_src
# 按上游 README 下载 stdf.pth
```

> 注意:STDF 上游 forward 签名/帧数(默认 7)需在 Phase 2 用实际源码核对。

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/stdf
python export_stdf_onnx.py \
  --repo-dir ./stdf_src --ckpt ./stdf.pth \
  --output-dir ./stdf_onnx --device cpu --height 256 --width 256 --num-frames 7
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码目录(含 model.py) | `./stdf_src` |
| `--ckpt` | 权重 | `./stdf.pth` |
| `--height/--width` | 固定尺寸(须被 16 整除) | `256` / `256` |
| `--num-frames` | 输入帧数 | `7` |

```text
./stdf_onnx/
└── stdf.onnx   # 输入 seq [1,7,3,256,256],输出 enhanced [1,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_stdf_onnx.py --onnx ./stdf_onnx/stdf.onnx \
  --input ./lowquality.png --output ./enhanced_onnx.png \
  --height 256 --width 256 --device cpu
```

```log
（待 Phase 2 验证后填入真实输出：路径 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./stdf_onnx/stdf.onnx \
  --outputFile=./stdf_onnx/stdf --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
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
python infer_stdf_mslite.py --mindir ./stdf_onnx/stdf.mindir \
  --input ./lowquality.png --output ./enhanced_mslite.png \
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
| 推理(256×256,7帧,mean) | （待填） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. 无法导入 STDF → 确认已 clone RyanXingHL/STDF 且 `--repo-dir` 含 `model.py`。
2. shape 不匹配 → 统一 `--height/--width` 与 `config.ini`,重导出+转换。
3. 内存守护告警 → 释放内存或降帧/降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/RyanXingHL/STDF>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
