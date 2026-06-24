# FLAVR ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **FLAVR**(Spatio-Temporal Deblocking Filter,视频压缩伪影去除)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/tarun005/FLAVR.git ./flavr_src
# 按上游 README 下载 flavr.pth
```

> 注意:FLAVR 上游 forward 签名/帧数(默认 7)需在 Phase 2 用实际源码核对。

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/flavr
python export_flavr_onnx.py \
  --repo-dir ./flavr_src --ckpt ./flavr.pth \
  --output-dir ./flavr_onnx --device cpu --height 256 --width 256 --num-frames 4
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码目录(含 model.py) | `./flavr_src` |
| `--ckpt` | 权重 | `./flavr.pth` |
| `--height/--width` | 固定尺寸(须被 16 整除) | `256` / `256` |
| `--num-frames` | 输入帧数 | `4` |

```text
./flavr_onnx/
└── flavr.onnx   # 输入 seq [1,4,3,256,256],输出 interp [1,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_flavr_onnx.py --onnx ./flavr_onnx/flavr.onnx \
  --input ./lowquality.png --output ./interp_onnx.png \
  --height 256 --width 256 --device cpu
```

```log
（待 Phase 2 验证后填入真实输出：路径 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./flavr_onnx/flavr.onnx \
  --outputFile=./flavr_onnx/flavr --optimize=ascend_oriented \
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
python infer_flavr_mslite.py --mindir ./flavr_onnx/flavr.mindir \
  --input ./lowquality.png --output ./interp_mslite.png \
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

> ⚠️ **实测转换阻塞(2026-06-19)**:FLAVR 解码器的 `ConvTranspose3D` 上采样 converter 报错 `Conv2dTransposeFusion: dilation must be 2, but got 5`。Conv3D 本身可转,瓶颈在 3D 转置卷积上采样;需改写为 `F.interpolate + Conv` 后重导出。性能数据待该问题解决后填入。

## 7. 常见问题

1. 无法导入 FLAVR → 确认已 clone tarun005/FLAVR 且 `--repo-dir` 含 `model.py`。
2. shape 不匹配 → 统一 `--height/--width` 与 `config.ini`,重导出+转换。
3. 内存守护告警 → 释放内存或降帧/降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/tarun005/FLAVR>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
