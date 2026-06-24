# VRT ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **VRT**(Video Restoration Transformer)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。VRT 用大量自注意力做视频超分/修复,显存占用重。

> ⚠️ **DCN 阻塞告警(修正)**:VRT 的 `network_vrt.py` 含 `DCNv2PackFlowGuided`/`ModulatedDeformConv`,即 Transformer **+ DCN**;DCN converter 不支持,转换会受阻(需自定义 AscendC 算子)。此前归为"纯 Transformer 可转换"有误。另:自注意力图大,300I Duo(44GB/卡)需小窗口/分块,在内存<80% 守护下运行;`--num-frames`/分辨率需调小。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/jingyunliang/vrt.git ./vrt_src
# 按上游下载 vrt_x4.pth
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/vrt
python export_vrt_onnx.py \
  --repo-dir ./vrt_src --model-file archs.vrt_arch.VRT --ckpt ./vrt_x4.pth \
  --output-dir ./vrt_onnx --device cpu \
  --num-frames 7 --lr-height 64 --lr-width 64
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码目录 | `./vrt_src` |
| `--model-file` | 架构类 dotted 路径 | `archs.vrt_arch.VRT` |
| `--ckpt` | 权重 | `./vrt_x4.pth` |
| `--num-frames` | 输入帧数(显存敏感,必要时调小) | `7` |
| `--lr-height/--lr-width` | 低清尺寸(须被 4 整除) | `64` / `64` |

```text
./vrt_onnx/
└── vrt.onnx   # 输入 lr_seq [1,7,3,64,64],输出 sr_frame [1,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_vrt_onnx.py --onnx ./vrt_onnx/vrt.onnx \
  --input ./lr.png --output ./sr_onnx.png --device cpu
```

```log
（待 Phase 2 验证后填入真实输出）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./vrt_onnx/vrt.onnx \
  --outputFile=./vrt_onnx/vrt --optimize=ascend_oriented \
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
python infer_vrt_mslite.py --mindir ./vrt_onnx/vrt.mindir \
  --input ./lr.png --output ./sr_mslite.png --device ascend --device-id 0
```

- 内存守护:建模型前检查 RAM/HBM,>80% 告警退出。
- 固定 shape:`--num-frames/--lr-height/--lr-width` 须与导出/转换一致。

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

1. 导出/推理 OOM → VRT 自注意力显存重;调小 `--num-frames` 或分辨率,或分块/滑窗。
2. 无法导入 VRT → 确认 `--repo-dir`/`--model-file` 路径(VRT 构造参数多,可能需 config-based build)。
3. shape 不匹配 → 统一帧数/分辨率与 `config.ini`。
4. 内存守护告警 → 释放内存或降帧/降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/jingyunliang/vrt>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
