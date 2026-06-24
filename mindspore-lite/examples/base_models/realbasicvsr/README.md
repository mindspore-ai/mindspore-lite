# RealBasicVSR ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **RealBasicVSR**(CVPR2021 视频超分)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。RealBasicVSR 用双向循环传播 + SpyNet 光流对齐,4x 超分。固定 N 帧输入,trace 时展开循环(无显式 recurrent 状态 I/O)。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
# RealBasicVSR 架构类来自 mmagic(或 clone 上游)
git clone https://github.com/open-mmlab/mmagic.git ./mmagic_src
# 按上游下载 realbasicvsr_x4.pth
```

> 注意:`--model-file` 的 dotted 路径随 mmagic 版本变化,Phase 2 用实际环境核对。

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/realbasicvsr
python export_realbasicvsr_onnx.py \
  --repo-dir ./mmagic_src \
  --model-file mmagic.models.realbasicvsr_net.RealBasicVSRNet \
  --ckpt ./realbasicvsr_x4.pth \
  --output-dir ./realbasicvsr_onnx --device cpu \
  --num-frames 10 --lr-height 64 --lr-width 64
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | mmagic/上游源码目录(用于 import) | `./mmagic_src` |
| `--model-file` | 架构类 dotted 路径 | `mmagic.models.realbasicvsr_net.RealBasicVSRNet` |
| `--ckpt` | 权重 | `./realbasicvsr_x4.pth` |
| `--num-frames` | 固定帧数(展开循环) | `10` |
| `--lr-height/--lr-width` | 低清尺寸(须被 4 整除) | `64` / `64` |

```text
./realbasicvsr_onnx/
└── realbasicvsr.onnx   # 输入 lr_seq [1,10,3,64,64],输出 sr_seq [1,10,3,256,256]
```

固定 shape 约束:改 N/分辨率须同步 `config.ini` 重新导出+转换。N 增大会线性增大图规模与显存。

---

## 3. ONNX 推理

```bash
python infer_realbasicvsr_onnx.py --onnx ./realbasicvsr_onnx/realbasicvsr.onnx \
  --input ./lr.png --output ./sr_onnx.png --device cpu
```

```log
（待 Phase 2 验证后填入真实输出：路径 + latency_ms_mean/p50 + proc_rss_mb）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./realbasicvsr_onnx/realbasicvsr.onnx \
  --outputFile=./realbasicvsr_onnx/realbasicvsr --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

```ini
[acl_build_options]
input_format="ND"
input_shape="lr_seq:1,10,3,64,64"
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
python infer_realbasicvsr_mslite.py --mindir ./realbasicvsr_onnx/realbasicvsr.mindir \
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
| 推理(10帧,64×64→256×256,mean) | （待填） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. 无法导入 RealBasicVSRNet → 确认 mmagic 已装或 `--repo-dir` 正确,`--model-file` 路径匹配版本。
2. 导出图过大/慢 → 降低 `--num-frames`(循环展开规模)。
3. shape 不匹配 → 统一 N/分辨率与 `config.ini`。
4. 内存守护告警 → 释放内存或降帧/降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/open-mmlab/mmagic>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
