# EDVR-L ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **EDVR-L**(Video Restoration,DCNv1 对齐 + 时空注意力,中等规模)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。输入 7 帧低清,输出 4x 超分中心帧。EDVR-L 同源仅换规模。

> ⚠️ **DCN 阻塞告警**:EDVR 的 deformable conv(DCNv1)目前 `converter_lite` **不原生支持**,ONNX→MindIR 转换预计受阻。Phase 2 需自定义 AscendC 算子或导出侧等价改写。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/open-mmlab/mmagic.git ./mmagic_src
# 按上游下载 edvr_l_x4.pth
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/edvr_l
python export_edvr_l_onnx.py \
  --repo-dir ./mmagic_src --model-file mmagic.models.edvr_net.EDVRNet \
  --ckpt ./edvr_l_x4.pth --output-dir ./edvr_l_onnx --device cpu \
  --num-frames 7 --lr-height 64 --lr-width 64
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | mmagic/上游源码目录 | `./mmagic_src` |
| `--model-file` | 架构类 dotted 路径 | `mmagic.models.edvr_net.EDVRNet` |
| `--ckpt` | 权重 | `./edvr_l_x4.pth` |
| `--num-frames` | 输入帧数 | `7` |
| `--lr-height/--lr-width` | 低清尺寸(须被 4 整除) | `64` / `64` |

```text
./edvr_l_onnx/
└── edvr_l.onnx   # 输入 lr_seq [1,7,3,64,64],输出 sr_frame [1,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_edvr_l_onnx.py --onnx ./edvr_l_onnx/edvr_l.onnx \
  --input ./lr.png --output ./sr_onnx.png --device cpu
```

```log
（待 Phase 2 验证后填入真实输出）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./edvr_l_onnx/edvr_l.onnx \
  --outputFile=./edvr_l_onnx/edvr_l --optimize=ascend_oriented \
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
（⚠️ 预计因 DCNv1 算子不支持而转换受阻;Phase 2 需自定义 AscendC 算子或等价改写）
```

---

## 5. MindSpore Lite 推理

```bash
python infer_edvr_l_mslite.py --mindir ./edvr_l_onnx/edvr_l.mindir \
  --input ./lr.png --output ./sr_mslite.png --device ascend --device-id 0
```

- 内存守护:建模型前检查 RAM/HBM,>80% 告警退出。

```log
（待 DCN 问题解决、MindIR 生成后填入）
```

---

## 6. 性能数据

测试环境:Atlas 300I Duo(310P3),CANN 8.5.0,MindSpore Lite 2.9.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 推理(7帧,64×64→256×256,mean) | （待填,依赖 DCN 解决） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. **converter 报 deformable conv / DCN 不支持** → 最高优先级阻塞;需自定义 AscendC 算子或导出侧等价改写。
2. 无法导入 EDVRNet → 确认 mmagic 版本与 `--model-file` 路径(EDVRNet 构造参数较多,Phase 2 可能需 config-based build)。
3. shape 不匹配 → 统一帧数/分辨率与 `config.ini`。
4. 内存守护告警 → 释放内存或降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/open-mmlab/mmagic> / <https://github.com/xinntao/EDVR>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
