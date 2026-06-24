# BasicVSR++ ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **BasicVSR++**(CVPR2022 视频超分,DCNv2 对齐 + 二阶传播)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。输入 N 帧低清,输出 4x 超分序列。

> ⚠️ **DCN 阻塞告警**:BasicVSR++ 的 deformable conv(DCNv2)目前 `converter_lite` **不原生支持**,ONNX→MindIR 转换预计会受阻。Phase 2 需自定义 AscendC 算子或导出侧等价改写后才能完成转换。本目录提供完整的导出/推理脚本骨架。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/open-mmlab/mmagic.git ./mmagic_src
# 按上游下载 basicvsr_pp_x4.pth
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/basicvsr_pp
python export_basicvsr_pp_onnx.py \
  --repo-dir ./mmagic_src \
  --model-file mmagic.models.basicvsr_pp_net.BasicVSRPlusPlusNet \
  --ckpt ./basicvsr_pp_x4.pth \
  --output-dir ./basicvsr_pp_onnx --device cpu \
  --num-frames 10 --lr-height 64 --lr-width 64
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | mmagic/上游源码目录 | `./mmagic_src` |
| `--model-file` | 架构类 dotted 路径 | `mmagic.models.basicvsr_pp_net.BasicVSRPlusPlusNet` |
| `--ckpt` | 权重 | `./basicvsr_pp_x4.pth` |
| `--num-frames` | 固定帧数 | `10` |
| `--lr-height/--lr-width` | 低清尺寸(须被 4 整除) | `64` / `64` |

```text
./basicvsr_pp_onnx/
└── basicvsr_pp.onnx   # 输入 lr_seq [1,10,3,64,64],输出 sr_seq [1,10,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_basicvsr_pp_onnx.py --onnx ./basicvsr_pp_onnx/basicvsr_pp.onnx \
  --input ./lr.png --output ./sr_onnx.png --device cpu
```

```log
（待 Phase 2 验证后填入真实输出）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./basicvsr_pp_onnx/basicvsr_pp.onnx \
  --outputFile=./basicvsr_pp_onnx/basicvsr_pp --optimize=ascend_oriented \
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
（⚠️ 预计因 DCNv2 算子不支持而转换受阻;Phase 2 需自定义 AscendC 算子或等价改写）
```

---

## 5. MindSpore Lite 推理

```bash
python infer_basicvsr_pp_mslite.py --mindir ./basicvsr_pp_onnx/basicvsr_pp.mindir \
  --input ./lr.png --output ./sr_mslite.png --device ascend --device-id 0
```

- 内存守护:建模型前检查 RAM/HBM,>80% 告警退出。
- 固定 shape:`--num-frames/--lr-height/--lr-width` 须与导出/转换一致。

```log
（待 DCN 问题解决、MindIR 生成后填入）
```

---

## 6. 性能数据

测试环境:Atlas 300I Duo(310P3),CANN 8.5.0,MindSpore Lite 2.9.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 推理(10帧,64×64→256×256,mean) | （待填,依赖 DCN 解决） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. **converter 报 deformable conv / DCN 不支持** → 当前最高优先级阻塞;需自定义 AscendC 算子(参考 [[cloud-whl-ascend-custom-op-integration]] ascend_ops 扩展点)或导出侧等价改写。
2. 无法导入 BasicVSRPlusPlusNet → 确认 mmagic 版本与 `--model-file` 路径。
3. 导出图过大 → 降低 `--num-frames`。
4. 内存守护告警 → 释放内存或降帧/降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/open-mmlab/mmagic>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
