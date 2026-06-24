# DAIN ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **DAIN**(Depth-Aware Video Frame Interpolation)导出为单 ONNX,经 ONNX Runtime 验证后转换为 MindIR,在 **Atlas 300I Duo(310P3)** 上推理测速。DAIN 用深度估计 + deformable conv 做插帧(结构重、慢)。

> ⚠️ **DCN 阻塞告警**:DAIN 的 deformable conv 目前 `converter_lite` **不原生支持**,ONNX→MindIR 转换预计受阻。Phase 2 需自定义 AscendC 算子或导出侧等价改写。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / pillow | 3.11 / 2.x / 1.17+ / 1.17+ / 近期 / 近期 |
| CANN / mindspore-lite | 8.5.0 / 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
git clone https://github.com/baowenbo/DAIN.git ./dain_src
# 按上游下载 dain.pth(约 90MB+,含 DepthNet)
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/dain
python export_dain_onnx.py \
  --repo-dir ./dain_src --model-file model.DAIN --ckpt ./dain.pth \
  --output-dir ./dain_onnx --device cpu --height 256 --width 256
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--repo-dir` | 上游源码目录 | `./dain_src` |
| `--model-file` | 模型模块路径 | `model.DAIN` |
| `--ckpt` | 权重 | `./dain.pth` |
| `--height/--width` | 固定尺寸(须被 16 整除) | `256` / `256` |

```text
./dain_onnx/
└── dain.onnx   # 输入 img0/img1 [1,3,256,256],输出 mid_frame [1,3,256,256]
```

---

## 3. ONNX 推理

```bash
python infer_dain_onnx.py --onnx ./dain_onnx/dain.onnx \
  --img0 ./frame0.png --img1 ./frame1.png --output ./dain_mid_onnx.png --device cpu
```

```log
（待 Phase 2 验证后填入真实输出）
```

---

## 4. MindSpore Lite 转换

```bash
converter_lite --fmk=ONNX --modelFile=./dain_onnx/dain.onnx \
  --outputFile=./dain_onnx/dain --optimize=ascend_oriented \
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
（⚠️ 预计因 DCN 算子不支持而转换受阻;Phase 2 需自定义 AscendC 算子或等价改写）
```

---

## 5. MindSpore Lite 推理

```bash
python infer_dain_mslite.py --mindir ./dain_onnx/dain.mindir \
  --img0 ./frame0.png --img1 ./frame1.png --output ./dain_mid_mslite.png \
  --height 256 --width 256 --device ascend --device-id 0
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
| 推理(256×256,mean) | （待填,依赖 DCN 解决） |
| 进程 RSS 峰值 (MB) | （待填） |

## 7. 常见问题

1. **converter 报 deformable conv / DCN 不支持** → 最高优先级阻塞;需自定义 AscendC 算子或导出侧等价改写。
2. 无法导入 DAIN → 确认 `--repo-dir`/`--model-file`(DAIN 含 DepthNet 等子模块,构造复杂)。
3. shape 不匹配 → 统一 `--height/--width` 与 `config.ini`。
4. 内存守护告警 → 释放内存或降分辨率。

## 8. 参考资源

- 上游仓库:<https://github.com/baowenbo/DAIN>
- MindSpore Lite:<https://www.mindspore.cn/lite>

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证;上游模型许可证以其仓库为准。
