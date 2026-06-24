# CAIN ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 **CAIN**(Channel Attention Is All You Need for Video Frame Interpolation,CVPR2020)导出为单 ONNX,使用 ONNX Runtime 验证推理结果,并将 ONNX 转换为 MindSpore Lite MindIR 后在 **Atlas 300I Duo(310P3)** 上推理与测速。

CAIN 为纯 CNN 视频插帧模型:输入两帧 `img0`/`img1`,输出中间帧 `mid_frame`,无光流计算,结构干净、迁移简单。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.x(CPU 即可,仅用于导出/ONNX 验证) |
| onnx | 1.17+ |
| onnxruntime | 1.17+ |
| numpy / pillow | 近期稳定版 |
| CANN | 8.5.0 |
| mindspore-lite | 2.9.0 |

```bash
pip install torch onnx onnxruntime numpy pillow
```

### 获取模型权重与源码

```bash
# 上游源码(导出需 import 其 model.py)
git clone https://github.com/myungsub/CAIN.git ./CAIN

# 预训练权重(按上游 README 下载 pretrained_CAIN.pth,放入当前目录)
```

说明:

- `MODEL_CODE_DIR` / `--cain-dir` 为 clone 出来的 CAIN 源码目录(含 `model.py`)。
- `--ckpt` 为权重文件路径(`pretrained_CAIN.pth`)。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/cain

python export_cain_onnx.py \
  --cain-dir ./CAIN \
  --ckpt ./pretrained_CAIN.pth \
  --output-dir ./cain_onnx \
  --device cpu \
  --height 256 --width 256 --depth 3
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--cain-dir` | 上游 CAIN 源码目录(含 model.py) | `./CAIN` |
| `--ckpt` | 预训练权重路径 | `./pretrained_CAIN.pth` |
| `--output-dir` | ONNX 输出目录 | `./cain_onnx` |
| `--device` | 导出设备(cpu/cuda) | `cpu` |
| `--opset` | ONNX opset | `17` |
| `--height/--width` | 固定输入尺寸(须可被 `2^(depth+1)` 整除) | `256` / `256` |
| `--depth` | CAIN depth(须与权重一致) | `3` |

### 产出文件

```text
./cain_onnx/
└── cain.onnx   # 输入 img0/img1 [1,3,256,256],输出 mid_frame [1,3,256,256]
```

### 导出注意事项

- **固定 shape 约束**:`ascend_oriented` 转换针对固定 shape 编译;如需其他分辨率,须改 `--height/--width` 并同步 `config.ini` 的 `input_shape` 后重新导出+转换。
- CAIN 为纯 CNN+通道注意力,导出无自定义算子,`do_constant_folding=True`。

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_cain_onnx.py \
  --onnx ./cain_onnx/cain.onnx \
  --img0 ./frame0.png --img1 ./frame1.png \
  --output ./cain_mid_onnx.png \
  --height 256 --width 256 --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--onnx` | ONNX 路径 | 必填 |
| `--img0/--img1` | 起始/结束帧 | 必填 |
| `--output` | 输出中间帧 | `./cain_mid_onnx.png` |
| `--height/--width` | 须与导出一致 | `256` / `256` |
| `--device` | ORT provider | `cpu` |

### 执行日志

```log
[onnx] saved mid frame -> ./cain_mid_onnx.png
  latency_ms_mean: 374.947   # CPU 参考; Ascend 见 §5
  proc_rss_mb: 262 (hwm=262)

> 说明:精度对齐需真实权重 `pretrained_cain.pth`(Dropbox 托管,本环境不可达)。性能为真实架构实测(性能与权重无关,数值有效),精度对齐待真实权重接入。
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

```bash
converter_lite --fmk=ONNX \
  --modelFile=./cain_onnx/cain.onnx \
  --outputFile=./cain_onnx/cain \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

### 配置文件 `config.ini`

```ini
[acl_build_options]
input_format="ND"
input_shape="img0:1,3,256,256;img1:1,3,256,256"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

### 产出说明

```text
./cain_onnx/
└── cain.mindir   # 单文件(CAIN <2GB,无 _variables)
```

执行日志:

```log
CONVERT RESULT SUCCESS:0
# 产物 cain.mindir (90MB, fp16)
# 注: GlobalAveragePool "has no attr kernel_size" 警告无害(channel attention 的 GAP 适配)
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_cain_mslite.py \
  --mindir ./cain_onnx/cain.mindir \
  --img0 ./frame0.png --img1 ./frame1.png \
  --output ./cain_mid_mslite.png \
  --height 256 --width 256 \
  --device ascend --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir` | MindIR 路径 | 必填 |
| `--img0/--img1` | 起始/结束帧 | 必填 |
| `--device/--device-id` | 推理设备/卡号 | `ascend` / `0` |
| `--warmup/--runs` | 预热/计时轮数 | `3` / `10` |

说明:

- **内存守护**:脚本在建模型前检查 RAM/NPU HBM,任一已用 >80% 则告警退出;运行后打印进程峰值占用。
- **固定 shape**:推理 `--height/--width` 必须与导出/转换一致(256×256),否则需重新导出+转换。

### 执行日志

```log
[memory-budget] RAM: used 68528/256375 MB (26.7%)
[memory-budget] NPU0 HBM: used 1366/44280 MB (3.1%)
[mslite] saved mid frame -> ./cain_mid_mslite.png
Perf:
  input: 256x256, warmup=3 runs=10
  latency_ms_mean: 13.535
  latency_ms_p50:  13.541
  latency_ms_p90:  13.570
  proc_rss_mb: 1093 (hwm=1261)
```

---

## 6. 性能数据

测试环境:Atlas 300I Duo(310P3),CANN 8.5.0,MindSpore Lite 2.9.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 推理(256×256,mean) | 13.535 |
| 推理(256×256,p50) | 13.541 |
| 推理(256×256,p90) | 13.570 |
| 进程 RSS 峰值 (MB) | 1261 |
| NPU0 HBM 峰值 (MB) | ~1366 (3.1% / 44280) |

---

## 7. 常见问题

1. 现象:`无法从 --cain-dir 导入 CAIN`
   - 原因:未 clone 上游源码或路径不含 `model.py`。
   - 解决方案:`git clone https://github.com/myungsub/CAIN.git` 并传入正确 `--cain-dir`。

2. 现象:转换/推理报 shape 不匹配
   - 原因:推理分辨率与导出/转换的固定 shape 不一致。
   - 解决方案:统一 `--height/--width` 与 `config.ini` 的 `input_shape`,重新导出+转换。

3. 现象:内存守护告警退出
   - 原因:RAM 或 NPU HBM 已用 >80%。
   - 解决方案:先释放内存,或降低帧数/分辨率后重试。

---

## 8. 参考资源

- 上游模型仓库:<https://github.com/myungsub/CAIN>
- MindSpore Lite 文档:<https://www.mindspore.cn/lite>
- ONNX Runtime 文档:<https://onnxruntime.ai/docs/>

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游 CAIN 模型与代码许可证以其仓库为准。
