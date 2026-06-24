# FCOS3D ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 FCOS3D（FCOSMono3D，单目相机 3D 目标检测，nuScenes 10 类）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。

FCOS3D 是纯卷积（ResNet-101 + FPN + DCN）的单目 3D 检测基线，算子友好，预期可直接转换。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmcv-full      | 1.7.0     |
| mmdet          | 2.28.2    |
| mmdetection3d  | 1.0.0rc4  |
| onnx           | 1.14.0    |
| onnxruntime    | 1.16.0    |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation
pip install mmdet==2.28.2
# 从 mmdetection3d 源码安装 1.0.0rc4
```

权重与 config 来源：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 的 `configs/fcos3d/`。

---

## 2. 模型说明

```log
输入图像 (1,3,H,W) → ResNet-101(+DCN) backbone → FPN neck → FCOSMono3D head → 多分支输出
                                                                    (cls / bbox / centerness / dir)
```

### 输入输出（导出固定 shape，H=320, W=800，可调）

| 类型 | 名称          | Shape                       | 说明                   |
| ---- | ------------- | --------------------------- | ---------------------- |
| 输入 | img           | \[1, 3, 320, 800]           | 前视相机图像           |
| 输出 | cls_score     | \[1, num_cls, h, w]         | 类别分数（最深 FPN 层）|
| 输出 | bbox_pred     | \[1, C, h, w]               | 3D 框回归（多通道）    |
| 输出 | centerness    | \[1, 1, h, w]               | 中心度                 |
| 输出 | dir_cls       | \[1, 2, h, w]               | 方向分类               |

> 说明：导出为 bbox_head 的**原始多分支输出**（最深 FPN level）。3D box 解码（含 cam2img 投影）留待后处理，阶段 2 验证时在 numpy 中实现（与 ONNX/MSLite 共用，便于精度对齐）。

---

## 3. ONNX 导出

```bash
cd examples/base_models/fcos3d

python export_fcos3d_onnx.py \
  --config /path/to/mmdetection3d/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py \
  --checkpoint /path/to/fcos3d_r101_..._nus-mono3d.pth \
  --output fcos3d_onnx/fcos3d.onnx \
  --img-h 320 --img-w 800 --opset 17
```

| 参数            | 说明                          | 默认值                      |
| --------------- | ----------------------------- | --------------------------- |
| `--config`      | mmdet3d config 路径           | 必填                        |
| `--checkpoint`  | 权重路径                      | 必填                        |
| `--output`      | 输出 ONNX 路径                | `fcos3d_onnx/fcos3d.onnx`   |
| `--img-h/w`     | 导出固定输入高/宽             | `320 / 800`                 |
| `--opset`       | ONNX opset                    | `17`                        |

产出：`fcos3d_onnx/fcos3d.onnx`

---

## 4. ONNX 推理

```bash
python infer_fcos3d_onnx.py --model ./fcos3d_onnx/fcos3d.onnx --img-h 320 --img-w 800
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input, shape=(1, 3, 320, 800), seed=1024
Output shapes:
  cls_score: (1, 10, h, w)
  bbox_pred: (1, C, h, w)
  ...
Performance:
  latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

$Converter --fmk=ONNX \
  --modelFile=./fcos3d_onnx/fcos3d.onnx \
  --outputFile=./fcos3d_onnx/fcos3d_ascend \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`（默认；精度不足再改 fp32）。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_fcos3d_mslite.py \
  --model ./fcos3d_onnx/fcos3d_ascend.mindir \
  --device ascend --device-id 0 --img-h 320 --img-w 800
```

执行日志（占位）：

```log
Using random input, shape=(1, 3, 320, 800), seed=1024
Performance:
  latency_ms_mean: TBD
Memory:
  VmRSS: TBD KB
```

---

## 7. 性能数据

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：输入 `(1, 3, 320, 800)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 备注        |
| -------------- | ----- | --------------------- | ----------- |
| ONNX Runtime   | CPU   | TBD                   | 精度基准    |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | force_fp16  |

---

## 8. 常见问题

1. **现象**：导出时报 `extract_feat` 或 bbox_head 输出 tuple 长度不符
   - **原因**：mmdet3d 不同版本 FCOSMono3D head forward 返回 3 或 4 个分量
   - **解决方案**：按实际版本调整 `FCOS3DWrapper.forward` 的解包（阶段 2 验证）

2. **阶段 2 阻塞（已记录暂搁）**：FCOS3D 主干 ResNet-101 末段使用 DCN（`MMCVModulatedDeformConv2d`）。实测该自定义算子：
   - ONNX 导出可完成（mmcv 注册了 symbolic），但 ONNX Runtime **无法推理**（`mmcv:MMCVModulatedDeformConv2d is not a registered function/op`）；
   - MSLite `converter_lite` **转换失败/崩溃**（DCN 算子不支持）。
   - 故 FCOS3D（DCN 版）当前无法在 Ascend 跑通。解决方案：① 用 AscendC 实现 `MMCVModulatedDeformConv2d` 自定义算子；② 或导出非 DCN 版（需对应无 DCN 的 checkpoint，当前 mmdetection3d 仅提供 DCN 版权重）。暂按记录暂搁。

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [mmdetection3d / FCOS3D](https://github.com/open-mmlab/mmdetection3d)

本目录脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 mmdetection3d 上游为准（Apache-2.0）。
