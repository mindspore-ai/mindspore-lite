# FlashOCC ONNX导出与MindSpore Lite 推理教程

本教程介绍如何将 FlashOCC 整网模型导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上进行推理 benchmark 与精度评估。

## 1. 环境准备

### 依赖版本

| 软件包            | 版本       |
| -------------- |----------|
| Python         | 3.10     |
| torch          | 2.0.0    |
| torchvision    | 0.15.0   |
| mmdet3d        | 1.0.0rc4 |
| mmdet          | 2.25.1   |
| mmcv-full      | 1.5.3    |
| mmsegmentation | 0.25.0   |
| numba          | >=0.55   |
| numpy          | <2.0     |
| onnx           | 1.14.0   |
| CANN           | 8.5.1    |
| mindspore-lite | 2.9.0    |

### 环境安装

> 请先下载[FlasshOCC源码](https://github.com/Yzichen/FlashOCC)，后续昇腾适配、ONNX导出以及benchmark测试都需要此源码。环境安装请参考 [Environment Setup](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md)，其中CUDA相关无需安装。

```bash
conda create --name FlashOcc python=3.10
conda activate FlashOcc
pip install torch==2.0.0 torchvision==0.15.0
pip install mmcv-full==1.5.3
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

pip install lyft_dataset_sdk
pip install networkx==2.8.8
pip install numba==0.56.4
pip install numpy==1.23.5
pip install nuscenes-devkit
pip install plyfile
pip install scikit-image
pip install tensorboard
pip install trimesh==2.35.39
pip install setuptools==59.5.0
pip install yapf==0.40.1

cd Path_to_FlashOcc
git clone git@github.com:Yzichen/FlashOCC.git

cd Path_to_FlashOcc/FlashOcc
git clone https://github.com/open-mmlab/mmdetection3d.git

cd Path_to_FlashOcc/FlashOcc/mmdetection3d
git checkout v1.0.0rc4
pip install -v -e .

# 安装前请先按照昇腾适配指示修改源码
cd Path_to_FlashOcc/FlashOcc/projects
pip install -v -e .
```

### 昇腾适配（修改 FlashOCC 源码）

FlashOCC 源码依赖 CUDA 扩展（`bev_pool_v2_ext`、`bev_pool_ext`、 `nearest_assign_ext`）和 CUDA 编译的 `ray_metrics`，在纯 CPU / Ascend 环境下导入会失败。需做以下修改：

#### 1. CUDA 扩展导入改为可选

`projects/mmdet3d_plugin/ops/__init__.py`：

```python
try:
    from .bev_pool import bev_pool
except (ImportError, ModuleNotFoundError):
    bev_pool = None
from .bev_pool_v2 import bev_pool_v2, TRTBEVPoolv2
try:
    from .nearest_assign import nearest_assign
except (ImportError, ModuleNotFoundError):
    nearest_assign = None
```

`projects/mmdet3d_plugin/ops/bev_pool_v2/bev_pool.py`：

```python
try:
    from . import bev_pool_v2_ext
except ImportError:
    bev_pool_v2_ext = None
```

#### 2. ray_metrics 延迟导入

`projects/mmdet3d_plugin/datasets/nuscenes_dataset_occ.py` 中将 `ray_metrics` 和 `ego_pose_dataset` 的顶层 import 移到 `evaluate()` 方法内部，避免 import 时编译 CUDA 扩展。

修改前（顶层 import，模块加载时即触发 CUDA 扩展编译）：

```python
# 顶层 import 区域（第 9-15 行）
from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from .ego_pose_dataset import EgoPoseDataset                    # ← 删除
from torch.utils.data import DataLoader
from ..core.evaluation.ray_metrics import main as calc_rayiou   # ← 删除
from ..core.evaluation.ray_metrics import main_raypq            # ← 删除
import torch
import glob
```

修改后（顶层移除，延迟到 evaluate 方法内）：

```python
# 顶层 import 区域（第 9-14 行）
from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore
from torch.utils.data import DataLoader
import torch
import glob

# ... 类定义 ...

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        metric = eval_kwargs['metric'][0]
        print("metric = ", metric)
        if metric == 'ray-iou':
            # 延迟导入：仅在使用 ray-iou 评估时才加载
            # 避免 mIoU 路径触发 CUDA 编译
            from .ego_pose_dataset import EgoPoseDataset
            from ..core.evaluation.ray_metrics import main as calc_rayiou
            from ..core.evaluation.ray_metrics import main_raypq
            occ_gts = []
            ...
```

> **原因**：`ray_metrics.py` 顶层执行 `dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], ...)` 即时编译 CUDA 扩展。顶层 import 会在 Python 启动时触发编译，而 Ascend 环境无 `nvcc`/`ninja` 导致崩溃。延迟到 `evaluate()` 内部后，只有实际运行 ray-iou 评估时才会触发，mIoU 评估路径不受影响。

### 数据准备

> NuScenes 数据集准备请参考FlashOCC安装文档中的 [step 3][step 3-link]。

[step 3-link]: https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md

数据集放置到 FlashOCC 数据目录：

```text
FlashOCC/
└── data/
    └── nuscenes/
        ├── bevdetv2-nuscenes_infos_val.pkl   # 验证集标注（81 样本）
        ├── bevdetv2-nuscenes_infos_train.pkl  # 训练集标注（323 样本）
        ├── gts/                               # 占用预测 GT（10 场景）
        ├── maps/
        ├── samples/
        └── sweeps/
```

### 模型权重

下载 FlashOCC-R50 (M1) 预训练权重：

```bash
# 放置到 FlashOCC/ 目录
# 下载地址:
# https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view
cp flashocc-r50-256x704.pth FlashOCC/
```

> [FlashOCC 代码仓和权重下载](https://github.com/Yzichen/FlashOCC)

### 目录结构

> 下载本工程，将工程中脚本benchmark_flashocc_mslite.py及export_flashocc_onnx.py拷贝到FlashOCC/scripts目录下，将config目录拷贝到FlashOCC目录下。

```text
path/to/flashocc/
├── FlashOCC/                               # FlashOCC 源码
│   ├── projects/configs/flashocc/
│   │   ├── flashocc-r50.py                 # M1 配置（out_dim=256）
│   │   └── flashocc-r50-trt.py             # TRT 配置
│   ├── scripts/
│   │   ├── export_flashocc_onnx.py         # ONNX 导出脚本
│   │   └── benchmark_flashocc_mslite.py    # MindIR benchmark 脚本
│   ├── config/
│   │   ├── config.ini                      # 转换配置（混合精度+融合+BN）
│   │   └── fusion_switch.config            # 融合规则配置（BN 融合）
│   └── output/                             # ONNX / MindIR 产出
├── flashocc-r50-256x704.pth                # 模型权重
└── results/
    └── benchmark_mslite_result.txt         # NPU benchmark 结果
```

### 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LITE_HOME=/path/to/mindspore-lite-2.9.0-linux-aarch64
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PATH=$LITE_HOME/tools/converter/converter:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
```

---

## 2. UnsortedSegmentSum 算子方案

### 方案概述

FlashOCC 原始的 BEV Pool 使用 CUDA 算子 `TRTBEVPoolv2`，CPU/Ascend 环境无法运行。本方案使用 CANN 内置算子 `UnsortedSegmentSum` 替代：

```text
原始 CUDA 路径: depth × feat → TRTBEVPoolv2 (CUDA) → BEV 特征
本方案路径:     depth × feat → Gather+Gather+Mul
                + Custom(UnsortedSegmentSum) → BEV 特征
```

### 算子定义

```python
class BEVPoolSegmentSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, segment_ids, num_segments):
        # forward 使用 scatter_add_ 保证 eager 模式正确
        N, C = x.shape
        ns = int(num_segments.item())
        out = torch.zeros(ns, C, dtype=x.dtype, device=x.device)
        out.scatter_add_(0, segment_ids.long().unsqueeze(-1).expand(-1, C), x)
        return out

    @staticmethod
    def symbolic(g, x, segment_ids, num_segments):
        # symbolic 发出 Custom(UnsortedSegmentSum) ONNX 节点
        return g.op("Custom", x, segment_ids, num_segments,
                    type_s="UnsortedSegmentSum",
                    input_names_s=["x", "segment_ids", "num_segments"],
                    optional_input_names_s=[],
                    output_names_s=["y"],
                    output_num_i=1,
                    input_index_i=[0, 1, 2])
```

### FlashOCC vs BEVDet 方案差异

| 对比项         | BEVDet               | FlashOCC                   |
| -------------- | -------------------- | -------------------------- |
| 模型类型       | `BEVDetTRT`          | `BEVDetOCCTRT`             |
| 任务           | 3D 目标检测          | 占用预测 (OCC)             |
| Neck 返回类型  | tensor (`outs[0]`)   | list，需取 `x[0]`          |
| BEV Encoder    | 单个 `bev_encoder()` | `backbone` + `neck` 分开   |
| Head           | `pts_bbox_head`      | `occ_head` (BEVOCCHead2D)  |
| 输出           | 6 个检测特征图       | 1 个占用 logits            |
| grid_size      | 128×128×1            | 200×200×1                  |
| ranks N_Points | ~179832              | ~300974                    |
| 评估指标       | mAP / NDS            | mIoU                       |
| 额外标志       | 无                   | `wocc`/`wdet3d`/`upsample` |
| 导出模式       | 1 个 ONNX            | 2 个 ONNX                  |

### 整网架构

```log
输入 (6张相机图像)
  → Image Backbone (ResNet-50)
  → Image Neck (CustomFPN, 返回 list → 取 x[0])
  → LSS View Transformer (depth_net + softmax
    + Gather+Gather+Mul + UnsortedSegmentSum)
  → BEV Encoder Backbone (CustomResNet)
  → BEV Encoder Neck (FPN_LSS)
  → OCC Head (BEVOCCHead2D: Conv2d 3×3 + Linear predicter)
  → 输出 occ logits (1, 200, 200, 16, 18)
  → [可选] ArgMax → occ labels (1, 200, 200, 16)
```

### 输入输出

| 类型          | 名称          | Shape             | 说明              |
| ------------- | ------------- | ----------------- | ----------------- |
| 输入          | img           | [1,6,3,256,704]   | 6 张相机图像      |
| 输入          | ranks_depth   | [N_Points]        | BEV pool 深度索引 |
| 输入          | ranks_feat    | [N_Points]        | BEV pool 特征索引 |
| 输入          | ranks_bev     | [N_Points]        | BEV pool 输出索引 |
| 输出 (ori)    | output_0      | [1,200,200,16,18] | 18 类占用 logits  |
| 输出 (argmax) | cls_occ_label | [1,200,200,16]    | 每 voxel 预测类别 |

其中 `N_Points ≈ 300974`（根据相机内外参和 BEV 网格配置计算，每个样本略有不同：300909~302189）。

---

## 3. 导出 ONNX 模型

### 导出命令

```bash
cd FlashOCC

python scripts/export_flashocc_onnx.py \
    --config projects/configs/flashocc/flashocc-r50.py \
    --checkpoint flashocc-r50-256x704.pth \
    --work_dir output \
    --prefix flashocc_r50
```

### 导出参数说明

| 参数           | 说明              | 默认值         |
| -------------- | ----------------- | -------------- |
| `--config`     | FlashOCC 配置文件 | 必填           |
| `--checkpoint` | 权重文件路径      | 必填           |
| `--work_dir`   | 输出目录          | 必填           |
| `--prefix`     | 输出文件前缀      | `flashocc_r50` |
| `--opset`      | ONNX opset 版本   | `17`           |
| `--device`     | 导出设备          | `cpu`          |

### 关键技术细节

1. **BEV Pool 算子替换**：使用 `BEVPoolSegmentSum` （`Custom::UnsortedSegmentSum`）替代 CUDA `TRTBEVPoolv2`
2. **Neck 返回类型适配**：FlashOCC 的 `CustomFPN` 返回 list，取 `x[0]`
3. **BEV Encoder 分开调用**：`img_bev_encoder_backbone` + `img_bev_encoder_neck`（BEVDet 是单个 `bev_encoder()`）
4. **wdet3d/wocc 标志**：非 TRT 配置不设置这些标志，导出脚本自动设置 `wdet3d=False, wocc=True`
5. **两种导出模式**：`forward_ori`（occ logits）+ `forward_with_argmax`（occ labels）
6. **禁用 with_cp**：ResNet-50 的梯度检查点不兼容 ONNX 导出
7. **ONNX_FALLTHROUGH**：跳过 Custom 算子注册检查

### 导出产出

```log
output/
├── flashocc_r50.onnx              # forward_ori (~171MB)
└── flashocc_r50_with_argmax.onnx  # forward_with_argmax (~171MB)
```

> **注意：** 导出的 ONNX 包含 `Custom::UnsortedSegmentSum` 节点，不支持 ONNX Runtime 推理，需通过 MindSpore Lite 转换为 MindIR。

---

## 4. ONNX 转换 MindIR

### 配置文件

配置文件 `config/config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="img:1,6,3,256,704;ranks_depth:300974;ranks_feat:300974;ranks_bev:300974"
ge.exec.precision_mode=allow_mix_precision

[acl_init_options]
ge.fusionSwitchFile="config/fusion_switch.config"

[ascend_context]
plugin_custom_ops=All
```

融合规则配置 `config/fusion_switch.config`：

```json
{
  "Switch": {
    "GraphFusion": {
      "ALL": "on",
      "BatchNormBnInferOnlyFusionPass": "on"
    },
    "UBFusion": {
      "ALL": "on"
    }
  }
}
```

> **注意：** 如需导出动态 ranks，将 input_shape 中 `ranks_depth:300974` 等改为 `ranks_depth:-1` 等。

### 转换命令

```bash

converter_lite \
    --fmk=ONNX \
    --modelFile=output/flashocc_r50_with_argmax.onnx \
    --outputFile=output/flashocc_r50_with_argmax \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=config/config.ini
```

### 配置项说明

| 配置项                   | 值                    | 作用                  |
| ------------------------ | --------------------- | --------------------- |
| `input_format`           | `ND`                  | 输入格式为 N 维       |
| `input_shape`            | 固定 shape            | img 和 ranks 的维度   |
| `ge.exec.precision_mode` | `allow_mix_precision` | Conv/MatMul FP16      |
| `ge.fusionSwitchFile`    | 融合规则文件          | 开启 BN 融合          |
| `plugin_custom_ops`      | `All`                 | 使能所有内置融合 pass |
| `--optimize`             | `ascend_oriented`     | Ascend 离线优化       |

### 转换产出

```log
output/
└── flashocc_r50_with_argmax.mindir   # MindIR (~87MB)
```

> 转换日志中出现 `CONVERT RESULT SUCCESS:0` 即表示成功。 `WARNING`（如 `Unsupported custom type: UnsortedSegmentSum`）可忽略。

---

## 5. MindSpore Lite 推理

### 执行 benchmark

```bash
cd FlashOCC

# 完整 benchmark + mIoU 评估
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/benchmark_flashocc_mslite.py \
    --config projects/configs/flashocc/flashocc-r50.py \
    --checkpoint flashocc-r50-256x704.pth \
    --model output/flashocc_r50_with_argmax.mindir \
    --eval \
    --output results/benchmark_mslite_result.txt
```

### 推理参数说明

| 参数               | 说明                                | 默认值 |
| ------------------ | ----------------------------------- | ------ |
| `--config`         | FlashOCC 配置文件                   | 必填   |
| `--checkpoint`     | 权重文件                            | 必填   |
| `--model`          | MindIR 模型路径                     | 必填   |
| `--shape-mode`     | `auto`/`fixed`/`dynamic`/`gear`     | `auto` |
| `--device-id`      | Ascend 设备 ID                      | `0`    |
| `--warmup`         | 预热迭代次数                        | `5`    |
| `--samples`        | benchmark 样本数（0=全部）          | `0`    |
| `--postprocessing` | 运行后处理                          | 关     |
| `--eval`           | 运行评估（自动启用 postprocessing） | 关     |
| `--output`         | 结果输出文件                        | 无     |

### 后处理流程

```text
NPU 推理输出
  ├─ forward_with_argmax 模型:
  │    cls_occ_label (1,200,200,16) → 直接作为 occ 预测结果
  │
  └─ forward_ori 模型:
       output_0 (1,200,200,16,18)
       → occ_head.get_occ() → softmax + argmax → occ 预测结果

占用预测结果 (200,200,16)
→ dataset.evaluate(metric=['mIoU']) → mIoU
```

---

## 6. 测试结果

### 测试环境

| 项目     | 配置                                 |
| -------- | ------------------------------------ |
| NPU      | Atlas 300IDuo + MindSpore Lite 2.9.0 |
| 数据集   | NuScenes 验证集 (81 样本)            |
| 评估指标 | mIoU (18 类占用预测)                 |

### 性能

| 指标 | MSLite (NPU) |
| ---- | ------------ |
| FPS  | 31.93        |
| 延迟 | 31.32 ms     |

### 精度对比

| 指标     | TRT (GPU) | MSLite (NPU) | 绝对差    |
| -------- | --------- | ------------ | --------- |
| **mIoU** | **24.42** | **24.42**    | **+0.00** |

### 逐类别 IoU 对比

| 类别                 | MSLite (NPU) | TRT (GPU) | 差异  |
| -------------------- | ------------ | --------- | ----- |
| others               | 27.67        | 27.69     | -0.02 |
| barrier              | nan          | nan       | N/A   |
| bicycle              | 0.70         | 0.70      | +0.00 |
| bus                  | 31.31        | 31.33     | -0.02 |
| car                  | 38.76        | 38.76     | +0.00 |
| construction_vehicle | 0.00         | 0.00      | +0.00 |
| motorcycle           | 7.57         | 7.61      | -0.04 |
| pedestrian           | 12.75        | 12.79     | -0.04 |
| traffic_cone         | 2.33         | 2.33      | +0.00 |
| trailer              | nan          | nan       | N/A   |
| truck                | 19.79        | 19.81     | -0.02 |
| driveable_surface    | 72.82        | 72.82     | +0.00 |
| other_flat           | 0.00         | 0.00      | +0.00 |
| sidewalk             | 43.01        | 43.01     | +0.00 |
| terrain              | 35.56        | 35.56     | +0.00 |
| manmade              | 44.18        | 44.18     | +0.00 |
| vegetation           | 29.74        | 29.74     | +0.00 |
| free                 | 86.94        | 86.93     | +0.01 |

### 结论

- **精度**：mIoU 误差 0.00，逐类别 IoU 差异均在 ±0.04 以内，精度与 TRT 完全对齐
- **性能**：NPU FPS 达到 31.93

---

## 7. 常见问题 FAQ

### 7.1 不支持 ONNX Runtime 推理

**问题：** `onnxruntime.InvalidGraph: No Op registered for Custom`

**原因：** ONNX 包含 `Custom::UnsortedSegmentSum` 节点，ONNX Runtime无法识别。

**解决：** 需通过 `converter_lite` 转换为 MindIR 后在 Ascend NPU 上运行。

### 7.2 转换日志出现 WARNING

**问题：** `Unsupported custom type: UnsortedSegmentSum`

**解决：** 这些 WARNING 可忽略。转换最终出现 `CONVERT RESULT SUCCESS:0` 即表示成功。CANN 在推理阶段通过 `ASCEND_CUSTOM_OPP_PATH` 加载自定义算子实现。

### 7.3 CUDA 扩展导入失败

**问题：** `ImportError: cannot import name 'bev_pool_v2_ext'`

**原因：** CUDA 扩展未编译，Ascend 环境无 CUDA 工具链。

**解决：** 将 CUDA 扩展导入改为 `try/except` 可选导入。

### 7.4 ray_metrics 导入时编译 CUDA 失败

**问题：** `RuntimeError: Ninja is required to load C++ extensions`

**原因：** `ray_metrics.py` 在 import 时编译 CUDA 扩展（`dvr.cpp`/`dvr.cu`）。

**解决：** 将 `ray_metrics` 和 `ego_pose_dataset` 的 import 移到 `evaluate()` 方法内部（延迟导入）。

### 7.5 wdet3d 标志导致导出失败

**问题：** `AttributeError: 'NoneType' object has no attribute 'task_heads'`

**原因：** `flashocc-r50.py`（非 TRT 配置）不设置 `wdet3d`/`wocc` 标志，`BEVDetOCCTRT` 默认 `wdet3d=True`，但 `BEVDetOCC` 将 `pts_bbox_head` 设为 `None`。

**解决：** 导出脚本中自动设置 `cfg.model.wdet3d = False`。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [FlashOCC 项目](https://github.com/Yzichen/FlashOCC)
- [FlashOCC 论文](https://arxiv.org/abs/2311.12058)
- [BEVDet 项目](https://github.com/HuangJunJie2017/BEVDet)
- [mmdetection3d 文档](https://mmdetection3d.readthedocs.io/)
- [CANN 文档](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373)
- [DrivingSDK (mx_driving)](https://gitcode.com/Ascend/DrivingSDK)

---

## 9. 许可证

本教程遵循 FlashOCC 项目的许可证（Apache License 2.0）。
