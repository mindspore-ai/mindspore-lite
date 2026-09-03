# BEVDet ONNX导出与MindSpore Lite 推理教程

本教程介绍如何将 BEVDet 整网模型导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上进行推理 benchmark 与精度评估。

## 1. 环境准备

### 依赖版本

| 软件包            | 版本       |
| -------------- |----------|
| Python         | 3.10     |
| torch          | 2.0.0    |
| torchvision    | 0.15.0   |
| mmdet3d        | 1.0.0rc4 |
| mmdet          | 2.28.2   |
| mmcv-full      | 1.7.0    |
| mmsegmentation | 0.30.0   |
| numba          | >=0.55   |
| numpy          | <2.0     |
| onnx           | 1.14.0   |
| CANN           | 8.5.1    |
| mindspore-lite | 2.9.0    |

### 环境安装

> 安装 mmdet3d、导出和推理需要下载[BEVDet源码](https://github.com/HuangJunJie2017/BEVDet)

```bash
# 安装 PyTorch 2.0.0 CPU 版本
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 OpenMMLab 依赖
pip install 'setuptools<70'
pip install mmcv-full==1.7.0 --no-build-isolation
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

# 安装 mmdet3d
# 注意：需要先修改 requirements/runtime.txt 中不兼容 Python 3.10 的版本限制
cd BEVDet/requirements
sed -i 's/numba==0.53.0/numba>=0.56.0/' runtime.txt
sed -i 's/networkx>=2.2,<2.3/networkx>=2.2/' runtime.txt
cd ..
pip install -v -e . --no-build-isolation
```

> **注意：** `numpy<2.0` 是必须的，否则 `torch` 和 `torchvision` 会出现 `NumPy 1.x compiled with NumPy 2.x` 的兼容性问题。安装完成后请确认 `pip install "numpy<2.0"`。

### 昇腾适配（修改 BEVDet 源码）

BEVDet 源码依赖 CUDA 扩展（`bev_pool_v2_ext`、`spconv`），在纯 CPU / Ascend 环境下导入会失败。需将 CUDA 扩展导入改为可选：

**1. `mmdet3d/ops/bev_pool_v2/bev_pool.py`**

```python
try:
    from . import bev_pool_v2_ext
except ImportError:
    bev_pool_v2_ext = None
```

**2. `mmdet3d/models/detectors/bevdet.py`**

```python
try:
    from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
except ImportError:
    TRTBEVPoolv2 = None
```

**3. `mmdet3d/models/utils/spconv_voxelize.py`**

```python
try:
    from spconv.pytorch.utils import PointToVoxel  # spconv-cu111 2.1.21
except ImportError:
    PointToVoxel = None
```

**4. `setup.py` 跳过 CUDA 扩展编译**

```python
ext_modules=[],  # 移除 CUDA 扩展，跳过编译
```

### 模型权重

下载 BEVDet-R50 预训练权重：

```bash
# 放置到 examples/base_models/bevdet/ 目录下
wget <权重下载地址>/bevdet-r50.pth
```

> [BEVDet 代码仓和权重下载](https://github.com/HuangJunJie2017/BEVDet)

***

## 2. UnsortedSegmentSum 算子方案

### 方案概述

BEVDet 原始的 BEV Pool 使用 CUDA 算子 `TRTBEVPoolv2`，NPU 环境无法运行。本方案使用 CANN 内置算子 `UnsortedSegmentSum` 替代：

```text
原始 CUDA 路径:  depth × feat → scatter_add (CUDA)  → BEV 特征
本方案路径:      depth × feat → Custom(UnsortedSegmentSum) → BEV 特征
```

### 算子定义

```python
class BEVPoolSegmentSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, segment_ids, num_segments):
        # forward 使用 scatter_add_ 保证 eager 模式正确
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

### BEVDet 整网架构

```log
原始输入 (6张相机图像)    →    Image Backbone    →    Image Neck    →    LSS View Transformer    →    BEV特征    →    BEV Encoder    →    Detection Head
   (1, 6, 3, 256, 704)       (ResNet-50)           (CustomFPN)        (depth_net + BEVPool)       (1, 64, 128, 128)   (CustomResNet+FPN_LSS)   (CenterHead)
                                                                                                                                                    ↓
                                                                                                                                                输出检测结果
```

### 整网导出说明

| 模块                   | 说明                           |
| -------------------- | ---------------------------- |
| Image Backbone       | ResNet-50，输入6张相机图像           |
| Image Neck           | CustomFPN，融合多尺度特征            |
| LSS View Transformer | depth_net + CustomBEVPoolV3 |
| BEV Encoder          | CustomResNet + FPN_LSS      |
| Detection Head       | CenterHead，输出10类3D检测结果       |

### 输入输出

| 类型 | 名称           | Shape                    | 说明                                       |
| -- | ------------ | ------------------------ | ---------------------------------------- |
| 输入 | img          | [batch, 6, 3, 256, 704] | 6 张相机图像，每张 256x704 像素                    |
| 输入 | ranks_depth  | [N_Points]              | 每个有效点在 `depth.reshape(-1)` 中的扁平索引（host 侧算） |
| 输入 | ranks_feat   | [N_Points]              | 每个有效点在 `feat.reshape(-1, C)` 中的扁平索引（host 侧算） |
| 输入 | ranks_bev    | [N_Points]              | 每个有效点在 BEV 输出中的扁平索引（host 侧算）            |
| 输出 | reg          | [batch, 2, H/2, W/2]    | 中心点偏移 (x, y)                             |
| 输出 | height       | [batch, 1, H/2, W/2]    | Z 坐标                                     |
| 输出 | dim          | [batch, 3, H/2, W/2]    | 3D 尺寸 (长, 宽, 高)                          |
| 输出 | rot          | [batch, 2, H/2, W/2]    | 偏航角 (sin, cos)                           |
| 输出 | vel          | [batch, 2, H/2, W/2]    | 速度 (vx, vy)                              |
| 输出 | heatmap      | [batch, 10, H/2, W/2]   | 10 类热力图                                  |

> **N_Points** 是动态维度（约 20 万，随相机参数变化），通过 `dynamic_axes` 标记为 `n_points`。ranks 三个张量共享同一个 N_Points 维度。

### 检测目标类别（10类）

| 索引 | 类别                    | 索引 | 类别            |
| -- | --------------------- | -- | ------------- |
| 0  | car                   | 5  | barrier       |
| 1  | truck                 | 6  | motorcycle    |
| 2  | construction_vehicle | 7  | bicycle       |
| 3  | bus                   | 8  | pedestrian    |
| 4  | trailer               | 9  | traffic_cone |

***

## 3. 导出 ONNX 模型

### 导出命令

```bash
cd examples/base_models/bevdet

python export_bevdet_onnx.py \
    --config BEVDet/configs/bevdet/bevdet-r50.py \
    --checkpoint bevdet-r50.pth \
    --output output \
    --prefix bevdet_r50
```

### 参数说明

| 参数             | 说明            | 默认值                          |
| -------------- | ------------- | ---------------------------- |
| `--config`     | BEVDet 配置文件   | `BEVDet/configs/bevdet/bevdet-r50.py` |
| `--checkpoint` | 权重文件路径        | `bevdet-r50.pth`              |
| `--output`     | 输出目录          | `output`                     |
| `--prefix`     | 输出文件前缀        | `bevdet_r50`                 |
| `--opset`      | ONNX opset 版本 | `17`                         |
| `--num-points` | 随机 ranks 长度（仅用于结构导出，不参与精度） | `179832`     |
| `--device`     | 导出设备          | `cpu`                        |

### 关键技术细节

1. **BEV Pool 算子替换**：使用 `BEVPoolSegmentSum`（`Custom::UnsortedSegmentSum`）替代 CUDA `TRTBEVPoolv2`
2. **禁用 with_cp**：ResNet-50 的梯度检查点不兼容 ONNX 导出，设为 `False`
3. **ONNX_FALLTHROUGH**：使用 `operator_export_type=ONNX_FALLTHROUGH` 跳过 Custom 算子注册检查
4. **固定 shape 导出**：不使用 `dynamic_axes`，让 converter 完成常量折叠优化
5. **随机 dummy 输入**：用随机张量作为导出输入；`ranks_*` 的 randint 上界由 pre-pass 探测真实 `D / H_feat / W_feat` 后生成，避免 gather 越界。随机 ranks 仅用于结构导出，导出的 ONNX 数值无意义，不能用于精度评估。

### 产出

```log
output/
└── bevdet_r50.onnx          # ONNX 模型 (~169MB, 253 nodes, 1 Custom, 0 Scatter)
```

> **注意：** 导出的 ONNX 包含 `Custom::UnsortedSegmentSum` 节点，不支持 ONNX Runtime 推理，需通过 MindSpore Lite 转换为 MindIR。

***

## 4. ONNX 转换 MindIR

### 配置文件

提供两种 shape 模式的转换配置：

**固定 shape（`config/config_fixed.ini`）**：

```ini
[acl_build_options]
input_format="ND"
input_shape="img:1,6,3,256,704;ranks_depth:179832;ranks_feat:179832;ranks_bev:179832"
ge.exec.precision_mode=allow_mix_precision
[ascend_context]
plugin_custom_ops=All
```

**动态 shape（`config/config_dynamic.ini`）**：

```ini
[acl_build_options]
input_format="ND"
input_shape="img:1,6,3,256,704;ranks_depth:-1;ranks_feat:-1;ranks_bev:-1"
ge.exec.precision_mode=allow_mix_precision
[ascend_context]
plugin_custom_ops=All
```

两种配置的差异仅在于 `ranks_*` 的 shape：固定为 `179832` vs 动态为 `-1`。

### 转换命令

```bash
# 按CANN安装的实际路径进行修改
source /path/to/Ascend/ascend-toolkit/set_env.sh
# 配置环境变量(LITE_HOME路径需要用户自行修改)
export LITE_HOME=/path/to/mindspore-lite-2.9.0-linux-aarch64
export LD_LIBRARY_PATH=${LITE_HOME}/runtime/third_party/dnnl:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LITE_HOME}/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export Convert=${LITE_HOME}/tools/converter/converter/converter_lite

# 固定 shape
$Convert \
    --fmk=ONNX \
    --modelFile=output/bevdet_r50.onnx \
    --outputFile=output/bevdet_r50 \
    --configFile=config/config_fixed.ini \
    --optimize=ascend_oriented \
    --saveType=MINDIR

# 动态 shape
$Convert \
    --fmk=ONNX \
    --modelFile=output/bevdet_r50.onnx \
    --outputFile=output/bevdet_r50_dynamic \
    --configFile=config/config_dynamic.ini \
    --optimize=ascend_oriented \
    --saveType=MINDIR
```

### 配置项说明

| 配置项 | 值 | 作用 |
| ------ | -- | ---- |
| `input_format` | `ND` | 输入格式为 N 维 |
| `input_shape` | `img:1,6,3,256,704;ranks_depth:...` | 输入 shape，`-1` 表示动态 |
| `ge.exec.precision_mode` | `allow_mix_precision` | Conv/MatMul 走 FP16，精度敏感算子保持 FP32 |
| `plugin_custom_ops` | `All` | 使能所有内置融合 pass |
| `--optimize` | `ascend_oriented` | Ascend 离线优化（必须） |

### 固定 vs 动态 shape 对比

| 维度 | 固定 shape | 动态 shape |
| ---- | --------- | --------- |
| ranks shape | `[179832]`（编译期固定） | `[-1]`（运行时 resize） |
| 模型文件 | `bevdet_r50.mindir` | `bevdet_r50_dynamic.mindir` |
| 推理时 ranks | 复用 sample 0 的 ranks | 每帧计算真实 ranks |
| 适用场景 | 追求最大吞吐 | 追求最高精度 |
| mAP | 0.2800 | 0.2886 |
| NDS | 0.2928 | 0.2989 |

### MindIR 产出

```log
output/
├── bevdet_r50.mindir              # 固定 shape MindIR
└── bevdet_r50_dynamic.mindir      # 动态 shape MindIR
```

> 转换日志中出现 `CONVERT RESULT SUCCESS:0` 即表示成功。`WARNING` 可忽略。

***

## 5. MindSpore Lite 推理验证

> 该脚本仅用于验证 MindIR 模型推理功能正常，用固定种子的随机张量作为输入；性能和精度由第 6 节的 benchmark 体现。

```bash
cd examples/base_models/bevdet

python infer_bevdet_mslite.py \
    --model output/bevdet_r50.mindir \
    --device-id 0
```

### 参数说明

| 参数             | 说明            | 默认值                          |
| -------------- | ------------- | ---------------------------- |
| `--model`      | MindIR 模型路径   | 必填                           |
| `--device-id`  | Ascend 设备 ID | `0`                          |
| `--seed`       | 随机种子          | `42`                         |
| `--depth-size` | ranks_depth randint 上界（B·N·D·H_feat·W_feat） | `498432`（BEVDet-R50） |
| `--feat-size`  | ranks_feat randint 上界（B·N·H_feat·W_feat） | `4224`（BEVDet-R50）   |
| `--bev-size`   | ranks_bev randint 上界（bev_z·bev_h·bev_w） | `16384`（BEVDet-R50）  |

脚本会在推理前校验输入 shape 不含 `-1`，否则提示用动态模型。

***

## 6. Benchmark 与精度评估

### 数据准备

> `bevdetv3-nuscenes_infos_val.pkl` 数据集是由 [NuScenes 开源数据集 Full dataset (v1.0) Mini](https://www.nuscenes.org/nuscenes#download) 经过 BEVDet 代码仓中步骤 step 3. 处理得到。

将 NuScenes 数据按以下结构放置到 BEVDet 仓中：

```text
BEVDet/
└── data/
    └── nuscenes/
        ├── bevdetv3-nuscenes_infos_val.pkl   # 验证集标注文件（81 样本）
        ├── maps/
        ├── samples/
        └── sweeps/
```

Benchmark 推理时通过 `--data-root BEVDet/data/nuscenes` 指向该目录。

### 手动执行

```bash
cd examples/base_models/bevdet

# 固定 shape
ASCEND_RT_VISIBLE_DEVICES=0 python benchmark_bevdet_mslite.py \
    --config BEVDet/configs/bevdet/bevdet-r50.py \
    --checkpoint bevdet-r50.pth \
    --model output/bevdet_r50.mindir \
    --data-root BEVDet/data/nuscenes \
    --shape-mode fixed \
    --device-id 0 \
    --eval \
    --output results/benchmark_mslite_fixed_result.txt

# 动态 shape
ASCEND_RT_VISIBLE_DEVICES=0 python benchmark_bevdet_mslite.py \
    --config BEVDet/configs/bevdet/bevdet-r50.py \
    --checkpoint bevdet-r50.pth \
    --model output/bevdet_r50_dynamic.mindir \
    --data-root BEVDet/data/nuscenes \
    --shape-mode dynamic \
    --device-id 0 \
    --eval \
    --output results/benchmark_mslite_dynamic_result.txt
```

### 参数说明

| 参数              | 说明                              | 默认值          |
| --------------- | ------------------------------- | ------------ |
| `--config`      | BEVDet 配置文件                     | 必填           |
| `--checkpoint`  | 权重文件                            | 必填           |
| `--model`       | MindIR 模型路径                     | 必填           |
| `--data-root`   | nuscenes 数据根目录（覆盖 cfg.data.test.data_root / ann_file，避免拷贝脚本到 BEVDet 仓） | 无 |
| `--shape-mode`  | shape 模式：`fixed` / `dynamic` / `gear` / `auto` | `fixed` |
| `--device-id`   | Ascend 设备 ID                    | `0`          |
| `--warmup`      | 预热迭代次数                          | `5`          |
| `--samples`     | benchmark 样本数（0=全部）             | `0`          |
| `--postprocessing` | 运行后处理（decode + NMS）          | 关            |
| `--eval`        | 运行 mAP/NDS 评估（自动启用 postprocessing）| 关            |
| `--output`      | 结果输出文件                          | 无            |

> `--data-root` 指向 BEVDet 仓中的 `data/nuscenes/` 目录（包含 `bevdetv3-nuscenes_infos_val.pkl`、`maps/`、`samples/`、`sweeps/`），脚本会自动改写配置里的相对路径，无需拷贝脚本到 BEVDet 仓。

***

## 7. 测试结果

### 测试环境

| 项目 | 配置 |
| ---- | --- |
| NPU | Atlas 300IDuo + MindSpore Lite 2.9.0 |
| 数据集 | NuScenes 验证集 (81 样本) |
| 评估指标 | NuScenes mAP / NDS |

### 性能

| 指标     | MSLite Fixed (NPU) |
|--------| ------------------- |
| FPS    | 34.01               |
| 延迟     | 29.40 ms            |

### 精度对比

> 精度数据基于固定 shape MindIR（`bevdet_r50.mindir`，`--shape-mode fixed`）执行得到。

| 指标 | TRT Fixed (GPU) | MSLite Fixed (NPU) | 绝对差 | 相对差 |
| ---- |-----------------| ------------------- | ------ | ------ |
| mAP  | 0.2805          | 0.2802              | 0.0003 | 0.11%  |
| NDS  | 0.2928          | 0.2928              | 0.0000 | 0.00%  |
| mATE | 0.8022          | 0.8014              | 0.0008 | 0.10%  |
| mASE | 0.4738          | 0.4737              | 0.0001 | 0.02%  |
| mAOE | 0.8016          | 0.8016              | 0.0000 | 0.00%  |
| mAVE | 1.0282          | 1.0266              | 0.0016 | 0.16%  |
| mAAE | 0.3967          | 0.3966              | 0.0001 | 0.03%  |

### 结论

- **精度**：mAP 误差 0.0003（0.11%），NDS 误差 0.0000（0.00%），精度与 TRT 完全对齐
- **性能**：NPU FPS 达到 34.01

***

## 8. 常见问题 FAQ

### 8.1 不支持 ONNX Runtime 推理

**问题：** `onnxruntime.InvalidGraph: No Op registered for Custom`

**原因：** ONNX 包含 `Custom::UnsortedSegmentSum` 节点，ONNX Runtime 无法识别。

**解决：** 需通过 `converter_lite` 转换为 MindIR 后在 Ascend NPU 上运行。

### 8.2 转换日志出现 WARNING

**问题：** `Unsupported custom type: UnsortedSegmentSum`

**解决：** 这些 WARNING 可忽略。转换最终出现 `CONVERT RESULT SUCCESS:0` 即表示成功。CANN 在推理阶段通过 `ASCEND_CUSTOM_OPP_PATH` 加载自定义算子实现。

### 8.3 固定 shape 模型推理其他样本报错

**问题：** `RuntimeError: data size not equal! Numpy size: 719424, Tensor size: 719328`

**原因：** 固定 shape MindIR 的 ranks 长度固定为 179832（sample 0），其他样本的 ranks 长度不同（如 sample 1 = 179856）。

**解决：** 使用动态 shape MindIR（`--model output/bevdet_r50_dynamic.mindir --shape-mode dynamic`）。

### 8.4 梯度检查点导致导出失败

**问题：** `RuntimeError: _Map_base::at`

**解决：** 导出脚本已自动设置 `cfg.model.img_backbone.with_cp = False` 禁用 ResNet 梯度检查点。

***

## 9. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [BEVDet 项目](https://github.com/HuangJunJie2017/BEVDet)
- [mmdetection3d 文档](https://mmdetection3d.readthedocs.io/)
- [CANN 文档](https://support.huaweicloud.com/cann/index.html)
- [NuScenes 数据集下载](https://www.nuscenes.org/nuscenes#download)

## 10. 许可证

本教程遵循 BEVDet 项目的许可证（MIT License）。
