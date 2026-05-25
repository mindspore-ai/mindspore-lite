# BEVDet 整网 ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 BEVDet 完整模型（含自定义 BEVPoolV3 算子）导出为 ONNX 格式，并转换为 MindSpore Lite MindIR 格式进行推理部署。

> **注意：** 导出的 ONNX 模型中包含自定义算子 `Custom::BEVPoolV3`，因此**不支持 ONNX Runtime 推理**。需要通过 MindSpore Lite 转换为 MindIR 格式后在 Ascend NPU 上运行。

## 1. 环境准备

### 依赖版本

| 软件包            | 版本       |
| -------------- | -------- |
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
| CANN           | 9.0.0    |
| mindspore-lite | 2.9.0    |

### 环境安装

```bash
# 安装 PyTorch 2.0.0 CPU 版本
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 OpenMMLab 依赖
pip install 'setuptools<70'
pip install mmcv-full==1.7.0 --no-build-isolation
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

# 安装 mmdet3d（从 BEVDet 目录以 editable 模式安装）
# 注意：需要先修改 requirements/runtime.txt 中不兼容 Python 3.10 的版本限制
cd BEVDet/requirements
sed -i 's/numba==0.53.0/numba>=0.56.0/' runtime.txt
sed -i 's/networkx>=2.2,<2.3/networkx>=2.2/' runtime.txt
cd ..
pip install -v -e . --no-build-isolation
```

> **注意：** `numpy<2.0` 是必须的，否则 `torch` 和 `torchvision` 会出现 `NumPy 1.x compiled with NumPy 2.x` 的兼容性问题。安装完成后请确认 `pip install "numpy<2.0"`。

***

## 2. 模型说明

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
| LSS View Transformer | depth\_net + CustomBEVPoolV3 |
| BEV Encoder          | CustomResNet + FPN\_LSS      |
| Detection Head       | CenterHead，输出10类3D检测结果       |

### 输入输出

| 类型 | 名称      | Shape                    | 说明                 |
| -- | ------- | ------------------------ | ------------------ |
| 输入 | img     | \[batch, 6, 3, 256, 704] | 6张相机图像，每张256x704像素 |
| 输出 | reg     | \[batch, 2, H/2, W/2]    | 中心点偏移 (x, y)       |
| 输出 | height  | \[batch, 1, H/2, W/2]    | Z 坐标               |
| 输出 | dim     | \[batch, 3, H/2, W/2]    | 3D 尺寸 (长, 宽, 高)    |
| 输出 | rot     | \[batch, 2, H/2, W/2]    | 偏航角 (sin, cos)     |
| 输出 | vel     | \[batch, 2, H/2, W/2]    | 速度 (vx, vy)        |
| 输出 | heatmap | \[batch, 10, H/2, W/2]   | 10 类热力图            |

### 检测目标类别（10类）

| 索引 | 类别                    | 索引 | 类别            |
| -- | --------------------- | -- | ------------- |
| 0  | car                   | 5  | barrier       |
| 1  | truck                 | 6  | motorcycle    |
| 2  | construction\_vehicle | 7  | bicycle       |
| 3  | bus                   | 8  | pedestrian    |
| 4  | trailer               | 9  | traffic\_cone |

***

## 3. 自定义 BEVPoolV3 算子

原始 BEVDet 使用 CUDA 自定义算子 `TRTBEVPoolv2`，在 CPU 环境下无法使用。本教程使用 `CustomBEVPoolV3` 昇腾融合算子替代，其定义在 `bev_pool_v3_ops.py` 中：

### 算子接口

```python
class CustomBEVPoolV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, feat, ranks_bev, with_depth, b, d, h, w, c):

    @staticmethod
    def symbolic(g, depth, feat, ranks_bev, with_depth, b, d, h, w, c):
        return g.op("Custom", depth, feat, ranks_bev,
                    with_depth_s=with_depth, b_i=b, d_i=d, h_i=h, w_i=w, c_i=c,
                    input_names_s=["depth", "feat", "ranks_depth", "ranks_feat", "ranks_bev"],
                    optional_input_names_s=["depth", "ranks_depth", "ranks_feat"],
                    type_s="BEVPoolV3",
                    input_index_i=[0, 1, 4],
                    output_names_s=["out"])
```

### 关键说明

- `g.op("Custom", ...)` 使用自定义算子域
- `ranks_depth`、`ranks_feat` 作为可选输入（optional\_input\_names\_s）
- `input_index_i=[0, 1, 4]` 对应 depth、feat、ranks\_bev 三个 Tensor 输入
- ONNX 导出时使用 `operator_export_type=ONNX_FALLTHROUGH` 跳过 Custom 算子注册检查

### 2D 输入接口

BEVPoolV3 算子接受 2D 张量作为输入，这些张量是从原始的 5D 张量通过索引展平得到的：

**原始 5D 张量：**

- `depth_5d`: `[B, N, D_depth, H_feat, W_feat]` = `[1, 6, 59, 16, 44]`
- `tran_feat_5d`: `[B, N, H_feat, W_feat, C]` = `[1, 6, 16, 44, 64]`

**展平为 2D 张量：**

```python
# depth 展平
depth_flat = depth_5d.reshape(-1)  # [B*N*D_depth*H_feat*W_feat]
base_idx = (ranks_depth.long() // D) * D
depth_2d = torch.stack([torch.gather(depth_flat, 0, base_idx + d) for d in range(D)], dim=1)
# depth_2d.shape: [N_RANKS, D_depth] = [120, 59]

# feat 展平
feat_flat = tran_feat_5d.reshape(-1, C)  # [B*N*H_feat*W_feat, C]
feat_2d = torch.gather(feat_flat, 0, ranks_feat.long().unsqueeze(-1).expand(-1, C))
# feat_2d.shape: [N_RANKS, C] = [120, 64]
```

**实际输入 Shape（以当前配置为例）：**

| 张量         | Shape           | 说明               |
| ---------- | --------------- | ---------------- |
| depth\_2d  | \[N\_RANKS, 59] | 每个有效点的所有深度bin概率值 |
| feat\_2d   | \[N\_RANKS, 64] | 每个有效点的所有通道特征值    |
| ranks\_bev | \[N\_RANKS]     | 每个有效点在BEV空间中的索引  |

其中 N\_RANKS ≈ 120（根据相机内外参和BEV网格配置计算得出的有效点数）。

### 参数含义

| 参数 | 说明                   | 默认值                |
| -- | -------------------- | ------------------ |
| b  | Batch size           | int(B)，动态获取        |
| d  | BEV空间Z轴高度维度 (bev\_z) | 1（从grid\_config计算） |
| h  | BEV Height           | 128                |
| w  | BEV Width            | 128                |
| c  | BEV Channels         | 64                 |

其中 `h=128`, `w=128` 来源于 `grid_config` 的 x,y 范围计算：

```python
grid_config = {
    'x': [-51.2, 51.2, 0.8],   # (51.2 - (-51.2)) / 0.8 = 128
    'y': [-51.2, 51.2, 0.8],   # (51.2 - (-51.2)) / 0.8 = 128
}
```

***

## 4. 导出 ONNX 模型

### 导出命令

```bash
cd examples/base_models/bevdet

python export_bevdet_onnx.py \
  --config BEVDet/configs/bevdet/bevdet-r50.py \
  --checkpoint bevdet-dev2.1/bevdet-r50.pth \
  --device cpu \
  --output bevdet_onnx/bevdet_r50_all.onnx
```

### 参数说明

| 参数             | 说明            | 默认值                                   |
| -------------- | ------------- | ------------------------------------- |
| `--config`     | BEVDet 配置文件路径 | `BEVDet/configs/bevdet/bevdet-r50.py` |
| `--checkpoint` | 权重文件路径        | `bevdet-dev2.1/bevdet-r50.pth`        |
| `--device`     | 设备类型          | `cpu`                                 |
| `--output`     | 输出 ONNX 路径    | `bevdet_onnx/bevdet_r50_all.onnx`     |
| `--opset`      | ONNX opset 版本 | `17`                                  |
| `--ncams`      | 相机数量          | `6`                                   |
| `--img_h`      | 图像高度          | `256`                                 |
| `--img_w`      | 图像宽度          | `704`                                 |

> **注意：** [BEVDet代码仓和权重下载链接](https://github.com/HuangJunJie2017/BEVDet)

### 关键技术细节

1. **禁用 with\_cp**：ResNet-50 的梯度检查点 (`with_cp=True`) 不兼容 ONNX 导出，需要在构建模型时设为 `False`
2. **跳过 ONNX checker**：由于 Custom 算子不在标准 ONNX opset 中，使用 `operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH` 跳过注册检查
3. **预计算 BEV Pool 索引**：通过 `model.get_bev_pool_input()` 预计算 `ranks_bev`、`ranks_depth`、`ranks_feat`、`interval_starts`、`interval_lengths`

### 产出

```log
bevdet/
├── bevdet_onnx/
│   └── bevdet_r50_all.onnx         # ONNX 模型 (~169MB)
├── bevdet-dev2.1/
│   └── bevdet-r50.pth              # 原始权重文件
└── BEVDet/                         # BEVDet 源码
```

***

## 5. MindSpore Lite 转换

### 配置文件

创建 `config.ini`：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 转换命令

```bash
Converter=mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

$Converter --fmk=ONNX \
  --modelFile=./bevdet_onnx/bevdet_r50_all.onnx \
  --outputFile=./bevdet_onnx/bevdet_r50_all_ascend \
  --optimize=ascend_oriented \
  --configFile=config.ini
```

### 参数说明

| 参数             | 说明                          |
| -------------- | --------------------------- |
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径                      |

### 产出

```log
bevdet_onnx/
├── bevdet_r50_all.onnx                     # ONNX 模型
└── bevdet_r50_all_ascend.mindir            # Ascend 优化版 MindIR
```

***

## 6. MindSpore Lite 推理

### 使用随机输入测试（本教程使用随机数测试）

```bash
cd examples/base_models/bevdet

python mindir_infer_bevdet.py \
  --model bevdet_onnx/bevdet_r50_all_ascend.mindir \
  --device ascend
```

### 使用 NuScenes 验证集真实数据

```bash
python mindir_infer_bevdet.py \
  --model bevdet_onnx/bevdet_r50_all_ascend.mindir \
  --device ascend \
  --ann-file data/nuscenes/bevdetv3-nuscenes_infos_val.pkl \
  --data-root data/nuscenes/ \
  --sample-idx 0
```

### 参数说明

| 参数             | 说明              | 默认值            |
| -------------- | --------------- | -------------- |
| `--model`      | MindIR 模型路径     | 必填             |
| `--device`     | 设备类型            | `cpu`          |
| `--device-id`  | Ascend 设备ID     | `0`            |
| `--batch`      | 批大小             | `1`            |
| `--warmup`     | 预热次数            | `5`            |
| `--runs`       | 测试次数            | `50`           |
| `--ann-file`   | NuScenes 标注文件路径 | `None`（使用随机输入） |
| `--data-root`  | NuScenes 数据根目录  | `None`         |
| `--sample-idx` | 样本索引            | `0`            |

### 执行日志（随机输入数据）

```log
=== BEVDet All-in-One MindIR Inference ===
Model: bevdet_onnx/bevdet_r50_all_ascend.mindir
Device: ascend
Using random input for testing
Input shape: (1, 6, 3, 256, 704)
Seed: 1024

--- Performance Benchmark (50 runs, warmup=5) ---
  Mean latency: 11.18 ms
  Min latency:  11.14 ms
  Max latency:  11.61 ms

--- Detection Output Shapes ---
  reg: (1, 2, 128, 64)
  height: (1, 1, 128, 64)
  dim: (1, 3, 128, 64)
  rot: (1, 2, 128, 64)
  vel: (1, 2, 128, 64)
  heatmap: (1, 10, 128, 64)
```

***

## 7. 性能数据

### 性能测试结果（Atlas 800 A2（313T）、Atals 300I Duo）

测试模型：BEVDet 整网\
测试条件：输入形状 (1, 6, 3, 256, 704)，固定随机种子 1024，50 次运行取平均

| 设备                      | 平均延迟 (ms) |
| ----------------------- | --------- |
| Atlas 800 A2（313T） | 11.18       |
| Atals 300I Duo | 18.28       |

***

## 8. 常见问题 FAQ

### 8.1 不支持 ONNX Runtime 推理

**问题：**

```log
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph:
[ONNXRuntimeError] : No Op registered for Custom with domain_version of 17
```

**原因：**
导出的 ONNX 模型中包含自定义算子 `Custom::BEVPoolV3`，ONNX Runtime 无法识别该算子。

**解决方案：**
需要通过 MindSpore Lite 的 `converter_lite` 工具将 ONNX 模型转换为 MindIR 格式，然后在 Ascend NPU 上运行。

### 8.2 Custom Op not registered (ONNX 导出时)

**问题：**

```log
RuntimeError: No Op registered for Custom with domain_version of 17
```

**解决方案：**
在 `torch.onnx.export` 中添加 `operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH` 参数。

### 8.3 梯度检查点导致导出失败

**问题：**

```log
RuntimeError: _Map_base::at
```

**解决方案：**
设置 `cfg.model.img_backbone.with_cp = False` 禁用 ResNet 的梯度检查点。

### 8.4 mmcv 版本不兼容

**问题：**

```log
ModuleNotFoundError: No module named 'mmcv'
```

**解决方案：**

```bash
pip uninstall mmcv mmcv-full mmcv-lite -y
pip install 'setuptools<70'
pip install mmcv-full==1.7.0 --no-build-isolation
```

### 8.5 numba 版本不兼容

**问题：**

```log
ValueError: numba 0.53.0 is not compatible with Python 3.10
```

**解决方案：**

```bash
pip install 'numba>=0.55'
```

### 8.6 mmdet3d 导入失败（CUDA 扩展缺失）

**问题：**

```log
ImportError: cannot import name 'TRTBEVPoolv2' from 'mmdet3d.ops.bev_pool_v2'
```

**原因：**\
BEVDet 包含 CUDA 自定义算子（`bev_pool_v2_ext`），在 CPU 环境下无法编译。

**解决方案：**\
修改 BEVDet 源码，将 CUDA 扩展导入改为可选：

```python
# 修改 mmdet3d/ops/bev_pool_v2/bev_pool.py
try:
    from . import bev_pool_v2_ext
except ImportError:
    bev_pool_v2_ext = None

# 修改 mmdet3d/models/detectors/bevdet.py
try:
    from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
except ImportError:
    TRTBEVPoolv2 = None

# 修改 mmdet3d/models/utils/spconv_voxelize.py
try:
    from spconv.pytorch.utils import PointToVoxel  # spconv-cu111  2.1.22
except ImportError:
    PointToVoxel = None

#修改setup.py 中跳过 CUDA 扩展编译
ext_modules=[],  # 移除 CUDA 扩展
```

***

## 9. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [BEVDet 项目](https://github.com/HuangJunJie2017/BEVDet)
- [mmdet3d 文档](https://mmdetection3d.readthedocs.io/)
- [CANN 文档](https://support.huaweicloud.com/cann/index.html)

***

## 10. 许可证

本教程遵循 BEVDet 项目的许可证（MIT License）。
