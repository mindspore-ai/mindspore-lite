# BEVDet 整网 ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 BEVDet 完整模型（含自定义 BEVPoolV3 算子）导出为 ONNX 格式，并转换为 MindSpore Lite MindIR 格式进行推理部署。

> **注意：**
> 1. 导出的 ONNX 模型中包含自定义算子 `BEVPoolV3`，因此**不支持 ONNX Runtime 推理**。需要通过 MindSpore Lite 转换为 MindIR 格式后在 Ascend NPU 上运行。`BEVPoolV3` 算子的安装和使能可以[参考链接](https://atomgit.com/Ascend/DrivingSDK/blob/master/README.md)。

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

## 3. 自定义 BEVPoolV3 算子

原始 BEVDet 使用 CUDA 自定义算子 `TRTBEVPoolv2`，在 CPU/Ascend 环境下无法使用。本教程使用 `BEVPoolV3` 昇腾融合算子替代，定义在 `export_bevdet_onnx.py` 中。

### 算子接口

```python
class CustomBEVPoolV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                with_depth, b, d, h, w, c):
        # 纯 torch 实现：先按 ranks_bev 排序，用 cumsum 分段求和，
        # 数学等价于 CUDA bev_pool_v2 / QuickCumsumCuda
        B, Z, Y, X, C = b, d, h, w, c
        depth_flat = depth.reshape(-1)
        feat_flat = feat.reshape(-1, C)
        contrib = depth_flat[ranks_depth.long()].unsqueeze(-1) * \
            feat_flat[ranks_feat.long()]

        sorted_idx = torch.argsort(ranks_bev)
        sorted_ranks = ranks_bev[sorted_idx].long()
        sorted_contrib = contrib[sorted_idx]

        N = sorted_contrib.shape[0]
        cumsum = sorted_contrib.cumsum(dim=0)
        cumsum_padded = torch.cat(
            [torch.zeros(1, C, dtype=cumsum.dtype), cumsum], dim=0
        )
        changes = sorted_ranks[1:] != sorted_ranks[:-1]
        starts = torch.cat([torch.zeros(1, dtype=torch.long),
                            torch.nonzero(changes).squeeze(-1) + 1])
        ends = torch.cat([starts[1:], torch.tensor([N])])

        seg_sums = cumsum_padded[ends] - cumsum_padded[starts]
        unique_ranks = sorted_ranks[starts]

        out = torch.zeros(B * Z * Y * X, C, device=depth.device, dtype=depth.dtype)
        out.scatter_(0, unique_ranks.unsqueeze(-1).expand(-1, C), seg_sums)
        return out.view(B, Z, Y, X, C).contiguous()

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                 with_depth, b, d, h, w, c):
        return g.op("Custom", depth, feat, ranks_depth, ranks_feat, ranks_bev,
                    with_depth_s=with_depth, b_i=b, d_i=d, h_i=h, w_i=w, c_i=c,
                    input_names_s=["depth", "feat", "ranks_depth", "ranks_feat", "ranks_bev"],
                    optional_input_names_s=["depth", "ranks_depth", "ranks_feat"],
                    type_s="BEVPoolV3",
                    input_index_i=[0, 1, 2, 3, 4],
                    output_names_s=["out"])
```

### 关键说明

- **forward 按 ranks_bev 排序后分段求和**：数学等价于原版 CUDA `bev_pool_v2`，在 PyTorch eager 下也能跑出正确数值（可用于精度对齐验证）
- **symbolic 注册为 `BEVPoolV3`**：ONNX 导出时该节点是 opaque，实际计算由 DrivingSDK 仓侧注册的 BEVPoolV3 C++ 算子完成
- **`input_index_i=[0, 1, 2, 3, 4]`**：部署侧 C++ 算子必须接受 **5 个 Tensor 输入**（depth, feat, ranks_depth, ranks_feat, ranks_bev）。
- **ONNX 导出使用 `ONNX_FALLTHROUGH`** 跳过 Custom 算子注册检查

### 5D 输入接口

Custom BEVPoolV3 直接接收 5D 张量 + 3 个 ranks 索引：

| 张量           | Shape                       | 说明                                   |
| ------------ | --------------------------- | ------------------------------------ |
| depth        | [B, N, D, H_feat, W_feat]  | softmax 后的深度概率分布                     |
| feat         | [B, N, H_feat, W_feat, C]  | 特征向量（已 permute 把 C 放最后一维）            |
| ranks_depth  | [N_Points] int             | 每个有效点在 `depth.reshape(-1)` 中的扁平索引   |
| ranks_feat   | [N_Points] int             | 每个有效点在 `feat.reshape(-1, C)` 中的扁平索引 |
| ranks_bev    | [N_Points] int             | 每个有效点在 BEV 输出中的扁平索引                  |

**实际 Shape**（bevdet-r50，B=1）：

| 张量         | Shape              |
| ---------- | ------------------ |
| depth      | [1, 6, 59, 16, 44] |
| feat       | [1, 6, 16, 44, 64] |
| ranks_*    | [N_Points]（约 20 万，动态） |

> **ranks 是 ONNX 的动态输入**（不再是固化的 buffer），由 host 侧通过 `model.get_bev_pool_input(camera_params)` 实时计算。一份 ONNX 适配任意相机布局/车型。

### 参数含义

| 参数          | 说明                   | 取值                                   |
| ----------- | -------------------- | ------------------------------------ |
| b           | Batch size           | int(B)，trace 时固化为 1（已知限制）             |
| d           | BEV Z 维大小 (bev_z)   | 1（grid_config['z'] 计算）             |
| h           | BEV Y 维大小 (bev_h)   | 128                                  |
| w           | BEV X 维大小 (bev_w)   | 128                                  |
| c           | BEV 通道数 (bev_c)     | 64                                   |
| with_depth  | 是否传入depth张量            | "true"                               |

`d/h/w/c` 来源于 `grid_config` 和模型架构（**与输入数据、相机参数无关，是模型常量**）：

```python
grid_config = {
    'x': [-51.2, 51.2, 0.8],   # (51.2 - (-51.2)) / 0.8 = 128  → bev_w
    'y': [-51.2, 51.2, 0.8],   # 128                              → bev_h
    'z': [-5, 3, 8],           # (3 - (-5)) / 8 = 1              → bev_z
}
numC_Trans = 64                                                  → bev_c
```

***

## 4. 导出 ONNX 模型

### 导出命令

```bash
cd examples/base_models/bevdet
mkdir bevdet_onnx

python export_bevdet_onnx.py \
  --config BEVDet/configs/bevdet/bevdet-r50.py \
  --checkpoint bevdet-dev2.1/bevdet-r50.pth \
  --device cpu \
  --output bevdet_onnx/bevdet_r50.onnx
```

### 参数说明

| 参数             | 说明            | 默认值                                       |
| -------------- | ------------- | ----------------------------------------- |
| `--config`     | BEVDet 配置文件路径 | `BEVDet/configs/bevdet/bevdet-r50.py`     |
| `--checkpoint` | 权重文件路径        | `bevdet-dev2.1/bevdet-r50.pth`            |
| `--device`     | 设备类型          | `cpu`                                     |
| `--output`     | 输出 ONNX 路径    | `bevdet_onnx/bevdet_r50.onnx` |
| `--opset`      | ONNX opset 版本 | `17`                                      |

> **注意：** [BEVDet代码仓和权重下载链接](https://github.com/HuangJunJie2017/BEVDet)

### 关键技术细节

1. **禁用 with_cp**：ResNet-50 的梯度检查点 (`with_cp=True`) 不兼容 ONNX 导出，构建模型时设为 `False`
2. **跳过 ONNX checker**：Custom 算子不在标准 ONNX opset 中，使用 `operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH` 跳过注册检查
3. **ranks 作为动态输入导出**：导出阶段通过 `load_first_sample()` 加载一条真实 NuScenes 数据作为 trace 的形状示例；导出后 ranks 成为 ONNX 的 4 个动态输入之一（img + 3 ranks），不再固化为 buffer
4. **collapse_z 对齐 BEVDet**：用 `torch.cat(x.unbind(dim=2), 1)` 替代 `view`，保证 Z>1 时通道顺序（Z 外 C 内）和 BEVDet 仓一致

### 产出

```log
bevdet/
├── bevdet_onnx/
│   └── bevdet_r50.onnx    # ONNX 模型（含 BEVPoolV3 节点）
├── bevdet-dev2.1/
│   └── bevdet-r50.pth              # 原始权重文件
└── BEVDet/                         # BEVDet 源码
```

***

## 5. MindSpore Lite 转换

### 配置文件

创建 `configs/config.ini`：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
[acl_build_options]
input_format="ND"
input_shape="img:1,6,3,256,704;ranks_depth:-1;ranks_feat:-1;ranks_bev:-1"
```

### 转换命令

```bash
Converter=mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

$Converter --fmk=ONNX \
  --modelFile=./bevdet_onnx/bevdet_r50.onnx \
  --outputFile=./bevdet_onnx/bevdet_r50_ascend \
  --optimize=ascend_oriented \
  --configFile=./configs/config.ini
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
├── bevdet_r50.onnx                     # ONNX 模型
└── bevdet_r50_ascend.mindir            # Ascend 优化版 MindIR
```

***

## 6. MindSpore Lite 推理

推理脚本 `infer_bevdet_mindir.py` 是**自包含的部署脚本**，包含：相机参数解析（支持 NuScenes 四元数 / 矩阵两种格式自动探测）、完整 BEVDet 测试模式图像预处理（PIL + ImageNet 归一化 + post_rot/post_tran）、host 侧 ranks 计算、MindSpore Lite 推理 + 性能 benchmark。

> **安全说明：** 脚本本身不包含 `pickle.load()`。用户需先运行下方「数据预处理」的 Python 代码，将 NuScenes 标注 pkl 文件转换为安全的 npz 格式（纯数值数组，无代码执行风险），再传给推理脚本。

### 推理流程

```log
[预处理后的 NuScenes npz + 6 张图像]
        ↓
1. 构建 BEVDet 模型（用于算 ranks）
2. 加载 NuScenes 样本：
   ├── 6 相机 PIL 加载 + resize/crop + ImageNet norm → imgs
   ├── npz 文件解析（四元数/矩阵） → sensor2egos, ego2globals
   ├── cam_intrinsic → intrins
   └── post_rot/post_tran 计算
3. compute_ranks：model.get_bev_pool_input(camera_params)
                                  → ranks_depth, ranks_feat, ranks_bev
4. MindSpore Lite 推理（4 个输入：img + 3 ranks）
                                  → 6 个检测头输出
```

### 数据预处理：将 pkl 转换为安全 npz 格式

`infer_bevdet_mindir.py` **不包含** `pickle.load()`。用户需先自行将 NuScenes 标注 pkl 文件中的单条样本提取为安全的 npz 文件，再传给推理脚本。

运行以下 Python 代码完成转换（将 unsafe 的 `pickle.load` 限制在此一步）：

```python
import pickle
import numpy as np
from pathlib import Path


def quat_to_rotation(quat) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)


def assemble_transform_from_quat(quat, tran) -> np.ndarray:
    """4x4 homogeneous transform from quaternion + 3-vec translation."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_to_rotation(quat)
    T[:3, 3] = np.asarray(tran, dtype=np.float32)
    return T


# 修改为实际路径
ann_file = "data/nuscenes/bevdetv3-nuscenes_infos_val.pkl"
sample_idx = 0
output_path = "sample_0.npz"

cam_order = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]

with open(ann_file, "rb") as f:
    data = pickle.load(f)  # unsafe — 仅限此预处理步骤
infos = data.get("infos", []) if isinstance(data, dict) else data
sample = infos[sample_idx]

cam_names, data_paths = [], []
cam_intrinsics, sensor2egos, ego2globals = [], [], []

for cam_name in cam_order:
    cam = sample["cams"][cam_name]
    cam_names.append(cam_name)
    data_paths.append(cam["data_path"])
    cam_intrinsics.append(np.asarray(cam["cam_intrinsic"], dtype=np.float32))
    # Build 4x4 transform matrices
    if "sensor2ego_rotation" in cam:
        sensor2ego = assemble_transform_from_quat(
            cam["sensor2ego_rotation"],
            cam["sensor2ego_translation"])
        ego2global = assemble_transform_from_quat(
            cam["ego2global_rotation"],
            cam["ego2global_translation"])
    elif "sensor2ego" in cam:
        sensor2ego = np.asarray(cam["sensor2ego"], dtype=np.float32)
        ego2global = np.asarray(cam["ego2global"], dtype=np.float32)
    else:
        raise KeyError(f"Unknown transform format for cam '{cam_name}'")
    sensor2egos.append(sensor2ego)
    ego2globals.append(ego2global)

np.savez(output_path,
         cam_names=np.array(cam_names, dtype=str),
         data_paths=np.array(data_paths, dtype=str),
         cam_intrinsics=np.stack(cam_intrinsics),
         sensor2egos=np.stack(sensor2egos),
         ego2globals=np.stack(ego2globals))
print(f"Saved to {output_path}")
```

| 变量             | 说明                         |
| -------------- | --------------------------- |
| `ann_file`     | NuScenes 标注 pkl 路径          |
| `sample_idx`   | 提取的样本索引                    |
| `output_path`  | 输出 npz 路径                  |

> **注意：** 上述代码内部使用 `pickle.load()`，但这仅限于此一步数据预处理。输出的 npz 文件只包含 numpy 数值数组，不携带任意代码执行风险，可安全地用于后续推理。
>
> `bevdetv3-nuscenes_infos_val.pkl` 数据集是由 [NuScenes 开源数据集 Full dataset (v1.0) Mini](https://www.nuscenes.org/nuscenes#download) 经过 BEVDet 代码仓中步骤 step 3. 处理得到。

### 使用 NuScenes 验证集真实数据

```bash
cd examples/base_models/bevdet
cd BEVDet
# # 从 NuScenes 官网下载开源数据集 Full dataset (v1.0) Mini
tar -zxvf v1.0-mini.tgz
mv v1.0-mini ./data/nuscenes
# # 运行 create_data_bevdet.py 之前需要修改 tools/create_data_bevdet.py 的 main 函数中 version = 'v1.0-mini'
python tools/create_data_bevdet.py
cd ..
ln -s BEVDet/data data

# 1. 预处理 pkl → npz（使用上方「数据预处理」小节的 Python 代码）
# 2. 推理
python infer_bevdet_mindir.py \
  --model bevdet_onnx/bevdet_r50_ascend.mindir \
  --config BEVDet/configs/bevdet/bevdet-r50.py \
  --checkpoint bevdet-dev2.1/bevdet-r50.pth \
  --device ascend \
  --sample-npz sample_0.npz \
  --data-root . \
  --postproc
```

### 参数说明

| 参数             | 说明                                | 默认值                                       |
| -------------- | --------------------------------- | ----------------------------------------- |
| `--model`      | MindIR 模型路径                       | 必填                                        |
| `--config`     | BEVDet 配置（用于 host 侧构建模型算 ranks）   | `BEVDet/configs/bevdet/bevdet-r50.py`     |
| `--checkpoint` | BEVDet 权重（同上）                     | `bevdet-dev2.1/bevdet-r50.pth`            |
| `--device`     | 设备类型                              | `ascend`                                  |
| `--device-id`  | Ascend 设备 ID                      | `0`                                       |
| `--batch`      | 批大小（仅随机模式生效）                      | `1`                                       |
| `--num-cams`   | 相机数量                              | `6`                                       |
| `--imH`        | 图像高度                              | `256`                                     |
| `--imW`        | 图像宽度                              | `704`                                     |
| `--seed`       | 随机输入种子（仅随机模式生效）                   | `1024`                                    |
| `--warmup`     | 预热次数                              | `5`                                       |
| `--runs`       | 测试次数                              | `50`                                      |
| `--sample-npz` | 预处理后的 NuScenes 样本 npz（需先通过上方数据预处理步骤生成） | `None`（使用随机输入） |
| `--data-root`  | NuScenes 数据根目录                    | `None`                                    |
| `--postproc`   | 启用后处理解码（result_deserialize → get_bboxes → bbox3d2result） | `False`（不启用） |

> **注意：**
> 预处理步骤将 pkl 中的单条样本提取为 npz，后续推理不再依赖 pkl 文件。如果只需随机输入进行功能测试，可省略预处理步骤直接运行推理脚本（`--sample-npz` 默认为 `None`，自动使用随机输入）。

### 执行日志（NuScenes 真实数据）

1、使用 NuScenes 验证集 sample 0 的实测结果：

```log
=== BEVDet MindIR Inference ===
  Model:      bevdet_r50_ascend.mindir
  Config:     BEVDet/configs/bevdet/bevdet-r50.py
  Checkpoint: bevdet-dev2.1/bevdet-r50.pth
  Device:     ascend
  Data mode:  NuScenes (sample_npz=sample_0.npz)

[1/4] Building BEVDet model (for ranks + postproc) ...
  task_heads: 1
[2/4] Preparing inputs ...
  Mode:       NuScenes
  imgs shape: (1, 6, 3, 256, 704)
  Sample npz: sample_0.npz
[3/4] Computing ranks from camera params ...
  ranks N_Points: 179832
[4/4] MindSpore Lite inference (warmup=5, runs=50) ...
  Mean latency: 16.69 ms
  Min latency:  16.49 ms
  Max latency:  18.04 ms

--- Raw Output Shapes ---
  task0_reg: (1, 2, 128, 128)
  task0_height: (1, 1, 128, 128)
  task0_dim: (1, 3, 128, 128)
  task0_rot: (1, 2, 128, 128)
  task0_vel: (1, 2, 128, 128)
  task0_heatmap: (1, 10, 128, 128)

--- Postprocessing Results ---
  Task 0: 387 boxes
    [pedestrian] score=1.000
    [pedestrian] score=1.000
    [pedestrian] score=0.981
    [pedestrian] score=0.976
    [pedestrian] score=0.969
```

2、整个nuscenes mini数据集Ascend与GPU的精度对比结果：

| 设备(模型类型)            | Ascend(mindir model) | GPU(TRT model) |
| ------------- | --------- | --------- |
| mAp | 0.2735     | 0.2805     |
| mATE | 0.8068     | 0.8022     |
| mASE | 0.4763     | 0.4738    |
| mAOE | 0.7853     | 0.8016      |
| mAVE | 1.1006     | 1.0282     |
| mAAE | 0.3994    | 0.3967    |
| NDS | 0.2900     | 0.2928      |

> **注意**：当前MindIR模型在Atlas 800I A2服务器上运行nuscenes mini的81个数据上输出的误差指标相比于GPU会劣化一些。

***

## 7. 性能数据

### 性能测试结果

测试模型：BEVDet 整网\
测试条件：输入形状 (1, 6, 3, 256, 704)，NuScenes 验证集真实数据 sample_idx=0，50 次运行取平均

| 设备            | 平均延迟 (ms) | 最小延迟 (ms) | 最大延迟 (ms) |
| ------------- | --------- | --------- | --------- |
| Atlas 800I A2 | 16.69     | 16.49     | 18.04     |

***

## 8. 常见问题 FAQ

### 8.1 不支持 ONNX Runtime 推理

**问题：**

```log
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph:
[ONNXRuntimeError] : No Op registered for Custom with domain_version of 17
```

**原因：**
导出的 ONNX 模型中包含自定义算子 `BEVPoolV3`，ONNX Runtime 无法识别该算子。

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
- [NuScenes 数据集下载](https://www.nuscenes.org/nuscenes#download)

***

## 10. 许可证

本教程遵循 BEVDet 项目的许可证（MIT License）。
