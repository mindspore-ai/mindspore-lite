---
name: third-party-custom-operator-integration
description: 安装第三方自定义算子包并通过Custom算子对接到MindSpore Lite推理链路。用户需要安装第三方算子包、配置ASCEND_CUSTOM_OPP_PATH、确认算子定义、在导出脚本中实现Custom算子改写、完成转换与验证时调用。
---

# 第三方自定义算子包安装与 Custom 算子对接

本技能覆盖第三方自定义算子包对接到 MindSpore Lite 推理链路的流程，聚焦于第三方算子包特有的**安装与环境配置**环节；Custom 改写、转换验证、精度对齐等通用流程直接引用现有 skill。

## 适用范围

- **适用**：用户或第三方团队开发的算子包（如 DrivingSDK、其他领域专用算子包等），需要通过 Custom 节点接入 MindSpore Lite 推理链路
- **不适用**：CANN 自带算子（CANN Toolkit/ops 已覆盖的标准算子）；MindSpore Lite 基于Ascend C实现的自定义算子

> CANN 内置算子无需额外安装和对接；MindSpore Lite 基于Ascend C实现的自定义算子不在本技能范围内。

## 何时调用

- 用户需要安装第三方算子包并对接到 MindSpore Lite
- 用户需要将某个第三方融合算子通过 Custom 节点接入推理链路
- 用户遇到 Custom 节点转换/推理失败，需要排查第三方算子包环境配置问题

---

## 通用对接流程

### 总体流程

```
安装第三方算子包 → 配置 ASCEND_CUSTOM_OPP_PATH → 确认算子定义 → Custom 改写 → 转换验证 → 精度对齐
      ↑ Step 1~2（本技能）                  ↑ Step 3~6（引用 custom_operator_fusion）
```

### Step 1：安装第三方算子包

按照算子包提供的安装方式完成安装。常见的安装方式：

- **pip 安装**：`pip install <package_name>`
- **源码编译安装**：下载源码 → 编译 → 安装 whl 包

安装后验证：

```bash
python3 -c "import <package_name>; print('<package_name> installed successfully')"
```

> 每个第三方算子包的安装流程、依赖要求和编译方式各不相同，**必须参考对应算子包的官方文档**，本技能的示例仅供参考。下方以 DrivingSDK 为例展示一种典型的源码编译安装流程。

### Step 2：配置 ASCEND_CUSTOM_OPP_PATH（必须）

安装第三方算子包后，需要配置 `ASCEND_CUSTOM_OPP_PATH` 使 MindSpore Lite 转换和推理时能找到算子包的融合算子实现：

```bash
# 查找算子包的安装路径
<PACKAGE>_PATH=$(python3 -c "import <package_name>; import os; print(os.path.dirname(<package_name>.__file__))")

# 配置环境变量（转换和推理前必须执行）
export ASCEND_CUSTOM_OPP_PATH=${<PACKAGE>_PATH}/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
```

> `vendors/customize` 是 CANN 自定义算子的标准目录结构，第三方算子包通常遵循此结构。若算子包使用了不同的路径，需根据其文档调整。

### Step 3~6：确认算子定义、Custom 改写、转换验证、精度对齐

以上步骤与 CANN 内置算子的 Custom 对接流程完全一致，详见：

**[custom_operator_fusion.md](../performance-optimization/references/custom_operator_fusion.md)**

该文档覆盖：
- 确认算子定义（规格对齐）
- `torch.autograd.Function` 实现 Custom 节点替换、属性规范、多输出处理
- patch 模型、重新导出 ONNX、ONNX 静态检查
- converter_lite 转换验证
- 推理功能与精度验证
- 常见故障排障

> 唯一区别：确认算子定义时，第三方算子包的定义文件位于 `${<PACKAGE>_PATH}/vendors/customize/op_impl/ai_core/tbe/config/`，而非 CANN 的 `opp/built-in/op_impl/ai_core/tbe/config/`。

---

## 示例：DrivingSDK（mx_driving）对接

> **注意**：本示例仅作参考，展示通用流程在具体算子包中的落地方式。不同第三方算子包的安装、配置和对接细节以各自的官方文档为准。

> 另有镜像安装和在线安装（pip install mx-driving）两种方式，有需要可参考官网：https://gitcode.com/Ascend/DrivingSDK

### Step 1 示例：DrivingSDK 源码安装

#### 硬件配套

| 产品 | 架构 | 是否支持 |
|------|------|---------|
| Atlas A2 训练系列产品 | x86_64, aarch64 | √ |
| Atlas A3 训练系列产品 | x86_64, aarch64 | √ |
| Ascend 950 系列产品 | x86_64, aarch64 | √ |

经过验证的操作系统：Ubuntu 22.04、openEuler 24.03。

#### 安装前准备

1. **安装 CANN**：安装配套版本的 NPU 驱动固件、CANN 软件（Toolkit、ops，可选 NNAL）并配置 CANN 环境变量，参考 [CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html)。
2. **安装 PyTorch 及 TorchNPU**：参考 [安装 TorchNPU](https://www.hiascend.com/document/detail/zh/Pytorch/2610/installguide/swinstall/docs/zh/installation_guide/installation_via_binary_package.md)。
3. **安装 Python 依赖**：`pip3 install -r requirements.txt`（`requirements.txt` 位于 DrivingSDK 项目根目录）。
4. **（可选）安装 protobuf-devel-3.14.0**：如需编译 ONNX 插件请安装；否则将 `CMakePresets.json` 中的 `ENABLE_ONNX` 改为 `FALSE`。
5. **设置运行环境权限**：建议使用非 root 用户，umask 设为 `0027`。

> 推荐使用 GCC 10.2 版本编译。

#### 下载与编译

```bash
git clone https://gitcode.com/Ascend/DrivingSDK.git
```

- **Atlas A2/A3 训练系列产品**：

  ```bash
  bash ci/build.sh --python=3.10
  ```
  或：
  ```bash
  python3.10 setup.py bdist_wheel
  ```

- **Ascend 950 系列产品**：

  ```bash
  bash ci/build.sh --a5 --python=3.10
  ```
  或：
  ```bash
  python3.10 setup.py bdist_wheel --a5
  ```

> `--python` 参数指定编译使用的 Python 版本，支持 3.8 及以上，缺省值为 3.8。生成的 whl 包位于 `dist/` 目录，命名规则为 `mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl`。

#### 安装与验证

```bash
cd dist
pip3 install mx_driving-1.0.0+git{commit_id}-cp{Python_version}-linux_{arch}.whl
```

```bash
python3 -c "import mx_driving; print('mx_driving installed successfully')"
```

### Step 2 示例：配置 DrivingSDK 环境变量

```bash
# 查找 mx_driving 包的安装路径
MX_DRIVING_PATH=$(python3 -c "import mx_driving; import os; print(os.path.dirname(mx_driving.__file__))")

# 配置环境变量（转换和推理前必须执行）
export ASCEND_CUSTOM_OPP_PATH=${MX_DRIVING_PATH}/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
```

### Step 3 示例：确认 DrivingSDK 算子定义

第三方算子包的定义文件位于包安装目录下，而非 CANN 的 `opp/built-in/op_impl/ai_core/tbe/config/` 路径：

```bash
# 查找算子定义文件
ls ${MX_DRIVING_PATH}/vendors/customize/op_impl/ai_core/tbe/config/
```

在该目录下 `***.json` 文件中找到目标算子的定义，确认输入/输出/属性规格。

确认 mx_driving Python 接口：

```python
import mx_driving
print(dir(mx_driving))

import inspect
print(inspect.signature(mx_driving.some_fusion_op))
```

确认算子的输入顺序、名称、类型，这将直接决定 Custom 节点中 `input_names_s` 的填写。

### Step 4 示例：Custom 改写中的 DrivingSDK 算子名

在 `symbolic` 方法中，`type_s` 填写 DrivingSDK 中的算子名：

```python
y = g.op(
    "Custom",
    *args,
    type_s="<DrivingSDK_OpName>",         # 如 "BEVPoolV3"、"VoxelPooling" 等
    input_names_s=["input1", "input2"],   # 按 DrivingSDK 算子定义的输入名顺序
    output_names_s=["output"],
    output_num_i=1,
    input_index_i=[0, 1],
)
```

### Step 6 示例：精度对齐标杆

使用 mx_driving API 在 Torch NPU 上跑出标杆：

```python
import torch
import torch_npu
import mx_driving

input_tensor = torch.from_numpy(input_data[0]).npu()

with torch.no_grad():
    ref_output_npu = mx_driving.target_fusion_op(input_tensor, ...)

ref_output_a = ref_output_npu.cpu().numpy()
```

---

## 常见故障排障

通用 Custom 改写相关的排障见 [custom_operator_fusion.md](../performance-optimization/references/custom_operator_fusion.md)。

第三方算子包特有的故障：

| 故障现象 | 原因 | 处理方式 |
|---------|------|---------|
| 转换时 Custom 节点未被识别 | 第三方算子包未安装或 `ASCEND_CUSTOM_OPP_PATH` 未配置 | 确认算子包 import 成功，检查 `ASCEND_CUSTOM_OPP_PATH` 是否指向算子包的 `vendors/customize` 目录 |
| 转换成功但推理精度异常 | Custom 算子输入顺序/类型与算子包定义不匹配 | 对照算子包定义文件（`${<PACKAGE>_PATH}/vendors/customize/op_impl/ai_core/tbe/config/` 下的 `***.json` ）逐项检查输入顺序、dtype |

## 执行检查清单

1. 第三方算子包安装成功（`import <package_name>` 无报错）
2. `ASCEND_CUSTOM_OPP_PATH` 已配置（指向算子包的 `vendors/customize` 目录）
3. 算子定义已确认（输入/输出/属性与 Custom 节点属性一致）
4. ONNX 中 Custom 节点属性完整（`type/input_names/output_names/output_num`）——详见 [custom_operator_fusion.md](../performance-optimization/references/custom_operator_fusion.md)
5. `input_index_i` 与实际输入顺序一致（有可选输入时）
6. converter_lite 转换成功（`CONVERT RESULT SUCCESS:0`）
7. MindIR 推理功能可用
8. 精度对齐验证通过——详见 [custom_operator_fusion.md](../performance-optimization/references/custom_operator_fusion.md) 的「转换与验证策略」
9. 性能收益明确且可重复
