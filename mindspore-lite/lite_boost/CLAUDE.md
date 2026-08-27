# LiteBoost

MindSpore Lite 推理加速工具包，面向昇腾 NPU 提供高性能自定义算子与多卡并行推理能力。

## 架构概览

项目由 **C++ 自定义算子层** 和 **Python 加速层** 组成：

- **C++ 算子**（[src/ops/](src/ops/)）：通过 PyTorch `TORCH_LIBRARY` 机制注册自定义算子，编译为 `liblite_boost_ops.so`，调用昇腾 CANN `aclnn` 接口（如 `aclnnRainFusionAttention`）。
- **Python 层**（[python/](python/)）：封装 C++ 算子的 Python 绑定、优化后的 attention/RoPE layer、以及基于 HCCL 的 Ulysses Sequence Parallel 多卡并行方案。

## 目录结构

| 路径 | 说明 |
|------|------|
| [src/ops/](src/ops/) | C++ 自定义算子源码 |
| [src/ops/register_ops.cc](src/ops/register_ops.cc) | PyTorch 自定义算子注册入口（`TORCH_LIBRARY(lite_boost, ...)`） |
| [src/ops/plugin/rain_fusion_attention.cc](src/ops/plugin/rain_fusion_attention.cc) | RainFusionAttention 算子 C++ 实现 |
| [src/ops/plugin/pytorch_npu_helper.cc](src/ops/plugin/pytorch_npu_helper.cc) | NPU 算子调用辅助工具 |
| [python/ops/](python/ops/) | Python 算子绑定（`rain_fusion_attention`、`sparse_attention`） |
| [python/layers/](python/layers/) | 优化后的 layer 实现（FlashAttention、RoPE） |
| [python/parallel/](python/parallel/) | Ulysses Sequence Parallel 多卡并行（`context_parallel.py`、`data_parallel.py`、`_initializer.py`） |
| [python/manager.py](python/manager.py) | BoostManager 并行入口（`from lite_boost import BoostManager`），负责解析 YAML 优化配置并透传给各模型 boost 函数 |
| [python/model/](python/model/) | 模型适配器注册表（当前支持 Wan2.1 / Wan2.2 / Qwen-Image-Edit） |
| [test/](test/) | 测试用例 |
| [CMakeLists.txt](CMakeLists.txt) | CMake 构建配置 |
| [build.sh](build.sh) | 一键构建脚本（CMake + wheel 打包） |

## 环境依赖

- Python >= 3.8
- PyTorch（含 NPU 支持）
- torch_npu
- 昇腾 CANN 工具链（需设置 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME` 或 `ASCEND_CUSTOM_PATH` 环境变量）

## 构建与安装

### 从项目根目录编译（推荐）

```bash
bash build.sh -O lite_boost -j 32
```

主构建脚本会自动将编译线程数、Debug/Release 等参数传递给 lite_boost。

### 独立编译

```bash
cd mindspore-lite/lite_boost
bash build.sh -j 32
```

### build.sh 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d` | Debug 模式 | Release |
| `-r` | Release 模式 | Release |
| `-v` | 显示完整编译命令 | 关闭 |
| `-i` | 增量编译 | 关闭 |
| `-j[n]` | 编译线程数 | 8 |
| `-h` | 打印帮助 | - |

### 环境变量（覆盖 CMake 默认值）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_PATH` | 昇腾 CANN 根路径 | 自动检测：`ASCEND_CUSTOM_PATH` > `ASCEND_TOOLKIT_HOME` > `ASCEND_HOME_PATH` |
| `PYTORCH_INSTALL_PATH` | PyTorch 安装路径 | 从 site-packages 自动检测 |
| `PYTORCH_NPU_INSTALL_PATH` | torch_npu 安装路径 | 从 site-packages 自动检测 |
| `Python3_EXECUTABLE` | Python3 路径 | 自动检测 |
| `CXX_STANDARD` | C++ 标准版本 | `17` |
| `ENABLE_GLIBCXX` | CXX11 ABI 控制（`ON`=ABI 1, `OFF`=ABI 0） | `ON` |

构建流程：`build.sh` → 校验 CANN 环境变量 → CMake 编译 `liblite_boost_ops.so` → `setup.py bdist_wheel` 将 `.so` 打包进 wheel 的 `lite_boost/lib/` 目录。

## 关键模块

### RainFusionAttention

基于 `aclnnRainFusionAttention` 的块级稀疏注意力算子，支持 TND 布局和可配置的 block sparse mask。

- C++ 实现：[src/ops/plugin/rain_fusion_attention.cc](src/ops/plugin/rain_fusion_attention.cc)
- Python 绑定：[python/ops/rain_fusion.py](python/ops/rain_fusion.py)
- 使用方式：`torch.ops.lite_boost.rain_fusion_attention(...)`，通过 `torch.ops.load_library` 加载 `.so`

### FlashAttention

NPU 兼容的 FlashAttention，按优先级自动选择后端：FA3 → FA2 → `npu_prompt_flash_attention`。

- 实现：[python/layers/attention.py](python/layers/attention.py)

### 优化 RoPE

float32 实运算 + cos/sin 表缓存，RoPE 耗时降低约 88%。需要分布式环境（依赖 `dist.get_world_size()`）。

- 实现：[python/layers/rope.py](python/layers/rope.py)

### Ulysses Sequence Parallel

基于 HCCL 的多卡序列并行方案，核心流程：RoPE → all_to_all（scatter heads / gather seq）→ attention → all_to_all reverse → 输出投影。

- 加速管理：[python/manager.py](python/manager.py)
- 分布式初始化：[python/parallel/_initializer.py](python/parallel/_initializer.py)
- 通信原语：[python/parallel/context_parallel.py](python/parallel/context_parallel.py)

### 模型适配

通过 `BoostManager` 一键将支持模型转为 USP 推理模式（`boost_manager = BoostManager(); pipe = boost_manager(pipe)`，也支持 `BoostManager(pipe)` 一步式）。模型注册表在 [python/model/__init__.py](python/model/__init__.py)。

可选传入 YAML 配置文件按模块选择优化：`pipe = boost_manager(pipe, config="boost.yaml")`。BoostManager 将其解析为 dict 后透传给各模型的 boost 函数，各模型只读取自己支持的配置段，未配置的模块默认采用性能最优配置（如 DiT CP / VAE DP @ 分布式 world_size）。以 Qwen-Image-Edit 为例（[python/model/qwen_image_edit/qwen_image_edit.yaml](python/model/qwen_image_edit/qwen_image_edit.yaml)）：

```yaml
Parallel:
  dit:
    alg: CP  # current support [CP]
    world_size: 2
  vae:
    alg: DP  # current support [DP]
    world_size: 2
```

适配新模型：在 `python/model/` 下创建目录，实现 `boost_xxx(model, config=None)` 函数（自行解析 config 中支持的段），在 `_MODEL_MATCH_TABLE` 与 `_BOOST_REGISTRY` 中注册。

## 编码规范

- C++ 算子使用 C++17 标准，ABI 通过 `ENABLE_GLIBCXX` 控制（默认 ABI=1）
- Python 文件头部包含 Apache 2.0 License 版权声明
- 共享库通过 `torch.ops.load_library` 延迟加载，首次 import `lite_boost.ops` 时自动触发
- 可通过 `LITE_BOOST_OPS_LIB` 环境变量指定 `.so` 路径，覆盖默认搜索逻辑
