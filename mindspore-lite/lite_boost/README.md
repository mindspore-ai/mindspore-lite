# LiteBoost

MindSpore Lite 推理加速工具包，面向昇腾 NPU 提供高性能自定义算子与多卡并行推理能力。

## 功能概览

| 模块 | 说明 |
|------|------|
| **RainFusionAttention** | 基于 `aclnnRainFusionAttention` 的稀疏注意力自定义算子，支持块级稀疏 mask，降低 Attention 计算量 |
| **FlashAttention** | NPU 兼容的 FlashAttention（FA3 → FA2 → NPU `npu_prompt_flash_attention` 自动选择） |
| **优化 RoPE** | float32 实运算 + cos/sin 表缓存，RoPE 耗时降低约 88% |
| **Ulysses Sequence Parallel** | 基于 HCCL 的多卡序列并行方案，一键将模型转为 USP 推理模式 |

## 目录结构

```text
lite_boost/
├── CMakeLists.txt          # C++ 算子编译配置
├── build.sh                # 一键构建脚本（CMake + wheel 打包）
├── version.txt             # 版本号
├── src/
│   └── ops/
│       ├── register_ops.cc               # PyTorch 自定义算子注册
│       └── plugin/
│           ├── rain_fusion_attention.cc   # RainFusionAttention C++ 实现
│           ├── rain_fusion_attention.h
│           ├── pytorch_npu_helper.cc      # NPU 算子调用辅助
│           └── pytorch_npu_helper.h
├── python/
│   ├── setup.py             # Python wheel 打包配置
│   ├── manager.py           # BoostManager
│   ├── ops/
│   │   └── rain_fusion.py   # RainFusionAttention Python 绑定
│   ├── layers/
│   │   ├── attention.py     # NPU 兼容 FlashAttention
│   │   └── rope.py          # 优化 RoPE 实现
│   ├── parallel/
│   │   ├── _initializer.py      # initialize_usp（HCCL 分布式初始化）
│   │   └── context_parallel.py  # all_to_all_4d 通信原语
│   └── model/
│       ├── __init__.py      # 模型适配器注册表
│       ├── wan2_1/          # Wan2.1 USP 适配
│       │   ├── boost.py     # 一键打补丁入口
│       │   ├── model.py     # USP attention / DiT forward 替换
│       │   └── README.md    # Wan2.1 适配详细说明
│       ├── wan2_2/          # Wan2.2 USP/DP 适配
│       ├── qwenimage/       # Qwen-Image VAE 并行实现
│       └── qwen_image_edit/ # Qwen-Image-Edit 适配
│           ├── boost.py               # 一键打补丁入口
│           └── qwen_image_edit.yaml   # 按模块优化配置示例
└── test/
    └── ops/
        └── test_rain_fusion_attention.py  # RainFusionAttention 测试
```

## 环境要求

- Python >= 3.8
- PyTorch（含 NPU 支持）
- torch_npu
- 昇腾 CANN 工具链（需设置 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME` 或 `ASCEND_CUSTOM_PATH` 环境变量）

## 编译安装

### 方式一：从项目根目录编译（推荐）

```bash
bash build.sh -O lite_boost -j 32
```

主构建脚本会自动将编译线程数、Debug/Release 等参数传递给 lite_boost。

### 方式二：独立编译

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
| `-i` | 增量编译（不清理 build 目录） | 关闭 |
| `-j[n]` | 编译线程数 | 8 |
| `-h` | 打印帮助信息 | - |

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASCEND_PATH` | 昇腾 CANN 根路径 | 按优先级自动检测：`ASCEND_CUSTOM_PATH` > `ASCEND_TOOLKIT_HOME` > `ASCEND_HOME_PATH` |
| `PYTORCH_INSTALL_PATH` | PyTorch 安装路径 | 从 Python site-packages 自动检测 |
| `PYTORCH_NPU_INSTALL_PATH` | torch_npu 安装路径 | 从 Python site-packages 自动检测 |
| `Python3_EXECUTABLE` | Python3 可执行文件路径 | 自动检测 |
| `CXX_STANDARD` | C++ 标准版本 | `17` |
| `ENABLE_GLIBCXX` | CXX11 ABI 控制（`ON`=ABI 1, `OFF`=ABI 0） | `ON` |

示例：

```bash
# 使用 ABI=0 编译
ENABLE_GLIBCXX=OFF bash build.sh -j 16

# 指定自定义 CANN 路径和 Python
ASCEND_PATH=/usr/local/Ascend/ascend-toolkit Python3_EXECUTABLE=/usr/bin/python3.9 bash build.sh
```

### 安装

```bash
pip install output/lite_boost-<version>-<tag>.whl
```

构建产物为 `liblite_boost_ops.so`，打包进 wheel 的 `lite_boost/lib/` 目录，安装后自动加载。

## 快速使用

### RainFusionAttention

```python
import lite_boost.ops as lite_ops

output, softmax_lse = lite_ops.rain_fusion_attention(
    query,               # [T, N, D]  TND 布局
    key,                 # [T, N, D]
    value,               # [T, N, D]
    select_idx,          # 稀疏 block 索引
    select_num_idx,      # 每个 query block 选择的 KV block 数
    block_shape=[128, 128],
    q_input_layout="TND",
    kv_input_layout="TND",
    num_key_value_heads=num_heads,
    scale_value=head_dim ** -0.5,
)
```

### Wan2.1 多卡并行推理

```python
from lite_boost import BoostManager
from lite_boost.parallel import initialize_usp
from wan import WanModel

# 1. 初始化 HCCL 分布式环境（读 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT）
initialize_usp()

# 2. 加载模型
model = WanModel.from_pretrained(checkpoint_dir)

# 3. 一键转为 USP 多卡推理（原地修改，返回模型本身）
boost_manager = BoostManager()
model = boost_manager(model)

# 4. 正常推理
output = model(x, t, context, seq_len)
```

```bash
# 多卡启动
export ASCEND_RT_VISIBLE_DEVICES=4,5
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503

torchrun --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    generate.py --task t2v-1.3B --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B
```

### 按模块配置优化（可选）

`BoostManager` 支持通过 YAML 配置文件按模块选择优化：配置被解析为 dict 后透传给对应模型的 boost 函数，各模型只读取自己支持的配置段，未配置的模块默认采用性能最优配置（如 DiT CP / VAE DP @ 分布式 world_size）：

```python
boost_manager = BoostManager()
pipe = boost_manager(pipe, config="boost.yaml")  # 一步式：BoostManager(pipe, config="boost.yaml")
```

以 Qwen-Image-Edit 为例（完整示例见 [python/model/qwen_image_edit/qwen_image_edit.yaml](python/model/qwen_image_edit/qwen_image_edit.yaml)）：

```yaml
Parallel:
  dit:                # DiT 上下文并行
    alg: CP           # current support [CP]
    world_size: 2
  vae:                # VAE 数据并行
    alg: DP           # current support [DP]
    world_size: 2
```

| 配置段 | 说明 |
|--------|------|
| `Parallel.dit.alg` | DiT 并行算法，当前仅支持 `CP`，缺省 `CP` |
| `Parallel.dit.world_size` | DiT 并行度，缺省为分布式 world_size，配置时需与其一致 |
| `Parallel.vae.alg` | VAE 并行算法，当前仅支持 `DP`，缺省 `DP` |
| `Parallel.vae.world_size` | VAE 并行度，缺省为 dit 的 world_size，配置时需与其一致 |

未配置的模块（含不传 `config`）默认采用性能最优配置（DiT CP + VAE DP @ 分布式 world_size）。

### 直接使用 Layers

```python
from lite_boost.layers import flash_attention, rope_apply

# NPU 兼容的 FlashAttention（自动选择最优后端）
x = flash_attention(q, k, v, q_lens=seq_lens, k_lens=seq_lens)

# 优化 RoPE
q = rope_apply(q, grid_sizes, freqs)
```

## 适配新模型

LiteBoost 采用模型注册机制，支持扩展到新的扩散模型：

1. 在 `python/model/` 下创建模型目录（如 `my_model/`）
2. 实现 `boost_my_model(model)` 函数，替换模型的 attention 和 forward
3. 在 `python/model/__init__.py` 的 `SUPPORTED_MODELS` 中注册模型类名
4. 在 `setup_model()` 中添加分发分支

核心流程：RoPE → all_to_all（scatter heads / gather seq）→ attention → all_to_all reverse → 输出投影。

## 许可证

Apache License 2.0
