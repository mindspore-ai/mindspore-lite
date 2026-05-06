# Wan2.2 LiteBoost 并行推理适配

## 1. 部署环境

### 硬件要求

- 华为 Ascend NPU（A2 系列或更新）
- 推荐 ≥2 卡以启用 CP/DP 并行

### 软件依赖

| 组件 | 版本要求 |
|------|----------|
| CANN | 8.5 及以上 |
| torch | ≥ 2.8.0 |
| torch_npu | ≥ 2.8.0 |
| lite_boost | 0.0.1 |

> Wan2.2 原始依赖列表见：[requirements.txt](https://github.com/Wan-Video/Wan2.2/blob/main/requirements.txt)

### 安装

```bash
# 1. 设置 CANN 环境
source /path/to/cann/set_env.sh

# 2. 安装 Wan2.2
cd /path/to/Wan2.2
pip install -r requirements.txt

# 3. 编译 lite_boost ops
cd /path/to/mindspore-lite/lite_boost && mkdir -p build && cd build
cmake .. && make -j$(nproc)
export LITE_BOOST_OPS_LIB=$(pwd)/liblite_boost_ops.so

# 4. 安装 lite_boost Python 包
cd /path/to/mindspore-lite/lite_boost/python
pip install -e .
```

---

## 2. 使用教程

### 基本用法

LiteBoost 通过 `ParallelManager` 一行接入，自动对 Wan2.2 pipeline 应用 DiT CP 和 VAE DP。

以下示例为简化版推理脚本（非 Wan2.2 自带的 `generate.py`，后者包含复杂的命令行参数处理，此处仅为说明 LiteBoost 的接入方式）：

```python
import torch_npu
from lite_boost.parallel import initialize_usp, ParallelManager
from wan.configs import WAN_CONFIGS
from wan.textimage2video import WanTI2V

# 1. 初始化分布式环境
initialize_usp()

# 2. 加载 Wan2.2 pipeline
cfg = WAN_CONFIGS["ti2v-5B"]
cfg.param_dtype = torch.float32   # required for Ascend 310, optional for A2
cfg.t5_dtype = torch.float32
pipe = WanTI2V(config=cfg, checkpoint_dir=ckpt, device_id=local_rank,
               rank=rank, t5_fsdp=False, dit_fsdp=False, use_sp=False,
               t5_cpu=True, convert_model_dtype=True)

# 3. 一行启用并行
ParallelManager(pipe)

# 4. 正常推理
video = pipe.generate(prompt, img=img, size=(832, 480),
                      max_area=832 * 480, frame_num=81,  # 输出544×704 (由宽高比+VAE spatial_scale=16决定)
                      shift=3.0, sample_solver='unipc',
                      sampling_steps=20, guide_scale=5.0,
                      seed=42, offload_model=True)
```

### 运行命令

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# 单卡基线
ASCEND_RT_VISIBLE_DEVICES=0 python my_generate.py

# 4 卡 CP + DP
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 my_generate.py
```

### Pipeline 补丁

Wan2.2 官方 pipeline 中 `WanTI2V.i2v()` 和 `WanTI2V.t2v()` 将 VAE decode 包裹在 `if self.rank == 0:` 内。VAE DP 需要所有 rank 参与 decode 的 `all_gather` 通信，需去掉此条件：

```diff
-            if self.rank == 0:
-                videos = self.vae.decode(x0)
+            videos = self.vae.decode(x0)
```

### 性能

测试条件：Wan2.2-TI2V-5B、Ascend A2、sp=4、81 帧、20 steps、I2V 输入 832×1104、输出 544×704。

| 指标 | 优化前（单卡） | 优化后（4 卡 CP+DP） |
|------|----------------|-----------------------|
| 总耗时 (s) | 50.0 | 18.4 |
| DiT 耗时 (s) | 39.9 (40 calls, 1.00s/call) | 12.5 (40 calls, 0.31s/call) |
| VAE encode 耗时 (s) | 0.0 | 0.0 |
| VAE decode 耗时 (s) | 9.7 | 5.4 |
| 其他耗时 (s) | 0.4 | 0.5 |

> DiT 部分通过 USP（sp=4）实现 **3.2× 加速**（39.9s → 12.5s）；VAE decode 通过 DP 时间切片实现 **1.8× 加速**（9.7s → 5.4s）。VAE encode 为 0.0s 因 I2V 模式下输入为单帧图片无时间维。总耗时 **2.7× 加速**（50.0s → 18.4s）。

---

## 3. 优化技术

### 3.1 Context Parallel (CP)

采用 Ulysses Sequence Parallel 将 DiT 的序列维度切分到多卡，每卡持有完整模型权重，仅在 attention 层通过 `all_to_all` 交换激活。

详见 → [lite_boost/docs/parallel/context_parallel.md](../../../docs/parallel/context_parallel.md)

Wan2.2 适配要点：

- **Pad 策略**：序列末尾追加零 token 以对齐 `world_size`，all_to_all 后从 dim=1 末尾直接截取/补回
- **RoPE 缓存**：按 grid shape + rank + world_size 预计算 cos/sin 表，padding 位置填零
- **Flash Attention**：自动回退链 FA3 → FA2 → `npu_prompt_flash_attention`

### 3.2 Data Parallel (DP)

VAE 时间轴 DP 切分：视频沿 T 维度重叠切块，各卡独立处理连续的帧片段，all_gather 拼接后剥离重叠边界。

详见 → [lite_boost/docs/parallel/data_parallel.md](../../../docs/parallel/data_parallel.md)

Wan2.2 适配参数：

```python
apply_vae_dp(vae,
    spatial_scale=16,     # Wan2.2 VAE 空间压缩比
    temporal_stride=4,    # VAE 时间步长
    chunk_frames=12,      # 每 chunk 帧数（含重叠）
    overlap_frames=8,     # 重叠帧数
)
```

### 3.3 RoPE 优化

`wan2_2/rope.py` 针对 CP 场景的 RoPE 实现：

- **缓存机制**：按 grid shape、rank、world_size 索引 cos/sin 表，同配置无需重算
- **Padding 保护**：当序列不能整除 world_size 时，padding 位置填零 cos/sin（RoPE 输出恒为零）
- **实数运算**：float32 实数和替代 complex128 复数乘法

---

## 4. 许可证

本目录下的代码修改自 [Wan2.2](https://github.com/Wan-Video/Wan2.2)，原始代码版权归 Alibaba Wan Team 所有（Apache 2.0 License）。lite_boost 修改部分同样适用 Apache 2.0 License。
