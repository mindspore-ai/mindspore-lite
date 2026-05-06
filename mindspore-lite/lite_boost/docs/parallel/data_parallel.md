# VAE 数据并行时间切片 (Data Parallel)

## 原理

数据并行（DP）时间切片是一种面向视频 VAE 编码器/解码器的数据维度并行策略。核心思想是：将视频沿时间（T）维度切成多个连续帧片段，均匀分发给各张卡独立处理，最后收集拼接为完整结果。

### 核心流程

1. **时间维切分**：将输入视频沿 T 维度切成多个重叠的连续帧片段（chunk）
2. **均匀分发**：chunk 按连续分配策略分发到各卡（rank 0 拿第一段，rank 1 拿第二段……），保持全局顺序
3. **独立处理**：每卡对其分配的帧片段独立执行 VAE 编码/解码（每卡持有完整 VAE 模型权重，无需通信）
4. **全局拼接**：`all_gather` 收集所有卡的输出，沿 T 维度拼接为完整视频

```text
输入 [T, C, H, W]
    │  分区：1D 重叠切块
    ├── rank=0: frames [chunk_0]  (含 overlap_start)
    ├── rank=1: frames [chunk_1]  (含 overlap_start)
    ├── ...
    └── rank=N-1: ...
    │  分散：连续分配，保持全局顺序
    ↓
各卡独立处理
    ↓
    │  收集：all_gather (tensor-based, 无序列化)
    ↓
拼接 + 裁剪重叠 → 输出 [T_out, C, H, W]
```

### 为什么需要重叠帧？

视频 VAE 内部使用因果卷积（causal convolution），每帧的输出依赖前序帧的卷积缓存状态。在 chunk 边界处，缓存在新 chunk 开头是"冷"的，导致前几帧输出不正确。通过在 chunk 前端增加重叠帧并丢弃这些不正确的边界输出，可以保证拼接结果与单卡完整处理后裁剪一致。

### 约束

- 每个 chunk 必须有足够重叠帧以覆盖 VAE 的时间感受野
- `world_size == 1` 时退化为单卡，无分布式开销

## lite_boost 通用基础设施

### 五层基元：`data_parallel.py`

文件 `lite_boost/parallel/data_parallel.py` 提供**完全模型无关**的五层 DP 时间切片基元：

| 层 | 组件 | 功能 |
|----|------|------|
| ① 1D 分块几何 | `Chunk`, `compute_1d_chunks` | 纯数学：将 1D 域均匀切分为重叠块，末尾 padding 对齐 |
| ② 输出长度 | `compute_compress_len`, `compute_expand_len` | 纯数学：编码压缩 / 解码展开的长度计算 |
| ③ 混合 | `blend_along_axis` | 沿指定轴的线性交叉淡化 |
| ④ 分发/收集 | `scatter_evenly`, `gather_and_concat` | 连续块分配 + `all_gather` 收集（tensor-based，无序列化） |
| ⑤ 编排 | `dp_temporal_process` | 完整循环：分区 → 分发 → 处理 → 收集 → 裁剪 |

### `dp_temporal_process` 入口

模型适配器唯一需要调用的接口，接受一个模型相关的 `chunk_fn` 回调：

```python
from lite_boost.parallel.data_parallel import dp_temporal_process

def process_chunk(chunk):
    """模型相关的单 chunk 处理函数（无分布式逻辑）。"""
    with torch.amp.autocast("npu", dtype=dtype):
        return model(chunk)  # 纯本地计算

result = dp_temporal_process(
    input_tensor,            # 输入 tensor
    process_chunk,           # 模型相关回调 f(chunk) → output_chunk
    t_dim=1,                 # 时间维度索引
    chunk_frames=12,         # 每 chunk 帧数（含重叠）
    overlap_frames=8,        # 重叠帧数
    temporal_stride=4,       # 时间压缩比（vae_stride[0]）
    world_size=ws,
    rank=rank,
    device=device,
)
```

### DP 编排流程

```text
输入 tensor: [..., T, ...]  (T 维需切分)
  │
  ├─ ① 分区  compute_1d_chunks(T, chunk_frames, overlap_frames)
  │       ┌──────────┬──────────┬──────────┬──────┐
  │       │  chunk 0 │  chunk 1 │  chunk 2 │ ...  │  ← 均匀重叠分块
  │       └──────────┴──────────┴──────────┴──────┘
  │          ↑ overlap：chunk 间共享 overlap_frames 帧
  │
  ├─ ② 分发  scatter_evenly(chunks, world_size, rank)
  │       ┌─────────────────────────────────────────┐
  │       │ rank=0: [chunk 0, chunk 1]              │
  │       │ rank=1: [chunk 2, chunk 3]              │
  │       │ ...     (连续分配，padding 对齐数量)      │
  │       └─────────────────────────────────────────┘
  │
  ├─ ③ 处理  各卡独立调用 chunk_fn (无通信)
  │       每卡对本地 chunk 执行 VAE encode/decode
  │       首个真实 chunk ← 确定输出 shape
  │       padding chunk  → 补零 tensor
  │
  ├─ ④ 收集  gather_and_concat(all_results, all_chunks, ...)
  │       all_gather → 每卡拿到全局结果
  │       按 (rank, local_index) 天然全局顺序拼接
  │       每个 chunk 剥离 overlap_start 后沿 T 维连接
  │
  └─ ⑤ 裁剪  窄化到精确目标长度 target_len
          编码: target_len = compress_len(T, stride)
          解码: target_len = expand_len(T, stride)
```

> 重叠帧仅影响 chunk 边界：**chunk 开头的 overlap_start 帧被丢弃**（因果卷积冷缓存），chunk 末尾的 overlap_end 帧保留（缓存已预热）。

### 模型适配模式

各模型的 VAE DP 适配遵循统一模式：

1. 定义 `dp_encode` / `dp_decode` 函数，内部调用 `dp_temporal_process`
2. 通过 `apply_vae_dp(vae, ...)` 原地绑定到 VAE 实例（`types.MethodType`），替换 `vae.encode` / `vae.decode`

```python
# 通用模式
def dp_encode(self, videos):
    """替换 vae.encode — 内部调用 dp_temporal_process"""
    for u in videos:
        def enc(chunk):
            return self.model.encode(chunk.unsqueeze(0), self.scale).float().squeeze(0)
        results.append(dp_temporal_process(u, enc, t_dim=1, ...))
    return results

# 原地绑定
vae.encode = types.MethodType(dp_encode, vae)
```

具体模型的适配参数（`spatial_scale`、`temporal_stride` 等）因模型而异，详见各模型目录下的 README。

### 模型特定混合策略

当前 `blend_along_axis` 提供**线性交叉淡化**（linear cross-fade），适用于多数 VAE。如果模型需要在重叠区域使用不同的混合方式（如加权平均、指数淡化、或完全不混合仅裁剪），可在模型的 `chunk_fn` 回调中自行实现，`dp_temporal_process` 不强制使用内置 `blend_along_axis`。

### H/W 空间轴切分（规划中）

当前实现仅支持 **1D 切分**（沿 T 维度）。空间轴的 DP 切分通常为 **2D tile 切分**（如将每帧切分为 3×3 的 tile 网格分发到多卡），涉及跨 tile 边界的卷积 padding 对齐和 2D 分区策略，超出了现有 1D 基元的能力范围。此特性计划在后续版本中支持。
