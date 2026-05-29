# RainFusionAttention

RainFusionAttention 是 lite_boost 组件中面向华为昇腾 NPU 的高性能融合注意力算子体系，提供两层接口：

| 接口 | 层级 | 说明 |
|------|------|------|
| `rain_fusion_attention` | 底层算子 | 直接封装 NPU 原生 `aclnnRainFusionAttention` 算子，提供块级稀疏融合注意力计算 |
| `sparse_attention` | 上层封装 | 在 `rain_fusion_attention` 基础上补充了完整的 token 重排、块级池化、稀疏 mask 生成和逆重排等前后处理逻辑 |

> API 使用请参见 [RainFusionAttention API 参考](../api/ops/RainFusionAttention.md)。

---

## 1. rf_v2 稀疏注意力流程

`sparse_type="rf_v2"` 时，会依次执行以下步骤：

```text
┌──────────┐   ┌──────────────────┐   ┌─────────────────────┐
│ 1. Token │──▶│ 2. Block-wise    │──▶│ 3. Top-k Sparse     │
│   重排   │   │    池化          │   │    Mask 生成        │
└──────────┘   └──────────────────┘   └────────┬────────────┘
                                               │
┌──────────┐   ┌──────────────────┐   ┌────────▼────────────┐
│ 5. 逆重排│◀──│ 4. rain_fusion_  │◀──│   (select_idx,      │
│          │   │    attention     │   │    select_num_idx)  │
└──────────┘   └──────────────────┘   └─────────────────────┘
```

### Step 1 — Token 重排（`rearrange_with_remaining`）

将 token 从平面排列 `(frame, h, w)` 转换为块交错排列 `(frame, hn, wn, hb, wb)`。子块大小为 8×8，有利于提高 NPU 上的数据局部性和计算效率。当 `h` 或 `w` 不能被 8 整除时，自动走 remainder 路径。

### Step 2 — 块级池化（`avgpool`）

对重排后的 Q/K/V 按 `block_size` 在序列维度上做平均池化，得到降采样的池化表示，用于高效计算稀疏 mask。

### Step 3 — Top-k 稀疏 Mask 生成（`get_blockwise_mask`）

- 在池化后的 QK 上计算 attention score 矩阵
- 对 score 做 softmax，基于 `sparsity` 比率进行 top-k 阈值截断
- 自动保护 text block（双向 attention）和首帧 block（可选）
- 输出 `select_idx` 和 `select_num_idx` 供底层算子使用

### Step 4 — Rain Fusion Attention（`rain_fusion_attention`）

传入重排后的 Q/K/V 和稀疏 mask，调用 NPU 原生 `aclnnRainFusionAttention` 执行块级稀疏融合注意力。

### Step 5 — 逆重排（`inv_rearrange_with_remaining`）

将注意力输出从块交错排列还原为原始的 `(frame, h, w)` 排列。

> **当 `txt_len > 0`** 时，文本 token 在 Step 1 中被分离，跳过空间重排，在 Step 5 中直接拼接回最终输出，确保文本 token 不会受到空间重排的影响。

---

## 2. 内部辅助函数

`sparse_attention.py` 提供了一系列可供高级用户直接调用的辅助函数：

### 2.1 avgpool

```python
def avgpool(input_tensor: Tensor, pool_size: int = 128, input_layout: str = 'BNSD') -> Tensor:
```

对输入张量在序列维度上按 `pool_size` 做块级均值池化。支持对齐序列的快速路径和非对齐序列的 remainder 路径。返回池化后的张量，序列长度缩减为 `ceil(seq_len / pool_size)`。

### 2.2 get_blockwise_mask

```python
def get_blockwise_mask(
    qkv_pool: Tensor,
    txt_len: int,
    sparsity: float,
    scale: float,
    pool_size: int,
    latent_shape_q: list,
    latent_shape_k: Optional[list] = None,
    input_layout: Optional[str] = None,
    return_binary: bool = False,
    protect_first_frame: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
```

在池化后的 QKV 上通过 softmax + top-k 阈值截断生成块级稀疏 mask。

- **`qkv_pool`**：`torch.cat([q_pool, k_pool, v_pool], dim=0)` 拼接的池化 QKV
- **`return_binary=True`** 时返回 `int8` 类型的二值 mask `[B, N, q_blocks, kv_blocks]`
- **`return_binary=False`** 时返回 `(select_idx, select_num_idx)` 供 `rain_fusion_attention` 使用

### 2.3 空间重排与逆重排

```python
def rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k=None, input_layout=None) -> Tensor:
def inv_rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k=None, input_layout=None) -> Tensor:
```

`rearrange_with_remaining` 将 token 从 `(frame, h, w)` 排列转换为 `(frame, hn, wn, 8, 8)` 块交错排列，子块固定为 8×8。支持 remainder 路径处理不能被 8 整除的维度。

`inv_rearrange_with_remaining` 是其逆操作，将块交错排列还原为原始空间排列。

### 2.4 do_tensor_rearrange_pooling

```python
def do_tensor_rearrange_pooling(query, key, value, text_len, pool_size,
                                latent_shape_q, latent_shape_k, input_layout) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
```

将 Q/K/V 进行重排和池化的一体化操作。当 `text_len > 0` 时自动分离文本 token。返回重排后的 `query_, key_, value_` 以及拼接的池化表示 `tensor_pool`。

### 2.5 do_tensor_inv_rearrange

```python
def do_tensor_inv_rearrange(tensor, text_len, latent_shape_q, latent_shape_k, input_layout) -> Tensor:
```

注意力输出的逆重排操作。当 `text_len > 0` 时自动分离文本 token 并跳过逆重排后拼接回来。

### 2.6 check_params

```python
def check_params(input_layout: str, sparse_type: Optional[str]) -> None:
```

参数校验，确保 `input_layout` 为 `"BSND"` 或 `"BNSD"`，`sparse_type` 为 `None` 或 `"rf_v2"`（rf_v3 / ada_bsa 待后续支持），否则抛出 `ValueError`。

---

## 3. 环境依赖与配置

### 3.1 硬件要求

- 华为昇腾 NPU（Atlas 800I A2及以上）
- 已安装 CANN 软件包（版本 ≥ 8.5）和 `torch_npu`

### 3.2 共享库加载

`rain_fusion_attention` 依赖 `liblite_boost_ops.so` 共享库，加载优先级如下：

1. **环境变量** `LITE_BOOST_OPS_LIB`：直接指定 `.so` 路径
2. **相对路径**：`python/ops/../lib/liblite_boost_ops.so` 或 `lite_boost_ops.so`
3. **系统路径**：在 `sys.path` 中搜索 `lite_boost/lib/` 下的 `.so` 文件

若加载失败，将抛出 `FileNotFoundError`。

### 3.3 自定义算子注册

底层算子通过 PyTorch custom op 机制注册：

```text
TORCH_LIBRARY(lite_boost, m)
└─ rain_fusion_attention → aclnnRainFusionAttention
```

NPU 后端实现绑定在 `PrivateUse1` dispatch key 上，仅在 NPU 设备上生效。

---

## 4. 注意事项

1. **设备限制**：`rain_fusion_attention` 仅在 NPU 上可用，不能在 CPU/GPU 上运行。
2. **输入 layout 一致性**：`q_input_layout` 和 `kv_input_layout` 目前仅支持 `"TND"` 和 `"BNSD"`，不支持 `"BSH"` 等格式。
3. **block_shape**：`block_shape[0]` 和 `block_shape[1]` 建议取相同值（如 `[128, 128]`），过小会增加 kernel launch 开销，过大会降低稀疏度灵活性。
4. **select_idx 对齐**：`select_idx` 的 `kv_blocks` 维度必须能够容纳最大的有效 KV 块数量，填充位必须为 `-1`。
5. **sparsity 参数**：`sparse_attention` 的 `sparsity` 仅控制池化后 attention score 的 top-k 阈值。设置为 `0.0` 不引入稀疏性，`0.5` 剪枝 50% 的块。由于 token 重排本身已带来性能提升，即使 `sparsity=0.0` 也可能比 dense attention 更快。
6. **latent_shape_q**：rf_v2 路径要求提供 `latent_shape_q = (t, h, w)`，即使 `t=1`。该参数用于将一维序列映射回空间维度以进行正确的重排。
