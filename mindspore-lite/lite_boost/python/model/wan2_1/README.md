# Wan2.1 Ulysses Sequence Parallel (USP) Adapter

Wan2.1 模型的 NPU 多卡 Ulysses Sequence Parallel 适配器，作为 `lite_boost.parallel.ParallelManager` 的模型后端。

## 概述

通过 `ParallelManager` 一键将 Wan2.1 模型替换为 USP 版本，实现序列维度上的多卡并行推理：

```python
from lite_boost.parallel import initialize_usp, ParallelManager

# 1. 初始化 HCCL 分布式环境
initialize_usp()

# 2. 加载模型
model = WanModel.from_pretrained(checkpoint_dir)

# 3. 一键替换为 USP 版本（原地修改）
ParallelManager(model)

# 4. 正常推理
output = model(x, t, context, seq_len)
```

## 工作原理

### Ulysses Sequence Parallel

将输入序列沿 `seq_len` 维度切分到多张 NPU 卡，每张卡持有 `S/P` 长度的 tokens。在 self-attention 层通过 `all_to_all` 通信交换 head 和 sequence 维度，做完 attention 后再换回。

```log
Forward all_to_all (scatter=heads, gather=seq):
    [B, S/P, H, D]  ──→  [B, S, H/P, D]
    切 heads，拼 seq          全局序列做 attention

Reverse all_to_all (scatter=seq, gather=heads):
    [B, S, H/P, D]  ──→  [B, S/P, H, D]
    切 seq，拼 heads          回到本地分片
```

### 本目录文件

| 文件 | 说明 |
|------|------|
| `model.py` | `usp_attn_forward`（self-attention 替换）、`usp_dit_forward`（model.forward 替换）、`usp_dit_forward_vace`（VACE forward_vace 替换） |
| `boost.py` | `boost_wan2_1(model)` — 对 WanModel 原地打补丁（替换 flash_attention、self_attn、model.forward） |
| `__init__.py` | 从 `model.py` 导出公开接口 |

依赖的通用层：

| 路径 | 说明 |
|------|------|
| `lite_boost.layers.rope` | 优化的 RoPE（float32 实运算 + 缓存） |
| `lite_boost.layers.attention` | NPU 兼容的 `flash_attention`（FA3 → FA2 → NPU 自动选择），统一返回 4D `[B, L, N, D]` |
| `lite_boost.parallel.context_parallel` | `all_to_all_4d` 通信原语 |

### ParallelManager 做了什么

```log
ParallelManager(model)
├── 替换 flash_attention（wan.modules.attention + wan.modules.model）
│   └── 影响：self-attention + cross-attention 全部使用 NPU 兼容版本
├── 替换每个 block.self_attn.forward → usp_attn_forward
│   └── 影响：self-attention 增加 all_to_all 通信
├── 替换 model.forward → usp_dit_forward
│   └── 影响：增加 seq chunk / all_gather / seq_pad
└── 如果是 VACE 模型 (VaceWanModel)：
    ├── 替换 vace_blocks[*].self_attn.forward → usp_attn_forward
    └── 替换 model.forward_vace → usp_dit_forward_vace
```

## 前提条件

1. `num_heads % world_size == 0`（如 1.3B 模型的 12 heads 可以被 2/3/4/6 卡整除）
2. 序列长度自动 pad 到 `world_size` 的倍数
3. NPU 环境需安装 `torch_npu`，HCCL 通信库可用

## 推理入口修改

### 修改前（xfuser + NCCL）

```python
import torch.distributed as dist

if world_size > 1:
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", ...)
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(...)
    initialize_model_parallel(...)

# pipeline 中：
if use_usp:
    from .distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward
    for block in self.model.blocks:
        block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
    self.model.forward = types.MethodType(usp_dit_forward, self.model)
```

### 修改后（lite_boost + HCCL）

```python
from lite_boost.parallel import initialize_usp, ParallelManager

if world_size > 1:
    initialize_usp()  # HCCL backend, 自动读 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT

# pipeline 中：
if use_usp:
    ParallelManager(self.model)
    self.sp_size = dist.get_world_size()
```

## 配置文件修改

NPU 对 `bfloat16` 支持不完善，需将 dtype 改为 `float32`：

```python
# shared_config.py
wan_shared_cfg.t5_dtype = torch.float32      # 原 torch.bfloat16
wan_shared_cfg.param_dtype = torch.float32   # 原 torch.bfloat16
```

推理入口需添加：

```python
# generate.py
import torch_npu

def generate(args):
    torch.npu.set_compile_mode(jit_compile=False)  # 必须
    ...
```

## 运行

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503

torchrun --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    generate.py --task t2v-1.3B --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B --sample_shift 8 --sample_guide_scale 6 \
    --prompt "Your prompt here."
```

## 适配新模型

如果要将此 USP 方案迁移到其他扩散模型（如 Stable Diffusion 3、Flux 等），需要：

1. 在 `lite_boost/model/` 下创建新模型目录
2. 实现对应模型的 `usp_attn_forward` 和 `usp_dit_forward`
3. 在 `model/__init__.py` 的 `SUPPORTED_MODELS` 中注册并添加分发分支
4. 核心逻辑与 wan2_1 相同：RoPE → all_to_all → attention → all_to_all reverse
