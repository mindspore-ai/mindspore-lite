# Wan2.1

Wan2.1 模型的 NPU 多卡 Ulysses Sequence Parallel 适配器，基于 `lite_boost.parallel.ParallelManager` 实现序列维度上的多卡并行推理。

---

## 部署环境（以 Wan2.1-T2V-1.3B 为例）

| 组件 | 版本要求 |
|------|----------|
| PyTorch | 2.9.0 |
| torch_npu | 2.9.0 |
| Wan2.1 | [T2V-1.3B](https://github.com/Wan-Video/Wan2.1) |
| CANN | ≥ 8.5 |
| lite_boost | 0.1.0 |

> 硬件要求：华为昇腾 NPU（Ascend atlas 800I A2 及以上），已安装 HCCL 通信库。

---

## 使用教程

### 1. 快速开始

以下代码简要介绍了如何使用 `lite_boost` 进行 Wan2.1 模型的多卡并行推理。以WanT2V模式举例，会涉及到修改generate.py和wan/text2video.py 2个文件。

```python
from lite_boost.parallel import initialize_usp, ParallelManager

# 1. 初始化 HCCL 分布式环境
initialize_usp()

# 2. 加载模型
wan_t2v = wan.WanT2V(config)

# 3. 一键替换为 USP 版本（原地修改）
ParallelManager(wan_t2v)

# 4. 正常推理
output = wan_t2v.generate(*args)
```

`ParallelManager` 会自动完成以下替换：

```text
ParallelManager(model)
├── 替换 flash_attention       → NPU 兼容版本
├── 替换 self_attn.forward     → usp_attn_forward（含 all_to_all 通信）
└── 替换 model.forward         → usp_dit_forward（含序列分片 / 聚合）
```

### 2. 推理入口迁移

**修改前**（xfuser + NCCL）：

```python
# generate.py脚本
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

**修改后**（lite_boost + HCCL）：

```python
from lite_boost.parallel import initialize_usp, ParallelManager

if world_size > 1:
    initialize_usp()  # HCCL backend, 自动读 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
    from lite_boost.parallel import ParallelManager
    ParallelManager(wan_t2v)
# pipeline 中：
# if use_usp:
#     from .distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward
#     for block in self.model.blocks:
#         block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
#     self.model.forward = types.MethodType(usp_dit_forward, self.model)
```

另外推理入口需添加：

```python
# generate.py
import torch_npu

def generate(args):
    torch.npu.set_compile_mode(jit_compile=False)  # 必须
    ...
```

### 3. 运行命令

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503

torchrun --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    generate.py --task t2v-1.3B --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B --sample_shift 8 --sample_guide_scale 6 \
    --prompt "Your prompt here."
```

### 4. 约束条件

- `num_heads % world_size == 0`（1.3B 模型 12 heads 可被 2/3/4/6 卡整除）
- 序列长度自动 pad 到 `world_size` 的倍数

### 5. 性能数据

测试条件：Wan2.1-T2V-1.3B 、Ascend A2、官网demo样例 480P。

| 指标           | 优化前（单卡） | 优化后（4 卡） |
|--------------|---------|----------|
| 总耗时 (s)      | 559.5s  | 326s     |
| DiT 单步性能 (s) | 5.98s   | 1.32s    |

---

## 优化特性

### Context Parallel (CP)

采用 Ulysses Sequence Parallel 将 DiT 的序列维度切分到多卡，每卡持有完整模型权重，仅在 attention 层通过 `all_to_all` 交换激活。

详见 → [lite_boost/docs/parallel/context_parallel.md](../../../docs/parallel/context_parallel.md)

Wan2.1 适配要点：

- **RoPE 缓存**：按 grid shape + rank + world_size 预计算 cos/sin 表，padding 位置填零
- **Flash Attention**：自动回退链 FA3 → FA2 → `npu_prompt_flash_attention`

---

## 许可

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
