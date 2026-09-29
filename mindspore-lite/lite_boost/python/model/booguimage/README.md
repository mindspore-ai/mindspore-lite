# BooguImage (T2I-Turbo)

Boogu-Image 模型的 NPU 多卡并行适配器，基于 `lite_boost.BoostManager` 实现文生图流水线的一键多卡并行推理：DiT 按区域（double-stream / single-stream blocks）选择张量并行（TP）或 Ulysses 序列并行（SP），并可叠加通
信-计算重叠（cc_overlap）与 BSND 融合注意力。当前最快组合为双区域 SP + 双区域 cc_overlap + bsnd（2 卡 1.84 s，1024²）。

---

## 部署环境

| 组件 | 版本要求 |
|------|----------|
| CANN | 9.0.0 |
| PyTorch | 2.10.0 |
| torch_npu | 2.10.0rc2 |
| torchvision | 0.25.0 |
| diffusers | >= 0.35.2, < 0.39 |
| transformers | >= 4.57.3, < 6 |
| accelerate | >= 1.0 |
| einops | >= 0.7 |
| numpy | >= 1.26 |
| pillow | >= 10 |
| Boogu-Image | [T2I-Turbo](https://github.com/boogu-project/Boogu-Image)（`npu` 分支） |
| lite_boost | 0.2.0 |

> 硬件要求：华为昇腾 NPU，已安装 HCCL 通信库。
>
> **torchvision 安装注意**：torchvision 请从 **CPU index** 安装（不要使用 NVIDIA index），并加 `--no-deps` 避免覆盖已安装的 torch：
>
> ```bash
> pip install torchvision==0.25.0 --no-deps --index-url https://download.pytorch.org/whl/cpu
> ```

### 安装 Boogu-Image

lite_boost 的 NPU 兼容补丁依赖 `boogu` 包（设备校验、RoPE、SwiGLU、SDPA 等）。请从 Boogu-Image 官方仓库下载并安装 **`npu` 分支**：

```bash
git clone -b npu https://github.com/boogu-project/Boogu-Image.git
cd Boogu-Image
pip install -e .
```

未安装时，`BoostManager` 会在日志中提示 `The 'boogu' package is required ...`，NPU 补丁将被跳过。

---

## 使用教程

### 1. 快速开始

以下代码简要介绍了如何使用 `lite_boost` 进行 Boogu-Image T2I-Turbo 的多卡并行推理（以 T2I-Turbo 流水线为例）：

```python
import os

import torch
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

from lite_boost import BoostManager
from lite_boost.parallel import initialize_usp

# 1. 初始化 HCCL 分布式环境（读 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT）
initialize_usp()

# 2. 加载流水线
pipe = BooguImageTurboPipeline.from_pretrained(
    "models/Boogu-Image-0.1-Turbo",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 3. 一键替换为并行版本（原地修改，config 指向 YAML 配置文件）
boost_manager = BoostManager()
pipe = boost_manager(pipe, config="booguimage.yaml")

# 4. 正常推理（VAE 仅在 rank 0 上解码）
image = pipe(
    instruction=["..."],
    negative_instruction="",
    empty_instruction="",
    height=1024,
    width=1024,
    num_inference_steps=4,
    text_guidance_scale=1.0,
    image_guidance_scale=1.0,
    empty_instruction_guidance_scale=0.0,
    use_dmd_student_inference=True,
    dmd_conditioning_sigma=0.001,
    generator=torch.Generator(f"npu:{os.getenv('RANK', '0')}").manual_seed(42),
).images[0]
```

`BoostManager` 会自动完成以下替换：

```text
boost_manager(pipe, config="booguimage.yaml")
├── NPU 兼容补丁：设备校验器接受 npu、RoPE 频率 gather 安全化、apply_rotary_emb
│   （complex64 优先，不支持时回退融合算子）、SwiGLU 融合算子、SDPA mask 广播安全化
├── double/single 区域按 YAML 选择：
│   ├── TP：权重按 rank 分片（colwise/rowwise/overlap-KV）+ 融合 matmul-all-reduce
│   └── SP：权重不动，序列分片 + all_to_all 交换 head/seq（Ulysses）
└── VAE：仅 rank 0 解码，其余 rank 返回零张量
```

### 2. YAML 配置说明

`config` 指向的 YAML 文件用于按 transformer 区域选择并行算法与并行度（完整示例见 [booguimage.yaml](booguimage.yaml)）：

```yaml
Parallel:
  dit:
    double:            # double-stream blocks
      alg: SP          # current support [TP, SP]
      cc_overlap: true # SP only: FFN matmul 与 SP 通信重叠
    single:            # single-stream blocks
      alg: SP          # current support [TP, SP]
      cc_overlap: true # SP only: QKV/输出投影与 SP 通信重叠
    world_size: 2
bsnd: true             # BSND 融合注意力（顶层键，与 Parallel 平级）
```

| 配置项 | 合法值 / 约束 | 缺省值 |
|--------|---------------|--------|
| `Parallel.dit.double.alg` | `TP` / `SP`（SP 假定 batch_size==1、每 forward 序列布局固定） | `TP` |
| `Parallel.dit.double.cc_overlap` | `true` / `false`（要求 `alg: SP`；FFN linear_1/linear_3 与 a2a(Q)/gather(K,V) 重叠，swiglu+linear_2 与反向 a2a 重叠；bit-exact） | `false` |
| `Parallel.dit.single.alg` | `TP` / `SP` | `TP` |
| `Parallel.dit.single.cc_overlap` | `true` / `false`（要求 `alg: SP`；a2a(Q) 与 K/V 投影 + RoPE 重叠，反向 a2a 与 to_out 重叠；bit-exact） | `false` |
| `Parallel.dit.world_size` | 必须等于分布式 world_size（通信跑在全局进程组） | 分布式 world_size |
| `bsnd` | `true` / `false`（`npu_fusion_attention` BSND 布局，去掉 BNSD 转置；可与 1P / SP / cc_overlap 组合，**不可与 TP 同用**，见 BSND 小节说明） | `false` |

- **缺省即最优**：不传 `config`、或配置文件缺某段/缺某键时，均采用缺省配置 —— double/single 均 TP @ 分布式 world_size；
- **非法配置快速失败**：alg 不在白名单、world_size 不一致时直接抛 `ValueError`，报错信息指向应修改的配置项；
- **注意**：YAML 文件禁止使用 tab 缩进（PyYAML 会报 `ScannerError`），请使用空格缩进。

### 3. 运行命令

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503

torchrun --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    booguimage_2card.py
```

### 4. 约束条件

- TP 区域：`num_attention_heads % world_size == 0`（head 数需能被并行卡数整除）
- SP 区域：序列长度自动 pad 到 `2*world_size` 的倍数（保证 S_local 为偶数，规避 HCCL a2a 奇数分片的 ~2 倍传输时间断崖），attention 输出后去除 padding
- double 区域 SP：假定 `batch_size == 1`、每个 forward 的 instruct/img 序列布局固定
- 配置的 `world_size` 必须与实际分布式 world_size 一致（校验失败直接报错）

### 5. 性能数据

测试硬件：Ascend A2，Boogu-Image-0.1-Turbo，输出尺寸 1024x1024，4 步，
中文提示词（69 token，S_joint=4165），warmup 2 + 3 次取均值

| 方案 | 端到端耗时 (s) | s/step | PSNR vs 1P 基线 |
|------|---------------|-------|-----------------|
| 1P 基线（无 lite_boost） | 2.51 | 0.628 | ref |
| 1P + lite_boost | 2.51 | 0.627 | inf（逐像素一致） |
| 1P + lite_boost + bsnd | 2.35 | 0.587 | inf（逐像素一致） |
| SP(single)（2 卡） | 2.03 | 0.508 | 23.19 dB |
| SP(single+double)（2 卡） | 1.91 | 0.479 | 21.88 dB |
| + CC overlap（单流） | 1.87 | 0.468 | bit-exact |
| + CC overlap（双流 FFN 重叠） | 1.83 | 0.458 | bit-exact |

---

## 优化特性

### Tensor Parallel (TP)

将 transformer 权重按 rank 分片（Q colwise、KV overlap 分区、输出 rowwise、FFN colwise/rowwise），前向中通过 `npu_mm_all_reduce_base` 融合 matmul 与 all-reduce：

- **KV overlap 分区**：GQA 模型下 KV head 采用带重叠的分区方式，每卡多取一个 head，前向时按全局 query head 索引选择本地 KV，避免跨卡通信；
- **VATP 融合**：attention 输出投影与 FFN 下投影通过 `mm_all_reduce` 融合通信与计算，减少调度开销。

### Sequence Parallel (SP, single-stream / double-stream)

Ulysses 序列并行：权重保持完整，序列按 rank 分片。attention 前经 `all_to_all` 将 `[B, S/P, H, D]` 交换为 `[B, S, H/P, D]`，attention 后反向 `all_to_all` 恢复本地序列分片；首个 SP 区域块负责序列 pad + 切分，最后>一个块负责去 pad 聚合。单流区域按联合序列整体切分；双流区域（`usp_double_stream.py`）instruct/img 两路分别 pad 到 `2*world` 后切分，attention 内合并处理，假定 `batch_size == 1`。

序列 pad 对齐到 `2*world_size` 的倍数：S_local 恒为偶数，规避 HCCL a2a 奇数分片的 ~2 倍传输时间断崖（S_local=2083 奇数 0.997ms vs 2084 偶数 0.519ms）。

### 通信-计算重叠（cc_overlap, SP only）

SP 的三段通信（a2a(Q)、gather(K/V)、反向 a2a）投递到独立通信流，用 npu Event 与计算流同步，等待期间穿插无依赖的计算：

- **单流**（`comm_compute_overlap.sp_attention_overlap`）：a2a(Q) 与 K/V 投影 + RoPE 重叠；反向 a2a 在 `to_out` 之前投递。逐位一致（bit-exact）。
- **双流**（`comm_compute_overlap.sp_joint_attention_ffn_interleaved` / `sp_img_self_attention_ffn_interleaved`）：FFN 为重叠车道 —— linear_1/linear_3 与 a2a(Q)/gather(K,V) 并行，swiglu+linear_2 与反向 a2a 并行。FFN 始终走 `feed_forward.linear_1/2/3` 模块属性，便于后续 Quant 替换。逐位一致。

### BSND 融合注意力（bsnd）

attention 直接以 `npu_fusion_attention(input_layout="BSND")` 调用，去掉 SDPA 路径的 BNSD 双向转置（约占 DiT 时间 8%）。mask 语义取反（SDPA True=有效，融合核 True=屏蔽），全有效时跳过 mask。1P 全量块生效（逐像素>一致，单卡 2.51→2.35 s）。

**与 SP / cc_overlap 组合**：`bsnd` 是注意力内核的独立轴——统一内核（`attention_common.py`）的每个流水线体都接受 `backend` 参数（`attention_bnsd` SDPA 路径 / `attention_bsnd` 融合核路径），exchanger（Identity / Sync / CC）决定通信方式，backend 决定注意力求值方式，二者正交。GQA 统一为显式 `kv_idx` 头选择（`arange(Nq) // ratio`，与 Ulysses 头交换及单卡 `enable_gqa` 广播逐位一致）。1P 走 processor 替换（`attention_bsnd.py`，现为统一内核的薄封装）；SP/CC 下按 `bsnd` 选项将 SP/CC 内核绑定到对应 backend。

> **TODO（已知限制）**：`bsnd` 与 `alg: TP` 互斥，请勿同时使用。TP 对 processor 类打的 `__call__` 补丁会被 `install_bsnd_processors` 覆盖为 `_bsnd_single_stream_call`，该路径不含 `_lb_mm_all_reduce` 跨卡求和、也不感知 TP 的 KV 重叠分区，输出错误（2 卡实测复现）。待后续在 TP 内核中接入 `attention_bsnd` backend（`_tp_gqa_expand` 后头数已匹配，融合核可用）后解除此限制。

---

## 许可

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
