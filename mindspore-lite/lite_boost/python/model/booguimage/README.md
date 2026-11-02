# BooguImage (T2I-Turbo)

Boogu-Image 模型的 NPU 多卡并行适配器，基于 `lite_boost.BoostManager` 实现文生图流水线的一键多卡并行推理：DiT 按区域（double-stream / single-stream blocks）选择张量并行（TP）或 Ulysses 序列并行（SP），并可叠加通
信-计算重叠（cc_overlap）与 BSND 融合注意力。当前最快组合为双区域 SP + 双区域 cc_overlap（2 卡，1024²）；bsnd 仅可用于 1P（与 TP / SP 均不可组合，见 BSND 小节说明）。

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

lite_boost 的 NPU 兼容补丁依赖 `boogu` 包（设备校验、RoPE、SwiGLU、SDPA 等）。请从 Boogu-Image 官方仓库下载并安装 **`npu` 分支**（源码仓库：[GitHub - boogu-project/Boogu-Image](https://github.com/boogu-project/Boogu-Image)，权重同源见下一节）：

```bash
git clone -b npu https://github.com/boogu-project/Boogu-Image.git
cd Boogu-Image
pip install -e .
```

未安装时，`BoostManager` 会在日志中提示 `The 'boogu' package is required ...`，NPU 补丁将被跳过。

### 下载权重

测试脚本通过 `--model` 参数读取权重目录（缺省为运行目录下的 `models/Boogu-Image-0.1-Turbo`，推荐显式配置，见运行命令一节）。在 `lite_boost/test/models/booguimage/` 下执行：

```bash
# HuggingFace
huggingface-cli download Boogu/Boogu-Image-0.1-Turbo --local-dir models/Boogu-Image-0.1-Turbo

# 或 ModelScope
modelscope download --model Boogu/Boogu-Image-0.1-Turbo --local_dir models/Boogu-Image-0.1-Turbo
```

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
bsnd: false            # BSND 融合注意力（顶层键；仅可与 1P 同用，见 BSND 小节说明）
```

| 配置项 | 合法值 / 约束 | 缺省值 |
|--------|---------------|--------|
| `Parallel.dit.double.alg` | `TP` / `SP`（SP 假定 batch_size==1、每 forward 序列布局固定） | `TP` |
| `Parallel.dit.double.cc_overlap` | `true` / `false`（要求 `alg: SP`；FFN linear_1/linear_3 与 a2a(Q)/gather(K,V) 重叠，swiglu+linear_2 与反向 a2a 重叠；bit-exact） | `false` |
| `Parallel.dit.single.alg` | `TP` / `SP` | `TP` |
| `Parallel.dit.single.cc_overlap` | `true` / `false`（要求 `alg: SP`；a2a(Q) 与 K/V 投影 + RoPE 重叠，反向 a2a 与 to_out 重叠；bit-exact） | `false` |
| `Parallel.dit.world_size` | 必须等于分布式 world_size（通信跑在全局进程组） | 分布式 world_size |
| `bsnd` | `true` / `false`（`npu_fusion_attention` BSND 布局，去掉 BNSD 转置；**仅可与 1P 同用**，与 TP / SP 组合均不可用，见 BSND 小节说明） | `false` |

- **缺省即最优**：不传 `config`、或配置文件缺某段/缺某键时，均采用缺省配置 —— double/single 均 TP @ 分布式 world_size；
- **非法配置快速失败**：alg 不在白名单、world_size 不一致时直接抛 `ValueError`，报错信息指向应修改的配置项；
- **注意**：YAML 文件禁止使用 tab 缩进（PyYAML 会报 `ScannerError`），请使用空格缩进。

### 3. 运行命令

批量测试用例脚本位于 `lite_boost/test/models/booguimage/`（单脚本 `booguimage_test.py`，boost 配置直接使用同目录随库的
[booguimage_sp2.yaml](../../../test/models/booguimage/booguimage_sp2.yaml)、
[booguimage_tp2.yaml](../../../test/models/booguimage/booguimage_tp2.yaml) 与
[booguimage_1p.yaml](../../../test/models/booguimage/booguimage_1p.yaml)；固定测试提示词、seed 42、4 步 DMD 推理，不修改 lite_boost 组件代码）：

- 性能用例（缺省）：`--resolution` 指定分辨率（缺省 512），一次运行跑完 `sp2`（SP 双区域 + cc_overlap 全开）、`tp2`（TP 双区域）与 `1p`（1P + bsnd）三种形态，顺序 torchrun 拉起，汇总至 `<out-dir>/summary.txt`；
- 精度用例（`--baseline` 指定 npz 参考图时）：liteboost 1p / sp2 / tp2 各出一张图与参考图做 PSNR ≥ 20 判定（`--psnr-min` 可调），不达标退出码 1。多卡数值差异随分辨率变化（见"精度对比"一节），精度判定固定 1024。

测试脚本参数：

| 参数 | 说明 | 缺省 |
|------|------|------|
| `--resolution` | 方形出图边长；性能 512/1024/2048 各跑一次，精度固定 1024 | 512 |
| `--model` | 权重目录 | models/Boogu-Image-0.1-Turbo |
| `--out-dir` | 出图、日志与 `summary.txt` 输出目录 | output |
| `--steps` | DMD 推理步数 | 4 |
| `--seed` | generator seed（统一 rank-NPU generator，与参考 npz 同流） | 42 |
| `--modes` | 性能用例要跑的形态，逗号分隔（`sp2` SP 双卡 / `tp2` TP 双卡 / `1p` 单卡） | sp2,1p |
| `--warmup` / `--runs` | 性能用例预热 / 计次轮数 | 2 / 3 |
| `--prompt` | 测试提示词（固定文案见下） | 脚本内置 |
| `--baseline` | npz 参考图路径，给定即进入精度用例 | 关 |
| `--psnr-min` | 精度门禁阈值（dB） | 20 |
| `--offload` | 编码器/VAE 主机换入换出，显存不足的开发卡验证用（见下文） | 关 |

各形态均不搬移 mllm：编码器与 DiT 同卡部署（多卡路径由 master 分支的 `tp_pipeline_call` 原生放置，编码器落在 rank 0），要求单卡显存能同时容纳 DiT 与 mllm（TP 形态 DiT 权重按卡切分，显存压力更小）。

显存放不下时（如开发验证卡）可加 `--offload`：每次编码前把 transformer 换出到主机内存、mllm 换入，编码完成再换回，VAE 同理在解码前后换入换出，单卡即可跑通用例。出图与不搬移部署逐像素一致，但计时含权重主机往返——多卡形态的文本编码有 cache、实测轮基本不受影响，1p 每轮重编码、实测轮含全套往返，因此 `--offload` 只用于开发卡验证链路与精度，正式性能数据一律不带它采集（A2 313T 显存充足，无需该选项）。

按顺序执行（每步不可省）：

```bash
# 1) 环境：CANN 9.0.0 与依赖版本见「部署环境」一节
source <CANN 安装路径>/set_env.sh
conda activate boogu-image

# 2) 进入测试目录
cd lite_boost/test/models/booguimage

# 3) 模型权重与输出目录参数（--model / --out-dir 均有缺省值，此处显式写出）
MODEL=$PWD/models/Boogu-Image-0.1-Turbo   # 权重目录（按「下载权重」一节下载；已有权重时直接指向该目录）
OUT=$PWD/output                           # 出图、日志与汇总表的输出目录

# 4) 性能：每个分辨率跑一次
python booguimage_test.py --resolution 512  --model "$MODEL" --out-dir "$OUT"
python booguimage_test.py --resolution 1024 --model "$MODEL" --out-dir "$OUT"
python booguimage_test.py --resolution 2048 --model "$MODEL" --out-dir "$OUT"   # 每次打印 RESULT 行并刷新汇总 $OUT/summary.txt

# 5) 精度：--baseline 提供 npz 参考图即进入精度对比，PSNR>=20 门禁，不达标退出码 1
python booguimage_test.py --resolution 1024 --model "$MODEL" --out-dir "$OUT" --baseline reference.npz
```

参数缺省值见上表，完整列表见 `python booguimage_test.py --help`。测试提示词缺省写在脚本内：

```text
生成一张街拍的照片，要有垃圾桶，红绿灯。画面中心有一个拾荒的老人，满脸沧桑。照片要有摄影质感，以及徕卡相机的街拍质感。
```

**npz 参考图（可选）**：`--baseline` 指向的 npz 内保存一张纯 PyTorch 流水线的输出图，要求与测试用例同提示词、seed 42、4 步、同分辨率，且使用 NPU generator（与测试用例同流）。生成方式（昇腾 NPU 环境）：

```python
import os

import numpy as np
import torch
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

device = "npu:0"
os.environ["device"] = device   # 官方 inference_npu.py 同款；boogu 按 import 时的该值选择 NPU 融合实现，必须在 import boogu 前设置
pipe = BooguImageTurboPipeline.from_pretrained(
    "models/Boogu-Image-0.1-Turbo", torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device)
image = pipe(
    instruction=["生成一张街拍的照片，要有垃圾桶，红绿灯。画面中心有一个拾荒的老人，"
                 "满脸沧桑。照片要有摄影质感，以及徕卡相机的街拍质感。"],
    negative_instruction="",
    empty_instruction="",
    height=1024,
    width=1024,
    max_input_image_pixels=1024 * 1024,
    max_input_image_side_length=2048,
    num_inference_steps=4,
    text_guidance_scale=1.0,
    image_guidance_scale=1.0,
    empty_instruction_guidance_scale=0.0,
    use_dmd_student_inference=True,
    dmd_conditioning_sigma=0.001,
    generator=torch.Generator("npu:0").manual_seed(42),   # NPU generator，与测试用例一致
    device=device,
).images[0]
np.savez("reference.npz", image=np.asarray(image))   # (H, W, 3) uint8；脚本也接受 (1,H,W,3)/(3,H,W)/(1,3,H,W) 与 float 0-1
```

### 4. 约束条件

- TP 区域：`num_attention_heads % world_size == 0`（head 数需能被并行卡数整除）
- SP 区域：序列长度自动 pad 到 `2*world_size` 的倍数（保证 S_local 为偶数，规避 HCCL a2a 奇数分片的 ~2 倍传输时间断崖），attention 输出后去除 padding
- double 区域 SP：假定 `batch_size == 1`、每个 forward 的 instruct/img 序列布局固定
- 配置的 `world_size` 必须与实际分布式 world_size 一致（校验失败直接报错）
- `bsnd` 仅可与 1P 同用：与 `alg: TP` 互斥；与 `alg: SP` 组合当前不可用（双流 SP 下 GQA 头按 rank 切分为非整数，`npu_fusion_attention` 报 aclnn EZ1001 参数错误，2 卡实测复现），见 BSND 小节说明

### 5. 性能数据

sp2 = SP 双区域 + cc_overlap 全开，tp2 = TP 双区域，1p = 1P + bsnd；mllm 与 DiT 同卡部署（见运行命令一节）。下表为 Ascend A2 313T 实测（不带 `--offload`）；双卡收益随分辨率增大——512 档通信开销高于并行收益（sp2 慢于 1p），1024 起反超约 22%。

| 用例 | 分辨率 | 端到端 (s) |
|------|--------|-----------|
| sp2 (SP+cc) | 512 | 1.115 |
| sp2 (SP+cc) | 1024 | 1.808 |
| sp2 (SP+cc) | 2048 | 9.819 |
| 1p (+bsnd) | 512 | 0.887 |
| 1p (+bsnd) | 1024 | 2.322 |
| 1p (+bsnd) | 2048 | 13.1887 |

> tp2（TP）形态性能暂不采集；精度实测 22.82 dB，见精度对比一节。
>
> 注：开发验证卡 A2 280T 单卡无法同时容纳 DiT 与 mllm（权重约 19.2 + 15.2 GiB，512 即 OOM），加 `--offload` 方可跑通用例（计时含权重主机往返，仅验证链路与精度，数据不入上表）。

### 6. 精度对比

同一请求分别用纯 torch 流水线（参考图，不 import lite_boost；脚本内生成或 `--baseline` 提供 npz，命令见运行命令一节）与 liteboost 1p / sp2 出图，PSNR ≥ 20 判定：

| 对比 | PSNR | 阈值 | 判定 |
|------|------|------|------|
| 参考图 vs liteboost 1p | inf | 20 | PASS |
| 参考图 vs liteboost sp2 | 22.42 | 20 | PASS |
| 参考图 vs liteboost tp2 | 22.82 | 20 | PASS |

> 注意：generator 全局同流由脚本保证——所有用例统一 rank-NPU generator（同 seed 下 NPU RNG 各卡逐位同流），npz 参考图同样用 npu:0 generator（生成代码见运行命令一节），三方同流，出图可直接对比。同流下实测（1024，A2 313T）：1p 与纯 torch 参考逐像素一致（PSNR=inf）；sp2 的差距是 SP 的 bf16 固有数值差异而非逻辑错误——SP 路径 attention 以"每卡 14 头 × 全序列"形状调用 NPU 融合内核，与 dense 路径 28 头 varlen 内核数学等价（fp32 仿真偏差 6e-08）但累加顺序不同，每个 transformer block 注入约 1 个 bf16 ULP，被 4 步 DMD 轨迹逐块放大（在逐位一致的 1p 路径注入 1 ULP 噪声可复现同量级差距）。放大倍数随分辨率/内容变化：512 约 15.8dB（不过 20 门禁），1024 约 22.4dB（过线），因此精度门禁固定在 1024 判定。

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

attention 直接以 `npu_fusion_attention(input_layout="BSND")` 调用，去掉 SDPA 路径的 BNSD 双向转置（约占 DiT 时间 8%）。mask 语义取反（SDPA True=有效，融合核 True=屏蔽），全有效时跳过 mask。1P 全量块生效（逐像素一致）。

**与 SP / cc_overlap 组合**：`bsnd` 是注意力内核的独立轴——统一内核（`attention_common.py`）的每个流水线体都接受 `backend` 参数（`attention_bnsd` SDPA 路径 / `attention_bsnd` 融合核路径），exchanger（Identity / Sync / CC）决定通信方式，backend 决定注意力求值方式，二者正交。GQA 统一为显式 `kv_idx` 头选择（`arange(Nq) // ratio`，与 Ulysses 头交换及单卡 `enable_gqa` 广播逐位一致）。1P 走 processor 替换（`attention_bsnd.py`，现为统一内核的薄封装）；SP/CC 下按 `bsnd` 选项将 SP/CC 内核绑定到对应 backend。

> **TODO（已知限制）**：`bsnd` 目前**仅可与 1P 同用**，请勿与 TP / SP 组合。
>
> - 与 `alg: TP` 互斥：TP 对 processor 类打的 `__call__` 补丁会被 `install_bsnd_processors` 覆盖为 `_bsnd_single_stream_call`，该路径不含 `_lb_mm_all_reduce` 跨卡求和、也不感知 TP 的 KV 重叠分区，输出错误（2 卡实测复现）。待后续在 TP 内核中接入 `attention_bsnd` backend（`_tp_gqa_expand` 后头数已匹配，融合核可用）后解除此限制。
> - 与 `alg: SP` 组合当前不可用：双流 SP 下 GQA 头（28:7）按 rank 切分为非整数，`npu_fusion_attention` 报 aclnn EZ1001 参数错误（2 卡实测复现）。待后续修正 SP 下的头切分后解除此限制。

---

## 许可

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
