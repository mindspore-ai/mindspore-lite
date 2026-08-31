# Qwen-Image-Edit

Qwen-Image-Edit 模型的 NPU 多卡并行适配器，基于 `lite_boost.BoostManager` 实现图像编辑流水线的一键多卡并行推理：DiT 采用 Context Parallel（USP 序列并行），VAE 采用 Data Parallel（分块 encode/decode）。

---

## 部署环境

<!-- 环境版本信息由维护者补充 -->

| 组件 | 版本要求 |
|------|----------|
| PyTorch | 2.9.0 |
| torch_npu | 2.9.0 |
| torchvision | 0.24.0 |
| pyaml | 26.2.1 |
| diffusers | 0.37.1 |
| transformers | 4.52.4 |
| CANN | 9.2.0 |
| lite_boost | 0.2.0 |

> 硬件要求：华为昇腾 NPU，已安装 HCCL 通信库。

---

## 使用教程

### 1. 快速开始

以下代码简要介绍了如何使用 `lite_boost` 进行 Qwen-Image-Edit 的多卡并行推理（完整可运行脚本见 [test/models/qwen_image_edit/edit_usp_2card.py](../../../test/models/qwen_image_edit/edit_usp_2card.py)）：

```python
from lite_boost import BoostManager
from lite_boost.parallel import initialize_usp
from diffusers import QwenImageEditPipeline

# 1. 初始化 HCCL 分布式环境（读 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT）
initialize_usp()

# 2. 加载流水线
pipe = QwenImageEditPipeline.from_pretrained("qwen-image-edit", torch_dtype=torch.bfloat16)

# 3. 一键替换为并行版本（原地修改，config 指向 YAML 配置文件）
boost_manager = BoostManager()
pipe = boost_manager(pipe, config="qwen_image_edit.yaml")

# 4. 正常推理
output = pipe(image=..., prompt=...)
```

`BoostManager` 会自动完成以下替换：

```text
boost_manager(pipe, config="qwen_image_edit.yaml")
├── 替换 F.scaled_dot_product_attention → eager 实现（NPU 融合 SDPA 算子不支持）
├── 替换 block.attn.processor           → USPQwenDoubleStreamAttnProcessor（联合注意力 + all_to_all）
├── 替换 transformer.forward            → usp_dit_forward（图像隐变量序列分片 / 聚合）
└── 替换 pipe.vae                       → AutoencoderKLQwenImage（DP 分块 encode/decode）
```

### 2. YAML 配置说明

`config` 指向的 YAML 文件用于按模块选择优化方式与并行度（完整示例见 [qwen_image_edit.yaml](qwen_image_edit.yaml)）：

```yaml
Parallel:
  dit:                # DiT 上下文并行
    alg: CP           # current support [CP]
    world_size: 2
  vae:                # VAE 数据并行
    alg: DP           # current support [DP]
    world_size: 2
```

| 配置项 | 合法值 / 约束 | 缺省值 |
|--------|---------------|--------|
| `Parallel.dit.alg` | 仅支持 `CP`（上下文并行，即 USP 序列并行） | `CP` |
| `Parallel.dit.world_size` | 必须等于分布式 world_size（CP 通信跑在全局进程组） | 分布式 world_size |
| `Parallel.vae.alg` | 仅支持 `DP`（分块 encode/decode 数据并行） | `DP` |
| `Parallel.vae.world_size` | 必须等于 dit 的 world_size | dit 的 world_size |

- **缺省即最优**：不传 `config`、或配置文件缺某段/缺某键时，均采用性能最优默认 —— DiT CP @ 分布式 world_size + 并行 VAE 开启；
- **非法配置快速失败**：alg 不在白名单、world_size 不一致时直接抛 `ValueError`，报错信息指向应修改的配置项；
- **注意**：YAML 文件禁止使用 tab 缩进（PyYAML 会报 `ScannerError`），请使用空格缩进。

### 3. 运行命令

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503

torchrun --nproc_per_node=2 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    edit_usp_2card.py
```

### 4. 约束条件

- `num_attention_heads % world_size == 0`（head 数需能被并行卡数整除）
- 图像隐变量序列长度自动 pad 到 `world_size` 的倍数，attention 输出后去除 padding
- 配置的 `world_size` 必须与实际分布式 world_size 一致（校验失败直接报错）
- 推理前需将文本编码器的 attention 实现设为 eager：`pipe.text_encoder.config._attn_implementation = "eager"`（参考测试脚本）

### 5. 性能数据

<!-- 性能数据由维护者补充 -->

测试硬件： Ascend A2， 输入图片尺寸: 1080x1441

| 指标 | 优化前（单卡） | 优化后（2 卡） | 优化后（4 卡） |
|------|---------------|---------------| ---------------|
| 总耗时 (s) | 267s | 76s | 44s |
| DiT 单步性能 (s) | 6.12s | 1.9s | 1.08s |

---

## 优化特性

### Context Parallel (CP)

采用 Ulysses Sequence Parallel 将 DiT 的联合注意力（[文本 token | 图像隐变量 token]）切分到多卡：

- **文本流**：每卡持有完整文本序列，仅做本地 head 切分（无通信，attention 后通过 `all_gather` 恢复全部 head）；
- **图像流**：隐变量序列按 rank 切分，经 `all_to_all` 交换为完整序列 × H/P head 的布局参与联合 attention，输出后再反向 `all_to_all` 恢复本地序列分片；
- **RoPE**：图像 token 使用本地切片的频率表（含 padding 对齐），文本 token 使用全量频率表；
- **SDPA**：NPU 融合 SDPA 算子不支持该场景，全局替换为 eager matmul 实现作为兜底。

### Data Parallel (DP)

VAE 替换为 `AutoencoderKLQwenImage`（支持分布式多卡 tiled encode/decode），开启 `enable_tiling()` 后沿空间维度分块并行处理；替换后自动将 fp32 参数恢复为目标 dtype（规避 torch_npu 部分低比特参数 `.to("npu")` 时回退 fp32 的问题）。

---

## 许可

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
