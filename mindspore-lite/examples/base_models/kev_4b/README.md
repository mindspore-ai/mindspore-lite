# Kev-4B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 Kev-4B 导出为两张 ONNX 图，转换为 MindSpore Lite MindIR，并在 Atlas 800I A2 上推理与测速。

Kev-4B 不是文本生成模型。它把 Qwen3.5-4B-Base 的隐藏状态接到一个指针头（pointer head）上，对一个问题下的候选选项逐个打分并 softmax，因此服务侧需要的是"按行种类"拆分的两张图，而不是 prefill/decode：

1. **State 图**（`kev_state.onnx`）：读入上下文状态行 `[<|fim_prefix|>] + state`，输出后面复用的三份 cache（conv_state / recurrent_state / KV cache），不输出 logits。
2. **Branch 图**（`kev_branch.onnx`）：读入一个问句行（含各选项与 `<|fim_suffix|>`）和三份 cache，输出该行的 `last_hidden_state`；指针头在 Host 侧用 numpy 完成打分。

两个图都是 batch=1、序列维纯动态，因此**同一份 MindIR 不需要重新转换即可服务任意上下文长度**（见第 4 节「动态 shape 约束」）。

## 示例文件

| 文件 | 作用 |
|------|------|
| `export_kev_4b_onnx.py` | 把合并后的 checkpoint 导出成 state / branch 两张 ONNX（第 2 节） |
| `infer_kev_4b_mslite.py` | 端到端推理与 `--perf` 形状扫描（第 4、5 节） |
| `configs/config_state.ini` / `configs/config_branch.ini` | 两张图的转换配置（第 3 节） |

## 模型架构

Kev-4B = Qwen3.5-4B-Base + 已合并进基座的 LoRA 适配器 + 独立的指针头。

- **基座**：Qwen3.5 混合注意力骨干，32 层中 24 层线性注意力（GatedDeltaNet）+ 8 层全注意力。
- **线性注意力层**：靠 conv_state + recurrent_state 传递状态，导出为 Ascend `ChunkGatedDeltaRule` Custom 算子。
- **全注意力层**：靠 KV cache 传递状态（8 层 × k/v = 16 份）。
- **LoRA**：r=16、alpha=32，推理期已按 `scale = alpha / r = 2.0` 合并，导出的权重是合并后的普通 `nn.Linear`。
- **指针头**：`z_j = (W_k·h_opt_j + b_k) · (W_q·h_dec + b_q) / sqrt(256)`，再按 checkpoint 自带温度 `T = 2.406050072164233` 做 `softmax(z / T)`。只有 4 组参数（2 个 256×2560 矩阵 + 2 个 bias 向量），留在 Host 的 numpy 里。

| 参数 | 值 |
|------|-----|
| hidden_size | 2560 |
| num_hidden_layers | 32 |
| num_attention_heads / num_key_value_heads | 16 / 4 |
| head_dim（全注意力） | 256 |
| vocab_size | 248320（tied，未导出） |
| linear_attention 层数 | 24 |
| full_attention 层数 | 8 |
| conv_kernel_size / conv_dim | 4 / 8192 |
| 线性注意力 key/value heads | 16 / 32 |
| 线性注意力 k/v head_dim | 128 / 128 |
| LoRA rank / alpha | 16 / 32 |
| 指针头维度 | 256 |
| 温度 T | 2.406050072164233 |

各层类型由 `config.json` 中的 `layer_types` 字段定义，每 4 层中有 3 个 linear_attention 和 1 个 full_attention。

---

## 1. 环境准备

本教程涉及两个目录，下文所有命令都**在示例目录下执行**：

```bash
export KEV_ROOT=/path/to/kev-4b          # 权重、源码、outputs/、scripts/ 的根目录
cd "$KEV_ROOT/mslite/mindspore-lite/mindspore-lite/examples/base_models/kev_4b"
```

- 仓库内的文件（`configs/` 与本目录脚本）用相对路径引用；
- 合并权重、ONNX、MindIR 等产物放在 `$KEV_ROOT/outputs/` 下。

### 依赖版本

| 软件包 | 版本 |
|--------|------|
| Python | 3.12.16 |
| torch | 2.10.0+cpu |
| transformers | 5.19.0（需包含 `qwen3_5` 架构） |
| peft | 0.21.2（仅合并 LoRA 时使用） |
| onnx | 1.23.2 |
| numpy | 2.5.4 |
| CANN | 9.0.0 |
| mindspore-lite | 2.12.0 |
| 硬件 | Atlas 800I A2（280T） |

```bash
pip install torch==2.10.0 transformers==5.19.0 peft==0.21.2 onnx==1.23.2 numpy==2.5.4
pip install /path/to/mindspore_lite-2.12.0-*.whl
```

导出全程在 CPU 上完成，不需要 torch-npu。Kev 上游声明 `torch<2.9`（训练侧约束），本教程只做推理，`torch 2.10.0` 实测可用；Kev 包按 `--no-deps` 安装，见下文。

### 获取模型权重与源码

```bash
pip install modelscope

# 基座（Apache-2.0）
modelscope download --model Qwen/Qwen3.5-4B-Base --localdir ./Qwen3.5-4B-Base

# Kev-4B 适配器 + 指针头 + tokenizer
modelscope download --model jaredpalmer/kev-4b --localdir ./kev-4b

# Kev 上游源码（参考实现、encode 规则）
git clone https://github.com/jaredpalmer/kev.git
```

`kev-4b/` 需要包含 `adapter_config.json`、`adapter_model.safetensors`、`head.pt` 和 tokenizer 文件；`Qwen3.5-4B-Base/` 需要包含 `config.json`、两个 safetensors 分片和 `model.safetensors.index.json`。两个目录放到 `$KEV_ROOT/weights/` 下，与下文命令一致。

权重来自 Hugging Face 的 `Qwen/Qwen3.5-4B-Base@1001bb4d` 与 `jaredpalmer/kev-4b@957b91e7`，ModelScope 上这两个仓库按 `master` 提供，文件与上游一致；内网访问不到 `huggingface.co` 时用上面的 ModelScope 命令即可，后续导出、转换、推理**全程不需要联网**。

### 合并 LoRA 并导出指针头

`merge_lora.py` 是 Kev 适配侧的辅助脚本（不在 mslite 仓库内），位于 `$KEV_ROOT/scripts/`，它 `import kev`，先按 `--no-deps` 安装 Kev 包（上游声明 `torch<2.9`，这里只做推理，绕开版本钉死）：

```bash
pip install --no-deps -e "$KEV_ROOT/code/kev-main"

python -I "$KEV_ROOT/scripts/merge_lora.py" \
  --base "$KEV_ROOT/weights/Qwen3.5-4B-Base" \
  --adapter "$KEV_ROOT/weights/kev-4b" \
  --out "$KEV_ROOT/outputs/merged_fp32" \
  --head-out "$KEV_ROOT/outputs/ref/head.npz"
```

必须在 fp32 上合并：发布的基座权重是 bf16 存储，先转 bf16 再合并会二次舍入，与 Kev 的数值不一致。产物：

- `$KEV_ROOT/outputs/merged_fp32/`：合并后的 `Qwen3_5ForCausalLM`（safetensors + config + tokenizer），供导出与 tokenizer 使用；
- `$KEV_ROOT/outputs/ref/head.npz`：指针头的 `q_weight / q_bias / k_weight / k_bias` 与 `temperature`。

### 环境变量与 ChunkGatedDeltaRule Custom 算子

24 层线性注意力导出为 `ChunkGatedDeltaRule` Custom 节点。转换与推理进程都必须能发现该算子：

```bash
source /path/to/CANN/set_env.sh
export MSLITE_HOME_PATH=/path/to/mindspore-lite-2.12.0-linux-aarch64

# 随发布包提供的 Custom 算子 vendor（已含 ChunkGatedDeltaRule）
bash "$MSLITE_HOME_PATH/tools/custom_kernels/install.sh"
source "$ASCEND_OPP_PATH/vendors/mslite_custom_ops/bin/set_env.bash"

export PATH="$MSLITE_HOME_PATH/tools/converter/converter:$PATH"
export LD_LIBRARY_PATH="$MSLITE_HOME_PATH/tools/converter/lib:$MSLITE_HOME_PATH/runtime/lib:$LD_LIBRARY_PATH"
```

若使用的发布包不带 vendor，可用源码构建（不写入 CANN 安装目录）：

```bash
cd /path/to/mindspore-lite-source
bash mindspore-lite/tools/custom_kernels/ascend_ops/build_all_ops.sh "$PWD/output/custom_ops"
export ASCEND_CUSTOM_OPP_PATH="$(find output/custom_ops -mindepth 2 -maxdepth 2 -type d -name mslite_custom_ops -print -quit)"
```

`converter_lite` 日志里的 `FindAdapter] Can't find OpAdapter for ChunkGatedDeltaRule` 是正常输出：Custom 算子不经过 GE 的 OpAdapter，而是按 `ASCEND_CUSTOM_OPP_PATH` 透传。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
python -I export_kev_4b_onnx.py \
  --merged "$KEV_ROOT/outputs/merged_fp32" \
  --output-dir "$KEV_ROOT/outputs/onnx"
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--merged` | 合并后的 fp32 checkpoint 目录 | 必填 |
| `--output-dir` | ONNX 输出目录（state/branch 各一个子目录） | `outputs/onnx` |
| `--device` | 导出设备（`cpu` / `cuda`） | `cpu` |
| `--dummy-state-len` | state 图 trace 用长度 | `64` |
| `--dummy-branch-len` | branch 图 trace 用长度 | `16` |
| `--dummy-past-len` | branch 图 trace 用的 cache 长度 | 与 `--dummy-state-len` 相同 |
| `--enable-cgdr-custom` / `--no-enable-cgdr-custom` | 线性注意力接 `ChunkGatedDeltaRule` Custom 算子 | `True` |

`--merged` 必须是 `merge_lora.py` 写出的**本地目录**。脚本会先检查该目录下有没有 `config.json`，找不到就直接报错退出；否则 `transformers` 会把不存在的路径当成 Hub 仓库名，去访问 `huggingface.co`（典型报错 `HEAD https://huggingface.co/<路径>/resolve/main/config.json`、`[Errno 101] Network is unreachable`）。相对路径按**当前工作目录**解析，建议照上文用 `"$KEV_ROOT/..."` 的绝对路径。导出本身不需要联网。

### 产出文件

```text
$KEV_ROOT/outputs/onnx/
├── state/
│   ├── kev_state.onnx                # 图定义 ~964 KB，24 个 Custom 节点
│   └── ...（外置权重，约 7.7 GiB）
└── branch/
    ├── kev_branch.onnx               # 图定义 ~942 KB，24 个 Custom 节点
    └── ...（外置权重，约 7.9 GiB）
```

导出日志会打印每张图的 Custom 节点数：

```log
--- exporting kev_state.onnx (dummy tokens: 64)
--- exporting kev_branch.onnx (dummy tokens: 16)
    kev_state.onnx: 24 ChunkGatedDeltaRule nodes
    kev_branch.onnx: 24 ChunkGatedDeltaRule nodes
```

#### State 图 I/O

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | input_ids | `[1, seq_len]` | int64 | `<\|fim_prefix\|>` + state token |
| Input | position_ids | `[1, seq_len]` | int64 | 从 0 连续编号 |
| Input | attention_mask | `[1, 1, seq_len, seq_len]` | float16 | 加性 mask，可见处 0，不可见处 `-10000` |
| Output | present_conv_states | `[24, 1, 8192, min(seq_len, 4)]` | float16 | 24 层线性注意力的卷积状态（state 少于 4 token 时只有 `seq_len` 列） |
| Output | present_recurrent_states | `[24, 1, 32, 128, 128]` | float32 | 24 层线性注意力的循环状态 |
| Output | present_kv_cache | `[16, 1, 4, seq_len, 256]` | float16 | 8 层全注意力的 K/V（16 = 8×2） |

`present_conv_states` 只取 conv 之前的最后 4 列；state 行本身不足 4 个 token 时只能给出 `seq_len` 列，推理脚本会按 Kev（transformers 零初始化 cache）的语义在左侧补零到 4 列后再喂给 branch 图。缓存维的符号名由 tensor 值推导（ONNX 里是 `Concat...` 形式的自动符号），不能只靠 ONNX 做静态 shape 推断；MindSpore Lite 侧按动态维 + `Model.resize` 处理，实测可用。

#### Branch 图 I/O

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | input_ids | `[1, branch_len]` | int64 | `<\|fim_middle\|>` + instr + 各 `<\|box_start\|>opt<\|box_end\|>` + `<\|fim_suffix\|>` |
| Input | position_ids | `[1, branch_len]` | int64 | 从 `state_len` 续接编号 |
| Input | attention_mask | `[1, 1, branch_len, state_len + branch_len]` | float16 | 前缀全可见 + 本行因果 |
| Input | past_conv_states | `[24, 1, 8192, 4]` | float16 | state 图输出 |
| Input | past_recurrent_states | `[24, 1, 32, 128, 128]` | float32 | state 图输出 |
| Input | past_kv_cache | `[16, 1, 4, past_len, 256]` | float16 | state 图输出，`past_len = state_len` |
| Output | last_hidden_state | `[1, branch_len, 2560]` | float16 | 指针头读取的隐藏状态 |

两个图都不输出 logits/lm_head，指针头权重不进图。

### 导出注意事项

- **动态 shape 依赖 Custom 算子**：参考实现的 chunked delta rule 在 trace 时把 `seq_len // 64` 个 chunk 展开，会把 dummy 长度固化进图。接 `ChunkGatedDeltaRule` 后线性注意力保持单节点，序列维才能是真符号维。
- **mask 作为输入而不是图内构造**：`torch.arange` / `past_len` 在 trace 期是常量，换长度即错。加性 mask 由 Host 用固定 `-10000.0` 构造，也避免 `finfo.min + finfo.min` 在 fp16 下溢出成 `-inf`。
- **全注意力 softmax 用 fp32**：`F.softmax(..., dtype=torch.float32)` 后再转回 fp16，与基座实现一致。
- **batch 维固定为 1**：转换配置里 batch 是常量 `1`，只有序列维是 `-1`；这与 Kev 的单请求路径一致。

---

## 3. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

```bash
Convert=/path/to/mindspore-lite-2.12.0-linux-aarch64/tools/converter/converter/converter_lite

$Convert --fmk=ONNX \
  --modelFile="$KEV_ROOT/outputs/onnx/state/kev_state.onnx" \
  --outputFile="$KEV_ROOT/outputs/mindir/kev_state" \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/config_state.ini

$Convert --fmk=ONNX \
  --modelFile="$KEV_ROOT/outputs/onnx/branch/kev_branch.onnx" \
  --outputFile="$KEV_ROOT/outputs/mindir/kev_branch" \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/config_branch.ini
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--modelFile` | 输入 ONNX |
| `--outputFile` | 输出前缀 |
| `--optimize=ascend_oriented` | Ascend 定向优化（转换期完成图编译，运行时不再构图） |
| `--saveType=MINDIR` | 输出 MindIR |
| `--configFile` | 精度模式与 `input_format` 配置 |

### 配置文件

`configs/config_state.ini`（保持默认精度）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;position_ids:1,-1;attention_mask:1,1,-1,-1"
```

`configs/config_branch.ini`（末尾 RMSNorm 必须留在 fp32，原因见 6.1）：

```ini
[acl_init_options]
ge.exec.precision_mode=must_keep_origin_dtype

[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;position_ids:1,-1;attention_mask:1,1,-1,-1;past_conv_states:24,1,8192,4;past_recurrent_states:24,1,32,128,128;past_kv_cache:16,1,4,-1,256"
```

`must_keep_origin_dtype` 约束的是"保持导出时的 dtype"，不是"统一用 fp32"：ONNX 中声明为 fp32 的算子（本模型是 branch 收尾 RMSNorm，以及全注意力 softmax）留在 fp32，fp16 的 MatMul 与 Custom 算子仍是 fp16。如果导出源本身是 bf16，它同样按 bf16 走，不会转成 fp32。

配置文件按行用 `=` 切分并去掉两侧空白，`key = value` 与 `key=value` 都能解析，这里统一写成不带空格的 `key=value`。

`BatchMatmulToMatmul` 插件不配置：它只处理图里的 BatchMatMul（或可判定为批量形式的 MatMulFusion）节点，而本导出两张 ONNX 的 BatchMatMul 节点数都是 0（state 243 个 MatMul、branch 264 个，均为普通二维 MatMul）。A/B 实测加与不加插件，转换产物输出逐位一致、branch 时延 31.8 vs 31.4 ms（在噪声内），没有收益。

`input_shape` 里的 `-1` 都是序列维（batch 固定为 1）。`past_kv_cache` 的序列维也保持动态：branch 图的 cache 长度等于任意 state 长度，枚举档位不可行，运行时用 `Model.resize` 适配。

### 产出说明

两张图的权重都超过 2 GB，converter 会拆成图定义 + 外置权重目录：

```text
$KEV_ROOT/outputs/mindir/
├── kev_state_graph.mindir       # ~2.2 KB
├── kev_state_variables/         # 7.8 GiB
├── kev_branch_graph.mindir      # ~1.3 KB
└── kev_branch_variables/        # 8.0 GiB
```

converter 还会在**执行命令时的当前目录**留下一个 `air_weight/`（约 16 GB 的 Ascend 编译中间产物）。MindIR 的权重已经外置到 `*_variables/`，`air_weight/` 不参与运行，确认日志里是 `CONVERT RESULT SUCCESS:0` 后可以直接删除。

### 执行日志

```log
--- converting kev_state
[WARNING] ... FindAdapter] Can't find OpAdapter for ChunkGatedDeltaRule
E0000 ... message_lite.cc:411] ge.proto.ModelDef exceeded maximum protobuf size of 2GB: 8219196097
[WARNING] ... GetShapeVectorAndIdxFromCNode] Shape is empty custom_0_getitem_0
[WARNING] ... GetShapeVectorAndIdxFromCNode] Shape is empty custom_0_getitem_1
[WARNING] ... GetShapeVectorAndIdxFromCNode] Shape is empty custom_0_getitem_2
CONVERT RESULT SUCCESS:0
--- converting kev_branch
[WARNING] ... FindAdapter] Can't find OpAdapter for ChunkGatedDeltaRule
E0000 ... message_lite.cc:411] ge.proto.ModelDef exceeded maximum protobuf size of 2GB: 8423848998
[WARNING] ... GetShapeVectorAndIdxFromCNode] Shape is empty custom_0
CONVERT RESULT SUCCESS:0
```

`ModelDef exceeded maximum protobuf size of 2GB` 是无害信息：权重被外置到 `*_variables/`，MindIR 只保留图元数据。单张图约 2–3 分钟。

---

## 4. MindSpore Lite 推理

### 推理命令

不提供输入时，脚本使用内置的 demo 记录（一个问题、5 个候选选项）：

```bash
python -I infer_kev_4b_mslite.py \
  --state-mindir "$KEV_ROOT/outputs/mindir/kev_state_graph.mindir" \
  --branch-mindir "$KEV_ROOT/outputs/mindir/kev_branch_graph.mindir" \
  --head-npz "$KEV_ROOT/outputs/ref/head.npz" \
  --tokenizer "$KEV_ROOT/outputs/merged_fp32" \
  --device-id 0
```

提供真实输入时按真实输入走，支持 Kev 的 JSON/JSONL 记录：

```bash
python -I infer_kev_4b_mslite.py \
  --state-mindir "$KEV_ROOT/outputs/mindir/kev_state_graph.mindir" \
  --branch-mindir "$KEV_ROOT/outputs/mindir/kev_branch_graph.mindir" \
  --head-npz "$KEV_ROOT/outputs/ref/head.npz" \
  --tokenizer "$KEV_ROOT/outputs/merged_fp32" \
  --record-json "$KEV_ROOT/record.json" \
  --device-id 0
```

打分模式只输出选项概率，不做计时；时延统一由 `--perf` 测（见第 5 节）。`--perf` 不需要 tokenizer、指针头和记录，只要求两张 MindIR 与测试长度：

```bash
python -I infer_kev_4b_mslite.py \
  --state-mindir "$KEV_ROOT/outputs/mindir/kev_state_graph.mindir" \
  --branch-mindir "$KEV_ROOT/outputs/mindir/kev_branch_graph.mindir" \
  --perf --state-lengths 128 256 512 1024 2048 --branch-lengths 128 256 512 1024 2048
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--state-mindir` | state 图 MindIR（指向 `*_graph.mindir`） | 必填 |
| `--branch-mindir` | branch 图 MindIR（指向 `*_graph.mindir`） | 必填 |
| `--head-npz` | 指针头数组（`--perf` 时不需要） | 打分时必填 |
| `--tokenizer` | tokenizer 目录（`--perf` 时不需要） | 打分时必填 |
| `--record-json` | Kev 记录（`.json` 或 `.jsonl`）；不给则用内置 demo | — |
| `--record-index` | JSONL 中取第几条 | `0` |
| `--max-state` | state 行最大 token 数，超出截断 | `8192` |
| `--max-branch` | `state + branch` 行最大 token 数，超出报错 | `16384` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--perf` / `--iters` | 合成长度的形状扫描，只输出时延 | 关 / `5` |
| `--state-lengths` | `--perf` 的上下文长度 | `128 256 512 1024 2048` |
| `--branch-lengths` | `--perf` 的问题行 token 长度 | `128 256 512 1024 2048` |

`--state-mindir/--branch-mindir` 必须写成 `kev_state_graph.mindir` 这样的**文件路径**；`mindspore_lite.model.build_from_file` 会做 `os.path.exists` 检查，传目录或前缀会直接报 `model_path does not exist`。

MindIR 的整型输入是 int32（converter 会把 ONNX 声明的 int64 统一改写），推理脚本按图声明自动 cast，喂 int64 也可以。

### 执行日志

```log
--- Which banking intent best describes this customer message?
     0.00033  activate_my_card
     0.00069  age_limit
     0.01515  apple_pay_or_google_pay: Issue concerning apple pay or google pay
  *  0.97077  supported_cards_and_currencies
     0.01306  verify_source_of_funds
```

打分路径不做计时：同一条记录首次运行时，state pass 包含 `resize` 与形状准备，单次值没有代表性。要看时延请用第 5 节的 `--perf`，它每档预热后再取平均。

### 推理流程

从模型语义看，一次请求分三步：

1. **读上下文（state）**：把上下文行 `[<|fim_prefix|>] + state` 读完，得到后续可复用的内部状态——线性注意力层留下卷积状态和循环状态，全注意力层留下 KV cache。这一步只做一次，成本随上下文长度增长。
2. **读问题（branch）**：每个问题写成一条独立行 `<|fim_middle|> + instr + (box_start opt box_end)* + <|fim_suffix|>`，在这份状态之上续读。模型只取该行的决策位置（`<|fim_suffix|>`）和每个选项末尾（`<|box_end|>`）的 hidden。问题之间互不可见，所以一个问题一次读入；同一个问题的所有选项在同一条行里，不按选项拆分。
3. **打分**：指针头把决策位置的 hidden 与各选项位置的 hidden 做双线性点积，除以温度 `T` 后 softmax，得到选项概率。只有几个小矩阵，放在 Host 上完成。

因此一次请求的成本可以写成：

```text
T_request ≈ T_state(S) + n × T_branch(S, Lb)
```

其中 `S` 是上下文 token 数，`n` 是问题数，`Lb` 是单个问题行的 token 数。`T_state` 是固定的“建状态”成本，`T_branch` 是每个问题各自的成本；选项数只影响 `Lb`，不额外增加 pass 次数。

### 动态 shape 约束

`ascend_oriented` 会在转换期完成图编译，所以 shape 必须是转换时就确定的形式：

- **不枚举档位**：branch 图的 `past_kv_cache` 长度等于任意 state 长度，档位无法穷举；两张图的序列维都配成纯动态（`-1`），运行时每次 `Model.resize` 到本次 shape。
- **同一份 MindIR 服务任意长度**：实测 `S ∈ {128, 384, 1024, 2048}`、`(state_len, branch_len) ∈ {(384,40),(1024,300),(2048,400)}` 不重新转换即可跑通，所有输出 `isfinite`。
- **batch 固定为 1**：与 Kev 的单请求路径一致。

---

## 5. 性能数据

测试环境：Atlas 800I A2，CANN 9.0.0，MindSpore Lite 2.12.0。

下面的数字全部由本目录的 `infer_kev_4b_mslite.py --perf` 产出，可直接复现：

```bash
python -I infer_kev_4b_mslite.py \
  --state-mindir "$KEV_ROOT/outputs/mindir/kev_state_graph.mindir" \
  --branch-mindir "$KEV_ROOT/outputs/mindir/kev_branch_graph.mindir" \
  --perf --state-lengths 128 256 512 1024 2048 --branch-lengths 128 256 512 1024 2048 --iters 7
```

`--perf` 只需要两张 MindIR：它用固定种子的随机 token 构造输入，按给定的 state 长度逐档建一次 cache，再在同一份 cache 上测各档 branch 长度，因此不需要 tokenizer、指针头或真实记录。

计时口径：每档先预热 2 次（第 2 次执行真正的 `Model.resize`），再取 `--iters` 次（默认 5）的平均；包含 `Model.predict` 的 Host↔Device 拷贝，不含模型加载与首次图准备。表中为多轮重复测量的中位数，同长度重复测量波动约 1–2%。

### State 图

| 上下文长度 S | State pass（ms） |
| ---: | ---: |
| 128 | 54.5 |
| 256 | 66.3 |
| 512 | 84.2 |
| 1024 | 123.5 |
| 2048 | 200.4 |

### Branch 图

| 上下文长度 S | Lb=128 | Lb=256 | Lb=512 | Lb=1024 | Lb=2048 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 35.2 ms | 42.6 ms | 60.8 ms | 96.5 ms | 173.2 ms |
| 256 | 35.5 ms | 42.9 ms | 60.9 ms | 96.1 ms | 173.4 ms |
| 512 | 36.2 ms | 43.4 ms | 62.2 ms | 97.0 ms | 174.7 ms |
| 1024 | 37.2 ms | 44.9 ms | 63.5 ms | 98.9 ms | 179.1 ms |
| 2048 | 39.4 ms | 47.4 ms | 66.5 ms | 105.2 ms | 186.5 ms |

`--state-lengths` 和 `--branch-lengths` 是两组长度，跑的是**网格**而不是一一配对：每个 state 长度只建一次 cache，然后在这份 cache 上依次测所有 branch 长度。默认 5 个 state 点 + 5×5 个 branch 点，共 30 个计时点；整轮在几十秒内跑完（不含模型加载）。

### State 与 Branch 的关系

两张表的形状不一样，原因是两个 pass 受限于不同的资源：

- **state pass 随 S 增长，但增长慢于 S。** 固定成本是把 8 GB 权重流一遍，约 45 ms；之后每多一个上下文 token 约 0.075 ms（32 层的权重矩阵乘 + 24 层线性注意力递推 + 8 层全注意力）。S 从 128 到 2048 翻了 16 倍，时延只从 54.5 到 200.4 ms（约 3.7 倍），即固定成本被摊薄。
- **branch pass 对 S 几乎不敏感，主要随问题行长度 Lb 增长。** 每个问题只新读 Lb 个 token，固定成本同样是约 30 ms 的权重流，之后每 token 约 0.07 ms，与 state 的每 token 计算成本基本一致。所以 Lb 从 128 加到 2048（16 倍）时延从 35.2 涨到 173.2 ms（约 4.9 倍）；而 S 从 128 加到 2048（16 倍）时延仅涨 4 ms——新增的只是从 KV cache 里读上下文，约 0.002 ms/token，远小于权重流。固定的权重流部分受限于显存带宽（8 GB / 30 ms ≈ 270 GB/s），不是算力。
- **两者不能互换或合并。** state 的产物是三份 cache，branch 必须等 cache 建好后才能读；而每个问题行都要在**同一份**循环状态上续写（`past_recurrent_states` 被原地推进），所以 n 个问题不能拼成一次 branch pass——拼行会让它们互相污染。多问只能串行走 n 次 branch。

### 端到端组合

按 `T_request ≈ T_state(S) + n × T_branch(S, Lb)` 组合：

| 场景 | 计算 | 耗时 |
|---|---|---:|
| 短上下文（S=128），单问 Lb=128 | 54.5 + 35.2 | 89.7 ms |
| 常规（S=1024），3 问 Lb=256 | 123.5 + 3×44.9 | 258.2 ms |
| 长上下文（S=2048），3 问 Lb=512 | 200.4 + 3×66.5 | 399.9 ms |

这个式子说明：**一次请求只付一次 state 成本，之后每个问题各付一次 branch 成本。** 设 S=1024、Lb=256，则

| 问题数 n | 总耗时 | 平均每问 |
| ---: | ---: | ---: |
| 1 | 123.5 + 1×44.9 = 168.4 ms | 168.4 ms |
| 3 | 123.5 + 3×44.9 = 258.2 ms | 86.1 ms |
| 10 | 123.5 + 10×44.9 = 572.5 ms | 57.3 ms |

n 越大，state 的 123.5 ms 被摊得越薄，平均每问越接近单次 branch 的 44.9 ms。这也是为什么"多问共享一次 state pass"对长上下文、多问题的场景收益最大：问题数从 1 到 10，平均每问成本降到约 1/3；反过来，单问短上下文时 state 与 branch 各占约一半，省 state 没有太大空间。

需要区分的是，这是**同一请求内**的复用：不同请求的 state 不同，缓存不能跨请求共享（除非上下文前缀完全相同，那属于前缀缓存，不在本示例范围）。

### 精度

Kev-4B 是非生成式打分模型，端到端产物是每个选项的 softmax 概率与 top-1，因此以**选项概率最大偏差**和 **top-1 一致**为主判据，张量余弦用于定位误差来源。`max_abs` 不作为硬门槛：本 checkpoint 的 hidden 绝对值最大到 282，个别极值会把 max_abs 放大；这与精度 skill 对 LLM/生成类模型"以端到端效果为准"的口径一致。

同一份 fixture（state 128 token、2 选项问题），与 Kev 自身 fp32 CPU 路径对比：

| 级 | 对比 | conv | recurrent | kv | hidden | 选项概率最大偏差 | top-1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| L1 | torch fp16（导出器自身） | 0.9999997 | 0.9999995 | 0.9999995 | 0.9999990 | 4.10e-04 | 一致 |
| L2 | ONNX Runtime CPU（无 Custom 参考图） | 0.9998078 | 0.9998256 | 0.9997898 | 0.9994491 | 9.79e-03 | 一致 |
| L3 | MindIR @A2（部署图） | 0.9999937 | 0.9999955 | 0.9999921 | **0.9999968** | **7.58e-04** | 一致 |

表中 conv / recurrent / kv / hidden 均为余弦相似度。判定容差：L1 cos ≥ 0.9999、Δp ≤ 2e-3；L2/L3 cos ≥ 0.999、Δp ≤ 2e-2。三级 top-1 全部一致。

L2 的 hidden 余弦低于 L3 属正常：ONNX Runtime 的 fp16 累加顺序与 torch 差异更大，而 Ascend 部署图在 `must_keep_origin_dtype` 下保留了收尾 RMSNorm 的 fp32 区间（见第 6 节问题 1）。

### 算子级分布（msprof）

branch 图在 20 token 分支上的 op_summary 汇总（kernel 时间，总计约 30.7 ms）：

| OP Type | 次数 | 累计耗时（ms） |
|---|---:|---:|
| MatMulV2 | 248 | 14.18 |
| ChunkGatedDeltaRule | 24 | 3.35 |
| EltwiseBroadcastFusionOp | 255 | 3.17 |
| Transpose | 104 | 2.00 |
| StridedSliceV3 | 116 | 1.55 |
| ReduceMean | 105 | 1.39 |
| TransData | 48 | 0.85 |
| Conv2D | 24 | 0.76 |

24 个 `ChunkGatedDeltaRule` 就是 24 层线性注意力，确认 Custom 算子在 A2 上真实命中内核，而不是回退到 CPU/展开实现。

---

## 6. 常见问题

1. 现象：MindIR 图能跑通，三份 cache 的余弦都在 0.99999，但 branch 的 hidden 与 fp32 参考差得很远，选项概率随之偏移
   - 原因：末尾 RMSNorm 在 ONNX 里是 fp32 区间（Cast → Pow → ReduceMean → rsqrt → Mul），GE 默认 `allow_fp32_to_fp16` 把它降到 fp16。该层输入是最后一层的 hidden，本 checkpoint 上 `|h|` 最大到 282，`h² = 79524` 超过 fp16 上限 65504，对应 token 的方差变成 `inf`。
   - 解决方案：给 branch 图加 `ge.exec.precision_mode=must_keep_origin_dtype`（见 `configs/config_branch.ini`）。它只保留"导出时就是 fp32"的算子，MatMulV2 与 `ChunkGatedDeltaRule` 仍是 fp16，所以不是 `force_fp32` 的整图降速；同一分支形状上代价约 +6.5%（34.91 → 37.18 ms）。state 图不需要。
   - 定位方法：把这段 RMSNorm 单独导成小图在 device 上跑，与 torch 的 fp32 数学、fp16 数学两个参考比较，看 device 输出贴近哪一种，即可判定是精度模式而不是算子实现。

2. 现象：`RuntimeError: build_from_file failed, model_path does not exist!`
   - 原因：`mindspore_lite.model.build_from_file` 对路径做 `os.path.exists` 检查。
   - 解决方案：传 `.../kev_state_graph.mindir` 文件，而不是目录或前缀。

3. 现象：转换日志出现 `ge.proto.ModelDef exceeded maximum protobuf size of 2GB`
   - 原因：`ascend_oriented` 要把 8 GB 量级权重过一遍 GE 序列化。
   - 解决方案：无视即可，权重会外置到 `*_variables/`；以最终 `CONVERT RESULT SUCCESS:0` 为准。

4. 现象：不知道该不该用 `--optimize=none`
   - 原因：`--optimize=none` 只是格式转换，不做 Ascend 定向编译。
   - 解决方案：本模型走 `--optimize=ascend_oriented`（转换期完成图编译，由 ACL 执行），转换完成后运行时不再构图；不要用 `--optimize=none`。

5. 现象：看到别人转换前 `export KEEP_ORIGIN_DTYPE=1`，不确定自己要不要设
   - 原因：converter 默认会把 int64/fp64 的 Parameter、Cast、Shape 输出改写成 int32/fp32；`KEEP_ORIGIN_DTYPE=1` 是转换器读取的**环境变量**（没有对应的 config 选项），置 1 后跳过这些改写。
   - 解决方案：本模型不需要设。实测开与关都能 `CONVERT RESULT SUCCESS:0`，两张 MindIR 的 `input_ids`/`position_ids` 都是 int32，三份 cache 的余弦差在 1e-5 量级（转换本身的舍入噪声）。它和 `must_keep_origin_dtype` 是两回事：后者是 GE 的编译精度选项，只决定 fp32 算子不被降成 fp16，与整数类型无关。

6. 现象：想在 ONNX Runtime 里跑导出的 ONNX，报找不到 `ChunkGatedDeltaRule`
   - 原因：部署图含 `Custom` 节点，ORT 没有该算子实现。
   - 解决方案：部署图只交给 MindSpore Lite 执行，不要用 ONNX Runtime 跑它。

7. 现象：转换时出现 `FindAdapter] Can't find OpAdapter for ChunkGatedDeltaRule`
   - 原因：这是 Custom 算子的正常日志（它不经过 GE 的 OpAdapter）；只有配合后续 Traceback 或 `CONVERT RESULT FAILED` 才是 vendor 缺失。
   - 解决方案：确认 `ASCEND_CUSTOM_OPP_PATH` 指向 `mslite_custom_ops` 中包含 `chunk_gated_delta_rule` 内核实现的目录（位于 `op_impl/ai_core/tbe/kernel/` 下），只要最终 `CONVERT RESULT SUCCESS:0` 且推理时 msprof 里能看到 `ChunkGatedDeltaRule` 任务即可。

8. 现象：`aclmdlLoadFromMem failed`
   - 原因：目标 device 被其他进程占用（共享机器上常见）。
   - 解决方案：换 `--device-id`，或先 `npu-smi info` 确认空闲卡。

9. 现象：安装 Kev 包时报 `torch<2.9` 版本冲突
   - 原因：上游声明的是训练侧约束。
   - 解决方案：`pip install --no-deps -e .` 安装；本教程只做 CPU 导出与推理，torch 2.10.0 实测可用。

10. 现象：照着 ONNX 图自己构造 mask 时出现 `-inf` 或 NaN
    - 原因：用 `finfo.min` 构造加性 mask，两个 `finfo.min` 相加在 fp16 下溢出。
    - 解决方案：固定用 `-10000.0`（与推理脚本 `additive_mask` 一致）。

---

## 7. 参考资源

- 上游模型仓库：[jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- Kev-4B：[ModelScope](https://modelscope.cn/models/jaredpalmer/kev-4b) ｜ [Hugging Face](https://huggingface.co/jaredpalmer/kev-4b)
- Qwen3.5-4B-Base：[ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-4B-Base) ｜ [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [Transformers 文档](https://huggingface.co/docs/transformers)

---

## 8. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求（Apache-2.0）。
- 上游 Kev 代码与权重的许可证以其仓库为准（Apache-2.0），基座 Qwen3.5-4B-Base 为 Apache-2.0。
