# .msl Model Format Specification v1

> 运行时侧格式契约：`src/manifest/msl_format.h`（C++ 常量/键枚举/static_assert）。
> 打包侧实现：`export/msl_pack.py`（Python，无外部工具依赖）。
> 字节级锁定：`tests/data/golden_v1.msl` + `tests/ut/test_msl_golden.cpp`（C++）+
> `tests/py/test_msl_golden.py`（Python）——四侧必须同步，改布局即红。

## Overview

`.msl` 是 lite_llm 的单文件自包含模型格式：Python 导出工具链（`export/`）产出，
C++ 运行时（`src/manifest/`）消费。v1 的核心变化：**元数据以 KV 键值内嵌在文件里，
没有外部 `manifest.json`**——部署交付物就是这一个文件。

当前交付路径为 **NPU（Kirin NNRT）**：单个 `.omc` 图（prefill/decode 双档位，
`npu.chunk_size` / 1），KV 在设备上就地更新。

## File Layout（小端，全部多字节字段为 u32/u64 LE）

```text

+------------------+-------------------------------------------------+
| MslHeader (24B)  | magic ".MSL" | version u32 | kv_count u32       |
|                  | resource_count u32 | alignment u32 | reserved u32|
+------------------+-------------------------------------------------+
| KV 区            | kv_count 条，每条：                              |
|                  |   key_len u32 | key(UTF-8) | type u32           |
|                  |   value_len u32 | value(按 type 编码)           |
+------------------+-------------------------------------------------+
| 资源表           | resource_count 条 × 88B：                        |
|                  |   name[64] NUL-padded | offset u64 | size u64   |
|                  |   access u32 | reserved u32                     |
+------------------+-------------------------------------------------+
| 数据区           | 各资源 payload，offset 对齐到 alignment          |
|                  | （v1 = 4096），payload 按 size 紧排              |
+------------------+-------------------------------------------------+

```

Header 字段：

| 字段 | 值 | 说明 |
|------|----|------|
| magic | `2E 4D 53 4C`（".MSL"） | 直接按字节比较，无大小端歧义 |
| version | `1` | 未知 version 运行时直接拒绝 |
| kv_count | N | KV 条数 |
| resource_count | M | 资源条数 |
| alignment | `4096` | payload offset 对齐粒度（运行时校验 `offset % alignment == 0`） |
| reserved | `0` | 保留 |

## KV Value Types（v1 封闭集合）

| type | 名称 | 编码 |
|------|------|------|
| 0 | bool | 1 字节（0/1） |
| 1 | uint32 | 4B LE |
| 2 | uint64 | 8B LE |
| 3 | float32 | 4B IEEE-754 |
| 4 | string | UTF-8 原始字节 |
| 5 | string[] | `count u32` + count × (`len u32` + bytes) |

**扩展语义**：

- **未知 type** → 运行时**拒绝**（布局契约被破坏，必须升 version 才能加类型）。
- **未知 key** → 运行时**跳过**（容忍，不报错）——加键不升 version，
  新键只被理解它的新版本运行时消费，旧运行时照常加载。

## KV Keys（v1，均有运行时消费者）

`model.*`：

| 键 | 类型 | 说明 |
|----|------|------|
| `model.name` | string | 模型名 |
| `model.version` | string | 模型版本 |
| `model.format_version` | string | 导出格式版本（`"1.0"`） |
| `model.dtype` | string | `"fp16"` / `"fp32"` / `"int8"` 等（`ParseDTypeName`） |

`arch.*`（NNRTBackend 消费）：

| 键 | 类型 |
|----|------|
| `arch.num_layers` / `hidden_size` / `intermediate_size` / `num_heads` / `num_kv_heads` / `head_dim` / `vocab_size` / `max_position_embeddings` / `tie_word_embeddings` | uint32 |
| `arch.rope_theta` / `norm_eps` | float32 |

`litert.*`（图定位）：

| 键 | 类型 | 说明 |
|----|------|------|
| `litert.prefill.path` | string | prefill 图资源名（NPU 单图即 `.omc`） |
| `litert.prefill.seq_len` | uint32 | prefill 档位 seq 长度（可选） |
| `litert.decode.path` | string | decode 图资源名（可选） |
| `litert.decode.dynamic_past_len` | bool | |
| `litert.decode.past_len` / `max_past_len` | uint32 | |
| `litert.decode_variants` | string | JSON：`[{"past_len":..,"path":..}]`（复用现有 JSON 解析器） |

`npu.*`（导出时固化，与 `.omc` 档位 shape 绑定）：

| 键 | 类型 | 说明 |
|----|------|------|
| `npu.max_length` | uint32 | 必须 > 0 且为 chunk_size 整数倍 |
| `npu.chunk_size` | uint32 | prefill 档位 seq 长度（decode 恒为 1） |
| `npu.embedding_quant` | bool | 是否 W4A8/W4A16 int4 打包 embedding |
| `npu.scale_gp_size` | uint32 | 量化分组大小（默认 32） |
| `npu.q4_0_weight_layout` | string | W4A16 必需为 `q4_0_nzf_compact_phase4`；旧 padded `q4_0_nzf_phase4`、`q4_0_nzf`、planar 或缺失标记的量化包须重新导出 |

`asset.*`（值 = 资源表中的资源名）：

| 键 | 说明 |
|----|------|
| `asset.tokenizer` | 词表资源（`vocab/vocab.bin`） |
| `asset.embedding` | 量化 embedding（`assets/embedding_quant.bin`） |
| `asset.embedding_fp16` | 备用 fp16 权重（可选） |
| `asset.rope_sin` / `asset.rope_cos` | RoPE 表 |
| `asset.attention_mask` | mask 表 |

`gen.*`：

| 键 | 类型 | 说明 |
|----|------|------|
| `gen.eos_token_id` | uint32 | eos token（NNRTBackend 的 `eos_id`） |

## Resource Table

- `name`：**完整相对路径**（如 `npu_offline/x.omc`、`vocab/vocab.bin`），
  64 字节 NUL 填充。禁止 `\`、`..` 穿越、控制字符；打包端与运行时统一按
  完整路径匹配（v1 修复了旧 mspacker「打包存 basename、运行时查全路径」的不一致）。
- `offset`：payload 的绝对文件偏移，必须 `offset % alignment == 0`。
- `size`：payload 字节数。
- `access`：`0 = mmap`（`.omc`、embedding：运行时整文件 mmap 后返回内部指针，
  `.omc` 经 NNRT `Compilation_ConstructWithOfflineModelBuffer` 零拷贝喂入）；
  `1 = read`（vocab/rope/attention_mask：拷贝读出）。

## Runtime Loading Flow

1. `MslPackageReader::Open`：mmap 整个文件，校验 magic/version/alignment，
   解析 KV 区（未知键跳过、未知类型拒绝）+ 资源表（对齐/范围/access 校验）。
2. `BuildModelManifestFromKv`：KV → `ModelManifest`（arch/npu/litert/asset/gen 各段；
   `gen.eos_token_id` → `generation.stop_token_ids[0]`）。
3. `LoadModelResourcesFromSingleFile`：按 `asset.*`/`litert.*` 键值在资源表中
   `Lookup` 定位资源（完整路径匹配）。
4. `NPUBackend::Init`：NNRT 单图 + 设备 KV；`.omc` 经 mmap 指针喂 buffer API。

## Binary Format Reference（资源内容格式）

- **embedding_quant.bin**（W4A16 `q4_0_nzf_compact_phase4`）：与 decoder 权重使用同一格式。
  逻辑权重为 `Q[N,K]`（embedding 中 `N = vocab_size`、`K = hidden_size`），
  N/K 必须为正数，分别为 16/32 的整数倍。cell 按 N64、K1024 划分，
  但尾 cell 只保存有效元素，不补齐 N 或 K。
  blob 先存全部 packed int4 cell，再存全部 fp16 scale cell，无 zero point；
  总容量严格为 `N * (K/32) * 18` 字节，其中 packed 区为 `N*K/2` 字节。
    - cell 按 `(nt,ks)` 排序；令 `n0=nt*64`、`k0=ks*1024`、
      `nc=min(64,N-n0)`、`kc=min(1024,K-k0)`。
      packed cell 起始字节为 `n0*K/2 + nc*k0/2`，
      scale cell 起始字节为 `N*K/2 + n0*K/16 + nc*k0/16`。
    - 每个 packed cell 含 `[kc/16,nc/16]` 个 16×16 分形，每个分形 128B。
      int4 为 signed 二补码（`-8..7`），相邻 K 元素占 low/high nibble。
      phase4 排列仍为 `dst[4*i+p] = src[32*p+i]`，`i=0..31`、`p=0..3`；
      compact 仅移除尾 cell padding，不改变分形内的编码。
    - 对逻辑坐标 `(n,k)`，取其所在 cell，令
      `fractal=((k-k0)//16)*(nc//16)+(n-n0)//16`、
      `j=((n%16)*16+k%16)//2`。
      packed 字节位于 `packed_cell_offset + fractal*128 + 4*(j%32) + j//32`；
      `k%2 == 0` 取 low nibble，否则取 high nibble，按 signed int4 解码。
    - scale 在每个 cell 内按 `[nc,kc/32]` 存储，为 little-endian fp16。
      `(n,k)` 的 scale 字节位于
      `scale_cell_offset + 2*((n-n0)*(kc/32)+(k-k0)//32)`。
      反量化值为 signed int4 × scale，CPU 按运行时 FP16 规则舍入。
    - 标准 GGUF Q4_0 block 为 2B scale + 16B split-half nibble。
      导入时逐位保留 scale（含符号及特殊位模式）和量化码值，仅将 unsigned nibble
      以 XOR 8 转为 signed wire 编码并重排；不重新量化。
    - CPU embedding 按需直接解码供 NPU 使用的同一 ION blob，不展开全量 fp16 embedding。
      字节数、维度合法性及容量计算溢出由运行时校验。
    - W4A16 必须携带 `npu.q4_0_weight_layout = "q4_0_nzf_compact_phase4"`。
      旧 padded `q4_0_nzf_phase4`、相邻字节 `q4_0_nzf`、planar 和缺失当前标记的包
      必须用匹配算子与当前导出器重新导出、编译整包。旧字段 `npu.weight_layout`
      不作为别名或回退；仅当前键的值参与布局校验，未知键仍按 KV 协议忽略。
      对齐 shape 的新旧容量可能相同，不能靠字节数识别布局，也不能只改标记。
      FP16 不受影响；本契约不覆盖 W4A8。

- **SubGraph_0.weight**：Qwen2.5 使用已有 external decoder weights 路径；
  “external” 指相对编译图的权重资源，仍打包在同一个 `.msl` 内，用户无需手工维护双文件。
- **rope_cos.bin / rope_sin.bin**：`[max_length, head_dim]` fp16，`max_length = npu.max_length`。
- **attention_mask.bin**：`[max_length, max_length]` fp16 上三角 mask。
- **vocab.bin**：tokenizer 词表 + 内嵌 chat template（受限 IR 指令流，v1）
    - 嵌 stop/suppress token 策略段（`LoadSpecialTokenPolicy`）。

## 与 v0（KCAP）的差异备忘

| | v0（已删除） | v1 |
|--|--|--|
| magic | KCAP（16B header） | .MSL（24B header） |
| 元数据 | 外部 `manifest.json`（目录逻辑视图） | KV 内嵌，无外部文件 |
| entry 名 | basename（打包端）vs 全路径（运行时）——不一致 | 统一完整相对路径 |
| 打包工具 | C++ `mspacker_tool` | Python `msl_pack.py` |
| 扩展性 | 改字段 = 改布局 | KV 加键不升版本 |

## Future Evolution（未实现，勿依赖）

- 动态 shape / 多 batch：当前 `max_batch_size=1`。
- 类型系统扩展（int32[]/int64[] 等）：需要升 version（未知 type 拒绝语义保证
  旧运行时会拒绝新类型文件）。
