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

## Binary Format Reference（资源内容格式，与 v0 一致）

- **embedding_quant.bin**（W4A16 Q4N0Group32）：容量 =
  `vocab_size * (hidden_size/2 + hidden_size/32 * 2)` 字节；每组 32 元素打包为
  16B int4 数据 + 2B scale + 2B zero point。
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
