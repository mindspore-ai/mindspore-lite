# lite_llm 导出工具（export/）

一键把 Qwen2.5-0.5B 的 HF bfloat16 / GGUF Q4_0 模型导出为单文件 `.msl`
（v1 格式：KV 元数据 + 资源表，无外部 manifest.json）。打包由内置的
`msl_pack.py`（Python）完成，不依赖外部 C++ 工具。设计见
`../DESIGN.md` 与本文档。

## 目录结构

```text

export/
├── mslite_llm_export.py      # 唯一对外入口（唯一带 CLI 的导出脚本）
├── models/
│   ├── _base/                # 跨模型公共 NNRT wrapper 基类
│   │   └── nnrt_decoder_wrapper.py  # NnrtDecoderWrapper（forward 循环）+ NnrtOpSet（算子集策略）
│   ├── qwen2_5/              # 模型专属（纯内部 API，无 CLI）
│   │   ├── qwen2_5_wrapper.py     # Qwen2NnrtWrapper（继承基类，无 per-head norm）
│   │   ├── qwen2_5_exporter.py    # Qwen2Onnx 类 + export_qwen2_5 编排
│   │   └── qwen2_5_gguf_loader.py # GGUF Q4_0 权重注入（qwen2 层名映射）
│   ├── qwen3/                # MiniMind-3（Qwen3 dense, head_dim=96），结构同 qwen2_5/
│   │   ├── qwen3_wrapper.py      # Qwen3NnrtWrapper（per-head q_norm/k_norm）
│   │   ├── qwen3_exporter.py     # Qwen3Onnx 类 + export_qwen3 编排
│   │   └── qwen3_gguf_loader.py  # MiniMind-3 GGUF Q4_0 权重注入
│   └── minicpm/              # MiniCPM-2B（LLaMA 派生, scale_emb=12），结构同 qwen2_5/
│       ├── minicpm_wrapper.py    # MiniCpmNnrtWrapper（scale_emb 输入缩放）
│       ├── minicpm_exporter.py   # MiniCpmOnnx 类 + export_minicpm 编排
│       └── minicpm_gguf_loader.py  # GGUF Q4_0 权重注入（llama.cpp 标准命名，复用族内 map）
└── utils/                    # 跨模型通用（均为内部 API，无 CLI）
    ├── onnx_postprocess.py   # MsAddRmsNorm 融合 / NNRT 契约校验 / 共享 initializer 复制
    ├── gguf_mapping.py       # Q4_0 g32 重排等模型无关 GGUF 张量映射
    ├── export_quant.py       # 量化导出：配置类 + 位打包内核 + 图变换
    ├── export_tokenizer.py   # tokenizer → vocab.bin + policy（含 chat template IR 编译器）
    ├── omc_compiler.py       # DDK omg → .omc
    └── msl_pack.py           # .msl v1 单文件容器 pack/unpack + 资源组装（内置工具）

```

对外接口只有一个：`mslite_llm_export.py`（一键导出）。
`models/`、`utils/` 下其余模块均为内部 API
（不承诺兼容，签名见各模块 docstring）。新增模型时在
`models/<model>/` 下建目录（文件以 `<model>_` 前缀命名），复用 `utils/` 通用件。

## 环境

```bash

pip install -r ../requirements.txt
# transformers >= 4.57：GGUF 骨架/tokenizer 加载依赖 4.57 的 gguf_file= 支持。
# forward 逻辑在独立 NNRT wrapper（models/_base + 子类）中，仅触碰稳定公开
# transformers API，不再 monkey-patch transformers 内部，故无 4.57.x 单 minor 锁定。

```text

omg 编译需要 Huawei DDK（AscendC）环境，且 DDK 里必须已注册 Ms\* 自定义算子，
见下方「附录 A：DDK 自定义算子安装」。

## 已支持模型与加载约束

| 模型 | model_type | 加载方式 |
|---|---|---|
| Qwen2.5-0.5B | `qwen2` | HF 目录 / GGUF（内置实现） |
| MiniMind-3（Qwen3 dense） | `qwen3` | HF 目录 / GGUF（内置实现） |
| MiniCPM-2B | `minicpm` | HF 目录 / GGUF，均需 `trust_remote_code=True`（模型仓库自带 modeling） |

注意：transformers 已移除内置 `minicpm`（4.57 与 5.x 均无），MiniCPM 的 HF/GGUF
加载都依赖模型仓库自带的 `modeling_minicpm.py`（仅 4.5x 系列验证过）；导出侧
wrapper（`models/minicpm/`）本身不依赖该 custom code，仅加载层受影响。

## 一键导出（issue #416）

`mslite_llm_export.py` 是唯一入口，自动编排骨架导出 → GGUF 权重注入（GGUF 路径）→
omg 编译 → tokenizer 导出 → msl_pack 打包，直接产出单文件 `.msl`：

```bash

# GGUF Q4_0（可跑 NPU，<200MB）
python mslite_llm_export.py --target kirin9020 \
    --model ./Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
    --output ./Qwen2.5-0.5B-Instruct-Q4-0.msl

# HF bfloat16（>1GB，仅支持导出，不满足 NPU <200MB 推理限制）
python mslite_llm_export.py --target kirin9020 \
    --model ./Qwen2.5-0.5B-Instruct \
    --output ./Qwen2.5-0.5B-Instruct-fp16.msl

```text

```

必需：
  --model MODEL     输入：GGUF 文件(.gguf) 或 HF 模型目录（自动识别）
  --output OUTPUT   输出单文件 .msl 路径
可选（均带默认值）：
  --target TARGET   部署目标芯片（默认 kirin9020），当前仅支持 kirin9020
  --max-length N    最大序列长度（默认 1024）
  --chunk-size N    prefill chunk 大小（默认 64）
  --verbose         详细日志

```text

前置：`pip install -r ../requirements.txt`、DDK 已 source（omg 可用）。
打包由内置 `msl_pack.py` 完成（无外部工具依赖）。中间产物落在输出同目录的
临时目录，打包成功后自动清理。

---

## 附录 A：DDK 自定义算子安装（custom_ops 算子仓）

omg 编译需要 DDK 里注册 Ms\* 算子。算子源码统一由 **custom_ops 算子库**提供，
已 vendor 到本仓库（`../custom_ops`），取代旧 douyin
自包含 `ddk_append_files/` 的手工拷贝流程。

### A1. 构建并安装算子到 DDK

```bash

# 1. 进入算子库目录
cd ../custom_ops

# 2. source DDK 环境（与 omg 编译同一套 DDK）
source $DDK/tools/tools_ascendc/set_ascendc_env.sh

# 3. 一键构建 + 安装 Ms* 算子到 DDK（kirin9020 平台）
./build.py \
    --ops MsRmsNorm MsAddRmsNorm MsAddSoftmax MsGroupMatmul \
         MsQuant4N0Group32 MsRotaryPosEmb MsScatterND MsFloatCastInt \
    --install "$DDK"

```text

W4A16 量化链路需要 7 个算子：`MsQuant4N0Group32` / `MsRotaryPosEmb` /
`MsScatterND` / `MsGroupMatmul` / `MsAddSoftmax` / `MsAddRmsNorm` / `MsRmsNorm`；
`MsFloatCastInt` 为 W4A8（`--embedding-quant W4A8`）额外所需。算子能力矩阵见
算子仓 `README.md`。

### A2. 验证

```bash

source $DDK/tools/tools_ascendc/set_ascendc_env.sh
export PATH=$DDK/tools/tools_omg:$PATH
# 跑一键导出（omg 步骤），日志里应无 undefined symbol / fatal error，且 exit=0

```

首次编译约 10–30 分钟（约 360 个自定义算子逐个 tiling/编译），之后 kernel cache
落在 `~/atc_data/kernel_cache/kirin9020/`，二次编译秒级复用。
