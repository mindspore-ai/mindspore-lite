---
name: "nnrt-torch-custom-export"
description: "将 HuggingFace/GGUF 大模型导出为带 Ms* torch_custom 算子的 NNRT ONNX（麒麟 NPU）。当需要新增/迁移 LLM 导出、替换融合算子集、或排查导出图节点名/量化/GGUF 注入问题时调用。"
---

# NNRT torch_custom 算子 ONNX 导出

适用目录：`mindspore-lite/lite_llm/export/`。目标：把 HF 权重或 GGUF 权重的解码器网络导出为符合 NNRT（Kirin NPU）执行器 I/O 契约、内含 `Ms*` 自定义算子的 ONNX。

## 1. 架构总览（两条正交变化轴）

| 变化轴 | 承载者 | 位置 |
|---|---|---|
| 模型架构（Qwen2/Qwen3/MiniCPM/后续家族…） | `NnrtDecoderWrapper` 子类 + `attn_module` 类属性选 attention 适配器 | `export/models/_base/nnrt_decoder_wrapper.py` |
| 融合算子集（哪些 torch_custom 核实现每个原语） | `NnrtOpSet` 子类，经 `wrapper(op_set=...)` 注入 | 同上 |

基类提供：`NnrtOpSet`（rope/kv_scatter/qk_matmul/pv_matmul/mask_softmax/rmsnorm 六原语）、`NnrtRmsNorm`、`NnrtAttention`（`apply_qk_norm` hook，默认 no-op）、`NnrtDecoderLayer`、`NnrtDecoderCore`、`NnrtDecoderWrapper`（`attn_module = NnrtAttention` 类属性，子类换适配器只需一行，如 `Qwen3NnrtWrapper.attn_module = Qwen3Attention`）。

**铁律 1 — 禁止 monkey-patch transformers**。旧 `*_patch.py` 方案已删除：它依赖 `modeling_qwen2.*` 内部 forward 签名，锁死 transformers 版本。wrapper 只在加载层经公开 API（`AutoConfig` / `AutoModelForCausalLM.from_pretrained` / `gguf_file=`）使用 transformers；forward 只读 HF 子模块的普通属性（`weight`/`variance_epsilon`/`q_proj`…），绝不调用其 `forward`。

**铁律 2 — wrapper 必须是镜像 `*ForCausalLM` 层级的真实 `nn.Module` 树**（`wrapper.model.layers.{i}.self_attn.q_proj`）。子模块经 `__call__` 调用才能产生 scope 链；若用普通方法/flat 持有，trace 出的节点名会退化为 `/q_proj/MatMul`，而 GGUF loader 按 `/model/layers.{i}/self_attn/q_proj/MatMul_quant` 这类**节点名硬编码映射**注入权重，会直接 KeyError。禁止把共享子模块塞进 `nn.ModuleList` 之外随意改名。

## 2. 新增模型五步（Step 0 每次前置）

### Step 0 刷新算子支持清单（每次 export 前必做，不可跳过）

任何 export 任务（新模型对接、重导出、换算子集、排障）动手前，先遍历 `mindspore-lite/lite_llm/custom_ops/torch_custom/`，确认算子支持状态最新，再做模型 export：

1. **枚举**：遍历目录全部 `ms_*.py`——每文件一个算子（模块内 `Ms<Op>` 类 = eager 参考实现 + ONNX symbolic），并核对 `torch_custom/__init__.py` 的 `__all__` 与目录一致（漏注册 → 包级 import 断）。
2. **比对快照**：目录实况 vs 本 skill §3 算子清单与 `EXPORTER-GUIDE.md` §4 算子表。目录有而文档无 → **先补文档再导出**；文档有而目录无（算子删除/改名）→ 同步清理两份文档与代码引用，防 `ImportError`。
3. **新算子归位**：新算子并非都进 `NnrtOpSet`——接入路径有三：OpSet 原语（forward 链路调用，如 `MsAddSoftmax`）、图后融合 pass（`fuse_add_rmsnorm` 产出 `MsAddRmsNorm`）、quant pass 直接建节点（`MsFloatCastInt`）。按语义归位后才动 wrapper/后处理。
4. **编译侧边界**：`ms_*.py` 只定义 ONNX 契约与 eager 参考，omg 编译还要求 DDK 侧 `libcustom_op.so` 有对应核实现——新算子过不了 omg 先查 DDK 支持，别改导出图迁就。

清单确认最新后：新模型对接走 Step 1–5；重导出/换算子集/排障直接执行对应操作。

### Step 1 兼容性判定（先问四个问题）
1. 层属性名是否 `input_layernorm / self_attn.{q,k,v,o}_proj / mlp.{gate,up,down}_proj / norm / lm_head`？
2. 数学是否 RMSNorm + pre-norm 残差 + 标准 RoPE + GQA？
3. KV cache 是否逐层 K/V、BNSD `[1, num_kv_heads, max_len, head_dim]`？
4. **加载层前置检查（MiniCPM 演练实锤的卡点）**：目标 `model_type` 在受支持的 transformers 区间内有内置实现吗？用 `python -c "import transformers.models.<type>"` 或查源码树即可确认。无内置（如 `minicpm`：4.57 与 5.x 均已移除）→ HF/GGUF 加载都退化为 `trust_remote_code` 依赖模型仓库自带 modeling，且该 custom code 可能只兼容旧版（MiniCPM 4.x 在 5.x 下 KV-cache 崩溃）；GGUF `gguf_file=` 路径同样断（依赖内置 arch 映射）。此时要么锁旧版 transformers，要么换同族 native 架构后继（MiniCPM5 = native llama）。**此问不过，后面四步全白走。**

前 3 问 yes 且第 4 问（加载层）有解 → wrapper 为空子类（Qwen2.5 即如此）；"一行级"数值差异（MiniCPM 的 `scale_emb`、per-layer rotary 表位置）→ 薄子类/ exporter 局部适配。仅 per-head Q/K norm 差异 → 子类化 `NnrtAttention` 加 `q_norm/k_norm` 适配器（照抄 `qwen3_wrapper.py` 的 `Qwen3Attention`）。架构级不同（MLA/NeoX/ALiBi）→ 不硬塞，在 `_base` 旁另立 sibling 基类，复用 `NnrtOpSet` 与 I/O 契约。

### Step 2 建 `export/models/<name>/` 四件套
- `<name>_wrapper.py`：薄子类（见上）
- `<name>_exporter.py`：照抄 `qwen2_5_exporter.py` 结构 —— `load()`（HF 目录走 `AutoConfig`+`from_pretrained`；`.gguf` 走 `from_pretrained(dir, gguf_file=basename, dtype=fp16, attn_implementation="eager")`，transformers>=4.57）→ `export()`（构造 wrapper，`torch.onnx.export(..., opset_version=18, operator_export_type=ONNX_FALLTHROUGH, dynamo=False)`，**dynamo=False 必需**：legacy exporter 才走 `Ms*` 的 custom symbolic）→ `slim(skip_fusion_patterns=["FusionGemm"])` → `fuse_add_rmsnorm` → `duplicate_shared_initializers` → `apply_quant`（W4A16）或 `apply_shared_weight`（插入 `embedding_weight` 图输入到 index 6）→ `validate_contract`
- `<name>_gguf_loader.py`：节点名→GGUF tensor 名映射三张表（`QUANT_MATMUL_MAP` / `FP16_WEIGHT_MAP` / `MODEL_WEIGHT_MAP`），key 是量化后节点名（`MatMul_quant`），value 是 `blk.{i}.attn_q.weight` 等
- `__init__.py`

### Step 3 资产导出（exporter 内完成）
`embedding(.bin/_quant.bin)`、`rope_cos/sin.bin`（调 HF 模型自身 `rotary_emb` 预计算）、`attention_mask.bin`、`<name>_config.json`（architecture/generation/npu/sampling/assets）。

### Step 4 注册入口
`export/mslite_llm_export.py`：`MODEL_TYPES["<name>"] = {exporter, gguf_loader, onnx_name, ...}` + `MODEL_TYPE_BY_ARCH["<config.model_type>"] = "<name>"`。

### Step 5 验证（不可跳过）
1. `cd mindspore-lite/lite_llm && python -m pytest tests/py/ -q`（38 个；golden fixture 由 `tests/py/conftest.py` 自动生成）
2. `validate_contract` 通过（7 输入 + 2L KV 输出契约）
3. fake 小模型对比法（迁移/重构时）：同一份随机权重分别跑参考实现与新 wrapper，断言 logits+KV `torch.equal` 逐位一致、ONNX 节点名集合一致、算子 multiset 一致、量化后 GGUF map key 全部可解析。已固化模式：`tests/py/test_minicpm_wrapper.py`（tiny LlamaForCausalLM 参考实现 + wrapper 数值/结构/量化/GGUF key 四层断言）。

## 3. 替换/扩展算子集（不同规格用不同融合算子）

子类化 `NnrtOpSet`，只覆写差异原语，注入 `Wrapper(model, config, op_set=MyOpSet())`：

```python
class UnfusedSpecOpSet(NnrtOpSet):
    def mask_softmax(self, weights, mask):
        return (weights + mask).softmax(dim=-1)
```

注意：`utils/onnx_postprocess.py` 的 `fuse_add_rmsnorm` 假设默认算子集的 `Add + MsRmsNorm` 模式；改 `rmsnorm`/matmul 原语时须同步导出侧融合 pass（跳过或扩展）。NNRT 图 I/O 契约（由 NPU 运行时固定）与算子集无关，换算子集不影响运行时契约。

现有 `custom_ops/torch_custom/` 算子（**快照，以 Step 0 遍历实况为准**）：`ms_rotary_pos_emb` `ms_scatter_nd` `ms_rms_norm` `ms_group_matmul` `ms_add_softmax` `ms_add_rms_norm`（图后融合）`ms_quant4_n0_group32`（量化）`ms_float_cast_int`（W4A8 激活量化：FP16 截断饱和到 INT8，由 `export_quant.py` quant pass 直接建节点）。每个都是 `torch.autograd.Function` + ONNX symbolic。

## 4. 关键契约速查

- **图输入**（顺序固定）：`valid_seq_len, lmhead_idx, rope_cos, rope_sin, inputs_embeds, attention_mask, embedding_weight` + `past_key_{i}/past_val_{i}`；**输出**：`logits` + `out_key_{i}/out_val_{i}`（KV 设备侧原地写回）。embedding lookup 在 CPU 侧，故 `input_ids` 不是图输入。
- **KV scatter**：纯 `MsScatterND`，`past[:, :, pos:pos+seq, :] = state`，无 mask 输入，状态 cast fp16，layout 参数 `"BNSD"`。
- **版本约束**：`transformers>=4.57,<5`（GGUF `gguf_file=` 为下限来源）、`torch>=2.0`；`custom_ops` 由 wrapper 内 `sys.path` bootstrap 定位（装 wheel 后 no-op）。
- **下游**：量化后的 ONNX 经 `omg` 编译为 `.omc` 离线模型（`--framework=5 --target=omc`），再由 `utils/msl_pack.py` 打包 `.msl`。

## 5. 常见坑

| 症状 | 根因 | 修法 |
|---|---|---|
| GGUF 注入 KeyError | wrapper 层级/命名偏离镜像结构，节点名丢 scope | 保持 `model.layers.{i}...` 属性名与 `__call__` 调用链 |
| 导出图无 `custom::` 节点 | 用了 dynamo exporter 或未设 `ONNX_FALLTHROUGH` | `dynamo=False` + `operator_export_type=ONNX_FALLTHROUGH` |
| transformers 升级后崩 | 某处调了 HF 模块 `forward` 或内部 API | 只读属性；GGUF 映射探测包 try/except 降级 |
| golden 测试 FileNotFoundError | fixture 被 gitignore | 已由 conftest/CMake 自动生成，无需手工跑 `gen_golden.py` |
| `max_length` 校验失败 | 非 chunk_size 整数倍 | NNRT chunked prefill 要求整除 |
