# Exporter 模块技术赋能文档

> 适用范围：`mindspore-lite/lite_llm/export/`
> 读者：需要新增/迁移大模型导出的开发者、CI 维护者、NPU 部署联调人员
> 关联文档：[DESIGN.md](DESIGN.md)（.msl 格式与运行时设计）、[PROTOCOL.md](PROTOCOL.md)（C API 协议）、export/README.md（快速上手）

---

## 1. 模块定位与端到端流水线

Exporter 是 lite_llm 的**模型接入层**：把 HuggingFace bfloat16 权重或 GGUF Q4_0 权重的大语言模型，转换成麒麟 NPU（NNRT 执行器）可加载的单文件 `.msl` 推理产物。

唯一对外入口是 `export/mslite_llm_export.py`：

```bash
python mslite_llm_export.py \
    --target kirin9020 \
    --model  /path/to/model_dir_or_file.gguf \
    --output /path/to/out.msl \
    --max-length 1024 \
    --chunk-size 64
```

`run_pipeline` 内部四步（顺序固定）：

| 步骤 | 做什么 | 关键产物 |
|---|---|---|
| Step 1 skeleton export | HF/GGUF 加载 → trace ONNX → 后处理/量化 → 资产导出 | `*.onnx` + `embedding*.bin` + `rope_*.bin` + `attention_mask.bin` + `*_config.json` |
| Step 2 omg compile | DDK `omg` 把 ONNX 编译为离线模型 | `model/*.omc`（`--framework=5 --target=omc`） |
| Step 3 tokenizer | tokenizer → `vocab.bin` + chat template IR 编译 | `tokenizer/vocab.bin` |
| Step 4 package | `msl_pack.py` 打包为单文件 `.msl`（v1 格式，STORE-only，无外部 manifest） | `out.msl` |

模型类型经 `config.model_type` 自动路由（`MODEL_TYPE_BY_ARCH`），无需手工指定。

---

## 2. 架构设计：两轴正交解耦

### 2.1 为什么这样设计

历史方案（monkey-patch transformers 的 `forward`）依赖 `modeling_qwen2.*` 内部签名，把整个导出链锁死在单一 transformers 版本上，CI 无法统一。现行方案把「模型长什么样」和「用什么融合核计算」拆成两条**正交变化轴**，transformers 只在加载层经公开 API 使用（`AutoConfig` / `AutoModelForCausalLM.from_pretrained` / `gguf_file=`），forward 逻辑零 transformers 依赖。

### 2.2 两条变化轴

| 变化轴 | 承载者 | 扩展方式 |
|---|---|---|
| **架构轴**（模型算什么） | `NnrtDecoderWrapper` 子类 + `attn_module` 类属性 | 子类覆写 hook |
| **算子轴**（用什么核实现） | `NnrtOpSet` 六原语，经 `wrapper(op_set=...)` 注入 | 策略对象覆写单个原语 |

两轴独立组合：N 个架构 × M 个算子集不需要 N×M 个类。

### 2.3 类分层（`models/_base/nnrt_decoder_wrapper.py`）

```text
NnrtOpSet                      # 算子集策略：rope / kv_scatter / qk_matmul /
                               #   pv_matmul / mask_softmax / rmsnorm 六原语，
                               #   默认实现 = 标准 Ms* 核（fp16 cast、BNSD 等契约细节封装在原语内）
├── NnrtRmsNorm                # norm 适配器：只读 HF 子模块 weight/variance_epsilon
├── NnrtAttention              # attention 适配器：apply_qk_norm hook（默认 no-op）
│                              #   GQA reshape、RoPE、KV scatter、softmax 全在此
├── NnrtDecoderLayer           # 单层：pre-norm 残差 + attn + mlp(SwiGLU)
├── NnrtDecoderCore            # 层循环容器（镜像 hf_model.model 层级）
└── NnrtDecoderWrapper         # 顶层：lm_head + NNRT 契约 forward 签名
                               #   （embed_tokens 刻意不持有：查表在 CPU 侧，
                               #    经 inputs_embeds 入图；lm_head 绑定权重
                               #    由后处理转为 embedding_weight 图输入）
```

### 2.4 两条铁律

1. **禁止 monkey-patch transformers**。wrapper 只读 HF 子模块的普通属性（`weight` / `variance_epsilon` / `q_proj`…），绝不调用其 `forward`。这是版本解耦的根基。
2. **wrapper 必须是镜像 `*ForCausalLM` 层级的真实 `nn.Module` 树**（`wrapper.model.layers.{i}.self_attn.q_proj`），子模块经 `__call__` 调用。若用 flat 方法持有，trace 出的节点名会退化为 `/q_proj/MatMul`，而 GGUF loader 按 `/model/layers.{i}/self_attn/q_proj/MatMul_quant` 这类**节点名硬编码映射**注入权重，会直接 KeyError。

---

## 3. NNRT 图契约

| | 名称 | 形状/类型 | 说明 |
|---|---|---|---|
| 输入 0 | `valid_seq_len` | `[1]` int32 | 当前有效序列长度 |
| 输入 1 | `lmhead_idx` | `[1]` int32 | logits 取出行索引 |
| 输入 2 | `rope_cos` | `[1, chunk, head_dim]` fp16 | RoPE 表（图外预计算） |
| 输入 3 | `rope_sin` | 同上 | |
| 输入 4 | `inputs_embeds` | `[1, chunk, hidden]` fp16 | embedding lookup 在 CPU 侧，`input_ids` 不是图输入 |
| 输入 5 | `attention_mask` | `[1,1,chunk,max_len]` fp16 | 面向**全量 KV cache** 的 causal mask |
| 输入 6 | `embedding_weight` | `[hidden, vocab]` | 仅共享权重路径；W4A16 量化路径无此项 |
| 输入 7+ | `past_key_{i}` / `past_val_{i}` | `[1, num_kv_heads, max_len, head_dim]` fp16 | 逐层交错排列（BNSD） |
| 输出 0 | `logits` | `[1, vocab]` | |
| 输出 1+ | `out_key_{i}` / `out_val_{i}` | 同 past | KV 设备侧原地写回 |

契约由 `utils/onnx_postprocess.py::validate_contract` 强制校验，任何后处理改动后必须过它。约束：`max_length` 必须是 `chunk_size` 的正整数倍（NPU chunked prefill 要求）。

导出 trace 的 `valid_seq_len` 从 0 开始，确保 `chunk_size == max_length` 时首个 chunk
仍完整落在 cache 内。自定义算子 shape inference 保留 Qwen3 逐头 RMSNorm 的 BSND
形状，并更新已有输出/中间张量注解而不追加重复 `value_info`；量化图重建保留这些
注解、原模型元数据及 opset imports，缺失时补充 `custom` opset。

---

## 4. 融合算子清单（`custom_ops/torch_custom/`）

每个算子 = `torch.autograd.Function` + ONNX symbolic，导出时经 `ONNX_FALLTHROUGH` 落为 `custom::Ms*` 节点。

| 算子 | OpSet 原语 | 语义 |
|---|---|---|
| `ms_rotary_pos_emb` | `rope` | RoPE 旋转 |
| `ms_scatter_nd` | `kv_scatter` | `past[:,:,pos:pos+seq,:] = state`，layout `"BNSD"`，状态 cast fp16 |
| `ms_group_matmul` | `qk_matmul` / `pv_matmul` | GQA 分组 matmul |
| `ms_add_softmax` | `mask_softmax` | `(w+mask).softmax()` 融合 |
| `ms_rms_norm` | `rmsnorm` | RMSNorm |
| `ms_add_rms_norm` | —（图后融合） | `fuse_add_rmsnorm` pass 把 `Add+MsRmsNorm` 模式融合为单节点 |
| `ms_quant4_n0_group32` | —（量化） | W4A16 反量化核 |
| `ms_float_cast_int` | —（量化） | W4A8 激活量化：FP16 截断饱和到 INT8 |

`kv_scatter(past, current_pos, current)` 按绝对位置把当前 BNSD K/V 写入完整 cache；
Q/K 在 BNSD 布局下通过 `rope(query, key, cos, sin)` 完成 RoPE。Qwen3 的逐头 Q/K
norm 在转置前的 BSND 投影上执行。

换算子集示例（同架构不同规格用不同融合粒度）：

```python
class UnfusedSpecOpSet(NnrtOpSet):
    def mask_softmax(self, weights, mask):
        return (weights + mask).softmax(dim=-1)

wrapper = Qwen2NnrtWrapper(model, config, op_set=UnfusedSpecOpSet())
```

注意：`fuse_add_rmsnorm` 假设默认算子集的 `Add + MsRmsNorm` 模式；改 `rmsnorm`/matmul 原语时须同步导出侧融合 pass（跳过或扩展）。图 I/O 契约由 NPU 运行时固定，与算子集无关——换算子集不影响运行时契约。

---

## 5. 模型迁移对接指南（五步）

### Step 1 兼容性判定（先问四个问题）

1. 层属性名是否 `input_layernorm / self_attn.{q,k,v,o}_proj / mlp.{gate,up,down}_proj / norm / lm_head`？
2. 数学是否 RMSNorm + pre-norm 残差 + 标准 RoPE + GQA？
3. KV cache 是否逐层 K/V、BNSD？
4. **加载层前置检查**：目标 `model_type` 在受支持的 transformers 区间内有内置实现吗？（`python -c "import transformers.models.<type>"`）无内置 → HF/GGUF 加载都退化为 `trust_remote_code`，且 custom code 可能只兼容旧版；GGUF `gguf_file=` 路径同样断（依赖内置 arch 映射）。**此问不过，后面四步全白走。**

判定分级：前 3 问 yes 且第 4 问有解 → 空子类；「一行级」数值差异（`scale_emb`、rotary 表位置）→ 薄子类/exporter 局部适配；仅 per-head Q/K norm → 子类化 `NnrtAttention` 覆写 `apply_qk_norm`；架构级不同（MLA/NeoX/ALiBi）→ 不硬塞，在 `_base` 旁另立 sibling 基类，复用 `NnrtOpSet` 与契约。

### Step 2 建 `export/models/<name>/` 四件套

```text
<name>_wrapper.py      # 薄子类（见 Step 1 分级）
<name>_exporter.py     # <Name>Onnx 类（load/export/资产方法）+ export_<name> 编排
<name>_gguf_loader.py  # 节点名→GGUF tensor 名映射三张表 + gguf_loader()
__init__.py
```

exporter 内部流程（照抄 `qwen2_5_exporter.py`）：

```text
load()   → AutoConfig / from_pretrained（GGUF: gguf_file=, dtype=fp16, attn_implementation="eager"）
export() → 构造 wrapper → torch.onnx.export(opset_version=18,
             operator_export_type=ONNX_FALLTHROUGH, dynamo=False)   # dynamo=False 必需：
                                                                    # legacy exporter 才走 Ms* symbolic
         → slim(skip_fusion_patterns=["FusionGemm"])
         → fuse_add_rmsnorm → duplicate_shared_initializers
         → apply_quant(W4A16) 或 apply_shared_weight(插入 embedding_weight 到 index 6)   # 二选一，互斥
         → validate_contract
```

### Step 3 资产导出（exporter 内完成）

`embedding(.bin/_quant.bin)`、`rope_cos/sin.bin`（调 HF 模型自身 `rotary_emb` 预计算；注意 LLaMA≤4.4x/MiniCPM 系把 rotary 挂在每个 attention 上，需 `layers[0].self_attn.rotary_emb` fallback，旧式签名返回 `[S,D]`）、`attention_mask.bin`、`<name>_config.json`（architecture/generation/npu/sampling/assets）。

### Step 4 注册入口 `mslite_llm_export.py`

```python
MODEL_TYPE_BY_ARCH["<config.model_type>"] = "<name>"
MODEL_TYPES["<name>"] = {
    "exporter": export_<name>, "gguf_loader": <name>_gguf_loader,
    "layers": N, "onnx_name": ..., "quant_name": ..., "gguf_name": ...,
    "model_name": ...,
    "chat_template": ...,   # 仅当非 ChatML 时（如 MiniCPM 的 <用户>/<AI> 格式）
}
```

chat template 被限制在 v1 IR 子集（消息循环 + `add_generation_prompt`），工具调用类 Jinja 不支持。

### Step 5 验证（不可跳过）

1. `python -m pytest tests/py/ -q` 全绿（golden fixture 由 conftest/CMake 自动生成）
2. `validate_contract` 通过
3. eager 数值对比：**无需真实权重**——用同构家族的 transformers 内置小模型当参考（MiniCPM→tiny `LlamaForCausalLM` + 手动施加差异项），断言 logits 相对误差（fp16 < 2e-2）+ top1 一致 + KV `allclose`；再 trace 验证节点名 scope、算子 multiset、量化后 GGUF map key 全可解析。已固化模式见 `tests/py/test_minicpm_wrapper.py`。

---

## 6. 量化链路（`utils/export_quant.py`）

- 配置类：`QuantizationConfig`（"W4A8"/"W4A16"/None）、`ModelConfig`、`LiteTurboConfig`
- embedding：W4A8 仍走 `quantize_weight_g128_4bit_nz`；W4A16 统一调用 `MsQuant4N0Group32.quantize_weight_g32_4bit`，输出 compact signed int4 phase4 NZF。CPU 按需解码 NPU 使用的同一 ION blob，不展开全量 fp16 embedding。
- decoder：`apply_quant` 图变换把 `MatMul` 换成 `MatMul_quant` + `MsQuant4N0Group32`；W4A16 与 embedding 使用同一个类的权重准备方法及 `q4_0_nzf_compact_phase4` 布局。
- **互斥规则**：quant 分支直接 `apply_quant`；`apply_shared_weight` 是 FP16 分支专属（先插共享权重再 quant 会 KeyError）
- **W4A16 二进制契约**：逻辑 `[N,K]`，N/K 为正且分别为 16/32 的整数倍。cell 按 N64/K1024 划分，但只保存有效尾部，不补齐；先存全部 signed int4，再存 cell 内 `[nc,kc/32]` little-endian fp16 scales。分形内 phase4 字节排列不变，总字节数严格为 `N*(K/32)*18`。精确偏移见 [PROTOCOL.md](PROTOCOL.md) 的 Binary Format Reference；OMG 的 `embedding_weight` 输入使用同一精确尺寸。
- **兼容边界**：量化包必须标记 `npu.q4_0_weight_layout = "q4_0_nzf_compact_phase4"`。旧 padded `q4_0_nzf_phase4`、`q4_0_nzf`、planar 和缺失当前标记的包均被拒绝，须重新导出、编译整包；不能只改元数据。旧字段 `npu.weight_layout` 不作为别名或回退，未知键仍按协议忽略。对齐 shape 的容量可能相同，禁止按大小猜测布局。FP16 不受影响；本契约不覆盖 W4A8。
- **Torch 权重 API**：使用 `torch_custom.ms_quant4_n0_group32.MsQuant4N0Group32` 的静态方法 `weight_blob_size(K,N)`、`repack_q4_0_to_nzf(blocks,(K,N))`、`quantize_q4_0_blocks(weight)`、`quantize_weight_g32_4bit(weight)` 和 `dequantize_weight_g32_4bit(blob,(K,N))`。浮点矩阵输入为 `[K,N]`；原始 block 为 `[N,K/32,18]` 的 `bytes` 或 `uint8` 数组。重排逐位保留 scale 与码值，工作缓冲限于一个 cell；eager 路径消费同一 compact blob。旧模块级权重函数及 exporter 的重复 packer 已移除。
- **浮点量化规则**：采用 canonical Q4_0，首个带符号绝对最大值确定 `d=vmax/-8`，FP32 reciprocal/multiply/add 后截断并限制码值至 15，最后才将 scale 存为 FP16；不再使用旧 Torch helper 的 abs-max/7。已有 GGUF Q4_0 必须走无损 repack，不重新量化。
- **配置产物**：Qwen2.5、Qwen3、MiniCPM 的独立导出配置和统一 CLI 均声明 compact 布局；通用 packager 只保留声明，不猜测或升级旧数据。FP16 / W4A8 不产生该 Q4_0 标记。
- Qwen2.5 使用已有 external decoder weights 打包路径；`SubGraph_0.weight` 仍是单文件 `.msl` 内的资源，“external” 不要求用户手工维护第二个文件。

## 7. GGUF 权重注入（`*_gguf_loader.py`）

先导出图及占位量化权重，再按三张表注入 GGUF 原始 Q4_0 block；bias 和 norm 保留真实权重：
`QUANT_MATMUL_MAP`（量化 matmul）/ `FP16_WEIGHT_MAP`（bias+norm）/ `MODEL_WEIGHT_MAP`（最终 norm）。key 是**量化后节点名**，value 是 llama.cpp 标准命名（`blk.{i}.attn_q.weight`…）。同族模型 map 可直接复用（MiniCPM 即零改动复用 Qwen2.5 map，仅层数 40 不同）。无 bias 的模型靠 `None` 防护跳过可选条目。

GGUF Q4_0 的 unsigned split-half nibble 不能原样作为 NPU 权重：注入时经 `MsQuant4N0Group32.repack_q4_0_to_nzf` 转成 compact signed phase4 NZF，逐位保留各组 fp16 scale（包括其符号）；embedding 与 decoder 遵循同一契约。HF 下载的 GGUF 已完成量化，此处不执行 `max/-8` 或 `abs_max/7` 量化；浮点量化规则的差异不能直接用于推断这条原始 block 导入链路的数值差异。

## 8. 版本约束与环境

| 依赖 | 约束 | 来源 |
|---|---|---|
| transformers | `>=4.57,<5` | 下限=GGUF `gguf_file=` kwarg；上限=4.x→5.x 移除了多款内置模型（含 minicpm） |
| torch | `>=2.0` | |
| onnxslim / gguf / onnx | 见 `requirements.txt` | |

源码树通过 `utils.ensure_custom_ops` 按需定位 `custom_ops`，已安装的 wheel 优先直接使用其 `torch_custom` 包；独立导入 OMG 编译工具不加载 Torch。wheel 同时打包所有 CLI 导入的模型模块（含 `models.minicpm`）。**禁止**在 forward 逻辑里 import `transformers.models.*` 内部模块。

### Kirin 9020 W4A16 算子前置条件

Kirin 9020 W4A16 要求 DDK 中的 `MsQuant4N0Group32` 支持 `q4_0_nzf_compact_phase4` 布局，并与导出器、Torch 权重接口和 CPU embedding 解码保持一致。布局及兼容性要求见 [PROTOCOL.md](PROTOCOL.md)。

使用同一套 DDK 完成算子注册和 omg 编译，确保整网所需算子均已安装；不要用 Q4-only 安装包覆盖整网共享 host 库和注册配置。算子依赖见 [导出工具说明](../export/README.md#附录-addk-自定义算子依赖)。

NNRT 在 `SetDevice` 后、`Compilation_Build` 前设置 `EXTREME=4`；缺少接口或设置失败时明确报错，不静默回退。该策略可能增加功耗和温度。

## 9. 已接入模型矩阵

| 模型 | 目录 | wrapper 改动量 | 特殊点 |
|---|---|---|---|
| Qwen2.5-0.5B | `models/qwen2_5/` | 空子类 | 基准实现 |
| MiniMind-3 (Qwen3 dense) | `models/qwen3/` | 覆写 `apply_qk_norm`（15 行） | per-head `q_norm/k_norm`，head_dim=96；GGUF head_dim 修复 |
| MiniCPM-2B | `models/minicpm/` | 薄子类（~15 行）+ exporter rotary fallback | `scale_emb=12` 图入口 Mul；rotary 挂 per-attention；**加载层卡点**：transformers 无内置 minicpm，需 `trust_remote_code` 且社区 custom code 在 5.x 下 KV-cache 崩溃（导出链路本身已验证全通：数值 rel_err 2.2e-4、GGUF key 14/14，回归固化于 `tests/py/test_minicpm_wrapper.py`） |

## 10. 常见坑速查

| 症状 | 根因 | 修法 |
|---|---|---|
| GGUF 注入 KeyError | wrapper 层级/命名偏离镜像结构，节点名丢 scope | 保持 `model.layers.{i}...` 属性名与 `__call__` 调用链 |
| 导出图无 `custom::` 节点 | dynamo exporter 或未设 FALLTHROUGH | `dynamo=False` + `operator_export_type=ONNX_FALLTHROUGH` |
| transformers 升级后崩 | 某处调了 HF forward/内部 API | 只读属性；映射探测包 try/except 降级 |
| MsAddSoftmax mask 维度不匹配 | eager 验证给了 `[1,1,S,S]` | mask 必须是 `[1,1,chunk,max_seq_len]` |
| apply_quant KeyError `embedding_weight_transpose` | 先 shared_weight 再 quant | 两路径互斥 |
| RoPE 取不到 `model.rotary_emb` | rotary 挂 per-attention | `layers[0].self_attn.rotary_emb` fallback，旧式签名返回 `[S,D]` |
| 新家族对话模板错乱 | 复用了 ChatML | MODEL_TYPES 加 per-model `chat_template` |
| golden 测试 FileNotFoundError | fixture gitignore | conftest/CMake 已自动生成，无需手工 |
