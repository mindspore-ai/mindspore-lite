# BooguImage 专属 W8A8 量化

BooguImage 量化实现通过 `BooguImageQuantizer.quantize(...)` 提供离线量化类接口。
调用方传入模型目录、输出目录和量化配置，接口负责标定、量化和保存流程。
当前支持单张 Ascend NPU、BooguImageTurboPipeline 的 T2I 推理。

## 离线量化与保存

`config` 为调用方提供的量化 YAML 文件路径，标定提示词路径由该文件指定。
需要从本地 Boogu 仓库导入模型时，可传入 `boogu_path="/path/to/Boogu-Image"`。
输出必须是不存在或为空的目录，且不能与原模型目录重叠。

```python
from lite_boost.model.booguimage.quant.quantization_stage import BooguImageQuantizer

manifest = BooguImageQuantizer.quantize(
    model_path="/path/to/Boogu-Image-0.1-Turbo",
    output="/path/to/quant_model",
    config="/path/to/quant.yaml",
    device="npu:0",
)
```

类接口依次加载模型、应用 NPU 适配、采集激活统计、执行策略、
生成未 Pad 的 INT8 张量、保存模型快照并回读校验。离线量化需要 torch、torch_npu、PyYAML、
safetensors 和 BooguImage 的模型依赖；须先初始化 CANN 环境。

## 两档策略

```yaml
Quant:
  alg: W8A8SQFusedFFNFull
  alpha: 0.6
  calib_prompts: calib_prompts.json
```

`calib_prompts` 是非空字符串数组的 JSON 文件路径，相对路径按 YAML 文件所在目录解析。
`alpha` 必须在 `(0, 1]`。标定固定为 1024 分辨率、4 步，seed 为 42 加样本序号。
标定只保存逐输入通道 absmax，不保存完整激活；只收到空输入的目标保留 BF16。

| 策略 | DiT 量化范围 |
| --- | --- |
| W8A8SQFusedFFN | 融合量化 FFN linear_1 和 linear_3，linear_2 保留 BF16 |
| W8A8SQFusedFFNFull | 进一步量化 linear_2 和注意力 q/k/v/out 投影 |

两档均量化文本编码器的 gate_proj、up_proj。
融合 FFN 要求 gate/up 同形状、无 bias，拼接后的输出维度对齐 256；完整档校验 256 个注意力投影。

## 在线加载和推理

在线只导入 `w8a8_quant_linear.py`，不导入离线算法、标定和量化产物保存实现。
从量化模型目录加载 pipeline，应用 NPU 适配和量化层替换，然后搬到 NPU：

```python
import torch
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
from lite_boost import BoostManager
from lite_boost.model.booguimage.quant.w8a8_quant_linear import apply_w8a8_sq_offline

pipe = BooguImageTurboPipeline.from_pretrained(
    "/path/to/quant_model", torch_dtype=torch.bfloat16, trust_remote_code=True,
)
BoostManager(pipe)
apply_w8a8_sq_offline(pipe, "/path/to/quant_model")
pipe.to("npu:0")
```

在线不需要指定 `alg` 或 `alpha`，量化策略、层清单、形状和量化参数均从量化模型目录读取。
在线仍会校验量化产物格式、支持的策略及模型结构。
部署替换必须早于 `pipe.to(device)`，以替换快照裁剪后留下的占位张量。

## 量化模型目录与 Pad

输出目录由调用方通过 `output` 指定，下面使用 `quant_model/` 作为示例名称。
离线保存会保留源模型目录结构，裁剪已被 INT8 替代的 BF16 张量，新增 `int8/` 和
`manifest.json`。以下按[官方 Boogu-Image-0.1-Turbo 模型目录](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo/tree/main)
完整列出顶层目录；源模型存在 `assets/` 时也会保留该目录。
文件名中的 `*` 表示随源模型版本或分片数量变化的一组文件。

```text
quant_model/
├── manifest.json
├── model_index.json
├── int8/
│   └── shard_*.pt
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model-*.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── transformer_boogu.py
├── mllm/
│   ├── config.json
│   ├── generation_config.json
│   ├── model-*.safetensors
│   ├── model.safetensors.index.json
│   ├── chat_template.json
│   ├── merges.txt
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── video_preprocessor_config.json
│   ├── vocab.json
│   ├── README.md
│   └── .gitattributes
├── processor/
│   ├── added_tokens.json
│   ├── chat_template.jinja
│   ├── merges.txt
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── video_preprocessor_config.json
│   └── vocab.json
├── scheduler/
│   ├── scheduler_config.json
│   └── scheduling_flow_match_euler_discrete_time_shifting.py
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── assets/
│   ├── *.png
│   ├── *.svg
│   └── .ms_upload_cache
├── README.md
├── README_CN.md
└── .gitattributes
```

`transformer/`、`mllm/` 中的文件名称以上游
[hotfix-20260625 的 Transformer](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo/tree/hotfix-20260625/transformer)
和 [MLLM](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo/tree/hotfix-20260625/mllm)
目录为例。实际输出沿用调用方提供的源目录文件名；裁剪后没有剩余张量的权重分片会被移除。
源目录中的其他文件和子目录也会保留，保存流程不会自动筛选模型运行必需文件。

| 目录或文件 | 内容与用途 |
| --- | --- |
| `int8/` | 新生成的量化张量分片。以 `shard_000.pt` 起始编号，按约 4 GiB 的目标大小分片；每片保存以层路径和张量名称为 key 的字典，包含 INT8 权重、权重 scale、逆平滑 scale 及可选 bias。 |
| `transformer/` | DiT 结构配置、模型实现代码，以及裁剪后保留的 BF16 权重。已量化目标的原 BF16 张量被删除，权重索引同步更新；保留哪些层取决于量化策略和有效标定统计。 |
| `mllm/` | 多模态语言模型配置、裁剪后保留的原精度权重，以及源目录自带的分词、预处理和说明文件。已量化的文本 gate/up 投影按实际 checkpoint key 裁剪，索引同步更新。 |
| `processor/` | 文本与图像输入处理配置，包括词表、分词规则、特殊 token、对话模板，以及图像和视频预处理配置。沿用源模型文件。 |
| `scheduler/` | 调度器配置和源模型自带的调度器实现代码，保留原有采样配置。 |
| `vae/` | VAE 结构配置和原精度权重，用于图像与潜变量之间的转换；当前量化策略不处理 VAE。 |
| `assets/` | 源目录中的示例图片、图表等说明资源，以及源目录可能自带的上传记录。仅在源目录存在时保留，不参与量化层计算。 |
| `manifest.json` | 新生成的量化清单，记录格式版本、策略、alpha、标定提示词内容、分片列表、目标组件和层路径、原始维度、张量名称、保留 BF16 的目标及快照裁剪统计。实际权重和 scale 张量位于 `int8/`。 |
| `model_index.json` | 源模型的 pipeline 组件配置，声明 Transformer、MLLM、Processor、Scheduler 和 VAE 的加载类型。 |
| 根目录的 `README.md`、`README_CN.md`、`.gitattributes` | 源模型自带的说明和仓库属性文件，存在时原样保留。 |

格式版本为 3，`tensor_layout` 为 `unpadded_kn`。张量按原始维度保存，
Linear 权重布局为 `[in_features, out_features]`，融合 FFN 为 `[dim, 2 * inner_dim]`。
量化模型目录中保留 weight_scale、逆平滑 scale、可选 bias，以及原始形状和策略信息。

权重、scale、bias 的 Pad 在在线构建量化层时执行一次，不写入离线量化产物。
融合 FFN 的输入激活先动态量化，再在前向中补零；输出投影会切掉 Pad 对应的通道。
文本投影保持原形状。

模型快照只删除已量化的 BF16 张量，其他权重、配置和模型代码保留。
MLLM 裁剪按实际 checkpoint 匹配 `language_model...` 或 `model.language_model...`，
缺失或歧义直接拒绝；权重索引与 total_size 同步更新，safetensors metadata 保留。
源模型文件不会被修改；部署时从生成的量化模型目录读取模型资产，不依赖原始权重目录，
运行环境仍需提供 BooguImage 模型代码及相关 Python 依赖。

## 四个 Python 文件

| 文件 | 阶段 | 职责 |
| --- | --- | --- |
| quantization_stage.py | 离线 | 公开类、策略、目标选择、标定和量化流程编排 |
| smooth_quant.py | 离线 | 平滑与权重量化，生成未 Pad 张量 |
| package_w8a8.py | 离线 | 保存量化产物、快照裁剪、索引更新和回读验证 |
| w8a8_quant_linear.py | 在线 | 加载量化产物、校验、替换、Pad 和前向计算 |

量化实现位于 `model/booguimage/quant/`，目录中保留上述四个业务 Python 文件和本文档。
离线与在线分别实现量化产物的写入和读取，通过磁盘格式对接。
