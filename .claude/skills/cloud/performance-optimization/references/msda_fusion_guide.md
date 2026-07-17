# MSDA（MultiScaleDeformableAttention）融合适配（以Grounding-DINO模型为例）

> 本文档是 `performance-optimization` skill 的案例文档之一。
> 命令模板（benchmark/profiling/精度对齐）见 [other_opt_methods.md](other_opt_methods.md)。
> Custom 改写通用规范见 [custom_operator_fusion.md](custom_operator_fusion.md)。

## 1. 背景与目标

Grounding-DINO / Deformable-DETR 系模型中，MSDA（MultiScale Deformable Attention）在导出 ONNX 后常表现为大量 `GridSample` + gather/reshape/transpose 组合：

- 节点数多、访存重
- profiling 中 `GridSample` 往往成为热点
- 端到端时延容易被该段链路拉高

目标是在 ONNX 图中把 MSDA 子图“上收”为单个融合算子节点：

- non-fuse：图里存在较多 `GridSample`
- fuse：图里出现 `MultiScaleDeformableAttnFunction`（通常每个 encoder layer 1 个），并显著减少/消除 `GridSample`

本案例采用“导出侧 emit 融合算子 op_type”的方式实现融合：

- 在 PyTorch → ONNX 导出阶段，使用 `torch.autograd.Function.symbolic()` 直接生成 `MultiScaleDeformableAttnFunction` 节点
- 运行时由后端/转换器识别该节点并匹配目标融合 kernel

## 2. 适用前提与注意事项

1. 硬件/后端是否支持该融合算子是前提条件。
   - 若后端不支持，转换/编译阶段可能失败（常见于部分 Atlas 300I Duo 环境）
   - 建议始终保留回退开关：导出 non-fuse（GridSample）版本用于“保证可转可跑”，fuse 版本用于“支持硬件（如 Atlas 800I A2）验证性能”
2. 本案例不是 `Custom` 节点（`g.op("Custom", ...)`）改写，而是输出特定 `op_type`：`MultiScaleDeformableAttnFunction`。
3. `symbolic()` 必须补齐输出形状/类型信息（至少 setType），否则 infershape 信息缺失可能导致转换失败或 fallback 拆解。

## 3. 核心适配思路（导出侧 patch + symbolic emit）

### 3.1 需要 patch 的位置

在 HuggingFace transformers 的 Grounding-DINO 实现中，MSDA 对应模块通常为：

- `transformers.models.grounding_dino.modeling_grounding_dino.MultiScaleDeformableAttention`

导出脚本需要做的事：

1. patch `MultiScaleDeformableAttention.forward`，让其 forward 不再调用原始实现
2. forward 内调用 `_MSDAFusionFn.apply(...)`：
   - `forward()`：走纯 PyTorch fallback（确保 tracing 时能跑、语义正确）
   - `symbolic()`：生成 `MultiScaleDeformableAttnFunction` ONNX 节点

### 3.2 核心代码（可直接复用的最小实现）

下面给出最小可用版本的核心片段（导出脚本内定义即可）：

```python
import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_dim_size
from transformers.models.grounding_dino import modeling_grounding_dino as gm


def msda_forward_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    batch_size, _, num_heads, hidden_dim = value.shape
    _, _, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([int(h * w) for h, w in value_spatial_shapes.tolist()], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (h, w) in enumerate(value_spatial_shapes.tolist()):
        value_l = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, h, w)
        )
        sampling_grid_l = (
            sampling_grids[:, :, :, level_id]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        sampling_value_l = torch.nn.functional.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, -1, num_levels * num_points
    )
    sampling_value = (
        torch.stack(sampling_value_list, dim=-2)
        .flatten(-2)
    )
    output = (sampling_value * attention_weights).sum(-1)
    output = output.view(batch_size, num_heads * hidden_dim, -1).transpose(1, 2)
    return output


class _MSDAFusionFn(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights):
        del ctx, value_level_start_index
        return msda_forward_pytorch(value, value_spatial_shapes,
                                    sampling_locations, attention_weights)

    @staticmethod
    def symbolic(g, value, value_spatial_shapes, value_level_start_index,
                 sampling_locations, attention_weights):
        y = g.op(
            "MultiScaleDeformableAttnFunction",
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            input_names_s=[
                "value",
                "value_spatial_shapes",
                "value_level_start_index",
                "sampling_locations",
                "attention_weights",
            ],
            output_names_s=["output"],
            type_s="MultiScaleDeformableAttnFunction",
        )

        value_sizes = value.type().sizes() or []
        if len(value_sizes) >= 4:
            bs = value_sizes[0]
            num_query = _get_tensor_dim_size(sampling_locations, 1) or 0
            num_heads = value_sizes[2]
            embed_dims = value_sizes[3]
            y.setType(value.type().with_sizes([bs, num_query, num_heads * embed_dims]))

        return y


def patch_msda_fusion():
    def new_forward(self, value, value_spatial_shapes,
                    value_spatial_shapes_list, level_start_index,
                    sampling_locations, attention_weights, im2col_step):
        del self, value_spatial_shapes_list, im2col_step
        return _MSDAFusionFn.apply(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )

    gm.MultiScaleDeformableAttention.forward = new_forward
```

#### 关键解释（为什么这些点不能省）

1. `symbolic()` 用 `g.op("MultiScaleDeformableAttnFunction", ...)`：
   - 这是让 ONNX 图里出现融合算子的唯一关键点。
2. `input_names_s/output_names_s/type_s`：
   - 用于给后续 parser/后端提供语义信息（不同链路可能对属性依赖程度不同，建议保持与已验证案例一致）。
3. `setType(with_sizes([...]))`：
   - 为 infershape 提供足够信息，避免转换阶段“形状推导缺失”引起的失败或退化。
4. `forward()` 删除 `value_level_start_index`：
   - forward 仅用于导出时跑通，实际部署依赖 symbolic 的融合算子；但必须确保 forward 语义正确，否则导出过程可能报错或输出错误形状。

### 3.3 回退开关（强烈建议保留）

导出脚本建议提供开关，例如：

- `--disable-msda-fusion` 或环境变量 `DISABLE_MSDA_FUSION=1`

行为：

- 开启：patch 到 `grid_sample` 路径（non-fuse，适配更多硬件，保证可转可跑）
- 关闭：patch 到 `_MSDAFusionFn`（fuse，用于支持硬件验证性能）

## 4. 如何确认“融合真的生效”

### 4.1 ONNX 静态检查（推荐先做）

统计 ONNX 中的节点类型数量：

- non-fuse：`GridSample > 0` 且 `MultiScaleDeformableAttnFunction == 0`
- fuse：`MultiScaleDeformableAttnFunction > 0` 且 `GridSample ≈ 0`

### 4.2 Profiling 检查（用 msprof 抓 CSV）

按 [other_opt_methods.md](other_opt_methods.md) 的 profiling 模板抓取：

- `op_statistic_*.csv`：看热点算子类型是否从 `GridSample` 迁移到融合算子
- `op_summary_*.csv`：看节点名、dtype/layout 是否符合预期，检查是否发生 fallback 拆解

## 5. 精度对齐建议

1. 统一基线：
   - non-fuse ONNX（ORT CPU）作为“数学语义基线”
   - fused MindIR 作为“部署产物”
2. 指标建议：
   - `max_abs_diff / mean_abs_diff / rmse / cosine`
   - 必要时对关键中间节点做截断对齐（避免误差扩散导致难定位）

## 6. 常见失败形态与处理

1. 转换失败（tiling/impl 不支持）：
   - 现象：converter/编译阶段报不支持融合算子
   - 处理：启用回退开关导出 non-fuse；融合版本交给支持硬件（如 Atlas 800I A2）验证
2. 融合成功但无性能收益或回归：
   - 处理：按同口径 benchmark 对比 fused vs non-fuse；profiling 排查是否引入 `TransData/Cast/Concat` 等副作用
3. 融合成功但精度不达标：
   - 处理：先用 non-fuse ORT 作为基线对齐；必要时对 MSDA 前后截断节点做误差定位，检查输入顺序/shape 推导是否一致
