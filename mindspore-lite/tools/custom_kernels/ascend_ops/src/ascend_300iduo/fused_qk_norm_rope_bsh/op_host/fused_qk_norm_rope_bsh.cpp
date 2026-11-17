/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/op_def_registry.h"

namespace ops {
// FusedQKNormRopeBSH: per-head RMSNorm (qk_norm) + NeoX RoPE on flat
// BSH-layout q/k rows (Atlas 300I Duo). Pure-vector kernel (AICore task).
//
// Differences vs fused_qk_norm_rope (BNSD variant):
//   - no v path (attention consumes q/k directly in BSH),
//   - q/k in as [M, H*D] flat rows and out in the SAME shape (no BNSD
//     transpose, no deinterleave-to-transposed output),
//   - q|k head groups are concatenated per row inside the kernel so the
//     rms reduction / norm broadcast / gamma scale run as flat contiguous
//     vector ops over one [rows, G, 16*D] tile.
//
// Inputs (ND, row-major):
//   q       float16 [M..., Hq*D]   head-major within the row
//   k       float16 [M..., Hkv*D]
//   gamma_q float16 [D]            qk_norm q weight
//   gamma_k float16 [D]            qk_norm k weight
//   cos     float16 [M..., D/2]    precomputed rotary cos table (half form)
//   sin     float16 [M..., D/2]    precomputed rotary sin table (half form)
// Outputs (ND, row-major):
//   q_out   float16 [M..., Hq*D]   mirrors q
//   k_out   float16 [M..., Hkv*D]  mirrors k
//
// Math (numerics aligned with fused_qk_norm_rope: fp32 norm accumulation, raw Sqrt/Reciprocal
// chain, one fp16 rounding at the very end -- better than the network chain,
// which multiplies x*invRms in fp16):
//   For each head h and row m:
//     rms    = sqrt(mean_d(x^2) + epsilon)          (epsilon attr, 1e-6)
//     xn     = x * rsqrt(...) * gamma[d]
//     out[0:D/2]   = xn1*cos - xn2*sin              (NeoX rotate-half)
//     out[D/2:D]   = xn2*cos + xn1*sin
class FusedQKNormRopeBSH : public OpDef {
 public:
  explicit FusedQKNormRopeBSH(const char *name) : OpDef(name) {
    this->Input("q")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("k")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gamma_q")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gamma_k")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("cos")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("sin")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("q_out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Output("k_out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);

    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .ExtendCfgInfo("softsync.flag", "true");
    this->AICore().AddConfig("ascend310p", aicConfig);
  }
};

OP_ADD(FusedQKNormRopeBSH);
}  // namespace ops
