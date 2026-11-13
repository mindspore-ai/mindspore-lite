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

/*!
 * \file fused_attn_out_transpose_quant.cpp
 * \brief OpDef registration for FusedAttnOutTransposeQuant
 * (Atlas 300I Duo W8A8 LLM graphs).
 */

#include "register/op_def_registry.h"

namespace ops {
// FusedAttnOutTransposeQuant: fused attention-output head-major->token-major
// fold + per-channel smooth + per-token dynamic quant for Atlas 300I Duo
// W8A8 LLM graphs. Pure-vector kernel (AICore task).
//
// Local compute core of report section 5.4 op K1
// (FusedAttnOutExchangeTransposeQuant): the
// multi-rank AG exchange stays outside (single-device contract); this op
// fuses the local tail the exchange wraps -- the FA output layout transposes,
// the smooth multiply and the dynamic quant:
//   Transpose#1 + AG (exchange) + Transpose#2(236us) + slice(102us)
//   + Mul-smooth(18us) + DynamicQuant(69us) + TransData(39us)
// (local core without the AG ~590us/layer at suffix=1536).
//
// Inputs (ND, row-major):
//   attn         float16 [H, M, D]  FA/PFA head-major output (BNSD flattened;
//                                   leading dims fold into H, e.g. [B,H,S,D])
//   smooth_scale float16 [H*D]      optional per-channel smooth scale
// Outputs:
//   y            int8    [M, H*D]   W8A8 activation (ND; feeds
//                                   QuantBatchMatmulV3 x1, no TransData)
//   token_scale  float32 [M]        dequant scale per row, absmax/127 (same
//                                   contract as the 310P DynamicQuant output)
//
// Formula (report section 5.4):
//   a[m, h*D+d] = attn[h, m, d] * smooth_scale[h*D+d]   (fp32 mul, one fp16
//                                                           rounding)
//   token_scale[m] = absmax_row(a) / 127
//   y[m, j]   = round_to_int8(a[m, j] * 127 / absmax_row(a))
//
// Constraint: D must be 128 (the 2D MTE2 fold gathers 128-element head runs;
// other head dims would need the FusedQKNormRope interleave machinery).
class FusedAttnOutTransposeQuant : public OpDef {
 public:
  explicit FusedAttnOutTransposeQuant(const char *opName) : OpDef(opName) {
    auto &inAttn = this->Input("attn");
    inAttn.ParamType(REQUIRED);
    inAttn.DataType({ge::DT_FLOAT16});
    inAttn.Format({ge::FORMAT_ND});
    inAttn.UnknownShapeFormat({ge::FORMAT_ND});

    auto &inSmoothScale = this->Input("smooth_scale");
    inSmoothScale.ParamType(OPTIONAL);
    inSmoothScale.DataType({ge::DT_FLOAT16});
    inSmoothScale.Format({ge::FORMAT_ND});
    inSmoothScale.UnknownShapeFormat({ge::FORMAT_ND});

    auto &outY = this->Output("y");
    outY.ParamType(REQUIRED);
    outY.DataType({ge::DT_INT8});
    outY.Format({ge::FORMAT_ND});
    outY.UnknownShapeFormat({ge::FORMAT_ND});
    outY.AutoContiguous();

    auto &outTokenScale = this->Output("token_scale");
    outTokenScale.ParamType(REQUIRED);
    outTokenScale.DataType({ge::DT_FLOAT});
    outTokenScale.Format({ge::FORMAT_ND});
    outTokenScale.UnknownShapeFormat({ge::FORMAT_ND});
    outTokenScale.AutoContiguous();

    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true);
    aicConfig.DynamicShapeSupportFlag(true);
    aicConfig.NeedCheckSupportFlag(false);
    aicConfig.DynamicFormatFlag(true);
    aicConfig.DynamicRankSupportFlag(false);
    aicConfig.ExtendCfgInfo("softsync.flag", "true");

    this->AICore().AddConfig("ascend310p", aicConfig);
  }
};

OP_ADD(FusedAttnOutTransposeQuant);
}  // namespace ops
