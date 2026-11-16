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
 * \file fused_add_rms_norm_smooth_quant.cpp
 * \brief OpDef registration for FusedAddRmsNormSmoothQuant
 * (Atlas 300I Duo W8A8 LLM graphs).
 */

#include "register/op_def_registry.h"

namespace ops {
// FusedAddRmsNormSmoothQuant: fused residual-add + RMSNorm + per-channel
// smooth + per-token dynamic quant for Atlas 300I Duo W8A8 LLM graphs
// Pure-vector kernel (AICore task).

//
// Unlike CANN's AddRmsNormQuantV2 (static per-channel scales, which does not
// match this network's per-token dynamic quant and has no smooth_scale), the
// hand-written op keeps the network's dynamic-quant semantics and folds the
// smooth multiply into the norm constant, so the single int8 output feeds
// BOTH the gate and up GEMMs (removing their duplicated quantization).
//
// Inputs/outputs (leading dims are flattened into M by tiling), all ND:
//   x1           float16 [M, H]  residual carry-in (previous norm site's
//                                res_out or the embedding, ND)
//   x2           float16 [M, H]  sub-layer GEMM output (QBMV3 NZ output ->
//                                GE-inserted NZ->ND TransData feeds this port)
//   gamma        float16 [H]     rmsnorm weight
//   smooth_scale float16 [H]     optional per-channel smooth scale
// Outputs:
//   res_out      float16 [M, H]  r = x1 + x2, the next residual (feeds the
//                                next site's x1, ND direct)
//   y            int8    [M, H]  W8A8 activation (feeds QuantBatchMatmulV3;
//                                GE inserts the int8 ND->NZ TransData)
//   token_scale  float32 [M]     dequant scale per row, absmax/127 (matches
//                                the 310P DynamicQuant output contract and
//                                QBMV3's ND scale input)
//
// Port formats stay ND by measured verdict (2026-09-20, msprof task-level
// same-window A/B, dual-vendor methodology). The NZ registration variant
// (x1/x2/res_out/y FRACTAL_NZ, the gelu/rms_norm_nz precedent) was fully
// implemented and bit-identical-verified (snapshot bit-compare 6/6 shapes
// incl. all three outputs, 3x-repeat bit-identical 6/6, MTE3 <=4-descriptor
// gating zero copy loss), but is NET-NEGATIVE at every deployment shape:
// the NZ kernel's fractal gather/scatter I/O (mte2 87->569us, mte3
// 22->523us @1536x4096; scattered store ~30GB/s, the rms_norm_nz v2 law)
// costs +89..+556us more than the ND kernel, while the site only carries
// two removable TransData ops (~140-155us @768x4096 SP site: x2 fp16
// NZ->ND in + y int8 ND->NZ out) -- res_out has no ND counterpart to
// remove, so it pays the scatter cost for nothing. Net ~-95us/site
// @768x4096, -249us @1536x4096. The kernel and tiling keep their verified
// NZ branches (unreachable under ND-only registration); if the MTE3
// scatter bottleneck is ever solved (fractal-staged store), flipping the
// four GEMM-side ports back to FRACTAL_NZ re-enables the zero-TransData
// chain with no kernel change.
//
// Formula (report section 5.2):
//   r[m, j]   = x1[m, j] + x2[m, j]
//   rms[m]    = sqrt(mean_j(r[m, :]^2) + epsilon)
//   n[m, j]   = r[m, j] / rms[m] * gamma[j] * smooth_scale[j]
//   token_scale[m] = absmax_row(n) / 127
//   y[m, j]   = round_to_int8(n[m, j] * 127 / absmax_row(n))
class FusedAddRmsNormSmoothQuant : public OpDef {
 public:
  explicit FusedAddRmsNormSmoothQuant(const char *opName) : OpDef(opName) {
    // ND-only registration (see the verdict note above the port table): ND
    // is the measured optimum for this op -- the NZ variant's scattered
    // MTE3 store on res_out (no removable TransData counterpart) plus the
    // fractal gather overhead exceed the two site TransData ops it could
    // remove. The kernel/tiling NZ branches stay as verified dead code.
    auto &inX1 = this->Input("x1");
    inX1.ParamType(REQUIRED);
    inX1.DataType({ge::DT_FLOAT16});
    inX1.Format({ge::FORMAT_ND});
    inX1.UnknownShapeFormat({ge::FORMAT_ND});

    auto &inX2 = this->Input("x2");
    inX2.ParamType(REQUIRED);
    inX2.DataType({ge::DT_FLOAT16});
    inX2.Format({ge::FORMAT_ND});
    inX2.UnknownShapeFormat({ge::FORMAT_ND});

    auto &inGamma = this->Input("gamma");
    inGamma.ParamType(REQUIRED);
    inGamma.DataType({ge::DT_FLOAT16});
    inGamma.Format({ge::FORMAT_ND});
    inGamma.UnknownShapeFormat({ge::FORMAT_ND});

    auto &inSmoothScale = this->Input("smooth_scale");
    inSmoothScale.ParamType(OPTIONAL);
    inSmoothScale.DataType({ge::DT_FLOAT16});
    inSmoothScale.Format({ge::FORMAT_ND});
    inSmoothScale.UnknownShapeFormat({ge::FORMAT_ND});

    auto &outResOut = this->Output("res_out");
    outResOut.ParamType(REQUIRED);
    outResOut.DataType({ge::DT_FLOAT16});
    outResOut.Format({ge::FORMAT_ND});
    outResOut.UnknownShapeFormat({ge::FORMAT_ND});
    outResOut.AutoContiguous();

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

    this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);

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

OP_ADD(FusedAddRmsNormSmoothQuant);
}  // namespace ops
