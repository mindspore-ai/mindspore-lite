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
 * \file fused_swi_glu_act_quant_nz.cpp
 * \brief OpDef registration for FusedSwiGLUActQuantNZ
 * (Atlas 300I Duo W8A8 LLM graphs).
 */

#include "register/op_def_registry.h"

namespace ops {
// FusedSwiGLUActQuantNZ: fused FFN activation + per-token dynamic quant for
// Atlas 300I Duo W8A8 LLM activation-quant fusion (pure-vector AICore kernel).
//
// Replaces the unfused per-layer chain
//   ConcatD(434us) + SwiGlu(600us) + Mul-smooth(190us) + DynamicQuant(373us)
//   + TransData(123us) ~= 1718us
// with one pure-vector kernel that reads the two QuantBatchMatmulV3 outputs
// directly, so the 12288-wide concat intermediate and the TransData re-layout
// never exist.
//
// Inputs (leading dims are flattened into M by tiling). The GEMM-side ports
// are FRACTAL_NZ-native (NZ-only registration, the gelu/rms_norm_nz
// precedent -- CANN 8.5.0 opbuild does not support multi-value Format lists
// per port, so every port declares exactly one format):
//   gate         float16 [M, N]  FRACTAL_NZ  gate branch (first QBMV3 output,
//                                N 即 inter 维; NZ direct-read removes the
//                                NZ->ND TransData on both GEMM outputs)
//   up           float16 [M, N]  FRACTAL_NZ  up branch (second QBMV3 output)
//   smooth_scale float16 [N]     ND          optional per-channel smooth scale
// Outputs:
//   y            int8   [M, N]  FRACTAL_NZ  W8A8 activation (feeds
//                                QuantBatchMatmulV3 x1 in its native layout,
//                                removing the int8 ND->NZ TransData)
//   token_scale  float32 [M]     ND          dequant scale per row, absmax/127
//                                (matches the 310P DynamicQuant output
//                                contract and QBMV3's ND scale input)
//
// Formula (report section 5.1; same gate/up roles as the CANN built-in
// SwiGlu, i.e. silu on the FIRST/gate branch):
//   act[m, j] = silu(gate[m, j]) * up[m, j] * smooth_scale[j]
//   token_scale[m] = absmax_row(act) / 127
//   y[m, j] = round_to_int8(act[m, j] * 127 / absmax_row(act))
//
// Measured verdict (2026-09-20, msprof task-level dual-vendor same-window
// A/B, medians n=7): ADOPT the NZ registration. Precision is bit-identical
// 6/6 shapes (snapshot bit-compare + 3x repeat; MTE3 <=4-descriptor gating
// held, zero copy loss — NZ is pure data movement, the UB compute is
// unchanged). The NZ kernel is slower than the ND kernel (995.3 vs 520.1us
// @1536x6144; 1817.2 vs 1280.7us @1768x12288 — fractal gather/scatter on
// 32B segments), but the ND registration carries three site TransData ops
// that NZ eliminates (gate/up NZ->ND fp16 in + y int8 ND->NZ out: 730us
// @1536x6144, 1382us @1768x12288): net +255us/site (+845 @1768x12288),
// and ~1229us/site vs the unfused chain incl. its GEMM-epilogue
// TransData. In-kernel I/O overlaps the vector pipe; separate TransData
// graph ops serialize — that asymmetry is the win.
class FusedSwiGLUActQuantNZ : public OpDef {
 public:
  explicit FusedSwiGLUActQuantNZ(const char *opName) : OpDef(opName) {
    // NZ-only registration on the GEMM-side ports (gelu/rms_norm_nz
    // precedent): CANN 8.5.0's opbuild CfgDtypeFormatCheck rejects
    // multi-value Format lists per port, so each port declares exactly one
    // format. At the deployment site (QBMV3 -> swiglu -> QBMV3, all NZ on
    // those edges) the op connects with zero TransData; ND producers
    // connect through a GE-inserted TransData (bench graphs with ND
    // placeholder inputs show this artifact — judge the kernel by msprof
    // task time, not end-to-end bench time). The kernel keeps an ND-storage
    // branch as defense (unreachable under NZ-only registration except via
    // legacy pre-NZ tilings).
    auto &inGate = this->Input("gate");
    inGate.ParamType(REQUIRED);
    inGate.DataType({ge::DT_FLOAT16});
    inGate.Format({ge::FORMAT_FRACTAL_NZ});
    inGate.UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});

    auto &inUp = this->Input("up");
    inUp.ParamType(REQUIRED);
    inUp.DataType({ge::DT_FLOAT16});
    inUp.Format({ge::FORMAT_FRACTAL_NZ});
    inUp.UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});

    auto &inSmoothScale = this->Input("smooth_scale");
    inSmoothScale.ParamType(OPTIONAL);
    inSmoothScale.DataType({ge::DT_FLOAT16});
    inSmoothScale.Format({ge::FORMAT_ND});
    inSmoothScale.UnknownShapeFormat({ge::FORMAT_ND});

    auto &outY = this->Output("y");
    outY.ParamType(REQUIRED);
    outY.DataType({ge::DT_INT8});
    outY.Format({ge::FORMAT_FRACTAL_NZ});
    outY.UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
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

OP_ADD(FusedSwiGLUActQuantNZ);
}  // namespace ops
