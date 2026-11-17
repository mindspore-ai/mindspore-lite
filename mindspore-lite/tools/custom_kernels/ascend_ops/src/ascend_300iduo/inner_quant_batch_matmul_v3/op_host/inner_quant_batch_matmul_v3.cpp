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
 * \file inner_quant_batch_matmul_v3.cpp
 * \brief OpDef registration for InnerQuantBatchMatmulV3
 * (Atlas 300I Duo W8A8 LLM graphs).
 */

#include "register/op_def_registry.h"

namespace ops {
// InnerQuantBatchMatmulV3: W8A8 quantized GEMM with per-token output
// dequant for Atlas 300I Duo (Ascend310P3) LLM graphs, drop-in for the
// CANN built-in QuantBatchMatmulV3 NZ_NZ int8 combo.
//
// Why this op exists:
// the built-in 310P registration matrix has NO NZ_NZ combo on CANN 8.5
// (the 4 registered combos all take x1 ND; the NZ-native producer chain
// SwiGLu-act-quant -> GEMM needs two TransData hops per site), and on
// 9.2 the NZ_NZ combo dispatches the arch20 raw-CCE kernel whose host
// tiling underfills L1 for int8 (k0 = 131072/(m0+n0) elements, i.e. it
// budgets A-ping and B-ping as one shared pool, while the kernel
// allocates four separate 131072-element L1 buffers -- per-buffer budget
// is 131072/max(m0,n0)). This op vendors that exact kernel (bit-identical
// math) with a shape-specialized tiling that doubles k0 for the
// suffix_1536 W8A8 shapes (k-part round trips halved).
//
// Contract (mirrors CANN 9.2 combo 5 exactly; validated in tiling):
//   x1 int8 FRACTAL_NZ [.., M, K]  REQUIRED  activations, transpose_x1=false
//   x2 int8 FRACTAL_NZ [.., N, K]  REQUIRED  W8 weights,   transpose_x2=true
//   scale      float  ND [N]       REQUIRED  per-channel weight dequant
//   offset     float  ND [N]       OPTIONAL  accepted, ignored (the
//                                built-in per-token path ignores it too;
//                                when offset is present the built-in also
//                                drops bias -- mirrored here)
//   bias       int32 ND [N]        OPTIONAL  int32 bias is added in the
//                                int32 L0C domain (exact beyond fp24);
//                                the built-in also accepts float bias,
//                                but a two-value DataType list on ANY
//                                port makes CANN 9.2 op_build drop the
//                                whole op's dtype/format info (see the
//                                attr note below), so this def admits
//                                int32 only -- the W8A8 deployment
//                                recipes (ascend-style quant bias) are
//                                int32 anyway
//   pertoken_scale float ND [M]    OPTIONAL  per-row activation dequant;
//                                when present the offset term is dropped
//                                (aclnnQuantMatmulWeightNz formula set)
//   y          float16 FRACTAL_NZ [.., M, N] REQUIRED
//
// out = cast_fp16( cast_fp32(acc_i32 [+ bias_i32]) * scale[n]
//                  * pertoken_scale[m] )
//
// Ports are NZ-only (gelu/rms_norm_nz precedent: CANN 8.5.0 opbuild
// rejects multi-value Format lists), so the deployment chain
// FusedSwiGLuActQuantNZ -> InnerQuantBatchMatmulV3 connects with zero
// TransData; ND producers connect through a GE-inserted TransData.
class InnerQuantBatchMatmulV3 : public OpDef {
 public:
  explicit InnerQuantBatchMatmulV3(const char *opName) : OpDef(opName) {
    auto &inX1 = this->Input("x1");
    inX1.ParamType(REQUIRED);
    inX1.DataType({ge::DT_INT8});
    inX1.Format({ge::FORMAT_FRACTAL_NZ});
    inX1.UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});

    auto &inX2 = this->Input("x2");
    inX2.ParamType(REQUIRED);
    inX2.DataType({ge::DT_INT8});
    inX2.Format({ge::FORMAT_FRACTAL_NZ});
    inX2.UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});

    auto &inScale = this->Input("scale");
    inScale.ParamType(REQUIRED);
    inScale.DataType({ge::DT_FLOAT});
    inScale.Format({ge::FORMAT_ND});
    inScale.UnknownShapeFormat({ge::FORMAT_ND});

    auto &inOffset = this->Input("offset");
    inOffset.ParamType(OPTIONAL);
    inOffset.DataType({ge::DT_FLOAT});
    inOffset.Format({ge::FORMAT_ND});
    inOffset.UnknownShapeFormat({ge::FORMAT_ND});

    auto &inBias = this->Input("bias");
    inBias.ParamType(OPTIONAL);
    inBias.DataType({ge::DT_INT32});
    inBias.Format({ge::FORMAT_ND});
    inBias.UnknownShapeFormat({ge::FORMAT_ND});

    auto &inPertokenScale = this->Input("pertoken_scale");
    inPertokenScale.ParamType(OPTIONAL);
    inPertokenScale.DataType({ge::DT_FLOAT});
    inPertokenScale.Format({ge::FORMAT_ND});
    inPertokenScale.UnknownShapeFormat({ge::FORMAT_ND});

    auto &outY = this->Output("y");
    outY.ParamType(REQUIRED);
    outY.DataType({ge::DT_FLOAT16});
    outY.Format({ge::FORMAT_FRACTAL_NZ});
    outY.UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
    outY.AutoContiguous();

    // Attrs mirror the built-in's contract, with two deliberate deviations
    // forced by the CANN 9.2 op_build ops-info generator:
    //  1. bias is int32-only (see the contract note above): a two-value
    //     DataType list on any port makes the generator resolve dtype/format
    //     for ALL ports of the op to empty ("The dtype size of input[0] ...
    //     is 0"), which then breaks the autogen aclnn wrapper (undeclared
    //     socSupportInfo0) and the kernel bin-param builder (IndexError).
    //  2. the built-in's "dtype" attr is renamed quant_dtype. Kept renamed
    //     out of caution for the same generator (the original failure was
    //     observed with an attr literally named "dtype"; the rename predates
    //     finding the real trigger and single-dtype ports now generate
    //     correctly, so the name itself was NOT the trigger -- but the
    //     rename is harmless and stays: the tiling reads the attr by
    //     declaration index, and the only emitter, the bench script's Custom
    //     node, uses the same name).
    //
    // transpose_x1/transpose_x2 are declared Int (0/1) instead of Bool: INT
    // attrs are the proven-lossless type through the mslite ONNX-Custom ->
    // mindspore schema -> ge NamedAttrs chain (every repo emitter's *_i attr
    // reaches op_host intact), while Bool has no in-repo precedent and the
    // CANN ONNX plugin itself encodes these two attrs as ints (i == 1 ->
    // true). Values validated in tiling: quant_dtype must be 1
    // (ge::DT_FLOAT16; y is fixed fp16 by the def), transpose_x1 must be 0,
    // transpose_x2 must be 1, group_size must be 0.
    this->Attr("quant_dtype").AttrType(REQUIRED).Int();
    this->Attr("transpose_x1").AttrType(OPTIONAL).Int(0);
    this->Attr("transpose_x2").AttrType(OPTIONAL).Int(0);
    this->Attr("group_size").AttrType(OPTIONAL).Int(0);

    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true);
    aicConfig.DynamicShapeSupportFlag(true);
    aicConfig.NeedCheckSupportFlag(false);
    aicConfig.DynamicFormatFlag(true);
    aicConfig.DynamicRankSupportFlag(true);
    aicConfig.ExtendCfgInfo("softsync.flag", "true");

    this->AICore().AddConfig("ascend310p", aicConfig);
  }
};

OP_ADD(InnerQuantBatchMatmulV3);
}  // namespace ops
