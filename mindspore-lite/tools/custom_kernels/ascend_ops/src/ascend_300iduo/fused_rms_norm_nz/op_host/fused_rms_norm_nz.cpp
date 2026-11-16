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
// FusedRmsNormNZ: single-kernel RMSNorm for Atlas 300I Duo: fp16 boundary
// + fp32 statistics, FRACTAL_NZ-native.
//
// Inputs/outputs are FRACTAL_NZ-native (NZ-only registration, the gelu
// precedent -- CANN 8.5.0 opbuild does not support multi-value Format lists
// per port): the deployment site (GEMM epilogue -> norm -> GEMM) is all NZ,
// so the op connects with zero TransData. ND producers connect through a
// GE-inserted TransData; the tiling detects the storage layout and the
// kernel keeps a flat-ND branch as defense.
//
// Inputs:
//   x     float16 [M..., N]  FRACTAL_NZ (or ND); upstream GEMM output
//   gamma float16 ND [N]     per-channel scale
// Outputs:
//   y     float16 [M..., N]  same shape and format as x
// Attr:
//   epsilon float, optional, default 1e-6
//
// Formula (per row; statistics accumulated in fp32 inside UB -- the fp32
// red line -- with a single fp16 rounding at the output,
// the same single-rounding semantics the chain has):
//   y = x * rsqrt(mean(x^2) + epsilon) * gamma
class FusedRmsNormNZ : public OpDef {
 public:
  explicit FusedRmsNormNZ(const char *name) : OpDef(name) {
    // NZ-only registration (the gelu precedent): CANN 8.5.0's opbuild
    // CfgDtypeFormatCheck rejects multi-value Format lists on a port (it
    // drops the dtype/format keys from the generated ops-info and the whole
    // vendor build breaks downstream) -- every port declares exactly one
    // format. ND producers connect through a GE-inserted TransData; at the
    // deployment site (GEMM epilogue, all NZ) the op connects with no
    // TransData at all. The kernel keeps an ND-storage branch as defense.
    this->Input("x")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_FRACTAL_NZ})
      .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
    this->Input("gamma")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("y")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_FRACTAL_NZ})
      .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ})
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

OP_ADD(FusedRmsNormNZ);
}  // namespace ops
