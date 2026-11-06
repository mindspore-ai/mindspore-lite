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
 * ============================================================================
 */
// InnerBatchMatMul: hand-written Cube BatchMatMul custom operator for Atlas 300I Duo
// (Ascend 310P / dav_m200).
//
// Public contract: y = x1 @ x2 [+ bias]
//   - x1/x2 are declared FRACTAL_NZ (GE inserts TransData after an ND producer) and y is ND;
//   - bias is OPTIONAL (FRACTAL_NZ) and supports row broadcast [batch...,1,N] and full size
//     [batch...,M,N]. An ONNX Custom node always declares the three input names and marks bias
//     optional; when bias is absent the node passes only the first two input indexes and the
//     converter fills the missing slot with None, so GetOptionalInputShape(2) is empty,
//     biasMode=0 and the kernel receives nullptr;
//   - batch dims: right-aligned numpy broadcast (a 1 broadcasts);
//   - adj_x1/adj_x2 are not supported (declared INT and rejected by the tiling when non-zero);
//   - the Ascend 310P Cube only supports fp16, so only fp16 is registered (fp32 accumulation
//     happens inside the Cube).
#include "register/op_def_registry.h"

namespace ops {
class InnerBatchMatMul : public OpDef {
 public:
  explicit InnerBatchMatMul(const char *name) : OpDef(name) {
    this->Input("x1")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_FRACTAL_NZ})
      .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
    this->Input("x2")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_FRACTAL_NZ})
      .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
    this->Input("bias")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_FRACTAL_NZ})
      .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
    this->Output("y")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    // Transposed multiply is not supported and the tiling must be able to see and reject it.
    // ONNX has no BOOL attribute type, so adj_x*=1 arrives at the Custom node as INT; declaring
    // Bool here makes GetAttrPointer<bool> return nothing and the true value would be silently
    // treated as false (non-transposed). Declare INT and let the tiling fail on any non-zero value.
    this->Attr("adj_x1").AttrType(OPTIONAL).Int(0);
    this->Attr("adj_x2").AttrType(OPTIONAL).Int(0);
    // Tuning knobs: 0 means "pick automatically from the shape".
    this->Attr("base_m").AttrType(OPTIONAL).Int(0);
    this->Attr("base_n").AttrType(OPTIONAL).Int(0);

    // opbuild requires every operator to register at least one SoC, otherwise no kernel binary is generated.
    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(false)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .ExtendCfgInfo("softsync.flag", "true");
    this->AICore().AddConfig("ascend310p", aicConfig);
  }
};

OP_ADD(InnerBatchMatMul);
}  // namespace ops
