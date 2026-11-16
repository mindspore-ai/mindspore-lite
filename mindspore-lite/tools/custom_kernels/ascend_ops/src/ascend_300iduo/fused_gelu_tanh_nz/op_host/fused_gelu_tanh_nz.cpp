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
// FusedGeluTanhNZ: single-kernel gelu_pytorch_tanh for Atlas 300I Duo.
//
// Inputs/outputs are registered FRACTAL_NZ only (single-format registration):
// the deployment site (BatchMatMul -> activation -> BatchMatMul) is all NZ, so
// the op connects without any TransData. A single format is also a hard build
// constraint of this repo's CANN 8.5.0 toolchain -- a Format list (e.g.
// {FRACTAL_NZ, ND}) breaks BOTH the aclnn autogen (it skips the
// socSupportInfo0 declaration in autogen/aclnn_<op>.cpp -> compile error) and
// the kernel opbuild (two parallel ascendc_compile_kernel.py invocations race
// on the same working dir -> FileExistsError); every other op in this repo
// registers exactly one format. In an ND graph GE simply inserts a TransData
// on each side (layout-only, bit-preserving), so ND single-op benches still
// work -- they just measure op + 2 TransData (a conservative lower bound; the
// kernel itself is a flat elementwise walk, see the tiling source).
//
// Inputs:
//   x  float16 [M..., N]  FRACTAL_NZ; upstream output
// Outputs:
//   y  float16 [M..., N]  FRACTAL_NZ, same shape as x; downstream input
//
// Formula (exp reformulation of gelu_pytorch_tanh, math-identical to
// 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) via tanh(y)=2/(1+e^(-2y))-1;
// same recipe as the CANN GeluV2(approximate="tanh") TBE source and
// gelu_quant_base.h::ComputeGeluTanh):
//   y = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3)))
// computed in fp32 inside the kernel (the chain's fp32 legs equivalent),
// with a single fp16 rounding at the output vs the chain's three.
class FusedGeluTanhNZ : public OpDef {
 public:
  explicit FusedGeluTanhNZ(const char *name) : OpDef(name) {
    // x is AutoContiguous like y: the kernel walks the storage as one flat
    // elementwise stream (see the tiling source), so the op must never be
    // handed a non-contiguous input view -- let GE materialize a contiguous
    // operand instead of silently reading the wrong flat range.
    this->Input("x")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_FRACTAL_NZ})
      .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ})
      .AutoContiguous();
    this->Output("y")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_FRACTAL_NZ})
      .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ})
      .AutoContiguous();

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

OP_ADD(FusedGeluTanhNZ);
}  // namespace ops
