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

/* fused_causal_conv_silu_def.cpp Op prototype for FusedCausalConvSilu (ascend910b). */

#include "register/op_def_registry.h"

namespace ops {
class FusedCausalConvSilu : public OpDef {
 public:
  explicit FusedCausalConvSilu(const char *name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    this->Input("seq_len").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});

    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Output("conv_state").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();

    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false);

    this->AICore().AddConfig("ascend910b", aicoreConfig);
  }
};
OP_ADD(FusedCausalConvSilu);
}  // namespace ops
