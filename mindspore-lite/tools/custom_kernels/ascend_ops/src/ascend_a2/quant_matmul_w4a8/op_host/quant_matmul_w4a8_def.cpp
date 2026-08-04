/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class QuantMatmulW4a8 : public OpDef {
 public:
  explicit QuantMatmulW4a8(const char *name) : OpDef(name) {
    this->Input("act")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT8})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("weight")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("scale")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("bias")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("x_scale")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("output_bias")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_BF16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false);
    this->AICore().AddConfig("ascend910b", aicConfig);
  }
};
OP_ADD(QuantMatmulW4a8);
}  // namespace ops
