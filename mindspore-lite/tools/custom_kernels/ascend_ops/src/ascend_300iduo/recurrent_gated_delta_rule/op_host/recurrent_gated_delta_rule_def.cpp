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
class RecurrentGatedDeltaRule : public OpDef {
 public:
  explicit RecurrentGatedDeltaRule(const char *name) : OpDef(name) {
    this->Input("query")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("value")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("beta")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("state")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("actual_seq_lengths")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("ssm_state_indices")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("g")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gk")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("num_accepted_tokens")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_INT32})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Output("state")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);

    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .ExtendCfgInfo("softsync.flag", "true");
    this->AICore().AddConfig("ascend310p", aicConfig);
  }
};

OP_ADD(RecurrentGatedDeltaRule);
}  // namespace ops
