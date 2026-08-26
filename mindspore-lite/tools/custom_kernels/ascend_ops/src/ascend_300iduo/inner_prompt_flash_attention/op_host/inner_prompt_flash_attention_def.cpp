/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 * \file inner_prompt_flash_attention_def.cpp
 * \brief Enhanced PFA fusion operator for Ascend 310P (FP16-only).
 *
 * Compared to the CANN built-in PFA, this version supports:
 *   - attenMask when S1 != S2
 *   - GQA/MQA (numKeyValueHeads < numHeads) combined with S1 != S2 + mask
 */

#include "register/op_def_registry.h"

namespace ops {
class InnerPromptFlashAttention : public OpDef {
 public:
  explicit InnerPromptFlashAttention(const char *name) : OpDef(name) {
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
    this->Input("pse_shift")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("atten_mask")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_BOOL})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("actual_seq_lengths")
      .ParamType(OPTIONAL)
      .ValueDepend(OPTIONAL)
      .DataType({ge::DT_INT64})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("actual_seq_lengths_kv")
      .ParamType(OPTIONAL)
      .ValueDepend(OPTIONAL)
      .DataType({ge::DT_INT64})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("deq_scale1")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_UINT64})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("quant_scale1")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("deq_scale2")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_UINT64})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("quant_scale2")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("quant_offset2")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("attention_out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("num_heads").AttrType(REQUIRED).Int(1);
    this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
    this->Attr("pre_tokens").AttrType(OPTIONAL).Int(214748647);
    this->Attr("next_tokens").AttrType(OPTIONAL).Int(0);
    this->Attr("input_layout").AttrType(OPTIONAL).String("BSH");
    this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(0);
    this->Attr("sparse_mode").AttrType(OPTIONAL).Int(0);
    this->Attr("inner_precise").AttrType(OPTIONAL).Int(1);
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(true)
      .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
      .ExtendCfgInfo("opFile.value", "inner_prompt_flash_attention")
      .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
    this->AICore().AddConfig("ascend310p", aicore_config);
  }
};

OP_ADD(InnerPromptFlashAttention);
}  // namespace ops
