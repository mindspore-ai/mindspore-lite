/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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
#include "src/litert/delegate/coreml/pass/coreml_trans_extend_pass.h"
#include <algorithm>
#include <set>
#include <string>
#include "src/litert/delegate/coreml/pass/coreml_pass_utils.h"
#include "src/litert/delegate/fusion_pass_utils.h"

// Adapter for CoreML Pass Utils to work with Transform Pass templates
struct CoreMLTransformPassUtils {
  static bool IsNchw2Nhwc(CoreMLOp *op) { return CoreMLPassUtils::IsNchw2Nhwc(op); }
  static bool IsNhwc2Nchw(CoreMLOp *op) { return CoreMLPassUtils::IsNhwc2Nchw(op); }
  static std::vector<mindspore::MSTensor> GetNonConstInputs(CoreMLOp *op) {
    return CoreMLPassUtils::GetNonConstInputs(op);
  }
  static CoreMLOp *OpInputFromOp(CoreMLOp *op, const mindspore::MSTensor &tensor) {
    return CoreMLPassUtils::OpInputFromOp(op, tensor);
  }
  static CoreMLOp *CreateNhwc2NchwOp(const std::vector<mindspore::MSTensor> &inputs,
                                     const std::vector<mindspore::MSTensor> &outputs, const std::string &name) {
    return CoreMLPassUtils::CreateNhwc2NchwOp(inputs, outputs, name);
  }
  static CoreMLOp *CreateNchw2NhwcOp(const std::vector<mindspore::MSTensor> &inputs,
                                     const std::vector<mindspore::MSTensor> &outputs, const std::string &name) {
    return CoreMLPassUtils::CreateNchw2NhwcOp(inputs, outputs, name);
  }
  static void UpdateOp(CoreMLOp *op, const std::vector<CoreMLOp *> &in_ops, const std::vector<CoreMLOp *> &out_ops,
                       const std::vector<mindspore::MSTensor> &inputs,
                       const std::vector<mindspore::MSTensor> &outputs) {
    return CoreMLPassUtils::UpdateOp(op, in_ops, out_ops, inputs, outputs);
  }
  static void UpdateNH2NCTransNodePreOp(CoreMLOp *cur_op, CoreMLOp *nh2nc_op, CoreMLOp *post_op) {
    return CoreMLPassUtils::UpdateNH2NCTransNodePreOp(cur_op, nh2nc_op, post_op);
  }
  static void UpdateNC2NHTransNodePostOp(CoreMLOp *cur_op, CoreMLOp *nc2nh_op, CoreMLOp *post_op,
                                         const mindspore::MSTensor &trans_in_tensor) {
    return CoreMLPassUtils::UpdateNC2NHTransNodePostOp(cur_op, nc2nh_op, post_op, trans_in_tensor);
  }
};

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
std::set<mindspore::schema::PrimitiveType> format_depend_nodes = {
  schema::PrimitiveType_Conv2DFusion,  schema::PrimitiveType_Conv2dTransposeFusion,
  schema::PrimitiveType_MaxPoolFusion, schema::PrimitiveType_AvgPoolFusion,
  schema::PrimitiveType_CropAndResize, schema::PrimitiveType_InstanceNorm,
  schema::PrimitiveType_ArgMaxFusion,  schema::PrimitiveType_FullConnection,
  schema::PrimitiveType_ScaleFusion,   schema::PrimitiveType_ExpandDims,
  schema::PrimitiveType_Unsqueeze,     schema::PrimitiveType_SliceFusion,
  schema::PrimitiveType_BroadcastTo,   schema::PrimitiveType_TileFusion,
  schema::PrimitiveType_Resize,        schema::PrimitiveType_MatMulFusion,
  schema::PrimitiveType_Gather,        schema::PrimitiveType_Gather,
  schema::PrimitiveType_Squeeze,       schema::PrimitiveType_Reshape,
  schema::PrimitiveType_Unsqueeze,     schema::PrimitiveType_Transpose,
};

// this pass goal is to minimize subgraphs generated
// by inserting nchw2nhwc or nhwc2nchw before or after the operator (e.g. concat, add, etc..) together with
// fusion pass. If transpose inserted are more than half of input output, we will insert remaining input
// output with transpose and hopefully do a fusion pass. Otherwise, we don't insert anything.

// Typically concat accept output from nchw2nhwc, we fill other input with nh2nc and nc2nh so that inputs to concat are
// format same and then fusion all nchw2nhwc op.
// e.g.
// original     (conv->nchw2nhwc, add(format nhwc)) -> concat-> (nhwc2nchw->conv)
// current pass (conv->nchw2nhwc, add->nhwc2nchw->nchw2nhwc) -> concat -> (nhwc2nchw->conv)
// fusion pass  (conv, add->nhwc2nchw) -> concat -> conv
// original 2 cpusubgraph, after 2 pass, only 1 cpu subgraph

// Such ops require inputs all have same format, could be nchw or nhwc or other format.
// Their inputs outputs may not be 4d, or are already format ok,
// so we won't insert nc2nh or nh2nc when op's in ops and out ops contains no nc2nh or nh2nc.
// This pass should be run after npu_transform_pass, which insert transpose for nchw-input-limited op like conv2d.
// Note: GetInsertState, InsertTransNode, InsertPreNodes, InsertPostNodes are now provided by
// delegate::TransformPassRun template function via CoreMLTransformPassUtils adapter.

// Note: InsertTransNode, InsertPreNodes, InsertPostNodes methods are now provided by
// delegate::TransformPassRun template function via CoreMLTransformPassUtils adapter.

int CoreMLTransExtendPass::Run(CoreMLGraph *subgraph) {
  return delegate::TransformPassRun<CoreMLOp, CoreMLGraph, CoreMLTransformPassUtils>(subgraph, format_depend_nodes,
                                                                                     name_);
}
}  // namespace mindspore::lite
