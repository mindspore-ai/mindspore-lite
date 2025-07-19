/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/converter/graph_kernel_expander_lite.h"

#include "mindspore/ops/op_def/conv_pool_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "backend/common/graph_kernel/model/node.h"
#include "backend/common/graph_kernel/model/op_node.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "utils/anf_utils.h"
#include "tools/graph_kernel/converter/basic_op_infer_shape.h"
#include "utils/ms_context.h"
#include "tools/graph_kernel/converter/preprocess_weight.h"
#include "tools/graph_kernel/common/utils.h"
#include "utils/check_convert_utils.h"
#include "common/kernel_build_info.h"
#include "include/backend/kernel_info.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"

namespace mindspore::graphkernel {
AnfNodePtr FixFormatDeco::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  std::vector<std::string> format = GetFixedFormat(node);
  auto current_kernel_build_info = GetKernelInfo(node);
  if (current_kernel_build_info == nullptr) {
    MS_LOG(ERROR) << "Kernel info from " << cnode->fullname_with_scope() << "is nullptr";
    return nullptr;
  }
  auto ori_format = current_kernel_build_info->GetAllOutputFormats();
  current_kernel_build_info->SetOutputsFormat(format);
  auto ret = decorated_->Run(cnode);
  if (ret == nullptr) {
    return nullptr;
  }
  auto fg = GetCNodeFuncGraph(ret);
  for (auto sub_cnode : fg->GetOrderedCnodes()) {
    SetAnfKernelInfoFormatFromAToB(node, sub_cnode, ori_format);
  }
  auto ret_node = ret->cast<CNodePtr>();
  SetAnfKernelInfoFormatFromAToB(node, ret_node, ori_format);
  return ret;
}

std::vector<std::string> FixFormatDeco::GetFixedFormat(const AnfNodePtr &node) const {
  auto cnode = node->cast<CNodePtr>();
  auto out_num = AnfUtils::GetOutputTensorNum(cnode);
  std::vector<std::string> format(out_num, kOpFormat_DEFAULT);
  return format;
}

std::vector<std::string> UseInputFormatDeco::GetFixedFormat(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<std::string> format;
  for (size_t i = 1; i < cnode->size(); i++) {
    if (cnode->input(i)->isa<CNode>()) {
      auto kernel_with_index = AnfUtils::VisitKernel(cnode->input(i), 0);
      auto input_cnode = kernel_with_index.first->cast<CNodePtr>();
      auto input_kernel_build_info = GetKernelInfo(input_cnode);
      if (input_cnode != nullptr && input_kernel_build_info != nullptr) {
        auto input_format = input_kernel_build_info->GetOutputFormat(kernel_with_index.second);
        format.push_back(input_format);
        break;
      }
    }
  }
  if (format.empty()) {
    format.push_back(kOpFormat_DEFAULT);
  }
  return format;
}

AnfNodePtr InferValueDeco::Run(const AnfNodePtr &node) {
  // operators must infer value
  std::unordered_set<std::string> akg_exclude_nodes = {prim::kPrimGather->name(), prim::kPrimShape->name(),
                                                       prim::kPrimConcat->name(), prim::kPrimConstantOfShape->name(),
                                                       "StridedSliceOnnx"};
  auto cnode = QuickCloneCNode(node);
  auto ret = decorated_->Run(cnode);
  if (ret == nullptr) {
    return nullptr;
  }
  auto fg = GetCNodeFuncGraph(ret);
  AnfNodePtrList inputs = ret->cast<CNodePtr>()->inputs();
  inner::LiteGraphPtr litegraph = GkUtils::AnfGraph2LiteGraph(fg);
  auto ops_list = litegraph->GetOrderedNodes();
  auto iter = ops_list.begin();
  while (iter != ops_list.end()) {
    auto this_op = std::static_pointer_cast<inner::PrimOp>(*iter);
    auto value = this_op->InferValue(this_op->inputs(), this_op->attrs());
    if (value != nullptr) {
      (*iter)->ReplaceWith(value);
      ops_list = litegraph->GetOrderedNodes();
      iter = ops_list.begin();
    } else {
      ++iter;
    }
  }
  auto &outputs = litegraph->GetOutputs();
  std::vector<AnfNodePtr> output_const;
  for (auto &output : outputs) {
    if (output->NodeType() == inner::NType::Tensor) {
      auto value = std::static_pointer_cast<inner::ConstTensorNode>(outputs[0])->data();
      auto valuenode = NewValueNode(value);
      valuenode->set_abstract(value->ToAbstract());
      (void)output_const.emplace_back(valuenode);
    }
  }
  if (outputs.size() == output_const.size()) {
    return node->func_graph()->NewCNode(output_const);
  }
  bool cannot_expand = std::any_of(ops_list.begin(), ops_list.end(), [&akg_exclude_nodes](const inner::NodePtr &node) {
    return akg_exclude_nodes.count(std::static_pointer_cast<inner::PrimOp>(node)->op()) > 0;
  });
  if (cannot_expand) {
    return nullptr;
  } else {
    auto new_fg = GkUtils::LiteGraph2AnfGraph(litegraph, Callback::Instance());
    (void)ConvertTensorToParameter(new_fg, &inputs);
    AnfNodePtrList new_inputs = {NewValueNode(new_fg)};
    (void)new_inputs.insert(new_inputs.end(), inputs.cbegin() + 1, inputs.cend());
    return node->func_graph()->NewCNode(new_inputs);
  }
}

AnfNodePtr PoolLayoutDeco::Run(const AnfNodePtr &node) {
  MS_CHECK_TRUE_MSG(node != nullptr, nullptr, "node is a nullptr.");
  auto cnode = QuickCloneCNode(node);
  MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "cnode is a nullptr.");
  auto prev_node = AnfUtils::VisitKernel(node->cast<CNodePtr>()->input(1), 0).first;
  if (prev_node != nullptr) {
    auto sub_graph = GetCNodeFuncGraph(prev_node);
    if (sub_graph != nullptr) {
      auto sub_nodes = TopoSort(sub_graph->get_return());
      for (auto sub_node : sub_nodes) {
        if (IsPrimitiveCNode(sub_node, prim::kPrimConv2D)) {
          auto prim = GetCNodePrimitive(sub_node);
          MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is a nullptr.");
          AnfUtils::SetNodeAttr("layout_axis", prim->GetAttr("weight_coi"), cnode);
          break;
        }
      }
    }
  }
  return decorated_->Run(cnode);
}

std::vector<PrimitivePtr> GraphKernelExpanderLite::ConvTuningExpanderOps() {
  std::vector<PrimitivePtr> conv_tuning_ops = {prim::kPrimConv2DFusion, prim::kPrimAvgPoolFusion,
                                               prim::kPrimMaxPoolFusion};
  return conv_tuning_ops;
}

bool GraphKernelExpanderLite::DisableConvTuning() {
  const auto &flags = GraphKernelFlags::GetInstance();
  auto flag_enable_only_list = flags.enable_expand_ops_only;
  auto flag_disable_list = flags.disable_expand_ops;
  return std::find(flag_disable_list.begin(), flag_disable_list.end(), prim::kPrimConv2DFusion->name()) !=
           flag_disable_list.end() ||
         (!flag_enable_only_list.empty() &&
          std::find(flag_enable_only_list.begin(), flag_enable_only_list.end(), prim::kPrimConv2DFusion->name()) ==
            flag_enable_only_list.end()) ||
         !flags.enable_lite_conv_tuning;
}

std::vector<PrimitivePtr> GraphKernelExpanderLite::InitOpList() {
  std::vector<OpWithLevel> expand_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimGeLU},
    {kAllTarget, OpLevel_0, prim::kPrimSquare},
    {kAllTarget, OpLevel_0, prim::kPrimSquaredDifference},
    {kAllTarget, OpLevel_0, prim::kPrimTile},
    // ascend device
    {kAscendDevice, OpLevel_0, prim::kPrimReduceMean},
    {kAscendDevice, OpLevel_0, prim::kPrimTile},
    {kAscendDevice, OpLevel_1, prim::kPrimLayerNorm},
    {kAscendDevice, OpLevel_0, prim::kPrimSigmoidCrossEntropyWithLogits},
    {kAscendDevice, OpLevel_0, prim::kPrimSquaredDifference},
    {kAscendDevice, OpLevel_0, prim::kPrimSquareSumAll},
    {kAscendDevice, OpLevel_1, prim::kPrimSoftsign},
    // cpu device
    {kCPUDevice, OpLevel_1, prim::kPrimExpandDims},
    {kCPUDevice, OpLevel_1, prim::kPrimSqueeze},
    {kCPUDevice, OpLevel_1, prim::kPrimTranspose},
    {kCPUDevice, OpLevel_1, prim::kPrimReshape},
    {kCPUDevice, OpLevel_1, prim::kPrimGather},
    {kCPUDevice, OpLevel_1, prim::kPrimShape},
    {kCPUDevice, OpLevel_1, prim::kPrimConcat},
    {kCPUDevice, OpLevel_1, prim::kPrimFusedBatchNorm},
    {kCPUDevice, OpLevel_1, prim::kPrimSoftmax},
    {kCPUDevice, OpLevel_0, prim::kPrimAddFusion},
    {kCPUDevice, OpLevel_0, prim::kPrimMulFusion},
    {kCPUDevice, OpLevel_0, prim::kPrimSubFusion},
    {kCPUDevice, OpLevel_1, prim::kPrimReduceFusion},
    {kCPUDevice, OpLevel_0, prim::kPrimActivation},
    {kCPUDevice, OpLevel_0, prim::kPrimDivFusion},
    {kCPUDevice, OpLevel_0, prim::kPrimExpFusion},
    {kCPUDevice, OpLevel_1, prim::kPrimUnsqueeze},
    {kCPUDevice, OpLevel_1, prim::kPrimConstantOfShape},
    {kCPUDevice, OpLevel_1, prim::kPrimLayerNormFusion},
    {kCPUDevice, OpLevel_1, prim::kPrimInstanceNorm},
    {kCPUDevice, OpLevel_1, prim::kPrimStridedSlice},
    {kCPUDevice, OpLevel_1, prim::kPrimScaleFusion}};
  const auto &flags = GraphKernelFlags::GetInstance();
  auto valid_op_list = GkUtils::GetValidOps(expand_ops_with_level, flags.fusion_ops_level, flags.enable_expand_ops_only,
                                            flags.enable_expand_ops, flags.disable_expand_ops);
  return valid_op_list;
}

bool GraphKernelExpanderLite::CanExpand(const CNodePtr &node) const {
  if (!GraphKernelExpander::CanExpand(node)) {
    return false;
  }
  auto cb = Callback::Instance();
  for (size_t i = 0; i < node->size() - 1; i++) {
    if (!node->input(i + 1)->isa<Parameter>() && !node->input(i + 1)->isa<ValueNode>() &&
        cb->GetInputShape(node, i).size() == 0) {
      MS_LOG(INFO) << "cnode with no input info can not expand now, node is " << node->fullname_with_scope();
      return false;
    }
  }
  return true;
}

ExpanderPtr GraphKernelExpanderLite::InitExpander(const AnfNodePtr &node) {
  auto expander = std::make_shared<LitegraphExpander>(Callback::Instance());
  ExpanderCreatorFuncList decos = {InferValueDeco::Creator};
  std::map<std::string, ExpanderCreatorFuncList> creators = {
    {prim::kPrimReduceFusion->name(), {DependValueDeco::GetCreator({1}), FixFormatDeco::Creator}},
    {prim::kPrimExpandDims->name(), {{DependValueDeco::GetCreator({1})}, FixFormatDeco::Creator}},
    {prim::kPrimUnsqueeze->name(), {FixFormatDeco::Creator}},
    {prim::kPrimSqueeze->name(), {FixFormatDeco::Creator}},
    {prim::kPrimShape->name(), {FixFormatDeco::Creator}},
    {prim::kPrimReshape->name(), {DependValueDeco::GetCreator({1}), FixFormatDeco::Creator}},
    {prim::kPrimConstantOfShape->name(), {DependValueDeco::GetCreator({0}), FixFormatDeco::Creator}},
    {prim::kPrimTranspose->name(), {DependValueDeco::GetCreator({1})}},
    {prim::kPrimGather->name(), {DependValueDeco::GetCreator({2}), FixFormatDeco::Creator}},
    {prim::kPrimReduceMean->name(), {DependValueDeco::GetCreator({1}), FixFormatDeco::Creator}},
    {prim::kPrimConcat->name(), {FixFormatDeco::Creator}},
    {prim::kPrimStridedSlice->name(), {FixFormatDeco::Creator}},
    {prim::kPrimMatMulFusion->name(), {MatmulPackB::Creator}},
    {prim::kPrimTile->name(), {{DependValueDeco::GetCreator({1})}, UseInputFormatDeco::Creator}},
  };
  auto prim = GetCNodePrimitive(node);
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is a nullptr.");
  auto iter = creators.find(prim->name());
  if (iter != creators.end()) {
    (void)decos.insert(decos.end(), iter->second.begin(), iter->second.end());
  }
  return WrapExpander(expander, decos);
}

void GraphKernelExpanderLite::PreProcessAllNode(const CNodePtr &node) {
  if (Callback::Instance()->GetTargetFromContext() == "CPU" && !AnfUtils::IsGraphKernel(node)) {
    BasicOpInferShape().Infer(node);
  }
}
}  // namespace mindspore::graphkernel
