/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/conv2d_fusion_mapper.h"
#include <vector>
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "src/common/log_util.h"
#include "ops_utils/op_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "infer/unsqueeze.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNumIndex0 = 0;
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;
constexpr size_t kNumIndex3 = 3;
constexpr size_t kSize_0 = 0;
constexpr size_t kSize_1 = 1;
constexpr size_t kSize_2 = 2;
constexpr size_t kSize_3 = 3;
constexpr size_t kSize_4 = 4;
}  // namespace

std::shared_ptr<CNode> Conv2DFusionMapper::CreateTransQuantParamV2(const FuncGraphPtr &func_graph,
                                                                   const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto quant_param_holder = mindspore::lite::GetCNodeQuantHolder(cnode);
  MS_CHECK_TRUE_RET(quant_param_holder != nullptr, nullptr);
  auto quant_params_vec = quant_param_holder->get_input_quant_params();
  if (quant_params_vec.empty()) {
    return nullptr;
  }
  auto quant_params_x1 = quant_params_vec.at(kNumIndex0);
  if (quant_params_x1.size() != kSize_1) {
    MS_LOG(ERROR) << "For active quantization, only per_tensor mode is currently supported."
                  << " Scale size should be 1, but get scale size is: " << quant_params_x1.size();
    return nullptr;
  }
  auto quant_param_x1 = quant_params_x1.front();
  auto scale_x1 = quant_param_x1.scale;
  auto zero_point_x1 = quant_param_x1.zeroPoint;
  if (zero_point_x1 != 0) {
    MS_LOG(ERROR) << "Only support zero_point = 0! zero_point_x1 is: " << zero_point_x1;
    return nullptr;
  }
  auto quant_params_x2 = quant_params_vec.at(kNumIndex1);
  if (quant_params_x2.empty()) {
    return nullptr;
  }
  auto per_channel_size = quant_params_x2.size();
  std::vector<int64_t> shape_vector = {static_cast<int64_t>(per_channel_size)};
  auto buf = std::make_unique<float[]>(per_channel_size);
  MS_CHECK_TRUE_RET(buf != nullptr, nullptr);
  for (uint64_t i = 0; i < per_channel_size; i++) {
    buf[i] = scale_x1 * quant_params_x2.at(i).scale;
    if (quant_params_x2.at(i).zeroPoint != 0) {
      MS_LOG(ERROR) << "Only support zero_point = 0!";
      return nullptr;
    }
  }
  auto tensor_info = lite::CreateTensorInfo(buf.get(), per_channel_size * sizeof(float), shape_vector,
                                            mindspore::TypeId::kNumberTypeFloat32);
  MS_CHECK_TRUE_RET(tensor_info != nullptr, nullptr);
  auto scale_param_node =
    opt::BuildParameterNode(func_graph, tensor_info, cnode->fullname_with_scope() + "_scale", false);
  MS_CHECK_TRUE_RET(scale_param_node != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode->abstract() != nullptr, nullptr);
  scale_param_node->set_abstract(cnode->abstract()->Clone());
  // insert TransQuantParamV2
  auto trans_quant_param_prim = std::make_shared<mindspore::lite::acl::TransQuantParamV2>();
  MS_CHECK_TRUE_RET(trans_quant_param_prim != nullptr, nullptr);
  std::vector<AnfNodePtr> trans_quant_param_inputs = {NewValueNode(trans_quant_param_prim), scale_param_node};
  auto trans_quant_param_cnode = func_graph->NewCNode(trans_quant_param_inputs);
  MS_CHECK_TRUE_RET(trans_quant_param_cnode != nullptr, nullptr);
  trans_quant_param_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_TransQuantParamV2");
  trans_quant_param_cnode->set_abstract(cnode->abstract()->Clone());
  return trans_quant_param_cnode;
}

CNodePtr Conv2DFusionMapper::InsertAdd(const FuncGraphPtr &func_graph, const CNodePtr &qaunt_conv_cnode,
                                       const AnfNodePtr &bias) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(qaunt_conv_cnode != nullptr, nullptr);

  auto prim_unsqueeze = std::make_unique<ops::Unsqueeze>();
  MS_CHECK_TRUE_MSG(prim_unsqueeze != nullptr, nullptr, "Create unsqueeze prim failed!");
  auto prim_unsqueeze_c = prim_unsqueeze->GetPrim();
  MS_CHECK_TRUE_MSG(prim_unsqueeze_c != nullptr, nullptr, "Get prim_c is nullptr!");
  std::vector<int64_t> axis = {kNumIndex0, kNumIndex2, kNumIndex3};
  prim_unsqueeze_c->AddAttr("axis", MakeValue(axis));
  std::vector<AnfNodePtr> unsqueeze_input = {NewValueNode(prim_unsqueeze_c), bias};
  auto unsqueeze_cnode = func_graph->NewCNode(unsqueeze_input);
  MS_CHECK_TRUE_MSG(unsqueeze_cnode != nullptr, nullptr, "Create unsqueeze CNode failed!");
  unsqueeze_cnode->set_fullname_with_scope(qaunt_conv_cnode->fullname_with_scope() + "_unsqueeze");
  MS_CHECK_TRUE_RET(qaunt_conv_cnode->abstract() != nullptr, nullptr);
  unsqueeze_cnode->set_abstract(qaunt_conv_cnode->abstract()->Clone());
  auto prim = std::make_unique<ops::Add>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "Create add prim failed!");
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "Get prim_c is nullptr!");
  auto add_primitive = NewValueNode(prim_c);
  MS_CHECK_TRUE_RET(add_primitive != nullptr, nullptr);
  auto add_fusion = func_graph->NewCNode({add_primitive, qaunt_conv_cnode, unsqueeze_cnode->cast<AnfNodePtr>()});
  MS_CHECK_TRUE_MSG(add_fusion != nullptr, nullptr, "Create add CNode failed!");
  add_fusion->set_fullname_with_scope(qaunt_conv_cnode->fullname_with_scope() + "_add");
  MS_CHECK_TRUE_RET(qaunt_conv_cnode->abstract() != nullptr, nullptr);
  add_fusion->set_abstract(qaunt_conv_cnode->abstract()->Clone());
  return add_fusion;
}

int Conv2DFusionMapper::ReplaceConvToQuantConv(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(cnode->size() >= kSize_3, RET_ERROR);
  auto input1 = cnode->input(kNumIndex1);
  MS_CHECK_TRUE_RET(input1 != nullptr, RET_ERROR);
  auto input2 = cnode->input(kNumIndex2);
  MS_CHECK_TRUE_RET(input2 != nullptr, RET_ERROR);
  auto quant_conv_prim = std::make_shared<mindspore::lite::acl::QuantConv2D>();
  MS_CHECK_TRUE_RET(quant_conv_prim != nullptr, RET_ERROR);
  auto conv_prim = GetValueNode<PrimitivePtr>(cnode->input(kNumIndex0));
  MS_CHECK_TRUE_RET(conv_prim != nullptr, RET_ERROR);
  auto dilation = conv_prim->GetAttr(ops::kDilation);
  auto group = conv_prim->GetAttr(ops::kGroup);
  if (dilation != nullptr) {
    quant_conv_prim->AddAttr("dilations", dilation);
    auto status = AttrAdjust(quant_conv_prim, "dilations");
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "adjust dilation failed!";
      return RET_ERROR;
    }
  }
  if (group != nullptr) {
    quant_conv_prim->AddAttr("groups", group);
  }
  quant_conv_prim->AddAttr("data_format", MakeValue("NCHW"));
  quant_conv_prim->SetAttrs(conv_prim->attrs());
  auto pad_ptr = conv_prim->GetAttr(ops::kPadList);
  if (pad_ptr != nullptr) {
    quant_conv_prim->AddAttr("pads", pad_ptr);
  }
  auto stride_ptr = conv_prim->GetAttr(ops::kStride);
  if (stride_ptr != nullptr) {
    quant_conv_prim->AddAttr("strides", stride_ptr);
    auto status = AttrAdjust(quant_conv_prim, "strides");
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "adjust stride failed!";
      return RET_ERROR;
    }
  }
  quant_conv_prim->AddAttr(kAttrDType, MakeValue(kFloat16));
  auto trans_quant_param_cnode = CreateTransQuantParamV2(func_graph, cnode);
  MS_CHECK_TRUE_RET(trans_quant_param_cnode != nullptr, RET_ERROR);
  auto none_value_node_offset = NewValueNode(std::make_shared<mindspore::None>());
  MS_CHECK_TRUE_RET(none_value_node_offset != nullptr, RET_ERROR);
  none_value_node_offset->set_abstract(std::make_shared<abstract::AbstractNone>());
  auto none_value_node_bias = NewValueNode(std::make_shared<mindspore::None>());
  MS_CHECK_TRUE_RET(none_value_node_bias != nullptr, RET_ERROR);
  none_value_node_bias->set_abstract(std::make_shared<abstract::AbstractNone>());
  std::vector<AnfNodePtr> quant_op_inputs = {
    NewValueNode(quant_conv_prim), input1, input2, trans_quant_param_cnode, none_value_node_bias,
    none_value_node_offset,
  };
  auto qaunt_conv_cnode = func_graph->NewCNode(quant_op_inputs);
  MS_CHECK_TRUE_RET(qaunt_conv_cnode != nullptr, RET_ERROR);
  qaunt_conv_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_quant_conv");
  MS_CHECK_TRUE_RET(cnode->abstract() != nullptr, RET_ERROR);
  auto abs_shape_ptr = cnode->abstract()->GetShape();
  MS_CHECK_TRUE_RET(abs_shape_ptr != nullptr, RET_ERROR);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(TypeId::kNumberTypeFloat16), abs_shape_ptr);
  qaunt_conv_cnode->set_abstract(abstract);
  MS_LOG(INFO) << "QuantConv name: " << qaunt_conv_cnode->fullname_with_scope()
               << ", prim name: " << quant_conv_prim->name() << ", input1: " << input1->DebugString()
               << ", input2: " << input2->DebugString();
  auto manager = Manage(func_graph);
  MS_CHECK_TRUE_RET(manager != nullptr, RET_ERROR);
  if (cnode->size() == kSize_4) {
    // Conv(prim, input1, input2, bias) -> QuantConv(prim, input1, input2, scale) + add(bias)
    auto bias = cnode->input(kNumIndex3);
    MS_CHECK_TRUE_RET(bias != nullptr, RET_ERROR);
    auto add_fusion = InsertAdd(func_graph, qaunt_conv_cnode, bias);
    MS_CHECK_TRUE_RET(add_fusion != nullptr, RET_ERROR);
    (void)manager->Replace(cnode, add_fusion);
    return RET_OK;
  }
  (void)manager->Replace(cnode, qaunt_conv_cnode);
  return RET_OK;
}

int Conv2DFusionMapper::QuantConvMapper(const CNodePtr &cnode) {
  auto func_graph = cnode->func_graph();
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  auto graph_manager = func_graph->manager();
  MS_CHECK_TRUE_RET(graph_manager != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(cnode->abstract() != nullptr, RET_ERROR);
  // size(prim, input1, input2) = 3 or size(prim, input1, input2, bias) = 4
  if (cnode->size() < kSize_3 || cnode->size() > kSize_4) {
    MS_LOG(ERROR) << "The number of inputs of conv cnode can only be 3 or 4, but get " << cnode->size();
    return RET_ERROR;
  }
  auto quant_param_holder = mindspore::lite::GetCNodeQuantHolder(cnode);
  MS_CHECK_TRUE_RET(quant_param_holder != nullptr, RET_ERROR);
  if (!quant_param_holder->IsInputQuantParamsInited()) {
    MS_LOG(INFO) << "InputQuantParamsInited is false, this node is " << cnode->fullname_with_scope();
    return RET_OK;
  }
  auto quant_params_vec = quant_param_holder->get_input_quant_params();
  lite::quant::InsertQuantNodeManager insert_node_manager;
  auto input_2_node = cnode->input(kNumIndex2);
  MS_CHECK_TRUE_RET(input_2_node != nullptr, RET_ERROR);
  if (quant_params_vec.size() == kSize_0) {
    MS_LOG(INFO) << "This node has no quantization parameter. Skip it. node name: " << cnode->fullname_with_scope();
    return RET_OK;
  } else if (quant_params_vec.size() == kSize_2 &&
             (input_2_node->isa<mindspore::CNode>() || input_2_node->isa<mindspore::Parameter>())) {
    MS_LOG(INFO) << "Start do double per_tensor(A&W) pass or Start do per_tensor(A) + per_channel(W) pass(The weight "
                    "is already of the int8 type.).";
    return ReplaceConvToQuantConv(func_graph, cnode);
  } else if (quant_params_vec.size() == kSize_3) {
    MS_LOG(ERROR) << "quant_params_vec size is 3. The conv node conversion with quantized bias is not supported now!";
    return RET_ERROR;
  } else {
    MS_LOG(ERROR) << "Don't support! The number of quantization parameters is " << quant_params_vec.size();
    return RET_ERROR;
  }
}

STATUS Conv2DFusionMapper::Mapper(const CNodePtr &cnode) {
  auto quant_holder = GetCNodeQuantHolder(cnode);
  MS_CHECK_TRUE_MSG(quant_holder != nullptr, RET_NULL_PTR, "quant holder is nullptr.");
  if (quant_holder->IsInputQuantParamsInited()) {
    auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_MSG(cnode_primitive != nullptr, RET_NULL_PTR, "Primitive is nullptr.");
    MS_LOG(INFO) << "quant_holder IsInputQuantParamsInited is true. quant_holder->quant_type(): "
                 << quant_holder->quant_type()
                 << ", cnode_primitive->HasAttr(quant_type): " << cnode_primitive->HasAttr(quant::kQuantType);
    return QuantConvMapper(cnode);
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  ops::Conv2D conv2d_op;
  PrimitivePtr dst_prim = conv2d_op.GetPrim();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  auto status = AttrAdjust(dst_prim, ops::kStride);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust stride failed.";
    return status;
  }
  status = AttrAdjust(dst_prim, ops::kDilation);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust dilation failed.";
    return status;
  }
  status = AdjustAttrPad(dst_prim);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust pad failed.";
    return status;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConv2DFusion, Conv2DFusionMapper)
}  // namespace lite
}  // namespace mindspore
