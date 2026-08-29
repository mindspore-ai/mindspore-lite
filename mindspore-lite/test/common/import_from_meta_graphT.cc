/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "test/common/import_from_meta_graphT.h"
#include <vector>
#include <algorithm>
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "tools/converter/converter_context.h"
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "ir/tensor_new.h"
#include "ir/value.h"
#include "ir/primitive.h"
#include "src/common/ops/primitive/activation.h"
#include "src/common/ops/primitive/add_fusion.h"
#include "src/common/ops/primitive/sub_fusion.h"
#include "src/common/ops/primitive/mul_fusion.h"
#include "src/common/ops/primitive/mat_mul_fusion.h"
#include "src/common/ops/primitive/conv2d_fusion.h"
#include "src/common/ops/primitive/scale_fusion.h"
#include "src/common/ops/primitive/slice_fusion.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "mindspore/ops/infer/fused_batch_norm.h"
#include "infer/stack.h"
#include "infer/tuple_get_item.h"
#include "ut/utils/build_func_graph.h"

namespace mindspore::lite {
namespace {
constexpr size_t kConvPadSize = 4;
}  // namespace

AnfNodePtr AnfImporterFromMetaGraphT::GetNode(int tensor_id) {
  auto n = nodes_.find(tensor_id);
  if (n == nodes_.end()) {
    return nullptr;
  }
  return n->second;
}

void AnfImporterFromMetaGraphT::AddNode(int tensor_id, AnfNodePtr node) { nodes_[tensor_id] = std::move(node); }

int AnfImporterFromMetaGraphT::ConverterConstTensor() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  for (size_t i = 0; i < meta_graph_->allTensors.size(); i++) {
    auto &tensor = meta_graph_->allTensors.at(i);
    MS_ASSERT(tensor != nullptr);
    if (tensor->nodeType != NodeType_ValueNode) {
      continue;
    }
    auto parameter = func_graph_->add_parameter();
    std::vector<int> shape(tensor->dims.size());
    std::copy(tensor->dims.begin(), tensor->dims.end(), shape.begin());
    auto type_id = static_cast<TypeId>(tensor->dataType);
    auto type_ptr = TypeIdToType(type_id);
    std::vector<int64_t> shape_vector(shape.begin(), shape.end());
    if (!tensor->name.empty()) {
      parameter->set_name(tensor->name);
    } else {
      parameter->set_name("const-" + std::to_string(i));
    }
    tensor::TensorPtr tensor_info = tensor::from_spec(type_id, shape_vector, device::DeviceType::kCPU);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "create tensor info failed.";
      return RET_ERROR;
    }
    int status = RET_OK;
    if (!tensor->data.empty()) {
      auto tensor_data = static_cast<char *>(tensor_info->data_c());
      // tensor->dataType may disagree with the real size of tensor->data (e.g. cast test data);
      // clamp the copy to the destination buffer to avoid heap corruption.
      auto copy_size = std::min(tensor->data.size(), static_cast<size_t>(tensor_info->Size()));
      auto ret = memcpy_s(tensor_data, tensor_info->Size(), tensor->data.data(), copy_size);
      if (EOK != ret) {
        MS_LOG(ERROR) << "memcpy_s error";
        return RET_MEMORY_FAILED;
      }
      status = lite::InitParameterFromTensorInfo(parameter, tensor_info);
    } else if (std::find(meta_graph_->inputIndex.begin(), meta_graph_->inputIndex.end(), i) ==
               meta_graph_->inputIndex.end()) {
      status = lite::InitParameterFromTensorInfo(parameter, tensor_info);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "init parameter from tensor info failed";
      return RET_ERROR;
    }
    AddNode(i, parameter);
  }
  return RET_OK;
}

ValueNodePtr AnfImporterFromMetaGraphT::ConvertPrimitive(const std::unique_ptr<schema::CNodeT> &cNode) {
  if (cNode == nullptr || cNode->primitive == nullptr) {
    MS_LOG(ERROR) << "cnode or primitive is nullptr";
    return nullptr;
  }
  const auto &prim_value = cNode->primitive->value;
  PrimitivePtr prim = nullptr;
  switch (prim_value.type) {
    case schema::PrimitiveType_Activation: {
      auto attr = prim_value.AsActivation();
      auto op = std::make_shared<ops::Activation>();
      op->set_activation_type(static_cast<mindspore::ActivationType>(attr->activation_type));
      op->set_alpha(attr->alpha);
      op->set_min_val(attr->min_val);
      op->set_max_val(attr->max_val);
      op->set_approximate(attr->approximate);
      op->set_beta(attr->beta);
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_AddFusion: {
      auto attr = prim_value.AsAddFusion();
      auto op = std::make_shared<ops::AddFusion>();
      op->set_activation_type(static_cast<mindspore::ActivationType>(attr->activation_type));
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_SubFusion: {
      auto attr = prim_value.AsSubFusion();
      auto op = std::make_shared<ops::SubFusion>();
      op->set_activation_type(static_cast<mindspore::ActivationType>(attr->activation_type));
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_MulFusion: {
      auto attr = prim_value.AsMulFusion();
      auto op = std::make_shared<ops::MulFusion>();
      op->set_activation_type(static_cast<mindspore::ActivationType>(attr->activation_type));
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_MatMulFusion: {
      auto attr = prim_value.AsMatMulFusion();
      auto op = std::make_shared<ops::MatMulFusion>();
      op->set_transpose_a(attr->transpose_a);
      op->set_transpose_b(attr->transpose_b);
      op->set_activation_type(static_cast<mindspore::ActivationType>(attr->activation_type));
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Conv2DFusion: {
      auto attr = prim_value.AsConv2DFusion();
      auto op = std::make_shared<ops::Conv2DFusion>();
      op->set_format(static_cast<mindspore::Format>(attr->format));
      op->set_kernel_size(attr->kernel_size);
      op->set_stride(attr->stride);
      op->set_dilation(attr->dilation);
      auto pad_mode = static_cast<mindspore::PadMode>(attr->pad_mode);
      // core Conv2D::set_pad_mode reads the "pad" attr and requires {0,0,0,0} unless pad_mode is PAD,
      // so pad must be set first; schema defaults (mode/group/channel = 0) are rejected by core setters.
      auto pad_list = attr->pad_list;
      if (pad_list.size() != kConvPadSize) {
        pad_list = {0, 0, 0, 0};
      }
      op->set_pad(pad_mode == mindspore::PAD ? pad_list : std::vector<int64_t>{0, 0, 0, 0});
      op->set_pad_mode(pad_mode);
      op->set_pad_list(pad_list);
      if (attr->mode > 0) {
        op->set_mode(attr->mode);
      }
      // nnacl CheckConvAttr rejects group == 0 (schema default), normalize to plain convolution
      op->set_group(attr->group > 0 ? attr->group : 1);
      if (attr->in_channel > 0) {
        op->set_in_channel(attr->in_channel);
      }
      if (attr->out_channel > 0) {
        op->set_out_channel(attr->out_channel);
      }
      op->set_activation_type(static_cast<mindspore::ActivationType>(attr->activation_type));
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_ScaleFusion: {
      auto attr = prim_value.AsScaleFusion();
      auto op = std::make_shared<ops::ScaleFusion>();
      op->set_axis(attr->axis);
      op->set_activation_type(static_cast<mindspore::ActivationType>(attr->activation_type));
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_SliceFusion: {
      auto attr = prim_value.AsSliceFusion();
      auto op = std::make_shared<ops::SliceFusion>();
      op->set_axes(attr->axes);
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_BiasAdd: {
      auto op = std::make_shared<ops::BiasAdd>();
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_BatchNorm: {
      auto attr = prim_value.AsBatchNorm();
      auto op = std::make_shared<ops::BatchNorm>();
      op->set_epsilon(attr->epsilon);
      op->set_is_training(attr->is_training);
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_FusedBatchNorm: {
      auto attr = prim_value.AsFusedBatchNorm();
      auto op = std::make_shared<ops::FusedBatchNorm>();
      op->set_epsilon(attr->epsilon);
      op->set_momentum(attr->momentum);
      op->set_mode(attr->mode);
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Stack: {
      auto attr = prim_value.AsStack();
      auto op = std::make_shared<ops::Stack>();
      op->set_axis(attr->axis);
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Split: {
      auto attr = prim_value.AsSplit();
      auto op = std::make_shared<ops::Split>();
      op->set_output_num(attr->output_num);
      op->set_axis(attr->axis);
      prim = op->GetPrim();
      prim->AddAttr("size_splits", MakeValue(attr->size_splits));
      break;
    }
    case schema::PrimitiveType_Concat: {
      auto attr = prim_value.AsConcat();
      auto op = std::make_shared<ops::Concat>();
      op->set_axis(attr->axis);
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Range: {
      auto attr = prim_value.AsRange();
      auto op = std::make_shared<ops::Range>();
      prim = op->GetPrim();
      prim->AddAttr("d_type", MakeValue(attr->d_type));
      prim->AddAttr("start", MakeValue(attr->start));
      prim->AddAttr("limit", MakeValue(attr->limit));
      prim->AddAttr("delta", MakeValue(attr->delta));
      break;
    }
    case schema::PrimitiveType_Cast: {
      auto op = std::make_shared<ops::Cast>();
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Transpose: {
      auto op = std::make_shared<ops::Transpose>();
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Reshape: {
      auto op = std::make_shared<ops::Reshape>();
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Rsqrt: {
      auto op = std::make_shared<ops::Rsqrt>();
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_Shape: {
      auto op = std::make_shared<ops::Shape>();
      prim = op->GetPrim();
      break;
    }
    case schema::PrimitiveType_ExpandDims: {
      auto op = std::make_shared<ops::ExpandDims>();
      prim = op->GetPrim();
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported schema primitive type: " << prim_value.type;
      return nullptr;
  }
  if (prim == nullptr) {
    MS_LOG(ERROR) << "create primitive failed for node " << cNode->name;
    return nullptr;
  }
  return NewValueNode(prim);
}

abstract::AbstractTensorPtr AnfImporterFromMetaGraphT::ConvertTensorToAbstractTensor(
  const std::unique_ptr<schema::TensorT> &tensor) {
  MS_ASSERT(nullptr != tensor);
  std::vector<int> shape(tensor->dims.size());
  std::copy(tensor->dims.begin(), tensor->dims.end(), shape.begin());
  auto type_id = static_cast<TypeId>(tensor->dataType);
  auto type_ptr = TypeIdToType(type_id);
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto ptr = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  MS_ASSERT(nullptr != ptr);
  return ptr;
}

int AnfImporterFromMetaGraphT::ConvertAbstract(const std::unique_ptr<schema::CNodeT> &src_cnode,
                                               const CNodePtr &dst_cnode) {
  if (src_cnode->outputIndex.empty()) {
    MS_LOG(ERROR) << "cnode " << src_cnode->name << " has no output tensor";
    return RET_ERROR;
  }
  if (src_cnode->outputIndex.size() == 1) {
    auto &tensor = meta_graph_->allTensors.at(src_cnode->outputIndex.front());
    dst_cnode->set_abstract(ConvertTensorToAbstractTensor(tensor));
    return RET_OK;
  }
  abstract::AbstractBasePtrList abstract_list;
  (void)std::transform(src_cnode->outputIndex.begin(), src_cnode->outputIndex.end(), std::back_inserter(abstract_list),
                       [this](uint32_t idx) { return ConvertTensorToAbstractTensor(meta_graph_->allTensors.at(idx)); });
  dst_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return RET_OK;
}

AnfNodePtr AnfImporterFromMetaGraphT::BuildGraphInput(uint32_t tensor_index) {
  if (tensor_index >= meta_graph_->allTensors.size()) {
    MS_LOG(ERROR) << "input tensor index out of range: " << tensor_index;
    return nullptr;
  }
  auto &tensor = meta_graph_->allTensors.at(tensor_index);
  if (tensor->nodeType == NodeType_ValueNode) {
    // const tensors are handled by ConverterConstTensor
    return nullptr;
  }
  auto parameter = func_graph_->add_parameter();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "add graph input parameter failed";
    return nullptr;
  }
  if (!tensor->name.empty()) {
    parameter->set_name(tensor->name);
  } else {
    parameter->set_name("input-" + std::to_string(tensor_index));
  }
  parameter->set_abstract(ConvertTensorToAbstractTensor(tensor));
  AddNode(tensor_index, parameter);
  return parameter;
}

int AnfImporterFromMetaGraphT::ConverterCNode() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  for (const auto &cNode : meta_graph_->nodes) {
    MS_ASSERT(nullptr != cNode);
    auto anf_primitive = ConvertPrimitive(cNode);
    if (anf_primitive == nullptr) {
      MS_LOG(ERROR) << "cannot obtain anf primitive";
      return RET_NULL_PTR;
    }
    std::vector<AnfNodePtr> op_inputs = {anf_primitive};
    for (int j : cNode->inputIndex) {
      auto node = GetNode(j);
      if (nullptr == node) {
        node = BuildGraphInput(j);
      }
      if (nullptr == node) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NULL_PTR;
      }
      op_inputs.push_back(node);
    }
    auto new_cnode = func_graph_->NewCNode(op_inputs);
    MS_ASSERT(nullptr != new_cnode);
    new_cnode->set_fullname_with_scope(cNode->name);
    auto status = ConvertAbstract(cNode, new_cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ConvertAbstract failed.";
      return status;
    }
    if (cNode->outputIndex.size() == 1) {
      AddNode(cNode->outputIndex.front(), new_cnode);
    } else {
      // multi-output ops are consumed through TupleGetItem cnodes
      for (size_t out_idx = 0; out_idx < cNode->outputIndex.size(); out_idx++) {
        auto tuple_prim = std::make_shared<ops::TupleGetItem>();
        if (tuple_prim == nullptr) {
          MS_LOG(ERROR) << "new TupleGetItem failed";
          return RET_NULL_PTR;
        }
        std::vector<AnfNodePtr> tuple_inputs = {NewValueNode(tuple_prim->GetPrim()), new_cnode,
                                                NewValueNode(MakeValue(static_cast<int64_t>(out_idx)))};
        auto get_item_cnode = func_graph_->NewCNode(tuple_inputs);
        if (get_item_cnode == nullptr) {
          MS_LOG(ERROR) << "new tuple get item cnode failed";
          return RET_NULL_PTR;
        }
        get_item_cnode->set_fullname_with_scope(cNode->name + "_getitem_" + std::to_string(out_idx));
        auto &out_tensor = meta_graph_->allTensors.at(cNode->outputIndex[out_idx]);
        get_item_cnode->set_abstract(ConvertTensorToAbstractTensor(out_tensor));
        AddNode(cNode->outputIndex[out_idx], get_item_cnode);
      }
    }
  }
  return RET_OK;
}

int AnfImporterFromMetaGraphT::AddReturnCNode() {
  func_graph_->set_attr("fmk", MakeValue(static_cast<int64_t>(meta_graph_->fmkType)));
  if (meta_graph_->name.empty()) {
    func_graph_->set_attr("graph_name", MakeValue(std::string("main")));
  } else {
    func_graph_->set_attr("graph_name", MakeValue(meta_graph_->name));
  }
  std::vector<AnfNodePtr> return_inputs;
  for (auto idx : meta_graph_->outputIndex) {
    auto node = GetNode(idx);
    if (node == nullptr) {
      MS_LOG(ERROR) << "graph output tensor not found: " << idx;
      return RET_NULL_PTR;
    }
    return_inputs.push_back(node);
  }
  auto return_cnode = lite::AddReturn(func_graph_, return_inputs);
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "add return cnode failed";
    return RET_ERROR;
  }
  (void)Manage(func_graph_, true);
  return RET_OK;
}

FuncGraphPtr AnfImporterFromMetaGraphT::Fb2Anf(schema::MetaGraphT *meta_graph) {
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta_graph is null";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
    return nullptr;
  }
  AnfImporterFromMetaGraphT anfImporterFromMetaGraphT(meta_graph);
  auto ret = anfImporterFromMetaGraphT.ConverterConstTensor();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "ConverterConstTensor failed " << ret;
    return nullptr;
  }
  ret = anfImporterFromMetaGraphT.ConverterCNode();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "ConverterCNode failed " << ret;
    return nullptr;
  }
  ret = anfImporterFromMetaGraphT.AddReturnCNode();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "AddReturnCNode failed " << ret;
    return nullptr;
  }
  return anfImporterFromMetaGraphT.func_graph_;
}
}  // namespace mindspore::lite
