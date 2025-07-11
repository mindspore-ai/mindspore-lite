/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * distributed under the License is distributed on an AS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the Lictf_logical_ense.
 */

#include "tools/converter/parser/tf/tf_model_parser.h"
#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include "abstract/utils.h"
#include "include/registry/node_parser_registry.h"
#include "ir/anf.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "infer/make_tuple.h"
#include "infer/return.h"
#include "infer/tuple_get_item.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/parser/lite_model_parser_creator.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/parser/tf/functionalize_control_op_pass.h"
#include "tools/converter/parser/tf/remove_ineffective_control_flow.h"
#include "tools/converter/parser/tf/tf_fake_quant_adjust.h"
#include "tools/converter/parser/tf/tf_input_adjust.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/parser/tf/tf_util.h"
#include "tools/converter/parser/unify_format.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/parser/unused_node_remove_pass.h"
#include "tools/converter/parser/einsum_adjust.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_w.h"

using mindspore::converter::kFmkTypeTf;
namespace mindspore {
namespace lite {
namespace {
bool IsTensorListOp(const AnfNodePtr &anf_node) {
  return opt::CheckPrimitiveType(anf_node, prim::kPrimTensorListFromTensor) ||
         opt::CheckPrimitiveType(anf_node, prim::kPrimTensorListSetItem) ||
         opt::CheckPrimitiveType(anf_node, prim::kPrimTensorListReserve);
}

constexpr size_t kConvWeightIndex = 2;

AnfNodePtr GetAnfNode(const std::string &name, const std::unordered_map<std::string, AnfNodePtr> &anf_node_map,
                      int index = 0) {
  AnfNodePtr ret = nullptr;
  auto flat_anf_name = TensorFlowUtils::GetFlattenNodeName(name);
  if (anf_node_map.find(flat_anf_name) != anf_node_map.end()) {
    ret = anf_node_map.at(flat_anf_name);
  } else if (anf_node_map.find(name + ":" + std::to_string(index)) != anf_node_map.end()) {
    ret = anf_node_map.at(flat_anf_name + ":" + std::to_string(index));
  }
  return ret;
}

std::string GetOriginInputName(const tensorflow::NodeDef &node,
                               const std::map<std::string, const tensorflow::NodeDef *> &tf_graph_nodes) {
  if (!TensorFlowUtils::OutputIsInputOp(node.op())) {
    return node.name();
  }
  auto tmp_node = &node;
  while (TensorFlowUtils::OutputIsInputOp(tmp_node->op())) {
    auto flatten_input_name = TensorFlowUtils::GetFlattenNodeName(tmp_node->input(0));
    if (tf_graph_nodes.find(flatten_input_name) == tf_graph_nodes.end()) {
      return flatten_input_name;
    }
    tmp_node = tf_graph_nodes.at(flatten_input_name);
  }
  return tmp_node->name();
}

STATUS CheckStrView(std::string_view str_view, uint64_t *scratch) {
  if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
    return RET_ERROR;
  }
  for (size_t i = 0; i < static_cast<size_t>(*scratch); ++i) {
    if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
      return RET_ERROR;
    }
  }
  if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
    return RET_ERROR;
  }
  if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
    return RET_ERROR;
  }
  return RET_OK;
}

int GetShapeSize(const tensorflow::TensorProto &tensor_proto) {
  auto &tensor_shape = tensor_proto.tensor_shape();
  int shape_size = 1;
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(shape_size, tensor_shape.dim(i).size(), 0);
    shape_size *= tensor_shape.dim(i).size();
  }
  return shape_size;
}

STATUS SetFloatTensorInfo(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info) {
  auto shape_size = GetShapeSize(tensor_proto);
  auto &tensor_shape = tensor_proto.tensor_shape();
  ShapeVector shape_vector{};
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector.push_back(tensor_shape.dim(i).size());
  }
  *tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, kNumberTypeFloat32);
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor data failed.";
    return RET_ERROR;
  }
  auto tensor_data = reinterpret_cast<float *>((*tensor_info)->data_c());
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    return RET_ERROR;
  }

  if (tensor_proto.float_val_size() == 1) {
    for (int i = 0; i < shape_size; i++) {
      tensor_data[i] = tensor_proto.float_val(0);
    }
  }
  if (INT_MUL_OVERFLOW_THRESHOLD(shape_size, sizeof(float), SIZE_MAX)) {
    MS_LOG(ERROR) << "data_size overflow.";
    return RET_ERROR;
  }
  if (tensor_proto.tensor_content().size() == shape_size * sizeof(float)) {
    const auto addr = reinterpret_cast<const float *>(tensor_proto.tensor_content().data());
    if (::memcpy_s(tensor_data, (*tensor_info)->Size(), addr, shape_size * sizeof(float)) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

STATUS SetFloat16TensorInfo(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info) {
  auto shape_size = GetShapeSize(tensor_proto);
  auto &tensor_shape = tensor_proto.tensor_shape();
  ShapeVector shape_vector{};
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector.push_back(tensor_shape.dim(i).size());
  }
  *tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, kNumberTypeFloat16);
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor data failed.";
    return RET_ERROR;
  }
  auto tensor_data = reinterpret_cast<int16_t *>((*tensor_info)->data_c());
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    return RET_ERROR;
  }

  if (tensor_proto.half_val_size() == 1) {
    for (int i = 0; i < shape_size; i++) {
      tensor_data[i] = tensor_proto.half_val(0);
    }
  }
  if (INT_MUL_OVERFLOW_THRESHOLD(shape_size, sizeof(int16_t), SIZE_MAX)) {
    MS_LOG(ERROR) << "data_size overflow.";
    return RET_ERROR;
  }
  if (tensor_proto.tensor_content().size() == shape_size * sizeof(int16_t)) {
    const auto addr = tensor_proto.tensor_content().data();
    if (::memcpy_s(tensor_data, (*tensor_info)->Size(), addr, shape_size * sizeof(int16_t)) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

STATUS SetInt32TensorInfo(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info) {
  auto shape_size = GetShapeSize(tensor_proto);
  auto &tensor_shape = tensor_proto.tensor_shape();
  ShapeVector shape_vector{};
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector.push_back(tensor_shape.dim(i).size());
  }
  *tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, kNumberTypeInt32);
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor data failed.";
    return RET_ERROR;
  }
  auto tensor_data = reinterpret_cast<int *>((*tensor_info)->data_c());
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    return RET_ERROR;
  }
  if (shape_size == 0) {
    return RET_OK;
  }
  if (tensor_proto.tensor_content().empty()) {
    const auto &origin_data = tensor_proto.int_val();
    if (tensor_proto.int_val_size() == 1) {
      for (int i = 0; i < shape_size; ++i) {
        tensor_data[i] = origin_data[0];
      }
    } else {
      MS_CHECK_GE(tensor_proto.int_val_size(), shape_size, RET_ERROR);
      for (int i = 0; i < shape_size; ++i) {
        tensor_data[i] = origin_data[i];
      }
    }
  } else {
    if (INT_MUL_OVERFLOW_THRESHOLD(shape_size, sizeof(int32_t), SIZE_MAX)) {
      MS_LOG(ERROR) << "data_size overflow.";
      return RET_ERROR;
    }
    MS_CHECK_GE(tensor_proto.tensor_content().size(), shape_size * sizeof(int32_t), RET_ERROR);
    const auto addr = reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data());
    if (::memcpy_s(tensor_data, (*tensor_info)->Size(), addr, shape_size * sizeof(int32_t)) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS SetBoolTensorInfo(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info) {
  auto shape_size = GetShapeSize(tensor_proto);
  auto &tensor_shape = tensor_proto.tensor_shape();
  ShapeVector shape_vector{};
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector.push_back(tensor_shape.dim(i).size());
  }
  *tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, kNumberTypeBool);
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor data failed.";
    return RET_ERROR;
  }
  auto tensor_data = reinterpret_cast<bool *>((*tensor_info)->data_c());
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    return RET_ERROR;
  }
  if (tensor_proto.bool_val_size() != shape_size) {
    MS_LOG(ERROR) << "shape size:[" << shape_size << "] not equal bool val size:[" << tensor_proto.bool_val_size()
                  << "]";
    return RET_ERROR;
  }
  for (int i = 0; i < shape_size; i++) {
    int value = tensor_proto.bool_val(i);
    tensor_data[i] = value;
  }
  return RET_OK;
}

STATUS SetStringTensorInfo(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info) {
  auto &tensor_shape = tensor_proto.tensor_shape();
  ShapeVector shape_vector{};
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector.push_back(tensor_shape.dim(i).size());
  }

  if (shape_vector.empty()) {
    *tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, kObjectTypeString);
    if (*tensor_info == nullptr) {
      MS_LOG(ERROR) << "create tensor info failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }

  std::string shape_str;
  shape_str += std::to_string(shape_vector.size()) + ",";
  for (auto &dim : shape_vector) {
    shape_str += std::to_string(dim) + ",";
  }

  auto tensor_data = new (std::nothrow) string;
  CHECK_NULL_RETURN(tensor_data);
  if (tensor_proto.string_val_size() == 1) {
    *tensor_data = tensor_proto.string_val(0);
  } else {
    MS_LOG(ERROR) << "string size bigger than one, not support.";
    delete tensor_data;
    return RET_ERROR;
  }
  if (INT_ADD_OVERFLOW(shape_str.size(), (*tensor_data).size())) {
    MS_LOG(ERROR) << "data_size overflow.";
    delete tensor_data;
    return RET_ERROR;
  }
  shape_vector = {static_cast<int64_t>(shape_str.size() + (*tensor_data).size())};
  *tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, kObjectTypeString);
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    delete tensor_data;
    return RET_ERROR;
  }
  auto tensor_info_data = reinterpret_cast<uint8_t *>((*tensor_info)->data_c());
  if (memcpy_s(tensor_info_data, (*tensor_info)->Size(), shape_str.data(), shape_str.size()) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    delete tensor_data;
    return RET_ERROR;
  }
  MS_CHECK_TRUE_RET((*tensor_info)->Size() >= (*tensor_data).size(), RET_ERROR);
  if (memcpy_s(tensor_info_data + shape_str.size(), (*tensor_info)->Size() - (*tensor_data).size(),
               (*tensor_data).data(), (*tensor_data).size()) != EOK) {
    MS_LOG(ERROR) << "memcpy failed.";
    delete tensor_data;
    return RET_ERROR;
  }

  delete tensor_data;
  return RET_OK;
}

FuncGraphPtr ConvertGraph(api::FuncGraphPtr func_graph) {
  auto impl = func_graph->impl();
  return std::dynamic_pointer_cast<FuncGraph>(impl);
}
}  // namespace

STATUS TFModelParser::SetInt64TensorToInt64Tensor(const tensorflow::TensorProto &tensor_proto,
                                                  tensor::TensorPtr *tensor_info) {
  auto &tensor_shape = tensor_proto.tensor_shape();
  ShapeVector shape_vector{};
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector.push_back(tensor_shape.dim(i).size());
  }
  tensor::TensorPtr tensor_info_int64;
  if (tensor_proto.tensor_content().empty()) {
    tensor_info_int64 = CreateTensorInfo(nullptr, 0, shape_vector, kNumberTypeInt64);
    if (tensor_info_int64 == nullptr) {
      MS_LOG(ERROR) << "CreateTensorInfo failed.";
      return RET_ERROR;
    }
    auto tensor_int64_data = reinterpret_cast<int64_t *>(tensor_info_int64->data_c());
    if (tensor_int64_data == nullptr) {
      MS_LOG(ERROR) << "new data failed";
      return RET_ERROR;
    }
    const auto &origin_data = tensor_proto.int64_val();
    for (int i = 0; i < tensor_proto.int64_val_size(); ++i) {
      tensor_int64_data[i] = origin_data[i];
    }
  } else {
    const auto origin_data = reinterpret_cast<const int64_t *>(tensor_proto.tensor_content().data());
    tensor_info_int64 =
      CreateTensorInfo(origin_data, tensor_proto.tensor_content().size(), shape_vector, kNumberTypeInt64);
    if (tensor_info_int64 == nullptr) {
      MS_LOG(ERROR) << "CreateTensorInfo failed.";
      return RET_ERROR;
    }
  }
  *tensor_info = tensor_info_int64;
  return RET_OK;
}

STATUS TFModelParser::SetInt64TensorInfo(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info,
                                         const std::string &node_name) {
  if (SetInt64TensorToInt64Tensor(tensor_proto, tensor_info) != RET_OK) {
    MS_LOG(ERROR) << "SetInt64TensorInfoMap failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

STATUS TFModelParser::ConvertConstVariant(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info) {
  if (tensor_proto.variant_val_size() != 1) {
    MS_LOG(ERROR) << "only support variant_val_size == 1 now";
    return RET_ERROR;
  }
  auto &variant = tensor_proto.variant_val(0);
  if (variant.type_name() != "tensorflow::TensorList" || variant.tensors_size() <= 0) {
    MS_LOG(DEBUG) << "Only nonempty TensorList type is supported now";
  }
  auto descriptor = variant.GetMetadata().descriptor;
  auto reflection = variant.GetMetadata().reflection;
  if (descriptor == nullptr || reflection == nullptr) {
    MS_LOG(ERROR) << "descriptor or reflection is nullptr";
    return RET_ERROR;
  }
  auto field_descriptor = descriptor->field(1);
  if (field_descriptor == nullptr) {
    MS_LOG(ERROR) << "field_descriptor is nullptr";
    return RET_ERROR;
  }
  if (field_descriptor->type() != google::protobuf::FieldDescriptor::TYPE_BYTES) {
    MS_LOG(ERROR) << "metadata type is not TYPE_BYTES";
    return RET_ERROR;
  }
  auto origin_str = reflection->GetString(variant, field_descriptor);
  std::string_view str_view(origin_str);
  uint64_t scratch;
  if (CheckStrView(str_view, &scratch) != RET_OK) {
    return RET_ERROR;
  }
  auto element_dtype = static_cast<size_t>(scratch);

  tensorflow::TensorShapeProto element_shape_proto;
  element_shape_proto.ParseFromString(origin_str);
  auto dim_size = element_shape_proto.dim_size();
  std::vector<int> tensor_list_data(dim_size + 2);
  tensor_list_data[0] = TensorFlowUtils::GetTFDataType(tensorflow::DataType(element_dtype));
  if (tensor_list_data[0] == kNumberTypeFloat64) {
    tensor_list_data[0] = kNumberTypeFloat32;
  }
  tensor_list_data[1] = element_shape_proto.dim_size();
  for (int i = 0; i < dim_size; i++) {
    auto dim = element_shape_proto.dim(i).size();
    if (dim > static_cast<int64_t>(INT32_MAX) || dim < static_cast<int64_t>(INT32_MIN)) {
      MS_LOG(ERROR) << "int64 data " << dim << " too big to fit into int32";
      return RET_ERROR;
    } else {
      tensor_list_data[i + 2] = static_cast<int>(dim);
    }
  }
  tensor_list_data.emplace_back(variant.tensors_size());
  for (const auto &tensor : variant.tensors()) {
    std::vector<int> single_tensor_data;
    single_tensor_data.emplace_back(tensor.tensor_shape().dim_size());
    for (int i = 0; i < tensor.tensor_shape().dim_size(); i++) {
      single_tensor_data.emplace_back(tensor.tensor_shape().dim(i).size());
    }
    tensor_list_data.insert(tensor_list_data.end(), single_tensor_data.begin(), single_tensor_data.end());
  }
  if (INT_MUL_OVERFLOW_THRESHOLD(tensor_list_data.size(), sizeof(int), INT_MAX)) {
    MS_LOG(ERROR) << "tensor_list_data's size overflow.";
    return RET_ERROR;
  }
  *tensor_info = CreateTensorInfo(tensor_list_data.data(), tensor_list_data.size() * sizeof(int),
                                  {static_cast<int64_t>(tensor_list_data.size())}, kObjectTypeTensorType);
  if (*tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor data failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TFModelParser::SetTensorInfoFromType(const tensorflow::TensorProto &tensor_proto, tensor::TensorPtr *tensor_info,
                                            const std::string &node_name) {
  auto type = (*tensor_info)->data_type();
  if (type == kNumberTypeFloat32 || type == kNumberTypeFloat) {
    return SetFloatTensorInfo(tensor_proto, tensor_info);
  } else if (type == kNumberTypeInt32 || type == kNumberTypeInt) {
    return SetInt32TensorInfo(tensor_proto, tensor_info);
  } else if (type == kNumberTypeInt64) {
    return SetInt64TensorInfo(tensor_proto, tensor_info, node_name);
  } else if (type == kNumberTypeBool) {
    return SetBoolTensorInfo(tensor_proto, tensor_info);
  } else if (type == kObjectTypeTensorType) {
    return ConvertConstVariant(tensor_proto, tensor_info);
  } else if (type == kObjectTypeString) {
    return SetStringTensorInfo(tensor_proto, tensor_info);
  } else if (type == kNumberTypeFloat16) {
    return SetFloat16TensorInfo(tensor_proto, tensor_info);
  } else {
    MS_LOG(ERROR) << "Unsupported dataType: " << type;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertConstTensor(const tensorflow::NodeDef &node_def, const tensorflow::AttrValue &attr_value,
                                         const TypeId &type, const ParameterPtr &parameter,
                                         std::vector<int64_t> *shape_vector) {
  MSLITE_CHECK_PTR(parameter);
  MSLITE_CHECK_PTR(shape_vector);
  const tensorflow::TensorProto &tensor_proto = attr_value.tensor();
  const tensorflow::TensorShapeProto &tensor_shape = tensor_proto.tensor_shape();
  shape_vector->clear();
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector->push_back(tensor_shape.dim(i).size());
  }
  auto tensor_info = std::make_shared<tensor::Tensor>(type, *shape_vector);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor info is nullptr";
    return RET_ERROR;
  }
  auto status = SetTensorInfoFromType(tensor_proto, &tensor_info, node_def.name());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "set tensor data from type failed.";
    return RET_ERROR;
  }
  status = InitParameterFromTensorInfo(parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertParameter(const tensorflow::NodeDef &node, const ParameterPtr &parameter,
                                       std::unordered_map<std::string, AnfNodePtr> *anf_node_map, bool root_graph) {
  MSLITE_CHECK_PTR(parameter);
  MSLITE_CHECK_PTR(anf_node_map);

  tensorflow::AttrValue attr_value;
  TypeId type = kNumberTypeFloat32;
  if (TensorFlowUtils::FindAttrValue(node, "dtype", &attr_value)) {
    type = TensorFlowUtils::GetTFDataType(attr_value.type());
  }

  std::vector<int64_t> shape;
  if (TensorFlowUtils::FindAttrValue(node, "shape", &attr_value)) {
    shape = ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(node.name());
    if (ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 && shape.empty()) {
      MS_LOG(WARNING) << "Can not find name in map. name is " << node.name();
    }
    if (shape.empty()) {
      auto &shape_attr = attr_value.shape();
      for (int i = 0; i < shape_attr.dim_size(); ++i) {
        shape.push_back(shape_attr.dim(i).size());
      }
    }
  }

  if (TensorFlowUtils::FindAttrValue(node, "value", &attr_value)) {
    MS_LOG(INFO) << "Found value attr, means it has default value";
    auto status = ConvertConstTensor(node, attr_value, type, parameter, &shape);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "convert const tensor failed.";
      return status;
    }
  } else {
    if (root_graph) {
      graph_input_names_.emplace_back(node.name());  // only root graph need set graph input names
    }
  }

  auto abstract_tensor = CreateTensorAbstract(shape, type);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return RET_ERROR;
  }
  parameter->set_name(node.name());
  parameter->set_abstract(abstract_tensor);

  (*anf_node_map)[node.name()] = parameter;
  (*anf_node_map)[node.name() + ":0"] = parameter;
  return RET_OK;
}

STATUS TFModelParser::ConvertGraphInputsAndConsts(const std::vector<const tensorflow::NodeDef *> &tf_graph_nodes,
                                                  const FuncGraphPtr &anf_graph,
                                                  std::unordered_map<std::string, AnfNodePtr> *anf_node_map,
                                                  bool root_graph) {
  MSLITE_CHECK_PTR(anf_graph);
  MSLITE_CHECK_PTR(anf_node_map);
  for (auto &node : tf_graph_nodes) {
    bool have_data_depend = false;
    for (int i = 0; i < node->input_size(); ++i) {
      auto name = node->input(i);
      if (!name.empty() && name[0] != '^') {  // control_depend input start with "^"
        have_data_depend = true;
        break;
      }
    }
    if (!have_data_depend && node->op() != "NoOp") {
      auto parameter = anf_graph->add_parameter();
      CHECK_NULL_RETURN(parameter);
      if (ConvertParameter(*node, parameter, anf_node_map, root_graph) != RET_OK) {
        MS_LOG(ERROR) << "convert Parameter Node failed";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

api::FuncGraphPtr TFModelParser::Parse(const converter::ConverterParameters &flag) {
  auto modelFile = flag.model_file;
  NotSupportOp::GetInstance()->set_fmk_type("TF");
  auto status = ValidateFileStr(modelFile, ".pb");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.pb";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  tf_root_graph_ = std::make_unique<tensorflow::GraphDef>();
  if (tf_root_graph_ == nullptr) {
    MS_LOG(ERROR) << "tf_root_graph_ is nullptr";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  status = ReadProtoFromBinaryFile(modelFile, tf_root_graph_.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Open modelFile for TF converter failed!";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(graph != nullptr, nullptr, "create FuncGraph failed");
  res_graph_ = api::MakeShared<api::FuncGraph>(graph);
  if (res_graph_ == nullptr) {
    MS_LOG(ERROR) << "funGraphPtr is nullptr";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  graph->set_attr("graph_name", MakeValue("main_graph"));
  graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeTf)));

  for (int i = 0; i < tf_root_graph_->node_size(); i++) {
    auto &node_def = tf_root_graph_->node(i);
    tf_root_graph_nodes_[node_def.name()] = &node_def;
    tf_root_graph_nodes_vec_.emplace_back(&node_def);
  }

  status = ConvertGraphInputsAndConsts(tf_root_graph_nodes_vec_, graph, &anf_root_node_map_, true);
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  bool success_flag = true;
  ineffective_if_op_map_.clear();
  for (int i = 0; i < tf_root_graph_->node_size(); i++) {
    auto &node_def = tf_root_graph_->node(i);
    status = ConvertOps(node_def, tf_root_graph_nodes_, graph, &anf_root_node_map_);
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    if (status != RET_OK) {
      success_flag = false;
    }
  }
  if (!success_flag) {
    MS_LOG(ERROR) << "Convert ops failed.";
    return nullptr;
  }

  if (!nodes_with_null_input_.empty()) {
    status = ConnectNullInput();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Connect null inputs failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return nullptr;
    }
  }

  if ((status = ConvertRootGraphOutputs()) != RET_OK) {
    MS_LOG(ERROR) << "Convert graph outputs failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ConvertSubgraph();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert subgraph failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  if ((status = CommonAnfAdjust(graph)) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(graph, &all_func_graphs);
  if ((status = TF2AnfAdjust(all_func_graphs, &ineffective_if_op_map_)) != RET_OK) {
    MS_LOG(ERROR) << "TF2AnfAdjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto unify_format = std::make_shared<UnifyFormatToNHWC>(kFmkTypeTf, false, flag.save_type);
  MS_CHECK_TRUE_RET(unify_format != nullptr, nullptr);
  if (!unify_format->Run(graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }
  graph->set_manager(nullptr);
  static auto root_func_manager = Manage(graph);
  if (root_func_manager == nullptr) {
    MS_LOG(ERROR) << "root_func_manager is nullptr.";
    return nullptr;
  }
  return res_graph_;
}

STATUS TFModelParser::ConvertSubgraphInputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                                            std::unordered_map<std::string, AnfNodePtr> *anf_sub_node_map,
                                            const tensorflow::FunctionDef &tf_sub_fuction, const CNodePtr &cnode,
                                            const FuncGraphPtr &sub_func_graph) {
  MSLITE_CHECK_PTR(anf_sub_node_map);
  MSLITE_CHECK_PTR(cnode);
  MSLITE_CHECK_PTR(sub_func_graph);
  MSLITE_CHECK_PTR(tf_sub_node_map);
  std::vector<ParameterPtr> sub_graph_inputs;
  auto &tf_sub_signature = tf_sub_fuction.signature();
  auto &sub_graph_name = tf_sub_signature.name();
  auto input_arg_size = tf_sub_signature.input_arg_size();
  for (int j = 0; j < input_arg_size; j++) {
    auto &input_arg = tf_sub_signature.input_arg(j);
    auto parameter = sub_func_graph->add_parameter();
    CHECK_NULL_RETURN(parameter);
    parameter->set_name(input_arg.name());
    (*anf_sub_node_map)[input_arg.name()] = parameter;
    auto root_inputs = cnode->inputs();
    if (opt::CheckPrimitiveType(cnode, prim::kPrimWhile)) {
      parameter->set_abstract(root_inputs[j + 1]->abstract());
    } else {
      parameter->set_abstract(root_inputs[j + 2]->abstract());
    }
    sub_graph_inputs.emplace_back(parameter);
  }
  std::vector<const tensorflow::NodeDef *> subgraph_tf_node_vec;
  for (int j = 0; j < tf_sub_fuction.node_def_size(); j++) {
    auto &node_def = tf_sub_fuction.node_def(j);
    (*tf_sub_node_map)[node_def.name()] = &node_def;
    subgraph_tf_node_vec.emplace_back(&node_def);
  }
  if (ConvertGraphInputsAndConsts(subgraph_tf_node_vec, sub_func_graph, anf_sub_node_map, false) != RET_OK) {
    MS_LOG(ERROR) << "Convert subgraph consts failed";
    return RET_ERROR;
  }

  // hardcode subgraph inputs name
  for (size_t j = 0; j < sub_graph_inputs.size(); j++) {
    sub_graph_inputs[j]->set_name(sub_graph_name + "_input_" + std::to_string(j) + "_parameter");
  }

  return RET_OK;
}

STATUS TFModelParser::ConvertSubgraphOutputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                                             const std::unordered_map<std::string, AnfNodePtr> &anf_sub_node_map,
                                             const tensorflow::FunctionDef &tf_sub_fuction,
                                             const FuncGraphPtr &sub_func_graph) {
  MSLITE_CHECK_PTR(sub_func_graph);
  MSLITE_CHECK_PTR(tf_sub_node_map);
  auto &tf_sub_signature = tf_sub_fuction.signature();
  auto &sub_graph_name = tf_sub_signature.name();

  std::vector<AnfNodePtr> sub_output_nodes;
  auto &subgraph_ret = tf_sub_fuction.ret();
  for (auto &output_arg : tf_sub_signature.output_arg()) {
    auto &signature_name = output_arg.name();
    if (subgraph_ret.find(signature_name) == subgraph_ret.end()) {
      MS_LOG(ERROR) << "can't find signature_name: " << signature_name;
      return RET_ERROR;
    }
    auto t = subgraph_ret.find(signature_name);
    MS_LOG(INFO) << "subret " << t->first << " " << t->second;
    auto tf_output_name = TensorFlowUtils::GetFlattenNodeName(t->second);
    AnfNodePtr anf_node = nullptr;
    if (tf_sub_node_map->find(tf_output_name) == tf_sub_node_map->end()) {
      anf_node = GetAnfNode(tf_output_name, anf_sub_node_map);
    } else {
      auto tf_real_name = GetOriginInputName(*tf_sub_node_map->at(tf_output_name), *tf_sub_node_map);
      anf_node = GetAnfNode(tf_real_name, anf_sub_node_map);
    }
    if (anf_node == nullptr) {
      MS_LOG(ERROR) << "can't find anf node,tf node flatten name" << tf_output_name;
      return RET_ERROR;
    }
    sub_output_nodes.push_back(anf_node);
  }
  if (MakeAnfGraphOutputs(sub_output_nodes, sub_func_graph) != RET_OK) {
    MS_LOG(ERROR) << "cmake anf graph outputs node error";
    return RET_ERROR;
  }

  // hardcode subgraph outputs name
  for (size_t j = 0; j < sub_output_nodes.size(); j++) {
    if (utils::isa<CNodePtr>(sub_output_nodes[j])) {
      sub_output_nodes[j]->cast<CNodePtr>()->set_fullname_with_scope(sub_graph_name + "_output_" + std::to_string(j) +
                                                                     "_cnode");
    } else if (utils::isa<ParameterPtr>(sub_output_nodes[j])) {
      sub_output_nodes[j]->cast<ParameterPtr>()->set_name(sub_graph_name + "_output_" + std::to_string(j) +
                                                          "_parameter");
    }
  }
  return RET_OK;
}

void TFModelParser::UpdateMap(const CNodePtr &cnode, const FuncGraphPtr &sub_func_graph,
                              const std::string &sub_graph_name) {
  CHECK_NULL_RETURN_VOID(cnode);
  CHECK_NULL_RETURN_VOID(sub_func_graph);
  if (opt::CheckPrimitiveType(cnode, prim::kPrimWhile)) {
    if (find(while_cond_branch_name_.begin(), while_cond_branch_name_.end(), sub_graph_name) !=
        while_cond_branch_name_.end()) {
      while_cond_map_[cnode] = sub_func_graph;
    } else {
      while_body_map_[cnode] = sub_func_graph;
    }
  }
  if (opt::CheckPrimitiveType(cnode, prim::kPrimIf)) {
    if (find(if_then_branch_name_.begin(), if_then_branch_name_.end(), sub_graph_name) != if_then_branch_name_.end()) {
      if_then_map_[cnode] = sub_func_graph;
    } else {
      if_else_map_[cnode] = sub_func_graph;
    }
  }
}

STATUS TFModelParser::ConvertSubgraph() {
  bool success_flag = true;
  std::queue<int> tf_graph_index_q{};
  for (int i = 0; i < tf_root_graph_->library().function_size(); i++) {
    tf_graph_index_q.push(i);
  }
  int max_move_times = tf_root_graph_->library().function_size();
  // key is graph index, value is the time move to the queue back.
  std::unordered_map<int, int> move_times_map{};
  while (!tf_graph_index_q.empty()) {
    auto cur_index = tf_graph_index_q.front();
    tf_graph_index_q.pop();
    auto &tf_sub_fuction = tf_root_graph_->library().function(cur_index);
    auto &tf_sub_signature = tf_sub_fuction.signature();
    auto input_arg_size = tf_sub_signature.input_arg_size();
    auto &sub_graph_name = tf_sub_signature.name();
    CNodePtr cnode = nullptr;
    if (function_while_map_.count(sub_graph_name)) {
      cnode = function_while_map_[sub_graph_name]->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(cnode != nullptr && static_cast<int>(cnode->size()) == input_arg_size + 1, RET_ERROR,
                        "while cnode  not equal input arg size!");
    } else if (function_if_map_.count(sub_graph_name)) {
      cnode = function_if_map_[sub_graph_name]->cast<CNodePtr>();
      if (cnode == nullptr || static_cast<int>(cnode->size()) != input_arg_size + 2) {
        MS_LOG(ERROR) << "if cnode  not equal input arg size";
        return RET_ERROR;
      }
    } else {
      if (move_times_map.find(cur_index) == move_times_map.end()) {
        move_times_map[cur_index] = 1;
        tf_graph_index_q.push(cur_index);
      } else {
        move_times_map[cur_index]++;
        if (move_times_map[cur_index] >= max_move_times) {
          MS_LOG(WARNING) << "This function is not belong to any while op or if op, graph name: " << sub_graph_name;
        } else {
          tf_graph_index_q.push(cur_index);
        }
      }
      continue;
    }
    FuncGraphPtr sub_func_graph = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_RET(sub_func_graph != nullptr, RET_ERROR);
    sub_func_graph->set_attr("graph_name", MakeValue(sub_graph_name));
    sub_func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeTf)));
    std::unordered_map<std::string, AnfNodePtr> anf_sub_node_map;
    std::map<std::string, const tensorflow::NodeDef *> tf_sub_node_map;

    if (ConvertSubgraphInputs(&tf_sub_node_map, &anf_sub_node_map, tf_sub_fuction, cnode, sub_func_graph) != RET_OK) {
      MS_LOG(ERROR) << "Convert subgraph inputs failed.";
      return RET_ERROR;
    }

    // convert sub graph ops
    STATUS status = RET_OK;
    for (int j = 0; j < tf_sub_fuction.node_def_size(); j++) {
      auto &node_def = tf_sub_fuction.node_def(j);
      status = ConvertOps(node_def, tf_sub_node_map, sub_func_graph, &anf_sub_node_map);
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Convert subgraph ops failed.";
        success_flag = false;
      }
    }
    if (!success_flag) {
      MS_LOG(ERROR) << "Convert subgraph is failed.";
      return RET_ERROR;
    }

    if (ConvertSubgraphOutputs(&tf_sub_node_map, anf_sub_node_map, tf_sub_fuction, sub_func_graph) != RET_OK) {
      MS_LOG(ERROR) << "Convert subgraph outputs failed.";
      return RET_ERROR;
    }

    // add while cond body function to while node input
    UpdateMap(cnode, sub_func_graph, sub_graph_name);
  }

  if (ControlFlowNodePostProcess(while_cond_map_, while_body_map_) != RET_OK ||
      (ControlFlowNodePostProcess(if_then_map_, if_else_map_) != RET_OK)) {
    MS_LOG(ERROR) << "while/if node post process failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TFModelParser::ControlFlowNodePostProcess(const std::map<CNodePtr, FuncGraphPtr> &first_func_map,
                                                 const std::map<CNodePtr, FuncGraphPtr> &second_func_map) {
  if (first_func_map.size() != second_func_map.size()) {
    MS_LOG(ERROR) << "first_func_map.size(): " << first_func_map.size()
                  << " second_func_map.size(): " << second_func_map.size();
    return RET_ERROR;
  }
  auto main_graph = ConvertGraph(res_graph_);
  MS_CHECK_TRUE_RET(main_graph != nullptr, RET_ERROR);
  static auto root_func_manager = Manage(main_graph);
  MS_CHECK_TRUE_RET(root_func_manager != nullptr, RET_ERROR);

  for (auto &kv : first_func_map) {
    auto control_flow_node = kv.first;
    MS_CHECK_TRUE_RET(control_flow_node != nullptr, RET_ERROR);
    auto func_graph = control_flow_node->func_graph();
    MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);

    auto &first_sub_graph = kv.second;
    auto &second_sub_graph = second_func_map.at(control_flow_node);
    CHECK_NULL_RETURN(control_flow_node);
    CHECK_NULL_RETURN(first_sub_graph);
    CHECK_NULL_RETURN(second_sub_graph);
    first_sub_graph->set_manager(root_func_manager);
    second_sub_graph->set_manager(root_func_manager);
    auto first_value_node = NewValueNode(first_sub_graph);
    CHECK_NULL_RETURN(first_value_node);
    auto second_value_node = NewValueNode(second_sub_graph);
    CHECK_NULL_RETURN(second_value_node);
    auto inputs = control_flow_node->inputs();
    inputs.insert(inputs.begin() + 1, {first_value_node, second_value_node});
    auto new_node = func_graph->NewCNode(inputs);  // must create new node, otherwise node_users won't update
    if (new_node == nullptr) {
      MS_LOG(ERROR) << "new node failed";
      return RET_ERROR;
    }
    new_node->set_abstract(control_flow_node->abstract()->Clone());
    new_node->set_fullname_with_scope(control_flow_node->fullname_with_scope());
    if (!root_func_manager->Replace(control_flow_node, new_node)) {
      MS_LOG(ERROR) << "replace new node failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertInputNodes(const tensorflow::NodeDef &node_def,
                                        const std::vector<std::string> &input_names,
                                        const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                                        const std::unordered_map<std::string, AnfNodePtr> &anf_node_map,
                                        std::vector<AnfNodePtr> *inputs,
                                        std::vector<std::string> *input_name_not_found) {
  CHECK_NULL_RETURN(inputs);
  CHECK_NULL_RETURN(input_name_not_found);
  // parse inputs
  for (size_t j = 0; j < input_names.size(); j++) {
    std::string input_name = input_names[j];  // input may be produced by multi-outputs node
    // subgraph input name x:output:index,need flatten
    auto flatten_input_name = TensorFlowUtils::GetFlattenNodeName(input_name);
    if (tf_node_map.find(flatten_input_name) != tf_node_map.end()) {
      auto input_node = tf_node_map.at(flatten_input_name);
      flatten_input_name = GetOriginInputName(*input_node, tf_node_map);
    }
    auto input = GetAnfNode(flatten_input_name, anf_node_map);
    if (input == nullptr) {
      MS_LOG(WARNING) << node_def.name() << " input " << j << ": " << input_name << " can't find parsed in_nodes";
      (*input_name_not_found).push_back(flatten_input_name);
    }
    inputs->emplace_back(input);
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertOutputTensor(const tensorflow::NodeDef &op, const CNodePtr &anf_node,
                                          std::unordered_map<std::string, AnfNodePtr> *anf_node_map,
                                          const FuncGraphPtr &anf_graph, int output_size) {
  MSLITE_CHECK_PTR(anf_node);
  MSLITE_CHECK_PTR(anf_node_map);
  MSLITE_CHECK_PTR(anf_graph);
  if (IsTensorListOp(anf_node) && output_size != 1) {
    MS_LOG(ERROR) << "tensorlist output op output_size !=1";
    return RET_ERROR;
  }
  if (output_size == 0) {
    return RET_OK;
  } else if (output_size == 1) {
    auto type = kNumberTypeFloat32;
    if (IsTensorListOp(anf_node)) {
      type = kObjectTypeTensorType;
    }
    auto abstract_tensor = CreateTensorAbstract({}, type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    anf_node->set_abstract(abstract_tensor);
    anf_node_map->insert(std::pair(op.name(), anf_node));
  } else {
    AbstractBasePtrList abstract_list;
    for (int output_idx = 0; output_idx < output_size; output_idx++) {
      auto abstract_tensor = CreateTensorAbstract({}, kNumberTypeFloat32);
      if (abstract_tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      abstract_list.emplace_back(abstract_tensor);
      auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "new TupleGetItem failed";
        return RET_NULL_PTR;
      }
      auto prim_c = tuple_get_item_prim_ptr->GetPrim();
      CHECK_NULL_RETURN(prim_c);
      auto tuple_get_item_prim = NewValueNode(prim_c);
      CHECK_NULL_RETURN(tuple_get_item_prim);
      auto get_item_value = NewValueNode(MakeValue<int64_t>(output_idx));
      CHECK_NULL_RETURN(get_item_value);
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, anf_node, get_item_value};
      CNodePtr get_item_cnode = anf_graph->NewCNode(inputs);
      CHECK_NULL_RETURN(get_item_cnode);
      std::string output_item_name = anf_node->fullname_with_scope() + "_getitem_" + std::to_string(output_idx);
      auto get_item_abstract = CreateTensorAbstract({}, kNumberTypeFloat32);
      if (get_item_abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      get_item_cnode->set_abstract(get_item_abstract);
      get_item_cnode->set_fullname_with_scope(output_item_name);
      anf_node_map->insert(std::pair(op.name() + ":" + std::to_string(output_idx), get_item_cnode));
    }
    anf_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return RET_OK;
}

STATUS TFModelParser::RecordNullInput(const CNodePtr &node, const std::vector<std::string> &input_name_not_found) {
  CHECK_NULL_RETURN(node);
  nodes_with_null_input_.emplace_back(node, input_name_not_found);
  return RET_OK;
}

STATUS TFModelParser::ConnectNullInput() {
  for (auto &it : nodes_with_null_input_) {
    auto &cnode = it.first;
    auto &input_name_not_found = it.second;
    auto &inputs = cnode->inputs();
    int i = 0;
    for (size_t j = 0; j < inputs.size(); ++j) {
      if (inputs[j] == nullptr) {
        cnode->set_input(j, GetAnfNode(input_name_not_found[i], anf_root_node_map_));
        ++i;
      }
    }
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertOps(const tensorflow::NodeDef &node_def,
                                 const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                                 const FuncGraphPtr &func_graph_ptr,
                                 std::unordered_map<std::string, AnfNodePtr> *anf_node_map) {
  MS_ASSERT(node_def != nullptr);
  MSLITE_CHECK_PTR(func_graph_ptr);
  MSLITE_CHECK_PTR(anf_node_map);
  STATUS status = RET_OK;
  const auto &op_type = node_def.op();
  if (TensorFlowUtils::OutputIsInputOp(op_type)) {
    return RET_OK;
  } else if (op_type == "Placeholder" || op_type == "Const") {
    node_output_num_[node_def.name()] = 1;
    return RET_OK;
  }
  MS_LOG(INFO) << "parse op : " << op_type;
  ops::PrimitiveCPtr primitive_c;
  auto node_parser = registry::NodeParserRegistry::GetNodeParser(kFmkTypeTf, op_type);
  int output_size;
  std::vector<std::string> input_names;
  if (node_parser != nullptr) {
    auto parser_result = node_parser->Parse(node_def, tf_node_map, &input_names, &output_size);
    if (parser_result == nullptr) {
      MS_LOG(ERROR) << "Node parse result nullptr!Please check system memory.";
      return RET_ERROR;
    }
    primitive_c = parser_result->GetPrim();
  } else {
    auto node_parser_builtin = TFNodeParserRegistry::GetInstance()->GetNodeParser(op_type);
    if (node_parser_builtin == nullptr) {
      NotSupportOp::GetInstance()->InsertOp(op_type);
      MS_LOG(ERROR) << "cannot find node parser: " << node_def.name() << " in "
                    << func_graph_ptr->get_attr("graph_name")->ToString();
      return RET_NOT_FIND_OP;
    }
    primitive_c = node_parser_builtin->Parse(node_def, tf_node_map, &input_names, &output_size);
  }
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "node " << op_type << " parser failed!";
    return RET_ERROR;
  }
  node_output_num_[node_def.name()] = output_size;
  for (int i = 0; i < output_size; i++) {
    node_output_num_[node_def.name() + ":" + std::to_string(i)] = 1;
  }
  auto value_node = NewValueNode(primitive_c);
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "value_node is nullptr";
    return RET_ERROR;
  }

  std::vector<AnfNodePtr> inputs = {value_node};
  std::vector<std::string> input_name_not_found{};
  status = ConvertInputNodes(node_def, input_names, tf_node_map, *anf_node_map, &inputs, &input_name_not_found);
  if (status != RET_OK) {
    return status;
  }
  // control_depends are not processed currently
  auto anf_node = func_graph_ptr->NewCNode(inputs);
  CHECK_NULL_RETURN(anf_node);
  anf_node->set_fullname_with_scope(node_def.name());
  status = ProcessControlFlowOp(anf_node, op_type, node_def);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ProcessControlFlowOp failed.";
    return RET_ERROR;
  }

  if (!input_name_not_found.empty()) {
    status = RecordNullInput(anf_node, input_name_not_found);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "RecordNullInput for " << anf_node->fullname_with_scope() << " failed.";
      return status;
    }
  }

  status = ConvertOutputTensor(node_def, anf_node, anf_node_map, func_graph_ptr, output_size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert output tensors for " << anf_node->fullname_with_scope() << " failed.";
    return status;
  }

  return status;
}

bool TFModelParser::IsEmptyTfFunction(const CNodePtr &anf_node, std::string branch_name) {
  for (int i = 0; i < tf_root_graph_->library().function_size(); i++) {
    auto &tf_sub_fuction = tf_root_graph_->library().function(i);
    auto &tf_sub_signature = tf_sub_fuction.signature();
    auto &sub_graph_name = tf_sub_signature.name();

    if (branch_name != sub_graph_name) {
      continue;
    }
    auto &tf_sub_signature_output_arg = tf_sub_signature.output_arg();
    if (tf_sub_signature_output_arg.size() != 1) {
      return false;
    }
    auto &tf_sub_signature_output_name = tf_sub_signature_output_arg.Get(0).name();
    auto input_arg_size = tf_sub_signature.input_arg_size();
    if (tf_sub_fuction.node_def_size() == 0) {
      for (int index = 0; index < input_arg_size; index++) {
        auto &input_arg = tf_sub_signature.input_arg(index);
        if (input_arg.name() == tf_sub_signature_output_name &&
            ineffective_if_op_map_.find(anf_node) == ineffective_if_op_map_.end()) {
          ineffective_if_op_map_[anf_node] = index + C2NUM;
          return true;
        }
      }
    } else if (tf_sub_fuction.node_def_size() == 1) {
      auto &node_def = tf_sub_fuction.node_def(0);
      if (!TensorFlowUtils::OutputIsInputOp(node_def.name())) {
        return false;
      }
      for (int index = 0; index < input_arg_size; index++) {
        auto &input_arg = tf_sub_signature.input_arg(index);
        if (input_arg.name() == node_def.input(0)) {
          auto output_name = node_def.name();
          std::transform(output_name.begin(), output_name.end(), output_name.begin(), ::tolower);
          if (output_name == tf_sub_signature_output_name &&
              ineffective_if_op_map_.find(anf_node) == ineffective_if_op_map_.end()) {
            ineffective_if_op_map_[anf_node] = index + C2NUM;
            return true;
          }
        }
      }
    }
  }
  return false;
}  // namespace lite

bool TFModelParser::IsIneffectiveIfOp(const CNodePtr &anf_node, const string &op_type,
                                      const tensorflow::NodeDef &node_def) {
  if (op_type != "If") {
    return false;
  }
  lite::DataInfo if_cond_info;
  auto if_cond = anf_node->input(1);
  if (if_cond == nullptr) {
    return false;
  }
  int status = lite::RET_ERROR;
  if (if_cond->isa<Parameter>()) {
    status = lite::FetchDataFromParameterNode(anf_node, 1, converter::kFmkTypeMs, &if_cond_info, true);
  } else if (utils::isa<CNodePtr>(if_cond)) {
    auto input_cnode = if_cond->cast<CNodePtr>();
    if (input_cnode == nullptr) {
      return false;
    }
    if (!opt::CheckPrimitiveType(input_cnode, prim::kPrimConstant)) {
      return false;
    }

    auto input_cnode_in1 = input_cnode->input(1);
    if (input_cnode_in1 == nullptr) {
      return false;
    }
    if (input_cnode_in1->isa<Parameter>()) {
      status = lite::FetchDataFromParameterNode(input_cnode, 1, converter::kFmkTypeMs, &if_cond_info, true);
    } else if (input_cnode_in1->isa<ValueNode>()) {
      status = lite::FetchDataFromValueNode(input_cnode, 1, converter::kFmkTypeMs, false, &if_cond_info, true);
    }
  }

  if (status != lite::RET_OK) {
    return false;
  }
  if (static_cast<TypeId>(if_cond_info.data_type_) == kNumberTypeBool && if_cond_info.data_.size() == 1) {
    tensorflow::AttrValue attr_value;
    if (static_cast<bool>(if_cond_info.data_[0])) {
      if (TensorFlowUtils::FindAttrValue(node_def, "then_branch", &attr_value)) {
        auto then_name = attr_value.func().name();
        if (IsEmptyTfFunction(anf_node, then_name)) {
          return true;
        }
      }
    } else {
      if (TensorFlowUtils::FindAttrValue(node_def, "else_branch", &attr_value)) {
        auto else_name = attr_value.func().name();
        if (IsEmptyTfFunction(anf_node, else_name)) {
          return true;
        }
      }
    }
  }

  return false;
}

STATUS TFModelParser::ProcessControlFlowOp(const CNodePtr &anf_node, const string &op_type,
                                           const tensorflow::NodeDef &node_def) {
  MSLITE_CHECK_PTR(anf_node);
  if (IsIneffectiveIfOp(anf_node, op_type, node_def)) {
    return RET_OK;
  }
  if (op_type == "StatelessWhile" || op_type == "While") {
    MS_LOG(INFO) << "find while node:" << node_def.name();
    tensorflow::AttrValue attr_value;
    if (TensorFlowUtils::FindAttrValue(node_def, "body", &attr_value)) {
      auto body_name = attr_value.func().name();
      function_while_map_[body_name] = anf_node;
      MS_LOG(DEBUG) << "parse body name:" << body_name;
    }
    if (TensorFlowUtils::FindAttrValue(node_def, "cond", &attr_value)) {
      auto cond_name = attr_value.func().name();
      function_while_map_[cond_name] = anf_node;
      while_cond_branch_name_.push_back(cond_name);
      MS_LOG(DEBUG) << "parse cond name:" << cond_name;
    }
  } else if (op_type == "StatelessIf" || op_type == "If") {
    MS_LOG(INFO) << "find if node:" << node_def.name();
    tensorflow::AttrValue attr_value;
    if (TensorFlowUtils::FindAttrValue(node_def, "then_branch", &attr_value)) {
      auto then_name = attr_value.func().name();
      if_then_branch_name_.push_back(then_name);
      function_if_map_[then_name] = anf_node;
      MS_LOG(DEBUG) << "parse then name:" << then_name;
    }
    if (TensorFlowUtils::FindAttrValue(node_def, "else_branch", &attr_value)) {
      auto else_name = attr_value.func().name();
      function_if_map_[else_name] = anf_node;
      MS_LOG(DEBUG) << "parse else name:" << else_name;
    }
  }
  return RET_OK;
}

std::set<std::string> TFModelParser::GetAllNodeInputs() {
  std::set<std::string> all_node_inputs;
  for (auto &node : tf_root_graph_nodes_vec_) {
    for (int i = 0; i < node->input_size(); ++i) {
      all_node_inputs.insert(TensorFlowUtils::GetNodeName(node->input(i)));
      auto input_name = node->input(i);
      if (input_name[0] == '^') {
        input_name.erase(0, 1);
      }
      all_node_inputs.insert(input_name);
    }
  }
  return all_node_inputs;
}

STATUS TFModelParser::GetGraphOutputNames(std::vector<AnfNodePtr> *output_nodes) {
  MS_CHECK_TRUE_RET(output_nodes->empty(), RET_ERROR);
  std::set<std::string> all_node_inputs = GetAllNodeInputs();
  for (auto &node : tf_root_graph_nodes_vec_) {
    if (node->op() == "Assert") {
      continue;
    }
    auto it = all_node_inputs.find(node->name());
    if (it != all_node_inputs.end() || node->input_size() <= 0) {  // output node not constraint to Identity
      continue;
    }
    auto origin_name = GetOriginInputName(*(node), tf_root_graph_nodes_);
    // node with multiple outputs has been changed to tupleGetItem, and the original name changes to be name:idx.
    for (int i = 0; i < node_output_num_[origin_name]; i++) {
      auto anf_node = GetAnfNode(origin_name, anf_root_node_map_, i);
      if (anf_node == nullptr) {
        MS_LOG(ERROR) << "can't find anf node: " << origin_name;
        return RET_ERROR;
      }
      output_nodes->push_back(anf_node);
      if (TensorFlowUtils::OutputIsInputOp(node->op())) {
        auto tmp_node = node;
        bool found_input = true;
        while (tmp_node->name().empty() && TensorFlowUtils::OutputIsInputOp(tmp_node->op())) {
          auto flatten_input_name = TensorFlowUtils::GetFlattenNodeName(tmp_node->input(0));
          if (tf_root_graph_nodes_.find(flatten_input_name) != tf_root_graph_nodes_.end()) {
            tmp_node = tf_root_graph_nodes_.at(flatten_input_name);
          } else {
            found_input = false;
            break;
          }
        }
        origin_name = found_input ? tmp_node->name() : origin_name;
      }
      graph_output_names_.push_back(origin_name);
    }
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertRootGraphOutputs() {
  // because output of intermediate node in anf graph may also be output tensors, we search output tensors in
  // tf_root_graph_nodes_ but not anf_root_node_map_
  std::vector<AnfNodePtr> output_nodes;
  auto status = GetGraphOutputNames(&output_nodes);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get graph outputs node error";
    return status;
  }
  auto func_graph = ConvertGraph(res_graph_);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "unc graph is invalid.";
    return RET_ERROR;
  }
  status = MakeAnfGraphOutputs(output_nodes, func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "make anf graph outputs node error";
    return status;
  }
  // save original output tensor names.
  ConverterInnerContext::GetInstance()->SetGraphOutputTensorNames(graph_output_names_);
  return RET_OK;
}
STATUS TFModelParser::MakeAnfGraphOutputs(const std::vector<AnfNodePtr> &output_nodes, const FuncGraphPtr &anf_graph) {
  if (output_nodes.empty() || anf_graph == nullptr) {
    MS_LOG(ERROR) << "anf output nodes empty or  null anf graph";
    return RET_ERROR;
  }
  if (output_nodes.size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs = output_nodes;
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
    CHECK_NULL_RETURN(make_tuple_prim_c);
    auto make_tuple_prim = NewValueNode(make_tuple_prim_c);
    CHECK_NULL_RETURN(make_tuple_prim);
    make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
    auto make_tuple_cnode = anf_graph->NewCNode(make_tuple_inputs);
    CHECK_NULL_RETURN(make_tuple_cnode);
    make_tuple_cnode->set_fullname_with_scope("return_tuple");

    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto return_prim_c = return_prim_ptr->GetPrim();
    CHECK_NULL_RETURN(return_prim_c);
    auto value_node = NewValueNode(return_prim_c);
    CHECK_NULL_RETURN(value_node);
    std::vector<AnfNodePtr> op_inputs = {value_node, make_tuple_cnode};
    auto cnode = anf_graph->NewCNode(op_inputs);
    CHECK_NULL_RETURN(cnode);
    cnode->set_fullname_with_scope("Return");
    anf_graph->set_return(cnode);
  } else {
    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto return_prim_c = return_prim_ptr->GetPrim();
    CHECK_NULL_RETURN(return_prim_c);
    auto value_node = NewValueNode(return_prim_c);
    CHECK_NULL_RETURN(value_node);
    std::vector<AnfNodePtr> op_inputs{value_node, output_nodes.front()};
    auto return_cnode = anf_graph->NewCNode(op_inputs);
    CHECK_NULL_RETURN(return_cnode);
    return_cnode->set_fullname_with_scope("Return");
    anf_graph->set_return(return_cnode);
  }
  return RET_OK;
}

int TFModelParser::TF2AnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs,
                                std::map<AnfNodePtr, int> *ineffective_if_op_map) {
  MSLITE_CHECK_PTR(ineffective_if_op_map);
  for (const auto &func_graph : all_func_graphs) {
    if (!TfInputAdjust::Adjust(func_graph)) {
      MS_LOG(ERROR) << "Do TfInputAdjust failed.";
      return RET_ERROR;
    }
    auto einsum_adjust = std::make_shared<EinsumAdjust>();
    MS_CHECK_TRUE_MSG(einsum_adjust != nullptr, RET_NULL_PTR, "einsum_adjust is nullptr.");
    if (!einsum_adjust->Adjust(func_graph)) {
      MS_LOG(ERROR) << "Adjust einsum failed!";
      return RET_ERROR;
    }
    auto remove_ineffective_control_flow = std::make_shared<RemoveIneffectiveControlFlow>();
    MS_CHECK_TRUE_RET(remove_ineffective_control_flow != nullptr, RET_ERROR);
    if (!remove_ineffective_control_flow->Run(func_graph, ineffective_if_op_map)) {
      MS_LOG(ERROR) << "Do RemoveIneffectiveControlFlow failed.";
      return RET_ERROR;
    }
    auto functionalize_control_op_pass = std::make_shared<opt::FunctionalizeControlOpPass>();
    MS_CHECK_TRUE_RET(functionalize_control_op_pass != nullptr, RET_ERROR);
    if (!functionalize_control_op_pass->Run(func_graph)) {
      MS_LOG(ERROR) << "functionalize control op pass failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
    auto fake_quant_adjust = std::make_shared<TFFakeQuantAdjust>();
    if (!fake_quant_adjust->Adjust(func_graph)) {
      MS_LOG(ERROR) << "tf fake quant adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REG_MODEL_PARSER(kFmkTypeTf, LiteModelParserCreator<TFModelParser>)
}  // namespace lite
}  // namespace mindspore
