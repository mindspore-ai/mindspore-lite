/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_UTILS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_UTILS_H_

#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
TypeId GetSingleNodeOutputTypeId(const mindspore::AnfNodePtr &node);
AnfNodePtr GetCastNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node, const TypeId &dst_type_id);
AnfNodePtr GetReshapeNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node,
                          const ShapeVector &dst_shape);
AnfNodePtr GetBroadcastToNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node,
                              const ShapeVector &dst_shape);
AnfNodePtr GetMatMulNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &input,
                         const mindspore::AnfNodePtr &other, const bool &transpose_a, const bool &transpose_b);
template <typename T>
ValueNodePtr GetCastedScalar(const T number, const TypeId &dst_type_id) {
  ValuePtr value_ptr = nullptr;
  switch (dst_type_id) {
    case kNumberTypeBool:
      value_ptr = MakeValue<bool>(static_cast<bool>(number));
      break;
    case kNumberTypeInt16:
      value_ptr = MakeValue<int16_t>(static_cast<int16_t>(number));
      break;
    case kNumberTypeUInt16:
      value_ptr = MakeValue<uint16_t>(static_cast<uint16_t>(number));
      break;
    case kNumberTypeInt8:
      value_ptr = MakeValue<int8_t>(static_cast<int8_t>(number));
      break;
    case kNumberTypeUInt8:
      value_ptr = MakeValue<uint8_t>(static_cast<uint8_t>(number));
      break;
    case kNumberTypeInt32:
      value_ptr = MakeValue<int32_t>(static_cast<int32_t>(number));
      break;
    case kNumberTypeUInt32:
      value_ptr = MakeValue<uint32_t>(static_cast<uint32_t>(number));
      break;
    case kNumberTypeInt64:
      value_ptr = MakeValue<int64_t>(static_cast<int64_t>(number));
      break;
    case kNumberTypeUInt64:
      value_ptr = MakeValue<uint64_t>(static_cast<uint64_t>(number));
      break;
    case kNumberTypeFloat32:
      value_ptr = MakeValue<float>(static_cast<float>(number));
      break;
    case kNumberTypeFloat64:
      value_ptr = MakeValue<double>(static_cast<double>(number));
      break;
    default:
      MS_LOG(ERROR) << "Not support scalar type:" << dst_type_id;
      return nullptr;
  }
  auto value_node = NewValueNode(value_ptr);
  value_node->set_abstract(value_ptr->ToAbstract());
  return value_node;
}
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_UTILS_H_
