/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/custom_mapper.h"
#include <vector>
#include <memory>
#include <string>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "nnacl_c/op_base.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_c.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameInputNum = 4;
constexpr auto kNameCustom = "Custom";
constexpr auto kMoeInitRouting = "MoeInitRouting";
}  // namespace

CustomMapper::CustomMapper() : PrimitiveMapper(kNameCustom) {}
STATUS CustomMapper::Mapper(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "Cnode is nullptr.");
  auto prim = GetCNodePrimitive(cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_NULL_PTR, "prim is nullptr.");
  auto attr_type = prim->GetAttr("type");
  MS_CHECK_TRUE_MSG(attr_type != nullptr, lite::RET_NULL_PTR, "attr_type is nullptr.");
  auto type = GetValue<std::string>(attr_type);
  if (type != kMoeInitRouting) {
    return lite::RET_OK;
  }
  if (cnode->size() != kNameInputNum) {
    MS_LOG(ERROR) << "MoeInitRouting inputs num must be 3, node is " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto attr_active_num = prim->GetAttr("active_num");
  MS_CHECK_TRUE_MSG(attr_active_num != nullptr, lite::RET_NULL_PTR, "attr_active_num of MoeInitRouting is nullptr.");
  auto value_node = NewValueNode(attr_active_num);
  MS_CHECK_TRUE_MSG(value_node != nullptr, lite::RET_NULL_PTR, "Create a value_node failed");
  std::vector<int64_t> shape = {};
  auto abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape);
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "Create a abstract failed");
  value_node->set_abstract(abstract);
  auto inputs = cnode->inputs();
  inputs.push_back(value_node);
  cnode->set_inputs(inputs);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameCustom, CustomMapper)
}  // namespace lite
}  // namespace mindspore
