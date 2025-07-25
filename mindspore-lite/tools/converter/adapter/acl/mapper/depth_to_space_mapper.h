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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_DEPTH_TO_SPACE_MAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_DEPTH_TO_SPACE_MAPPER_H_

#include "tools/converter/adapter/acl/mapper/primitive_mapper.h"
#include "infer/depth_to_space.h"

namespace mindspore {
namespace lite {
using mindspore::ops::kNameDepthToSpace;

class DepthToSpaceMapper : public PrimitiveMapper {
 public:
  DepthToSpaceMapper() : PrimitiveMapper(kNameDepthToSpace) {}

  ~DepthToSpaceMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_DEPTH_TO_SPACE_MAPPER_H_
