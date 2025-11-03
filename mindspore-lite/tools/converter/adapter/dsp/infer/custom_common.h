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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DSP_CUSTOM_COMMON_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DSP_CUSTOM_COMMON_H

#include <vector>
#include "include/api/types.h"
#include "include/api/status.h"

namespace mindspore::lite {
namespace common {
// verify that the tensors' shape is inferred successfully when inferring current node.
Status CheckIsDynamicShape(const std::vector<mindspore::MSTensor> &tensors);
}  // namespace common
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DSP_CUSTOM_COMMON_H
