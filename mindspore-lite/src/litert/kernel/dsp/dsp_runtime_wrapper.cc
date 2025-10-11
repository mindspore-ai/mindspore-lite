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

#include "src/litert/kernel/dsp/dsp_runtime_wrapper.h"
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "src/common/file_utils.h"
#include "src/litert/kernel/dsp/dsp_allocator.h"
#include "src/litert/kernel/dsp/dsp_runtime.h"

namespace mindspore::registry::dsp {
std::shared_ptr<Allocator> DSPRuntimeWrapper::GetAllocator() {
  lite::dsp::DSPRuntimeInnerWrapper dsp_runtime_wrapper;
  lite::dsp::DSPRuntime *dsp_runtime = dsp_runtime_wrapper.GetInstance();
  return dsp_runtime->GetAllocator();
}
}  // namespace mindspore::registry::dsp
