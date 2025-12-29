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

#include "src/litert/kernel/dsp/ft04/ft04_add.h"
#include <algorithm>
#include <map>
#include <string>
#include <set>
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AddFusion;

namespace mindspore::kernel {

int AddFT04DSPKernel::Prepare() {
  auto data_type = in_tensors_[0]->data_type();
  static const std::set<TypeId> supported_types = {kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeInt16,
                                                   kNumberTypeInt32, kNumberTypeComplex64};
  if (supported_types.find(data_type) == supported_types.end()) {
    MS_LOG(ERROR) << "AddFT04DSPKernel does not support data type: " << static_cast<int>(data_type);
    return RET_ERROR;
  }
  constexpr int kAllCoresMask = 0xf;
  SetCoreMask(kAllCoresMask);
  auto ret = AddBaseDSPKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AddFT04DSPKernel prepare failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_AddFusion, DSPKernelCreator<AddFT04DSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_AddFusion, DSPKernelCreator<AddFT04DSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_AddFusion, DSPKernelCreator<AddFT04DSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_AddFusion, DSPKernelCreator<AddFT04DSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_AddFusion, DSPKernelCreator<AddFT04DSPKernel>)
}  // namespace mindspore::kernel
