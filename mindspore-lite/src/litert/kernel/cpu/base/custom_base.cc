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

#include "src/litert/kernel/cpu/base/custom_base.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "src/litert/kernel_registry.h"
#include "nnacl_c/op_base.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore::kernel {
int CustomBaseCPUKernel::Prepare() { return RET_OK; }

int CustomBaseCPUKernel::ReSize() { return RET_OK; }

int CustomBaseCPUKernel::Run() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeInt32, PrimType_Inner_ThirdPartyModel, LiteKernelCreator<CustomBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_ThirdPartyModel, LiteKernelCreator<CustomBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimType_Inner_ThirdPartyModel, LiteKernelCreator<CustomBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt8, PrimType_Inner_ThirdPartyModel, LiteKernelCreator<CustomBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimType_Inner_ThirdPartyModel, LiteKernelCreator<CustomBaseCPUKernel>)
}  // namespace mindspore::kernel
