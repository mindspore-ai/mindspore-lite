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

#ifndef MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_
#define MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_

#include <iostream>
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "common/common_test.h"
#include "nnacl_c/arithmetic_parameter.h"

namespace mindspore::lite::dsp::test {

class DSPCommonTest : public CommonTest {
 public:
  void InitDSPRuntime() {
    dsp_runtime_wrapper_ = new (std::nothrow) dsp::DSPRuntimeInnerWrapper();
    if (dsp_runtime_wrapper_ == nullptr) {
      MS_LOG(ERROR) << "create DSPRuntimeInnerWrapper failed.";
    }
    auto dsp_runtime = dsp_runtime_wrapper_->GetInstance();
    if (dsp_runtime->Init() != RET_OK) {
      MS_LOG(ERROR) << "Init DSP runtime failed.";
    }
    allocator_ = dsp_runtime->GetAllocator();
  }

  void UninitDSPRuntime() {
    delete dsp_runtime_wrapper_;
    dsp_runtime_wrapper_ = nullptr;
  }

 protected:
  dsp::DSPRuntimeInnerWrapper *dsp_runtime_wrapper_{nullptr};
  std::shared_ptr<DSPAllocator> allocator_;
};
}  // namespace mindspore::lite::dsp::test

#endif  // MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_DSP_DSP_TEST_H_
