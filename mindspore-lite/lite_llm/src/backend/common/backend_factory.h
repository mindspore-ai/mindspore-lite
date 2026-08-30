/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#ifndef MSLLM_BACKEND_FACTORY_H
#define MSLLM_BACKEND_FACTORY_H

#include <functional>
#include <memory>

#include "backend/common/backend.h"

namespace mslite_llm {

/// Factory that produces a Backend instance. The production default is the
/// NNRT backend; tests override it with a FakeBackend to drive the C API
/// state machine without NPU hardware.
using BackendFactory = std::function<std::unique_ptr<Backend>()>;

/// Test seam: set a backend factory override. Pass a null factory to restore
/// the production default. Not part of the public C API.
void SetBackendFactory(BackendFactory factory);

/// Returns the current override (null when the production default is active).
const BackendFactory &GetBackendFactoryOverride();

/// Registers a backend creator for a backend type; called by
/// MSLLM_REGISTER_BACKEND during static initialisation. Returns false (and
/// keeps the first registration) when the type is already registered. The
/// registry is a function-local static, so registration order across
/// translation units is irrelevant.
bool RegisterBackend(MSLlmBackendType type, BackendFactory factory);

/// Creates a backend by type: honours the test override first, then the
/// registry. Returns nullptr when neither is available.
std::unique_ptr<Backend> CreateBackend(MSLlmBackendType type);

}  // namespace mslite_llm

/// Registers a backend implementation at static-init time, mirroring the
/// MindSpore Lite REGISTER_KERNEL pattern: an anonymous-namespace static
/// object whose construction registers the creator. Each backend backend
/// calls it once, e.g. MSLLM_REGISTER_BACKEND(MSLLM_BACKEND_NNRT,
/// CreateNNRTBackend).
#define MSLLM_REGISTER_BACKEND(backend_type, creator)                                                           \
  namespace {                                                                                                   \
  static const bool g_msllm_##backend_type##_registered = ::mslite_llm::RegisterBackend(backend_type, creator); \
  }  // namespace

#endif  // MSLLM_BACKEND_FACTORY_H
