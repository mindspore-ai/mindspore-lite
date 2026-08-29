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
#include "backend/common/backend_factory.h"

#include <unordered_map>

namespace mslite_llm {
namespace {

BackendFactory g_backend_factory_override;

std::unordered_map<MSLlmBackendType, BackendFactory> &BackendRegistry() {
  static std::unordered_map<MSLlmBackendType, BackendFactory> registry;
  return registry;
}

}  // namespace

void SetBackendFactory(BackendFactory factory) { g_backend_factory_override = std::move(factory); }

const BackendFactory &GetBackendFactoryOverride() { return g_backend_factory_override; }

bool RegisterBackend(MSLlmBackendType type, BackendFactory factory) {
  return BackendRegistry().emplace(type, std::move(factory)).second;
}

std::unique_ptr<Backend> CreateBackend(MSLlmBackendType type) {
  if (const auto &override = GetBackendFactoryOverride()) {
    return override();
  }
  auto &registry = BackendRegistry();
  auto it = registry.find(type);
  return it == registry.end() ? nullptr : it->second();
}

}  // namespace mslite_llm
