/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/infer_session.h"

#include "common/ms_factory.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/delegate/plugin/litert_executor_plugin.h"
#include "extendrt/delegate/plugin/ascend_ge_executor_plugin.h"
#include "extendrt/delegate/plugin/ascend_acl_executor_plugin.h"
#include "nnacl_c/op_base.h"

namespace mindspore {
namespace {
void AscendPluginRegistration(const std::shared_ptr<AscendDeviceInfo> &ascend_device, bool use_experimental_rts) {
  constexpr auto default_npu_provider = "ge";
  auto provider = ascend_device->GetProvider();
  if (provider == default_npu_provider) {
    if (!lite::AscendGeExecutorPlugin::GetInstance().Register()) {
      MS_LOG(WARNING) << "Failed to register AscendGe plugin";
      return;
    }
  }
  if (!use_experimental_rts) {
    if (provider == "litert") {
      if (!lite::AscendAclExecutorPlugin::GetInstance().Register()) {
        MS_LOG(WARNING) << "Failed to register Ascend ACL plugin";
        return;
      }
    }
  }
}
}  // namespace
std::shared_ptr<InferSession> InferSession::CreateSession(const std::shared_ptr<Context> &context,
                                                          const ConfigInfos &config_info) {
  static const char *const env = std::getenv("ENABLE_MULTI_BACKEND_RUNTIME");
  bool use_experimental_rts = env != nullptr && strcmp(env, "on") == 0;
  HandleContext(context, use_experimental_rts);
  auto session_type = SelectSession(context, use_experimental_rts);
  return SessionRegistry::GetInstance().GetSession(session_type, context, config_info);
}

void InferSession::HandleContext(const std::shared_ptr<Context> &context, bool use_experimental_rts) {
  if (!context) {
    return;
  }
  constexpr auto default_cpu_provider = "litert";

  auto device_infos = context->MutableDeviceInfo();
  for (auto &device_info : device_infos) {
    if (!device_info) {
      MS_LOG(WARNING) << "device info is nullptr.";
      continue;
    }
    if (device_info->GetDeviceType() == kAscend) {
      auto ascend_device = device_info->Cast<AscendDeviceInfo>();
      if (!ascend_device) {
        MS_LOG(WARNING) << "not ascend device.";
        continue;
      }
      AscendPluginRegistration(ascend_device, use_experimental_rts);
      continue;
    }
    if (device_info->GetDeviceType() == kCPU) {
      auto cpu_device = device_info->Cast<CPUDeviceInfo>();
      if (!cpu_device) {
        MS_LOG(WARNING) << "cpu_device";
        continue;
      }
      auto provider = cpu_device->GetProvider();
      if (provider.empty() || provider == default_cpu_provider) {
        if (!infer::LiteRTExecutorPlugin::GetInstance().Register()) {
          MS_LOG(WARNING) << "Failed to register LiteRT plugin";
          return;
        }
        cpu_device->SetProvider(default_cpu_provider);
      }
      continue;
    }
    if (device_info->GetDeviceType() == kAllDevice) {
      // Auto Device: MSLite will detect available device and run graph/sub-graph on suitable device by its scheduler
      continue;
    }
  }
}

SessionType InferSession::SelectSession(const std::shared_ptr<Context> &context, bool use_experimental_rts) {
  if (context != nullptr) {
    if (MS_LIKELY((!use_experimental_rts))) {
      auto &device_contexts = context->MutableDeviceInfo();
      for (const auto &device_context : device_contexts) {
        MS_EXCEPTION_IF_NULL(device_context);
        if (device_context->GetDeviceType() == kAscend) {
          if (device_context->GetProvider() == "ge") {
            return kDelegateSession;
          }
          return kDelegateSession;
        }
        return kDelegateSession;
      }
    } else {
      return kDefaultSession;
    }
  }
  return kDefaultSession;
}

Status InferSession::Finalize() {
  MS_LOG(INFO) << "Finalize is only implemented in single_op_session now.";
  return kLiteError;
}
}  // namespace mindspore
