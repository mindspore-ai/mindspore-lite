/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/mslite_model_state.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

/**
 * Backend that demonstrates using in-process C-API to execute inferences
 * within the backend.
 */

namespace triton {
namespace backend {
namespace mslite {
extern "C" {
/**
 * Triton calls TRITONBACKEND_Initialize when a backend is loaded into
 * Triton to allow the backend to create and initialize any state that
 * is intended to be shared across all models and model instances that
 * use the backend. The backend should also verify version
 * compatibility with Triton in this function.
 */
TRITONSERVER_Error *TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend) {
  const char *cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  /**
   * Check the backend API version that Triton supports vs. what this
   * backend was compiled against. Make sure that the Triton major
   * version is the same and the minor version is >= what this backend
   * uses.
   */
  uint32_t api_version_major;
  uint32_t api_version_minor;
  RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Triton TRITONBACKEND API version: ") +
                                      std::to_string(api_version_major) + "." + std::to_string(api_version_minor))
                                       .c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("'") + name + "' TRITONBACKEND API version: " +
               std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." + std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                .c_str());
  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) || (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    LOG_MESSAGE(TRITONSERVER_LOG_WARN, "triton backend API version does not support this backend");
  }

  /**
   * The backend configuration may contain information needed by the
   * backend, such as tritonserver command-line arguments. This
   * backend doesn't use any such configuration but for this example
   * print whatever is available.
   */
  TRITONSERVER_Message *backend_config_message;
  RETURN_IF_ERROR(TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char *buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());

  return nullptr;  // success
}

TRITONSERVER_Error *TRITONBACKEND_Finalize(TRITONBACKEND_Backend *backend) {
  return nullptr;  // success
}

/**
 * Implementing TRITONBACKEND_ModelInitialize is optional. The backend
 * should initialize any state that is intended to be shared across
 * all instances of the model.
 */
TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model) {
  const char *const_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &const_name));
  uint64_t version;
  std::string name(const_name);
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));
  LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("TRITONBACKEND_ModelInitialize: ") + name + " ( version " + std::to_string(version) + ")").c_str());
  ModelState *model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void *>(model_state)));

  return nullptr;  // success
}
/**
 * Implementing TRITONBACKEND_ModelFinalize is optional unless state
 * is set using TRITONBACKEND_ModelSetState. The backend must free
 * this state and perform any other cleanup.
 */
TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

/**
 * Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
 * backend should initialize any state that is required for a model
 * instance.
 */
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *model_instance) {
  const char *const_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(model_instance, &const_name));
  std::string name(const_name);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(model_instance, &device_id));
  TRITONSERVER_InstanceGroupKind group_kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(model_instance, &group_kind));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
               TRITONSERVER_InstanceGroupKindString(group_kind) + " device " + std::to_string(device_id) + ")")
                .c_str());
  TRITONBACKEND_Model *triton_backend_model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(model_instance, &triton_backend_model));

  void *vmodel_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(triton_backend_model, &vmodel_state));
  ModelState *model_state = reinterpret_cast<ModelState *>(vmodel_state);
  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(ModelInstanceState::Create(model_state, model_instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(model_instance, reinterpret_cast<void *>(instance_state)));

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                                                     "initialization successful ") +
                                         name + " (device " + std::to_string(device_id) + ")")
                                          .c_str());

  return nullptr;  // success
}

/**
 * Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
 * state is set using TRITONBACKEND_ModelInstanceSetState. The backend
 * must free this state and perform any other cleanup.
 */
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState *instance_state = reinterpret_cast<ModelInstanceState *>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                                       TRITONBACKEND_Request **requests, const uint32_t request_count) {
  ModelInstanceState *model_instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void **>(&model_instance_state)));
  ModelState *model_state = reinterpret_cast<ModelState *>(model_instance_state->Model());

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("model ") + model_state->Name() + ", instance " + model_instance_state->Name() +
               ", executing " + std::to_string(request_count) + " requests")
                .c_str());

  model_instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}
}  // extern "C"
}  // namespace mslite
}  // namespace backend
}  // namespace triton
