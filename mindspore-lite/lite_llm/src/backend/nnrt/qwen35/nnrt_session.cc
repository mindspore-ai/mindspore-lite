// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#include "backend/nnrt/qwen35/nnrt_session.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "manifest/artifact_file.h"
#include "neural_network_runtime/neural_network_core.h"

namespace mslite_llm::qwen35 {
namespace {

bool SelectNpuDevice(size_t *device_id, std::string *error) {
  const size_t *ids = nullptr;
  uint32_t count = 0;
  if (OH_NNDevice_GetAllDevicesID(&ids, &count) != OH_NN_SUCCESS || ids == nullptr || count == 0) {
    if (error) *error = "OH_NNDevice_GetAllDevicesID failed";
    return false;
  }
  for (uint32_t index = 0; index < count; ++index) {
    const char *name = nullptr;
    if (OH_NNDevice_GetName(ids[index], &name) == OH_NN_SUCCESS && name != nullptr &&
        std::string(name).find("HIAI_F") != std::string::npos) {
      *device_id = ids[index];
      return true;
    }
  }
  *device_id = ids[0];
  return true;
}

bool TensorName(NN_TensorDesc *desc, std::string *name) {
  const char *raw = nullptr;
  if (desc == nullptr || OH_NNTensorDesc_GetName(desc, &raw) != OH_NN_SUCCESS || raw == nullptr) {
    return false;
  }
  *name = raw;
  return true;
}

bool SetShape(NN_TensorDesc *desc, const std::vector<int32_t> &shape) {
  return desc != nullptr && !shape.empty() &&
         OH_NNTensorDesc_SetShape(desc, shape.data(), shape.size()) == OH_NN_SUCCESS;
}

std::string ShapeText(const std::vector<int32_t> &shape) {
  std::ostringstream stream;
  stream << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) stream << ',';
    stream << shape[i];
  }
  stream << ']';
  return stream.str();
}

std::string CanonicalOutputName(size_t index, size_t output_count, const std::string &descriptor_name,
                                const TensorShapes &output_shapes) {
  if (output_shapes.find(descriptor_name) != output_shapes.end()) {
    return descriptor_name;
  }
  if (output_count == 1 && output_shapes.size() == 1) {
    return output_shapes.begin()->first;
  }
  // OMG's multi-shape pass renames outputs to
  // output_N_enum_shape_graph/caseN_N. The text OMC ABI is stable by index:
  // logits, then K/V for each of the 28 layers.
  if (output_count == 57) {
    if (index == 0) return "logits";
    const size_t cache_index = index - 1;
    const size_t layer = cache_index / 2;
    return std::string("present_") + ((cache_index & 1U) == 0U ? "key_" : "value_") + std::to_string(layer);
  }
  // Resident MoE layer templates have a stable index ABI, while OMG's
  // multi-shape pass replaces their output names with generated case names.
  if (output_count == 6) {
    static constexpr const char *kResidentOutputCandidates[][2] = {{"attention_residual", nullptr},
                                                                   {"expert_input", nullptr},
                                                                   {"router_logits", nullptr},
                                                                   {"shared_expert_output", nullptr},
                                                                   {"present_conv_state", "present_key"},
                                                                   {"present_recurrent_state_bits", "present_value"}};
    for (const char *candidate : kResidentOutputCandidates[index]) {
      if (candidate != nullptr && output_shapes.find(candidate) != output_shapes.end()) {
        return candidate;
      }
    }
  }
  return descriptor_name;
}

}  // namespace

class NnrtSession::Impl {
 public:
  ~Impl() { Release(); }

  bool Load(const std::string &model_path, const std::string &weight_path, std::string *error) {
    if (executor_ != nullptr) {
      if (error) *error = "NNRT session is already loaded";
      return false;
    }
    model_path_ = model_path;
    std::string weight_directory;
    if (!weight_path.empty() && !weight_directory_.Materialize(weight_path, &weight_directory, error)) {
      return false;
    }
    if (!MapModel(model_path, error)) {
      return false;
    }
    foundation_ = dlopen("libhiai_foundation.so", RTLD_NOW | RTLD_LOCAL);
    if (foundation_ == nullptr) {
      if (error) {
        *error = std::string("dlopen libhiai_foundation.so failed: ") + dlerror();
      }
      return false;
    }
    if (!SelectNpuDevice(&device_id_, error)) return false;

    if (!CompileModel(model_path, error)) return false;
    if (!InitializeWeights(weight_path, weight_directory, error)) return false;
    return true;
  }

  bool CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes, const TensorShapes &output_shapes,
                       const std::unordered_map<std::string, std::string> &output_aliases,
                       const SharedInputKeys &shared_input_keys, const ExternalInputHandles &external_inputs,
                       const FileInputBindings &file_inputs, std::string *error) {
    if (executor_ == nullptr || sets_.find(set_name) != sets_.end()) {
      if (error) *error = "invalid or duplicate NNRT tensor set: " + set_name;
      return false;
    }
    size_t input_count = 0;
    size_t output_count = 0;
    if (OH_NNExecutor_GetInputCount(executor_, &input_count) != OH_NN_SUCCESS ||
        OH_NNExecutor_GetOutputCount(executor_, &output_count) != OH_NN_SUCCESS || input_count != input_shapes.size() ||
        output_count != output_shapes.size()) {
      if (error) {
        *error =
          "unexpected OMC ABI: inputs=" + std::to_string(input_count) +
          " expected_inputs=" + std::to_string(input_shapes.size()) + " outputs=" + std::to_string(output_count) +
          " expected_outputs=" + std::to_string(output_shapes.size()) + " set=" + set_name + " model=" + model_path_;
      }
      return false;
    }

    TensorSet set;
    set.inputs.reserve(input_count);
    set.outputs.reserve(output_count);
    set.input_bytes.reserve(input_count);
    const TensorSetOptions options{input_shapes,      output_shapes,   output_aliases,
                                   shared_input_keys, external_inputs, file_inputs};
    for (size_t index = 0; index < input_count; ++index) {
      if (!CreateInput(index, options, set, error)) return false;
    }
    for (size_t index = 0; index < output_count; ++index) {
      if (!CreateOutput(index, options, set, error)) return false;
    }
    sets_.emplace(set_name, std::move(set));
    return true;
  }

  bool ReplaceFileInput(const std::string &set_name, const std::string &tensor_name, const FileInputBinding &binding,
                        std::string *error) {
    const auto set_it = sets_.find(set_name);
    if (set_it == sets_.end() || binding.fd < 0 || binding.size == 0 || binding.offset > binding.size) {
      if (error) *error = "invalid file input replacement: " + tensor_name;
      return false;
    }
    auto input_it = set_it->second.input_by_name.find(tensor_name);
    if (input_it == set_it->second.input_by_name.end()) {
      if (error) *error = "unknown input for replacement: " + tensor_name;
      return false;
    }
    const size_t index = set_it->second.input_index_by_name.at(tensor_name);
    const size_t expected_bytes = set_it->second.input_bytes.at(tensor_name);
    if (expected_bytes > binding.size - binding.offset) {
      if (error) *error = "file input is smaller than tensor: " + tensor_name;
      return false;
    }
    NN_TensorDesc *desc = OH_NNExecutor_CreateInputTensorDesc(executor_, index);
    if (desc == nullptr) {
      if (error) *error = "cannot create replacement descriptor: " + tensor_name;
      return false;
    }
    if (!ValidateReplacementShape(desc, set_it->second, tensor_name, expected_bytes, error)) return false;
    return RebindFileInput(set_it->second, tensor_name, desc, binding, error);
  }

  bool DestroyTensorSet(const std::string &set_name, std::string *error) {
    const auto found = sets_.find(set_name);
    if (found == sets_.end()) {
      if (error) *error = "unknown NNRT tensor set: " + set_name;
      return false;
    }
    std::unordered_set<NN_Tensor *> candidates;
    candidates.insert(found->second.inputs.begin(), found->second.inputs.end());
    candidates.insert(found->second.outputs.begin(), found->second.outputs.end());
    sets_.erase(found);

    for (NN_Tensor *tensor : candidates) {
      const bool referenced = std::any_of(sets_.begin(), sets_.end(), [tensor](const auto &item) {
        return std::find(item.second.inputs.begin(), item.second.inputs.end(), tensor) != item.second.inputs.end() ||
               std::find(item.second.outputs.begin(), item.second.outputs.end(), tensor) != item.second.outputs.end();
      });
      if (referenced) continue;
      for (auto shared = shared_tensors_.begin(); shared != shared_tensors_.end();) {
        if (shared->second == tensor) {
          shared = shared_tensors_.erase(shared);
        } else {
          ++shared;
        }
      }
      auto owned = std::find(owned_tensors_.begin(), owned_tensors_.end(), tensor);
      if (owned == owned_tensors_.end()) continue;
      file_backed_tensors_.erase(tensor);
      NN_Tensor *destroyed = tensor;
      OH_NNTensor_Destroy(&destroyed);
      owned_tensors_.erase(owned);
    }
    return true;
  }

  TensorView Input(const std::string &set_name, const std::string &tensor_name) const {
    const auto set = sets_.find(set_name);
    if (set == sets_.end()) return {};
    const auto tensor = set->second.input_by_name.find(tensor_name);
    if (tensor == set->second.input_by_name.end()) return {};
    return TensorViewOf(tensor->second);
  }

  TensorView Output(const std::string &set_name, const std::string &tensor_name) const {
    const auto set = sets_.find(set_name);
    if (set == sets_.end()) return {};
    const auto tensor = set->second.output_by_name.find(tensor_name);
    return tensor == set->second.output_by_name.end() ? TensorView{} : TensorViewOf(tensor->second);
  }

  void *InputHandle(const std::string &set_name, const std::string &tensor_name) const {
    const auto set = sets_.find(set_name);
    if (set == sets_.end()) return nullptr;
    const auto tensor = set->second.input_by_name.find(tensor_name);
    return tensor == set->second.input_by_name.end() ? nullptr : tensor->second;
  }

  bool Run(const std::string &set_name, double *elapsed_ms, std::string *error) {
    const auto set = sets_.find(set_name);
    if (set == sets_.end()) {
      if (error) *error = "unknown NNRT tensor set: " + set_name;
      return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const OH_NN_ReturnCode status =
      OH_NNExecutor_RunSync(executor_, set->second.inputs.data(), set->second.inputs.size(), set->second.outputs.data(),
                            set->second.outputs.size());
    const auto end = std::chrono::steady_clock::now();
    if (elapsed_ms != nullptr) {
      *elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    }
    if (status != OH_NN_SUCCESS) {
      if (error) *error = "OH_NNExecutor_RunSync failed: " + std::to_string(status);
      return false;
    }
    return true;
  }

 private:
  struct TensorSet {
    std::vector<NN_Tensor *> inputs;
    std::vector<NN_Tensor *> outputs;
    std::unordered_map<std::string, NN_Tensor *> input_by_name;
    std::unordered_map<std::string, size_t> input_index_by_name;
    std::unordered_map<std::string, size_t> input_bytes;
    TensorShapes input_shapes;
    std::unordered_map<std::string, NN_Tensor *> output_by_name;
  };

  struct TensorSetOptions {
    const TensorShapes &input_shapes;
    const TensorShapes &output_shapes;
    const std::unordered_map<std::string, std::string> &output_aliases;
    const SharedInputKeys &shared_input_keys;
    const ExternalInputHandles &external_inputs;
    const FileInputBindings &file_inputs;
  };

  bool ConfigureProfiling(OH_NNCompilation *compilation, const std::string &model_path, std::string *error) {
    const char *profiling_dir = std::getenv("QWEN35_NNRT_PROFILING_DIR");
    const char *profiling_model = std::getenv("QWEN35_NNRT_PROFILING_MODEL");
    const bool profile_this_model =
      profiling_model == nullptr || profiling_model[0] == '\0' || model_path.find(profiling_model) != std::string::npos;
    if (profiling_dir != nullptr && profiling_dir[0] != '\0' && profile_this_model) {
      using SetOmOptions = OH_NN_ReturnCode (*)(OH_NNCompilation *, int32_t, const char *);
      auto set_om_options = reinterpret_cast<SetOmOptions>(dlsym(foundation_, "HMS_HiAIOptions_SetOmOptions"));
      if (set_om_options == nullptr || set_om_options(compilation, 1, profiling_dir) != OH_NN_SUCCESS) {
        if (error) *error = "cannot enable NNRT profiling";
        return false;
      }
    }
    return true;
  }

  bool InitializeWeights(const std::string &weight_path, const std::string &weight_directory, std::string *error) {
    if (!weight_path.empty()) {
      using InitWeights = OH_NN_ReturnCode (*)(OH_NNExecutor *, const char *);
      auto init_weights = reinterpret_cast<InitWeights>(dlsym(foundation_, "HMS_HiAIExecutor_InitWeights"));
      if (init_weights == nullptr) {
        if (error) *error = "HMS_HiAIExecutor_InitWeights is unavailable";
        return false;
      }
      const OH_NN_ReturnCode status = init_weights(executor_, weight_directory.c_str());
      if (status != OH_NN_SUCCESS) {
        if (error) {
          *error = "external weight initialization failed: " + std::to_string(status);
        }
        return false;
      }
    }
    return true;
  }

  bool AllocateInput(const std::string &name, NN_TensorDesc *desc, size_t expected_bytes,
                     const TensorSetOptions &options, NN_Tensor **tensor, std::string *error) {
    const auto external = options.external_inputs.find(name);
    if (external != options.external_inputs.end()) {
      *tensor = static_cast<NN_Tensor *>(external->second);
    }
    const auto shared_key = options.shared_input_keys.find(name);
    if (*tensor == nullptr && shared_key != options.shared_input_keys.end()) {
      const auto shared = shared_tensors_.find(shared_key->second);
      if (shared != shared_tensors_.end()) *tensor = shared->second;
    }
    const auto file = options.file_inputs.find(name);
    if (*tensor == nullptr && file != options.file_inputs.end()) {
      const FileInputBinding &binding = file->second;
      if (binding.fd < 0 || binding.size == 0 || binding.offset > binding.size ||
          expected_bytes > binding.size - binding.offset) {
        OH_NNTensorDesc_Destroy(&desc);
        if (error) *error = "invalid file-backed OMC input: " + name;
        return false;
      }
      *tensor = OH_NNTensor_CreateWithFd(device_id_, desc, binding.fd, binding.size, binding.offset);
      if (*tensor != nullptr) {
        owned_tensors_.push_back(*tensor);
        file_backed_tensors_.insert(*tensor);
        if (shared_key != options.shared_input_keys.end()) {
          shared_tensors_[shared_key->second] = *tensor;
        }
      }
    }
    if (*tensor == nullptr) {
      *tensor = OH_NNTensor_Create(device_id_, desc);
      if (*tensor != nullptr) {
        owned_tensors_.push_back(*tensor);
        if (shared_key != options.shared_input_keys.end()) {
          shared_tensors_[shared_key->second] = *tensor;
        }
      }
    }
    return true;
  }

  bool CreateInput(size_t index, const TensorSetOptions &options, TensorSet &set, std::string *error) {
    NN_TensorDesc *desc = OH_NNExecutor_CreateInputTensorDesc(executor_, index);
    std::string name;
    if (!TensorName(desc, &name)) {
      if (desc != nullptr) OH_NNTensorDesc_Destroy(&desc);
      if (error) *error = "cannot inspect OMC input " + std::to_string(index);
      return false;
    }
    const auto shape = options.input_shapes.find(name);
    if (shape == options.input_shapes.end() || !SetShape(desc, shape->second)) {
      OH_NNTensorDesc_Destroy(&desc);
      if (error) {
        *error = "missing or invalid input shape: " + name;
        if (shape != options.input_shapes.end()) {
          *error += " requested=" + ShapeText(shape->second);
        }
      }
      return false;
    }
    size_t expected_bytes = 0;
    if (OH_NNTensorDesc_GetByteSize(desc, &expected_bytes) != OH_NN_SUCCESS) {
      OH_NNTensorDesc_Destroy(&desc);
      if (error) *error = "cannot inspect input byte size: " + name;
      return false;
    }
    NN_Tensor *tensor = nullptr;
    if (!AllocateInput(name, desc, expected_bytes, options, &tensor, error)) return false;
    const auto file = options.file_inputs.find(name);
    OH_NNTensorDesc_Destroy(&desc);
    const TensorView tensor_view = TensorViewOf(tensor);
    if (tensor == nullptr || tensor_view.data == nullptr || tensor_view.bytes != expected_bytes) {
      if (error) {
        *error = "cannot allocate OMC input tensor: " + name + " expected=" + std::to_string(expected_bytes) +
                 " file_backed=" + (file != options.file_inputs.end() ? "1" : "0") +
                 " tensor=" + (tensor == nullptr ? "null" : "non-null") +
                 " data=" + (tensor_view.data == nullptr ? "null" : "non-null") +
                 " actual=" + std::to_string(tensor_view.bytes);
      }
      return false;
    }
    set.inputs.push_back(tensor);
    set.input_by_name[name] = tensor;
    set.input_index_by_name[name] = index;
    set.input_bytes[name] = expected_bytes;
    set.input_shapes[name] = shape->second;
    return true;
  }

  bool CreateOutput(size_t index, const TensorSetOptions &options, TensorSet &set, std::string *error) {
    NN_TensorDesc *desc = OH_NNExecutor_CreateOutputTensorDesc(executor_, index);
    std::string name;
    if (!TensorName(desc, &name)) {
      if (desc != nullptr) OH_NNTensorDesc_Destroy(&desc);
      if (error) *error = "cannot inspect OMC output " + std::to_string(index);
      return false;
    }
    name = CanonicalOutputName(index, options.output_shapes.size(), name, options.output_shapes);
    const auto shape = options.output_shapes.find(name);
    if (shape == options.output_shapes.end() || !SetShape(desc, shape->second)) {
      OH_NNTensorDesc_Destroy(&desc);
      if (error) *error = "missing or invalid output shape: " + name;
      return false;
    }
    NN_Tensor *tensor = nullptr;
    const auto alias = options.output_aliases.find(name);
    if (alias != options.output_aliases.end()) {
      const auto input = set.input_by_name.find(alias->second);
      if (input != set.input_by_name.end()) tensor = input->second;
    } else {
      tensor = OH_NNTensor_Create(device_id_, desc);
      if (tensor != nullptr) owned_tensors_.push_back(tensor);
    }
    OH_NNTensorDesc_Destroy(&desc);
    if (tensor == nullptr || TensorViewOf(tensor).data == nullptr) {
      if (error) *error = "cannot allocate or alias OMC output tensor: " + name;
      return false;
    }
    set.outputs.push_back(tensor);
    set.output_by_name[name] = tensor;
    return true;
  }

  bool ValidateReplacementShape(NN_TensorDesc *desc, const TensorSet &set, const std::string &tensor_name,
                                size_t expected_bytes, std::string *error) {
    const auto shape_it = set.input_shapes.find(tensor_name);
    const bool shape_ok = shape_it != set.input_shapes.end() && SetShape(desc, shape_it->second);
    size_t descriptor_bytes = 0;
    const bool size_ok = shape_ok && OH_NNTensorDesc_GetByteSize(desc, &descriptor_bytes) == OH_NN_SUCCESS &&
                         descriptor_bytes == expected_bytes;
    if (!size_ok) {
      OH_NNTensorDesc_Destroy(&desc);
      if (error) {
        *error = "replacement descriptor mismatch: " + tensor_name + " shape_ok=" + (shape_ok ? "1" : "0") +
                 " expected=" + std::to_string(expected_bytes) + " descriptor=" + std::to_string(descriptor_bytes);
      }
      return false;
    }
    return true;
  }

  bool CompileModel(const std::string &model_path, std::string *error) {
    OH_NNCompilation *compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(mapping_, mapping_bytes_);
    if (compilation == nullptr) {
      if (error) *error = "offline model compilation construction failed";
      return false;
    }
    auto fail = [&](const std::string &message) {
      if (error) *error = message;
      OH_NNCompilation_Destroy(&compilation);
      return false;
    };
    if (OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_EXTREME) != OH_NN_SUCCESS ||
        OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_HIGH) != OH_NN_SUCCESS) {
      return fail("cannot configure NNRT compilation");
    }
    using SetAsyncMode = OH_NN_ReturnCode (*)(OH_NNCompilation *, bool);
    auto set_async = reinterpret_cast<SetAsyncMode>(dlsym(foundation_, "HMS_HiAIOptions_SetAsyncModeEnable"));
    if (set_async == nullptr || set_async(compilation, false) != OH_NN_SUCCESS) {
      return fail("NNRT offline model build failed");
    }
    if (!ConfigureProfiling(compilation, model_path, error)) {
      OH_NNCompilation_Destroy(&compilation);
      return false;
    }
    if (OH_NNCompilation_Build(compilation) != OH_NN_SUCCESS) {
      return fail("NNRT offline model build failed");
    }
    executor_ = OH_NNExecutor_Construct(compilation);
    OH_NNCompilation_Destroy(&compilation);
    if (executor_ == nullptr) {
      if (error) *error = "NNRT executor construction failed";
      return false;
    }
    return true;
  }

  bool RebindFileInput(TensorSet &set, const std::string &tensor_name, NN_TensorDesc *desc,
                       const FileInputBinding &binding, std::string *error) {
    auto input_it = set.input_by_name.find(tensor_name);
    const size_t index = set.input_index_by_name.at(tensor_name);
    const size_t expected_bytes = set.input_bytes.at(tensor_name);
    NN_Tensor *old = input_it->second;
    // The Kirin runtime has a tight limit for simultaneously mapped regular
    // file tensors. Reclaim the old physical slot before mapping its
    // replacement, instead of transiently creating a ninth shard tensor.
    auto old_it = std::find(owned_tensors_.begin(), owned_tensors_.end(), old);
    if (old_it == owned_tensors_.end()) {
      OH_NNTensorDesc_Destroy(&desc);
      if (error) *error = "file replacement does not own tensor: " + tensor_name;
      return false;
    }
    file_backed_tensors_.erase(old);
    OH_NNTensor_Destroy(&old);
    owned_tensors_.erase(old_it);
    NN_Tensor *replacement = OH_NNTensor_CreateWithFd(device_id_, desc, binding.fd, binding.size, binding.offset);
    OH_NNTensorDesc_Destroy(&desc);
    if (replacement == nullptr || TensorViewOf(replacement).data == nullptr ||
        TensorViewOf(replacement).bytes != expected_bytes) {
      if (replacement != nullptr) OH_NNTensor_Destroy(&replacement);
      if (error) {
        *error = "cannot create file-backed replacement: " + tensor_name + " fd=" + std::to_string(binding.fd) +
                 " size=" + std::to_string(binding.size) + " offset=" + std::to_string(binding.offset) +
                 " expected=" + std::to_string(expected_bytes);
      }
      return false;
    }
    // Zero-copy fused boundaries share the eight shard tensors within one
    // executor. Replace every set that references the old tensor so the
    // shared object remains valid for subsequent layer execution.
    for (auto &item : sets_) {
      auto item_input = item.second.input_by_name.find(tensor_name);
      if (item_input == item.second.input_by_name.end() || item_input->second != old) continue;
      const size_t item_index = item.second.input_index_by_name.at(tensor_name);
      item_input->second = replacement;
      item.second.inputs[item_index] = replacement;
    }
    input_it->second = replacement;
    set.inputs[index] = replacement;
    set.input_by_name[tensor_name] = replacement;
    owned_tensors_.push_back(replacement);
    file_backed_tensors_.insert(replacement);
    for (auto &item : shared_tensors_) {
      if (item.second == old) item.second = replacement;
    }
    return true;
  }

  TensorView TensorViewOf(NN_Tensor *tensor) const {
    if (tensor == nullptr) return {};
    NN_TensorDesc *desc = OH_NNTensor_GetTensorDesc(tensor);
    size_t allocation = 0;
    size_t offset = 0;
    size_t bytes = 0;
    auto *base = static_cast<uint8_t *>(OH_NNTensor_GetDataBuffer(tensor));
    if (desc == nullptr || base == nullptr || OH_NNTensor_GetSize(tensor, &allocation) != OH_NN_SUCCESS ||
        OH_NNTensor_GetOffset(tensor, &offset) != OH_NN_SUCCESS ||
        OH_NNTensorDesc_GetByteSize(desc, &bytes) != OH_NN_SUCCESS) {
      return {};
    }
    if (file_backed_tensors_.find(tensor) != file_backed_tensors_.end()) {
      // Kirin NNRT maps a file-backed tensor at the requested file offset;
      // GetDataBuffer already points to the first tensor byte.
      return bytes <= allocation ? TensorView{base, bytes} : TensorView{};
    }
    if (offset > allocation || bytes > allocation - offset) return {};
    return {base + offset, bytes};
  }

  bool MapModel(const std::string &path, std::string *error) {
    size_t resource_offset = 0;
    model_fd_ = OpenArtifact(path, &mapping_bytes_, &resource_offset, error);
    if (model_fd_ < 0 || mapping_bytes_ == 0) {
      if (error) *error = "cannot open OMC: " + path;
      return false;
    }
    mapping_ = mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, model_fd_, static_cast<off_t>(resource_offset));
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      if (error) *error = "cannot mmap OMC: " + path;
      return false;
    }
    return true;
  }

  void Release() {
    for (NN_Tensor *&tensor : owned_tensors_) {
      if (tensor != nullptr) OH_NNTensor_Destroy(&tensor);
    }
    if (executor_ != nullptr) OH_NNExecutor_Destroy(&executor_);
    if (foundation_ != nullptr) dlclose(foundation_);
    if (mapping_ != nullptr) munmap(mapping_, mapping_bytes_);
    if (model_fd_ >= 0) close(model_fd_);
    owned_tensors_.clear();
    file_backed_tensors_.clear();
    shared_tensors_.clear();
    sets_.clear();
    executor_ = nullptr;
    foundation_ = nullptr;
    mapping_ = nullptr;
    mapping_bytes_ = 0;
    model_fd_ = -1;
  }

  int model_fd_ = -1;
  ArtifactWeightDirectory weight_directory_;
  void *mapping_ = nullptr;
  size_t mapping_bytes_ = 0;
  void *foundation_ = nullptr;
  size_t device_id_ = 0;
  OH_NNExecutor *executor_ = nullptr;
  std::vector<NN_Tensor *> owned_tensors_;
  std::unordered_set<NN_Tensor *> file_backed_tensors_;
  std::unordered_map<std::string, NN_Tensor *> shared_tensors_;
  std::unordered_map<std::string, TensorSet> sets_;
  std::string model_path_;
};

NnrtSession::NnrtSession() : impl_(std::make_unique<Impl>()) {}
NnrtSession::~NnrtSession() = default;

bool NnrtSession::Load(const std::string &model_path, const std::string &weight_path, std::string *error) {
  return impl_->Load(model_path, weight_path, error);
}

bool NnrtSession::CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes,
                                  const TensorShapes &output_shapes,
                                  const std::unordered_map<std::string, std::string> &output_aliases,
                                  const std::unordered_set<std::string> &shared_inputs, std::string *error) {
  SharedInputKeys keys;
  keys.reserve(shared_inputs.size());
  for (const std::string &name : shared_inputs) keys.emplace(name, name);
  return impl_->CreateTensorSet(set_name, input_shapes, output_shapes, output_aliases, keys, {}, {}, error);
}

bool NnrtSession::CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes,
                                  const TensorShapes &output_shapes,
                                  const std::unordered_map<std::string, std::string> &output_aliases,
                                  const SharedInputKeys &shared_input_keys, std::string *error) {
  return impl_->CreateTensorSet(set_name, input_shapes, output_shapes, output_aliases, shared_input_keys, {}, {},
                                error);
}

bool NnrtSession::CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes,
                                  const TensorShapes &output_shapes,
                                  const std::unordered_map<std::string, std::string> &output_aliases,
                                  const SharedInputKeys &shared_input_keys, const ExternalInputHandles &external_inputs,
                                  std::string *error) {
  return impl_->CreateTensorSet(set_name, input_shapes, output_shapes, output_aliases, shared_input_keys,
                                external_inputs, {}, error);
}

bool NnrtSession::CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes,
                                  const TensorShapes &output_shapes,
                                  const std::unordered_map<std::string, std::string> &output_aliases,
                                  const SharedInputKeys &shared_input_keys, const ExternalInputHandles &external_inputs,
                                  const FileInputBindings &file_inputs, std::string *error) {
  return impl_->CreateTensorSet(set_name, input_shapes, output_shapes, output_aliases, shared_input_keys,
                                external_inputs, file_inputs, error);
}

bool NnrtSession::ReplaceFileInput(const std::string &set_name, const std::string &tensor_name,
                                   const FileInputBinding &binding, std::string *error) {
  return impl_->ReplaceFileInput(set_name, tensor_name, binding, error);
}

bool NnrtSession::DestroyTensorSet(const std::string &set_name, std::string *error) {
  return impl_->DestroyTensorSet(set_name, error);
}

TensorView NnrtSession::Input(const std::string &set_name, const std::string &tensor_name) const {
  return impl_->Input(set_name, tensor_name);
}

TensorView NnrtSession::Output(const std::string &set_name, const std::string &tensor_name) const {
  return impl_->Output(set_name, tensor_name);
}

void *NnrtSession::InputHandle(const std::string &set_name, const std::string &tensor_name) const {
  return impl_->InputHandle(set_name, tensor_name);
}

bool NnrtSession::Run(const std::string &set_name, double *elapsed_ms, std::string *error) {
  return impl_->Run(set_name, elapsed_ms, error);
}

}  // namespace mslite_llm::qwen35
