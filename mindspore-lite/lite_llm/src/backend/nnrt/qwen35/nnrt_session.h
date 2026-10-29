// Copyright 2026 Huawei Technologies Co., Ltd
// Adapted from HMOS-Device-AI llm/engine/runtime; preserves its paged Qwen3.5 execution.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mslite_llm::qwen35 {

struct TensorView {
  void *data = nullptr;
  size_t bytes = 0;
};

using TensorShapes = std::unordered_map<std::string, std::vector<int32_t>>;
using SharedInputKeys = std::unordered_map<std::string, std::string>;
using ExternalInputHandles = std::unordered_map<std::string, void *>;
struct FileInputBinding {
  int fd = -1;
  size_t size = 0;
  size_t offset = 0;
};
using FileInputBindings = std::unordered_map<std::string, FileInputBinding>;

class NnrtSession {
 public:
  NnrtSession();
  ~NnrtSession();
  NnrtSession(const NnrtSession &) = delete;
  NnrtSession &operator=(const NnrtSession &) = delete;

  bool Load(const std::string &model_path, const std::string &weight_path, std::string *error);
  bool CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes, const TensorShapes &output_shapes,
                       const std::unordered_map<std::string, std::string> &output_aliases,
                       const std::unordered_set<std::string> &shared_inputs, std::string *error);
  bool CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes, const TensorShapes &output_shapes,
                       const std::unordered_map<std::string, std::string> &output_aliases,
                       const SharedInputKeys &shared_input_keys, std::string *error);
  bool CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes, const TensorShapes &output_shapes,
                       const std::unordered_map<std::string, std::string> &output_aliases,
                       const SharedInputKeys &shared_input_keys, const ExternalInputHandles &external_inputs,
                       std::string *error);
  bool CreateTensorSet(const std::string &set_name, const TensorShapes &input_shapes, const TensorShapes &output_shapes,
                       const std::unordered_map<std::string, std::string> &output_aliases,
                       const SharedInputKeys &shared_input_keys, const ExternalInputHandles &external_inputs,
                       const FileInputBindings &file_inputs, std::string *error);
  // Replaces one input tensor while retaining the executor and tensor set.
  // The caller keeps binding.fd open for the lifetime of the tensor set.
  bool ReplaceFileInput(const std::string &set_name, const std::string &tensor_name, const FileInputBinding &binding,
                        std::string *error);
  bool DestroyTensorSet(const std::string &set_name, std::string *error);
  TensorView Input(const std::string &set_name, const std::string &tensor_name) const;
  TensorView Output(const std::string &set_name, const std::string &tensor_name) const;
  void *InputHandle(const std::string &set_name, const std::string &tensor_name) const;
  bool Run(const std::string &set_name, double *elapsed_ms, std::string *error);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mslite_llm::qwen35
