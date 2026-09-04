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

#include "backend/nnrt/nnrt_executor.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "backend/nnrt/nnrt_log.h"
#include "backend/nnrt/nnrt_wrapper.h"
#include "manifest/msl_package_reader.h"
#include "backend/nnrt/nnrt_embedding_dequant.h"
namespace mslite {
namespace backend {
namespace nnrt {

NnrtExecutor::~NnrtExecutor() {
  const auto &api = NNRTWrapper::GetApi();
  auto destroy_tensor = [&](NN_Tensor *t) {
    if (t != nullptr && api.Tensor_Destroy != nullptr) {
      api.Tensor_Destroy(&t);
    }
  };
  for (auto *t : prefill_inputs_) {
    destroy_tensor(t);
  }
  // embedding_weight (idx6) is shared between prefill and decode; avoid double-free.
  if (decode_inputs_.size() > 6) {
    decode_inputs_[6] = nullptr;
  }
  for (auto *t : decode_inputs_) {
    destroy_tensor(t);
  }
  destroy_tensor(logits_tensor_);
  // KV tensors freed by kv_cache_manager_ destructor

  if (embedding_table_ != nullptr) {
    ::munmap(embedding_table_, embedding_table_elems_ * sizeof(uint16_t));
    embedding_table_ = nullptr;
    embedding_table_elems_ = 0;
  }
  if (nn_executor_ != nullptr && api.Executor_Destroy != nullptr) {
    api.Executor_Destroy(&nn_executor_);
    nn_executor_ = nullptr;
  }
  if (nn_compilation_ != nullptr && api.Compilation_Destroy != nullptr) {
    api.Compilation_Destroy(&nn_compilation_);
    nn_compilation_ = nullptr;
  }
  if (omc_mapping_ != nullptr) {
    ::munmap(omc_mapping_, omc_mapping_size_);
    omc_mapping_ = nullptr;
    omc_mapping_size_ = 0;
  }
}

namespace {

int ArgMax(const std::vector<float> &logits) {
  if (logits.empty()) return -1;
  return static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

}  // namespace

bool NnrtExecutor::Build(const NnrtConfig &config) {
  auto perf_t0 = std::chrono::steady_clock::now();
  auto log_phase = [&](const char *tag) {
    auto t1 = std::chrono::steady_clock::now();
    MS_LOG(INFO) << "[perf] " << tag << ": " << std::chrono::duration<double, std::milli>(t1 - perf_t0).count() << "ms";
    perf_t0 = t1;
  };
  if (config.vocab_size <= 0 || config.num_layers <= 0 || config.head_dim <= 0 || config.max_length <= 0 ||
      config.hidden_size <= 0 || config.chunk_size <= 0) {
    MS_LOG(ERROR) << "Invalid NnrtConfig";
    return false;
  }
  // Prefill copies whole chunks of [start, start+chunk_size) rows from the max_length_-row
  // rope/mask buffers, so a non-divisible max_length would read out of bounds on the last chunk.
  if (config.max_length % config.chunk_size != 0) {
    MS_LOG(ERROR) << "max_length " << config.max_length << " must be a multiple of chunk_size " << config.chunk_size;
    return false;
  }
  vocab_size_ = config.vocab_size;
  head_dim_ = config.head_dim;
  max_length_ = config.max_length;
  chunk_size_ = config.chunk_size;
  eos_id_ = config.eos_id;
  hidden_size_ = config.hidden_size;
  num_key_value_heads_ = config.num_key_value_heads;
  num_layers_ = config.num_layers;
  embedding_quant_ = config.embedding_quant;
  scale_gp_size_ = config.scale_gp_size;
  single_file_ = config.single_file;
  package_reader_ = config.package_reader;
  embedding_path_ = config.embedding_path;
  // device_id_ defaults to 0 (first NPU). Single-NPU Kirin is the current target;
  // multi-device name->id resolution via OH_NNDevice_GetAllDevicesID is TODO.

  omc_path_ = config.prefill_path;
  if (omc_path_.empty()) {
    MS_LOG(ERROR) << "model_path empty";
    return false;
  }

  if (NNRTWrapper::GetInstance() == nullptr) {
    MS_LOG(ERROR) << "NNRTWrapper init failed";
    return false;
  }
  log_phase("GetInstance(dlopen)");

  if (!BuildModel()) {
    return false;
  }
  log_phase("BuildModel");
  if (!LoadCpuBuffers(config)) {
    return false;
  }
  log_phase("LoadCpuBuffers");
  if (!kv_cache_manager_.Alloc(num_layers_, num_key_value_heads_, static_cast<int>(max_length_), head_dim_, device_id_,
                               nn_executor_)) {
    MS_LOG(ERROR) << "KV Cache allocation failed";
    return false;
  }
  log_phase("KVCacheAlloc");
  if (!CreateTensors()) {
    return false;
  }
  log_phase("CreateTensors");

  // Sampler is owned by Pipeline (see Task 5); NNRT executor only runs graphs.

  kv_cache_manager_.Reset();  // start of sequence
  log_phase("KVReset");
  MS_LOG(INFO) << "NnrtExecutor::Build succeeded";
  return true;
}

bool NnrtExecutor::BuildModel() {
  const auto &api = NNRTWrapper::GetApi();
  if (!ConstructCompilation()) {
    return false;
  }
  if (api.Compilation_SetDevice(nn_compilation_, device_id_) != 0) {
    MS_LOG(ERROR) << "Compilation_SetDevice failed";
    return false;
  }
  if (api.HIAIOptions_SetAsyncModeEnable(nn_compilation_, false) != 0) {
    MS_LOG(ERROR) << "HIAIOptions_SetAsyncModeEnable failed";
    return false;
  }
  if (api.Compilation_Build(nn_compilation_) != 0) {
    MS_LOG(ERROR) << "Compilation_Build failed";
    return false;
  }
  nn_executor_ = api.Executor_Construct(nn_compilation_);
  if (nn_executor_ == nullptr) {
    MS_LOG(ERROR) << "Executor_Construct failed";
    return false;
  }
  if (!ValidateModelContract()) {
    return false;
  }
  if (!ReadModelVocab()) {
    return false;
  }
  ReclaimOfflineModelPages();
  return true;
}

bool NnrtExecutor::ConstructCompilation() {
  const auto &api = NNRTWrapper::GetApi();
  if (single_file_) {
    // The .omc lives inside the .msl. Hand NNRT the mmap'd entry directly via
    // the offline-model BUFFER API — the /proc/self/fd/N memfd path is rejected
    // by the app sandbox at Build time (verified on kirin9020).
    if (package_reader_ == nullptr || api.Compilation_ConstructWithOfflineModelBuffer == nullptr) {
      MS_LOG(ERROR) << "single-file mode requires package reader + offline-model buffer API";
      return false;
    }
    const uint8_t *data = nullptr;
    size_t size = 0;
    if (!package_reader_->Mmap(omc_path_, &data, &size) || data == nullptr || size == 0) {
      MS_LOG(ERROR) << "Failed to mmap .omc entry: " << omc_path_;
      return false;
    }
    nn_compilation_ = api.Compilation_ConstructWithOfflineModelBuffer(data, size);
    MS_LOG(INFO) << "Loaded .omc entry " << omc_path_ << " (" << size << " bytes) via offline-model buffer API";
  } else if (api.Compilation_ConstructWithOfflineModelBuffer != nullptr && MapOfflineModelFile(omc_path_)) {
    nn_compilation_ = api.Compilation_ConstructWithOfflineModelBuffer(omc_mapping_, omc_mapping_size_);
    if (nn_compilation_ != nullptr) {
      MS_LOG(INFO) << "Loaded directory .omc " << omc_path_ << " (" << omc_mapping_size_
                   << " bytes) via mmap + offline-model buffer API";
    } else {
      MS_LOG(WARNING) << "offline-model buffer API rejected directory .omc; falling back to file API";
      ::munmap(omc_mapping_, omc_mapping_size_);
      omc_mapping_ = nullptr;
      omc_mapping_size_ = 0;
    }
  } else {
    MS_LOG(WARNING) << (api.Compilation_ConstructWithOfflineModelBuffer == nullptr
                          ? "offline-model buffer API unavailable; falling back to file API"
                          : "Failed to mmap directory .omc; falling back to file API");
  }
  if (!single_file_ && nn_compilation_ == nullptr) {
    nn_compilation_ = api.Compilation_ConstructWithOfflineModelFile(omc_path_.c_str());
  }
  if (nn_compilation_ == nullptr) {
    MS_LOG(ERROR) << "ConstructWithOfflineModel failed";
    return false;
  }
  return true;
}

bool NnrtExecutor::MapOfflineModelFile(const std::string &path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }
  struct stat st {};
  const bool valid = ::fstat(fd, &st) == 0 && st.st_size > 0;
  if (!valid || static_cast<uintmax_t>(st.st_size) > std::numeric_limits<size_t>::max()) {
    (void)::close(fd);
    return false;
  }
  omc_mapping_size_ = static_cast<size_t>(st.st_size);
  omc_mapping_ = ::mmap(nullptr, omc_mapping_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  (void)::close(fd);
  if (omc_mapping_ == MAP_FAILED) {
    omc_mapping_ = nullptr;
    omc_mapping_size_ = 0;
    return false;
  }
  return true;
}

void NnrtExecutor::ReclaimOfflineModelPages() const {
  if (single_file_) {
    if (package_reader_ != nullptr && !omc_path_.empty() && !package_reader_->Reclaim(omc_path_)) {
      MS_LOG(WARNING) << "Failed to reclaim package .omc pages";
    }
    return;
  }
  if (omc_mapping_ != nullptr && ::madvise(omc_mapping_, omc_mapping_size_, MADV_DONTNEED) != 0) {
    MS_LOG(WARNING) << "Failed to reclaim directory .omc pages";
  }
}

void NnrtExecutor::ReclaimEmbeddingWeightPages() const {
  if (single_file_ && package_reader_ != nullptr && !embedding_path_.empty() &&
      !package_reader_->Reclaim(embedding_path_)) {
    MS_LOG(WARNING) << "Failed to reclaim embedding package pages";
  }
}

bool NnrtExecutor::ValidateModelContract() {
  const auto &api = NNRTWrapper::GetApi();
  // Expected contract, derived from config.num_layers: 7 non-KV inputs + interleaved
  // past_key_i/past_val_i per layer; logits + interleaved out_key_i/out_val_i per layer.
  // Deriving from num_layers_ (rather than a fixed model) also catches config/model mismatch.
  constexpr size_t kNonKvInputs = 7;
  const size_t kv_layers = static_cast<size_t>(num_layers_);
  const size_t expected_input_count = kNonKvInputs + 2 * kv_layers;
  const size_t expected_output_count = 1 + 2 * kv_layers;
  static const char *kBaseNames[kNonKvInputs] = {"valid_seq_len", "lmhead_idx",     "rope_cos",        "rope_sin",
                                                 "inputs_embeds", "attention_mask", "embedding_weight"};

  // Count check (skipped with a warning if the NNRT is too old to expose I/O counts).
  if (api.Executor_GetInputCount != nullptr && api.Executor_GetOutputCount != nullptr) {
    size_t in_count = 0;
    size_t out_count = 0;
    if (api.Executor_GetInputCount(nn_executor_, &in_count) != 0 || in_count != expected_input_count) {
      MS_LOG(ERROR) << "Contract check failed: model input count " << in_count << " != expected "
                    << expected_input_count;
      return false;
    }
    if (api.Executor_GetOutputCount(nn_executor_, &out_count) != 0 || out_count != expected_output_count) {
      MS_LOG(ERROR) << "Contract check failed: model output count " << out_count << " != expected "
                    << expected_output_count;
      return false;
    }
  } else {
    MS_LOG(WARNING) << "NNRT does not expose I/O counts; skipping count contract check";
  }

  // Name check (degrade to count-only on old NNRT without TensorDesc_GetName).
  if (api.Executor_CreateInputTensorDesc == nullptr || api.Executor_CreateOutputTensorDesc == nullptr ||
      api.TensorDesc_GetName == nullptr) {
    MS_LOG(WARNING) << "NNRT does not expose tensor names; skipping name contract check";
    return true;
  }
  auto check_name = [&](NN_TensorDesc *desc, const std::string &expected, const char *io, size_t index) {
    const char *name = nullptr;
    bool ok = desc != nullptr && api.TensorDesc_GetName(desc, &name) == 0 && name != nullptr && expected == name;
    if (!ok) {
      MS_LOG(ERROR) << "Contract check failed: " << io << " " << index << " name \""
                    << (name != nullptr ? name : "<null>") << "\" != expected \"" << expected << "\"";
    }
    if (desc != nullptr) {
      api.TensorDesc_Destroy(&desc);
    }
    return ok;
  };
  // Inputs keep their semantic names on device, so they are checked by name.
  for (size_t i = 0; i < kNonKvInputs; ++i) {
    if (!check_name(api.Executor_CreateInputTensorDesc(nn_executor_, i), kBaseNames[i], "input", i)) {
      return false;
    }
  }
  for (size_t i = 0; i < 2 * kv_layers; ++i) {
    std::string expected = std::string((i % 2 == 0) ? "past_key_" : "past_val_") + std::to_string(i / 2);
    if (!check_name(api.Executor_CreateInputTensorDesc(nn_executor_, kNonKvInputs + i), expected, "input",
                    kNonKvInputs + i)) {
      return false;
    }
  }
  // The Kirin DDK renames every output (enum-shape artifact, e.g.
  // "output_0_enum_shape_graph/case0_0"), so output NAME checks can never pass on device:
  // outputs are verified by count only (above) and the actual names are logged for forensics.
  for (size_t i = 0; i < expected_output_count; ++i) {
    NN_TensorDesc *desc = api.Executor_CreateOutputTensorDesc(nn_executor_, i);
    const char *name = nullptr;
    const bool has_name = desc != nullptr && api.TensorDesc_GetName(desc, &name) == 0 && name != nullptr;
    MS_LOG(INFO) << "Model output " << i << " name: " << (has_name ? name : "<unavailable>");
    if (desc != nullptr) {
      api.TensorDesc_Destroy(&desc);
    }
  }
  MS_LOG(INFO) << "Model contract check passed (" << expected_input_count << " inputs / " << expected_output_count
               << " outputs, interleaved KV)";
  return true;
}

bool NnrtExecutor::ReadModelVocab() {
  // The tokenizer/sampling vocab (config) may be narrower than the model's logits width
  // (e.g. cropped-vocab models: 114300 vs 151936). Logits tensors must use the model width;
  // sampling keeps reading only the first vocab_size_ entries.
  model_vocab_ = vocab_size_;  // fallback when the shape cannot be read
  const auto &api = NNRTWrapper::GetApi();
  if (api.Executor_CreateOutputTensorDesc == nullptr || api.TensorDesc_GetShape == nullptr) {
    MS_LOG(WARNING) << "NNRT cannot report logits shape; assuming model vocab == config vocab_size " << vocab_size_;
    return true;
  }
  NN_TensorDesc *desc = api.Executor_CreateOutputTensorDesc(nn_executor_, 0);
  if (desc == nullptr) {
    MS_LOG(WARNING) << "CreateOutputTensorDesc(0) failed; assuming model vocab == config vocab_size " << vocab_size_;
    return true;
  }
  int32_t *shape = nullptr;
  size_t shape_len = 0;
  if (api.TensorDesc_GetShape(desc, &shape, &shape_len) == 0 && shape != nullptr && shape_len > 0 &&
      shape[shape_len - 1] > 0) {
    model_vocab_ = shape[shape_len - 1];
  } else {
    MS_LOG(WARNING) << "Failed to read logits shape; assuming model vocab == config vocab_size " << vocab_size_;
  }
  api.TensorDesc_Destroy(&desc);
  // Config vocab may exceed on-graph logits width when the external embedding bin
  // and tokenizer are full-Qwen (151936) while the omc lm_head is cropped (e.g.
  // 114300). Embedding lookup uses vocab_size_; sampling uses model_vocab_.
  if (vocab_size_ > model_vocab_) {
    MS_LOG(WARNING) << "config vocab_size " << vocab_size_ << " > model logits width " << model_vocab_
                    << "; embedding/tokenizer use full vocab, sampling capped at model width";
  }
  MS_LOG(INFO) << "Model logits width " << model_vocab_ << ", embedding vocab " << vocab_size_ << ", sampling vocab "
               << std::min(vocab_size_, model_vocab_);
  return true;
}

bool NnrtExecutor::ReadAsset(const std::string &path_or_entry, std::vector<uint8_t> *out) const {
  if (out == nullptr) {
    return false;
  }
  out->clear();
  if (single_file_) {
    return package_reader_ != nullptr && package_reader_->Read(path_or_entry, out);
  }
  std::ifstream ifs(path_or_entry, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    return false;
  }
  const auto size = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  out->resize(size);
  if (size > 0) {
    ifs.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(size));
    if (!ifs.good()) {
      return false;
    }
  }
  return true;
}

bool NnrtExecutor::LoadCpuBuffers(const NnrtConfig &config) {
  auto t0 = std::chrono::steady_clock::now();
  auto mark = [&](const char *tag) {
    auto t1 = std::chrono::steady_clock::now();
    MS_LOG(INFO) << "[perf] LoadCpuBuffers/" << tag << ": "
                 << std::chrono::duration<double, std::milli>(t1 - t0).count() << "ms";
    t0 = t1;
  };
  size_t embed_size = static_cast<size_t>(vocab_size_) * hidden_size_;
  if (!config.embedding_path.empty()) {
    if (config.embedding_quant) {
      // Keep only a temporary upload source until idx6 ION Tensor is initialized.
      // After CreateTensors, EmbeddingRow reads the shared ION buffer directly.
      if (single_file_) {
        if (package_reader_ == nullptr ||
            !package_reader_->Mmap(config.embedding_path, &embedding_weight_data_, &embedding_weight_size_)) {
          MS_LOG(ERROR) << "Failed to mmap embedding weight entry";
          return false;
        }
      } else {
        if (!ReadAsset(config.embedding_path, &embedding_weight_buffer_)) {
          MS_LOG(ERROR) << "Failed to read embedding weight bin";
          return false;
        }
        embedding_weight_data_ = embedding_weight_buffer_.data();
        embedding_weight_size_ = embedding_weight_buffer_.size();
      }
      if (embedding_weight_data_ == nullptr || embedding_weight_size_ == 0) {
        MS_LOG(ERROR) << "Embedding weight bin is empty";
        return false;
      }
      mark("LoadEmbeddingWeight");
      if (!DequantizeEmbeddingTable(config.scale_gp_size)) {
        return false;
      }
      mark("DequantizeEmbeddingTable");
    } else {
      // fp16: the bin IS the table. mmap (lazy pages) avoids the 272MB eager zero-fill.
      embedding_table_elems_ = embed_size;
      embedding_table_ = static_cast<uint16_t *>(
        ::mmap(nullptr, embed_size * sizeof(uint16_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      if (embedding_table_ == MAP_FAILED) {
        embedding_table_ = nullptr;
        embedding_table_elems_ = 0;
        MS_LOG(ERROR) << "mmap embedding_table failed";
        return false;
      }
      std::vector<uint8_t> bytes;
      if (!ReadAsset(config.embedding_path, &bytes)) {
        MS_LOG(ERROR) << "Failed to read embedding weight bin";
        return false;
      }
      mark("ReadAsset(embedding)");
      if (bytes.size() != embed_size * sizeof(uint16_t)) {
        MS_LOG(ERROR) << "Embedding bin size " << bytes.size() << " != expected " << embed_size * sizeof(uint16_t);
        return false;
      }
      std::memcpy(embedding_table_, bytes.data(), bytes.size());
      mark("memcpy(embedding_table)");
    }
  }

  auto load_fp16 = [&](const std::string &path, const char *what, std::vector<uint16_t> *dst, size_t count) {
    if (path.empty()) {
      return true;
    }
    std::vector<uint8_t> bytes;
    if (!ReadAsset(path, &bytes)) {
      MS_LOG(ERROR) << "Failed to read " << what;
      return false;
    }
    if (bytes.size() != count * sizeof(uint16_t)) {
      MS_LOG(ERROR) << "Unexpected " << what << " size " << bytes.size() << " (expected " << count * sizeof(uint16_t)
                    << ")";
      return false;
    }
    std::memcpy(dst->data(), bytes.data(), bytes.size());
    return true;
  };

  size_t rope_size = static_cast<size_t>(max_length_) * head_dim_;
  sin_buffer_.assign(rope_size, 0);
  if (!load_fp16(config.rope_sin_path, "sin bin", &sin_buffer_, rope_size)) {
    return false;
  }
  mark("rope_sin");
  cos_buffer_.assign(rope_size, 0);
  if (!load_fp16(config.rope_cos_path, "cos bin", &cos_buffer_, rope_size)) {
    return false;
  }
  mark("rope_cos");
  size_t mask_size = static_cast<size_t>(max_length_) * max_length_;
  attention_mask_buffer_.assign(mask_size, 0);
  if (!load_fp16(config.attention_mask_path, "attn mask", &attention_mask_buffer_, mask_size)) {
    return false;
  }
  mark("attention_mask");
  return true;
}

bool NnrtExecutor::DequantizeEmbeddingTable(int group_size) {
  // W4A16 planar layout over file_rows. The packed bin stays resident; individual
  // rows are dequantized on demand by EmbeddingRow (no fp16 table is built).
  if (group_size <= 0 || hidden_size_ <= 0 || hidden_size_ % 2 != 0 || hidden_size_ % group_size != 0) {
    MS_LOG(ERROR) << "Invalid hidden_size " << hidden_size_ << " or scale_gp_size " << group_size;
    return false;
  }
  const size_t packed_per_row = static_cast<size_t>(hidden_size_ / 2);
  const size_t scales_per_row = static_cast<size_t>(hidden_size_ / group_size);
  const size_t row_bytes = packed_per_row + scales_per_row * sizeof(uint16_t);
  if (embedding_weight_size_ % row_bytes != 0) {
    MS_LOG(ERROR) << "Embedding bin size " << embedding_weight_size_ << " is not a multiple of row bytes " << row_bytes;
    return false;
  }
  const size_t rows = embedding_weight_size_ / row_bytes;
  if (rows != static_cast<size_t>(vocab_size_)) {
    MS_LOG(ERROR) << "Embedding bin rows " << rows << " != vocab_size " << vocab_size_;
    return false;
  }
  embed_file_rows_ = rows;
  embed_packed_per_row_ = packed_per_row;
  embed_scales_per_row_ = scales_per_row;
  embed_group_size_ = group_size;
  MS_LOG(INFO) << "W4A16 embedding layout: file rows " << rows << ", group_size " << group_size << ", packed "
               << packed_per_row << "B/row, scales " << scales_per_row << "/row; dequantized on demand";
  return true;
}

bool NnrtExecutor::EmbeddingRow(int tid, uint16_t *dst) {
  if (tid < 0 || tid >= vocab_size_ || dst == nullptr) {
    return false;
  }
  if (!embedding_quant_) {
    // fp16 path: the table is the single resident copy.
    if (embedding_table_ == nullptr) {
      return false;
    }
    std::memcpy(dst, embedding_table_ + static_cast<size_t>(tid) * hidden_size_, hidden_size_ * sizeof(uint16_t));
    return true;
  }
  if (embedding_weight_data_ == nullptr || embed_file_rows_ == 0) {
    return false;
  }
  const auto *packed_base = embedding_weight_data_;
  const auto *scales_base = reinterpret_cast<const uint16_t *>(packed_base + embed_file_rows_ * embed_packed_per_row_);
  DequantizeEmbeddingRow(packed_base + static_cast<size_t>(tid) * embed_packed_per_row_,
                         scales_base + static_cast<size_t>(tid) * embed_scales_per_row_, hidden_size_,
                         embed_group_size_, dst);
  return true;
}

bool NnrtExecutor::PrepareEmbeddingWeightWrite(NN_Tensor *tensor, const void **data, size_t *size) const {
  if (data == nullptr || size == nullptr || embedding_weight_data_ == nullptr || embedding_weight_size_ == 0) {
    return false;
  }
  *data = embedding_weight_data_;
  *size = embedding_weight_size_;
  if (!embedding_quant_) {
    return true;
  }
  auto it = tensor_byte_sizes_.find(tensor);
  if (it == tensor_byte_sizes_.end()) {
    MS_LOG(ERROR) << "idx6 embedding_weight logical capacity unavailable";
    return false;
  }
  if (it->second != embedding_weight_size_) {
    MS_LOG(ERROR) << "idx6 embedding_weight capacity " << it->second << " != embedding bin size "
                  << embedding_weight_size_;
    return false;
  }
  return true;
}

void NnrtExecutor::RecordTensorCapacity(NN_Tensor *tensor, const NN_TensorDesc *desc, size_t index) {
  if (tensor == nullptr || desc == nullptr) {
    return;
  }
  const auto &api = NNRTWrapper::GetApi();
  if (api.TensorDesc_GetByteSize == nullptr) {
    MS_LOG(WARNING) << "NNRT cannot report tensor byte size; WriteTensor capacity guard disabled for input " << index;
    return;
  }
  size_t byte_size = 0;
  if (api.TensorDesc_GetByteSize(desc, &byte_size) == 0) {
    tensor_byte_sizes_[tensor] = byte_size;
  }
}

NN_Tensor *NnrtExecutor::CreateInputTensor(size_t index, const int32_t *shape, size_t dim_count, int32_t dtype) {
  const auto &api = NNRTWrapper::GetApi();
  NN_TensorDesc *desc = api.Executor_CreateInputTensorDesc(nn_executor_, index);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "CreateInputTensorDesc " << index << " failed";
    return nullptr;
  }
  if (api.TensorDesc_SetShape(desc, shape, dim_count) != 0 || api.TensorDesc_SetDataType(desc, dtype) != 0) {
    MS_LOG(ERROR) << "TensorDesc config failed for input " << index;
    api.TensorDesc_Destroy(&desc);
    return nullptr;
  }
  NN_Tensor *t = api.Tensor_Create(device_id_, desc);
  RecordTensorCapacity(t, desc, index);
  api.TensorDesc_Destroy(&desc);
  if (t == nullptr) {
    MS_LOG(ERROR) << "Tensor_Create failed for input " << index;
    return nullptr;
  }
  return t;
}

NN_Tensor *NnrtExecutor::CreateInputTensorFromOmc(size_t index, int32_t fallback_capacity, int32_t fallback_dtype) {
  const auto &api = NNRTWrapper::GetApi();
  NN_TensorDesc *desc = api.Executor_CreateInputTensorDesc(nn_executor_, index);
  if (desc == nullptr) {
    int32_t fallback_shape[1] = {fallback_capacity};
    return CreateInputTensor(index, fallback_shape, 1, fallback_dtype);
  }
  // The device enum-shape gear matcher compares input desc shapes against the ORIGINAL
  // ONNX ranks at RunSync (proven on kirin9020, 2026-07-25: rc=0 for both the seq=128 and
  // seq=1 gears). embedding_weight is 1-dim INT8 [capacity] in the ONNX, but the NNRT/omc
  // desc is padded to [capacity,1,1,1] — passing that padded desc to Tensor_Create is what
  // broke gear matching. Keep only the CAPACITY (byte size) from the omc desc and force
  // the original 1-dim shape; without a byte-size query (old NNRT), keep the caller's
  // fallback (the embedding bin size).
  int32_t flat_shape[1] = {fallback_capacity};
  if (api.TensorDesc_GetByteSize != nullptr) {
    size_t byte_size = 0;
    if (api.TensorDesc_GetByteSize(desc, &byte_size) == 0 && byte_size > 0 &&
        byte_size <= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      flat_shape[0] = static_cast<int32_t>(byte_size);
    }
  }
  if (api.TensorDesc_SetShape(desc, flat_shape, 1) != 0 || api.TensorDesc_SetDataType(desc, fallback_dtype) != 0) {
    MS_LOG(ERROR) << "TensorDesc config failed for input " << index;
    api.TensorDesc_Destroy(&desc);
    return nullptr;
  }
  NN_Tensor *t = api.Tensor_Create(device_id_, desc);
  RecordTensorCapacity(t, desc, index);
  api.TensorDesc_Destroy(&desc);
  return t;
}

bool NnrtExecutor::CreateTensors() {
  const int cs = static_cast<int>(chunk_size_);
  const int hd = static_cast<int>(head_dim_);
  const int hs = hidden_size_;
  const int ml = static_cast<int>(max_length_);

  // On-device gear contract (kirin9020 enum-shape, seq in {1, chunk_size}): the gear
  // matcher compares input desc shapes against the ORIGINAL ONNX ranks at RunSync (proven
  // on kirin9020, 2026-07-25, rc=0 for both gears): scalars INT32 [1] (1-dim),
  // rope/inputs_embeds F16 [1,seq,*] (3-dim), attention_mask F16 [1,1,seq,max_len] (4-dim
  // IS its ONNX rank), embedding_weight INT8 [capacity] (1-dim). NNRT-padded 4-dim descs
  // are rejected by the matcher. Gear matching is shape-based, so the prefill set uses
  // seq = chunk_size and the decode set seq = 1.
  prefill_inputs_.resize(7);
  int32_t s1[1] = {1};
  int32_t s_rope_p[3] = {1, cs, hd};
  int32_t s_embed_p[3] = {1, cs, hs};
  int32_t s_mask_p[4] = {1, 1, cs, ml};
  prefill_inputs_[0] = CreateInputTensor(0, s1, 1, kOhNnInt32);           // valid_seq_len
  prefill_inputs_[1] = CreateInputTensor(1, s1, 1, kOhNnInt32);           // lmhead_idx
  prefill_inputs_[2] = CreateInputTensor(2, s_rope_p, 3, kOhNnFloat16);   // rope_cos
  prefill_inputs_[3] = CreateInputTensor(3, s_rope_p, 3, kOhNnFloat16);   // rope_sin
  prefill_inputs_[4] = CreateInputTensor(4, s_embed_p, 3, kOhNnFloat16);  // input_embeds
  prefill_inputs_[5] = CreateInputTensor(5, s_mask_p, 4, kOhNnFloat16);   // attn_mask
  if (embedding_weight_size_ > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    MS_LOG(ERROR) << "embedding_weight size exceeds NNRT INT32 capacity";
    return false;
  }
  const auto embed_capacity = static_cast<int32_t>(embedding_weight_size_);
  prefill_inputs_[6] = CreateInputTensorFromOmc(6, embed_capacity, kOhNnUint8);  // embedding_weight

  // Decode group (seq = 1 gear). input_embeds fp16.
  decode_inputs_.resize(7);
  int32_t s_rope_d[3] = {1, 1, hd};
  int32_t s_embed_d[3] = {1, 1, hs};
  int32_t s_mask_d[4] = {1, 1, 1, ml};
  decode_inputs_[0] = CreateInputTensor(0, s1, 1, kOhNnInt32);
  decode_inputs_[1] = CreateInputTensor(1, s1, 1, kOhNnInt32);
  decode_inputs_[2] = CreateInputTensor(2, s_rope_d, 3, kOhNnFloat16);
  decode_inputs_[3] = CreateInputTensor(3, s_rope_d, 3, kOhNnFloat16);
  decode_inputs_[4] = CreateInputTensor(4, s_embed_d, 3, kOhNnFloat16);  // decode fp16
  decode_inputs_[5] = CreateInputTensor(5, s_mask_d, 4, kOhNnFloat16);
  decode_inputs_[6] = prefill_inputs_[6];  // embedding_weight shared constant

  // Logits output tensor [1,1,1,model_vocab] fp32. The model's logits width may exceed the
  // tokenizer vocab (two-vocab cropped model); sampling reads only the first vocab_size_.
  {
    const auto &api = NNRTWrapper::GetApi();
    NN_TensorDesc *desc = api.Executor_CreateOutputTensorDesc(nn_executor_, 0);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "CreateOutputTensorDesc failed";
      return false;
    }
    int32_t s_log[4] = {1, 1, 1, static_cast<int32_t>(model_vocab_)};
    if (api.TensorDesc_SetShape(desc, s_log, 4) != 0 || api.TensorDesc_SetDataType(desc, kOhNnFloat32) != 0) {
      api.TensorDesc_Destroy(&desc);
      return false;
    }
    logits_tensor_ = api.Tensor_Create(device_id_, desc);
    api.TensorDesc_Destroy(&desc);
    if (logits_tensor_ == nullptr) {
      MS_LOG(ERROR) << "logits Tensor_Create failed";
      return false;
    }
  }

  if (std::any_of(prefill_inputs_.begin(), prefill_inputs_.end(), [](const auto *t) { return t == nullptr; }) ||
      std::any_of(decode_inputs_.begin(), decode_inputs_.end(), [](const auto *t) { return t == nullptr; })) {
    return false;
  }

  // Write the W4A16 constant once, then use this CPU-accessible ION Tensor as the
  // sole packed embedding owner for both NNRT and per-token CPU dequantization.
  if (embedding_quant_) {
    const void *embed_data = nullptr;
    size_t embed_size = 0;
    if (!PrepareEmbeddingWeightWrite(prefill_inputs_[6], &embed_data, &embed_size) ||
        !WriteTensor(prefill_inputs_[6], embed_data, embed_size)) {
      MS_LOG(ERROR) << "Failed to write embedding_weight input tensor";
      return false;
    }
    const auto &api = NNRTWrapper::GetApi();
    embedding_weight_data_ = static_cast<const uint8_t *>(api.Tensor_GetDataBuffer(prefill_inputs_[6]));
    if (embedding_weight_data_ == nullptr) {
      MS_LOG(ERROR) << "embedding_weight ION buffer is unavailable";
      return false;
    }
    embedding_weight_buffer_.clear();
    embedding_weight_buffer_.shrink_to_fit();
    ReclaimEmbeddingWeightPages();
  }

  // Assemble I/O arrays. KV layout is interleaved: K0,V0,K1,V1,... starting at index 7.
  // Output KV entries are the SAME tensor objects as inputs (in-place model update).
  auto assemble = [&](std::vector<NN_Tensor *> &in, std::vector<NN_Tensor *> &out,
                      const std::vector<NN_Tensor *> &non_kv) {
    in = non_kv;  // 7 non-KV (including embedding_weight at idx6)
    for (int i = 0; i < num_layers_; ++i) {
      in.push_back(kv_cache_manager_.GetKeyTensor(i));    // K_i
      in.push_back(kv_cache_manager_.GetValueTensor(i));  // V_i (interleaved)
    }
    out.clear();
    out.push_back(logits_tensor_);
    for (int i = 0; i < num_layers_; ++i) {
      out.push_back(kv_cache_manager_.GetKeyTensor(i));    // out_key_i (same object → in-place)
      out.push_back(kv_cache_manager_.GetValueTensor(i));  // out_val_i
    }
  };
  assemble(prefill_in_, prefill_out_, prefill_inputs_);
  assemble(decode_in_, decode_out_, decode_inputs_);
  return true;
}

bool NnrtExecutor::WriteTensor(NN_Tensor *tensor, const void *data, size_t size) {
  if (tensor == nullptr || data == nullptr || size == 0) {
    MS_LOG(ERROR) << "WriteTensor: invalid argument (null tensor/data or zero size)";
    return false;
  }
  auto it = tensor_byte_sizes_.find(tensor);
  if (it != tensor_byte_sizes_.end() && size > it->second) {
    MS_LOG(ERROR) << "WriteTensor: size " << size << " exceeds tensor capacity " << it->second;
    return false;
  }
  const auto &api = NNRTWrapper::GetApi();
  void *buf = api.Tensor_GetDataBuffer(tensor);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "WriteTensor: null data buffer";
    return false;
  }
  std::memcpy(buf, data, size);
  return true;
}

bool NnrtExecutor::Forward(const std::vector<int> &input_ids, int *output_ids, bool is_prefill,
                           std::vector<float> *logits_out) {
  if (nn_executor_ == nullptr) {
    MS_LOG(ERROR) << "Executor not built";
    return false;
  }
  if (output_ids == nullptr || input_ids.empty()) {
    MS_LOG(ERROR) << "Forward invalid params";
    return false;
  }
  if (is_prefill) {
    kv_cache_manager_.Reset();  // fresh sequence: clear KV before prefill
    history_ = 0;
    const bool ok = Prefill(input_ids, output_ids, logits_out);
    if (ok) {
      history_ = static_cast<int64_t>(input_ids.size());
    }
    return ok;
  }
  return Decode(input_ids, output_ids, logits_out);
}

bool NnrtExecutor::Reset() {
  if (nn_executor_ == nullptr) {
    MS_LOG(ERROR) << "Executor not built";
    return false;
  }
  // Same KV clear that prefill performs at the start of a fresh sequence.
  kv_cache_manager_.Reset();
  history_ = 0;
  return true;
}

bool NnrtExecutor::Prefill(const std::vector<int> &input_ids, int *output_ids, std::vector<float> *logits_out) {
  const auto &api = NNRTWrapper::GetApi();
  const int cs = static_cast<int>(chunk_size_);
  const int hd = static_cast<int>(head_dim_);
  const int hs = hidden_size_;
  const int ml = static_cast<int>(max_length_);
  const int total_len = static_cast<int>(input_ids.size());
  if (total_len > static_cast<int>(max_length_)) {
    MS_LOG(ERROR) << "Prefill: input length " << total_len << " exceeds max_length " << max_length_;
    return false;
  }
  const int num_chunks = (total_len + cs - 1) / cs;

  const int64_t sample_vocab = std::min(vocab_size_, model_vocab_);
  std::vector<float> logits(static_cast<size_t>(sample_vocab));

  // reusable staging buffers at chunk_size (right-padded)
  std::vector<uint16_t> embed_buf(cs * hs);
  std::vector<uint16_t> rope_buf(cs * hd);
  std::vector<uint16_t> mask_buf(cs * ml);

  for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    int start = chunk_id * cs;
    int valid = std::min(cs, total_len - start);  // real tokens in this chunk

    // valid_seq_len = global write offset for MsScatterND (device: past[:,:,pos:pos+L]=state).
    // Prefill chunk k must write at start=k*chunk_size (first chunk → 0), NOT the
    // chunk-local valid count. lmhead_idx stays chunk-local: last real token in the
    // right-padded chunk buffer is at index valid-1.
    int32_t vsl = start;
    int32_t lmh = valid - 1;
    if (!WriteTensor(prefill_inputs_[0], &vsl, sizeof(int32_t)) ||
        !WriteTensor(prefill_inputs_[1], &lmh, sizeof(int32_t))) {
      MS_LOG(ERROR) << "Prefill: failed to write scalar inputs";
      return false;
    }

    // input_embeds [1, cs, hs] fp16: valid embeddings + zero right-padding
    for (int i = 0; i < valid; ++i) {
      int tid = input_ids[start + i];
      if (tid < 0 || tid >= vocab_size_) {
        MS_LOG(ERROR) << "token oob tid=" << tid << " vocab=" << vocab_size_;
        return false;
      }
      if (!EmbeddingRow(tid, embed_buf.data() + i * hs)) {
        MS_LOG(ERROR) << "EmbeddingRow failed for tid=" << tid;
        return false;
      }
    }
    std::memset(embed_buf.data() + valid * hs, 0, (cs - valid) * hs * sizeof(uint16_t));
    if (!WriteTensor(prefill_inputs_[4], embed_buf.data(), cs * hs * sizeof(uint16_t))) {
      MS_LOG(ERROR) << "Prefill: failed to write inputs_embeds";
      return false;
    }

    // rope_cos/sin [1, cs, hd] from positions [start, start+cs)
    std::memcpy(rope_buf.data(), cos_buffer_.data() + static_cast<size_t>(start) * hd, cs * hd * sizeof(uint16_t));
    if (!WriteTensor(prefill_inputs_[2], rope_buf.data(), cs * hd * sizeof(uint16_t))) {
      MS_LOG(ERROR) << "Prefill: failed to write rope_cos";
      return false;
    }
    std::memcpy(rope_buf.data(), sin_buffer_.data() + static_cast<size_t>(start) * hd, cs * hd * sizeof(uint16_t));
    if (!WriteTensor(prefill_inputs_[3], rope_buf.data(), cs * hd * sizeof(uint16_t))) {
      MS_LOG(ERROR) << "Prefill: failed to write rope_sin";
      return false;
    }

    // attn_mask [1,1,cs,ml] from attention_mask_buffer_ rows [start, start+cs)
    std::memcpy(mask_buf.data(), attention_mask_buffer_.data() + static_cast<size_t>(start) * ml,
                cs * ml * sizeof(uint16_t));
    if (!WriteTensor(prefill_inputs_[5], mask_buf.data(), cs * ml * sizeof(uint16_t))) {
      MS_LOG(ERROR) << "Prefill: failed to write attention_mask";
      return false;
    }

    // KV inputs already in prefill_in_ arrays (Reset cleared them before prefill)

    NnrtReturnCode ret = api.Executor_RunSync(nn_executor_, prefill_in_.data(), prefill_in_.size(), prefill_out_.data(),
                                              prefill_out_.size());
    if (ret != 0) {
      MS_LOG(ERROR) << "RunSync failed prefill chunk " << chunk_id << ": " << ret;
      return false;
    }

    // logits = output[0] GetDataBuffer (KV outputs are same objects → already updated in place)
    std::memcpy(logits.data(), api.Tensor_GetDataBuffer(logits_tensor_), logits.size() * sizeof(float));
  }

  if (logits_out != nullptr) {
    *logits_out = logits;  // host-side sampling owns the argmax
  } else {
    *output_ids = ArgMax(logits);
  }
  return true;
}

bool NnrtExecutor::Decode(const std::vector<int> &input_ids, int *output_ids, std::vector<float> *logits_out) {
  const auto &api = NNRTWrapper::GetApi();
  const int hd = static_cast<int>(head_dim_);
  const int hs = hidden_size_;
  const int ml = static_cast<int>(max_length_);
  const int history = static_cast<int>(history_);
  if (history < 0 || history >= ml) {
    MS_LOG(ERROR) << "Decode: history out of range: " << history;
    return false;
  }
  const int last = input_ids.back();
  if (last < 0 || last >= vocab_size_) {
    MS_LOG(ERROR) << "token oob tid=" << last << " vocab=" << vocab_size_;
    return false;
  }

  // MsScatterND write offset = absolute index of the new token (= history).
  // rope_cos/sin/mask are already absolute rows fed separately; valid_seq_len is
  // not used for RoPE in the .omc graph (only Scatter consumes it). Decode state
  // has seq_len=1, so past[:,:,history:history+1,:] = state.
  int32_t vsl = history;
  int32_t lmh = 0;
  if (!WriteTensor(decode_inputs_[0], &vsl, sizeof(int32_t)) ||
      !WriteTensor(decode_inputs_[1], &lmh, sizeof(int32_t))) {
    MS_LOG(ERROR) << "Decode: failed to write scalar inputs";
    return false;
  }

  // input_embeds [1,1,hs] fp16: dequantize (W4A16) or copy (fp16) this token's row
  // straight into the tensor buffer — no fp16 table lookup, no extra memcpy.
  void *embeds_buf = api.Tensor_GetDataBuffer(decode_inputs_[4]);
  if (embeds_buf == nullptr) {
    MS_LOG(ERROR) << "Decode: null inputs_embeds buffer";
    return false;
  }
  if (!EmbeddingRow(last, static_cast<uint16_t *>(embeds_buf))) {
    MS_LOG(ERROR) << "Decode: EmbeddingRow failed for tid=" << last;
    return false;
  }

  // rope [1,1,hd] at history position
  if (!WriteTensor(decode_inputs_[2], cos_buffer_.data() + static_cast<size_t>(history) * hd, hd * sizeof(uint16_t)) ||
      !WriteTensor(decode_inputs_[3], sin_buffer_.data() + static_cast<size_t>(history) * hd, hd * sizeof(uint16_t))) {
    MS_LOG(ERROR) << "Decode: failed to write rope inputs";
    return false;
  }
  // attn_mask [1,1,1,ml] at history row
  if (!WriteTensor(decode_inputs_[5], attention_mask_buffer_.data() + static_cast<size_t>(history) * ml,
                   ml * sizeof(uint16_t))) {
    MS_LOG(ERROR) << "Decode: failed to write attention_mask";
    return false;
  }

  NnrtReturnCode ret =
    api.Executor_RunSync(nn_executor_, decode_in_.data(), decode_in_.size(), decode_out_.data(), decode_out_.size());
  if (ret != 0) {
    MS_LOG(ERROR) << "RunSync failed decode: " << ret << " history=" << history << " last=" << last
                  << " chunk_size=" << chunk_size_ << " max_len=" << ml << " hs=" << hs << " hd=" << hd;
    size_t in_count = 0;
    api.Executor_GetInputCount(nn_executor_, &in_count);
    for (size_t di = 0; di < in_count; ++di) {
      NN_TensorDesc *d = api.Executor_CreateInputTensorDesc(nn_executor_, di);
      if (d != nullptr) {
        int32_t *sh = nullptr;
        size_t sl = 0;
        api.TensorDesc_GetShape(d, &sh, &sl);
        std::string s;
        for (size_t k = 0; k < sl; ++k) s += (k ? "," : "") + std::to_string(sh[k]);
        MS_LOG(ERROR) << "  omc_in[" << di << "] shape=[" << s << "]";
        api.TensorDesc_Destroy(&d);
      }
    }
    return false;
  }

  const int64_t sample_vocab = std::min(vocab_size_, model_vocab_);
  std::vector<float> logits(static_cast<size_t>(sample_vocab));
  std::memcpy(logits.data(), api.Tensor_GetDataBuffer(logits_tensor_), logits.size() * sizeof(float));
  history_ = history + 1;
  if (logits_out != nullptr) {
    *logits_out = logits;  // host-side sampling owns the argmax
  } else {
    *output_ids = ArgMax(logits);
  }
  return true;
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
