// Copyright 2026 Huawei Technologies Co., Ltd
#pragma once

#include "backend/common/backend.h"
#include "backend/nnrt/qwen35/embedding.h"
#include "backend/nnrt/qwen35/qwen35_moe_model.h"

namespace mslite_llm::qwen35 {

// Runs the complete paged model behind the standard tokenizer/sampler pipeline.
class PagedBackend final : public Backend {
 public:
  MSLlmStatus Init(const BackendConfig &config) override;
  MSLlmStatus Prefill(const BackendInput &input, BackendOutput *output) override;
  MSLlmStatus Decode(const BackendInput &input, BackendOutput *output) override;
  MSLlmStatus Reset() override;

 private:
  bool RunChunk(const std::vector<int32_t> &tokens, ModelGear gear);
  bool PrepareInputs(const std::vector<int32_t> &tokens, ModelGear gear);
  bool ReadLogits(ModelGear gear, BackendOutput *output);
  void BuildRopeTable();

  Qwen35MoeModel model_;
  EmbeddingTable embedding_;
  std::vector<uint16_t> cosine_;
  std::vector<uint16_t> sine_;
  std::string error_;
  int32_t cache_position_ = 0;
  int32_t left_padding_ = 0;
  bool initialized_ = false;
};

}  // namespace mslite_llm::qwen35
