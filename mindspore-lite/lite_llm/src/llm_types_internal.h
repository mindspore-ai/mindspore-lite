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
/**
 * @file llm_types_internal.h
 * @brief Internal-only type definitions for the lite_llm C++ engine.
 *
 * These types are the legacy MSLlm* surface that the engine's internal
 * components (pipeline / manifest / sampler / tokenizer / backend) use
 * internally. They are NOT part of the public C API (see llm/llm_types.h)
 * and must not be included from public-facing headers.
 */

#ifndef MSLLM_TYPES_INTERNAL_H
#define MSLLM_TYPES_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "llm/llm_types.h"  // MSLLMRole (for the MSLlmChatRole alias)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MSLLM_SUCCESS = 0,
  MSLLM_ERROR_INVALID_ARGS = 1,
  MSLLM_ERROR_MODEL_LOAD = 2,
  MSLLM_ERROR_INFERENCE = 3,
  MSLLM_ERROR_OOM = 4,
  MSLLM_ERROR_NOT_SUPPORTED = 5,
  MSLLM_ERROR_IO = 6,
  MSLLM_ERROR_INTERNAL = 7,
} MSLlmStatus;

typedef enum {
  MSLLM_BACKEND_NNRT = 0,
  MSLLM_BACKEND_QNN = 1,
  MSLLM_BACKEND_METAL = 2,
  MSLLM_BACKEND_GPU = 3,
} MSLlmBackendType;

typedef enum {
  MSLLM_DTYPE_FLOAT32 = 0,
  MSLLM_DTYPE_FLOAT16 = 1,
  MSLLM_DTYPE_INT8 = 2,
  MSLLM_DTYPE_INT4 = 3,
  MSLLM_DTYPE_BFLOAT16 = 4,
} MSLlmDType;

typedef enum {
  MSLLM_SAMPLER_GREEDY = 0,
  MSLLM_SAMPLER_TOP_K = 1,
  MSLLM_SAMPLER_TOP_P = 2,
  MSLLM_SAMPLER_TEMPERATURE = 3,
} MSLlmSamplerStrategy;

typedef MSLLMRole MSLlmChatRole;

typedef struct {
  MSLLMRole role;
  const char *content;
} MSLlmChatMessage;

typedef struct {
  MSLlmBackendType backend_type;
  int32_t num_threads;
  int32_t npu_device_id;
  const char *cache_dir;
} MSLlmEngineConfig;

typedef struct {
  int32_t max_context_len;
  int32_t max_batch_size;
  MSLlmDType kv_cache_dtype;
  int32_t paged_attention_block_size;
  float kv_cache_mem_ratio;
  int32_t num_layers;
  int32_t num_heads;
  int32_t num_kv_heads;
  int32_t head_dim;
  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t vocab_size;
  int32_t max_position_embeddings;
  float rope_theta;
  float norm_eps;
  int32_t tie_word_embeddings;
} MSLlmModelConfig;

typedef struct {
  int32_t max_new_tokens;
  int32_t max_context_len;
  const char **stop_sequences;
  size_t num_stop_sequences;
  int32_t override_sampler;
  float temperature;
  int32_t top_k;
  float top_p;
  float repetition_penalty;
  float presence_penalty;
  float frequency_penalty;
  int32_t seed;
  const char *lora_name;
  float lora_scale;
  const int32_t *logit_bias_tokens;
  const float *logit_bias_values;
  size_t num_logit_biases;
} MSLlmGenerateConfig;

#ifdef __cplusplus
}
#endif

#endif  // MSLLM_TYPES_INTERNAL_H
