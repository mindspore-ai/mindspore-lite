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
 * @file llm_types.h
 * @brief Public type definitions for the MindSpore Lite LLM C API.
 */

#ifndef MSLLM_TYPES_H
#define MSLLM_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#ifdef MSLLM_BUILDING
#define MSLLM_API __declspec(dllexport)
#else
#define MSLLM_API __declspec(dllimport)
#endif
#else
#define MSLLM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** LLM model handle (lifecycle: Create → Build → Generate* → Destroy). */
typedef struct MSLLMModel *MSLLMModelHandle;

typedef enum {
  kMSLLM_SUCCESS = 0,
  /** invalid handle, null pointer, or out-of-range parameter. */
  kMSLLM_ERROR_INVALID_ARGS = 1,
  /** model package missing, corrupt, or incompatible. */
  kMSLLM_ERROR_MODEL_LOAD = 2,
  /** failure during prefill or decode execution. */
  kMSLLM_ERROR_INFERENCE = 3,
  /** memory or KV cache allocation failure. */
  kMSLLM_ERROR_OOM = 4,
  /** feature reserved in the ABI but not implemented by this runtime. */
  kMSLLM_ERROR_NOT_SUPPORTED = 5,
  /** file or IO error. */
  kMSLLM_ERROR_IO = 6,
  /** model already has an in-flight generation. */
  kMSLLM_ERROR_BUSY = 7,
  /** prompt alone exceeds max_context_len; nothing was generated. */
  kMSLLM_ERROR_CONTEXT_OVERFLOW = 8,
  /** unexpected internal error. */
  kMSLLM_ERROR_INTERNAL = 9,
  /** caller-provided buffer is too small; required_size (if non-NULL)
   *  is filled with the needed byte count. Nothing was written. */
  kMSLLM_ERROR_BUFFER_TOO_SMALL = 10,
} MSLLMStatus;

typedef enum {
  /** the LLM is running, not finished. */
  kMSLLM_RUNNING = 0,
  /** the LLM is finished by EOS token. */
  kMSLLM_FINISHED_BY_EOS = 1,
  /** the LLM is finished by exceeding max_context_length limit. */
  kMSLLM_FINISHED_BY_MAX_CONTEXT_LENGTH = 2,
  /** the LLM is finished by exceeding max_output_length limit. */
  kMSLLM_FINISHED_BY_MAX_OUTPUT_LENGTH = 3,
  /** the LLM is stopped by user. */
  kMSLLM_STOPPED_BY_USER = 4,
  /** the LLM is finished due to an inference error (INFERENCE / OOM);
   *  the exact error code is returned by the generate function. */
  kMSLLM_FINISHED_BY_ERROR = 5,
} MSLLMFinishReason;

/** Message roles for chat template rendering. */
typedef enum {
  MSLLM_ROLE_SYSTEM = 0,
  MSLLM_ROLE_USER = 1,
  MSLLM_ROLE_ASSISTANT = 2,
} MSLLMRole;

/** A single message, consumed by MSLLMApplyChatTemplate. */
typedef struct {
  MSLLMRole role;
  const char *content;
} MSLLMChatMessage;

typedef struct {
  /** Maximum number of new tokens to generate. 0 = no explicit cap (generate
   *  until EOS or the context window is exhausted); negative is invalid. */
  int32_t max_new_tokens;

  /** Sampling parameters. do_sample=false → greedy (argmax); all sampling
   *  fields below are then ignored. */
  bool do_sample;
  /** Sampling temperature, valid range [0, 2]. 0 = greedy. */
  float temperature;
  /** top_k, valid range >= 0; 0 = disabled. */
  int32_t top_k;
  /** top_p (nucleus), valid range [0, 1]; 0 or 1 = disabled. */
  float top_p;
  float repetition_penalty;
} MSLLMGenerationConfig;

/**
 * @brief Streaming callback, invoked once per generated token.
 *
 * Runs on the caller's thread, inside the MSLLMStreamGenerate call.
 * Contract:
 *   - MUST NOT throw: exceptions crossing the extern "C" boundary terminate.
 *   - @p token is only valid for the duration of the call; copy to retain.
 *   - MUST NOT call back into any other MSLLM* API from within the callback
 *     (including MSLLMAbort): behaviour is undefined and strongly discouraged.
 *     Request termination from the thread that invoked MSLLMStreamGenerate.
 *
 * @param token Incremental text, NULL on the final invocation.
 * @param reason kMSLLM_RUNNING while generating, terminal value on the last
 *        call (EOS / length limits / user abort / error).
 * @param user_data User pointer passed through from MSLLMStreamGenerate.
 */
typedef void (*MSLLMStreamCallback)(const char *token, MSLLMFinishReason reason, void *user_data);

#ifdef __cplusplus
}
#endif

#endif  // MSLLM_TYPES_H
