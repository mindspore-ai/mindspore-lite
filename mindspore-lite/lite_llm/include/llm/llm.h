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
 * @file llm.h
 * @brief Public C API for MindSpore Lite LLM text generation.
 */

#ifndef MSLLM_H
#define MSLLM_H

#include "llm/llm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create an empty LLM model handle.
 *
 * @return LLM model handle, or NULL on allocation failure.
 */
MSLLM_API MSLLMModelHandle MSLLMCreateModel(void);

/**
 * @brief Destroy the LLM model handle and release all resources.
 *
 * Refuses (kMSLLM_ERROR_BUSY) while a generation is in-flight on this model.
 * Correct teardown sequence: Abort → wait for StreamGenerate to return →
 * Destroy.
 *
 * @param llm_model LLM model handle.
 * @return kMSLLM_SUCCESS on success.
 */
MSLLM_API MSLLMStatus MSLLMDestroyModel(MSLLMModelHandle llm_model);

/**
 * @brief Load and build the LLM model from an .msl package.
 *
 * Loads the model graph, tokenizer, sampler config, and chat template from
 * the given .msl package. Must be called exactly once, after Create and
 * before any Generate call: re-building in READY returns
 * kMSLLM_ERROR_NOT_SUPPORTED, and building while generating returns
 * kMSLLM_ERROR_BUSY.
 *
 * @param llm_model LLM model handle.
 * @param model_path Path to the .msl model package.
 * @return kMSLLM_SUCCESS on success, or an error code.
 */
MSLLM_API MSLLMStatus MSLLMBuildModel(MSLLMModelHandle llm_model, const char *model_path);

/**
 * @brief Set the generation configuration.
 *
 * Unspecified fields keep the model's built-in defaults. Boundary validation:
 * max_new_tokens < 0, temperature outside [0, 2], top_k < 0, or top_p outside
 * [0, 1] returns kMSLLM_ERROR_INVALID_ARGS and preserves the previous config.
 * Returns kMSLLM_ERROR_BUSY while a generation is in-flight on this model.
 *
 * @param llm_model LLM model handle.
 * @param config Generation configuration (struct, not JSON).
 * @return kMSLLM_SUCCESS or error code.
 */
MSLLM_API MSLLMStatus MSLLMSetGenerationConfig(MSLLMModelHandle llm_model, const MSLLMGenerationConfig config);

/**
 * @brief Get the current generation configuration.
 *
 * @param llm_model LLM model handle.
 * @param config [out] Receives the current generation config (a copy).
 * @return kMSLLM_SUCCESS on success, kMSLLM_ERROR_INVALID_ARGS if
 *         llm_model or config is NULL.
 */
MSLLM_API MSLLMStatus MSLLMGetGenerationConfig(MSLLMModelHandle llm_model, MSLLMGenerationConfig *config);

/**
 * @brief Render a full prompt from role/content messages using the model's
 *        chat template (loaded from the model package).
 *
 * Pure text rendering: no tokenization is involved.
 *
 * @param llm_model LLM model handle.
 * @param messages Array of role/content messages (e.g. full multi-turn history).
 * @param num_messages Number of messages.
 * @param add_generation_prompt Non-zero to append the generation prompt
 *        (e.g. the trailing assistant-start marker); zero to render the
 *        conversation so far without it.
 * @param generated_prompt Caller-provided output buffer for the rendered text.
 * @param prompt_size Size of generated_prompt in bytes.
 * @return kMSLLM_SUCCESS on success, kMSLLM_ERROR_BUFFER_TOO_SMALL if the
 *         buffer is insufficient, kMSLLM_ERROR_MODEL_LOAD if the package has
 *         no chat template (runtime ships no builtin renderer), or
 *         another error code.
 */
MSLLM_API MSLLMStatus MSLLMApplyChatTemplate(MSLLMModelHandle llm_model, const MSLLMChatMessage *messages,
                                             int num_messages, int add_generation_prompt, char *generated_prompt,
                                             int prompt_size);

/**
 * @brief Generate text from a prompt, blocking until finished.
 *
 * Writes the complete generated text into the caller-provided buffer.
 * Returns kMSLLM_ERROR_BUFFER_TOO_SMALL if the buffer is too small;
 * retry with a larger buffer (full re-generation).
 *
 * Non-streaming: cannot be aborted via MSLLMAbort.
 *
 * @param llm_model LLM model handle.
 * @param prompt Input prompt text.
 * @param generated_text Caller-provided output buffer.
 * @param text_size Size of generated_text in bytes.
 * @return kMSLLM_SUCCESS or error code.
 */
MSLLM_API MSLLMStatus MSLLMGenerate(MSLLMModelHandle llm_model, const char *prompt, char *generated_text,
                                    int text_size);

/**
 * @brief Generate text streamly: one callback invocation per token.
 *
 * BLOCKS until generation completes. The decode loop runs on the calling
 * thread; the callback is invoked on that same thread. The function returns
 * once the last token is produced — return means completion.
 *
 * Call this from a Task/Worker on UI platforms to avoid blocking the UI.
 *
 * @param llm_model LLM model handle.
 * @param prompt Input prompt text.
 * @param callback Called once per generated token, and once at the end
 *        (token=NULL) with the terminal reason.
 * @param user_data Opaque pointer passed through to the callback.
 * @return kMSLLM_SUCCESS on success, or error code. Generation end
 *         reasons are reported via the callback.
 */
MSLLM_API MSLLMStatus MSLLMStreamGenerate(MSLLMModelHandle llm_model, const char *prompt, MSLLMStreamCallback callback,
                                          void *user_data);

/**
 * @brief Request early termination of an in-progress streaming generation.
 *
 * Only affects MSLLMStreamGenerate; the flag is set only while a generation
 * is in-flight, otherwise it is a no-op that still returns kMSLLM_SUCCESS.
 * Has no effect on MSLLMGenerate (non-streaming is not abortable).
 * Safe to call from any thread (except from inside the streaming callback
 * itself — calling back into any MSLLM* API there is undefined, see
 * MSLLMStreamCallback). Non-blocking: only sets a flag.
 *
 * The blocked MSLLMStreamGenerate call will return kMSLLM_SUCCESS,
 * and the final callback will have reason = kMSLLM_STOPPED_BY_USER.
 *
 * @param llm_model LLM model handle.
 * @return kMSLLM_SUCCESS.
 */
MSLLM_API MSLLMStatus MSLLMAbort(MSLLMModelHandle llm_model);

#ifdef __cplusplus
}
#endif

#endif  // MSLLM_H
