/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INNER_INNER_PROMPT_FLASH_ATTENTION_UTILS_H
#define INNER_INNER_PROMPT_FLASH_ATTENTION_UTILS_H

#include <cstdio>
#include <cstring>

/*!
 * \file inner_prompt_flash_attention_utils.h
 * \brief Self-contained host-side (tiling/infershape) helper macros for the
 *        InnerPromptFlashAttention custom op.
 *
 *        CANN's op_common/log/log.h normally provides the full OpLog machinery,
 *        but it ships under pkg_inc/op_common/ which is NOT on this project's
 *        include path (only ${ASCEND_CANN_PACKAGE_PATH}/include is). This stub
 *        therefore provides the minimum macro surface the tiling/infershape
 *        sources use: OP_LOG* (fprintf to stderr), OP_CHECK_IF,
 *        OP_CHECK_NULL_WITH_CONTEXT, and the no-op OPS_REPORT_* error-report
 *        macros (mirroring ops-transformer's err/ops_err.h).
 *
 *        opName passed to OP_LOG* may be a const char*, std::string or a gert
 *        context pointer, so it is deliberately discarded ((void)(opName))
 *        rather than formatted -- only the level tag + the caller's format
 *        string reach stderr.
 */

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x) __builtin_expect((x), 1)

#define OP_CHECK_IF(condition, log, return_expr) \
  do {                                           \
    if (unlikely(condition)) {                   \
      return_expr;                               \
    }                                            \
  } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
  do {                                           \
    if (unlikely((ptr) == nullptr)) {            \
      return ge::GRAPH_FAILED;                   \
    }                                            \
  } while (0)

#define OP_LOGI(opName, ...)                          \
  do {                                                \
    (void)(opName);                                   \
    fprintf(stderr, "[InnerPFA][INFO] " __VA_ARGS__); \
    fprintf(stderr, "\n");                            \
  } while (0)
#define OP_LOGW(opName, ...)                          \
  do {                                                \
    (void)(opName);                                   \
    fprintf(stderr, "[InnerPFA][WARN] " __VA_ARGS__); \
    fprintf(stderr, "\n");                            \
  } while (0)
#define OP_LOGE(opName, ...)                           \
  do {                                                 \
    (void)(opName);                                    \
    fprintf(stderr, "[InnerPFA][ERROR] " __VA_ARGS__); \
    fprintf(stderr, "\n");                             \
  } while (0)
#define OP_LOGD(opName, ...)                           \
  do {                                                 \
    (void)(opName);                                    \
    fprintf(stderr, "[InnerPFA][DEBUG] " __VA_ARGS__); \
    fprintf(stderr, "\n");                             \
  } while (0)

// No-op error-report macros (mirror ops-transformer err/ops_err.h; the report
// channel is not wired in standalone builds).
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) \
  do {                                             \
  } while (0)

#endif  // INNER_INNER_PROMPT_FLASH_ATTENTION_UTILS_H
