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

#ifndef INNER_INNER_QUANT_BATCH_MATMUL_V3_UTILS_H_
#define INNER_INNER_QUANT_BATCH_MATMUL_V3_UTILS_H_

#include <cstdio>

// Self-contained host-side (tiling/infershape) logging macros for the
// InnerQuantBatchMatmulV3 custom op.
//
// CANN's op_common/log/log.h provides the full OpLog machinery, but nothing
// in the opbuild include closure (register/tiling headers) pulls it in on
// CANN 9.2, and its pkg_inc location is not on this project's include path
// (the opbuild passes only ${ASCEND_CANN_PACKAGE_PATH}/include). This stub
// provides the minimum macro surface the tiling source uses (fprintf to
// stderr), mirroring the delivered inner_prompt_flash_attention utils stub.
//
// opName passed to OP_LOG* is deliberately discarded ((void)(opName)); only
// the level tag + the caller's format string reach stderr.
#define OP_LOGW(opName, ...)                         \
  do {                                               \
    (void)(opName);                                  \
    fprintf(stderr, "[IQBMMV3][WARN] " __VA_ARGS__); \
    fprintf(stderr, "\n");                           \
  } while (0)
#define OP_LOGE(opName, ...)                          \
  do {                                                \
    (void)(opName);                                   \
    fprintf(stderr, "[IQBMMV3][ERROR] " __VA_ARGS__); \
    fprintf(stderr, "\n");                            \
  } while (0)
#define OP_LOGD(opName, ...)                          \
  do {                                                \
    (void)(opName);                                   \
    fprintf(stderr, "[IQBMMV3][DEBUG] " __VA_ARGS__); \
    fprintf(stderr, "\n");                            \
  } while (0)

#endif  // INNER_INNER_QUANT_BATCH_MATMUL_V3_UTILS_H_
