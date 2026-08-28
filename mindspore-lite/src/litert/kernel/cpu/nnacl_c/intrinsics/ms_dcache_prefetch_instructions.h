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

#ifndef NNACL_INTRINSICS_MS_DCACHE_PREFETCH_INSTRUCTIONS_H_
#define NNACL_INTRINSICS_MS_DCACHE_PREFETCH_INSTRUCTIONS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Set Linx131 data cache auto prefetch configuration and return previous configuration.
uint32_t SetDCacheAutoPrefetchLinx131(uint32_t csr_val);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_INTRINSICS_MS_DCACHE_PREFETCH_INSTRUCTIONS_H_
