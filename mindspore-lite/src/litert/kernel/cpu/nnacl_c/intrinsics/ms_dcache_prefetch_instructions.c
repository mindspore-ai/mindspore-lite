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

#include "nnacl_c/intrinsics/ms_dcache_prefetch_instructions.h"

#if defined(__riscv)
// Linx131 APREFD: data cache auto prefetch control register.
#define LINX131_DCACHE_AUTO_PREFETCH_CSR 0x7C7
#endif

// Set Linx131 data cache auto prefetch configuration and return previous configuration.
uint32_t SetDCacheAutoPrefetchLinx131(uint32_t csr_val) {
#if defined(__riscv)
  uint32_t prev_val = 0;
  asm volatile("csrrw %0, %1, %2" : "=r"(prev_val) : "i"(LINX131_DCACHE_AUTO_PREFETCH_CSR), "r"(csr_val) : "memory");
  return prev_val;
#else
  // Keep host builds linkable; the coder should emit real calls only for matching RISC-V targets.
  (void)csr_val;
  return 0;
#endif
}
