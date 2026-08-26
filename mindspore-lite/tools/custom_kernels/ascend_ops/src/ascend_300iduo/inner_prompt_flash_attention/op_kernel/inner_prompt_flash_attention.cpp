/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file inner_prompt_flash_attention.cpp
 * \brief Slim 310P entry for the vendored InnerPromptFlashAttention custom op.
 *
 * This is a trimmed copy of the upstream ops-transformer kernel entry that keeps
 * ONLY the Atlas-inference-series (Ascend310P) FP16 path implemented by
 * InnerPromptFlashAttentionS1s2Bns1X310. The upstream QINT8/GQA prefill path
 * (PromptAttentionPrefill) is intentionally dropped so we don't pull in
 * unpad_flash_attention_common.h / prompt_attention_prefill.h and the heavy
 * act-template iterator headers — the FP16 aclnn path never hits those keys.
 *
 * 4 FP16 tiling keys are dispatched (see prompt_flash_attention_tilingkey.h):
 *   12288 -> BNSD, high-performance      22288 -> BSH,  high-performance
 *   12888 -> BNSD, high-precision        22888 -> BSH,  high-precision
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"   // NOLINT(build/include_subdir)
#include "kernel_cube_intf.h"  // NOLINT(build/include_subdir)
#else
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#endif
#include "lib/matmul_intf.h"

#include "inner_prompt_flash_attention_tilingkey.h"            // NOLINT(build/include_subdir)
#include "inner_prompt_flash_attention_s1s2_bns1_x310_base.h"  // NOLINT(build/include_subdir)
#include "inner_prompt_flash_attention_s1s2_bns1_x310.h"       // NOLINT(build/include_subdir)

using namespace matmul;  // NOLINT(build/namespaces)

#define PFA310_TILING_DATA(tiling)                                                          \
  GET_TILING_DATA_WITH_STRUCT(InnerPromptFlashAttentionTilingData, tiling_data_in, tiling); \
  const InnerPromptFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in;      \
  const TCubeTiling *__restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);            \
  const TCubeTiling *__restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect)

#define PFA310_INVOKE(templateClass, ...)                                                                            \
  TPipe tPipe;                                                                                                       \
  do {                                                                                                               \
    if (query == nullptr) {                                                                                          \
      return;                                                                                                        \
    }                                                                                                                \
    PFA310_TILING_DATA(tiling);                                                                                      \
    templateClass<__VA_ARGS__> op;                                                                                   \
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                         \
    op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, nullptr, nullptr, nullptr, \
            nullptr, nullptr, nullptr, attentionOut, nullptr, user, tiling_data, tiling, &tPipe);                    \
    op.InitMsd(nullptr, nullptr, nullptr, nullptr);                                                                  \
    op.Process();                                                                                                    \
  } while (0)

extern "C" __global__ __aicore__ void inner_prompt_flash_attention(
  __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
  __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
  __gm__ uint8_t *deq_scale1, __gm__ uint8_t *quant_scale1, __gm__ uint8_t *deq_scale2, __gm__ uint8_t *quant_scale2,
  __gm__ uint8_t *quant_offset2, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  __gm__ uint8_t *user = GetUserWorkspace(workspace);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  TILING_KEY_IS(QINT8_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);     // 12288
  TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);     // 22288
  TILING_KEY_IS(QFP4E1M2_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);  // 12888
  TILING_KEY_IS(QFP4E1M2_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);  // 22888

#if TILING_KEY_VAR == QINT8_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
  PFA310_INVOKE(InnerPromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half>);
#elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
  PFA310_INVOKE(InnerPromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half>);
#elif TILING_KEY_VAR == QFP4E1M2_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
  PFA310_INVOKE(InnerPromptFlashAttentionS1s2Bns1X310,
                PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
#elif TILING_KEY_VAR == QFP4E1M2_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
  PFA310_INVOKE(InnerPromptFlashAttentionS1s2Bns1X310,
                PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
#endif
}
