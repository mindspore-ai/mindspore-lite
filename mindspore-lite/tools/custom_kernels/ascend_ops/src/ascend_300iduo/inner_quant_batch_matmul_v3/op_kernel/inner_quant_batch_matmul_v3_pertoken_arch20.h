/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __QUANT_BATCH_MATMUL_V3_PERTOKEN_ARCH20_H__
#define __QUANT_BATCH_MATMUL_V3_PERTOKEN_ARCH20_H__

#include "./inner_quant_batch_matmul_v3_tiling_data.h"
// VENDORED ADAPTATION (no origin counterpart): the origin's build
// environment (atcos -I over adv_api) provided CubeFormat; the CANN
// tikcfw kernel_operator.h chain does not, so the enum is vendored
// verbatim from CANN's adv_api/matmul/
// matmul_config.h -- see inner_quant_batch_matmul_v3_cube_format.h.
#include "./inner_quant_batch_matmul_v3_cube_format.h"
#include "utils/tbmm_common.h"
#include "utils/tbmm_common_func.h"
#include "utils/mem.h"
#include "utils/iterator.h"
#include "utils/utils.h"

namespace PpMatMulNS {

constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_INT8 = 32768;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_8 = 8;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_64 = 64;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t MAX_REPEAT_TIMES = 255;
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t L1_PINGPONG_BUFFER_LEN_INT8 = 131072;
constexpr uint32_t MAX_NUMEL_INST_B32 = 255 * 64;
constexpr uint32_t UB_SCALE_BLOCK_SIZE = 8;

constexpr uint32_t UB_C_OFFSET = 8 * 1024;
constexpr uint32_t UB_C_TEMP_OFFSET = 8 * 1024 + 64 * 1024;
constexpr uint32_t UB_BIAS_OFFSET = 8 * 1024 + 192 * 1024;
constexpr uint32_t UB_PERTOKEN_SCALE_OFFSET = 210 * 1024;
constexpr uint32_t UB_PERTOKEN_SCALE_CALC_OFFSET = 214 * 1024;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200

template <typename T, typename U>
struct Arch20IsSame {
  static constexpr bool VALUE = false;
};
template <typename T>
struct Arch20IsSame<T, T> {
  static constexpr bool VALUE = true;
};

template <uint32_t SwizzleDir, bool BiasFlag, CubeFormat FormatA = CubeFormat::NZ, CubeFormat FormatB = CubeFormat::NZ,
          CubeFormat FormatY = CubeFormat::NZ, typename IN_DTYPE = int8_t, typename DESCALE_TYPE = float,
          typename BIAS_TYPE = int32_t, typename OUT_TYPE = half>
class QuantBatchMatMulPertokenArch20 {
 public:
  static constexpr bool IS_FP32_BIAS = Arch20IsSame<BIAS_TYPE, float>::VALUE;
  __aicore__ explicit QuantBatchMatMulPertokenArch20() {
    SetPadding<uint64_t>((uint64_t)0x0);
    SetNdpara(1, 0, 0);
    SetAtomicnone();
    SetMasknorm();
  }

  __aicore__ FORCE_INLINE void Init(GM_ADDR A, GM_ADDR B, GM_ADDR bias, GM_ADDR scale, GM_ADDR pertoken_scale,
                                    GM_ADDR C, const QuantMatmulPertokenTilingDataArch20 *tilingData) {
    gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(A));
    gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(B));
    gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_TYPE *>(C));
    gm_bias.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_TYPE *>(bias));
    gm_scale.SetGlobalBuffer(reinterpret_cast<__gm__ DESCALE_TYPE *>(scale));
    gm_pertoken_scale.SetGlobalBuffer(reinterpret_cast<__gm__ DESCALE_TYPE *>(pertoken_scale));
    b_ = tilingData->batchSize;
    m_ = tilingData->m;
    k_ = tilingData->k;
    n_ = tilingData->n;
    m0_ = tilingData->m0;
    k0_ = tilingData->k0;
    n0_ = tilingData->n0;
    m_loop_ = tilingData->mLoop;
    k_loop_ = tilingData->kLoop;
    n_loop_ = tilingData->nLoop;
    core_loop_ = tilingData->coreLoop;
    swizzle_cnt_ = tilingData->swizzleCount;
    BiasWithBatch = tilingData->biasWithBatch;
    l1_a_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
    l1_a_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(L1_PINGPONG_BUFFER_LEN_INT8);
    l1_b_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(L1_PINGPONG_BUFFER_LEN_INT8 * CONST_2);
    l1_b_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(L1_PINGPONG_BUFFER_LEN_INT8 * CONST_3);
    l0_a_ping = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
    l0_a_pong = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(L0AB_PINGPONG_BUFFER_LEN_INT8);
    l0_b_ping = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
    l0_b_pong = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(L0AB_PINGPONG_BUFFER_LEN_INT8);

    ub_scale = buf.template GetBuffer<BufferType::ASCEND_UB, DESCALE_TYPE>(0);
    ub_c = buf.template GetBuffer<BufferType::ASCEND_UB, half>(UB_C_OFFSET);
    ub_c_i32 = buf.template GetBuffer<BufferType::ASCEND_UB, int32_t>(UB_C_TEMP_OFFSET);
    ub_c_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_C_TEMP_OFFSET);
    if constexpr (BiasFlag) {
      ub_bias = buf.template GetBuffer<BufferType::ASCEND_UB, BIAS_TYPE>(UB_BIAS_OFFSET);
    }
    ub_pertoken_scale = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_PERTOKEN_SCALE_OFFSET);
    ub_pertoken_scale_calc = buf.template GetBuffer<BufferType::ASCEND_UB, float>(UB_PERTOKEN_SCALE_CALC_OFFSET);
  }

  __aicore__ FORCE_INLINE void CopyTileA(AscendC::LocalTensor<IN_DTYPE> &dstTensor,
                                         const AscendC::GlobalTensor<IN_DTYPE> &srcTensor, const uint64_t m_actual,
                                         const uint64_t m_round, const uint64_t k_actual, const uint64_t k_round) {
    if (((m_ == 1) || (m_actual == 1)) && FormatA == CubeFormat::ND) {
      gm_to_l1<ArchKind::ASCEND_V220, IN_DTYPE, DataFormat::ND, DataFormat::ND>(
        dstTensor,                        // dst
        srcTensor,                        // src
        1,                                // nTileActual
        RoundUp<BLOCK_SIZE_16>(1),        // nTileCeil
        1,                                // nVal
        k_round,                          // dTileActual
        RoundUp<BLOCK_SIZE_16>(k_round),  // dTileCeil
        k_round);                         // dVal
    } else {
      if constexpr (FormatA == CubeFormat::ND) {
        int dstOffset = 0;
        int srcOffset = 0;
        for (int i = 0; i < k_round / BLOCK_SIZE_32; i++) {
          AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset],
                            AscendC::DataCopyParams(m_round, 1, k_ / BLOCK_SIZE_32 - 1, 0));
          dstOffset += m_round * BLOCK_SIZE_32;
          srcOffset += BLOCK_SIZE_32;
        }
      } else {
        gm_to_l1<ArchKind::ASCEND_V200, IN_DTYPE, DataFormat::NZ, DataFormat::NZ>(
          dstTensor, srcTensor, m_actual, m_round, RoundUp<BLOCK_SIZE_16>(m_), k_actual, k_round,
          RoundUp<BLOCK_SIZE_32>(k_));
      }
    }
  }

  __aicore__ FORCE_INLINE void GetIdx(uint32_t loop_idx, uint64_t &m_idx, uint64_t &n_idx) {
    uint32_t in_batch_idx = loop_idx % (m_loop_ * n_loop_);
    if constexpr (SwizzleDir == 0) {  // Zn
      uint32_t tile_block_loop = (m_loop_ + swizzle_cnt_ - 1) / swizzle_cnt_;
      uint32_t tile_block_idx = in_batch_idx / (swizzle_cnt_ * n_loop_);
      uint32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt_ * n_loop_);

      uint32_t n_row = swizzle_cnt_;
      if (tile_block_idx == tile_block_loop - 1) {
        n_row = m_loop_ - swizzle_cnt_ * tile_block_idx;
      }
      m_idx = tile_block_idx * swizzle_cnt_ + in_tile_block_idx % n_row;
      n_idx = in_tile_block_idx / n_row;
    } else if constexpr (SwizzleDir == 1) {  // Nz
      uint32_t tile_block_loop = (n_loop_ + swizzle_cnt_ - 1) / swizzle_cnt_;
      uint32_t tile_block_idx = in_batch_idx / (swizzle_cnt_ * m_loop_);
      uint32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt_ * m_loop_);

      uint32_t n_col = swizzle_cnt_;
      if (tile_block_idx == tile_block_loop - 1) {
        n_col = n_loop_ - swizzle_cnt_ * tile_block_idx;
      }
      m_idx = in_tile_block_idx / n_col;
      n_idx = tile_block_idx * swizzle_cnt_ + in_tile_block_idx % n_col;
    }
  }

  __aicore__ FORCE_INLINE void CubeMmad(AscendC::LocalTensor<int32_t> l0cTensor,
                                        AscendC::LocalTensor<IN_DTYPE> l0aTensor,
                                        AscendC::LocalTensor<IN_DTYPE> l0bTensor, uint32_t mTileActual,
                                        uint32_t nTileActual, uint32_t kPartActual, bool initC, uint8_t unitFlag = 0) {
    AscendC::Mmad(l0cTensor,                        // C
                  l0aTensor,                        // A
                  l0bTensor,                        // B
                  AscendC::MmadParams(mTileActual,  // m
                                      nTileActual,  // n
                                      kPartActual,  // k
                                      unitFlag,     // unitFlag
                                      false,        // cmatrixSource
                                      initC));      // cmatrixInitVal
  }

  struct LoopCtx {
    uint64_t b_idx = 0;
    uint64_t m_idx = 0;
    uint64_t n_idx = 0;
    uint32_t m_actual = 0;
    uint32_t n_actual = 0;
    uint32_t m_round = 0;
    uint32_t n_round = 0;
    uint32_t k_actual = 0;
    uint32_t k_round = 0;
    uint32_t m_org_up = 0;
    uint32_t n_org_up = 0;
    uint32_t k_org_up = 0;
    uint64_t offset_pertoken_scalar = 0;
    AscendC::LocalTensor<IN_DTYPE> l1_a;
    AscendC::LocalTensor<IN_DTYPE> l1_b;
    event_t l1_a_event = EVENT_ID0;
    event_t l1_b_event = EVENT_ID0;
    uint32_t k_part_len = 0;
    uint32_t last_kloop = 0;
  };

  __aicore__ FORCE_INLINE void LoadFirstTiles(uint32_t loop_idx, uint32_t l1_ping_pong, LoopCtx &ctx) {
    uint64_t &b_idx = ctx.b_idx;
    uint64_t &m_idx = ctx.m_idx;
    uint64_t &n_idx = ctx.n_idx;
    uint32_t &m_actual = ctx.m_actual;
    uint32_t &n_actual = ctx.n_actual;
    uint32_t &m_round = ctx.m_round;
    uint32_t &n_round = ctx.n_round;
    uint32_t &k_actual = ctx.k_actual;
    uint32_t &k_round = ctx.k_round;
    uint32_t &m_org_up = ctx.m_org_up;
    uint32_t &n_org_up = ctx.n_org_up;
    uint32_t &k_org_up = ctx.k_org_up;
    uint64_t &offset_pertoken_scalar = ctx.offset_pertoken_scalar;
    AscendC::LocalTensor<IN_DTYPE> &l1_a = ctx.l1_a;
    AscendC::LocalTensor<IN_DTYPE> &l1_b = ctx.l1_b;
    event_t &l1_a_event = ctx.l1_a_event;
    event_t &l1_b_event = ctx.l1_b_event;
    uint32_t &k_part_len = ctx.k_part_len;
    uint32_t &last_kloop = ctx.last_kloop;
    b_idx = loop_idx / (m_loop_ * n_loop_);
    m_idx = 0;
    n_idx = 0;
    GetIdx(loop_idx, m_idx, n_idx);
    m_actual = (m_idx == (m_loop_ - 1)) ? (m_ - m_idx * m0_) : m0_;
    n_actual = (n_idx == (n_loop_ - 1)) ? (n_ - n_idx * n0_) : n0_;
    m_round = RoundUp<BLOCK_SIZE_16>(m_actual);
    n_round = RoundUp<BLOCK_SIZE_16>(n_actual);
    k_actual = (k_loop_ == 1) ? k_ : k0_;
    k_round = RoundUp<BLOCK_SIZE_32>(k_actual);

    m_org_up = RoundUp<BLOCK_SIZE_16>(m_);
    n_org_up = RoundUp<BLOCK_SIZE_16>(n_);
    k_org_up = RoundUp<BLOCK_SIZE_32>(k_);

    uint64_t batch_m_k_offset = b_idx * m_org_up * k_org_up;
    uint64_t batch_k_n_offset = b_idx * n_org_up * k_org_up;
    uint64_t offset_a = 0;
    if constexpr (FormatA == CubeFormat::ND) {
      offset_a = b_idx * m_ * k_ + m_idx * m0_ * k_;
    } else {
      offset_a = batch_m_k_offset + m_idx * m0_ * BLOCK_SIZE_32;
    }
    uint64_t offset_b = batch_k_n_offset + n_idx * n0_ * BLOCK_SIZE_32;
    offset_pertoken_scalar = m_idx * m0_;

    l1_a = l1_ping_pong ? l1_a_ping : l1_a_pong;
    l1_b = l1_ping_pong ? l1_b_ping : l1_b_pong;

    l1_a_event = l1_ping_pong ? EVENT_ID0 : EVENT_ID1;
    l1_b_event = l1_ping_pong ? EVENT_ID2 : EVENT_ID3;
    WAIT_FLAG(MTE1, MTE2, l1_a_event);
    CopyTileA(l1_a, gm_a[offset_a], m_actual, m_round, k_actual, k_round);
    SET_FLAG(MTE2, MTE1, l1_a_event);

    WAIT_FLAG(MTE1, MTE2, l1_b_event);
    gm_to_l1<ArchKind::ASCEND_V200, IN_DTYPE, DataFormat::NZ, DataFormat::NZ>(l1_b, gm_b[offset_b], n_actual, n_round,
                                                                              n_org_up, k_actual, k_round, k_org_up);
    SET_FLAG(MTE2, MTE1, l1_b_event);

    if constexpr (BiasFlag && !IS_FP32_BIAS) {
      WAIT_FLAG(V, MTE2, EVENT_ID0);
      uint64_t bias_offset = BiasWithBatch ? b_idx * n_ + n_idx * n0_ : n_idx * n0_;
      gm_to_ub<ArchKind::ASCEND_V200, BIAS_TYPE>(ub_bias, gm_bias[bias_offset],
                                                 0,                  // sid
                                                 1,                  // nBurst
                                                 n_round / CONST_8,  // lenBurst
                                                 0,                  // srcStride
                                                 0);                 // dstStride
      SET_FLAG(MTE2, V, EVENT_ID0);
    }

    uint32_t mn_max = m_round > n_round ? m_round : n_round;
    k_part_len = L0AB_PINGPONG_BUFFER_LEN_INT8 / mn_max / BLOCK_SIZE_32 * BLOCK_SIZE_32;

    last_kloop = k_loop_ - 1;
  }

  __aicore__ FORCE_INLINE void EpilogueStore(const LoopCtx &ctx, uint32_t hb, uint32_t passBlocks, uint32_t nPassRound,
                                             uint64_t gmPassOff, bool lastPass) {
    const uint64_t &b_idx = ctx.b_idx;
    const uint64_t &m_idx = ctx.m_idx;
    const uint64_t &n_idx = ctx.n_idx;
    const uint32_t &m_actual = ctx.m_actual;
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &m_org_up = ctx.m_org_up;
    const uint32_t &n_org_up = ctx.n_org_up;
    uint32_t dst_offset = 0;
    uint32_t repeat_count = (m_round * nPassRound / CONST_64) / MAX_REPEAT_TIMES;
    uint32_t repeat_remainder = (m_round * nPassRound / CONST_64) % MAX_REPEAT_TIMES;
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    for (uint32_t i = 0; i < repeat_count; ++i) {
      AscendC::Cast<half, float, false>(ub_c[i * MAX_NUMEL_INST_B32],                         // dst
                                        ub_c_fp32[i * MAX_NUMEL_INST_B32],                    // src
                                        AscendC::RoundMode::CAST_NONE,                        // roundMode
                                        (uint64_t)0,                                          // count(unuse)
                                        MAX_REPEAT_TIMES,                                     // repeatTime
                                        AscendC::UnaryRepeatParams(1, 1, CONST_4, CONST_8));  // repeatParams
    }
    AscendC::Cast<half, float, false>(ub_c[repeat_count * MAX_NUMEL_INST_B32],              // dst
                                      ub_c_fp32[repeat_count * MAX_NUMEL_INST_B32],         // src
                                      AscendC::RoundMode::CAST_NONE,                        // roundMode
                                      (uint64_t)0,                                          // count(unuse)
                                      repeat_remainder,                                     // repeatTime
                                      AscendC::UnaryRepeatParams(1, 1, CONST_4, CONST_8));  // repeatParams

    if (lastPass) {
      // scale/pertoken UB regions are shared across passes:
      // released to the next work item's MTE2 reload only
      // after the last pass consumed them.
      SET_FLAG(V, MTE2, EVENT_ID1);
    }
    if (m_actual == 1) {
      SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
      half zero = 0;
      for (uint32_t i = 0; i < passBlocks; i++) {
        uint64_t curr_offset_c = i * m_round * BLOCK_SIZE_16 + m_actual * BLOCK_SIZE_16;
        AscendC::Muls<half, false>(ub_c[curr_offset_c],                                  // dst
                                   ub_c[curr_offset_c],                                  // src0
                                   zero,                                                 // src1
                                   (uint64_t)0,                                          // count(unuse)
                                   1,                                                    // repeatTime
                                   AscendC::UnaryRepeatParams(1, 1, CONST_2, CONST_2));  // repeatParams
      }
    }
    if (lastPass) {
      // L0C release to the next work item's Mmad: this position
      // is past every pass's fixpipe read (conservative, same
      // release point as the vendor sequence).
      SET_FLAG(V, M, EVENT_ID0);
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);

    if constexpr (FormatY == CubeFormat::ND) {
      dst_offset = b_idx * m_org_up * n_org_up + m_idx * m0_ * n_org_up + n_idx * n0_ + hb * BLOCK_SIZE_16;
      AscendC::DataCopy(gm_c[dst_offset], ub_c, AscendC::Nz2NdParamsFull(1, m_round, nPassRound, 0, m_round, n_, 0));
    } else {
      dst_offset = b_idx * m_org_up * n_org_up + n_idx * n0_ * m_org_up + m_idx * m0_ * BLOCK_SIZE_16;
      ub_to_gm<ArchKind::ASCEND_V200, OUT_TYPE, DataFormat::NZ, DataFormat::NZ>(
        gm_c[dst_offset + gmPassOff], ub_c, m_round, m_round, m_org_up, nPassRound, nPassRound, n_org_up);
    }

    SET_FLAG(MTE3, V, EVENT_ID0);
  }

  __aicore__ FORCE_INLINE void EpilogueScale(const LoopCtx &ctx, uint32_t passBlocks, uint32_t scaleBase) {
    const uint32_t &m_round = ctx.m_round;
    uint32_t count = passBlocks;
    uint32_t m_repeat_count = m_round / MAX_REPEAT_TIMES;
    uint32_t m_repeat_remainder = m_round % MAX_REPEAT_TIMES;
    SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
    for (uint32_t i = 0; i < count; ++i) {
      if (m_repeat_count > 0) {  // m_round == 256 is split in two halves (UB caps at 128K elements)
        AscendC::Mul<float, false>(ub_c_fp32[i * m_round * BLOCK_SIZE_16],   // dst
                                   ub_c_fp32[i * m_round * BLOCK_SIZE_16],   // src0
                                   ub_scale[scaleBase + i * BLOCK_SIZE_16],  // src1
                                   (uint64_t)0,                              // count(unuse)
                                   MAX_REPEAT_TIMES,                         // repeatTime
                                   AscendC::BinaryRepeatParams(1,            // dstBlockStride
                                                               1,            // src0BlockStride
                                                               1,            // src1BlockStride
                                                               CONST_2,      // dstRepeatStride
                                                               CONST_2,      // src0RepeatStride
                                                               0));          // src1RepeatStride
      }

      // when m_repeat_count is 0 this single Mul covers all m_round repeats
      AscendC::Mul<float, false>(
        ub_c_fp32[i * m_round * BLOCK_SIZE_16 + m_repeat_count * MAX_REPEAT_TIMES * BLOCK_SIZE_16],  // dst
        ub_c_fp32[i * m_round * BLOCK_SIZE_16 + m_repeat_count * MAX_REPEAT_TIMES * BLOCK_SIZE_16],  // src0
        ub_scale[scaleBase + i * BLOCK_SIZE_16],                                                     // src1
        (uint64_t)0,                                                                                 // count(unuse)
        m_repeat_remainder,                                                                          // repeatTime
        AscendC::BinaryRepeatParams(1,                                                               // dstBlockStride
                                    1,                                                               // src0BlockStride
                                    1,                                                               // src1BlockStride
                                    CONST_2,                                                         // dstRepeatStride
                                    CONST_2,                                                         // src0RepeatStride
                                    0));                                                             // src1RepeatStride
    }
    AscendC::PipeBarrier<PIPE_V>();

    // dequant x2 scale perToken
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    for (uint32_t i = 0; i < count; ++i) {
      AscendC::Mul<float, false>(ub_c_fp32[i * m_round * BLOCK_SIZE_16],  // dst
                                 ub_c_fp32[i * m_round * BLOCK_SIZE_16],  // src0
                                 ub_pertoken_scale_calc,                  // src1
                                 (uint64_t)0,                             // count(unuse)
                                 m_round / CONST_4,                       // repeatTime
                                 AscendC::BinaryRepeatParams(1,           // dstBlockStride
                                                             1,           // src0BlockStride
                                                             1,           // src1BlockStride
                                                             CONST_8,     // dstRepeatStride
                                                             CONST_8,     // src0RepeatStride
                                                             CONST_8));   // src1RepeatStride
    }

    AscendC::PipeBarrier<PIPE_V>();

    // add fp32 bias after dequant, reuse ub_bias; same per-N broadcast as x2Scale Mul
    if constexpr (BiasFlag && IS_FP32_BIAS) {
      SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
      for (uint32_t i = 0; i < count; ++i) {
        if (m_repeat_count > 0) {
          AscendC::Add<float, false>(ub_c_fp32[i * m_round * BLOCK_SIZE_16],  // dst
                                     ub_c_fp32[i * m_round * BLOCK_SIZE_16],  // src0
                                     ub_bias[scaleBase + i * BLOCK_SIZE_16],  // src1
                                     (uint64_t)0,                             // count(unuse)
                                     MAX_REPEAT_TIMES,                        // repeatTime
                                     AscendC::BinaryRepeatParams(1,           // dstBlockStride
                                                                 1,           // src0BlockStride
                                                                 1,           // src1BlockStride
                                                                 CONST_2,     // dstRepeatStride
                                                                 CONST_2,     // src0RepeatStride
                                                                 0));         // src1RepeatStride
        }
        AscendC::Add<float, false>(
          ub_c_fp32[i * m_round * BLOCK_SIZE_16 + m_repeat_count * MAX_REPEAT_TIMES * BLOCK_SIZE_16],
          ub_c_fp32[i * m_round * BLOCK_SIZE_16 + m_repeat_count * MAX_REPEAT_TIMES * BLOCK_SIZE_16],
          ub_bias[scaleBase + i * BLOCK_SIZE_16],
          (uint64_t)0,                          // count(unuse)
          m_repeat_remainder,                   // repeatTime
          AscendC::BinaryRepeatParams(1,        // dstBlockStride
                                      1,        // src0BlockStride
                                      1,        // src1BlockStride
                                      CONST_2,  // dstRepeatStride
                                      CONST_2,  // src0RepeatStride
                                      0));      // src1RepeatStride
      }
      AscendC::PipeBarrier<PIPE_V>();
    }

    // cast fp32->fp16
  }

  __aicore__ FORCE_INLINE void EpiloguePass(const LoopCtx &ctx, uint32_t hb, uint32_t passBlocks, uint32_t nPassRound,
                                            uint32_t l0cPassOff, uint64_t gmPassOff, uint32_t scaleBase,
                                            bool lastPass) {
    const uint64_t &b_idx = ctx.b_idx;
    const uint64_t &m_idx = ctx.m_idx;
    const uint64_t &n_idx = ctx.n_idx;
    const uint32_t &m_actual = ctx.m_actual;
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &m_org_up = ctx.m_org_up;
    const uint32_t &n_org_up = ctx.n_org_up;
    uint32_t dst_offset = 0;

    WAIT_FLAG(MTE3, V, EVENT_ID0);
    // copy mmOut l0C->UB (this pass's column strips)
    l0c_to_ub<ArchKind::ASCEND_V200, int32_t, int32_t>(ub_c_i32, l0c.ReinterpretCast<int32_t>()[l0cPassOff],
                                                       (uint16_t)passBlocks,                 // nBurst
                                                       (uint16_t)(m_round / BLOCK_SIZE_16),  // lenBurst
                                                       (uint16_t)0,                          // srcStride
                                                       (uint16_t)0);                         // dstStride
    AscendC::PipeBarrier<PIPE_V>();

    // cast int32->fp32
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    uint32_t repeat_count = (m_round * nPassRound / CONST_64) / MAX_REPEAT_TIMES;
    uint32_t repeat_remainder = (m_round * nPassRound / CONST_64) % MAX_REPEAT_TIMES;
    for (uint32_t i = 0; i < repeat_count; ++i) {
      AscendC::Cast<float, int32_t, false>(ub_c_fp32[i * MAX_NUMEL_INST_B32],                    // dst
                                           ub_c_i32[i * MAX_NUMEL_INST_B32],                     // src
                                           AscendC::RoundMode::CAST_NONE,                        // roundMode
                                           (uint64_t)0,                                          // count(unuse)
                                           MAX_REPEAT_TIMES,                                     // repeatTime
                                           AscendC::UnaryRepeatParams(1, 1, CONST_8, CONST_8));  // repeatParams
    }
    AscendC::Cast<float, int32_t, false>(ub_c_fp32[repeat_count * MAX_NUMEL_INST_B32],         // dst
                                         ub_c_i32[repeat_count * MAX_NUMEL_INST_B32],          // src
                                         AscendC::RoundMode::CAST_NONE,                        // roundMode
                                         (uint64_t)0,                                          // count(unuse)
                                         repeat_remainder,                                     // repeatTime
                                         AscendC::UnaryRepeatParams(1, 1, CONST_8, CONST_8));  // repeatParams

    AscendC::PipeBarrier<PIPE_V>();

    // dequant x1 scale perChannel (ub_scale indexed by GLOBAL
    // n-block: this pass covers [scaleBase, scaleBase+nPassRound))
    EpilogueScale(ctx, passBlocks, scaleBase);
    EpilogueStore(ctx, hb, passBlocks, nPassRound, gmPassOff, lastPass);
  }

  __aicore__ FORCE_INLINE void RunEpiloguePasses(const LoopCtx &ctx) {
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &n_round = ctx.n_round;
    const uint32_t &m_org_up = ctx.m_org_up;
    uint32_t totalBlocks = n_round / BLOCK_SIZE_16;
    uint32_t blocksPerPass = (UB_BIAS_OFFSET - UB_C_TEMP_OFFSET) / (m_round * CONST_4 * BLOCK_SIZE_16);
    if (blocksPerPass == 0) {  // paranoia: m_round is host-capped at 256 -> >= 8
      blocksPerPass = 1;
    }
    for (uint32_t hb = 0; hb < totalBlocks; hb += blocksPerPass) {
      const uint32_t passBlocks = hb + blocksPerPass < totalBlocks ? blocksPerPass : totalBlocks - hb;
      const uint32_t nPassRound = passBlocks * BLOCK_SIZE_16;
      const uint32_t l0cPassOff = hb * m_round * BLOCK_SIZE_16;  // int32 elems, column-strip major
      const uint64_t gmPassOff = static_cast<uint64_t>(hb) * BLOCK_SIZE_16 * m_org_up;  // fp16 elems
      const uint32_t scaleBase = hb * BLOCK_SIZE_16;                                    // this pass's global n offset
      const bool lastPass = hb + blocksPerPass >= totalBlocks;
      EpiloguePass(ctx, hb, passBlocks, nPassRound, l0cPassOff, gmPassOff, scaleBase, lastPass);
    }
  }
  __aicore__ FORCE_INLINE void PrepareDequant(const LoopCtx &ctx) {
    const uint64_t &b_idx = ctx.b_idx;
    const uint64_t &n_idx = ctx.n_idx;
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &n_round = ctx.n_round;
    const uint64_t &offset_pertoken_scalar = ctx.offset_pertoken_scalar;
    const uint32_t &m_org_up = ctx.m_org_up;
    AscendC::PipeBarrier<PIPE_MTE2>();
    WAIT_FLAG(V, MTE2, EVENT_ID1);
    // copy x1Scale
    gm_to_ub<ArchKind::ASCEND_V200, DESCALE_TYPE>(ub_scale, gm_scale[n_idx * n0_],
                                                  0,                              // sid
                                                  1,                              // nBurst
                                                  n_round / UB_SCALE_BLOCK_SIZE,  // lenBurst
                                                  0,                              // srcStride
                                                  0);                             // dstStride
    // copy x2Scale
    gm_to_ub<ArchKind::ASCEND_V220, float>(ub_pertoken_scale, gm_pertoken_scale[offset_pertoken_scalar],
                                           0,                              // sid
                                           1,                              // nBurst
                                           m_round / UB_SCALE_BLOCK_SIZE,  // lenBurst
                                           0,                              // srcStride
                                           0);                             // dstStride
    if constexpr (BiasFlag && IS_FP32_BIAS) {
      uint64_t bias_offset = BiasWithBatch ? b_idx * n_ + n_idx * n0_ : n_idx * n0_;
      gm_to_ub<ArchKind::ASCEND_V200, BIAS_TYPE>(ub_bias, gm_bias[bias_offset],
                                                 0,                  // sid
                                                 1,                  // nBurst
                                                 n_round / CONST_8,  // lenBurst
                                                 0,                  // srcStride
                                                 0);                 // dstStride
    }
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    // broc x2Scale
    Brcb(ub_pertoken_scale_calc, ub_pertoken_scale, CeilDiv<CONST_8>(m_round), {CONST_2, CONST_16});
    AscendC::DataCopy(ub_pertoken_scale_calc[CONST_8], ub_pertoken_scale_calc,
                      AscendC::DataCopyParams(m_round, 1, 1, 1));

    SET_FLAG(M, V, EVENT_ID0);
    WAIT_FLAG(M, V, EVENT_ID0);
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);

    // WIDE-TILE COLUMN-PASS EPILOGUE (vendor fork
    // of this kernel): the int32/fp32 work region
    // [UB_C_TEMP_OFFSET, UB_BIAS_OFFSET) = 128KB and the fp16 out
    // region [UB_C_OFFSET, UB_C_TEMP_OFFSET) = 64KB both cap one
    // epilogue pass at m_round*n_round <= 32768 elements.
    // Search-derived tilings (host kUbLimitPertoken parity,
    // m0*n0*4 <= 128KB) always fit a single pass and emit the vendor
    // op sequence verbatim; wide tiles (m0*n0*4 <= l0cSize, host
    // env/override channel only -- the (256,256) AI=128 lever) drain
    // in column-block passes sized to the budget. L0C is
    // column-block-major ([n/16][m_round*16] int32 strips; addressing
    // proven by the bias broadcast path and by S2/S4 (128,256)
    // bit-identity), so pass p reads strips [hb, hb+passBlocks) at
    // l0c[hb*m_round*16] and stores GM column strips at
    // +hb*16*m_org_up (NZ GM layout [n/16][m_org_up*16]). All math
    // is element-wise on identical int32 accumulations -> bit-identity
    // holds by construction.
  }

  __aicore__ FORCE_INLINE void MmadAccum(const LoopCtx &ctx, uint32_t k_idx, uint32_t k_part_idx,
                                         uint32_t m_mmad_actual, uint32_t n_actual, uint32_t k0_actual,
                                         AscendC::LocalTensor<IN_DTYPE> l0_a, AscendC::LocalTensor<IN_DTYPE> l0_b) {
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &n_round = ctx.n_round;
    bool init_c = (k_idx == 0 && k_part_idx == 0);

    if (init_c) {
      // L0C release gate hoisted BEFORE the first (tile-overwriting)
      // Mmad. The vendor sequence consumed the previous work item's
      // epilogue release (SET_FLAG(V, M, EVENT_ID0)) only AFTER this
      // first Mmad, leaving the overwrite gated by nothing but the
      // L1/L0 reload-chain timing (~4.5us) racing the epilogue V
      // chain (~2.7us) -- a margin race the single-pass vendor wins
      // in steady state. The column-pass epilogue (wide tiles)
      // doubles the V chain and loses it (L0C read/write conflict: vec
      // read vs cube write on the same address). Consuming the same flag one
      // Mmad earlier is free in steady state (the flag is pre-set by
      // the time the M queue arrives -- identical margin to the
      // vendor's post-Mmad gate) and makes the hazard explicit for
      // every tile size.
      WAIT_FLAG(V, M, EVENT_ID0);
      if constexpr (BiasFlag && !IS_FP32_BIAS) {
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        for (uint32_t i = 0; i < n_round / BLOCK_SIZE_16; i++) {
          for (uint32_t j = 0; j < m_round / BLOCK_SIZE_16; j++) {
            AscendC::BroadCastVecToMM(l0c[i * m_round * BLOCK_SIZE_16 + j * BLOCK_SIZE_16 * BLOCK_SIZE_16],
                                      ub_bias[i * BLOCK_SIZE_16], 1, 1, 0, 0);
          }
        }
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, M, EVENT_ID1);
        WAIT_FLAG(V, M, EVENT_ID1);
        CubeMmad(l0c, l0_a, l0_b, m_mmad_actual, n_actual, k0_actual, false);
      } else {
        CubeMmad(l0c, l0_a, l0_b, m_mmad_actual, n_actual, k0_actual, init_c);
      }
    } else {
      CubeMmad(l0c, l0_a, l0_b, m_mmad_actual, n_actual, k0_actual, init_c);
    }
  }

  __aicore__ FORCE_INLINE void MmadKPart(const LoopCtx &ctx, uint32_t k_idx, uint32_t l1_ping_pong) {
    const uint32_t &m_actual = ctx.m_actual;
    const uint32_t &n_actual = ctx.n_actual;
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &n_round = ctx.n_round;
    const uint32_t &k_part_len = ctx.k_part_len;
    const uint32_t &last_kloop = ctx.last_kloop;
    uint32_t k_actual = (k_idx == k_loop_ - 1) ? k_ - k_idx * k0_ : k0_;
    uint32_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);
    AscendC::LocalTensor<IN_DTYPE> l1_a = l1_ping_pong ? l1_a_ping : l1_a_pong;
    AscendC::LocalTensor<IN_DTYPE> l1_b = l1_ping_pong ? l1_b_ping : l1_b_pong;
    event_t l1_a_event = l1_ping_pong ? EVENT_ID0 : EVENT_ID1;
    event_t l1_b_event = l1_ping_pong ? EVENT_ID2 : EVENT_ID3;
    uint32_t k_part_loop = CeilDiv<uint32_t>(k_actual, k_part_len);
    for (uint32_t k_part_idx = 0; k_part_idx < k_part_loop; ++k_part_idx) {
      uint32_t k0_round = k_part_idx < k_part_loop - 1 ? k_part_len : k_round - k_part_idx * k_part_len;
      uint32_t k0_actual = k_part_idx < k_part_loop - 1 ? k_part_len : k_actual - k_part_idx * k_part_len;
      uint32_t l0_ping_pong = 1 - k_part_idx % CONST_2;
      event_t l0_event = l0_ping_pong ? EVENT_ID0 : EVENT_ID1;
      AscendC::LocalTensor<IN_DTYPE> l0_a = l0_ping_pong ? l0_a_ping : l0_a_pong;
      AscendC::LocalTensor<IN_DTYPE> l0_b = l0_ping_pong ? l0_b_ping : l0_b_pong;
      if (k_part_idx == 0) {
        WAIT_FLAG(MTE2, MTE1, l1_a_event);
      }
      WAIT_FLAG(M, MTE1, l0_event);
      l1_to_l0_a<ArchKind::ASCEND_V200, IN_DTYPE, false, DataFormat::ZN, DataFormat::ZZ>(
        l0_a, l1_a[k_part_idx * k_part_len * m_round], m_round, k0_round, 1, m_round / BLOCK_SIZE_16,
        k0_round / BLOCK_SIZE_32, 1);
      if (k_part_idx == k_part_loop - 1) {
        SET_FLAG(MTE1, MTE2, l1_a_event);
      }
      if (k_part_idx == 0) {
        WAIT_FLAG(MTE2, MTE1, l1_b_event);
      }

      l1_to_l0_b<ArchKind::ASCEND_V200, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
        l0_b,                                     // dst
        l1_b[k_part_idx * k_part_len * n_round],  // src
        0,
        k0_round * n_round / CONST_512,  // repeat
        0,
        1,  // srcStride
        0,
        0);  // dstStride
      if (k_part_idx == k_part_loop - 1) {
        SET_FLAG(MTE1, MTE2, l1_b_event);
      }
      uint32_t m_mmad_actual = (m_actual == 1) ? CONST_2 : m_actual;
      SET_FLAG(MTE1, M, l0_event);
      WAIT_FLAG(MTE1, M, l0_event);

      MmadAccum(ctx, k_idx, k_part_idx, m_mmad_actual, n_actual, k0_actual, l0_a, l0_b);
      AscendC::PipeBarrier<PIPE_M>();
      SET_FLAG(M, MTE1, l0_event);
    }
  }

  __aicore__ FORCE_INLINE void PrefetchNextTiles(const LoopCtx &ctx, uint32_t k_idx, uint32_t l1_ping_pong) {
    const uint64_t &b_idx = ctx.b_idx;
    const uint64_t &m_idx = ctx.m_idx;
    const uint64_t &n_idx = ctx.n_idx;
    const uint32_t &m_actual = ctx.m_actual;
    const uint32_t &n_actual = ctx.n_actual;
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &n_round = ctx.n_round;
    const uint32_t &m_org_up = ctx.m_org_up;
    const uint32_t &n_org_up = ctx.n_org_up;
    const uint32_t &k_org_up = ctx.k_org_up;
    const uint32_t &last_kloop = ctx.last_kloop;
    uint64_t offset_a_next = 0;
    uint64_t offset_b_next = 0;
    if (k_idx < last_kloop) {
      if constexpr (FormatA == CubeFormat::ND) {
        offset_a_next = b_idx * m_ * k_ + m_idx * m0_ * k_ + (k_idx + 1) * k0_;
      } else {
        offset_a_next = b_idx * m_org_up * k_org_up + (k_idx + 1) * k0_ * m_org_up + m_idx * m0_ * BLOCK_SIZE_32;
      }

      offset_b_next = b_idx * n_org_up * k_org_up + (k_idx + 1) * k0_ * n_org_up + n_idx * n0_ * BLOCK_SIZE_32;
      AscendC::LocalTensor<IN_DTYPE> l1_buf_a_next = (1 - l1_ping_pong) ? l1_a_ping : l1_a_pong;
      AscendC::LocalTensor<IN_DTYPE> l1_buf_b_next = (1 - l1_ping_pong) ? l1_b_ping : l1_b_pong;
      event_t l1_a_event_next = (1 - l1_ping_pong) ? EVENT_ID0 : EVENT_ID1;
      event_t l1_b_event_next = (1 - l1_ping_pong) ? EVENT_ID2 : EVENT_ID3;

      uint32_t k_actual_next = ((k_idx + 1) == last_kloop) ? (k_ - (k_idx + 1) * k0_) : k0_;
      uint32_t k_round_next = RoundUp<BLOCK_SIZE_32>(k_actual_next);

      WAIT_FLAG(MTE1, MTE2, l1_a_event_next);
      CopyTileA(l1_buf_a_next, gm_a[offset_a_next], m_actual, m_round, k_actual_next, k_round_next);
      SET_FLAG(MTE2, MTE1, l1_a_event_next);

      WAIT_FLAG(MTE1, MTE2, l1_b_event_next);
      gm_to_l1<ArchKind::ASCEND_V200, IN_DTYPE, DataFormat::NZ, DataFormat::NZ>(
        l1_buf_b_next, gm_b[offset_b_next], n_actual, n_round, n_org_up, k_actual_next, k_round_next, k_org_up);
      SET_FLAG(MTE2, MTE1, l1_b_event_next);
    }
  }

  __aicore__ FORCE_INLINE void RunGemm(const LoopCtx &ctx, uint32_t &l1_ping_pong) {
    const uint32_t &last_kloop = ctx.last_kloop;
    const uint32_t &m_actual = ctx.m_actual;
    const uint32_t &n_actual = ctx.n_actual;
    const uint32_t &m_round = ctx.m_round;
    const uint32_t &n_round = ctx.n_round;
    const uint32_t &k_part_len = ctx.k_part_len;
    const uint64_t &b_idx = ctx.b_idx;
    const uint64_t &m_idx = ctx.m_idx;
    const uint64_t &n_idx = ctx.n_idx;
    const uint32_t &k_org_up = ctx.k_org_up;
    const uint32_t &m_org_up = ctx.m_org_up;
    const uint32_t &n_org_up = ctx.n_org_up;
    for (uint32_t k_idx = 0; k_idx < k_loop_; ++k_idx) {
      uint32_t k_actual = (k_idx == k_loop_ - 1) ? k_ - k_idx * k0_ : k0_;
      uint32_t k_round = RoundUp<BLOCK_SIZE_32>(k_actual);

      AscendC::LocalTensor<IN_DTYPE> l1_a = l1_ping_pong ? l1_a_ping : l1_a_pong;
      AscendC::LocalTensor<IN_DTYPE> l1_b = l1_ping_pong ? l1_b_ping : l1_b_pong;

      event_t l1_a_event = l1_ping_pong ? EVENT_ID0 : EVENT_ID1;
      event_t l1_b_event = l1_ping_pong ? EVENT_ID2 : EVENT_ID3;

      PrefetchNextTiles(ctx, k_idx, l1_ping_pong);

      MmadKPart(ctx, k_idx, l1_ping_pong);
      l1_ping_pong = 1 - l1_ping_pong;
    }
  }

  __aicore__ FORCE_INLINE void Process() {
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(V, M, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);
    SET_FLAG(V, MTE2, EVENT_ID0);
    uint32_t l1_ping_pong = 1;
    for (uint32_t loop_idx = block_idx; loop_idx < core_loop_; loop_idx += block_num) {
      LoopCtx ctx;
      LoadFirstTiles(loop_idx, l1_ping_pong, ctx);
      uint64_t offset_a_next = 0;
      uint64_t offset_b_next = 0;
      uint32_t k_actual_next = 0;
      uint32_t k_round_next = 0;

      RunGemm(ctx, l1_ping_pong);
      PrepareDequant(ctx);
      RunEpiloguePasses(ctx);
    }
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(V, M, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

 public:
  OnChipBuffer<ArchKind::ASCEND_V200> buf;

 private:
  AscendC::GlobalTensor<IN_DTYPE> gm_a;
  AscendC::GlobalTensor<IN_DTYPE> gm_b;
  AscendC::GlobalTensor<OUT_TYPE> gm_c;
  AscendC::GlobalTensor<BIAS_TYPE> gm_bias;
  AscendC::GlobalTensor<DESCALE_TYPE> gm_scale;
  AscendC::GlobalTensor<DESCALE_TYPE> gm_pertoken_scale;

  AscendC::LocalTensor<IN_DTYPE> l1_a_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
  AscendC::LocalTensor<IN_DTYPE> l1_a_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
  AscendC::LocalTensor<IN_DTYPE> l1_b_ping = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
  AscendC::LocalTensor<IN_DTYPE> l1_b_pong = buf.template GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(0);
  AscendC::LocalTensor<IN_DTYPE> l0_a_ping = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
  AscendC::LocalTensor<IN_DTYPE> l0_a_pong = buf.template GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
  AscendC::LocalTensor<IN_DTYPE> l0_b_ping = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
  AscendC::LocalTensor<IN_DTYPE> l0_b_pong = buf.template GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
  AscendC::LocalTensor<int32_t> l0c = buf.template GetBuffer<BufferType::ASCEND_L0C, int32_t>(0);
  AscendC::LocalTensor<half> ub_c = buf.template GetBuffer<BufferType::ASCEND_UB, half>(0);
  AscendC::LocalTensor<float> ub_c_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(0);
  AscendC::LocalTensor<int32_t> ub_c_i32 = buf.template GetBuffer<BufferType::ASCEND_UB, int32_t>(0);
  AscendC::LocalTensor<DESCALE_TYPE> ub_scale = buf.template GetBuffer<BufferType::ASCEND_UB, DESCALE_TYPE>(0);
  AscendC::LocalTensor<BIAS_TYPE> ub_bias = buf.template GetBuffer<BufferType::ASCEND_UB, BIAS_TYPE>(0);
  AscendC::LocalTensor<float> ub_pertoken_scale = buf.template GetBuffer<BufferType::ASCEND_UB, float>(0);
  AscendC::LocalTensor<float> ub_pertoken_scale_calc = buf.template GetBuffer<BufferType::ASCEND_UB, float>(0);

  uint32_t b_{0};
  uint32_t m_{0};
  uint32_t k_{0};
  uint32_t n_{0};
  uint32_t m0_{0};
  uint32_t k0_{0};
  uint32_t n0_{0};
  uint32_t m_loop_{0};
  uint32_t k_loop_{0};
  uint32_t n_loop_{0};
  uint32_t core_loop_{0};
  uint32_t swizzle_cnt_{0};
  bool BiasWithBatch{false};
};

#endif

}  // namespace PpMatMulNS

#endif
