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

#ifndef FUSED_ATTN_OUT_TRANSPOSE_QUANT_H
#define FUSED_ATTN_OUT_TRANSPOSE_QUANT_H

#include "kernel_operator.h"                             // NOLINT(build/include_subdir)
#include "fused_attn_out_transpose_quant_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

namespace FusedAttnOutTransposeQuant {

// ---------------------------------------------------------------------------
// FusedAttnOutTransposeQuant: FA-output head-major->token-major fold + smooth
// + per-token dynamic quant for Atlas 300I Duo (dav_m200) W8A8 LLM prefill
// graphs. Pure-vector kernel (AICore task, vector pipe only, no Cube).
//
// Local compute core of report section 5.4 op K1
// (FusedAttnOutExchangeTransposeQuant): the
// multi-rank AG exchange stays outside (single-device contract); this kernel
// fuses the local tail the exchange wraps. At suffix=1536 the unfused local
// chain costs ~590us/layer (transpose#1 126 + transpose#2 236 + slice 102 +
// mul-smooth 18 + DynamicQuant 69 + TransData 39) and the whole 993us K1
// site additionally carries the AG (~403us) which remains unchanged.
//
// Math (numerics aligned op-for-op with the chain, quant tail identical to
// FusedAddRmsNormSmoothQuant / the 310P DynamicQuant contract):
//   a[m, h*D+d] = fp16(fp32(attn[h, m, d]) * fp32(smooth[h*D+d]))
//                 (without smooth: a = attn bit-identical, no op at all)
//   absmax[m]   = max_j |a[m, j]|        (fp16 abs + native vcgmax/vcmax
//                 reduce, v6)
//   token_scale[m] = absmax[m] / 127
//   y[m, j]     = int8_round(a[m, j] * (127 / absmax[m]))   (fp16 Muls +
//                 CAST_ROUND; absmax <= 1e-12 quantizes the whole row to 0)
//
// Layout notes for dav_m200:
//   - The fold [H, M, D] -> [M, H*D] is ONE 2D DataCopyParams MTE2 gather
//     per token row (tiling sets fold2d=1 when m <= 8192): H blocks of
//     D elements each, blockLen 8 / srcGap (m-1)*8 / dstGap 0 in 32B units
//     (DataCopyParams blockLen/strides are 32B-unit gaps, not elements --
//     same semantics as fused_rms_norm_nz). At m > 8192 the uint16 stride
//     field overflows and the kernel falls back to H plain count-form
//     copies per row (verified-safe form). The 2D form was v1's original
//     load and its "failure" was later traced to the ApplySmooth OOB
//     broadcast bug (see there); msprof reinstated it as a perf experiment
//     once precision was green -- @1536x8 the scalar pipe ran 0.86 busy
//     (descriptor-issue-bound) with the fold load as R*H of the issues,
//     so collapsing H descriptors into one is the single biggest win.
//   - D must be 128: 256B slice runs keep every copy and row base 32B
//     aligned; other head dims would need the FusedQKNormRope
//     interleave/deinterleave machinery (future work).
//   - DataCopyPad is unavailable; D=128 and n = H*128 keep every copy, row
//     base and table 32B aligned.
//   - No cross-core reduce: rows are disjoint.
//
// Execution model (op-B v3 issue-count discipline):
//   8-row groups striped over cores (token_scale stays whole 32B blocks);
//   within a core, rowBlock R (<= 16, a multiple of 8 or a divisor of 8 so
//   block boundaries never straddle a group) rows per block, ~4 event
//   round-trips per block (MTE2_V in, V_MTE2 handback, V_S absmax extract,
//   V_MTE3/MTE3_V out). Bigger blocks amortize the round-trips -- the v1/v2
//   kernels' small tiles (R=8 at n=1024, R=2-4 at n=4096) left the op
//   sync-bound at 0.77-0.95x the unfused chain. The next block's fold rows
//   are prefetched right after the handback so the loads drain under the
//   scalar/quant/MTE3 tail.
//
//   v5: the V pipe executes instructions in order (established by
//   FusedQKNormRoPEBSH's rope chain and validated at scale by op B v3:
//   same-pipe dependent ops need no PipeBarrier between them), so a block's
//   whole V section is ONE in-order stream with a single barrier at the
//   V->scalar readback -- v4 kept ~14 per block (one after every tree Max,
//   one per Mul/Cast batch), which msprof showed as scalar-pipe issue
//   overhead. Issue count is also cut where batching is free: the smooth
//   table is loaded R-times-tiled at Init so ApplySmooth is ONE whole-tile
//   Mul (v4's per-row Muls were R issues -- vector Mul has no broadcast, a
//   single rows*n call against an n-table reads OOB, which is why the table
//   must be tiled), and the quant Casts collapse to ONE whole-tile Cast
//   (the per-row Muls stay: their fp16 quantMul scalar differs per row).
//
//   v6: the halving-Max absmax tree is replaced by the native reduce
//   instructions. msprof at 1536x32 showed vec at 0.9 (near-saturated) with
//   ~72% of the vector work in the tree -- the count-form fold
//   Max(tmp, tmp, tmp[h], rows*n - h) processes ~the whole tile every fold
//   while only rows*h live lanes remain (fold k is 2^-k useful), so the
//   12-fold tree at n=4096 does ~11 dead passes. dav_m200 has native
//   vcgmax/vcmax (Ascend310P3.ini Intrinsic_vcmax/Intrinsic_vcgmax) and
//   CANN's quant_matmul_dequant kernel (production on this exact arch:
//   Ascend310P3.ini CCEC_AIC_version=dav-m200) reduces per-token absmax
//   as: Abs -> BlockReduceMax (vcgmax: each 16-element 32B block -> 1 max
//   packed at the source prefix; counter-mode mask spans the whole chunk
//   with repeatTime=1) -> one more vcgmax pass while > 128 maxima remain
//   -> WholeReduceMax (vcmax: norm-mode bit mask over the final <= 128
//   maxima, repeatTime=1, ORDER_VALUE_INDEX -> dst[0]=value, dst[1]=index).
//   The reduce here is PER ROW with exactly those shapes and stride tuples
//   (dstRepStride=1, srcBlkStride=1, srcRepStride=8=one repeat width;
//   HALF_PERBLOCK=16): the in-place vcgmax stages write each row's maxima
//   on the row's own prefix so rows never share a 16-slot group, and
//   per-row counters stay at n (<= 4096, inside the production envelope --
//   a tile-wide counter of rows*n=32768 hung the bench in a first attempt,
//   and ORDER_ONLY_VALUE is arch-2002-illegal: the int32-mask vcmax checks
//   order in [0,1], so the readback takes dst[0] of [value,index] pairs).
//   absmax values are exact fp16 maxima either way -- numerics identical
//   to the tree. The stage-2 shape needs 16-multiple maxima counts
//   (n <= 2048 or n % 256 == 0) and the vcmax mask caps n at 32768; the
//   tiling rejects the rest (n = h * 128 -> h <= 16 or h even; 1/2/8/32
//   all pass).
//
// absmax staging (v6) runs in tmpH_ (the quant fp16 scratch -- in-order
// reuse: the V_S readback consumes the reduce results before the quant
// Muls overwrite the tile). Mask-state discipline (from CANN's kernel): a
// pending V instruction reads the mask registers at execution time, so a
// PipeBarrier must sit between the last old-mask V op and the first
// new-mask V op; the count-form Abs/Muls/Cast calls around the reduce set
// their own counter masks per call, and the trailing SetMaskCount restores
// counter mode for them (the caller's barrier separates it from the vcmax,
// CANN's post-vcmax order).
// ---------------------------------------------------------------------------
constexpr uint32_t kScaleGroupRows = 8;   // fp32 x 8 = one 32B GM block
constexpr uint32_t kMaxRowBlock = 16;     // R upper bound (8 or 16 = whole groups)
constexpr uint32_t kHalfPerBlock = 16;    // fp16 elements in one 32B block (HALF_PERBLOCK)
constexpr uint32_t kHalfPerRepeat = 128;  // fp16 elements one vcmax repeat spans (HALF_PERREPEAT)
constexpr uint32_t kVcgSrcRepStride = 8;  // production vcgmax tuple: 8 x 32B = one repeat width
constexpr uint32_t kVcmaxPairStride = 2;  // vcmax packs [value, index]; value = lane 0
constexpr float kQuantMaxValue = 127.0f;
constexpr float kQuantEpsilon = 1e-12f;  // DynamicQuant 310P epsilon

class FusedAttnOutTransposeQuant {
 public:
  __aicore__ inline void Init(GM_ADDR attn, GM_ADDR smooth, GM_ADDR y, GM_ADDR tokenScale,
                              const FusedAttnOutTransposeQuantTilingData *__restrict tiling, TPipe *pipe) {
    blockIdx_ = GetBlockIdx();
    coreNum_ = tiling->coreNum;
    m_ = tiling->m;
    h_ = tiling->h;
    d_ = tiling->d;
    n_ = tiling->n;
    rowBlock_ = tiling->rowBlock;
    hasSmooth_ = tiling->hasSmooth;
    fold2d_ = tiling->fold2d;
    foldLen32_ = d_ * sizeof(half) / 32;  // 256B head slices = 8 x 32B blocks
    // 8-row-group striping: group boundaries are integers, so every core's
    // row range starts at a multiple of 8 and no group is shared. R is a
    // multiple of 8 (16/8: whole groups, flushed as 1-2 merged copies) or a
    // divisor of 8 (4/2/1: sub-group blocks never straddle a boundary).
    uint32_t groups = (m_ + kScaleGroupRows - 1) / kScaleGroupRows;
    uint32_t gStart = blockIdx_ * groups / coreNum_;
    uint32_t gEnd = (blockIdx_ + 1) * groups / coreNum_;
    rowStart_ = gStart * kScaleGroupRows;
    uint32_t rowEnd = gEnd * kScaleGroupRows;
    if (rowEnd > m_) {
      rowEnd = m_;
    }
    myRows_ = rowEnd - rowStart_;
    groupRowBase_ = rowStart_;

    attnGm_.SetGlobalBuffer((__gm__ half *)attn);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
    scaleGm_.SetGlobalBuffer((__gm__ float *)tokenScale);
    if (hasSmooth_ != 0) {
      smoothGm_.SetGlobalBuffer((__gm__ half *)smooth);
    }

    pipe->InitBuffer(foldBuf_, rowBlock_ * n_ * sizeof(half));
    pipe->InitBuffer(outI8Buf_, rowBlock_ * n_ * sizeof(int8_t));
    pipe->InitBuffer(tmpHBuf_, rowBlock_ * n_ * sizeof(half));
    pipe->InitBuffer(scaleStageBuf_, kMaxRowBlock * sizeof(float));
    pipe->InitBuffer(redBuf_, kMaxRowBlock * kVcmaxPairStride * sizeof(half));
    if (hasSmooth_ != 0) {
      pipe->InitBuffer(actBuf_, rowBlock_ * n_ * sizeof(half));
      pipe->InitBuffer(smoothBuf_, rowBlock_ * n_ * sizeof(half));
    }

    foldH_ = foldBuf_.Get<half>();
    outI8_ = outI8Buf_.Get<int8_t>();
    tmpH_ = tmpHBuf_.Get<half>();
    scaleStage_ = scaleStageBuf_.Get<float>();
    redH_ = redBuf_.Get<half>();
    if (hasSmooth_ != 0) {
      actH_ = actBuf_.Get<half>();
      smoothH_ = smoothBuf_.Get<half>();
    }
    LoadSmoothTable();
  }

  __aicore__ inline void Process() {
    if (myRows_ == 0) {
      return;
    }
    // First block's fold rows load here; every later block prefetches its own
    // rows at the previous block's handback (drains under the compute tail).
    uint32_t firstRows = (myRows_ < rowBlock_) ? myRows_ : rowBlock_;
    LoadFoldRow(rowStart_, firstRows);
    for (uint32_t m0 = rowStart_; m0 < rowStart_ + myRows_; m0 += rowBlock_) {
      uint32_t rowsInBlock = rowStart_ + myRows_ - m0;
      if (rowsInBlock > rowBlock_) {
        rowsInBlock = rowBlock_;
      }
      ProcessBlock(m0, rowsInBlock);
    }
  }

 private:
  // One-time per core (smooth path): the smooth table loaded R-times-tiled
  // (every tile row = the same [n] table) so the block smooth below is ONE
  // whole-tile Mul. R count-form copies, one MTE2 drain; the first block's
  // MTE2_V wait covers them (all issued before it).
  __aicore__ inline void LoadSmoothTable() {
    if (hasSmooth_ == 0) {
      return;
    }
    for (uint32_t r = 0; r < rowBlock_; ++r) {
      DataCopy(smoothH_[r * n_], smoothGm_, n_);
    }
    PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
  }

  // Fold load. 2D form (default, m <= 8192): ONE DataCopyParams descriptor
  // per token row -- H blocks of D fp16 (blockLen 8 in 32B units), src gap
  // between head blocks (m-1)*D elements ((m-1)*8 in 32B units), dst
  // contiguous (dstStride 0). transpose#1/#2/slice/TransData of the chain
  // collapse into these R descriptors. Plain form (m > 8192): H count-form
  // copies per row -- the uint16 stride field cannot hold the 2D src gap.
  // The loaded tile is byte-identical either way (pure data movement).
  // NO PipeBarrier here (FusedQKNormRope pattern: issue all copies, sync at
  // the consumer's MTE2_V wait): a barrier stalls the scalar issue thread
  // until the batch fully drains (GM latency), serializing load and compute
  // -- the v1 mid-batch barriers cost ~30% of the op. The next block's
  // MTE2_V SetFlag/WaitFlag (in-order issue) is the real sync point.
  __aicore__ inline void LoadFoldRow(uint32_t m0, uint32_t rows) {
    if (fold2d_ != 0) {
      // blockCount/srcStride are uint16 fields (tiling guarantees
      // (m-1)*foldLen32_ <= 65535 when fold2d=1); blockLen in 32B units.
      DataCopyParams params(static_cast<uint16_t>(h_), foldLen32_, static_cast<uint16_t>((m_ - 1) * foldLen32_), 0);
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(foldH_[r * n_], attnGm_[(uint64_t)(m0 + r) * d_], params);
      }
      return;
    }
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t h = 0; h < h_; ++h) {
        DataCopy(foldH_[r * n_ + h * d_], attnGm_[(uint64_t)h * m_ * d_ + (uint64_t)(m0 + r) * d_], d_);
      }
    }
  }

  // Whole-block smooth in ONE call: act = fp16-mul(fold, smooth-tiled) over
  // rows*n contiguous elements. AscendC fp16 vector Mul computes at fp32
  // internal precision with a single fp16 rounding -- the same single
  // rounding as the chain's fp16 te_mul (verified bit-exact by the precision
  // gate). The table must be ROWS-REPEATED (vector Mul does NOT broadcast --
  // a single rows*n call against an n-element table reads OOB, the v1 (13,8)
  // precision bug); the tiled Init load provides exactly that, and a ragged
  // tail block reads only its live rows (the first `rows` table copies).
  // No PipeBarrier: the consumers (absmax reduce, V_MTE2 handback event) are
  // all issued after it and the V pipe executes in order.
  __aicore__ inline void ApplySmooth(uint32_t rows) { Mul(actH_, foldH_, smoothH_, rows * n_); }

  // Whole-block absmax on the [rows, n] tile via the native reduce
  // instructions (v6; see the header comment). |act| lands in tmpH_, then
  // PER ROW: BlockReduceMax (vcgmax) collapses each 16-element block to one
  // max packed at the row prefix (counter mask = the row length, in-place
  // exactly like CANN's kernel), once more while a row still has more than
  // kHalfPerRepeat maxima (n=4096: 256 -> 16), and WholeReduceMax (vcmax)
  // reduces the final <= 128 maxima with a norm-mode bit mask -- dst[0] is
  // the max value, dst[1] its index (unread). Every stride tuple is CANN
  // quant_matmul_dequant's verbatim. Rows in a phase issue back-to-back
  // with no barriers: same mask state, in-order V pipe, disjoint row
  // prefixes. PipeBarriers sit only where the mask state changes (a pending
  // V instruction reads the mask registers at execution time; CANN's kernel
  // barriers the same spots), and the trailing SetMaskCount restores
  // counter mode for the caller's count-form ops (safe because the
  // caller's readback barrier separates it from the vcmax).
  __aicore__ inline void AbsMaxBlock(uint32_t rows, const LocalTensor<half> &act) {
    Abs(tmpH_, act, rows * n_);
    // Phase 1: one vcgmax per row, counter = the row length; the n/16 maxima
    // land in place on the row's own prefix.
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(n_);
    for (uint32_t r = 0; r < rows; ++r) {
      BlockReduceMax<half, false>(tmpH_[r * n_], tmpH_[r * n_], 1, MASK_PLACEHOLDER, 1, 1, kVcgSrcRepStride);
    }
    uint32_t slots = n_ / kHalfPerBlock;
    if (slots > kHalfPerRepeat) {
      // Phase 2 (n > 2048): collapse the maxima once more (256 -> 16 at
      // n=4096); the counter is the maxima count, same in-place tuple.
      PipeBarrier<PIPE_V>();
      SetVectorMask<half, MaskMode::COUNTER>(slots);
      for (uint32_t r = 0; r < rows; ++r) {
        BlockReduceMax<half, false>(tmpH_[r * n_], tmpH_[r * n_], 1, MASK_PLACEHOLDER, 1, 1, kVcgSrcRepStride);
      }
      slots /= kHalfPerBlock;
    }
    // Phase 3: one vcmax per row over the final `slots` maxima (norm-mode
    // bit mask, repeatTime=1, srcRepStride=0 -- the production stage-3
    // tuple; ORDER_VALUE_INDEX is the only arch-2002-legal no-tail order).
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    for (uint32_t r = 0; r < rows; ++r) {
      WholeReduceMax<half>(redH_[r * kVcmaxPairStride], tmpH_[r * n_], static_cast<int32_t>(slots), 1, 1, 1, 0,
                           ReduceOrder::ORDER_VALUE_INDEX);
    }
    PipeBarrier<PIPE_V>();  // pending vcmax must read the norm-mode mask
    SetMaskCount();
  }

  // token_scale[row] = absmax / 127, staged per 8-row group; the flush
  // happens in the block's MTE3 batch whenever the group completes. A ragged
  // final group (< kScaleGroupRows rows) is written straight to GM with 4B
  // scalar stores: flushing it as a whole 32B block would write past [M].
  __aicore__ inline void StageTokenScale(uint32_t row, float absmax) {
    float scaleOut = absmax / kQuantMaxValue;
    if (m_ - groupRowBase_ < kScaleGroupRows) {
      scaleGm_.SetValue(row, scaleOut);
      return;
    }
    uint32_t slot = row - groupRowBase_;
    Duplicate(scaleStage_[slot], scaleOut, 1);
  }

  // ---------------------------------------------------------------------------
  // One row block (R rows starting at m0):
  //   MTE2 fold rows   : 2D gather (prefetched by the previous block)
  //   smooth (opt.)    : act = fp16(fold * gs), batched over the block
  //   absmax           : native vcgmax/vcmax reduce, R GetValue for the block
  //   scale + quant    : token_scale staging + int8 quant into the y tile
  //   handback + MTE3  : fold tile freed -> prefetch next block; contiguous
  //                      y block (+ any completed 8-row scale group) out
  // ---------------------------------------------------------------------------
  __aicore__ inline void ProcessBlock(uint32_t m0, uint32_t rowsInBlock) {
    // Fold rows landed (prefetched or loaded in Process).
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    const LocalTensor<half> &act = (hasSmooth_ != 0) ? actH_ : foldH_;
    if (hasSmooth_ != 0) {
      ApplySmooth(rowsInBlock);
      // foldH_'s last reader was ApplySmooth: free it now and prefetch the
      // next block's fold rows (drains under absmax/quant/MTE3).
      SetFlag<HardEvent::V_MTE2>(0);
      WaitFlag<HardEvent::V_MTE2>(0);
      uint32_t nextM0 = m0 + rowBlock_;
      if (nextM0 < rowStart_ + myRows_) {
        uint32_t nextRows = rowStart_ + myRows_ - nextM0;
        if (nextRows > rowBlock_) {
          nextRows = rowBlock_;
        }
        LoadFoldRow(nextM0, nextRows);
      }
    }

    AbsMaxBlock(rowsInBlock, act);
    // Drain the reduce before the V_S readback consumes redH_. This barrier
    // also separates the vcmax from the next count-form V op, which makes
    // the SetMaskCount tail flip inside AbsMaxBlock safe (CANN's kernel
    // flips the mask mode right after its vcmax the same way).
    PipeBarrier<PIPE_V>();
    event_t evMax = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evMax);
    WaitFlag<HardEvent::V_S>(evMax);
    float absmax[kMaxRowBlock];
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      // vcmax ORDER_VALUE_INDEX packs [value, index] per call -- the value
      // is the even lane of each row's pair.
      absmax[r] = static_cast<float>(redH_.GetValue(r * kVcmaxPairStride));
    }

    // Scale staging (disjoint 1-elem writes, no barrier needed -- the MTE3
    // consumer is ordered by the V_MTE3 event), then the quant tail: all the
    // per-row Muls into the contiguous [rows, n] tmpH_ staging tile, then
    // ONE whole-tile Cast. The per-row Muls stay per-row (each row's fp16
    // quantMul scalar differs); the Casts collapse because the staging is
    // contiguous and the V pipe is in order (v4 issued R Casts + two
    // barrier pairs for the same data).
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      StageTokenScale(m0 + r, absmax[r]);
    }
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      float quantMul = (absmax[r] > kQuantEpsilon) ? (kQuantMaxValue / absmax[r]) : 0.0f;
      // fp16 Muls + CAST_ROUND (the 310P DynamicQuant quant tail, op-B
      // parity).
      Muls(tmpH_[r * n_], act[r * n_], static_cast<half>(quantMul), n_);
    }
    Cast(outI8_, tmpH_, RoundMode::CAST_ROUND, rowsInBlock * n_);

    if (hasSmooth_ == 0) {
      // foldH_ (as act) stayed live through the quant Muls: free it now and
      // prefetch the next block's fold rows.
      SetFlag<HardEvent::V_MTE2>(0);
      WaitFlag<HardEvent::V_MTE2>(0);
      uint32_t nextM0 = m0 + rowBlock_;
      if (nextM0 < rowStart_ + myRows_) {
        uint32_t nextRows = rowStart_ + myRows_ - nextM0;
        if (nextRows > rowBlock_) {
          nextRows = rowBlock_;
        }
        LoadFoldRow(nextM0, nextRows);
      }
    }

    // Batched MTE3 out: the whole y block is ONE contiguous copy (token-major
    // fold == o_proj x1 layout), plus all the 8-row scale groups the block
    // completed as ONE merged copy (R<=16 -> <=2 groups -> 64B, two 32B
    // blocks; a ragged final group is written row-by-row instead of as a
    // whole 32B block) -- at most 2 copies per fence (310P MTE3 copy-loss law:
    // <= 4 descriptors per burst, op-B batch pattern).
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(yGm_[(uint64_t)m0 * n_], outI8_, rowsInBlock * n_);
    uint32_t lastRow = m0 + rowsInBlock;
    uint32_t groupsDone = (lastRow - groupRowBase_) / kScaleGroupRows;
    const bool raggedTail = (lastRow == rowStart_ + myRows_) && ((lastRow - groupRowBase_) % kScaleGroupRows != 0);
    if (groupsDone > 0) {
      DataCopy(scaleGm_[groupRowBase_], scaleStage_, groupsDone * kScaleGroupRows);
      groupRowBase_ += groupsDone * kScaleGroupRows;
    }
    if (raggedTail) {
      // Those rows were already written individually by StageTokenScale.
      groupRowBase_ = lastRow;
    }
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    // The prefetched next-block fold rows (if any) are waited for at the top
    // of the next ProcessBlock via the shared MTE2_V flag.
  }

  uint32_t blockIdx_ = 0;
  uint32_t coreNum_ = 1;
  uint32_t m_ = 0;
  uint32_t h_ = 0;
  uint32_t d_ = 0;
  uint32_t n_ = 0;
  uint32_t rowBlock_ = 1;
  uint32_t hasSmooth_ = 0;
  uint32_t fold2d_ = 1;
  uint32_t foldLen32_ = 8;
  uint32_t rowStart_ = 0;
  uint32_t myRows_ = 0;
  uint32_t groupRowBase_ = 0;

  GlobalTensor<half> attnGm_;
  GlobalTensor<half> smoothGm_;
  GlobalTensor<int8_t> yGm_;
  GlobalTensor<float> scaleGm_;

  TBuf<TPosition::VECCALC> foldBuf_;
  TBuf<TPosition::VECCALC> outI8Buf_;
  TBuf<TPosition::VECCALC> tmpHBuf_;
  TBuf<TPosition::VECCALC> scaleStageBuf_;
  TBuf<TPosition::VECCALC> redBuf_;
  TBuf<TPosition::VECCALC> actBuf_;
  TBuf<TPosition::VECCALC> smoothBuf_;

  LocalTensor<half> foldH_;
  LocalTensor<int8_t> outI8_;
  LocalTensor<half> tmpH_;
  LocalTensor<float> scaleStage_;
  LocalTensor<half> redH_;
  LocalTensor<half> actH_;
  LocalTensor<half> smoothH_;
};

}  // namespace FusedAttnOutTransposeQuant

#endif  // FUSED_ATTN_OUT_TRANSPOSE_QUANT_H
