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

#ifndef FUSED_QK_NORM_ROPE_BSH_H
#define FUSED_QK_NORM_ROPE_BSH_H

#include "kernel_operator.h"                     // NOLINT(build/include_subdir)
#include "fused_qk_norm_rope_bsh_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

namespace FusedQKNormRopeBSH {

// ---------------------------------------------------------------------------
// FusedQKNormRopeBSH: per-head RMSNorm (qk_norm) + NeoX RoPE on flat BSH
// q/k rows for Atlas 300I Duo (dav_m200). Pure-vector kernel (AICore task).
//
// Math per (head h, row m); fp32 accumulation, single fp16 rounding at the
// very end (the network chain rounds twice and multiplies x*invRms in fp16):
//   rms[h,m]  = rsqrt(mean_d(x^2) + epsilon)
//   xn        = x * rms * gamma[d]
//   out[0:D/2]     = xn1*cos[m,d] - xn2*sin[m,d]      (NeoX rotate-half)
//   out[D/2:D]     = xn2*cos[m,d] + xn1*sin[m,d]
//   cos/sin contract: [M, D/2]; the B half reuses the same table.
//
// Compute layout: q|k 16-head group tiles are CONCATENATED per row --
// slot s = r*G + g at offset s*span, G = ceil(Hq/16)+ceil(Hkv/16) groups.
// Each slot is interleaved in UB to a [D, 16] fp32 view (p = d*16 + lane):
//   - the square-sum runs as a lane-preserving halving tree with PER-SLOT
//     folds (v2): fold h of slot s computes only the live region
//     [s*span, s*span+h) -- the v1 batched fold recomputed the whole tile
//     every fold (the late folds were 85-99% garbage); live values see the
//     same (p, p+h) pair sequence, so the sums are bit-identical while fp32
//     vector work drops -39% (55312 -> 8128 elements per block, deploy
//     shape); per-slot lane sums land at [s*span, +16)),
//   - the invRms chain is a 4-op count=16 sequence per slot, phase-batched,
//   - the norm broadcast is 2 stride-0 Mul calls per slot (even/odd blocks),
//   - gamma applies per ROW as ONE plain Mul over G*span against a concat
//     [G, span] fp32 table ([gamEq x Gq | gamEk x Gk]) built once per core,
//   - RoPE is 6 plain Muls/Sub/Add per (row, group) against per-row
//     [D/2, 16] Brcb16 cos/sin tables (B half indexes the same table),
//   - interleave/deinterleave use scatter_vnchwconv (sibling-kernel-proven strides).
//
// Dummy-head padding: groups with fewer than 16 real heads leave stale data
// in the unloaded lanes; garbage stays strictly in-lane through all lanewise
// ops and MTE3 stores whole BSH rows (hq*d / hkv*d elements) so dummies
// (which sit after the real heads in the last group) never leave the kernel.
//
// MTE traffic (all plain 1D contiguous copies, 32B aligned):
//   MTE2-in per block: cos/sin rows + every (row, group) nat tile, issued
//   upfront with ONE drain -- no per-group MTE2 phases (unlike the BNSD variant), because
//   the concat tile needs all groups resident anyway.
//   MTE3-out per block (v2): out16 is laid out [all q slots | all k slots]
//   so every row's strips and consecutive rows are contiguous -- the whole
//   block leaves in TWO descriptors (one q strip, one k strip; 16-divisible
//   heads) or per-row copies (ragged heads), chunk-4 fenced (310P MTE3 law:
//   >4 DMA descriptors per unfenced burst drops copies; the law caps
//   descriptors, not bytes).
//
// Sync model (v2): ZERO intra-V PipeBarriers. The V pipe executes in issue
// order, so every same-pipe hazard (Brcb->Cast staging chains, tree folds,
// invRms phases, bcast/gamma/rope RAW chains, TransDataTo5HD pointer-table
// scatter) needs no fencing (sibling-kernel precedent, incl. repeat-form Brcb and
// vnchwconv scatter). Cross-pipe safety is unchanged: buffer discipline
// (MTE2-in targets nat16/cos16/sin16 are disjoint from the MTE3-out source
// out16) + event channels (each strictly 1:1 set/wait): E0 MTE2_V (immediate
// pairs: gamma-load and block-start drains), E1 V_MTE3 + E2 MTE3_V (store
// gates/drains), E3 V_MTE2 (long-lived arm covering this block's V readers of
// the MTE2-in buffers; waited only by the next prefetch or the kernel-end
// drain).
// ---------------------------------------------------------------------------
constexpr uint32_t kGroupHeads = 16;  // vnchwconv stream count
// MTE3 store copies per fence (310P law: >4 unfenced DMA descriptors drop).
constexpr uint32_t kMte3ChunkCopies = 4;
constexpr uint8_t kEvtMte2ToV = 0;            // MTE2_V: loads landed
constexpr uint8_t kEvtVToMte3 = 1;            // V_MTE3: V output ready
constexpr uint8_t kEvtMte3ToV = 2;            // MTE3_V: out16 drained
constexpr uint8_t kEvtVToMte2 = 3;            // V_MTE2: in-buffers free
constexpr uint32_t kRopeHalves = 2;           // NeoX rotate-half (each head splits in two)
constexpr uint32_t kF32PerBlk = 8;            // fp32 elements per 32B block
constexpr uint32_t kBrcbBlocksPerRepeat = 8;  // Brcb default repeat writes 8 32B blocks
constexpr uint32_t kF32RepeatUnit = 64;       // plain-count fp32 op repeat unit, in elements

class FusedQKNormRopeBSH {
 public:
  __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR gammaQ, GM_ADDR gammaK, GM_ADDR cos, GM_ADDR sin,
                              GM_ADDR qOut, GM_ADDR kOut, const FusedQKNormRopeBSHTilingData *__restrict tiling,
                              TPipe *pipe) {
    blockIdx_ = GetBlockIdx();
    coreNum_ = tiling->coreNum;
    m_ = tiling->m;
    hq_ = tiling->hq;
    hkv_ = tiling->hkv;
    d_ = tiling->d;
    rowBlock_ = tiling->rowBlock;
    epsilon_ = tiling->epsilon;
    invD_ = tiling->invD;
    span_ = kGroupHeads * d_;     // interleaved elements per slot
    half_ = span_ / kRopeHalves;  // rope half span (= 8*d)
    qGroups_ = (hq_ + kGroupHeads - 1) / kGroupHeads;
    kGroups_ = (hkv_ + kGroupHeads - 1) / kGroupHeads;
    groups_ = qGroups_ + kGroups_;

    // Row blocks striped over cores: [blockIdx*B/coreNum, (blockIdx+1)*B/coreNum).
    uint32_t blocks = (m_ + rowBlock_ - 1) / rowBlock_;
    bStart_ = blockIdx_ * blocks / coreNum_;
    bEnd_ = (blockIdx_ + 1) * blocks / coreNum_;
    rowEnd_ = m_;

    qGm_.SetGlobalBuffer((__gm__ half *)q);
    kGm_.SetGlobalBuffer((__gm__ half *)k);
    gqGm_.SetGlobalBuffer((__gm__ half *)gammaQ);
    gkGm_.SetGlobalBuffer((__gm__ half *)gammaK);
    cosGm_.SetGlobalBuffer((__gm__ half *)cos);
    sinGm_.SetGlobalBuffer((__gm__ half *)sin);
    qoGm_.SetGlobalBuffer((__gm__ half *)qOut);
    koGm_.SetGlobalBuffer((__gm__ half *)kOut);

    uint32_t tile = rowBlock_ * groups_ * span_;  // concat tile elements
    pipe->InitBuffer(natBuf_, tile * sizeof(half));
    pipe->InitBuffer(ilvBuf_, tile * sizeof(half));
    pipe->InitBuffer(outBuf_, tile * sizeof(half));
    pipe->InitBuffer(xBuf_, tile * sizeof(float));
    pipe->InitBuffer(sqBuf_, tile * sizeof(float));
    pipe->InitBuffer(cosEBuf_, rowBlock_ * half_ * sizeof(float));
    pipe->InitBuffer(sinEBuf_, rowBlock_ * half_ * sizeof(float));
    pipe->InitBuffer(cosBuf_, rowBlock_ * (d_ / kRopeHalves) * sizeof(half));
    pipe->InitBuffer(sinBuf_, rowBlock_ * (d_ / kRopeHalves) * sizeof(half));
    pipe->InitBuffer(tmp1Buf_, half_ * sizeof(float));
    pipe->InitBuffer(tmp2Buf_, half_ * sizeof(float));
    pipe->InitBuffer(tmp3Buf_, half_ * sizeof(float));
    pipe->InitBuffer(gqBuf_, d_ * sizeof(half));
    pipe->InitBuffer(gkBuf_, d_ * sizeof(half));
    pipe->InitBuffer(gEqBuf_, span_ * sizeof(half));
    pipe->InitBuffer(gEkBuf_, span_ * sizeof(half));
    pipe->InitBuffer(gamCatBuf_, groups_ * span_ * sizeof(float));

    nat16_ = natBuf_.Get<half>();
    ilv16_ = ilvBuf_.Get<half>();
    out16_ = outBuf_.Get<half>();
    x32_ = xBuf_.Get<float>();
    sq32_ = sqBuf_.Get<float>();
    cosE32_ = cosEBuf_.Get<float>();
    sinE32_ = sinEBuf_.Get<float>();
    cos16_ = cosBuf_.Get<half>();
    sin16_ = sinBuf_.Get<half>();
    tmp1_ = tmp1Buf_.Get<float>();
    tmp2_ = tmp2Buf_.Get<float>();
    tmp3_ = tmp3Buf_.Get<float>();
    gq16_ = gqBuf_.Get<half>();
    gk16_ = gkBuf_.Get<half>();
    gEq16_ = gEqBuf_.Get<half>();
    gEk16_ = gEkBuf_.Get<half>();
    gamCat32_ = gamCatBuf_.Get<float>();
  }

  __aicore__ inline void Process() {
    if (bStart_ >= bEnd_) {
      return;
    }
    BuildGammaTables();
    // First block's inputs load here; later blocks prefetch their own at the
    // end of the previous block (drain under this block's MTE3 stores).
    uint32_t firstM0 = bStart_ * rowBlock_;
    LoadBlockInputs(firstM0, BlockRows(firstM0));
    for (uint32_t b = bStart_; b < bEnd_; ++b) {
      uint32_t m0 = b * rowBlock_;
      ProcessBlock(m0, BlockRows(m0), b + 1 < bEnd_);
    }
    // Drain the last block's V_MTE2 arm. MTE3 bursts are fully drained at
    // each store site, so nothing else is left set at kernel exit.
    WaitFlag<HardEvent::V_MTE2>(kEvtVToMte2);
  }

 private:
  __aicore__ inline uint32_t BlockRows(uint32_t m0) {
    uint32_t rows = rowEnd_ - m0;
    if (rows > rowBlock_) {
      rows = rowBlock_;
    }
    return rows;
  }

  // MTE2 upfront for the whole block: cos/sin rows ([M, D/2] rows are
  // contiguous in GM) + every (row, group) nat tile (each row's group slice
  // is contiguous in GM; dummy head lanes keep stale data). All 32B aligned.
  __aicore__ inline void LoadBlockInputs(uint32_t m0, uint32_t rows) {
    DataCopy(cos16_, cosGm_[(uint64_t)m0 * (d_ / kRopeHalves)], rows * (d_ / kRopeHalves));
    DataCopy(sin16_, sinGm_[(uint64_t)m0 * (d_ / kRopeHalves)], rows * (d_ / kRopeHalves));
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t g = 0; g < groups_; ++g) {
        bool isK = g >= qGroups_;
        uint32_t hTot = isK ? hkv_ : hq_;
        uint32_t h0 = (isK ? g - qGroups_ : g) * kGroupHeads;
        uint32_t vh = hTot - h0;
        if (vh > kGroupHeads) {
          vh = kGroupHeads;
        }
        const GlobalTensor<half> &inGm = isK ? kGm_ : qGm_;
        DataCopy(nat16_[(r * groups_ + g) * span_], inGm[((uint64_t)(m0 + r) * hTot + h0) * d_], vh * d_);
      }
    }
  }

  // One-time per core: [D,16] gamma tables (Brcb16 broadcasts each scalar to
  // a full 16-half block, gE[16*j+i] = gamma[j]), then the CONCAT fp32 table
  // gamCat32_ [G, span] = [gamEq x Gq | gamEk x Gk] so the gamma mul is one
  // plain Mul per row over the whole G*span concat region.
  __aicore__ inline void BuildGammaTables() {
    DataCopy(gq16_, gqGm_, d_);
    DataCopy(gk16_, gkGm_, d_);
    SetFlag<HardEvent::MTE2_V>(kEvtMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvtMte2ToV);
    // Default BrcbRepeatParams: dstBlkStride=1, dstRepStride=8 blocks ->
    // the d/8 repeats write d blocks back-to-back (contiguous [D,16]).
    Brcb(gEq16_, gq16_, static_cast<uint8_t>(d_ / kBrcbBlocksPerRepeat), BrcbRepeatParams());
    Brcb(gEk16_, gk16_, static_cast<uint8_t>(d_ / kBrcbBlocksPerRepeat), BrcbRepeatParams());
    for (uint32_t s = 0; s < groups_; ++s) {
      const LocalTensor<half> &src = (s < qGroups_) ? gEq16_ : gEk16_;
      Cast(gamCat32_[s * span_], src, RoundMode::CAST_NONE, span_);
    }
  }

  // Per-block rotary tables, BATCHED over rows (v2): Brcb16 doubles each
  // [D/2] row into a [D/2, 16] half table (cE[16*d' + lane] = cos[m, d']),
  // cast to fp32; the B rope half reads the same table (NeoX doubling).
  // repeatTimes is uint8: chunk rows so (rows * D/2 / 8) <= 255.
  // Staging aliases out16_[0, 2*rows*half_): out16_ is dead at block entry
  // (first real writer is the deinterleave, later in V issue order) and the
  // previous block's MTE3 reads fully drained at their store-site fences.
  __aicore__ inline void BuildRotaryTables(uint32_t rows) {
    LocalTensor<half> bigCE16 = out16_;
    LocalTensor<half> bigSE16 = out16_[rows * half_];
    uint32_t repsPerRow = (d_ / kRopeHalves) / kBrcbBlocksPerRepeat;
    uint32_t rowsPerCall = 255 / repsPerRow;
    uint32_t done = 0;
    while (done < rows) {
      uint32_t n = rows - done;
      if (n > rowsPerCall) {
        n = rowsPerCall;
      }
      Brcb(bigCE16[done * half_], cos16_[done * (d_ / kRopeHalves)], static_cast<uint8_t>(n * repsPerRow),
           BrcbRepeatParams());
      Brcb(bigSE16[done * half_], sin16_[done * (d_ / kRopeHalves)], static_cast<uint8_t>(n * repsPerRow),
           BrcbRepeatParams());
      done += n;
    }
    Cast(cosE32_, bigCE16, RoundMode::CAST_NONE, rows * half_);
    Cast(sinE32_, bigSE16, RoundMode::CAST_NONE, rows * half_);
  }

  // [16, D] natural -> [D, 16] interleaved (per slot). scatter_vnchwconv:
  // srcList[n] = nat[n*D] (head rows, pitch D), dstList[m] = ilv[16*m]
  // (d-rows, pitch 16), repeat r = d/16 -> dst row d = 16r+m holds head m's
  // value at d. Proven stride pair.
  __aicore__ inline void InterleaveRow(const LocalTensor<half> &nat, const LocalTensor<half> &ilv) {
    uint64_t dstList[kGroupHeads];
    uint64_t srcList[kGroupHeads];
    for (uint32_t m = 0; m < kGroupHeads; ++m) {
      dstList[m] = (uint64_t)ilv[kGroupHeads * m].GetPhyAddr();
    }
    for (uint32_t n = 0; n < kGroupHeads; ++n) {
      srcList[n] = (uint64_t)nat[d_ * n].GetPhyAddr();
    }
    TransDataTo5HDParams params;
    params.dstHighHalf = false;
    params.srcHighHalf = false;
    params.repeatTimes = static_cast<uint8_t>(d_ / kGroupHeads);
    params.dstRepStride = kGroupHeads;
    params.srcRepStride = 1;
    TransDataTo5HD<half>(dstList, srcList, params);
  }

  // [D, 16] interleaved -> [16, D] natural (per slot), MIRRORED single call
  // (v2): exact stride-swap of InterleaveRow.
  __aicore__ inline void DeinterleaveRow(const LocalTensor<half> &ilv, const LocalTensor<half> &out) {
    uint64_t dstList[kGroupHeads];
    uint64_t srcList[kGroupHeads];
    for (uint32_t m = 0; m < kGroupHeads; ++m) {
      dstList[m] = (uint64_t)out[d_ * m].GetPhyAddr();
    }
    for (uint32_t n = 0; n < kGroupHeads; ++n) {
      srcList[n] = (uint64_t)ilv[kGroupHeads * n].GetPhyAddr();
    }
    TransDataTo5HDParams params;
    params.dstHighHalf = false;
    params.srcHighHalf = false;
    params.repeatTimes = static_cast<uint8_t>(d_ / kGroupHeads);
    params.dstRepStride = 1;
    params.srcRepStride = kGroupHeads;
    TransDataTo5HD<half>(dstList, srcList, params);
  }

  // iterations (pipe order makes reuse safe).
  __aicore__ inline void NormRopeSlots(uint32_t rows) {
    uint32_t slots = rows * groups_;
    // Packed slabs at the tile head, all inside post-fold dead workspace:
    //   x = ms+eps [0, n)   y = rsqrt [n, 2n)   t = Newton temp [2n, 3n)
    // with n = slots*16 (3n <= slots*span always, since 3 <= d). Once the
    // gather below has consumed every [s*span, +16) sum, nothing reads
    // sq32_ again except the broadcast (now on yT), so the only ordering
    // constraint is that these writes issue after the gather -- guaranteed
    // by in-order V execution (within the gather itself, the write that
    // would touch slot s' sums comes at step s'*d, far after its read at
    // step s'; s=0 reuses its own sum slot in place, the v2 idiom).
    uint32_t n = slots * kGroupHeads;
    LocalTensor<float> xT = sq32_;                   // x = ms+eps, packed
    LocalTensor<float> yT = sq32_[n];                // y = invRms, packed
    LocalTensor<float> tT = sq32_[kRopeHalves * n];  // Newton temp, packed
    for (uint32_t s = 0; s < slots; ++s) {
      Muls(xT[s * kGroupHeads], sq32_[s * span_], invD_, kGroupHeads);
    }
    Adds(xT, xT, epsilon_, n);
    // Newton-refine invRms. dav_m200 Sqrt/Reciprocal are fast approximations
    // (~2.8e-3 relative each) and the combined error is COMMON-MODE per head:
    // Two Newton steps y <- y*(1.5 - 0.5*x*y^2) on
    // x = ms+eps drive the error to fp32 rounding (~3e-9). Same elementwise
    // ops on the same inputs in the same order as the v3 per-slot chain ->
    // bit-identical invRms; only the addresses are packed.
    Sqrt(yT, xT, n);
    Reciprocal(yT, yT, n);
    for (uint32_t it = 0; it < kRopeHalves; ++it) {
      Mul(tT, yT, yT, n);  // t = y*y
      Mul(tT, xT, tT, n);  // t = x*y*y
      Muls(tT, tT, -0.5f, n);
      Adds(tT, tT, 1.5f, n);
      Mul(yT, yT, tT, n);  // y *= 1.5 - 0.5*x*y*y
    }
    BinaryRepeatParams bcast(kRopeHalves, kRopeHalves, 0, kGroupHeads, kGroupHeads, 0);
    for (uint32_t s = 0; s < slots; ++s) {
      LocalTensor<float> rmsSlot = yT[s * kGroupHeads];
      LocalTensor<float> xSlot = x32_[s * span_];
      Mul(xSlot, xSlot, rmsSlot, kF32RepeatUnit, static_cast<uint8_t>(span_ / (kRopeHalves * kF32RepeatUnit)), bcast);
      Mul(xSlot[kF32PerBlk], xSlot[kF32PerBlk], rmsSlot[kF32PerBlk], kF32RepeatUnit,
          static_cast<uint8_t>(span_ / (kRopeHalves * kF32RepeatUnit)), bcast);
    }
    // gamma: ONE plain Mul per row over the concat region vs the [G, span]
    // concat table (slot s lives at x32_[r*G*span + ...] = row offset + the
    // same s*span offset inside gamCat32_).
    for (uint32_t r = 0; r < rows; ++r) {
      Mul(x32_[r * groups_ * span_], x32_[r * groups_ * span_], gamCat32_, groups_ * span_);
    }
    // NeoX rope per (row, group): A = first half, B = second half.
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t g = 0; g < groups_; ++g) {
        LocalTensor<float> xRow = x32_[(r * groups_ + g) * span_];
        LocalTensor<float> cosE = cosE32_[r * half_];
        LocalTensor<float> sinE = sinE32_[r * half_];
        Mul(tmp1_, xRow, cosE, half_);          // A*cos
        Mul(tmp2_, xRow[half_], sinE, half_);   // B*sin
        Mul(tmp3_, xRow, sinE, half_);          // A*sin (A still live)
        Sub(xRow, tmp1_, tmp2_, half_);         // outA = A*cos - B*sin
        Mul(tmp2_, xRow[half_], cosE, half_);   // B*cos (B still live)
        Add(xRow[half_], tmp2_, tmp3_, half_);  // outB = B*cos + A*sin
      }
    }
  }

  __aicore__ inline LocalTensor<half> OutSlot(uint32_t r, uint32_t g, uint32_t rows) {
    if (g < qGroups_) {
      return out16_[(r * qGroups_ + g) * span_];
    }
    return out16_[rows * qGroups_ * span_ + (r * kGroups_ + (g - qGroups_)) * span_];
  }

  __aicore__ inline void Mte3Gate() {
    SetFlag<HardEvent::V_MTE3>(kEvtVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvtVToMte3);
  }
  __aicore__ inline void Mte3Drain() {
    SetFlag<HardEvent::MTE3_V>(kEvtMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvtMte3ToV);
  }

  __aicore__ inline void ProcessBlock(uint32_t m0, uint32_t rows, bool hasNextBlock) {
    // Inputs landed (prefetched by the previous block or loaded in Process).
    SetFlag<HardEvent::MTE2_V>(kEvtMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvtMte2ToV);
    BuildRotaryTables(rows);

    uint32_t slots = rows * groups_;
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t g = 0; g < groups_; ++g) {
        InterleaveRow(nat16_[(r * groups_ + g) * span_], ilv16_[(r * groups_ + g) * span_]);
      }
    }
    // Arm: once V drains the interleaves (the only nat16 readers) and the
    // rotary-table builds (cos16/sin16 readers, issued before), the next
    // MTE2 prefetch into nat16/cos16/sin16 is safe.
    SetFlag<HardEvent::V_MTE2>(kEvtVToMte2);

    // Batched cast to fp32 + squares over the whole concat tile (same-pipe
    // RAW chain, no barriers -- same-pipe discipline).
    Cast(x32_, ilv16_, RoundMode::CAST_NONE, slots * span_);
    Mul(sq32_, x32_, x32_, slots * span_);
    // Lane-preserving halving tree, PER-SLOT folds (v2). The v1 batched fold
    // (count = slots*span - h) recomputed the whole tile every fold, but
    // only [s*span, s*span+h) of each slot is live -- at the deploy shape
    // that was 55312 computed vs 8128 live elements per block (-39% of all
    // fp32 vector work; the late folds were 85-99% garbage). Per-slot fold h
    // computes exactly the live outputs: dst/src0 = [s*span, +h), src1 =
    // [s*span+h, +h), all inside slot s. The live values see the IDENTICAL
    // (p, p+h) pair sequence as the batched form -- bit-identical by
    // construction. Same-pipe RAW chain, no barriers between folds.
    for (uint32_t h = span_ / kRopeHalves; h >= kGroupHeads; h /= kRopeHalves) {
      for (uint32_t s = 0; s < slots; ++s) {
        Add(sq32_[s * span_], sq32_[s * span_], sq32_[s * span_ + h], h);
      }
    }
    NormRopeSlots(rows);
    Cast(ilv16_, x32_, RoundMode::CAST_NONE, slots * span_);

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t g = 0; g < groups_; ++g) {
        DeinterleaveRow(ilv16_[(r * groups_ + g) * span_], OutSlot(r, g, rows));
      }
    }

    // MTE3 out (v2 batched): the [q | k] out16 split makes every row's q and
    // k strip contiguous, and consecutive rows back-to-back. A 16-divisible
    // head count makes the strip length equal the GM row length, so that
    // side leaves as ONE whole-block descriptor; ragged counts (dummy tail
    // inside the last group tile, e.g. hkv=8 at the deploy shape) keep
    // per-row copies that skip the dummies. The law caps descriptors per
    // unfenced burst, not bytes (op-B 96KB single-fenced precedent), so the
    // deploy shape (hq=16, hkv=8, rows=2) is 3 descriptors under ONE
    // gate/drain pair (v1: 4 per-row copies). Chunk-4 fenced throughout.
    uint32_t fenced = 0;
    if (hq_ % kGroupHeads == 0) {
      Mte3Gate();
      DataCopy(qoGm_[(uint64_t)m0 * hq_ * d_], out16_, rows * hq_ * d_);
      fenced = 1;
    } else {
      for (uint32_t r = 0; r < rows; ++r) {
        if (fenced == 0) {
          Mte3Gate();
        }
        DataCopy(qoGm_[(uint64_t)(m0 + r) * hq_ * d_], OutSlot(r, 0, rows), hq_ * d_);
        if (++fenced == kMte3ChunkCopies) {
          Mte3Drain();
          fenced = 0;
        }
      }
    }
    if (hkv_ % kGroupHeads == 0) {
      if (fenced == 0) {
        Mte3Gate();
      }
      DataCopy(koGm_[(uint64_t)m0 * hkv_ * d_], out16_[rows * qGroups_ * span_], rows * hkv_ * d_);
      ++fenced;
    } else {
      for (uint32_t r = 0; r < rows; ++r) {
        if (fenced == 0) {
          Mte3Gate();
        }
        DataCopy(koGm_[(uint64_t)(m0 + r) * hkv_ * d_], OutSlot(r, qGroups_, rows), hkv_ * d_);
        if (++fenced == kMte3ChunkCopies) {
          Mte3Drain();
          fenced = 0;
        }
      }
    }
    if (fenced != 0) {
      Mte3Drain();
    }

    // Prefetch the next block's inputs: waits the V_MTE2 arm (covers all of
    // this block's V readers of nat16/cos16/sin16), then the copies drain
    // under the MTE3 stores. Block-based guard: only prefetch when THIS core
    // runs the next block (a stray prefetch would consume the arm and the
    // end-of-kernel drain would deadlock).
    if (hasNextBlock) {
      WaitFlag<HardEvent::V_MTE2>(kEvtVToMte2);
      LoadBlockInputs(m0 + rowBlock_, BlockRows(m0 + rowBlock_));
    }
  }

  uint32_t blockIdx_ = 0;
  uint32_t coreNum_ = 1;
  uint32_t m_ = 0;
  uint32_t hq_ = 0;
  uint32_t hkv_ = 0;
  uint32_t d_ = 0;
  uint32_t rowBlock_ = 1;
  float epsilon_ = 1e-6;
  float invD_ = 0.0f;
  uint32_t span_ = 0;  // 16 * d
  uint32_t half_ = 0;  // 8 * d
  uint32_t qGroups_ = 0;
  uint32_t kGroups_ = 0;
  uint32_t groups_ = 0;  // qGroups_ + kGroups_
  uint32_t bStart_ = 0;
  uint32_t bEnd_ = 0;
  uint32_t rowEnd_ = 0;

  GlobalTensor<half> qGm_;
  GlobalTensor<half> kGm_;
  GlobalTensor<half> gqGm_;
  GlobalTensor<half> gkGm_;
  GlobalTensor<half> cosGm_;
  GlobalTensor<half> sinGm_;
  GlobalTensor<half> qoGm_;
  GlobalTensor<half> koGm_;

  TBuf<TPosition::VECCALC> natBuf_;
  TBuf<TPosition::VECCALC> ilvBuf_;
  TBuf<TPosition::VECCALC> outBuf_;
  TBuf<TPosition::VECCALC> xBuf_;
  TBuf<TPosition::VECCALC> sqBuf_;
  TBuf<TPosition::VECCALC> cosEBuf_;
  TBuf<TPosition::VECCALC> sinEBuf_;
  TBuf<TPosition::VECCALC> cosBuf_;
  TBuf<TPosition::VECCALC> sinBuf_;
  TBuf<TPosition::VECCALC> tmp1Buf_;
  TBuf<TPosition::VECCALC> tmp2Buf_;
  TBuf<TPosition::VECCALC> tmp3Buf_;
  TBuf<TPosition::VECCALC> gqBuf_;
  TBuf<TPosition::VECCALC> gkBuf_;
  TBuf<TPosition::VECCALC> gEqBuf_;
  TBuf<TPosition::VECCALC> gEkBuf_;
  TBuf<TPosition::VECCALC> gamCatBuf_;

  LocalTensor<half> nat16_;
  LocalTensor<half> ilv16_;
  LocalTensor<half> out16_;
  LocalTensor<float> x32_;
  LocalTensor<float> sq32_;
  LocalTensor<float> cosE32_;
  LocalTensor<float> sinE32_;
  LocalTensor<half> cos16_;
  LocalTensor<half> sin16_;
  LocalTensor<float> tmp1_;
  LocalTensor<float> tmp2_;
  LocalTensor<float> tmp3_;
  LocalTensor<half> gq16_;
  LocalTensor<half> gk16_;
  LocalTensor<half> gEq16_;
  LocalTensor<half> gEk16_;
  LocalTensor<float> gamCat32_;
};

}  // namespace FusedQKNormRopeBSH

#endif  // FUSED_QK_NORM_ROPE_BSH_H
