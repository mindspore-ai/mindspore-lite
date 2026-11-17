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
 * ============================================================================
 */
// Bias helpers (row-bias Mmad, full-size rank-16 decomposition) and the 2x2 write-back
// quadrant of the InnerBatchMatMul kernel.
// Included from inner_batch_matmul_kernel.h after the class declaration (not standalone).
#ifndef INNER_BATCH_MATMUL_KERNEL_BIAS_H_
#define INNER_BATCH_MATMUL_KERNEL_BIAS_H_

namespace InnerBatchMatMul {

// Row bias: build the synthetic A' (all ones) / B' (row 0 = bias) and run one extra K=16 Mmad
// that accumulates bias into L0C. This is equivalent to the built-in "preload bias into L0C"
// (bias is added inside the Cube fp32 accumulator) but only uses already-validated primitives.
__aicore__ inline void InnerBatchMatMulKernel::BuildBiasAux(uint64_t biasOff, uint32_t nLo, uint32_t n1) {
  LocalTensor<half> ubA = ubAux_.Get<half>();
  LocalTensor<half> ubB = ubAux_.Get<half>()[tileM_ * kFractalDim];
  Duplicate(ubA, static_cast<half>(1), tileM_ * kFractalDim);
  Duplicate(ubB, static_cast<half>(0), tileN_ * kFractalDim);
  Sync<HardEvent::V_MTE2>();
  // B': row 0 of every n-fractal holds the bias of those 16 columns (the first row of each NZ
  // fractal is exactly 16 values).
  for (uint32_t j = 0; j < n1; ++j) {
    DataCopy(ubB[j * kFractalElems], biasGm_[biasOff + (static_cast<uint64_t>(nLo) / kFractalDim + j) * kFractalElems],
             kFractalDim);
  }
  Sync<HardEvent::MTE2_V>();
  Sync<HardEvent::V_MTE3>();
  DataCopyParams pc;  // UB->L1, blockLen in 32B units
  pc.blockCount = 1;
  pc.blockLen = static_cast<uint16_t>(tileM_ * kFractalDim * sizeof(half) / kBlockBytes);
  pc.srcStride = 0;
  pc.dstStride = 0;
  DataCopy(b1AuxA_.Get<half>(), ubA, pc);
  pc.blockLen = static_cast<uint16_t>(tileN_ * kFractalDim * sizeof(half) / kBlockBytes);
  DataCopy(b1AuxB_.Get<half>(), ubB, pc);
}

__aicore__ inline void InnerBatchMatMulKernel::IssueBiasMmad(LocalTensor<float> &c1, uint32_t m1, uint32_t n1,
                                                             uint32_t qm) {
  const uint32_t m1sub = p_->baseM / kFractalDim;
  // L1 -> L0: A' is [k][m] (kf = 1), B' is [n][k] (kf = 1).
  Sync<HardEvent::MTE3_MTE1>();
  LocalTensor<half> aL1 = b1AuxA_.Get<half>();
  LocalTensor<half> bL1 = b1AuxB_.Get<half>();
  LocalTensor<half> a2 = L0A(0);
  LocalTensor<half> b2 = L0B(0);
  LoadData2DParams lda;
  lda.startIndex = 0;
  lda.repeatTimes = static_cast<uint8_t>(m1);
  lda.srcStride = 1;
  lda.dstGap = 0;  // kf = 1
  lda.ifTranspose = false;
  LoadData(a2[0], aL1[0], lda);
  LoadData2DParams ldb;
  ldb.startIndex = 0;
  ldb.repeatTimes = static_cast<uint8_t>(n1);
  ldb.srcStride = 1;
  ldb.dstGap = 0;
  ldb.ifTranspose = true;
  LoadData(b2[0], bL1[0], ldb);
  Sync<HardEvent::MTE1_M>();
  for (uint32_t qi = 0; qi < qm; ++qi) {
    const uint32_t mq = (qi == 0) ? ((m1 < m1sub) ? m1 : m1sub) : (m1 - m1sub);
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mq * kFractalDim);
    mp.n = static_cast<uint16_t>(n1 * kFractalDim);
    mp.k = static_cast<uint16_t>(kFractalDim);
    mp.cmatrixInitVal = false;  // accumulate into the existing C
    mp.isBias = false;
    mp.unitFlag = 0;
    Mmad(c1[qi * tileM_ * p_->baseN], a2[qi * m1sub * kFractalElems], b2, mp);
  }
  Sync<HardEvent::M_MTE1>();
}

// Full-size bias rank-16 decomposition (2x2 path): build A' = 128x128 block-diagonal identity in
// L1. The layout matches one K chunk of the real A ([kfrac][mfrac][16(m)][16(k)]): only the
// diagonal fractal (r,r) holds a 16x16 identity and the other 56 fractals must be exactly zero
// (otherwise C is silently polluted). Then the "I x bias" Mmad is equivalent to moving bias into
// L0C unchanged. Executed only once in Init: clear the block, then write the 8 diagonal entries
// (one Duplicate with repeatTime across the diagonal fractals).
__aicore__ inline void InnerBatchMatMulKernel::BuildBiasIdentL1() {
  const uint32_t idDim = p_->baseM / kFractalDim;  // identity edge in fractals (baseM=128 -> 8)
  LocalTensor<half> ub = ubIdent_.Get<half>();
  Duplicate(ub, static_cast<half>(0), static_cast<int32_t>(idDim * idDim * kFractalElems));
  // Diagonal element (row, row) sits at row*16+row inside the diagonal fractal (r*idDim+r): the
  // dst points at the row start (32B aligned) and the mask only sets bit=row, so the value lands
  // at dst+row. Neighbouring diagonal fractals are idDim+1 fractals apart, which is
  // (idDim+1)*16 in level-0 repeat units (one repeat is 16 half elements).
  const uint8_t repStride = static_cast<uint8_t>((idDim + 1U) * kFractalElems / kFractalDim);
  for (uint32_t row = 0; row < kFractalDim; ++row) {
    uint64_t maskBits[2] = {1ULL << row, 0ULL};
    Duplicate<half, true>(ub[row * kFractalDim], static_cast<half>(1), maskBits, static_cast<uint8_t>(idDim), 1,
                          repStride);
  }
  Sync<HardEvent::V_MTE3>();
  DataCopyParams ip;  // UB->L1, blockLen in 32B units
  ip.blockCount = 1;
  ip.blockLen = static_cast<uint16_t>(idDim * idDim * kFractalElems * sizeof(half) / kBlockBytes);
  ip.srcStride = 0;
  ip.dstStride = 0;
  DataCopy(b1Ident_.Get<half>(), ub, ip);
  Sync<HardEvent::MTE3_MTE1>();  // no L1->L0A may start before A' has landed in L1
}

// Full-size bias rank-16 decomposition (2x2 path): one C = I x bias Mmad per (row, column)
// quadrant writes bias into L0C.
// - The shape is identical to the K-loop Mmad of the same row quadrant (m = mq*16, k = mq*16,
//   n = nq*16), so the [n][m] fractal layout of L0C matches the K-loop accumulator exactly
//   (including tails: when mq shrinks the n-frac stride shrinks with it);
// - cmatrixInitVal=true: the first block is cleared, so C = 0 + I*bias = bias (I elements are
//   exactly 1.0 and bias is fp16, so there is no rounding);
// - A' is loaded once per row quadrant (mq x mq fractals <= 32KB, one L0A slot) and B' once per
//   column quadrant (<= 32KB, one L0B slot).
__aicore__ inline void InnerBatchMatMulKernel::IssueBiasMmadFull(LocalTensor<float> &c1, uint32_t m1, uint32_t n1,
                                                                 uint32_t qm, uint32_t qn) {
  const uint32_t m1sub = p_->baseM / kFractalDim;
  const uint32_t n1sub = p_->baseN / kFractalDim;
  LocalTensor<half> identL1 = b1Ident_.Get<half>();
  LocalTensor<half> a2 = L0A(0);
  LocalTensor<half> b2 = L0B(0);
  for (uint32_t qi = 0; qi < qm; ++qi) {
    const uint32_t mq = (qi == 0) ? ((m1 < m1sub) ? m1 : m1sub) : (m1 - m1sub);
    // Start of this row quadrant in the m-fractal axis of the tile (0 or m1sub, i.e. the
    // accumulated offset, not qi*mq which only coincides with it for a full tile).
    const uint32_t mqStart = (qi == 0) ? 0U : m1sub;
    // A': the first mq k-fractals x the first mq m-fractals of the identity
    // (L1 [kfrac][mfrac] -> L0A [mfrac][kfrac]). Any diagonal mq x mq block of an identity is an
    // identity itself, so the top-left block also serves row quadrant 1 without an m offset.
    LoadData2DParams lda;
    lda.startIndex = 0;
    lda.repeatTimes = static_cast<uint8_t>(mq);
    lda.srcStride = 1;
    lda.dstGap = static_cast<uint16_t>(mq - 1);  // same shape as the real A path: dstGap = kf-1
    lda.ifTranspose = false;
    for (uint32_t kk = 0; kk < mq; ++kk) {
      LoadData(a2[kk * kFractalElems], identL1[kk * m1sub * kFractalElems], lda);
    }
    for (uint32_t qj = 0; qj < qn; ++qj) {
      const uint32_t nq = (qj == 0) ? ((n1 < n1sub) ? n1 : n1sub) : (n1 - n1sub);
      // B': the whole tile is stored contiguously as [n-frac][m-frac][16][16] (m1 m-fracs each);
      // this (row, column) quadrant starts at (qj*n1sub*m1 + mqStart) fractals and neighbouring
      // n-fractals are m1 fractals apart.
      LocalTensor<half> biasL1 =
        b1Bias_.Get<half>()[curBiasSlot_ * 2U * p_->baseM * tileN_ + (qj * n1sub * m1 + mqStart) * kFractalElems];
      LoadData2DParams ldb;
      ldb.startIndex = 0;
      ldb.repeatTimes = static_cast<uint8_t>(nq);
      ldb.srcStride = static_cast<uint16_t>(m1);  // neighbouring n-fractals are m1 fractals apart in L1
      ldb.dstGap = 0;
      ldb.ifTranspose = true;  // same as the real B path: the L0B fractal layout is the transposed [16n][16k]
      for (uint32_t kk = 0; kk < mq; ++kk) {
        LoadData(b2[kk * nq * kFractalElems], biasL1[kk * kFractalElems], ldb);
      }
      Sync<HardEvent::MTE1_M>();
      MmadParams mp;
      mp.m = static_cast<uint16_t>(mq * kFractalDim);
      mp.n = static_cast<uint16_t>(nq * kFractalDim);
      mp.k = static_cast<uint16_t>(mq * kFractalDim);
      mp.cmatrixInitVal = true;  // clear the first block -> C = I*bias
      mp.isBias = false;
      mp.unitFlag = 0;
      Mmad(c1[qi * tileM_ * p_->baseN + qj * n1sub * mq * kFractalElems], a2, b2, mp);
      Sync<HardEvent::M_MTE1>();  // only after the Mmad has read L0B may the next column quadrant overwrite b2
    }
  }
}

// 2x2 bias prefetch (full-size bias): move the fp16 bias of the whole tile into L1 with one
// strided copy. The tile bias is contiguous as [n-frac][m-frac][16][16], so the burst is
// m1*32B (8KB at m1 = 16). Compared with "two 4KB bursts per row quadrant" this halves the
// number of bursts, doubles each burst and needs a single DataCopy.
__aicore__ inline void InnerBatchMatMulKernel::PrefetchBiasL1(uint32_t slot, uint32_t m1, uint32_t n1,
                                                              uint64_t biasOffTile) {
  const uint32_t m1All = p_->m / kFractalDim;
  LocalTensor<half> dst = b1Bias_.Get<half>()[slot * 2U * p_->baseM * tileN_];
  DataCopyParams bp;
  bp.blockCount = static_cast<uint16_t>(n1);              // number of n-fractals
  bp.blockLen = static_cast<uint16_t>(m1 * kFractalDim);  // 32B units
  bp.srcStride = static_cast<uint16_t>((m1All - m1) * kFractalDim);
  bp.dstStride = 0;
  DataCopy(dst, biasGm_[biasOffTile], bp);
}

// Vector flattening NZ([n][m][16][16]) -> ND([m][n]); the layout mirrors the built-in
// TransNZ2NDByVec from copy_cube_out_datacopy_wrapper.h, item by item:
//   - one repeat moves 8 32B blocks (= 8 rows x 16 half = 128 half, exactly one half vector mask);
//   - neighbouring rows of dst are n1 blocks apart (dst row width = n1*16 half), so dstBlkStride = n1;
//   - the 8 source rows are contiguous in UB ([m][16] contiguous inside one n-fractal), so
//     srcBlkStride = 1 and srcRepStride = 8.
// Compared with the per-n-fractal "blockLen=1" UB->UB DataCopy (one instruction per 32B), this
// cuts the copy instruction count of a quadrant by about 8x and changes neither the data layout
// nor the synchronization (V pipe -> the trailing Sync<V_MTE3> still hands over to the GM write).
__aicore__ inline void InnerBatchMatMulKernel::Nz2NdByVec(LocalTensor<half> dst, LocalTensor<half> src, uint32_t m1,
                                                          uint32_t n1) {
  const uint32_t perFrac = m1 * kFractalElems;                // half elements of one n-fractal
  const uint8_t repeatTimes = static_cast<uint8_t>(m1 * 2U);  // m1*16 rows / 8 rows per repeat
  UnaryRepeatParams rp;
  rp.dstBlkStride = static_cast<uint16_t>(n1);      // dst rows are one full row apart (n1 32B blocks)
  rp.srcBlkStride = 1;                              // the 8 source rows are contiguous
  rp.dstRepStride = static_cast<uint8_t>(n1 * 8U);  // next group of 8 rows
  rp.srcRepStride = 8;                              // source advances by 8 rows
  uint64_t maskAll[2] = {~0ULL, ~0ULL};             // 128 half per repeat (bit mode, full mask)
  for (uint32_t kk = 0; kk < n1; ++kk) {
    Muls<half, true>(dst[kk * kFractalDim], src[kk * perFrac], static_cast<half>(1), maskAll, repeatTimes, rp);
  }
}

// 2x2 write-back of one quadrant: L0C->UB(fp32) -> Cast(fp16) -> NZ2ND -> GM
// (bias was already written into L0C by the bias Mmad at the start of the tile).
__aicore__ inline void InnerBatchMatMulKernel::WriteBackQuad(const QuadDesc &q) {
  LocalTensor<float> ub32 = ubFp32_.Get<float>();
  LocalTensor<half> ub16 = ubFp16_.Get<half>();
  LocalTensor<half> ubNd = ubNd_.Get<half>();
  LocalTensor<float> c1 = c1Buf_.Get<float>();

  DataCopyParams cp;
  cp.blockCount = static_cast<uint16_t>(q.n1);
  cp.blockLen = static_cast<uint16_t>(q.m1);
  cp.srcStride = 0;
  cp.dstStride = 0;
  DataCopyEnhancedParams ep;
  ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
  DataCopy(ub32, c1[q.l0cOff], cp, ep);
  Sync<HardEvent::MTE2_V>();
  // The full-size bias was already written into L0C by the "I x bias" Mmad at the start of the
  // tile, so there is no L1->UB / Cast / Add here.
  Cast(ub16, ub32, RoundMode::CAST_NONE, q.m1 * q.n1 * kFractalElems);
  Sync<HardEvent::V_MTE2>();  // ub32 may be reused by the next quadrant's L0C->UB

  // NZ([n][m][16][16]) -> ND([m][n]): vector flattening (see Nz2NdByVec), then one 2D GM write.
  Sync<HardEvent::MTE3_V>();  // the previous quadrant's GM write finished, ubNd is free again
  Nz2NdByVec(ubNd, ub16, q.m1, q.n1);
  Sync<HardEvent::V_MTE3>();

  DataCopyParams gout;
  gout.blockCount = static_cast<uint16_t>(q.m1 * kFractalDim);
  gout.blockLen = static_cast<uint16_t>(q.n1);
  gout.srcStride = 0;
  gout.dstStride = static_cast<uint16_t>((p_->n - q.n1 * kFractalDim) / kFractalDim);
  DataCopy(cGm_[q.cOff + static_cast<uint64_t>(q.mRowOff) * p_->n + q.nColOff], ubNd, gout);
}

}  // namespace InnerBatchMatMul

#endif  // INNER_BATCH_MATMUL_KERNEL_BIAS_H_
