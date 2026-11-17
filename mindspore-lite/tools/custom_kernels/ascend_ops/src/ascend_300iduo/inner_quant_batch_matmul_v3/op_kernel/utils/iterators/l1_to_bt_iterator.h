/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file l1_to_bt_iterator.h
 * \brief
 */
#ifndef L1_TO_BT_ITERATOR_H
#define L1_TO_BT_ITERATOR_H

/////////////////////////////////////////////////////
// l1_to_bt
/////////////////////////////////////////////////////

// Partial specialization for V220
template <typename DataType>
struct l1_to_bt<ArchKind::ASCEND_V220, DataType> {
  __aicore__ l1_to_bt(uint64_t dst, const AscendC::LocalTensor<DataType> &src, uint16_t convControl, uint16_t nBurst,
                      uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap) {
    AscendC::LocalTensor<DataType> dstTensor;
    dstTensor.InitBuffer(dst, nBurst * lenBurst);
    dstTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::DataCopy(dstTensor, src,
                      AscendC::DataCopyParams(nBurst,    // nBurst
                                              lenBurst,  // lenBurst
                                              srcGap,    // srcGap
                                              dstGap));  // dstGap
  }
};

#endif
