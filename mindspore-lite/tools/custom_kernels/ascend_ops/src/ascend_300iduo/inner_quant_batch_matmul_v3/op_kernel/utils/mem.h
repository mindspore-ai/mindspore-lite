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
 * \file mem.h
 * \brief
 */

#ifndef INCLUDE_MEM_H
#define INCLUDE_MEM_H

#include <kernel_operator_block_sync_intf.h>

#include "./hardware.h"

namespace PpMatMulNS {
enum class BufferType : uint32_t { ASCEND_UB, ASCEND_CB, ASCEND_L0A, ASCEND_L0B, ASCEND_L0C, ASCEND_MAX };

template <ArchKind archKind>
struct OnChipBuffer {
  template <typename T>
  using Tensor = AscendC::LocalTensor<T>;

 public:
  __aicore__ inline OnChipBuffer() {
    constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
      HardwareInfo<archKind>::ubSize, HardwareInfo<archKind>::l1Size, HardwareInfo<archKind>::l0ASize,
      HardwareInfo<archKind>::l0BSize, HardwareInfo<archKind>::l0CSize};
#if defined(__DAV_C220_VEC__)
    buffer_[(uint32_t)BufferType::ASCEND_UB] =
      Tensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
#elif defined(__DAV_C220_CUBE__)
    buffer_[(uint32_t)BufferType::ASCEND_CB] =
      Tensor<uint8_t>(AscendC::TPosition::A1, 0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
    buffer_[(uint32_t)BufferType::ASCEND_L0A] =
      Tensor<uint8_t>(AscendC::TPosition::A2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
    buffer_[(uint32_t)BufferType::ASCEND_L0B] =
      Tensor<uint8_t>(AscendC::TPosition::B2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
    buffer_[(uint32_t)BufferType::ASCEND_L0C] =
      Tensor<uint8_t>(AscendC::TPosition::CO1, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
#else
    buffer_[(uint32_t)BufferType::ASCEND_UB] =
      Tensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
    buffer_[(uint32_t)BufferType::ASCEND_CB] =
      Tensor<uint8_t>(AscendC::TPosition::A1, 0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
    buffer_[(uint32_t)BufferType::ASCEND_L0A] =
      Tensor<uint8_t>(AscendC::TPosition::A2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
    buffer_[(uint32_t)BufferType::ASCEND_L0B] =
      Tensor<uint8_t>(AscendC::TPosition::B2, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
    buffer_[(uint32_t)BufferType::ASCEND_L0C] =
      Tensor<uint8_t>(AscendC::TPosition::CO1, 0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
#endif
  }

  template <BufferType bufferType, typename Dtype = half>
  __aicore__ inline Tensor<Dtype> GetBuffer(const uint32_t offset) const {
    return buffer_[(uint32_t)bufferType][offset].template ReinterpretCast<Dtype>();
  }

 private:
  AscendC::LocalTensor<uint8_t> buffer_[(uint32_t)BufferType::ASCEND_MAX];
};

}  // namespace PpMatMulNS
#endif
