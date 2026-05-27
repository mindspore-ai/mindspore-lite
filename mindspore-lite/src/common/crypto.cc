/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/common/crypto.h"
#include <unordered_set>
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "include/securec.h"

#ifndef SECUREC_MEM_MAX_LEN
#define SECUREC_MEM_MAX_LEN 0x7fffffffUL
#endif

namespace mindspore::lite {
std::unique_ptr<Byte[]> Encrypt(size_t *, const Byte *, size_t, const Byte *, size_t, const std::string &) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return nullptr;
}

std::unique_ptr<Byte[]> Decrypt(size_t *, const std::string &, const Byte *, size_t, const std::string &) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return nullptr;
}

std::unique_ptr<Byte[]> Decrypt(size_t *, const Byte *, size_t, const Byte *, size_t, const std::string &) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return nullptr;
}
}  // namespace mindspore::lite
