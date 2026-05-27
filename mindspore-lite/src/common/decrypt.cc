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

#include "src/common/decrypt.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"

#ifndef SECUREC_MEM_MAX_LEN
#define SECUREC_MEM_MAX_LEN 0x7fffffffUL
#endif

namespace mindspore::lite {
std::unique_ptr<Byte[]> Decrypt(const std::string &lib_path, size_t *, const Byte *, const size_t, const Byte *,
                                const size_t, const std::string &) {
  MS_LOG(ERROR) << "The feature is only supported on the Linux platform "
                   "when the OPENSSL compilation option is enabled.";
  return nullptr;
}
}  // namespace mindspore::lite
