/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_LOG_ADAPTER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_LOG_ADAPTER_H_

#if !defined(ENABLE_ANDROID) || defined(ENABLE_MINDDATA_PYTHON)
#include "src/common/log_adapter.h"
#define DATASET_SRC_FILE_NAME FILE_NAME
#else
#include "mindspore-lite/src/common/log_adapter.h"
#define DATASET_SRC_FILE_NAME LITE_FILE_NAME
#endif

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_LOG_ADAPTER_H_
