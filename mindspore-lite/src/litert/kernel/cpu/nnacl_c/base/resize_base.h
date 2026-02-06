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
#ifndef NNACL_BASE_RESIZE_BASE_H_
#define NNACL_BASE_RESIZE_BASE_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int RoundPreferFloor(float x);
int RoundPreferCeil(float x);
int ResizeNearestModeSelect(float x_actual, bool align_corners, int nearest_mode);
int ResizeNearestModeSelectInt8(float x_actual, int nearest_mode);
int ClampIndex(int x, int low_bound, int high_bound);
float CalculateAsymmetric(int x_resized, int length_original, int length_resized);
float CalculateAlignCorners(int x_resized, int length_original, int length_resized);
float CalculateHalfPixel(int x_resized, int length_original, int length_resized);
float CalculateHalfPixelTfliteNearest(int x_resized, int length_original, int length_resized);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_BASE_RESIZE_BASE_H_
