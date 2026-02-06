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
#include "nnacl_c/base/resize_base.h"
#include <float.h>
#include <math.h>

int RoundPreferFloor(float x) {
  float frac = x - floorf(x);
  int base = (int)floorf(x);
  if (frac > 0.5f + FLT_EPSILON) {
    return base + 1;
  }
  return base;
}

int RoundPreferCeil(float x) {
  float frac = x - floorf(x);
  int base = (int)floorf(x);
  if (frac > 0.5f - FLT_EPSILON) {
    return base + 1;
  }
  return base;
}

int ResizeNearestModeSelect(float x_actual, bool align_corners, int nearest_mode) {
  int x_actual_nearest = 0;
  // 0:tflite resize, 1:round_prefer_floor, 2:round_prefer_ceil, 3:floor, 4:ceil
  switch (nearest_mode) {
    case 0:
      if (align_corners) {
        x_actual_nearest = (int)(roundf(x_actual));
      } else {
        x_actual_nearest = (int)(floorf(x_actual));
      }
      break;
    case 1:
      x_actual_nearest = RoundPreferFloor(x_actual);
      break;
    case 2:
      x_actual_nearest = RoundPreferCeil(x_actual);
      break;
    case 3:
      x_actual_nearest = (int)(floorf(x_actual));
      break;
    case 4:
      x_actual_nearest = (int)(ceilf(x_actual));
      break;
    default:
      x_actual_nearest = (int)(floorf(x_actual));
      break;
  }
  return x_actual_nearest;
}

int ResizeNearestModeSelectInt8(float x_actual, int nearest_mode) {
  int x_actual_nearest = 0;
  // 1:round_prefer_floor, 2:round_prefer_ceil, 3:floor, 4:ceil
  switch (nearest_mode) {
    case 1:
      x_actual_nearest = RoundPreferFloor(x_actual);
      return x_actual_nearest;
    case 2:
      x_actual_nearest = RoundPreferCeil(x_actual);
      return x_actual_nearest;
    case 3:
      x_actual_nearest = (int)(floorf(x_actual));
      return x_actual_nearest;
    case 4:
      x_actual_nearest = (int)(ceilf(x_actual));
      return x_actual_nearest;
    default:
      x_actual_nearest = (int)(floorf(x_actual));
      return x_actual_nearest;
  }
}

int ClampIndex(int x, int low_bound, int high_bound) {
  if (x < low_bound) {
    return low_bound;
  } else if (x > high_bound) {
    return high_bound;
  }
  return x;
}

float CalculateAsymmetric(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized) / (float)(length_original);
  return (float)(x_resized) / scale;
}

float CalculateAlignCorners(int x_resized, int length_original, int length_resized) {
  float scale;
  if (length_resized != 1) {
    scale = (float)(length_original - 1) / (float)(length_resized - 1);
  } else {
    scale = (float)(length_original) / (float)(length_resized);
  }
  return (float)(x_resized)*scale;
}

float CalculateHalfPixel(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized) / (float)(length_original);
  float actual = (float)(x_resized + 0.5f) / scale - 0.5f;
  return actual > 0 ? actual : 0;
}

float CalculateHalfPixelTfliteNearest(int x_resized, int length_original, int length_resized) {
  float scale = (float)(length_resized) / (float)(length_original);
  float actual = (float)(x_resized + 0.5f) / scale;
  return actual > 0 ? actual : 0;
}
