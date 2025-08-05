static const char *one_hot_source ="\n" 
"#ifdef cl_khr_fp16\n" \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#endif\n" \
"\n" \
"#define C4NUM 4\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"\n" \
"#define SET_ON_OR_OFF_VALUE(RESULT, POSITION, INDICES, ON_VALUE, OFF_VALUE) \\\n" \
"  if (POSITION == INDICES) {                                                \\\n" \
"    RESULT = (float)(ON_VALUE);                                             \\\n" \
"  } else {                                                                  \\\n" \
"    RESULT = (float)(OFF_VALUE);                                            \\\n" \
"  }\n" \
"\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void OneHotAxis0(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,\n" \
"                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int in_index = (H * out_shape.z + Y) * out_shape.w + X;\n" \
"  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));\n" \
"  int *indices_int = (int *)&indices;\n" \
"  for (int i = 0; i < C4NUM; i++) {\n" \
"    if (support_neg_index != 0 && indices_int[i] < 0) {\n" \
"      indices_int[i] += depth;\n" \
"    }\n" \
"  }\n" \
"  float4 result = (float4)(0.f);\n" \
"  if (4 * X < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.x, N, indices_int[0], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 1 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.y, N, indices_int[1], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 2 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.z, N, indices_int[2], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 3 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.w, N, indices_int[3], on_value, off_value);\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void OneHotAxis1(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,\n" \
"                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int in_index = (N * out_shape.z + Y) * out_shape.w + X;\n" \
"  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));\n" \
"  int *indices_int = (int *)&indices;\n" \
"  for (int i = 0; i < C4NUM; i++) {\n" \
"    if (support_neg_index != 0 && indices_int[i] < 0) {\n" \
"      indices_int[i] += depth;\n" \
"    }\n" \
"  }\n" \
"  float4 result = (float4)(0.f);\n" \
"  if (4 * X < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.x, H, indices_int[0], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 1 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.y, H, indices_int[1], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 2 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.z, H, indices_int[2], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 3 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.w, H, indices_int[3], on_value, off_value);\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void OneHotAxis2(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,\n" \
"                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int in_index = (N * out_shape.y + H) * out_shape.w + X;\n" \
"  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(in_index % in_image2d_shape.x, in_index / in_image2d_shape.x));\n" \
"  int *indices_int = (int *)&indices;\n" \
"  for (int i = 0; i < C4NUM; i++) {\n" \
"    if (support_neg_index != 0 && indices_int[i] < 0) {\n" \
"      indices_int[i] += depth;\n" \
"    }\n" \
"  }\n" \
"  float4 result = (float4)(0.f);\n" \
"  if (4 * X < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.x, Y, indices_int[0], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 1 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.y, Y, indices_int[1], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 2 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.z, Y, indices_int[2], on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 3 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.w, Y, indices_int[3], on_value, off_value);\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void OneHotAxis3(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,\n" \
"                          int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int ci4_size = UP_DIV(out_shape.z, C4NUM);\n" \
"  int in_index_c4 = (N * out_shape.y + H) * ci4_size + Y / 4;\n" \
"  int in_index_c4_remainder = Y % 4;\n" \
"  int4 indices =\n" \
"    READ_IMAGE(src_data, smp_zero, (int2)(in_index_c4 % in_image2d_shape.x, in_index_c4 / in_image2d_shape.x));\n" \
"  int *indices_int = (int *)&indices;\n" \
"  int index_one = indices_int[in_index_c4_remainder];\n" \
"  if (support_neg_index != 0 && index_one < 0) {\n" \
"    index_one += depth;\n" \
"  }\n" \
"  float4 result = (float4)(0.f);\n" \
"  if (4 * X < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.x, 4 * X, index_one, on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 1 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.y, 4 * X + 1, index_one, on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 2 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.z, 4 * X + 2, index_one, on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 3 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.w, 4 * X + 3, index_one, on_value, off_value);\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void OneHot2DAxis3(__read_only image2d_t src_data, __write_only image2d_t dst_data, int2 in_image2d_shape,\n" \
"                            int4 out_shape, int depth, float on_value, float off_value, int C, int support_neg_index) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W (out_shape.w is 1, Y is always 0)\n" \
"  int Z = get_global_id(2);  // H * N (out_shape.h is 1, so N == Z)\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  int in_index_c4_remainder = Z % 4;\n" \
"  int4 indices = READ_IMAGE(src_data, smp_zero, (int2)(Z / C4NUM, 0));\n" \
"  int *indices_int = (int *)&indices;\n" \
"  int index_one = indices_int[in_index_c4_remainder];\n" \
"  if (support_neg_index != 0 && index_one < 0) {\n" \
"    index_one += depth;\n" \
"  }\n" \
"  float4 result = (float4)(0.f);\n" \
"  if (4 * X < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.x, 4 * X, index_one, on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 1 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.y, 4 * X + 1, index_one, on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 2 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.z, 4 * X + 2, index_one, on_value, off_value);\n" \
"  }\n" \
"  if (4 * X + 3 < C) {\n" \
"    SET_ON_OR_OFF_VALUE(result.w, 4 * X + 3, index_one, on_value, off_value);\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result);\n" \
"}\n" \
;
