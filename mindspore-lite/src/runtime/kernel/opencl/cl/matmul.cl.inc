static const char *matmul_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define C4NUM 4\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void MatMul_2d(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,\n" \
"                        __read_only image2d_t bias, int4 in_shape, int4 out_shape) {\n" \
"  int gidx = get_global_id(0);  // CO4\n" \
"  int gidz = get_global_id(2);  // N\n" \
"  int lidx = get_local_id(0);\n" \
"  int lidy = get_local_id(1);\n" \
"  int ci4 = UP_DIV(in_shape.w, C4NUM);\n" \
"  int co4 = UP_DIV(out_shape.w, C4NUM);\n" \
"  int n = out_shape.z;\n" \
"  bool inside = gidx < co4 && gidz < n;\n" \
"  FLT4 result = (FLT4)(0.0f);\n" \
"  for (uint i = lidy; i < ci4 && inside; i += 4) {\n" \
"    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(i, gidz));\n" \
"    FLT16 w = weight[i * co4 + gidx];\n" \
"    result.x += dot(v, w.s0123);\n" \
"    result.y += dot(v, w.s4567);\n" \
"    result.z += dot(v, w.s89ab);\n" \
"    result.w += dot(v, w.scdef);\n" \
"  }\n" \
"  __local FLT4 temp[32][4];\n" \
"  temp[lidx][lidy] = result;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (lidy == 0 && inside) {\n" \
"    result += temp[lidx][1];\n" \
"    result += temp[lidx][2];\n" \
"    result += temp[lidx][3];\n" \
"    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));\n" \
"    WRITE_IMAGE(output, (int2)(gidx, gidz), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void MatMul_4d(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,\n" \
"                        __read_only image2d_t bias, int4 in_shape, int4 out_shape) {\n" \
"  int gidx = get_global_id(0);  // CO4\n" \
"  int gidy = get_global_id(1);  // N * H * 4\n" \
"  int gidz = get_global_id(2);  // W\n" \
"  int lidx = get_local_id(0);\n" \
"  int lidy = get_local_id(1);\n" \
"  int ci4 = UP_DIV(in_shape.w, C4NUM);\n" \
"  int co4 = UP_DIV(out_shape.w, C4NUM);\n" \
"  int n = out_shape.x;\n" \
"  int h = out_shape.y;\n" \
"  int w = out_shape.z;\n" \
"  int nh_index = gidy / 4;\n" \
"  bool inside = gidx < co4 && gidz < w && nh_index < n * h;\n" \
"  FLT4 result = (FLT4)(0.0f);\n" \
"  for (uint i = lidy; i < ci4 && inside; i += 4) {\n" \
"    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(gidz * ci4 + i, nh_index));\n" \
"    FLT16 weight_value = weight[nh_index * ci4 * co4 + i * co4 + gidx];\n" \
"    result.x += dot(v, weight_value.s0123);\n" \
"    result.y += dot(v, weight_value.s4567);\n" \
"    result.z += dot(v, weight_value.s89ab);\n" \
"    result.w += dot(v, weight_value.scdef);\n" \
"  }\n" \
"  __local FLT4 temp[32][4];\n" \
"  temp[lidx][lidy] = result;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (lidy == 0 && inside) {\n" \
"    result += temp[lidx][1];\n" \
"    result += temp[lidx][2];\n" \
"    result += temp[lidx][3];\n" \
"    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));\n" \
"    WRITE_IMAGE(output, (int2)(gidz * co4 + gidx, nh_index), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void MatMulActWeightTransposeB_4d(__read_only image2d_t input, __write_only image2d_t output,\n" \
"                                           __read_only image2d_t weight, __read_only image2d_t bias, int4 in_shape,\n" \
"                                           int4 out_shape) {\n" \
"  int gidx = get_global_id(0);  // CO4\n" \
"  int gidy = get_global_id(1);  // N * H * 4\n" \
"  int gidz = get_global_id(2);  // W\n" \
"  int lidx = get_local_id(0);\n" \
"  int lidy = get_local_id(1);\n" \
"  int ci4 = UP_DIV(in_shape.w, C4NUM);\n" \
"  int co4 = UP_DIV(out_shape.w, C4NUM);\n" \
"  int n = out_shape.x;\n" \
"  int h = out_shape.y;\n" \
"  int w = out_shape.z;\n" \
"  int nh_index = gidy / 4;\n" \
"  bool inside = gidx < co4 && gidz < w && nh_index < n * h;\n" \
"  FLT4 result = (FLT4)(0.0f);\n" \
"  for (uint i = lidy; i < ci4 && inside; i += 4) {\n" \
"    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(gidz * ci4 + i, nh_index));\n" \
"    FLT4 weight_value0 = READ_IMAGE(weight, smp_zero, (int2)(gidx * 4 * ci4 + i, nh_index));\n" \
"    result.x += dot(v, weight_value0);\n" \
"    FLT4 weight_value1 = READ_IMAGE(weight, smp_zero, (int2)((gidx * 4 + 1) * ci4 + i, nh_index));\n" \
"    result.y += dot(v, weight_value1);\n" \
"    FLT4 weight_value2 = READ_IMAGE(weight, smp_zero, (int2)((gidx * 4 + 2) * ci4 + i, nh_index));\n" \
"    result.z += dot(v, weight_value2);\n" \
"    FLT4 weight_value3 = READ_IMAGE(weight, smp_zero, (int2)((gidx * 4 + 3) * ci4 + i, nh_index));\n" \
"    result.w += dot(v, weight_value3);\n" \
"  }\n" \
"  __local FLT4 temp[32][4];\n" \
"  temp[lidx][lidy] = result;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (lidy == 0 && inside) {\n" \
"    result += temp[lidx][1];\n" \
"    result += temp[lidx][2];\n" \
"    result += temp[lidx][3];\n" \
"    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));\n" \
"    WRITE_IMAGE(output, (int2)(gidz * co4 + gidx, nh_index), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void MatMulActWeight_4d(__read_only image2d_t input, __write_only image2d_t output,\n" \
"                                 __read_only image2d_t weight, __read_only image2d_t bias, int4 in_shape,\n" \
"                                 int4 out_shape) {\n" \
"  int gidx = get_global_id(0);  // CO4\n" \
"  int gidy = get_global_id(1);  // N * H * 4\n" \
"  int gidz = get_global_id(2);  // W\n" \
"  int lidx = get_local_id(0);\n" \
"  int lidy = get_local_id(1);\n" \
"  int ci4 = UP_DIV(in_shape.w, C4NUM);\n" \
"  int co4 = UP_DIV(out_shape.w, C4NUM);\n" \
"  int n = out_shape.x;\n" \
"  int h = out_shape.y;\n" \
"  int w = out_shape.z;\n" \
"  int nh_index = gidy / 4;\n" \
"  bool inside = gidx < co4 && gidz < w && nh_index < n * h;\n" \
"  FLT4 result = (FLT4)(0.0f);\n" \
"  for (uint i = lidy; i < ci4 && inside; i += 4) {\n" \
"    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(gidz * ci4 + i, nh_index));\n" \
"    FLT4 weight_value0 = READ_IMAGE(weight, smp_zero, (int2)(i * 4 * co4 + gidx, nh_index));\n" \
"    result += v.x * weight_value0;\n" \
"    FLT4 weight_value1 = READ_IMAGE(weight, smp_zero, (int2)((i * 4 + 1) * co4 + gidx, nh_index));\n" \
"    result += v.y * weight_value1;\n" \
"    FLT4 weight_value2 = READ_IMAGE(weight, smp_zero, (int2)((i * 4 + 2) * co4 + gidx, nh_index));\n" \
"    result += v.z * weight_value2;\n" \
"    FLT4 weight_value3 = READ_IMAGE(weight, smp_zero, (int2)((i * 4 + 3) * co4 + gidx, nh_index));\n" \
"    result += v.w * weight_value3;\n" \
"  }\n" \
"  __local FLT4 temp[32][4];\n" \
"  temp[lidx][lidy] = result;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (lidy == 0 && inside) {\n" \
"    result += temp[lidx][1];\n" \
"    result += temp[lidx][2];\n" \
"    result += temp[lidx][3];\n" \
"    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));\n" \
"    WRITE_IMAGE(output, (int2)(gidz * co4 + gidx, nh_index), result);\n" \
"  }\n" \
"}\n" \
;
