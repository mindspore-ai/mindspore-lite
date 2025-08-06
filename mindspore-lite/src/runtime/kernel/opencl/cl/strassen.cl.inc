static const char *strassen_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define C4NUM 4\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void MatMul_Strassen_NHWC4_2d(__read_only image2d_t input, __write_only image2d_t output, __global FLT *weight,\n" \
"                                       int4 in_shape, int4 out_shape) {\n" \
"  int gidx = get_global_id(0);  // CO4\n" \
"  int gidz = get_global_id(2);  // N\n" \
"  int lidx = get_local_id(0);\n" \
"  int lidy = get_local_id(1);\n" \
"  int ci4 = UP_DIV(in_shape.w, C4NUM);\n" \
"  int co4 = UP_DIV(out_shape.w, C4NUM);\n" \
"  int weight_stride = in_shape.w;\n" \
"  FLT sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"  bool inside = gidx < co4 && gidz < weight_stride;\n" \
"  for (uint i = lidy; i < ci4 && inside; i += 4) {\n" \
"    FLT4 result_in = READ_IMAGE(input, smp_zero, (int2)(i, gidz));\n" \
"    int index_x = (i * C4NUM) * weight_stride + gidx * C4NUM;\n" \
"    int index_y = index_x + weight_stride;\n" \
"    int index_z = index_y + weight_stride;\n" \
"    int index_w = index_z + weight_stride;\n" \
"    for (int j = 0; j < C4NUM; ++j) {\n" \
"      FLT4 result_weight = {weight[index_x + j], weight[index_y + j], weight[index_z + j], weight[index_w + j]};\n" \
"      sum[j] += dot(result_in, result_weight);\n" \
"    }\n" \
"  }\n" \
"  FLT4 result = {sum[0], sum[1], sum[2], sum[3]};\n" \
"  __local FLT4 temp[32][4];\n" \
"  temp[lidx][lidy] = result;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (lidy == 0 && inside) {\n" \
"    result += temp[lidx][1];\n" \
"    result += temp[lidx][2];\n" \
"    result += temp[lidx][3];\n" \
"    WRITE_IMAGE(output, (int2)(gidx, gidz), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"// flag = 0 : represent add, otherwise, sub\n" \
"__kernel void MatMul_BUF_Add_Sub_2(__global FLT4 *input, __global FLT4 *output, int4 shape, int4 offset, int flag) {\n" \
"  int gidy = get_global_id(0);  // W*C4\n" \
"  int gidx = get_global_id(2);  // N*H\n" \
"  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  int ci_co_4 = shape.w;\n" \
"  const int origin_shape = 2 * ci_co_4;\n" \
"  int index_1 = (gidx + offset.x) * origin_shape + gidy + offset.y;\n" \
"  int index_2 = (gidx + offset.z) * origin_shape + gidy + offset.w;\n" \
"  FLT4 result1 = input[index_1];\n" \
"  FLT4 result2 = input[index_2];\n" \
"  FLT4 result;\n" \
"  if (flag == 0) {\n" \
"    result = result1 + result2;\n" \
"  } else {\n" \
"    result = result1 - result2;\n" \
"  }\n" \
"  output[gidx * ci_co_4 + gidy] = result;\n" \
"}\n" \
"\n" \
"__kernel void MatMul_IMG_Add_Sub_2(__read_only image2d_t input, __write_only image2d_t output, int4 shape, int4 offset,\n" \
"                                   int flag) {\n" \
"  int gidy = get_global_id(0);  // W*C4\n" \
"  int gidx = get_global_id(2);  // N*H\n" \
"  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result1 = READ_IMAGE(input, smp_zero, (int2)(gidy + offset.y, gidx + offset.x));\n" \
"  FLT4 result2 = READ_IMAGE(input, smp_zero, (int2)(gidy + offset.w, gidx + offset.z));\n" \
"  FLT4 result;\n" \
"  if (flag == 0) {\n" \
"    result = result1 + result2;\n" \
"  } else {\n" \
"    result = result1 - result2;\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(gidy, gidx), result);\n" \
"}\n" \
"\n" \
"__kernel void Strassen_Back_Result(__read_only image2d_t input1, __read_only image2d_t input2,\n" \
"                                   __read_only image2d_t input3, __read_only image2d_t input4,\n" \
"                                   __read_only image2d_t input5, __read_only image2d_t input6,\n" \
"                                   __read_only image2d_t input7, __write_only image2d_t output, int4 shape) {\n" \
"  int gidy = get_global_id(0);  // W*C4\n" \
"  int gidx = get_global_id(2);  // N*H\n" \
"  int offset_x = shape.x * shape.y, offset_y = shape.z * shape.w;\n" \
"  if (gidx >= offset_x || gidy >= offset_y) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result_M1 = READ_IMAGE(input1, smp_zero, (int2)(gidy, gidx));\n" \
"  FLT4 result_M2 = READ_IMAGE(input2, smp_zero, (int2)(gidy, gidx));\n" \
"  FLT4 result_M3 = READ_IMAGE(input3, smp_zero, (int2)(gidy, gidx));\n" \
"  FLT4 result_M4 = READ_IMAGE(input4, smp_zero, (int2)(gidy, gidx));\n" \
"  FLT4 result_M5 = READ_IMAGE(input5, smp_zero, (int2)(gidy, gidx));\n" \
"  FLT4 result_M6 = READ_IMAGE(input6, smp_zero, (int2)(gidy, gidx));\n" \
"  FLT4 result_M7 = READ_IMAGE(input7, smp_zero, (int2)(gidy, gidx));\n" \
"  FLT4 result1 = result_M4 + result_M5 + result_M6 - result_M2;  // C11\n" \
"  FLT4 result2 = result_M1 + result_M2;                          // C12\n" \
"  FLT4 result3 = result_M3 + result_M4;                          // C21\n" \
"  FLT4 result4 = result_M1 + result_M5 - result_M3 - result_M7;  // C22\n" \
"  WRITE_IMAGE(output, (int2)(gidy, gidx), result1);\n" \
"  WRITE_IMAGE(output, (int2)(gidy + offset_y, gidx), result2);\n" \
"  WRITE_IMAGE(output, (int2)(gidy, gidx + offset_x), result3);\n" \
"  WRITE_IMAGE(output, (int2)(gidy + offset_y, gidx + offset_x), result4);\n" \
"}\n" \
"\n" \
"__kernel void MatMul_IMG_Filled(__read_only image2d_t input, __write_only image2d_t output, int4 shape, int2 offset) {\n" \
"  int gidy = get_global_id(0);  // W*C4\n" \
"  int gidx = get_global_id(2);  // N*H\n" \
"  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input, smp_zero, (int2)(gidy + offset.y, gidx + offset.x));\n" \
"  WRITE_IMAGE(output, (int2)(gidy, gidx), result);\n" \
"}\n" \
"\n" \
"__kernel void MatMul_BUF_Filled(__global FLT4 *input, __global FLT4 *output, int4 shape, int2 offset) {\n" \
"  int gidy = get_global_id(0);  // W*C4\n" \
"  int gidx = get_global_id(2);  // N*H\n" \
"  if (gidx >= shape.x * shape.y || gidy >= shape.z * shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  int stride_out = shape.z * shape.w;\n" \
"  int index_out = gidx * stride_out + gidy;\n" \
"  const int stride_origin = 2 * stride_out;\n" \
"  int index_in = (gidx + offset.x) * stride_origin + gidy + offset.y;\n" \
"  FLT4 result = input[index_in];\n" \
"  output[index_out] = result;\n" \
"}\n" \
;
