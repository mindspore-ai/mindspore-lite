static const char *power_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"#define C4NUM 4\n" \
"#define CHECK_IDX                                                                                                  \\\n" \
"  int X = get_global_id(0);                                                                                        \\\n" \
"  int Y = get_global_id(1);                                                                                        \\\n" \
"  int Z = get_global_id(2);                                                                                        \\\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w || output_shape.y == 0) { \\\n" \
"    return;                                                                                                        \\\n" \
"  }\n" \
"\n" \
"FLT OptimizedPowerImpl(FLT x, int exponent) {\n" \
"  int exp = abs(exponent);\n" \
"  FLT result = 1.0f;\n" \
"  FLT iterator = x;\n" \
"  while (exp) {\n" \
"    if (exp % 2) {\n" \
"      result *= iterator;\n" \
"    }\n" \
"    iterator *= iterator;\n" \
"    exp = exp / 2;\n" \
"  }\n" \
"  return exponent >= 0 ? result : 1 / result;\n" \
"}\n" \
"\n" \
"__kernel void power(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,\n" \
"                    int4 output_shape, FLT4 parameter) {\n" \
"  CHECK_IDX;\n" \
"  int n = X / output_shape.y;\n" \
"  int h = X % output_shape.y;\n" \
"  FLT4 result;\n" \
"  FLT4 result0 = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)));\n" \
"  FLT4 result1 = READ_IMAGE(input1, smp_none, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)));\n" \
"\n" \
"  FLT tmp_result[4];\n" \
"  FLT tmp_result0[4] = {result0.x, result0.y, result0.z, result0.w};\n" \
"  FLT tmp_result1[4] = {result1.x, result1.y, result1.z, result1.w};\n" \
"\n" \
"  for (int i = 0; i < 4; ++i) {\n" \
"    tmp_result0[i] = tmp_result0[i] * parameter.z + parameter.y;\n" \
"    if (floor(tmp_result1[i]) == tmp_result1[i]) {\n" \
"      int exponent = tmp_result1[i];\n" \
"      tmp_result[i] = OptimizedPowerImpl(tmp_result0[i], exponent);\n" \
"    } else {\n" \
"      tmp_result[i] = pow(tmp_result0[i], tmp_result1[i]);\n" \
"    }\n" \
"  }\n" \
"  result.x = tmp_result[0];\n" \
"  result.y = tmp_result[1];\n" \
"  result.z = tmp_result[2];\n" \
"  result.w = tmp_result[3];\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)), result);\n" \
"}\n" \
"\n" \
"__kernel void power_broadcast(__read_only image2d_t input, __write_only image2d_t output, int4 output_shape,\n" \
"                              FLT4 parameter) {\n" \
"  CHECK_IDX;\n" \
"  int n = X / output_shape.y;\n" \
"  int h = X % output_shape.y;\n" \
"  FLT4 result;\n" \
"  FLT4 result0 = READ_IMAGE(input, smp_none, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)));\n" \
"  FLT tmp_result0[4] = {result0.x, result0.y, result0.z, result0.w};\n" \
"  FLT tmp_result[4];\n" \
"\n" \
"  bool flag = floor(parameter.x) == parameter.x ? false : true;\n" \
"  for (int i = 0; i < 4; ++i) {\n" \
"    tmp_result0[i] = tmp_result0[i] * parameter.z + parameter.y;\n" \
"    if (flag) {\n" \
"      int exponent = parameter.x;\n" \
"      tmp_result[i] = OptimizedPowerImpl(tmp_result0[i], exponent);\n" \
"    } else {\n" \
"      tmp_result[i] = pow(tmp_result0[i], parameter.x);\n" \
"    }\n" \
"  }\n" \
"  result.x = tmp_result[0];\n" \
"  result.y = tmp_result[1];\n" \
"  result.z = tmp_result[2];\n" \
"  result.w = tmp_result[3];\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (n * output_shape.y + h)), result);\n" \
"}\n" \
;
