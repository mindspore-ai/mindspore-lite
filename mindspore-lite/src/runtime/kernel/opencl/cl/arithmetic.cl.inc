static const char *arithmetic_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define divide_no_check(a, b) (a / b)\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void ElementAdd(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,\n" \
"                         const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a + b;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementSub(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,\n" \
"                         const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a - b;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementMul(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,\n" \
"                         const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a * b;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementDiv(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,\n" \
"                         const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = divide_no_check(a, b);\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementLogicalAnd(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = AS_FLT4(AS_UINT4(a) & AS_UINT4(b));\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementLogicalOr(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                               __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = AS_FLT4(AS_UINT4(a) | AS_UINT4(b));\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementMaximum(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = max(a, b);\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementMinimum(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = min(a, b);\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementFloorDiv(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                              __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = floor(divide_no_check(a, b));\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementFloorMod(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                              __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a - floor(divide_no_check(a, b)) * b;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementSquaredDifference(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                       __write_only image2d_t output, const int2 output_shape, float act_min,\n" \
"                                       float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = pown((a - b), (int4)2);\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementEqual(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,\n" \
"                           const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a == b ? (FLT4)1.f : (FLT4).0f;\n" \
"  // error?\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementNotEqual(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                              __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a != b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementLess(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t output,\n" \
"                          const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a < b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementLessEqual(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                               __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a <= b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementGreater(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a > b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void ElementGreaterEqual(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                  __write_only image2d_t output, const int2 output_shape, float act_min,\n" \
"                                  float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a >= b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastNHWC4Add(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,\n" \
"                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // N * H\n" \
"  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {\n" \
"    return;\n" \
"  }\n" \
"  int H = Z % output_shape.y;\n" \
"  int N = Z / output_shape.y;\n" \
"  int a_c = X < a_shape.w ? X : 0;\n" \
"  int a_w = Y < a_shape.z ? Y : 0;\n" \
"  int a_h = H < a_shape.y ? H : 0;\n" \
"  int a_n = N < a_shape.x ? N : 0;\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));\n" \
"  int b_c = X < b_shape.w ? X : 0;\n" \
"  int b_w = Y < b_shape.z ? Y : 0;\n" \
"  int b_h = H < b_shape.y ? H : 0;\n" \
"  int b_n = N < b_shape.x ? N : 0;\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));\n" \
"  FLT4 result;\n" \
"  if (broadcastC_flag == 0) {\n" \
"    result = a + b;\n" \
"  } else if (broadcastC_flag == 1) {\n" \
"    result = a.x + b;\n" \
"  } else {\n" \
"    result = a + b.x;\n" \
"  }\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastNHWC4BiasAdd(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                    __write_only image2d_t output, const int4 a_shape, const int4 b_shape,\n" \
"                                    const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // N * H\n" \
"  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {\n" \
"    return;\n" \
"  }\n" \
"  int H = Z % output_shape.y;\n" \
"  int N = Z / output_shape.y;\n" \
"  int a_c = X < a_shape.w ? X : 0;\n" \
"  int a_w = Y < a_shape.z ? Y : 0;\n" \
"  int a_h = H < a_shape.y ? H : 0;\n" \
"  int a_n = N < a_shape.x ? N : 0;\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));\n" \
"  int b_c = X < b_shape.w ? X : 0;\n" \
"  int b_w = Y < b_shape.z ? Y : 0;\n" \
"  int b_h = H < b_shape.y ? H : 0;\n" \
"  int b_n = N < b_shape.x ? N : 0;\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));\n" \
"  FLT4 result;\n" \
"  if (broadcastC_flag == 0) {\n" \
"    result = a + b;\n" \
"  } else if (broadcastC_flag == 1) {\n" \
"    result = a.x + b;\n" \
"  } else {\n" \
"    result = a + b.x;\n" \
"  }\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastNHWC4Sub(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,\n" \
"                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // N * H\n" \
"  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {\n" \
"    return;\n" \
"  }\n" \
"  int H = Z % output_shape.y;\n" \
"  int N = Z / output_shape.y;\n" \
"  int a_c = X < a_shape.w ? X : 0;\n" \
"  int a_w = Y < a_shape.z ? Y : 0;\n" \
"  int a_h = H < a_shape.y ? H : 0;\n" \
"  int a_n = N < a_shape.x ? N : 0;\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));\n" \
"  int b_c = X < b_shape.w ? X : 0;\n" \
"  int b_w = Y < b_shape.z ? Y : 0;\n" \
"  int b_h = H < b_shape.y ? H : 0;\n" \
"  int b_n = N < b_shape.x ? N : 0;\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));\n" \
"  FLT4 result;\n" \
"  if (broadcastC_flag == 0) {\n" \
"    result = a - b;\n" \
"  } else if (broadcastC_flag == 1) {\n" \
"    result = a.x - b;\n" \
"  } else {\n" \
"    result = a - b.x;\n" \
"  }\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastNHWC4Mul(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,\n" \
"                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // N * H\n" \
"  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {\n" \
"    return;\n" \
"  }\n" \
"  int H = Z % output_shape.y;\n" \
"  int N = Z / output_shape.y;\n" \
"  int a_c = X < a_shape.w ? X : 0;\n" \
"  int a_w = Y < a_shape.z ? Y : 0;\n" \
"  int a_h = H < a_shape.y ? H : 0;\n" \
"  int a_n = N < a_shape.x ? N : 0;\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));\n" \
"  int b_c = X < b_shape.w ? X : 0;\n" \
"  int b_w = Y < b_shape.z ? Y : 0;\n" \
"  int b_h = H < b_shape.y ? H : 0;\n" \
"  int b_n = N < b_shape.x ? N : 0;\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));\n" \
"  FLT4 result;\n" \
"  if (broadcastC_flag == 0) {\n" \
"    result = a * b;\n" \
"  } else if (broadcastC_flag == 1) {\n" \
"    result = a.x * b;\n" \
"  } else {\n" \
"    result = a * b.x;\n" \
"  }\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastNHWC4Div(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                __write_only image2d_t output, const int4 a_shape, const int4 b_shape,\n" \
"                                const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // N * H\n" \
"  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {\n" \
"    return;\n" \
"  }\n" \
"  int H = Z % output_shape.y;\n" \
"  int N = Z / output_shape.y;\n" \
"  int a_c = X < a_shape.w ? X : 0;\n" \
"  int a_w = Y < a_shape.z ? Y : 0;\n" \
"  int a_h = H < a_shape.y ? H : 0;\n" \
"  int a_n = N < a_shape.x ? N : 0;\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));\n" \
"  int b_c = X < b_shape.w ? X : 0;\n" \
"  int b_w = Y < b_shape.z ? Y : 0;\n" \
"  int b_h = H < b_shape.y ? H : 0;\n" \
"  int b_n = N < b_shape.x ? N : 0;\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));\n" \
"  FLT4 result;\n" \
"  if (broadcastC_flag == 0) {\n" \
"    result = a / b;\n" \
"  } else if (broadcastC_flag == 1) {\n" \
"    result = a.x / b;\n" \
"  } else {\n" \
"    result = a / b.x;\n" \
"  }\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastLogicalAnd(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                                  const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = AS_FLT4(AS_UINT4(a) & (UINT4)((FLT)b));\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastLogicalOr(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                                 const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = AS_FLT4(AS_UINT4(a) | (UINT4)((FLT)b));\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastMaximum(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                               const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = max(a, (FLT4)b);\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastMinimum(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                               const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = min(a, (FLT4)b);\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastFloorDiv(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                                const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = floor(divide_no_check(a, (FLT4)b));\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"__kernel void BroadcastNHWC4FloorMod(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                     __write_only image2d_t output, const int4 a_shape, const int4 b_shape,\n" \
"                                     const int4 output_shape, const int broadcastC_flag, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H\n" \
"  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(Y * a_shape.w + X, Z));\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, 0));\n" \
"  FLT4 result = a - floor(divide_no_check(a, b)) * b;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastNHWC4SquaredDifference(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                                              __write_only image2d_t output, const int4 a_shape, const int4 b_shape,\n" \
"                                              const int4 output_shape, const int broadcastC_flag, float act_min,\n" \
"                                              float act_max) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // w\n" \
"  int Z = get_global_id(2);  // H\n" \
"\n" \
"  if (X >= output_shape.w || Y >= output_shape.z || Z >= output_shape.y * output_shape.x) {\n" \
"    return;\n" \
"  }\n" \
"  int H = Z % output_shape.y;\n" \
"  int N = Z / output_shape.y;\n" \
"  int a_c = X < a_shape.w ? X : 0;\n" \
"  int a_w = Y < a_shape.z ? Y : 0;\n" \
"  int a_h = H < a_shape.y ? H : 0;\n" \
"  int a_n = N < a_shape.x ? N : 0;\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(a_w * a_shape.w + a_c, a_n * a_shape.y + a_h));\n" \
"  int b_c = X < b_shape.w ? X : 0;\n" \
"  int b_w = Y < b_shape.z ? Y : 0;\n" \
"  int b_h = H < b_shape.y ? H : 0;\n" \
"  int b_n = N < b_shape.x ? N : 0;\n" \
"  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(b_w * b_shape.w + b_c, b_n * b_shape.y + b_h));\n" \
"  FLT4 result;\n" \
"  if (broadcastC_flag == 0) {\n" \
"    result = pown((a - b), (int4)2);\n" \
"  } else if (broadcastC_flag == 1) {\n" \
"    result = pown((a.x - b), (int4)2);\n" \
"  } else {\n" \
"    result = pown((a - b.x), (int4)2);\n" \
"  }\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + X, Z), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                             const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a == (FLT4)b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastNotEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                                const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a != (FLT4)b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastLess(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                            const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a < (FLT4)b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastLessEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                                 const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a <= (FLT4)b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastGreater(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                               const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a > (FLT4)b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void BroadcastGreaterEqual(__read_only image2d_t input_a, float b, __write_only image2d_t output,\n" \
"                                    const int2 output_shape, float act_min, float act_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n" \
"  FLT4 result = a >= (FLT4)b ? (FLT4)1.f : (FLT4).0f;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
;
