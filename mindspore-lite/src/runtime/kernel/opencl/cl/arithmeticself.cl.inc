static const char *arithmeticself_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementAbs_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                              int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = result.x >= 0 ? result.x : -result.x;\n" \
"  result.y = result.y >= 0 ? result.y : -result.y;\n" \
"  result.z = result.z >= 0 ? result.z : -result.z;\n" \
"  result.w = result.w >= 0 ? result.w : -result.w;\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementCos_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                              int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = cos(result.x);\n" \
"  result.y = cos(result.y);\n" \
"  result.z = cos(result.z);\n" \
"  result.w = cos(result.w);\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementSin_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                              int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = sin(result.x);\n" \
"  result.y = sin(result.y);\n" \
"  result.z = sin(result.z);\n" \
"  result.w = sin(result.w);\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementNeg_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                              int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = -result.x;\n" \
"  result.y = -result.y;\n" \
"  result.z = -result.z;\n" \
"  result.w = -result.w;\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementExp_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                              int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = exp(result.x);\n" \
"  result.y = exp(result.y);\n" \
"  result.z = exp(result.z);\n" \
"  result.w = exp(result.w);\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementLog_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                              int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = result.x > 0 ? log(result.x) : HUGE_VALF;\n" \
"  result.y = result.y > 0 ? log(result.y) : HUGE_VALF;\n" \
"  result.z = result.z > 0 ? log(result.z) : HUGE_VALF;\n" \
"  result.w = result.w > 0 ? log(result.w) : HUGE_VALF;\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementSquare_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                                 int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = result.x * result.x;\n" \
"  result.y = result.y * result.y;\n" \
"  result.z = result.z * result.z;\n" \
"  result.w = result.w * result.w;\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementSqrt_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                               int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = result.x > 0 ? sqrt(result.x) : HUGE_VALF;\n" \
"  result.y = result.y > 0 ? sqrt(result.y) : HUGE_VALF;\n" \
"  result.z = result.z > 0 ? sqrt(result.z) : HUGE_VALF;\n" \
"  result.w = result.w > 0 ? sqrt(result.w) : HUGE_VALF;\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementRsqrt_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                                int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = result.x > 0 ? 1.0f / sqrt(result.x) : HUGE_VALF;\n" \
"  result.y = result.y > 0 ? 1.0f / sqrt(result.y) : HUGE_VALF;\n" \
"  result.z = result.z > 0 ? 1.0f / sqrt(result.z) : HUGE_VALF;\n" \
"  result.w = result.w > 0 ? 1.0f / sqrt(result.w) : HUGE_VALF;\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementLogicalNot_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                                     int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = result.x > 0 || result.x < 0 ? false : true;\n" \
"  result.y = result.y > 0 || result.y < 0 ? false : true;\n" \
"  result.z = result.z > 0 || result.z < 0 ? false : true;\n" \
"  result.w = result.w > 0 || result.w < 0 ? false : true;\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementFloor_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                                int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = floor(result.x);\n" \
"  result.y = floor(result.y);\n" \
"  result.z = floor(result.z);\n" \
"  result.w = floor(result.w);\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementCeil_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                               int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = ceil(result.x);\n" \
"  result.y = ceil(result.y);\n" \
"  result.z = ceil(result.z);\n" \
"  result.w = ceil(result.w);\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
"\n" \
"__kernel void ArithmeticSelf_ElementRound_NHWC4(__read_only image2d_t input0, __write_only image2d_t output,\n" \
"                                                int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // c/4\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));\n" \
"  result.x = round(result.x);\n" \
"  result.y = round(result.y);\n" \
"  result.z = round(result.z);\n" \
"  result.w = round(result.w);\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);\n" \
"}\n" \
;
