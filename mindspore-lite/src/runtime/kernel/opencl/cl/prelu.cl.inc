static const char *prelu_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"#define NHWC4 2\n" \
"\n" \
"__kernel void PRelu_scalar(__read_only image2d_t input, __write_only image2d_t output, float weight, int4 shape,\n" \
"                           int data_format) {\n" \
"  int nh = get_global_id(0);\n" \
"  int w = get_global_id(1);\n" \
"  int c = get_global_id(2);\n" \
"  if (nh >= shape.x * shape.y || w >= shape.z || c >= shape.w || shape.y == 0) {\n" \
"    return;\n" \
"  }\n" \
"  int n = nh / shape.y;\n" \
"  int h = nh % shape.y;\n" \
"  int x = w * shape.w + c;\n" \
"  int y = n * shape.y + h;\n" \
"  FLT4 out = READ_IMAGE(input, smp_zero, (int2)(x, y));\n" \
"  if (out.x < 0) {\n" \
"    out.x *= weight;\n" \
"  }\n" \
"  if (out.y < 0) {\n" \
"    out.y *= weight;\n" \
"  }\n" \
"  if (out.z < 0) {\n" \
"    out.z *= weight;\n" \
"  }\n" \
"  if (out.w < 0) {\n" \
"    out.w *= weight;\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(x, y), out);\n" \
"}\n" \
"\n" \
"__kernel void PRelu_vector(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight_vector,\n" \
"                           int4 shape, int data_format) {\n" \
"  int nh = get_global_id(0);\n" \
"  int w = get_global_id(1);\n" \
"  int c = get_global_id(2);\n" \
"  if (nh >= shape.x * shape.y || w >= shape.z || c >= shape.w || shape.y == 0) {\n" \
"    return;\n" \
"  }\n" \
"  int n = nh / shape.y;\n" \
"  int h = nh % shape.y;\n" \
"  int x = w * shape.w + c;\n" \
"  int y = n * shape.y + h;\n" \
"  FLT4 weight = weight_vector[c];\n" \
"\n" \
"  FLT4 out = READ_IMAGE(input, smp_zero, (int2)(x, y));\n" \
"  if (out.x < 0) {\n" \
"    out.x *= weight.x;\n" \
"  }\n" \
"  if (out.y < 0) {\n" \
"    out.y *= weight.y;\n" \
"  }\n" \
"  if (out.z < 0) {\n" \
"    out.z *= weight.z;\n" \
"  }\n" \
"  if (out.w < 0) {\n" \
"    out.w *= weight.w;\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(x, y), out);\n" \
"}\n" \
;
