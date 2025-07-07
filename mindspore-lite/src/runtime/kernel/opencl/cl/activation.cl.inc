static const char *activation_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void LeakyRelu(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape,\n" \
"                        const float alpha) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= img_shape.x || Y >= img_shape.y) return;\n" \
"  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  FLT4 tmp;\n" \
"  FLT alpha_f = TO_FLT(alpha);\n" \
"  tmp.x = in_c4.x > 0.0f ? in_c4.x : in_c4.x * alpha_f;\n" \
"  tmp.y = in_c4.y > 0.0f ? in_c4.y : in_c4.y * alpha_f;\n" \
"  tmp.z = in_c4.z > 0.0f ? in_c4.z : in_c4.z * alpha_f;\n" \
"  tmp.w = in_c4.w > 0.0f ? in_c4.w : in_c4.w * alpha_f;\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), tmp);\n" \
"}\n" \
"\n" \
"__kernel void Relu(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= img_shape.x || Y >= img_shape.y) return;\n" \
"  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  in_c4 = max(in_c4, (FLT)(0.f));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), in_c4);\n" \
"}\n" \
"\n" \
"__kernel void Relu6(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= img_shape.x || Y >= img_shape.y) return;\n" \
"  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  in_c4 = clamp(in_c4, (FLT)(0.f), (FLT)(6.f));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), in_c4);\n" \
"}\n" \
"\n" \
"__kernel void Sigmoid(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape, const int c4,\n" \
"                      const int last_c4) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= img_shape.x || Y >= img_shape.y || c4 == 0) return;\n" \
"  int C4 = X % c4;\n" \
"  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  if (C4 < c4 - 1) {\n" \
"    in_c4 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-in_c4));\n" \
"  } else {\n" \
"    in_c4.x = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.x));\n" \
"    if (last_c4 > 1) {\n" \
"      in_c4.y = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.y));\n" \
"    }\n" \
"    if (last_c4 > 2) {\n" \
"      in_c4.z = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.z));\n" \
"    }\n" \
"    if (last_c4 > 3) {\n" \
"      in_c4.w = (FLT)(1.f) / ((FLT)(1.f) + exp(-in_c4.w));\n" \
"    }\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), in_c4);\n" \
"}\n" \
"\n" \
"__kernel void Tanh(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= img_shape.x || Y >= img_shape.y) return;\n" \
"  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  FLT4 exp0 = exp(in_c4);\n" \
"  FLT4 exp1 = exp(-in_c4);\n" \
"  in_c4 = (exp0 - exp1) / (exp0 + exp1);\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), in_c4);\n" \
"}\n" \
"\n" \
"__kernel void Swish(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= img_shape.x || Y >= img_shape.y) return;\n" \
"  FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  in_c4 = in_c4 * ((FLT4)(1.f) / ((FLT4)(1.f) + exp(-in_c4)));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), in_c4);\n" \
"}\n" \
"\n" \
"__kernel void HSwish(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {\n" \
"  int X = get_global_id(0);  // w*c\n" \
"  int Y = get_global_id(1);  // n*h\n" \
"  if (X >= img_shape.x || Y >= img_shape.y) return;\n" \
"  FLT4 temp = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  FLT4 result = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  result.x = temp.x * (temp.x <= -3 ? 0 : (temp.x >= 3 ? 1 : temp.x / 6 + 0.5f));\n" \
"  result.y = temp.y * (temp.y <= -3 ? 0 : (temp.y >= 3 ? 1 : temp.y / 6 + 0.5f));\n" \
"  result.z = temp.z * (temp.z <= -3 ? 0 : (temp.z >= 3 ? 1 : temp.z / 6 + 0.5f));\n" \
"  result.w = temp.w * (temp.w <= -3 ? 0 : (temp.w >= 3 ? 1 : temp.w / 6 + 0.5f));\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
"\n" \
"__kernel void HSigmoid(__read_only image2d_t input, __write_only image2d_t output, const int2 img_shape) {\n" \
"  int X = get_global_id(0);  // w*c\n" \
"  int Y = get_global_id(1);  // n*h\n" \
"  if (X >= img_shape.x || Y >= img_shape.y) return;\n" \
"  FLT4 temp = READ_IMAGE(input, smp_zero, (int2)(X, Y));\n" \
"  FLT4 result = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  result.x = temp.x <= -3 ? 0 : (temp.x >= 3 ? 1 : temp.x / 6 + 0.5f);\n" \
"  result.y = temp.y <= -3 ? 0 : (temp.y >= 3 ? 1 : temp.y / 6 + 0.5f);\n" \
"  result.z = temp.z <= -3 ? 0 : (temp.z >= 3 ? 1 : temp.z / 6 + 0.5f);\n" \
"  result.w = temp.w <= -3 ? 0 : (temp.w >= 3 ? 1 : temp.w / 6 + 0.5f);\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), result);\n" \
"}\n" \
;
