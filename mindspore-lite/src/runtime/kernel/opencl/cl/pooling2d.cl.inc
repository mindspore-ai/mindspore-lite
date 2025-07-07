static const char *pooling2d_source ="\n" 
"#ifdef cl_khr_fp16\n" \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#endif\n" \
"#define divide_no_check(a, b) (a / b)\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void AvgPooling2d_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,\n" \
"                                     const int4 output_shape, const int2 stride, const int2 kernel_size,\n" \
"                                     const int2 padding) {\n" \
"  // axis to dst tensor coordinate\n" \
"  int X = get_global_id(2);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(0);  // C4\n" \
"  int N = X / output_shape.y;\n" \
"  X = X % output_shape.y;\n" \
"  // boundary check\n" \
"  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 r = (FLT4)(0.0f);\n" \
"  FLT window_size = 0.0f;\n" \
"  int xs = X * stride.x - padding.x;\n" \
"  int ys = Y * stride.y - padding.y;\n" \
"\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = ys + ky;\n" \
"    bool outside_y = y_c < 0 || y_c >= input_shape.z;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = xs + kx;\n" \
"      bool outside = outside_y || x_c < 0 || x_c >= input_shape.y;\n" \
"      r +=\n" \
"        !outside ? READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c)) : (FLT4)(0.0f);\n" \
"      window_size += !outside ? 1.0f : 0.0f;\n" \
"    }\n" \
"  }\n" \
"  FLT4 result = TO_FLT4(divide_no_check(r, window_size));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), result);\n" \
"}\n" \
"\n" \
"__kernel void AvgPooling2d_ReLU_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output,\n" \
"                                          const int4 input_shape, const int4 output_shape, const int2 stride,\n" \
"                                          const int2 kernel_size, const int2 padding) {\n" \
"  // axis to dst tensor coordinate\n" \
"  int X = get_global_id(2);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(0);  // C4\n" \
"  int N = X / output_shape.y;\n" \
"  X = X % output_shape.y;\n" \
"  // boundary check\n" \
"  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 r = (FLT4)(0.0f);\n" \
"  FLT window_size = 0.0f;\n" \
"  int xs = X * stride.x - padding.x;\n" \
"  int ys = Y * stride.y - padding.y;\n" \
"\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = ys + ky;\n" \
"    bool outside_y = y_c < 0 || y_c >= input_shape.z;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = xs + kx;\n" \
"      bool outside = outside_y || x_c < 0 || x_c >= input_shape.y;\n" \
"      r +=\n" \
"        !outside ? READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c)) : (FLT4)(0.0f);\n" \
"      window_size += !outside ? 1.0f : 0.0f;\n" \
"    }\n" \
"  }\n" \
"  FLT4 result = TO_FLT4(divide_no_check(r, window_size));\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), max(result, (FLT4)(0.f)));\n" \
"}\n" \
"\n" \
"__kernel void MaxPooling2d_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output, const int4 input_shape,\n" \
"                                     const int4 output_shape, const int2 stride, const int2 kernel_size,\n" \
"                                     const int2 padding) {\n" \
"  // axis to dst tensor coordinate\n" \
"  int X = get_global_id(2);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(0);  // C4\n" \
"  int N = X / output_shape.y;\n" \
"  X = X % output_shape.y;\n" \
"  // boundary check\n" \
"  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 maximum = (FLT4)(-10000.0f);\n" \
"  int xs = X * stride.x - padding.x;\n" \
"  int ys = Y * stride.y - padding.y;\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = ys + ky;\n" \
"    if (y_c < 0 || y_c >= input_shape.z) continue;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = xs + kx;\n" \
"      if (x_c < 0 || x_c >= input_shape.y) continue;\n" \
"      FLT4 src = READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c));\n" \
"      maximum = max(src, maximum);\n" \
"    }\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), maximum);\n" \
"}\n" \
"\n" \
"__kernel void MaxPooling2d_ReLU_NHWC4_IMG(__read_only image2d_t input, __write_only image2d_t output,\n" \
"                                          const int4 input_shape, const int4 output_shape, const int2 stride,\n" \
"                                          const int2 kernel_size, const int2 padding) {\n" \
"  // axis to dst tensor coordinate\n" \
"  int X = get_global_id(2);  // N*H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(0);  // C4\n" \
"  int N = X / output_shape.y;\n" \
"  X = X % output_shape.y;\n" \
"  // boundary check\n" \
"  if (N >= output_shape.x || X >= output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 maximum = (FLT4)(-10000.0f);\n" \
"  int xs = X * stride.x - padding.x;\n" \
"  int ys = Y * stride.y - padding.y;\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = ys + ky;\n" \
"    if (y_c < 0 || y_c >= input_shape.z) continue;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = xs + kx;\n" \
"      if (x_c < 0 || x_c >= input_shape.y) continue;\n" \
"      FLT4 src = READ_IMAGE(input, smp_zero, (int2)(y_c * input_shape.w + Z, N * input_shape.y + x_c));\n" \
"      maximum = max(src, maximum);\n" \
"    }\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(Y * output_shape.w + Z, N * output_shape.y + X), max(maximum, (FLT4)(0.f)));\n" \
"}\n" \
;
