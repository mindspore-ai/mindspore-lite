static const char *softmax_source ="\n" 
"#ifdef cl_khr_fp16\n" \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#endif\n" \
"#define divide_no_check(a, b) (a / b)\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void SoftmaxAxis3_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,\n" \
"                                 const int4 input_shape) {\n" \
"  int X = get_global_id(1);  // H\n" \
"  int Y = get_global_id(0);  // W\n" \
"  int n = get_global_id(2);  // N\n" \
"  int H = input_shape.y;\n" \
"  int W = input_shape.z;\n" \
"  int C4 = input_shape.w;\n" \
"\n" \
"  if (n >= input_shape.x || X >= H || Y >= W) return;\n" \
"\n" \
"  // get max\n" \
"  float4 last = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, n * H + X)));\n" \
"  float input_max = last.x;\n" \
"  if (mask.y > 0.5f) input_max = max(input_max, last.y);\n" \
"  if (mask.z > 0.5f) input_max = max(input_max, last.z);\n" \
"  if (mask.w > 0.5f) input_max = max(input_max, last.w);\n" \
"  for (int d = 0; d < C4 - 1; ++d) {\n" \
"    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, n * H + X)));\n" \
"    input_max = max(input_max, t.x);\n" \
"    input_max = max(input_max, t.y);\n" \
"    input_max = max(input_max, t.z);\n" \
"    input_max = max(input_max, t.w);\n" \
"  }\n" \
"  float4 input_max_f4 = (float4)(input_max, input_max, input_max, input_max);\n" \
"\n" \
"  float sum = 0.0f;\n" \
"  for (int d = 0; d < C4 - 1; ++d) {\n" \
"    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, n * H + X)));\n" \
"    sum += dot(exp(t - input_max_f4), (float4)(1.f));\n" \
"  }\n" \
"  float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, n * H + X)));\n" \
"  sum += dot(exp(min(t - input_max_f4, (float4)(0.f))), mask);\n" \
"  for (int d = 0; d < C4 - 1; ++d) {\n" \
"    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + d, n * H + X)));\n" \
"    result = exp(result - input_max_f4) / sum;\n" \
"    WRITE_IMAGEOUT(output, (int2)(Y * C4 + d, n * H + X), OUT_FLT4(result));\n" \
"  }\n" \
"  float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(Y * C4 + C4 - 1, n * H + X)));\n" \
"  result = exp(min(result - input_max_f4, (float4)(0.f))) / sum;\n" \
"  result = result * mask;\n" \
"  WRITE_IMAGEOUT(output, (int2)(Y * C4 + C4 - 1, n * H + X), OUT_FLT4(result));\n" \
"}\n" \
"\n" \
"__kernel void SoftmaxAxis1_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,\n" \
"                                 const int4 input_shape) {\n" \
"  int X = get_global_id(1);  // W\n" \
"  int Y = get_global_id(0);  // C4\n" \
"  int n = get_global_id(2);  // N\n" \
"  int H = input_shape.y;\n" \
"  int W = input_shape.z;\n" \
"  int C4 = input_shape.w;\n" \
"\n" \
"  if (n >= input_shape.x || X >= W || Y >= C4) return;\n" \
"\n" \
"  float4 sum = 0.0f;\n" \
"  for (int d = 0; d < H; ++d) {\n" \
"    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(X * C4 + Y, n * H + d)));\n" \
"    sum += exp(t);\n" \
"  }\n" \
"  for (int d = 0; d < H; ++d) {\n" \
"    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(X * C4 + Y, n * H + d)));\n" \
"    result = exp(result) / sum;\n" \
"    WRITE_IMAGEOUT(output, (int2)(X * C4 + Y, n * H + d), OUT_FLT4(result));\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void SoftmaxAxis2_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,\n" \
"                                 const int4 input_shape) {\n" \
"  int X = get_global_id(1);  // H\n" \
"  int Y = get_global_id(0);  // C4\n" \
"  int n = get_global_id(2);  // n\n" \
"  int H = input_shape.y;\n" \
"  int W = input_shape.z;\n" \
"  int C4 = input_shape.w;\n" \
"\n" \
"  if (n >= input_shape.x || X >= H || Y >= C4) return;\n" \
"\n" \
"  float4 sum = 0.0f;\n" \
"  for (int d = 0; d < W; ++d) {\n" \
"    float4 t = convert_float4(READ_IMAGE(input, smp_zero, (int2)(d * C4 + Y, n * H + X)));\n" \
"    sum += exp(t);\n" \
"  }\n" \
"  for (int d = 0; d < W; ++d) {\n" \
"    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(d * C4 + Y, n * H + X)));\n" \
"    result = exp(result) / sum;\n" \
"    WRITE_IMAGEOUT(output, (int2)(d * C4 + Y, n * H + X), OUT_FLT4(result));\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Softmax1x1_NHWC4(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,\n" \
"                               const int4 input_shape) {\n" \
"  int tid = get_local_id(0);\n" \
"  int n = get_global_id(1);\n" \
"  if (n >= input_shape.x) return;\n" \
"  int C4 = input_shape.w;\n" \
"  float sum = 0.0f;\n" \
"  for (size_t i = tid; i < C4 - 1; i += 32) {\n" \
"    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, n)));\n" \
"    sum += dot((float4)(1.0f), exp(src));\n" \
"  }\n" \
"  if ((C4 - 1) % 32 == tid) {\n" \
"    float4 src = convert_float4(READ_IMAGE(input, smp_zero, (int2)(C4 - 1, n)));\n" \
"    sum += dot(convert_float4(mask), exp(src));\n" \
"  }\n" \
"\n" \
"  __local float4 tmp[8];\n" \
"  __local float *tmpx1 = (__local float *)tmp;\n" \
"  tmpx1[tid] = sum;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (tid == 0) {\n" \
"    sum = dot((float4)(1.0f), tmp[0]);\n" \
"    sum += dot((float4)(1.0f), tmp[1]);\n" \
"    sum += dot((float4)(1.0f), tmp[2]);\n" \
"    sum += dot((float4)(1.0f), tmp[3]);\n" \
"    sum += dot((float4)(1.0f), tmp[4]);\n" \
"    sum += dot((float4)(1.0f), tmp[5]);\n" \
"    sum += dot((float4)(1.0f), tmp[6]);\n" \
"    sum += dot((float4)(1.0f), tmp[7]);\n" \
"    tmpx1[0] = divide_no_check(1.0f, sum);\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  sum = tmpx1[0];\n" \
"  for (size_t i = tid; i < C4; i += 32) {\n" \
"    float4 result = convert_float4(READ_IMAGE(input, smp_zero, (int2)(i, n)));\n" \
"    result = exp(result) * sum;\n" \
"    WRITE_IMAGEOUT(output, (int2)(i, n), OUT_FLT4(result));\n" \
"  }\n" \
"}\n" \
;
