static const char *conv2d_transpose_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void conv2d_transpose(__read_only image2d_t src_data, __write_only image2d_t dst_data, __global FLT16 *weight,\n" \
"                               __read_only image2d_t biases, int2 kernel_size, int2 stride, int2 padding, int4 src_size,\n" \
"                               int4 dst_size, int act_type) {\n" \
"  int dst_h = get_global_id(0);\n" \
"  int rem_h = dst_h % stride.x;\n" \
"  int ceil_h = dst_h / stride.x;\n" \
"  dst_h = ceil_h * stride.x * 2 + rem_h;\n" \
"  int dst_w = get_global_id(1);\n" \
"  int rem_w = dst_w % stride.y;\n" \
"  int ceil_w = dst_w / stride.y;\n" \
"  dst_w = ceil_w * stride.y * 2 + rem_w;\n" \
"  int dst_c = get_global_id(2);  // n * c4\n" \
"  int n = dst_c / dst_size.z;\n" \
"  dst_c = dst_c % dst_size.z;\n" \
"  if (dst_h >= dst_size.x || dst_w >= dst_size.y || dst_c >= dst_size.z || n >= dst_size.w) return;\n" \
"  int weight_base = dst_c * src_size.z * kernel_size.x * kernel_size.y;\n" \
"  FLT4 r0 = (FLT4)(0.f);\n" \
"  FLT4 r1 = (FLT4)(0.f);\n" \
"  FLT4 r2 = (FLT4)(0.f);\n" \
"  FLT4 r3 = (FLT4)(0.f);\n" \
"  int kh_start = dst_h + padding.x;\n" \
"  int kw_start = dst_w + padding.y;\n" \
"  int kh_end = kh_start - kernel_size.x;\n" \
"  int kw_end = kw_start - kernel_size.y;\n" \
"  int src_h = kh_start / stride.x;\n" \
"  int kh = src_h * stride.x;\n" \
"  int src_w = kw_start / stride.y;\n" \
"  int kw = src_w * stride.y;\n" \
"  for (; kh > kh_end; src_h -= 1, kh -= stride.x) {\n" \
"    int out0_src_h = src_h;\n" \
"    int out1_src_h = src_h + 1;\n" \
"    int kernel_h = kh_start - kh;\n" \
"    int src_w_copy = src_w;\n" \
"    int kw_copy = kw;\n" \
"    for (; kw_copy > kw_end; src_w_copy -= 1, kw_copy -= stride.y) {\n" \
"      int out0_src_w = src_w_copy;\n" \
"      int out1_src_w = src_w_copy + 1;\n" \
"      int kernel_w = kw_start - kw_copy;\n" \
"      int weight_offset = weight_base + (kernel_h * kernel_size.y + kernel_w) * src_size.z;\n" \
"      for (int ci = 0; ci < src_size.z; ++ci) {\n" \
"        FLT4 x0 = (FLT4)0.f;\n" \
"        FLT4 x2 = (FLT4)0.f;\n" \
"        if (out0_src_h < src_size.x && out0_src_h >= 0) {\n" \
"          x0 = READ_IMAGE(src_data, smp_zero, (int2)(out0_src_w * src_size.z + ci, n * src_size.x + out0_src_h));\n" \
"          x2 = READ_IMAGE(src_data, smp_zero, (int2)(out1_src_w * src_size.z + ci, n * src_size.x + out0_src_h));\n" \
"        }\n" \
"        FLT4 x1 = (FLT4)0.f;\n" \
"        FLT4 x3 = (FLT4)0.f;\n" \
"        if (out1_src_h < src_size.x && out1_src_h >= 0) {\n" \
"          x1 = READ_IMAGE(src_data, smp_zero, (int2)(out0_src_w * src_size.z + ci, n * src_size.x + out1_src_h));\n" \
"          x3 = READ_IMAGE(src_data, smp_zero, (int2)(out1_src_w * src_size.z + ci, n * src_size.x + out1_src_h));\n" \
"        }\n" \
"        FLT16 weight_cache = weight[weight_offset++];\n" \
"        r0 += x0.x * weight_cache.s0123;\n" \
"        r0 += x0.y * weight_cache.s4567;\n" \
"        r0 += x0.z * weight_cache.s89ab;\n" \
"        r0 += x0.w * weight_cache.scdef;\n" \
"\n" \
"        r1 += x1.x * weight_cache.s0123;\n" \
"        r1 += x1.y * weight_cache.s4567;\n" \
"        r1 += x1.z * weight_cache.s89ab;\n" \
"        r1 += x1.w * weight_cache.scdef;\n" \
"\n" \
"        r2 += x2.x * weight_cache.s0123;\n" \
"        r2 += x2.y * weight_cache.s4567;\n" \
"        r2 += x2.z * weight_cache.s89ab;\n" \
"        r2 += x2.w * weight_cache.scdef;\n" \
"\n" \
"        r3 += x3.x * weight_cache.s0123;\n" \
"        r3 += x3.y * weight_cache.s4567;\n" \
"        r3 += x3.z * weight_cache.s89ab;\n" \
"        r3 += x3.w * weight_cache.scdef;\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"  FLT4 bias_val = READ_IMAGE(biases, smp_zero, (int2)(dst_c, 0));\n" \
"  r0 += bias_val;\n" \
"  r1 += bias_val;\n" \
"  r2 += bias_val;\n" \
"  r3 += bias_val;\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    r0 = max(r0, (FLT4)(0.0f));\n" \
"    r1 = max(r1, (FLT4)(0.0f));\n" \
"    r2 = max(r2, (FLT4)(0.0f));\n" \
"    r3 = max(r3, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    r0 = clamp(r0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    r1 = clamp(r1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    r2 = clamp(r2, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    r3 = clamp(r3, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  }\n" \
"\n" \
"  WRITE_IMAGE(dst_data, (int2)(dst_w * dst_size.z + dst_c, n * dst_size.x + dst_h), r0);\n" \
"  if (dst_h + stride.x < dst_size.x && dst_w < dst_size.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(dst_w * dst_size.z + dst_c, n * dst_size.x + dst_h + stride.x), r1);\n" \
"  }\n" \
"  if (dst_h < dst_size.x && dst_w + stride.y < dst_size.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((dst_w + stride.y) * dst_size.z + dst_c, n * dst_size.x + dst_h), r2);\n" \
"  }\n" \
"  if (dst_h + stride.x < dst_size.x && dst_w + stride.y < dst_size.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((dst_w + stride.y) * dst_size.z + dst_c, n * dst_size.x + dst_h + stride.x), r3);\n" \
"  }\n" \
"}\n" \
;
