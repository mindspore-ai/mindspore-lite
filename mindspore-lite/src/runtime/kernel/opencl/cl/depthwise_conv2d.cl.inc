static const char *depthwise_conv2d_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                        __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                        int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                        float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int Y = get_global_id(2);\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int batch = Y / dst_size.y;\n" \
"  int y_offset = (Y % dst_size.y) * stride.y + padding.y;\n" \
"  int fx_c = Z * kernel_size.x * kernel_size.y;\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    bool outside_y = y_c < 0 || y_c >= src_size.y;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      bool outside_x = x_c < 0 || x_c >= src_size.x;\n" \
"      if (!outside_x && !outside_y) {\n" \
"        FLT4 flt_p = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));\n" \
"        FLT4 src_p = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r += TO_FLT4(src_p * flt_p);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  FLT4 bias_p = bias[Z];\n" \
"  FLT4 res = TO_FLT4(r) + bias_p;\n" \
"  res = clamp(res, (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), res);\n" \
"}\n" \
"\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_1x1(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                            __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                            int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                            float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int Y = get_global_id(2);\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int batch = Y / dst_size.y;\n" \
"  int y_offset = (Y % dst_size.y) * stride.y + padding.y;\n" \
"  int fx_c = Z;\n" \
"  {\n" \
"    int y_c = y_offset;\n" \
"    bool outside_y = y_c < 0 || y_c >= src_size.y;\n" \
"    {\n" \
"      int x_c = x_offset;\n" \
"      bool outside_x = x_c < 0 || x_c >= src_size.x;\n" \
"      if (!outside_x && !outside_y) {\n" \
"        FLT4 flt_p = READ_IMAGE(filter, smp_zero, (int2)(0, Z));\n" \
"        FLT4 src_p = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r += TO_FLT4(src_p * flt_p);\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"  FLT4 bias_p = bias[Z];\n" \
"  FLT4 res = TO_FLT4(r) + bias_p;\n" \
"  res = clamp(res, (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), res);\n" \
"}\n" \
"__kernel void DepthwiseConv2d_FIWI_NHWC4_b222(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                              __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                              int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                              float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1) * 2;\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0) * 2;\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[8] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  bool last_c = (get_global_id(0) == (dst_size.z >> 1)) && ((dst_size.z & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.z : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.z : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));\n" \
"      FLT4 flt_p1 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z + 1));\n" \
"      {\n" \
"        FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));\n" \
"        FLT4 src_p00_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"        r[1] += TO_FLT4(src_p00_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p01_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c));\n" \
"        FLT4 src_p01_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_a1_sign + x_c_a1 * src_size.z, y_c + batch * src_size.y));\n" \
"        r[2] += TO_FLT4(src_p01_c0 * flt_p0);\n" \
"        r[3] += TO_FLT4(src_p01_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1));\n" \
"        FLT4 src_p10_c1 = READ_IMAGE(src_data, smp_zero, (int2)(Z + 1 + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"        r[4] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"        r[5] += TO_FLT4(src_p10_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p11_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1));\n" \
"        FLT4 src_p11_c1 = READ_IMAGE(src_data, smp_zero,\n" \
"                                     (int2)((Z + 1) * x_a1_sign + x_c_a1 * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"        r[6] += TO_FLT4(src_p11_c0 * flt_p0);\n" \
"        r[7] += TO_FLT4(src_p11_c1 * flt_p1);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z + 1];\n" \
"  r[2] += bias[Z];\n" \
"  r[3] += bias[Z + 1];\n" \
"  r[4] += bias[Z];\n" \
"  r[5] += bias[Z + 1];\n" \
"  r[6] += bias[Z];\n" \
"  r[7] += bias[Z + 1];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[4] = clamp(r[4], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[5] = clamp(r[5], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[6] = clamp(r[6], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[7] = clamp(r[7], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_c) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y), r[1]);\n" \
"  }\n" \
"  if (!last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[2]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z + 1, Y), r[3]);\n" \
"    }\n" \
"  }\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[4]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y + 1), r[5]);\n" \
"    }\n" \
"  }\n" \
"  if (!last_y && !last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[6]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z + 1, Y + 1), r[7]);\n" \
"    }\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_FIWI_NHWC4_b221(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                              __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                              int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                              float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1) * 2;\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.w : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.w : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));\n" \
"      FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"      r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"      FLT4 src_p01_c0 =\n" \
"        READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c + batch * src_size.y));\n" \
"      r[1] += TO_FLT4(src_p01_c0 * flt_p0);\n" \
"      FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"      r[2] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"      FLT4 src_p11_c0 =\n" \
"        READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"      r[3] += TO_FLT4(src_p11_c0 * flt_p0);\n" \
"\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z];\n" \
"  r[2] += bias[Z];\n" \
"  r[3] += bias[Z];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[1]);\n" \
"  }\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[2]);\n" \
"  }\n" \
"  if (!last_y && !last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[3]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_FIWI_NHWC4_b212(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                              __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                              int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                              float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0) * 2;\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  bool last_c = (get_global_id(0) == (dst_size.z >> 1)) && ((dst_size.z & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.z : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.z : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));\n" \
"      FLT4 flt_p1 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z + 1));\n" \
"      {\n" \
"        FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));\n" \
"        FLT4 src_p00_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"        r[1] += TO_FLT4(src_p00_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1));\n" \
"        FLT4 src_p10_c1 = READ_IMAGE(src_data, smp_zero, (int2)(Z + 1 + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"        r[2] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"        r[3] += TO_FLT4(src_p10_c1 * flt_p1);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z + 1];\n" \
"  r[2] += bias[Z];\n" \
"  r[3] += bias[Z + 1];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_c) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y), r[1]);\n" \
"  }\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[2]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y + 1), r[3]);\n" \
"    }\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_FIWI_NHWC4_b211(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                              __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                              int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                              float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[2] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.w : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.w : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));\n" \
"      FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"      r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"      FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"      r[1] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[1]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_FIWI_NHWC4_b121(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                              __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                              int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                              float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1) * 2;\n" \
"  int batch = get_global_id(2) / (dst_size.y);\n" \
"  int Y = (get_global_id(2) - dst_size.y * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.w : y_c;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));\n" \
"      FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"      r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"      FLT4 src_p01_c0 =\n" \
"        READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c + batch * src_size.y));\n" \
"      r[1] += TO_FLT4(src_p01_c0 * flt_p0);\n" \
"\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[1]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_FIWI_NHWC4_b112(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                              __read_only image2d_t filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                              int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                              float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int batch = get_global_id(2) / (dst_size.y);\n" \
"  int Y = (get_global_id(2) - dst_size.y * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0) * 2;\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_c = (get_global_id(0) == (dst_size.z >> 1)) && ((dst_size.z & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.z : y_c;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z));\n" \
"      FLT4 flt_p1 = READ_IMAGE(filter, smp_zero, (int2)(ky * kernel_size.x + kx, Z + 1));\n" \
"      {\n" \
"        FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));\n" \
"        FLT4 src_p00_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"        r[1] += TO_FLT4(src_p00_c1 * flt_p1);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z + 1];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_c) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y), r[1]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_b222(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,\n" \
"                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                             float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1) * 2;\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0) * 2;\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[8] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  bool last_c = (get_global_id(0) == (dst_size.z >> 1)) && ((dst_size.z & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.z : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.z : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = filter[fx_c];\n" \
"      FLT4 flt_p1 = filter[fx_c + f_len];\n" \
"      {\n" \
"        FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));\n" \
"        FLT4 src_p00_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"        r[1] += TO_FLT4(src_p00_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p01_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c));\n" \
"        FLT4 src_p01_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_a1_sign + x_c_a1 * src_size.z, y_c + batch * src_size.y));\n" \
"        r[2] += TO_FLT4(src_p01_c0 * flt_p0);\n" \
"        r[3] += TO_FLT4(src_p01_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1));\n" \
"        FLT4 src_p10_c1 = READ_IMAGE(src_data, smp_zero, (int2)(Z + 1 + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"        r[4] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"        r[5] += TO_FLT4(src_p10_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p11_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1));\n" \
"        FLT4 src_p11_c1 = READ_IMAGE(src_data, smp_zero,\n" \
"                                     (int2)((Z + 1) * x_a1_sign + x_c_a1 * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"        r[6] += TO_FLT4(src_p11_c0 * flt_p0);\n" \
"        r[7] += TO_FLT4(src_p11_c1 * flt_p1);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z + 1];\n" \
"  r[2] += bias[Z];\n" \
"  r[3] += bias[Z + 1];\n" \
"  r[4] += bias[Z];\n" \
"  r[5] += bias[Z + 1];\n" \
"  r[6] += bias[Z];\n" \
"  r[7] += bias[Z + 1];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[4] = clamp(r[4], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[5] = clamp(r[5], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[6] = clamp(r[6], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[7] = clamp(r[7], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_c) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y), r[1]);\n" \
"  }\n" \
"  if (!last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[2]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z + 1, Y), r[3]);\n" \
"    }\n" \
"  }\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[4]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y + 1), r[5]);\n" \
"    }\n" \
"  }\n" \
"  if (!last_y && !last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[6]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z + 1, Y + 1), r[7]);\n" \
"    }\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_b221(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,\n" \
"                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                             float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1) * 2;\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.w : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.w : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = filter[fx_c];\n" \
"      FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"      r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"      FLT4 src_p01_c0 =\n" \
"        READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c + batch * src_size.y));\n" \
"      r[1] += TO_FLT4(src_p01_c0 * flt_p0);\n" \
"      FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"      r[2] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"      FLT4 src_p11_c0 =\n" \
"        READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"      r[3] += TO_FLT4(src_p11_c0 * flt_p0);\n" \
"\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z];\n" \
"  r[2] += bias[Z];\n" \
"  r[3] += bias[Z];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[1]);\n" \
"  }\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[2]);\n" \
"  }\n" \
"  if (!last_y && !last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[3]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_b212(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,\n" \
"                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                             float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0) * 2;\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  bool last_c = (get_global_id(0) == (dst_size.z >> 1)) && ((dst_size.z & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.z : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.z : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = filter[fx_c];\n" \
"      FLT4 flt_p1 = filter[fx_c + f_len];\n" \
"      {\n" \
"        FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));\n" \
"        FLT4 src_p00_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"        r[1] += TO_FLT4(src_p00_c1 * flt_p1);\n" \
"      }\n" \
"      {\n" \
"        FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1));\n" \
"        FLT4 src_p10_c1 = READ_IMAGE(src_data, smp_zero, (int2)(Z + 1 + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"        r[2] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"        r[3] += TO_FLT4(src_p10_c1 * flt_p1);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z + 1];\n" \
"  r[2] += bias[Z];\n" \
"  r[3] += bias[Z + 1];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_c) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y), r[1]);\n" \
"  }\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[2]);\n" \
"    if (!last_c) {\n" \
"      WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y + 1), r[3]);\n" \
"    }\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_b211(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,\n" \
"                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                             float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[2] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    int y_c_a1 = y_c + stride.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.w : y_c;\n" \
"    y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.w : y_c_a1;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = filter[fx_c];\n" \
"      FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"      r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"      FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"      r[1] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[1]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_b121(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,\n" \
"                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                             float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1) * 2;\n" \
"  int batch = get_global_id(2) / (dst_size.y);\n" \
"  int Y = (get_global_id(2) - dst_size.y * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.w : y_c;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      int x_a1_sign = x_c_a1 < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = filter[fx_c];\n" \
"      FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"      r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"      FLT4 src_p01_c0 =\n" \
"        READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c + batch * src_size.y));\n" \
"      r[1] += TO_FLT4(src_p01_c0 * flt_p0);\n" \
"\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[1]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_b112(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                             __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size, int2 stride,\n" \
"                                             int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                             float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1);\n" \
"  int batch = get_global_id(2) / (dst_size.y);\n" \
"  int Y = (get_global_id(2) - dst_size.y * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0) * 2;\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_c = (get_global_id(0) == (dst_size.z >> 1)) && ((dst_size.z & 0x1) == 1);\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.z : y_c;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      int x_c_a1 = x_c + stride.x;\n" \
"      int x_sign = x_c < 0 ? -1 : 1;\n" \
"      FLT4 flt_p0 = filter[fx_c];\n" \
"      FLT4 flt_p1 = filter[fx_c + f_len];\n" \
"      {\n" \
"        FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c));\n" \
"        FLT4 src_p00_c1 =\n" \
"          READ_IMAGE(src_data, smp_zero, (int2)((Z + 1) * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"        r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"        r[1] += TO_FLT4(src_p00_c1 * flt_p1);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z + 1];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_c) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z + 1, Y), r[1]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_IMG_NHWC4_1x1_b221(__write_only image2d_t dst_data, __read_only image2d_t src_data,\n" \
"                                                 __global FLT4 *filter, __global FLT4 *bias, int2 kernel_size,\n" \
"                                                 int2 stride, int2 padding, int2 dilation, int4 src_size, int4 dst_size,\n" \
"                                                 float relu_clip_min, float relu_clip_max) {\n" \
"  int X = get_global_id(1) * 2;\n" \
"  int heightOfBlock = dst_size.y + (dst_size.y & 0x1);\n" \
"  int batch = get_global_id(2) / (heightOfBlock >> 1);\n" \
"  int Y = (get_global_id(2) * 2 - heightOfBlock * batch) + batch * dst_size.y;\n" \
"  int Z = get_global_id(0);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r[4] = {(FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f), (FLT4)(0.0f, 0.0f, 0.0f, 0.0f),\n" \
"               (FLT4)(0.0f, 0.0f, 0.0f, 0.0f)};\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int f_len = kernel_size.x * kernel_size.y;\n" \
"  int fx_c = Z * f_len;\n" \
"  bool last_x = (get_global_id(1) == (dst_size.x >> 1)) && ((dst_size.x & 0x1) == 1);\n" \
"  bool last_y =\n" \
"    ((get_global_id(2) - batch * (heightOfBlock >> 1)) == ((heightOfBlock >> 1) - 1)) && ((dst_size.y & 0x1) == 1);\n" \
"  int y_c = y_offset;\n" \
"  int y_c_a1 = y_c + stride.y;\n" \
"  int x_c = x_offset;\n" \
"  int x_c_a1 = x_c + stride.x;\n" \
"  int x_sign = x_c < 0 ? -1 : 1;\n" \
"  int x_a1_sign = x_c_a1 < 0 ? -1 : 1;\n" \
"  FLT4 flt_p0 = filter[fx_c];\n" \
"  y_c = y_c < 0 || y_c >= src_size.y ? src_size.y * src_size.z : y_c;\n" \
"  y_c_a1 = y_c_a1 < 0 || y_c_a1 >= src_size.y ? src_size.y * src_size.z : y_c_a1;\n" \
"  FLT4 src_p00_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z * x_sign + x_c * src_size.z, y_c + batch * src_size.y));\n" \
"  r[0] += TO_FLT4(src_p00_c0 * flt_p0);\n" \
"  FLT4 src_p01_c0 =\n" \
"    READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c + batch * src_size.y));\n" \
"  r[1] += TO_FLT4(src_p01_c0 * flt_p0);\n" \
"  FLT4 src_p10_c0 = READ_IMAGE(src_data, smp_zero, (int2)(Z + x_c * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"  r[2] += TO_FLT4(src_p10_c0 * flt_p0);\n" \
"  FLT4 src_p11_c0 =\n" \
"    READ_IMAGE(src_data, smp_zero, (int2)(Z * x_a1_sign + x_c_a1 * src_size.z, y_c_a1 + batch * src_size.y));\n" \
"  r[3] += TO_FLT4(src_p11_c0 * flt_p0);\n" \
"\n" \
"  r[0] += bias[Z];\n" \
"  r[1] += bias[Z];\n" \
"  r[2] += bias[Z];\n" \
"  r[3] += bias[Z];\n" \
"  r[0] = clamp(r[0], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[1] = clamp(r[1], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[2] = clamp(r[2], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  r[3] = clamp(r[3], (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y), r[0]);\n" \
"  if (!last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y), r[1]);\n" \
"  }\n" \
"  if (!last_y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, Y + 1), r[2]);\n" \
"  }\n" \
"  if (!last_y && !last_x) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((X + 1) * dst_size.z + Z, Y + 1), r[3]);\n" \
"  }\n" \
"}\n" \
"__kernel void DepthwiseConv2d_BUF_NC4HW4(__global FLT4 *dst_data, __global FLT4 *src_data, __global FLT4 *filter,\n" \
"                                         __global FLT4 *bias, int2 kernel_size, int2 stride, int2 padding,\n" \
"                                         int2 dilation, int4 src_size, int4 dst_size, float relu_clip_min,\n" \
"                                         float relu_clip_max) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  int Z = get_global_id(2);\n" \
"  if (X >= dst_size.x || Y >= (dst_size.y * dst_size.w) || Z >= dst_size.z) return;\n" \
"  FLT4 r = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  int x_offset = X * stride.x + padding.x;\n" \
"  int batch = Y / dst_size.y;\n" \
"  int y_offset = (Y - batch * dst_size.y) * stride.y + padding.y;\n" \
"  int fx_c = Z * kernel_size.x * kernel_size.y;\n" \
"  for (int ky = 0; ky < kernel_size.y; ++ky) {\n" \
"    int y_c = y_offset + ky * dilation.y;\n" \
"    bool outside_y = y_c < 0 || y_c >= src_size.y;\n" \
"    for (int kx = 0; kx < kernel_size.x; ++kx) {\n" \
"      int x_c = x_offset + kx * dilation.x;\n" \
"      bool outside_x = x_c < 0 || x_c >= src_size.x;\n" \
"      if (!outside_x && !outside_y) {\n" \
"        FLT4 flt_p = filter[fx_c];\n" \
"        FLT4 src_p = src_data[(((Z)*src_size.y + (y_c + batch * src_size.y)) * src_size.x + (x_c))];\n" \
"        r += TO_FLT4(src_p * flt_p);\n" \
"      }\n" \
"      fx_c++;\n" \
"    }\n" \
"  }\n" \
"  FLT4 bias_p = bias[Z];\n" \
"  FLT4 res = TO_FLT4(r) + bias_p;\n" \
"  res = clamp(res, (FLT)(relu_clip_min), (FLT)(relu_clip_max));\n" \
"  dst_data[(((Z)*dst_size.y + (Y)) * dst_size.x + (X))] = res;\n" \
"}\n" \
;
