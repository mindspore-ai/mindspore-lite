static const char *space_to_depth_source ="\n" 
"#ifdef cl_khr_fp16\n" \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#endif\n" \
"\n" \
"#define C4NUM 4\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void SpaceToDepth(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,\n" \
"                           int4 out_shape, int block_size, int ci_size) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  if (out_shape.y == 0 || ci_size == 0 || block_size == 0) return;\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int co_base = X * C4NUM;\n" \
"  FLT result[C4NUM] = {0.f};\n" \
"  for (int i = 0; i < C4NUM; i++) {\n" \
"    int co = co_base + i;\n" \
"    int ci = co % ci_size;\n" \
"    int hw_block = co / ci_size;\n" \
"    int hi = H * block_size + hw_block / block_size;\n" \
"    int wi = Y * block_size + hw_block % block_size;\n" \
"    int ci4 = ci / C4NUM;\n" \
"    int ci4_ramainder = ci % C4NUM;\n" \
"    FLT4 tmp = READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci4, N * in_shape.y + hi));\n" \
"    if (ci4_ramainder == 0) {\n" \
"      result[i] = tmp.x;\n" \
"    } else if (ci4_ramainder == 1) {\n" \
"      result[i] = tmp.y;\n" \
"    } else if (ci4_ramainder == 2) {\n" \
"      result[i] = tmp.z;\n" \
"    } else {\n" \
"      result[i] = tmp.w;\n" \
"    }\n" \
"  }\n" \
"  FLT4 result_flt4 = {result[0], result[1], result[2], result[3]};\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result_flt4);\n" \
"}\n" \
"\n" \
"__kernel void SpaceToDepthAlign(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,\n" \
"                                int4 out_shape, int block_size, int ci_size) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  if (out_shape.y == 0 || in_shape.w == 0 || block_size == 0) return;\n" \
"\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int ni = N;\n" \
"  int ci = X % in_shape.w;\n" \
"  int hw_block = X / in_shape.w;\n" \
"  int hi = H * block_size + hw_block / block_size;\n" \
"  int wi = Y * block_size + hw_block % block_size;\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z),\n" \
"              READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci, ni * in_shape.y + hi)));\n" \
"}\n" \
"\n" \
"__kernel void DepthToSpace(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,\n" \
"                           int4 out_shape, int block_size, int co_size) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  if (out_shape.y == 0 || block_size == 0) return;\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int co_base = X * C4NUM;\n" \
"  FLT result[C4NUM] = {0.f};\n" \
"  for (int i = 0; i < C4NUM; i++) {\n" \
"    int co = co_base + i;\n" \
"    int bh = H % block_size;\n" \
"    int hi = H / block_size;\n" \
"    int bw = Y % block_size;\n" \
"    int wi = Y / block_size;\n" \
"    int ci = (bh * block_size + bw) * co_size + co;\n" \
"    int ci4 = ci / C4NUM;\n" \
"    int ci4_ramainder = ci % C4NUM;\n" \
"    FLT4 tmp = READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci4, N * in_shape.y + hi));\n" \
"    if (ci4_ramainder == 0) {\n" \
"      result[i] = tmp.x;\n" \
"    } else if (ci4_ramainder == 1) {\n" \
"      result[i] = tmp.y;\n" \
"    } else if (ci4_ramainder == 2) {\n" \
"      result[i] = tmp.z;\n" \
"    } else {\n" \
"      result[i] = tmp.w;\n" \
"    }\n" \
"  }\n" \
"  FLT4 result_flt4 = {result[0], result[1], result[2], result[3]};\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z), result_flt4);\n" \
"}\n" \
"\n" \
"__kernel void DepthToSpaceAlign(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 in_shape,\n" \
"                                int4 out_shape, int block_size, int co_size) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // H * N\n" \
"  if (X >= out_shape.w || Y >= out_shape.z || Z >= out_shape.x * out_shape.y) return;\n" \
"  if (out_shape.y == 0 || block_size == 0) return;\n" \
"  int N = Z / out_shape.y;\n" \
"  int H = Z % out_shape.y;\n" \
"  int ni = N;\n" \
"  int bh = H % block_size;\n" \
"  int hi = H / block_size;\n" \
"  int bw = Y % block_size;\n" \
"  int wi = Y / block_size;\n" \
"  int ci = (bh * block_size + bw) * out_shape.w + X;\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * out_shape.w + X, Z),\n" \
"              READ_IMAGE(src_data, smp_zero, (int2)(wi * in_shape.w + ci, ni * in_shape.y + hi)));\n" \
"}\n" \
;
