static const char *transpose_source ="\n" 
"#ifdef cl_khr_fp16\n" \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#endif\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void transpose_0312_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {\n" \
"  int X = get_global_id(0);  // H4, C4 for src\n" \
"  int Y = get_global_id(1);  // W, H for src\n" \
"  int Z = get_global_id(2);  // C4, W4 for src\n" \
"  if (4 * X >= shape.y || Y >= shape.z || 4 * Z >= shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  int H4 = UP_DIV(shape.y, 4);\n" \
"  int C4 = UP_DIV(shape.w, 4);\n" \
"  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z * H4 + X, Y));\n" \
"  FLT4 src1 = (FLT4)0.f;\n" \
"  if (4 * Z + 1 < shape.w) {\n" \
"    src1 = READ_IMAGE(src_data, smp_zero, (int2)((4 * Z + 1) * H4 + X, Y));\n" \
"  }\n" \
"  FLT4 src2 = (FLT4)0.f;\n" \
"  if (4 * Z + 2 < shape.w) {\n" \
"    src2 = READ_IMAGE(src_data, smp_zero, (int2)((4 * Z + 2) * H4 + X, Y));\n" \
"  }\n" \
"  FLT4 src3 = (FLT4)0.f;\n" \
"  if (4 * Z + 3 < shape.w) {\n" \
"    src3 = READ_IMAGE(src_data, smp_zero, (int2)((4 * Z + 3) * H4 + X, Y));\n" \
"  }\n" \
"  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);\n" \
"  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);\n" \
"  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);\n" \
"  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X), dst0);\n" \
"  if (4 * X + 1 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 1), dst1);\n" \
"  }\n" \
"  if (4 * X + 2 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 2), dst2);\n" \
"  }\n" \
"  if (4 * X + 3 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 3), dst3);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void transpose_0312_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {\n" \
"  int X = get_global_id(0);  // H4, C4 for src\n" \
"  int Y = get_global_id(1);  // W, H for src\n" \
"  int Z = get_global_id(2);  // C4, W4 for src\n" \
"  if (4 * X >= shape.y || Y >= shape.z || 4 * Z >= shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z, X * shape.z + Y));\n" \
"  FLT4 src1 = (FLT4)0.f;\n" \
"  if (4 * Z + 1 < shape.w) {\n" \
"    src1 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z + 1, X * shape.z + Y));\n" \
"  }\n" \
"  FLT4 src2 = (FLT4)0.f;\n" \
"  if (4 * Z + 2 < shape.w) {\n" \
"    src2 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z + 2, X * shape.z + Y));\n" \
"  }\n" \
"  FLT4 src3 = (FLT4)0.f;\n" \
"  if (4 * Z + 3 < shape.w) {\n" \
"    src3 = READ_IMAGE(src_data, smp_zero, (int2)(4 * Z + 3, X * shape.z + Y));\n" \
"  }\n" \
"  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);\n" \
"  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);\n" \
"  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);\n" \
"  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X), dst0);\n" \
"  if (4 * X + 1 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X + 1), dst1);\n" \
"  }\n" \
"  if (4 * X + 2 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X + 2), dst2);\n" \
"  }\n" \
"  if (4 * X + 3 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y, Z * shape.y + 4 * X + 3), dst3);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void transpose_0312_oversize_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data,\n" \
"                                            int4 shape) {\n" \
"  int X = get_global_id(0);  // H4, C4 for src\n" \
"  int Y = get_global_id(1);  // W, H for src\n" \
"  int Z = get_global_id(2);  // C4, W4 for src\n" \
"  if (4 * X >= shape.y || Y >= shape.z || 4 * Z >= shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  int H4 = UP_DIV(shape.y, 4);\n" \
"  int C4 = UP_DIV(shape.w, 4);\n" \
"  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z));\n" \
"  FLT4 src1 = (FLT4)0.f;\n" \
"  if (4 * Z + 1 < shape.w) {\n" \
"    src1 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 1));\n" \
"  }\n" \
"  FLT4 src2 = (FLT4)0.f;\n" \
"  if (4 * Z + 2 < shape.w) {\n" \
"    src2 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 2));\n" \
"  }\n" \
"  FLT4 src3 = (FLT4)0.f;\n" \
"  if (4 * Z + 3 < shape.w) {\n" \
"    src3 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 3));\n" \
"  }\n" \
"  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);\n" \
"  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);\n" \
"  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);\n" \
"  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X), dst0);\n" \
"  if (4 * X + 1 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 1), dst1);\n" \
"  }\n" \
"  if (4 * X + 2 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 2), dst2);\n" \
"  }\n" \
"  if (4 * X + 3 < shape.y) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(Y * C4 + Z, 4 * X + 3), dst3);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void transpose_0231_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {\n" \
"  int X = get_global_id(0);  // H, W for src\n" \
"  int Y = get_global_id(1);  // W4, C4 for src\n" \
"  int Z = get_global_id(2);  // C4, H4 for src\n" \
"  if (X >= shape.y || 4 * Y >= shape.z || 4 * Z >= shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  int W4 = UP_DIV(shape.z, 4);\n" \
"  int C4 = UP_DIV(shape.w, 4);\n" \
"  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z));\n" \
"  FLT4 src1 = (FLT4)0.f;\n" \
"  if (4 * Z + 1 < shape.w) {\n" \
"    src1 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z + 1));\n" \
"  }\n" \
"  FLT4 src2 = (FLT4)0.f;\n" \
"  if (4 * Z + 2 < shape.w) {\n" \
"    src2 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z + 2));\n" \
"  }\n" \
"  FLT4 src3 = (FLT4)0.f;\n" \
"  if (4 * Z + 3 < shape.w) {\n" \
"    src3 = READ_IMAGE(src_data, smp_zero, (int2)(X * W4 + Y, 4 * Z + 3));\n" \
"  }\n" \
"  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);\n" \
"  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);\n" \
"  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);\n" \
"  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);\n" \
"  WRITE_IMAGE(dst_data, (int2)(4 * Y * C4 + Z, X), dst0);\n" \
"  if (4 * Y + 1 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((4 * Y + 1) * C4 + Z, X), dst1);\n" \
"  }\n" \
"  if (4 * Y + 2 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((4 * Y + 2) * C4 + Z, X), dst2);\n" \
"  }\n" \
"  if (4 * Y + 3 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((4 * Y + 3) * C4 + Z, X), dst3);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void transpose_0231_oversize_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data,\n" \
"                                            int4 shape) {\n" \
"  int X = get_global_id(0);  // H, W for src\n" \
"  int Y = get_global_id(1);  // W4, C4 for src\n" \
"  int Z = get_global_id(2);  // C4, H4 for src\n" \
"  if (X >= shape.y || 4 * Y >= shape.z || 4 * Z >= shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  int W4 = UP_DIV(shape.z, 4);\n" \
"  int C4 = UP_DIV(shape.w, 4);\n" \
"  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(Y, 4 * Z * shape.y + X));\n" \
"  FLT4 src1 = (FLT4)0.f;\n" \
"  if (4 * Z + 1 < shape.w) {\n" \
"    src1 = READ_IMAGE(src_data, smp_zero, (int2)(Y, (4 * Z + 1) * shape.y + X));\n" \
"  }\n" \
"  FLT4 src2 = (FLT4)0.f;\n" \
"  if (4 * Z + 2 < shape.w) {\n" \
"    src2 = READ_IMAGE(src_data, smp_zero, (int2)(Y, (4 * Z + 2) * shape.y + X));\n" \
"  }\n" \
"  FLT4 src3 = (FLT4)0.f;\n" \
"  if (4 * Z + 3 < shape.w) {\n" \
"    src3 = READ_IMAGE(src_data, smp_zero, (int2)(Y, (4 * Z + 3) * shape.y + X));\n" \
"  }\n" \
"  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);\n" \
"  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);\n" \
"  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);\n" \
"  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);\n" \
"  WRITE_IMAGE(dst_data, (int2)(4 * Y * C4 + Z, X), dst0);\n" \
"  if (4 * Y + 1 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((4 * Y + 1) * C4 + Z, X), dst1);\n" \
"  }\n" \
"  if (4 * Y + 2 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((4 * Y + 2) * C4 + Z, X), dst2);\n" \
"  }\n" \
"  if (4 * Y + 3 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)((4 * Y + 3) * C4 + Z, X), dst3);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void transpose_0231_NC4HW4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 shape) {\n" \
"  int X = get_global_id(0);  // H, W for src\n" \
"  int Y = get_global_id(1);  // W4, C4 for src\n" \
"  int Z = get_global_id(2);  // C4, H4 for src\n" \
"  if (X >= shape.y || 4 * Y >= shape.z || 4 * Z >= shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 src0 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z));\n" \
"  FLT4 src1 = (FLT4)0.f;\n" \
"  if (4 * Z + 1 < shape.w) {\n" \
"    src1 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 1));\n" \
"  }\n" \
"  FLT4 src2 = (FLT4)0.f;\n" \
"  if (4 * Z + 2 < shape.w) {\n" \
"    src2 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 2));\n" \
"  }\n" \
"  FLT4 src3 = (FLT4)0.f;\n" \
"  if (4 * Z + 3 < shape.w) {\n" \
"    src3 = READ_IMAGE(src_data, smp_zero, (int2)(X, Y * shape.w + 4 * Z + 3));\n" \
"  }\n" \
"  FLT4 dst0 = (FLT4)(src0.x, src1.x, src2.x, src3.x);\n" \
"  FLT4 dst1 = (FLT4)(src0.y, src1.y, src2.y, src3.y);\n" \
"  FLT4 dst2 = (FLT4)(src0.z, src1.z, src2.z, src3.z);\n" \
"  FLT4 dst3 = (FLT4)(src0.w, src1.w, src2.w, src3.w);\n" \
"  WRITE_IMAGE(dst_data, (int2)(4 * Y, Z * shape.y + X), dst0);\n" \
"  if (4 * Y + 1 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(4 * Y + 1, Z * shape.y + X), dst1);\n" \
"  }\n" \
"  if (4 * Y + 2 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(4 * Y + 2, Z * shape.y + X), dst2);\n" \
"  }\n" \
"  if (4 * Y + 3 < shape.z) {\n" \
"    WRITE_IMAGE(dst_data, (int2)(4 * Y + 3, Z * shape.y + X), dst3);\n" \
"  }\n" \
"}\n" \
"\n" \
"typedef union FLT4_array {\n" \
"  FLT c_array[4];\n" \
"  FLT4 vector;\n" \
"} FLT4_array;\n" \
"\n" \
"__kernel void transpose_general_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 out_shape,\n" \
"                                      int4 de_perm, int4 in_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);\n" \
"  int Z = get_global_id(2);\n" \
"  if (X >= out_shape.y * out_shape.x || Y >= out_shape.z || 4 * Z >= out_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  int N = X / out_shape.y;\n" \
"  int H = X % out_shape.y;\n" \
"  int CI4_SIZE = UP_DIV(in_shape.w, 4);\n" \
"  FLT4_array result_tmp;\n" \
"  result_tmp.vector = (FLT4)(0.f);\n" \
"  FLT *result_ptr = result_tmp.c_array;\n" \
"  for (int i = 0; i < 4; i++) {\n" \
"    if (Z * 4 + i < out_shape.w) {\n" \
"      int out_index[4] = {N, H, Y, Z * 4 + i};\n" \
"      FLT4 src = READ_IMAGE(src_data, smp_zero,\n" \
"                            (int2)(out_index[de_perm.z] * CI4_SIZE + out_index[de_perm.w] / 4,\n" \
"                                   out_index[de_perm.x] * in_shape.y + out_index[de_perm.y]));\n" \
"      FLT4_array src_tmp;\n" \
"      src_tmp.vector = src;\n" \
"      result_tmp.c_array[i] = src_tmp.c_array[out_index[de_perm.w] % 4];\n" \
"    }\n" \
"  }\n" \
"  int CO4_SIZE = UP_DIV(out_shape.w, 4);\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y * CO4_SIZE + Z, X), result_tmp.vector);\n" \
"}\n" \
;
