static const char *reshape_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define C4NUM 4\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void reshape_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 src_size,\n" \
"                            int4 dst_size) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  int CO4 = UP_DIV(dst_size.z, C4NUM);\n" \
"  int CO4_rem = dst_size.z % C4NUM;\n" \
"  if (X >= dst_size.x || Y > dst_size.y) {\n" \
"    return;\n" \
"  }\n" \
"  int CI4 = UP_DIV(src_size.x, C4NUM);\n" \
"  int CI4_rem = src_size.x % C4NUM;\n" \
"  CI4_rem = (CI4_rem == 0) ? C4NUM : CI4_rem;\n" \
"  int in_img_x = CI4 * src_size.y;\n" \
"  DTYPE4 res = (DTYPE4)(0.0f);\n" \
"  DTYPE tmp[4];\n" \
"  DTYPE res_tmp[4];\n" \
"  int gcnt = 0;\n" \
"  if (CO4_rem == 0 && ((CI4_rem & 0x3) == 0)) {\n" \
"    gcnt = X + dst_size.x * Y;\n" \
"    res = READ_IMAGE(src_data, smp_zero, (int2)(gcnt % in_img_x, gcnt / in_img_x));\n" \
"    WRITE_IMAGE(dst_data, (int2)(X, Y), res);\n" \
"  } else {\n" \
"    int start = ((X / CO4 * dst_size.z + min(dst_size.z, (X % CO4) * C4NUM)) + dst_size.w * Y);\n" \
"    gcnt = start / src_size.x * CI4 + (start % src_size.x) / C4NUM;\n" \
"    start = start % src_size.x % C4NUM;\n" \
"    for (int i = 0, n = 0, j = start; i < C4NUM; ++n, j = 0) {\n" \
"      int X_src = (gcnt + n) % in_img_x;\n" \
"      res = READ_IMAGE(src_data, smp_zero, (int2)(X_src, (gcnt + n) / in_img_x));\n" \
"      tmp[0] = res.x;\n" \
"      tmp[1] = res.y;\n" \
"      tmp[2] = res.z;\n" \
"      tmp[3] = res.w;\n" \
"      int k = (X_src % CI4) == (CI4 - 1) ? CI4_rem : C4NUM;\n" \
"      for (; j < k && i < C4NUM; ++j, ++i) {\n" \
"        res_tmp[i] = tmp[j];\n" \
"      }\n" \
"    }\n" \
"    res.x = res_tmp[0];\n" \
"    res.y = res_tmp[1];\n" \
"    res.z = res_tmp[2];\n" \
"    res.w = res_tmp[3];\n" \
"    WRITE_IMAGE(dst_data, (int2)(X, Y), res);\n" \
"  }\n" \
"}\n" \
;
