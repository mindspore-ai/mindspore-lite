static const char *gather_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define C4NUM 4\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void gather(__write_only image2d_t dst_data, __read_only image2d_t src_data, __global int *indices,\n" \
"                     int4 src_size, int4 dst_size, int indices_num, int axis) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  int Z = get_global_id(2);\n" \
"  if (X >= dst_size.x || Y >= dst_size.y * dst_size.w || Z >= dst_size.z || dst_size.y == 0) {\n" \
"    return;\n" \
"  }\n" \
"  DTYPE4 res_data = (DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  int batch = Y / dst_size.y;\n" \
"  int height = Y % dst_size.y;\n" \
"  if (axis == 0) {\n" \
"    res_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + Z, indices[batch] * src_size.y + height));\n" \
"  } else if (axis == 1) {\n" \
"    res_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + Z, batch * src_size.y + indices[height]));\n" \
"  } else if (axis == 2) {\n" \
"    res_data = READ_IMAGE(src_data, smp_zero, (int2)(indices[X] * src_size.z + Z, batch * src_size.y + height));\n" \
"  } else if (axis == 3) {\n" \
"    int offset[4] = {indices[Z * 4] / 4, indices[Z * 4 + 1] / 4, indices[Z * 4 + 2] / 4, indices[Z * 4 + 3] / 4};\n" \
"    DTYPE tmp[4];\n" \
"    DTYPE res_tmp[4];\n" \
"    for (int i = 0; i < indices_num; ++i) {\n" \
"      DTYPE4 rd_data = (DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"      rd_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + offset[i], batch * src_size.y + height));\n" \
"      if (i >= 1 && offset[i] != offset[i - 1]) {\n" \
"        rd_data = READ_IMAGE(src_data, smp_zero, (int2)(X * src_size.z + offset[i], batch * src_size.y + height));\n" \
"      }\n" \
"      tmp[0] = rd_data.x;\n" \
"      tmp[1] = rd_data.y;\n" \
"      tmp[2] = rd_data.z;\n" \
"      tmp[3] = rd_data.w;\n" \
"      res_tmp[i] = tmp[indices[Z * 4 + i] % 4];\n" \
"    }\n" \
"    res_data.x = res_tmp[0];\n" \
"    res_data.y = res_tmp[1];\n" \
"    res_data.z = res_tmp[2];\n" \
"    res_data.w = res_tmp[3];\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(X * dst_size.z + Z, batch * dst_size.y + height), res_data);\n" \
"}\n" \
;
