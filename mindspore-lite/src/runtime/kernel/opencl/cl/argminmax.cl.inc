static const char *argminmax_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define swap(a, b, c) \\\n" \
"  c = a;              \\\n" \
"  a = b;              \\\n" \
"  b = c;\n" \
"#define swap_atomic(a, b, c) \\\n" \
"  c = atomic_xchg(a, *(b));  \\\n" \
"  c = atomic_xchg(b, c);\n" \
"#define UP_ROUND(a, b) (((a + b - 1) / b) * b)\n" \
"#define C4NUM 4\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void argminmax(__global FLT *src_data, __global FLT *dst_data, __global FLT *buf, __global int *ids,\n" \
"                        int4 shape, int4 src_size, int4 cus_size, int4 strides, int4 flags) {\n" \
"  int X = get_global_id(0);  // lower reduce stride\n" \
"  int Y = get_global_id(1);  // upper axis accumulation\n" \
"  if (X >= src_size.x || Y >= src_size.y) {\n" \
"    return;\n" \
"  }\n" \
"  bool keep_dims = cus_size.y;\n" \
"  int width = shape.z * shape.w;\n" \
"  int offset = X + Y * src_size.z;\n" \
"  int align_c4_in = (flags.z != 3) ? (X / shape.w) * (C4NUM - shape.w & 0x00000003) : 0;\n" \
"  int align_c4_out =\n" \
"    (flags.z == 3 && flags.w == 1 && !keep_dims) ? (Y / shape.z) * (C4NUM - shape.z & 0x00000003) : align_c4_in;\n" \
"  int align_in = 0;\n" \
"  int align_out = 0;\n" \
"  if (flags.z == 3) {\n" \
"    align_in = (Y / shape.z) * cus_size.z;\n" \
"    align_out = (Y / ((flags.w > 1 || keep_dims) ? shape.z : shape.z * shape.y)) * cus_size.w;\n" \
"  }\n" \
"  if (flags.z == 0) {\n" \
"    align_in = X / (width)*cus_size.z;\n" \
"    align_out = align_in;\n" \
"  }\n" \
"  if (flags.z == 2 && !keep_dims) {\n" \
"    align_out = (Y / shape.y) * cus_size.w;\n" \
"  }\n" \
"  for (int k = 0; k < src_size.w; ++k) {\n" \
"    int idx0 = (X + k * strides.x) + Y * strides.y + (align_c4_in + align_in);\n" \
"    int idx1 = offset + k * src_size.x;\n" \
"    ids[idx1] = k;\n" \
"    buf[idx1] = src_data[idx0];\n" \
"  }\n" \
"  for (unsigned int i = 2; i <= cus_size.x; i <<= 1) {\n" \
"    for (unsigned int j = i >> 1; j > 0; j >>= 1) {\n" \
"      for (int tid = 0; tid < src_size.w; ++tid) {\n" \
"        unsigned int tid_comp = tid + j;\n" \
"        if (tid_comp < src_size.w) {\n" \
"          int lk = offset + tid * src_size.x;\n" \
"          int rk = offset + tid_comp * src_size.x;\n" \
"          if ((tid & i) == 0) {  // ascending\n" \
"            if (buf[lk] > buf[rk]) {\n" \
"              FLT tmpf;\n" \
"              swap(buf[lk], buf[rk], tmpf);\n" \
"              int tmpi;\n" \
"              swap(ids[lk], ids[rk], tmpi);\n" \
"            }\n" \
"          } else {  // desending\n" \
"            if (buf[lk] < buf[rk]) {\n" \
"              FLT tmpf;\n" \
"              swap(buf[lk], buf[rk], tmpf);\n" \
"              int tmpi;\n" \
"              swap(ids[lk], ids[rk], tmpi);\n" \
"            }\n" \
"          }\n" \
"        }\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"  for (int k = 0; k < flags.w; ++k) {\n" \
"    int idx0 = (X + k * strides.z) + Y * strides.w + (align_c4_out + align_out);\n" \
"    int idx1 = flags.y ? (offset + (src_size.w - k - 1) * src_size.x) : (offset + k * src_size.x);\n" \
"    if (flags.x) {\n" \
"      dst_data[idx0] = buf[idx1];\n" \
"    } else {\n" \
"      dst_data[idx0] = ids[idx1];\n" \
"    }\n" \
"  }\n" \
"}\n" \
;
