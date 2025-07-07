static const char *to_format_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"#define BUF_to_IMG(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, WRITE_IMAGE_OUT)                                          \\\n" \
"  __kernel void BUF_to_IMG_##src_dtype##_##dst_dtype(__global SRC_TYPE##4 * src_data, __write_only image2d_t dst_data, \\\n" \
"                                                     int4 size, int4 shape) {                                          \\\n" \
"    int X = get_global_id(0);                                                                                          \\\n" \
"    int Y = get_global_id(1);                                                                                          \\\n" \
"    int Z = get_global_id(2);                                                                                          \\\n" \
"    if (X >= size.x || Y >= size.y || Z >= size.z) {                                                                   \\\n" \
"      return;                                                                                                          \\\n" \
"    }                                                                                                                  \\\n" \
"    DST_TYPE##4 data = (DST_TYPE##4)(0.f);                                                                             \\\n" \
"    int offset = (X * shape.z + Y) * shape.w + Z * 4;                                                                  \\\n" \
"    __global SRC_TYPE *src_addr = (__global SRC_TYPE *)src_data;                                                       \\\n" \
"    src_addr += offset;                                                                                                \\\n" \
"    if ((Z + 1) * 4 <= shape.w) {                                                                                      \\\n" \
"      data = convert_##DST_TYPE##4(((__global SRC_TYPE##4 *)src_addr)[0]);                                             \\\n" \
"    } else {                                                                                                           \\\n" \
"      if ((shape.w - Z * 4) >= 1) {                                                                                    \\\n" \
"        data.x = (DST_TYPE)src_addr[0];                                                                                \\\n" \
"      }                                                                                                                \\\n" \
"      if ((shape.w - Z * 4) >= 2) {                                                                                    \\\n" \
"        data.y = (DST_TYPE)src_addr[1];                                                                                \\\n" \
"      }                                                                                                                \\\n" \
"      if ((shape.w - Z * 4) >= 3) {                                                                                    \\\n" \
"        data.z = (DST_TYPE)src_addr[2];                                                                                \\\n" \
"      }                                                                                                                \\\n" \
"    }                                                                                                                  \\\n" \
"    if (size.y * size.z <= MAX_IMAGE2D_WIDTH)                                                                          \\\n" \
"      WRITE_IMAGE_OUT(dst_data, (int2)(Y * size.z + Z, X), data);                                                      \\\n" \
"    else                                                                                                               \\\n" \
"      WRITE_IMAGE_OUT(dst_data, (int2)(Z, X * size.y + Y), data);                                                      \\\n" \
"  }\n" \
"\n" \
"// BUF_to_IMG(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, WRITE_IMAGE_OUT)\n" \
"BUF_to_IMG(float32, float32, float, float, write_imagef);\n" \
"BUF_to_IMG(float32, float16, float, half, write_imageh);\n" \
"BUF_to_IMG(float16, float16, half, half, write_imageh);\n" \
"BUF_to_IMG(int32, int32, int, int, write_imagei);\n" \
"BUF_to_IMG(uint32, uint32, int, int, write_imagei);\n" \
"BUF_to_IMG(int8, int8, char, int, write_imagei);\n" \
"\n" \
"#define IMG_to_BUF(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, READ_IMAGE_IN)                                           \\\n" \
"  __kernel void IMG_to_BUF_##src_dtype##_##dst_dtype(__read_only image2d_t src_data, __global DST_TYPE##4 * dst_data, \\\n" \
"                                                     int4 size, int4 shape) {                                         \\\n" \
"    int X = get_global_id(0);                                                                                         \\\n" \
"    int Y = get_global_id(1);                                                                                         \\\n" \
"    int Z = get_global_id(2);                                                                                         \\\n" \
"    if (X >= size.x || Y >= size.y || Z >= size.z) {                                                                  \\\n" \
"      return;                                                                                                         \\\n" \
"    }                                                                                                                 \\\n" \
"    DST_TYPE##4 data;                                                                                                 \\\n" \
"    if (size.y * size.z <= MAX_IMAGE2D_WIDTH)                                                                         \\\n" \
"      data = convert_##DST_TYPE##4(READ_IMAGE_IN(src_data, smp_zero, (int2)(Y * size.z + Z, X)));                     \\\n" \
"    else                                                                                                              \\\n" \
"      data = convert_##DST_TYPE##4(READ_IMAGE_IN(src_data, smp_zero, (int2)(Z, X * size.y + Y)));                     \\\n" \
"    int offset = (X * shape.z + Y) * shape.w + Z * 4;                                                                 \\\n" \
"    __global DST_TYPE *dst_addr = (__global DST_TYPE *)dst_data;                                                      \\\n" \
"    dst_addr += offset;                                                                                               \\\n" \
"    if ((Z + 1) * 4 <= shape.w) {                                                                                     \\\n" \
"      ((__global DST_TYPE##4 *)dst_addr)[0] = data;                                                                   \\\n" \
"    } else {                                                                                                          \\\n" \
"      if (shape.w - Z * 4 >= 1) {                                                                                     \\\n" \
"        dst_addr[0] = data.x;                                                                                         \\\n" \
"      }                                                                                                               \\\n" \
"      if (shape.w - Z * 4 >= 2) {                                                                                     \\\n" \
"        dst_addr[1] = data.y;                                                                                         \\\n" \
"      }                                                                                                               \\\n" \
"      if (shape.w - Z * 4 >= 3) {                                                                                     \\\n" \
"        dst_addr[2] = data.z;                                                                                         \\\n" \
"      }                                                                                                               \\\n" \
"    }                                                                                                                 \\\n" \
"  }\n" \
"\n" \
"// IMG_to_BUF(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, READ_IMAGE_IN)\n" \
"IMG_to_BUF(float32, float32, float, float, read_imagef);\n" \
"IMG_to_BUF(float16, float32, half, float, read_imageh);\n" \
"IMG_to_BUF(float16, float16, half, half, read_imageh);\n" \
"IMG_to_BUF(int32, int32, int, int, read_imagei);\n" \
"IMG_to_BUF(uint32, uint32, int, int, read_imagei);\n" \
"IMG_to_BUF(int32, float32, int, float, read_imagei);\n" \
"IMG_to_BUF(int8, int8, char, char, read_imagei);\n" \
;
