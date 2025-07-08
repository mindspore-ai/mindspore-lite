static const char *reduce_source ="\n" 
"#ifdef cl_khr_fp16\n" \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#endif\n" \
"#define LOCAL_CACHE_THREAD 16\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"__kernel void GlobalHWMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  if (X >= size.z) {\n" \
"    return;\n" \
"  }\n" \
"  float4 result = (float4)0.f;\n" \
"  for (int h = 0; h < size.x; h++) {\n" \
"    for (int w = 0; w < size.y; w++) {\n" \
"      result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));\n" \
"    }\n" \
"  }\n" \
"  result /= size.x * size.y;\n" \
"  WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));\n" \
"}\n" \
"\n" \
"__kernel void LocalHWMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {\n" \
"  int X = get_global_id(0);  // C4\n" \
"  int localy = get_local_id(1);\n" \
"  int localz = get_local_id(2);\n" \
"  if (X >= size.z) return;\n" \
"  __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];\n" \
"  temp[localy][localz] = (float4)0.f;\n" \
"  for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {\n" \
"    for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {\n" \
"      temp[localy][localz] += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (localz == 0) {\n" \
"    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"      temp[localy][0] += temp[localy][i];\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  float4 result = temp[0][0];\n" \
"  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"    result += temp[i][0];\n" \
"  }\n" \
"  result /= size.x * size.y;\n" \
"  WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));\n" \
"}\n" \
"\n" \
"__kernel void GlobalWCMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, float4 mask) {\n" \
"  int X = get_global_id(0);  // H\n" \
"  if (X >= size.x) {\n" \
"    return;\n" \
"  }\n" \
"  float4 result = (float4)0.f;\n" \
"  for (int w = 0; w < size.y; w++) {\n" \
"    for (int c = 0; c < size.z; c++) {\n" \
"      result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));\n" \
"    }\n" \
"  }\n" \
"\n" \
"  result /= size.y * size.w;\n" \
"  FLT4 result2 = (FLT4)(0.f);\n" \
"  result2.x = dot(TO_FLT4(result), (FLT4)(1.f));\n" \
"  WRITE_IMAGE(dst_data, (int2)(0, X), result2);\n" \
"}\n" \
"\n" \
"__kernel void LocalWCMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, float4 mask) {\n" \
"  int X = get_global_id(0);  // H\n" \
"  int localy = get_local_id(1);\n" \
"  int localz = get_local_id(2);\n" \
"  if (X >= size.x) return;\n" \
"  __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];\n" \
"  temp[localy][localz] = (float4)0.f;\n" \
"  for (int w = localy; w < size.y; w += LOCAL_CACHE_THREAD) {\n" \
"    for (int c = localz; c < size.z; c += LOCAL_CACHE_THREAD) {\n" \
"      temp[localy][localz] += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (localz == 0) {\n" \
"    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"      temp[localy][0] += temp[localy][i];\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  float4 result = temp[0][0];\n" \
"  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"    result += temp[i][0];\n" \
"  }\n" \
"  result /= size.y * size.w;\n" \
"  FLT4 result2 = (FLT4)(0.f);\n" \
"  result2.x = dot(TO_FLT4(result), (FLT4)(1.f));\n" \
"  WRITE_IMAGE(dst_data, (int2)(0, X), result2);\n" \
"}\n" \
"\n" \
"__kernel void GlobalHWSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {\n" \
"  int X = get_global_id(0);\n" \
"  if (X >= size.z) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = (FLT4)0.f;\n" \
"  for (int h = 0; h < size.x; h++) {\n" \
"    for (int w = 0; w < size.y; w++) {\n" \
"      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h));\n" \
"      result += current * current;\n" \
"    }\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(X, 0), result);\n" \
"}\n" \
"\n" \
"__kernel void LocalHWSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {\n" \
"  int X = get_global_id(0);\n" \
"  int localy = get_local_id(1);\n" \
"  int localz = get_local_id(2);\n" \
"  if (X >= size.z) return;\n" \
"  __local FLT4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];\n" \
"  temp[localy][localz] = (FLT4)0.f;\n" \
"  for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {\n" \
"    for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {\n" \
"      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h));\n" \
"      temp[localy][localz] += current * current;\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (localz == 0) {\n" \
"    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"      temp[localy][0] += temp[localy][i];\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  FLT4 result = temp[0][0];\n" \
"  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"    result += temp[i][0];\n" \
"  }\n" \
"  WRITE_IMAGE(dst_data, (int2)(X, 0), result);\n" \
"}\n" \
"\n" \
"__kernel void GlobalWCSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size,\n" \
"                                float4 mask) {\n" \
"  int X = get_global_id(0);\n" \
"  if (X >= size.x) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = (FLT4)0.f;\n" \
"  for (int w = 0; w < size.y; w++) {\n" \
"    for (int c = 0; c < size.z; c++) {\n" \
"      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));\n" \
"      result += current * current;\n" \
"    }\n" \
"  }\n" \
"\n" \
"  FLT4 result2 = (FLT4)(0.f);\n" \
"  result2.x = dot(result, (FLT4)(1.f));\n" \
"  WRITE_IMAGE(dst_data, (int2)(0, X), result2);\n" \
"}\n" \
"\n" \
"__kernel void LocalWCSumSquare(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size,\n" \
"                               float4 mask) {\n" \
"  int X = get_global_id(0);\n" \
"  int localy = get_local_id(1);\n" \
"  int localz = get_local_id(2);\n" \
"  if (X >= size.x) return;\n" \
"  __local FLT4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];\n" \
"  temp[localy][localz] = (FLT4)0.f;\n" \
"  for (int w = localy; w < size.y; w += LOCAL_CACHE_THREAD) {\n" \
"    for (int c = localz; c < size.z; c += LOCAL_CACHE_THREAD) {\n" \
"      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));\n" \
"      temp[localy][localz] += current * current;\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (localz == 0) {\n" \
"    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"      temp[localy][0] += temp[localy][i];\n" \
"    }\n" \
"  }\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  FLT4 result = temp[0][0];\n" \
"  for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {\n" \
"    result += temp[i][0];\n" \
"  }\n" \
"  FLT4 result2 = (FLT4)(0.f);\n" \
"  result2.x = dot(result, (FLT4)(1.f));\n" \
"  WRITE_IMAGE(dst_data, (int2)(0, X), result2);\n" \
"}\n" \
"\n" \
"__kernel void GlobalCMean(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, float4 mask) {\n" \
"  int X = get_global_id(0);  // H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  if (X >= size.x || Y >= size.y) {\n" \
"    return;\n" \
"  }\n" \
"  float4 result = (float4)0.f;\n" \
"  for (int c = 0; c < size.z; c++) {\n" \
"    result += convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + c, X)));\n" \
"  }\n" \
"\n" \
"  result /= size.w;\n" \
"  FLT4 result2 = (FLT4)(0.f);\n" \
"  result2.x = dot(TO_FLT4(result), (FLT4)(1.f));\n" \
"  WRITE_IMAGE(dst_data, (int2)(Y, X), result2);\n" \
"}\n" \
"\n" \
"#define GlobalHW(Method)                                                                                       \\\n" \
"  __kernel void GlobalHW##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) { \\\n" \
"    int X = get_global_id(0);                                                                                  \\\n" \
"    if (X >= size.z) {                                                                                         \\\n" \
"      return;                                                                                                  \\\n" \
"    }                                                                                                          \\\n" \
"    FLT4 result = (FLT4)Init##Method;                                                                          \\\n" \
"    for (int h = 0; h < size.x; h++) {                                                                         \\\n" \
"      for (int w = 0; w < size.y; w++) {                                                                       \\\n" \
"        FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h));                              \\\n" \
"        Do##Method(result, current);                                                                           \\\n" \
"      }                                                                                                        \\\n" \
"    }                                                                                                          \\\n" \
"    WRITE_IMAGE(dst_data, (int2)(X, 0), result);                                                               \\\n" \
"  }\n" \
"\n" \
"#define GlobalWC(Method)                                                                                     \\\n" \
"  __kernel void GlobalWC##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, \\\n" \
"                                 float4 mask) {                                                              \\\n" \
"    int X = get_global_id(0);                                                                                \\\n" \
"    if (X >= size.x) {                                                                                       \\\n" \
"      return;                                                                                                \\\n" \
"    }                                                                                                        \\\n" \
"    FLT4 result = (FLT4)Init##Method;                                                                        \\\n" \
"    FLT4 maskFLT4 = TO_FLT4(mask);                                                                           \\\n" \
"    for (int w = 0; w < size.y; w++) {                                                                       \\\n" \
"      int c = 0;                                                                                             \\\n" \
"      for (; c < size.z - 1; c++) {                                                                          \\\n" \
"        FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));                            \\\n" \
"        Do##Method(result, current);                                                                         \\\n" \
"      }                                                                                                      \\\n" \
"      FLT4 current = READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X));                              \\\n" \
"      current += maskFLT4;                                                                                   \\\n" \
"      Do##Method(result, current);                                                                           \\\n" \
"    }                                                                                                        \\\n" \
"    Do##Method(result.x, result.y);                                                                          \\\n" \
"    Do##Method(result.x, result.z);                                                                          \\\n" \
"    Do##Method(result.x, result.w);                                                                          \\\n" \
"    FLT4 result2 = (FLT4)(0.f);                                                                              \\\n" \
"    result2.x = TO_FLT(result.x);                                                                            \\\n" \
"    WRITE_IMAGE(dst_data, (int2)(0, X), result2);                                                            \\\n" \
"  }\n" \
"\n" \
"#define LocalHW(Method)                                                                                       \\\n" \
"  __kernel void LocalHW##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) { \\\n" \
"    int X = get_global_id(0);                                                                                 \\\n" \
"    int localy = get_local_id(1);                                                                             \\\n" \
"    int localz = get_local_id(2);                                                                             \\\n" \
"    if (X >= size.z) return;                                                                                  \\\n" \
"    __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];                                              \\\n" \
"    temp[localy][localz] = (float4)Init##Method;                                                              \\\n" \
"    for (int h = localy; h < size.x; h += LOCAL_CACHE_THREAD) {                                               \\\n" \
"      for (int w = localz; w < size.y; w += LOCAL_CACHE_THREAD) {                                             \\\n" \
"        float4 current = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + X, h)));           \\\n" \
"        Do##Method(temp[localy][localz], current);                                                            \\\n" \
"      }                                                                                                       \\\n" \
"    }                                                                                                         \\\n" \
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                             \\\n" \
"    if (localz == 0) {                                                                                        \\\n" \
"      for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                          \\\n" \
"        Do##Method(temp[localy][0], temp[localy][i]);                                                         \\\n" \
"      }                                                                                                       \\\n" \
"    }                                                                                                         \\\n" \
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                             \\\n" \
"    float4 result = temp[0][0];                                                                               \\\n" \
"    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                            \\\n" \
"      Do##Method(result, temp[i][0]);                                                                         \\\n" \
"    }                                                                                                         \\\n" \
"    WRITE_IMAGE(dst_data, (int2)(X, 0), TO_FLT4(result));                                                     \\\n" \
"  }\n" \
"\n" \
"#define LocalWC(Method)                                                                                     \\\n" \
"  __kernel void LocalWC##Method(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size, \\\n" \
"                                float4 mask) {                                                              \\\n" \
"    int X = get_global_id(0);                                                                               \\\n" \
"    int localy = get_local_id(1);                                                                           \\\n" \
"    int localz = get_local_id(2);                                                                           \\\n" \
"    if (X >= size.x) return;                                                                                \\\n" \
"    __local float4 temp[LOCAL_CACHE_THREAD][LOCAL_CACHE_THREAD];                                            \\\n" \
"    temp[localy][localz] = (float4)Init##Method;                                                            \\\n" \
"    for (int w = localy; w < size.y; w += LOCAL_CACHE_THREAD) {                                             \\\n" \
"      int c = localz;                                                                                       \\\n" \
"      for (; c < size.z - 1; c += LOCAL_CACHE_THREAD) {                                                     \\\n" \
"        float4 current = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));         \\\n" \
"        Do##Method(temp[localy][localz], current);                                                          \\\n" \
"      }                                                                                                     \\\n" \
"      if (c == size.z - 1) {                                                                                \\\n" \
"        float4 current = convert_float4(READ_IMAGE(src_data, smp_zero, (int2)(w * size.z + c, X)));         \\\n" \
"        current += mask;                                                                                    \\\n" \
"        Do##Method(temp[localy][localz], current);                                                          \\\n" \
"      }                                                                                                     \\\n" \
"    }                                                                                                       \\\n" \
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                           \\\n" \
"    if (localz == 0) {                                                                                      \\\n" \
"      for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                        \\\n" \
"        Do##Method(temp[localy][0], temp[localy][i]);                                                       \\\n" \
"      }                                                                                                     \\\n" \
"    }                                                                                                       \\\n" \
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                           \\\n" \
"    float4 result = temp[0][0];                                                                             \\\n" \
"    for (int i = 1; i < LOCAL_CACHE_THREAD; i++) {                                                          \\\n" \
"      Do##Method(result, temp[i][0]);                                                                       \\\n" \
"    }                                                                                                       \\\n" \
"    Do##Method(result.x, result.y);                                                                         \\\n" \
"    Do##Method(result.x, result.z);                                                                         \\\n" \
"    Do##Method(result.x, result.w);                                                                         \\\n" \
"    FLT4 result2 = (FLT4)(0.f);                                                                             \\\n" \
"    result2.x = TO_FLT(result.x);                                                                           \\\n" \
"    WRITE_IMAGE(dst_data, (int2)(0, X), result2);                                                           \\\n" \
"  }\n" \
"\n" \
"#define DoSum(A, B) A += B\n" \
"#define InitSum 0.f\n" \
"GlobalHW(Sum) GlobalWC(Sum) LocalHW(Sum) LocalWC(Sum)\n" \
"#define DoMin(A, B) A = min(A, B)\n" \
"#define InitMin 10000.f\n" \
"  GlobalHW(Min) GlobalWC(Min) LocalHW(Min) LocalWC(Min)\n" \
"\n" \
"#define DoMax(A, B) A = max(A, B)\n" \
"#define InitMax -10000.f\n" \
"    GlobalHW(Max) GlobalWC(Max) LocalHW(Max) LocalWC(Max)\n" \
"\n" \
"#define DoProd(A, B) A *= B\n" \
"#define InitProd 1.f\n" \
"      GlobalHW(Prod) GlobalWC(Prod) LocalHW(Prod) LocalWC(Prod)\n" \
;
