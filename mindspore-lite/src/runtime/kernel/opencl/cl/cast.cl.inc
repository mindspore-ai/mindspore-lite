static const char *cast_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void Cast_fp32_to_fp16(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {\n" \
"  int x = get_global_id(0);\n" \
"  int y = get_global_id(1);\n" \
"  if (x >= XY.x || y >= XY.y) {\n" \
"    return;\n" \
"  }\n" \
"  half4 result = convert_half4(read_imagef(input, smp_none, (int2)(x, y)));\n" \
"  write_imageh(output, (int2)(x, y), result);\n" \
"}\n" \
"\n" \
"__kernel void Cast_fp32_to_fp32(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {\n" \
"  int x = get_global_id(0);\n" \
"  int y = get_global_id(1);\n" \
"  if (x >= XY.x || y >= XY.y) {\n" \
"    return;\n" \
"  }\n" \
"  float4 result = read_imagef(input, smp_none, (int2)(x, y));\n" \
"  write_imagef(output, (int2)(x, y), result);\n" \
"}\n" \
"\n" \
"__kernel void Cast_fp16_to_fp16(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {\n" \
"  int x = get_global_id(0);\n" \
"  int y = get_global_id(1);\n" \
"  if (x >= XY.x || y >= XY.y) {\n" \
"    return;\n" \
"  }\n" \
"  half4 result = read_imageh(input, smp_none, (int2)(x, y));\n" \
"  write_imageh(output, (int2)(x, y), result);\n" \
"}\n" \
"\n" \
"__kernel void Cast_fp16_to_fp32(__read_only image2d_t input, __write_only image2d_t output, int2 XY) {\n" \
"  int x = get_global_id(0);\n" \
"  int y = get_global_id(1);\n" \
"  if (x >= XY.x || y >= XY.y) {\n" \
"    return;\n" \
"  }\n" \
"  float4 result = convert_float4(read_imageh(input, smp_none, (int2)(x, y)));\n" \
"  write_imagef(output, (int2)(x, y), result);\n" \
"}\n" \
;
