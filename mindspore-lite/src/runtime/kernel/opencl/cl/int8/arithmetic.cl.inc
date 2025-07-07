static const char *arithmetic_source ="\n" 
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void ElementAddInt8(__read_only image2d_t input_a, __read_only image2d_t input_b,\n" \
"                             __write_only image2d_t output, const int2 output_shape, float act_min, float act_max,\n" \
"                             const float4 scale, const char4 zero_point) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"  char4 a = convert_char4(read_imagei(input_a, smp_none, (int2)(X, Y)));\n" \
"  char4 b = convert_char4(read_imagei(input_b, smp_none, (int2)(X, Y)));\n" \
"\n" \
"  float4 real_a = convert_float4(a - zero_point.x) * scale.x;\n" \
"  float4 real_b = convert_float4(b - zero_point.y) * scale.y;\n" \
"  int4 result = convert_int4(round((real_a + real_b) / scale.z)) + zero_point.z;\n" \
"  result = clamp(result, (FLT)(act_min), (FLT)(act_max));\n" \
"  write_imagei(output, (int2)(X, Y), result);\n" \
"}\n" \
;
