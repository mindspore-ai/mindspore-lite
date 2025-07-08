static const char *scale_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n" \
"#define C4NUM 4\n" \
"\n" \
"__kernel void Scale_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,\n" \
"                        __write_only image2d_t output, const int2 output_shape, const int act_type) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));\n" \
"  FLT4 s = READ_IMAGE(scale, smp_none, (int2)(X, Y));\n" \
"  FLT4 o = READ_IMAGE(offset, smp_none, (int2)(X, Y));\n" \
"  FLT4 out = in * s + o;\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out = max(out, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), out);\n" \
"}\n" \
"\n" \
"__kernel void BoardcastScale_IMG(__read_only image2d_t input, float scale, float offset, __write_only image2d_t output,\n" \
"                                 const int2 output_shape, const int act_type) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));\n" \
"  FLT4 out = in * (FLT)scale + (FLT)offset;\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out = max(out, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), out);\n" \
"}\n" \
"\n" \
"__kernel void Scale_C_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,\n" \
"                          __write_only image2d_t output, const int2 output_shape, const int C, const int act_type) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y || C == 0) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));\n" \
"  FLT4 s = READ_IMAGE(scale, smp_none, (int2)(X % C, 0));\n" \
"  FLT4 o = READ_IMAGE(offset, smp_none, (int2)(X % C, 0));\n" \
"  FLT4 out = in * s + o;\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out = max(out, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), out);\n" \
"}\n" \
"\n" \
"__kernel void Scale_H_IMG(__read_only image2d_t input, __read_only image2d_t scale, __read_only image2d_t offset,\n" \
"                          __write_only image2d_t output, const int2 output_shape, const int H, const int act_type) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= output_shape.x || Y >= output_shape.y || H == 0) {\n" \
"    return;\n" \
"  }\n" \
"  int h = Y % H;\n" \
"  int h_quotient = h / C4NUM;\n" \
"  int h_remainder = h % C4NUM;\n" \
"  FLT4 in = READ_IMAGE(input, smp_none, (int2)(X, Y));\n" \
"  FLT4 s = READ_IMAGE(scale, smp_none, (int2)(h_quotient, 0));\n" \
"  FLT4 o = READ_IMAGE(offset, smp_none, (int2)(h_quotient, 0));\n" \
"  FLT s_real;\n" \
"  FLT o_real;\n" \
"  if (h_remainder == 0) {\n" \
"    s_real = s.x;\n" \
"    o_real = o.x;\n" \
"  } else if (h_remainder == 1) {\n" \
"    s_real = s.y;\n" \
"    o_real = o.y;\n" \
"  } else if (h_remainder == 2) {\n" \
"    s_real = s.z;\n" \
"    o_real = o.z;\n" \
"  } else {\n" \
"    s_real = s.w;\n" \
"    o_real = o.w;\n" \
"  }\n" \
"  FLT4 out = in * s_real + o_real;\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out = max(out, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out = clamp(out, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)(X, Y), out);\n" \
"}\n" \
;
