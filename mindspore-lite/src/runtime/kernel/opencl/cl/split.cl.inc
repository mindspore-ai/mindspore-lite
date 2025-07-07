static const char *split_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"#define C4NUM 4\n" \
"\n" \
"#define CHECK_IDX_ALIGN                                                                     \\\n" \
"  const int X = get_global_id(0);                                                           \\\n" \
"  const int Y = get_global_id(1);                                                           \\\n" \
"  const int Z = get_global_id(2);                                                           \\\n" \
"  if (X > in_shape.x * in_shape.y || Y > in_shape.z || Z > in_shape.w || in_shape.y == 0) { \\\n" \
"    return;                                                                                 \\\n" \
"  }\n" \
"\n" \
"#define ARGS_ALIGN                         \\\n" \
"  const int IN = X / in_shape.y;           \\\n" \
"  const int IH = X % in_shape.y;           \\\n" \
"  int coordinate_x = IN * in_shape.y + IH; \\\n" \
"  int coordinate_y = Y * in_shape.w + Z;   \\\n" \
"  FLT4 result = READ_IMAGE(input, smp_none, (int2)(coordinate_y, coordinate_x));\n" \
"\n" \
"__kernel void split_out2_axis3(__read_only image2d_t input, __write_only image2d_t output1,\n" \
"                               __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,\n" \
"                               int4 out_shape1, int4 out_shape2) {\n" \
"  CHECK_IDX_ALIGN;\n" \
"  ARGS_ALIGN;\n" \
"  int boundary = UP_DIV(split_sizes_[0], C4NUM);\n" \
"  if (Z < boundary) {\n" \
"    coordinate_x = IN * out_shape1.y + IH;\n" \
"    coordinate_y = Y * out_shape1.w + Z;\n" \
"    WRITE_IMAGE(output1, (int2)(coordinate_y, coordinate_x), result);\n" \
"  } else {\n" \
"    coordinate_x = IN * out_shape2.y + IH;\n" \
"    coordinate_y = Y * out_shape2.w + Z - boundary;\n" \
"    WRITE_IMAGE(output2, (int2)(coordinate_y, coordinate_x), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void split_out2_axis2(__read_only image2d_t input, __write_only image2d_t output1,\n" \
"                               __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,\n" \
"                               int4 out_shape1, int4 out_shape2) {\n" \
"  CHECK_IDX_ALIGN;\n" \
"  ARGS_ALIGN;\n" \
"  if (Y < split_sizes_[0]) {\n" \
"    coordinate_x = IN * out_shape1.y + IH;\n" \
"    coordinate_y = Y * out_shape1.w + Z;\n" \
"    WRITE_IMAGE(output1, (int2)(coordinate_y, coordinate_x), result);\n" \
"  } else {\n" \
"    coordinate_x = IN * out_shape2.y + IH;\n" \
"    coordinate_y = (Y - split_sizes_[0]) * out_shape2.w + Z;\n" \
"    WRITE_IMAGE(output2, (int2)(coordinate_y, coordinate_x), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void split_out2_axis1(__read_only image2d_t input, __write_only image2d_t output1,\n" \
"                               __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,\n" \
"                               int4 out_shape1, int4 out_shape2) {\n" \
"  CHECK_IDX_ALIGN;\n" \
"  ARGS_ALIGN;\n" \
"  if (IH < split_sizes_[0]) {\n" \
"    coordinate_x = IN * out_shape1.y + IH;\n" \
"    coordinate_y = Y * out_shape1.w + Z;\n" \
"    WRITE_IMAGE(output1, (int2)(coordinate_y, coordinate_x), result);\n" \
"  } else {\n" \
"    coordinate_x = IN * out_shape2.y + IH - split_sizes_[0];\n" \
"    coordinate_y = Y * out_shape2.w + Z;\n" \
"    WRITE_IMAGE(output2, (int2)(coordinate_y, coordinate_x), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"// UnAlign in Axis C for concat\n" \
"#define CHECK_IDX_UNALIGN                                                   \\\n" \
"  const int X = get_global_id(0);                                           \\\n" \
"  const int Y = get_global_id(1);                                           \\\n" \
"  if (X >= in_shape.x * in_shape.y || Y >= in_shape.z || in_shape.y == 0) { \\\n" \
"    return;                                                                 \\\n" \
"  }\n" \
"\n" \
"#define ARGS_UNALIGN                                   \\\n" \
"  const int IN = X / in_shape.y, IH = X % in_shape.y;  \\\n" \
"  const int IW = Y;                                    \\\n" \
"  const int Align_inShape = UP_DIV(in_shape.w, C4NUM); \\\n" \
"  int index_input = (IN * in_shape.y + IH) * stride_w + IW * Align_inShape * C4NUM;\n" \
"\n" \
"int dosplit(__global FLT *input, __write_only image2d_t output, int4 out_shape, int IN, int IH, int IW,\n" \
"            int index_input) {\n" \
"  int Remainder = out_shape.w % C4NUM;\n" \
"  int coordinate_x = IN * out_shape.y + IH;\n" \
"  int align_w = UP_DIV(out_shape.w, C4NUM);\n" \
"  for (int i = 0; i < align_w; ++i) {\n" \
"    int coordinate_y = IW * align_w + i;\n" \
"    if ((i + 1) * C4NUM <= out_shape.w) {\n" \
"      FLT4 result = {input[index_input], input[index_input + 1], input[index_input + 2], input[index_input + 3]};\n" \
"      WRITE_IMAGE(output, (int2)(coordinate_y, coordinate_x), result);\n" \
"      index_input += 4;\n" \
"    } else {\n" \
"      FLT result_temp[4] = {};\n" \
"      for (int j = 0; j < Remainder; ++j) {\n" \
"        result_temp[j] = input[index_input++];\n" \
"      }\n" \
"      FLT4 result = {result_temp[0], result_temp[1], result_temp[2], result_temp[3]};\n" \
"      WRITE_IMAGE(output, (int2)(coordinate_y, coordinate_x), result);\n" \
"    }\n" \
"  }\n" \
"  return index_input;\n" \
"}\n" \
"\n" \
"__kernel void split_out2_axis3_unalign(__global FLT *input, __write_only image2d_t output1,\n" \
"                                       __write_only image2d_t output2, __global int *split_sizes_, int4 in_shape,\n" \
"                                       int4 out_shape1, int4 out_shape2, int stride_w) {\n" \
"  CHECK_IDX_UNALIGN;\n" \
"  ARGS_UNALIGN;\n" \
"  index_input = dosplit(input, output1, out_shape1, IN, IH, IW, index_input);\n" \
"  index_input = dosplit(input, output2, out_shape2, IN, IH, IW, index_input);\n" \
"}\n" \
;
