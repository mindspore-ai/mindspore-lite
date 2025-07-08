static const char *stack_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define INT4 int4\n" \
"#define C4NUM 4\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"#define CHECK_IDX_FOR_STACK                                                                 \\\n" \
"  int X = get_global_id(0);                                                                 \\\n" \
"  int Y = get_global_id(1);                                                                 \\\n" \
"  int Z = get_global_id(2);                                                                 \\\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \\\n" \
"    return;                                                                                 \\\n" \
"  }                                                                                         \\\n" \
"  FLT4 result;\n" \
"\n" \
"// input -1D\n" \
"__kernel void stack_2input_3axis_1inshape(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {\n" \
"  int X = get_global_id(0);  // N*H\n" \
"  int Y = get_global_id(1);  // W*C\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z * output_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result1 = READ_IMAGE(input0, smp_none, (int2)(X, 0));\n" \
"  FLT result1_temp[4] = {result1.x, result1.y, result1.z, result1.w};\n" \
"  FLT4 result2 = READ_IMAGE(input1, smp_none, (int2)(X, 0));\n" \
"  FLT result2_temp[4] = {result2.x, result2.y, result2.z, result2.w};\n" \
"  for (int i = 0; i < C4NUM; ++i) {\n" \
"    FLT4 result = {result1_temp[i], result2_temp[i], 0, 0};\n" \
"    WRITE_IMAGE(output, (int2)(Y, (X * C4NUM + i)), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"// input -2D -axis = 1\n" \
"__kernel void stack_2input_1axis_2inshape(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {\n" \
"  CHECK_IDX_FOR_STACK;\n" \
"  int IN = X / output_shape.y;\n" \
"  int IH = X % output_shape.y;\n" \
"  int boundary0 = input_shape.z;\n" \
"  if (Y < boundary0) {\n" \
"    int coordinate_x = Y * input_shape.w + Z;\n" \
"    int coordinate_y = IN * input_shape.y + IH;\n" \
"    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y));\n" \
"  } else {\n" \
"    int coordinate_x = (Y - boundary0) * input_shape.w + Z;\n" \
"    int coordinate_y = IN * input_shape.y + IH;\n" \
"    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y));\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (IN * output_shape.y + IH)), result);\n" \
"}\n" \
"\n" \
"// input -3D -axis = 1\n" \
"__kernel void stack_2input_1axis_3inshape(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {\n" \
"  CHECK_IDX_FOR_STACK;\n" \
"  int IN = X / output_shape.y;\n" \
"  int IH = X % output_shape.y;\n" \
"  int boundary0 = input_shape.y;\n" \
"  if (IH < boundary0) {\n" \
"    int coordinate_x = Y * input_shape.w + Z;\n" \
"    int coordinate_y = IN * input_shape.y + IH;\n" \
"    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y));\n" \
"  } else {\n" \
"    int coordinate_x = Y * input_shape.w + Z;\n" \
"    int coordinate_y = IN * input_shape.y + IH - boundary0;\n" \
"    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y));\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (IN * output_shape.y + IH)), result);\n" \
"}\n" \
"\n" \
"// input -3D -axis = 2\n" \
"__kernel void stack_2input_2axis_3inshape(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                          __write_only image2d_t output, int4 input_shape, int4 output_shape) {\n" \
"  CHECK_IDX_FOR_STACK;\n" \
"  int boundary0 = input_shape.y;\n" \
"  int IN = X / output_shape.y;\n" \
"  int IW = X % output_shape.y;\n" \
"  int IC = Z;\n" \
"  int coordinate_x = IW * input_shape.w + IC;\n" \
"  int coordinate_y = IN * input_shape.y;\n" \
"  if (Y < boundary0) {\n" \
"    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y));\n" \
"  } else {\n" \
"    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y));\n" \
"  }\n" \
"  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, IN * output_shape.y + IW), result);\n" \
"}\n" \
"\n" \
"// input -3D -axis = 3  and input -2D -axis = 2  boundary stack\n" \
"__kernel void stack_2input_boundary(__global float *input0, __global float *input1, __global float *output,\n" \
"                                    int4 input_shape, int4 output_shape, int2 stride_w) {\n" \
"  int X = get_global_id(0);  // N\n" \
"  int Y = get_global_id(1);  // H\n" \
"  if (X >= output_shape.x || Y >= output_shape.y) {\n" \
"    return;\n" \
"  }\n" \
"  int IW = output_shape.z;\n" \
"  int Align_out = output_shape.w * C4NUM;\n" \
"  int Align_in = input_shape.w * C4NUM;\n" \
"  int index_out = X * output_shape.y * stride_w.x + Y * stride_w.x;\n" \
"  int index_in = X * input_shape.y * stride_w.y + Y * Align_in;\n" \
"  for (int iw = 0; iw < IW; iw++) {\n" \
"    int index_out_tmp = index_out + iw * Align_out;\n" \
"    int index_in_tmp = index_in + iw;\n" \
"    output[index_out_tmp] = input0[index_in_tmp];\n" \
"    index_out_tmp++;\n" \
"    output[index_out_tmp] = input1[index_in_tmp];\n" \
"  }\n" \
"}\n" \
;
