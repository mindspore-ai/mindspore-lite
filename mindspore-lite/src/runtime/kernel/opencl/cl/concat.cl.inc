static const char *concat_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"#define C4NUM 4\n" \
"\n" \
"// Align in Axis C for concat\n" \
"#define CHECK_IDX                                                                           \\\n" \
"  int X = get_global_id(0);                                                                 \\\n" \
"  int Y = get_global_id(1);                                                                 \\\n" \
"  int Z = get_global_id(2);                                                                 \\\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) { \\\n" \
"    return;                                                                                 \\\n" \
"  }                                                                                         \\\n" \
"  DTYPE4 result;\n" \
"\n" \
"// axis = 1\n" \
"#define DOConcat2inputaxis1_NHWC4                                              \\\n" \
"  int IN = X / output_shape.y;                                                 \\\n" \
"  int IH = X % output_shape.y;                                                 \\\n" \
"  int boundary0 = input_shape0.y;                                              \\\n" \
"  int boundary1 = boundary0 + input_shape1.y;                                  \\\n" \
"  if (IH < boundary0) {                                                        \\\n" \
"    int coordinate_x = Y * input_shape0.w + Z;                                 \\\n" \
"    int coordinate_y = IN * input_shape0.y + IH;                               \\\n" \
"    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  } else if (IH < boundary1) {                                                 \\\n" \
"    int coordinate_x = Y * input_shape1.w + Z;                                 \\\n" \
"    int coordinate_y = IN * input_shape1.y + IH - boundary0;                   \\\n" \
"    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat3inputaxis1_NHWC4                                              \\\n" \
"  DOConcat2inputaxis1_NHWC4;                                                   \\\n" \
"  int boundary2 = boundary1 + input_shape2.y;                                  \\\n" \
"  if (IH >= boundary1 && IH < boundary2) {                                     \\\n" \
"    int coordinate_x = Y * input_shape2.w + Z;                                 \\\n" \
"    int coordinate_y = IN * input_shape2.y + IH - boundary1;                   \\\n" \
"    result = READ_IMAGE(input2, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat4inputaxis1_NHWC4                                              \\\n" \
"  DOConcat3inputaxis1_NHWC4;                                                   \\\n" \
"  int boundary3 = boundary2 + input_shape3.y;                                  \\\n" \
"  if (IH >= boundary2 && IH < boundary3) {                                     \\\n" \
"    int coordinate_x = Y * input_shape3.w + Z;                                 \\\n" \
"    int coordinate_y = IN * input_shape3.y + IH - boundary2;                   \\\n" \
"    result = READ_IMAGE(input3, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat5inputaxis1_NHWC4                                              \\\n" \
"  DOConcat4inputaxis1_NHWC4;                                                   \\\n" \
"  int boundary4 = boundary3 + input_shape4.y;                                  \\\n" \
"  if (IH >= boundary3 && IH < boundary4) {                                     \\\n" \
"    int coordinate_x = Y * input_shape4.w + Z;                                 \\\n" \
"    int coordinate_y = IN * input_shape4.y + IH - boundary3;                   \\\n" \
"    result = READ_IMAGE(input4, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat6inputaxis1_NHWC4                                              \\\n" \
"  DOConcat5inputaxis1_NHWC4;                                                   \\\n" \
"  int boundary5 = boundary4 + input_shape5.y;                                  \\\n" \
"  if (IH >= boundary4 && IH < boundary5) {                                     \\\n" \
"    int coordinate_x = Y * input_shape5.w + Z;                                 \\\n" \
"    int coordinate_y = IN * input_shape5.y + IH - boundary4;                   \\\n" \
"    result = READ_IMAGE(input5, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"// axis = 2\n" \
"#define DOConcat2inputaxis2_NHWC4                                              \\\n" \
"  int boundary0 = input_shape0.z;                                              \\\n" \
"  int boundary1 = boundary0 + input_shape1.z;                                  \\\n" \
"  if (Y < boundary0) {                                                         \\\n" \
"    int coordinate_x = Y * input_shape0.w + Z;                                 \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  } else if (Y < boundary1) {                                                  \\\n" \
"    int coordinate_x = (Y - boundary0) * input_shape1.w + Z;                   \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat3inputaxis2_NHWC4                                              \\\n" \
"  DOConcat2inputaxis2_NHWC4;                                                   \\\n" \
"  int boundary2 = boundary1 + input_shape2.z;                                  \\\n" \
"  if (Y >= boundary1 && Y < boundary2) {                                       \\\n" \
"    int coordinate_x = (Y - boundary1) * input_shape2.w + Z;                   \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input2, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat4inputaxis2_NHWC4                                              \\\n" \
"  DOConcat3inputaxis2_NHWC4;                                                   \\\n" \
"  int boundary3 = boundary2 + input_shape3.z;                                  \\\n" \
"  if (Y >= boundary2 && Y < boundary3) {                                       \\\n" \
"    int coordinate_x = (Y - boundary2) * input_shape3.w + Z;                   \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input3, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat5inputaxis2_NHWC4                                              \\\n" \
"  DOConcat4inputaxis2_NHWC4;                                                   \\\n" \
"  int boundary4 = boundary3 + input_shape4.z;                                  \\\n" \
"  if (Y >= boundary3 && Y < boundary4) {                                       \\\n" \
"    int coordinate_x = (Y - boundary3) * input_shape4.w + Z;                   \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input4, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat6inputaxis2_NHWC4                                              \\\n" \
"  DOConcat5inputaxis2_NHWC4;                                                   \\\n" \
"  int boundary5 = boundary4 + input_shape5.z;                                  \\\n" \
"  if (Y >= boundary4 && Y < boundary5) {                                       \\\n" \
"    int coordinate_x = (Y - boundary4) * input_shape5.w + Z;                   \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input5, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"// axis = 3\n" \
"#define DOConcat2inputaxis3_NHWC4                                              \\\n" \
"  int boundary0 = input_shape0.w;                                              \\\n" \
"  int boundary1 = boundary0 + input_shape1.w;                                  \\\n" \
"  if (Z < boundary0) {                                                         \\\n" \
"    int coordinate_x = Y * input_shape0.w + Z;                                 \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input0, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  } else if (Z < boundary1) {                                                  \\\n" \
"    int coordinate_x = Y * input_shape1.w + Z - boundary0;                     \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input1, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat3inputaxis3_NHWC4                                              \\\n" \
"  DOConcat2inputaxis3_NHWC4;                                                   \\\n" \
"  int boundary2 = boundary1 + input_shape2.w;                                  \\\n" \
"  if (Z >= boundary1 && Z < boundary2) {                                       \\\n" \
"    int coordinate_x = Y * input_shape2.w + Z - boundary1;                     \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input2, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat4inputaxis3_NHWC4                                              \\\n" \
"  DOConcat3inputaxis3_NHWC4;                                                   \\\n" \
"  int boundary3 = boundary2 + input_shape3.w;                                  \\\n" \
"  if (Z >= boundary2 && Z < boundary3) {                                       \\\n" \
"    int coordinate_x = Y * input_shape3.w + Z - boundary2;                     \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input3, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat5inputaxis3_NHWC4                                              \\\n" \
"  DOConcat4inputaxis3_NHWC4;                                                   \\\n" \
"  int boundary4 = boundary3 + input_shape4.w;                                  \\\n" \
"  if (Z >= boundary3 && Z < boundary4) {                                       \\\n" \
"    int coordinate_x = Y * input_shape4.w + Z - boundary3;                     \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input4, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define DOConcat6inputaxis3_NHWC4                                              \\\n" \
"  DOConcat5inputaxis3_NHWC4;                                                   \\\n" \
"  int boundary5 = boundary4 + input_shape5.w;                                  \\\n" \
"  if (Z >= boundary4 && Z < boundary5) {                                       \\\n" \
"    int coordinate_x = Y * input_shape5.w + Z - boundary4;                     \\\n" \
"    int coordinate_y = X;                                                      \\\n" \
"    result = READ_IMAGE(input5, smp_none, (int2)(coordinate_x, coordinate_y)); \\\n" \
"  }\n" \
"\n" \
"#define CONCAT6(Inputnum, Axis, ToFormat)                                                                      \\\n" \
"  __kernel void Concat##Inputnum##Axis##ToFormat(                                                              \\\n" \
"    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,                  \\\n" \
"    __read_only image2d_t input3, __read_only image2d_t input4, __read_only image2d_t input5,                  \\\n" \
"    __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2, int4 input_shape3, \\\n" \
"    int4 input_shape4, int4 input_shape5, int4 output_shape) {                                                 \\\n" \
"    CHECK_IDX;                                                                                                 \\\n" \
"    DOConcat##Inputnum##Axis##ToFormat;                                                                        \\\n" \
"    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                          \\\n" \
"  }\n" \
"\n" \
"#define CONCAT5(Inputnum, Axis, ToFormat)                                                                         \\\n" \
"  __kernel void Concat##Inputnum##Axis##ToFormat(                                                                 \\\n" \
"    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,                     \\\n" \
"    __read_only image2d_t input3, __read_only image2d_t input4, __write_only image2d_t output, int4 input_shape0, \\\n" \
"    int4 input_shape1, int4 input_shape2, int4 input_shape3, int4 input_shape4, int4 output_shape) {              \\\n" \
"    CHECK_IDX;                                                                                                    \\\n" \
"    DOConcat##Inputnum##Axis##ToFormat;                                                                           \\\n" \
"    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                             \\\n" \
"  }\n" \
"\n" \
"#define CONCAT4(Inputnum, Axis, ToFormat)                                                                             \\\n" \
"  __kernel void Concat##Inputnum##Axis##ToFormat(__read_only image2d_t input0, __read_only image2d_t input1,          \\\n" \
"                                                 __read_only image2d_t input2, __read_only image2d_t input3,          \\\n" \
"                                                 __write_only image2d_t output, int4 input_shape0, int4 input_shape1, \\\n" \
"                                                 int4 input_shape2, int4 input_shape3, int4 output_shape) {           \\\n" \
"    CHECK_IDX                                                                                                         \\\n" \
"    DOConcat##Inputnum##Axis##ToFormat;                                                                               \\\n" \
"    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                                 \\\n" \
"  }\n" \
"\n" \
"#define CONCAT3(Inputnum, Axis, ToFormat)                                                                        \\\n" \
"  __kernel void Concat##Inputnum##Axis##ToFormat(                                                                \\\n" \
"    __read_only image2d_t input0, __read_only image2d_t input1, __read_only image2d_t input2,                    \\\n" \
"    __write_only image2d_t output, int4 input_shape0, int4 input_shape1, int4 input_shape2, int4 output_shape) { \\\n" \
"    CHECK_IDX                                                                                                    \\\n" \
"    DOConcat##Inputnum##Axis##ToFormat;                                                                          \\\n" \
"    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                            \\\n" \
"  }\n" \
"\n" \
"#define CONCAT2(Inputnum, Axis, ToFormat)                                                                             \\\n" \
"  __kernel void Concat##Inputnum##Axis##ToFormat(__read_only image2d_t input0, __read_only image2d_t input1,          \\\n" \
"                                                 __write_only image2d_t output, int4 input_shape0, int4 input_shape1, \\\n" \
"                                                 int4 output_shape) {                                                 \\\n" \
"    CHECK_IDX                                                                                                         \\\n" \
"    DOConcat##Inputnum##Axis##ToFormat;                                                                               \\\n" \
"    WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);                                                 \\\n" \
"  }\n" \
"\n" \
"// axis = 1\n" \
"CONCAT6(6input, axis1, _NHWC4)\n" \
"CONCAT5(5input, axis1, _NHWC4)\n" \
"CONCAT4(4input, axis1, _NHWC4)\n" \
"CONCAT3(3input, axis1, _NHWC4)\n" \
"CONCAT2(2input, axis1, _NHWC4)\n" \
"\n" \
"// axis = 2\n" \
"CONCAT6(6input, axis2, _NHWC4)\n" \
"CONCAT5(5input, axis2, _NHWC4)\n" \
"CONCAT4(4input, axis2, _NHWC4)\n" \
"CONCAT3(3input, axis2, _NHWC4)\n" \
"CONCAT2(2input, axis2, _NHWC4)\n" \
"\n" \
"// axis = 3\n" \
"CONCAT6(6input, axis3, _NHWC4)\n" \
"CONCAT5(5input, axis3, _NHWC4)\n" \
"CONCAT4(4input, axis3, _NHWC4)\n" \
"CONCAT3(3input, axis3, _NHWC4)\n" \
"CONCAT2(2input, axis3, _NHWC4)\n" \
"\n" \
"// UnAlign in Axis C for concat\n" \
"#define CHECK_IDX_UNALIGN                                                                         \\\n" \
"  int X = get_global_id(0);                                                                       \\\n" \
"  int Y = get_global_id(1);                                                                       \\\n" \
"  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z) {                              \\\n" \
"    return;                                                                                       \\\n" \
"  }                                                                                               \\\n" \
"  int IN = X / output_shape.y, IH = X % output_shape.y;                                           \\\n" \
"  int IW = Y;                                                                                     \\\n" \
"  int Align_Shape0 = UP_DIV(input_shape0.w, C4NUM), Align_Shape1 = UP_DIV(input_shape1.w, C4NUM); \\\n" \
"  int Align_OutShape = output_shape.w;                                                            \\\n" \
"  int index_output = (IN * output_shape.y + IH) * stride_w + IW * Align_OutShape * C4NUM;\n" \
"\n" \
"int doconcat(__read_only image2d_t input, __global DTYPE *output, int Align_Shape, int4 input_shape, int IN, int IH,\n" \
"             int Y, int index_output) {\n" \
"  int Remainder = input_shape.w % C4NUM;\n" \
"  for (int i = 0; i < Align_Shape; ++i) {\n" \
"    DTYPE4 result = READ_IMAGE(input, smp_none, (int2)((Y * Align_Shape + i), (IN * input_shape.y + IH)));\n" \
"    DTYPE result_temp[4] = {result.x, result.y, result.z, result.w};\n" \
"    if ((i + 1) * C4NUM <= input_shape.w) {\n" \
"      for (int j = 0; j < C4NUM; ++j) {\n" \
"        output[index_output++] = result_temp[j];\n" \
"      }\n" \
"    } else {\n" \
"      for (int j = 0; j < Remainder; ++j) {\n" \
"        output[index_output++] = result_temp[j];\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"  return index_output;\n" \
"}\n" \
"\n" \
"__kernel void ConcatInput2UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                        __global DTYPE *output, int4 input_shape0, int4 input_shape1, int stride_w,\n" \
"                                        int4 output_shape) {\n" \
"  CHECK_IDX_UNALIGN;\n" \
"  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);\n" \
"}\n" \
"\n" \
"__kernel void ConcatInput3UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                        __read_only image2d_t input2, __global DTYPE *output, int4 input_shape0,\n" \
"                                        int4 input_shape1, int4 input_shape2, int stride_w, int4 output_shape) {\n" \
"  CHECK_IDX_UNALIGN;\n" \
"  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM);\n" \
"  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);\n" \
"}\n" \
"\n" \
"__kernel void ConcatInput4UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                        __read_only image2d_t input2, __read_only image2d_t input3,\n" \
"                                        __global DTYPE *output, int4 input_shape0, int4 input_shape1, int4 input_shape2,\n" \
"                                        int4 input_shape3, int stride_w, int4 output_shape) {\n" \
"  CHECK_IDX_UNALIGN;\n" \
"  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM), Align_Shape3 = UP_DIV(input_shape3.w, C4NUM);\n" \
"  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input3, output, Align_Shape3, input_shape3, IN, IH, Y, index_output);\n" \
"}\n" \
"\n" \
"__kernel void ConcatInput5UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                        __read_only image2d_t input2, __read_only image2d_t input3,\n" \
"                                        __read_only image2d_t input4, __global DTYPE *output, int4 input_shape0,\n" \
"                                        int4 input_shape1, int4 input_shape2, int4 input_shape3, int4 input_shape4,\n" \
"                                        int stride_w, int4 output_shape) {\n" \
"  CHECK_IDX_UNALIGN;\n" \
"  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM), Align_Shape3 = UP_DIV(input_shape3.w, C4NUM);\n" \
"  int Align_Shape4 = UP_DIV(input_shape4.w, C4NUM);\n" \
"  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input3, output, Align_Shape3, input_shape3, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input4, output, Align_Shape4, input_shape4, IN, IH, Y, index_output);\n" \
"}\n" \
"\n" \
"__kernel void ConcatInput6UnAlign_NHWC4(__read_only image2d_t input0, __read_only image2d_t input1,\n" \
"                                        __read_only image2d_t input2, __read_only image2d_t input3,\n" \
"                                        __read_only image2d_t input4, __read_only image2d_t input5,\n" \
"                                        __global DTYPE *output, int4 input_shape0, int4 input_shape1, int4 input_shape2,\n" \
"                                        int4 input_shape3, int4 input_shape4, int4 input_shape5, int stride_w,\n" \
"                                        int4 output_shape) {\n" \
"  CHECK_IDX_UNALIGN;\n" \
"  int Align_Shape2 = UP_DIV(input_shape2.w, C4NUM), Align_Shape3 = UP_DIV(input_shape3.w, C4NUM);\n" \
"  int Align_Shape4 = UP_DIV(input_shape4.w, C4NUM), Align_Shape5 = UP_DIV(input_shape5.w, C4NUM);\n" \
"  index_output = doconcat(input0, output, Align_Shape0, input_shape0, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input1, output, Align_Shape1, input_shape1, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input2, output, Align_Shape2, input_shape2, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input3, output, Align_Shape3, input_shape3, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input4, output, Align_Shape4, input_shape4, IN, IH, Y, index_output);\n" \
"  index_output = doconcat(input5, output, Align_Shape5, input_shape5, IN, IH, Y, index_output);\n" \
"}\n" \
;
