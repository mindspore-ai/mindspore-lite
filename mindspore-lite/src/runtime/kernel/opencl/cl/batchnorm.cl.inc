static const char *batchnorm_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define INT4 int4\n" \
"#define INT2 int2\n" \
"#define C4NUM 4\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"__kernel void Batch_normalization_NHWC4(__read_only image2d_t input, __global FLT *scale, __global FLT *offset,\n" \
"                                        __global FLT *mean, __global FLT *variance, __write_only image2d_t output,\n" \
"                                        const INT4 input_shape, float epsilon, int unalign_input_w) {\n" \
"  int X = get_global_id(0);  // H\n" \
"  int Y = get_global_id(1);  // W\n" \
"  int Z = get_global_id(2);  // C/4\n" \
"  if (X >= input_shape.y || Y >= input_shape.z || Z >= input_shape.w) {\n" \
"    return;\n" \
"  }\n" \
"  FLT4 result = READ_IMAGE(input, smp_none, (int2)((Y)*input_shape.w + Z, (X)));\n" \
"\n" \
"  FLT result_mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"  FLT result_var[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"  FLT result_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};\n" \
"  FLT result_offset[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"  if ((Z + 1) * C4NUM <= unalign_input_w) {\n" \
"    for (int i = 0; i < C4NUM; ++i) {\n" \
"      result_mean[i] = mean[Z * C4NUM + i];\n" \
"      result_var[i] = variance[Z * C4NUM + i];\n" \
"      result_scale[i] = scale[Z * C4NUM + i];\n" \
"      result_offset[i] = offset[Z * C4NUM + i];\n" \
"    }\n" \
"  } else {\n" \
"    for (int i = 0; i < unalign_input_w % C4NUM; ++i) {\n" \
"      result_mean[i] = mean[Z * C4NUM + i];\n" \
"      result_var[i] = variance[Z * C4NUM + i];\n" \
"      result_scale[i] = scale[Z * C4NUM + i];\n" \
"      result_offset[i] = offset[Z * C4NUM + i];\n" \
"    }\n" \
"  }\n" \
"  result.x = result_scale[0] * ((result.x - result_mean[0]) / sqrt(result_var[0] + epsilon)) + result_offset[0];\n" \
"  result.y = result_scale[1] * ((result.y - result_mean[1]) / sqrt(result_var[1] + epsilon)) + result_offset[1];\n" \
"  result.z = result_scale[2] * ((result.z - result_mean[2]) / sqrt(result_var[2] + epsilon)) + result_offset[2];\n" \
"  result.w = result_scale[3] * ((result.w - result_mean[3]) / sqrt(result_var[3] + epsilon)) + result_offset[3];\n" \
"  WRITE_IMAGE(output, (int2)((Y)*input_shape.w + Z, (X)), result);\n" \
"}\n" \
;
