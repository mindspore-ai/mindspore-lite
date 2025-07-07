static const char *layer_norm_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"#define C4NUM 4\n" \
"\n" \
"__kernel void ComputeMeanVarAxis3NHWC4(__read_only image2d_t src_data, __global FLT *mean_, __global FLT *variance_,\n" \
"                                       int4 in_shape, int normalized_shape_size) {\n" \
"  int X = get_global_id(0);  // n*h\n" \
"  int Y = get_global_id(1);  // w\n" \
"  if (X > in_shape.x * in_shape.y || Y > in_shape.z || in_shape.y == 0 || normalized_shape_size == 0) {\n" \
"    return;\n" \
"  }\n" \
"  int n = X / in_shape.y;\n" \
"  int h = X % in_shape.y;\n" \
"  int w = Y;\n" \
"  int ci4 = UP_DIV(in_shape.w, C4NUM);\n" \
"  int remainder = in_shape.w % C4NUM;\n" \
"  FLT4 mean_temp = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"  FLT4 var_temp = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"  FLT mean = 0.0f;\n" \
"  FLT var = 0.0f;\n" \
"\n" \
"  // compute mean\n" \
"  for (int i = 0; i < ci4; ++i) {\n" \
"    FLT4 result_temp = READ_IMAGE(src_data, smp_none, (int2)(w * ci4 + i, n * in_shape.y + h));\n" \
"    mean_temp += result_temp;\n" \
"  }\n" \
"  mean = (mean_temp.x + mean_temp.y + mean_temp.z + mean_temp.w) / normalized_shape_size;\n" \
"  mean_temp.x = mean_temp.y = mean_temp.z = mean_temp.w = mean;\n" \
"\n" \
"  // compute var\n" \
"  for (int i = 0; i < ci4; ++i) {\n" \
"    FLT4 result_temp = READ_IMAGE(src_data, smp_none, (int2)(w * ci4 + i, n * in_shape.y + h));\n" \
"    if ((i + 1) * C4NUM <= in_shape.w) {\n" \
"      var_temp += (result_temp - mean_temp) * (result_temp - mean_temp);\n" \
"    } else {\n" \
"      if (remainder == 1) {\n" \
"        mean_temp.x = mean;\n" \
"        mean_temp.y = mean_temp.z = mean_temp.w = 0.0f;\n" \
"      } else if (remainder == 2) {\n" \
"        mean_temp.x = mean_temp.y = mean;\n" \
"        mean_temp.z = mean_temp.w = 0.0f;\n" \
"      } else {\n" \
"        mean_temp.x = mean_temp.y = mean_temp.z = mean;\n" \
"        mean_temp.w = 0.0f;\n" \
"      }\n" \
"      var_temp += (result_temp - mean_temp) * (result_temp - mean_temp);\n" \
"    }\n" \
"  }\n" \
"  var = (var_temp.x + var_temp.y + var_temp.z + var_temp.w) / normalized_shape_size;\n" \
"\n" \
"  // write result to dst\n" \
"  int position = (n * in_shape.y + h) * in_shape.z + w;\n" \
"  mean_[position] = mean;\n" \
"  variance_[position] = var;\n" \
"}\n" \
"\n" \
"__kernel void LayerNormalization_NHWC4(__read_only image2d_t src_data, __write_only image2d_t dst_data,\n" \
"                                       __global FLT *mean_, __global FLT *variance_, __global FLT *gamma_,\n" \
"                                       __global FLT *beta_, int4 in_shape, float epsilon_, int begin_params_axis_) {\n" \
"  int X = get_global_id(0);  // n*h\n" \
"  int Y = get_global_id(1);  // w\n" \
"  int Z = get_global_id(2);  // c4\n" \
"  if (X >= in_shape.x * in_shape.y || Y >= in_shape.z || Z >= in_shape.w || in_shape.y == 0) {\n" \
"    return;\n" \
"  }\n" \
"  int n = X / in_shape.y;\n" \
"  int h = X % in_shape.y;\n" \
"  int w = Y;\n" \
"  int c = Z;\n" \
"  int ci4 = UP_DIV(in_shape.w, C4NUM);\n" \
"  int postion_mv = 0;\n" \
"  int postion_gb = 0;\n" \
"  if (begin_params_axis_ == 1) {\n" \
"    postion_mv = n;\n" \
"    postion_gb = (h * in_shape.z + w) * ci4 * C4NUM + c * C4NUM;\n" \
"  } else if (begin_params_axis_ == 2) {\n" \
"    postion_mv = n * in_shape.y + h;\n" \
"    postion_gb = w * ci4 * C4NUM + c * C4NUM;\n" \
"  } else if (begin_params_axis_ == 3) {\n" \
"    postion_mv = (n * in_shape.y + h) * in_shape.z + w;\n" \
"    postion_gb = c * C4NUM;\n" \
"  }\n" \
"  FLT4 result = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"  FLT4 result_in = READ_IMAGE(src_data, smp_none, (int2)(w * ci4 + c, n * in_shape.y + h));\n" \
"  result.x = ((result_in.x - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb] +\n" \
"             beta_[postion_gb];\n" \
"  result.y = ((result_in.y - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb + 1] +\n" \
"             beta_[postion_gb + 1];\n" \
"  result.z = ((result_in.z - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb + 2] +\n" \
"             beta_[postion_gb + 2];\n" \
"  result.w = ((result_in.w - mean_[postion_mv]) / sqrt(variance_[postion_mv] + epsilon_)) * gamma_[postion_gb + 3] +\n" \
"             beta_[postion_gb + 3];\n" \
"  WRITE_IMAGE(dst_data, (int2)((w * ci4 + c), (n * in_shape.y + h)), result);\n" \
"}\n" \
;
