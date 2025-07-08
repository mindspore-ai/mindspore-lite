static const char *fullconnection_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define C4NUM 4\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void FullConnection(__read_only image2d_t input, __write_only image2d_t output, __global FLT16 *weight,\n" \
"                             __read_only image2d_t bias, int N, int CI4, int CO4, int2 in_img_shape, int act_type) {\n" \
"  int gidx = get_global_id(0);  // CO4\n" \
"  int gidz = get_global_id(2);  // N\n" \
"  int lidx = get_local_id(0);\n" \
"  int lidy = get_local_id(1);\n" \
"  bool inside = gidx < CO4 && gidz < N;\n" \
"  FLT4 result = (FLT4)(0.0f);\n" \
"  for (uint i = lidy; i < CI4 && inside; i += 4) {\n" \
"    int index = gidz * CI4 + i;\n" \
"    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(index % in_img_shape.y, index / in_img_shape.y));\n" \
"    FLT16 w = weight[i * CO4 + gidx];\n" \
"    result.x += dot(v, w.s0123);\n" \
"    result.y += dot(v, w.s4567);\n" \
"    result.z += dot(v, w.s89ab);\n" \
"    result.w += dot(v, w.scdef);\n" \
"  }\n" \
"  __local FLT4 temp[32][4];\n" \
"  temp[lidx][lidy] = result;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (lidy == 0 && inside) {\n" \
"    result += temp[lidx][1];\n" \
"    result += temp[lidx][2];\n" \
"    result += temp[lidx][3];\n" \
"    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));\n" \
"    if (act_type == ActivationType_RELU) {\n" \
"      result = max(result, (FLT4)(0.0f));\n" \
"    } else if (act_type == ActivationType_RELU6) {\n" \
"      result = clamp(result, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    } else if (act_type == ActivationType_TANH) {\n" \
"      result = tanh(result);\n" \
"    }\n" \
"    WRITE_IMAGE(output, (int2)(gidx, gidz), result);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void FullConnectionWeightVar(__read_only image2d_t input, __write_only image2d_t output,\n" \
"                                      __read_only image2d_t weight, __read_only image2d_t bias, int N, int CI4, int CO4,\n" \
"                                      int2 in_img_shape, int act_type) {\n" \
"  int gidx = get_global_id(0);  // CO4\n" \
"  int gidz = get_global_id(2);  // N\n" \
"  int lidx = get_local_id(0);\n" \
"  int lidy = get_local_id(1);\n" \
"  bool inside = gidx < CO4 && gidz < N;\n" \
"  FLT4 result = (FLT4)(0.0f);\n" \
"  for (uint i = lidy; i < CI4 && inside; i += 4) {\n" \
"    int index = gidz * CI4 + i;\n" \
"    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(index % in_img_shape.y, index / in_img_shape.y));\n" \
"    FLT4 weight0 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4));\n" \
"    result.x += dot(v, weight0);\n" \
"    FLT4 weight1 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4 + 1));\n" \
"    result.y += dot(v, weight1);\n" \
"    FLT4 weight2 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4 + 2));\n" \
"    result.z += dot(v, weight2);\n" \
"    FLT4 weight3 = READ_IMAGE(weight, smp_zero, (int2)(i, gidx * 4 + 3));\n" \
"    result.w += dot(v, weight3);\n" \
"  }\n" \
"  __local FLT4 temp[32][4];\n" \
"  temp[lidx][lidy] = result;\n" \
"  barrier(CLK_LOCAL_MEM_FENCE);\n" \
"  if (lidy == 0 && inside) {\n" \
"    result += temp[lidx][1];\n" \
"    result += temp[lidx][2];\n" \
"    result += temp[lidx][3];\n" \
"    result += READ_IMAGE(bias, smp_zero, (int2)(gidx, 0));\n" \
"    if (act_type == ActivationType_RELU) {\n" \
"      result = max(result, (FLT4)(0.0f));\n" \
"    } else if (act_type == ActivationType_RELU6) {\n" \
"      result = clamp(result, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    } else if (act_type == ActivationType_TANH) {\n" \
"      result = tanh(result);\n" \
"    }\n" \
"    WRITE_IMAGE(output, (int2)(gidx, gidz), result);\n" \
"  }\n" \
"}\n" \
;
