static const char *sparse_to_dense_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"#define C4NUM 4\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void SparseToDenseScalar(__read_only image2d_t input, __global DTYPE *output, float weight, int2 inputshape,\n" \
"                                  int4 outputshape, float default_value, int stride_w, int inshapeindex1_dim) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= inputshape.x || Y >= inputshape.y) {\n" \
"    return;\n" \
"  }\n" \
"  int4 index_input = read_imagei(input, smp_zero, (int2)(Y, X));\n" \
"  int4 index_input_int = *((int4 *)&index_input);\n" \
"  int index = 0;\n" \
"  if (inshapeindex1_dim == 1) {\n" \
"    index = (index_input_int.x) * stride_w;\n" \
"  } else if (inshapeindex1_dim == 2) {\n" \
"    index = (index_input_int.x) * stride_w + (index_input_int.y);\n" \
"  } else if (inshapeindex1_dim == 3) {\n" \
"    index = (index_input_int.x) * stride_w + (index_input_int.y) * outputshape.w * C4NUM + (index_input_int.z);\n" \
"  } else {\n" \
"    index = (index_input_int.x) * outputshape.y * stride_w + (index_input_int.y) * stride_w +\n" \
"            (index_input_int.z) * outputshape.w * C4NUM + index_input_int.w;\n" \
"  }\n" \
"  output[index] = weight;\n" \
"}\n" \
"\n" \
"__kernel void SparseToDenseVector(__read_only image2d_t input, __global DTYPE *output, __global float *weight_vector,\n" \
"                                  int2 inputshape, int4 outputshape, float default_value, int stride_w,\n" \
"                                  int inshapeindex1_dim) {\n" \
"  int X = get_global_id(0);\n" \
"  int Y = get_global_id(1);\n" \
"  if (X >= inputshape.x || Y >= inputshape.y) {\n" \
"    return;\n" \
"  }\n" \
"  int4 index_input = read_imagei(input, smp_zero, (int2)(Y, X));\n" \
"  int4 index_input_int = *((int4 *)&index_input);\n" \
"  int index = 0;\n" \
"  if (inshapeindex1_dim == 1) {\n" \
"    index = (index_input_int.x) * stride_w;\n" \
"  } else if (inshapeindex1_dim == 2) {\n" \
"    index = (index_input_int.x) * stride_w + index_input_int.y;\n" \
"  } else if (inshapeindex1_dim == 3) {\n" \
"    index = (index_input_int.x) * stride_w + (index_input_int.y) * outputshape.w * C4NUM + index_input_int.z;\n" \
"  } else {\n" \
"    index = (index_input_int.x) * outputshape.y * stride_w + (index_input_int.y) * stride_w +\n" \
"            (index_input_int.z) * outputshape.w * C4NUM + index_input_int.w;\n" \
"  }\n" \
"  output[index] = weight_vector[X];\n" \
"}\n" \
;
