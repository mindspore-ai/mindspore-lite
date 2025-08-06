static const char *conv2d_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"#define CI_TILE 4\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"\n" \
"#define DEFINE_ARGS                                                         \\\n" \
"  int N = input_shape.x;                                                    \\\n" \
"  int IH = input_shape.y, IW = input_shape.z, CI_SLICES = input_shape.w;    \\\n" \
"  int OH = output_shape.y, OW = output_shape.z, CO_SLICES = output_shape.w; \\\n" \
"  int KH = kernel_stride.x, KW = kernel_stride.y;                           \\\n" \
"  int strideH = kernel_stride.z, strideW = kernel_stride.w;                 \\\n" \
"  int padTop = pad.x, padBottom = pad.y, padLeft = pad.z, padRight = pad.w; \\\n" \
"  int dilationH = dilation.x, dilationW = dilation.y;                       \\\n" \
"                                                                            \\\n" \
"  int n_oh = get_global_id(0);                                              \\\n" \
"  int ow = get_global_id(1) * BlockW;                                       \\\n" \
"  int co_slice = get_global_id(2) * BlockC;                                 \\\n" \
"  int OH_SLICES = UP_DIV(OH, BlockH);                                       \\\n" \
"  int n = n_oh / OH_SLICES;                                                 \\\n" \
"  int oh = (n_oh % OH_SLICES) * BlockH;                                     \\\n" \
"  if (n >= N || oh >= OH || ow >= OW || co_slice >= CO_SLICES) {            \\\n" \
"    return;                                                                 \\\n" \
"  }\n" \
"\n" \
"#define DO_TANH(data) \\\n" \
"  exp0 = exp(data);   \\\n" \
"  exp1 = exp(-data);  \\\n" \
"  data = (exp0 - exp1) / (exp0 + exp1);\n" \
"\n" \
"#define DO_LEAKY_RELU(data, alpha)               \\\n" \
"  data.x = data.x > 0 ? data.x : data.x * alpha; \\\n" \
"  data.y = data.y > 0 ? data.y : data.y * alpha; \\\n" \
"  data.z = data.z > 0 ? data.z : data.z * alpha; \\\n" \
"  data.w = data.w > 0 ? data.w : data.w * alpha;\n" \
"\n" \
"__kernel void Conv2D_H1W1C1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                            int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 1;\n" \
"  const int BlockW = 1;\n" \
"  const int BlockC = 1;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int ow0 = ow + 0;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  for (int kh = 0; kh < KH; ++kh) {\n" \
"    int ih0 = kh * dilationH + oh0 * strideH - padTop;\n" \
"    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"\n" \
"    for (int kw = 0; kw < KW; ++kw) {\n" \
"      int iw0 = kw * dilationW + ow0 * strideW - padLeft;\n" \
"      int x_idx0 = iw0 * CI_SLICES;\n" \
"\n" \
"      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"        x_idx0++;\n" \
"\n" \
"        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"\n" \
"        weight_ptr += 4;\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"  }\n" \
"\n" \
"  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"  } else {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W1C1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                            int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 1;\n" \
"  const int BlockC = 1;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  for (int kh = 0; kh < KH; ++kh) {\n" \
"    int ih0 = kh * dilationH + oh0 * strideH - padTop;\n" \
"    // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"    int ih1 = kh * dilationH + oh1 * strideH - padTop;\n" \
"    // check ih0 and ih1\n" \
"    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"    for (int kw = 0; kw < KW; ++kw) {\n" \
"      int iw0 = kw * dilationW + ow0 * strideW - padLeft;\n" \
"      int x_idx0 = iw0 * CI_SLICES;\n" \
"\n" \
"      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"        x_idx0++;\n" \
"\n" \
"        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"        out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"        out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"        out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"        out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"\n" \
"        weight_ptr += 4;\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"  }\n" \
"\n" \
"  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"    }  // end if (oh1 < OH)\n" \
"  } else {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"    }  // end (oh1 < OH)\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W1C2(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                            int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 1;\n" \
"  const int BlockC = 2;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"  int co_slice1 = co_slice + 1;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  for (int kh = 0; kh < KH; ++kh) {\n" \
"    int ih0 = kh * dilationH + oh0 * strideH - padTop;\n" \
"    // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"    int ih1 = kh * dilationH + oh1 * strideH - padTop;\n" \
"    // check ih0 and ih1\n" \
"    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"    for (int kw = 0; kw < KW; ++kw) {\n" \
"      int iw0 = kw * dilationW + ow0 * strideW - padLeft;\n" \
"      int x_idx0 = iw0 * CI_SLICES;\n" \
"\n" \
"      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"        x_idx0++;\n" \
"\n" \
"        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"        out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"        out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"        out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"        out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"\n" \
"        out_h0_w0_c1 += weight_ptr[4] * in_h0_w0.x;\n" \
"        out_h1_w0_c1 += weight_ptr[4] * in_h1_w0.x;\n" \
"        out_h0_w0_c1 += weight_ptr[5] * in_h0_w0.y;\n" \
"        out_h1_w0_c1 += weight_ptr[5] * in_h1_w0.y;\n" \
"        out_h0_w0_c1 += weight_ptr[6] * in_h0_w0.z;\n" \
"        out_h1_w0_c1 += weight_ptr[6] * in_h1_w0.z;\n" \
"        out_h0_w0_c1 += weight_ptr[7] * in_h0_w0.w;\n" \
"        out_h1_w0_c1 += weight_ptr[7] * in_h1_w0.w;\n" \
"\n" \
"        weight_ptr += 8;\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w0_c1 += bias[co_slice1];\n" \
"    out_h1_w0_c1 += bias[co_slice1];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));\n" \
"    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h0_w0_c1);\n" \
"    DO_TANH(out_h1_w0_c1);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c1, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));\n" \
"    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));\n" \
"  }\n" \
"\n" \
"  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"    }  // end if (oh1 < OH)\n" \
"    if (co_slice1 < CO_SLICES) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);\n" \
"      if (oh1 < OH) {\n" \
"        WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);\n" \
"      }  // end if (oh1 < OH)\n" \
"    }    // end if (co_slice1 < CO_SLICES)\n" \
"  } else {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"    }  // end (oh1 < OH)\n" \
"    if (co_slice1 < CO_SLICES) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);\n" \
"      if (oh1 < OH) {\n" \
"        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);\n" \
"      }  // end if (oh1 < OH)\n" \
"    }    // end if (co_slice1 < CO_SLICES)\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W2C1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                            int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 2;\n" \
"  const int BlockC = 1;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int ow1 = ow + 1;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  for (int kh = 0; kh < KH; ++kh) {\n" \
"    int ih0 = kh * dilationH + oh0 * strideH - padTop;\n" \
"    // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"    int ih1 = kh * dilationH + oh1 * strideH - padTop;\n" \
"    // check ih0 and ih1\n" \
"    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"    for (int kw = 0; kw < KW; ++kw) {\n" \
"      int iw0 = kw * dilationW + ow0 * strideW - padLeft;\n" \
"      int iw1 = (ow1 < OW) ? kw * dilationW + ow1 * strideW - padLeft : -2;\n" \
"      int x_idx0 = iw0 * CI_SLICES;\n" \
"      int x_idx1 = iw1 * CI_SLICES;\n" \
"\n" \
"      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"        FLT4 in_h0_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx0));\n" \
"        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"        FLT4 in_h1_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx1));\n" \
"        x_idx0++;\n" \
"        x_idx1++;\n" \
"\n" \
"        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"        out_h0_w1_c0 += weight_ptr[0] * in_h0_w1.x;\n" \
"        out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"        out_h1_w1_c0 += weight_ptr[0] * in_h1_w1.x;\n" \
"        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"        out_h0_w1_c0 += weight_ptr[1] * in_h0_w1.y;\n" \
"        out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"        out_h1_w1_c0 += weight_ptr[1] * in_h1_w1.y;\n" \
"        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"        out_h0_w1_c0 += weight_ptr[2] * in_h0_w1.z;\n" \
"        out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"        out_h1_w1_c0 += weight_ptr[2] * in_h1_w1.z;\n" \
"        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"        out_h0_w1_c0 += weight_ptr[3] * in_h0_w1.w;\n" \
"        out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"        out_h1_w1_c0 += weight_ptr[3] * in_h1_w1.w;\n" \
"\n" \
"        weight_ptr += 4;\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w1_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w1_c0 += bias[co_slice0];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));\n" \
"\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h0_w1_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h1_w1_c0);\n" \
"\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c0, alpha);\n" \
"\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));\n" \
"  }\n" \
"\n" \
"  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh0), out_h0_w1_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh1), out_h1_w1_c0);\n" \
"    }  // end if (oh1 < OH)\n" \
"  } else {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);\n" \
"    }  // end (oh1 < OH)\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W2C2(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                            __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                            int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 2;\n" \
"  const int BlockC = 2;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int ow1 = ow + 1;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"  int co_slice1 = co_slice + 1;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  for (int kh = 0; kh < KH; ++kh) {\n" \
"    int ih0 = kh * dilationH + oh0 * strideH - padTop;\n" \
"    // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"    int ih1 = kh * dilationH + oh1 * strideH - padTop;\n" \
"    // check ih0 and ih1\n" \
"    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"    for (int kw = 0; kw < KW; ++kw) {\n" \
"      int iw0 = kw * dilationW + ow0 * strideW - padLeft;\n" \
"      int iw1 = (ow1 < OW) ? kw * dilationW + ow1 * strideW - padLeft : -2;\n" \
"      int x_idx0 = iw0 * CI_SLICES;\n" \
"      int x_idx1 = iw1 * CI_SLICES;\n" \
"\n" \
"      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"        FLT4 in_h0_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx0));\n" \
"        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"        FLT4 in_h1_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx1));\n" \
"        x_idx0++;\n" \
"        x_idx1++;\n" \
"\n" \
"        out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"        out_h0_w1_c0 += weight_ptr[0] * in_h0_w1.x;\n" \
"        out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"        out_h1_w1_c0 += weight_ptr[0] * in_h1_w1.x;\n" \
"        out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"        out_h0_w1_c0 += weight_ptr[1] * in_h0_w1.y;\n" \
"        out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"        out_h1_w1_c0 += weight_ptr[1] * in_h1_w1.y;\n" \
"        out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"        out_h0_w1_c0 += weight_ptr[2] * in_h0_w1.z;\n" \
"        out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"        out_h1_w1_c0 += weight_ptr[2] * in_h1_w1.z;\n" \
"        out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"        out_h0_w1_c0 += weight_ptr[3] * in_h0_w1.w;\n" \
"        out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"        out_h1_w1_c0 += weight_ptr[3] * in_h1_w1.w;\n" \
"\n" \
"        out_h0_w0_c1 += weight_ptr[4] * in_h0_w0.x;\n" \
"        out_h0_w1_c1 += weight_ptr[4] * in_h0_w1.x;\n" \
"        out_h1_w0_c1 += weight_ptr[4] * in_h1_w0.x;\n" \
"        out_h1_w1_c1 += weight_ptr[4] * in_h1_w1.x;\n" \
"        out_h0_w0_c1 += weight_ptr[5] * in_h0_w0.y;\n" \
"        out_h0_w1_c1 += weight_ptr[5] * in_h0_w1.y;\n" \
"        out_h1_w0_c1 += weight_ptr[5] * in_h1_w0.y;\n" \
"        out_h1_w1_c1 += weight_ptr[5] * in_h1_w1.y;\n" \
"        out_h0_w0_c1 += weight_ptr[6] * in_h0_w0.z;\n" \
"        out_h0_w1_c1 += weight_ptr[6] * in_h0_w1.z;\n" \
"        out_h1_w0_c1 += weight_ptr[6] * in_h1_w0.z;\n" \
"        out_h1_w1_c1 += weight_ptr[6] * in_h1_w1.z;\n" \
"        out_h0_w0_c1 += weight_ptr[7] * in_h0_w0.w;\n" \
"        out_h0_w1_c1 += weight_ptr[7] * in_h0_w1.w;\n" \
"        out_h1_w0_c1 += weight_ptr[7] * in_h1_w0.w;\n" \
"        out_h1_w1_c1 += weight_ptr[7] * in_h1_w1.w;\n" \
"\n" \
"        weight_ptr += 8;\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w1_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w1_c0 += bias[co_slice0];\n" \
"    out_h0_w0_c1 += bias[co_slice1];\n" \
"    out_h0_w1_c1 += bias[co_slice1];\n" \
"    out_h1_w0_c1 += bias[co_slice1];\n" \
"    out_h1_w1_c1 += bias[co_slice1];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));\n" \
"    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));\n" \
"    out_h0_w1_c1 = max(out_h0_w1_c1, (FLT4)(0.0f));\n" \
"    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));\n" \
"    out_h1_w1_c1 = max(out_h1_w1_c1, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c1 = clamp(out_h0_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c1 = clamp(out_h1_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h0_w1_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h1_w1_c0);\n" \
"    DO_TANH(out_h0_w0_c1);\n" \
"    DO_TANH(out_h0_w1_c1);\n" \
"    DO_TANH(out_h1_w0_c1);\n" \
"    DO_TANH(out_h1_w1_c1);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c1, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));\n" \
"    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));\n" \
"    out_h0_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c1));\n" \
"    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));\n" \
"    out_h1_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c1));\n" \
"  }\n" \
"\n" \
"  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh0), out_h0_w1_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh1), out_h1_w1_c0);\n" \
"    }  // end if (oh1 < OH)\n" \
"    if (co_slice1 < CO_SLICES) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh0), out_h0_w1_c1);\n" \
"      if (oh1 < OH) {\n" \
"        WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);\n" \
"        WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh1), out_h1_w1_c1);\n" \
"      }  // end if (oh1 < OH)\n" \
"    }    // end if (co_slice1 < CO_SLICES)\n" \
"  } else {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);\n" \
"    }  // end (oh1 < OH)\n" \
"    if (co_slice1 < CO_SLICES) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow1), out_h0_w1_c1);\n" \
"      if (oh1 < OH) {\n" \
"        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);\n" \
"        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow1), out_h1_w1_c1);\n" \
"      }  // end if (oh1 < OH)\n" \
"    }    // end if (co_slice1 < CO_SLICES)\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W2C2_Img(__read_only image2d_t input, __write_only image2d_t output,\n" \
"                                __read_only image2d_t weight, __global FLT4 *bias, int4 input_shape, int4 output_shape,\n" \
"                                int4 kernel_stride, int4 pad, int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 2;\n" \
"  const int BlockC = 2;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int ow1 = ow + 1;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"  int co_slice1 = co_slice + 1;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  int filter_offset = 0;\n" \
"  for (int kh = 0; kh < KH; ++kh) {\n" \
"    int ih0 = kh * dilationH + oh0 * strideH - padTop;\n" \
"    // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"    int ih1 = kh * dilationH + oh1 * strideH - padTop;\n" \
"    // check ih0 and ih1\n" \
"    int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"    int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"    for (int kw = 0; kw < KW; ++kw) {\n" \
"      int iw0 = kw * dilationW + ow0 * strideW - padLeft;\n" \
"      int iw1 = (ow1 < OW) ? kw * dilationW + ow1 * strideW - padLeft : -2;\n" \
"      int x_idx0 = iw0 * CI_SLICES;\n" \
"      int x_idx1 = iw1 * CI_SLICES;\n" \
"\n" \
"      for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"        FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"        FLT4 in_h0_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx0));\n" \
"        FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"        FLT4 in_h1_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx1));\n" \
"        x_idx0++;\n" \
"        x_idx1++;\n" \
"\n" \
"        FLT4 filter_ci0_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 0));\n" \
"        FLT4 filter_ci1_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 1));\n" \
"        FLT4 filter_ci2_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 2));\n" \
"        FLT4 filter_ci3_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 3));\n" \
"        FLT4 filter_ci0_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 0));\n" \
"        FLT4 filter_ci1_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 1));\n" \
"        FLT4 filter_ci2_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 2));\n" \
"        FLT4 filter_ci3_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 3));\n" \
"        filter_offset += 4;\n" \
"\n" \
"        out_h0_w0_c0 += filter_ci0_co0 * in_h0_w0.x;\n" \
"        out_h0_w1_c0 += filter_ci0_co0 * in_h0_w1.x;\n" \
"        out_h1_w0_c0 += filter_ci0_co0 * in_h1_w0.x;\n" \
"        out_h1_w1_c0 += filter_ci0_co0 * in_h1_w1.x;\n" \
"        out_h0_w0_c0 += filter_ci1_co0 * in_h0_w0.y;\n" \
"        out_h0_w1_c0 += filter_ci1_co0 * in_h0_w1.y;\n" \
"        out_h1_w0_c0 += filter_ci1_co0 * in_h1_w0.y;\n" \
"        out_h1_w1_c0 += filter_ci1_co0 * in_h1_w1.y;\n" \
"        out_h0_w0_c0 += filter_ci2_co0 * in_h0_w0.z;\n" \
"        out_h0_w1_c0 += filter_ci2_co0 * in_h0_w1.z;\n" \
"        out_h1_w0_c0 += filter_ci2_co0 * in_h1_w0.z;\n" \
"        out_h1_w1_c0 += filter_ci2_co0 * in_h1_w1.z;\n" \
"        out_h0_w0_c0 += filter_ci3_co0 * in_h0_w0.w;\n" \
"        out_h0_w1_c0 += filter_ci3_co0 * in_h0_w1.w;\n" \
"        out_h1_w0_c0 += filter_ci3_co0 * in_h1_w0.w;\n" \
"        out_h1_w1_c0 += filter_ci3_co0 * in_h1_w1.w;\n" \
"\n" \
"        out_h0_w0_c1 += filter_ci0_co1 * in_h0_w0.x;\n" \
"        out_h0_w1_c1 += filter_ci0_co1 * in_h0_w1.x;\n" \
"        out_h1_w0_c1 += filter_ci0_co1 * in_h1_w0.x;\n" \
"        out_h1_w1_c1 += filter_ci0_co1 * in_h1_w1.x;\n" \
"        out_h0_w0_c1 += filter_ci1_co1 * in_h0_w0.y;\n" \
"        out_h0_w1_c1 += filter_ci1_co1 * in_h0_w1.y;\n" \
"        out_h1_w0_c1 += filter_ci1_co1 * in_h1_w0.y;\n" \
"        out_h1_w1_c1 += filter_ci1_co1 * in_h1_w1.y;\n" \
"        out_h0_w0_c1 += filter_ci2_co1 * in_h0_w0.z;\n" \
"        out_h0_w1_c1 += filter_ci2_co1 * in_h0_w1.z;\n" \
"        out_h1_w0_c1 += filter_ci2_co1 * in_h1_w0.z;\n" \
"        out_h1_w1_c1 += filter_ci2_co1 * in_h1_w1.z;\n" \
"        out_h0_w0_c1 += filter_ci3_co1 * in_h0_w0.w;\n" \
"        out_h0_w1_c1 += filter_ci3_co1 * in_h0_w1.w;\n" \
"        out_h1_w0_c1 += filter_ci3_co1 * in_h1_w0.w;\n" \
"        out_h1_w1_c1 += filter_ci3_co1 * in_h1_w1.w;\n" \
"      }\n" \
"    }\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w1_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w1_c0 += bias[co_slice0];\n" \
"    out_h0_w0_c1 += bias[co_slice1];\n" \
"    out_h0_w1_c1 += bias[co_slice1];\n" \
"    out_h1_w0_c1 += bias[co_slice1];\n" \
"    out_h1_w1_c1 += bias[co_slice1];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));\n" \
"    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));\n" \
"    out_h0_w1_c1 = max(out_h0_w1_c1, (FLT4)(0.0f));\n" \
"    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));\n" \
"    out_h1_w1_c1 = max(out_h1_w1_c1, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c1 = clamp(out_h0_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c1 = clamp(out_h1_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h0_w1_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h1_w1_c0);\n" \
"    DO_TANH(out_h0_w0_c1);\n" \
"    DO_TANH(out_h0_w1_c1);\n" \
"    DO_TANH(out_h1_w0_c1);\n" \
"    DO_TANH(out_h1_w1_c1);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c1, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));\n" \
"    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));\n" \
"    out_h0_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c1));\n" \
"    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));\n" \
"    out_h1_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c1));\n" \
"  }\n" \
"\n" \
"  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh0), out_h0_w1_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh1), out_h1_w1_c0);\n" \
"    }  // end if (oh1 < OH)\n" \
"    if (co_slice1 < CO_SLICES) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh0), out_h0_w1_c1);\n" \
"      if (oh1 < OH) {\n" \
"        WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);\n" \
"        WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh1), out_h1_w1_c1);\n" \
"      }  // end if (oh1 < OH)\n" \
"    }    // end if (co_slice1 < CO_SLICES)\n" \
"  } else {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"      WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);\n" \
"    }  // end (oh1 < OH)\n" \
"    if (co_slice1 < CO_SLICES) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow1), out_h0_w1_c1);\n" \
"      if (oh1 < OH) {\n" \
"        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);\n" \
"        WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow1), out_h1_w1_c1);\n" \
"      }  // end if (oh1 < OH)\n" \
"    }    // end if (co_slice1 < CO_SLICES)\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H1W1C1_1x1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                                __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                                int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 1;\n" \
"  const int BlockW = 1;\n" \
"  const int BlockC = 1;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int ow0 = ow + 0;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  int ih0 = oh0 * strideH - padTop;\n" \
"  int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"\n" \
"  int iw0 = ow0 * strideW - padLeft;\n" \
"  int x_idx0 = iw0 * CI_SLICES;\n" \
"\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"    x_idx0++;\n" \
"\n" \
"    out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"    out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"    out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"    out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"\n" \
"    weight_ptr += 4;\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"  }\n" \
"\n" \
"  if (OW * CO_SLICES <= MAX_IMAGE2D_WIDTH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"  } else {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"  }\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W1C1_1x1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                                __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                                int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 1;\n" \
"  const int BlockC = 1;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  int ih0 = oh0 * strideH - padTop;\n" \
"  // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"  int ih1 = oh1 * strideH - padTop;\n" \
"  // check ih0 and ih1\n" \
"  int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"  int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"  int iw0 = ow0 * strideW - padLeft;\n" \
"  int x_idx0 = iw0 * CI_SLICES;\n" \
"\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"    FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"    x_idx0++;\n" \
"\n" \
"    out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"    out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"    out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"    out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"    out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"    out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"    out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"    out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"\n" \
"    weight_ptr += 4;\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"  }\n" \
"\n" \
"#ifndef EXCEDD_MAX_IMAGE2D_WIDTH\n" \
"  WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"  }  // end if (oh1 < OH)\n" \
"#else\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"  }  // end (oh1 < OH)\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W1C2_1x1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                                __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                                int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 1;\n" \
"  const int BlockC = 2;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"  int co_slice1 = co_slice + 1;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  int ih0 = oh0 * strideH - padTop;\n" \
"  // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"  int ih1 = oh1 * strideH - padTop;\n" \
"  // check ih0 and ih1\n" \
"  int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"  int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"  int iw0 = ow0 * strideW - padLeft;\n" \
"  int x_idx0 = iw0 * CI_SLICES;\n" \
"\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"    FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"    x_idx0++;\n" \
"\n" \
"    out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"    out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"    out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"    out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"    out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"    out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"    out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"    out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"\n" \
"    out_h0_w0_c1 += weight_ptr[4] * in_h0_w0.x;\n" \
"    out_h1_w0_c1 += weight_ptr[4] * in_h1_w0.x;\n" \
"    out_h0_w0_c1 += weight_ptr[5] * in_h0_w0.y;\n" \
"    out_h1_w0_c1 += weight_ptr[5] * in_h1_w0.y;\n" \
"    out_h0_w0_c1 += weight_ptr[6] * in_h0_w0.z;\n" \
"    out_h1_w0_c1 += weight_ptr[6] * in_h1_w0.z;\n" \
"    out_h0_w0_c1 += weight_ptr[7] * in_h0_w0.w;\n" \
"    out_h1_w0_c1 += weight_ptr[7] * in_h1_w0.w;\n" \
"\n" \
"    weight_ptr += 8;\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w0_c1 += bias[co_slice1];\n" \
"    out_h1_w0_c1 += bias[co_slice1];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));\n" \
"    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h0_w0_c1);\n" \
"    DO_TANH(out_h1_w0_c1);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c1, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));\n" \
"    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));\n" \
"  }\n" \
"\n" \
"#ifndef EXCEDD_MAX_IMAGE2D_WIDTH\n" \
"  WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"  }  // end if (oh1 < OH)\n" \
"  if (co_slice1 < CO_SLICES) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);\n" \
"    }  // end if (oh1 < OH)\n" \
"  }    // end if (co_slice1 < CO_SLICES)\n" \
"#else\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"  }  // end (oh1 < OH)\n" \
"  if (co_slice1 < CO_SLICES) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);\n" \
"    }  // end if (oh1 < OH)\n" \
"  }    // end if (co_slice1 < CO_SLICES)\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W2C1_1x1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                                __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                                int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 2;\n" \
"  const int BlockC = 1;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int ow1 = ow + 1;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  int ih0 = oh0 * strideH - padTop;\n" \
"  // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"  int ih1 = oh1 * strideH - padTop;\n" \
"  // check ih0 and ih1\n" \
"  int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"  int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"  int iw0 = ow0 * strideW - padLeft;\n" \
"  int iw1 = (ow1 < OW) ? ow1 * strideW - padLeft : -2;\n" \
"  int x_idx0 = iw0 * CI_SLICES;\n" \
"  int x_idx1 = iw1 * CI_SLICES;\n" \
"\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"    FLT4 in_h0_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx0));\n" \
"    FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"    FLT4 in_h1_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx1));\n" \
"    x_idx0++;\n" \
"    x_idx1++;\n" \
"\n" \
"    out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"    out_h0_w1_c0 += weight_ptr[0] * in_h0_w1.x;\n" \
"    out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"    out_h1_w1_c0 += weight_ptr[0] * in_h1_w1.x;\n" \
"    out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"    out_h0_w1_c0 += weight_ptr[1] * in_h0_w1.y;\n" \
"    out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"    out_h1_w1_c0 += weight_ptr[1] * in_h1_w1.y;\n" \
"    out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"    out_h0_w1_c0 += weight_ptr[2] * in_h0_w1.z;\n" \
"    out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"    out_h1_w1_c0 += weight_ptr[2] * in_h1_w1.z;\n" \
"    out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"    out_h0_w1_c0 += weight_ptr[3] * in_h0_w1.w;\n" \
"    out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"    out_h1_w1_c0 += weight_ptr[3] * in_h1_w1.w;\n" \
"\n" \
"    weight_ptr += 4;\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w1_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w1_c0 += bias[co_slice0];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h0_w1_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h1_w1_c0);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c0, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));\n" \
"  }\n" \
"\n" \
"#ifndef EXCEDD_MAX_IMAGE2D_WIDTH\n" \
"  WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"  WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh0), out_h0_w1_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh1), out_h1_w1_c0);\n" \
"  }  // end if (oh1 < OH)\n" \
"#else\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);\n" \
"  }  // end (oh1 < OH)\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W2C2_1x1(__read_only image2d_t input, __write_only image2d_t output, __global FLT4 *weight,\n" \
"                                __global FLT4 *bias, int4 input_shape, int4 output_shape, int4 kernel_stride, int4 pad,\n" \
"                                int2 dilation, int act_type, float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 2;\n" \
"  const int BlockC = 2;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int ow1 = ow + 1;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"  int co_slice1 = co_slice + 1;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  __global FLT4 *weight_ptr = weight + co_slice / BlockC * KH * KW * CI_SLICES * BlockC * CI_TILE;\n" \
"\n" \
"  int ih0 = oh0 * strideH - padTop;\n" \
"  // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"  int ih1 = oh1 * strideH - padTop;\n" \
"  // check ih0 and ih1\n" \
"  int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"  int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"  int iw0 = ow0 * strideW - padLeft;\n" \
"  int iw1 = (ow1 < OW) ? ow1 * strideW - padLeft : -2;\n" \
"  int x_idx0 = iw0 * CI_SLICES;\n" \
"  int x_idx1 = iw1 * CI_SLICES;\n" \
"\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"    FLT4 in_h0_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx0));\n" \
"    FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"    FLT4 in_h1_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx1));\n" \
"    x_idx0++;\n" \
"    x_idx1++;\n" \
"\n" \
"    out_h0_w0_c0 += weight_ptr[0] * in_h0_w0.x;\n" \
"    out_h0_w1_c0 += weight_ptr[0] * in_h0_w1.x;\n" \
"    out_h1_w0_c0 += weight_ptr[0] * in_h1_w0.x;\n" \
"    out_h1_w1_c0 += weight_ptr[0] * in_h1_w1.x;\n" \
"    out_h0_w0_c0 += weight_ptr[1] * in_h0_w0.y;\n" \
"    out_h0_w1_c0 += weight_ptr[1] * in_h0_w1.y;\n" \
"    out_h1_w0_c0 += weight_ptr[1] * in_h1_w0.y;\n" \
"    out_h1_w1_c0 += weight_ptr[1] * in_h1_w1.y;\n" \
"    out_h0_w0_c0 += weight_ptr[2] * in_h0_w0.z;\n" \
"    out_h0_w1_c0 += weight_ptr[2] * in_h0_w1.z;\n" \
"    out_h1_w0_c0 += weight_ptr[2] * in_h1_w0.z;\n" \
"    out_h1_w1_c0 += weight_ptr[2] * in_h1_w1.z;\n" \
"    out_h0_w0_c0 += weight_ptr[3] * in_h0_w0.w;\n" \
"    out_h0_w1_c0 += weight_ptr[3] * in_h0_w1.w;\n" \
"    out_h1_w0_c0 += weight_ptr[3] * in_h1_w0.w;\n" \
"    out_h1_w1_c0 += weight_ptr[3] * in_h1_w1.w;\n" \
"\n" \
"    out_h0_w0_c1 += weight_ptr[4] * in_h0_w0.x;\n" \
"    out_h0_w1_c1 += weight_ptr[4] * in_h0_w1.x;\n" \
"    out_h1_w0_c1 += weight_ptr[4] * in_h1_w0.x;\n" \
"    out_h1_w1_c1 += weight_ptr[4] * in_h1_w1.x;\n" \
"    out_h0_w0_c1 += weight_ptr[5] * in_h0_w0.y;\n" \
"    out_h0_w1_c1 += weight_ptr[5] * in_h0_w1.y;\n" \
"    out_h1_w0_c1 += weight_ptr[5] * in_h1_w0.y;\n" \
"    out_h1_w1_c1 += weight_ptr[5] * in_h1_w1.y;\n" \
"    out_h0_w0_c1 += weight_ptr[6] * in_h0_w0.z;\n" \
"    out_h0_w1_c1 += weight_ptr[6] * in_h0_w1.z;\n" \
"    out_h1_w0_c1 += weight_ptr[6] * in_h1_w0.z;\n" \
"    out_h1_w1_c1 += weight_ptr[6] * in_h1_w1.z;\n" \
"    out_h0_w0_c1 += weight_ptr[7] * in_h0_w0.w;\n" \
"    out_h0_w1_c1 += weight_ptr[7] * in_h0_w1.w;\n" \
"    out_h1_w0_c1 += weight_ptr[7] * in_h1_w0.w;\n" \
"    out_h1_w1_c1 += weight_ptr[7] * in_h1_w1.w;\n" \
"\n" \
"    weight_ptr += 8;\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w1_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w1_c0 += bias[co_slice0];\n" \
"    out_h0_w0_c1 += bias[co_slice1];\n" \
"    out_h0_w1_c1 += bias[co_slice1];\n" \
"    out_h1_w0_c1 += bias[co_slice1];\n" \
"    out_h1_w1_c1 += bias[co_slice1];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));\n" \
"    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));\n" \
"    out_h0_w1_c1 = max(out_h0_w1_c1, (FLT4)(0.0f));\n" \
"    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));\n" \
"    out_h1_w1_c1 = max(out_h1_w1_c1, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c1 = clamp(out_h0_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c1 = clamp(out_h1_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h0_w1_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h1_w1_c0);\n" \
"    DO_TANH(out_h0_w0_c1);\n" \
"    DO_TANH(out_h0_w1_c1);\n" \
"    DO_TANH(out_h1_w0_c1);\n" \
"    DO_TANH(out_h1_w1_c1);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c1, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));\n" \
"    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));\n" \
"    out_h0_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c1));\n" \
"    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));\n" \
"    out_h1_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c1));\n" \
"  }\n" \
"\n" \
"#ifndef EXCEDD_MAX_IMAGE2D_WIDTH\n" \
"  WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"  WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh0), out_h0_w1_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh1), out_h1_w1_c0);\n" \
"  }  // end if (oh1 < OH)\n" \
"  if (co_slice1 < CO_SLICES) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh0), out_h0_w1_c1);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh1), out_h1_w1_c1);\n" \
"    }  // end if (oh1 < OH)\n" \
"  }    // end if (co_slice1 < CO_SLICES)\n" \
"#else\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);\n" \
"  }  // end (oh1 < OH)\n" \
"  if (co_slice1 < CO_SLICES) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow1), out_h0_w1_c1);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow1), out_h1_w1_c1);\n" \
"    }  // end if (oh1 < OH)\n" \
"  }    // end if (co_slice1 < CO_SLICES)\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel void Conv2D_H2W2C2_Img_1x1(__read_only image2d_t input, __write_only image2d_t output,\n" \
"                                    __read_only image2d_t weight, __global FLT4 *bias, int4 input_shape,\n" \
"                                    int4 output_shape, int4 kernel_stride, int4 pad, int2 dilation, int act_type,\n" \
"                                    float alpha) {\n" \
"  const int BlockH = 2;\n" \
"  const int BlockW = 2;\n" \
"  const int BlockC = 2;\n" \
"  DEFINE_ARGS;\n" \
"\n" \
"  int oh0 = oh + 0;\n" \
"  int oh1 = oh + 1;\n" \
"  int n_oh0 = n * OH + oh0;\n" \
"  int n_oh1 = n * OH + oh1;\n" \
"  int ow0 = ow + 0;\n" \
"  int ow1 = ow + 1;\n" \
"  int co_slice0 = co_slice + 0;\n" \
"  int co_slice1 = co_slice + 1;\n" \
"\n" \
"  FLT4 out_h0_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h0_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w0_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out_h1_w1_c1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  int filter_offset = 0;\n" \
"  int ih0 = oh0 * strideH - padTop;\n" \
"  // no need to check oh1, finally write out will check (oh1 < OH)\n" \
"  int ih1 = oh1 * strideH - padTop;\n" \
"  // check ih0 and ih1\n" \
"  int y_idx0 = (ih0 >= 0 && ih0 < IH) ? n * IH + ih0 : -1;\n" \
"  int y_idx1 = (ih1 >= 0 && ih1 < IH) ? n * IH + ih1 : -1;\n" \
"\n" \
"  int iw0 = ow0 * strideW - padLeft;\n" \
"  int iw1 = (ow1 < OW) ? ow1 * strideW - padLeft : -2;\n" \
"  int x_idx0 = iw0 * CI_SLICES;\n" \
"  int x_idx1 = iw1 * CI_SLICES;\n" \
"\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in_h0_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx0));\n" \
"    FLT4 in_h0_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx0));\n" \
"    FLT4 in_h1_w0 = READ_IMAGE(input, smp_zero, (int2)(x_idx0, y_idx1));\n" \
"    FLT4 in_h1_w1 = READ_IMAGE(input, smp_zero, (int2)(x_idx1, y_idx1));\n" \
"    x_idx0++;\n" \
"    x_idx1++;\n" \
"\n" \
"    FLT4 filter_ci0_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 0));\n" \
"    FLT4 filter_ci1_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 1));\n" \
"    FLT4 filter_ci2_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 2));\n" \
"    FLT4 filter_ci3_co0 = READ_IMAGE(weight, smp_zero, (int2)(co_slice0, filter_offset + 3));\n" \
"    FLT4 filter_ci0_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 0));\n" \
"    FLT4 filter_ci1_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 1));\n" \
"    FLT4 filter_ci2_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 2));\n" \
"    FLT4 filter_ci3_co1 = READ_IMAGE(weight, smp_zero, (int2)(co_slice1, filter_offset + 3));\n" \
"    filter_offset += 4;\n" \
"\n" \
"    out_h0_w0_c0 += filter_ci0_co0 * in_h0_w0.x;\n" \
"    out_h0_w1_c0 += filter_ci0_co0 * in_h0_w1.x;\n" \
"    out_h1_w0_c0 += filter_ci0_co0 * in_h1_w0.x;\n" \
"    out_h1_w1_c0 += filter_ci0_co0 * in_h1_w1.x;\n" \
"    out_h0_w0_c0 += filter_ci1_co0 * in_h0_w0.y;\n" \
"    out_h0_w1_c0 += filter_ci1_co0 * in_h0_w1.y;\n" \
"    out_h1_w0_c0 += filter_ci1_co0 * in_h1_w0.y;\n" \
"    out_h1_w1_c0 += filter_ci1_co0 * in_h1_w1.y;\n" \
"    out_h0_w0_c0 += filter_ci2_co0 * in_h0_w0.z;\n" \
"    out_h0_w1_c0 += filter_ci2_co0 * in_h0_w1.z;\n" \
"    out_h1_w0_c0 += filter_ci2_co0 * in_h1_w0.z;\n" \
"    out_h1_w1_c0 += filter_ci2_co0 * in_h1_w1.z;\n" \
"    out_h0_w0_c0 += filter_ci3_co0 * in_h0_w0.w;\n" \
"    out_h0_w1_c0 += filter_ci3_co0 * in_h0_w1.w;\n" \
"    out_h1_w0_c0 += filter_ci3_co0 * in_h1_w0.w;\n" \
"    out_h1_w1_c0 += filter_ci3_co0 * in_h1_w1.w;\n" \
"\n" \
"    out_h0_w0_c1 += filter_ci0_co1 * in_h0_w0.x;\n" \
"    out_h0_w1_c1 += filter_ci0_co1 * in_h0_w1.x;\n" \
"    out_h1_w0_c1 += filter_ci0_co1 * in_h1_w0.x;\n" \
"    out_h1_w1_c1 += filter_ci0_co1 * in_h1_w1.x;\n" \
"    out_h0_w0_c1 += filter_ci1_co1 * in_h0_w0.y;\n" \
"    out_h0_w1_c1 += filter_ci1_co1 * in_h0_w1.y;\n" \
"    out_h1_w0_c1 += filter_ci1_co1 * in_h1_w0.y;\n" \
"    out_h1_w1_c1 += filter_ci1_co1 * in_h1_w1.y;\n" \
"    out_h0_w0_c1 += filter_ci2_co1 * in_h0_w0.z;\n" \
"    out_h0_w1_c1 += filter_ci2_co1 * in_h0_w1.z;\n" \
"    out_h1_w0_c1 += filter_ci2_co1 * in_h1_w0.z;\n" \
"    out_h1_w1_c1 += filter_ci2_co1 * in_h1_w1.z;\n" \
"    out_h0_w0_c1 += filter_ci3_co1 * in_h0_w0.w;\n" \
"    out_h0_w1_c1 += filter_ci3_co1 * in_h0_w1.w;\n" \
"    out_h1_w0_c1 += filter_ci3_co1 * in_h1_w0.w;\n" \
"    out_h1_w1_c1 += filter_ci3_co1 * in_h1_w1.w;\n" \
"  }\n" \
"\n" \
"  if (bias != 0) {\n" \
"    out_h0_w0_c0 += bias[co_slice0];\n" \
"    out_h0_w1_c0 += bias[co_slice0];\n" \
"    out_h1_w0_c0 += bias[co_slice0];\n" \
"    out_h1_w1_c0 += bias[co_slice0];\n" \
"    out_h0_w0_c1 += bias[co_slice1];\n" \
"    out_h0_w1_c1 += bias[co_slice1];\n" \
"    out_h1_w0_c1 += bias[co_slice1];\n" \
"    out_h1_w1_c1 += bias[co_slice1];\n" \
"  }\n" \
"\n" \
"  if (act_type == ActivationType_RELU) {\n" \
"    out_h0_w0_c0 = max(out_h0_w0_c0, (FLT4)(0.0f));\n" \
"    out_h0_w1_c0 = max(out_h0_w1_c0, (FLT4)(0.0f));\n" \
"    out_h1_w0_c0 = max(out_h1_w0_c0, (FLT4)(0.0f));\n" \
"    out_h1_w1_c0 = max(out_h1_w1_c0, (FLT4)(0.0f));\n" \
"    out_h0_w0_c1 = max(out_h0_w0_c1, (FLT4)(0.0f));\n" \
"    out_h0_w1_c1 = max(out_h0_w1_c1, (FLT4)(0.0f));\n" \
"    out_h1_w0_c1 = max(out_h1_w0_c1, (FLT4)(0.0f));\n" \
"    out_h1_w1_c1 = max(out_h1_w1_c1, (FLT4)(0.0f));\n" \
"  } else if (act_type == ActivationType_RELU6) {\n" \
"    out_h0_w0_c0 = clamp(out_h0_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c0 = clamp(out_h0_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c0 = clamp(out_h1_w0_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c0 = clamp(out_h1_w1_c0, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w0_c1 = clamp(out_h0_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h0_w1_c1 = clamp(out_h0_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w0_c1 = clamp(out_h1_w0_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"    out_h1_w1_c1 = clamp(out_h1_w1_c1, (FLT4)(0.0f), (FLT4)(6.0f));\n" \
"  } else if (act_type == ActivationType_TANH) {\n" \
"    FLT4 exp0, exp1;\n" \
"    DO_TANH(out_h0_w0_c0);\n" \
"    DO_TANH(out_h0_w1_c0);\n" \
"    DO_TANH(out_h1_w0_c0);\n" \
"    DO_TANH(out_h1_w1_c0);\n" \
"    DO_TANH(out_h0_w0_c1);\n" \
"    DO_TANH(out_h0_w1_c1);\n" \
"    DO_TANH(out_h1_w0_c1);\n" \
"    DO_TANH(out_h1_w1_c1);\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) {\n" \
"    DO_LEAKY_RELU(out_h0_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c0, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h0_w1_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w0_c1, alpha);\n" \
"    DO_LEAKY_RELU(out_h1_w1_c1, alpha);\n" \
"  } else if (act_type == ActivationType_SIGMOID) {\n" \
"    out_h0_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c0));\n" \
"    out_h0_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c0));\n" \
"    out_h1_w0_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c0));\n" \
"    out_h1_w1_c0 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c0));\n" \
"    out_h0_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w0_c1));\n" \
"    out_h0_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h0_w1_c1));\n" \
"    out_h1_w0_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w0_c1));\n" \
"    out_h1_w1_c1 = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-out_h1_w1_c1));\n" \
"  }\n" \
"\n" \
"#ifndef EXCEDD_MAX_IMAGE2D_WIDTH\n" \
"  WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh0), out_h0_w0_c0);\n" \
"  WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh0), out_h0_w1_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice0, n_oh1), out_h1_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice0, n_oh1), out_h1_w1_c0);\n" \
"  }  // end if (oh1 < OH)\n" \
"  if (co_slice1 < CO_SLICES) {\n" \
"    WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh0), out_h0_w0_c1);\n" \
"    WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh0), out_h0_w1_c1);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(ow0 * CO_SLICES + co_slice1, n_oh1), out_h1_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(ow1 * CO_SLICES + co_slice1, n_oh1), out_h1_w1_c1);\n" \
"    }  // end if (oh1 < OH)\n" \
"  }    // end if (co_slice1 < CO_SLICES)\n" \
"#else\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow0), out_h0_w0_c0);\n" \
"  WRITE_IMAGE(output, (int2)(co_slice0, n_oh0 * OW + ow1), out_h0_w1_c0);\n" \
"  if (oh1 < OH) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow0), out_h1_w0_c0);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice0, n_oh1 * OW + ow1), out_h1_w1_c0);\n" \
"  }  // end (oh1 < OH)\n" \
"  if (co_slice1 < CO_SLICES) {\n" \
"    WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow0), out_h0_w0_c1);\n" \
"    WRITE_IMAGE(output, (int2)(co_slice1, n_oh0 * OW + ow1), out_h0_w1_c1);\n" \
"    if (oh1 < OH) {\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow0), out_h1_w0_c1);\n" \
"      WRITE_IMAGE(output, (int2)(co_slice1, n_oh1 * OW + ow1), out_h1_w1_c1);\n" \
"    }  // end if (oh1 < OH)\n" \
"  }    // end if (co_slice1 < CO_SLICES)\n" \
"#endif\n" \
"}\n" \
;
