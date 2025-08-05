static const char *pad_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void Pad(__read_only image2d_t input, __write_only image2d_t output, int4 input_shape, int4 output_shape,\n" \
"                  int2 io_slices, int4 pad_before, float constant_value) {\n" \
"  int IN = input_shape.x, IH = input_shape.y, IW = input_shape.z, CI = input_shape.w;\n" \
"  int ON = output_shape.x, OH = output_shape.y, OW = output_shape.z, CO = output_shape.w;\n" \
"  int CI_SLICES = io_slices.x, CO_SLICES = io_slices.y;\n" \
"  int on_oh = get_global_id(0);\n" \
"  int ow = get_global_id(1);\n" \
"  int co_slice = get_global_id(2);\n" \
"  int on = on_oh / OH;\n" \
"  int oh = on_oh % OH;\n" \
"  if (on >= ON || oh >= OH || ow >= OW || co_slice >= CO_SLICES) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  int in = on - pad_before.x;\n" \
"  int ih = oh - pad_before.y;\n" \
"  int iw = ow - pad_before.z;\n" \
"  int ci = co_slice * 4 - pad_before.w;\n" \
"  if (in < 0 || in >= IN || ih < 0 || ih >= IH || iw < 0 || iw >= IW || ci + 3 < 0 || ci >= CI) {\n" \
"    WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), (FLT4)(constant_value));\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  int offset = ci % 4;\n" \
"  if (offset < 0) {\n" \
"    offset += 4;\n" \
"  }\n" \
"  FLT4 src0 = READ_IMAGE(input, smp_zero, (int2)(iw * CI_SLICES + ci / 4, in * IH + ih));\n" \
"  if (offset == 0 && ci >= 0 && ci + 3 < CI) {\n" \
"    WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), src0);\n" \
"    return;\n" \
"  }\n" \
"  FLT4 src1 = READ_IMAGE(input, smp_zero, (int2)(iw * CI_SLICES + (ci + 4) / 4, in * IH + ih));\n" \
"  FLT4 src_f4;\n" \
"  if (offset == 0) {\n" \
"    src_f4 = (FLT4)(src0.x, src0.y, src0.z, src0.w);\n" \
"  } else if (offset == 1) {\n" \
"    src_f4 = (FLT4)(src0.y, src0.z, src0.w, src1.x);\n" \
"  } else if (offset == 2) {\n" \
"    src_f4 = (FLT4)(src0.z, src0.w, src1.x, src1.y);\n" \
"  } else {  // if (offset==3)\n" \
"    src_f4 = (FLT4)(src0.w, src1.x, src1.y, src1.z);\n" \
"  }\n" \
"  FLT src[4] = {src_f4.x, src_f4.y, src_f4.z, src_f4.w};\n" \
"  FLT out[4] = {constant_value, constant_value, constant_value, constant_value};\n" \
"  for (int i = 0; i < 4; ++i) {\n" \
"    if (ci + i >= 0 && ci + i < CI) {\n" \
"      out[i] = src[i];\n" \
"    }\n" \
"  }\n" \
"  FLT4 out_f4 = (FLT4)(out[0], out[1], out[2], out[3]);\n" \
"  WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), out_f4);\n" \
"}\n" \
;
