static const char *strided_slice_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"\n" \
"__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n" \
"\n" \
"__kernel void strided_slice(__read_only image2d_t input, __write_only image2d_t output, int4 input_shape,\n" \
"                            int4 output_shape, int2 io_slices, int4 begin, int4 stride, int4 size) {\n" \
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
"  FLT tmp[4];\n" \
"  for (int i = 0; i < 4; ++i) {\n" \
"    // output_shape idx -> size idx. because squeeze(output_shape)=squeeze(size)\n" \
"    // for example:\n" \
"    // python code: B = A[1, 1:16, 2:16, 3:16]\n" \
"    // input_shape  = [16, 16, 16, 16]\n" \
"    // begin        = [ 1,  1,  2,  3]\n" \
"    // end          = [ 2, 16, 16, 16]\n" \
"    // stride       = [ 1,  1,  1,  1]\n" \
"    // size         = [ 1, 15, 14, 13] = ceil((end - begin) / stride)\n" \
"    // output_shape = [    15, 14, 13]\n" \
"    int idx = ((on * OH + oh) * OW + ow) * CO + co_slice * 4 + i;\n" \
"    int co_ = idx % size.w;\n" \
"    idx /= size.w;\n" \
"    int ow_ = idx % size.z;\n" \
"    idx /= size.z;\n" \
"    int oh_ = idx % size.y;\n" \
"    idx /= size.y;\n" \
"    int on_ = idx;\n" \
"\n" \
"    int in = begin.x + stride.x * on_;\n" \
"    int ih = begin.y + stride.y * oh_;\n" \
"    int iw = begin.z + stride.z * ow_;\n" \
"    int ci = begin.w + stride.w * co_;\n" \
"\n" \
"    FLT4 src = READ_IMAGE(input, smp_none, (int2)(iw * CI_SLICES + ci / 4, in * IH + ih));\n" \
"    int offset = ci % 4;\n" \
"    if (offset == 0) {\n" \
"      tmp[i] = src.x;\n" \
"    } else if (offset == 1) {\n" \
"      tmp[i] = src.y;\n" \
"    } else if (offset == 2) {\n" \
"      tmp[i] = src.z;\n" \
"    } else {\n" \
"      tmp[i] = src.w;\n" \
"    }\n" \
"  }\n" \
"\n" \
"  FLT4 out = (FLT4)(tmp[0], tmp[1], tmp[2], tmp[3]);\n" \
"  WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), out);\n" \
"}\n" \
;
