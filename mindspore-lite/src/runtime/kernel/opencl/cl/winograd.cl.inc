static const char *winograd_source ="\n" 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" \
"\n" \
"__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" \
"\n" \
"#define CI_TILE 4\n" \
"#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n" \
"\n" \
"constant FLT Bt[36] = {\n" \
"  1.0000000000f, 0.0000000000f,  -2.5000004768f, -0.0000001192f, 1.0000001192f,  0.0000000000f,\n" \
"  0.0000000000f, 0.9428091049f,  1.3333333731f,  -0.4714044929f, -0.6666667461f, 0.0000000000f,\n" \
"  0.0000000000f, -0.9428089857f, 1.3333334923f,  0.4714045525f,  -0.6666667461f, 0.0000000000f,\n" \
"  0.0000000000f, -0.1178511307f, -0.0833333358f, 0.2357022613f,  0.1666666865f,  0.0000000000f,\n" \
"  0.0000000000f, 0.1178511307f,  -0.0833333507f, -0.2357022911f, 0.1666666865f,  0.0000000000f,\n" \
"  0.0000000000f, 0.9999998808f,  -0.0000000596f, -2.5000000000f, 0.0000000000f,  1.0000000000f,\n" \
"};\n" \
"\n" \
"__kernel void Winograd4x4To36(__read_only image2d_t input,    // height=N*H             width=W*CI_SLICES\n" \
"                              __write_only image2d_t output,  // height=CI_SLICES*36    width=H/4*W/4\n" \
"                              int4 input_shape,               // N H W CI_SLICES\n" \
"                              int TILE_HW, int pad_u, int pad_l) {\n" \
"  int tile_hw = get_global_id(0);\n" \
"  int row = get_global_id(1);\n" \
"  int ci_slice = get_global_id(2);\n" \
"  int H = input_shape.y;\n" \
"  int W = input_shape.z;\n" \
"  int CI_SLICES = input_shape.w;\n" \
"  if (tile_hw >= TILE_HW || row >= 6 || ci_slice >= CI_SLICES) {\n" \
"    return;\n" \
"  }\n" \
"  int TILE_W = UP_DIV(W, 4);\n" \
"  int tile_w = tile_hw % TILE_W;\n" \
"  int tile_h = tile_hw / TILE_W;\n" \
"\n" \
"  constant FLT *Bt_row = Bt + row * 6;\n" \
"  FLT4 BtD_row[6] = {0};\n" \
"  int h = tile_h * 4 - pad_u;\n" \
"  int w = tile_w * 4 - pad_l;\n" \
"\n" \
"  int x_idx = w * CI_SLICES + ci_slice;\n" \
"  FLT bt0 = Bt_row[0], bt1 = Bt_row[1], bt2 = Bt_row[2], bt3 = Bt_row[3], bt4 = Bt_row[4], bt5 = Bt_row[5];\n" \
"  BtD_row[0] =\n" \
"    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +\n" \
"    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +\n" \
"    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));\n" \
"  x_idx += CI_SLICES;\n" \
"  BtD_row[1] =\n" \
"    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +\n" \
"    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +\n" \
"    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));\n" \
"  x_idx += CI_SLICES;\n" \
"  BtD_row[2] =\n" \
"    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +\n" \
"    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +\n" \
"    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));\n" \
"  x_idx += CI_SLICES;\n" \
"  BtD_row[3] =\n" \
"    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +\n" \
"    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +\n" \
"    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));\n" \
"  x_idx += CI_SLICES;\n" \
"  BtD_row[4] =\n" \
"    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +\n" \
"    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +\n" \
"    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));\n" \
"  x_idx += CI_SLICES;\n" \
"  BtD_row[5] =\n" \
"    bt0 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 0)) + bt1 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 1)) +\n" \
"    bt2 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 2)) + bt3 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 3)) +\n" \
"    bt4 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 4)) + bt5 * READ_IMAGE(input, smp_zero, (int2)(x_idx, h + 5));\n" \
"\n" \
"#if FP16_ENABLE\n" \
"#ifndef HALF_MAX  // adreno not exist\n" \
"#define HALF_MAX 0x1.ffcp15h\n" \
"#endif\n" \
"#define LimitAcc()          \\\n" \
"  acc = min(acc, HALF_MAX); \\\n" \
"  acc = max(acc, -HALF_MAX);\n" \
"#else\n" \
"#define LimitAcc() \\\n" \
"  {}\n" \
"#endif\n" \
"\n" \
"  int y_idx = ci_slice * 36 + row * 6;\n" \
"  FLT4 acc = BtD_row[0] + (FLT)(-2.5f) * BtD_row[2] + BtD_row[4];\n" \
"  LimitAcc();\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);\n" \
"\n" \
"  FLT4 tmp0 = (FLT)(0.9428091049f) * BtD_row[1] + (FLT)(-0.4714044929f) * BtD_row[3];\n" \
"  FLT4 tmp1 = (FLT)(1.3333333731f) * BtD_row[2] + (FLT)(-0.6666667461f) * BtD_row[4];\n" \
"  acc = tmp0 + tmp1;\n" \
"  LimitAcc();\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);\n" \
"\n" \
"  acc = -tmp0 + tmp1;\n" \
"  LimitAcc();\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);\n" \
"\n" \
"  tmp0 = (FLT)(-0.1178511307f) * BtD_row[1] + (FLT)(0.2357022613f) * BtD_row[3];\n" \
"  tmp1 = (FLT)(-0.0833333358f) * BtD_row[2] + (FLT)(0.1666666865f) * BtD_row[4];\n" \
"  acc = tmp0 + tmp1;\n" \
"  LimitAcc();\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);\n" \
"\n" \
"  acc = -tmp0 + tmp1;\n" \
"  LimitAcc();\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);\n" \
"\n" \
"  acc = BtD_row[1] + (FLT)(-2.5f) * BtD_row[3] + BtD_row[5];\n" \
"  LimitAcc();\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw, y_idx++), acc);\n" \
"}\n" \
"\n" \
"__kernel void WinogradConv2D(__read_only image2d_t input,    // height=CI_SLICES*36    width=TILE_HW\n" \
"                             __write_only image2d_t output,  // height=CO_SLICES*36    width=TILE_HW\n" \
"                             __global FLT16 *weight, int TILE_HW, int CI_SLICES, int CO_SLICES) {\n" \
"  int tile_hw = get_global_id(0) * 2;\n" \
"  int h = get_global_id(1);\n" \
"  int co_slice = get_global_id(2) * 2;\n" \
"  if (h >= 36 || tile_hw >= TILE_HW || co_slice >= CO_SLICES) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  FLT4 out00 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out01 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out10 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out11 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  int y_idx = h;\n" \
"  __global FLT16 *weight_ptr = weight + (co_slice / 2 * 36 + h) * CI_SLICES * 2;\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in0 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 0, y_idx));\n" \
"    FLT4 in1 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 1, y_idx));\n" \
"    y_idx += 36;\n" \
"\n" \
"    FLT16 weight0 = weight_ptr[0], weight1 = weight_ptr[1];\n" \
"    weight_ptr += 2;\n" \
"\n" \
"    out00 += in0.x * weight0.s0123;\n" \
"    out00 += in0.y * weight0.s4567;\n" \
"    out00 += in0.z * weight0.s89ab;\n" \
"    out00 += in0.w * weight0.scdef;\n" \
"\n" \
"    out01 += in1.x * weight0.s0123;\n" \
"    out01 += in1.y * weight0.s4567;\n" \
"    out01 += in1.z * weight0.s89ab;\n" \
"    out01 += in1.w * weight0.scdef;\n" \
"\n" \
"    out10 += in0.x * weight1.s0123;\n" \
"    out10 += in0.y * weight1.s4567;\n" \
"    out10 += in0.z * weight1.s89ab;\n" \
"    out10 += in0.w * weight1.scdef;\n" \
"\n" \
"    out11 += in1.x * weight1.s0123;\n" \
"    out11 += in1.y * weight1.s4567;\n" \
"    out11 += in1.z * weight1.s89ab;\n" \
"    out11 += in1.w * weight1.scdef;\n" \
"  }\n" \
"\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 0) * 36 + h), out00);\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 0) * 36 + h), out01);\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 1) * 36 + h), out10);\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 1) * 36 + h), out11);\n" \
"}\n" \
"\n" \
"__kernel void WinogradConv2D_Img(__read_only image2d_t input,    // height=CI_SLICES*36    width=TILE_HW\n" \
"                                 __write_only image2d_t output,  // height=CO_SLICES*36    width=TILE_HW\n" \
"                                 __read_only image2d_t weight, int TILE_HW, int CI_SLICES, int CO_SLICES) {\n" \
"  int tile_hw = get_global_id(0) * 2;\n" \
"  int h = get_global_id(1);\n" \
"  int co_slice = get_global_id(2) * 2;\n" \
"  if (h >= 36 || tile_hw >= TILE_HW || co_slice >= CO_SLICES) {\n" \
"    return;\n" \
"  }\n" \
"  int CI_ALIGN = CI_SLICES * CI_TILE;\n" \
"\n" \
"  FLT4 out00 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out01 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out10 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"  FLT4 out11 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n" \
"\n" \
"  int y_idx = h;\n" \
"  for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++) {\n" \
"    FLT4 in0 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 0, y_idx));\n" \
"    FLT4 in1 = READ_IMAGE(input, smp_zero, (int2)(tile_hw + 1, y_idx));\n" \
"    y_idx += 36;\n" \
"\n" \
"    FLT4 filter_ci0_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 0, co_slice + 0));\n" \
"    FLT4 filter_ci1_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 1, co_slice + 0));\n" \
"    FLT4 filter_ci2_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 2, co_slice + 0));\n" \
"    FLT4 filter_ci3_co0 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 3, co_slice + 0));\n" \
"    FLT4 filter_ci0_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 0, co_slice + 1));\n" \
"    FLT4 filter_ci1_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 1, co_slice + 1));\n" \
"    FLT4 filter_ci2_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 2, co_slice + 1));\n" \
"    FLT4 filter_ci3_co1 = READ_IMAGE(weight, smp_zero, (int2)(h * CI_ALIGN + ci_slice * CI_TILE + 3, co_slice + 1));\n" \
"\n" \
"    out00 += in0.x * filter_ci0_co0;\n" \
"    out00 += in0.y * filter_ci1_co0;\n" \
"    out00 += in0.z * filter_ci2_co0;\n" \
"    out00 += in0.w * filter_ci3_co0;\n" \
"\n" \
"    out01 += in1.x * filter_ci0_co0;\n" \
"    out01 += in1.y * filter_ci1_co0;\n" \
"    out01 += in1.z * filter_ci2_co0;\n" \
"    out01 += in1.w * filter_ci3_co0;\n" \
"\n" \
"    out10 += in0.x * filter_ci0_co1;\n" \
"    out10 += in0.y * filter_ci1_co1;\n" \
"    out10 += in0.z * filter_ci2_co1;\n" \
"    out10 += in0.w * filter_ci3_co1;\n" \
"\n" \
"    out11 += in1.x * filter_ci0_co1;\n" \
"    out11 += in1.y * filter_ci1_co1;\n" \
"    out11 += in1.z * filter_ci2_co1;\n" \
"    out11 += in1.w * filter_ci3_co1;\n" \
"  }\n" \
"\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 0) * 36 + h), out00);\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 0) * 36 + h), out01);\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 0, (co_slice + 1) * 36 + h), out10);\n" \
"  WRITE_IMAGE(output, (int2)(tile_hw + 1, (co_slice + 1) * 36 + h), out11);\n" \
"}\n" \
"\n" \
"#define DO_LEAKY_RELU(data, alpha)               \\\n" \
"  data.x = data.x > 0 ? data.x : data.x * alpha; \\\n" \
"  data.y = data.y > 0 ? data.y : data.y * alpha; \\\n" \
"  data.z = data.z > 0 ? data.z : data.z * alpha; \\\n" \
"  data.w = data.w > 0 ? data.w : data.w * alpha;\n" \
"\n" \
"constant FLT At[24] = {1.0000000000f, 1.0000000000f, 1.0000000000f,  1.0000000000f, 1.0000000000f,  0.0000000000f,\n" \
"                       0.0000000000f, 0.7071067691f, -0.7071067691f, 1.4142135382f, -1.4142135382f, 0.0000000000f,\n" \
"                       0.0000000000f, 0.4999999702f, 0.4999999702f,  1.9999998808f, 1.9999998808f,  0.0000000000f,\n" \
"                       0.0000000000f, 0.3535533845f, -0.3535533845f, 2.8284270763f, -2.8284270763f, 1.0000000000f};\n" \
"\n" \
"#define UpdateAcc()                                   \\\n" \
"  if (bias != 0) acc += bias[co_slice];               \\\n" \
"  if (act_type == ActivationType_RELU) {              \\\n" \
"    acc = max(acc, (FLT4)(0.0f));                     \\\n" \
"  } else if (act_type == ActivationType_RELU6) {      \\\n" \
"    acc = clamp(acc, (FLT4)(0.0f), (FLT4)(6.0f));     \\\n" \
"  } else if (act_type == ActivationType_TANH) {       \\\n" \
"    FLT4 exp0 = exp(acc);                             \\\n" \
"    FLT4 exp1 = exp(-acc);                            \\\n" \
"    acc = (exp0 - exp1) / (exp0 + exp1);              \\\n" \
"  } else if (act_type == ActivationType_LEAKY_RELU) { \\\n" \
"    DO_LEAKY_RELU(acc, alpha);                        \\\n" \
"  } else if (act_type == ActivationType_SIGMOID) {    \\\n" \
"    acc = (FLT4)(1.f) / ((FLT4)(1.f) + exp(-acc));    \\\n" \
"  }\n" \
"\n" \
"__kernel void Winograd36To4x4(__read_only image2d_t input,    // height=CO_SLICES*36    width=TILE_HW\n" \
"                              __write_only image2d_t output,  // height=N*H             width=W*CO_SLICES\n" \
"                              __global FLT4 *bias,\n" \
"                              int4 output_shape,  // N H W CO_SLICES\n" \
"                              int TILE_HW, int act_type, float alpha) {\n" \
"  int tile_hw = get_global_id(0);\n" \
"  int row = get_global_id(1);\n" \
"  int co_slice = get_global_id(2);\n" \
"  int H = output_shape.y;\n" \
"  int W = output_shape.z;\n" \
"  int CO_SLICES = output_shape.w;\n" \
"  if (tile_hw >= TILE_HW || row >= 4 || co_slice >= CO_SLICES) {\n" \
"    return;\n" \
"  }\n" \
"\n" \
"  constant FLT *At_row = At + row * 6;\n" \
"  FLT4 AtM_row[6] = {0};\n" \
"  int idx = co_slice * 36;\n" \
"  FLT at = At_row[0];\n" \
"  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  at = At_row[1];\n" \
"  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  at = At_row[2];\n" \
"  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  at = At_row[3];\n" \
"  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  at = At_row[4];\n" \
"  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  at = At_row[5];\n" \
"  AtM_row[0] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[1] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[2] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[3] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[4] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"  AtM_row[5] += at * READ_IMAGE(input, smp_zero, (int2)(tile_hw, idx++));\n" \
"\n" \
"  int TILE_W = UP_DIV(W, 4);\n" \
"  int tile_w = tile_hw % TILE_W;\n" \
"  int tile_h = tile_hw / TILE_W;\n" \
"  int h = tile_h * 4 + row;\n" \
"  int w = tile_w * 4;\n" \
"  int x_idx = w * CO_SLICES + co_slice;\n" \
"\n" \
"  FLT4 acc = AtM_row[0] + AtM_row[1] + AtM_row[2] + AtM_row[3] + AtM_row[4];\n" \
"  UpdateAcc();\n" \
"  WRITE_IMAGE(output, (int2)(x_idx, h), acc);\n" \
"  x_idx += CO_SLICES;\n" \
"\n" \
"  acc = (FLT)(0.7071067691f) * (AtM_row[1] - AtM_row[2]) + (FLT)(1.4142135382f) * (AtM_row[3] - AtM_row[4]);\n" \
"  UpdateAcc();\n" \
"  WRITE_IMAGE(output, (int2)(x_idx, h), acc);\n" \
"  x_idx += CO_SLICES;\n" \
"\n" \
"  acc = (FLT)(0.5f) * (AtM_row[1] + AtM_row[2]) + (FLT)(2.0f) * (AtM_row[3] + AtM_row[4]);\n" \
"  UpdateAcc();\n" \
"  WRITE_IMAGE(output, (int2)(x_idx, h), acc);\n" \
"  x_idx += CO_SLICES;\n" \
"\n" \
"  acc =\n" \
"    (FLT)(0.3535533845f) * (AtM_row[1] - AtM_row[2]) + (FLT)(2.8284270763f) * (AtM_row[3] - AtM_row[4]) + AtM_row[5];\n" \
"  UpdateAcc();\n" \
"  WRITE_IMAGE(output, (int2)(x_idx, h), acc);\n" \
"}\n" \
;
