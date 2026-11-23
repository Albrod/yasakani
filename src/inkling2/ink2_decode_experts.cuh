// SPDX-License-Identifier: Apache-2.0
// NVFP4 down projection for short runs of Inkling's 1152-wide routed experts.
// A half warp owns one output and caches its five weight groups in registers
// across all the tokens routed to that expert. There is no block-wide barrier.
namespace {
__global__ void decode_expert_down(ExpMArgs a) {
  const int expert_index = blockIdx.y;
  if (expert_index >= *a.gcount) return;
  const int expert = a.gexp[expert_index];
  const int channel = blockIdx.x * 16 + threadIdx.x / 16;
  const int lane = threadIdx.x % 16;
  constexpr int width = 1152;
  constexpr int groups = width / 16;
  const size_t row = static_cast<size_t>(expert) * a.rows + channel;
  const u64* weights = reinterpret_cast<const u64*>(a.w) + row * groups;
  const u8* scales = a.sc + row * groups;
  u64 codes[5];
  float scale[5];
#pragma unroll
  for (int slice = 0; slice < 5; ++slice) {
    const int unit = slice * 16 + lane;
    if (unit < groups) {
      codes[slice] = weights[unit];
      scale[slice] = a.e4m3[scales[unit]];
    }
  }
  for (int i = a.goff[expert_index]; i < a.goff[expert_index + 1]; ++i) {
    const int pair = a.gpairs[i];
    const float* x = a.x + static_cast<size_t>(pair) * width;
    float sum = 0;
#pragma unroll
    for (int slice = 0; slice < 5; ++slice) {
      const int unit = slice * 16 + lane;
      if (unit >= groups) continue;
      const float4* input = reinterpret_cast<const float4*>(x + unit * 16);
      float d[16];
#pragma unroll
      for (int j = 0; j < 16; ++j)
        d[j] = k_fp4x2(static_cast<unsigned>(codes[slice] >> (4 * j)));
      const float part = k_dot16(d, input[0], input[1], input[2], input[3]);
      sum = fmaf(part, scale[slice], sum);
    }
    sum = k_half_sum(sum);
    if (!lane) a.y[static_cast<size_t>(pair) * a.rows + channel] = sum * a.factor[expert];
  }
}
}  // namespace
