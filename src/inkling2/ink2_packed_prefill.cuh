// SPDX-License-Identifier: Apache-2.0
// Wide prompt products. A block keeps eight input rows in registers while it
// walks successive output-row tiles. Retain pgemvcm's eight FMAs per thread,
// warp shuffle tree and ordered sum of warps.
namespace {
constexpr int prompt_out = 16;
constexpr int prompt_in = 8;
constexpr int prompt_warps = 12;

template<int Format>
__device__ __forceinline__ void prompt_tile(const GemvJob& job, int out0, int in0,
    int column, const float (&x)[prompt_in][8], float (&partials)[prompt_out][prompt_in][prompt_warps + 1]) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int base = 0; base < prompt_out; base += 4) {
    K_PKW packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      packed[i] = k_pkload(reinterpret_cast<const u8*>(job.w), job.sc, Format,
                           out0 + base + i, job.cols, column);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float w[8], partial[prompt_in];
      const float scale = k_pkdec(packed[i], Format, w);
#pragma unroll
      for (int token = 0; token < prompt_in; ++token) {
        float sum = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) sum = fmaf(w[e], x[token][e], sum);
        partial[token] = sum * scale;
      }
      float sum = k_warp_sum8t(partial);
      if (!(lane & 3)) partials[base + i][lane / 4][warp] = sum;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < prompt_out * prompt_in; i += blockDim.x) {
    const int row = i / prompt_in, token = i % prompt_in;
    float sum = 0.0f;
    for (int warp_index = 0; warp_index < blockDim.x / 32; ++warp_index)
      sum += partials[row][token][warp_index];
    job.y[static_cast<size_t>(in0 + token) * job.rows + out0 + row] = k_jout(job, sum);
  }
  __syncthreads();
}

__global__ void __launch_bounds__(384, 1) prompt_packed(GemvArgs a) {
  __shared__ float partials[prompt_out][prompt_in][prompt_warps + 1];
  const int groups = a.nrhs / prompt_in;
  const int in0 = (blockIdx.x % groups) * prompt_in;
  const int column = threadIdx.x * 8;
  const float* previous = nullptr;
  float x[prompt_in][8];
  for (int tile = blockIdx.x / groups; tile < a.blocks; tile += gridDim.x / groups) {
    int row = tile * prompt_out, j = 0;
    while (row >= a.job[j].rows) row -= a.job[j++].rows;
    const GemvJob& job = a.job[j];
    if (previous != job.x) {
      previous = job.x;
#pragma unroll
      for (int i = 0; i < prompt_in; ++i) {
        const float* v = job.x + static_cast<size_t>(in0 + i) * job.cols + column;
        const float4 lo = KLOADF4(v), hi = KLOADF4(v + 4);
        x[i][0] = lo.x; x[i][1] = lo.y; x[i][2] = lo.z; x[i][3] = lo.w;
        x[i][4] = hi.x; x[i][5] = hi.y; x[i][6] = hi.z; x[i][7] = hi.w;
      }
    }
    if (job.fmt == INK2_DFMT_NVFP4)
      prompt_tile<INK2_DFMT_NVFP4>(job, row, in0, column, x, partials);
    else
      prompt_tile<INK2_DFMT_MXFP8>(job, row, in0, column, x, partials);
  }
}

bool packed_prompt_shape(const GemvArgs& a, int threads) {
  if (a.nrhs <= 32 || a.nrhs % prompt_in || threads > prompt_warps * 32 || threads % 32) return false;
  for (int i = 0; i < a.njobs; ++i) {
    const GemvJob& j = a.job[i];
    if (j.rows % prompt_out || j.cols != threads * 8 || j.split != j.cols ||
        (j.fmt != INK2_DFMT_NVFP4 && j.fmt != INK2_DFMT_MXFP8)) return false;
  }
  return true;
}
}  // namespace
