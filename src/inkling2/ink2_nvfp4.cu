// SPDX-License-Identifier: Apache-2.0
// Prompt-only NVFP4 expert products on a Blackwell GeForce GPU.
// Storage is the existing row-major E2M1 weight with one E4M3 scale per 16.
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "ink2_expert_tile.cuh"

namespace {
constexpr int threads = 128;
constexpr int warp_channel_rows = 8;
constexpr int block_channel_rows = threads / 32 * warp_channel_rows;

// Each eight lanes pack one block of 16 activations. Further planes encode
// the residual times eight to keep its scale in the E4M3 range.
template<int Planes>
__global__ void pack_input(const float* x, u8* scratch, int elements) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= elements / 2) return;  // elements is a multiple of 64
  float2 v = reinterpret_cast<const float2*>(x)[pair];
  const int packed_bytes = elements / 2;
  const int plane_bytes = packed_bytes + elements / 16;
#pragma unroll
  for (int p = 0; p < Planes; ++p) {
    float m = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
    for (int offset = 4; offset; offset >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset, 8));
    u8 sc = __nv_cvt_float_to_fp8(m / 6.0f, __NV_SATFINITE, __NV_E4M3);
    // Even a denormal input has a well-defined representation.
    if (!sc) sc = 1;
    float scale = __half2float(__half(__nv_cvt_fp8_to_halfraw(sc, __NV_E4M3)));
    float inv = 1.0f / scale;
    u8 byte = __nv_cvt_float2_to_fp4x2(make_float2(v.x * inv, v.y * inv), __NV_E2M1, cudaRoundNearest);
    u8* plane = scratch + p * plane_bytes;
    plane[pair] = byte;
    if (!(pair & 7)) plane[packed_bytes + pair / 8] = sc;
    if (p + 1 < Planes) {
      float2 q = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1)));
      v.x = (v.x - q.x * scale) * 8.0f;
      v.y = (v.y - q.y * scale) * 8.0f;
    }
  }
}

__device__ __forceinline__ unsigned word(const u8* base, int index) {
  return reinterpret_cast<const unsigned*>(base)[index];
}

__device__ __forceinline__ void multiply(float (&sum)[4], unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
      : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// One warp writes up to 16 routed pairs by eight adjacent weight rows.
// Row pairs in w13 are interleaved gate and up. w2 writes the original
// (token, router slot) so that the existing combiner keeps its ordering.
template<int Planes, bool GateUp>
__global__ void products(ExpMArgs a) {
  int expert_index = blockIdx.y;
  if (expert_index >= *a.gcount) return;
  int lane = threadIdx.x & 31, group = lane / 4, sub = lane % 4;
  int channel = blockIdx.x * block_channel_rows + (threadIdx.x / 32) * warp_channel_rows;
  if (channel >= a.rows) return;
  const int expert = a.gexp[expert_index];
  const int end = a.goff[expert_index + 1];
  const size_t weight_row = static_cast<size_t>(expert) * a.rows + channel + group;
  const u8* w = a.w + weight_row * (a.cols / 2);
  const u8* scales = a.sc + weight_row * (a.cols / 16);
  const int packed_bytes = a.input_rows * a.cols / 2;
  const int plane_bytes = packed_bytes + a.input_rows * a.cols / 16;
  const float factor = 2.0f * a.factor[expert];

  for (int g = a.goff[expert_index]; g < end; g += 16) {
    int row[2] = {0, 0};
    int pair[2] = {0, 0};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int i = g + group + 8 * half;
      if (i < end) pair[half] = a.gpairs[i];
      row[half] = GateUp ? pair[half] / a.k : pair[half];
    }
    float sums[Planes][4] = {};
    for (int k = 0; k < a.cols; k += 64) {
      unsigned b0 = word(w, k / 8 + sub);
      unsigned b1 = word(w, k / 8 + 4 + sub);
      unsigned sb = word(scales, k / 64);
#pragma unroll
      for (int p = 0; p < Planes; ++p) {
        const u8* plane = a.nvfp4_scratch + p * plane_bytes;
        const u8* x0 = plane + row[0] * (a.cols / 2);
        const u8* x1 = plane + row[1] * (a.cols / 2);
        const u8* xs = plane + packed_bytes + row[lane & 1] * (a.cols / 16);
        int offset = k / 8 + sub;
        multiply(sums[p], word(x0, offset), word(x1, offset),
                 word(x0, offset + 4), word(x1, offset + 4),
                 b0, b1, word(xs, k / 64), sb);
      }
    }
    float values[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = sums[Planes - 1][i];
#pragma unroll
      for (int p = Planes - 2; p >= 0; --p) v = sums[p][i] + v * 0.125f;
      values[i] = v * factor;
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      if (g + group + 8 * half >= end) continue;
      float v0 = values[2 * half], v1 = values[2 * half + 1];
      int col = channel + 2 * sub;
      ink2_store_pair<GateUp>(a, pair[half], col, v0, v1);
    }
  }
}

template<int Planes>
cudaError_t launch(const ExpMArgs& a, bool gate_up, cudaStream_t stream) {
  int elements = a.input_rows * a.cols;
  pack_input<Planes><<<(elements / 2 + threads - 1) / threads, threads, 0, stream>>>(
      a.x, a.nvfp4_scratch, elements);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  // grid / nchunk is only an estimate in the scalar kernel. Launch the full
  // index bound here, including experts with unusually low routing frequency.
  dim3 grid((a.rows + block_channel_rows - 1) / block_channel_rows, a.expert_slots);
  if (gate_up) products<Planes, true><<<grid, threads, 0, stream>>>(a);
  else products<Planes, false><<<grid, threads, 0, stream>>>(a);
  return cudaGetLastError();
}
} // namespace

bool ink2_nvfp4_available() {
  int dev = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
      prop.major != 12 || prop.minor != 0) return false;
  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, products<1, true>) == cudaSuccess) return true;
  cudaGetLastError();
  return false;
}

cudaError_t ink2_nvfp4_experts(const ExpMArgs& a, bool gate_up, cudaStream_t stream) {
  switch (a.nvfp4_planes) {
    case 1: return launch<1>(a, gate_up, stream);
    case 2: return launch<2>(a, gate_up, stream);
    case 3: return launch<3>(a, gate_up, stream);
    case 4: return launch<4>(a, gate_up, stream);
    default: return cudaErrorInvalidValue;
  }
}
