// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Included by q35_g6.cu. One CTA owns a complete output tile and accumulates
// all K groups in registers. The INT8 operands and INT64 arithmetic keep
// the same layout and rounding as mt_gemm6_kernel.

template <int RowFragments, int RowWarps, int ColWarps, int Capacity = G6_GP> struct G6WholeTile {
  static constexpr int Rows = 16 * RowFragments * RowWarps;
  static constexpr int Cols = 32 * ColWarps;
  static constexpr int Threads = 32 * RowWarps * ColWarps;
  static constexpr int ABytes = Rows * G6_BK;
  static constexpr int BBytes = Cols * G6_BK;
  static constexpr int OperandBytes = G6_S * (ABytes + BBytes);
  static constexpr size_t Shared = OperandBytes + Rows * (Capacity * sizeof(int) + 2 * sizeof(float));
};

template <int RM, int WM, int WN, int Capacity = G6_GP>
__global__ void __launch_bounds__(G6WholeTile<RM, WM, WN, Capacity>::Threads)
g6_whole_kernel(const __grid_constant__ Q35G6Args P, int row_tiles, int groups) {
  using Tile = G6WholeTile<RM, WM, WN, Capacity>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto* const stage_a = storage;
  auto* const stage_b = stage_a + G6_S * Tile::ABytes;
  auto* const scale = reinterpret_cast<int*>(storage + Tile::OperandBytes);
  auto* const unscale = reinterpret_cast<float*>(scale + Tile::Rows * Capacity);
  auto* const norm = unscale + Tile::Rows;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int row_warp = warp / WN, col_warp = warp % WN;
  const int column_tile = blockIdx.x / row_tiles;
  const int row = (blockIdx.x - column_tile * row_tiles) * Tile::Rows;
  int segment = 0;
  if (P.n_seg > 1 && column_tile >= P.seg[1].cta_begin) segment = 1;
  if (P.n_seg > 2 && column_tile >= P.seg[2].cta_begin) segment = 2;
  const Q35G6Seg& seg = P.seg[segment];
  const int col = (column_tile - seg.cta_begin) * Tile::Cols, width = seg.t.Kc;
  const auto* const a_base = seg.qx + static_cast<size_t>(row) * width;
  const auto* const b_base = seg.w8 + static_cast<size_t>(col / 16) * width;
  const auto* const alpha = seg.alpha + static_cast<size_t>(row) * groups;
  const unsigned a_shared = static_cast<unsigned>(__cvta_generic_to_shared(stage_a));
  const unsigned b_shared = static_cast<unsigned>(__cvta_generic_to_shared(stage_b));

  auto load_stage = [&](int slot, int step) {
    #pragma unroll
    for (int batch = 0; batch < Tile::ABytes / (16 * Tile::Threads); ++batch) {
      const int chunk = tid + batch * Tile::Threads, ar = chunk / 4, ak = chunk % 4;
      const int swizzled = ak ^ ((ar >> 1) & 3);
      mt_cp16s(a_shared + slot * Tile::ABytes + ar * G6_BK + 16 * swizzled,
               a_base + static_cast<size_t>(ar) * width + step * G6_BK + 16 * ak);
    }
    #pragma unroll
    for (int batch = 0; batch < Tile::BBytes / (16 * Tile::Threads); ++batch) {
      const int chunk = tid + batch * Tile::Threads;
      mt_cp16s(b_shared + slot * Tile::BBytes + 16 * chunk,
               b_base + static_cast<size_t>(chunk / 64) * width + step * G6_BK + chunk % 64);
    }
  };

  if constexpr (Capacity == 34) {
    // One warp reads one contiguous row of 34 scales. The maximum of
    // nonnegative scales is identical regardless of reduction order.
    for (int r = warp; r < Tile::Rows; r += Tile::Threads / 32) {
      const float* row_alpha = alpha + r * Capacity;
      const float first = __ldg(row_alpha + lane);
      const float last = lane < 2 ? __ldg(row_alpha + 32 + lane) : 0.f;
      float largest = fmaxf(0.f, fmaxf(first, last));
      #pragma unroll
      for (int delta = 16; delta; delta /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, delta));
      const float multiplier = largest > 0.f ? __fdiv_rn(16777216.f, largest) : 0.f;
      scale[r * Capacity + lane] = __float2int_rn(__fmul_rn(first, multiplier));
      if (lane < 2) scale[r * Capacity + 32 + lane] = __float2int_rn(__fmul_rn(last, multiplier));
      if (lane == 0) {
        const float factor = ldexpf(largest, -24);
        unscale[r] = factor;
        if (column_tile == 0 && P.us_out) P.us_out[row + r] = factor;
        if (seg.S) {
          const float* parts = seg.S + static_cast<size_t>(row + r) * P.xsm_sst;
          float sum = 0.f;
          for (int b = 0; b < P.xsm_nb; ++b) sum = __fadd_rn(sum, parts[b]);
          const float mean = __fadd_rn(__fdiv_rn(sum, P.xsm_n), P.eps);
          norm[r] = P.pm ? pm_rsqrt(mean) : rsqrtf(mean);
        } else {
          norm[r] = 0.f;
        }
      }
    }
  } else {
    if (tid < Tile::Rows) {
      const float* const row_alpha = alpha + tid * groups;
      float largest = 0.f;
      for (int g = 0; g < groups; ++g) largest = fmaxf(largest, __ldg(row_alpha + g));
      const float multiplier = largest > 0.f ? __fdiv_rn(16777216.f, largest) : 0.f;
      for (int g = 0; g < groups; ++g)
        scale[tid * Capacity + g] = __float2int_rn(__fmul_rn(__ldg(row_alpha + g), multiplier));
      const float factor = ldexpf(largest, -24);
      unscale[tid] = factor;
      if (column_tile == 0 && P.us_out) P.us_out[row + tid] = factor;
      if (seg.S) {
        const auto* const parts = seg.S + static_cast<size_t>(row + tid) * P.xsm_sst;
        float sum = 0.f;
        for (int b = 0; b < P.xsm_nb; ++b) sum = __fadd_rn(sum, parts[b]);
        const float mean = __fadd_rn(__fdiv_rn(sum, P.xsm_n), P.eps);
        norm[tid] = P.pm ? pm_rsqrt(mean) : rsqrtf(mean);
      } else {
        norm[tid] = 0.f;
      }
    }
  }

  #pragma unroll
  for (int slot = 0; slot < G6_S - 1; ++slot) {
    load_stage(slot, slot);
    mt_cp_commit();
  }

  const int local_row = lane & 15, row_group = lane >> 2, lane_col = lane & 3;
  const unsigned a_warp = a_shared + (16 * RM * row_warp + local_row) * G6_BK;
  const unsigned b_warp = b_shared + (128 * col_warp + lane) * 16;
  long long totals[RM][4][4] = {};
  int dots[RM][4][4];
  auto consume = [&](auto chunk_tag) {
    constexpr int chunk = decltype(chunk_tag)::value;
    constexpr int slot = chunk % G6_S;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint4 first, second;
      const unsigned b = b_warp + slot * Tile::BBytes + 512 * half;
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                   : "=r"(first.x), "=r"(first.y), "=r"(first.z), "=r"(first.w) : "r"(b));
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                   : "=r"(second.x), "=r"(second.y), "=r"(second.z), "=r"(second.w) : "r"(b + 1024));
      const uint2 weights[4] = {make_uint2(first.x, first.y), make_uint2(first.z, first.w),
                               make_uint2(second.x, second.y), make_uint2(second.z, second.w)};
      #pragma unroll
      for (int i = 0; i < RM; ++i) {
        unsigned a[4];
        const unsigned addr = a_warp + slot * Tile::ABytes + i * 1024 +
                              16 * ((2 * half + (lane >> 4)) ^ ((local_row >> 1) & 3));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (chunk == 0 && half == 0) g6_mma0(dots[i][j], a, weights[j]);
          else mt_mma(dots[i][j], a, weights[j]);
        }
      }
    }
  };

  for (int g = 0; g < groups; ++g) {
    auto step = [&](auto chunk_tag) {
      constexpr int chunk = decltype(chunk_tag)::value;
      constexpr int ahead = chunk + G6_S - 1;
      g6_cp_wait<G6_S - 2>();
      __syncthreads();
      if (ahead < 8 || g + 1 < groups) load_stage(ahead % G6_S, g * 8 + ahead);
      mt_cp_commit();
      consume(chunk_tag);
    };
    step(std::integral_constant<int, 0>{}); step(std::integral_constant<int, 1>{});
    step(std::integral_constant<int, 2>{}); step(std::integral_constant<int, 3>{});
    step(std::integral_constant<int, 4>{}); step(std::integral_constant<int, 5>{});
    step(std::integral_constant<int, 6>{}); step(std::integral_constant<int, 7>{});

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
      const int r = 16 * (RM * row_warp + i) + row_group;
      const int low = scale[r * Capacity + g], high = scale[(r + 8) * Capacity + g];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        mt_mad_wide(totals[i][j][0], low, dots[i][j][0]);
        mt_mad_wide(totals[i][j][1], low, dots[i][j][1]);
        mt_mad_wide(totals[i][j][2], high, dots[i][j][2]);
        mt_mad_wide(totals[i][j][3], high, dots[i][j][3]);
      }
    }
  }

  const int output_col = col + 32 * col_warp + 2 * lane_col;
  auto store = [&](int r, int c, long long v0, long long v1) {
    float low = __fmul_rn(__ll2float_rn(v0), unscale[r]);
    float high = __fmul_rn(__ll2float_rn(v1), unscale[r]);
    if (seg.S) {
      low = __fmul_rn(low, norm[r]);
      high = __fmul_rn(high, norm[r]);
    }
    auto* const output = seg.y + static_cast<size_t>(row + r) * seg.ldy + c;
    *reinterpret_cast<float2*>(output) = make_float2(low, high);
  };
  #pragma unroll
  for (int i = 0; i < RM; ++i) {
    const int r = 16 * (RM * row_warp + i) + row_group;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      store(r, output_col + 8 * j, totals[i][j][0], totals[i][j][1]);
      store(r + 8, output_col + 8 * j, totals[i][j][2], totals[i][j][3]);
    }
  }
}

template <int RM, int WM, int WN, int Capacity = G6_GP> static void g6_prepare_whole(Q35G6Workspace& W, int sms) {
  using Tile = G6WholeTile<RM, WM, WN, Capacity>;
  static_assert(Tile::ABytes % (16 * Tile::Threads) == 0);
  static_assert(Tile::BBytes % (16 * Tile::Threads) == 0);
  auto kernel = g6_whole_kernel<RM, WM, WN, Capacity>;
  cudaFuncAttributes attributes{};
  int resident = 0;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Tile::Shared) != cudaSuccess ||
      cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, Tile::Threads, Tile::Shared) != cudaSuccess ||
      resident < 1) {
    W.tile = 0;
    return;
  }
  W.tile_rows = Tile::Rows;
  W.tile_cols = Tile::Cols;
  W.tile_resident = resident * sms;
  W.tile_registers = attributes.numRegs;
  W.tile_shared = Tile::Shared;
}

template <int RM, int WM, int WN, int Capacity = G6_GP> static void g6_launch_whole(
    const Q35G6Args& P, int cols, int groups, cudaStream_t stream) {
  using Tile = G6WholeTile<RM, WM, WN, Capacity>;
  const int rows = P.R / Tile::Rows;
  g6_whole_kernel<RM, WM, WN, Capacity><<<rows * cols, Tile::Threads, Tile::Shared, stream>>>(P, rows, groups);
}
