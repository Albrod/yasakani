// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// Multi-stream pass (experimental): the sampling kernels of q35_sample.cu for rows of several streams, each row with the position and
// the sampling settings of its own stream.  They are compiled in a translation unit of their own, so that the single-request kernels
// of q35_sample.cu are made exactly as without them (kj_pop keeps its shared memory in the kernel only while one kernel of the unit
// uses it).  q35_sample.cu is included for its helpers (kj_pop, kj_uniform, k2_warp_pop; kh_gumbel and pm_exp_d come with it); its
// kernels and launchers are left out by YSKN_SAMPLE_MS_TU.
//
// Each kernel is the kernel of q35_sample.cu with the same name without "_ms", line for line, except where a value comes from:
//   single request                                      multi-stream row r
//   A.prm (this engine's settings, set_opt 10..13/16/29) prm[A.row0 + r] (the settings of the row's stream, ms_set_sampling)
//   pos = *A.d_pos + depth + 1                          pos = pos_tab[A.row0 + r] + 1 (the row's position in its stream, + 1)
//   P.seed                                              P.seed of the row's stream
// A row of a greedy stream (temp 0) keeps the argmax that d_tok_out already holds: the kernel returns before it picks a token (and,
// when no top list is asked for, right at the start).
#define YSKN_SAMPLE_MS_TU 1
#include "q35_sample.cu"
namespace yskn {
namespace {
// smp_kernel of q35_sample.cu, with the position and the settings of each row's stream
__global__ void __launch_bounds__(KJ_T) smp_ms_kernel(Q35SmpArgs A, const int* pos_tab, const Q35SmpParams* prm) { pdl_wait(); pdl_trigger();
  const Q35SmpParams P = prm[A.row0 + blockIdx.y];                       // the row's stream
  if (!(P.temp > 0.0)) return;                                           // greedy: the row keeps the argmax (this kernel writes no top list)
  const int r = blockIdx.y, k = P.top_k; const float* lg = A.logits + (size_t)r * A.V;
  float tv[KJ_PT]; int ti[KJ_PT];
  #pragma unroll
  for (int q = 0; q < KJ_PT; ++q) { tv[q] = -FLT_MAX; ti[q] = 0x7fffffff; }
  #pragma unroll
  for (int q = 0; q < KJ_PT; ++q) { const int v = (q * KJ_G + blockIdx.x) * KJ_T + threadIdx.x;
    if (v < A.V) { float x = lg[v]; int ix = v;
      #pragma unroll
      for (int s = 0; s < KJ_PT; ++s) if (kj_better(x, ix, tv[s], ti[s])) { const float t0 = tv[s]; const int i0 = ti[s]; tv[s] = x; ti[s] = ix; x = t0; ix = i0; } } }
  const size_t c = ((size_t)r * KJ_G + blockIdx.x) * Q35_SMP_K;
  kj_pop(tv, ti, k, A.cand_val + c, A.cand_idx + c);
  __shared__ int s_last; __threadfence(); __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(A.cnt + r, 1u) & (KJ_G - 1)) == KJ_G - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  float cv[Q35_SMP_K]; int ci[Q35_SMP_K];
  { const size_t c2 = ((size_t)r * KJ_G + threadIdx.x) * Q35_SMP_K;
    #pragma unroll
    for (int q = 0; q < Q35_SMP_K; ++q) { const bool in = q < k; cv[q] = in ? __ldcg(A.cand_val + c2 + q) : -FLT_MAX; ci[q] = in ? __ldcg(A.cand_idx + c2 + q) : 0x7fffffff; } }
  __shared__ float s_v[Q35_SMP_K]; __shared__ int s_i[Q35_SMP_K];
  kj_pop(cv, ci, k, s_v, s_i);
  if (threadIdx.x != 0) return;
  const long long pos = (long long)pos_tab[A.row0 + r] + 1;
  double w[Q35_SMP_K], tot = 0.0;
  for (int j = 0; j < k; ++j) { const double z = ((double)s_v[j] - (double)s_v[0]) / P.temp; w[j] = P.pm ? pm_exp_d(z) : exp(z); tot += w[j]; }
  int n = k; double run = 0.0;
  for (int j = 0; j < k; ++j) { run += w[j]; if (run >= P.top_p * tot) { n = j + 1; break; } }
  if (P.min_p > 0.0) { int m = 1; while (m < n && w[m] >= P.min_p) ++m; n = m; }
  int pick = n - 1;
  if (P.version == 3) {
    double best = -INFINITY;
    for (int j = 0; j < n; ++j) { const double g = ((double)s_v[j] - (double)s_v[0]) / P.temp + kh_gumbel(P.seed, pos, s_i[j], P.pm); if (g > best) { best = g; pick = j; } }
  } else {
    double kept = 0.0; for (int j = 0; j < n; ++j) kept += w[j];
    const double u = kj_uniform(P.seed, pos) * kept;
    run = 0.0;
    for (int j = 0; j < n; ++j) { run += w[j]; if (u < run) { pick = j; break; } }
  }
  const int id = s_i[pick];
  A.d_tok_out[r] = id;
  if (A.adv_pos) { *A.adv_pos += 1; *A.adv_tok = id; }
}
// smp2_kernel of q35_sample.cu, with the position and the settings of each row's stream
__global__ void __launch_bounds__(K2_T) smp2_ms_kernel(Q35SmpArgs A, const int* pos_tab, const Q35SmpParams* prm) { pdl_wait(); pdl_trigger();
  const Q35SmpParams P = prm[A.row0 + blockIdx.y];                       // the row's stream
  const bool on = P.temp > 0.0;                                          // greedy: the row keeps the argmax; only the top lists are written
  if (!on && !A.top_val && !A.topk_val) return;
  __shared__ float s_wv[K2_T / 32][Q35_SMP_K]; __shared__ int s_wi[K2_T / 32][Q35_SMP_K];
  __shared__ float s_v[Q35_SMP_K]; __shared__ int s_i[Q35_SMP_K]; __shared__ int s_last;
  const int r = blockIdx.y, k = P.top_k, kk = A.top_val ? (k > 8 ? k : 8) : k;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* lg = A.logits + (size_t)r * A.V;
  {
    float v[K2_PT]; int id[K2_PT];
    #pragma unroll
    for (int q = 0; q < K2_PT; ++q) { const int x = (blockIdx.x * K2_PT + q) * K2_T + threadIdx.x;
      v[q] = x < A.V ? lg[x] : -FLT_MAX; id[q] = x < A.V ? x : 0x7fffffff; }
    k2_warp_pop(v, id, kk, s_wv[warp], s_wi[warp]);
  }
  __syncthreads();
  if (warp == 0) {
    float c[5]; int ci[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) { const int f = lane + 32 * j; const bool in = f < (K2_T / 32) * kk;
      c[j] = in ? s_wv[f / kk][f % kk] : -FLT_MAX; ci[j] = in ? s_wi[f / kk][f % kk] : 0x7fffffff; }
    const size_t cb = ((size_t)r * K2_G + blockIdx.x) * Q35_SMP_K;
    k2_warp_pop(c, ci, kk, A.cand_val + cb, A.cand_idx + cb);
  }
  __threadfence(); __syncthreads();
  if (threadIdx.x == 0) s_last = (atomicAdd(A.cnt + r, 1u) & (K2_G - 1)) == K2_G - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  {
    float c[5]; int ci[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) { const int f = lane + 32 * j; const bool in = f < 8 * kk;
      const size_t src = ((size_t)r * K2_G + warp * 8 + (in ? f / kk : 0)) * Q35_SMP_K + (in ? f % kk : 0);
      c[j] = in ? __ldcg(A.cand_val + src) : -FLT_MAX; ci[j] = in ? __ldcg(A.cand_idx + src) : 0x7fffffff; }
    __syncthreads();
    k2_warp_pop(c, ci, kk, s_wv[warp], s_wi[warp]);
  }
  __syncthreads();
  if (warp != 0) return;
  {
    float c[5]; int ci[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) { const int f = lane + 32 * j; const bool in = f < (K2_T / 32) * kk;
      c[j] = in ? s_wv[f / kk][f % kk] : -FLT_MAX; ci[j] = in ? s_wi[f / kk][f % kk] : 0x7fffffff; }
    k2_warp_pop(c, ci, kk, s_v, s_i);
  }
  __syncwarp();
  if (A.top_val && lane < 8) { A.top_val[(size_t)r * 8 + lane] = s_v[lane]; A.top_idx[(size_t)r * 8 + lane] = s_i[lane]; }
  if (A.topk_val && lane < k) { A.topk_val[(size_t)r * Q35_SMP_K + lane] = s_v[lane]; A.topk_idx[(size_t)r * Q35_SMP_K + lane] = s_i[lane]; }
  if (!on) return;
  const long long pos = (long long)pos_tab[A.row0 + r] + 1;
  __shared__ double s_w[Q35_SMP_K], s_gz[Q35_SMP_K];
  if (lane < k) { const double z = ((double)s_v[lane] - (double)s_v[0]) / P.temp; s_w[lane] = P.pm ? pm_exp_d(z) : exp(z);
                  if (P.version == 3) s_gz[lane] = z + kh_gumbel(P.seed, pos, s_i[lane], P.pm); }
  __syncwarp();
  if (lane != 0) return;
  double w[Q35_SMP_K], tot = 0.0;
  for (int j = 0; j < k; ++j) { w[j] = s_w[j]; tot += w[j]; }
  int n = k; double run = 0.0;
  for (int j = 0; j < k; ++j) { run += w[j]; if (run >= P.top_p * tot) { n = j + 1; break; } }
  if (P.min_p > 0.0) { int m = 1; while (m < n && w[m] >= P.min_p) ++m; n = m; }
  int pick = n - 1;
  if (P.version == 3) {
    double best = -INFINITY;
    for (int j = 0; j < n; ++j) { const double g = s_gz[j]; if (g > best) { best = g; pick = j; } }
  } else {
    double kept = 0.0; for (int j = 0; j < n; ++j) kept += w[j];
    const double u = kj_uniform(P.seed, pos) * kept;
    run = 0.0;
    for (int j = 0; j < n; ++j) { run += w[j]; if (u < run) { pick = j; break; } }
  }
  const int id = s_i[pick];
  A.d_tok_out[r] = id;
  if (A.adv_pos) { *A.adv_pos += 1; *A.adv_tok = id; }
}
}
void launch_q35_smp_ms(const Q35SmpArgs& A, const int* pos, const Q35SmpParams* prm, cudaStream_t st) {
  if (A.V > KJ_G * KJ_T * KJ_PT) throw std::runtime_error("SMP: vocabulary larger than 262,144");
  if (!pos || !prm) throw std::runtime_error("launch_q35_smp_ms: no stream tables");
  if (q35_smp2_on()) { launch_k(smp2_ms_kernel, dim3(K2_G, A.rows), dim3(K2_T), 0, st, A, pos, prm); return; }
  launch_k(smp_ms_kernel, dim3(KJ_G, A.rows), dim3(KJ_T), 0, st, A, pos, prm);
}
}
