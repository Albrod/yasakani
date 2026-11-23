// SPDX-License-Identifier: Apache-2.0
// Tensor-core form of attn_flash (flash_attn = 2 or 3): BF16 mma.sync m16n8k16 with FP32 sums.
// One block of W warps shares one key/value head across 16 W queries (16 W / 3 rows x 3 heads); warp w owns the
// queries 16 w .. 16 w + 15 (an m16 tile). Keys stream in tiles of 64: K and V are read once per tile (FP32 or MXFP8
// cache), rounded to BF16 into shared memory; S = Q K^T and O += P V run on the tensor cores (Q and P rounded to
// BF16), softmax in FP32 registers. Same grid, range and split outputs as attn_flash (ink2_flash_attn.inc).
// Needs sm_80 or newer (mma.sync bf16, ldmatrix); every architecture of the build is (8.6, 8.9, 12.0).
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned ftc_pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);  // .x (lo) in the low half
  return *reinterpret_cast<unsigned*>(&v);
}

__device__ __forceinline__ void ftc_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void ftc_ldsm4t(unsigned* r, const void* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

template <int W>
__global__ void __launch_bounds__(32 * W) kk_attn_flash_tc(AttnArgs a) {
  constexpr int NQ = 16 * W, BR = NQ / 3, KST = 136;  // row stride 272 bytes: conflict-free fragments and ldmatrix
  __shared__ __align__(16) u16 ks[64 * KST];
  __shared__ __align__(16) u16 vs[64 * KST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
  int b = blockIdx.x;
  const int kh = b % a.nkv;
  b /= a.nkv;
  const int sp = b % a.splits;
  b /= a.splits;
  const int r0 = b * BR;
  int rows = a.rows - r0;
  if (rows > BR) rows = BR;
  const int nq = rows * 3;
  int pmin, begin, end;
  flash_range(a, r0, rows, sp, &pmin, &begin, &end);
  const int pn = flash_pnear(a, r0, rows);  // MXFP8 cache near copies
  int qp[2];
  size_t qi[2];
  bool qv[2];
  float m[2], l[2];
  for (int e = 0; e < 2; ++e) {
    int q = warp * 16 + g + 8 * e;
    qv[e] = q < nq;
    int row = r0 + (qv[e] ? q / 3 : 0);
    qi[e] = (size_t)row * a.nh + kh * 3 + q % 3;
    qp[e] = qv[e] ? a.st[row].pos : -1;
    bool seed = a.splits == 1 && qv[e];
    m[e] = seed ? a.sink[qi[e]] : -3.402823466e38f;
    l[e] = seed && tig == 0 ? 1.0f : 0.0f;  // the quad's partial sums are added at the end
  }
  unsigned qf[8][4];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    int d = k * 16 + tig * 2;
    float2 z = make_float2(0.0f, 0.0f);
    float2 x0 = qv[0] ? *(const float2*)(a.q + qi[0] * 128 + d) : z;
    float2 x1 = qv[1] ? *(const float2*)(a.q + qi[1] * 128 + d) : z;
    float2 x2 = qv[0] ? *(const float2*)(a.q + qi[0] * 128 + d + 8) : z;
    float2 x3 = qv[1] ? *(const float2*)(a.q + qi[1] * 128 + d + 8) : z;
    qf[k][0] = ftc_pack(x0.x, x0.y);
    qf[k][1] = ftc_pack(x1.x, x1.y);
    qf[k][2] = ftc_pack(x2.x, x2.y);
    qf[k][3] = ftc_pack(x3.x, x3.y);
  }
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  const int C = a.nkv * 128;
  const float neg_inf = __int_as_float(0xff800000);
  for (int t0 = begin; t0 < end; t0 += 64) {
    int n = end - t0;
    if (n > 64) n = 64;
    __syncthreads();
#pragma unroll 4
    for (int i = tid; i < 64 * 32; i += 32 * W) {
      int kk = i >> 5, d4 = (i & 31) * 4;
      float4 kx = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vx = kx;
      if (kk < n) {
        int t = t0 + kk, slot = a.ring ? t % a.ring : t;
        const int ns = flash_ns(a, t, pn);
        kx = kv_read4(a.kv, a.kc, 0, slot, ns, C, kh * 128 + d4);
        vx = kv_read4(a.kv, a.vc, 1, slot, ns, C, kh * 128 + d4);
      }
      *(uint2*)(ks + kk * KST + d4) = make_uint2(ftc_pack(kx.x, kx.y), ftc_pack(kx.z, kx.w));
      *(uint2*)(vs + kk * KST + d4) = make_uint2(ftc_pack(vx.x, vx.y), ftc_pack(vx.z, vx.w));
    }
    __syncthreads();
    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
      const u16* kr = ks + (nt * 8 + g) * KST + tig * 2;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        ftc_mma(s[nt], qf[k], *(const unsigned*)(kr + k * 16), *(const unsigned*)(kr + k * 16 + 8));
    }
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = e >> 1, key = nt * 8 + tig * 2 + (e & 1), t = t0 + key, p = qp[r];
        float v = neg_inf;
        if (qv[r] && key < n && t <= p && (!a.window || p - t < a.window)) {
          int dist = p - t;
          v = s[nt][e] * a.scale;
          if (dist < a.rel_extent) v = v + a.rel_logits[qi[r] * a.rel_stride + dist];
        }
        s[nt][e] = v;
        mx[r] = fmaxf(mx[r], v);
      }
    }
    float al[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
      al[r] = expf(m[r] - mx[r]);
      m[r] = mx[r];
      l[r] *= al[r];
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float p = expf(s[nt][e] - m[e >> 1]);
        l[e >> 1] += p;
        s[nt][e] = p;
      }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      o[i][0] *= al[0]; o[i][1] *= al[0];
      o[i][2] *= al[1]; o[i][3] *= al[1];
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      unsigned pa[4] = {ftc_pack(s[2 * kk][0], s[2 * kk][1]), ftc_pack(s[2 * kk][2], s[2 * kk][3]),
                        ftc_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), ftc_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
      const u16* vrow = vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KST + (lane >> 4) * 8;
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        unsigned r[4];
        ftc_ldsm4t(r, vrow + dp * 16);
        ftc_mma(o[2 * dp], pa, r[0], r[1]);
        ftc_mma(o[2 * dp + 1], pa, r[2], r[3]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    if (!qv[r]) continue;
    if (a.splits == 1) {
#pragma unroll
      for (int dn = 0; dn < 16; ++dn) {
        int d = dn * 8 + tig * 2;
        float2 gg = *(const float2*)(a.gate + qi[r] * 128 + d);
        *(float2*)(a.out + qi[r] * 128 + d) =
            make_float2((o[dn][2 * r] / l[r]) * gg.x, (o[dn][2 * r + 1] / l[r]) * gg.y);
      }
    } else {
      float* p = a.partial + (qi[r] * a.splits + sp) * 130;
#pragma unroll
      for (int dn = 0; dn < 16; ++dn) {
        int d = dn * 8 + tig * 2;
        p[d] = o[dn][2 * r];
        p[d + 1] = o[dn][2 * r + 1];
      }
      if (tig == 0) {
        p[128] = m[r];
        p[129] = l[r];
      }
    }
  }
}

// The CUDA backend runs this instead of the FP32 form when flash_tc is 3 or 6. Returns false for other values.
static bool ink2_flash_tc_launch(const AttnArgs& a, int grid, cudaStream_t stream) {
  if (a.flash_tc == 3) kk_attn_flash_tc<3><<<grid, 96, 0, stream>>>(a);
  else if (a.flash_tc == 6) kk_attn_flash_tc<6><<<grid, 192, 0, stream>>>(a);
  else return false;
  return true;
}
