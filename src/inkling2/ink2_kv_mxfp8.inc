// SPDX-License-Identifier: Apache-2.0
// Two small helpers shared by every attention kernel (CUDA and the block emulator).
//
// 1. The MXFP8 cache (option kv_mxfp8): the full-attention visits keep E4M3 codes with one E8M0 (power of two)
//    scale per group of 32 elements; the sliding visits stay FP32. The scale is the smallest power
//    of two with amax / scale <= 448, so no element saturates and x / scale is exact; the only
//    rounding is the E4M3 code. The first sinks and the last kv_hot positions also keep a BF16
//    copy (near_k / near_v). Byte layout per visit: codes [cap, nkv*hd] u8 and
//    exponents [cap, nkv*hd / 32] u8 (KvStorage::ks / vs).
//    kv_mx8_read4 is the read helper for other kernels (split-K, flash): 4 adjacent elements
//    starting at element index i (a multiple of 4) of the codes.
// 2. kv_rel_bias: the relative-position bias of lane (< drel) at distance dist (< rel_extent),
//    from the transposed copy when the engine uploaded it (same value, same bits).

KINL u8 kv_mx8_exp(float amax) {
  unsigned u = kv_bits(amax) & 0x7fffffffu;
  int e = (int)(u >> 23);
  if (e == 0) return 1;  // zero or subnormal group: the codes are zero
  // amax = 1.m * 2^(e - 127) and 448 = 1.75 * 2^8 (mantissa 0x600000)
  int code = e - 8 + ((u & 0x7fffffu) > 0x600000u ? 1 : 0);
  return (u8)(code < 1 ? 1 : code > 253 ? 253 : code);
}
KINL float kv_mx8_inv(unsigned code) { return KBITS2F((254u - code) << 23); }  // 2^-k for scale 2^k

KINL KF4 kv_mx8_read4(const u8* codes, const u8* exps, size_t i) {
  float s = k_exp2v(exps[i >> 5]);
  unsigned u = KLOAD32(codes + i);
  KF4 a;
  a.x = k_e4m3v(u) * s; a.y = k_e4m3v(u >> 8) * s;
  a.z = k_e4m3v(u >> 16) * s; a.w = k_e4m3v(u >> 24) * s;
  return a;
}

KINL float kv_rel_bias(const AttnArgs& a, int lane, int dist) {
  return k_bf(a.rel_proj_t ? a.rel_proj_t[(size_t)dist * a.drel + lane]
                           : a.rel_proj[(size_t)lane * a.rel_extent + dist]);
}
