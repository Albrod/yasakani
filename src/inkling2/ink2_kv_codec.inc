// SPDX-License-Identifier: Apache-2.0
// MXFP8 cache encoding and reading; the same code runs in CUDA and the block emulator.
KINL unsigned kv_bits(float x) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(x);
#else
  unsigned bits;
  memcpy(&bits, &x, sizeof(bits));
  return bits;
#endif
}
KINL u16 kv_bfloat(float x) {
  unsigned u = kv_bits(x);
  return (u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
KINL int kv_rne(float x) {  // nonnegative and small
  int i = (int)x;
  float f = x - (float)i;
  return i + (f > 0.5f || (f == 0.5f && (i & 1)));
}
KINL u8 kv_e4(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
#else
  unsigned u = kv_bits(x), sign = (u >> 24) & 128u;
  u &= 0x7fffffffu;
  unsigned code;
  if (u >= 0x43e00000u) code = 126;  // finite saturation at 448
  else if (u < 0x3c800000u) code = (unsigned)kv_rne(KBITS2F(u) * 512.0f);
  else code = ((u + 0x7ffffu + ((u >> 20) & 1u)) >> 20) - 960u;
  return (u8)(code | sign);
#endif
}
#include "ink2_kv_mxfp8.inc"

// One value of the MXFP8 cache (a warp writes 32 adjacent values: one group with one exponent), and its BF16 copy
// when the position is a sink or recent.
KINL void kv_store(KCTXP const KvStorage& k, int isv, int pos, int slot, int C, int c, float x) {
  const size_t i = (size_t)slot * C + c;
  u8* data = isv ? k.value : k.key;
  const int lane = c & 31;
  const size_t group = i >> 5;
  const u8 e = kv_mx8_exp(k_warp_max(KCTXA fabsf(x)));
  data[i] = kv_e4(x * kv_mx8_inv(e));
  if (lane == 0) (isv ? k.vs : k.ks)[group] = e;
  int near_slot;
  if (pos < k.sinks) near_slot = pos;
  else if (k.hot_slots) {
    int h = pos % k.hot_slots;
    if (!isv && c == 0) k.tags[h] = pos + 1;
    near_slot = k.sinks + h;
  } else return;
  (isv ? k.near_v : k.near_k)[(size_t)near_slot * C + c] = kv_bfloat(x);
}

// Use the logical query position to choose precision. The larger physical ring
// preserves the previous tail while a pass writes future rows or rolls back.
KINL int kv_near(const KvStorage& k, int t, int pos) {
  if (t < k.sinks) return t;
  if (!k.hot_slots || t <= pos - k.hot) return -1;
  int slot = t % k.hot_slots;
  return k.tags[slot] == t + 1 ? k.sinks + slot : -1;
}
KINL KF4 kv_bf4(const u16* p) {
  u64 v = KLOAD64(p);
  KF4 a;
  a.x = k_bf((u16)v); a.y = k_bf((u16)(v >> 16));
  a.z = k_bf((u16)(v >> 32)); a.w = k_bf((u16)(v >> 48));
  return a;
}
// Four adjacent values at column col of cache slot `slot`: FP32 from fp, else the BF16 copy (near_slot >= 0) or MXFP8.
KINL KF4 kv_read4(const KvStorage& k, const float* fp, int isv, int slot, int near_slot, int C, int col) {
  size_t i = (size_t)slot * C + col;
  if (k.mode == INK2_KV_FP32) return KLOADF4(fp + i);
  const u8* data = isv ? k.value : k.key;
  if (near_slot >= 0)
    return kv_bf4((isv ? k.near_v : k.near_k) + (size_t)near_slot * C + col);
  return kv_mx8_read4(data, isv ? k.vs : k.ks, i);
}
