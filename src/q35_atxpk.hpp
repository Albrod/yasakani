// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
#pragma once







#include "q35.hpp"
#include <string>
namespace yskn {
struct Q35PkArgs { const uint8_t* kp = nullptr; const uint8_t* vp = nullptr; const float* kps = nullptr; const float* vps = nullptr; const unsigned char* rec = nullptr; int layer = 0; int kb = 4; int variant = 0; };
void launch_q35_atx_pk(const Q35AttnArgs&, const uint8_t* wk, const Q35PkArgs&, cudaStream_t);
// multi-stream pass (at most 16 rows): the KV containers of each stream (by slot); packed K is required, V is packed (vp / vps)
// or INT8 (v8 / vs)
struct Q35PkMs { Q35MsRows ms; const uint8_t* const* kp = nullptr; const uint8_t* const* vp = nullptr; const float* const* kps = nullptr;
                 const float* const* vps = nullptr; const int8_t* const* v8 = nullptr; const float* const* vs = nullptr; };
void launch_q35_atx_pk_ms(const Q35AttnArgs&, const Q35PkArgs&, const Q35PkMs&, cudaStream_t);
// "" when the attention settings of this process (YSKN_ATX_PK3_*) and kb take a multi-stream kernel the library has, else the reason
std::string q35_atx_pk_ms_unsupported(int kb);
bool launch_q35_prefill_atx_pk(const Q35AttnArgs&, const uint8_t* wk, const Q35PkArgs&, cudaStream_t);
void launch_q35_head_first_combine(const Q35CombineArgs&, cudaStream_t);
}  
