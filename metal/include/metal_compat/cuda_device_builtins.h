// cuda_device_builtins.h - the CUDA device built-ins, declared for the HOST compile only.
//
// The DeepSeek kernels are compiled twice in this port: as C++ here, to enumerate exactly which API the device code uses,
// and as MSL (kernels/*.metal) where the real work happens.  These declarations let the first pass run; nothing here is a
// working kernel.  Metal spells the same things thread_position_in_grid, threadgroup_barrier, simd_shuffle and so on.
#pragma once

#include <cstdint>

struct strata_dim3 { unsigned x, y, z; };
static strata_dim3 threadIdx{0, 0, 0}, blockIdx{0, 0, 0}, blockDim{1, 1, 1}, gridDim{1, 1, 1};

static inline void __syncthreads() {}
static inline void __syncwarp(unsigned = 0xffffffffu) {}
static inline unsigned __shfl_down_sync(unsigned, unsigned v, int) { return v; }
static inline unsigned __shfl_xor_sync(unsigned, unsigned v, int) { return v; }
static inline float __shfl_down_sync(unsigned, float v, int) { return v; }
static inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
static inline unsigned __ballot_sync(unsigned, int) { return 0; }
static inline int __popc(unsigned v) { return __builtin_popcount(v); }
static inline int __ffs(unsigned v) { return __builtin_ffs((int) v); }

// the bit and integer intrinsics the block decoders use.  Each is the exact CUDA semantics, computed on the host here; the
// MSL kernels implement the same operations with Metal's own select/rotate/permute built-ins.
static inline unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
    unsigned out = 0;
    const unsigned long long src = ((unsigned long long) y << 32) | x;
    for (int i = 0; i < 4; ++i) {
        const unsigned sel = (s >> (4 * i)) & 0x7u;
        out |= (unsigned) ((src >> (8 * sel)) & 0xffu) << (8 * i);
    }
    return out;
}
static inline int __mul24(int a, int b) { return a * b; }
static inline unsigned __umul24(unsigned a, unsigned b) { return a * b; }
static inline unsigned __brev(unsigned v) { return __builtin_bitreverse32(v); }
static inline int __clz(int v) { return v ? __builtin_clz((unsigned) v) : 32; }
static inline int __clzll(long long v) { return v ? __builtin_clzll((unsigned long long) v) : 64; }
static inline float __fdividef(float a, float b) { return a / b; }
static inline float __saturatef(float a) { return a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a); }
static inline float __int_as_float(int v) { float f; __builtin_memcpy(&f, &v, 4); return f; }
static inline int __float_as_int(float f) { int v; __builtin_memcpy(&v, &f, 4); return v; }
static inline unsigned __float_as_uint(float f) { unsigned v; __builtin_memcpy(&v, &f, 4); return v; }
static inline float __uint_as_float(unsigned v) { float f; __builtin_memcpy(&f, &v, 4); return f; }
static inline void __threadfence() {}
static inline void __threadfence_block() {}
static inline float atomicAdd(float* p, float v) { const float o = *p; *p = o + v; return o; }
static inline int atomicAdd(int* p, int v) { const int o = *p; *p = o + v; return o; }
static inline unsigned atomicAdd(unsigned* p, unsigned v) { const unsigned o = *p; *p = o + v; return o; }
static inline int atomicMax(int* p, int v) { const int o = *p; if (v > o) *p = v; return o; }
static inline int atomicCAS(int* p, int c, int v) { const int o = *p; if (o == c) *p = v; return o; }

// the SIMD video instructions (packed 8-bit lanes) the i-quant block decoders use: on CUDA these are one instruction, on
// Apple Silicon NEON has them directly (vsubq_u8, vpaddlq / vdotq_s32) and the MSL kernels use those.  Here they are the
// exact per-lane semantics, so the host compile reproduces the same numbers.
static inline unsigned __vsub4(unsigned a, unsigned b) {
    unsigned out = 0;
    for (int i = 0; i < 4; ++i) {
        const int d = (int) ((a >> (8 * i)) & 0xffu) - (int) ((b >> (8 * i)) & 0xffu);
        out |= (unsigned) ((d < 0 ? 0 : d) & 0xff) << (8 * i);   // unsigned saturated
    }
    return out;
}
static inline unsigned __vsubss4(unsigned a, unsigned b) {
    unsigned out = 0;
    for (int i = 0; i < 4; ++i) {
        int d = (int) ((a >> (8 * i)) & 0xffu) - (int) ((b >> (8 * i)) & 0xffu);
        if (d < -128) d = -128;
        if (d > 127) d = 127;
        out |= (unsigned) (d & 0xff) << (8 * i);
    }
    return out;
}
static inline unsigned __vadd4(unsigned a, unsigned b) {
    unsigned out = 0;
    for (int i = 0; i < 4; ++i) {
        const int d = (int) ((a >> (8 * i)) & 0xffu) + (int) ((b >> (8 * i)) & 0xffu);
        out |= (unsigned) ((d > 255 ? 255 : d) & 0xff) << (8 * i);
    }
    return out;
}
static inline unsigned __vaddss4(unsigned a, unsigned b) {
    unsigned out = 0;
    for (int i = 0; i < 4; ++i) {
        int d = (int) ((a >> (8 * i)) & 0xffu) + (int) ((b >> (8 * i)) & 0xffu);
        if (d > 127) d = 127;
        out |= (unsigned) (d & 0xff) << (8 * i);
    }
    return out;
}
static inline unsigned __vcmpeq4(unsigned a, unsigned b) {
    unsigned out = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned ea = (a >> (8 * i)) & 0xffu, eb = (b >> (8 * i)) & 0xffu;
        out |= (ea == eb ? 0xffu : 0u) << (8 * i);
    }
    return out;
}
static inline unsigned __vcmpgtu4(unsigned a, unsigned b) {
    unsigned out = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned ea = (a >> (8 * i)) & 0xffu, eb = (b >> (8 * i)) & 0xffu;
        out |= (ea > eb ? 0xffu : 0u) << (8 * i);
    }
    return out;
}
static inline int __dp4a(unsigned a, unsigned b, int c) {
    for (int i = 0; i < 4; ++i) c += (int) (signed char) ((a >> (8 * i)) & 0xffu) * (int) (signed char) ((b >> (8 * i)) & 0xffu);
    return c;
}

// the remaining scalar math helpers CUDA provides and the decoders use (round-to-nearest, plus min/max for integers - CUDA
// brings those in from its own headers, the C++ standard library does not for every type)
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __frcp_rn_(float a) { return 1.0f / a; }
template <typename T> static inline T strata_min(T a, T b) { return a < b ? a : b; }
template <typename T> static inline T strata_max(T a, T b) { return a > b ? a : b; }
