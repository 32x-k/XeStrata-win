
// the round-to-nearest float helpers: on Apple Silicon the FP ops are already IEEE and un-fused here
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fmaf_rn(float a, float b, float c) { return a * b + c; }
static inline float __frcp_rn(float a) { return 1.0f / a; }
static inline float __fsqrt_rn(float a) { return __builtin_sqrtf(a); }
