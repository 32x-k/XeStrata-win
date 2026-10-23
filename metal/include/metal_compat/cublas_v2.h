// metal/include/metal_compat/cublas_v2.h - the cuBLAS surface the dsv41 sources use, over the Metal device runtime.
//
// The engine makes 9 GEMM calls (7 cublasGemmEx, 2 cublasSgemm) for the dense projections, the LM head and the vision
// tower's matmuls.  On Metal the dense path is kernels/dense_mm.metal; this header gives the calls the CUDA names so the
// shared sources compile unchanged, and routes them to strata::gpu::dense_gemm(), which the metal host runtime provides.
//
// Compile-audit finding (M5 Max, 2026-10-23): the dsv41 core needs exactly cublasCreate/Destroy/SetStream/SetWorkspace,
// cublasSgemm and cublasGemmEx, plus cublasStatus_t/cublasHandle_t/cublasOperation_t/cublasComputeType_t/cudaDataType_t.
#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"

typedef void* cublasHandle_t;

enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED = 1, CUBLAS_STATUS_ALLOC_FAILED = 3,
                      CUBLAS_STATUS_INVALID_VALUE = 7, CUBLAS_STATUS_ARCH_MISMATCH = 8, CUBLAS_STATUS_NOT_SUPPORTED = 14 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
enum cublasComputeType_t { CUBLAS_COMPUTE_16F = 64, CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_32F_FAST_16F = 74 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1, CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99 };
enum cudaDataType_t { CUDA_R_32F = 0, CUDA_R_16F = 2, CUDA_R_16BF = 14, CUDA_R_8F_E4M3 = 28 };

static inline cublasStatus_t cublasCreate(cublasHandle_t* h) {
    *h = (void*) 1;   // the Metal runtime is a singleton: the handle carries no state
    return CUBLAS_STATUS_SUCCESS;
}
static inline cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }
static inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; }
static inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) { return CUBLAS_STATUS_SUCCESS; }

// C = alpha * op(A) * op(B) + beta * C, all row-major floats.
// NOT WIRED YET: the metal host runtime has no dense GEMM entry point.  This aborts with a message instead of returning a
// wrong result, so the shared sources compile and a run fails where the work is, not somewhere confusing.  The real
// implementation drives kernels/dense_mm.metal (the kernel exists and passes its parity test).
[[noreturn]] static inline void strata_metal_dense_gemm_missing(const char* what) {
    std::fprintf(stderr, "strata metal: %s is not wired to kernels/dense_mm.metal yet\n", what);
    std::abort();
}

static inline cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                         const float*, const float*, int, const float*, int, const float*, float*, int) {
    strata_metal_dense_gemm_missing("cublasSgemm");
}

// the mixed-precision form: any argument list, because the engine passes cuBLAS's full 19-argument spelling; the body
// names kernels/dense_mm.metal, which is where the real implementation goes.
template <typename... A>
static inline cublasStatus_t cublasGemmEx(A...) {
    strata_metal_dense_gemm_missing("cublasGemmEx");
}
