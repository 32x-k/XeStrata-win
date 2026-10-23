// Strata. The few HIP names the DeepSeek-V4.1 kernel tests use, on CUDA (the tests are written against HIP).
#pragma once
#include <cuda_runtime.h>
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventRecord(e) cudaEventRecord(e, 0)
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
