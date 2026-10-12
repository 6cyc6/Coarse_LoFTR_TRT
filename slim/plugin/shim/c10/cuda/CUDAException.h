// Stand-in for c10/cuda/CUDAException.h: errors are not thrown but left to cudaGetLastError, which the plugin checks
// after every launch.
#pragma once
#include <cuda_runtime.h>

#define C10_CUDA_CHECK(EXPR) ((void)(EXPR))
#define C10_CUDA_KERNEL_LAUNCH_CHECK() ((void)0)
