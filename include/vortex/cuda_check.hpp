#pragma once

#include "vortex/cuda_sync.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "[VORTEX][CUDA] "                                 \
                      << cudaGetErrorString(error)                          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CUDA_KERNEL_CHECK() \
    do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK( \
            ::vortex::detail::maybe_synchronize_after_kernel() \
        ); \
    } while (0)

#define CUDA_KERNEL_LAUNCH_CHECK()                              \
    do {                                                        \
        CUDA_CHECK(cudaGetLastError());                         \
    } while (0)

#define CUDA_EXECUTION_BOUNDARY_CHECK() \
    do { \
        CUDA_CHECK( \
            ::vortex::detail::synchronize_execution_boundary() \
        ); \
    } while (0)
