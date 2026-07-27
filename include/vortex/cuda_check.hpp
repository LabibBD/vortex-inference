#pragma once

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

#define CUDA_KERNEL_CHECK()                                                \
    do {                                                                   \
        CUDA_CHECK(cudaGetLastError());                                    \
        CUDA_CHECK(cudaDeviceSynchronize());                               \
    } while (0)

#define CUDA_KERNEL_LAUNCH_CHECK()                              \
    do {                                                        \
        CUDA_CHECK(cudaGetLastError());                         \
    } while (0)
