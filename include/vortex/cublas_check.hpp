#pragma once

#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>

inline const char* cublas_status_string(
    cublasStatus_t status
) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "NOT_SUPPORTED";

        default:
            return "UNKNOWN";
    }
}

#define CUBLAS_CHECK(call)                                      \
    do {                                                        \
        const cublasStatus_t status = (call);                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                  \
            std::cerr                                           \
                << "[VORTEX][cuBLAS] "                          \
                << cublas_status_string(status)                 \
                << " (" << __FILE__                             \
                << ":" << __LINE__ << ")\n";                    \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)
