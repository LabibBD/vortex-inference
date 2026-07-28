#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace vortex::detail {

enum class KernelSyncMode {
    Boundary,
    PerKernel
};

inline KernelSyncMode
initial_kernel_sync_mode() noexcept {
    const char* value =
        std::getenv(
            "VORTEX_SYNC_MODE"
        );

    if (value == nullptr) {
        return KernelSyncMode::Boundary;
    }

    if (
        std::strcmp(
            value,
            "per-kernel"
        ) == 0 ||
        std::strcmp(
            value,
            "legacy"
        ) == 0 ||
        std::strcmp(
            value,
            "1"
        ) == 0
    ) {
        return KernelSyncMode::PerKernel;
    }

    return KernelSyncMode::Boundary;
}

inline std::atomic<KernelSyncMode>&
kernel_sync_storage() noexcept {
    static std::atomic<KernelSyncMode>
        mode{
            initial_kernel_sync_mode()
        };

    return mode;
}

inline KernelSyncMode
kernel_sync_mode() noexcept {
    return kernel_sync_storage().load(
        std::memory_order_relaxed
    );
}

inline void set_kernel_sync_mode(
    KernelSyncMode mode
) noexcept {
    kernel_sync_storage().store(
        mode,
        std::memory_order_relaxed
    );
}

inline cudaError_t
maybe_synchronize_after_kernel() noexcept {
    if (
        kernel_sync_mode() ==
        KernelSyncMode::PerKernel
    ) {
        return cudaDeviceSynchronize();
    }

    return cudaSuccess;
}

inline cudaError_t
synchronize_execution_boundary() noexcept {
    if (
        kernel_sync_mode() ==
        KernelSyncMode::Boundary
    ) {
        return cudaDeviceSynchronize();
    }

    return cudaSuccess;
}

} // namespace vortex::detail
