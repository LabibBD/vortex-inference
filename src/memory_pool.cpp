#include "vortex/memory_pool.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace vortex {

MemoryPool::~MemoryPool() {
    clear();
}

MemoryPool& MemoryPool::global() {
    static MemoryPool pool;
    return pool;
}

std::size_t MemoryPool::size_class(
    std::size_t bytes
) {
    if (bytes == 0) {
        throw std::invalid_argument(
            "MemoryPool allocation size must be greater than zero"
        );
    }

    // Minimum allocation class: 256 bytes.
    std::size_t size = 256;

    // Power-of-two size classes:
    // 256, 512, 1024, 2048, ...
    while (size < bytes) {
        size <<= 1;
    }

    return size;
}

void* MemoryPool::acquire(
    std::size_t bytes
) {
    const std::size_t block_size =
        size_class(bytes);

    std::lock_guard<std::mutex> lock(
        mutex_
    );

    auto& free_list =
        free_lists_[block_size];

    void* pointer = nullptr;

    if (!free_list.empty()) {
        pointer =
            free_list.back();

        free_list.pop_back();

        cached_bytes_ -=
            block_size;

        ++reuse_count_;
    } else {
        CUDA_CHECK(
            cudaMalloc(
                &pointer,
                block_size
            )
        );

        allocated_bytes_ +=
            block_size;

        ++cuda_allocations_;
    }

    active_bytes_ +=
        block_size;

    peak_active_bytes_ =
        std::max(
            peak_active_bytes_,
            active_bytes_
        );

    return pointer;
}

void MemoryPool::release(
    void* pointer,
    std::size_t bytes
) noexcept {
    if (pointer == nullptr || bytes == 0) {
        return;
    }

    const std::size_t block_size =
        size_class(bytes);

    std::lock_guard<std::mutex> lock(
        mutex_
    );

    free_lists_[block_size].push_back(
        pointer
    );

    active_bytes_ -=
        block_size;

    cached_bytes_ +=
        block_size;
}

void MemoryPool::clear() noexcept {
    std::lock_guard<std::mutex> lock(
        mutex_
    );

    for (
        auto& [size, blocks] :
        free_lists_
    ) {
        for (void* pointer : blocks) {
            cudaFree(pointer);
        }
    }

    free_lists_.clear();

    allocated_bytes_ =
        active_bytes_;

    cached_bytes_ = 0;
}

MemoryPoolStats MemoryPool::stats() const {
    std::lock_guard<std::mutex> lock(
        mutex_
    );

    std::size_t cached_blocks = 0;

    for (
        const auto& [size, blocks] :
        free_lists_
    ) {
        cached_blocks +=
            blocks.size();
    }

    return MemoryPoolStats{
        allocated_bytes_,
        active_bytes_,
        cached_bytes_,
        peak_active_bytes_,
        cuda_allocations_,
        reuse_count_,
        cached_blocks
    };
}

} // namespace vortex
