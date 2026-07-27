#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vortex {

struct MemoryPoolStats {
    std::size_t allocated_bytes{0};
    std::size_t active_bytes{0};
    std::size_t cached_bytes{0};
    std::size_t peak_active_bytes{0};

    std::size_t cuda_allocations{0};
    std::size_t reuse_count{0};
    std::size_t cached_blocks{0};
};

class MemoryPool {
public:
    MemoryPool() = default;
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    static MemoryPool& global();

    void* acquire(std::size_t bytes);

    void release(
        void* pointer,
        std::size_t bytes
    ) noexcept;

    void clear() noexcept;

    MemoryPoolStats stats() const;

private:
    static std::size_t size_class(
        std::size_t bytes
    );

    mutable std::mutex mutex_;

    std::unordered_map<
        std::size_t,
        std::vector<void*>
    > free_lists_;

    std::size_t allocated_bytes_{0};
    std::size_t active_bytes_{0};
    std::size_t cached_bytes_{0};
    std::size_t peak_active_bytes_{0};

    std::size_t cuda_allocations_{0};
    std::size_t reuse_count_{0};
};

} // namespace vortex
