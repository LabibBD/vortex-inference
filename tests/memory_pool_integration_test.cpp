#include "vortex/memory_pool.hpp"
#include "vortex/tensor.hpp"

#include <iostream>

int main() {
    using namespace vortex;

    auto& pool =
        MemoryPool::global();

    // Start from a known empty cache.
    pool.clear();

    void* first_pointer = nullptr;

    // ==================================================
    // First Tensor
    // ==================================================

    {
        Tensor first(
            {1024, 1024},
            DataType::Float32,
            Device::CUDA
        );

        first_pointer =
            first.data();

        const auto stats =
            pool.stats();

        std::cout
            << "After first Tensor:\n";

        std::cout
            << "CUDA allocations: "
            << stats.cuda_allocations
            << '\n';

        std::cout
            << "Active bytes:     "
            << stats.active_bytes
            << '\n';
    }

    const auto after_release =
        pool.stats();

    std::cout
        << "\nAfter first Tensor destruction:\n";

    std::cout
        << "Cached blocks:    "
        << after_release.cached_blocks
        << '\n';

    std::cout
        << "Cached bytes:     "
        << after_release.cached_bytes
        << '\n';

    // ==================================================
    // Second Tensor — same size
    // ==================================================

    void* second_pointer = nullptr;

    {
        Tensor second(
            {1024, 1024},
            DataType::Float32,
            Device::CUDA
        );

        second_pointer =
            second.data();

        const auto stats =
            pool.stats();

        std::cout
            << "\nAfter second Tensor:\n";

        std::cout
            << "CUDA allocations: "
            << stats.cuda_allocations
            << '\n';

        std::cout
            << "Reuse count:      "
            << stats.reuse_count
            << '\n';
    }

    const auto final_stats =
        pool.stats();

    const bool same_pointer =
        first_pointer == second_pointer;

    std::cout
        << "\nSame GPU pointer reused: "
        << (same_pointer ? "YES" : "NO")
        << '\n';

    std::cout
        << "Peak active bytes:       "
        << final_stats.peak_active_bytes
        << '\n';

    std::cout
        << "Final cached blocks:     "
        << final_stats.cached_blocks
        << '\n';

    const bool passed =
        same_pointer &&
        final_stats.cuda_allocations == 1 &&
        final_stats.reuse_count >= 1 &&
        final_stats.cached_blocks == 1;

    pool.clear();

    if (passed) {
        std::cout
            << "\n[VORTEX] Memory pool integration PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Memory pool integration FAILED\n";

    return 1;
}
