#include "vortex/memory_pool.hpp"

#include <iostream>

int main() {
    using namespace vortex;

    MemoryPool pool;

    std::cout
        << "=================================\n"
        << " VORTEX Memory Pool Test\n"
        << "=================================\n\n";

    void* a =
        pool.acquire(1000);

    void* b =
        pool.acquire(1000);

    auto first =
        pool.stats();

    std::cout
        << "After two allocations:\n"
        << "CUDA allocations: "
        << first.cuda_allocations
        << '\n'
        << "Active bytes:     "
        << first.active_bytes
        << '\n';

    pool.release(
        a,
        1000
    );

    auto released =
        pool.stats();

    std::cout
        << "\nAfter release:\n"
        << "Cached blocks:    "
        << released.cached_blocks
        << '\n'
        << "Cached bytes:     "
        << released.cached_bytes
        << '\n';

    void* c =
        pool.acquire(1000);

    auto reused =
        pool.stats();

    std::cout
        << "\nAfter reuse:\n"
        << "CUDA allocations: "
        << reused.cuda_allocations
        << '\n'
        << "Reuse count:      "
        << reused.reuse_count
        << '\n';

    const bool pointer_reused =
        c == a;

    pool.release(
        b,
        1000
    );

    pool.release(
        c,
        1000
    );

    auto final_stats =
        pool.stats();

    std::cout
        << "\nPointer reused:   "
        << (pointer_reused ? "YES" : "NO")
        << '\n';

    std::cout
        << "Peak active bytes:"
        << " "
        << final_stats.peak_active_bytes
        << '\n';

    std::cout
        << "Final cached blocks: "
        << final_stats.cached_blocks
        << '\n';

    const bool passed =
        first.cuda_allocations == 2 &&
        reused.cuda_allocations == 2 &&
        reused.reuse_count == 1 &&
        pointer_reused &&
        final_stats.cached_blocks == 2;

    if (passed) {
        std::cout
            << "\n[VORTEX] Memory pool test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Memory pool test FAILED\n";

    return 1;
}
