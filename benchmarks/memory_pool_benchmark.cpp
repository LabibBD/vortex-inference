#include "vortex/benchmark.hpp"
#include "vortex/cuda_check.hpp"
#include "vortex/memory_pool.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

double elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point stop
) {
    return std::chrono::duration<
        double,
        std::milli
    >(stop - start).count();
}

} // namespace

int main() {
    using namespace vortex;

    constexpr std::size_t BYTES =
        4 * 1024 * 1024; // 4 MiB

    constexpr int WARMUP = 20;
    constexpr int ITERATIONS = 500;

    std::cout
        << "=================================\n"
        << " VORTEX Memory Allocation Benchmark\n"
        << "=================================\n\n";

    std::cout
        << "Allocation size: "
        << BYTES / (1024.0 * 1024.0)
        << " MiB\n";

    std::cout
        << "Warmup:         "
        << WARMUP << '\n';

    std::cout
        << "Iterations:     "
        << ITERATIONS << "\n\n";

    // ==================================================
    // RAW cudaMalloc / cudaFree
    // ==================================================

    for (int i = 0; i < WARMUP; ++i) {
        void* ptr = nullptr;

        CUDA_CHECK(
            cudaMalloc(&ptr, BYTES)
        );

        CUDA_CHECK(
            cudaFree(ptr)
        );
    }

    std::vector<double> raw_samples;
    raw_samples.reserve(ITERATIONS);

    for (int i = 0; i < ITERATIONS; ++i) {
        void* ptr = nullptr;

        const auto start =
            std::chrono::steady_clock::now();

        CUDA_CHECK(
            cudaMalloc(&ptr, BYTES)
        );

        CUDA_CHECK(
            cudaFree(ptr)
        );

        const auto stop =
            std::chrono::steady_clock::now();

        raw_samples.push_back(
            elapsed_ms(start, stop)
        );
    }

    const BenchmarkStats raw =
        summarize_benchmark(
            raw_samples,
            WARMUP
        );

    // ==================================================
    // MemoryPool reuse
    // ==================================================

    auto& pool =
        MemoryPool::global();

    pool.clear();

    // Seed one cached block first.
    {
        void* ptr =
            pool.acquire(BYTES);

        pool.release(
            ptr,
            BYTES
        );
    }

    for (int i = 0; i < WARMUP; ++i) {
        void* ptr =
            pool.acquire(BYTES);

        pool.release(
            ptr,
            BYTES
        );
    }

    std::vector<double> pool_samples;
    pool_samples.reserve(ITERATIONS);

    for (int i = 0; i < ITERATIONS; ++i) {
        const auto start =
            std::chrono::steady_clock::now();

        void* ptr =
            pool.acquire(BYTES);

        pool.release(
            ptr,
            BYTES
        );

        const auto stop =
            std::chrono::steady_clock::now();

        pool_samples.push_back(
            elapsed_ms(start, stop)
        );
    }

    const BenchmarkStats pooled =
        summarize_benchmark(
            pool_samples,
            WARMUP
        );

    const auto stats =
        pool.stats();

    // ==================================================
    // Results
    // ==================================================

    const double speedup =
        raw.mean_ms /
        pooled.mean_ms;

    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "Raw cudaMalloc/cudaFree\n"
        << "  Mean:    "
        << raw.mean_ms << " ms\n"
        << "  Median:  "
        << raw.median_ms << " ms\n"
        << "  P95:     "
        << raw.p95_ms << " ms\n";

    std::cout
        << "\nVORTEX MemoryPool\n"
        << "  Mean:    "
        << pooled.mean_ms << " ms\n"
        << "  Median:  "
        << pooled.median_ms << " ms\n"
        << "  P95:     "
        << pooled.p95_ms << " ms\n";

    std::cout
        << "\nAllocation speedup: "
        << speedup << "x\n";

    std::cout
        << "\nPool statistics\n"
        << "  CUDA allocations: "
        << stats.cuda_allocations << '\n'
        << "  Reuse count:      "
        << stats.reuse_count << '\n'
        << "  Cached blocks:    "
        << stats.cached_blocks << '\n'
        << "  Cached bytes:     "
        << stats.cached_bytes << '\n';

    pool.clear();

    return 0;
}
