#include "vortex/benchmark.hpp"
#include "vortex/operators/matmul.hpp"
#include "vortex/tensor.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void benchmark_size(std::size_t size) {
    using namespace vortex;

    const std::size_t elements =
        size * size;

    std::vector<float> a(elements, 0.1f);
    std::vector<float> b(elements, 0.2f);

    Tensor d_a(
        {size, size},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_b(
        {size, size},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_c(
        {size, size},
        DataType::Float32,
        Device::CUDA
    );

    d_a.copy_from_host(
        a.data(),
        elements * sizeof(float)
    );

    d_b.copy_from_host(
        b.data(),
        elements * sizeof(float)
    );

    auto run =
        [&](auto operation) {
            return benchmark_cuda(
                [&]() {
                    operation(
                        d_a,
                        d_b,
                        d_c
                    );
                },
                20,
                100
            );
        };

    const auto naive =
        run(vortex::ops::matmul_naive);

    const auto tile8 =
        run(vortex::ops::matmul_tiled_8);

    const auto tile16 =
        run(vortex::ops::matmul_tiled);

    const auto tile32 =
        run(vortex::ops::matmul_tiled_32);

    const auto cublas =
        run(vortex::ops::matmul_cublas);

    const double operations =
        2.0 *
        static_cast<double>(size) *
        static_cast<double>(size) *
        static_cast<double>(size);

    auto gflops =
        [&](double ms) {
            return operations /
                (ms * 1e-3) /
                1e9;
        };

    auto print =
        [&](const char* name,
            const BenchmarkStats& stats) {

            std::cout
                << name
                << "\n  Mean:    "
                << stats.mean_ms
                << " ms\n  GFLOP/s: "
                << gflops(stats.mean_ms)
                << '\n';
        };

    std::cout
        << "\nMatrix: "
        << size
        << " x "
        << size
        << '\n';

    print("Naive", naive);
    print("Tile 8x8", tile8);
    print("Tile 16x16", tile16);
    print("Tile 32x32", tile32);
    print("cuBLAS", cublas);

    const double best_custom_ms =
        std::min(
            std::min(
                tile8.mean_ms,
                tile16.mean_ms
            ),
            tile32.mean_ms
        );

    std::cout
        << "Best custom / cuBLAS gap: "
        << best_custom_ms / cublas.mean_ms
        << "x slower than cuBLAS\n";

    std::cout
        << "16x16 speedup vs naive: "
        << naive.mean_ms / tile16.mean_ms
        << "x\n";
}

} // namespace

int main() {
    std::cout
        << "=================================\n"
        << " VORTEX GEMM Tile Benchmark\n"
        << "=================================\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    benchmark_size(256);
    benchmark_size(512);
    benchmark_size(1024);

    return 0;
}
