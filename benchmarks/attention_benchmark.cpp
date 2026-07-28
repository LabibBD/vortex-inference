#include "vortex/benchmark.hpp"
#include "vortex/operators/attention.hpp"
#include "vortex/tensor.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void run_case(
    std::size_t batch,
    std::size_t tokens,
    std::size_t head_dim
) {
    using namespace vortex;

    const std::size_t elements =
        batch *
        tokens *
        head_dim;

    std::vector<float> host(
        elements,
        0.01f
    );

    Tensor query(
        {batch, tokens, head_dim},
        DataType::Float32,
        Device::CUDA
    );

    Tensor key(
        {batch, tokens, head_dim},
        DataType::Float32,
        Device::CUDA
    );

    Tensor value(
        {batch, tokens, head_dim},
        DataType::Float32,
        Device::CUDA
    );

    Tensor output(
        {batch, tokens, head_dim},
        DataType::Float32,
        Device::CUDA
    );

    query.copy_from_host(
        host.data(),
        host.size() * sizeof(float)
    );

    key.copy_from_host(
        host.data(),
        host.size() * sizeof(float)
    );

    value.copy_from_host(
        host.data(),
        host.size() * sizeof(float)
    );

    const BenchmarkStats stats =
        benchmark_cuda(
            [&]() {
                ops::scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    output
                );
            },
            20,
            100
        );

    const double operations =
        4.0 *
        static_cast<double>(batch) *
        static_cast<double>(tokens) *
        static_cast<double>(tokens) *
        static_cast<double>(head_dim);

    const double gflops =
        operations /
        (stats.mean_ms * 1e-3) /
        1e9;

    const double score_memory_mib =
        static_cast<double>(
            batch *
            tokens *
            tokens *
            sizeof(float)
        ) /
        (1024.0 * 1024.0);

    std::cout
        << "\nShape: B="
        << batch
        << ", T="
        << tokens
        << ", D="
        << head_dim
        << '\n';

    std::cout
        << "Mean:             "
        << stats.mean_ms
        << " ms\n";

    std::cout
        << "Median:           "
        << stats.median_ms
        << " ms\n";

    std::cout
        << "P95:              "
        << stats.p95_ms
        << " ms\n";

    std::cout
        << "Approx throughput:"
        << " "
        << gflops
        << " GFLOP/s\n";

    std::cout
        << "Score matrix:     "
        << score_memory_mib
        << " MiB\n";
}

} // namespace

int main() {
    std::cout
        << "=================================\n"
        << " VORTEX Self-Attention Benchmark\n"
        << "=================================\n";

    std::cout
        << std::fixed
        << std::setprecision(6);

    run_case(8, 32, 64);
    run_case(8, 64, 64);
    run_case(8, 128, 64);

    return 0;
}
