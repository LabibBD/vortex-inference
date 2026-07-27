#include "vortex/benchmark.hpp"
#include "vortex/operators/activations.hpp"
#include "vortex/operators/fused.hpp"
#include "vortex/operators/linear.hpp"
#include "vortex/tensor.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void run_case(
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    using namespace vortex;

    std::vector<float> h_input(
        m * k,
        0.01f
    );

    std::vector<float> h_weight(
        k * n,
        0.02f
    );

    std::vector<float> h_bias(
        n,
        0.01f
    );

    Tensor input(
        {m, k},
        DataType::Float32,
        Device::CUDA
    );

    Tensor weight(
        {k, n},
        DataType::Float32,
        Device::CUDA
    );

    Tensor bias(
        {n},
        DataType::Float32,
        Device::CUDA
    );

    Tensor linear_output(
        {m, n},
        DataType::Float32,
        Device::CUDA
    );

    Tensor baseline_output(
        {m, n},
        DataType::Float32,
        Device::CUDA
    );

    Tensor fused_output(
        {m, n},
        DataType::Float32,
        Device::CUDA
    );

    input.copy_from_host(
        h_input.data(),
        h_input.size() * sizeof(float)
    );

    weight.copy_from_host(
        h_weight.data(),
        h_weight.size() * sizeof(float)
    );

    bias.copy_from_host(
        h_bias.data(),
        h_bias.size() * sizeof(float)
    );

    const BenchmarkStats baseline =
        benchmark_cuda(
            [&]() {
                ops::linear(
                    input,
                    weight,
                    bias,
                    linear_output
                );

                ops::gelu(
                    linear_output,
                    baseline_output
                );
            },
            20,
            200
        );

    const BenchmarkStats fused =
        benchmark_cuda(
            [&]() {
                ops::linear_gelu_fused(
                    input,
                    weight,
                    bias,
                    fused_output
                );
            },
            20,
            200
        );

    const double speedup =
        baseline.mean_ms /
        fused.mean_ms;

    std::cout
        << "\nShape: "
        << m << " x "
        << k << " x "
        << n << '\n';

    std::cout
        << "Baseline Linear + GELU\n"
        << "  Mean:   "
        << baseline.mean_ms
        << " ms\n"
        << "  Median: "
        << baseline.median_ms
        << " ms\n"
        << "  P95:    "
        << baseline.p95_ms
        << " ms\n";

    std::cout
        << "Fused Linear+Bias+GELU\n"
        << "  Mean:   "
        << fused.mean_ms
        << " ms\n"
        << "  Median: "
        << fused.median_ms
        << " ms\n"
        << "  P95:    "
        << fused.p95_ms
        << " ms\n";

    std::cout
        << "Speedup: "
        << speedup
        << "x\n";
}

} // namespace

int main() {
    std::cout
        << "=================================\n"
        << " VORTEX Kernel Fusion Benchmark\n"
        << "=================================\n";

    std::cout
        << std::fixed
        << std::setprecision(6);

    run_case(
        32,
        512,
        512
    );

    run_case(
        128,
        512,
        512
    );

    run_case(
        512,
        512,
        512
    );

    return 0;
}
