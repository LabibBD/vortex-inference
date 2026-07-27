#include "vortex/benchmark.hpp"
#include "vortex/benchmark_csv.hpp"
#include "vortex/operators/activations.hpp"
#include "vortex/tensor.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N =
        1 << 20;

    constexpr int WARMUP =
        50;

    constexpr int ITERATIONS =
        500;

    std::vector<float> input(N);

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        input[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 2001
                ) - 1000
            ) * 0.01f;
    }

    Tensor d_input(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    d_input.copy_from_host(
        input.data(),
        N * sizeof(float)
    );

    const BenchmarkStats relu =
        benchmark_cuda(
            [&]() {
                ops::relu(
                    d_input,
                    d_output
                );
            },
            WARMUP,
            ITERATIONS
        );

    const BenchmarkStats sigmoid =
        benchmark_cuda(
            [&]() {
                ops::sigmoid(
                    d_input,
                    d_output
                );
            },
            WARMUP,
            ITERATIONS
        );

    const BenchmarkStats gelu =
        benchmark_cuda(
            [&]() {
                ops::gelu(
                    d_input,
                    d_output
                );
            },
            WARMUP,
            ITERATIONS
        );

    auto print_stats =
        [](const char* name,
           const BenchmarkStats& stats,
           std::size_t elements) {

        const double throughput =
            static_cast<double>(
                elements
            ) /
            (
                stats.mean_ms *
                1e-3
            );

        std::cout
            << "\n"
            << name
            << "\n";

        std::cout
            << "  Mean:      "
            << stats.mean_ms
            << " ms\n";

        std::cout
            << "  Median:    "
            << stats.median_ms
            << " ms\n";

        std::cout
            << "  P95:       "
            << stats.p95_ms
            << " ms\n";

        std::cout
            << "  Std dev:   "
            << stats.stddev_ms
            << " ms\n";

        std::cout
            << "  Min:       "
            << stats.min_ms
            << " ms\n";

        std::cout
            << "  Max:       "
            << stats.max_ms
            << " ms\n";

        std::cout
            << "  Throughput:"
            << " "
            << throughput / 1e6
            << " M elements/s\n";
    };

    std::cout
        << "=================================\n";

    std::cout
        << " VORTEX Activation Benchmark\n";

    std::cout
        << "=================================\n\n";

    std::cout
        << "Elements:    "
        << N
        << '\n';

    std::cout
        << "Warmup:      "
        << WARMUP
        << '\n';

    std::cout
        << "Iterations:  "
        << ITERATIONS
        << '\n';

    std::cout
        << std::fixed
        << std::setprecision(6);

    print_stats(
        "ReLU",
        relu,
        N
    );

    print_stats(
        "Sigmoid",
        sigmoid,
        N
    );

    print_stats(
        "GELU",
        gelu,
        N
    );


    const double relu_throughput =
        static_cast<double>(N) /
        (relu.mean_ms * 1e-3) /
        1e6;

    const double sigmoid_throughput =
        static_cast<double>(N) /
        (sigmoid.mean_ms * 1e-3) /
        1e6;

    const double gelu_throughput =
        static_cast<double>(N) /
        (gelu.mean_ms * 1e-3) /
        1e6;

    append_benchmark_csv(
        "../results/csv/activation_benchmark.csv",
        "ReLU",
        N,
        relu,
        relu_throughput
    );

    append_benchmark_csv(
        "../results/csv/activation_benchmark.csv",
        "Sigmoid",
        N,
        sigmoid,
        sigmoid_throughput
    );

    append_benchmark_csv(
        "../results/csv/activation_benchmark.csv",
        "GELU",
        N,
        gelu,
        gelu_throughput
    );

    std::cout
        << "\nResults saved to "
        << "results/csv/activation_benchmark.csv\n\n";

    return 0;
}
