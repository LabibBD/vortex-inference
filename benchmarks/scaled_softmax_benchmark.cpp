#include "vortex/benchmark.hpp"
#include "vortex/operators/scaled_softmax.hpp"
#include "vortex/operators/softmax.hpp"
#include "vortex/operators/transformer.hpp"
#include "vortex/tensor.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void run_case(
    std::size_t rows,
    std::size_t columns
) {
    using namespace vortex;

    constexpr float SCALE =
        0.125f;

    const std::size_t elements =
        rows *
        columns;

    std::vector<float> host(
        elements
    );

    for (
        std::size_t index = 0;
        index < elements;
        ++index
    ) {
        host[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 101
                ) -
                50
            ) *
            0.02f;
    }

    Tensor input(
        {rows, columns},
        DataType::Float32,
        Device::CUDA
    );

    Tensor scaled(
        {rows, columns},
        DataType::Float32,
        Device::CUDA
    );

    Tensor separate_output(
        {rows, columns},
        DataType::Float32,
        Device::CUDA
    );

    Tensor fused_output(
        {rows, columns},
        DataType::Float32,
        Device::CUDA
    );

    input.copy_from_host(
        host.data(),
        host.size() *
            sizeof(float)
    );

    const BenchmarkStats separate =
        benchmark_cuda(
            [&]() {
                ops::scale(
                    input,
                    SCALE,
                    scaled
                );

                ops::softmax(
                    scaled,
                    separate_output
                );
            },
            30,
            300
        );

    const BenchmarkStats fused =
        benchmark_cuda(
            [&]() {
                ops::scaled_softmax(
                    input,
                    SCALE,
                    fused_output
                );
            },
            30,
            300
        );

    const double speedup =
        separate.mean_ms /
        fused.mean_ms;

    std::cout
        << "\nShape: "
        << rows
        << " x "
        << columns
        << '\n';

    std::cout
        << "Separate mean: "
        << separate.mean_ms
        << " ms\n";

    std::cout
        << "Fused mean:    "
        << fused.mean_ms
        << " ms\n";

    std::cout
        << "Speedup:       "
        << speedup
        << "x\n";
}

} // namespace

int main() {
    std::cout
        << "=================================\n"
        << " VORTEX Scaled-Softmax Benchmark\n"
        << "=================================\n";

    std::cout
        << std::fixed
        << std::setprecision(6);

    run_case(128, 64);
    run_case(512, 128);
    run_case(1024, 256);

    return 0;
}
