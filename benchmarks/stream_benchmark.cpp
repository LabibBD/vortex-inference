#include "vortex/cuda_stream.hpp"
#include "vortex/operators/activations.hpp"
#include "vortex/tensor.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock =
    std::chrono::steady_clock;

double milliseconds(
    Clock::time_point start,
    Clock::time_point end
) {
    return std::chrono::duration<
        double,
        std::milli
    >(end - start).count();
}

} // namespace

int main() {
    using namespace vortex;

    constexpr std::size_t N =
        1 << 24;

    constexpr int WARMUP =
        20;

    constexpr int ITERATIONS =
        100;

    std::vector<float> host(
        N,
        0.5f
    );

    Tensor input_a(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor input_b(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor output_a(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor output_b(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    input_a.copy_from_host(
        host.data(),
        host.size() * sizeof(float)
    );

    input_b.copy_from_host(
        host.data(),
        host.size() * sizeof(float)
    );

    CudaStream stream_a;
    CudaStream stream_b;

    // ==================================================
    // Warmup
    // ==================================================

    for (int i = 0; i < WARMUP; ++i) {
        ops::gelu_async(
            input_a,
            output_a,
            stream_a.get()
        );

        ops::gelu_async(
            input_b,
            output_b,
            stream_b.get()
        );

        stream_a.synchronize();
        stream_b.synchronize();
    }

    // ==================================================
    // Sequential execution using one stream
    // ==================================================

    double sequential_total =
        0.0;

    for (
        int i = 0;
        i < ITERATIONS;
        ++i
    ) {
        const auto start =
            Clock::now();

        ops::gelu_async(
            input_a,
            output_a,
            stream_a.get()
        );

        stream_a.synchronize();

        ops::gelu_async(
            input_b,
            output_b,
            stream_a.get()
        );

        stream_a.synchronize();

        const auto stop =
            Clock::now();

        sequential_total +=
            milliseconds(
                start,
                stop
            );
    }

    // ==================================================
    // Concurrent launch on two streams
    // ==================================================

    double concurrent_total =
        0.0;

    for (
        int i = 0;
        i < ITERATIONS;
        ++i
    ) {
        const auto start =
            Clock::now();

        ops::gelu_async(
            input_a,
            output_a,
            stream_a.get()
        );

        ops::gelu_async(
            input_b,
            output_b,
            stream_b.get()
        );

        stream_a.synchronize();
        stream_b.synchronize();

        const auto stop =
            Clock::now();

        concurrent_total +=
            milliseconds(
                start,
                stop
            );
    }

    const double sequential_ms =
        sequential_total /
        ITERATIONS;

    const double concurrent_ms =
        concurrent_total /
        ITERATIONS;

    const double speedup =
        sequential_ms /
        concurrent_ms;

    std::cout
        << "=================================\n"
        << " VORTEX CUDA Stream Benchmark\n"
        << "=================================\n\n";

    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "Elements per branch: "
        << N << '\n';

    std::cout
        << "Sequential:         "
        << sequential_ms
        << " ms\n";

    std::cout
        << "Two streams:        "
        << concurrent_ms
        << " ms\n";

    std::cout
        << "Speedup:            "
        << speedup
        << "x\n";

    return 0;
}
